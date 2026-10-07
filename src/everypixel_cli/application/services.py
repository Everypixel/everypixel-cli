"""Application services for API operations, tasks, downloads, and media payloads."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, get_args

from pydantic import BaseModel, TypeAdapter, ValidationError

from ..errors import (
    APIResponseError,
    FileReadError,
    InputParsingError,
    TaskFailedError,
    TaskPollingError,
    ValidationCLIError,
)
from ..files import (
    download_urls,
    is_url,
    media_value,
    result_downloads,
    save_inline_result,
)
from ..openapi import (
    find_operation,
    load_schema,
    operation_help,
    validate_payload_against_operation,
)
from ..schemas import (
    ChatRequest,
    TTS_CREATE_PAYLOADS,
    TTS_DESIGN_PAYLOADS,
    TTSClonePayload,
    TTSDefaultCreatePayload,
    TTSVoiceSavePayload,
    TTSVoicesPayload,
    TTSVoiceDeletePayload,
    GPT_IMAGE_MODEL_QUALITIES,
    ImageEditPayload,
    ImageGeneratePayload,
    ImageUpscalePayload,
    ImageVectorizePayload,
    LipsyncImagePayload,
    LipsyncVideoPayload,
    RecraftGeneratePayload,
    TaskResponse,
    VideoEditRequest,
    VideoGenerateRequest,
    VideoGenerateModel,
    VideoUpscalePayload,
    normalize_grok_model,
    normalize_image_quality,
)
from .chat import ChatClientProtocol, ChatService, build_chat_payload
from .models import (
    ExecutionOptions,
    OperationCancelled,
    OperationRequest,
    OperationResult,
)


_VIDEO_GENERATE_ADAPTER: TypeAdapter[VideoGenerateRequest] = TypeAdapter(
    VideoGenerateRequest
)
_VIDEO_EDIT_ADAPTER: TypeAdapter[VideoEditRequest] = TypeAdapter(VideoEditRequest)


class EverypixelClientProtocol(ChatClientProtocol, Protocol):
    """Small infrastructure seam used by application services."""

    base_url: str

    def get_status(self, task_id: str) -> Any: ...

    def check_auth(self) -> None: ...


def raise_for_task_failure(payload: dict[str, Any], *, fallback_task_id: str) -> None:
    """Raise a typed error when a status response reached ``FAILURE``."""

    if payload.get("status") != "FAILURE":
        return
    task_id = payload.get("task_id")
    resolved_task_id = (
        task_id if isinstance(task_id, str) and task_id else fallback_task_id
    )
    message, details = extract_task_failure(payload)
    raise TaskFailedError(
        message,
        task_id=resolved_task_id,
        details=details,
    )


def extract_task_failure(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Extract a safe error message and structured diagnostics from task data."""

    error = payload.get("error")
    details: dict[str, Any] = {}
    message = extract_failure_message(error)
    if not message:
        for key in ("message", "detail", "reason"):
            message = extract_failure_message(payload.get(key))
            if message:
                break

    code = extract_failure_code(error)
    if not code:
        for key in ("error_code", "code"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                code = value
                break
    if code:
        details["api_error_code"] = code

    if isinstance(error, dict):
        error_details = error.get("details") or error.get("metadata")
        if isinstance(error_details, (dict, list, str, int, float, bool)):
            details["api_details"] = error_details
    else:
        for key in ("details", "metadata"):
            value = payload.get(key)
            if isinstance(value, (dict, list, str, int, float, bool)):
                details["api_details"] = value
                break
    return message or "Task failed", details


def extract_failure_message(value: Any) -> str | None:
    """Find a human-readable failure text without stringifying whole payloads."""

    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("message", "detail", "reason", "error"):
            message = extract_failure_message(value.get(key))
            if message:
                return message
    return None


def extract_failure_code(value: Any) -> str | None:
    """Find an optional stable API error code in a structured failure."""

    if not isinstance(value, dict):
        return None
    for key in ("code", "error_code", "type"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


class DownloadService:
    """Save completed task results without presentation concerns."""

    def save(
        self,
        task: Mapping[str, Any],
        directory: Path,
        *,
        fallback_extension: str,
        fallback_task_id: str,
        cancel_check: Callable[[], None] | None = None,
    ) -> tuple[Path, ...]:
        result = task.get("result")
        task_id = task.get("task_id")
        resolved_task_id = (
            task_id if isinstance(task_id, str) and task_id else fallback_task_id
        )
        downloads = result_downloads(result, fallback_ext=fallback_extension)
        if downloads:
            paths = download_urls(
                downloads,
                directory,
                fallback_ext=fallback_extension,
                task_id=resolved_task_id,
                cancel_check=cancel_check,
            )
        else:
            paths = save_inline_result(
                result,
                directory,
                task_id=resolved_task_id,
                fallback_ext=fallback_extension,
                cancel_check=cancel_check,
            )
        return tuple(Path(path) for path in paths)


class TaskService:
    """Interpret and wait for task lifecycle states."""

    def __init__(self, client: EverypixelClientProtocol) -> None:
        self.client = client

    def status(self, task_id: str) -> dict[str, Any]:
        payload = self.client.get_status(task_id)
        if not isinstance(payload, Mapping):
            raise APIResponseError("API returned invalid task status response")
        payload = dict(payload)
        raise_for_task_failure(payload, fallback_task_id=task_id)
        return payload

    def wait(self, task_id: str, options: ExecutionOptions) -> dict[str, Any]:
        started = time.monotonic()
        while True:
            options.check_cancelled()
            payload = self.status(task_id)
            options.check_cancelled()
            if payload.get("status") == "SUCCESS":
                payload["elapsed_sec"] = round(time.monotonic() - started, 2)
                return payload
            if time.monotonic() - started >= options.timeout:
                raise TaskPollingError(
                    f"Timed out waiting for task {task_id}",
                    code="timeout",
                    details=payload,
                )
            if options.cancel_event is None:
                time.sleep(options.poll_interval)
            elif options.cancel_event.wait(options.poll_interval):
                raise OperationCancelled


class OperationService:
    """Execute sync and async operations through one typed pipeline."""

    def __init__(
        self,
        client: EverypixelClientProtocol,
        *,
        download_service: DownloadService | None = None,
    ) -> None:
        self.client = client
        self.tasks = TaskService(client)
        self.downloads = download_service or DownloadService()

    def execute(self, request: OperationRequest) -> OperationResult:
        method = request.method.upper()
        payload = dict(request.payload)
        use_query = method == "GET" or (
            method == "DELETE" and request.endpoint == "/v1/tts_voice"
        )
        response = self.client.request(
            method,
            request.endpoint,
            json=None if use_query else payload,
            params=payload if use_query else None,
        )
        if not isinstance(response, Mapping) or "task_id" not in response:
            return OperationResult(value=response)
        try:
            created = TaskResponse.model_validate(response).model_dump(
                exclude_none=True
            )
        except ValidationError as exc:
            raise TaskPollingError(
                "API response does not contain a valid task ID"
            ) from exc
        if not request.execution.wait_for_result:
            return OperationResult(value=created, task=created)
        task_id = str(created["task_id"])
        completed = self.tasks.wait(task_id, request.execution)
        saved_files: tuple[Path, ...] = ()
        if request.execution.download_directory is not None:
            saved_files = self.downloads.save(
                completed,
                request.execution.download_directory,
                fallback_extension=request.fallback_extension,
                fallback_task_id=task_id,
                cancel_check=request.execution.check_cancelled,
            )
        return OperationResult(value=completed, task=completed, saved_files=saved_files)

    def status(
        self,
        task_id: str,
        options: ExecutionOptions,
        *,
        fallback_extension: str = ".bin",
    ) -> OperationResult:
        task = (
            self.tasks.wait(task_id, options)
            if options.wait_for_result
            else self.tasks.status(task_id)
        )
        saved_files: tuple[Path, ...] = ()
        if options.download_directory is not None:
            saved_files = self.downloads.save(
                task,
                options.download_directory,
                fallback_extension=fallback_extension,
                fallback_task_id=task_id,
                cancel_check=options.check_cancelled,
            )
        return OperationResult(value=task, task=task, saved_files=saved_files)

    def wait(self, task_id: str, options: ExecutionOptions) -> OperationResult:
        task = self.tasks.wait(task_id, options)
        saved_files: tuple[Path, ...] = ()
        if options.download_directory is not None:
            saved_files = self.downloads.save(
                task,
                options.download_directory,
                fallback_extension=".bin",
                fallback_task_id=task_id,
                cancel_check=options.check_cancelled,
            )
        return OperationResult(value=task, task=task, saved_files=saved_files)


def execution_options(
    *, wait: bool, download_directory: Path | None, timeout: float, poll_interval: float
) -> ExecutionOptions:
    """Build options once so download-implies-wait lives outside the CLI."""

    return ExecutionOptions(
        wait=wait,
        download_directory=download_directory,
        timeout=timeout,
        poll_interval=poll_interval,
    )


def fallback_extension_for_endpoint(path: str) -> str:
    """Infer the existing conservative extension fallback from an endpoint path."""

    lowered = path.lower()
    if lowered == "/v1/image_vectorize":
        return ".svg"
    if "transcribe" in lowered or "asr" in lowered:
        return ".txt"
    if "video" in lowered or "lipsync" in lowered:
        return ".mp4"
    if "tts" in lowered or "speech" in lowered or "audio" in lowered:
        return ".mp3"
    if "image" in lowered:
        return ".png"
    return ".bin"


@dataclass(frozen=True)
class _VideoEditInputs:
    prompt: str | None
    model: str
    images: list[str]
    video: str | None
    audio: str | None
    duration: int | None
    resolution: str | None
    aspect_ratio: str | None
    seed: int | None
    generate_audio: bool | None
    callback_url: str | None
    keyframes: list[str]
    public_figure_threshold: str | None


def _validated_video_edit_payload(values: Mapping[str, Any]) -> dict[str, Any]:
    validated = _VIDEO_EDIT_ADAPTER.validate_python(
        {key: value for key, value in values.items() if value is not None}
    )
    return validated.model_dump(mode="json", exclude_none=True)


def _reject_video_edit_options(model: str, options: Mapping[str, Any]) -> None:
    provided = [
        f"--{name.replace('_', '-')}"
        for name, value in options.items()
        if value is not None and value != []
    ]
    if provided:
        joined = ", ".join(provided)
        raise ValidationCLIError(f"{model} video edit does not support {joined}")


def _build_content_video_edit_payload(
    inputs: _VideoEditInputs, *, include_roles: bool
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": inputs.prompt}]
    for value in inputs.images:
        item = {
            "type": "image_url",
            "image_url": {"url": media_value(value)},
        }
        if include_roles:
            item["role"] = "reference_image"
        content.append(item)
    if inputs.video:
        item = {
            "type": "video_url",
            "video_url": {"url": media_value(inputs.video)},
        }
        if include_roles:
            item["role"] = "reference_video"
        content.append(item)
    if inputs.audio:
        item = {
            "type": "audio_url",
            "audio_url": {"url": media_value(inputs.audio)},
        }
        if include_roles:
            item["role"] = "reference_audio"
        content.append(item)
    return _validated_video_edit_payload(
        {
            "prompt": inputs.prompt,
            "model": inputs.model,
            "duration": inputs.duration,
            "resolution": inputs.resolution,
            "aspect_ratio": inputs.aspect_ratio,
            "generate_audio": inputs.generate_audio,
            "callback_url": inputs.callback_url,
            "content": content,
        }
    )


def _build_seedance_video_edit_payload(
    inputs: _VideoEditInputs,
) -> dict[str, Any]:
    _reject_video_edit_options(
        inputs.model,
        {
            "seed": inputs.seed,
            "keyframe": inputs.keyframes,
            "public_figure_threshold": inputs.public_figure_threshold,
        },
    )
    return _build_content_video_edit_payload(inputs, include_roles=True)


def _build_kling_video_edit_payload(inputs: _VideoEditInputs) -> dict[str, Any]:
    _reject_video_edit_options(
        inputs.model,
        {
            "audio": inputs.audio,
            "seed": inputs.seed,
            "keyframe": inputs.keyframes,
            "public_figure_threshold": inputs.public_figure_threshold,
        },
    )
    return _build_content_video_edit_payload(inputs, include_roles=False)


def _build_wan_video_edit_payload(inputs: _VideoEditInputs) -> dict[str, Any]:
    _reject_video_edit_options(
        inputs.model,
        {
            "audio": inputs.audio,
            "generate_audio": inputs.generate_audio,
            "keyframe": inputs.keyframes,
            "public_figure_threshold": inputs.public_figure_threshold,
        },
    )
    return _validated_video_edit_payload(
        {
            "prompt": inputs.prompt,
            "model": inputs.model,
            "duration": inputs.duration,
            "resolution": inputs.resolution,
            "aspect_ratio": inputs.aspect_ratio,
            "video_url": media_value(inputs.video) if inputs.video else None,
            "reference_image_urls": [media_value(value) for value in inputs.images],
            "seed": inputs.seed,
            "callback_url": inputs.callback_url,
        }
    )


def _build_flux3_video_edit_payload(inputs: _VideoEditInputs) -> dict[str, Any]:
    _reject_video_edit_options(
        inputs.model,
        {
            "image": inputs.images,
            "audio": inputs.audio,
            "seed": inputs.seed,
            "keyframe": inputs.keyframes,
            "public_figure_threshold": inputs.public_figure_threshold,
        },
    )
    return _validated_video_edit_payload(
        {
            "prompt": inputs.prompt,
            "model": inputs.model,
            "duration": inputs.duration,
            "resolution": inputs.resolution,
            "aspect_ratio": inputs.aspect_ratio,
            "video_url": media_value(inputs.video) if inputs.video else None,
            "generate_audio": inputs.generate_audio,
            "callback_url": inputs.callback_url,
        }
    )


def _parse_aleph_keyframe(value: str) -> dict[str, Any]:
    try:
        keyframe = json.loads(value)
    except json.JSONDecodeError as exc:
        raise InputParsingError("Unable to parse Aleph 2 keyframe JSON") from exc
    if not isinstance(keyframe, dict):
        raise InputParsingError("Aleph 2 keyframe must be a JSON object")
    image_url = keyframe.get("image_url")
    if isinstance(image_url, str):
        keyframe["image_url"] = media_value(image_url)
    return keyframe


def _build_aleph_video_edit_payload(inputs: _VideoEditInputs) -> dict[str, Any]:
    _reject_video_edit_options(
        inputs.model,
        {
            "image": inputs.images,
            "audio": inputs.audio,
            "duration": inputs.duration,
            "resolution": inputs.resolution,
            "generate_audio": inputs.generate_audio,
        },
    )
    return _validated_video_edit_payload(
        {
            "prompt": inputs.prompt,
            "model": inputs.model,
            "video_url": media_value(inputs.video) if inputs.video else None,
            "keyframes": [_parse_aleph_keyframe(value) for value in inputs.keyframes],
            "seed": inputs.seed,
            "target_aspect_ratio": inputs.aspect_ratio,
            "public_figure_threshold": inputs.public_figure_threshold,
            "callback_url": inputs.callback_url,
        }
    )


_VIDEO_EDIT_BUILDERS: dict[str, Callable[[_VideoEditInputs], dict[str, Any]]] = {
    "seedance2": _build_seedance_video_edit_payload,
    "seedance2-mini": _build_seedance_video_edit_payload,
    "seedance2.5": _build_seedance_video_edit_payload,
    "kling-3-omni": _build_kling_video_edit_payload,
    "wan2.7": _build_wan_video_edit_payload,
    "aleph2": _build_aleph_video_edit_payload,
    "flux3": _build_flux3_video_edit_payload,
}


def build_video_edit_payload(
    *,
    prompt: str | None,
    images: list[str],
    video: str | None,
    audio: str | None,
    duration: int | None,
    resolution: str | None,
    aspect_ratio: str | None,
    generate_audio: bool | None,
    callback_url: str | None,
    model: str = "seedance2",
    seed: int | None = None,
    keyframes: list[str] | None = None,
    public_figure_threshold: str | None = None,
) -> dict[str, Any]:
    """Build a model-specific JSON ``video_edit`` payload."""

    inputs = _VideoEditInputs(
        prompt=prompt,
        model=model,
        images=images,
        video=video,
        audio=audio,
        duration=duration,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        seed=seed,
        generate_audio=generate_audio,
        callback_url=callback_url,
        keyframes=keyframes or [],
        public_figure_threshold=public_figure_threshold,
    )
    builder = _VIDEO_EDIT_BUILDERS.get(model)
    if builder is None:
        raise ValidationCLIError(
            "Unsupported video edit model",
            details={"model": model},
        )
    return builder(inputs)


_IMAGE_GENERATE_BUILDERS: dict[str, type[BaseModel]] = {
    "recraftv4_1_vector": RecraftGeneratePayload,
    "recraftv4_1_pro_vector": RecraftGeneratePayload,
}


def build_image_generate_payload(**values: Any) -> dict[str, Any]:
    """Validate and build image generation payloads including local media."""

    image = values.pop("image", None)
    if image is not None:
        values["image_url"] = media_value(image)
    controls = values.get("controls")
    if isinstance(controls, str):
        try:
            values["controls"] = json.loads(controls)
        except json.JSONDecodeError as exc:
            raise InputParsingError("Unable to parse Recraft controls JSON") from exc
    builder = _IMAGE_GENERATE_BUILDERS.get(
        values.get("model", "zimage"), ImageGeneratePayload
    )
    return builder.model_validate(
        {key: value for key, value in values.items() if value is not None}
    ).model_dump(mode="json", exclude_none=True)


def build_image_edit_payload(**values: Any) -> dict[str, Any]:
    """Validate image edit payloads and encode every local source image."""

    images = values.pop("images")
    return ImageEditPayload(
        **{key: value for key, value in values.items() if value is not None},
        image_urls=[media_value(item) for item in images],
    ).model_dump(mode="json", exclude_none=True)


def build_image_vectorize_payload(
    *, image: str, callback_url: str | None
) -> dict[str, Any]:
    return ImageVectorizePayload.model_validate(
        {"image_url": media_value(image), "callback_url": callback_url}
    ).model_dump(mode="json", exclude_none=True)


def build_image_upscale_payload(**values: Any) -> dict[str, Any]:
    """Build an image upscale request from one URL/file/task source."""

    image = values.pop("image", None)
    return ImageUpscalePayload(
        **values,
        image_url=media_value(image) if image else None,
    ).model_dump(exclude_none=True)


def build_video_generate_payload(**values: Any) -> dict[str, Any]:
    """Build text/image video generation payloads with typed model validation."""

    image = values.pop("image", None)
    last_image = values.pop("last_image", None)
    reference_images = values.pop("reference_images", [])
    reference_videos = values.pop("reference_videos", [])
    reference_audios = values.pop("reference_audios", [])
    values = normalize_grok_model(values)
    if values.get("duration") is None:
        model = values.get("model")
        required_duration_defaults = {
            "minimax-h3-turbo": 5,
            "minimax-h3": 5,
            "ltx23": 5,
            "grok-imagine": 5,
            "grok-imagine-1.5": 5,
        }
        values["duration"] = (
            required_duration_defaults.get(model) if isinstance(model, str) else None
        )
    payload = {
        **values,
        "image_url": media_value(image) if image else None,
        "image_last_url": media_value(last_image) if last_image else None,
    }
    if reference_images:
        payload["reference_image_urls"] = [
            media_value(value) for value in reference_images
        ]
    if reference_videos:
        payload["reference_video_urls"] = [
            media_value(value) for value in reference_videos
        ]
    if reference_audios:
        payload["reference_audio_urls"] = [
            media_value(value) for value in reference_audios
        ]
    validated = _VIDEO_GENERATE_ADAPTER.validate_python(
        {key: value for key, value in payload.items() if value is not None}
    )
    return validated.model_dump(mode="json", exclude_none=True)


def build_video_upscale_payload(**values: Any) -> dict[str, Any]:
    """Build a video upscale request."""

    video = values.pop("video", None)
    return VideoUpscalePayload(
        **values,
        video_url=media_value(video) if video else None,
    ).model_dump(exclude_none=True)


def build_lipsync_video_payload(**values: Any) -> dict[str, Any]:
    """Build a lipsync video request."""

    video = values.pop("video")
    audio = values.pop("audio")
    return LipsyncVideoPayload(
        **values, video_url=media_value(video), audio_url=media_value(audio)
    ).model_dump(exclude_none=True)


def build_lipsync_image_payload(**values: Any) -> dict[str, Any]:
    """Build a lipsync image request."""

    image = values.pop("image")
    audio = values.pop("audio")
    return LipsyncImagePayload(
        **values, image_url=media_value(image), audio_url=media_value(audio)
    ).model_dump(exclude_none=True)


def build_media_payload(
    *, media_key: str, media: str, values: Mapping[str, Any]
) -> dict[str, Any]:
    """Build simple JSON endpoints that accept one local or remote media value."""

    return {media_key: media_value(media), **dict(values)}


def build_image_angles_payload(**values: Any) -> dict[str, Any]:
    """Build the image angle request and omit only absent optional values."""

    image = values.pop("image")
    return {
        "image_url": media_value(image),
        **{key: value for key, value in values.items() if value is not None},
    }


def build_image_colors_payload(*, image: str, reference: str) -> dict[str, Any]:
    """Build the two-image color transfer payload."""

    return {
        "image_url": media_value(image),
        "image_reference_url": media_value(reference),
    }


def read_text_input(text: str | None, text_file: Path | None) -> str:
    """Read a text argument in the application layer."""

    if text_file:
        try:
            return text_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise FileReadError(
                "Unable to read text input", details={"path": str(text_file)}
            ) from exc
    if text is None:
        raise ValidationCLIError("--text or --text-file is required")
    return text


def validate_tts_payload(
    path: str, values: dict[str, Any], *, generic: bool = False
) -> BaseModel:
    """Select a provider schema without dropping unsupported controls."""
    if generic:
        schemas: list[type[BaseModel]] = [
            *TTS_CREATE_PAYLOADS.values(),
            *TTS_DESIGN_PAYLOADS.values(),
            TTSClonePayload,
            TTSDefaultCreatePayload,
        ]
        known_fields = {field for schema in schemas for field in schema.model_fields}
        values = {key: value for key, value in values.items() if key in known_fields}
    if path == "/v1/tts_create" and "model" not in values:
        return TTSDefaultCreatePayload.model_validate(values)
    registries = {
        "/v1/tts_create": TTS_CREATE_PAYLOADS,
        "/v1/tts_design": TTS_DESIGN_PAYLOADS,
    }
    if path in registries:
        model = values.get("model", "qwen3")
        schema = registries[path].get(model) if isinstance(model, str) else None
        if schema is None:
            raise ValidationCLIError("Unsupported TTS model")
        return schema.model_validate(values)
    return TTSClonePayload.model_validate(values)


def _build_tts_payload(path: str, values: dict[str, Any]) -> dict[str, Any]:
    text = values.pop("text")
    text_file = values.pop("text_file")
    values = {key: value for key, value in values.items() if value is not None}
    values["text"] = read_text_input(text, text_file)
    return validate_tts_payload(path, values).model_dump(mode="json", exclude_none=True)


def build_tts_create_payload(**values: Any) -> dict[str, Any]:
    return _build_tts_payload("/v1/tts_create", values)


def build_tts_design_payload(**values: Any) -> dict[str, Any]:
    return _build_tts_payload("/v1/tts_design", values)


def build_tts_clone_payload(**values: Any) -> dict[str, Any]:
    audio = values.pop("audio")
    if audio is not None:
        values["audio_url"] = media_value(audio)
    return _build_tts_payload("/v1/tts_clone", values)


def build_tts_voice_payload(**values: Any) -> dict[str, Any]:
    audio = values.pop("audio")
    if audio:
        values["audio_urls"] = [media_value(sample) for sample in audio]
    return TTSVoiceSavePayload.model_validate(
        {key: value for key, value in values.items() if value is not None}
    ).model_dump(mode="json", exclude_none=True)


@dataclass
class ApplicationServices:
    """Small composition root usable by CLI and future MCP handlers."""

    _operations: OperationService

    @classmethod
    def with_client(cls, client: EverypixelClientProtocol) -> "ApplicationServices":
        return cls(_operations=OperationService(client))

    def close(self) -> None:
        """Close owned transport resources when the client exposes a lifecycle."""

        close = getattr(self._operations.client, "close", None)
        if callable(close):
            close()

    def _execute_async(
        self,
        *,
        endpoint: str,
        payload: Mapping[str, Any],
        execution: ExecutionOptions,
        fallback_extension: str,
    ) -> OperationResult:
        return self._operations.execute(
            OperationRequest(
                endpoint=endpoint,
                payload=payload,
                execution=execution,
                fallback_extension=fallback_extension,
            )
        )

    def execute_image_generate(
        self,
        *,
        prompt: str,
        model: str,
        image_size: str,
        style: str | None,
        image: str | None,
        resolution: str | None,
        seed: int,
        callback_url: str | None,
        execution: ExecutionOptions,
        lora_url: str | None = None,
        controls: dict[str, Any] | str | None = None,
        quality: str | None = None,
    ) -> OperationResult:
        payload = build_image_generate_payload(
            prompt=prompt,
            model=model,
            image_size=image_size,
            style=style,
            image=image,
            resolution=resolution,
            seed=seed,
            callback_url=callback_url,
            lora_url=lora_url,
            controls=controls,
            quality=quality,
        )
        return self._execute_async(
            endpoint="/v1/image_generate",
            payload=payload,
            execution=execution,
            fallback_extension=(
                ".svg" if model in _IMAGE_GENERATE_BUILDERS else ".png"
            ),
        )

    def execute_image_vectorize(
        self,
        *,
        image: str,
        callback_url: str | None = None,
        execution: ExecutionOptions,
    ) -> OperationResult:
        return self._execute_async(
            endpoint="/v1/image_vectorize",
            payload=build_image_vectorize_payload(
                image=image,
                callback_url=callback_url,
            ),
            execution=execution,
            fallback_extension=".svg",
        )

    def execute_image_edit(
        self,
        *,
        prompt: str,
        images: list[str],
        model: str,
        image_size: str | None,
        resolution: str | None,
        seed: int,
        callback_url: str | None,
        execution: ExecutionOptions,
        megapixel_ratio: float = 1.0,
        quality: str | None = None,
    ) -> OperationResult:
        payload = build_image_edit_payload(
            prompt=prompt,
            images=images,
            model=model,
            image_size=image_size,
            resolution=resolution,
            megapixel_ratio=megapixel_ratio,
            quality=quality,
            seed=seed,
            callback_url=callback_url,
        )
        return self._execute_async(
            endpoint="/v1/image_edit",
            payload=payload,
            execution=execution,
            fallback_extension=".png",
        )

    def execute_image_upscale(
        self,
        *,
        image: str | None,
        task_id: str | None,
        model: str,
        callback_url: str | None,
        execution: ExecutionOptions,
    ) -> OperationResult:
        payload = build_image_upscale_payload(
            image=image,
            image_from_task_id=task_id,
            model=model,
            callback_url=callback_url,
        )
        return self._execute_async(
            endpoint="/v1/image_upscale",
            payload=payload,
            execution=execution,
            fallback_extension=".jpg",
        )

    def execute_image_angles(
        self,
        *,
        image: str,
        azimuth: str,
        elevation: str,
        distance: str,
        prompt: str | None,
        execution: ExecutionOptions,
    ) -> OperationResult:
        payload = build_image_angles_payload(
            image=image,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            prompt=prompt,
        )
        return self._execute_async(
            endpoint="/v1/image_edit_angles",
            payload=payload,
            execution=execution,
            fallback_extension=".png",
        )

    def execute_image_colors(
        self,
        *,
        image: str,
        reference: str,
        execution: ExecutionOptions,
    ) -> OperationResult:
        return self._execute_async(
            endpoint="/v1/image_edit_colors",
            payload=build_image_colors_payload(image=image, reference=reference),
            execution=execution,
            fallback_extension=".png",
        )

    def execute_video_generate(
        self,
        *,
        prompt: str,
        model: str,
        duration: int | None,
        resolution: str | None,
        aspect_ratio: str | None,
        reference_images: list[str] | None = None,
        reference_videos: list[str] | None = None,
        reference_audios: list[str] | None = None,
        image: str | None = None,
        last_image: str | None = None,
        seed: int | None = None,
        generate_audio: bool | None = None,
        callback_url: str | None = None,
        execution: ExecutionOptions,
    ) -> OperationResult:
        payload = build_video_generate_payload(
            prompt=prompt,
            model=model,
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            reference_images=reference_images or [],
            reference_videos=reference_videos or [],
            reference_audios=reference_audios or [],
            image=image,
            last_image=last_image,
            seed=seed,
            generate_audio=generate_audio,
            callback_url=callback_url,
        )
        return self._execute_async(
            endpoint="/v1/video_generate",
            payload=payload,
            execution=execution,
            fallback_extension=".mp4",
        )

    def execute_video_edit(
        self,
        *,
        prompt: str | None,
        model: str,
        images: list[str],
        video: str | None,
        audio: str | None,
        duration: int | None,
        resolution: str | None,
        aspect_ratio: str | None,
        seed: int | None,
        generate_audio: bool | None,
        callback_url: str | None,
        keyframes: list[str],
        public_figure_threshold: str | None,
        execution: ExecutionOptions,
    ) -> OperationResult:
        payload = build_video_edit_payload(
            prompt=prompt,
            model=model,
            images=images,
            video=video,
            audio=audio,
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            seed=seed,
            generate_audio=generate_audio,
            callback_url=callback_url,
            keyframes=keyframes,
            public_figure_threshold=public_figure_threshold,
        )
        return self._execute_async(
            endpoint="/v1/video_edit",
            payload=payload,
            execution=execution,
            fallback_extension=".mp4",
        )

    def execute_video_upscale(
        self,
        *,
        video: str | None,
        task_id: str | None,
        resolution: str,
        execution: ExecutionOptions,
        model: str = "seedvr2",
        callback_url: str | None = None,
    ) -> OperationResult:
        payload = build_video_upscale_payload(
            video=video,
            video_from_task_id=task_id,
            resolution=resolution,
            model=model,
            callback_url=callback_url,
        )
        return self._execute_async(
            endpoint="/v1/video_upscale",
            payload=payload,
            execution=execution,
            fallback_extension=".mp4",
        )

    def execute_lipsync_video(
        self,
        *,
        video: str,
        audio: str,
        resolution: str,
        prompt: str | None,
        seed: int,
        callback_url: str | None,
        execution: ExecutionOptions,
    ) -> OperationResult:
        payload = build_lipsync_video_payload(
            video=video,
            audio=audio,
            resolution=resolution,
            prompt=prompt,
            seed=seed,
            callback_url=callback_url,
        )
        return self._execute_async(
            endpoint="/v1/video_lipsync",
            payload=payload,
            execution=execution,
            fallback_extension=".mp4",
        )

    def execute_lipsync_image(
        self,
        *,
        image: str,
        audio: str,
        model: str,
        resolution: str,
        prompt: str | None,
        seed: int,
        callback_url: str | None,
        execution: ExecutionOptions,
    ) -> OperationResult:
        payload = build_lipsync_image_payload(
            image=image,
            audio=audio,
            model=model,
            resolution=resolution,
            prompt=prompt,
            seed=seed,
            callback_url=callback_url,
        )
        return self._execute_async(
            endpoint="/v1/image_lipsync",
            payload=payload,
            execution=execution,
            fallback_extension=".mp4",
        )

    def execute_audio_transcribe(
        self,
        *,
        audio: str,
        language: str,
        hints: str,
        denoise: bool,
        execution: ExecutionOptions,
    ) -> OperationResult:
        payload = build_media_payload(
            media_key="audio_url",
            media=audio,
            values={"language": language, "hints": hints, "denoise": denoise},
        )
        return self._execute_async(
            endpoint="/v1/transcribe",
            payload=payload,
            execution=execution,
            fallback_extension=".txt",
        )

    def execute_tts_create(
        self,
        *,
        text: str | None,
        text_file: Path | None,
        execution: ExecutionOptions,
        model: str | None = None,
        voice_id: str | None = None,
        speaker: str | None = None,
        style: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        seed: int | None = None,
    ) -> OperationResult:
        return self._execute_async(
            endpoint="/v1/tts_create",
            payload=build_tts_create_payload(
                text=text,
                text_file=text_file,
                model=model,
                voice_id=voice_id,
                speaker=speaker,
                style=style,
                language=language,
                prompt=prompt,
                seed=seed,
            ),
            execution=execution,
            fallback_extension=".mp3",
        )

    def execute_tts_clone(
        self,
        *,
        text: str | None,
        text_file: Path | None,
        execution: ExecutionOptions,
        audio: str | None = None,
        voice_id: str | None = None,
        model: str = "qwen3",
        language: str | None = None,
        seed: int | None = None,
    ) -> OperationResult:
        return self._execute_async(
            endpoint="/v1/tts_clone",
            payload=build_tts_clone_payload(
                audio=audio,
                text=text,
                text_file=text_file,
                voice_id=voice_id,
                model=model,
                language=language,
                seed=seed,
            ),
            execution=execution,
            fallback_extension=".mp3",
        )

    def execute_tts_design(
        self,
        *,
        text: str | None,
        text_file: Path | None,
        execution: ExecutionOptions,
        model: str = "qwen3",
        character: str | None = None,
        style: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        seed: int | None = None,
    ) -> OperationResult:
        return self._execute_async(
            endpoint="/v1/tts_design",
            payload=build_tts_design_payload(
                text=text,
                text_file=text_file,
                model=model,
                character=character,
                style=style,
                language=language,
                prompt=prompt,
                seed=seed,
            ),
            execution=execution,
            fallback_extension=".mp3",
        )

    def execute_tts_voice(
        self,
        *,
        name: str,
        audio: list[str],
        provider: str | None = None,
        preview_id: str | None = None,
        description: str = "",
    ) -> OperationResult:
        return self._operations.execute(
            OperationRequest(
                endpoint="/v1/tts_voice",
                method="POST",
                payload=build_tts_voice_payload(
                    name=name,
                    audio=audio,
                    provider=provider,
                    preview_id=preview_id,
                    description=description,
                ),
                execution=ExecutionOptions(),
            )
        )

    def list_tts_voices(
        self,
        *,
        provider: str = "labs",
        offset: int = 0,
        limit: int = 50,
    ) -> OperationResult:
        return self._operations.execute(
            OperationRequest(
                endpoint="/v1/tts_voices",
                method="GET",
                payload=TTSVoicesPayload.model_validate(
                    {"provider": provider, "offset": offset, "limit": limit}
                ).model_dump(),
                execution=ExecutionOptions(),
            )
        )

    def delete_tts_voice(self, *, voice_id: str) -> OperationResult:
        return self._operations.execute(
            OperationRequest(
                endpoint="/v1/tts_voice",
                method="DELETE",
                payload=TTSVoiceDeletePayload.model_validate(
                    {"voice_id": voice_id}
                ).model_dump(mode="json"),
                execution=ExecutionOptions(),
            )
        )

    def execute_chat(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        input_file: Path | None = None,
        payload: dict[str, Any] | None = None,
        model: str | None = None,
        stream: bool | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: str | None = None,
        request_timeout: float = 240.0,
        execution: ExecutionOptions | None = None,
        on_chunk: Callable[[dict[str, Any]], None] | None = None,
    ) -> OperationResult:
        inputs = parse_generic_payload(prompt=None, items=[], input_file=input_file)
        inputs.update(payload or {})
        request = build_chat_payload(
            payload=inputs,
            prompt=prompt,
            system=system,
            options={
                "model": model,
                "stream": stream,
                "max_completion_tokens": max_completion_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "reasoning_effort": reasoning_effort,
            },
        )
        return ChatService(self._operations.client).execute(
            request,
            request_timeout=request_timeout,
            execution=execution,
            on_chunk=on_chunk,
        )

    def get_task_status(
        self, *, task_id: str, execution: ExecutionOptions
    ) -> OperationResult:
        return self._operations.status(task_id, execution)

    def wait_for_task(
        self, *, task_id: str, execution: ExecutionOptions
    ) -> OperationResult:
        return self._operations.wait(task_id, execution)

    def execute_generic(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any],
        method: str | None,
        execution: ExecutionOptions,
        dry_run: bool = False,
        help_schema: bool = False,
        client_id_present: bool = False,
        schema_loader: Callable[
            [EverypixelClientProtocol], tuple[dict[str, Any], str]
        ] = load_schema,
    ) -> OperationResult:
        schema_source: str | None = None
        resolved_method = method.upper() if method else None
        operation = None
        if not endpoint.startswith("/"):
            schema, schema_source = schema_loader(self._operations.client)
            operation = find_operation(schema, endpoint, resolved_method)
        path = endpoint if endpoint.startswith("/") else f"/v1/{endpoint}"
        if operation:
            path = operation.path
            resolved_method = operation.method
        resolved_method = resolved_method or "POST"
        if path in {"/v1/video_edit", "/v1/video_generate"}:
            normalize_video_edit_content(payload)
        if path == "/v1/image_vectorize" and isinstance(payload.get("image_url"), str):
            payload["image_url"] = media_value(payload["image_url"])
        if help_schema:
            return OperationResult(
                value=operation_help(
                    operation,
                    endpoint=endpoint,
                    method=resolved_method,
                    path=path,
                    schema_source=schema_source,
                )
            )
        if path in {"/v1/image_generate", "/v1/image_edit", "/v1/video_generate"}:
            payload = normalize_grok_model(payload)
        if path in {"/v1/image_generate", "/v1/image_edit"}:
            try:
                payload = normalize_image_quality(payload)
            except ValueError as exc:
                raise ValidationCLIError(str(exc)) from exc
            if (
                isinstance(payload.get("model"), str)
                and payload["model"] in GPT_IMAGE_MODEL_QUALITIES
            ):
                image_schema = (
                    ImageGeneratePayload
                    if path == "/v1/image_generate"
                    else ImageEditPayload
                )
                image_schema.model_validate(payload, extra="ignore")
        if path == "/v1/video_generate" and payload.get("model") in get_args(
            VideoGenerateModel
        ):
            _VIDEO_GENERATE_ADAPTER.validate_python(payload, extra="ignore")
        if path == "/v1/image_vectorize":
            ImageVectorizePayload.model_validate(payload)
        if path == "/v1/video_edit":
            # Generic callers may send forward-compatible fields that the local
            # specialized model does not know yet; validate known semantics
            # without rewriting or narrowing their payload.
            _VIDEO_EDIT_ADAPTER.validate_python(payload, extra="ignore")
        if path == "/v1/chat/completions":
            ChatRequest.model_validate(payload, extra="ignore")
        if resolved_method == "POST" and path in {
            "/v1/tts_create",
            "/v1/tts_clone",
            "/v1/tts_design",
        }:
            validate_tts_payload(path, payload, generic=True)
        if path == "/v1/tts_voice":
            if resolved_method == "POST":
                TTSVoiceSavePayload.model_validate(payload)
            elif resolved_method == "DELETE":
                TTSVoiceDeletePayload.model_validate(payload)
        if path == "/v1/tts_voices" and resolved_method == "GET":
            TTSVoicesPayload.model_validate(payload)
        validate_payload_against_operation(operation, payload)
        if dry_run:
            return OperationResult(
                value={
                    "method": resolved_method,
                    "path": path,
                    "body": payload,
                    "headers": {
                        "Authorization": "Basic ***" if client_id_present else None
                    },
                    "schema_source": schema_source,
                }
            )
        if path == "/v1/chat/completions" and resolved_method == "POST":
            return ChatService(self._operations.client).execute(
                payload, execution=execution
            )
        return self._operations.execute(
            OperationRequest(
                endpoint=path,
                method=resolved_method,
                payload=payload,
                execution=execution,
                fallback_extension=(
                    ".svg"
                    if path == "/v1/image_generate"
                    and payload.get("model") in _IMAGE_GENERATE_BUILDERS
                    else fallback_extension_for_endpoint(path)
                ),
            )
        )

    def _execute_classic(
        self, *, path: str, media: str, params: Mapping[str, Any] | None = None
    ) -> OperationResult:
        """Execute URL or upload based classic media endpoints."""

        method, request_params, local_file = build_classic_request(
            media=media, params=params
        )
        if local_file is None:
            return OperationResult(
                value=self._operations.client.request(
                    method, path, params=request_params
                )
            )
        try:
            with local_file.open("rb") as file_handle:
                response = self._operations.client.request(
                    method,
                    path,
                    params=request_params,
                    files={"data": (local_file.name, file_handle)},
                )
        except OSError as exc:
            raise FileReadError(
                "Unable to read input file", details={"path": str(local_file)}
            ) from exc
        return OperationResult(value=response)

    def execute_keywords(
        self,
        *,
        image: str,
        lang: str,
        num_keywords: int | None,
        colors: bool,
    ) -> OperationResult:
        return self._execute_classic(
            path="/v1/keywords",
            media=image,
            params={"lang": lang, "num_keywords": num_keywords, "colors": colors},
        )

    def execute_quality(self, *, image: str) -> OperationResult:
        return self._execute_classic(path="/v1/quality", media=image)

    def execute_quality_ugc(self, *, image: str) -> OperationResult:
        return self._execute_classic(path="/v1/quality_ugc", media=image)

    def execute_faces(self, *, image: str) -> OperationResult:
        return self._execute_classic(path="/v1/faces", media=image)

    def execute_captioning(self, *, image: str) -> OperationResult:
        return self._execute_classic(path="/v1/image_captioning", media=image)

    def execute_video_keywords(self, *, video: str) -> OperationResult:
        return self._execute_classic(path="/v1/video_keywords", media=video)

    def check_auth(self) -> OperationResult:
        """Run an authentication check without a CLI dependency."""

        self._operations.client.check_auth()
        return OperationResult(value={"status": "ok"})

    def openapi(self) -> OperationResult:
        """Read the selected OpenAPI schema through the infrastructure seam."""

        return OperationResult(value=load_schema(self._operations.client)[0])

    def refresh_openapi(self) -> tuple[OperationResult, Path]:
        """Refresh the OpenAPI schema cache through the infrastructure seam."""

        from ..openapi import refresh_schema

        schema, path = refresh_schema(self._operations.client)
        return OperationResult(value=schema), path


def parse_generic_payload(
    *, prompt: str | None, items: list[str], input_file: Path | None
) -> dict[str, Any]:
    """Parse generic CLI input without depending on Typer or rendering."""

    payload: dict[str, Any] = {}
    if input_file:
        try:
            loaded = json.loads(input_file.read_text(encoding="utf-8"))
        except OSError as exc:
            raise FileReadError(
                "Unable to read input file", details={"path": str(input_file)}
            ) from exc
        except json.JSONDecodeError as exc:
            raise InputParsingError(
                "Unable to parse JSON input", details={"path": str(input_file)}
            ) from exc
        if not isinstance(loaded, dict):
            raise InputParsingError(
                "JSON input must contain an object", details={"path": str(input_file)}
            )
        payload.update(loaded)
    for raw in items:
        key, separator, value = raw.partition("=")
        if not key or not separator:
            raise InputParsingError("Invalid input item", details={"input": raw})
        try:
            payload[key] = json.loads(value)
        except ValueError:
            payload[key] = value
    if prompt is not None:
        payload["prompt"] = prompt
    return payload


def normalize_video_edit_content(payload: dict[str, Any]) -> None:
    """Encode local generic video frames and references as data URIs."""

    for key in ("video_url", "image_url", "image_last_url"):
        value = payload.get(key)
        if isinstance(value, str):
            payload[key] = media_value(value)
    for key in ("reference_image_urls", "reference_video_urls", "reference_audio_urls"):
        values = payload.get(key)
        if isinstance(values, list):
            payload[key] = [
                media_value(value) if isinstance(value, str) else value
                for value in values
            ]
    keyframes = payload.get("keyframes")
    if isinstance(keyframes, list):
        for keyframe in keyframes:
            if not isinstance(keyframe, dict):
                continue
            image_url = keyframe.get("image_url")
            if isinstance(image_url, str):
                keyframe["image_url"] = media_value(image_url)

    content = payload.get("content")
    if not isinstance(content, list):
        return
    for item in content:
        if not isinstance(item, dict):
            continue
        media_key = item.get("type")
        if media_key not in {"image_url", "video_url", "audio_url"}:
            continue
        media = item.get(media_key)
        if isinstance(media, dict) and isinstance(media.get("url"), str):
            media["url"] = media_value(media["url"])


def build_classic_request(
    *, media: str, params: Mapping[str, Any] | None = None
) -> tuple[str, dict[str, Any], Path | None]:
    """Prepare a classic media operation while keeping file I/O out of CLI."""

    clean_params = {
        key: value for key, value in (params or {}).items() if value is not None
    }
    if is_url(media):
        return "GET", {"url": media, **clean_params}, None
    target = Path(media)
    if not target.is_file():
        raise FileReadError(
            "Unable to read input file",
            code="file_not_found",
            details={"path": str(target)},
        )
    return "POST", clean_params, target
