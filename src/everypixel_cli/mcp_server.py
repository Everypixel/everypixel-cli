"""MCP tools backed by the transport-independent application services."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import anyio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
    ToolAnnotations,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import __version__
from .application import (
    ApplicationServices,
    ExecutionOptions,
    OperationResult,
)
from .application.models import OperationCancelled
from .application.serialization import serialize_operation_result
from .client import APIClient
from .config import resolved_settings
from .errors import (
    ValidationCLIError,
    format_validation_errors,
    normalize_exception,
)
from .schemas import (
    ImageEditMedia,
    ImageEditModel,
    ImageGenerateModel,
    ImageResolution,
    ImageSize,
    ImageStyle,
    LipsyncImageModel,
    LipsyncImageResolution,
    LipsyncVideoResolution,
    PublicFigureThreshold,
    RecraftControls,
    VideoEditAspectRatio,
    VideoEditDuration,
    VideoEditModel,
    VideoEditResolution,
    VideoGenerateAspectRatio,
    VideoGenerateDuration,
    VideoGenerateModel,
    VideoGenerateResolution,
    VideoUpscaleResolution,
    VideoUpscaleModel,
)


ServiceFactory = Callable[[], ApplicationServices]
_CURRENT_CANCEL_EVENT: ContextVar[Event | None] = ContextVar(
    "everypixel_mcp_cancellation",
    default=None,
)

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    idempotent_hint=True,
    open_world_hint=True,
)
LOCAL_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=True,
)


class MCPExecutionOptions(BaseModel):
    """Execution controls shared by asynchronous Everypixel tools."""

    model_config = ConfigDict(extra="forbid")

    wait: bool = Field(
        default=True,
        description=(
            "Poll until the asynchronous task reaches SUCCESS; set false to "
            "return the created task immediately."
        ),
    )
    download_directory: str | None = Field(
        default=None,
        description="Local directory for completed result files; implies waiting.",
    )
    timeout: float = Field(
        default=300.0,
        gt=0,
        description="Maximum task wait time in seconds.",
    )
    poll_interval: float = Field(
        default=2.0,
        gt=0,
        description="Delay between task status requests in seconds.",
    )


def _execution(
    value: MCPExecutionOptions | None,
    *,
    force_wait: bool | None = None,
) -> ExecutionOptions:
    options = value or MCPExecutionOptions()
    return ExecutionOptions(
        wait=options.wait if force_wait is None else force_wait,
        download_directory=(
            Path(options.download_directory) if options.download_directory else None
        ),
        timeout=options.timeout,
        poll_interval=options.poll_interval,
        cancel_event=_CURRENT_CANCEL_EVENT.get(),
    )


def _tool_result(payload: Any, *, is_error: bool = False) -> CallToolResult:
    """Build one text and structured MCP result from a JSON-compatible value."""

    structured = dict(payload) if isinstance(payload, Mapping) else {"result": payload}
    return CallToolResult(
        content=[
            TextContent(
                text=json.dumps(payload, ensure_ascii=False, default=str),
            )
        ],
        structured_content=structured,
        is_error=is_error,
    )


def _invoke(action: Callable[[], Any]) -> CallToolResult:
    """Execute a service action through the MCP success/error boundary."""

    try:
        value = action()
        if isinstance(value, OperationResult):
            value = serialize_operation_result(value)
        return _tool_result(value)
    except OperationCancelled:
        raise
    except Exception as exc:  # The MCP boundary must never leak tracebacks or secrets.
        return _tool_result(normalize_exception(exc).to_payload(), is_error=True)


class MCPToolInput(BaseModel):
    """Strict base model shared by all public MCP tool inputs."""

    model_config = ConfigDict(extra="forbid")


class ImageGenerateInput(MCPToolInput):
    prompt: str
    model: ImageGenerateModel = "zimage"
    image_size: ImageSize = "square"
    style: ImageStyle | None = None
    image: str | None = None
    lora_url: str | None = None
    controls: RecraftControls | None = None
    resolution: ImageResolution | None = None
    seed: int = -1
    callback_url: str | None = None
    execution: MCPExecutionOptions | None = None


class ImageEditInput(MCPToolInput):
    prompt: str
    images: ImageEditMedia
    model: ImageEditModel = "flux2"
    image_size: ImageSize | None = None
    resolution: ImageResolution | None = None
    megapixel_ratio: float = Field(default=1.0, ge=0.5, le=1.5)
    seed: int = -1
    callback_url: str | None = None
    execution: MCPExecutionOptions | None = None


class ImageUpscaleInput(MCPToolInput):
    image: str | None = None
    task_id: str | None = None
    model: str = "seedvr2"
    callback_url: str | None = None
    execution: MCPExecutionOptions | None = None


class ImageAnglesInput(MCPToolInput):
    image: str
    azimuth: str = "front"
    elevation: str = "eye_level"
    distance: str = "medium"
    prompt: str | None = None
    execution: MCPExecutionOptions | None = None


class ImageVectorizeInput(MCPToolInput):
    image: str
    callback_url: str | None = None
    execution: MCPExecutionOptions | None = None


class ImageColorsInput(MCPToolInput):
    image: str
    reference: str
    execution: MCPExecutionOptions | None = None


class VideoGenerateInput(MCPToolInput):
    prompt: str
    model: VideoGenerateModel = "ltx23"
    duration: VideoGenerateDuration | None = None
    resolution: VideoGenerateResolution | None = None
    aspect_ratio: VideoGenerateAspectRatio | None = None
    reference_images: list[str] | None = None
    reference_videos: list[str] | None = None
    image: str | None = None
    last_image: str | None = None
    seed: int | None = None
    generate_audio: bool | None = None
    callback_url: str | None = None
    execution: MCPExecutionOptions | None = None


class VideoEditInput(MCPToolInput):
    prompt: str | None = None
    model: VideoEditModel = "seedance2"
    images: list[str] | None = None
    video: str | None = None
    audio: str | None = None
    duration: VideoEditDuration | None = None
    resolution: VideoEditResolution | None = None
    aspect_ratio: VideoEditAspectRatio | None = None
    seed: int | None = None
    generate_audio: bool | None = None
    callback_url: str | None = None
    keyframes: list[dict[str, Any]] | None = None
    public_figure_threshold: PublicFigureThreshold | None = None
    execution: MCPExecutionOptions | None = None


class VideoUpscaleInput(MCPToolInput):
    video: str | None = None
    task_id: str | None = None
    resolution: VideoUpscaleResolution = "1080p"
    model: VideoUpscaleModel = "seedvr2"
    callback_url: str | None = None
    execution: MCPExecutionOptions | None = None


class LipsyncVideoInput(MCPToolInput):
    video: str
    audio: str
    resolution: LipsyncVideoResolution = "480p"
    prompt: str | None = None
    seed: int = -1
    callback_url: str | None = None
    execution: MCPExecutionOptions | None = None


class LipsyncImageInput(MCPToolInput):
    image: str
    audio: str
    model: LipsyncImageModel = "inftalk"
    resolution: LipsyncImageResolution = "480p"
    prompt: str | None = None
    seed: int = -1
    callback_url: str | None = None
    execution: MCPExecutionOptions | None = None


class AudioTranscribeInput(MCPToolInput):
    audio: str
    language: str = "auto"
    hints: str = ""
    denoise: bool = True
    execution: MCPExecutionOptions | None = None


class TTSCreateInput(MCPToolInput):
    text: str | None = None
    text_file: str | None = None
    speaker: str = "Ryan"
    style: str = "Auto"
    language: str = "Auto"
    prompt: str = ""
    seed: int = -1
    execution: MCPExecutionOptions | None = None


class TTSCloneInput(MCPToolInput):
    audio: str
    text: str | None = None
    text_file: str | None = None
    language: str = "Auto"
    seed: int = -1
    execution: MCPExecutionOptions | None = None


class TTSVoiceInput(MCPToolInput):
    text: str | None = None
    text_file: str | None = None
    character: str = "Female"
    style: str = "Auto"
    language: str = "Auto"
    prompt: str = ""
    seed: int = -1
    execution: MCPExecutionOptions | None = None


class TaskStatusInput(MCPToolInput):
    task_id: str


class TaskWaitInput(MCPToolInput):
    task_id: str
    execution: MCPExecutionOptions | None = None


class KeywordsInput(MCPToolInput):
    image: str
    lang: str = "en"
    num_keywords: int | None = None
    colors: bool = False


class ImageInput(MCPToolInput):
    image: str


class VideoInput(MCPToolInput):
    video: str


class RunInput(MCPToolInput):
    endpoint: str
    payload: dict[str, Any]
    method: str | None = None
    dry_run: bool = False
    help_schema: bool = False
    execution: MCPExecutionOptions | None = None


class NoInput(MCPToolInput):
    pass


@dataclass(frozen=True)
class ToolSpec:
    """One MCP contract and its transport-independent application handler."""

    name: str
    description: str
    input_model: type[MCPToolInput]
    annotations: ToolAnnotations
    handler: Callable[[Any], Any]

    def describe(self) -> Tool:
        return Tool(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(by_alias=True),
            annotations=self.annotations,
        )


class ToolRegistry:
    """List, validate, and invoke MCP tools through one explicit boundary."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        input_model: type[MCPToolInput],
        *,
        annotations: ToolAnnotations,
    ) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
        def register(handler: Callable[[Any], Any]) -> Callable[[Any], Any]:
            self._tools[handler.__name__] = ToolSpec(
                name=handler.__name__,
                description=handler.__doc__ or "",
                input_model=input_model,
                annotations=annotations,
                handler=handler,
            )
            return handler

        return register

    def list_tools(self) -> list[Tool]:
        return [spec.describe() for spec in self._tools.values()]

    async def call_tool(self, params: CallToolRequestParams) -> CallToolResult:
        spec = self._tools.get(params.name)
        if spec is None:
            return self._validation_error()
        try:
            arguments = spec.input_model.model_validate(params.arguments or {})
        except ValidationError as exc:
            return self._validation_error(exc)
        cancel_event = Event()

        def invoke() -> CallToolResult:
            context_token = _CURRENT_CANCEL_EVENT.set(cancel_event)
            try:
                if cancel_event.is_set():
                    raise OperationCancelled
                result = _invoke(lambda: spec.handler(arguments))
                if cancel_event.is_set():
                    raise OperationCancelled
                return result
            finally:
                _CURRENT_CANCEL_EVENT.reset(context_token)

        try:
            return await anyio.to_thread.run_sync(
                invoke,
                abandon_on_cancel=True,
            )
        finally:
            cancel_event.set()

    @staticmethod
    def _validation_error(exc: ValidationError | None = None) -> CallToolResult:
        message = (
            format_validation_errors(exc.errors())
            if exc is not None
            else "Invalid MCP tool arguments"
        )
        return _tool_result(
            ValidationCLIError(message).to_payload(),
            is_error=True,
        )


def _configured_services_factory(
    *,
    profile: str | None = None,
    dev: bool = False,
    base_url: str | None = None,
) -> tuple[ServiceFactory, bool]:
    settings = resolved_settings(profile=profile, dev=dev, base_url=base_url)

    client = APIClient(
        base_url=settings["base_url"],
        client_id=settings["client_id"],
        client_secret=settings["client_secret"],
    )
    configured_services = ApplicationServices.with_client(client)

    def services() -> ApplicationServices:
        return configured_services

    return services, settings["client_id"] is not None


def create_mcp_server(
    services_factory: ServiceFactory | None = None,
    *,
    client_id_present: bool | None = None,
) -> Server[Any]:
    """Create an Everypixel MCP server with injectable application services."""

    if services_factory is None:
        services_factory, configured_client_id = _configured_services_factory()
        if client_id_present is None:
            client_id_present = configured_client_id
    client_id_present = bool(client_id_present)

    registry = ToolRegistry()

    @registry.tool(ImageGenerateInput, annotations=LOCAL_WRITE)
    def image_generate(arguments: ImageGenerateInput) -> Any:
        """Generate an image from a prompt, optionally using a source image."""

        return services_factory().execute_image_generate(
            prompt=arguments.prompt,
            model=arguments.model,
            image_size=arguments.image_size,
            style=arguments.style,
            image=arguments.image,
            lora_url=arguments.lora_url,
            controls=(
                arguments.controls.model_dump(mode="json", exclude_none=True)
                if arguments.controls is not None
                else None
            ),
            resolution=arguments.resolution,
            seed=arguments.seed,
            callback_url=arguments.callback_url,
            execution=_execution(arguments.execution),
        )

    @registry.tool(ImageEditInput, annotations=LOCAL_WRITE)
    def image_edit(arguments: ImageEditInput) -> Any:
        """Edit one or more images using a text instruction."""

        return services_factory().execute_image_edit(
            prompt=arguments.prompt,
            images=arguments.images,
            model=arguments.model,
            image_size=arguments.image_size,
            resolution=arguments.resolution,
            megapixel_ratio=arguments.megapixel_ratio,
            seed=arguments.seed,
            callback_url=arguments.callback_url,
            execution=_execution(arguments.execution),
        )

    @registry.tool(ImageVectorizeInput, annotations=LOCAL_WRITE)
    def image_vectorize(arguments: ImageVectorizeInput) -> Any:
        """Convert an image URL or local file to SVG."""

        return services_factory().execute_image_vectorize(
            image=arguments.image,
            callback_url=arguments.callback_url,
            execution=_execution(arguments.execution),
        )

    @registry.tool(ImageUpscaleInput, annotations=LOCAL_WRITE)
    def image_upscale(arguments: ImageUpscaleInput) -> Any:
        """Upscale an image supplied directly or by a completed task ID."""

        return services_factory().execute_image_upscale(
            image=arguments.image,
            task_id=arguments.task_id,
            model=arguments.model,
            callback_url=arguments.callback_url,
            execution=_execution(arguments.execution),
        )

    @registry.tool(ImageAnglesInput, annotations=LOCAL_WRITE)
    def image_angles(arguments: ImageAnglesInput) -> Any:
        """Render an image from a different camera angle and distance."""

        return services_factory().execute_image_angles(
            image=arguments.image,
            azimuth=arguments.azimuth,
            elevation=arguments.elevation,
            distance=arguments.distance,
            prompt=arguments.prompt,
            execution=_execution(arguments.execution),
        )

    @registry.tool(ImageColorsInput, annotations=LOCAL_WRITE)
    def image_colors(arguments: ImageColorsInput) -> Any:
        """Transfer colors from a reference image to a source image."""

        return services_factory().execute_image_colors(
            image=arguments.image,
            reference=arguments.reference,
            execution=_execution(arguments.execution),
        )

    @registry.tool(VideoGenerateInput, annotations=LOCAL_WRITE)
    def video_generate(arguments: VideoGenerateInput) -> Any:
        """Generate video from text, an image, or first and last frame images."""

        return services_factory().execute_video_generate(
            prompt=arguments.prompt,
            model=arguments.model,
            duration=arguments.duration,
            resolution=arguments.resolution,
            aspect_ratio=arguments.aspect_ratio,
            reference_images=arguments.reference_images,
            reference_videos=arguments.reference_videos,
            image=arguments.image,
            last_image=arguments.last_image,
            seed=arguments.seed,
            generate_audio=arguments.generate_audio,
            callback_url=arguments.callback_url,
            execution=_execution(arguments.execution),
        )

    @registry.tool(VideoEditInput, annotations=LOCAL_WRITE)
    def video_edit(arguments: VideoEditInput) -> Any:
        """Edit a video with model-specific image, video, audio, or keyframe inputs."""

        encoded_keyframes = [
            json.dumps(keyframe, ensure_ascii=False)
            for keyframe in arguments.keyframes or []
        ]
        return services_factory().execute_video_edit(
            prompt=arguments.prompt,
            model=arguments.model,
            images=arguments.images or [],
            video=arguments.video,
            audio=arguments.audio,
            duration=arguments.duration,
            resolution=arguments.resolution,
            aspect_ratio=arguments.aspect_ratio,
            seed=arguments.seed,
            generate_audio=arguments.generate_audio,
            callback_url=arguments.callback_url,
            keyframes=encoded_keyframes,
            public_figure_threshold=arguments.public_figure_threshold,
            execution=_execution(arguments.execution),
        )

    @registry.tool(VideoUpscaleInput, annotations=LOCAL_WRITE)
    def video_upscale(arguments: VideoUpscaleInput) -> Any:
        """Upscale a video supplied directly or by a completed task ID."""

        return services_factory().execute_video_upscale(
            video=arguments.video,
            task_id=arguments.task_id,
            resolution=arguments.resolution,
            model=arguments.model,
            callback_url=arguments.callback_url,
            execution=_execution(arguments.execution),
        )

    @registry.tool(LipsyncVideoInput, annotations=LOCAL_WRITE)
    def lipsync_video(arguments: LipsyncVideoInput) -> Any:
        """Synchronize lips in a video to an audio track."""

        return services_factory().execute_lipsync_video(
            video=arguments.video,
            audio=arguments.audio,
            resolution=arguments.resolution,
            prompt=arguments.prompt,
            seed=arguments.seed,
            callback_url=arguments.callback_url,
            execution=_execution(arguments.execution),
        )

    @registry.tool(LipsyncImageInput, annotations=LOCAL_WRITE)
    def lipsync_image(arguments: LipsyncImageInput) -> Any:
        """Create a lipsync video from a still image and audio track."""

        return services_factory().execute_lipsync_image(
            image=arguments.image,
            audio=arguments.audio,
            model=arguments.model,
            resolution=arguments.resolution,
            prompt=arguments.prompt,
            seed=arguments.seed,
            callback_url=arguments.callback_url,
            execution=_execution(arguments.execution),
        )

    @registry.tool(AudioTranscribeInput, annotations=LOCAL_WRITE)
    def audio_transcribe(arguments: AudioTranscribeInput) -> Any:
        """Transcribe speech from an audio URL, data URI, or local file."""

        return services_factory().execute_audio_transcribe(
            audio=arguments.audio,
            language=arguments.language,
            hints=arguments.hints,
            denoise=arguments.denoise,
            execution=_execution(arguments.execution),
        )

    @registry.tool(TTSCreateInput, annotations=LOCAL_WRITE)
    def tts_create(arguments: TTSCreateInput) -> Any:
        """Create speech from text using a selected speaker."""

        return services_factory().execute_tts_create(
            text=arguments.text,
            text_file=Path(arguments.text_file) if arguments.text_file else None,
            speaker=arguments.speaker,
            style=arguments.style,
            language=arguments.language,
            prompt=arguments.prompt,
            seed=arguments.seed,
            execution=_execution(arguments.execution),
        )

    @registry.tool(TTSCloneInput, annotations=LOCAL_WRITE)
    def tts_clone(arguments: TTSCloneInput) -> Any:
        """Create speech using a cloned voice sample."""

        return services_factory().execute_tts_clone(
            audio=arguments.audio,
            text=arguments.text,
            text_file=Path(arguments.text_file) if arguments.text_file else None,
            language=arguments.language,
            seed=arguments.seed,
            execution=_execution(arguments.execution),
        )

    @registry.tool(TTSVoiceInput, annotations=LOCAL_WRITE)
    def tts_voice(arguments: TTSVoiceInput) -> Any:
        """Create speech from text using a character voice."""

        return services_factory().execute_tts_voice(
            text=arguments.text,
            text_file=Path(arguments.text_file) if arguments.text_file else None,
            character=arguments.character,
            style=arguments.style,
            language=arguments.language,
            prompt=arguments.prompt,
            seed=arguments.seed,
            execution=_execution(arguments.execution),
        )

    @registry.tool(TaskStatusInput, annotations=READ_ONLY)
    def task_status(arguments: TaskStatusInput) -> Any:
        """Fetch the current state of an asynchronous Everypixel task once."""

        return services_factory().get_task_status(
            task_id=arguments.task_id,
            execution=ExecutionOptions(
                cancel_event=_CURRENT_CANCEL_EVENT.get(),
            ),
        )

    @registry.tool(TaskWaitInput, annotations=LOCAL_WRITE)
    def task_wait(arguments: TaskWaitInput) -> Any:
        """Poll a task until SUCCESS and optionally download its result files."""

        return services_factory().wait_for_task(
            task_id=arguments.task_id,
            execution=_execution(
                arguments.execution,
                force_wait=True,
            ),
        )

    @registry.tool(KeywordsInput, annotations=READ_ONLY)
    def keywords(arguments: KeywordsInput) -> Any:
        """Extract semantic keywords and optional colors from an image."""

        return services_factory().execute_keywords(
            image=arguments.image,
            lang=arguments.lang,
            num_keywords=arguments.num_keywords,
            colors=arguments.colors,
        )

    @registry.tool(ImageInput, annotations=READ_ONLY)
    def quality(arguments: ImageInput) -> Any:
        """Score the technical quality of an image."""

        return services_factory().execute_quality(image=arguments.image)

    @registry.tool(ImageInput, annotations=READ_ONLY)
    def quality_ugc(arguments: ImageInput) -> Any:
        """Score the user-generated-content quality of an image."""

        return services_factory().execute_quality_ugc(image=arguments.image)

    @registry.tool(ImageInput, annotations=READ_ONLY)
    def faces(arguments: ImageInput) -> Any:
        """Detect faces and facial attributes in an image."""

        return services_factory().execute_faces(image=arguments.image)

    @registry.tool(ImageInput, annotations=READ_ONLY)
    def captioning(arguments: ImageInput) -> Any:
        """Generate a natural-language caption for an image."""

        return services_factory().execute_captioning(image=arguments.image)

    @registry.tool(VideoInput, annotations=READ_ONLY)
    def video_keywords(arguments: VideoInput) -> Any:
        """Extract semantic keywords from a video."""

        return services_factory().execute_video_keywords(video=arguments.video)

    @registry.tool(RunInput, annotations=LOCAL_WRITE)
    def run(arguments: RunInput) -> Any:
        """Run any Everypixel operation by OpenAPI name or direct /v1 path."""

        return services_factory().execute_generic(
            endpoint=arguments.endpoint,
            payload=arguments.payload,
            method=arguments.method,
            execution=_execution(arguments.execution),
            dry_run=arguments.dry_run,
            help_schema=arguments.help_schema,
            client_id_present=client_id_present,
        )

    @registry.tool(NoInput, annotations=READ_ONLY)
    def auth_check(_arguments: NoInput) -> Any:
        """Verify that the configured Everypixel credentials are accepted."""

        return services_factory().check_auth()

    @registry.tool(NoInput, annotations=LOCAL_WRITE)
    def openapi(_arguments: NoInput) -> Any:
        """Return the selected live, cached, or bundled Everypixel OpenAPI schema."""

        return services_factory().openapi()

    @registry.tool(NoInput, annotations=LOCAL_WRITE)
    def openapi_refresh(_arguments: NoInput) -> Any:
        """Refresh the local Everypixel OpenAPI cache from the live API."""

        result, path = services_factory().refresh_openapi()
        return {
            "schema": serialize_operation_result(result),
            "cache_path": str(path),
        }

    async def list_tools(_ctx: Any, _params: Any) -> ListToolsResult:
        return ListToolsResult(tools=registry.list_tools())

    async def call_tool(_ctx: Any, params: CallToolRequestParams) -> CallToolResult:
        return await registry.call_tool(params)

    return Server(
        "everypixel",
        title="Everypixel API",
        description="Generate and analyze media through the Everypixel API.",
        instructions=(
            "Generation tools wait for the completed result by default. Pass "
            "execution.wait=false to return the created task immediately, then "
            "use the task status or task wait tool. "
            "Media arguments accept HTTP(S) URLs, data URIs, and local file paths. "
            "Use English for natural-language prompts sent to generation tools; "
            "translate prompts when needed while preserving the user's intent and "
            "any text that must appear verbatim in the generated result."
        ),
        version=__version__,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def _serve_stdio(server: Server[Any]) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run_mcp_server(server: Server[Any]) -> None:
    """Run an Everypixel MCP server using the official stdio transport."""

    anyio.run(_serve_stdio, server)


def main() -> None:
    """Run the configured server over stdio without importing the CLI layer."""

    run_mcp_server(create_mcp_server())


if __name__ == "__main__":
    main()
