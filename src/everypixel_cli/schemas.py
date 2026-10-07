"""Pydantic payload schemas for common generation commands.

These models validate CLI parameters before a request is sent to the API:
media sources, model limits, enum values, and numeric ranges.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, get_args
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


ImageSize = Literal[
    "square",
    "portrait_3_2",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_3_2",
    "landscape_4_3",
    "landscape_16_9",
]
ImageStyle = Literal["portrait", "transparent"]
ImageResolution = Literal["1k", "2k", "3k", "4k"]
ImageQuality = Literal["low", "medium", "high", "xhigh", "max"]
GPTImageModel = Literal["gpt-image-2", "gpt-image-2.5-sunburst"]
RasterImageGenerateModel = Literal[
    GPTImageModel,
    "zimage",
    "wan2.7",
    "wan2.7-pro",
    "flux2",
    "grok",
    "grok-imagine",
    "grok-imagine-2",
    "grok-imagine-2-low",
    "gemini-3.1-flash",
    "gemini-3-pro",
    "seedream-5-pro",
    "seedream-5",
]
ImageEditModel = Literal[
    GPTImageModel,
    "flux2",
    "qwen",
    "wan2.7",
    "wan2.7-pro",
    "grok",
    "grok-imagine",
    "grok-imagine-2",
    "grok-imagine-2-low",
    "gemini-3.1-flash",
    "gemini-3-pro",
    "seedream-5-pro",
    "seedream-5",
]

_GEMINI_IMAGE_MODELS = {"gemini-3.1-flash", "gemini-3-pro"}
_GROK_IMAGE_MODELS = {"grok-imagine", "grok-imagine-2", "grok-imagine-2-low"}
RecraftVectorModel = Literal["recraftv4_1_vector", "recraftv4_1_pro_vector"]
ImageGenerateModel = Literal[RasterImageGenerateModel, RecraftVectorModel]
_WAN_IMAGE_MODELS = {"wan2.7", "wan2.7-pro"}
_SEEDREAM_IMAGE_MODELS = {"seedream-5-pro", "seedream-5"}
GPT_IMAGE_MODEL_QUALITIES = {
    "gpt-image-2": ("low", "medium", "high"),
    "gpt-image-2.5-sunburst": ("low", "medium", "high", "xhigh", "max"),
}
_IMAGE_EDIT_MODEL_MAX_IMAGES = {
    "flux2": 5,
    "qwen": 3,
    "wan2.7": 9,
    "wan2.7-pro": 9,
    "grok-imagine": 3,
    "grok-imagine-2": 3,
    "grok-imagine-2-low": 3,
    "gemini-3.1-flash": 14,
    "gemini-3-pro": 14,
    "seedream-5-pro": 10,
    "seedream-5": 14,
    "gpt-image-2": 16,
    "gpt-image-2.5-sunburst": 16,
}
ImageEditMedia = Annotated[
    list[str],
    Field(min_length=1, max_length=max(_IMAGE_EDIT_MODEL_MAX_IMAGES.values())),
]


class TaskResponse(BaseModel):
    """Standard API response when an async task is created."""

    task_id: str
    status: str = "PENDING"
    result: Any | None = None
    queue: int | None = None
    error: str | None = None


def normalize_grok_model(value: Any) -> Any:
    """Preserve public Grok aliases while sending canonical model names."""

    if isinstance(value, dict) and value.get("model") in ("grok", "grok15"):
        return {
            **value,
            "model": {"grok": "grok-imagine", "grok15": "grok-imagine-1.5"}[
                value["model"]
            ],
        }
    return value


def normalize_image_quality(value: Any) -> Any:
    """Apply quality defaults and validate the selected model's quality contract."""

    if not isinstance(value, dict):
        return value
    value = dict(value)
    model = value.get("model")
    if isinstance(model, str) and model in GPT_IMAGE_MODEL_QUALITIES:
        quality = value.get("quality", "medium")
        if quality not in GPT_IMAGE_MODEL_QUALITIES[model]:
            raise ValueError(f"quality {quality} is not supported by model {model}")
        value["quality"] = quality
        return value
    if "quality" in value:
        raise ValueError("quality is supported only by GPT Image models")
    return value


class RecraftColor(BaseModel):
    rgb: tuple[
        Annotated[int, Field(ge=0, le=255)],
        Annotated[int, Field(ge=0, le=255)],
        Annotated[int, Field(ge=0, le=255)],
    ]
    weight: float | None = Field(default=None, ge=0, le=1)


class RecraftControls(BaseModel):
    model_config = ConfigDict(extra="forbid")

    colors: list[RecraftColor] = Field(default_factory=list, max_length=12)
    background_color: RecraftColor | None = None

    @model_validator(mode="after")
    def validate_weights(self) -> "RecraftControls":
        if sum(color.weight or 0 for color in self.colors) > 1:
            raise ValueError("The sum of Recraft color weights must not exceed 1")
        return self


class RecraftGeneratePayload(BaseModel):
    """Vector generation uses size and palette controls instead of raster options."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=10_000)
    model: RecraftVectorModel
    image_size: ImageSize = "square"
    seed: int = Field(default=-1, ge=-1, le=4_294_967_295)
    controls: RecraftControls | None = None
    callback_url: HttpUrl | None = None


class ImageVectorizePayload(BaseModel):
    image_url: str
    callback_url: HttpUrl | None = None

    @model_validator(mode="after")
    def validate_image(self) -> "ImageVectorizePayload":
        validate_frame_inputs(self.image_url, None)
        return self


class ImageGeneratePayload(BaseModel):
    """Payload for image generation."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    model: RasterImageGenerateModel = "zimage"
    image_size: ImageSize = "square"
    style: ImageStyle | None = None
    resolution: ImageResolution = "1k"
    quality: ImageQuality | None = None
    seed: int = -1
    image_url: str | None = None
    lora_url: HttpUrl | None = None
    callback_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_model(cls, value: Any) -> Any:
        return normalize_image_quality(normalize_grok_model(value))

    @model_validator(mode="after")
    def validate_model_constraints(self) -> "ImageGeneratePayload":
        """Validate provider-specific styles, inputs, and resolutions."""

        style = self.style
        if style is None and self.model == "zimage":
            self.style = "portrait"
        if style is not None:
            expected_models = {
                "portrait": "zimage",
                "transparent": "flux2",
            }
            if self.model != expected_models[style]:
                raise ValueError(
                    f"Model {self.model} is not compatible with style {style}"
                )

        if self.model in _GROK_IMAGE_MODELS and self.resolution not in {
            "1k",
            "2k",
        }:
            raise ValueError(
                f"resolution {self.resolution} is not supported by model {self.model}"
            )
        if self.model in _WAN_IMAGE_MODELS:
            if not 1 <= len(self.prompt) <= 5000:
                raise ValueError(
                    "Wan 2.7 prompt must contain between 1 and 5000 characters"
                )
            if not -1 <= self.seed <= 2_147_483_647:
                raise ValueError("Wan 2.7 seed must be -1 or between 0 and 2147483647")
            if self.image_url is not None:
                raise ValueError(
                    "image is not supported for Wan 2.7 image generate; use image edit"
                )
            if "resolution" not in self.model_fields_set:
                self.resolution = "2k"
            supported_resolutions = {"1k", "2k"}
            if self.model == "wan2.7-pro":
                supported_resolutions.add("4k")
            if self.resolution not in supported_resolutions:
                raise ValueError(
                    f"resolution {self.resolution} is not supported by model "
                    f"{self.model}"
                )
        if self.model in _GEMINI_IMAGE_MODELS and self.resolution not in {
            "1k",
            "2k",
            "4k",
        }:
            raise ValueError(
                f"resolution {self.resolution} is not supported by model {self.model}"
            )
        if self.model in GPT_IMAGE_MODEL_QUALITIES:
            if self.image_url is not None:
                raise ValueError(
                    "image is not supported for GPT Image image generate; "
                    "use image edit"
                )
            if self.resolution not in {"1k", "2k", "3k"}:
                raise ValueError(
                    f"resolution {self.resolution} is not supported by model "
                    f"{self.model}"
                )
            if self.resolution == "3k" and self.image_size != "square":
                raise ValueError("GPT Image 3k supports only square image_size")
        if self.model in _SEEDREAM_IMAGE_MODELS:
            if self.image_url is not None:
                raise ValueError(
                    "image is not supported for Seedream image generate; use image edit"
                )
            if "resolution" not in self.model_fields_set:
                self.resolution = "2k"
            supported_resolutions_by_model = {
                "seedream-5-pro": {"1k", "2k"},
                "seedream-5": {"2k", "3k", "4k"},
            }
            if self.resolution not in supported_resolutions_by_model[self.model]:
                raise ValueError(
                    f"resolution {self.resolution} is not supported by model "
                    f"{self.model}"
                )
        return self


class ImageEditPayload(BaseModel):
    """Payload for image editing."""

    prompt: str
    image_urls: ImageEditMedia
    model: ImageEditModel = "flux2"
    image_size: ImageSize | None = None
    resolution: ImageResolution = "1k"
    quality: ImageQuality | None = None
    megapixel_ratio: float = Field(default=1.0, ge=0.5, le=1.5)
    seed: int = -1
    callback_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_model(cls, value: Any) -> Any:
        return normalize_image_quality(normalize_grok_model(value))

    @model_validator(mode="after")
    def validate_model_limits(self) -> "ImageEditPayload":
        """Validate image count limits for the selected model."""

        max_images = _IMAGE_EDIT_MODEL_MAX_IMAGES.get(self.model)
        if max_images is None:
            raise ValueError(f"image edit model {self.model} has no configured limit")
        if len(self.image_urls) > max_images:
            raise ValueError(f"{self.model} supports a maximum of {max_images} images")

        if self.model in _GROK_IMAGE_MODELS and self.resolution not in {
            "1k",
            "2k",
        }:
            raise ValueError(
                f"resolution {self.resolution} is not supported by model {self.model}"
            )
        if self.model in _WAN_IMAGE_MODELS:
            if not 1 <= len(self.prompt) <= 5000:
                raise ValueError(
                    "Wan 2.7 prompt must contain between 1 and 5000 characters"
                )
            if not -1 <= self.seed <= 2_147_483_647:
                raise ValueError("Wan 2.7 seed must be -1 or between 0 and 2147483647")
            if "resolution" not in self.model_fields_set:
                self.resolution = "2k"
            if self.resolution not in {"1k", "2k"}:
                raise ValueError(
                    f"resolution {self.resolution} is not supported by model "
                    f"{self.model}"
                )
        if self.model in _GEMINI_IMAGE_MODELS and self.resolution not in {
            "1k",
            "2k",
            "4k",
        }:
            raise ValueError(
                f"resolution {self.resolution} is not supported by model {self.model}"
            )
        if self.model in _SEEDREAM_IMAGE_MODELS:
            if "resolution" not in self.model_fields_set:
                self.resolution = "2k"
            supported = {
                "seedream-5-pro": {"1k", "2k"},
                "seedream-5": {"2k", "3k", "4k"},
            }
            if self.resolution not in supported[self.model]:
                raise ValueError(
                    f"resolution {self.resolution} is not supported by model "
                    f"{self.model}"
                )
        if self.model in GPT_IMAGE_MODEL_QUALITIES:
            if self.resolution not in {"1k", "2k", "3k"}:
                raise ValueError(
                    f"resolution {self.resolution} is not supported by model "
                    f"{self.model}"
                )
            if self.image_size is None:
                self.image_size = "square"
            if self.resolution == "3k" and self.image_size != "square":
                raise ValueError("GPT Image 3k supports only square image_size")
        return self


class _StrictChatPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ChatFunctionCall(_StrictChatPayload):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    arguments: str


class ChatToolCall(_StrictChatPayload):
    id: str = Field(min_length=1)
    type: Literal["function"]
    function: ChatFunctionCall


class ChatTextPart(_StrictChatPayload):
    type: Literal["text"]
    text: str


class ChatMessage(_StrictChatPayload):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[ChatTextPart] | None = None
    name: str | None = Field(
        default=None, min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"
    )
    reasoning_content: str | None = None
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def validate_role(self) -> "ChatMessage":
        if self.content is None and not (self.role == "assistant" and self.tool_calls):
            raise ValueError(
                "content is required unless the assistant returns tool_calls"
            )
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id belongs to tool messages")
        if self.role != "assistant" and (
            self.tool_calls is not None or self.reasoning_content is not None
        ):
            raise ValueError(
                "tool_calls and reasoning_content belong to assistant messages"
            )
        return self


class ChatFunction(_StrictChatPayload):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class ChatTool(_StrictChatPayload):
    type: Literal["function"]
    function: ChatFunction


class ChatThinking(_StrictChatPayload):
    type: Literal["enabled"] = "enabled"
    clear_thinking: bool | None = None


class ChatStreamOptions(_StrictChatPayload):
    include_usage: bool = False


class ChatResponseFormat(_StrictChatPayload):
    type: Literal["text", "json_object"]


class ChatRequest(_StrictChatPayload):
    """Public GLM chat contract shared by specialized, generic, and MCP callers."""

    model: Literal["glm-5.3"]
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: ChatStreamOptions | None = None
    tool_stream: bool = False
    tools: list[ChatTool] | None = Field(default=None, min_length=1, max_length=128)
    tool_choice: Literal["auto"] | None = None
    thinking: ChatThinking | None = None
    reasoning_effort: Literal["low", "high", "max"] | None = None
    temperature: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    top_p: float | None = Field(default=None, ge=0.01, le=1, allow_inf_nan=False)
    do_sample: bool | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=131072)
    stop: str | Annotated[list[str], Field(min_length=1, max_length=1)] | None = None
    response_format: ChatResponseFormat | None = None
    n: int = Field(default=1, ge=1, le=1)
    user: str | None = Field(default=None, min_length=6, max_length=128)

    @model_validator(mode="after")
    def validate_options(self) -> "ChatRequest":
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("Use only one of max_tokens and max_completion_tokens")
        if (self.tool_stream or self.stream_options is not None) and not self.stream:
            raise ValueError("tool_stream and stream_options require stream=true")
        if self.tool_choice is not None and not self.tools:
            raise ValueError("tool_choice requires tools")
        if not any(message.role in ("user", "tool") for message in self.messages):
            raise ValueError("messages must contain a user or tool message")
        return self


CommonVideoResolution = Literal["360p", "480p", "720p"]
CommonVideoAspectRatio = Literal["16:9", "9:16", "1:1"]
Seedance2VideoResolution = Literal["480p", "720p", "1080p", "4k"]
Seedance2VideoAspectRatio = Literal["16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
Wan27VideoResolution = Literal["720p", "1080p"]
Wan27VideoAspectRatio = Literal["16:9", "9:16", "1:1", "4:3", "3:4"]
VeoVideoResolution = Literal["720p", "1080p", "4k"]
VeoVideoAspectRatio = Literal["16:9", "9:16"]
Aleph2AspectRatio = Literal["16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "21:9"]
VideoGenerateModel = Literal[
    "minimax-h3-turbo",
    "minimax-h3",
    "ltx23",
    "grok",
    "grok15",
    "grok-imagine",
    "grok-imagine-1.5",
    "flux3",
    "veo-3.1",
    "veo-3.1-fast",
    "seedance2",
    "seedance2-mini",
    "seedance2.5",
    "kling-2.6",
    "kling-3",
    "kling-3-turbo",
    "kling-3-omni",
    "wan2.7",
    "wan3.0",
]
VideoEditModel = Literal[
    "seedance2",
    "seedance2-mini",
    "seedance2.5",
    "kling-3-omni",
    "wan2.7",
    "aleph2",
    "flux3",
]
VideoGenerateResolution = Literal["360p", "480p", "720p", "768p", "1080p", "4k"]
VideoEditResolution = Literal["480p", "720p", "1080p", "4k"]
VideoGenerateAspectRatio = Literal[
    "16:9",
    "4:3",
    "1:1",
    "3:4",
    "9:16",
    "21:9",
    "2:1",
    "7:4",
    "4:7",
    "adaptive",
]
Flux3VideoAspectRatio = Literal["21:9", "2:1", "16:9", "4:3", "1:1", "3:4", "9:16"]
VideoEditAspectRatio = Literal[Aleph2AspectRatio, Flux3VideoAspectRatio]
VideoGenerateDuration = Annotated[int, Field(ge=1, le=30)]
VideoEditDuration = Annotated[int, Field(ge=2, le=30)]
PublicFigureThreshold = Literal["auto", "low"]


class _StrictVideoPayload(BaseModel):
    """Reject fields that do not belong to the selected API model."""

    model_config = ConfigDict(extra="forbid")


class MiniMaxH3VideoGenRequest(_StrictVideoPayload):
    prompt: str = Field(min_length=1)
    model: Literal["minimax-h3-turbo", "minimax-h3"] = "minimax-h3-turbo"
    duration: int = Field(ge=3, le=15)
    resolution: Literal["768p"] = "768p"
    aspect_ratio: Literal["7:4", "4:7", "1:1"] = "7:4"
    seed: int = Field(default=-1, ge=-1, le=18_446_744_073_709_551_615)
    image_url: str | None = None
    image_last_url: str | None = None
    reference_image_urls: list[str] = Field(default_factory=list, max_length=9)
    reference_video_urls: list[str] = Field(default_factory=list, max_length=3)
    reference_audio_urls: list[str] = Field(default_factory=list, max_length=3)
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_media(self) -> "MiniMaxH3VideoGenRequest":
        if not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        validate_frame_inputs(self.image_url, self.image_last_url)
        validate_reference_inputs(self.reference_image_urls, [])
        for media_type, values in (
            ("video", self.reference_video_urls),
            ("audio", self.reference_audio_urls),
        ):
            for value in values:
                if value.startswith("data:"):
                    if not re.fullmatch(
                        rf"data:{media_type}/[a-zA-Z0-9.+\-]+;base64,[A-Za-z0-9+/]+=*",
                        value,
                    ):
                        raise ValueError(
                            f"reference_{media_type}_urls requires {media_type} "
                            "base64 data URIs"
                        )
                else:
                    HttpUrl(value)
        references = (
            self.reference_image_urls
            or self.reference_video_urls
            or self.reference_audio_urls
        )
        if references and (self.image_url or self.image_last_url):
            raise ValueError("Frame images and reference media cannot be combined")
        return self


class LTX23VideoGenRequest(_StrictVideoPayload):
    prompt: str
    model: Literal["ltx23"] = "ltx23"
    duration: int = Field(ge=1, le=10)
    resolution: CommonVideoResolution = "720p"
    aspect_ratio: CommonVideoAspectRatio = "16:9"
    image_url: str | None = None
    image_last_url: str | None = None
    seed: int = -1
    callback_url: str | None = None


class GrokVideoGenRequest(_StrictVideoPayload):
    prompt: str
    model: Literal["grok-imagine"] = "grok-imagine"
    duration: int = Field(ge=1, le=15)
    resolution: Literal["480p", "720p"] = "480p"
    aspect_ratio: CommonVideoAspectRatio = "16:9"
    image_url: str | None = None
    callback_url: str | None = None


class Grok15VideoGenRequest(_StrictVideoPayload):
    prompt: str
    model: Literal["grok-imagine-1.5"] = "grok-imagine-1.5"
    duration: int = Field(ge=1, le=15)
    resolution: Literal["480p", "720p"] = "480p"
    aspect_ratio: CommonVideoAspectRatio = "16:9"
    image_url: str = Field(min_length=1)
    callback_url: str | None = None


class VeoVideoGenRequest(_StrictVideoPayload):
    prompt: str = Field(min_length=1)
    model: Literal["veo-3.1", "veo-3.1-fast"]
    duration: Literal[4, 6, 8] = 8
    resolution: VeoVideoResolution = "720p"
    aspect_ratio: VeoVideoAspectRatio = "16:9"
    image_url: str | None = None
    image_last_url: str | None = None
    reference_image_urls: list[str] = Field(default_factory=list, max_length=3)
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_media_and_duration(self) -> "VeoVideoGenRequest":
        for value in (self.image_url, self.image_last_url):
            if value is not None and not value.startswith(
                ("http://", "https://", "data:image/")
            ):
                raise ValueError(
                    "Veo frame input must be an HTTP(S) URL or image data URI"
                )
        for value in self.reference_image_urls:
            if not value.startswith(("http://", "https://", "data:image/")):
                raise ValueError(
                    "Veo reference image must be an HTTP(S) URL or image data URI"
                )
        if self.resolution != "720p" and self.duration != 8:
            raise ValueError("Veo 1080p and 4k generation requires duration 8")
        if self.image_last_url and not self.image_url:
            raise ValueError("Veo image_last_url requires image_url")
        if self.reference_image_urls and (self.image_url or self.image_last_url):
            raise ValueError(
                "Veo reference_image_urls cannot be combined with frame images"
            )
        if self.reference_image_urls and self.duration != 8:
            raise ValueError("Veo reference images require duration 8")
        return self


class Seedance2VideoGenRequest(_StrictVideoPayload):
    prompt: str
    model: Literal["seedance2", "seedance2-mini", "seedance2.5"] = "seedance2"
    duration: int = Field(default=5, ge=4, le=30)
    resolution: Seedance2VideoResolution = "720p"
    aspect_ratio: Seedance2VideoAspectRatio = "16:9"
    generate_audio: bool = True
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_model_resolution(self) -> "Seedance2VideoGenRequest":
        if self.model != "seedance2.5" and self.duration > 15:
            raise ValueError(f"{self.model} supports duration from 4 to 15 seconds")
        if self.model == "seedance2.5" and self.resolution == "4k":
            raise ValueError("seedance2.5 supports only 480p, 720p, and 1080p")
        if self.model == "seedance2-mini" and self.resolution not in {
            "480p",
            "720p",
        }:
            raise ValueError("seedance2-mini supports only 480p and 720p")
        return self


class KlingV26VideoGenRequest(_StrictVideoPayload):
    prompt: str
    model: Literal["kling-2.6"] = "kling-2.6"
    duration: Literal[5, 10] = 5
    resolution: Literal["720p", "1080p"] = "1080p"
    aspect_ratio: CommonVideoAspectRatio = "16:9"
    generate_audio: bool = True
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_audio_resolution(self) -> "KlingV26VideoGenRequest":
        if self.generate_audio and self.resolution != "1080p":
            raise ValueError("kling-2.6 audio generation is supported only at 1080p")
        return self


class KlingV3VideoGenRequest(_StrictVideoPayload):
    prompt: str
    model: Literal["kling-3"] = "kling-3"
    duration: int = Field(default=5, ge=3, le=15)
    resolution: Literal["720p", "1080p", "4k"] = "1080p"
    aspect_ratio: CommonVideoAspectRatio = "16:9"
    generate_audio: bool = True
    callback_url: str | None = None


class KlingV3TurboVideoGenRequest(_StrictVideoPayload):
    prompt: str
    model: Literal["kling-3-turbo"] = "kling-3-turbo"
    duration: int = Field(default=5, ge=3, le=15)
    resolution: Literal["720p", "1080p"] = "720p"
    aspect_ratio: CommonVideoAspectRatio = "16:9"
    generate_audio: Literal[False] = False
    callback_url: str | None = None


class KlingV3OmniVideoGenRequest(_StrictVideoPayload):
    prompt: str
    model: Literal["kling-3-omni"] = "kling-3-omni"
    duration: int = Field(default=5, ge=3, le=15)
    resolution: Literal["720p", "1080p", "4k"] = "1080p"
    aspect_ratio: CommonVideoAspectRatio = "16:9"
    generate_audio: bool = True
    callback_url: str | None = None


class Wan27VideoGenRequest(_StrictVideoPayload):
    prompt: str = Field(min_length=1, max_length=5000)
    model: Literal["wan2.7"] = "wan2.7"
    duration: int = Field(default=5, ge=2, le=15)
    resolution: Wan27VideoResolution = "720p"
    aspect_ratio: Wan27VideoAspectRatio = "16:9"
    image_url: str | None = None
    image_last_url: str | None = None
    reference_image_urls: list[str] = Field(default_factory=list, max_length=5)
    reference_video_urls: list[str] = Field(default_factory=list, max_length=3)
    seed: int = Field(default=-1, ge=-1, le=2_147_483_647)
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_media(self) -> "Wan27VideoGenRequest":
        for value in (self.image_url, self.image_last_url):
            if value is not None and not value.startswith(
                ("http://", "https://", "data:image/")
            ):
                raise ValueError(
                    "Wan 2.7 frame input must be an HTTP(S) URL or image data URI"
                )
        for value in self.reference_image_urls:
            if not value.startswith(("http://", "https://", "data:image/")):
                raise ValueError(
                    "Wan 2.7 reference image must be an HTTP(S) URL or image data URI"
                )
        for value in self.reference_video_urls:
            if not value.startswith(
                ("http://", "https://", "data:video/mp4;", "data:video/quicktime;")
            ):
                raise ValueError(
                    "Wan 2.7 reference video must be an HTTP(S) URL or MP4/MOV data URI"
                )
        if self.image_last_url and not self.image_url:
            raise ValueError("image_last_url requires image_url")
        if (self.image_url or self.image_last_url) and (
            self.reference_image_urls or self.reference_video_urls
        ):
            raise ValueError(
                "Wan 2.7 frame inputs cannot be combined with reference inputs"
            )
        reference_count = len(self.reference_image_urls) + len(
            self.reference_video_urls
        )
        if reference_count > 5:
            raise ValueError(
                "Wan 2.7 supports at most 5 reference images and videos in total"
            )
        if self.reference_video_urls and self.duration > 10:
            raise ValueError(
                "Wan 2.7 requests with reference videos support up to 10 seconds"
            )
        return self


def validate_frame_inputs(image: str | None, last_image: str | None) -> None:
    for value in (image, last_image):
        if value is not None and not value.startswith(
            ("http://", "https://", "data:image/")
        ):
            raise ValueError("Frame input must be an HTTP(S) URL or image data URI")
    if last_image and not image:
        raise ValueError("image_last_url requires image_url")


def validate_reference_inputs(images: list[str], videos: list[str]) -> None:
    for value in images:
        validate_frame_inputs(value, None)
    for value in videos:
        validate_provider_video(value)


def validate_provider_video(value: str) -> str:
    if not value.startswith(
        ("http://", "https://", "data:video/mp4;", "data:video/quicktime;")
    ):
        raise ValueError("Video input must be an HTTP(S) URL or MP4/MOV data URI")
    return value


class Wan30VideoGenRequest(_StrictVideoPayload):
    prompt: str = Field(min_length=1, max_length=20_000)
    model: Literal["wan3.0"] = "wan3.0"
    duration: int = Field(default=5, ge=2, le=30)
    resolution: Literal["480p", "720p", "1080p"] = "1080p"
    aspect_ratio: Literal["adaptive", Wan27VideoAspectRatio] = "adaptive"
    image_url: str | None = None
    image_last_url: str | None = None
    reference_image_urls: list[str] = Field(default_factory=list, max_length=10)
    reference_video_urls: list[str] = Field(default_factory=list, max_length=5)
    generate_audio: bool = True
    seed: int = Field(default=-1, ge=-1, le=2_147_483_647)
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_media(self) -> "Wan30VideoGenRequest":
        validate_frame_inputs(self.image_url, self.image_last_url)
        validate_reference_inputs(self.reference_image_urls, self.reference_video_urls)
        if (self.image_url or self.image_last_url) and (
            self.reference_image_urls or self.reference_video_urls
        ):
            raise ValueError("Wan 3.0 frame inputs cannot be combined with references")
        return self


class Flux3VideoGenRequest(_StrictVideoPayload):
    prompt: str = Field(min_length=1, max_length=5000)
    model: Literal["flux3"] = "flux3"
    duration: int = Field(default=5, ge=5, le=20)
    resolution: Literal["720p", "1080p"] = "720p"
    aspect_ratio: Flux3VideoAspectRatio = "16:9"
    image_url: str | None = None
    image_last_url: str | None = None
    generate_audio: bool = True
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_frames(self) -> "Flux3VideoGenRequest":
        validate_frame_inputs(self.image_url, self.image_last_url)
        return self


class Flux3VideoEditPayload(_StrictVideoPayload):
    """Continue a source video using FLUX.3."""

    prompt: str = Field(min_length=1, max_length=5000)
    model: Literal["flux3"] = "flux3"
    duration: int = Field(default=5, ge=5, le=15)
    resolution: Literal["720p", "1080p"] = "720p"
    aspect_ratio: Flux3VideoAspectRatio = "16:9"
    video_url: Annotated[str, AfterValidator(validate_provider_video)]
    generate_audio: bool = True
    callback_url: str | None = None


VideoGenerateRequest = Annotated[
    MiniMaxH3VideoGenRequest
    | LTX23VideoGenRequest
    | GrokVideoGenRequest
    | Grok15VideoGenRequest
    | VeoVideoGenRequest
    | Seedance2VideoGenRequest
    | KlingV26VideoGenRequest
    | KlingV3VideoGenRequest
    | KlingV3TurboVideoGenRequest
    | KlingV3OmniVideoGenRequest
    | Wan27VideoGenRequest
    | Wan30VideoGenRequest
    | Flux3VideoGenRequest,
    Discriminator("model"),
    BeforeValidator(normalize_grok_model),
]


class VideoEditPayload(Seedance2VideoGenRequest):
    """JSON request body for Seedance ``video_edit`` operations."""

    content: list[dict[str, Any]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_content(self) -> "VideoEditPayload":
        """Mirror the documented structural validation before making a request."""

        video_count = 0
        has_visual = False
        has_audio = False
        for item in self.content:
            item_type = item.get("type")
            if item_type == "text":
                if not isinstance(item.get("text"), str) or not item["text"]:
                    raise ValueError("text content item requires text")
                continue
            if item_type not in {"image_url", "video_url", "audio_url"}:
                raise ValueError("content item type is not supported")
            field = item_type
            media = item.get(field)
            if not isinstance(media, dict) or not isinstance(media.get("url"), str):
                raise ValueError(f"{field} content item requires a URL")
            url = media["url"]
            expected_role = f"reference_{item_type.removesuffix('_url')}"
            if item.get("role") != expected_role:
                raise ValueError(f"{field} content item has an invalid role")
            if item_type == "image_url" and not url.startswith(
                ("http://", "https://", "data:image/")
            ):
                raise ValueError("image_url must be an HTTP(S) URL or image data URI")
            if item_type == "video_url":
                if not url.startswith(
                    ("http://", "https://", "data:video/mp4;", "data:video/quicktime;")
                ):
                    raise ValueError(
                        "video_url must be an HTTP(S) URL or MP4/MOV data URI"
                    )
                video_count += 1
                has_visual = True
            if item_type == "audio_url":
                if not url.startswith(("http://", "https://", "data:audio/")):
                    raise ValueError(
                        "audio_url must be an HTTP(S) URL or audio data URI"
                    )
                has_audio = True
            if item_type == "image_url":
                has_visual = True
        if video_count > 1:
            raise ValueError("content supports at most one video_url item")
        if not has_visual and not has_audio:
            raise ValueError(
                "video_edit requires at least one image_url, video_url, "
                "or audio_url item"
            )
        if has_audio and not has_visual and self.model != "seedance2.5":
            raise ValueError(
                "audio_url requires at least one image_url or video_url item"
            )
        return self


class KlingV3OmniVideoEditPayload(_StrictVideoPayload):
    """JSON request body for Kling 3 Omni video-to-video editing."""

    prompt: str
    model: Literal["kling-3-omni"] = "kling-3-omni"
    duration: int = Field(default=5, ge=3, le=10)
    resolution: Literal["720p", "1080p", "4k"] = "1080p"
    aspect_ratio: CommonVideoAspectRatio = "16:9"
    generate_audio: Literal[False] = False
    callback_url: str | None = None
    content: list[dict[str, Any]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_content(self) -> "KlingV3OmniVideoEditPayload":
        """Require the provider's single video reference and content shapes."""

        video_count = 0
        for item in self.content:
            item_type = item.get("type")
            if item_type == "text":
                if not isinstance(item.get("text"), str) or not item["text"]:
                    raise ValueError("text content item requires text")
                continue
            if item_type not in {"image_url", "video_url"}:
                raise ValueError("content item type is not supported")
            media = item.get(item_type)
            if not isinstance(media, dict) or not isinstance(media.get("url"), str):
                raise ValueError(f"{item_type} content item requires a URL")
            url = media["url"]
            if item_type == "image_url" and not url.startswith(
                ("http://", "https://", "data:image/")
            ):
                raise ValueError("image_url must be an HTTP(S) URL or image data URI")
            if item_type == "video_url":
                if not url.startswith(
                    ("http://", "https://", "data:video/mp4;", "data:video/quicktime;")
                ):
                    raise ValueError(
                        "video_url must be an HTTP(S) URL or MP4/MOV data URI"
                    )
                video_count += 1
        if video_count != 1:
            raise ValueError(
                "kling-3-omni video edit requires exactly one video reference"
            )
        if self.resolution == "4k":
            raise ValueError(
                "kling-3-omni video edit does not support 4k with video input"
            )
        return self


class Wan27VideoEditPayload(_StrictVideoPayload):
    """JSON request body for Wan 2.7 video editing."""

    prompt: str = Field(min_length=1, max_length=5000)
    model: Literal["wan2.7"] = "wan2.7"
    duration: int = Field(default=5, ge=2, le=10)
    resolution: Wan27VideoResolution = "720p"
    aspect_ratio: Wan27VideoAspectRatio = "16:9"
    video_url: str
    reference_image_urls: list[str] = Field(default_factory=list, max_length=3)
    seed: int = Field(default=-1, ge=-1, le=2_147_483_647)
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_media(self) -> "Wan27VideoEditPayload":
        if not self.video_url.startswith(
            ("http://", "https://", "data:video/mp4;", "data:video/quicktime;")
        ):
            raise ValueError("Wan 2.7 video must be an HTTP(S) URL or MP4/MOV data URI")
        for value in self.reference_image_urls:
            if not value.startswith(("http://", "https://", "data:image/")):
                raise ValueError(
                    "Wan 2.7 reference image must be an HTTP(S) URL or image data URI"
                )
        return self


class Aleph2KeyframeRange(BaseModel):
    start_seconds: int = Field(ge=0)
    end_seconds: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "Aleph2KeyframeRange":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class Aleph2Keyframe(BaseModel):
    image_url: str
    seconds: float | None = Field(default=None, ge=0, le=30)
    at: float | None = Field(default=None, ge=0, le=1)
    range: Aleph2KeyframeRange | None = None

    @model_validator(mode="after")
    def validate_keyframe(self) -> "Aleph2Keyframe":
        if not self.image_url.startswith(("https://", "data:image/")):
            raise ValueError(
                "Aleph 2 keyframe image_url must use HTTPS or an image data URI"
            )
        if len(self.image_url) > 2048 and not self.image_url.startswith("data:"):
            raise ValueError("Aleph 2 keyframe image_url exceeds 2048 characters")
        if (self.seconds is None) == (self.at is None):
            raise ValueError("Aleph 2 keyframe requires exactly one of seconds or at")
        return self


class Aleph2VideoEditPayload(_StrictVideoPayload):
    model: Literal["aleph2"] = "aleph2"
    video_url: str
    prompt: str | None = Field(default=None, min_length=1)
    keyframes: list[Aleph2Keyframe] = Field(default_factory=list, max_length=5)
    seed: int = Field(default=-1, ge=-1, le=4_294_967_295)
    target_aspect_ratio: Aleph2AspectRatio | None = None
    public_figure_threshold: PublicFigureThreshold | None = None
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_request(self) -> "Aleph2VideoEditPayload":
        if not self.video_url.startswith(
            ("http://", "https://", "data:video/mp4;", "data:video/quicktime;")
        ):
            raise ValueError("Aleph 2 video must be an HTTP(S) URL or MP4/MOV data URI")
        if self.prompt is not None:
            code_units = len(self.prompt.encode("utf-16-le")) // 2
            if code_units > 1000:
                raise ValueError(
                    "Aleph 2 prompt must not exceed 1000 UTF-16 code units"
                )
        ranged = [keyframe.range is not None for keyframe in self.keyframes]
        if ranged and any(ranged) and not all(ranged):
            raise ValueError("All Aleph 2 keyframes must either set range or omit it")
        return self


VideoEditRequest = Annotated[
    VideoEditPayload
    | KlingV3OmniVideoEditPayload
    | Wan27VideoEditPayload
    | Aleph2VideoEditPayload
    | Flux3VideoEditPayload,
    Discriminator("model"),
]


class ImageUpscalePayload(BaseModel):
    """Payload for image upscale from URL or another task result."""

    image_url: str | None = None
    image_from_task_id: str | None = None
    model: str = "seedvr2"
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "ImageUpscalePayload":
        """Require exactly one source: --image or --task-id."""

        if bool(self.image_url) == bool(self.image_from_task_id):
            raise ValueError("Provide exactly one of --image or --task-id")
        return self


VideoUpscaleResolution = Literal["720p", "1080p", "1440p", "4k"]
VideoUpscaleModel = Literal["seedvr2", "topaz-prob-4", "topaz-slp-2.5", "topaz-ast-2"]
LipsyncVideoResolution = Literal["360p", "480p", "720p"]
LipsyncImageModel = Literal["inftalk", "ltx23"]
LipsyncImageResolution = Literal["360p", "480p", "720p"]


class VideoUpscalePayload(BaseModel):
    """Payload for video upscale from URL or another task result."""

    video_url: str | None = None
    video_from_task_id: str | None = None
    model: VideoUpscaleModel = "seedvr2"
    callback_url: str | None = None
    resolution: VideoUpscaleResolution = Field(
        default="1080p",
        description=(
            "Output resolution; SeedVR2 source limits: "
            "20s at 1080p, 10s at 1440p, inclusive."
        ),
    )

    @model_validator(mode="after")
    def validate_source(self) -> "VideoUpscalePayload":
        """Require exactly one source: --video or --task-id."""

        if bool(self.video_url) == bool(self.video_from_task_id):
            raise ValueError("Provide exactly one of --video or --task-id")
        if self.model == "seedvr2" and self.resolution == "4k":
            raise ValueError("seedvr2 video_upscale supports resolution up to 1440p")
        return self


class LipsyncVideoPayload(BaseModel):
    """Payload for lipsync from source video and audio."""

    video_url: str
    audio_url: str
    resolution: LipsyncVideoResolution = "480p"
    prompt: str | None = None
    seed: int = -1
    callback_url: str | None = None


class LipsyncImagePayload(BaseModel):
    """Payload for lipsync from a still image and audio."""

    image_url: str
    audio_url: str
    model: LipsyncImageModel = "inftalk"
    resolution: LipsyncImageResolution = "480p"
    prompt: str | None = None
    seed: int = -1
    callback_url: str | None = None


TTSSpeaker = Literal[
    "Aiden",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Ryan",
    "Serena",
    "Sohee",
    "Uncle_Fu",
    "Vivian",
]


TTSStyle = Literal[
    "Auto",
    "Warm",
    "Gentle",
    "Calm",
    "Cheerful",
    "Friendly",
    "Serious",
    "Sad",
    "Angry",
    "Excited",
    "Soft",
    "Deep",
    "Clear",
    "Emotional",
    "Dramatic",
    "Whisper",
    "Breathy",
    "Husky",
    "Authoritative",
    "Storytelling",
    "News Anchor",
    "Documentary",
    "Customer Support",
    "Teacher",
    "Audiobook",
    "Energetic",
    "Relaxed",
    "Playful",
    "Mysterious",
    "Romantic",
    "Inspirational",
    "Formal",
    "Casual",
    "ASMR",
    "Noir",
    "Cinematic",
    "Trailer",
    "Motivational",
    "Robotic",
    "Vintage Radio",
    "Lullaby",
    "Comedy",
    "Interview",
    "Poetic",
    "Philosophical",
    "Sportscaster",
    "Meditation",
]


TTSCharacter = Literal[
    "Auto",
    "Female",
    "Male",
    "Young Female",
    "Young Male",
    "Girl",
    "Boy",
    "Child",
    "Teen",
    "Adult",
    "Senior Female",
    "Senior Male",
    "Narrator",
    "Announcer",
]


TTSLanguage = Literal[
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
    "Italian",
]


TTSCreateModel = Literal[
    "qwen3",
    "eleven_v4",
    "eleven_v4_turbo",
    "eleven_v3",
    "eleven_v3_conversational",
    "eleven_multilingual_v2",
    "eleven_flash_v2_5",
]
TTSDesignModel = Literal["qwen3", "eleven_multilingual_ttv_v2", "eleven_ttv_v3"]
TTSProvider = Literal["labs", "elevenlabs"]


class _TTSPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    callback_url: HttpUrl | None = None


class _QwenTTSPayload(_TTSPayload):
    text: str = Field(min_length=1, max_length=200)
    language: TTSLanguage = "Auto"
    seed: int = -1

    @field_validator(
        "language", "speaker", "style", "character", mode="before", check_fields=False
    )
    @classmethod
    def normalize_enum(cls, value: Any, info: Any) -> Any:
        choices = {
            "language": TTSLanguage,
            "speaker": TTSSpeaker,
            "style": TTSStyle,
            "character": TTSCharacter,
        }
        for choice in get_args(choices[info.field_name]):
            if str(value).lower() in {choice.lower(), choice.lower().replace(" ", "_")}:
                return choice
        return value


class TTSDefaultCreatePayload(_TTSPayload):
    """Validate shared inputs while the API chooses the model and its defaults."""

    text: str = Field(min_length=1)
    voice_id: UUID | None = None
    speaker: str | None = None
    style: str | None = None
    language: str | None = None
    prompt: str | None = None
    seed: int | None = None


class QwenTTSCreatePayload(_QwenTTSPayload):
    model: Literal["qwen3"] = "qwen3"
    speaker: TTSSpeaker = "Ryan"
    style: TTSStyle = "Auto"
    prompt: str = Field(default="", max_length=1000)


class ElevenLabsTTSCreatePayload(_TTSPayload):
    model: Literal[
        "eleven_v4",
        "eleven_v4_turbo",
        "eleven_v3",
        "eleven_v3_conversational",
        "eleven_multilingual_v2",
        "eleven_flash_v2_5",
    ]
    text: str = Field(min_length=1, max_length=2048)
    voice_id: UUID


def validate_tts_media(value: str) -> str:
    if value.startswith(("http://", "https://")):
        return value
    if re.fullmatch(
        r"data:(audio|video)/[a-zA-Z0-9.+\-]+;base64,[A-Za-z0-9+/]+=*", value
    ):
        return value
    raise ValueError("Audio must be an HTTP(S) URL or a base64 media data URI")


class TTSClonePayload(_QwenTTSPayload):
    model: Literal["qwen3"]
    audio_url: str | None = None
    voice_id: UUID | None = None

    @field_validator("audio_url")
    @classmethod
    def validate_audio(cls, value: str | None) -> str | None:
        return validate_tts_media(value) if value is not None else None

    @model_validator(mode="after")
    def validate_reference(self) -> "TTSClonePayload":
        if (self.audio_url is None) == (self.voice_id is None):
            raise ValueError("Provide exactly one of --audio or --voice-id")
        return self


class QwenTTSDesignPayload(_QwenTTSPayload):
    model: Literal["qwen3"]
    character: TTSCharacter = "Female"
    style: TTSStyle = "Auto"
    prompt: str = Field(default="", max_length=1000)


class ElevenLabsTTSDesignPayload(_TTSPayload):
    model: Literal["eleven_multilingual_ttv_v2", "eleven_ttv_v3"]
    text: str = Field(min_length=100, max_length=1000)
    prompt: str = Field(min_length=20, max_length=1000)


TTS_CREATE_PAYLOADS: dict[str, type[BaseModel]] = {
    "qwen3": QwenTTSCreatePayload,
    **dict.fromkeys(
        (
            "eleven_v4",
            "eleven_v4_turbo",
            "eleven_v3",
            "eleven_v3_conversational",
            "eleven_multilingual_v2",
            "eleven_flash_v2_5",
        ),
        ElevenLabsTTSCreatePayload,
    ),
}
TTS_DESIGN_PAYLOADS: dict[str, type[BaseModel]] = {
    "qwen3": QwenTTSDesignPayload,
    "eleven_multilingual_ttv_v2": ElevenLabsTTSDesignPayload,
    "eleven_ttv_v3": ElevenLabsTTSDesignPayload,
}


class TTSVoiceSavePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)
    provider: TTSProvider | None = "labs"
    audio_urls: list[str] | None = Field(default=None, min_length=1, max_length=10)
    preview_id: UUID | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Voice name cannot be blank")
        return value.strip()

    @field_validator("audio_urls")
    @classmethod
    def validate_samples(cls, value: list[str] | None) -> list[str] | None:
        if value is not None:
            for url in value:
                validate_tts_media(url)
                if not url.startswith("data:") and len(url) > 2048:
                    raise ValueError("Sample URL must contain at most 2048 characters")
        return value

    @model_validator(mode="after")
    def validate_source(self) -> "TTSVoiceSavePayload":
        if self.preview_id is not None:
            if (
                self.provider is not None and "provider" in self.model_fields_set
            ) or self.audio_urls is not None:
                raise ValueError("Provide --preview-id without --provider or --audio")
            self.provider = None
        elif self.provider is None or self.audio_urls is None:
            raise ValueError("Provide --preview-id or audio samples")
        elif self.provider == "labs" and len(self.audio_urls) != 1:
            raise ValueError("Qwen requires exactly one audio reference")
        return self


class TTSVoicesPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: TTSProvider = "labs"
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)


class TTSVoiceDeletePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    voice_id: UUID
