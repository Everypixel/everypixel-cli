"""Pydantic payload schemas for common generation commands.

These models validate CLI parameters before a request is sent to the API:
media sources, model limits, enum values, and numeric ranges.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
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
ImageStyle = Literal[
    "instagram",
    "transparent",
    "replication",
    "basic",
]
ImageResolution = Literal["1k", "2k", "3k", "4k"]
ImageGenerateModel = Literal[
    "zimage",
    "wan22",
    "wan2.7",
    "wan2.7-pro",
    "flux2",
    "grok",
    "grok_quality",
    "gemini-3.1-flash",
    "gemini-3-pro",
    "seedream-5-pro",
    "seedream-5",
    "gpt-image-2-low",
    "gpt-image-2-medium",
    "gpt-image-2-high",
]
ImageEditModel = Literal[
    "flux2",
    "qwen",
    "wan2.7",
    "wan2.7-pro",
    "grok",
    "grok_quality",
    "gemini-3.1-flash",
    "gemini-3-pro",
    "seedream-5-pro",
    "seedream-5",
    "gpt-image-2-low",
    "gpt-image-2-medium",
    "gpt-image-2-high",
]

_GEMINI_IMAGE_MODELS = {"gemini-3.1-flash", "gemini-3-pro"}
_WAN_IMAGE_MODELS = {"wan2.7", "wan2.7-pro"}
_SEEDREAM_IMAGE_MODELS = {"seedream-5-pro", "seedream-5"}
_GPT_IMAGE_2_MODELS = {
    "gpt-image-2-low",
    "gpt-image-2-medium",
    "gpt-image-2-high",
}
_IMAGE_EDIT_MODEL_MAX_IMAGES = {
    "flux2": 5,
    "qwen": 3,
    "wan2.7": 9,
    "wan2.7-pro": 9,
    "grok": 3,
    "grok_quality": 3,
    "gemini-3.1-flash": 5,
    "gemini-3-pro": 5,
    "seedream-5-pro": 10,
    "seedream-5": 14,
    "gpt-image-2-low": 5,
    "gpt-image-2-medium": 5,
    "gpt-image-2-high": 5,
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


class ImageGeneratePayload(BaseModel):
    """Payload for image generation."""

    prompt: str
    model: ImageGenerateModel = "zimage"
    image_size: ImageSize = "square"
    style: ImageStyle | None = None
    resolution: ImageResolution = "1k"
    seed: int = -1
    image_url: str | None = None
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_model_constraints(self) -> "ImageGeneratePayload":
        """Validate provider-specific styles, inputs, and resolutions."""

        style = self.style
        if style is not None:
            expected_models = {
                "basic": "wan22",
                "instagram": "wan22",
                "replication": "wan22",
                "transparent": "flux2",
            }
            if self.model != expected_models[style]:
                raise ValueError(
                    f"Model {self.model} is not compatible with style {style}"
                )
            if style == "replication" and not self.image_url:
                raise ValueError(f"{style} style requires --image")

        if self.model in {"grok", "grok_quality"} and self.resolution not in {
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
        if self.model in _GPT_IMAGE_2_MODELS:
            if self.image_url is not None:
                raise ValueError(
                    "image is not supported for GPT Image 2 image generate; "
                    "use image edit"
                )
            if self.resolution not in {"1k", "2k", "3k"}:
                raise ValueError(
                    f"resolution {self.resolution} is not supported by model "
                    f"{self.model}"
                )
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
    seed: int = -1
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_model_limits(self) -> "ImageEditPayload":
        """Validate image count limits for the selected model."""

        max_images = _IMAGE_EDIT_MODEL_MAX_IMAGES.get(self.model)
        if max_images is None:
            raise ValueError(f"image edit model {self.model} has no configured limit")
        if len(self.image_urls) > max_images:
            raise ValueError(f"{self.model} supports a maximum of {max_images} images")

        if self.model in {"grok", "grok_quality"} and self.resolution not in {
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
        if self.model in _GPT_IMAGE_2_MODELS:
            if self.resolution not in {"1k", "2k", "3k"}:
                raise ValueError(
                    f"resolution {self.resolution} is not supported by model "
                    f"{self.model}"
                )
            if self.image_size is None:
                self.image_size = "square"
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
    "wan22",
    "ltx23",
    "grok",
    "grok15",
    "veo-3.1",
    "veo-3.1-fast",
    "seedance2",
    "seedance2-mini",
    "kling-2.6",
    "kling-3",
    "kling-3-turbo",
    "kling-3-omni",
    "wan2.7",
]
VideoEditModel = Literal[
    "seedance2",
    "seedance2-mini",
    "kling-3-omni",
    "wan2.7",
    "aleph2",
]
VideoGenerateResolution = Literal["360p", "480p", "720p", "1080p", "4k"]
VideoEditResolution = Literal["480p", "720p", "1080p", "4k"]
VideoGenerateAspectRatio = Literal[
    "16:9",
    "4:3",
    "1:1",
    "3:4",
    "9:16",
    "21:9",
]
VideoEditAspectRatio = Aleph2AspectRatio
VideoGenerateDuration = Annotated[int, Field(ge=1, le=15)]
VideoEditDuration = Annotated[int, Field(ge=2, le=15)]
PublicFigureThreshold = Literal["auto", "low"]


class _StrictVideoPayload(BaseModel):
    """Reject fields that do not belong to the selected API model."""

    model_config = ConfigDict(extra="forbid")


class WAN22VideoGenRequest(_StrictVideoPayload):
    prompt: str
    model: Literal["wan22"] = "wan22"
    duration: int = Field(ge=1, le=10)
    resolution: Literal["360p", "480p"] = "480p"
    aspect_ratio: CommonVideoAspectRatio = "16:9"
    seed: int = -1
    lora_high_url: str | None = None
    lora_low_url: str | None = None
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_lora_urls(self) -> "WAN22VideoGenRequest":
        if bool(self.lora_high_url) != bool(self.lora_low_url):
            raise ValueError(
                "Both lora_high_url and lora_low_url must be provided together"
            )
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
    model: Literal["grok"] = "grok"
    duration: int = Field(ge=1, le=15)
    resolution: Literal["480p", "720p"] = "480p"
    aspect_ratio: CommonVideoAspectRatio = "16:9"
    image_url: str | None = None
    callback_url: str | None = None


class Grok15VideoGenRequest(_StrictVideoPayload):
    prompt: str
    model: Literal["grok15"] = "grok15"
    duration: int = Field(ge=1, le=15)
    resolution: Literal["480p", "720p"] = "480p"
    aspect_ratio: CommonVideoAspectRatio = "16:9"
    image_url: str
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
    model: Literal["seedance2", "seedance2-mini"] = "seedance2"
    duration: int = Field(default=5, ge=4, le=15)
    resolution: Seedance2VideoResolution = "720p"
    aspect_ratio: Seedance2VideoAspectRatio = "16:9"
    generate_audio: bool = True
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_model_resolution(self) -> "Seedance2VideoGenRequest":
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


VideoGenerateRequest = Annotated[
    WAN22VideoGenRequest
    | LTX23VideoGenRequest
    | GrokVideoGenRequest
    | Grok15VideoGenRequest
    | VeoVideoGenRequest
    | Seedance2VideoGenRequest
    | KlingV26VideoGenRequest
    | KlingV3VideoGenRequest
    | KlingV3TurboVideoGenRequest
    | KlingV3OmniVideoGenRequest
    | Wan27VideoGenRequest,
    Discriminator("model"),
]


class VideoEditPayload(_StrictVideoPayload):
    """JSON request body for Seedance ``video_edit`` operations."""

    prompt: str
    model: Literal["seedance2", "seedance2-mini"] = "seedance2"
    duration: int = Field(default=5, ge=4, le=15)
    resolution: Literal["480p", "720p", "1080p", "4k"] = "720p"
    aspect_ratio: Literal["16:9", "4:3", "1:1", "3:4", "9:16", "21:9"] = "16:9"
    generate_audio: bool = True
    callback_url: str | None = None
    content: list[dict[str, Any]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_model_resolution(self) -> "VideoEditPayload":
        if self.model == "seedance2-mini" and self.resolution not in {
            "480p",
            "720p",
        }:
            raise ValueError("seedance2-mini supports only 480p and 720p")
        return self

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
        if has_audio and not has_visual:
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
        if self.image_url.startswith("data:") and len(self.image_url) > 5 * 1024 * 1024:
            raise ValueError("Aleph 2 keyframe data URI exceeds 5 MB")
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
    | Aleph2VideoEditPayload,
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


VideoUpscaleResolution = Literal["720p", "1080p", "1440p"]
LipsyncVideoResolution = Literal["360p", "480p"]
LipsyncImageModel = Literal["inftalk", "ltx23"]
LipsyncImageResolution = Literal["360p", "480p", "720p"]


class VideoUpscalePayload(BaseModel):
    """Payload for video upscale from URL or another task result."""

    video_url: str | None = None
    video_from_task_id: str | None = None
    resolution: VideoUpscaleResolution = Field(
        default="1080p",
        description="Output resolution; 1440p accepts videos up to 20 seconds.",
    )

    @model_validator(mode="after")
    def validate_source(self) -> "VideoUpscalePayload":
        """Require exactly one source: --video or --task-id."""

        if bool(self.video_url) == bool(self.video_from_task_id):
            raise ValueError("Provide exactly one of --video or --task-id")
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
