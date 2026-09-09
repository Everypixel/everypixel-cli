import pytest
from pydantic import TypeAdapter, ValidationError

from everypixel_cli.schemas import (
    ImageEditPayload,
    ImageGeneratePayload,
    ImageUpscalePayload,
    LipsyncVideoPayload,
    VideoEditPayload,
    VideoEditRequest,
    VideoGenerateRequest,
    VideoUpscalePayload,
)


_VIDEO_GENERATE_ADAPTER: TypeAdapter[VideoGenerateRequest] = TypeAdapter(
    VideoGenerateRequest
)
_VIDEO_EDIT_ADAPTER: TypeAdapter[VideoEditRequest] = TypeAdapter(VideoEditRequest)


def validate_video_generate(**values):
    return _VIDEO_GENERATE_ADAPTER.validate_python(values)


@pytest.mark.parametrize(
    ("model", "style"),
    [("zimage", "portrait"), ("flux2", "transparent")],
)
def test_image_generate_keeps_styles_for_supported_models(model, style):
    assert ImageGeneratePayload(prompt="x", model=model, style=style).style == style


def test_removed_flux_model_is_rejected_and_flux2_remains_valid():
    with pytest.raises(ValidationError):
        ImageGeneratePayload(prompt="x", model="flux")

    assert ImageGeneratePayload(prompt="x", model="flux2").model == "flux2"


def test_qwen_accepts_max_three_images():
    with pytest.raises(ValidationError):
        ImageEditPayload(prompt="x", model="qwen", image_urls=["1", "2", "3", "4"])


def test_new_image_models_apply_provider_constraints():
    assert ImageGeneratePayload(prompt="x", model="seedream-5").resolution == "2k"
    assert ImageGeneratePayload(prompt="x", model="wan2.7").resolution == "2k"
    assert (
        ImageGeneratePayload(prompt="x", model="wan2.7-pro", resolution="4k").model
        == "wan2.7-pro"
    )
    assert (
        ImageEditPayload(
            prompt="x",
            model="wan2.7-pro",
            image_urls=["https://img.test/source.png"],
        ).resolution
        == "2k"
    )
    assert (
        ImageEditPayload(
            prompt="x",
            model="gpt-image-2-medium",
            resolution="3k",
            image_urls=["https://img.test/source.png"],
        ).image_size
        == "square"
    )

    with pytest.raises(ValidationError):
        ImageGeneratePayload(prompt="x", model="gemini-3-pro", resolution="3k")
    with pytest.raises(ValidationError):
        ImageEditPayload(
            prompt="x",
            model="seedream-5",
            image_urls=[str(index) for index in range(15)],
        )
    with pytest.raises(ValidationError):
        ImageEditPayload(
            prompt="x",
            model="wan2.7",
            image_urls=[str(index) for index in range(10)],
        )


def test_retired_wan22_is_rejected():
    with pytest.raises(ValidationError):
        validate_video_generate(
            prompt="x", model="wan22", duration=5, resolution="480p"
        )


@pytest.mark.parametrize(
    ("model", "resolution", "aspect_ratio"),
    [
        ("minimax-h3", "1080p", "7:4"),
        ("ltx23", "4k", "16:9"),
        ("ltx23", "720p", "21:9"),
        ("grok", "360p", "16:9"),
        ("grok15", "720p", "4:3"),
    ],
)
def test_video_models_reject_other_provider_values(model, resolution, aspect_ratio):
    values = {
        "prompt": "x",
        "model": model,
        "duration": 5,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
    }
    if model == "grok15":
        values["image_url"] = "https://img.test/source.png"
    with pytest.raises(ValidationError):
        validate_video_generate(**values)


def test_video_models_apply_provider_defaults_and_fields():
    ltx23 = validate_video_generate(prompt="x", model="ltx23", duration=5)

    assert ltx23.model_dump(exclude_none=True)["resolution"] == "720p"
    assert "generate_audio" not in ltx23.model_dump(exclude_none=True)


@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        ("minimax-h3", "generate_audio", False),
        ("seedance2", "image_url", "https://img.test/source.png"),
        ("grok", "image_last_url", "https://img.test/last.png"),
        ("ltx23", "lora_high_url", "https://cdn.test/lora.safetensors"),
    ],
)
def test_video_models_reject_unsupported_fields(model, field, value):
    with pytest.raises(ValidationError):
        validate_video_generate(
            prompt="x",
            model=model,
            duration=5,
            **{field: value},
        )


def test_image_upscale_requires_exactly_one_source():
    with pytest.raises(ValidationError):
        ImageUpscalePayload()
    with pytest.raises(ValidationError):
        ImageUpscalePayload(
            image_url="https://img.test/in.png", image_from_task_id="abc"
        )


def test_video_upscale_requires_exactly_one_source_and_valid_resolution():
    with pytest.raises(ValidationError):
        VideoUpscalePayload()
    with pytest.raises(ValidationError):
        VideoUpscalePayload(video_from_task_id="abc", resolution="480p")


def test_lipsync_video_rejects_1080p():
    with pytest.raises(ValidationError):
        LipsyncVideoPayload(
            video_url="https://cdn.test/in.mp4",
            audio_url="https://cdn.test/voice.mp3",
            resolution="1080p",
        )


def test_seedance2_video_generation_contract():
    payload = validate_video_generate(
        prompt="x",
        model="seedance2",
        duration=4,
        resolution="4k",
        aspect_ratio="21:9",
        generate_audio=False,
    )

    assert payload.model_dump(exclude_none=True) == {
        "prompt": "x",
        "model": "seedance2",
        "duration": 4,
        "resolution": "4k",
        "aspect_ratio": "21:9",
        "generate_audio": False,
    }


@pytest.mark.parametrize(
    ("duration", "resolution", "aspect_ratio"),
    [(3, "720p", "16:9"), (5, "360p", "16:9"), (5, "720p", "2:1")],
)
def test_seedance2_rejects_values_outside_its_schema(
    duration, resolution, aspect_ratio
):
    with pytest.raises(ValidationError):
        validate_video_generate(
            prompt="x",
            model="seedance2",
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
        )


def test_seedance2_mini_accepts_only_480p_and_720p():
    payload = validate_video_generate(
        prompt="x", model="seedance2-mini", duration=4, resolution="480p"
    )

    assert payload.model == "seedance2-mini"
    with pytest.raises(ValidationError, match="supports only 480p and 720p"):
        validate_video_generate(
            prompt="x", model="seedance2-mini", duration=4, resolution="1080p"
        )


def test_wan27_video_contracts_validate_reference_modes():
    generated = validate_video_generate(
        prompt="move",
        model="wan2.7",
        duration=10,
        resolution="1080p",
        aspect_ratio="3:4",
        reference_image_urls=["https://img.test/reference.png"],
        reference_video_urls=["https://cdn.test/reference.mp4"],
    )
    edited = _VIDEO_EDIT_ADAPTER.validate_python(
        {
            "prompt": "change outfit",
            "model": "wan2.7",
            "video_url": "https://cdn.test/source.mp4",
            "reference_image_urls": ["https://img.test/outfit.png"],
        }
    )

    assert generated.model_dump()["seed"] == -1
    assert edited.model_dump()["duration"] == 5
    with pytest.raises(ValidationError, match="up to 10 seconds"):
        validate_video_generate(
            prompt="move",
            model="wan2.7",
            duration=11,
            reference_video_urls=["https://cdn.test/reference.mp4"],
        )
    with pytest.raises(ValidationError, match="cannot be combined"):
        validate_video_generate(
            prompt="move",
            model="wan2.7",
            image_url="https://img.test/start.png",
            reference_image_urls=["https://img.test/reference.png"],
        )


def test_veo_video_generation_contract():
    payload = validate_video_generate(prompt="x", model="veo-3.1-fast")

    assert payload.duration == 8
    assert payload.resolution == "720p"
    assert payload.aspect_ratio == "16:9"
    with pytest.raises(ValidationError, match="requires duration 8"):
        validate_video_generate(
            prompt="x",
            model="veo-3.1",
            duration=6,
            resolution="1080p",
        )
    with pytest.raises(ValidationError, match="cannot be combined"):
        validate_video_generate(
            prompt="x",
            model="veo-3.1",
            image_url="https://img.test/start.png",
            reference_image_urls=["https://img.test/reference.png"],
        )


def test_aleph2_video_edit_contract():
    payload = _VIDEO_EDIT_ADAPTER.validate_python(
        {
            "model": "aleph2",
            "video_url": "https://cdn.test/source.mp4",
            "keyframes": [
                {
                    "image_url": "https://img.test/guide.png",
                    "at": 0.5,
                    "range": {"start_seconds": 1, "end_seconds": 4},
                }
            ],
            "target_aspect_ratio": "3:2",
        }
    )

    assert payload.prompt is None
    assert payload.keyframes[0].at == 0.5
    with pytest.raises(ValidationError, match="exactly one"):
        _VIDEO_EDIT_ADAPTER.validate_python(
            {
                "model": "aleph2",
                "video_url": "https://cdn.test/source.mp4",
                "keyframes": [
                    {
                        "image_url": "https://img.test/guide.png",
                        "seconds": 2,
                        "at": 0.5,
                    }
                ],
            }
        )


def test_seedance_video_edit_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _VIDEO_EDIT_ADAPTER.validate_python(
            {
                "model": "seedance2",
                "prompt": "edit",
                "content": [
                    {
                        "type": "image_url",
                        "role": "reference_image",
                        "image_url": {"url": "https://img.test/guide.png"},
                    }
                ],
                "unknown_option": True,
            }
        )


@pytest.mark.parametrize(
    ("model", "resolution", "generate_audio"),
    [
        ("kling-2.6", "1080p", True),
        ("kling-3", "1080p", True),
        ("kling-3-turbo", "720p", False),
        ("kling-3-omni", "1080p", True),
    ],
)
def test_kling_video_models_apply_provider_defaults(model, resolution, generate_audio):
    payload = validate_video_generate(prompt="x", model=model, duration=5)

    assert payload.resolution == resolution
    assert payload.generate_audio is generate_audio


def test_kling_video_edit_requires_one_video_and_disables_4k():
    content = [
        {"type": "text", "text": "edit"},
        {
            "type": "video_url",
            "video_url": {"url": "https://cdn.test/source.mp4"},
        },
    ]
    payload = _VIDEO_EDIT_ADAPTER.validate_python(
        {"prompt": "x", "model": "kling-3-omni", "content": content}
    )
    assert payload.generate_audio is False

    with pytest.raises(ValidationError):
        _VIDEO_EDIT_ADAPTER.validate_python(
            {
                "prompt": "x",
                "model": "kling-3-omni",
                "resolution": "4k",
                "content": content,
            }
        )


def test_video_edit_requires_visual_reference_and_limits_video_count():
    with pytest.raises(ValidationError):
        VideoEditPayload(prompt="x", content=[{"type": "text", "text": "x"}])
    with pytest.raises(ValidationError):
        VideoEditPayload(
            prompt="x",
            content=[
                {"type": "text", "text": "x"},
                {
                    "type": "audio_url",
                    "role": "reference_audio",
                    "audio_url": {"url": "https://cdn.test/a.mp3"},
                },
            ],
        )
    with pytest.raises(ValidationError):
        VideoEditPayload(
            prompt="x",
            content=[
                {
                    "type": "video_url",
                    "role": "reference_video",
                    "video_url": {"url": "https://cdn.test/a.mp4"},
                },
                {
                    "type": "video_url",
                    "role": "reference_video",
                    "video_url": {"url": "https://cdn.test/b.mp4"},
                },
            ],
        )
