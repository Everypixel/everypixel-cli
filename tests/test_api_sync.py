import json
from pathlib import Path
from typing import get_args

from everypixel_cli.schemas import (
    ImageEditModel,
    ImageGenerateModel,
    VideoGenerateModel,
)


SCHEMA_PATH = (
    Path(__file__).parents[1] / "src" / "everypixel_cli" / "resources" / "openapi.json"
)


def test_bundled_openapi_has_current_video_edit_contract():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    operation = schema["paths"]["/v1/video_edit"]["post"]
    assert operation["operationId"] == "edit_video_v1_video_edit_post"
    assert set(operation["requestBody"]["content"]) == {"application/json"}
    assert (
        operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/TaskResponse"
    )
    edit_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert set(edit_schema["discriminator"]["mapping"]) == {
        "seedance2",
        "seedance2-mini",
        "seedance2.5",
        "kling-3-omni",
        "wan2.7",
        "aleph2",
        "flux3",
    }
    generate_schema = schema["paths"]["/v1/video_generate"]["post"]["requestBody"][
        "content"
    ]["application/json"]["schema"]
    assert set(generate_schema["discriminator"]["mapping"]) == {
        "minimax-h3-turbo",
        "minimax-h3",
        "ltx23",
        "grok-imagine",
        "grok-imagine-1.5",
        "flux3",
        "seedance2",
        "seedance2.5",
        "kling-2.6",
        "kling-3",
        "kling-3-turbo",
        "kling-3-omni",
        "seedance2-mini",
        "wan2.7",
        "wan3.0",
        "veo-3.1",
        "veo-3.1-fast",
    }
    assert schema["components"]["schemas"]["Seedance2VideoGenRequest"]["properties"][
        "model"
    ]["enum"] == ["seedance2", "seedance2-mini", "seedance2.5"]
    assert schema["components"]["schemas"]["VeoVideoGenRequest"]["properties"]["model"][
        "enum"
    ] == ["veo-3.1", "veo-3.1-fast"]
    assert "Aleph2VideoEditRequest" in schema["components"]["schemas"]
    assert set(generate_schema["discriminator"]["mapping"]) == set(
        get_args(VideoGenerateModel)
    ) - {"grok", "grok15"}
    for model, component in {
        "minimax-h3": "MiniMaxH3VideoGenRequest",
        "minimax-h3-turbo": "MiniMaxH3VideoGenRequest",
        "wan3.0": "Wan30VideoGenRequest",
        "flux3": "Flux3VideoGenRequest",
        "seedance2.5": "Seedance2VideoGenRequest",
    }.items():
        assert generate_schema["discriminator"]["mapping"][model] == (
            f"#/components/schemas/{component}"
        )
    assert edit_schema["discriminator"]["mapping"]["flux3"] == (
        "#/components/schemas/Flux3VideoEditRequest"
    )


def test_bundled_openapi_has_current_image_enums_and_quality_ugc():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    components = schema["components"]["schemas"]

    assert components["ImageStyleEnum"]["enum"] == [
        "portrait",
        "transparent",
    ]
    assert components["ImageGenerateModelEnum"]["enum"] == [
        "zimage",
        "wan2.7",
        "wan2.7-pro",
        "flux2",
        "grok-imagine",
        "grok-imagine-2",
        "grok-imagine-2-low",
        "gemini-3.1-flash",
        "gemini-3-pro",
        "seedream-5-pro",
        "seedream-5",
        "gpt-image-2-low",
        "gpt-image-2-medium",
        "gpt-image-2-high",
        "recraftv4_1_vector",
        "recraftv4_1_pro_vector",
    ]
    assert components["ImageEditModelEnum"]["enum"] == [
        "flux2",
        "qwen",
        "wan2.7",
        "wan2.7-pro",
        "grok-imagine",
        "grok-imagine-2",
        "grok-imagine-2-low",
        "gemini-3.1-flash",
        "gemini-3-pro",
        "seedream-5-pro",
        "seedream-5",
        "gpt-image-2-low",
        "gpt-image-2-medium",
        "gpt-image-2-high",
    ]
    assert components["ImageGenerateResolutionEnum"]["enum"] == [
        "1k",
        "2k",
        "3k",
        "4k",
    ]
    assert "/v1/quality_ugc" in schema["paths"]
    assert set(components["ImageGenerateModelEnum"]["enum"]) == set(
        get_args(ImageGenerateModel)
    ) - {"grok"}
    assert set(components["ImageEditModelEnum"]["enum"]) == set(
        get_args(ImageEditModel)
    ) - {"grok"}


def test_bundled_openapi_has_current_dev_endpoints_and_content_refs():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    components = schema["components"]["schemas"]

    assert "/v1/auth/check" in schema["paths"]
    assert "/v1/image_remove_background" in schema["paths"]
    assert "AuthCheckResponse" in components
    assert "ImageRemoveBackgroundRequest" in components
    assert "ProviderContentAudioUrl" in components
    assert (
        components["Seedance2ContentAudio"]["properties"]["audio_url"]["$ref"]
        == "#/components/schemas/ProviderContentAudioUrl"
    )
    content_schema = components["Seedance2VideoEditRequest"]["properties"]["content"][
        "items"
    ]
    assert content_schema["discriminator"]["propertyName"] == "type"
    assert len(content_schema["oneOf"]) == 4
    vectorize = schema["paths"]["/v1/image_vectorize"]["post"]
    assert vectorize["operationId"] == "image_vectorize_v1_image_vectorize_post"
    assert vectorize["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ImageVectorizeRequest"
    }
    assert {
        "RecraftColor",
        "RecraftControls",
        "ImageVectorizeRequest",
    } <= components.keys()
    assert components["VideoUpscaleModelEnum"]["enum"] == [
        "seedvr2",
        "topaz-prob-4",
        "topaz-slp-2.5",
        "topaz-ast-2",
    ]
    assert components["VideoUpscaleResolutionEnum"]["enum"] == [
        "720p",
        "1080p",
        "1440p",
        "4k",
    ]
    for name in ("TTSCreateRequest", "TTSCloneRequest", "TTSVoiceRequest"):
        assert components[name]["properties"]["text"]["maxLength"] == 200
