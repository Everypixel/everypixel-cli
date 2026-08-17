import json
from pathlib import Path


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
        "kling-3-omni",
        "wan2.7",
        "aleph2",
    }
    generate_schema = schema["paths"]["/v1/video_generate"]["post"]["requestBody"][
        "content"
    ]["application/json"]["schema"]
    assert set(generate_schema["discriminator"]["mapping"]) >= {
        "kling-2.6",
        "kling-3",
        "kling-3-turbo",
        "kling-3-omni",
        "seedance2-mini",
        "wan2.7",
        "veo-3.1",
        "veo-3.1-fast",
    }
    assert schema["components"]["schemas"]["Seedance2VideoGenRequest"]["properties"][
        "model"
    ]["enum"] == ["seedance2", "seedance2-mini"]
    assert schema["components"]["schemas"]["VeoVideoGenRequest"]["properties"]["model"][
        "enum"
    ] == ["veo-3.1", "veo-3.1-fast"]
    assert "Aleph2VideoEditRequest" in schema["components"]["schemas"]


def test_bundled_openapi_has_current_image_enums_and_quality_ugc():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    components = schema["components"]["schemas"]

    assert components["ImageStyleEnum"]["enum"] == [
        "basic",
        "instagram",
        "portrait",
        "transparent",
        "replication",
    ]
    assert components["ImageGenerateModelEnum"]["enum"] == [
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
    assert components["ImageEditModelEnum"]["enum"] == [
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
    assert components["ImageGenerateResolutionEnum"]["enum"] == [
        "1k",
        "2k",
        "3k",
        "4k",
    ]
    assert "/v1/quality_ugc" in schema["paths"]


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
