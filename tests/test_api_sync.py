import json
from pathlib import Path
from typing import get_args

from everypixel_cli.schemas import (
    ChatRequest,
    ImageEditModel,
    ImageGenerateModel,
    ImageSize,
    VideoGenerateModel,
    TTSCreateModel,
    TTSDesignModel,
    TTSSpeaker,
    TTSStyle,
    TTSLanguage,
    TTSCharacter,
)


SCHEMA_PATH = (
    Path(__file__).parents[1] / "src" / "everypixel_cli" / "resources" / "openapi.json"
)


def test_bundled_openapi_has_current_video_edit_contract():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    operation = schema["paths"]["/v1/video_edit"]["post"]
    assert operation["operationId"] == "edit_video_v1_video_edit_post"
    assert set(operation["requestBody"]["content"]) == {"application/json"}
    assert operation["responses"]["200"]["content"]["application/json"]["schema"][
        "anyOf"
    ] == [
        {"$ref": "#/components/schemas/TaskResponse"},
        {"$ref": "#/components/schemas/CostEstimateResponse"},
    ]
    edit_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert set(edit_schema["discriminator"]["mapping"]) == {
        "seedance2",
        "seedance2-mini",
        "seedance2.5",
        "kling-3-omni",
        "wan2.7",
        "aleph2",
        "gemini-omni-1.1-flash",
        "gemini-omni-1.2-flash",
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
        "gemini-omni-1.1-flash",
        "gemini-omni-1.2-flash",
    }
    assert schema["components"]["schemas"]["Seedance2VideoGenRequest"]["properties"][
        "model"
    ]["enum"] == ["seedance2", "seedance2-mini", "seedance2.5"]
    for component in ("OmniVideoGenRequest", "OmniVideoEditRequest"):
        model = schema["components"]["schemas"][component]["properties"]["model"]
        assert model["enum"] == ["gemini-omni-1.1-flash", "gemini-omni-1.2-flash"]
        assert model["default"] == "gemini-omni-1.2-flash"
    omni = schema["components"]["schemas"]["OmniVideoGenRequest"]["properties"]
    assert omni["duration"]["maximum"] == 15
    assert omni["reference_image_urls"]["maxItems"] == 6
    components = schema["components"]["schemas"]
    assert components["Grok15VideoResolutionEnum"]["enum"] == ["480p", "720p", "1080p"]
    assert (
        edit_schema["discriminator"]["mapping"]["gemini-omni-1.2-flash"]
        == "#/components/schemas/OmniVideoEditRequest"
    )
    for component in edit_schema["discriminator"]["mapping"].values():
        properties = components[component.rsplit("/", 1)[1]]["properties"]
        assert "video_from_task_id" in properties
    assert "Aleph2VideoEditRequest" in components
    assert set(generate_schema["discriminator"]["mapping"]) == set(
        get_args(VideoGenerateModel)
    ) - {"grok", "grok15"}
    for model, component in {
        "minimax-h3": "MiniMaxH3VideoGenRequest",
        "minimax-h3-turbo": "MiniMaxH3VideoGenRequest",
        "wan3.0": "Wan30VideoGenRequest",
        "flux3": "Flux3VideoGenRequest",
        "gemini-omni-1.2-flash": "OmniVideoGenRequest",
        "seedance2.5": "Seedance2VideoGenRequest",
    }.items():
        assert generate_schema["discriminator"]["mapping"][model] == (
            f"#/components/schemas/{component}"
        )
    assert edit_schema["discriminator"]["mapping"]["flux3"] == (
        "#/components/schemas/Flux3VideoEditRequest"
    )
    seedance = components["Seedance2VideoGenRequest"]
    for name in ("Seedance2VideoGenRequest", "Seedance2VideoEditRequest"):
        properties = components[name]["properties"]
        assert properties["seed"]["minimum"] == -1
        assert properties["seed"]["maximum"] == 2_147_483_647
        assert properties["draft"]["default"] is False
        assert properties["draft_from_task_id"]["anyOf"][0]["format"] == "uuid"
    assert {"seed", "draft", "draft_from_task_id", "draft_expires_at"} <= (
        components["ImgenStatusResponse"]["properties"].keys()
    )
    assert seedance["additionalProperties"] is False
    assert seedance["properties"]["content"]["items"] == {
        "$ref": "#/components/schemas/Seedance2GenerationImage"
    }
    image = components["Seedance2GenerationImage"]
    assert image["additionalProperties"] is False
    assert image["properties"]["role"]["default"] == "first_frame"
    assert image["properties"]["role"]["enum"] == [
        "first_frame",
        "last_frame",
        "reference_image",
    ]
    assert components["ProviderContentImageUrl"]["additionalProperties"] is False
    minimax = schema["components"]["schemas"]["MiniMaxH3VideoGenRequest"]
    for media_type, limit in (("image", 9), ("video", 3), ("audio", 3)):
        assert (
            minimax["properties"][f"reference_{media_type}_urls"]["maxItems"] == limit
        )


def test_bundled_openapi_has_current_image_enums_and_quality_ugc():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    components = schema["components"]["schemas"]

    for operation, model_type in (
        ("generate", ImageGenerateModel),
        ("edit", ImageEditModel),
    ):
        endpoint = schema["paths"][f"/v1/image_{operation}"]["post"]
        assert (
            endpoint["operationId"]
            == {
                "generate": "generate_v1_image_generate_post",
                "edit": "image_edit_v1_image_edit_post",
            }[operation]
        )
        body = endpoint["requestBody"]["content"]["application/json"]["schema"]
        mapping = body["discriminator"]["mapping"]
        assert set(mapping) == set(get_args(model_type)) - {"grok"}
        seedream = f"Seedream55Image{operation.title()}Request"
        assert mapping["seedream-5.5"] == f"#/components/schemas/{seedream}"
        assert components[seedream]["properties"]["resolution"] == {
            "$ref": "#/components/schemas/Seedream55Resolution",
            "default": "2k",
        }
        if operation == "generate":
            assert mapping["krea2"] == "#/components/schemas/Krea2ImageGenerateRequest"
        for version in ("2", "2.5", "2.6"):
            model = "gpt-image-2" if version == "2" else f"gpt-image-{version}-sunburst"
            name = f"GPTImage{version.replace('.', '')}{operation.title()}Request"
            assert mapping[model] == f"#/components/schemas/{name}"
            properties = components[name]["properties"]
            assert properties["quality"]["default"] == "medium"
            assert properties["resolution"]["enum"] == ["1k", "2k", "3k"]
        formats = components[f"GPTImage26{operation.title()}Request"]["properties"][
            "output_format"
        ]["enum"]
        assert formats == (
            ["png", "jpeg", "webp", "gif", "svg"]
            if operation == "generate"
            else ["png", "jpeg", "webp", "gif"]
        )
    assert components["ImageSizeEnum"]["enum"] == list(get_args(ImageSize))
    krea = components["Krea2ImageGenerateRequest"]["properties"]
    assert krea["style"]["const"] == "portrait"
    assert krea["quality"]["enum"] == ["low", "medium"]
    assert krea["quality"]["default"] == "medium"
    assert krea["resolution"]["enum"] == ["1k", "2k"]
    assert components["Seedream55Resolution"]["enum"] == ["1k", "1.5k", "2k"]
    assert (
        components["Seedream55ImageEditRequest"]["properties"]["image_urls"]["maxItems"]
        == 10
    )
    assert (
        components["Flux2ImageGenerateRequest"]["properties"]["style"]["anyOf"][0][
            "const"
        ]
        == "transparent"
    )
    assert "/v1/quality_ugc" in schema["paths"]
    assert components["ImageQualityEnum"]["enum"] == [
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]


def test_bundled_chat_contract_matches_local_request_schema():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    operation = schema["paths"]["/v1/chat/completions"]["post"]
    assert operation["operationId"] == "chat_completions_v1_chat_completions_post"
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ChatRequest"
    }
    local = ChatRequest.model_json_schema(ref_template="#/components/schemas/{model}")
    definitions = local.pop("$defs")

    def normalized(value):
        if isinstance(value, dict):
            return {
                key: normalized(item)
                for key, item in value.items()
                if key not in {"title", "description"}
                and not (key == "default" and item is None)
            }
        if isinstance(value, list):
            return [normalized(item) for item in value]
        if isinstance(value, str) and value.startswith("#/components/schemas/Chat"):
            return value.replace("/Chat", "/")
        return value

    for name, component in {"ChatRequest": local, **definitions}.items():
        live_name = name if name == "ChatRequest" else name.removeprefix("Chat")
        assert normalized(component) == normalized(
            schema["components"]["schemas"][live_name]
        )


def test_bundled_openapi_has_current_dev_endpoints_and_content_refs():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    components = schema["components"]["schemas"]

    assert "/v1/auth/check" in schema["paths"]
    assert "/v1/image_remove_background" in schema["paths"]
    assert schema["paths"]["/v1/tasks/{task_id}"]["delete"]["operationId"] == (
        "delete_task_v1_tasks__task_id__delete"
    )
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
        "topaz-slp-2.6",
        "topaz-ast-2",
    ]
    assert components["VideoUpscaleResolutionEnum"]["enum"] == [
        "720p",
        "1080p",
        "1440p",
        "4k",
    ]


def test_bundled_tts_contract_matches_local_models():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    components = schema["components"]["schemas"]
    for name, limit in (
        ("TTSCreateRequest", 2048),
        ("TTSCloneRequest", 200),
        ("QwenTTSDesignRequest", 200),
        ("ElevenLabsVoiceDesignRequest", 1000),
    ):
        assert components[name]["properties"]["text"]["maxLength"] == limit
    assert set(components["TTSCreateRequest"]["properties"]["model"]["enum"]) == set(
        get_args(TTSCreateModel)
    )
    assert components["TTSCreateRequest"]["properties"]["model"]["enum"] == [
        "qwen3",
        "eleven_v4",
        "eleven_v4_turbo",
        "eleven_v3",
        "eleven_v3_conversational",
        "eleven_multilingual_v2",
        "eleven_flash_v2_5",
    ]
    assert set(
        components["ElevenLabsVoiceDesignRequest"]["properties"]["model"]["enum"]
    ) | {"qwen3"} == set(get_args(TTSDesignModel))
    for name, local in (
        ("TTSSpeakerEnum", TTSSpeaker),
        ("TTSStyleEnum", TTSStyle),
        ("TTSLanguageEnum", TTSLanguage),
        ("TTSCharacterEnum", TTSCharacter),
    ):
        assert components[name]["enum"] == list(get_args(local))
    assert "TTSVoiceSaveRequest" in components
    for path, method, operation in (
        ("tts_create", "post", "tts_create"),
        ("tts_clone", "post", "tts_clone"),
        ("tts_design", "post", "tts_design"),
        ("tts_voice", "post", "tts_voice"),
        ("tts_voice", "delete", "tts_voice_delete"),
        ("tts_voices", "get", "tts_voices"),
    ):
        assert (
            schema["paths"][f"/v1/{path}"][method]["operationId"]
            == f"{operation}_v1_{path}_{method}"
        )

    design = schema["paths"]["/v1/tts_design"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert design["anyOf"] == [
        {"$ref": "#/components/schemas/QwenTTSDesignRequest"},
        {"$ref": "#/components/schemas/ElevenLabsVoiceDesignRequest"},
    ]
    assert components["TTSCloneRequest"]["required"] == ["model", "text"]
    assert schema["paths"]["/v1/tts_voice"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/TTSVoiceSaveRequest"}


def test_bundled_task_costs_are_optional_usd_strings():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    components = schema["components"]["schemas"]
    for name, fields in (
        ("TaskResponse", ("estimated_cost",)),
        ("ImgenStatusResponse", ("estimated_cost", "billed_cost")),
    ):
        for field in fields:
            assert components[name]["properties"][field]["anyOf"] == [
                {"type": "string"},
                {"type": "null"},
            ]
            assert field not in components[name]["required"]


def test_bundled_balance_and_cost_estimation_contract():
    from everypixel_cli.application.services import COST_ESTIMATE_ENDPOINTS

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    balance = schema["paths"]["/v1/account/balance"]["get"]
    assert balance["operationId"] == "get_balance_v1_account_balance_get"
    assert balance["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/BalanceResponse"
    }
    assert schema["components"]["schemas"]["BalanceResponse"]["required"] == [
        "balance",
        "email",
    ]
    endpoints = set()
    for path, methods in schema["paths"].items():
        operation = methods.get("post", {})
        flags = [
            p for p in operation.get("parameters", []) if p["name"] == "estimate_cost"
        ]
        if not flags:
            continue
        endpoints.add(path)
        assert flags[0]["in"] == "query"
        assert flags[0]["schema"]["type"] == "boolean"
        assert flags[0]["schema"]["default"] is False
        assert {"$ref": "#/components/schemas/CostEstimateResponse"} in (
            operation["responses"]["200"]["content"]["application/json"]["schema"][
                "anyOf"
            ]
        )
    assert endpoints == COST_ESTIMATE_ENDPOINTS


def test_bundled_heygen_contract():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    components = schema["components"]["schemas"]
    for path, method, operation_id in (
        ("video_translate", "post", "video_translate_v1_video_translate_post"),
        (
            "video_translate/languages",
            "get",
            "translation_languages_v1_video_translate_languages_get",
        ),
        ("heygen/voices", "get", "heygen_voices_v1_heygen_voices_get"),
    ):
        assert schema["paths"][f"/v1/{path}"][method]["operationId"] == operation_id
    assert components["LipsyncRequest"]["properties"]["model"]["enum"] == [
        "inftalk",
        "heygen-speed",
        "heygen-precision",
    ]
    avatar = components["HeyGenImageLipsyncRequest"]
    assert avatar["properties"]["model"]["const"] == "heygen-avatar-4"
    assert avatar["additionalProperties"] is False
    assert {"$ref": "#/components/schemas/HeyGenImageLipsyncRequest"} in schema[
        "paths"
    ]["/v1/image_lipsync"]["post"]["requestBody"]["content"]["application/json"][
        "schema"
    ]["anyOf"]

    from everypixel_cli.schemas import HeyGenAvatar6Payload

    def normalized(value):
        if isinstance(value, dict):
            return {
                key: normalized(item)
                for key, item in value.items()
                if key not in {"title", "description"}
                and not (key == "default" and item is None)
            }
        if isinstance(value, list):
            return [normalized(item) for item in value]
        return value

    assert normalized(HeyGenAvatar6Payload.model_json_schema()) == normalized(
        components["HeyGenAvatar6Request"]
    )
    assert {"$ref": "#/components/schemas/HeyGenAvatar6Request"} in schema["paths"][
        "/v1/image_lipsync"
    ]["post"]["requestBody"]["content"]["application/json"]["schema"]["anyOf"]


def test_image_provider_schemas_match_bundled_contract():
    from pydantic import TypeAdapter
    from everypixel_cli.schemas import ImageGeneratePayload, ImageEditPayload

    document = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def normalized(value, definitions):
        if isinstance(value, dict):
            if "$ref" in value:
                return normalized(
                    {
                        **definitions[value["$ref"].rsplit("/", 1)[1]],
                        **{key: item for key, item in value.items() if key != "$ref"},
                    },
                    definitions,
                )
            return {
                key: normalized(item, definitions)
                for key, item in value.items()
                if key not in {"title", "description", "additionalProperties"}
                and not (key == "default" and item is None)
            }
        if isinstance(value, list):
            return [normalized(item, definitions) for item in value]
        return value

    for operation, payload_type in (
        ("generate", ImageGeneratePayload),
        ("edit", ImageEditPayload),
    ):
        local = TypeAdapter(payload_type).json_schema()
        remote = document["paths"][f"/v1/image_{operation}"]["post"]["requestBody"][
            "content"
        ]["application/json"]["schema"]
        assert (
            local["discriminator"]["mapping"].keys()
            == remote["discriminator"]["mapping"].keys()
        )
        for model, ref in local["discriminator"]["mapping"].items():
            actual = normalized({"$ref": ref}, local["$defs"])
            expected = normalized(
                {"$ref": remote["discriminator"]["mapping"][model]},
                document["components"]["schemas"],
            )
            if model == "krea2":
                # The client sends an explicit model and defaults to the fast tier.
                expected["required"].append("model")
                expected["properties"]["model"].pop("default")
                expected["properties"]["quality"]["default"] = "low"
            assert actual == expected, model
