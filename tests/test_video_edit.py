import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from everypixel_cli import cli
from everypixel_cli.cli import app


runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def invoke(args: list[str], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    return runner.invoke(app, ["--base-url", "https://api.test", *args])


def pending_response():
    return httpx.Response(200, json={"task_id": "edit-123", "status": "PENDING"})


@respx.mock
def test_video_generate_seedance2_uses_its_api_contract(monkeypatch):
    route = respx.post("https://api.test/v1/video_generate").mock(
        return_value=pending_response()
    )

    result = invoke(
        [
            "--output-json",
            "--no-wait",
            "video",
            "generate",
            "city at night",
            "--model",
            "seedance2",
            "--duration",
            "15",
            "--resolution",
            "4k",
            "--aspect-ratio",
            "21:9",
            "--no-generate-audio",
        ],
        monkeypatch,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(route.calls[0].request.content) == {
        "prompt": "city at night",
        "model": "seedance2",
        "duration": 15,
        "resolution": "4k",
        "aspect_ratio": "21:9",
        "generate_audio": False,
    }


@respx.mock
def test_video_generate_kling_turbo_uses_provider_defaults(monkeypatch):
    route = respx.post("https://api.test/v1/video_generate").mock(
        return_value=pending_response()
    )

    result = invoke(
        [
            "--output-json",
            "--no-wait",
            "video",
            "generate",
            "city at night",
            "--model",
            "kling-3-turbo",
        ],
        monkeypatch,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(route.calls[0].request.content) == {
        "prompt": "city at night",
        "model": "kling-3-turbo",
        "duration": 5,
        "resolution": "720p",
        "aspect_ratio": "16:9",
        "generate_audio": False,
    }


@respx.mock
def test_video_generate_wan_uses_reference_media_contract(monkeypatch):
    route = respx.post("https://api.test/v1/video_generate").mock(
        return_value=pending_response()
    )

    result = invoke(
        [
            "--output-json",
            "--no-wait",
            "video",
            "generate",
            "consistent motion",
            "--model",
            "wan2.7",
            "--duration",
            "10",
            "--resolution",
            "1080p",
            "--aspect-ratio",
            "3:4",
            "--reference-image",
            str(FIXTURES / "sample.png"),
            "--reference-video",
            str(FIXTURES / "result-1.mp4"),
            "--seed",
            "123",
        ],
        monkeypatch,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(route.calls[0].request.content)
    assert payload["model"] == "wan2.7"
    assert payload["duration"] == 10
    assert payload["resolution"] == "1080p"
    assert payload["aspect_ratio"] == "3:4"
    assert payload["seed"] == 123
    assert payload["reference_image_urls"][0].startswith("data:image/png;base64,")
    assert payload["reference_video_urls"][0].startswith("data:video/mp4;base64,")


@respx.mock
def test_video_generate_veo_uses_reference_image_contract(monkeypatch):
    route = respx.post("https://api.test/v1/video_generate").mock(
        return_value=pending_response()
    )

    result = invoke(
        [
            "--output-json",
            "--no-wait",
            "video",
            "generate",
            "preserve the product",
            "--model",
            "veo-3.1-fast",
            "--resolution",
            "4k",
            "--reference-image",
            str(FIXTURES / "sample.png"),
        ],
        monkeypatch,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(route.calls[0].request.content)
    assert payload["model"] == "veo-3.1-fast"
    assert payload["duration"] == 8
    assert payload["resolution"] == "4k"
    assert payload["reference_image_urls"][0].startswith("data:image/png;base64,")


@respx.mock
def test_video_edit_posts_json_content_with_local_image(monkeypatch):
    route = respx.post("https://api.test/v1/video_edit").mock(
        return_value=pending_response()
    )

    result = invoke(
        [
            "--output-json",
            "--no-wait",
            "video",
            "edit",
            "make it cinematic",
            "--image",
            str(FIXTURES / "sample.png"),
        ],
        monkeypatch,
    )

    assert result.exit_code == 0, result.output
    assert route.calls[0].request.headers["content-type"].startswith("application/json")
    payload = json.loads(route.calls[0].request.content)
    assert payload["model"] == "seedance2"
    assert payload["content"][0] == {"type": "text", "text": "make it cinematic"}
    assert payload["content"][1]["role"] == "reference_image"
    assert payload["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert str(FIXTURES / "sample.png") not in route.calls[0].request.content.decode()


@respx.mock
def test_video_edit_kling_omni_uses_video_provider_contract(monkeypatch):
    route = respx.post("https://api.test/v1/video_edit").mock(
        return_value=pending_response()
    )

    result = invoke(
        [
            "--output-json",
            "--no-wait",
            "video",
            "edit",
            "make it cinematic",
            "--model",
            "kling-3-omni",
            "--video",
            "https://cdn.test/source.mp4",
        ],
        monkeypatch,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(route.calls[0].request.content)
    assert payload["model"] == "kling-3-omni"
    assert payload["resolution"] == "1080p"
    assert payload["generate_audio"] is False
    assert payload["content"][1] == {
        "type": "video_url",
        "video_url": {"url": "https://cdn.test/source.mp4"},
    }


@respx.mock
def test_video_edit_wan_uses_flat_provider_contract(monkeypatch):
    route = respx.post("https://api.test/v1/video_edit").mock(
        return_value=pending_response()
    )

    result = invoke(
        [
            "--output-json",
            "--no-wait",
            "video",
            "edit",
            "change outfit",
            "--model",
            "wan2.7",
            "--video",
            str(FIXTURES / "result-1.mp4"),
            "--image",
            str(FIXTURES / "sample.png"),
            "--seed",
            "321",
        ],
        monkeypatch,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(route.calls[0].request.content)
    assert "content" not in payload
    assert payload["model"] == "wan2.7"
    assert payload["video_url"].startswith("data:video/mp4;base64,")
    assert payload["reference_image_urls"][0].startswith("data:image/png;base64,")
    assert payload["seed"] == 321


@respx.mock
def test_video_edit_aleph_uses_keyframe_contract(monkeypatch):
    route = respx.post("https://api.test/v1/video_edit").mock(
        return_value=pending_response()
    )
    keyframe = json.dumps(
        {
            "image_url": str(FIXTURES / "sample.png"),
            "at": 0.5,
            "range": {"start_seconds": 1, "end_seconds": 4},
        }
    )

    result = invoke(
        [
            "--output-json",
            "--no-wait",
            "video",
            "edit",
            "--model",
            "aleph2",
            "--video",
            str(FIXTURES / "result-1.mp4"),
            "--keyframe",
            keyframe,
            "--seed",
            "321",
            "--aspect-ratio",
            "3:2",
            "--public-figure-threshold",
            "low",
        ],
        monkeypatch,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(route.calls[0].request.content)
    assert "prompt" not in payload
    assert "content" not in payload
    assert payload["model"] == "aleph2"
    assert payload["video_url"].startswith("data:video/mp4;base64,")
    assert payload["keyframes"][0]["image_url"].startswith("data:image/png;base64,")
    assert payload["keyframes"][0]["range"] == {
        "start_seconds": 1,
        "end_seconds": 4,
    }
    assert payload["seed"] == 321
    assert payload["target_aspect_ratio"] == "3:2"
    assert payload["public_figure_threshold"] == "low"


@pytest.mark.parametrize(
    ("option", "filename", "content_key", "role"),
    [
        ("--video", "result-1.mp4", "video_url", "reference_video"),
        ("--audio", "result-2.mp3", "audio_url", "reference_audio"),
    ],
)
@respx.mock
def test_video_edit_builds_reference_media(
    option, filename, content_key, role, monkeypatch
):
    route = respx.post("https://api.test/v1/video_edit").mock(
        return_value=pending_response()
    )
    args = [
        "--output-json",
        "--no-wait",
        "video",
        "edit",
        "edit",
        "--image",
        "https://img.test/ref.png",
        option,
        str(FIXTURES / filename),
    ]

    result = invoke(args, monkeypatch)

    assert result.exit_code == 0, result.output
    item = json.loads(route.calls[0].request.content)["content"][2]
    assert item["role"] == role
    assert item[content_key]["url"].startswith("data:")


@respx.mock
def test_video_edit_preserves_generate_audio_false_and_jq(monkeypatch):
    route = respx.post("https://api.test/v1/video_edit").mock(
        return_value=pending_response()
    )

    result = invoke(
        [
            "--no-wait",
            "video",
            "edit",
            "edit",
            "--image",
            "https://img.test/ref.png",
            "--no-generate-audio",
            "--jq",
            ".task_id",
        ],
        monkeypatch,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == "edit-123"
    assert json.loads(route.calls[0].request.content)["generate_audio"] is False


@respx.mock
def test_video_edit_wait_downloads_task_result(monkeypatch, make_temp_dir):
    target = make_temp_dir("success")
    respx.post("https://api.test/v1/video_edit").mock(return_value=pending_response())
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "edit-123",
                "status": "SUCCESS",
                "result": "https://cdn.test/edit",
            },
        )
    )
    respx.get("https://cdn.test/edit").mock(
        return_value=httpx.Response(
            200, content=b"video", headers={"content-type": "video/mp4"}
        )
    )

    result = invoke(
        [
            "--output-json",
            "--poll-interval",
            "0.001",
            "video",
            "edit",
            "edit",
            "--image",
            "https://img.test/ref.png",
            "--wait",
            "--download",
            str(target),
        ],
        monkeypatch,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["downloaded"] == [str(target / "edit-123.mp4")]


@respx.mock
def test_video_edit_download_uses_mp4_fallback_without_response_metadata(
    monkeypatch, make_temp_dir
):
    target = make_temp_dir("fallback-extension")
    respx.post("https://api.test/v1/video_edit").mock(return_value=pending_response())
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "edit-123",
                "status": "SUCCESS",
                "result": "https://cdn.test/result",
            },
        )
    )
    respx.get("https://cdn.test/result").mock(
        return_value=httpx.Response(200, content=b"video")
    )

    result = invoke(
        [
            "--output-json",
            "--poll-interval",
            "0.001",
            "video",
            "edit",
            "edit",
            "--image",
            "https://img.test/ref.png",
            "--download",
            str(target),
        ],
        monkeypatch,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["downloaded"] == [str(target / "edit-123.mp4")]


@respx.mock
def test_video_edit_failure_does_not_create_download(monkeypatch, tmp_path):
    target = tmp_path / "failure"
    respx.post("https://api.test/v1/video_edit").mock(return_value=pending_response())
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "edit-123",
                "status": "FAILURE",
                "error": "Provider rejected input",
            },
        )
    )

    result = invoke(
        [
            "--output-json",
            "--poll-interval",
            "0.001",
            "video",
            "edit",
            "edit",
            "--image",
            "https://img.test/ref.png",
            "--wait",
            "--download",
            str(target),
        ],
        monkeypatch,
    )

    assert result.exit_code == 4
    assert json.loads(result.stdout)["error"]["type"] == "TaskFailedError"
    assert not target.exists()


def test_video_edit_rejects_text_only_and_invalid_local_video(monkeypatch):
    text_only = invoke(
        ["--output-json", "video", "edit", "edit", "--no-wait"], monkeypatch
    )
    invalid_video = invoke(
        [
            "--output-json",
            "video",
            "edit",
            "edit",
            "--video",
            str(FIXTURES / "sample.png"),
            "--no-wait",
        ],
        monkeypatch,
    )

    assert text_only.exit_code == 2
    assert json.loads(text_only.stdout)["error"]["code"] == "validation_error"
    assert invalid_video.exit_code == 2
    assert "MP4/MOV" in json.loads(invalid_video.stdout)["error"]["message"]


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "text", "text": "edit"}],
        [
            {"type": "text", "text": "edit"},
            {
                "type": "audio_url",
                "role": "reference_audio",
                "audio_url": {"url": "https://cdn.test/audio.mp3"},
            },
        ],
    ],
)
@respx.mock
def test_generic_run_video_edit_applies_specialized_validation(content, monkeypatch):
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "src"
            / "everypixel_cli"
            / "resources"
            / "openapi.json"
        ).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(cli, "load_schema", lambda *_: (schema, "bundled"))
    route = respx.post("https://api.test/v1/video_edit").mock(
        return_value=pending_response()
    )

    result = invoke(
        [
            "--output-json",
            "--no-wait",
            "run",
            "video_edit",
            "-i",
            "prompt=edit",
            "-i",
            "model=seedance2",
            "-i",
            f"content={json.dumps(content)}",
        ],
        monkeypatch,
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "validation_error"
    assert not route.called


@respx.mock
def test_generic_run_video_edit_matches_specialized_payload(monkeypatch):
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "src"
            / "everypixel_cli"
            / "resources"
            / "openapi.json"
        ).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(cli, "load_schema", lambda *_: (schema, "bundled"))
    route = respx.post("https://api.test/v1/video_edit").mock(
        side_effect=[pending_response(), pending_response()]
    )
    content = [
        {"type": "text", "text": "edit"},
        {
            "type": "image_url",
            "role": "reference_image",
            "image_url": {"url": str(FIXTURES / "sample.png")},
        },
    ]

    specialized = invoke(
        [
            "--no-wait",
            "video",
            "edit",
            "edit",
            "--image",
            str(FIXTURES / "sample.png"),
        ],
        monkeypatch,
    )
    generic = invoke(
        [
            "--no-wait",
            "run",
            "video_edit",
            "-i",
            "prompt=edit",
            "-i",
            f"content={json.dumps(content)}",
            "-i",
            "model=seedance2",
            "-i",
            "duration=5",
            "-i",
            "resolution=720p",
            "-i",
            "aspect_ratio=16:9",
            "-i",
            "generate_audio=true",
        ],
        monkeypatch,
    )

    assert specialized.exit_code == 0, specialized.output
    assert generic.exit_code == 0, generic.output
    specialized_payload = json.loads(route.calls[0].request.content)
    generic_payload = json.loads(route.calls[1].request.content)
    assert generic_payload == specialized_payload


@respx.mock
def test_generic_run_video_edit_waits_and_downloads(monkeypatch, make_temp_dir):
    target = make_temp_dir("generic-success")
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "src"
            / "everypixel_cli"
            / "resources"
            / "openapi.json"
        ).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(cli, "load_schema", lambda *_: (schema, "bundled"))
    respx.post("https://api.test/v1/video_edit").mock(return_value=pending_response())
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "edit-123",
                "status": "SUCCESS",
                "result": "https://cdn.test/edit",
            },
        )
    )
    respx.get("https://cdn.test/edit").mock(
        return_value=httpx.Response(
            200, content=b"video", headers={"content-type": "video/mp4"}
        )
    )
    content = [
        {"type": "text", "text": "edit"},
        {
            "type": "image_url",
            "role": "reference_image",
            "image_url": {"url": "https://img.test/ref.png"},
        },
    ]

    result = invoke(
        [
            "--output-json",
            "--poll-interval",
            "0.001",
            "run",
            "video_edit",
            "-i",
            "prompt=edit",
            "-i",
            "model=seedance2",
            "-i",
            f"content={json.dumps(content)}",
            "--wait",
            "--download",
            str(target),
        ],
        monkeypatch,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["downloaded"] == [str(target / "edit-123.mp4")]
