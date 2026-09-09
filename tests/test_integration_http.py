import httpx
import json
import pytest
import respx
from pathlib import Path
from typer.testing import CliRunner

from everypixel_cli.cli import app
from everypixel_cli.client import APIClient
from everypixel_cli.errors import CLIError


FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


def invoke_json(args, monkeypatch):
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    result = runner.invoke(
        app,
        ["--base-url", "https://api.test", "--output-json", "--no-wait", *args],
    )
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def mock_async_post(path: str):
    return respx.post(f"https://api.test{path}").mock(
        return_value=httpx.Response(200, json={"task_id": "abc", "status": "PENDING"})
    )


def request_json(route):
    return json.loads(route.calls[0].request.content)


def request_multipart_body(route):
    return route.calls[0].request.content


@pytest.mark.parametrize(
    ("model", "canonical"),
    [
        ("grok", "grok-imagine"),
        ("grok-imagine", "grok-imagine"),
        ("grok-imagine-2", "grok-imagine-2"),
        ("grok-imagine-2-low", "grok-imagine-2-low"),
    ],
)
@respx.mock
def test_grok_image_models_and_generic_aliases(model, canonical, monkeypatch):
    route = mock_async_post("/v1/image_generate")
    invoke_json(
        ["image", "generate", "a lake", "--model", model, "--resolution", "2k"],
        monkeypatch,
    )
    specialized = request_json(route)
    assert specialized["model"] == canonical
    respx.get("https://api.test/v1/openapi.json").respond(
        200,
        json=json.loads(
            (
                Path(__file__).parents[1] / "src/everypixel_cli/resources/openapi.json"
            ).read_text()
        ),
    )
    invoke_json(
        [
            "run",
            "image_generate",
            "--input",
            f"model={model}",
            "--input",
            "prompt=a lake",
            "--input",
            "resolution=2k",
            "--input",
            "future_option=false",
        ],
        monkeypatch,
    )
    generic = json.loads(route.calls[1].request.content)
    assert generic["model"] == canonical
    assert generic["future_option"] is False


@pytest.mark.parametrize("model", ["topaz-prob-4", "topaz-slp-2.5", "topaz-ast-2"])
@respx.mock
def test_topaz_upscale_4k(model, monkeypatch):
    route = mock_async_post("/v1/video_upscale")
    invoke_json(
        [
            "video",
            "upscale",
            "--video",
            str(FIXTURES / "result-1.mp4"),
            "--model",
            model,
            "--resolution",
            "4k",
        ],
        monkeypatch,
    )
    body = request_json(route)
    assert body["model"] == model
    assert body["resolution"] == "4k"
    assert body["video_url"].startswith("data:video/mp4;base64,")


@pytest.mark.parametrize(
    "mode", ["recraftv4_1_vector", "recraftv4_1_pro_vector", "vectorize"]
)
@respx.mock
def test_vector_images_download_svg(mode, monkeypatch, make_temp_dir):
    endpoint = "/v1/image_vectorize" if mode == "vectorize" else "/v1/image_generate"
    route = mock_async_post(endpoint)
    status = respx.get("https://api.test/v1/status").mock(
        side_effect=[
            httpx.Response(200, json={"task_id": "abc", "status": "PENDING"}),
            httpx.Response(
                200,
                json={
                    "task_id": "abc",
                    "status": "SUCCESS",
                    "result": "https://cdn.test/vector",
                },
            ),
        ]
    )
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"/>'
    download = respx.get("https://cdn.test/vector").respond(200, content=svg)
    folder = make_temp_dir("vectors")
    if mode == "vectorize":
        args = ["image", "vectorize", str(FIXTURES / "sample.png")]
    else:
        args = [
            "image",
            "generate",
            "a mountain icon",
            "--model",
            mode,
            "--controls",
            '{"colors":[{"rgb":[0,128,255],"weight":0}]}',
            "--seed",
            "0",
        ]
    invoke_json(
        ["--poll-interval", "0.01", *args, "--download", str(folder)], monkeypatch
    )
    assert status.call_count == 2
    assert download.call_count == 1
    files = list(folder.iterdir())
    assert len(files) == 1 and files[0].suffix == ".svg"
    assert files[0].read_bytes() == svg
    body = request_json(route)
    if mode == "vectorize":
        assert body["image_url"].startswith("data:image/png;base64,")
    else:
        assert body["controls"]["colors"][0] == {"rgb": [0, 128, 255], "weight": 0}
        assert body["seed"] == 0
        assert "resolution" not in body


@pytest.mark.parametrize("command", ["tts-create", "tts-clone", "tts-voice"])
@pytest.mark.parametrize("source", ["text", "file", "generic"])
@respx.mock
def test_tts_sends_full_text_without_local_limit(
    command, source, monkeypatch, make_temp_dir
):
    route = mock_async_post(f"/v1/{command.replace('-', '_')}")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")

    def arguments(text):
        audio = (
            ["--audio", str(FIXTURES / "result-2.mp3")]
            if command == "tts-clone"
            else []
        )
        if source == "generic":
            return [
                "run",
                f"/v1/{command.replace('-', '_')}",
                "--input",
                f"text={text}",
                "--input",
                "audio_url=https://cdn.test/audio.mp3",
            ]
        if source == "file":
            path = folder / "text.txt"
            path.write_text(text, encoding="utf-8")
            return ["audio", command, *audio, "--text-file", str(path)]
        return ["audio", command, *audio, "--text", text]

    folder = make_temp_dir("tts-text")
    text = "я" * 200 + " "
    invoke_json(arguments(text), monkeypatch)
    assert request_json(route)["text"] == text
    assert route.call_count == 1


@pytest.mark.parametrize(
    "args",
    [
        ["image", "generate", "x", "--model", "wan22"],
        ["image", "generate", "x", "--model", "grok_quality"],
        ["image", "generate", "x", "--style", "instagram"],
        ["image", "generate", "x", "--controls", "{}"],
        [
            "image",
            "generate",
            "x",
            "--model",
            "recraftv4_1_vector",
            "--resolution",
            "2k",
        ],
        [
            "image",
            "generate",
            "x",
            "--model",
            "recraftv4_1_vector",
            "--controls",
            '{"colors":[{"rgb":[0,0,0],"weight":0.7},{"rgb":[255,255,255],"weight":0.4}]}',
        ],
        [
            "image",
            "edit",
            "x",
            "--image",
            "https://img.test/a.png",
            "--megapixel-ratio",
            "1.6",
        ],
        ["video", "upscale", "--task-id", "abc", "--resolution", "4k"],
        ["video", "generate", "x", "--model", "minimax-h3", "--duration", "2"],
        ["video", "generate", " ", "--model", "minimax-h3"],
        ["video", "generate", "x", "--model", "minimax-h3", "--no-generate-audio"],
        ["video", "generate", "x", "--model", "seedance2", "--duration", "30"],
        ["video", "generate", "x", "--model", "seedance2.5", "--resolution", "4k"],
        ["video", "generate", "x", "--model", "grok15"],
        [
            "video",
            "edit",
            "x",
            "--model",
            "flux3",
            "--video",
            "https://cdn.test/v.mp4",
            "--duration",
            "20",
        ],
        [
            "video",
            "edit",
            "x",
            "--model",
            "flux3",
            "--video",
            "https://cdn.test/v.mp4",
            "--image",
            "https://img.test/a.png",
        ],
    ],
)
@respx.mock
def test_updated_contract_rejects_invalid_cli_inputs_before_http(args):
    result = runner.invoke(
        app, ["--base-url", "https://api.test", "-j", "--no-wait", *args]
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "validation_error"
    assert result.stderr == ""
    assert not respx.calls


@respx.mock
def test_basic_auth_header_sent():
    route = respx.post("https://api.test/v1/image_generate").mock(
        return_value=httpx.Response(200, json={"task_id": "abc", "status": "PENDING"})
    )

    client = APIClient(
        base_url="https://api.test", client_id="id", client_secret="secret"
    )
    response = client.request("POST", "/v1/image_generate", json={"prompt": "x"})

    assert response["task_id"] == "abc"
    assert route.calls[0].request.headers["authorization"].startswith("Basic ")


@respx.mock
def test_status_success_response():
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "abc",
                "status": "SUCCESS",
                "result": "https://cdn.test/out.png",
            },
        )
    )

    client = APIClient(base_url="https://api.test", client_id=None, client_secret=None)

    assert client.get_status("abc")["status"] == "SUCCESS"


@respx.mock
def test_auth_check_uses_dedicated_endpoint():
    route = respx.get("https://api.test/v1/auth/check").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )

    client = APIClient(
        base_url="https://api.test", client_id="id", client_secret="secret"
    )

    client.check_auth()

    assert route.called


@respx.mock
def test_auth_check_does_not_fall_back_when_endpoint_is_missing():
    respx.get("https://api.test/v1/auth/check").mock(
        return_value=httpx.Response(404, json={"detail": "Not found"})
    )
    fallback = respx.get("https://api.test/v1/quality").mock(
        return_value=httpx.Response(200, json={"status": True})
    )

    client = APIClient(
        base_url="https://api.test", client_id="id", client_secret="secret"
    )

    with pytest.raises(CLIError) as exc_info:
        client.check_auth()

    assert exc_info.value.code == "not_found"
    assert not fallback.called


@respx.mock
def test_classic_ml_url_request_uses_get_params():
    route = respx.get("https://api.test/v1/keywords").mock(
        return_value=httpx.Response(200, json={"status": "ok", "keywords": []})
    )

    client = APIClient(
        base_url="https://api.test", client_id="id", client_secret="secret"
    )
    response = client.request(
        "GET",
        "/v1/keywords",
        params={"url": "https://img.test/sample.jpg", "lang": "ru"},
    )

    assert response["status"] == "ok"
    assert route.calls[0].request.url.params["url"] == "https://img.test/sample.jpg"
    assert route.calls[0].request.url.params["lang"] == "ru"


@respx.mock
def test_classic_ml_local_file_request_uses_multipart_data_field():
    route = respx.post("https://api.test/v1/keywords").mock(
        return_value=httpx.Response(200, json={"status": "ok", "keywords": []})
    )

    client = APIClient(
        base_url="https://api.test", client_id="id", client_secret="secret"
    )
    with (FIXTURES / "sample.png").open("rb") as file_handle:
        response = client.request(
            "POST", "/v1/keywords", files={"data": ("sample.png", file_handle)}
        )

    assert response["status"] == "ok"
    body = route.calls[0].request.content
    assert b'name="data"; filename="sample.png"' in body
    assert b"png-bytes" in body


@respx.mock
def test_keywords_cli_url_uses_get_with_params(monkeypatch):
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    route = respx.get("https://api.test/v1/keywords").mock(
        return_value=httpx.Response(
            200, json={"status": "ok", "keywords": [{"keyword": "cat", "score": 0.9}]}
        )
    )

    payload = invoke_json(
        [
            "keywords",
            "--image",
            "https://img.test/sample.jpg",
            "--lang",
            "ru",
            "--num-keywords",
            "10",
            "--colors",
        ],
        monkeypatch,
    )

    params = route.calls[0].request.url.params
    assert payload["keywords"][0]["keyword"] == "cat"
    assert params["url"] == "https://img.test/sample.jpg"
    assert params["lang"] == "ru"
    assert params["num_keywords"] == "10"
    assert params["colors"] == "true"


@respx.mock
def test_quality_cli_local_file_uses_multipart(monkeypatch):
    route = respx.post("https://api.test/v1/quality").mock(
        return_value=httpx.Response(
            200, json={"status": "ok", "quality": {"score": 0.5}}
        )
    )

    payload = invoke_json(
        ["quality", "--image", str(FIXTURES / "sample.png")], monkeypatch
    )

    assert payload["quality"]["score"] == 0.5
    assert b'name="data"; filename="sample.png"' in request_multipart_body(route)


@respx.mock
def test_quality_ugc_cli_url_uses_get(monkeypatch):
    route = respx.get("https://api.test/v1/quality_ugc").mock(
        return_value=httpx.Response(
            200, json={"status": "ok", "quality": {"score": 0.7, "class": "high"}}
        )
    )

    payload = invoke_json(
        ["quality-ugc", "--image", "https://img.test/sample.jpg"], monkeypatch
    )

    assert payload["quality"]["class"] == "high"
    assert route.calls[0].request.url.params["url"] == "https://img.test/sample.jpg"


@respx.mock
def test_faces_cli_local_file_uses_multipart(monkeypatch):
    route = respx.post("https://api.test/v1/faces").mock(
        return_value=httpx.Response(
            200, json={"status": "ok", "faces": [{"score": 0.91}]}
        )
    )

    payload = invoke_json(
        ["faces", "--image", str(FIXTURES / "sample.png")], monkeypatch
    )

    assert payload["faces"][0]["score"] == 0.91
    assert b'name="data"; filename="sample.png"' in request_multipart_body(route)


@respx.mock
def test_video_keywords_cli_local_file_uses_multipart(monkeypatch):
    route = respx.post("https://api.test/v1/video_keywords").mock(
        return_value=httpx.Response(
            200,
            json={"status": "ok", "keywords": [{"keyword": "motion", "score": 0.8}]},
        )
    )

    payload = invoke_json(
        ["video-keywords", "--video", str(FIXTURES / "sample.png")], monkeypatch
    )

    assert payload["keywords"][0]["keyword"] == "motion"
    assert b'name="data"; filename="sample.png"' in request_multipart_body(route)


@respx.mock
def test_captioning_cli_url_uses_get(monkeypatch):
    route = respx.get("https://api.test/v1/image_captioning").mock(
        return_value=httpx.Response(
            200, json={"status": True, "result": {"caption": "A cat"}}
        )
    )

    payload = invoke_json(
        ["captioning", "--image", "https://img.test/sample.jpg"], monkeypatch
    )

    assert payload["result"]["caption"] == "A cat"
    assert route.calls[0].request.url.params["url"] == "https://img.test/sample.jpg"


@respx.mock
def test_generic_run_raw_path_mode_posts_without_openapi_lookup(monkeypatch):
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    route = respx.post("https://api.test/v1/custom").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "--output-json",
            "run",
            "/v1/custom",
            "--method",
            "POST",
            "-i",
            "count=1",
        ],
    )

    assert result.exit_code == 0
    assert '"status": "ok"' in result.stdout
    assert route.calls[0].request.content == b'{"count":1}'


def test_generic_run_merge_order_in_dry_run():
    result = runner.invoke(
        app,
        [
            "--output-json",
            "run",
            "/v1/custom",
            "--method",
            "POST",
            "--input-file",
            str(FIXTURES / "payload.json"),
            "-i",
            "count=1",
            "-p",
            "from-prompt",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["body"] == {"prompt": "from-prompt", "count": 1}


@respx.mock
def test_generic_run_no_wait_returns_created_task(monkeypatch):
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    respx.post("https://api.test/v1/image_generate").mock(
        return_value=httpx.Response(200, json={"task_id": "abc", "status": "PENDING"})
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "run",
            "/v1/image_generate",
            "--method",
            "POST",
            "-i",
            "prompt=woman",
            "--no-wait",
            "--output-json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"task_id": "abc", "status": "PENDING"}


@respx.mock
def test_generic_run_waits_by_default(monkeypatch):
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    respx.post("https://api.test/v1/image_generate").mock(
        return_value=httpx.Response(200, json={"task_id": "abc", "status": "PENDING"})
    )
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "abc",
                "status": "SUCCESS",
                "result": "https://cdn.test/out.png",
            },
        )
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "--poll-interval",
            "0.001",
            "run",
            "/v1/image_generate",
            "--method",
            "POST",
            "-i",
            "prompt=woman",
            "--output-json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "SUCCESS"


@respx.mock
def test_generic_run_wait_output_json(monkeypatch):
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    respx.post("https://api.test/v1/image_generate").mock(
        return_value=httpx.Response(200, json={"task_id": "abc", "status": "PENDING"})
    )
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "abc",
                "status": "SUCCESS",
                "result": "https://cdn.test/out.png",
            },
        )
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "--poll-interval",
            "0.001",
            "run",
            "/v1/image_generate",
            "--method",
            "POST",
            "-i",
            "prompt=woman",
            "--wait",
            "--output-json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["result"] == "https://cdn.test/out.png"


@respx.mock
def test_generic_run_wait_download(monkeypatch, make_temp_dir):
    output_dir = make_temp_dir("generic-wait-download")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    respx.post("https://api.test/v1/image_generate").mock(
        return_value=httpx.Response(200, json={"task_id": "abc", "status": "PENDING"})
    )
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "abc",
                "status": "SUCCESS",
                "result": "https://cdn.test/out",
            },
        )
    )
    respx.get("https://cdn.test/out").mock(
        return_value=httpx.Response(
            200, content=b"generated-image", headers={"content-type": "image/png"}
        )
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "--poll-interval",
            "0.001",
            "run",
            "/v1/image_generate",
            "--method",
            "POST",
            "-i",
            "prompt=woman",
            "--wait",
            "--download",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "abc.png").read_bytes() == b"generated-image"


@respx.mock
def test_generic_run_wait_download_output_json(monkeypatch, make_temp_dir):
    output_dir = make_temp_dir("generic-wait-download-json")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    respx.post("https://api.test/v1/image_generate").mock(
        return_value=httpx.Response(200, json={"task_id": "abc", "status": "PENDING"})
    )
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "abc",
                "status": "SUCCESS",
                "result": "https://cdn.test/out",
            },
        )
    )
    respx.get("https://cdn.test/out").mock(
        return_value=httpx.Response(
            200, content=b"generated-image", headers={"content-type": "image/png"}
        )
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "--poll-interval",
            "0.001",
            "run",
            "/v1/image_generate",
            "--method",
            "POST",
            "-i",
            "prompt=woman",
            "--wait",
            "--download",
            str(output_dir),
            "--output-json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["downloaded"] == [str(output_dir / "abc.png")]


@respx.mock
def test_generic_run_jq(monkeypatch):
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    respx.post("https://api.test/v1/image_generate").mock(
        return_value=httpx.Response(200, json={"task_id": "abc", "status": "PENDING"})
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "run",
            "/v1/image_generate",
            "--method",
            "POST",
            "-i",
            "prompt=woman",
            "--no-wait",
            "--jq",
            ".task_id",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == "abc"


@respx.mock
def test_image_generate_payload(monkeypatch):
    route = mock_async_post("/v1/image_generate")

    payload = invoke_json(
        [
            "image",
            "generate",
            "product photo",
            "--model",
            "flux2",
            "--size",
            "landscape_4_3",
            "--style",
            "transparent",
            "--seed",
            "123",
        ],
        monkeypatch,
    )

    assert payload == {"task_id": "abc", "status": "PENDING"}
    assert request_json(route) == {
        "prompt": "product photo",
        "model": "flux2",
        "image_size": "landscape_4_3",
        "style": "transparent",
        "resolution": "1k",
        "seed": 123,
    }


@respx.mock
def test_seedream_image_generate_uses_provider_default_resolution(monkeypatch):
    route = mock_async_post("/v1/image_generate")

    invoke_json(
        ["image", "generate", "product photo", "--model", "seedream-5"],
        monkeypatch,
    )

    assert request_json(route)["resolution"] == "2k"


@respx.mock
def test_wan_image_generate_uses_provider_default_resolution(monkeypatch):
    route = mock_async_post("/v1/image_generate")

    invoke_json(
        ["image", "generate", "product photo", "--model", "wan2.7"],
        monkeypatch,
    )

    body = request_json(route)
    assert body["model"] == "wan2.7"
    assert body["resolution"] == "2k"


@pytest.mark.parametrize(
    ("model", "count", "resolution", "size"),
    [("gpt-image-2-high", 16, "3k", "square"), ("gemini-3-pro", 14, "4k", None)],
)
@respx.mock
def test_image_edit_accepts_expanded_input_limits(
    model, count, resolution, size, monkeypatch
):
    route = mock_async_post("/v1/image_edit")

    invoke_json(
        [
            "image",
            "edit",
            "make it brighter",
            *[
                value
                for _ in range(count)
                for value in ("--image", "https://img.test/source.png")
            ],
            "--model",
            model,
            "--resolution",
            resolution,
            "--megapixel-ratio",
            "1.5",
        ],
        monkeypatch,
    )

    body = request_json(route)
    assert body["model"] == model
    assert body["resolution"] == resolution
    assert body.get("image_size") == size
    assert len(body["image_urls"]) == count
    assert body["megapixel_ratio"] == 1.5


@respx.mock
def test_common_output_flag_before_command(monkeypatch):
    route = mock_async_post("/v1/image_generate")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "--output-json",
            "--no-wait",
            "image",
            "generate",
            "woman smiling",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"task_id": "abc", "status": "PENDING"}
    assert request_json(route) == {
        "prompt": "woman smiling",
        "model": "zimage",
        "style": "portrait",
        "image_size": "square",
        "resolution": "1k",
        "seed": -1,
    }


@respx.mock
def test_common_output_flag_after_command(monkeypatch):
    route = mock_async_post("/v1/image_generate")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "image",
            "generate",
            "woman smiling",
            "--model",
            "flux2",
            "--no-wait",
            "--output-json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["task_id"] == "abc"
    assert request_json(route)["prompt"] == "woman smiling"


@respx.mock
def test_multiple_common_flags_after_command(monkeypatch, make_temp_dir):
    output_dir = make_temp_dir("multiple-common-flags")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    mock_async_post("/v1/image_generate")
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "abc",
                "status": "SUCCESS",
                "result": "https://cdn.test/out",
            },
        )
    )
    respx.get("https://cdn.test/out").mock(
        return_value=httpx.Response(
            200, content=b"generated-image", headers={"content-type": "image/png"}
        )
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "--poll-interval",
            "0.001",
            "image",
            "generate",
            "woman smiling",
            "--model",
            "flux2",
            "--download",
            str(output_dir),
            "--output-json",
            "--jq",
            ".downloaded",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == [str(output_dir / "abc.png")]


@respx.mock
def test_image_generate_wait_polls_until_success(monkeypatch):
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    mock_async_post("/v1/image_generate")
    respx.get("https://api.test/v1/status").mock(
        side_effect=[
            httpx.Response(200, json={"task_id": "abc", "status": "PENDING"}),
            httpx.Response(
                200,
                json={
                    "task_id": "abc",
                    "status": "SUCCESS",
                    "result": "https://cdn.test/out.png",
                },
            ),
        ]
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "--output-json",
            "--poll-interval",
            "0.001",
            "image",
            "generate",
            "product photo",
            "--model",
            "flux2",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "SUCCESS"
    assert payload["result"] == "https://cdn.test/out.png"


@respx.mock
def test_image_generate_wait_downloads_result(monkeypatch, make_temp_dir):
    output_dir = make_temp_dir("image-generate-global-download")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    mock_async_post("/v1/image_generate")
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "abc",
                "status": "SUCCESS",
                "result": "https://cdn.test/out",
            },
        )
    )
    respx.get("https://cdn.test/out").mock(
        return_value=httpx.Response(
            200, content=b"generated-image", headers={"content-type": "image/png"}
        )
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "--output-json",
            "--poll-interval",
            "0.001",
            "--download",
            str(output_dir),
            "image",
            "generate",
            "product photo",
            "--model",
            "flux2",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["downloaded"] == [str(output_dir / "abc.png")]
    assert (output_dir / "abc.png").read_bytes() == b"generated-image"


@respx.mock
def test_image_generate_download_after_command_uses_task_id_extension(
    monkeypatch, make_temp_dir
):
    output_dir = make_temp_dir("image-generate-local-download")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    mock_async_post("/v1/image_generate")
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "abc",
                "status": "SUCCESS",
                "result": "https://cdn.test/out",
            },
        )
    )
    respx.get("https://cdn.test/out").mock(
        return_value=httpx.Response(
            200, content=b"generated-image", headers={"content-type": "image/png"}
        )
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "--poll-interval",
            "0.001",
            "image",
            "generate",
            "woman smiling",
            "--model",
            "flux2",
            "--download",
            str(output_dir),
            "--output-json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["downloaded"] == [str(output_dir / "abc.png")]
    assert (output_dir / "abc.png").read_bytes() == b"generated-image"


@respx.mock
def test_image_edit_payload_with_local_file(monkeypatch):
    route = mock_async_post("/v1/image_edit")

    invoke_json(
        [
            "image",
            "edit",
            "make it brighter",
            "--image",
            str(FIXTURES / "sample.png"),
            "--model",
            "qwen",
        ],
        monkeypatch,
    )

    body = request_json(route)
    assert body["prompt"] == "make it brighter"
    assert body["model"] == "qwen"
    assert body["image_urls"][0].startswith("data:image/png;base64,")


@respx.mock
def test_image_upscale_payload_from_task_id(monkeypatch):
    route = mock_async_post("/v1/image_upscale")

    invoke_json(
        ["image", "upscale", "--task-id", "source-task", "--model", "seedvr2"],
        monkeypatch,
    )

    assert request_json(route) == {
        "image_from_task_id": "source-task",
        "model": "seedvr2",
    }


@respx.mock
def test_image_angles_payload(monkeypatch):
    route = mock_async_post("/v1/image_edit_angles")

    invoke_json(
        [
            "image",
            "angles",
            "--image",
            "https://img.test/in.png",
            "--azimuth",
            "right",
            "--elevation",
            "eye_level",
            "--distance",
            "medium",
        ],
        monkeypatch,
    )

    assert request_json(route) == {
        "image_url": "https://img.test/in.png",
        "azimuth": "right",
        "elevation": "eye_level",
        "distance": "medium",
    }


@respx.mock
def test_image_colors_payload(monkeypatch):
    route = mock_async_post("/v1/image_edit_colors")

    invoke_json(
        [
            "image",
            "colors",
            "--image",
            "https://img.test/in.png",
            "--reference",
            "https://img.test/ref.png",
        ],
        monkeypatch,
    )

    assert request_json(route) == {
        "image_url": "https://img.test/in.png",
        "image_reference_url": "https://img.test/ref.png",
    }


@respx.mock
def test_video_generate_payload(monkeypatch):
    route = mock_async_post("/v1/video_generate")

    invoke_json(
        [
            "video",
            "generate",
            "drone shot",
            "--model",
            "ltx23",
            "--duration",
            "10",
            "--resolution",
            "720p",
            "--aspect-ratio",
            "16:9",
        ],
        monkeypatch,
    )

    assert request_json(route) == {
        "prompt": "drone shot",
        "model": "ltx23",
        "duration": 10,
        "resolution": "720p",
        "aspect_ratio": "16:9",
        "seed": -1,
    }


@pytest.mark.parametrize("model", ["minimax-h3-turbo", "minimax-h3"])
@respx.mock
def test_video_generate_minimax_reference_payload(model, monkeypatch):
    route = mock_async_post("/v1/video_generate")

    invoke_json(
        [
            "video",
            "generate",
            "product shot",
            "--model",
            model,
            "--reference-image",
            str(FIXTURES / "sample.png"),
            "--seed",
            "18446744073709551615",
            "--callback-url",
            "https://hooks.test/callback",
        ],
        monkeypatch,
    )

    body = request_json(route)
    assert body["model"] == model
    assert body["duration"] == 5
    assert body["resolution"] == "768p"
    assert body["aspect_ratio"] == "7:4"
    assert body["seed"] == 18446744073709551615
    assert body["reference_image_urls"][0].startswith("data:image/png;base64,")
    assert body["callback_url"] == "https://hooks.test/callback"


@respx.mock
def test_video_from_image_payload(monkeypatch):
    route = mock_async_post("/v1/video_generate")

    invoke_json(
        [
            "video",
            "from-image",
            "animate",
            "--image",
            "https://img.test/first.png",
            "--model",
            "ltx23",
            "--aspect-ratio",
            "9:16",
            "--seed",
            "42",
            "--callback-url",
            "https://hooks.test/callback",
        ],
        monkeypatch,
    )

    body = request_json(route)
    assert body["prompt"] == "animate"
    assert body["image_url"] == "https://img.test/first.png"
    assert body["aspect_ratio"] == "9:16"
    assert body["seed"] == 42
    assert body["callback_url"] == "https://hooks.test/callback"


@respx.mock
def test_video_first_last_payload(monkeypatch):
    route = mock_async_post("/v1/video_generate")

    invoke_json(
        [
            "video",
            "first-last",
            "transition",
            "--image",
            "https://img.test/first.png",
            "--last-image",
            "https://img.test/last.png",
            "--aspect-ratio",
            "1:1",
            "--seed",
            "7",
            "--callback-url",
            "https://hooks.test/callback",
        ],
        monkeypatch,
    )

    body = request_json(route)
    assert body["image_url"] == "https://img.test/first.png"
    assert body["image_last_url"] == "https://img.test/last.png"
    assert body["aspect_ratio"] == "1:1"
    assert body["seed"] == 7
    assert body["callback_url"] == "https://hooks.test/callback"


@respx.mock
def test_video_upscale_payload(monkeypatch):
    route = mock_async_post("/v1/video_upscale")

    invoke_json(
        ["video", "upscale", "--task-id", "source-task", "--resolution", "1080p"],
        monkeypatch,
    )

    assert request_json(route) == {
        "video_from_task_id": "source-task",
        "model": "seedvr2",
        "resolution": "1080p",
    }


@respx.mock
def test_lipsync_video_payload(monkeypatch):
    route = mock_async_post("/v1/video_lipsync")

    invoke_json(
        [
            "lipsync",
            "video",
            "--video",
            "https://cdn.test/in.mp4",
            "--audio",
            "https://cdn.test/voice.mp3",
            "--resolution",
            "720p",
            "--callback-url",
            "https://hooks.test/callback",
        ],
        monkeypatch,
    )

    assert request_json(route) == {
        "video_url": "https://cdn.test/in.mp4",
        "audio_url": "https://cdn.test/voice.mp3",
        "resolution": "720p",
        "seed": -1,
        "callback_url": "https://hooks.test/callback",
    }


@respx.mock
def test_lipsync_image_payload(monkeypatch):
    route = mock_async_post("/v1/image_lipsync")

    invoke_json(
        [
            "lipsync",
            "image",
            "--image",
            "https://img.test/face.jpg",
            "--audio",
            "https://cdn.test/voice.mp3",
            "--model",
            "inftalk",
            "--callback-url",
            "https://hooks.test/callback",
        ],
        monkeypatch,
    )

    body = request_json(route)
    assert body["image_url"] == "https://img.test/face.jpg"
    assert body["audio_url"] == "https://cdn.test/voice.mp3"
    assert body["model"] == "inftalk"
    assert body["callback_url"] == "https://hooks.test/callback"


@respx.mock
def test_audio_transcribe_payload(monkeypatch):
    route = mock_async_post("/v1/transcribe")

    invoke_json(
        [
            "audio",
            "transcribe",
            "--audio",
            "https://cdn.test/speech.mp3",
            "--language",
            "ru",
        ],
        monkeypatch,
    )

    assert request_json(route) == {
        "audio_url": "https://cdn.test/speech.mp3",
        "language": "ru",
        "hints": "",
        "denoise": True,
    }


@respx.mock
def test_audio_transcribe_downloads_plain_transcript_as_txt(monkeypatch, make_temp_dir):
    output_dir = make_temp_dir("asr-plain")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    mock_async_post("/v1/transcribe")
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200, json={"task_id": "abc", "status": "SUCCESS", "result": "hello world"}
        )
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "--poll-interval",
            "0.001",
            "audio",
            "transcribe",
            "--audio",
            "https://cdn.test/speech.mp3",
            "--download",
            str(output_dir),
            "--output-json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["downloaded"] == [str(output_dir / "abc.txt")]
    assert (output_dir / "abc.txt").read_text(encoding="utf-8") == "hello world"


@respx.mock
def test_audio_transcribe_structured_json_keeps_fields(monkeypatch, make_temp_dir):
    output_dir = make_temp_dir("asr-json")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")
    structured = {
        "text": "hello world",
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.2, "text": "hello world"}],
        "metadata": {"model": "asr"},
    }
    mock_async_post("/v1/transcribe")
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200, json={"task_id": "abc", "status": "SUCCESS", "result": structured}
        )
    )

    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "--poll-interval",
            "0.001",
            "audio",
            "transcribe",
            "--audio",
            "https://cdn.test/speech.mp3",
            "--download",
            str(output_dir),
            "--output-json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["result"]["segments"][0]["start"] == 0.0
    assert payload["result"]["language"] == "en"
    assert payload["downloaded"] == [str(output_dir / "abc.json")]
    saved = json.loads((output_dir / "abc.json").read_text(encoding="utf-8"))
    assert saved["metadata"]["model"] == "asr"


@respx.mock
def test_tts_create_payload(monkeypatch):
    route = mock_async_post("/v1/tts_create")

    invoke_json(
        [
            "audio",
            "tts-create",
            "--text",
            "Hello",
            "--speaker",
            "Female",
            "--style",
            "Warm",
            "--language",
            "en",
        ],
        monkeypatch,
    )

    assert request_json(route) == {
        "text": "Hello",
        "speaker": "Female",
        "style": "Warm",
        "language": "en",
        "prompt": "",
        "seed": -1,
    }


@respx.mock
def test_tts_clone_payload(monkeypatch):
    route = mock_async_post("/v1/tts_clone")

    invoke_json(
        [
            "audio",
            "tts-clone",
            "--audio",
            "https://cdn.test/sample.mp3",
            "--text",
            "Hello",
            "--language",
            "en",
        ],
        monkeypatch,
    )

    assert request_json(route) == {
        "audio_url": "https://cdn.test/sample.mp3",
        "text": "Hello",
        "language": "en",
        "seed": -1,
    }


@respx.mock
def test_tts_voice_payload(monkeypatch):
    route = mock_async_post("/v1/tts_voice")

    invoke_json(
        [
            "audio",
            "tts-voice",
            "--text",
            "Hello",
            "--character",
            "Narrator",
            "--style",
            "Storytelling",
            "--language",
            "en",
        ],
        monkeypatch,
    )

    assert request_json(route) == {
        "text": "Hello",
        "character": "Narrator",
        "style": "Storytelling",
        "language": "en",
        "prompt": "",
        "seed": -1,
    }
