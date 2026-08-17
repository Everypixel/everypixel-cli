import json

import httpx
import pytest
import respx
from typer.testing import CliRunner

from everypixel_cli.cli import app
from everypixel_cli.application.services import raise_for_task_failure
from everypixel_cli.errors import TaskFailedError


runner = CliRunner()


def invoke_json(args: list[str], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "test-id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "test-secret")
    return runner.invoke(
        app, ["--base-url", "https://api.test", "--output-json", *args]
    )


def failed_payload(error: object = "Input was rejected") -> dict:
    return {"task_id": "abc123", "status": "FAILURE", "error": error}


def assert_task_failure_json(result) -> dict:
    assert result.exit_code == 4
    assert result.stderr == ""
    assert "Traceback" not in result.output
    assert "\x1b[" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["task"] == {"id": "abc123", "status": "FAILURE"}
    assert payload["error"]["type"] == "TaskFailedError"
    assert payload["error"]["code"] == "task_failed"
    return payload


@respx.mock
def test_status_failure_human_shows_task_status_and_api_message(monkeypatch):
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "test-id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "test-secret")
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json=failed_payload(
                {"message": "Input was rejected", "code": "content_policy_violation"}
            ),
        )
    )

    result = runner.invoke(app, ["--base-url", "https://api.test", "status", "abc123"])

    assert result.exit_code == 4
    assert "Error [task_failed]: Input was rejected" in result.stderr
    assert "Task Id: abc123" in result.stderr
    assert "Status: FAILURE" in result.stderr
    assert "Api Error Code: content_policy_violation" in result.stderr
    assert "Traceback" not in result.output


@respx.mock
def test_status_failure_json_preserves_nested_error(monkeypatch):
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json=failed_payload(
                {
                    "detail": {"reason": "Model is unavailable"},
                    "error_code": "model_unavailable",
                    "details": {"retryable": True},
                }
            ),
        )
    )

    result = invoke_json(["status", "abc123"], monkeypatch)

    payload = assert_task_failure_json(result)
    assert payload["error"]["message"] == "Model is unavailable"
    assert payload["error"]["details"]["api_error_code"] == "model_unavailable"
    assert payload["error"]["details"]["api_details"] == {"retryable": True}


@respx.mock
def test_status_failure_without_api_message_uses_fallback(monkeypatch):
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(200, json=failed_payload({"code": "unknown"}))
    )

    payload = assert_task_failure_json(invoke_json(["status", "abc123"], monkeypatch))

    assert payload["error"]["message"] == "Task failed"


@respx.mock
def test_failure_details_mask_credentials_and_signed_url(monkeypatch):
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json=failed_payload(
                {
                    "message": "Upload failed",
                    "details": {
                        "token": "do-not-show",
                        "source": "https://cdn.test/file.png?signature=private",
                    },
                }
            ),
        )
    )

    result = invoke_json(["status", "abc123"], monkeypatch)

    payload = assert_task_failure_json(result)
    details = payload["error"]["details"]["api_details"]
    assert details["token"] == "***"
    assert details["source"] == "https://cdn.test/file.png"
    assert "do-not-show" not in result.output
    assert "signature=private" not in result.output


@respx.mock
def test_wait_stops_on_failure_without_download(monkeypatch, tmp_path):
    output_dir = tmp_path / "wait-download"
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(200, json=failed_payload("Render failed"))
    )

    result = invoke_json(["wait", "abc123", "--download", str(output_dir)], monkeypatch)

    payload = assert_task_failure_json(result)
    assert payload["error"]["message"] == "Render failed"
    assert not output_dir.exists()


@respx.mock
def test_generic_run_wait_stops_on_failure(monkeypatch):
    respx.post("https://api.test/v1/custom").mock(
        return_value=httpx.Response(
            200, json={"task_id": "abc123", "status": "PENDING"}
        )
    )
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(200, json=failed_payload("Generation failed"))
    )

    result = invoke_json(["run", "/v1/custom", "--wait"], monkeypatch)

    assert assert_task_failure_json(result)["error"]["message"] == "Generation failed"


@respx.mock
def test_async_command_wait_stops_on_failure(monkeypatch):
    respx.post("https://api.test/v1/image_generate").mock(
        return_value=httpx.Response(
            200, json={"task_id": "abc123", "status": "PENDING"}
        )
    )
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(200, json=failed_payload("Generation failed"))
    )

    result = invoke_json(
        ["image", "generate", "test prompt", "--model", "flux2", "--wait"],
        monkeypatch,
    )

    assert assert_task_failure_json(result)["error"]["message"] == "Generation failed"


def test_non_terminal_status_remains_a_regular_status():
    payload = {"task_id": "abc123", "status": "PENDING", "queue": 2}

    assert raise_for_task_failure(payload, fallback_task_id="abc123") is None


def test_task_failed_error_keeps_task_data():
    error = TaskFailedError("Render failed", task_id="abc123")

    assert error.to_payload()["task"] == {"id": "abc123", "status": "FAILURE"}


@respx.mock
def test_status_success_download_remains_available(monkeypatch, make_temp_dir):
    output_dir = make_temp_dir("status-success-download")
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200,
            json={
                "task_id": "abc123",
                "status": "SUCCESS",
                "result": "https://cdn.test/result",
            },
        )
    )
    respx.get("https://cdn.test/result").mock(
        return_value=httpx.Response(
            200, content=b"image", headers={"content-type": "image/png"}
        )
    )

    result = invoke_json(
        ["status", "abc123", "--download", str(output_dir)], monkeypatch
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["downloaded"] == [str(output_dir / "abc123.png")]
