import json

import httpx
import pytest
import respx
from typer.testing import CliRunner

from everypixel_cli.cli import app
from everypixel_cli.files import download_urls


runner = CliRunner()


def invoke_json(args: list[str], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "test-id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "test-secret")
    return runner.invoke(
        app, ["--base-url", "https://api.test", "--output-json", *args]
    )


def assert_json_error(result, code: str, error_type: str) -> dict:
    assert result.exit_code != 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == code
    assert payload["error"]["type"] == error_type
    assert "Traceback" not in result.output
    assert "\x1b[" not in result.stdout
    return payload


def test_missing_input_file_is_clean_in_human_mode():
    result = runner.invoke(app, ["keywords", "--image", "missing-image.png"])

    assert result.exit_code == 5
    assert "Error [file_not_found]: Unable to read input file" in result.stderr
    assert "Traceback" not in result.output


def test_missing_input_file_returns_json_error(monkeypatch):
    result = invoke_json(["keywords", "--image", "missing-image.png"], monkeypatch)

    payload = assert_json_error(result, "file_not_found", "FileReadError")
    assert payload["error"]["details"]["path"] == "missing-image.png"


def test_invalid_json_input_file_returns_json_error(monkeypatch, make_temp_dir):
    input_file = make_temp_dir("invalid-json") / "payload.json"
    input_file.write_text("{broken", encoding="utf-8")

    result = invoke_json(
        ["run", "/v1/custom", "--input-file", str(input_file)], monkeypatch
    )

    payload = assert_json_error(result, "input_parsing_error", "InputParsingError")
    assert payload["error"]["details"]["path"] == str(input_file)


def test_invalid_jq_expression_returns_json_error(monkeypatch):
    result = invoke_json(
        ["run", "/v1/custom", "--dry-run", "--jq", ".missing"], monkeypatch
    )

    payload = assert_json_error(result, "jq_expression_error", "JqExpressionError")
    assert payload["error"]["details"]["expression"] == ".missing"


@pytest.mark.parametrize("status_code", [400, 401, 422, 500])
@respx.mock
def test_http_errors_use_common_json_contract(monkeypatch, status_code):
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(status_code, json={"message": "Invalid model"})
    )

    result = invoke_json(["status", "task-1"], monkeypatch)

    expected_type = "AuthenticationError" if status_code == 401 else "APIRequestError"
    payload = assert_json_error(
        result,
        "auth_error"
        if status_code == 401
        else "api_validation_error"
        if status_code in (400, 422)
        else "api_error",
        expected_type,
    )
    assert payload["error"]["details"]["status_code"] == status_code
    assert payload["error"]["details"]["api_message"] == "Invalid model"
    assert "test-secret" not in result.output


@respx.mock
def test_timeout_returns_clean_json_error(monkeypatch):
    respx.get("https://api.test/v1/status").mock(
        side_effect=httpx.ReadTimeout("timeout")
    )

    result = invoke_json(["status", "task-1"], monkeypatch)

    assert_json_error(result, "api_timeout", "APIRequestError")


@respx.mock
def test_invalid_api_json_returns_clean_json_error(monkeypatch):
    respx.get("https://api.test/v1/status").mock(
        return_value=httpx.Response(
            200, content=b"not-json", headers={"content-type": "text/plain"}
        )
    )

    result = invoke_json(["status", "task-1"], monkeypatch)

    assert_json_error(result, "api_response_error", "APIResponseError")


def test_unknown_exception_has_no_traceback(monkeypatch):
    from everypixel_cli import cli

    monkeypatch.setattr(
        cli, "load_config", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    result = runner.invoke(app, ["config", "list"])

    assert result.exit_code == 1
    assert "Error [internal_error]: Unexpected internal error" in result.stderr
    assert "Traceback" not in result.output


def test_debug_reports_sanitized_stack_without_exception_message(monkeypatch):
    from everypixel_cli import cli

    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: (_ for _ in ()).throw(RuntimeError("secret backend detail")),
    )
    result = runner.invoke(app, ["--debug", "--output-json", "config", "list"])

    payload = assert_json_error(result, "internal_error", "InternalCLIError")
    details = payload["error"]["details"]
    assert details["exception_type"] == "RuntimeError"
    assert details["stack"]
    assert {"file", "line", "function"} <= details["stack"][-1].keys()
    assert "secret backend detail" not in result.output


def test_configuration_error_in_root_callback_is_json(monkeypatch):
    from everypixel_cli import cli

    monkeypatch.setattr(
        cli, "resolved_settings", lambda **_: (_ for _ in ()).throw(OSError("broken"))
    )
    result = runner.invoke(app, ["--output-json", "config", "list"])

    assert_json_error(result, "file_read_error", "FileReadError")


@respx.mock
def test_download_write_error_returns_json_error(monkeypatch, make_temp_dir):
    respx.get("https://cdn.test/result").mock(
        return_value=httpx.Response(200, content=b"x")
    )
    blocked = make_temp_dir("download-write") / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")

    with pytest.raises(Exception) as exc_info:
        download_urls(["https://cdn.test/result"], blocked)

    assert exc_info.value.code == "file_write_error"
