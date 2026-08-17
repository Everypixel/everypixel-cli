import json

from typer.testing import CliRunner

from everypixel_cli.cli import app


runner = CliRunner()


def test_missing_required_argument_exits_with_error():
    result = runner.invoke(app, ["image", "generate"])

    assert result.exit_code != 0
    assert "Missing argument" in result.output


def test_missing_required_argument_uses_json_error_contract():
    result = runner.invoke(app, ["--output-json", "image", "generate"])

    assert result.exit_code == 2
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["type"] == "ValidationCLIError"
    assert payload["error"]["code"] == "validation_error"
    assert "Missing argument" in payload["error"]["message"]


def test_unknown_root_option_uses_json_error_contract():
    result = runner.invoke(app, ["--output-json", "--unknown-option"])

    assert result.exit_code == 2
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["error"]["type"] == "ValidationCLIError"
    assert "No such option" in payload["error"]["message"]


def test_invalid_output_mode_is_rejected():
    result = runner.invoke(app, ["--output", "jsom", "config", "list"])

    assert result.exit_code == 2
    assert "Invalid value" in result.stderr


def test_invalid_output_mode_uses_json_error_contract_regardless_of_flag_order():
    result = runner.invoke(
        app,
        ["--output", "jsom", "--output-json", "config", "list"],
    )

    assert result.exit_code == 2
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "validation_error"


def test_wait_controls_must_be_positive_in_json_mode():
    for option, value in (("--timeout", "0"), ("--poll-interval", "-1")):
        result = runner.invoke(
            app,
            ["--output-json", option, value, "config", "list"],
        )

        assert result.exit_code == 2
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        assert payload["error"]["code"] == "validation_error"
        assert "greater than 0" in payload["error"]["message"]


def test_run_rejects_invalid_input_item_as_json_error():
    result = runner.invoke(
        app, ["--output-json", "run", "/v1/custom", "-i", "broken", "--dry-run"]
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["type"] == "InputParsingError"
    assert payload["error"]["code"] == "input_parsing_error"
    assert payload["error"]["message"] == "Invalid input item"


def test_video_upscale_help_documents_1440p_duration_limit():
    result = runner.invoke(app, ["video", "upscale", "--help"])

    assert result.exit_code == 0
    assert "1440p" in result.stdout
    assert "up to 20s at 1440p" in result.stdout
