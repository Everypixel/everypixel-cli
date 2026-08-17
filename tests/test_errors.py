import httpx

from everypixel_cli.client import extract_error_detail
from everypixel_cli.errors import (
    EXIT_API_VALIDATION,
    EXIT_AUTH,
    EXIT_BILLING,
    EXIT_NOT_FOUND,
    EXIT_RATE_LIMIT,
    http_exit_code,
    format_validation_errors,
    mask_secret,
)


def test_mask_secret():
    assert mask_secret("abcdef123456") == "abcd********"


def test_http_status_to_exit_code_mapping():
    assert http_exit_code(401) == (EXIT_AUTH, "auth_error")
    assert http_exit_code(404) == (EXIT_NOT_FOUND, "not_found")
    assert http_exit_code(429) == (EXIT_RATE_LIMIT, "rate_limit")
    assert http_exit_code(400, "Input should be a valid URL") == (
        EXIT_API_VALIDATION,
        "api_validation_error",
    )
    assert http_exit_code(422, "invalid payload") == (
        EXIT_API_VALIDATION,
        "api_validation_error",
    )
    assert http_exit_code(400, "billing limit exceeded") == (
        EXIT_BILLING,
        "billing_or_limit_error",
    )


def test_extract_error_detail_reads_backend_message_field():
    response = httpx.Response(
        400,
        json={"status": "error", "message": [{"msg": "Input should be a valid URL"}]},
    )

    assert extract_error_detail(response) == "Input should be a valid URL"


def test_extract_error_detail_reads_detail_and_error_fields():
    detail_response = httpx.Response(422, json={"detail": {"message": "Bad payload"}})
    error_response = httpx.Response(500, json={"error": "Internal failure"})

    assert extract_error_detail(detail_response) == "Bad payload"
    assert extract_error_detail(error_response) == "Internal failure"


def test_format_validation_errors_removes_pydantic_noise():
    message = format_validation_errors(
        [
            {"loc": (), "msg": "Value error, replication style requires --image"},
            {
                "loc": ("resolution",),
                "msg": "Input should be '720p', '1080p' or '1440p'",
            },
        ]
    )

    assert (
        message
        == "replication style requires --image; resolution: Input should be '720p', '1080p' or '1440p'"
    )
