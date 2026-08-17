"""Application error model and stable CLI exit codes."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError


EXIT_GENERAL = 1
EXIT_VALIDATION = 2
EXIT_AUTH = 3
EXIT_API = 4
EXIT_FILE = 5
EXIT_RESULT = 6

# Backwards-compatible names used by the existing public Python API.
EXIT_RATE_LIMIT = EXIT_API
EXIT_NOT_FOUND = EXIT_API
EXIT_BILLING = EXIT_API
EXIT_TIMEOUT = EXIT_API
EXIT_API_VALIDATION = EXIT_API
EXIT_TASK_FAILURE = EXIT_API


class EverypixelCLIError(Exception):
    """Base error rendered consistently by the CLI boundary."""

    code = "cli_error"
    exit_code = EXIT_GENERAL

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        exit_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.code
        self.exit_code = exit_code if exit_code is not None else self.exit_code
        self.details = safe_details(details or {})

    def to_payload(self) -> dict[str, Any]:
        """Return the stable JSON error contract."""

        return {
            "ok": False,
            "error": {
                "type": type(self).__name__,
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


# CLIError remains an alias so third-party callers and existing tests keep working.
CLIError = EverypixelCLIError


class ConfigurationError(EverypixelCLIError):
    code = "configuration_error"
    exit_code = EXIT_AUTH


class AuthenticationError(EverypixelCLIError):
    code = "authentication_error"
    exit_code = EXIT_AUTH


class ValidationCLIError(EverypixelCLIError):
    code = "validation_error"
    exit_code = EXIT_VALIDATION


class InputParsingError(ValidationCLIError):
    code = "input_parsing_error"


class FileReadError(EverypixelCLIError):
    code = "file_read_error"
    exit_code = EXIT_FILE


class FileWriteError(EverypixelCLIError):
    code = "file_write_error"
    exit_code = EXIT_FILE


class PayloadPreparationError(ValidationCLIError):
    code = "payload_preparation_error"


class APIRequestError(EverypixelCLIError):
    code = "api_request_error"
    exit_code = EXIT_API


class APIResponseError(EverypixelCLIError):
    code = "api_response_error"
    exit_code = EXIT_API


class TaskPollingError(APIRequestError):
    code = "task_polling_error"


class TaskFailedError(EverypixelCLIError):
    """A task reached the API-level FAILURE terminal state."""

    code = "task_failed"
    exit_code = EXIT_API

    def __init__(
        self,
        message: str,
        *,
        task_id: str,
        status: str = "FAILURE",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.task_id = task_id
        self.status = status
        super().__init__(
            message, details={"task_id": task_id, "status": status, **(details or {})}
        )

    def to_payload(self) -> dict[str, Any]:
        """Include task state alongside the shared error contract."""

        payload = super().to_payload()
        payload["task"] = {"id": self.task_id, "status": self.status}
        return payload


class DownloadError(FileWriteError):
    code = "download_error"


class JqExpressionError(EverypixelCLIError):
    code = "jq_expression_error"
    exit_code = EXIT_RESULT


class SerializationError(EverypixelCLIError):
    code = "serialization_error"
    exit_code = EXIT_RESULT


class UnsupportedResultError(EverypixelCLIError):
    code = "unsupported_result_error"
    exit_code = EXIT_RESULT


class InternalCLIError(EverypixelCLIError):
    code = "internal_error"
    exit_code = EXIT_GENERAL


def safe_details(value: dict[str, Any]) -> dict[str, Any]:
    """Make details JSON-safe and remove credential-like keys."""

    sensitive = {
        "authorization",
        "client_secret",
        "api_key",
        "secret",
        "token",
        "password",
    }

    def sanitize(item: Any, key: str | None = None) -> Any:
        if key and key.lower().replace("-", "_") in sensitive:
            return "***"
        if isinstance(item, dict):
            return {str(name): sanitize(raw, str(name)) for name, raw in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(raw) for raw in item]
        if isinstance(item, str) and item.startswith(("http://", "https://")):
            return item.split("?", 1)[0]
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return str(item)

    return sanitize(value)


def fallback_serialization_error() -> SerializationError:
    """Return the only error safe to emit when JSON serialization itself fails."""

    return SerializationError("Unable to serialize CLI response")


def serialize_error(error: EverypixelCLIError) -> str:
    """Serialize an error without allowing JSON failures to leak a traceback."""

    try:
        return json.dumps(error.to_payload(), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(fallback_serialization_error().to_payload())


def http_exit_code(status_code: int, detail: str = "") -> tuple[int, str]:
    """Map API status codes to stable remote-service categories."""

    normalized = detail.lower()
    if status_code in (401, 403):
        return EXIT_AUTH, "auth_error"
    if status_code == 404:
        return EXIT_API, "not_found"
    if status_code == 429:
        return EXIT_API, "rate_limit"
    if "billing" in normalized or "quota" in normalized or "limit" in normalized:
        return EXIT_API, "billing_or_limit_error"
    if status_code in (400, 422):
        return EXIT_API, "api_validation_error"
    return EXIT_API, "api_error"


def mask_secret(value: str | None, visible: int = 4) -> str | None:
    """Mask a secret for safe console output."""

    if value is None:
        return None
    if len(value) <= visible:
        return "*" * len(value)
    return f"{value[:visible]}{'*' * 8}"


def format_validation_errors(errors: list[Any]) -> str:
    """Format Pydantic validation errors for users."""

    messages: list[str] = []
    for item in errors:
        location = ".".join(
            str(part) for part in item.get("loc", ()) if part != "__root__"
        )
        message = str(item.get("msg", "Invalid value"))
        if message.startswith("Value error, "):
            message = message.removeprefix("Value error, ")
        messages.append(f"{location}: {message}" if location else message)
    return "; ".join(messages)


def normalize_exception(exc: Exception) -> EverypixelCLIError:
    """Convert expected technical exceptions into stable application errors."""

    if isinstance(exc, EverypixelCLIError):
        return exc
    if isinstance(exc, ValidationError):
        return ValidationCLIError(format_validation_errors(exc.errors()))
    if isinstance(exc, json.JSONDecodeError):
        return InputParsingError("Unable to parse JSON input")
    if isinstance(exc, (FileNotFoundError, PermissionError, OSError)):
        return FileReadError("Unable to access local file")
    if isinstance(exc, ValueError):
        return PayloadPreparationError("Unable to prepare request payload")
    return InternalCLIError("Unexpected internal error")
