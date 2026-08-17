"""OpenAPI schema support for the generic `run` command.

The CLI tries live API schema first, then cache, then the bundled snapshot.
This keeps `everypixel run ...` usable offline and up to date when API exists.
"""

from __future__ import annotations

import json
import time
from contextlib import suppress
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from jsonschema.exceptions import best_match
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from .config import config_dir
from .errors import CLIError, EXIT_VALIDATION, FileWriteError


SCHEMA_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class Operation:
    """Resolved OpenAPI operation: HTTP method, path, and schema fragment."""

    method: str
    path: str
    schema: dict[str, Any]
    document: dict[str, Any] | None = None


def schema_cache_path(base_url: str) -> Path:
    """Return the OpenAPI schema cache path for a base URL."""

    safe = base_url.replace("://", "_").replace("/", "_").replace(":", "_")
    return config_dir() / "schemas" / f"{safe}.json"


def read_cached_schema(
    base_url: str, *, ttl_seconds: int = SCHEMA_TTL_SECONDS
) -> dict[str, Any] | None:
    """Read a fresh cached schema or return None."""

    path = schema_cache_path(base_url)
    try:
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > ttl_seconds:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_cached_schema(base_url: str, schema: dict[str, Any]) -> Path:
    """Save an OpenAPI schema to local cache."""

    path = schema_cache_path(base_url)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise FileWriteError(
            "Unable to write OpenAPI schema cache",
            details={"path": str(path)},
        ) from exc
    return path


def bundled_schema() -> dict[str, Any]:
    """Read the bundled OpenAPI snapshot from package resources."""

    resource = files("everypixel_cli.resources").joinpath("openapi.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_schema(client, *, refresh: bool = False) -> tuple[dict[str, Any], str]:
    """Load schema from cache/live/bundled and return its source."""

    if not refresh:
        cached = read_cached_schema(client.base_url)
        if cached is not None:
            return cached, "cache"
    try:
        live = client.request("GET", "/v1/openapi.json", auth_required=False)
    except CLIError:
        cached = read_cached_schema(
            client.base_url, ttl_seconds=10 * 365 * 24 * 60 * 60
        )
        if cached is not None:
            return cached, "cache_stale"
        return bundled_schema(), "bundled"
    with suppress(FileWriteError):
        write_cached_schema(client.base_url, live)
    return live, "live"


def refresh_schema(client) -> tuple[dict[str, Any], Path]:
    """Force-load live OpenAPI schema and update cache."""

    schema = client.request("GET", "/v1/openapi.json", auth_required=False)
    path = write_cached_schema(client.base_url, schema)
    return schema, path


def find_operation(
    schema: dict[str, Any], endpoint: str, requested_method: str | None = None
) -> Operation | None:
    """Find an operation by path, short endpoint name, or operationId."""

    paths = schema.get("paths")
    if not isinstance(paths, dict):
        return None

    normalized_endpoint = endpoint.strip()
    if normalized_endpoint.startswith("/"):
        candidates = [normalized_endpoint]
    else:
        candidates = [f"/v1/{normalized_endpoint}", normalized_endpoint]

    requested = requested_method.lower() if requested_method else None
    for path, operations in paths.items():
        if not isinstance(operations, dict):
            continue
        for method, operation_schema in operations.items():
            if requested and method.lower() != requested:
                continue
            if (
                path in candidates
                or path.rstrip("/").split("/")[-1] == normalized_endpoint
            ):
                return Operation(
                    method=method.upper(),
                    path=path,
                    schema=operation_schema,
                    document=schema,
                )
            operation_id = (
                operation_schema.get("operationId")
                if isinstance(operation_schema, dict)
                else None
            )
            if operation_id == normalized_endpoint:
                return Operation(
                    method=method.upper(),
                    path=path,
                    schema=operation_schema,
                    document=schema,
                )
    return None


def operation_help(
    operation: Operation | None,
    *,
    endpoint: str,
    method: str,
    path: str,
    schema_source: str | None,
) -> dict[str, Any]:
    """Build a help payload for a resolved OpenAPI operation."""

    schema = operation.schema if operation else None
    return {
        "endpoint": endpoint,
        "schema_source": schema_source,
        "method": method,
        "path": path,
        "summary": schema.get("summary") if isinstance(schema, dict) else None,
        "operation_id": schema.get("operationId") if isinstance(schema, dict) else None,
        "parameters": summarize_parameters(schema),
        "request_body": summarize_request_body(schema),
        "schema": schema,
    }


def summarize_parameters(schema: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Simplify OpenAPI parameters for user output."""

    if not isinstance(schema, dict) or not isinstance(schema.get("parameters"), list):
        return []
    parameters = []
    for item in schema["parameters"]:
        if isinstance(item, dict):
            parameters.append(
                {
                    "name": item.get("name"),
                    "in": item.get("in"),
                    "required": item.get("required", False),
                    "schema": item.get("schema"),
                    "description": item.get("description"),
                }
            )
    return parameters


def summarize_request_body(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Simplify requestBody description for user output."""

    if not isinstance(schema, dict):
        return None
    request_body = schema.get("requestBody")
    if not isinstance(request_body, dict):
        return None
    content = request_body.get("content")
    if not isinstance(content, dict):
        return {"required": request_body.get("required", False), "content_types": []}
    return {
        "required": request_body.get("required", False),
        "content_types": [
            {
                "content_type": content_type,
                "schema": content_schema.get("schema")
                if isinstance(content_schema, dict)
                else None,
            }
            for content_type, content_schema in content.items()
        ],
    }


def validate_payload_against_operation(
    operation: Operation | None, payload: dict[str, Any]
) -> None:
    """Validate a payload against an OpenAPI 3.1 JSON request schema."""

    if operation is None:
        return
    schema = json_request_body_schema(operation.schema)
    if schema is None:
        return
    validator = openapi_validator(schema, operation.document)
    error = best_match(validator.iter_errors(payload))
    if error is None:
        return
    message, details = format_openapi_validation_error(error)
    raise CLIError(
        message,
        code="validation_error",
        exit_code=EXIT_VALIDATION,
        details=details,
    )


def json_request_body_schema(operation_schema: dict[str, Any]) -> dict[str, Any] | None:
    """Return the application/json schema for an OpenAPI operation."""

    request_body = operation_schema.get("requestBody")
    if not isinstance(request_body, dict):
        return None
    content = request_body.get("content")
    if not isinstance(content, dict):
        return None
    json_content = content.get("application/json")
    if not isinstance(json_content, dict):
        return None
    schema = json_content.get("schema")
    return schema if isinstance(schema, dict) else None


def openapi_validator(
    schema: dict[str, Any], document: dict[str, Any] | None
) -> Draft202012Validator:
    """Build a JSON Schema validator with local OpenAPI references available."""

    if document is None:
        return Draft202012Validator(schema)
    document_uri = "urn:everypixel:openapi"
    resource = Resource.from_contents(
        document,
        default_specification=DRAFT202012,
    )
    registry = Registry().with_resource(document_uri, resource)
    return Draft202012Validator(
        absolute_document_refs(schema, document_uri),
        registry=registry,
    )


def absolute_document_refs(value: Any, document_uri: str) -> Any:
    """Point fragment refs from a detached request schema at its document."""

    if isinstance(value, dict):
        result = {
            key: absolute_document_refs(item, document_uri)
            for key, item in value.items()
        }
        ref = result.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            result["$ref"] = f"{document_uri}{ref}"
        return result
    if isinstance(value, list):
        return [absolute_document_refs(item, document_uri) for item in value]
    return value


def format_openapi_validation_error(
    error: JSONSchemaValidationError,
) -> tuple[str, dict[str, Any]]:
    """Return a stable error without echoing arbitrary user payload values."""

    path = ".".join(str(part) for part in error.absolute_path)
    label = path or "payload"
    rule = str(error.validator)
    if error.validator == "required" and isinstance(error.validator_value, list):
        instance = error.instance if isinstance(error.instance, dict) else {}
        missing = [key for key in error.validator_value if key not in instance]
        message = f"Missing required input(s): {', '.join(map(str, missing))}"
    elif error.validator == "type":
        message = f"Invalid type for input '{label}': expected {error.validator_value}"
    elif error.validator == "enum":
        message = f"Invalid value for input '{label}': expected an allowed enum value"
    elif error.validator in {"oneOf", "anyOf"}:
        message = "Payload does not match an allowed request schema"
    elif error.validator in {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
    }:
        message = f"Input '{label}' is outside the allowed range"
    else:
        message = f"Input '{label}' does not satisfy the OpenAPI schema"
    details = {"rule": rule}
    if path:
        details["path"] = path
    return message, details
