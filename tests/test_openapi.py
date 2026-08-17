import os

import pytest

from everypixel_cli.errors import CLIError, EXIT_VALIDATION
from everypixel_cli.openapi import (
    Operation,
    bundled_schema,
    load_schema,
    find_operation,
    operation_help,
    validate_payload_against_operation,
    read_cached_schema,
    write_cached_schema,
)
from everypixel_cli.errors import FileWriteError


class FakeSchemaClient:
    base_url = "https://api.test"

    def __init__(self, response=None, error=False):
        self.response = response or {
            "paths": {"/v1/live": {"get": {"operationId": "live"}}}
        }
        self.error = error
        self.calls = 0

    def request(self, method, path, **kwargs):
        self.calls += 1
        if self.error:
            raise CLIError("network failed", code="network_error")
        assert method == "GET"
        assert path == "/v1/openapi.json"
        assert kwargs["auth_required"] is False
        return self.response


def test_find_operation_by_endpoint_name():
    schema = {
        "paths": {"/v1/image_generate": {"post": {"operationId": "image_generate"}}}
    }

    operation = find_operation(schema, "image_generate")

    assert operation is not None
    assert operation.method == "POST"
    assert operation.path == "/v1/image_generate"


def test_find_operation_by_operation_id():
    schema = {"paths": {"/v1/status": {"get": {"operationId": "get_status"}}}}

    operation = find_operation(schema, "get_status")

    assert operation is not None
    assert operation.method == "GET"
    assert operation.path == "/v1/status"


def test_find_operation_respects_requested_method():
    schema = {
        "paths": {
            "/v1/keywords": {
                "get": {"operationId": "keywords_get"},
                "post": {"operationId": "keywords_post"},
            }
        }
    }

    operation = find_operation(schema, "/v1/keywords", "POST")

    assert operation is not None
    assert operation.method == "POST"


def test_load_schema_uses_fresh_cache_without_live_request(monkeypatch):
    cached = {"paths": {"/v1/cache": {"get": {"operationId": "cache"}}}}
    client = FakeSchemaClient()

    monkeypatch.setattr(
        "everypixel_cli.openapi.read_cached_schema", lambda *args, **kwargs: cached
    )

    schema, source = load_schema(client)

    assert schema == cached
    assert source == "cache"
    assert client.calls == 0


def test_load_schema_uses_live_and_writes_cache(monkeypatch):
    written = {}
    client = FakeSchemaClient(
        response={"paths": {"/v1/live": {"get": {"operationId": "live"}}}}
    )

    monkeypatch.setattr(
        "everypixel_cli.openapi.read_cached_schema", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "everypixel_cli.openapi.write_cached_schema",
        lambda base_url, schema: written.update({base_url: schema}),
    )

    schema, source = load_schema(client)

    assert source == "live"
    assert schema == client.response
    assert written[client.base_url] == client.response


def test_load_schema_uses_stale_cache_when_live_fails(monkeypatch):
    stale = {"paths": {"/v1/stale": {"get": {"operationId": "stale"}}}}
    calls = []
    client = FakeSchemaClient(error=True)

    def fake_read_cached_schema(*args, **kwargs):
        calls.append(kwargs.get("ttl_seconds"))
        return None if len(calls) == 1 else stale

    monkeypatch.setattr(
        "everypixel_cli.openapi.read_cached_schema", fake_read_cached_schema
    )

    schema, source = load_schema(client)

    assert schema == stale
    assert source == "cache_stale"


def test_load_schema_uses_bundled_when_live_and_cache_fail(monkeypatch):
    bundled = {"paths": {"/v1/bundled": {"get": {"operationId": "bundled"}}}}
    client = FakeSchemaClient(error=True)

    monkeypatch.setattr(
        "everypixel_cli.openapi.read_cached_schema", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("everypixel_cli.openapi.bundled_schema", lambda: bundled)

    schema, source = load_schema(client)

    assert schema == bundled
    assert source == "bundled"


def test_operation_help_summarizes_parameters_and_request_body():
    operation = Operation(
        method="POST",
        path="/v1/image_generate",
        schema={
            "operationId": "image_generate",
            "summary": "Generate image",
            "parameters": [
                {
                    "name": "dry_run",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "boolean"},
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"prompt": {"type": "string"}},
                        }
                    }
                },
            },
        },
    )

    payload = operation_help(
        operation,
        endpoint="image_generate",
        method="POST",
        path="/v1/image_generate",
        schema_source="live",
    )

    assert payload["operation_id"] == "image_generate"
    assert payload["summary"] == "Generate image"
    assert payload["parameters"][0]["name"] == "dry_run"
    assert payload["request_body"]["required"] is True
    assert (
        payload["request_body"]["content_types"][0]["content_type"]
        == "application/json"
    )
    assert payload["schema"] == operation.schema


def test_validate_payload_against_operation_rejects_missing_required_input():
    operation = Operation(
        method="POST",
        path="/v1/image_generate",
        schema={
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["prompt"],
                            "properties": {"prompt": {"type": "string"}},
                        }
                    }
                }
            }
        },
    )

    with pytest.raises(CLIError) as exc_info:
        validate_payload_against_operation(operation, {})

    assert exc_info.value.exit_code == EXIT_VALIDATION
    assert exc_info.value.message == "Missing required input(s): prompt"


def test_validate_payload_against_operation_rejects_primitive_type_mismatch():
    operation = Operation(
        method="POST",
        path="/v1/video_generate",
        schema={
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"duration": {"type": "integer"}},
                        }
                    }
                }
            }
        },
    )

    with pytest.raises(CLIError) as exc_info:
        validate_payload_against_operation(operation, {"duration": "10"})

    assert exc_info.value.exit_code == EXIT_VALIDATION
    assert (
        exc_info.value.message == "Invalid type for input 'duration': expected integer"
    )


def test_bundled_ref_request_schema_is_validated():
    operation = find_operation(bundled_schema(), "image_generate")

    with pytest.raises(CLIError) as exc_info:
        validate_payload_against_operation(operation, {})

    assert exc_info.value.code == "validation_error"
    assert exc_info.value.message == "Missing required input(s): prompt"


def test_bundled_one_of_request_schema_is_validated():
    operation = find_operation(bundled_schema(), "video_generate")

    with pytest.raises(CLIError) as exc_info:
        validate_payload_against_operation(
            operation,
            {"model": "seedance2", "prompt": "x", "duration": "5"},
        )

    assert exc_info.value.code == "validation_error"


def test_openapi_validation_preserves_forward_compatible_fields():
    operation = find_operation(bundled_schema(), "image_generate")

    validate_payload_against_operation(
        operation,
        {"prompt": "x", "future_api_field": {"enabled": True}},
    )


def test_schema_cache_round_trip_and_ttl(monkeypatch, tmp_path):
    monkeypatch.setattr("everypixel_cli.openapi.config_dir", lambda: tmp_path)
    schema = {"openapi": "3.1.0", "paths": {}}

    path = write_cached_schema("https://api.test", schema)

    assert read_cached_schema("https://api.test") == schema
    os.utime(path, (0, 0))
    assert read_cached_schema("https://api.test", ttl_seconds=1) is None


def test_malformed_schema_cache_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr("everypixel_cli.openapi.config_dir", lambda: tmp_path)
    path = write_cached_schema("https://api.test", {"paths": {}})
    path.write_text("{broken", encoding="utf-8")

    assert read_cached_schema("https://api.test") is None


def test_schema_cache_write_failure_is_typed(monkeypatch, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr("everypixel_cli.openapi.config_dir", lambda: blocked)

    with pytest.raises(FileWriteError):
        write_cached_schema("https://api.test", {"paths": {}})


def test_live_schema_remains_usable_when_cache_is_not_writable(monkeypatch, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr("everypixel_cli.openapi.config_dir", lambda: blocked)
    client = FakeSchemaClient(response={"paths": {"/v1/live": {"get": {}}}})

    schema, source = load_schema(client)

    assert schema == client.response
    assert source == "live"
