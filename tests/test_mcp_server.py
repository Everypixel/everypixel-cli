from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import anyio
import pytest
import respx
from mcp import Client
from typer.testing import CliRunner

from everypixel_cli.application import ApplicationServices, OperationResult
from everypixel_cli.cli import app
from everypixel_cli.client import APIClient
from everypixel_cli.mcp_server import create_mcp_server


EXPECTED_TOOLS = {
    "chat",
    "image_generate",
    "image_vectorize",
    "image_edit",
    "image_upscale",
    "image_angles",
    "image_colors",
    "video_generate",
    "video_edit",
    "video_upscale",
    "lipsync_video",
    "lipsync_image",
    "audio_transcribe",
    "tts_create",
    "tts_clone",
    "tts_voice",
    "tts_design",
    "tts_voices",
    "tts_delete",
    "task_status",
    "task_wait",
    "keywords",
    "quality",
    "quality_ugc",
    "faces",
    "captioning",
    "video_keywords",
    "run",
    "auth_check",
    "openapi",
    "openapi_refresh",
}


class FakeServices:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute_image_generate(self, **kwargs: Any) -> OperationResult:
        self.calls.append(("execute_image_generate", kwargs))
        return OperationResult(value={"task_id": "task-1", "status": "PENDING"})

    def check_auth(self) -> OperationResult:
        raise RuntimeError("secret backend detail")


class PollingClient:
    base_url = "https://api.test"

    def __init__(self) -> None:
        self.calls = 0
        self.started = threading.Event()
        self.started_at: float | None = None

    def get_status(self, task_id: str) -> dict[str, Any]:
        self.calls += 1
        if self.started_at is None:
            self.started_at = time.monotonic()
            self.started.set()
        if time.monotonic() - self.started_at >= 0.25:
            return {
                "task_id": task_id,
                "status": "SUCCESS",
                "result": "completed",
            }
        return {"task_id": task_id, "status": "PROCESSING"}


def test_mcp_server_exposes_application_use_cases_with_generated_schemas() -> None:
    async def inspect_tools() -> dict[str, Any]:
        server = create_mcp_server(lambda: FakeServices())
        async with Client(server) as client:
            listed = await client.list_tools()
            return {tool.name: tool for tool in listed.tools}

    tools = asyncio.run(inspect_tools())

    assert set(tools) == EXPECTED_TOOLS
    assert all(
        tool.input_schema["additionalProperties"] is False for tool in tools.values()
    )
    image_generate_schema = tools["image_generate"].input_schema
    assert image_generate_schema["required"] == ["prompt"]
    assert "zimage" in image_generate_schema["properties"]["model"]["enum"]
    assert image_generate_schema["properties"]["image_size"]["enum"] == [
        "square",
        "portrait_3_2",
        "portrait_4_3",
        "portrait_16_9",
        "landscape_3_2",
        "landscape_4_3",
        "landscape_16_9",
    ]
    assert tools["image_edit"].input_schema["properties"]["images"]["minItems"] == 1
    video_duration_schema = tools["video_generate"].input_schema["properties"][
        "duration"
    ]["anyOf"][0]
    assert video_duration_schema["minimum"] == 1
    assert video_duration_schema["maximum"] == 30
    assert tools["image_edit"].input_schema["properties"]["images"]["maxItems"] == 16
    assert (
        "minimax-h3"
        in tools["video_generate"].input_schema["properties"]["model"]["enum"]
    )
    assert (
        "4k" in tools["video_upscale"].input_schema["properties"]["resolution"]["enum"]
    )
    assert tools["image_generate"].annotations.read_only_hint is False
    assert tools["image_generate"].annotations.destructive_hint is True
    assert tools["task_status"].annotations.read_only_hint is True
    assert tools["task_wait"].annotations.destructive_hint is True
    assert tools["openapi"].annotations.destructive_hint is True


def test_mcp_tool_delegates_to_application_services_and_serializes_result() -> None:
    services = FakeServices()

    async def call_tool():
        server = create_mcp_server(lambda: services)
        async with Client(server) as client:
            return await client.call_tool(
                "image_generate",
                {
                    "prompt": "product photo",
                    "model": "gpt-image-2.5-sunburst",
                    "quality": "max",
                    "execution": {
                        "download_directory": "/tmp/everypixel-output",
                        "timeout": 12,
                        "poll_interval": 0.25,
                    },
                },
            )

    result = asyncio.run(call_tool())

    assert result.is_error is False
    assert result.structured_content == {"task_id": "task-1", "status": "PENDING"}
    name, arguments = services.calls[0]
    assert name == "execute_image_generate"
    assert arguments["model"] == "gpt-image-2.5-sunburst"
    assert arguments["quality"] == "max"
    assert arguments["execution"].wait is True
    assert arguments["execution"].download_directory == Path("/tmp/everypixel-output")
    assert arguments["execution"].timeout == 12
    assert arguments["execution"].poll_interval == 0.25


@respx.mock
def test_mcp_minimax_accepts_audio_references_without_images() -> None:
    route = respx.post("https://api.test/v1/video_generate").respond(
        200, json={"task_id": "task-1", "status": "PENDING"}
    )
    services = ApplicationServices.with_client(
        APIClient(base_url="https://api.test", client_id="id", client_secret="secret")
    )
    audio = str(Path(__file__).parent / "fixtures" / "result-2.mp3")

    async def call_tool():
        server = create_mcp_server(lambda: services)
        async with Client(server) as client:
            return await client.call_tool(
                "video_generate",
                {
                    "prompt": "animate to this music",
                    "model": "minimax-h3-turbo",
                    "reference_audios": [audio],
                    "execution": {"wait": False},
                },
            )

    try:
        result = asyncio.run(call_tool())
    finally:
        services.close()
    assert result.is_error is False
    payload = json.loads(route.calls[0].request.content)
    assert payload["reference_audio_urls"][0].startswith("data:audio/")
    assert payload["reference_image_urls"] == []
    assert payload["reference_video_urls"] == []


@pytest.mark.parametrize("streaming", [False, True])
@respx.mock
def test_mcp_chat_returns_complete_response(streaming):
    completion = {
        "id": "chat-1",
        "object": "chat.completion",
        "model": "glm-5.3",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                    "reasoning_content": "Think",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    route = respx.post("https://api.test/v1/chat/completions")
    if streaming:
        event = {
            **completion,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": completion["choices"][0]["message"],
                    "finish_reason": "stop",
                }
            ],
        }
        route.respond(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n",
        )
    else:
        route.respond(200, json=completion)
    services = ApplicationServices.with_client(
        APIClient(base_url="https://api.test", client_id="id", client_secret="secret")
    )

    async def call_tool():
        server = create_mcp_server(lambda: services)
        async with Client(server) as client:
            return await client.call_tool(
                "chat",
                {
                    "model": "glm-5.3",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": streaming,
                    "temperature": 0,
                    "do_sample": False,
                },
            )

    try:
        result = asyncio.run(call_tool())
    finally:
        services.close()
    assert result.is_error is False
    assert result.structured_content == completion
    assert route.call_count == len(respx.calls) == 1
    payload = json.loads(route.calls[0].request.content)
    assert payload["temperature"] == 0
    assert payload["do_sample"] is False
    if streaming:
        assert payload["stream_options"] == {"include_usage": True}


def test_mcp_tool_returns_sanitized_structured_error() -> None:
    async def call_tool():
        server = create_mcp_server(lambda: FakeServices())
        async with Client(server) as client:
            return await client.call_tool("auth_check")

    result = asyncio.run(call_tool())

    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "internal_error"
    assert result.structured_content["error"]["message"] == "Unexpected internal error"
    assert "secret backend detail" not in result.content[0].text


@pytest.mark.parametrize(
    "invalid_arguments",
    [
        {"prompt": "product photo", "resoluton": "1080p"},
        {"prompt": "product photo", "model": "unknown-model"},
    ],
)
def test_mcp_rejects_invalid_arguments_before_starting_operation(
    invalid_arguments: dict[str, Any],
) -> None:
    services = FakeServices()

    async def call_tool():
        server = create_mcp_server(lambda: services)
        async with Client(server) as client:
            return await client.call_tool(
                "image_generate",
                invalid_arguments,
            )

    result = asyncio.run(call_tool())

    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "validation_error"
    assert result.structured_content["error"]["message"] != "Invalid MCP tool arguments"
    assert services.calls == []


def test_mcp_redacts_values_from_pre_handler_validation_errors() -> None:
    sensitive_value = "dont-echo-me"

    async def call_tool():
        server = create_mcp_server(lambda: FakeServices())
        async with Client(server) as client:
            return await client.call_tool(
                "image_generate",
                {"prompt": {"client_secret": sensitive_value}},
            )

    result = asyncio.run(call_tool())

    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "validation_error"
    assert sensitive_value not in result.content[0].text
    assert sensitive_value not in str(result.structured_content)


def test_mcp_cancellation_stops_polling_before_download(tmp_path: Path) -> None:
    polling_client = PollingClient()
    services = ApplicationServices.with_client(polling_client)
    existing_result = tmp_path / "task-cancel.txt"
    existing_result.write_text("existing", encoding="utf-8")

    async def cancel_call() -> None:
        server = create_mcp_server(lambda: services)
        async with Client(server) as client:
            cancelled_at = 0.0
            with anyio.fail_after(1):
                async with anyio.create_task_group() as task_group:
                    task_group.start_soon(
                        client.call_tool,
                        "task_wait",
                        {
                            "task_id": "task-cancel",
                            "execution": {
                                "download_directory": str(tmp_path),
                                "poll_interval": 0.01,
                            },
                        },
                    )
                    while not polling_client.started.is_set():
                        await anyio.sleep(0.001)
                    cancelled_at = time.monotonic()
                    task_group.cancel_scope.cancel()

            assert time.monotonic() - cancelled_at < 0.2
            calls_after_cancel = polling_client.calls
            await anyio.sleep(0.3)
            assert polling_client.calls == calls_after_cancel

    asyncio.run(cancel_call())

    assert existing_result.read_text(encoding="utf-8") == "existing"


def test_mcp_serve_cli_command_starts_stdio_server(monkeypatch) -> None:
    from everypixel_cli import mcp_server

    calls: dict[str, Any] = {}

    def create_server(services_factory, *, client_id_present):
        calls["services_factory"] = services_factory
        calls["client_id_present"] = client_id_present
        return object()

    def run_server(server) -> None:
        calls["server"] = server
        calls["transport"] = "stdio"

    monkeypatch.setattr(mcp_server, "create_mcp_server", create_server)
    monkeypatch.setattr(mcp_server, "run_mcp_server", run_server)
    result = CliRunner().invoke(
        app,
        ["mcp", "serve"],
        env={
            "EVERYPIXEL_CLIENT_ID": "test-id",
            "EVERYPIXEL_CLIENT_SECRET": "test-secret",
        },
    )

    assert result.exit_code == 0
    assert result.stdout == ""
    assert calls["transport"] == "stdio"
    assert calls["client_id_present"] is True
    assert callable(calls["services_factory"])


@respx.mock
def test_mcp_tts_design_preview_can_be_saved_as_voice() -> None:
    preview_id = "550e8400-e29b-41d4-a716-446655440000"
    design = respx.post("https://api.test/v1/tts_design").respond(
        202, json={"task_id": "task-1", "status": "PENDING"}
    )
    card = {"type": "voice", "voice": {"id": preview_id, "name": "Narrator"}}
    saved = respx.post("https://api.test/v1/tts_voice").respond(201, json=card)
    services = ApplicationServices.with_client(
        APIClient(base_url="https://api.test", client_id="id", client_secret="secret")
    )

    async def call_tools():
        server = create_mcp_server(lambda: services)
        async with Client(server) as client:
            designed = await client.call_tool(
                "tts_design",
                {
                    "model": "eleven_ttv_v3",
                    "text": "я" * 100,
                    "prompt": "A calm, deep narrator voice",
                    "execution": {"wait": False},
                },
            )
            saved_voice = await client.call_tool(
                "tts_voice",
                {
                    "name": "Narrator",
                    "preview_id": preview_id,
                },
            )
            return designed, saved_voice

    try:
        designed, saved_voice = asyncio.run(call_tools())
    finally:
        services.close()
    assert designed.is_error is False
    assert saved_voice.is_error is False
    assert json.loads(design.calls[0].request.content) == {
        "model": "eleven_ttv_v3",
        "text": "я" * 100,
        "prompt": "A calm, deep narrator voice",
    }
    assert json.loads(saved.calls[0].request.content) == {
        "name": "Narrator",
        "preview_id": preview_id,
        "description": "",
    }
    assert json.loads(saved_voice.content[0].text) == card


@respx.mock
def test_mcp_tts_create_leaves_default_model_to_api() -> None:
    route = respx.post("https://api.test/v1/tts_create").respond(
        202, json={"task_id": "task-1", "status": "PENDING"}
    )
    services = ApplicationServices.with_client(
        APIClient(base_url="https://api.test", client_id="id", client_secret="secret")
    )

    async def call_tool():
        server = create_mcp_server(lambda: services)
        async with Client(server) as client:
            return await client.call_tool(
                "tts_create",
                {
                    "text": "я" * 201,
                    "seed": 0,
                    "execution": {"wait": False},
                },
            )

    try:
        result = asyncio.run(call_tool())
    finally:
        services.close()
    assert result.is_error is False
    assert json.loads(route.calls[0].request.content) == {"text": "я" * 201, "seed": 0}
