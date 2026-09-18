import json

import httpx
import pytest
import respx
from typer.testing import CliRunner

from everypixel_cli.application import ApplicationServices, ExecutionOptions
from everypixel_cli.application.models import OperationCancelled
from everypixel_cli.cli import app
from everypixel_cli.client import APIClient
from everypixel_cli.openapi import bundled_schema


runner = CliRunner()
COMPLETION = {
    "id": "chat-1",
    "object": "chat.completion",
    "model": "glm-5.3",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello [world]!",
                "reasoning_content": "Thinking.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


@pytest.fixture(autouse=True)
def chat_credentials(monkeypatch):
    monkeypatch.setenv("EVERYPIXEL_CLIENT_ID", "id")
    monkeypatch.setenv("EVERYPIXEL_CLIENT_SECRET", "secret")


class EventStream(httpx.SyncByteStream):
    def __init__(self, data):
        self.data = data.encode("utf-8")
        self.closed = False

    def __iter__(self):
        # Exercise transport boundaries inside JSON, Unicode, and SSE delimiters.
        for offset in range(0, len(self.data), 7):
            yield self.data[offset : offset + 7]

    def close(self):
        self.closed = True


def frame(event):
    return "data: " + json.dumps(event, ensure_ascii=False) + "\r\n\r\n"


def chunk(delta, finish=None):
    return {
        "id": "chat-1",
        "object": "chat.completion.chunk",
        "model": "glm-5.3",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


@pytest.mark.parametrize("output", ["json", "human", "reasoning"])
@respx.mock
def test_chat_synchronous_history_tools_and_falsey_options(output, make_temp_dir):
    directory = make_temp_dir("chat-history")
    request_file = directory / "request.json"
    history = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Use weather",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "Sunny"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "strict": False,
                "parameters": {"type": "object"},
            },
        }
    ]
    request_file.write_text(
        json.dumps(
            {
                "messages": history,
                "tools": tools,
                "tool_choice": "auto",
                "do_sample": False,
                "temperature": 0.8,
                "thinking": {"type": "enabled", "clear_thinking": False},
            }
        )
    )
    route = respx.post("https://api.test/v1/chat/completions").respond(
        200, json=COMPLETION
    )
    options = (
        ["-j"]
        if output == "json"
        else ["--show-reasoning"]
        if output == "reasoning"
        else []
    )
    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "chat",
            "Summarize.",
            "--system",
            "Be brief.",
            "--input-file",
            str(request_file),
            "--temperature",
            "0",
            "--max-completion-tokens",
            "100",
            "--request-timeout",
            "75",
            *options,
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert route.call_count == len(respx.calls) == 1  # No async status requests.
    request = route.calls[0].request
    body = json.loads(request.content)
    assert request.headers["authorization"] == "Basic aWQ6c2VjcmV0"
    assert request.extensions["timeout"]["read"] == 75
    assert body["model"] == "glm-5.3"
    assert body["temperature"] == 0
    assert body["max_completion_tokens"] == 100
    assert body["do_sample"] is False
    assert body["thinking"]["clear_thinking"] is False
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}
    assert body["messages"][-1] == {"role": "user", "content": "Summarize."}
    assert body["messages"][2]["tool_calls"] == history[1]["tool_calls"]
    assert body["messages"][2]["reasoning_content"] == "Use weather"
    assert body["tools"] == tools
    if output == "json":
        assert json.loads(result.stdout) == COMPLETION
    else:
        assert (
            result.stdout
            == ("Thinking.\n" if output == "reasoning" else "") + "Hello [world]!\n"
        )


@pytest.mark.parametrize("output", ["json", "human", "reasoning"])
@respx.mock
def test_chat_stream_assembles_reasoning_tools_and_usage(output):
    events = [
        chunk({"role": "assistant", "reasoning_content": "Think."}),
        chunk({"content": "При"}),
        chunk(
            {
                "content": "вет!",
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "call-b",
                        "type": "function",
                        "function": {"name": "second", "arguments": ""},
                    },
                    {
                        "index": 0,
                        "id": "call-a",
                        "type": "function",
                        "function": {"name": "wea", "arguments": '{"city":'},
                    },
                ],
            }
        ),
        chunk(
            {
                "tool_calls": [
                    {"index": 0, "function": {"name": "ther", "arguments": '"Paris"}'}},
                    {"index": 1, "function": {"arguments": "{}"}},
                ]
            },
            "tool_calls",
        ),
        {"id": "chat-1", "choices": [], "usage": COMPLETION["usage"]},
    ]
    stream = EventStream(
        ": keepalive\r\n\r\n" + "".join(map(frame, events)) + "data: [DONE]\r\n\r\n"
    )
    route = respx.post("https://api.test/v1/chat/completions").respond(
        200,
        headers={"content-type": "text/event-stream; charset=utf-8"},
        stream=stream,
    )
    options = (
        ["-j"]
        if output == "json"
        else ["--show-reasoning"]
        if output == "reasoning"
        else []
    )
    result = runner.invoke(
        app, ["--base-url", "https://api.test", "chat", "Hi", "--stream", *options]
    )
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert stream.closed
    assert route.call_count == len(respx.calls) == 1
    assert json.loads(route.calls[0].request.content)["stream_options"] == {
        "include_usage": True
    }
    assert route.calls[0].request.extensions["timeout"]["read"] == 240
    if output == "json":
        body = json.loads(result.stdout)
        assert body["object"] == "chat.completion"
        assert body["usage"] == COMPLETION["usage"]
        choice = body["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"] == {
            "role": "assistant",
            "content": "Привет!",
            "reasoning_content": "Think.",
            "tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                },
                {
                    "id": "call-b",
                    "type": "function",
                    "function": {"name": "second", "arguments": "{}"},
                },
            ],
        }
    else:
        assert result.stdout.count("Привет!") == 1
        assert ("Think." in result.stdout) is (output == "reasoning")
        assert '"name": "weather"' in result.stdout


@pytest.mark.parametrize("endpoint", ["chat/completions", "/v1/chat/completions"])
@pytest.mark.parametrize("streaming", [False, True])
@respx.mock
def test_generic_chat_returns_completion_without_polling(endpoint, streaming):
    if not endpoint.startswith("/"):
        respx.get("https://api.test/v1/openapi.json").respond(
            200, json=bundled_schema()
        )
    route = respx.post("https://api.test/v1/chat/completions")
    if streaming:
        route.respond(
            200,
            headers={"content-type": "text/event-stream"},
            text=frame(chunk({"content": "ok"}, "stop")) + "data: [DONE]\n\n",
        )
    else:
        route.respond(200, json=COMPLETION)
    payload = {
        "model": "glm-5.3",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": streaming,
    }
    options = [
        part
        for key, value in payload.items()
        for part in ("--input", f"{key}={json.dumps(value)}")
    ]
    result = runner.invoke(
        app, ["--base-url", "https://api.test", "-j", "run", endpoint, *options]
    )
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert json.loads(result.stdout)["object"] == "chat.completion"
    assert json.loads(route.calls[0].request.content) == payload
    assert route.call_count == 1


@pytest.mark.parametrize(
    ("ending", "code"),
    [
        ("", "incomplete_stream"),
        ("data: nope\n\n", "api_response_error"),
        ("data: []\n\n", "api_response_error"),
        (
            frame(
                {"error": {"message": "Provider unavailable", "code": "unavailable"}}
            ),
            "chat_stream_error",
        ),
        ("data: [DONE]\n\n", "api_response_error"),
    ],
)
@respx.mock
def test_stream_errors_do_not_emit_partial_json_or_retry(ending, code):
    stream = EventStream(frame(chunk({"content": "partial"})) + ending)
    route = respx.post("https://api.test/v1/chat/completions").respond(
        200,
        headers={"content-type": "text/event-stream"},
        stream=stream,
    )
    result = runner.invoke(
        app, ["--base-url", "https://api.test", "chat", "Hi", "--stream", "-j"]
    )
    assert result.exit_code == 4, result.output
    assert result.stderr == ""
    assert json.loads(result.stdout)["error"]["code"] == code
    assert "partial" not in result.stdout
    assert route.call_count == 1
    assert stream.closed


@pytest.mark.parametrize("streaming", [False, True])
@respx.mock
def test_chat_http_errors_keep_json_contract(streaming):
    route = respx.post("https://api.test/v1/chat/completions").respond(
        401,
        json={
            "error": {"message": "Invalid credentials", "type": "authentication_error"}
        },
    )
    result = runner.invoke(
        app,
        [
            "--base-url",
            "https://api.test",
            "-j",
            "chat",
            "Hi",
            "--stream" if streaming else "--no-stream",
        ],
    )
    assert result.exit_code == 3
    assert result.stderr == ""
    assert (
        json.loads(result.stdout)["error"]["details"]["api_message"]
        == "Invalid credentials"
    )
    assert route.call_count == 1


@pytest.mark.parametrize("source", ["cli", "file", "generic"])
@respx.mock
def test_chat_invalid_input_fails_before_http(source, make_temp_dir):
    if source == "cli":
        command = ["chat", "Hi", "--temperature", "2"]
    else:
        directory = make_temp_dir("invalid-chat")
        request_file = directory / "request.json"
        request_file.write_text(
            json.dumps(
                {
                    "model": "glm-5.3",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                    "max_completion_tokens": 20,
                }
            )
        )
        command = ["chat"] if source == "file" else ["run", "/v1/chat/completions"]
        command += ["--input-file", str(request_file)]
    result = runner.invoke(app, ["--base-url", "https://api.test", "-j", *command])
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"]["code"] == "validation_error"
    assert result.stderr == ""
    assert not respx.calls


@respx.mock
def test_chat_cancellation_closes_stream_after_live_callback():
    from threading import Event

    cancelled = Event()
    stream = EventStream(
        frame(chunk({"content": "first"}))
        + frame(chunk({"content": "second"}, "stop"))
        + "data: [DONE]\n\n"
    )
    respx.post("https://api.test/v1/chat/completions").respond(
        200,
        headers={"content-type": "text/event-stream"},
        stream=stream,
    )
    services = ApplicationServices.with_client(
        APIClient(base_url="https://api.test", client_id="id", client_secret="secret")
    )
    received = []

    def on_chunk(event):
        received.append(event)
        cancelled.set()

    try:
        with pytest.raises(OperationCancelled):
            services.execute_chat(
                prompt="Hi",
                stream=True,
                execution=ExecutionOptions(cancel_event=cancelled),
                on_chunk=on_chunk,
            )
    finally:
        services.close()
    assert len(received) == 1
    assert stream.closed
