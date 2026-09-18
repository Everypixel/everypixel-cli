"""Chat request construction and completion assembly without presentation code."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any, Protocol

from ..errors import APIResponseError, ValidationCLIError
from ..schemas import ChatRequest
from .models import ExecutionOptions, OperationResult


class ChatClientProtocol(Protocol):
    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        auth_required: bool = True,
        timeout: float | None = None,
    ) -> Any: ...

    def stream_request(
        self, method: str, path: str, *, json: dict[str, Any], timeout: float
    ) -> Iterator[dict[str, Any]]: ...


def build_chat_payload(
    *,
    payload: dict[str, Any],
    prompt: str | None,
    system: str | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    values = {
        **payload,
        **{key: value for key, value in options.items() if value is not None},
    }
    values.setdefault("model", "glm-5.3")
    messages = values.get("messages", [])
    if not isinstance(messages, list):
        raise ValidationCLIError("messages must be an array")
    messages = list(messages)
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})
    values["messages"] = messages
    if values.get("stream") and "stream_options" not in values:
        values["stream_options"] = {"include_usage": True}
    return ChatRequest.model_validate(values).model_dump(mode="json", exclude_none=True)


class ChatCompletion:
    """Combine text, reasoning, and tool-call fragments into a completion."""

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.choices: dict[int, dict[str, Any]] = {}
        self.tools: dict[int, dict[int, dict[str, Any]]] = {}

    def add(self, event: dict[str, Any]) -> None:
        choices = event.get("choices")
        if not isinstance(choices, list):
            raise APIResponseError("API returned invalid chat choices")
        self.metadata.update(
            {key: value for key, value in event.items() if key != "choices"}
        )
        for item in choices:
            if not isinstance(item, dict):
                raise APIResponseError("API returned an invalid chat choice")
            index = item.get("index")
            delta = item.get("delta")
            if type(index) is not int or index < 0 or not isinstance(delta, dict):
                raise APIResponseError("API returned an invalid chat delta")
            choice = self.choices.setdefault(
                index,
                {
                    "index": index,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": None,
                },
            )
            choice.update(
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"delta", "finish_reason"}
                }
            )
            if item.get("finish_reason") is not None:
                choice["finish_reason"] = item["finish_reason"]
            message = choice["message"]
            for key in ("content", "reasoning_content", "refusal"):
                append_fragment(message, delta, key)
            if "role" in delta:
                message["role"] = delta["role"]
            tool_calls = delta.get("tool_calls")
            if tool_calls is not None:
                self._add_tools(index, tool_calls)

    def _add_tools(self, choice_index: int, calls: Any) -> None:
        if not isinstance(calls, list):
            raise APIResponseError("API returned invalid tool calls")
        tools = self.tools.setdefault(choice_index, {})
        for call in calls:
            if not isinstance(call, dict):
                raise APIResponseError("API returned an invalid tool call")
            index = call.get("index")
            if type(index) is not int or index < 0:
                raise APIResponseError("API returned an invalid tool call index")
            tool = tools.setdefault(index, {"function": {"name": "", "arguments": ""}})
            append_fragment(tool, call, "id")
            if "type" in call:
                tool["type"] = call["type"]
            function = call.get("function")
            if function is not None:
                if not isinstance(function, dict):
                    raise APIResponseError("API returned an invalid function call")
                for key in ("name", "arguments"):
                    append_fragment(tool["function"], function, key)

    def result(self) -> dict[str, Any]:
        if not self.choices or any(
            choice["finish_reason"] is None for choice in self.choices.values()
        ):
            raise APIResponseError("API returned an unfinished chat completion")
        for index, tools in self.tools.items():
            self.choices[index]["message"]["tool_calls"] = [
                tools[key] for key in sorted(tools)
            ]
        return {
            **self.metadata,
            "object": "chat.completion",
            "choices": [self.choices[index] for index in sorted(self.choices)],
        }


def append_fragment(target: dict[str, Any], delta: dict[str, Any], key: str) -> None:
    value = delta.get(key)
    if value is None:
        return
    if not isinstance(value, str):
        raise APIResponseError("API returned an invalid chat text fragment")
    target[key] = (target.get(key) or "") + value


class ChatService:
    def __init__(self, client: ChatClientProtocol) -> None:
        self.client = client

    def execute(
        self,
        payload: dict[str, Any],
        *,
        request_timeout: float = 240.0,
        execution: ExecutionOptions | None = None,
        on_chunk: Callable[[dict[str, Any]], None] | None = None,
    ) -> OperationResult:
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValidationCLIError("request timeout must be a finite positive number")
        execution = execution or ExecutionOptions()
        execution.check_cancelled()
        if not payload.get("stream"):
            value = self.client.request(
                "POST", "/v1/chat/completions", json=payload, timeout=request_timeout
            )
            execution.check_cancelled()
            return OperationResult(value=value)
        completion = ChatCompletion()
        events = self.client.stream_request(
            "POST", "/v1/chat/completions", json=payload, timeout=request_timeout
        )
        # Closing the generator also closes its HTTP response on cancellation
        # or a rendering failure, without starting another generation.
        try:
            for event in events:
                execution.check_cancelled()
                completion.add(event)
                if on_chunk is not None:
                    on_chunk(event)
                execution.check_cancelled()
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()
        execution.check_cancelled()
        return OperationResult(value=completion.result())
