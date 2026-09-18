"""Everypixel API HTTP transport.

The client stores connection settings, reuses one httpx connection pool, and
normalizes remote failures into typed application errors.
"""

from __future__ import annotations

import json as jsonlib
from collections.abc import Iterator
from typing import Any

import httpx

from .errors import (
    APIRequestError,
    APIResponseError,
    AuthenticationError,
    http_exit_code,
)


class APIClient:
    """Thin HTTP API wrapper with Basic Auth and normalized errors."""

    def __init__(
        self,
        *,
        base_url: str,
        client_id: str | None,
        client_secret: str | None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = timeout
        self._http = httpx.Client(timeout=timeout, follow_redirects=True)

    @property
    def auth(self) -> tuple[str, str] | None:
        """Return a Basic Auth pair or None."""

        if self.client_id and self.client_secret:
            return self.client_id, self.client_secret
        return None

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
    ) -> Any:
        """Perform an HTTP request and return an arbitrary decoded JSON value."""

        if auth_required and not self.auth:
            raise AuthenticationError(
                "Credentials are not configured", code="auth_missing"
            )
        url = self.base_url + (path if path.startswith("/") else f"/{path}")
        try:
            response = self._http.request(
                method,
                url,
                json=json,
                params=drop_none(params),
                files=files,
                auth=self.auth,
                timeout=self.timeout if timeout is None else timeout,
            )
        except httpx.TimeoutException as exc:
            raise APIRequestError("API request timed out", code="api_timeout") from exc
        except httpx.RequestError as exc:
            raise APIRequestError(
                "Unable to connect to API", code="network_error"
            ) from exc
        raise_for_response(response)
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise APIResponseError("API returned invalid JSON response") from exc

    def stream_request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> Iterator[dict[str, Any]]:
        """Decode an SSE response and require its explicit completion marker."""

        if not self.auth:
            raise AuthenticationError(
                "Credentials are not configured", code="auth_missing"
            )
        url = self.base_url + (path if path.startswith("/") else f"/{path}")
        try:
            with self._http.stream(
                method, url, json=json, auth=self.auth, timeout=timeout
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise_for_response(response)
                if response.headers.get("content-type", "").split(";")[0].strip() != (
                    "text/event-stream"
                ):
                    raise APIResponseError("API did not return an event stream")
                data: list[str] = []
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        data.append(line[5:].removeprefix(" "))
                    elif not line and data:
                        event = "\n".join(data)
                        data.clear()
                        if event == "[DONE]":
                            return
                        yield parse_stream_event(event)
                if data == ["[DONE]"]:
                    return
                raise APIResponseError(
                    "API stream ended before [DONE]", code="incomplete_stream"
                )
        except httpx.TimeoutException as exc:
            raise APIRequestError("API request timed out", code="api_timeout") from exc
        except httpx.RequestError as exc:
            raise APIRequestError(
                "Unable to connect to API", code="network_error"
            ) from exc

    def get_status(self, task_id: str) -> Any:
        """Fetch async task state by task_id."""

        return self.request(
            "GET",
            "/v1/status",
            params={"task_id": task_id},
            auth_required=False,
        )

    def check_auth(self) -> None:
        """Check that Basic Auth is accepted by the API."""

        if not self.auth:
            raise AuthenticationError(
                "Credentials are not configured", code="auth_missing"
            )
        self.request("GET", "/v1/auth/check")

    def close(self) -> None:
        """Close the shared HTTP connection pool."""

        self._http.close()

    def __enter__(self) -> "APIClient":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()


def drop_none(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop None values from query params."""

    if data is None:
        return None
    return {key: value for key, value in data.items() if value is not None}


def raise_for_response(response: httpx.Response) -> None:
    """Normalize HTTP failures consistently for JSON and streaming requests."""

    if response.status_code < 400:
        return
    detail = extract_error_detail(response)
    exit_code, code = http_exit_code(response.status_code, detail)
    error_type = (
        AuthenticationError if response.status_code in (401, 403) else APIRequestError
    )
    raise error_type(
        "API request failed",
        code=code,
        exit_code=exit_code,
        details={"status_code": response.status_code, "api_message": detail[:500]},
    )


def parse_stream_event(data: str) -> dict[str, Any]:
    try:
        event = jsonlib.loads(data)
    except ValueError as exc:
        raise APIResponseError("API returned invalid stream JSON") from exc
    if not isinstance(event, dict):
        raise APIResponseError("API returned an invalid stream event")
    if "error" in event:
        error = event["error"]
        message = error.get("message") if isinstance(error, dict) else None
        raise APIRequestError(
            "Chat stream failed",
            code="chat_stream_error",
            details={"api_message": message[:500]} if isinstance(message, str) else {},
        )
    return event


def extract_error_detail(response: httpx.Response) -> str:
    """Extract a human-readable error from an API response."""

    try:
        data = response.json()
    except ValueError:
        return response.text
    detail = data.get("detail") or data.get("message") or data.get("error") or data
    return format_error_detail(detail)


def format_error_detail(detail: Any) -> str:
    """Normalize different API error shapes into one string."""

    if isinstance(detail, list):
        return "; ".join(format_error_detail(item) for item in detail)
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("msg") or detail)
    return str(detail)
