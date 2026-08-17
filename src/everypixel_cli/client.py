"""Everypixel API HTTP transport.

The client stores connection settings, reuses one httpx connection pool, and
normalizes remote failures into typed application errors.
"""

from __future__ import annotations

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
            )
        except httpx.TimeoutException as exc:
            raise APIRequestError("API request timed out", code="api_timeout") from exc
        except httpx.RequestError as exc:
            raise APIRequestError(
                "Unable to connect to API", code="network_error"
            ) from exc
        if response.status_code >= 400:
            detail = extract_error_detail(response)
            exit_code, code = http_exit_code(response.status_code, detail)
            error_type = (
                AuthenticationError
                if response.status_code in (401, 403)
                else APIRequestError
            )
            raise error_type(
                "API request failed",
                code=code,
                exit_code=exit_code,
                details={
                    "status_code": response.status_code,
                    "api_message": detail[:500],
                },
            )
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise APIResponseError("API returned invalid JSON response") from exc

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
