from __future__ import annotations

from typing import Any

import httpx

from everypixel_cli.client import APIClient


class FakeHTTPClient:
    def __init__(self, **options: Any) -> None:
        self.options = options
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def request(self, method: str, url: str, **_kwargs: Any) -> httpx.Response:
        self.calls.append((method, url))
        return httpx.Response(200, json={"ok": True})

    def close(self) -> None:
        self.closed = True


def test_api_client_reuses_and_closes_one_http_client(monkeypatch):
    clients: list[FakeHTTPClient] = []

    def client_factory(**options: Any) -> FakeHTTPClient:
        client = FakeHTTPClient(**options)
        clients.append(client)
        return client

    monkeypatch.setattr("everypixel_cli.client.httpx.Client", client_factory)
    client = APIClient(base_url="https://api.test", client_id=None, client_secret=None)

    client.request("GET", "/v1/one", auth_required=False)
    client.request("GET", "/v1/two", auth_required=False)
    client.close()

    assert len(clients) == 1
    assert clients[0].calls == [
        ("GET", "https://api.test/v1/one"),
        ("GET", "https://api.test/v1/two"),
    ]
    assert clients[0].closed is True
