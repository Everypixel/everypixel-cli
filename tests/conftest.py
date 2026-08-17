from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_user_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep tests independent from developer profiles, credentials, and cache."""

    from everypixel_cli import config

    for name in (
        "EVERYPIXEL_PROFILE",
        "EVERYPIXEL_BASE_URL",
        "EVERYPIXEL_CLIENT_ID",
        "EVERYPIXEL_CLIENT_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    config_root = tmp_path / "user-config"
    monkeypatch.setattr(
        config,
        "user_config_dir",
        lambda *_args, **_kwargs: str(config_root),
    )
    secrets: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(
        config.keyring,
        "set_password",
        lambda service, username, password: secrets.__setitem__(
            (service, username), password
        ),
    )
    monkeypatch.setattr(
        config.keyring,
        "get_password",
        lambda service, username: secrets.get((service, username)),
    )
    monkeypatch.setattr(
        config.keyring,
        "delete_password",
        lambda service, username: secrets.pop((service, username), None),
    )


@pytest.fixture
def make_temp_dir(tmp_path: Path) -> Callable[[str], Path]:
    """Return a per-test directory factory safe for parallel test execution."""

    def create(name: str) -> Path:
        target = tmp_path / name
        target.mkdir(parents=True)
        return target

    return create
