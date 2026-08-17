"""Local CLI configuration and credential storage.

The module keeps public client_id values in the JSON profile config and stores
client_secret in the system keyring when possible. If keyring is unavailable,
config.json is used as a fallback so the CLI still works in minimal setups.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

import keyring
from platformdirs import user_config_dir
from pydantic import BaseModel, Field, HttpUrl

from .errors import ConfigurationError, FileWriteError, ValidationCLIError, mask_secret


DEFAULT_BASE_URL = "https://api.everypixel.com"
SERVICE_NAME = "everypixel-cli"


class ProfileConfig(BaseModel):
    """Settings for one Everypixel API profile."""

    base_url: str = DEFAULT_BASE_URL
    client_id: str | None = None
    client_secret: str | None = None


class AppConfig(BaseModel):
    """Full local CLI configuration."""

    current_profile: str = "default"
    profiles: dict[str, ProfileConfig] = Field(
        default_factory=lambda: {"default": ProfileConfig()}
    )


def config_dir() -> Path:
    """Return the user config directory."""

    return Path(user_config_dir("everypixel-cli", "Everypixel"))


def config_path() -> Path:
    """Return the config.json path."""

    return config_dir() / "config.json"


def load_config() -> AppConfig:
    """Read config.json or return defaults."""

    path = config_path()
    if not path.exists():
        return AppConfig()
    try:
        return AppConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(
            "Unable to read configuration",
            code="config_read_error",
            details={"path": str(path)},
        ) from exc


def save_config(config: AppConfig) -> None:
    """Save config and try to restrict file permissions."""

    path = config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(config.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with suppress(OSError):
            path.chmod(0o600)
    except OSError as exc:
        raise FileWriteError(
            "Unable to write configuration", details={"path": str(path)}
        ) from exc


def resolve_profile_name(
    config: AppConfig,
    profile: str | None = None,
    dev: bool = False,
) -> str:
    """Resolve the active profile name."""

    env_profile = os.getenv("EVERYPIXEL_PROFILE")
    if dev:
        return "dev"
    return profile or env_profile or config.current_profile


def get_or_create_profile(config: AppConfig, name: str) -> ProfileConfig:
    """Return an existing profile or create it."""

    if name not in config.profiles:
        config.profiles[name] = ProfileConfig()
    return config.profiles[name]


def keyring_name(profile: str) -> str:
    """Build a keyring service name for a profile."""

    return f"{SERVICE_NAME}:{profile}"


def save_credentials(profile: str, client_id: str, client_secret: str) -> bool:
    """Save profile credentials.

    Returns True when the secret was stored in keyring. False means the secret
    was stored in config.json as a fallback.
    """

    config = load_config()
    profile_config = get_or_create_profile(config, profile)
    profile_config.client_id = client_id
    profile_config.client_secret = None
    try:
        keyring.set_password(keyring_name(profile), client_id, client_secret)
        keyring_ok = True
    except Exception:  # noqa: BLE001
        profile_config.client_secret = client_secret
        keyring_ok = False
    save_config(config)
    return keyring_ok


def delete_credentials(profile: str) -> None:
    """Delete profile client_id/client_secret from config and keyring."""

    config = load_config()
    profile_config = get_or_create_profile(config, profile)
    client_id = profile_config.client_id
    if client_id:
        # Keyring backends differ in how they report an already-missing entry.
        with suppress(Exception):
            keyring.delete_password(keyring_name(profile), client_id)
    profile_config.client_id = None
    profile_config.client_secret = None
    save_config(config)


def resolved_settings(
    profile: str | None = None,
    dev: bool = False,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Resolve effective settings for one run.

    Source priority is CLI options, environment variables, then saved profile.
    The secret comes from env/config or is fetched from keyring.
    """

    config = load_config()
    profile_name = resolve_profile_name(config, profile=profile, dev=dev)
    profile_config = get_or_create_profile(config, profile_name)

    client_id = os.getenv("EVERYPIXEL_CLIENT_ID") or profile_config.client_id
    client_secret = (
        os.getenv("EVERYPIXEL_CLIENT_SECRET") or profile_config.client_secret
    )
    if client_id and not client_secret:
        try:
            client_secret = keyring.get_password(keyring_name(profile_name), client_id)
        except Exception:
            client_secret = None

    return {
        "profile": profile_name,
        "base_url": (
            base_url or os.getenv("EVERYPIXEL_BASE_URL") or profile_config.base_url
        ),
        "client_id": client_id,
        "client_secret": client_secret,
    }


def set_config_value(profile: str, key: str, value: str) -> None:
    """Set a supported profile option."""

    config = load_config()
    profile_config = get_or_create_profile(config, profile)
    if key != "base_url":
        raise ValidationCLIError("Unsupported configuration key", details={"key": key})
    profile_config.base_url = str(HttpUrl(value))
    save_config(config)


def use_profile(profile: str) -> None:
    """Make a profile active."""

    config = load_config()
    get_or_create_profile(config, profile)
    config.current_profile = profile
    save_config(config)


def safe_config_payload(config: AppConfig) -> dict[str, Any]:
    """Return config data without exposing secrets."""

    return {
        "current_profile": config.current_profile,
        "profiles": {
            name: {
                "base_url": profile.base_url,
                "client_id": profile.client_id,
                "client_secret": mask_secret(profile.client_secret),
            }
            for name, profile in config.profiles.items()
        },
    }
