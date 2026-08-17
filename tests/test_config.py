from everypixel_cli.cli import create_profile_payload
from everypixel_cli import config as config_module
from everypixel_cli.config import (
    AppConfig,
    ProfileConfig,
    config_path,
    delete_credentials,
    load_config,
    resolve_profile_name,
    safe_config_payload,
    save_credentials,
)


def test_safe_config_payload_masks_fallback_secret():
    config = AppConfig(
        current_profile="default",
        profiles={
            "default": ProfileConfig(
                base_url="https://api.test",
                client_id="client",
                client_secret="secret-value",
            )
        },
    )

    payload = safe_config_payload(config)

    assert payload["profiles"]["default"]["client_id"] == "client"
    assert payload["profiles"]["default"]["client_secret"] == "secr********"


def test_create_profile_does_not_copy_default_credentials(monkeypatch):
    config = AppConfig(
        profiles={
            "default": ProfileConfig(
                base_url="https://api.test",
                client_id="client",
                client_secret="secret-value",
            )
        }
    )
    saved = {}

    monkeypatch.setattr("everypixel_cli.cli.load_config", lambda: config)
    monkeypatch.setattr(
        "everypixel_cli.cli.save_config", lambda value: saved.update({"config": value})
    )

    payload = create_profile_payload("dev", "http://localhost:7862")

    assert payload == {
        "status": "ok",
        "profile": "dev",
        "base_url": "http://localhost:7862",
    }
    assert saved["config"].profiles["dev"].base_url == "http://localhost:7862"
    assert saved["config"].profiles["dev"].client_id is None
    assert saved["config"].profiles["dev"].client_secret is None


def test_explicit_profile_overrides_environment(monkeypatch):
    monkeypatch.setenv("EVERYPIXEL_PROFILE", "from-environment")

    profile = resolve_profile_name(AppConfig(current_profile="saved"), "explicit")

    assert profile == "explicit"


def test_dev_profile_overrides_other_profile_sources(monkeypatch):
    monkeypatch.setenv("EVERYPIXEL_PROFILE", "from-environment")

    profile = resolve_profile_name(
        AppConfig(current_profile="saved"), profile="explicit", dev=True
    )

    assert profile == "dev"


def test_save_credentials_uses_keyring_without_persisting_secret(monkeypatch):
    stored = {}
    monkeypatch.setattr(
        config_module.keyring,
        "set_password",
        lambda service, username, password: stored.update(
            service=service,
            username=username,
            password=password,
        ),
    )

    assert save_credentials("prod", "client", "secret") is True

    profile = load_config().profiles["prod"]
    assert profile.client_id == "client"
    assert profile.client_secret is None
    assert stored["password"] == "secret"


def test_save_credentials_falls_back_to_private_config_file(monkeypatch):
    monkeypatch.setattr(
        config_module.keyring,
        "set_password",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    assert save_credentials("prod", "client", "secret") is False

    profile = load_config().profiles["prod"]
    assert profile.client_secret == "secret"
    assert config_path().stat().st_mode & 0o777 == 0o600


def test_delete_credentials_clears_config_and_keyring(monkeypatch):
    monkeypatch.setattr(
        config_module.keyring,
        "set_password",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    save_credentials("prod", "client", "secret")
    deleted = {}
    monkeypatch.setattr(
        config_module.keyring,
        "delete_password",
        lambda service, username: deleted.update(service=service, username=username),
    )

    delete_credentials("prod")

    profile = load_config().profiles["prod"]
    assert profile.client_id is None
    assert profile.client_secret is None
    assert deleted["username"] == "client"
