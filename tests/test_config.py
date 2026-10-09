"""Config precedence tests."""

import yaml

from app.config.manager import ConfigManager


def test_defaults_are_written_on_first_load(tmp_path):
    path = tmp_path / "config.yaml"
    config = ConfigManager(path).load()
    assert path.exists()
    assert config.image.edit_model == "seedream-v4.5"
    assert config.image.upscale_model == "clarity-ai-pro-upscaler"


def test_saved_values_are_read_back(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"image": {"edit_model": "flux-2-pro-image-to-image"}}))
    assert ConfigManager(path).load().image.edit_model == "flux-2-pro-image-to-image"


def test_environment_overrides_the_saved_file(tmp_path, monkeypatch):
    # The regression that made every SPARKLE_* in docker-compose a no-op from
    # the second boot onwards, once the app had written its own config file.
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"image": {"edit_model": "from-file"}, "ui": {"server_port": 7860}}))
    monkeypatch.setenv("SPARKLE_IMAGE__EDIT_MODEL", "from-env")
    monkeypatch.setenv("SPARKLE_UI__SERVER_PORT", "7899")

    config = ConfigManager(path).load()
    assert config.image.edit_model == "from-env"
    assert config.ui.server_port == 7899


def test_file_still_wins_over_defaults_for_untouched_keys(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"image": {"edit_model": "from-file", "timeout": 444}}))
    monkeypatch.setenv("SPARKLE_IMAGE__EDIT_MODEL", "from-env")
    config = ConfigManager(path).load()
    assert config.image.edit_model == "from-env"
    assert config.image.timeout == 444


def test_auth_tuple_needs_both_halves(tmp_path):
    manager = ConfigManager(tmp_path / "config.yaml")
    config = manager.load()
    assert config.auth_tuple is None
    config.ui.auth_username = "joe"
    assert config.auth_tuple is None
    config.ui.auth_password = "secret"
    assert config.auth_tuple == ("joe", "secret")
