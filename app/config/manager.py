"""Configuration manager for loading and saving YAML settings."""

import os
from pathlib import Path
from typing import Optional

import yaml

from app.config.models import AppConfig


DEFAULT_CONFIG_PATH = Path("data/config.yaml")


class ConfigManager:
    """Manages application configuration persistence."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._config: Optional[AppConfig] = None

    def load(self) -> AppConfig:
        """Load configuration from YAML file or create defaults."""
        if self._config is not None:
            return self._config

        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._config = AppConfig.model_validate(data)
        else:
            self._config = AppConfig()
            self.save(self._config)

        return self._config

    def save(self, config: Optional[AppConfig] = None) -> None:
        """Save configuration to YAML file."""
        config = config or self._config
        if config is None:
            raise ValueError("No configuration to save")

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                config.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        self._config = config

    def reload(self) -> AppConfig:
        """Force reload from disk."""
        self._config = None
        return self.load()

    def update_from_env(self) -> AppConfig:
        """Reload config, allowing environment variables to override."""
        self._config = AppConfig()
        self.save(self._config)
        return self._config


# Global singleton instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get or create the global ConfigManager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> AppConfig:
    """Convenience function to get the current configuration."""
    return get_config_manager().load()
