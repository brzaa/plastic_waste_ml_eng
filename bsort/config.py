"""Configuration management for bsort."""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration manager for bsort application."""

    def __init__(self, config_path: str = "settings.yaml"):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self._config.get("logging", {})
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format=log_config.get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'training.epochs')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Get configuration section."""
        return self._config[key]

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()


def load_config(config_path: str = "settings.yaml") -> Config:
    """
    Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Config object
    """
    return Config(config_path)
