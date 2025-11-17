"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
import yaml

from bsort.config import Config, load_config


def test_config_load():
    """Test configuration loading from YAML file."""
    # Create temporary config file
    config_data = {
        "data": {"raw_dir": "data/raw", "processed_dir": "data/processed"},
        "training": {"epochs": 100, "batch_size": 16},
        "classes": {"names": ["light_blue", "dark_blue", "others"]},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        config = Config(temp_path)

        # Test direct access
        assert config["data"]["raw_dir"] == "data/raw"
        assert config["training"]["epochs"] == 100

        # Test dot notation
        assert config.get("training.epochs") == 100
        assert config.get("training.batch_size") == 16

        # Test default value
        assert config.get("nonexistent.key", "default") == "default"

    finally:
        Path(temp_path).unlink()


def test_config_missing_file():
    """Test error handling for missing config file."""
    with pytest.raises(FileNotFoundError):
        Config("nonexistent.yaml")


def test_config_nested_get():
    """Test nested key access."""
    config_data = {"level1": {"level2": {"level3": "value"}}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        config = Config(temp_path)
        assert config.get("level1.level2.level3") == "value"
        assert config.get("level1.level2.nonexistent") is None
        assert config.get("level1.level2.nonexistent", "default") == "default"

    finally:
        Path(temp_path).unlink()
