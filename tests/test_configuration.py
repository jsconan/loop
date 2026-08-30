"""Tests for project configuration loading and persistence."""

import pytest
from pydantic import ValidationError

from loop.configuration import ConfigurationManager


def test_initialize_creates_private_commented_defaults(tmp_path):
    """A new project receives a complete editable default configuration file."""
    manager = ConfigurationManager(tmp_path)

    path = manager.initialize()

    content = path.read_text(encoding="utf-8")
    assert "# Loop project configuration" in content
    assert 'api_key = "local-api-key"' in content
    assert path.stat().st_mode & 0o777 == 0o600
    assert manager.load().backend.api_key.get_secret_value() == "local-api-key"


def test_initialize_returns_existing_configuration_without_replacing_it(tmp_path):
    """Initializing an existing project leaves its configuration unchanged."""
    manager = ConfigurationManager(tmp_path)
    path = manager.initialize()
    original = path.read_text(encoding="utf-8")

    assert manager.initialize() == path
    assert path.read_text(encoding="utf-8") == original


def test_environment_overrides_persisted_values(tmp_path):
    """Environment values take precedence over configuration-file values."""
    manager = ConfigurationManager(tmp_path)
    manager.initialize()
    manager.set("backend.default_model", "file-model")
    manager.set("backend.temperature", 0.2)
    manager.set("backend.reasoning_effort", "high")
    manager.set("backend.hyperparameter_policy", "strict")

    settings = manager.load(
        {
            "DEFAULT_MODEL": "environment-model",
            "OPENAI_TEMPERATURE": "0.7",
            "OPENAI_REASONING_EFFORT": "low",
            "OPENAI_HYPERPARAMETER_POLICY": "fallback",
        }
    )

    assert settings.backend.default_model == "environment-model"
    assert settings.backend.temperature == 0.7
    assert settings.backend.reasoning_effort == "low"
    assert settings.backend.hyperparameter_policy == "fallback"


def test_set_preserves_existing_comments(tmp_path):
    """Saving a setting retains manually authored TOML comments."""
    manager = ConfigurationManager(tmp_path)
    manager.initialize()
    manager.path.write_text(
        manager.path.read_text(encoding="utf-8").replace(
            'default_model = "nvidia/Qwen3.6-35B-A3B-NVFP4"',
            '# Project-specific model\ndefault_model = "nvidia/Qwen3.6-35B-A3B-NVFP4"',
        ),
        encoding="utf-8",
    )

    manager.set("backend.default_model", "configured-model")

    assert "# Project-specific model" in manager.path.read_text(encoding="utf-8")
    assert manager.load().backend.default_model == "configured-model"


def test_manager_exposes_effective_values_sources_and_reload(tmp_path):
    """Loaded settings support redacted reads, provenance, and explicit reloads."""
    manager = ConfigurationManager(tmp_path)
    manager.initialize()

    with pytest.raises(RuntimeError, match="not been loaded"):
        _ = manager.effective
    settings = manager.load({"DEFAULT_MODEL": "environment-model"})

    assert manager.effective is settings
    assert manager.get("backend.default_model") == "environment-model"
    assert str(manager.get("backend.api_key")) == "**********"
    assert manager.source_for("backend.default_model") == "environment"
    assert manager.source_for("backend.api_key") == "config"
    assert (
        manager.reload({"DEFAULT_MODEL": "reloaded-model"}).backend.default_model
        == "reloaded-model"
    )


@pytest.mark.parametrize("path", ["backend", "backend.default_model.extra"])
def test_manager_rejects_malformed_configuration_paths(tmp_path, path):
    """Public configuration access requires exactly one section and field."""
    manager = ConfigurationManager(tmp_path)
    manager.initialize()
    manager.load()

    with pytest.raises(ValueError, match="section.field"):
        manager.get(path)


def test_manager_rejects_unknown_fields_and_invalid_values(tmp_path):
    """Edits validate schema membership and values before persisting changes."""
    manager = ConfigurationManager(tmp_path)
    manager.initialize()

    with pytest.raises(ValueError, match="Unknown configuration field"):
        manager.set("backend.unknown", "value")
    with pytest.raises(ValueError, match="Unknown configuration field"):
        manager.unset("backend.unknown")
    with pytest.raises(ValidationError):
        manager.set("backend.max_retries", -1)
    with pytest.raises(ValidationError):
        manager.set("backend.temperature", 3)


def test_unset_restores_default_and_missing_file_loads_defaults(tmp_path):
    """Removing a value restores its model default and a missing file remains usable."""
    manager = ConfigurationManager(tmp_path)
    assert manager.load().backend.default_model == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    manager.initialize()
    manager.set("backend.default_model", "configured-model")

    assert (
        manager.unset("backend.default_model").backend.default_model
        == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    )
