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
        manager.reset("backend.unknown")
    with pytest.raises(ValidationError):
        manager.set("backend.max_retries", -1)
    with pytest.raises(ValidationError):
        manager.set("backend.temperature", 3)


def test_reset_stores_default_and_missing_file_loads_defaults(tmp_path):
    """Reset stores a model default and a missing file remains usable."""
    manager = ConfigurationManager(tmp_path)
    assert manager.load().backend.default_model == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    manager.initialize()
    manager.set("backend.default_model", "configured-model")

    assert (
        manager.reset("backend.default_model").backend.default_model
        == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    )


def test_session_reset_overrides_environment_without_writing_the_file(tmp_path):
    """A session reset stores the default without changing the configuration file."""
    manager = ConfigurationManager(tmp_path)
    manager.initialize()
    manager.load({"DEFAULT_MODEL": "environment-model"})
    original = manager.path.read_text(encoding="utf-8")

    settings = manager.set_session("backend.default_model", "session-model")

    assert settings.backend.default_model == "session-model"
    assert manager.source_for("backend.default_model") == "session"
    assert manager.path.read_text(encoding="utf-8") == original
    assert (
        manager.reset_session("backend.default_model").backend.default_model
        == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    )


def test_file_edit_retains_environment_precedence_and_exposes_entries(tmp_path):
    """Durable edits retain remembered environment precedence and schema-backed metadata."""
    manager = ConfigurationManager(tmp_path)
    manager.initialize()
    manager.load({"DEFAULT_MODEL": "environment-model"})

    manager.set("backend.default_model", "file-model")

    entry = next(item for item in manager.entries if item.path == "backend.api_key")
    assert manager.effective.backend.default_model == "environment-model"
    assert manager.source_for("backend.default_model") == "environment"
    assert entry.secret is True
    assert str(entry.value) == "**********"
    assert next(item for item in manager.entries if item.path == "loop.stream").choices == (
        True,
        False,
    )
    assert next(
        item for item in manager.entries if item.path == "backend.file_input_mode"
    ).choices == (
        "text",
        "native",
    )


def test_reset_all_stores_defaults_in_each_scope(tmp_path):
    """File and session resets each retain explicitly stored built-in defaults."""
    manager = ConfigurationManager(tmp_path)
    manager.initialize()
    manager.load()
    manager.set("loop.debug", True)
    manager.set_session("loop.stream", False)

    assert manager.reset_all(scope="session").loop.stream is True
    assert manager.reset_all(scope="file").loop.debug is False
    assert 'api_key = "local-api-key"' in manager.path.read_text(encoding="utf-8")
    assert manager.source_for("loop.debug") == "session"
    assert manager.source_for("loop.stream") == "session"


def test_unset_removes_a_file_value_and_reveals_environment_precedence(tmp_path):
    """Removing a file value exposes the lower-precedence environment value."""
    manager = ConfigurationManager(tmp_path)
    manager.initialize()
    manager.load({"DEFAULT_MODEL": "environment-model"})
    manager.set("backend.default_model", "file-model")

    settings = manager.unset("backend.default_model")

    assert settings.backend.default_model == "environment-model"
    assert manager.source_for("backend.default_model") == "environment"
    assert "default_model" not in manager.path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration field"):
        manager.unset("backend.default_model")


def test_unset_session_removes_one_override_and_requires_an_existing_value(tmp_path):
    """Removing a session value exposes lower scopes and rejects a second removal."""
    manager = ConfigurationManager(tmp_path)
    manager.initialize()
    manager.load({"DEFAULT_MODEL": "environment-model"})
    manager.set_session("backend.default_model", "session-model")

    settings = manager.unset_session("backend.default_model")

    assert settings.backend.default_model == "environment-model"
    assert manager.source_for("backend.default_model") == "environment"
    with pytest.raises(ValueError, match="No session override"):
        manager.unset_session("backend.default_model")


def test_unset_all_removes_only_the_selected_scope(tmp_path):
    """Clearing a scope leaves higher-precedence session values in effect."""
    manager = ConfigurationManager(tmp_path)
    manager.initialize()
    manager.load()
    manager.set("loop.debug", True)
    manager.set_session("loop.stream", False)

    assert manager.unset_all(scope="file").loop.debug is False
    assert manager.effective.loop.stream is False
    assert manager.source_for("loop.debug") == "default"
    assert manager.source_for("loop.stream") == "session"
    document = manager.path.read_text(encoding="utf-8")
    assert "config_version" not in document
    assert "[loop]" not in document
    assert manager.unset_all(scope="session").loop.stream is True
