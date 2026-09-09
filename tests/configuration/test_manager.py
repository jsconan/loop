"""Tests for workspace configuration loading and persistence."""

import pytest
from pydantic import ValidationError

from loop.configuration import ConfigurationManager


def test_manager_snapshots_its_configuration_path(tmp_path):
    """Configuration owns one immutable destination selected at construction."""
    path = tmp_path / "first.toml"
    manager = ConfigurationManager(path)
    manager.initialize()
    manager.load()
    manager.save()

    assert manager.path == path
    assert path.is_file()


def test_user_defaults_merge_with_sparse_workspace_overrides(tmp_path):
    """A complete user document remains editable while the workspace stores only overrides."""
    user_path = tmp_path / "config" / "config.toml"
    workspace_path = tmp_path / "project" / ".loop" / "config.toml"
    manager = ConfigurationManager(user_path, workspace_path)
    manager.initialize()
    manager.load()

    manager.set("loop.debug", True)
    manager.set("loop.stream", False, scope="user")

    assert "stream = false" in user_path.read_text(encoding="utf-8")
    assert workspace_path.read_text(encoding="utf-8") == "[loop]\ndebug = true\n"
    assert manager.effective.loop.debug is True
    assert manager.effective.loop.stream is False
    assert manager.source_for("loop.debug") == "workspace"
    assert manager.source_for("loop.stream") == "user"

    manager.set("loop.debug", True, scope="user")
    manager.set("loop.stream", True, scope="workspace")
    manager.reset("loop.debug", scope="workspace")

    assert manager.effective.loop.debug is True
    assert manager.effective.loop.stream is True
    assert manager.source_for("loop.debug") == "user"
    workspace_document = workspace_path.read_text(encoding="utf-8")
    assert "debug" not in workspace_document
    assert "stream = true" in workspace_document

    manager.set("loop.debug", True, scope="workspace")
    assert manager.unset_all(scope="workspace").loop.debug is True
    assert not workspace_path.exists()


def test_sparse_workspace_reset_and_validation_cover_layer_edge_cases(tmp_path):
    """Workspace-wide reset clears overrides and malformed sparse operations fail safely."""
    user_path = tmp_path / "config.toml"
    workspace_path = tmp_path / ".loop" / "config.toml"
    manager = ConfigurationManager(user_path, workspace_path)
    manager.initialize()
    manager.load()
    manager.set("loop.debug", True, scope="workspace")
    manager.set("loop.stream", False, scope="workspace")
    manager.unset("loop.debug", scope="workspace")
    assert manager.effective.loop.stream is False

    assert manager.reset_all(scope="workspace").loop.debug is False
    assert manager.reset_all(scope="user").loop.debug is False
    with pytest.raises(ValueError, match="Unknown configuration field"):
        manager.unset("loop.debug", scope="workspace")

    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path.write_text("config_version = 1\nloop = true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        manager.load()


def test_single_path_manager_rejects_workspace_writes(tmp_path):
    """Managers without a workspace path reject writes to the workspace scope."""
    manager = ConfigurationManager(tmp_path / "config.toml")
    manager.initialize()
    manager.load()

    with pytest.raises(RuntimeError, match="Workspace path is not configured"):
        manager.set("loop.debug", True)


def test_user_reset_and_unset_restore_the_user_scope(tmp_path):
    """User-scope edits persist in the complete document and do not create workspace overrides."""
    user_path = tmp_path / "config.toml"
    workspace_path = tmp_path / ".loop" / "config.toml"
    manager = ConfigurationManager(user_path, workspace_path)
    manager.initialize()
    manager.load()
    manager.set("loop.debug", True, scope="user")

    manager.reset("loop.debug", scope="user")

    assert manager.effective.loop.debug is False
    assert manager.source_for("loop.debug") == "user"
    assert not workspace_path.exists()
    assert "debug = false" in user_path.read_text(encoding="utf-8")

    manager.set("loop.debug", True, scope="user")
    manager.unset("loop.debug", scope="user")

    assert manager.effective.loop.debug is False
    assert "debug = false" in user_path.read_text(encoding="utf-8")


def test_invalid_session_value_is_rolled_back(tmp_path):
    """Rejecting an invalid session override leaves the prior effective settings unchanged."""
    manager = ConfigurationManager(tmp_path / "config.toml")
    manager.initialize()
    manager.load()

    with pytest.raises(ValidationError):
        manager.set_session("loop.temperature", 3)

    assert manager.effective.loop.temperature is None
    with pytest.raises(ValueError, match="No session override"):
        manager.unset_session("loop.temperature")


def test_reset_rejects_unknown_default_field(tmp_path):
    """Reset rejects syntactically valid paths that are absent from the settings model."""
    manager = ConfigurationManager(tmp_path / "config.toml")
    manager.initialize()
    manager.load()

    with pytest.raises(ValueError, match="Unknown configuration field"):
        manager.reset("backend.unknown")


def test_initialize_creates_private_commented_defaults(tmp_path):
    """A new project receives a complete editable default configuration file."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")

    path = manager.initialize()

    content = path.read_text(encoding="utf-8")
    assert "# Loop workspace configuration" in content
    assert 'api_key = "local-api-key"' in content
    assert "# context_window = <unset>" in content
    assert "[tools]" in content
    assert "command_timeout = 30.0" in content
    assert path.stat().st_mode & 0o777 == 0o600
    assert manager.load().backend.api_key.get_secret_value() == "local-api-key"


def test_initialize_returns_existing_configuration_without_replacing_it(tmp_path):
    """Initializing an existing project leaves its configuration unchanged."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    path = manager.initialize()
    original = path.read_text(encoding="utf-8")

    assert manager.initialize() == path
    assert path.read_text(encoding="utf-8") == original


def test_environment_overrides_persisted_values(tmp_path):
    """Environment values take precedence over configuration-file values."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    manager.initialize()
    manager.set("backend.default_model", "file-model", scope="user")
    manager.set("loop.temperature", 0.2, scope="user")
    manager.set("loop.reasoning_effort", "high", scope="user")
    manager.set("backend.hyperparameter_policy", "strict", scope="user")

    settings = manager.load(
        {
            "DEFAULT_MODEL": "environment-model",
            "LOOP_COMMAND_TIMEOUT": "0.25",
            "LOOP_TEMPERATURE": "0.7",
            "LOOP_REASONING_EFFORT": "low",
            "OPENAI_HYPERPARAMETER_POLICY": "fallback",
        }
    )

    assert settings.backend.default_model == "environment-model"
    assert settings.tools.command_timeout == 0.25
    assert settings.loop.temperature == 0.7
    assert settings.loop.reasoning_effort == "low"
    assert settings.backend.hyperparameter_policy == "fallback"


def test_command_timeout_requires_a_positive_duration(tmp_path):
    """Configuration rejects a command lifecycle timeout that cannot make progress."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    manager.initialize()
    manager.load()

    with pytest.raises(ValidationError):
        manager.set_session("tools.command_timeout", 0)

    assert manager.effective.tools.command_timeout == 30.0


def test_set_preserves_existing_comments(tmp_path):
    """Saving a setting retains manually authored TOML comments."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    manager.initialize()
    manager.path.write_text(
        manager.path.read_text(encoding="utf-8").replace(
            'default_model = "nvidia/Qwen3.6-35B-A3B-NVFP4"',
            '# Project-specific model\ndefault_model = "nvidia/Qwen3.6-35B-A3B-NVFP4"',
        ),
        encoding="utf-8",
    )

    manager.set("backend.default_model", "configured-model", scope="user")

    assert "# Project-specific model" in manager.path.read_text(encoding="utf-8")
    assert manager.load().backend.default_model == "configured-model"


def test_independent_managers_merge_writes_under_the_interprocess_lock(tmp_path):
    """Two stale manager snapshots cannot lose each other's durable TOML updates."""
    path = tmp_path / "config.toml"
    first = ConfigurationManager(path)
    first.initialize()
    first.load()
    second = ConfigurationManager(path)
    second.load()

    first.set("loop.debug", True, scope="user")
    second.set("loop.stream", False, scope="user")

    settings = ConfigurationManager(path).load({})
    assert settings.loop.debug is True
    assert settings.loop.stream is False
    assert path.with_name(".config.toml.lock").stat().st_mode & 0o777 == 0o600


def test_manager_exposes_effective_values_sources_and_reload(tmp_path):
    """Loaded settings support redacted reads, provenance, and explicit reloads."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    manager.initialize()

    with pytest.raises(RuntimeError, match="not been loaded"):
        _ = manager.effective
    settings = manager.load({"DEFAULT_MODEL": "environment-model"})

    assert manager.effective is settings
    assert manager.get("backend.default_model") == "environment-model"
    assert str(manager.get("backend.api_key")) == "**********"
    assert manager.source_for("backend.default_model") == "environment"
    assert manager.source_for("backend.api_key") == "workspace"
    assert (
        manager.reload({"DEFAULT_MODEL": "reloaded-model"}).backend.default_model
        == "reloaded-model"
    )


def test_reload_reads_user_and_workspace_files_without_discarding_session_overrides(tmp_path):
    """Reload resolves external user and workspace edits beneath session precedence."""
    user_path = tmp_path / "config.toml"
    workspace_path = tmp_path / ".loop" / "config.toml"
    manager = ConfigurationManager(user_path, workspace_path)
    manager.initialize()
    manager.load()
    manager.set_session("loop.debug", True)
    user_path.write_text("[loop]\nstream = false\n", encoding="utf-8")
    workspace_path.parent.mkdir()
    workspace_path.write_text("[loop]\ntemperature = 0.4\n", encoding="utf-8")

    settings = manager.reload()

    assert settings.loop.debug is True
    assert settings.loop.stream is False
    assert settings.loop.temperature == 0.4
    assert manager.source_for("loop.debug") == "session"
    assert manager.source_for("loop.stream") == "user"
    assert manager.source_for("loop.temperature") == "workspace"


def test_reload_keeps_the_prior_snapshot_when_external_configuration_is_invalid(tmp_path):
    """A failed disk reload leaves the last valid documents and effective settings available."""
    manager = ConfigurationManager(tmp_path / "config.toml")
    manager.initialize()
    manager.load()
    manager.path.write_text('[loop]\ndebug = "invalid"\n', encoding="utf-8")

    with pytest.raises(ValidationError):
        manager.reload()

    assert manager.effective.loop.debug is False
    assert manager.source_for("loop.debug") == "workspace"


@pytest.mark.parametrize("path", ["backend", "backend.default_model.extra"])
def test_manager_rejects_malformed_configuration_paths(tmp_path, path):
    """Public configuration access requires exactly one section and field."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    manager.initialize()
    manager.load()

    with pytest.raises(ValueError, match="section.field"):
        manager.get(path)


def test_manager_rejects_unknown_fields_and_invalid_values(tmp_path):
    """Edits validate schema membership and values before persisting changes."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    manager.initialize()

    with pytest.raises(ValueError, match="Unknown configuration field"):
        manager.set("backend.unknown", "value")
    with pytest.raises(ValueError, match="Unknown configuration field"):
        manager.reset("backend.unknown")
    with pytest.raises(ValidationError):
        manager.set("backend.max_retries", -1)
    with pytest.raises(ValidationError):
        manager.set("loop.temperature", 3)


def test_invalid_workspace_edit_restores_the_previous_document(tmp_path):
    """A rejected sparse override does not contaminate later configuration operations."""
    workspace_path = tmp_path / "project" / ".loop" / "config.toml"
    manager = ConfigurationManager(tmp_path / "config.toml", workspace_path)
    manager.initialize()
    manager.load()

    with pytest.raises(ValidationError):
        manager.set("loop.temperature", 3, scope="workspace")

    assert manager.effective.loop.temperature is None
    assert not workspace_path.exists()


def test_reset_stores_default_and_missing_file_loads_defaults(tmp_path):
    """Reset stores a model default and a missing file remains usable."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    assert manager.load().backend.default_model == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    manager.initialize()
    manager.set("backend.default_model", "configured-model", scope="user")

    assert (
        manager.reset("backend.default_model", scope="user").backend.default_model
        == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    )


def test_session_reset_overrides_environment_without_writing_the_file(tmp_path):
    """A session reset stores the default without changing the configuration file."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
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
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    manager.initialize()
    manager.load({"DEFAULT_MODEL": "environment-model"})

    manager.set("backend.default_model", "file-model", scope="user")

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
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    manager.initialize()
    manager.load()
    manager.set("loop.debug", True, scope="user")
    manager.set("backend.api_key", "configured-secret", scope="user")
    manager.set_session("loop.stream", False)

    assert manager.reset_all(scope="session").loop.stream is True
    settings = manager.reset_all(scope="user")
    assert settings.loop.debug is False
    assert settings.backend.api_key.get_secret_value() == "local-api-key"
    assert 'api_key = "local-api-key"' in manager.path.read_text(encoding="utf-8")
    assert manager.source_for("loop.debug") == "session"
    assert manager.source_for("loop.stream") == "session"


def test_file_reset_all_retains_comments_and_formatting(tmp_path):
    """A file-wide reset restores values without replacing existing TOML layout."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    manager.initialize()
    manager.path.write_text(
        manager.path.read_text(encoding="utf-8").replace(
            'default_model = "nvidia/Qwen3.6-35B-A3B-NVFP4"',
            '# Project-specific model\ndefault_model = "configured-model"',
        ),
        encoding="utf-8",
    )
    manager.load()

    settings = manager.reset_all(scope="user")

    document = manager.path.read_text(encoding="utf-8")
    assert settings.backend.default_model == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert "# Project-specific model" in document
    assert 'default_model = "nvidia/Qwen3.6-35B-A3B-NVFP4"' in document


def test_file_reset_all_restores_missing_sections_and_removes_nullable_defaults(tmp_path):
    """A file-wide reset recreates absent sections and removes values defaulting to null."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    manager.path.parent.mkdir()
    manager.path.write_text("[backend]\ncontext_window = 4096\n", encoding="utf-8")
    manager.load()

    settings = manager.reset_all(scope="user")

    document = manager.path.read_text(encoding="utf-8")
    assert settings.backend.context_window is None
    assert "context_window" not in document
    assert "[telemetry]" in document


def test_unset_user_value_restores_the_stored_default(tmp_path):
    """Unsetting a user value stores its built-in default beneath environment precedence."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    manager.initialize()
    manager.load({"DEFAULT_MODEL": "environment-model"})
    manager.set("backend.default_model", "file-model", scope="user")

    settings = manager.unset("backend.default_model", scope="user")

    assert settings.backend.default_model == "environment-model"
    assert manager.source_for("backend.default_model") == "environment"
    assert 'default_model = "nvidia/Qwen3.6-35B-A3B-NVFP4"' in manager.path.read_text(
        encoding="utf-8"
    )


def test_unset_session_removes_one_override_and_requires_an_existing_value(tmp_path):
    """Removing a session value exposes lower scopes and rejects a second removal."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
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
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    manager.initialize()
    manager.load()
    manager.set("loop.debug", True, scope="user")
    manager.set_session("loop.stream", False)

    assert manager.unset_all(scope="user").loop.debug is False
    assert manager.effective.loop.stream is False
    assert manager.source_for("loop.debug") == "workspace"
    assert manager.source_for("loop.stream") == "session"
    document = manager.path.read_text(encoding="utf-8")
    assert "config_version" in document
    assert "[loop]" in document
    assert manager.unset_all(scope="session").loop.stream is True
