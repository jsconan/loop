"""Tests for workspace configuration loading and persistence."""

import os

import pytest
from pydantic import ValidationError

from loop.configuration import ApplicationSettings, ConfigurationManager
from loop.configuration import manager as configuration_module
from loop.configuration.models import BackendSettings
from loop.models import CommandCapability


@pytest.mark.parametrize(
    "values",
    [
        {"request_timeout_seconds": float("nan")},
        {"max_generation_seconds": float("inf")},
        {"repetition_penalty": float("inf")},
        {"repetition_min_pattern_size": 5, "repetition_max_pattern_size": 4},
        {
            "repetition_min_pattern_size": 0,
            "repetition_max_pattern_size": 1,
            "repetition_min_count": 0,
        },
    ],
)
def test_backend_settings_reject_nonfinite_and_incoherent_repetition_values(values):
    """Backend settings reject finite and threshold combinations Pydantic bounds cannot express."""
    with pytest.raises(ValidationError):
        BackendSettings.model_validate(values)


def test_manager_snapshots_its_configuration_path(tmp_path):
    """Configuration owns one immutable destination selected at construction."""
    path = tmp_path / "first.toml"
    manager = ConfigurationManager(path)
    manager.initialize()
    manager.load()
    manager.save()

    assert manager.path == path
    assert path.is_file()


def test_version_one_loop_sections_load_and_save_as_app_without_losing_overrides(tmp_path):
    """Saved user and workspace settings survive the application-section rename."""
    user_path = tmp_path / "config.toml"
    workspace_path = tmp_path / ".loop" / "config.toml"
    user_path.write_text(
        'config_version = 1\n[loop]\n# keep the model choice\nmodel = "saved"\nstream = false\n',
        encoding="utf-8",
    )
    workspace_path.parent.mkdir()
    workspace_path.write_text("[loop]\ndebug = true\n", encoding="utf-8")
    manager = ConfigurationManager(user_path, workspace_path)

    settings = manager.load({})

    assert settings.app.model == "saved"
    assert settings.app.stream is False
    assert settings.app.debug is True
    assert manager.source_for("app.model") == "user"
    assert manager.source_for("app.debug") == "workspace"
    assert "[loop]" in user_path.read_text(encoding="utf-8")
    manager.set("app.agent_name", "Renamed", scope="user")
    manager.set("app.debug", False, scope="workspace")
    assert "[loop]" not in user_path.read_text(encoding="utf-8")
    assert "[loop]" not in workspace_path.read_text(encoding="utf-8")
    assert "# keep the model choice" in user_path.read_text(encoding="utf-8")
    reloaded = ConfigurationManager(user_path, workspace_path).load({})
    assert reloaded.app.model == "saved"
    assert reloaded.app.stream is False
    assert reloaded.app.debug is False
    assert reloaded.app.agent_name == "Renamed"


@pytest.mark.parametrize("scope", ["user", "workspace"])
def test_legacy_and_current_app_sections_cannot_compete(tmp_path, scope):
    """A mixed version-one file fails closed instead of selecting one application section."""
    user_path = tmp_path / "config.toml"
    workspace_path = tmp_path / ".loop" / "config.toml"
    manager = ConfigurationManager(user_path, workspace_path)
    manager.initialize()
    selected = user_path if scope == "user" else workspace_path
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_text("[loop]\ndebug = true\n[app]\ndebug = false\n", encoding="utf-8")

    with pytest.raises(ValueError, match="both \\[loop\\] and \\[app\\]"):
        manager.load({})


def test_legacy_workspace_section_cannot_activate_yolo(tmp_path):
    """A renamed workspace section retains the user-only YOLO trust boundary."""
    user_path = tmp_path / "config.toml"
    workspace_path = tmp_path / ".loop" / "config.toml"
    manager = ConfigurationManager(user_path, workspace_path)
    manager.initialize()
    workspace_path.parent.mkdir()
    workspace_path.write_text("[loop]\nyolo = true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Workspace configuration cannot control app.yolo"):
        manager.load({})


def test_app_environment_does_not_activate_yolo_or_change_saved_defaults(tmp_path, monkeypatch):
    """An unrelated ambient APP value cannot activate YOLO or enter saved defaults."""
    monkeypatch.setenv("APP", '{"yolo":true}')
    monkeypatch.setenv("APP_YOLO", "true")
    monkeypatch.setenv("APP__YOLO", "true")
    manager = ConfigurationManager(tmp_path / "config.toml")
    manager.initialize()

    assert manager.load({}).app.yolo is False
    assert ApplicationSettings().app.yolo is False
    assert manager.load({"APP": '{"yolo":true}'}).app.yolo is False
    assert manager.source_for("app.yolo") == "user"
    assert manager.saved_yolo is False
    assert manager.load({"APP_YOLO": "true", "APP__YOLO": "true"}).app.yolo is False
    assert ConfigurationManager(manager.path).load({}).app.yolo is False
    monkeypatch.setenv("APP", "not JSON")
    assert ApplicationSettings().app.yolo is False
    assert manager.load().app.yolo is False


def test_yolo_defaults_off_and_session_and_user_disables_have_distinct_lifetimes(tmp_path):
    """Session choices override trusted user state without changing its restart value."""
    path = tmp_path / "config" / "config.toml"
    manager = ConfigurationManager(path)
    manager.initialize()
    manager.load({})
    assert manager.effective.app.yolo is False
    assert manager.source_for("app.yolo") == "user"

    manager.set_session("app.yolo", True)
    assert manager.effective.app.yolo is True
    assert manager.saved_yolo is False
    manager.set("app.yolo", True, scope="user")
    manager.set_session("app.yolo", False)
    assert manager.effective.app.yolo is False
    assert manager.source_for("app.yolo") == "session"
    assert manager.saved_yolo is True

    restarted = ConfigurationManager(path)
    restarted.load({})
    assert restarted.effective.app.yolo is True
    assert restarted.source_for("app.yolo") == "user"
    restarted.set_session("app.yolo", True)
    restarted.set("app.yolo", False, scope="user")
    assert restarted.effective.app.yolo is True
    assert restarted.saved_yolo is False
    restarted.set_session("app.yolo", False)
    assert restarted.effective.app.yolo is False
    assert ConfigurationManager(path).load({}).app.yolo is False


def test_yolo_rejects_workspace_control_and_ignores_unrelated_environment(tmp_path):
    """Only a trusted user or session value can activate YOLO."""
    path = tmp_path / "config" / "config.toml"
    workspace = tmp_path / "project" / ".loop" / "config.toml"
    manager = ConfigurationManager(path, workspace)
    manager.initialize()
    manager.load({})
    with pytest.raises(ValueError, match="session"):
        manager.set("app.yolo", True, scope="workspace")
    with pytest.raises(ValueError, match="Workspace"):
        manager.unset("app.yolo", scope="workspace")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    workspace.write_text("[app]\nyolo = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Workspace"):
        manager.reload()
    workspace.unlink()
    assert manager.load({"APP": '{"yolo":true}'}).app.yolo is False
    assert manager.source_for("app.yolo") == "user"
    assert manager.saved_yolo is False
    manager.set("app.yolo", True, scope="user")
    assert manager.load({"APP": '{"yolo":false}'}).app.yolo is True
    assert manager.saved_yolo is True
    manager.set("app.yolo", False, scope="user")
    assert manager.load({"LOOP__YOLO": "true"}).app.yolo is False
    assert manager.load({"APP": '{"yolo":1}'}).app.yolo is False


def test_named_environment_fields_precede_saved_values_and_session_overrides(tmp_path):
    """Named field variables override saved values and session remains highest."""
    manager = ConfigurationManager(tmp_path / "config.toml")
    manager.initialize()
    manager.load({})
    manager.set("app.model", "saved", scope="user")

    settings = manager.load(
        {"APP": '{"model":"json-model","debug":true}', "LOOP_MODEL": "named-model"}
    )

    assert settings.app.model == "named-model"
    assert settings.app.debug is False
    assert manager.source_for("app.model") == "environment"
    assert manager.source_for("app.debug") == "workspace"
    manager.set_session("app.model", "session-model")
    assert manager.effective.app.model == "session-model"
    assert manager.source_for("app.model") == "session"


def test_persistent_yolo_requires_stable_private_user_storage(tmp_path, monkeypatch):
    """Startup refuses aliased, writable, ACL-exposed, and changing activation sources."""
    path = tmp_path / "config" / "config.toml"
    manager = ConfigurationManager(path)
    manager.initialize()
    manager.load({})
    manager.set("app.yolo", True, scope="user")
    assert ConfigurationManager(path).load({}).app.yolo is True

    path.chmod(0o644)
    with pytest.raises(ValueError, match="owner-controlled"):
        ConfigurationManager(path).load({})
    path.chmod(0o600)
    alias = path.with_name("alias.toml")
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="aliased user config"):
        ConfigurationManager(alias).load({})
    with pytest.raises(ValueError, match="aliased user config"):
        ConfigurationManager(alias).set("app.yolo", True, scope="user")
    alias.unlink()

    monkeypatch.setattr(configuration_module, "has_extended_access_acl", lambda _fd: True)
    with pytest.raises(ValueError, match="extended access ACL"):
        ConfigurationManager(path).load({})
    monkeypatch.undo()

    original_open = configuration_module.open_selected_file

    def change_after_open(root, selected, identity):
        """Replace the selected path after obtaining its original bound descriptor."""
        descriptor = original_open(root, selected, identity)
        selected.write_bytes(selected.read_bytes() + b"\n")
        return descriptor

    monkeypatch.setattr(configuration_module, "open_selected_file", change_after_open)
    with pytest.raises(ValueError, match="changed"):
        ConfigurationManager(path).load({})
    monkeypatch.undo()

    monkeypatch.setattr(
        configuration_module, "open_selected_file", lambda *_: os.open("/absent", 0)
    )
    with pytest.raises(ValueError, match="source is untrusted"):
        ConfigurationManager(path).load({})


def test_yolo_rejects_ambiguous_boolean_values(tmp_path):
    """Only exact true and false values can activate trusted session or saved state."""
    path = tmp_path / "config" / "config.toml"
    manager = ConfigurationManager(path)
    manager.initialize()
    manager.load({})
    with pytest.raises(ValueError, match="requires true or false"):
        manager.set_session("app.yolo", "1")
    with pytest.raises(ValueError, match="requires true or false"):
        manager.set("app.yolo", 1, scope="user")
    assert manager.set_session("app.yolo", " TRUE ").app.yolo is True
    manager.set_session("app.yolo", False)

    path.write_text(path.read_text(encoding="utf-8").replace("yolo = false", 'yolo = "true"'))
    with pytest.raises(ValueError, match="TOML boolean"):
        ConfigurationManager(path).load({})


def test_yolo_save_requires_a_private_real_user_directory(tmp_path):
    """An explicit user activation cannot be written beneath an unsafe config parent."""
    path = tmp_path / "config" / "config.toml"
    manager = ConfigurationManager(path)
    manager.initialize()
    manager.load({})
    path.parent.chmod(0o755)
    with pytest.raises(ValueError, match="private real directory"):
        manager.set("app.yolo", True, scope="user")
    assert manager.effective.app.yolo is False


def test_yolo_user_source_must_be_outside_the_workspace(tmp_path):
    """A workspace dotenv cannot turn its own private directory into trusted user storage."""
    user_path = tmp_path / "config" / "config.toml"
    workspace_path = tmp_path / ".loop" / "config.toml"
    manager = ConfigurationManager(user_path, workspace_path)
    manager.initialize()
    manager.load({})
    with pytest.raises(ValueError, match="inside the workspace"):
        manager.set("app.yolo", True, scope="user")
    user_path.write_text(
        user_path.read_text(encoding="utf-8").replace("yolo = false", "yolo = true")
    )
    with pytest.raises(ValueError, match="inside the workspace"):
        ConfigurationManager(user_path, workspace_path).load({})


def test_conversation_input_mode_defaults_validates_and_loads_from_environment(tmp_path):
    """The active input route defaults to queue and accepts only the documented modes."""
    assert ApplicationSettings().conversation.input_mode == "queue"
    with pytest.raises(ValidationError):
        ApplicationSettings.model_validate({"conversation": {"input_mode": "interrupt"}})

    path = tmp_path / "config.toml"
    manager = ConfigurationManager(path)
    manager.initialize()
    settings = manager.load({"LOOP_CONVERSATION_INPUT_MODE": "steer"})

    assert settings.conversation.input_mode == "steer"
    assert manager.source_for("conversation.input_mode") == "environment"
    assert '[conversation]\ninput_mode = "queue"' in path.read_text(encoding="utf-8")
    entry = next(item for item in manager.entries if item.path == "conversation.input_mode")
    assert entry.default == "queue"
    assert entry.choices == ("queue", "steer")


def test_user_defaults_merge_with_sparse_workspace_overrides(tmp_path):
    """A complete user document remains editable while the workspace stores only overrides."""
    user_path = tmp_path / "config" / "config.toml"
    workspace_path = tmp_path / "project" / ".loop" / "config.toml"
    manager = ConfigurationManager(user_path, workspace_path)
    manager.initialize()
    manager.load()

    manager.set("app.debug", True)
    manager.set("app.stream", False, scope="user")

    assert "stream = false" in user_path.read_text(encoding="utf-8")
    assert workspace_path.read_text(encoding="utf-8") == "[app]\ndebug = true\n"
    assert manager.effective.app.debug is True
    assert manager.effective.app.stream is False
    assert manager.source_for("app.debug") == "workspace"
    assert manager.source_for("app.stream") == "user"

    manager.set("app.debug", True, scope="user")
    manager.set("app.stream", True, scope="workspace")
    manager.reset("app.debug", scope="workspace")

    assert manager.effective.app.debug is True
    assert manager.effective.app.stream is True
    assert manager.source_for("app.debug") == "user"
    workspace_document = workspace_path.read_text(encoding="utf-8")
    assert "debug" not in workspace_document
    assert "stream = true" in workspace_document

    manager.set("app.debug", True, scope="workspace")
    assert manager.unset_all(scope="workspace").app.debug is True
    assert not workspace_path.exists()


def test_sparse_workspace_reset_and_validation_cover_layer_edge_cases(tmp_path):
    """Workspace-wide reset clears overrides and malformed sparse operations fail safely."""
    user_path = tmp_path / "config.toml"
    workspace_path = tmp_path / ".loop" / "config.toml"
    manager = ConfigurationManager(user_path, workspace_path)
    manager.initialize()
    manager.load()
    manager.set("app.debug", True, scope="workspace")
    manager.set("app.stream", False, scope="workspace")
    manager.unset("app.debug", scope="workspace")
    assert manager.effective.app.stream is False

    assert manager.reset_all(scope="workspace").app.debug is False
    assert manager.reset_all(scope="user").app.debug is False
    with pytest.raises(ValueError, match="Unknown configuration field"):
        manager.unset("app.debug", scope="workspace")

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
        manager.set("app.debug", True)


def test_user_reset_and_unset_restore_the_user_scope(tmp_path):
    """User-scope edits persist in the complete document and do not create workspace overrides."""
    user_path = tmp_path / "config.toml"
    workspace_path = tmp_path / ".loop" / "config.toml"
    manager = ConfigurationManager(user_path, workspace_path)
    manager.initialize()
    manager.load()
    manager.set("app.debug", True, scope="user")

    manager.reset("app.debug", scope="user")

    assert manager.effective.app.debug is False
    assert manager.source_for("app.debug") == "user"
    assert not workspace_path.exists()
    assert "debug = false" in user_path.read_text(encoding="utf-8")

    manager.set("app.debug", True, scope="user")
    manager.unset("app.debug", scope="user")

    assert manager.effective.app.debug is False
    assert "debug = false" in user_path.read_text(encoding="utf-8")


def test_invalid_session_value_is_rolled_back(tmp_path):
    """Rejecting an invalid session override leaves the prior effective settings unchanged."""
    manager = ConfigurationManager(tmp_path / "config.toml")
    manager.initialize()
    manager.load()

    with pytest.raises(ValidationError):
        manager.set_session("app.temperature", 3)

    assert manager.effective.app.temperature is None
    with pytest.raises(ValueError, match="No session override"):
        manager.unset_session("app.temperature")


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
    assert "# Workspace configuration" in content
    assert 'api_key = "local-api-key"' in content
    assert "# context_window = <unset>" in content
    assert "[tools]" in content
    assert "command_timeout = 3600.0" in content
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
    manager.set("app.temperature", 0.2, scope="user")
    manager.set("app.reasoning_effort", "high", scope="user")
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
    assert settings.app.temperature == 0.7
    assert settings.app.reasoning_effort == "low"
    assert settings.backend.hyperparameter_policy == "fallback"


@pytest.mark.parametrize("value", [0, float("inf"), float("-inf"), float("nan")])
def test_command_timeout_requires_a_positive_duration(tmp_path, value):
    """Configuration rejects non-positive and non-finite command lifecycle timeouts."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    manager.initialize()
    manager.load()

    with pytest.raises(ValidationError):
        manager.set_session("tools.command_timeout", value)

    assert manager.effective.tools.command_timeout == 3600.0


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

    first.set("app.debug", True, scope="user")
    second.set("app.stream", False, scope="user")

    settings = ConfigurationManager(path).load({})
    assert settings.app.debug is True
    assert settings.app.stream is False
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
    manager.set_session("app.debug", True)
    user_path.write_text("[app]\nstream = false\n", encoding="utf-8")
    workspace_path.parent.mkdir()
    workspace_path.write_text("[app]\ntemperature = 0.4\n", encoding="utf-8")

    settings = manager.reload()

    assert settings.app.debug is True
    assert settings.app.stream is False
    assert settings.app.temperature == 0.4
    assert manager.source_for("app.debug") == "session"
    assert manager.source_for("app.stream") == "user"
    assert manager.source_for("app.temperature") == "workspace"


def test_reload_keeps_the_prior_snapshot_when_external_configuration_is_invalid(tmp_path):
    """A failed disk reload leaves the last valid documents and effective settings available."""
    manager = ConfigurationManager(tmp_path / "config.toml")
    manager.initialize()
    manager.load()
    manager.path.write_text('[app]\ndebug = "invalid"\n', encoding="utf-8")

    with pytest.raises(ValidationError):
        manager.reload()

    assert manager.effective.app.debug is False
    assert manager.source_for("app.debug") == "workspace"


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
        manager.set("app.temperature", 3)


def test_invalid_workspace_edit_restores_the_previous_document(tmp_path):
    """A rejected sparse override does not contaminate later configuration operations."""
    workspace_path = tmp_path / "project" / ".loop" / "config.toml"
    manager = ConfigurationManager(tmp_path / "config.toml", workspace_path)
    manager.initialize()
    manager.load()

    with pytest.raises(ValidationError):
        manager.set("app.temperature", 3, scope="workspace")

    assert manager.effective.app.temperature is None
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
    assert next(item for item in manager.entries if item.path == "app.stream").choices == (
        True,
        False,
    )
    assert next(
        item for item in manager.entries if item.path == "backend.file_input_mode"
    ).choices == (
        "text",
        "native",
    )
    live_update = next(item for item in manager.entries if item.path == "app.max_agent_turns")
    paused_update = next(item for item in manager.entries if item.path == "backend.base_url")
    assert live_update.required_capabilities == frozenset({CommandCapability.CONTEXT_UPDATE})
    assert paused_update.required_capabilities == frozenset({CommandCapability.SESSION_IDLE})


def test_reset_all_stores_defaults_in_each_scope(tmp_path):
    """File and session resets each retain explicitly stored built-in defaults."""
    manager = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    manager.initialize()
    manager.load()
    manager.set("app.debug", True, scope="user")
    manager.set("backend.api_key", "configured-secret", scope="user")
    manager.set_session("app.stream", False)

    assert manager.reset_all(scope="session").app.stream is True
    settings = manager.reset_all(scope="user")
    assert settings.app.debug is False
    assert settings.backend.api_key.get_secret_value() == "local-api-key"
    assert 'api_key = "local-api-key"' in manager.path.read_text(encoding="utf-8")
    assert manager.source_for("app.debug") == "session"
    assert manager.source_for("app.stream") == "session"


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
    manager.set("app.debug", True, scope="user")
    manager.set_session("app.stream", False)

    assert manager.unset_all(scope="user").app.debug is False
    assert manager.effective.app.stream is False
    assert manager.source_for("app.debug") == "workspace"
    assert manager.source_for("app.stream") == "session"
    document = manager.path.read_text(encoding="utf-8")
    assert "config_version" in document
    assert "[app]" in document
    assert manager.unset_all(scope="session").app.stream is True
