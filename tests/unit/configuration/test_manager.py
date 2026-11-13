"""Tests for workspace configuration loading and persistence."""

import os
from pathlib import Path

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


def test_recovery_defaults_cannot_initialize_a_missing_file(tmp_path):
    """A defaults bypass cannot create a persistent configuration through initialization."""
    path = tmp_path / "missing.toml"
    manager = ConfigurationManager(path)
    manager.use_defaults()

    with pytest.raises(ValueError, match="Repair and reload"):
        manager.initialize()
    assert not path.exists()


def test_recovery_reload_retries_the_failed_environment(tmp_path):
    """Bypassing a bad environment does not silently discard it on the next reload."""
    manager = ConfigurationManager(tmp_path / "config.toml")
    with pytest.raises(ValidationError):
        manager.load({"LOOP_DEBUG": "invalid"})
    manager.use_defaults()

    with pytest.raises(ValidationError):
        manager.reload()
    assert manager.reload({}).app.debug is False


def test_recovery_rejected_session_edit_preserves_the_previous_override(tmp_path):
    """A rejected edit cannot remove a prior session value during later updates."""
    manager = ConfigurationManager(tmp_path / "config.toml")
    manager.use_defaults()
    manager.set_session("app.debug", True)
    with pytest.raises(ValidationError):
        manager.set_session("app.debug", "invalid")
    assert manager.effective.app.debug is True
    assert manager.set_session("app.stream", False).app.debug is True


def test_recovery_entered_before_publication_preserves_the_saved_document(tmp_path):
    """A trusted pre-save callback cannot publish after it enters defaults-only recovery."""
    path = tmp_path / "config.toml"
    path.write_text("# preserve me\n[app]\ndebug = false\n", encoding="utf-8")
    manager = ConfigurationManager(path)
    manager.load({})
    original = path.read_bytes()

    with pytest.raises(ValueError, match="Repair and reload"):
        manager.set("app.debug", True, scope="user", before_save=manager.use_defaults)
    assert path.read_bytes() == original
    assert manager.saved_yolo is None
    assert manager.effective.app.debug is False


@pytest.mark.parametrize("operation", ["set", "unset", "reset_user", "reset_workspace", "save"])
def test_recovery_defaults_protect_saved_configuration_until_reload(tmp_path, operation):
    """A defaults bypass leaves the invalid document untouched and refuses durable writes."""
    user = tmp_path / "user.toml"
    user.write_text('[app]\ndebug = "invalid"\n', encoding="utf-8")
    manager = ConfigurationManager(user, tmp_path / "workspace.toml")
    original = user.read_bytes()

    assert manager.use_defaults().app.yolo is False
    assert manager.saved_yolo is None
    operations = {
        "set": lambda: manager.set("app.debug", True, scope="user"),
        "unset": lambda: manager.unset("app.debug", scope="workspace"),
        "reset_user": lambda: manager.reset_all(scope="user"),
        "reset_workspace": lambda: manager.reset_all(scope="workspace"),
        "save": manager.save,
    }
    with pytest.raises(ValueError, match="Repair and reload"):
        operations[operation]()
    assert user.read_bytes() == original
    assert manager.set_session("app.debug", True).app.debug is True
    user.write_text("[app]\ndebug = false\n", encoding="utf-8")
    assert manager.reload({}).app.debug is True
    manager.set("app.stream", False, scope="user")


@pytest.mark.parametrize("source", ["workspace", "environment"])
def test_invalid_section_type_is_reported_before_merging_overrides(tmp_path, source):
    """A non-table setting section fails validation instead of crashing during merge."""
    user = tmp_path / "user.toml"
    workspace = tmp_path / "workspace.toml"
    user.write_text("app = 1\n", encoding="utf-8")
    if source == "workspace":
        workspace.write_text("[app]\ndebug = true\n", encoding="utf-8")
    environment = {"LOOP_DEBUG": "true"} if source == "environment" else {}

    with pytest.raises(ValueError, match="section 'app' must be a table"):
        ConfigurationManager(user, workspace).load(environment)


@pytest.mark.parametrize("scope", ["user", "workspace"])
def test_invalid_toml_reports_source_without_exposing_content(tmp_path, scope):
    """Syntax errors name the file and position without including its source text."""
    user = tmp_path / "user.toml"
    workspace = tmp_path / "workspace.toml"
    selected = user if scope == "user" else workspace
    selected.write_text('[backend]\napi_key = "synthetic-secret\n', encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        ConfigurationManager(user, workspace).load({})
    assert str(selected) in str(raised.value)
    assert "line" in str(raised.value)
    assert "synthetic-secret" not in str(raised.value)


def test_duplicate_toml_key_is_reported_without_disclosing_its_name(tmp_path):
    """Duplicate keys produce a safe named-file syntax error."""
    path = tmp_path / "config.toml"
    path.write_text("[app]\nsynthetic_secret = 1\nsynthetic_secret = 2\n", encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        ConfigurationManager(path).load({})
    assert str(path) in str(raised.value)
    assert "KeyAlreadyPresent" in str(raised.value)
    assert "synthetic_secret" not in str(raised.value)


def test_invalid_utf8_configuration_reports_its_file_and_offset(tmp_path):
    """Invalid encoded input identifies its source and byte offset without showing bytes."""
    path = tmp_path / "config.toml"
    path.write_bytes(b"[app]\n\xff")

    with pytest.raises(ValueError, match="byte 6") as raised:
        ConfigurationManager(path).load({})
    assert str(path) in str(raised.value)


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


@pytest.mark.parametrize("scope", ["user", "workspace"])
def test_unsupported_configuration_version_is_rejected(tmp_path, scope):
    """Neither settings layer silently ignores a version it cannot interpret."""
    user = tmp_path / "user.toml"
    workspace = tmp_path / "workspace.toml"
    target = user if scope == "user" else workspace
    target.write_text("config_version = 999\n", encoding="utf-8")
    manager = ConfigurationManager(user, workspace)
    with pytest.raises(ValueError, match="config_version"):
        manager.load({})


def test_defaults_recovery_initialize_preserves_an_existing_file(tmp_path):
    """Recovery can report an existing path without writing over its contents."""
    path = tmp_path / "config.toml"
    path.write_text("invalid = true\n", encoding="utf-8")
    manager = ConfigurationManager(path)
    manager.use_defaults()
    assert manager.initialize() == path
    assert path.read_text(encoding="utf-8") == "invalid = true\n"


@pytest.mark.parametrize("scope", ["user", "workspace"])
def test_invalid_configuration_is_not_hidden_by_environment(tmp_path, scope):
    """An environment override cannot conceal a broken setting in either file."""
    user = tmp_path / "user.toml"
    workspace = tmp_path / "workspace.toml"
    target = user if scope == "user" else workspace
    target.write_text('[app]\ndebug = "invalid"\n', encoding="utf-8")
    manager = ConfigurationManager(user, workspace)
    with pytest.raises(ValidationError, match="app.debug") as error:
        manager.load({"LOOP_DEBUG": "true"})
    assert str(target) in error.value.title


@pytest.mark.parametrize("scope", ["user", "workspace"])
def test_configuration_accepts_natural_boolean_strings(tmp_path, scope):
    """Recognized boolean strings retain their meaning in either settings file."""
    user, workspace = tmp_path / "user.toml", tmp_path / "workspace.toml"
    target = user if scope == "user" else workspace
    target.write_text('[app]\ndebug = "yes"\nstream = "no"\n', encoding="utf-8")
    settings = ConfigurationManager(user, workspace).load({})
    assert settings.app.debug is True
    assert settings.app.stream is False


def test_invalid_environment_is_not_hidden_by_session_override(tmp_path):
    """An existing session value cannot conceal a broken environment value on reload."""
    manager = ConfigurationManager(tmp_path / "config.toml")
    manager.load({})
    manager.set_session("app.debug", False)
    with pytest.raises(ValidationError, match="app.debug") as error:
        manager.reload({"LOOP_DEBUG": "invalid"})
    assert "LOOP_DEBUG" in error.value.title


@pytest.mark.parametrize(
    "content", ["config_version = 999\n", "app = 7\n", "[app]\nunknown_secret = 1\n"]
)
def test_invalid_workspace_diagnostic_identifies_source_without_unknown_keys(tmp_path, content):
    """Schema diagnostics identify the file while redacting unrecognized names."""
    workspace = tmp_path / "workspace.toml"
    workspace.write_text(content, encoding="utf-8")
    manager = ConfigurationManager(tmp_path / "user.toml", workspace)
    with pytest.raises(ValueError) as error:
        manager.load({})
    detail = error.value.title if isinstance(error.value, ValidationError) else str(error.value)
    assert str(workspace) in detail
    assert "unknown_secret" not in detail


def test_workspace_can_declare_the_supported_configuration_version(tmp_path):
    """A workspace version is validated without overriding the user document version."""
    workspace = tmp_path / "workspace.toml"
    workspace.write_text("config_version = 1\n[app]\ndebug = true\n", encoding="utf-8")
    manager = ConfigurationManager(tmp_path / "user.toml", workspace)
    assert manager.load({}).app.debug is True


@pytest.mark.parametrize(
    "field", ["request_timeout_seconds", "max_generation_seconds", "repetition_penalty"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_backend_nonfinite_fields_remain_invalid(field, value):
    """Every duration and repetition penalty rejects all non-finite values."""
    with pytest.raises(ValidationError):
        BackendSettings.model_validate({field: value})


@pytest.mark.parametrize("scope", ["user", "workspace"])
@pytest.mark.parametrize("content", ["[app\n", '[app]\ndebug = "invalid"\n'])
def test_recovery_reset_archives_only_the_invalid_file(tmp_path, scope, content):
    """An identified invalid file is archived and replaced by the appropriate defaults."""
    user = tmp_path / "user.toml"
    workspace = tmp_path / "workspace.toml"
    target = user if scope == "user" else workspace
    target.write_text(content, encoding="utf-8")
    manager = ConfigurationManager(user, workspace)
    with pytest.raises(ValueError) as error:
        manager.load({})
    candidate = manager.recovery_candidate(error.value)
    assert candidate is not None
    backup = manager.reset_invalid_file(candidate)
    assert backup.read_text(encoding="utf-8") == content
    assert manager.load({}).app.debug is False
    assert target.exists() is (scope == "user")


def test_recovery_reset_does_not_offer_to_archive_changed_validation_failure(tmp_path):
    """A different invalid revision is not represented as the diagnosed failure."""
    target = tmp_path / "config.toml"
    target.write_text('[app]\ndebug = "bad"\n', encoding="utf-8")
    manager = ConfigurationManager(target)
    with pytest.raises(ValueError) as error:
        manager.load({})
    target.write_text('[app]\nstream = "bad"\n', encoding="utf-8")
    assert manager.recovery_candidate(error.value) is None


def test_recovery_reset_retains_concurrent_edits(tmp_path):
    """A change after the choice is archived and restored rather than discarded."""
    target = tmp_path / "config.toml"
    target.write_text('[app]\ndebug = "bad"\n', encoding="utf-8")
    manager = ConfigurationManager(target)
    with pytest.raises(ValueError) as error:
        manager.load({})
    candidate = manager.recovery_candidate(error.value)
    assert candidate is not None
    target.write_text("[app]\ndebug = true\n", encoding="utf-8")
    with pytest.raises(OSError, match="preserved at"):
        manager.reset_invalid_file(candidate)
    assert target.read_text(encoding="utf-8") == "[app]\ndebug = true\n"


def test_recovery_reset_is_unavailable_for_environment_failures(tmp_path):
    """The application cannot reset an invalid environment variable by archiving a file."""
    manager = ConfigurationManager(tmp_path / "config.toml")
    with pytest.raises(ValidationError) as error:
        manager.load({"LOOP_DEBUG": "bad"})
    assert manager.recovery_candidate(error.value) is None


def test_startup_can_reset_invalid_configuration(tmp_path):
    """Startup archives a verified invalid file before using newly created defaults."""
    import asyncio
    from unittest.mock import AsyncMock, Mock

    from loop.interaction import ConsoleInteraction

    path = tmp_path / "config.toml"
    path.write_text('[app]\ndebug = "bad"\n', encoding="utf-8")
    manager = ConfigurationManager(path)
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt_async = AsyncMock(return_value="reset")
    assert asyncio.run(manager.load_interactively_async(interaction)).app.debug is False
    assert next(tmp_path.glob(".config.toml.backup-*/config.toml")).read_text() == (
        '[app]\ndebug = "bad"\n'
    )


@pytest.mark.parametrize("selection", ["continue", "retry"])
def test_startup_reports_failed_reset_before_next_action(tmp_path, monkeypatch, selection):
    """A failed reset requires a fresh decision rather than automatically bypassing the file."""
    import asyncio
    from unittest.mock import AsyncMock, Mock

    from loop.interaction import ConsoleInteraction

    path = tmp_path / "config.toml"
    path.write_text('[app]\ndebug = "bad"\n', encoding="utf-8")
    manager = ConfigurationManager(path)
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt_async = AsyncMock(side_effect=["reset", selection, "continue"])

    def fail_reset(_candidate):
        """Refuse the attempted reset without changing the file."""
        raise OSError(13, "denied")

    monkeypatch.setattr(manager, "reset_invalid_file", fail_reset)
    assert asyncio.run(manager.load_interactively_async(interaction)).app.debug is False
    assert interaction.prompt_async.call_count == (3 if selection == "retry" else 2)
    assert path.read_text() == '[app]\ndebug = "bad"\n'
    assert any(
        call.args[0].code == "configuration.reset_failed"
        for call in interaction.report.call_args_list
    )


def test_recovery_candidate_supports_invalid_section_and_rejects_unrelated_error(tmp_path):
    """Only attributable section failures identify a file for reset."""
    path = tmp_path / "config.toml"
    path.write_text("app = 3\n", encoding="utf-8")
    manager = ConfigurationManager(path)
    with pytest.raises(ValueError) as error:
        manager.load({})
    assert manager.recovery_candidate(error.value) is not None
    assert manager.recovery_candidate(ValueError("unrelated")) is None


@pytest.mark.parametrize("replacement", ["symlink", "missing", "mixed", "valid"])
def test_recovery_candidate_rejects_changed_or_unreadable_file(tmp_path, replacement):
    """A missing, aliased or differently invalid file cannot be reset as the earlier failure."""
    path = tmp_path / "config.toml"
    path.write_text('[app]\ndebug = "bad"\n', encoding="utf-8")
    manager = ConfigurationManager(path)
    with pytest.raises(ValueError) as error:
        manager.load({})
    if replacement == "missing":
        path.unlink()
    elif replacement == "symlink":
        path.unlink()
        other = tmp_path / "other.toml"
        other.write_text('[app]\ndebug = "bad"\n', encoding="utf-8")
        path.symlink_to(other)
    elif replacement == "mixed":
        path.write_text("[loop]\ndebug = true\n[app]\ndebug = false\n", encoding="utf-8")
    else:
        path.write_text("[app]\ndebug = true\n", encoding="utf-8")
    assert manager.recovery_candidate(error.value) is None


def test_recovery_reset_rejects_a_different_managers_candidate(tmp_path):
    """A recovery decision cannot be applied to another configuration destination."""
    path = tmp_path / "config.toml"
    path.write_text('[app]\ndebug = "bad"\n', encoding="utf-8")
    manager = ConfigurationManager(path)
    with pytest.raises(ValueError) as error:
        manager.load({})
    candidate = manager.recovery_candidate(error.value)
    assert candidate is not None
    with pytest.raises(ValueError, match="does not belong"):
        ConfigurationManager(tmp_path / "other.toml").reset_invalid_file(candidate)
    assert path.exists()


@pytest.mark.parametrize("during", ["restore", "publish", "write"])
def test_recovery_reset_preserves_archive_during_io_races(tmp_path, monkeypatch, during):
    """Competing files are not overwritten and reset failures retain the archived source."""
    path = tmp_path / "config.toml"
    original_content = '[app]\ndebug = "bad"\n'
    path.write_text(original_content, encoding="utf-8")
    manager = ConfigurationManager(path)
    with pytest.raises(ValueError) as error:
        manager.load({})
    candidate = manager.recovery_candidate(error.value)
    assert candidate is not None
    original_link = os.link
    original_open = os.open
    if during == "restore":
        path.write_text("changed", encoding="utf-8")

    def competing_link(source, destination, *args, **kwargs):
        """Publish another file at the instant Loop attempts an exclusive link."""
        path.write_text("concurrent", encoding="utf-8")
        return original_link(source, destination, *args, **kwargs)

    def refused_open(target, flags, *args, **kwargs):
        """Refuse creation of the reset replacement."""
        if str(target).endswith("defaults.toml"):
            raise PermissionError("refused")
        return original_open(target, flags, *args, **kwargs)

    if during in {"restore", "publish"}:
        monkeypatch.setattr(configuration_module.os, "link", competing_link)
    else:
        monkeypatch.setattr(configuration_module.os, "open", refused_open)
    if during == "publish":
        backup = manager.reset_invalid_file(candidate)
        assert path.read_text() == "concurrent"
    else:
        with pytest.raises(OSError, match="preserved at"):
            manager.reset_invalid_file(candidate)
        backup = next(tmp_path.glob(".config.toml.backup-*/config.toml"))
    assert backup.read_text() == ("changed" if during == "restore" else original_content)


@pytest.mark.parametrize("scope", ["user", "workspace"])
def test_boolean_configuration_version_requires_recovery(tmp_path, scope):
    """A boolean in a TOML version field is rejected at the source file."""
    user = tmp_path / "user.toml"
    workspace = tmp_path / "workspace.toml"
    target = user if scope == "user" else workspace
    target.write_text("config_version = true\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="config_version"):
        ConfigurationManager(user, workspace).load({})


@pytest.mark.parametrize("scope", ["user", "workspace"])
def test_inaccessible_configuration_requires_interactive_recovery(tmp_path, monkeypatch, scope):
    """An unreadable existing file is reported rather than bypassed during startup."""
    import asyncio
    from unittest.mock import AsyncMock, Mock

    from loop.interaction import ConsoleInteraction

    user = tmp_path / "user.toml"
    workspace = tmp_path / "workspace.toml"
    target = user if scope == "user" else workspace
    target.write_text("[app]\ndebug = true\n", encoding="utf-8")
    target = target.resolve()
    original = Path.read_bytes

    def deny_read(path):
        """Refuse the selected configuration read with its source path."""
        if path == target:
            raise PermissionError(13, "synthetic-secret", str(path))
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", deny_read)
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt_async = AsyncMock(return_value="continue")
    manager = ConfigurationManager(user, workspace)
    assert asyncio.run(manager.load_interactively_async(interaction)).app.debug is False
    problem = interaction.report.call_args.args[0]
    assert "Permission denied" in problem.detail
    assert str(target) in problem.detail
    assert "synthetic-secret" not in problem.detail
    interaction.prompt_async.assert_awaited_once()
    assert original(target) == b"[app]\ndebug = true\n"


def test_initialization_does_not_overwrite_inaccessible_user_configuration(tmp_path, monkeypatch):
    """An existence-check denial is propagated before any default file is published."""
    user = tmp_path / "user.toml"
    user.write_text("[app]\ndebug = true\n", encoding="utf-8")
    original = Path.lstat

    def deny_stat(path, *args, **kwargs):
        """Deny only the user configuration's metadata lookup."""
        if path == user:
            raise PermissionError(13, "denied", str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", deny_stat)
    with pytest.raises(PermissionError, match="denied"):
        ConfigurationManager(user).initialize()
    assert user.read_text(encoding="utf-8") == "[app]\ndebug = true\n"


@pytest.mark.parametrize("scope", ["user", "workspace"])
def test_broken_configuration_symlink_is_not_treated_as_absent(tmp_path, scope):
    """A dangling configuration link is reported and left intact."""
    user = tmp_path / "user.toml"
    workspace = tmp_path / "workspace.toml"
    target = user if scope == "user" else workspace
    target.symlink_to(tmp_path / "missing.toml")
    manager = ConfigurationManager(user, workspace)
    if scope == "user":
        manager.initialize()
    with pytest.raises(FileNotFoundError):
        manager.load({})
    assert target.is_symlink()


@pytest.mark.parametrize("scope", ["user", "workspace"])
def test_broken_configuration_directory_symlink_is_not_treated_as_absent(tmp_path, scope):
    """A dangling configuration ancestor cannot hide a missing policy source."""
    directory = tmp_path / "alias"
    directory.symlink_to(tmp_path / "missing-directory", target_is_directory=True)
    target = directory / "config.toml"
    user = target if scope == "user" else tmp_path / "user.toml"
    workspace = target if scope == "workspace" else None
    with pytest.raises(FileNotFoundError):
        ConfigurationManager(user, workspace).load({})
    assert directory.is_symlink()


def test_missing_workspace_config_under_valid_symlink_remains_optional(tmp_path):
    """A missing file under a valid directory alias still uses the defaults."""
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    manager = ConfigurationManager(tmp_path / "user.toml", alias / "missing.toml")
    assert manager.load({}).app.debug is False


def test_workspace_alias_changed_after_selection_is_reported(tmp_path):
    """A new alias target cannot be silently skipped after the canonical path was selected."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (second / "config.toml").write_text("[app]\ndebug = true\n", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(first, target_is_directory=True)
    manager = ConfigurationManager(tmp_path / "user.toml", alias / "config.toml")
    alias.unlink()
    alias.symlink_to(second, target_is_directory=True)
    with pytest.raises(OSError, match="changed during loading"):
        manager.load({})


@pytest.mark.parametrize("scope", ["user", "workspace"])
@pytest.mark.parametrize(
    ("invalid", "override", "message"),
    [
        (
            "repetition_min_pattern_size = 10\nrepetition_max_pattern_size = 5\n",
            "repetition_min_pattern_size = 2\n",
            "Minimum repetition pattern size cannot exceed the maximum",
        ),
        (
            "repetition_max_pattern_size = 5\nrepetition_min_count = 1\n",
            "repetition_min_count = 3\n",
            "Repetition minimum count must be at least two",
        ),
    ],
)
def test_invalid_source_repetition_pair_cannot_be_masked(
    tmp_path, scope, invalid, override, message
):
    """An explicit contradictory pair is rejected before higher-precedence values merge."""
    user = tmp_path / "user.toml"
    workspace = tmp_path / "workspace.toml"
    if scope == "user":
        user.write_text("[backend]\n" + invalid, encoding="utf-8")
        workspace.write_text("[backend]\n" + override, encoding="utf-8")
        manager = ConfigurationManager(user, workspace)
        environment = {}
    else:
        workspace.write_text("[backend]\n" + invalid, encoding="utf-8")
        manager = ConfigurationManager(user, workspace)
        environment = {
            "LOOP_REPETITION_MIN_PATTERN_SIZE": "2",
            "LOOP_REPETITION_MIN_COUNT": "3",
        }
    with pytest.raises(ValidationError, match=message) as error:
        manager.load(environment)
    assert str(user if scope == "user" else workspace) in error.value.title


def test_sparse_repetition_values_can_be_completed_by_higher_layer(tmp_path):
    """Absent source fields are not treated as defaults when checking source-local pairs."""
    user = tmp_path / "user.toml"
    workspace = tmp_path / "workspace.toml"
    user.write_text("[backend]\nrepetition_min_pattern_size = 20\n", encoding="utf-8")
    workspace.write_text("[backend]\nrepetition_max_pattern_size = 25\n", encoding="utf-8")
    result = ConfigurationManager(user, workspace).load({})
    assert result.backend.repetition_min_pattern_size == 20
    assert result.backend.repetition_max_pattern_size == 25


def test_environment_repetition_pair_cannot_be_hidden_by_session_value(tmp_path):
    """An invalid pair within environment variables remains visible on reload."""
    manager = ConfigurationManager(tmp_path / "user.toml")
    manager.load({})
    manager.set_session("backend.repetition_min_pattern_size", 2)
    with pytest.raises(ValidationError, match="Minimum repetition pattern size") as error:
        manager.reload(
            {"LOOP_REPETITION_MIN_PATTERN_SIZE": "10", "LOOP_REPETITION_MAX_PATTERN_SIZE": "5"}
        )
    assert "environment variables" in error.value.title
