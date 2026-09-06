"""Verify application runtime composition and shutdown."""

from unittest.mock import Mock, call

import pytest

import loop.application.runtime as runtime_module
from loop.application import ApplicationPaths
from loop.application.runtime import ApplicationRuntime
from loop.configuration import ApplicationSettings
from loop.workspace import Workspace


@pytest.fixture
def assembled(tmp_path):
    """Return initialized workspace, immutable paths, and settings."""
    project = tmp_path / "project"
    project.mkdir()
    workspace = Workspace(project, project, "workspace-id", "project", "directory", 1, 1)
    paths = ApplicationPaths(tmp_path / "config", tmp_path / "data", tmp_path / "state")
    workspace_paths = paths.for_workspace(workspace.id, project)
    settings = ApplicationSettings()
    return workspace, paths, workspace_paths, settings


@pytest.fixture
def dependencies(monkeypatch):
    """Replace runtime collaborators with composition-boundary doubles."""
    names = (
        "OpenAIBackend",
        "SQLiteTelemetryAdapter",
        "Telemetry",
        "create_default_tool_registry",
        "SQLiteSessionStore",
        "SessionManager",
        "PermissionManager",
    )
    result = {name: Mock(return_value=Mock()) for name in names}
    loop = Mock()
    active_loop = Mock()
    active_loop.command_manager = Mock()
    loop.create_default.return_value = active_loop
    result["Loop"] = loop
    for name in ("configure_operational_logging", "set_telemetry", "telemetry_activity"):
        result[name] = Mock()
    for name, replacement in result.items():
        monkeypatch.setattr(runtime_module, name, replacement)
    return result


def test_create_composes_runtime_from_bound_references(dependencies, assembled):
    """One settings snapshot and reference graph configures every runtime component."""
    workspace, paths, workspace_paths, settings = assembled
    configuration = Mock()
    repository = Mock()
    interaction = Mock()

    runtime = ApplicationRuntime.create(
        workspace, paths, workspace_paths, settings, configuration, repository, interaction
    )

    logging_path = dependencies["configure_operational_logging"].call_args.args[0]
    assert logging_path == paths.operational_log
    telemetry_path = dependencies["SQLiteTelemetryAdapter"].call_args.args[0]
    session_path = dependencies["SQLiteSessionStore"].call_args.args[0]
    assert telemetry_path == paths.telemetry
    assert session_path == workspace_paths.sessions
    loop_kwargs = dependencies["Loop"].create_default.call_args.kwargs
    tool_kwargs = dependencies["create_default_tool_registry"].call_args.kwargs
    assert tool_kwargs["settings"].user_agent == settings.web.user_agent
    assert loop_kwargs["working_directory"] is workspace.working_directory
    assert loop_kwargs["stream"] is settings.loop.stream
    assert loop_kwargs["compaction_threshold"] == settings.loop.compaction_threshold
    loop_kwargs["on_model_select"]("selected")
    configuration.set.assert_called_once_with("loop.model", "selected")
    dependencies["set_telemetry"].assert_called_once_with(dependencies["Telemetry"].return_value)

    runtime.run()
    runtime.stop()
    runtime.close()

    dependencies["Loop"].create_default.return_value.run.assert_called_once_with()
    dependencies["telemetry_activity"].assert_has_calls(
        [
            call("application.stopping", severity="info", reason="interrupted"),
            call("application.stopped", severity="info", component="main"),
        ]
    )
    dependencies["set_telemetry"].assert_called_with(None)


def test_create_requires_identity_and_skips_environment_model_callback(dependencies, assembled):
    """Runtime assembly requires identity and does not persist environment-owned models."""
    workspace, paths, workspace_paths, settings = assembled
    configuration = Mock()
    repository = Mock()
    configuration.source_for.return_value = "environment"

    ApplicationRuntime.create(
        workspace, paths, workspace_paths, settings, configuration, repository, Mock()
    )
    assert dependencies["Loop"].create_default.call_args.kwargs["on_model_select"] is None

    with pytest.raises(ValueError, match="initialized"):
        ApplicationRuntime.create(
            Workspace(workspace.root, workspace.working_directory),
            paths,
            workspace_paths,
            settings,
            configuration,
            repository,
            Mock(),
        )


def test_apply_configuration_routes_aggregate_domain_and_scalar_changes(dependencies):
    """Backend and model operations remain explicit while scalar references update directly."""
    active_loop = Mock()
    settings = ApplicationSettings.model_validate({"loop": {"debug": True, "model": "model"}})
    active_loop.apply_runtime_settings.side_effect = lambda path, _settings: (
        "saved; restart required" if path.startswith("telemetry.") else "applied now"
    )
    runtime = ApplicationRuntime(active_loop, Mock(), 2.0)

    assert runtime.apply_configuration("backend.temperature", settings) == "applied now"
    assert runtime.apply_configuration("loop.model", settings) == "applied now"
    assert runtime.apply_configuration("loop.debug", settings) == "applied now"
    assert (
        runtime.apply_configuration("telemetry.batch_size", settings) == "saved; restart required"
    )

    active_loop.replace_backend.assert_called_once_with(dependencies["OpenAIBackend"].return_value)
    assert active_loop.apply_runtime_settings.call_count == 3


@pytest.mark.parametrize("after_telemetry", [False, True])
def test_create_closes_only_successfully_created_telemetry(
    dependencies, assembled, after_telemetry
):
    """Failed composition closes telemetry only after telemetry construction succeeded."""
    workspace, paths, workspace_paths, settings = assembled
    error = RuntimeError("composition failed")
    if after_telemetry:
        dependencies["Loop"].create_default.side_effect = error
    else:
        dependencies["OpenAIBackend"].side_effect = error

    with pytest.raises(RuntimeError, match="composition failed"):
        ApplicationRuntime.create(
            workspace, paths, workspace_paths, settings, Mock(), Mock(), Mock()
        )

    if after_telemetry:
        dependencies["Telemetry"].return_value.close.assert_called_once_with(
            timeout=settings.telemetry.shutdown_timeout
        )
    else:
        dependencies["Telemetry"].return_value.close.assert_not_called()


def test_close_without_owned_log_handler_remains_safe():
    """Injected runtimes can close telemetry without owning process logging."""
    telemetry = Mock()
    runtime = ApplicationRuntime(Mock(), telemetry, 1.0)

    runtime.close()

    telemetry.close.assert_called_once_with(timeout=1.0)
