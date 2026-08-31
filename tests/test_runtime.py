"""Tests for application runtime composition and shutdown."""

from pathlib import Path
from unittest.mock import ANY, Mock, call

import pytest

import loop.runtime as runtime_module
from loop.configuration import ApplicationSettings
from loop.runtime import ApplicationRuntime
from loop.workspace import Workspace, WorkspaceStorage


@pytest.fixture
def workspace() -> Workspace:
    """Return a workspace with distinct root and active directories."""
    return Workspace(
        root=Path("/project"),
        working_directory=Path("/project/workspace"),
        storage=WorkspaceStorage(Path("/project/.loop")),
    )


@pytest.fixture
def runtime_dependencies(monkeypatch):
    """Replace runtime collaborators with isolated composition-boundary doubles."""

    dependencies = {
        name: Mock(return_value=Mock())
        for name in (
            "OpenAIBackend",
            "SQLiteTelemetryAdapter",
            "Telemetry",
            "create_default_tool_registry",
            "SQLiteSessionStore",
            "SessionManager",
            "PermissionManager",
        )
    }
    dependencies["Loop"] = Mock()
    dependencies["Loop"].create_default.return_value = Mock()
    dependencies["configure_operational_logging"] = Mock()
    dependencies["set_telemetry"] = Mock()
    dependencies["telemetry_activity"] = Mock()
    for name, replacement in dependencies.items():
        monkeypatch.setattr(runtime_module, name, replacement)
    return dependencies


def test_create_composes_all_runtime_dependencies_from_one_settings_snapshot(
    runtime_dependencies, workspace
):
    """A settings snapshot configures every application-owned runtime component."""
    settings = ApplicationSettings()
    configuration = Mock()
    interaction = Mock()

    runtime = ApplicationRuntime.create(workspace, settings, configuration, interaction)

    runtime_dependencies["configure_operational_logging"].assert_called_once_with(
        Path("/project/.loop/loop.log"),
        level=settings.logging.level,
        max_bytes=settings.logging.max_bytes,
        backup_count=settings.logging.backup_count,
    )
    runtime_dependencies["OpenAIBackend"].assert_called_once_with(
        base_url=settings.backend.base_url,
        default_model=settings.backend.default_model,
        api_key=settings.backend.api_key.get_secret_value(),
        context_window=settings.backend.context_window,
        file_input_mode=settings.backend.file_input_mode,
        structured_output_mode=settings.backend.structured_output_mode,
        structured_output_max_retries=settings.backend.structured_output_max_retries,
        max_retries=settings.backend.max_retries,
        temperature=settings.backend.temperature,
        reasoning_effort=settings.backend.reasoning_effort,
        hyperparameter_policy=settings.backend.hyperparameter_policy,
    )
    runtime_dependencies["Telemetry"].assert_called_once_with(
        runtime_dependencies["SQLiteTelemetryAdapter"].return_value,
        queue_capacity=settings.telemetry.queue_capacity,
        batch_size=settings.telemetry.batch_size,
        flush_seconds=settings.telemetry.flush_seconds,
        workspace_root=Path("/project"),
    )
    runtime_dependencies["Loop"].create_default.assert_called_once_with(
        runtime_dependencies["OpenAIBackend"].return_value,
        interaction=interaction,
        tool_registry=runtime_dependencies["create_default_tool_registry"].return_value,
        working_directory=Path("/project/workspace"),
        permission_manager=runtime_dependencies["PermissionManager"].return_value,
        session_manager=runtime_dependencies["SessionManager"].return_value,
        agent_name=settings.loop.agent_name,
        model=settings.loop.model,
        on_model_select=ANY,
        stream=settings.loop.stream,
        debug=settings.loop.debug,
        compaction_threshold=settings.loop.compaction_threshold,
        prompt_on_recoverable_error=settings.loop.prompt_on_recoverable_error,
        max_agent_turns=settings.loop.max_agent_turns,
    )
    persist = runtime_dependencies["Loop"].create_default.call_args.kwargs["on_model_select"]
    persist("selected-model")
    configuration.set.assert_called_once_with("loop.model", "selected-model")
    runtime_dependencies["set_telemetry"].assert_called_once_with(
        runtime_dependencies["Telemetry"].return_value
    )
    runtime_dependencies["telemetry_activity"].assert_called_once_with(
        "application.started", severity="info", component="main"
    )

    runtime.run()
    runtime.stop()
    runtime.close()

    runtime_dependencies["Loop"].create_default.return_value.run.assert_called_once_with()
    runtime_dependencies["telemetry_activity"].assert_has_calls(
        [
            call("application.stopping", severity="info", reason="interrupted"),
            call("application.stopped", severity="info", component="main"),
        ]
    )
    runtime_dependencies["set_telemetry"].assert_called_with(None)
    runtime_dependencies["Telemetry"].return_value.close.assert_called_once_with(
        timeout=settings.telemetry.shutdown_timeout
    )


def test_create_does_not_persist_environment_selected_model(runtime_dependencies, workspace):
    """An environment-selected model does not receive a durable selection callback."""
    configuration = Mock()
    configuration.source_for.return_value = "environment"

    ApplicationRuntime.create(workspace, ApplicationSettings(), configuration, Mock())

    assert runtime_dependencies["Loop"].create_default.call_args.kwargs["on_model_select"] is None


def test_runtime_applies_backend_changes_by_replacing_the_shared_backend(runtime_dependencies):
    """Backend settings construct a replacement backend for all future loop work."""
    active_loop = Mock()
    active_loop.apply_runtime_settings.return_value = "saved; restart required"
    runtime = ApplicationRuntime(active_loop, Mock(), 2.0)
    settings = ApplicationSettings()

    assert runtime.apply_configuration("backend.temperature", settings) == "applied now"

    active_loop.replace_backend.assert_called_once_with(
        runtime_dependencies["OpenAIBackend"].return_value
    )
    assert (
        runtime.apply_configuration("telemetry.batch_size", settings) == "saved; restart required"
    )
    active_loop.apply_runtime_settings.assert_called_once_with("telemetry.batch_size", settings)


@pytest.mark.parametrize("failure_after_telemetry", [False, True])
def test_create_closes_telemetry_when_composition_fails(
    runtime_dependencies, workspace, failure_after_telemetry
):
    """A failed composition closes telemetry only when it was successfully created."""
    error = RuntimeError("composition failed")
    failing_dependency = "Loop" if failure_after_telemetry else "OpenAIBackend"
    if failing_dependency == "Loop":
        runtime_dependencies["Loop"].create_default.side_effect = error
    else:
        runtime_dependencies["OpenAIBackend"].side_effect = error

    with pytest.raises(RuntimeError, match="composition failed"):
        ApplicationRuntime.create(workspace, ApplicationSettings(), Mock(), Mock())

    telemetry = runtime_dependencies["Telemetry"].return_value
    if failure_after_telemetry:
        telemetry.close.assert_called_once_with(timeout=2.0)
    else:
        telemetry.close.assert_not_called()
