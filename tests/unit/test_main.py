"""Verify the application command-line composition root."""

import asyncio
import runpy
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call

import pytest

from loop import ShutdownRequested
from loop import main as application
from loop.configuration import ApplicationSettings
from loop.permissions import PermissionNamespaceError
from loop.workspace import Workspace, WorkspaceSwitchRequested


@pytest.fixture(autouse=True)
def isolate_main(monkeypatch):
    """Replace filesystem discovery and composition with isolated doubles."""
    monkeypatch.setattr(application, "load_dotenv", Mock())
    workspace = Workspace(Path("/project"), Path("/project"))
    initialized = Workspace(Path("/project"), Path("/project"), "id", "project", "directory", 1, 1)
    monkeypatch.setattr(application, "Workspace", Mock(discover=Mock(return_value=workspace)))
    paths = Mock()
    paths.user_configuration = Path("/config/config.toml")
    paths.workspace_catalog = Path("/data/workspaces.db")
    paths.for_workspace.return_value = Mock()
    monkeypatch.setattr(application, "ApplicationPaths", Mock(discover=Mock(return_value=paths)))
    configuration = Mock()
    configuration.load.return_value = ApplicationSettings()
    monkeypatch.setattr(application, "ConfigurationManager", Mock(return_value=configuration))
    repository = Mock()
    repository.initialize.return_value = initialized
    monkeypatch.setattr(application, "WorkspaceRepository", Mock(return_value=repository))
    monkeypatch.setattr(application, "ApplicationMigration", Mock())
    monkeypatch.setattr(application, "set_telemetry", Mock())


def test_run_application_async_closes_runtime(monkeypatch):
    """The async entry point runs and closes its composed runtime directly."""
    interaction = Mock()
    runtime = Mock()
    runtime.run_async = AsyncMock()
    runtime.close_async = AsyncMock()
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(application, "ApplicationRuntime", Mock(create=Mock(return_value=runtime)))
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())

    asyncio.run(application.run_application_async())

    interaction.info.assert_called_once_with("Hello from loop!")
    runtime.run_async.assert_awaited_once_with()
    runtime.close_async.assert_awaited_once_with()


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt, ShutdownRequested])
def test_run_application_gracefully_handles_shutdown(monkeypatch, interruption):
    """Interruptions stop and close a constructed runtime with a friendly message."""
    interaction = Mock()
    runtime = Mock()
    runtime.run_async = AsyncMock(side_effect=interruption)
    runtime.close_async = AsyncMock()
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(application, "ApplicationRuntime", Mock(create=Mock(return_value=runtime)))
    signals = Mock()
    monkeypatch.setattr(application, "register_shutdown_signals", signals)

    application.run_application()

    signals.assert_called_once_with()
    runtime.run.assert_not_called()
    runtime.stop.assert_called_once_with()
    runtime.close_async.assert_awaited_once_with()
    assert interaction.info.call_args_list == [
        call("Hello from loop!"),
        call("\nStopping loop. Goodbye!"),
    ]


def test_run_application_reports_runtime_failure_and_closes(monkeypatch):
    """Unexpected runtime failures become fatal problems before close."""
    interaction = Mock()
    runtime = Mock()
    runtime.run_async = AsyncMock(side_effect=RuntimeError("private"))
    runtime.close_async = AsyncMock()
    logger = Mock()
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(application, "ApplicationRuntime", Mock(create=Mock(return_value=runtime)))
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())
    monkeypatch.setattr(application, "_LOGGER", logger)

    application.run_application()

    logger.log.assert_called_once()
    interaction.report.assert_called_once()
    assert interaction.report.call_args.args[0].severity == "fatal"
    runtime.close_async.assert_awaited_once_with()


def test_policy_recovery_reports_repair_path_without_starting_model(monkeypatch):
    """CLI startup reports one actionable policy failure and ends before model work."""
    interaction = Mock()
    error = PermissionNamespaceError(Path("/policy"), "Restore or repair the policy.")
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    runtime = Mock(create=Mock(side_effect=error))
    monkeypatch.setattr(application, "ApplicationRuntime", runtime)
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())
    monkeypatch.setattr(application, "_LOGGER", Mock())

    application.run_application()

    problem = interaction.report.call_args.args[0]
    assert problem.code == "permission.policy_recovery_required"
    assert problem.severity == "fatal"
    assert "/policy" in problem.detail
    assert "Restore or repair" in problem.detail
    runtime.create.assert_called_once()


def test_policy_recovery_before_console_setup_still_closes_cleanly(monkeypatch):
    """An early policy failure needs no console to release process telemetry."""
    error = PermissionNamespaceError(Path("/policy"), "Restore the policy.")
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(side_effect=error))
    monkeypatch.setattr(application, "_LOGGER", Mock())

    application.run_application()

    application.set_telemetry.assert_called_once_with(None)


def test_run_application_clears_telemetry_when_failure_prevents_runtime(monkeypatch):
    """Failure before runtime ownership clears any process telemetry facade."""
    monkeypatch.setattr(
        application, "ConsoleInteraction", Mock(side_effect=RuntimeError("terminal"))
    )
    monkeypatch.setattr(application, "_LOGGER", Mock())

    application.run_application()

    application.set_telemetry.assert_called_once_with(None)


def test_run_application_rejects_an_active_event_loop():
    """The CLI root does not create a nested event loop or an unawaited coroutine."""

    async def scenario():
        """Attempt to start the synchronous CLI entry point from async code."""
        with pytest.raises(RuntimeError, match=r"await run_application_async\(\) instead"):
            application.run_application()

    asyncio.run(scenario())


def test_headless_policy_initialization_uses_explicit_host_cli(monkeypatch):
    """The direct CLI operation provisions a named namespace without a model or prompt."""
    manager = Mock()
    monkeypatch.setattr(application, "PermissionManager", Mock(return_value=manager))
    monkeypatch.setattr(application.sys, "argv", ["loop", "--initialize-policy", "workspace", "id"])

    application.run_application()

    manager.initialize_policy_namespace.assert_called_once_with("workspace", "id", authorized=True)
    manager.close.assert_called_once_with()


@pytest.mark.parametrize(
    "arguments", [["loop", "--initialize-policy"], ["loop", "--initialize-policy", "user"]]
)
def test_headless_policy_initialization_requires_scope_and_identity(monkeypatch, arguments):
    """Incomplete provisioning fails with a non-success CLI result."""
    monkeypatch.setattr(application.sys, "argv", arguments)
    with pytest.raises(SystemExit, match="Usage: loop --initialize-policy"):
        application.run_application()


def test_headless_policy_initialization_rejects_namespace_errors(monkeypatch):
    """A registry or identity failure remains a non-success, actionable CLI outcome."""
    manager = Mock()
    manager.initialize_policy_namespace.side_effect = PermissionNamespaceError(
        Path("/policy"), "Identity mismatch."
    )
    monkeypatch.setattr(application, "PermissionManager", Mock(return_value=manager))
    monkeypatch.setattr(application.sys, "argv", ["loop", "--initialize-policy", "user", "wrong"])

    with pytest.raises(SystemExit, match="Identity mismatch"):
        application.run_application()
    manager.close.assert_called_once_with()


def test_headless_policy_initialization_requires_workspace_identity(monkeypatch):
    """Provisioning cannot create policy without a registered workspace UUID."""
    repository = application.WorkspaceRepository.return_value
    repository.initialize.return_value.id = None
    with pytest.raises(ValueError, match="Workspace identity is unavailable"):
        application._initialize_policy_namespace("workspace", "id")


@pytest.mark.parametrize("with_interaction", [False, True])
def test_run_application_handles_shutdown_before_runtime_ownership(monkeypatch, with_interaction):
    """Early shutdown needs neither a runtime nor an initialized interaction."""
    if with_interaction:
        interaction = Mock()
        monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
        monkeypatch.setattr(
            application, "register_shutdown_signals", Mock(side_effect=ShutdownRequested)
        )
    else:
        monkeypatch.setattr(application, "ConsoleInteraction", Mock(side_effect=ShutdownRequested))

    application.run_application()

    application.set_telemetry.assert_called_once_with(None)
    if with_interaction:
        interaction.info.assert_called_once_with("\nStopping loop. Goodbye!")


def test_module_runs_application_entry_point(monkeypatch):
    """Executing the source module as a script invokes its entry point."""
    monkeypatch.setattr("loop.interaction.ConsoleInteraction", Mock(return_value=Mock()))
    runtime = Mock()
    runtime.run_async = AsyncMock()
    runtime.close_async = AsyncMock()
    monkeypatch.setattr("loop.application.ApplicationRuntime.create", Mock(return_value=runtime))
    monkeypatch.setattr("loop.utils.find_project_root", Mock(return_value=Path.cwd()))
    monkeypatch.setattr("loop.utils.register_shutdown_signals", Mock())
    configuration = Mock()
    configuration.load.return_value = ApplicationSettings()
    monkeypatch.setattr("loop.configuration.ConfigurationManager", Mock(return_value=configuration))
    monkeypatch.setattr("loop.telemetry.set_telemetry", Mock())

    with pytest.warns(RuntimeWarning, match="'loop.main' found in sys.modules"):
        runpy.run_module("loop.main", run_name="__main__")


def test_run_application_closes_runtime_before_workspace_switch(monkeypatch):
    """A switch signal tears down global owners before composing the target runtime."""
    target = Workspace(Path("/target"), Path("/target"), "target", "target", "directory", 2, 2)
    first = Mock()
    first.run_async = AsyncMock(side_effect=WorkspaceSwitchRequested(target))
    first.close_async = AsyncMock()
    second = Mock()
    second.close_async = AsyncMock()
    runtime_factory = Mock(side_effect=[first, second])
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=Mock()))
    monkeypatch.setattr(application, "ApplicationRuntime", Mock(create=runtime_factory))
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())

    application.run_application()

    first.close_async.assert_awaited_once_with()
    assert runtime_factory.call_args_list[1].args[0] is target
    second.close_async.assert_awaited_once_with()


def test_run_application_restores_workspace_when_switch_rebuild_fails(monkeypatch):
    """A failed target composition rebuilds and continues the previous workspace."""
    target = Workspace(Path("/target"), Path("/target"), "target", "target", "directory", 2, 2)
    first = Mock()
    first.run_async = AsyncMock(side_effect=WorkspaceSwitchRequested(target))
    first.close_async = AsyncMock()
    restored = Mock()
    restored.close_async = AsyncMock()
    runtime_factory = Mock(side_effect=[first, RuntimeError("target failed"), restored])
    interaction = Mock()
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(application, "ApplicationRuntime", Mock(create=runtime_factory))
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())

    application.run_application()

    assert runtime_factory.call_args_list[1].args[0] is target
    assert runtime_factory.call_args_list[2].args[0].id == "id"
    assert "restored workspace id" in interaction.warning.call_args.args[0]
    restored.close_async.assert_awaited_once_with()
