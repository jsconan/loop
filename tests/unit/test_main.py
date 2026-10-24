"""Verify the application command-line composition root."""

import asyncio
import runpy
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call

import pytest

from loop import ShutdownRequested
from loop import main as application
from loop.application import ApplicationRunOutcome
from loop.configuration import ApplicationSettings
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


def test_run_application_rebuilds_runtime_after_conversation_stop(monkeypatch):
    """Stopping one conversation rebuilds its runtime and keeps the application flow open."""
    interaction = Mock()
    first = Mock()
    first.run_async = AsyncMock(return_value=ApplicationRunOutcome.CONVERSATION_STOPPED)
    first.close_async = AsyncMock()
    second = Mock()
    second.run_async = AsyncMock(return_value=ApplicationRunOutcome.APPLICATION_EXITED)
    second.close_async = AsyncMock()
    create_runtime = Mock(side_effect=[first, second])
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(application, "ApplicationRuntime", Mock(create=create_runtime))
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())

    asyncio.run(application.run_application_async())

    assert create_runtime.call_count == 2
    first.close_async.assert_awaited_once_with()
    second.close_async.assert_awaited_once_with()


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
