"""Verify the application command-line composition root."""

import runpy
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from loop import ShutdownRequested, main
from loop.configuration import ApplicationSettings
from loop.workspace import Workspace, WorkspaceSwitchRequested


@pytest.fixture(autouse=True)
def isolate_main(monkeypatch):
    """Replace filesystem discovery and composition with isolated doubles."""
    monkeypatch.setattr(main, "load_dotenv", Mock())
    workspace = Workspace(Path("/project"), Path("/project"))
    initialized = Workspace(Path("/project"), Path("/project"), "id", "project", "directory", 1, 1)
    monkeypatch.setattr(main, "Workspace", Mock(discover=Mock(return_value=workspace)))
    paths = Mock()
    paths.user_configuration = Path("/config/config.toml")
    paths.workspace_catalog = Path("/data/workspaces.db")
    paths.for_workspace.return_value = Mock()
    monkeypatch.setattr(main, "ApplicationPaths", Mock(discover=Mock(return_value=paths)))
    configuration = Mock()
    configuration.load.return_value = ApplicationSettings()
    monkeypatch.setattr(main, "ConfigurationManager", Mock(return_value=configuration))
    repository = Mock()
    repository.initialize.return_value = initialized
    monkeypatch.setattr(main, "WorkspaceRepository", Mock(return_value=repository))
    monkeypatch.setattr(main, "ApplicationMigration", Mock())
    monkeypatch.setattr(main, "set_telemetry", Mock())


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt, ShutdownRequested])
def test_main_gracefully_handles_shutdown(monkeypatch, interruption):
    """Interruptions stop and close a constructed runtime with a friendly message."""
    interaction = Mock()
    runtime = Mock()
    runtime.run.side_effect = interruption
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "ApplicationRuntime", Mock(create=Mock(return_value=runtime)))
    signals = Mock()
    monkeypatch.setattr(main, "register_shutdown_signals", signals)

    main.main()

    signals.assert_called_once_with()
    runtime.stop.assert_called_once_with()
    runtime.close.assert_called_once_with()
    assert interaction.info.call_args_list == [
        call("Hello from loop!"),
        call("\nStopping loop. Goodbye!"),
    ]


def test_main_reports_runtime_failure_and_closes(monkeypatch):
    """Unexpected runtime failures become fatal problems before close."""
    interaction = Mock()
    runtime = Mock()
    runtime.run.side_effect = RuntimeError("private")
    logger = Mock()
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "ApplicationRuntime", Mock(create=Mock(return_value=runtime)))
    monkeypatch.setattr(main, "register_shutdown_signals", Mock())
    monkeypatch.setattr(main, "_LOGGER", logger)

    main.main()

    logger.log.assert_called_once()
    interaction.report.assert_called_once()
    assert interaction.report.call_args.args[0].severity == "fatal"
    runtime.close.assert_called_once_with()


def test_main_clears_telemetry_when_failure_prevents_runtime(monkeypatch):
    """Failure before runtime ownership clears any process telemetry facade."""
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(side_effect=RuntimeError("terminal")))
    monkeypatch.setattr(main, "_LOGGER", Mock())

    main.main()

    main.set_telemetry.assert_called_once_with(None)


@pytest.mark.parametrize("with_interaction", [False, True])
def test_main_handles_shutdown_before_runtime_ownership(monkeypatch, with_interaction):
    """Early shutdown needs neither a runtime nor an initialized interaction."""
    if with_interaction:
        interaction = Mock()
        monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
        monkeypatch.setattr(main, "register_shutdown_signals", Mock(side_effect=ShutdownRequested))
    else:
        monkeypatch.setattr(main, "ConsoleInteraction", Mock(side_effect=ShutdownRequested))

    main.main()

    main.set_telemetry.assert_called_once_with(None)
    if with_interaction:
        interaction.info.assert_called_once_with("\nStopping loop. Goodbye!")


def test_main_module_runs_entry_point(monkeypatch):
    """Executing the source module as a script invokes its entry point."""
    monkeypatch.setattr("loop.interaction.ConsoleInteraction", Mock(return_value=Mock()))
    monkeypatch.setattr("loop.application.ApplicationRuntime.create", Mock(return_value=Mock()))
    monkeypatch.setattr("loop.utils.find_project_root", Mock(return_value=Path.cwd()))
    monkeypatch.setattr("loop.utils.register_shutdown_signals", Mock())
    configuration = Mock()
    configuration.load.return_value = ApplicationSettings()
    monkeypatch.setattr("loop.configuration.ConfigurationManager", Mock(return_value=configuration))
    monkeypatch.setattr("loop.telemetry.set_telemetry", Mock())

    with pytest.warns(RuntimeWarning, match="'loop.main' found in sys.modules"):
        runpy.run_module("loop.main", run_name="__main__")


def test_main_closes_active_runtime_before_rebuilding_for_workspace_switch(monkeypatch):
    """A switch signal tears down global owners before composing the target runtime."""
    target = Workspace(Path("/target"), Path("/target"), "target", "target", "directory", 2, 2)
    first = Mock()
    first.run.side_effect = WorkspaceSwitchRequested(target)
    second = Mock()
    runtime_factory = Mock(side_effect=[first, second])
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=Mock()))
    monkeypatch.setattr(main, "ApplicationRuntime", Mock(create=runtime_factory))
    monkeypatch.setattr(main, "register_shutdown_signals", Mock())

    main.main()

    first.close.assert_called_once_with()
    assert runtime_factory.call_args_list[1].args[0] is target
    second.close.assert_called_once_with()


def test_main_restores_previous_runtime_when_switch_rebuild_fails(monkeypatch):
    """A failed target composition rebuilds and continues the previous workspace."""
    target = Workspace(Path("/target"), Path("/target"), "target", "target", "directory", 2, 2)
    first = Mock()
    first.run.side_effect = WorkspaceSwitchRequested(target)
    restored = Mock()
    runtime_factory = Mock(side_effect=[first, RuntimeError("target failed"), restored])
    interaction = Mock()
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "ApplicationRuntime", Mock(create=runtime_factory))
    monkeypatch.setattr(main, "register_shutdown_signals", Mock())

    main.main()

    assert runtime_factory.call_args_list[1].args[0] is target
    assert runtime_factory.call_args_list[2].args[0].id == "id"
    assert "restored workspace id" in interaction.warning.call_args.args[0]
    restored.close.assert_called_once_with()
