"""Tests for the command-line entry point."""

import runpy
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from loop import ShutdownRequested, main
from loop.configuration import ApplicationSettings
from loop.telemetry import set_telemetry as set_process_telemetry
from loop.workspace import Workspace, WorkspaceStorage


@pytest.fixture(autouse=True)
def isolate_main_environment(monkeypatch):
    """Keep CLI configuration independent of process variables and local dotenv files."""
    monkeypatch.setattr(main, "load_dotenv", Mock())
    configuration = Mock()
    configuration.load.return_value = ApplicationSettings()
    monkeypatch.setattr(main, "ConfigurationManager", Mock(return_value=configuration))
    monkeypatch.setattr(main, "set_telemetry", Mock())
    workspace = Workspace(
        Path("/project"), Path("/project/workspace"), WorkspaceStorage(Path("/project/.loop"))
    )
    monkeypatch.setattr(main, "Workspace", Mock(discover=Mock(return_value=workspace)))


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt, ShutdownRequested])
def test_main_gracefully_handles_shutdown_requests(monkeypatch, interruption):
    """Interruptions stop the CLI with a friendly message and close the runtime."""
    interaction = Mock()
    runtime = Mock()
    runtime.run.side_effect = interruption
    factory = Mock(return_value=runtime)
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "ApplicationRuntime", Mock(create=factory))
    register_shutdown_signals = Mock()
    monkeypatch.setattr(main, "register_shutdown_signals", register_shutdown_signals)

    main.main()

    register_shutdown_signals.assert_called_once_with()
    factory.assert_called_once_with(
        main.Workspace.discover.return_value,
        ApplicationSettings(),
        main.ConfigurationManager.return_value,
        interaction,
    )
    runtime.stop.assert_called_once_with()
    runtime.close.assert_called_once_with()
    assert interaction.info.call_args_list == [
        call("Hello from loop!"),
        call("\nStopping loop. Goodbye!"),
    ]


def test_main_reports_unexpected_failures(monkeypatch):
    """Runtime failures are converted into a fatal problem and close the runtime."""
    interaction = Mock()
    error = RuntimeError("runtime private")
    runtime = Mock()
    runtime.run.side_effect = error
    logger = Mock()
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "ApplicationRuntime", Mock(create=Mock(return_value=runtime)))
    monkeypatch.setattr(main, "register_shutdown_signals", Mock())
    monkeypatch.setattr(main, "_LOGGER", logger)

    main.main()

    logger.log.assert_called_once()
    assert logger.log.call_args.args[0] == 50
    assert logger.log.call_args.kwargs["extra"]["exception.type"] == "builtins.RuntimeError"
    interaction.report.assert_called_once()
    problem = interaction.report.call_args.args[0]
    assert problem.code == "internal.unexpected"
    assert problem.severity == "fatal"
    runtime.close.assert_called_once_with()


def test_main_reports_runtime_failures_to_initialized_telemetry(monkeypatch):
    """Runtime failures reach the telemetry service established by the application runtime."""
    interaction = Mock()
    error = RuntimeError("runtime private")
    runtime = Mock()
    runtime.run.side_effect = error
    telemetry = Mock()
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "ApplicationRuntime", Mock(create=Mock(return_value=runtime)))
    monkeypatch.setattr(main, "register_shutdown_signals", Mock())
    set_process_telemetry(telemetry)

    try:
        main.main()
    finally:
        set_process_telemetry(None)

    telemetry.error.assert_called_once()
    assert telemetry.error.call_args.kwargs["exception"] is error
    assert telemetry.error.call_args.args == ("problem.reported",)


def test_main_handles_failure_before_runtime_creation(monkeypatch):
    """Interaction initialization failures clear any previous global telemetry service."""
    error = RuntimeError("terminal private")
    logger = Mock()
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(side_effect=error))
    monkeypatch.setattr(main, "_LOGGER", logger)

    main.main()

    logger.log.assert_called_once()
    main.set_telemetry.assert_called_once_with(None)


@pytest.mark.parametrize("during_interaction", [False, True])
def test_main_handles_shutdown_before_runtime_creation(monkeypatch, during_interaction):
    """Shutdown before runtime construction does not require interaction or telemetry."""
    if during_interaction:
        monkeypatch.setattr(main, "ConsoleInteraction", Mock(side_effect=ShutdownRequested))
    else:
        interaction = Mock()
        monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
        monkeypatch.setattr(main, "register_shutdown_signals", Mock(side_effect=ShutdownRequested))

    main.main()

    main.set_telemetry.assert_called_once_with(None)


def test_main_module_runs_entry_point(monkeypatch):
    """Executing the source module as a script invokes its entry point."""
    monkeypatch.setattr("loop.interaction.ConsoleInteraction", Mock(return_value=Mock()))
    monkeypatch.setattr("loop.runtime.ApplicationRuntime.create", Mock(return_value=Mock()))
    monkeypatch.setattr("loop.utils.find_project_root", Mock(return_value=Path.cwd()))
    monkeypatch.setattr("loop.utils.register_shutdown_signals", Mock())
    configuration = Mock()
    configuration.load.return_value = ApplicationSettings()
    monkeypatch.setattr("loop.configuration.ConfigurationManager", Mock(return_value=configuration))
    monkeypatch.setattr("loop.telemetry.set_telemetry", Mock())

    with pytest.warns(RuntimeWarning, match="'loop.main' found in sys.modules"):
        runpy.run_module("loop.main", run_name="__main__")
