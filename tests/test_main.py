"""Tests for the command-line entry point."""

import runpy
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from loop import SessionManager, ShutdownRequested, main
from loop.configuration import ApplicationSettings
from loop.tooling import ToolRuntimeSettings


@pytest.fixture(autouse=True)
def isolate_main_environment(monkeypatch):
    """Keep CLI configuration independent of process variables and local dotenv files."""
    monkeypatch.setattr(main, "load_dotenv", Mock())
    monkeypatch.setattr(main, "configure_operational_logging", Mock())
    monkeypatch.setattr(main, "SQLiteTelemetryAdapter", Mock(return_value=Mock()))
    monkeypatch.setattr(main, "Telemetry", Mock(return_value=Mock()))
    configuration = Mock()
    configuration.load.return_value = ApplicationSettings()
    monkeypatch.setattr(main, "ConfigurationManager", Mock(return_value=configuration))
    for variable in (
        "BASE_URL",
        "DEFAULT_MODEL",
        "OPENAI_API_KEY",
        "CONTEXT_WINDOW",
        "OPENAI_MAX_RETRIES",
    ):
        monkeypatch.delenv(variable, raising=False)


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt, ShutdownRequested])
def test_main_gracefully_handles_shutdown_requests(monkeypatch, interruption):
    """Interrupts stop the CLI with a friendly message and no exception."""
    interaction = Mock()
    loop = Mock()
    loop.run.side_effect = interruption
    loop_factory = Mock(return_value=loop)
    loop_factory.create_default.return_value = loop
    backend = Mock()
    backend_factory = Mock(return_value=backend)
    tool_registry = Mock()
    registry_factory = Mock(return_value=tool_registry)
    session_store = Mock()
    session_store_factory = Mock(return_value=session_store)
    session_manager = Mock()
    session_manager_factory = Mock(return_value=session_manager)
    register_shutdown_signals = Mock()
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "Loop", loop_factory)
    monkeypatch.setattr(main, "OpenAIBackend", backend_factory)
    monkeypatch.setattr(main, "create_default_tool_registry", registry_factory)
    monkeypatch.setattr(main, "SQLiteSessionStore", session_store_factory)
    monkeypatch.setattr(main, "SessionManager", session_manager_factory)
    monkeypatch.setattr(main, "find_project_root", Mock(return_value=Path("/project")))
    monkeypatch.setattr(main, "register_shutdown_signals", register_shutdown_signals)

    main.main()

    register_shutdown_signals.assert_called_once_with()
    registry_factory.assert_called_once_with(
        interaction=interaction,
        settings=ToolRuntimeSettings(user_agent=ApplicationSettings().web.user_agent),
    )
    backend_factory.assert_called_once_with(
        base_url="http://localhost:8000/v1",
        default_model="nvidia/Qwen3.6-35B-A3B-NVFP4",
        api_key="local-api-key",
        context_window=None,
        file_input_mode=None,
        structured_output_mode="auto",
        structured_output_max_retries=1,
        max_retries=2,
    )
    session_store_factory.assert_called_once_with(Path("/project/.loop/sessions.db"))
    session_manager_factory.assert_called_once_with(
        interaction=interaction,
        session_store=session_store,
    )
    loop_factory.create_default.assert_called_once_with(
        backend,
        interaction=interaction,
        tool_registry=tool_registry,
        session_manager=session_manager,
        agent_name="Loop",
        model=None,
        stream=True,
        debug=False,
        compaction_threshold=0.8,
        prompt_on_recoverable_error=True,
        max_agent_turns=25,
        working_directory=Path.cwd(),
    )
    assert interaction.info.call_args_list == [
        call("Hello from loop!"),
        call("\nStopping loop. Goodbye!"),
    ]


def test_main_routes_startup_output_through_the_loop_interaction(monkeypatch, tmp_path):
    """A successful run uses the working directory when no Git project root exists."""
    interaction = Mock()
    loop = Mock()
    loop_factory = Mock(return_value=loop)
    loop_factory.create_default.return_value = loop
    backend = Mock()
    backend_factory = Mock(return_value=backend)
    tool_registry = Mock()
    registry_factory = Mock(return_value=tool_registry)
    session_store = Mock()
    session_store_factory = Mock(return_value=session_store)
    session_manager = Mock(spec=SessionManager)
    session_manager_factory = Mock(return_value=session_manager)
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "Loop", loop_factory)
    monkeypatch.setattr(main, "OpenAIBackend", backend_factory)
    monkeypatch.setattr(main, "create_default_tool_registry", registry_factory)
    monkeypatch.setattr(main, "SQLiteSessionStore", session_store_factory)
    monkeypatch.setattr(main, "SessionManager", session_manager_factory)
    monkeypatch.setattr(main, "find_project_root", Mock(return_value=None))
    monkeypatch.setattr(main, "register_shutdown_signals", Mock())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("DEFAULT_MODEL", "configured-model")
    monkeypatch.setenv("OPENAI_API_KEY", "configured-key")
    monkeypatch.setenv("CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "5")

    main.main()

    registry_factory.assert_called_once_with(
        interaction=interaction,
        settings=ToolRuntimeSettings(user_agent=ApplicationSettings().web.user_agent),
    )
    backend_factory.assert_called_once_with(
        base_url="http://localhost:8000/v1",
        default_model="nvidia/Qwen3.6-35B-A3B-NVFP4",
        api_key="local-api-key",
        context_window=None,
        file_input_mode=None,
        structured_output_mode="auto",
        structured_output_max_retries=1,
        max_retries=2,
    )
    session_store_factory.assert_called_once_with(tmp_path / ".loop" / "sessions.db")
    session_manager_factory.assert_called_once_with(
        interaction=interaction,
        session_store=session_store,
    )
    loop_factory.create_default.assert_called_once_with(
        backend,
        interaction=interaction,
        tool_registry=tool_registry,
        session_manager=session_manager,
        agent_name="Loop",
        model=None,
        stream=True,
        debug=False,
        compaction_threshold=0.8,
        prompt_on_recoverable_error=True,
        max_agent_turns=25,
        working_directory=tmp_path,
    )
    interaction.info.assert_called_once_with("Hello from loop!")
    loop.run.assert_called_once_with()


def test_main_handles_shutdown_before_telemetry_initialization(monkeypatch):
    """Early shutdown remains friendly when no structured telemetry exists yet."""
    interaction = Mock()
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "register_shutdown_signals", Mock())
    monkeypatch.setattr(main, "OpenAIBackend", Mock(side_effect=ShutdownRequested))

    main.main()

    interaction.info.assert_has_calls([call("Hello from loop!"), call("\nStopping loop. Goodbye!")])


def test_main_reports_unexpected_failures(monkeypatch):
    """Unexpected startup or runtime failures are converted into fatal problems."""
    interaction = Mock()
    error = RuntimeError("backend failed")
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "register_shutdown_signals", Mock())
    monkeypatch.setattr(main, "OpenAIBackend", Mock(side_effect=error))
    logger = Mock()
    monkeypatch.setattr(main, "_LOGGER", logger)

    main.main()

    logger.log.assert_called_once()
    assert logger.log.call_args.args[0] == 50
    assert logger.log.call_args.kwargs["extra"]["exception.type"] == "builtins.RuntimeError"
    assert "exc_info" not in logger.log.call_args.kwargs
    interaction.report.assert_called_once()
    problem = interaction.report.call_args.args[0]
    assert problem.code == "internal.unexpected"
    assert problem.severity == "fatal"


def test_main_logs_interaction_initialization_failure_without_presenting(monkeypatch):
    """A failure before interaction construction still reaches independent low-level logging."""
    error = RuntimeError("terminal private")
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(side_effect=error))
    logger = Mock()
    monkeypatch.setattr(main, "_LOGGER", logger)

    main.main()

    logger.log.assert_called_once()
    assert logger.log.call_args.kwargs["extra"]["exception.type"] == "builtins.RuntimeError"
    assert "terminal private" not in str(logger.log.call_args)


def test_main_handles_shutdown_during_interaction_initialization(monkeypatch):
    """A shutdown before interaction construction exits without presentation or failure."""
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(side_effect=ShutdownRequested))

    main.main()


def test_main_reports_runtime_failures_to_initialized_telemetry(monkeypatch):
    """Failures after startup reach both independent logging and structured telemetry."""
    interaction = Mock()
    loop = Mock()
    error = RuntimeError("runtime private")
    loop.run.side_effect = error
    loop_factory = Mock()
    loop_factory.create_default.return_value = loop
    telemetry = Mock()
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "register_shutdown_signals", Mock())
    monkeypatch.setattr(main, "OpenAIBackend", Mock(return_value=Mock()))
    monkeypatch.setattr(main, "create_default_tool_registry", Mock(return_value=Mock()))
    monkeypatch.setattr(main, "SQLiteSessionStore", Mock(return_value=Mock()))
    monkeypatch.setattr(main, "SessionManager", Mock(return_value=Mock()))
    monkeypatch.setattr(main, "Loop", loop_factory)
    monkeypatch.setattr(main, "find_project_root", Mock(return_value=Path("/project")))
    monkeypatch.setattr(main, "Telemetry", Mock(return_value=telemetry))

    main.main()

    telemetry.error.assert_called_once()
    assert telemetry.error.call_args.kwargs["exception"] is error
    assert telemetry.error.call_args.args == ("problem.reported",)
    telemetry.close.assert_called_once_with(timeout=2.0)


def test_main_module_runs_entry_point(monkeypatch):
    """Executing the source module as a script invokes its entry point."""
    monkeypatch.setattr("loop.interaction.ConsoleInteraction", Mock(return_value=Mock()))
    monkeypatch.setattr("loop.backend.OpenAIBackend", Mock(return_value=Mock()))
    monkeypatch.setattr("loop.loop.Loop", Mock(return_value=Mock()))
    monkeypatch.setattr("loop.session.SessionManager", Mock(return_value=Mock()))
    monkeypatch.setattr("loop.session.SQLiteSessionStore", Mock(return_value=Mock()))
    monkeypatch.setattr("loop.tools.create_default_tool_registry", Mock(return_value=Mock()))
    monkeypatch.setattr("loop.utils.find_project_root", Mock(return_value=Path.cwd()))
    monkeypatch.setattr("loop.utils.register_shutdown_signals", Mock())
    monkeypatch.setattr("loop.telemetry.Telemetry", Mock(return_value=Mock()))
    monkeypatch.setattr("loop.telemetry.SQLiteTelemetryAdapter", Mock(return_value=Mock()))
    monkeypatch.setattr("loop.telemetry.configure_operational_logging", Mock())
    monkeypatch.setattr("loop.telemetry.set_telemetry", Mock())

    monkeypatch.setattr("loop.configuration.ConfigurationManager", Mock())
    runpy.run_module("loop.main", run_name="__main__")
