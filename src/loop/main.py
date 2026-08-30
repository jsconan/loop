"""Run the interactive loop command-line application."""

import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .backend import OpenAIBackend
from .configuration import ConfigurationManager
from .constants import (
    APP_DIRECTORY,
    OPERATIONAL_LOG_FILENAME,
    SESSION_DATABASE_FILENAME,
    TELEMETRY_DATABASE_FILENAME,
)
from .errors import Problem, log_problem
from .interaction import ConsoleInteraction
from .loop import Loop
from .session import SessionManager, SQLiteSessionStore
from .telemetry import (
    SQLiteTelemetryAdapter,
    Telemetry,
    configure_operational_logging,
    set_telemetry,
    telemetry_activity,
)
from .tooling import ToolRuntimeSettings
from .tools import create_default_tool_registry
from .utils import ShutdownRequested, find_project_root, register_shutdown_signals

_LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run an interactive conversation with an LLM backend."""
    interaction = None
    telemetry = None

    try:
        load_dotenv(find_dotenv(usecwd=True))
        working_directory = Path.cwd()
        project_root = find_project_root(working_directory) or working_directory
        configuration = ConfigurationManager(project_root)
        configuration.initialize()
        settings = configuration.load()
        configure_operational_logging(
            project_root / APP_DIRECTORY / OPERATIONAL_LOG_FILENAME,
            level=settings.logging.level,
            max_bytes=settings.logging.max_bytes,
            backup_count=settings.logging.backup_count,
        )
        interaction = ConsoleInteraction()
        register_shutdown_signals()
        interaction.info("Hello from loop!")
        backend = OpenAIBackend(
            base_url=settings.backend.base_url,
            default_model=settings.backend.default_model,
            api_key=settings.backend.api_key.get_secret_value(),
            context_window=settings.backend.context_window,
            file_input_mode=settings.backend.file_input_mode,
            structured_output_mode=settings.backend.structured_output_mode,
            structured_output_max_retries=settings.backend.structured_output_max_retries,
            max_retries=settings.backend.max_retries,
        )
        telemetry = Telemetry(
            SQLiteTelemetryAdapter(
                project_root / APP_DIRECTORY / TELEMETRY_DATABASE_FILENAME,
                busy_timeout_ms=settings.telemetry.sqlite_busy_timeout_ms,
            ),
            queue_capacity=settings.telemetry.queue_capacity,
            batch_size=settings.telemetry.batch_size,
            flush_seconds=settings.telemetry.flush_seconds,
        )
        set_telemetry(telemetry)
        telemetry_activity("application.started", severity="info", component="main")
        session_manager = SessionManager(
            interaction=interaction,
            session_store=SQLiteSessionStore(
                project_root / APP_DIRECTORY / SESSION_DATABASE_FILENAME
            ),
        )
        loop = Loop.create_default(
            backend,
            interaction=interaction,
            tool_registry=create_default_tool_registry(
                interaction=interaction,
                settings=ToolRuntimeSettings(user_agent=settings.web.user_agent),
            ),
            working_directory=working_directory,
            session_manager=session_manager,
            agent_name=settings.loop.agent_name,
            model=settings.loop.model,
            stream=settings.loop.stream,
            debug=settings.loop.debug,
            compaction_threshold=settings.loop.compaction_threshold,
            prompt_on_recoverable_error=settings.loop.prompt_on_recoverable_error,
            max_agent_turns=settings.loop.max_agent_turns,
        )
        loop.run()
    except (EOFError, KeyboardInterrupt, ShutdownRequested):
        telemetry_activity("application.stopping", severity="info", reason="interrupted")
        if interaction is not None:
            interaction.info("\nStopping loop. Goodbye!")
    except Exception as error:  # noqa: BLE001  # pylint: disable=broad-except
        problem = Problem(
            code="internal.unexpected",
            title="Unexpected application error",
            detail="Loop encountered an unexpected error and must stop.",
            severity="fatal",
            operation="main",
        )
        log_problem(_LOGGER, problem, error)
        if interaction is not None:
            interaction.report(problem)
    finally:
        if telemetry is not None:
            telemetry_activity("application.stopped", severity="info", component="main")
            set_telemetry(None)
            telemetry.close(timeout=settings.telemetry.shutdown_timeout)
        else:
            set_telemetry(None)


if __name__ == "__main__":
    main()
