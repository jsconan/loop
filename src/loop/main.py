"""Run the interactive loop command-line application."""

import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from loop import (
    ConsoleInteraction,
    Loop,
    OpenAIBackend,
    Problem,
    SessionManager,
    ShutdownRequested,
    SQLiteSessionStore,
    create_default_tool_registry,
    find_project_root,
    log_problem,
    register_shutdown_signals,
)
from loop.constants import (
    APP_DIRECTORY,
    OPERATIONAL_LOG_FILENAME,
    SESSION_DATABASE_FILENAME,
    TELEMETRY_DATABASE_FILENAME,
)
from loop.telemetry import (
    SQLiteTelemetryAdapter,
    Telemetry,
    configure_operational_logging,
    set_telemetry,
    telemetry_activity,
)

_LOGGER = logging.getLogger(__name__)

_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
_API_KEY = "local-api-key"


def main() -> None:
    """Run an interactive conversation with an LLM backend."""
    configure_operational_logging(Path.cwd() / APP_DIRECTORY / OPERATIONAL_LOG_FILENAME)
    interaction = None
    telemetry = None

    try:
        interaction = ConsoleInteraction()
        load_dotenv(find_dotenv(usecwd=True))
        register_shutdown_signals()
        interaction.info("Hello from loop!")
        context_window = os.getenv("CONTEXT_WINDOW")
        max_retries = os.getenv("OPENAI_MAX_RETRIES")
        backend = OpenAIBackend(
            base_url=os.getenv("BASE_URL", _BASE_URL),
            default_model=os.getenv("DEFAULT_MODEL", _DEFAULT_MODEL),
            api_key=os.getenv("OPENAI_API_KEY", _API_KEY),
            context_window=int(context_window) if context_window else None,
            max_retries=int(max_retries) if max_retries else 2,
        )
        working_directory = Path.cwd()
        project_root = find_project_root(working_directory) or working_directory
        telemetry = Telemetry(
            SQLiteTelemetryAdapter(project_root / APP_DIRECTORY / TELEMETRY_DATABASE_FILENAME)
        )
        set_telemetry(telemetry)
        telemetry_activity("application.started", severity="info", component="main")
        session_manager = SessionManager(
            interaction=interaction,
            session_store=SQLiteSessionStore(
                project_root / APP_DIRECTORY / SESSION_DATABASE_FILENAME
            ),
        )
        loop = Loop(
            backend,
            interaction=interaction,
            tool_registry=create_default_tool_registry(interaction=interaction),
            working_directory=working_directory,
            session_manager=session_manager,
            stream=True,
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
            telemetry.close()
        set_telemetry(None)


if __name__ == "__main__":
    main()
