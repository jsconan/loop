"""Main entry point for the loop package."""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from loop import (
    ConsoleInteraction,
    Loop,
    OpenAIBackend,
    SessionManager,
    ShutdownRequested,
    SQLiteSessionStore,
    find_project_root,
    register_shutdown_signals,
)
from loop.constants import APP_DIRECTORY, SESSION_DATABASE_FILENAME

_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
_API_KEY = "local-api-key"


def main() -> None:
    """Run an interactive conversation with an LLM backend."""
    load_dotenv(find_dotenv(usecwd=True))
    register_shutdown_signals()
    interaction = ConsoleInteraction()

    try:
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
        session_manager = SessionManager(
            interaction=interaction,
            session_store=SQLiteSessionStore(
                project_root / APP_DIRECTORY / SESSION_DATABASE_FILENAME
            ),
        )
        loop = Loop(
            backend,
            interaction=interaction,
            working_directory=working_directory,
            session_manager=session_manager,
            stream=True,
        )
        loop.run()
    except EOFError, KeyboardInterrupt, ShutdownRequested:
        interaction.info("\nStopping loop. Goodbye!")
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-except
        interaction.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
