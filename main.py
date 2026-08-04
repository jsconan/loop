"""Main entry point for the loop package."""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from loop import (
    ConsoleInteraction,
    Loop,
    OpenAIBackend,
    ShutdownRequested,
    SQLiteSessionStore,
    find_project_root,
    register_shutdown_signals,
)

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
        backend = OpenAIBackend(
            base_url=os.getenv("BASE_URL", _BASE_URL),
            default_model=os.getenv("DEFAULT_MODEL", _DEFAULT_MODEL),
            api_key=os.getenv("OPENAI_API_KEY", _API_KEY),
            context_window=int(context_window) if context_window is not None else None,
        )
        working_directory = Path.cwd()
        project_root = find_project_root(working_directory) or working_directory
        loop = Loop(
            backend,
            interaction=interaction,
            working_directory=working_directory,
            session_store=SQLiteSessionStore(project_root / ".loop" / "sessions.db"),
            stream=True,
        )
        loop.run()
    except EOFError, KeyboardInterrupt, ShutdownRequested:
        interaction.info("\nStopping loop. Goodbye!")
    except Exception as e:  # pylint: disable=broad-except
        interaction.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
