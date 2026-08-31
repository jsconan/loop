"""Run the interactive loop command-line application."""

import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .configuration import ConfigurationManager
from .errors import Problem, log_problem
from .interaction import ConsoleInteraction
from .runtime import ApplicationRuntime
from .telemetry import set_telemetry
from .utils import ShutdownRequested, register_shutdown_signals
from .workspace import Workspace

_LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run an interactive conversation with an LLM backend."""
    interaction = None
    runtime = None

    try:
        load_dotenv(find_dotenv(usecwd=True))
        workspace = Workspace.discover(Path.cwd())
        configuration = ConfigurationManager(workspace.storage.configuration)
        configuration.initialize()
        settings = configuration.load()
        interaction = ConsoleInteraction()
        register_shutdown_signals()
        interaction.info("Hello from loop!")
        runtime = ApplicationRuntime.create(
            workspace,
            settings,
            configuration,
            interaction,
        )
        runtime.run()
    except (EOFError, KeyboardInterrupt, ShutdownRequested):
        if runtime is not None:
            runtime.stop()
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
        if runtime is not None:
            runtime.close()
        else:
            set_telemetry(None)


if __name__ == "__main__":
    main()
