"""Run the interactive loop application composition root."""

import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .application import ApplicationPaths, ApplicationRuntime
from .configuration import ConfigurationManager
from .errors import Problem, log_problem
from .interaction import ConsoleInteraction
from .telemetry import set_telemetry
from .utils import ShutdownRequested, register_shutdown_signals
from .workspace import Workspace, WorkspaceMigration, WorkspaceRepository

_LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run an interactive conversation with an LLM backend."""
    interaction = None
    runtime = None
    try:
        load_dotenv(find_dotenv(usecwd=True))
        paths = ApplicationPaths.discover()
        repository = WorkspaceRepository(paths.workspace_catalog)
        workspace = repository.initialize(Workspace.discover(Path.cwd()))
        workspace_paths = paths.for_workspace(workspace.id, workspace.root)

        configuration = ConfigurationManager(
            paths.user_configuration,
            workspace_paths.configuration,
        )
        configuration.initialize()
        settings = configuration.load()
        WorkspaceMigration(
            workspace,
            workspace_paths.sessions,
            workspace_paths.permissions,
        ).run()
        interaction = ConsoleInteraction()
        register_shutdown_signals()
        interaction.info("Hello from loop!")
        runtime = ApplicationRuntime.create(
            workspace,
            paths,
            workspace_paths,
            settings,
            configuration,
            repository,
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
