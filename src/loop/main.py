"""Run the interactive loop application composition root."""

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from . import constants
from .application import (
    ApplicationMigration,
    ApplicationPaths,
    ApplicationRuntime,
)
from .configuration import ConfigurationManager
from .errors import Problem, log_problem
from .interaction import ConsoleInteraction
from .permissions import PermissionManager, PermissionNamespaceError
from .telemetry import set_telemetry
from .utils import ShutdownRequested, has_running_event_loop, register_shutdown_signals
from .workspace import (
    Workspace,
    WorkspaceRepository,
    WorkspaceSwitchRequested,
)

_LOGGER = logging.getLogger(__name__)


def run_application() -> None:  # pylint: disable=too-many-branches
    """Run the interactive application or an explicit host policy-provisioning command.

    Raises:
        RuntimeError: If called while another event loop is running in this thread.
        SystemExit: If policy provisioning syntax, identity or storage validation fails.

    Returns:
        None: After the application exits or policy initialization succeeds.
    """
    if has_running_event_loop():
        raise RuntimeError(
            "CLI run_application() cannot run inside an event loop; "
            "await run_application_async() instead."
        )
    if len(sys.argv) > 1 and sys.argv[1] == "--initialize-policy":
        if len(sys.argv) != 4:
            raise SystemExit(
                f"Usage: {constants.APP_NAME} --initialize-policy <workspace|user> <identity>"
            )
        try:
            _initialize_policy_namespace(sys.argv[2], sys.argv[3])
        except (PermissionNamespaceError, ValueError, OSError) as error:
            raise SystemExit(f"Policy initialization failed: {error}") from error
        return
    asyncio.run(run_application_async())


def _initialize_policy_namespace(scope: str, identity: str) -> None:
    """Provision one durable policy namespace from an explicit host CLI invocation."""
    paths = ApplicationPaths.discover()
    repository = WorkspaceRepository(paths.workspace_catalog)
    workspace = repository.initialize(Workspace.discover(Path.cwd()))
    if workspace.id is None:
        raise ValueError("Workspace identity is unavailable.")
    workspace_paths = paths.for_workspace(workspace.id, workspace.root)
    manager = PermissionManager(
        workspace.root,
        configuration_path=workspace_paths.permissions,
        user_configuration_path=paths.user_permissions,
        protected_host_roots=(paths.configuration_root, paths.data_root, paths.state_root),
        workspace_id=workspace.id,
        namespace_catalog_path=paths.workspace_catalog,
        user_namespace_identity=str(paths.data_root),
    )
    try:
        manager.initialize_policy_namespace(scope, identity, authorized=True)
    finally:
        manager.close()


async def run_application_async() -> None:  # pylint: disable=too-many-branches
    """Compose, run, and release each workspace runtime before its event loop closes.

    Raises:
        RuntimeError: If conversation execution has not settled before runtime cleanup.
        TimeoutError: If telemetry cannot flush within its configured shutdown timeout.
        BaseException: The first owned-resource cleanup failure after every resource is tried.
        asyncio.CancelledError: After owned resource cleanup settles following cancellation.
    """
    interaction = None
    runtime = None
    try:
        load_dotenv(find_dotenv(usecwd=True))
        paths = ApplicationPaths.discover()
        repository = WorkspaceRepository(paths.workspace_catalog)
        workspace = repository.initialize(Workspace.discover(Path.cwd()))
        interaction = ConsoleInteraction()
        register_shutdown_signals()
        interaction.info(f"Hello from {constants.APP_NAME}!")
        while True:
            if runtime is None:
                runtime = _build_runtime(workspace, paths, repository, interaction)
            try:
                await runtime.run_async()
            except WorkspaceSwitchRequested as request:
                previous = workspace
                try:
                    await runtime.close_async()
                    runtime = None
                    workspace = request.workspace
                    try:
                        runtime = _build_runtime(workspace, paths, repository, interaction)
                    except Exception as error:  # noqa: BLE001  # pylint: disable=broad-except
                        workspace = previous
                        interaction.warning(
                            f"Workspace switch failed ({type(error).__name__}); restored "
                            f"workspace {previous.id}."
                        )
                        runtime = _build_runtime(workspace, paths, repository, interaction)
                finally:
                    request.complete()
                continue
            break
    except (EOFError, KeyboardInterrupt, ShutdownRequested):
        if runtime is not None:
            runtime.stop()
        if interaction is not None:
            interaction.info(f"\nStopping {constants.APP_NAME}. Goodbye!")
    except PermissionNamespaceError as error:
        problem = Problem.from_exception(
            error,
            code="permission.policy_recovery_required",
            title="Permission policy recovery required",
            severity="fatal",
            operation="startup",
        )
        log_problem(_LOGGER, problem, error)
        if interaction is not None:
            interaction.report(problem)
    except Exception as error:  # noqa: BLE001  # pylint: disable=broad-except
        problem = Problem(
            code="internal.unexpected",
            title="Unexpected application error",
            detail="The application encountered an unexpected error and must stop.",
            severity="fatal",
            operation="main",
        )
        log_problem(_LOGGER, problem, error)
        if interaction is not None:
            interaction.report(problem)
    finally:
        if runtime is not None:
            await runtime.close_async()
        else:
            set_telemetry(None)


def _build_runtime(
    workspace: Workspace,
    paths: ApplicationPaths,
    repository: WorkspaceRepository,
    interaction: ConsoleInteraction,
) -> ApplicationRuntime:
    """Build every workspace-owned service for one resolved identity."""
    workspace_paths = paths.for_workspace(workspace.id, workspace.root)
    configuration = ConfigurationManager(paths.user_configuration, workspace_paths.configuration)
    configuration.initialize()
    settings = configuration.load()
    ApplicationMigration(
        workspace,
        paths,
        workspace_paths,
        busy_timeout_ms=settings.telemetry.sqlite_busy_timeout_ms,
    ).run()
    return ApplicationRuntime.create(
        workspace,
        paths,
        workspace_paths,
        settings,
        configuration,
        repository,
        interaction,
        main_app=True,
    )


if __name__ == "__main__":
    run_application()
