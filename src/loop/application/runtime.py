"""Compose and manage one configured application runtime."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Self

from ..backend import OpenAIBackend
from ..configuration import ApplicationSettings, ConfigurationCommands, ConfigurationManager
from ..instructions import InstructionsManager
from ..interaction import Interaction
from ..loop import Loop
from ..permissions import PermissionManager
from ..session import SessionManager, SQLiteSessionStore
from ..telemetry import (
    SQLiteTelemetryAdapter,
    Telemetry,
    configure_operational_logging,
    get_telemetry,
    set_telemetry,
    telemetry_activity,
)
from ..tooling import ToolRuntimeSettings
from ..tools import create_default_tool_registry
from ..workspace import Workspace, WorkspaceCommands, WorkspaceRepository
from .commands import ApplicationCommands
from .paths import ApplicationPaths, WorkspacePaths


class ApplicationRuntime:
    """Manage the assembled application and process-wide telemetry lifecycle.

    Args:
        loop (Loop): Fully assembled interactive application.
        telemetry (Telemetry): Application telemetry service.
        shutdown_timeout (float): Maximum seconds allowed for telemetry shutdown.
        logging_handler (logging.Handler | None): Owned process-global log handler.
        cleanup_callbacks (list[Callable[[], object]] | None): Acquired-resource cleanup actions in
            acquisition order. Defaults to telemetry and optional logging cleanup.
    """

    _loop: Loop
    _telemetry: Telemetry
    _shutdown_timeout: float
    _logging_handler: logging.Handler | None
    _cleanup_callbacks: list[Callable[[], object]]
    _closed: bool

    def __init__(
        self,
        loop: Loop,
        telemetry: Telemetry,
        shutdown_timeout: float,
        *,
        logging_handler: logging.Handler | None = None,
        cleanup_callbacks: list[Callable[[], object]] | None = None,
    ) -> None:
        self._loop = loop
        self._telemetry = telemetry
        self._shutdown_timeout = shutdown_timeout
        self._logging_handler = logging_handler
        if cleanup_callbacks is None:
            self._cleanup_callbacks = [lambda: self._close_telemetry(telemetry, shutdown_timeout)]
            if logging_handler is not None:
                self._cleanup_callbacks.insert(
                    0, lambda: self._close_logging_handler(logging_handler)
                )
        else:
            self._cleanup_callbacks = cleanup_callbacks
        self._closed = False

    @classmethod
    def create(
        cls,
        workspace: Workspace,
        paths: ApplicationPaths,
        workspace_paths: WorkspacePaths,
        settings: ApplicationSettings,
        configuration: ConfigurationManager,
        workspace_repository: WorkspaceRepository,
        interaction: Interaction,
    ) -> ApplicationRuntime:
        """Build a runtime from initialized workspace and application-owned references.

        Args:
            workspace (Workspace): Initialized active workspace.
            paths (ApplicationPaths): Immutable global application paths.
            workspace_paths (WorkspacePaths): Immutable active-workspace storage paths.
            settings (ApplicationSettings): Validated immutable application configuration.
            configuration (ConfigurationManager): Persistent configuration owner.
            workspace_repository (WorkspaceRepository): Owner of workspace registry operations.
            interaction (Interaction): User interaction service.

        Returns:
            ApplicationRuntime: Fully composed active runtime.

        Raises:
            ValueError: If the workspace has not been initialized.
        """
        if workspace.id is None:
            raise ValueError("Application runtime requires an initialized workspace.")
        cleanup_callbacks = []
        try:
            root_logger = logging.getLogger()
            previous_logging_level = root_logger.level
            logging_handler = configure_operational_logging(
                paths.operational_log,
                level=settings.logging.level,
                max_bytes=settings.logging.max_bytes,
                backup_count=settings.logging.backup_count,
                workspace_id=workspace.id,
            )
            if logging_handler is not None:
                installed_logging_level = root_logger.level
                cleanup_callbacks.append(
                    lambda: cls._close_logging_handler(
                        logging_handler,
                        previous_level=previous_logging_level,
                        installed_level=installed_logging_level,
                    )
                )
            backend = cls._create_backend(settings)
            cls._track_close(backend, cleanup_callbacks)
            telemetry_adapter = SQLiteTelemetryAdapter(
                paths.telemetry,
                workspace_id=workspace.id,
                busy_timeout_ms=settings.telemetry.sqlite_busy_timeout_ms,
            )
            adapter_tracked = cls._track_close(telemetry_adapter, cleanup_callbacks)
            telemetry = Telemetry(
                telemetry_adapter,
                queue_capacity=settings.telemetry.queue_capacity,
                batch_size=settings.telemetry.batch_size,
                flush_seconds=settings.telemetry.flush_seconds,
                workspace_id=workspace.id,
            )
            if adapter_tracked:
                cleanup_callbacks.pop()
            cleanup_callbacks.append(
                lambda: cls._close_telemetry(telemetry, settings.telemetry.shutdown_timeout)
            )
            session_store = SQLiteSessionStore(
                workspace_paths.sessions,
                workspace_id=workspace.id,
            )
            cls._track_close(session_store, cleanup_callbacks)
            permission_manager = PermissionManager(
                workspace.root,
                configuration_path=workspace_paths.permissions,
                audit_path=paths.permissions_audit,
                workspace_id=workspace.id,
                interaction=interaction,
            )
            cls._track_close(permission_manager, cleanup_callbacks)
            loop = Loop.create_default(
                backend,
                interaction=interaction,
                tool_registry=create_default_tool_registry(
                    interaction=interaction,
                    settings=ToolRuntimeSettings(
                        user_agent=settings.web.user_agent,
                        command_timeout=settings.tools.command_timeout,
                    ),
                ),
                working_directory=workspace.working_directory,
                instructions_manager=InstructionsManager.discover(
                    workspace.working_directory.resolve(),
                    workspace_id=workspace.id,
                    workspace_root=workspace.root,
                ),
                permission_manager=permission_manager,
                session_manager=SessionManager(
                    interaction=interaction,
                    session_store=session_store,
                    workspace_id=workspace.id,
                ),
                agent_name=settings.loop.agent_name,
                model=settings.loop.model,
                temperature=settings.loop.temperature,
                reasoning_effort=settings.loop.reasoning_effort,
                on_model_select=(
                    None
                    if configuration.source_for("loop.model") == "environment"
                    else lambda model: configuration.set("loop.model", model)
                ),
                stream=settings.loop.stream,
                debug=settings.loop.debug,
                compaction_threshold=settings.loop.compaction_threshold,
                prompt_on_recoverable_error=settings.loop.prompt_on_recoverable_error,
                max_agent_turns=settings.loop.max_agent_turns,
            )
            runtime = cls(
                loop,
                telemetry,
                settings.telemetry.shutdown_timeout,
                logging_handler=logging_handler,
                cleanup_callbacks=cleanup_callbacks,
            )
            loop.command_manager.register_all(
                ConfigurationCommands(
                    configuration,
                    runtime.apply_configuration,
                    runtime.apply_configuration_changes,
                ).get_commands()
            )
            loop.command_manager.register_all(
                ApplicationCommands(paths, workspace_paths, workspace.id).get_commands()
            )
            loop.command_manager.register_all(
                WorkspaceCommands(workspace, workspace_repository).get_commands()
            )
            previous_telemetry = get_telemetry()
            set_telemetry(telemetry)
            cleanup_callbacks.append(lambda: cls._restore_telemetry(telemetry, previous_telemetry))
            telemetry_activity("application.started", severity="info", component="main")
            return runtime
        except BaseException as error:
            cls._cleanup(cleanup_callbacks, primary=error)
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def run(self) -> None:
        """Run the composed interactive loop."""
        self._loop.run()

    def apply_configuration(self, path: str, settings: ApplicationSettings) -> str:
        """Apply one validated configuration change to the active runtime.

        Args:
            path (str): Changed dot-separated configuration path.
            settings (ApplicationSettings): Newly validated effective configuration.

        Returns:
            str: Whether the setting was applied now or requires restart.
        """
        if path.startswith("backend."):
            self._loop.replace_backend(self._create_backend(settings))
            return "applied now"
        return self._loop.apply_runtime_settings(path, settings)

    def apply_configuration_changes(
        self, paths: tuple[str, ...], settings: ApplicationSettings
    ) -> Mapping[str, str]:
        """Apply a set of changed configuration paths from one reload transaction.

        Args:
            paths (tuple[str, ...]): Effective configuration paths changed by the reload.
            settings (ApplicationSettings): Newly validated effective configuration.

        Returns:
            Mapping[str, str]: Per-path statuses describing whether each change was applied now.
        """
        statuses = {}
        if any(path.startswith("backend.") for path in paths):
            self._loop.replace_backend(self._create_backend(settings))
            statuses.update({path: "applied now" for path in paths if path.startswith("backend.")})
        statuses.update(
            {
                path: self._loop.apply_runtime_settings(path, settings)
                for path in paths
                if not path.startswith("backend.")
            }
        )
        return statuses

    @staticmethod
    def _create_backend(settings: ApplicationSettings) -> OpenAIBackend:
        """Construct the backend described by one effective settings snapshot."""
        return OpenAIBackend(
            base_url=settings.backend.base_url,
            default_model=settings.backend.default_model,
            api_key=settings.backend.api_key.get_secret_value(),
            context_window=settings.backend.context_window,
            file_input_mode=settings.backend.file_input_mode,
            structured_output_mode=settings.backend.structured_output_mode,
            structured_output_max_retries=settings.backend.structured_output_max_retries,
            max_retries=settings.backend.max_retries,
            hyperparameter_policy=settings.backend.hyperparameter_policy,
        )

    def stop(self) -> None:
        """Record an interrupted application shutdown."""
        telemetry_activity("application.stopping", severity="info", reason="interrupted")

    def close(self) -> None:
        """Record shutdown and deterministically release every owned resource.

        Raises:
            TimeoutError: If telemetry cannot flush and shut down within its configured timeout.
            BaseException: The first resource-specific cleanup failure after all cleanup is tried.
        """
        if self._closed:
            return
        self._closed = True
        callbacks, self._cleanup_callbacks = self._cleanup_callbacks, []
        try:
            telemetry_activity("application.stopped", severity="info", component="main")
        except BaseException as error:
            self._cleanup(callbacks, primary=error)
            self._logging_handler = None
            raise
        self._cleanup(callbacks)
        self._logging_handler = None

    @staticmethod
    def _track_close(resource: object, callbacks: list[Callable[[], object]]) -> bool:
        """Track an acquired resource exposing an explicit close operation."""
        close = getattr(resource, "close", None)
        if callable(close):
            callbacks.append(close)
            return True
        return False

    @staticmethod
    def _close_logging_handler(
        handler: logging.Handler,
        *,
        previous_level: int | None = None,
        installed_level: int | None = None,
    ) -> None:
        """Remove one owned handler and restore its root level when still current."""
        root_logger = logging.getLogger()
        root_logger.removeHandler(handler)
        if (
            previous_level is not None
            and installed_level is not None
            and root_logger.level == installed_level
        ):
            root_logger.setLevel(previous_level)
        handler.close()

    @staticmethod
    def _restore_telemetry(telemetry: Telemetry, previous_telemetry: Telemetry | None) -> None:
        """Restore the prior facade only while this runtime still owns publication."""
        if get_telemetry() is telemetry:
            set_telemetry(previous_telemetry)

    @staticmethod
    def _close_telemetry(telemetry: Telemetry, timeout: float) -> None:
        """Flush and close telemetry, exposing an incomplete shutdown."""
        if not telemetry.close(timeout=timeout):
            raise TimeoutError(f"Telemetry did not shut down within {timeout} seconds.")

    @staticmethod
    def _cleanup(
        callbacks: list[Callable[[], object]], *, primary: BaseException | None = None
    ) -> None:
        """Run owned cleanup in reverse order without obscuring a primary failure."""
        cleanup_errors = []
        while callbacks:
            callback = callbacks.pop()
            try:
                callback()
            except BaseException as error:  # noqa: BLE001 - cleanup must survive cancellation
                cleanup_errors.append(error)
        if primary is not None:
            for error in cleanup_errors:
                primary.add_note(f"Runtime rollback also failed: {type(error).__name__}: {error}")
            return
        if cleanup_errors:
            first = cleanup_errors[0]
            for error in cleanup_errors[1:]:
                first.add_note(
                    f"Additional runtime cleanup failed: {type(error).__name__}: {error}"
                )
            raise first
