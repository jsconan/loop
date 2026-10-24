"""Compose and manage one configured application runtime."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Self

from ..backend import OpenAIBackend
from ..configuration import ApplicationSettings, ConfigurationCommands, ConfigurationManager
from ..conversation import ConversationCommands, SessionController
from ..execution import CommandExecutionService
from ..instructions import InstructionsManager
from ..interaction import AsyncInteractionInput, Interaction
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
from ..utils import run_in_worker_async
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
        session_controller (SessionController | None): Optional responsive conversation owner.
            Defaults to the synchronous loop for legacy interaction adapters.
    """

    _loop: Loop
    _telemetry: Telemetry
    _shutdown_timeout: float
    _logging_handler: logging.Handler | None
    _cleanup_callbacks: list[Callable[[], object]]
    _closed: bool
    _session_controller: SessionController | None

    def __init__(
        self,
        loop: Loop,
        telemetry: Telemetry,
        shutdown_timeout: float,
        *,
        logging_handler: logging.Handler | None = None,
        cleanup_callbacks: list[Callable[[], object]] | None = None,
        session_controller: SessionController | None = None,
    ) -> None:
        self._loop = loop
        self._session_controller = session_controller
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
        *,
        main_app: bool = False,
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
            main_app (bool): Enable direct main-app terminal management decisions; False
                (default) prevents library command dispatch from elevating.

        Returns:
            ApplicationRuntime: Fully composed active runtime.

        Raises:
            ValueError: If the workspace has not been initialized.
        """
        if workspace.id is None:
            raise ValueError("Application runtime requires an initialized workspace.")
        session_controller = (
            SessionController(interaction, input_mode=settings.conversation.input_mode)
            if isinstance(interaction, AsyncInteractionInput)
            else None
        )
        active_interaction = (
            session_controller.interaction if session_controller is not None else interaction
        )
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
                user_configuration_path=paths.user_permissions,
                audit_path=paths.permissions_audit,
                workspace_id=workspace.id,
                interaction=active_interaction,
            )
            cls._track_close(permission_manager, cleanup_callbacks)
            execution_service = CommandExecutionService.for_host()
            execution_tracked = cls._track_close(execution_service, cleanup_callbacks)
            tool_registry = create_default_tool_registry(
                interaction=active_interaction,
                permission_manager=permission_manager,
                settings=ToolRuntimeSettings(
                    user_agent=settings.web.user_agent,
                    command_timeout=settings.tools.command_timeout,
                ),
                execution_service=execution_service,
            )
            if execution_tracked:
                cleanup_callbacks.pop()
            cls._track_close(tool_registry, cleanup_callbacks)
            session_manager = SessionManager(
                interaction=active_interaction,
                session_store=session_store,
                workspace_id=workspace.id,
            )
            loop = Loop.create_default(
                backend,
                interaction=active_interaction,
                tool_registry=tool_registry,
                working_directory=workspace.working_directory,
                instructions_manager=InstructionsManager.discover(
                    workspace.working_directory.resolve(),
                    workspace_id=workspace.id,
                    workspace_root=workspace.root,
                ),
                permission_manager=permission_manager,
                session_manager=session_manager,
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
                run_control=session_controller.control if session_controller is not None else None,
                main_app=main_app,
                refresh_sandbox_backend=getattr(execution_service, "refresh_backend", None),
            )
            cls._track_close(loop.command_manager, cleanup_callbacks)
            runtime = cls(
                loop,
                telemetry,
                settings.telemetry.shutdown_timeout,
                logging_handler=logging_handler,
                cleanup_callbacks=cleanup_callbacks,
                session_controller=session_controller,
            )
            if session_controller is not None:
                session_controller.bind_session_manager(session_manager)
                loop.register_command_provider(ConversationCommands(session_controller))
                session_controller.bind_command_manager(loop.command_manager)
            loop.register_command_provider(
                ConfigurationCommands(
                    configuration,
                    runtime.apply_configuration,
                    runtime.apply_configuration_changes,
                    apply_async=runtime.apply_configuration_async,
                    apply_many_async=runtime.apply_configuration_changes_async,
                )
            )
            loop.register_command_provider(
                ApplicationCommands(paths, workspace_paths, workspace.id)
            )
            loop.register_command_provider(WorkspaceCommands(workspace, workspace_repository))
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
        """Run the composed loop through its native synchronous path.

        Raises:
            RuntimeError: If called from an active event loop.
            TypeError: If an agent run completes without metrics.
            KeyboardInterrupt: If the configured stop signal interrupts execution.
            WorkspaceSwitchRequested: If a command requests a workspace runtime rebuild.
        """
        self._loop.run()

    async def run_async(self) -> None:
        """Run the composed application through its async loop and session controller.

        Raises:
            RuntimeError: If the session controller has already run.
            BaseException: If the conversation or UI fails after owned execution settles.

        Returns:
            None: After the conversation and its owned operations have settled.
        """
        if self._session_controller is None:
            await self._loop.run_async()
        else:
            await self._session_controller.run_async(
                lambda receive: self._loop.run_async(input_source=receive),
                route_idle_commands=True,
            )

    def apply_configuration(self, path: str, settings: ApplicationSettings) -> str:
        """Apply one validated configuration change to the active runtime.

        Args:
            path (str): Changed dot-separated configuration path.
            settings (ApplicationSettings): Newly validated effective configuration.

        Returns:
            str: Whether the setting was applied now or requires restart.
        """
        if path.startswith("backend."):
            self._replace_backend(settings)
            return "applied now"
        if path == "conversation.input_mode":
            return self._apply_conversation_input_mode(settings)
        return self._loop.apply_runtime_settings(path, settings)

    async def apply_configuration_async(self, path: str, settings: ApplicationSettings) -> str:
        """Apply one configuration change through native async backend metadata APIs.

        Args:
            path (str): Changed dot-separated configuration path.
            settings (ApplicationSettings): Newly validated effective configuration.

        Returns:
            str: Whether the setting was applied now or requires restart.

        Raises:
            asyncio.CancelledError: If asynchronous model metadata lookup is cancelled.
        """
        if path.startswith("backend."):
            await self._replace_backend_async(settings)
            return "applied now"
        if path == "conversation.input_mode":
            return self._apply_conversation_input_mode(settings)
        return await self._loop.apply_runtime_settings_async(path, settings)

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
            self._replace_backend(settings)
            statuses.update({path: "applied now" for path in paths if path.startswith("backend.")})
        statuses.update(
            {
                path: self._loop.apply_runtime_settings(path, settings)
                for path in paths
                if not path.startswith("backend.") and path != "conversation.input_mode"
            }
        )
        if "conversation.input_mode" in paths:
            statuses["conversation.input_mode"] = self._apply_conversation_input_mode(settings)
        return statuses

    async def apply_configuration_changes_async(
        self, paths: tuple[str, ...], settings: ApplicationSettings
    ) -> Mapping[str, str]:
        """Apply changed settings using async model metadata when the backend changes.

        Args:
            paths (tuple[str, ...]): Effective configuration paths changed by this reload.
            settings (ApplicationSettings): Newly validated effective configuration.

        Returns:
            Mapping[str, str]: Per-path statuses describing whether each change was applied now.

        Raises:
            asyncio.CancelledError: If asynchronous model metadata lookup is cancelled.
        """
        statuses = {}
        if any(path.startswith("backend.") for path in paths):
            await self._replace_backend_async(settings)
            statuses.update({path: "applied now" for path in paths if path.startswith("backend.")})
        for path in paths:
            if not path.startswith("backend."):
                if path == "conversation.input_mode":
                    statuses[path] = self._apply_conversation_input_mode(settings)
                else:
                    statuses[path] = await self._loop.apply_runtime_settings_async(path, settings)
        return statuses

    def _apply_conversation_input_mode(self, settings: ApplicationSettings) -> str:
        """Apply the configured plain-text route to the active session controller."""
        if self._session_controller is None:
            return "saved; restart required"
        self._session_controller.input_mode = settings.conversation.input_mode
        return "applied now"

    def _replace_backend(self, settings: ApplicationSettings) -> None:
        """Track a replacement backend before publishing it to the active loop."""
        backend = self._create_backend(settings)
        self._track_close(backend, self._cleanup_callbacks)
        self._loop.replace_backend(backend)

    async def _replace_backend_async(self, settings: ApplicationSettings) -> None:
        """Track a replacement backend before publishing it through async restoration."""
        backend = self._create_backend(settings)
        self._track_close(backend, self._cleanup_callbacks)
        await self._loop.replace_backend_async(backend)

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
            request_timeout_seconds=settings.backend.request_timeout_seconds,
            max_generation_seconds=settings.backend.max_generation_seconds,
            max_response_chars=settings.backend.max_response_chars,
            max_output_tokens=settings.backend.max_output_tokens,
            repetition_detection=settings.backend.repetition_detection,
            repetition_penalty=settings.backend.repetition_penalty,
            repetition_min_pattern_size=settings.backend.repetition_min_pattern_size,
            repetition_max_pattern_size=settings.backend.repetition_max_pattern_size,
            repetition_min_count=settings.backend.repetition_min_count,
            hyperparameter_policy=settings.backend.hyperparameter_policy,
            retention_policy=settings.backend.retention_policy,
        )

    def stop(self) -> None:
        """Request graceful conversation shutdown and record the interruption."""
        if self._session_controller is not None:
            self._session_controller.stop()
        telemetry_activity("application.stopping", severity="info", reason="interrupted")

    def close(self) -> None:
        """Record shutdown and deterministically release every owned resource.

        Raises:
            RuntimeError: If conversation execution has not settled or async resources need
                ``close_async()``.
            TimeoutError: If telemetry cannot flush and shut down within its configured timeout.
            BaseException: The first resource-specific cleanup failure after all cleanup is tried.
        """
        if self._session_controller is not None and self._session_controller.running:
            raise RuntimeError(
                "Wait for conversation execution to settle before closing the runtime."
            )
        if self._closed:
            return
        if any(
            getattr(callback.__self__, "async_close_required", False)
            for callback in self._cleanup_callbacks
            if inspect.ismethod(callback)
        ):
            raise RuntimeError("Await ApplicationRuntime.close_async() to close async resources.")
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

    async def close_async(self) -> None:
        """Release runtime resources after async work has settled and before its loop closes.

        Native async close methods run on their owning loop. Sync-only cleanup callbacks run on a
        bounded worker and settle before shutdown returns.

        Raises:
            RuntimeError: If conversation execution has not settled, an async resource belongs to
                another event loop, or bounded worker capacity is full.
            TimeoutError: If telemetry cannot flush within its configured timeout.
            BaseException: The first cleanup failure after every owned resource is tried.
            asyncio.CancelledError: After asynchronous resource cleanup settles.
        """
        if self._session_controller is not None and self._session_controller.running:
            raise RuntimeError(
                "Wait for conversation execution to settle before closing the runtime."
            )
        if self._closed:
            return
        loop = asyncio.get_running_loop()
        for callback in self._cleanup_callbacks:
            resource = callback.__self__ if inspect.ismethod(callback) else None
            owner_loop = getattr(resource, "async_close_loop", None)
            if owner_loop is not None and owner_loop is not loop:
                raise RuntimeError("Await runtime cleanup on each async resource's owning loop.")
        self._closed = True
        callbacks, self._cleanup_callbacks = self._cleanup_callbacks, []
        try:
            telemetry_activity("application.stopped", severity="info", component="main")
        except BaseException as error:
            await self._cleanup_async(callbacks, primary=error)
            self._logging_handler = None
            raise
        await self._cleanup_async(callbacks)
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

    @staticmethod
    async def _cleanup_async(
        callbacks: list[Callable[[], object]], *, primary: BaseException | None = None
    ) -> None:
        """Close async resources on their owner and settle sync cleanup on bounded workers."""
        cleanup_errors = []
        cancellation = None
        while callbacks:
            callback = callbacks.pop()
            try:
                resource = callback.__self__ if inspect.ismethod(callback) else None
                close_async = getattr(resource, "close_async", None)
                if callable(close_async):
                    result = close_async()
                elif inspect.iscoroutinefunction(callback):
                    result = callback()
                else:
                    pending = asyncio.ensure_future(run_in_worker_async(callback))
                    while not pending.done():
                        try:
                            await asyncio.shield(pending)
                        except asyncio.CancelledError as error:
                            cancellation = cancellation or error
                    result = pending.result()
                if inspect.isawaitable(result):
                    pending = asyncio.ensure_future(result)
                    while not pending.done():
                        try:
                            await asyncio.shield(pending)
                        except asyncio.CancelledError as error:
                            cancellation = cancellation or error
                    pending.result()
            except BaseException as error:  # noqa: BLE001 - shutdown must try every owner
                cleanup_errors.append(error)
        if primary is not None:
            for error in cleanup_errors:
                primary.add_note(f"Runtime rollback also failed: {type(error).__name__}: {error}")
            if cancellation is not None:
                primary.add_note("Runtime cleanup completed after cancellation was requested.")
            return
        if cancellation is not None:
            for error in cleanup_errors:
                cancellation.add_note(
                    f"Runtime cleanup also failed: {type(error).__name__}: {error}"
                )
            raise cancellation
        if cleanup_errors:
            first = cleanup_errors[0]
            for error in cleanup_errors[1:]:
                first.add_note(
                    f"Additional runtime cleanup failed: {type(error).__name__}: {error}"
                )
            raise first
