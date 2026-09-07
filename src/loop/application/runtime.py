"""Compose and manage one configured application runtime."""

from __future__ import annotations

import logging

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
    """

    _loop: Loop
    _telemetry: Telemetry
    _shutdown_timeout: float
    _logging_handler: logging.Handler | None

    def __init__(
        self,
        loop: Loop,
        telemetry: Telemetry,
        shutdown_timeout: float,
        logging_handler: logging.Handler | None = None,
    ) -> None:
        self._loop = loop
        self._telemetry = telemetry
        self._shutdown_timeout = shutdown_timeout
        self._logging_handler = logging_handler

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
        logging_handler = configure_operational_logging(
            paths.operational_log,
            level=settings.logging.level,
            max_bytes=settings.logging.max_bytes,
            backup_count=settings.logging.backup_count,
        )
        telemetry = None
        try:
            backend = cls._create_backend(settings)
            telemetry = Telemetry(
                SQLiteTelemetryAdapter(
                    paths.telemetry,
                    workspace_id=workspace.id,
                    busy_timeout_ms=settings.telemetry.sqlite_busy_timeout_ms,
                ),
                queue_capacity=settings.telemetry.queue_capacity,
                batch_size=settings.telemetry.batch_size,
                flush_seconds=settings.telemetry.flush_seconds,
                workspace_id=workspace.id,
            )
            loop = Loop.create_default(
                backend,
                interaction=interaction,
                tool_registry=create_default_tool_registry(
                    interaction=interaction,
                    settings=ToolRuntimeSettings(user_agent=settings.web.user_agent),
                ),
                working_directory=workspace.working_directory,
                instructions_manager=InstructionsManager.discover(
                    workspace.working_directory.resolve(),
                    workspace_id=workspace.id,
                    workspace_root=workspace.root,
                ),
                permission_manager=PermissionManager(
                    workspace.root,
                    configuration_path=workspace_paths.permissions,
                    audit_path=paths.permissions_audit,
                    workspace_id=workspace.id,
                    interaction=interaction,
                ),
                session_manager=SessionManager(
                    interaction=interaction,
                    session_store=SQLiteSessionStore(
                        workspace_paths.sessions,
                        workspace_id=workspace.id,
                    ),
                    workspace_id=workspace.id,
                ),
                agent_name=settings.loop.agent_name,
                model=settings.loop.model,
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
        except Exception:
            if telemetry is not None:
                telemetry.close(timeout=settings.telemetry.shutdown_timeout)
            raise
        runtime = cls(loop, telemetry, settings.telemetry.shutdown_timeout, logging_handler)
        loop.command_manager.register_all(
            ConfigurationCommands(configuration, runtime.apply_configuration).get_commands()
        )
        loop.command_manager.register_all(
            ApplicationCommands(paths, workspace_paths, workspace.id).get_commands()
        )
        loop.command_manager.register_all(
            WorkspaceCommands(workspace, workspace_repository).get_commands()
        )
        set_telemetry(telemetry)
        telemetry_activity("application.started", severity="info", component="main")
        return runtime

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
            temperature=settings.backend.temperature,
            reasoning_effort=settings.backend.reasoning_effort,
            hyperparameter_policy=settings.backend.hyperparameter_policy,
        )

    def stop(self) -> None:
        """Record an interrupted application shutdown."""
        telemetry_activity("application.stopping", severity="info", reason="interrupted")

    def close(self) -> None:
        """Record shutdown and close process-wide telemetry."""
        telemetry_activity("application.stopped", severity="info", component="main")
        set_telemetry(None)
        self._telemetry.close(timeout=self._shutdown_timeout)
        if self._logging_handler is not None:
            logging.getLogger().removeHandler(self._logging_handler)
            self._logging_handler.close()
            self._logging_handler = None
