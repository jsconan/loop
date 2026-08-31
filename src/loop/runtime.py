"""Compose and manage one configured application runtime."""

from __future__ import annotations

from .backend import OpenAIBackend
from .configuration import ApplicationSettings, ConfigurationCommands, ConfigurationManager
from .interaction import Interaction
from .loop import Loop
from .permissions import PermissionManager
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
from .workspace import Workspace


class ApplicationRuntime:
    """Manage the assembled application and its process-wide telemetry lifecycle.

    Args:
        loop (Loop): Fully assembled interactive application.
        telemetry (Telemetry): Application telemetry service to start and close with the runtime.
        shutdown_timeout (float): Maximum seconds allowed for telemetry shutdown.
    """

    _loop: Loop
    _telemetry: Telemetry
    _shutdown_timeout: float

    def __init__(self, loop: Loop, telemetry: Telemetry, shutdown_timeout: float) -> None:
        self._loop = loop
        self._telemetry = telemetry
        self._shutdown_timeout = shutdown_timeout

    @classmethod
    def create(
        cls,
        workspace: Workspace,
        settings: ApplicationSettings,
        configuration: ConfigurationManager,
        interaction: Interaction,
    ) -> ApplicationRuntime:
        """Build an application runtime from one effective configuration snapshot.

        Args:
            workspace (Workspace): Active worktree and its durable artifact locations.
            settings (ApplicationSettings): Validated immutable application configuration.
            configuration (ConfigurationManager): Configuration owner used to persist model choices.
            interaction (Interaction): User interaction service for the assembled loop.

        Returns:
            ApplicationRuntime: Fully composed runtime with active process-wide telemetry.

        """
        configure_operational_logging(
            workspace.storage.operational_log,
            level=settings.logging.level,
            max_bytes=settings.logging.max_bytes,
            backup_count=settings.logging.backup_count,
        )
        telemetry = None
        try:
            backend = cls._create_backend(settings)
            telemetry = Telemetry(
                SQLiteTelemetryAdapter(
                    workspace.storage.telemetry,
                    busy_timeout_ms=settings.telemetry.sqlite_busy_timeout_ms,
                ),
                queue_capacity=settings.telemetry.queue_capacity,
                batch_size=settings.telemetry.batch_size,
                flush_seconds=settings.telemetry.flush_seconds,
                workspace_root=workspace.root,
            )
            loop = Loop.create_default(
                backend,
                interaction=interaction,
                tool_registry=create_default_tool_registry(
                    interaction=interaction,
                    settings=ToolRuntimeSettings(user_agent=settings.web.user_agent),
                ),
                working_directory=workspace.working_directory,
                permission_manager=PermissionManager(
                    workspace.root,
                    configuration_path=workspace.storage.permissions,
                    audit_path=workspace.storage.permissions_audit,
                    interaction=interaction,
                ),
                session_manager=SessionManager(
                    interaction=interaction,
                    session_store=SQLiteSessionStore(workspace.storage.sessions),
                    workspace_root=workspace.root,
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
        runtime = cls(loop, telemetry, settings.telemetry.shutdown_timeout)
        loop.command_manager.register_all(
            ConfigurationCommands(configuration, runtime.apply_configuration).get_commands()
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
