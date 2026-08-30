"""Compose and manage one configured application runtime."""

from __future__ import annotations

from pathlib import Path

from .backend import OpenAIBackend
from .configuration import ApplicationSettings, ConfigurationManager
from .constants import (
    APP_DIRECTORY,
    OPERATIONAL_LOG_FILENAME,
    SESSION_DATABASE_FILENAME,
    TELEMETRY_DATABASE_FILENAME,
)
from .interaction import Interaction
from .loop import Loop
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
        project_root: Path,
        working_directory: Path,
        settings: ApplicationSettings,
        configuration: ConfigurationManager,
        interaction: Interaction,
    ) -> ApplicationRuntime:
        """Build an application runtime from one effective configuration snapshot.

        Args:
            project_root (Path): Project directory that owns durable application artifacts.
            working_directory (Path): Directory used to resolve workspace-local behavior.
            settings (ApplicationSettings): Validated immutable application configuration.
            configuration (ConfigurationManager): Configuration owner used to persist model choices.
            interaction (Interaction): User interaction service for the assembled loop.

        Returns:
            ApplicationRuntime: Fully composed runtime with active process-wide telemetry.

        """
        configure_operational_logging(
            project_root / APP_DIRECTORY / OPERATIONAL_LOG_FILENAME,
            level=settings.logging.level,
            max_bytes=settings.logging.max_bytes,
            backup_count=settings.logging.backup_count,
        )
        telemetry = None
        try:
            backend = OpenAIBackend(
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
            telemetry = Telemetry(
                SQLiteTelemetryAdapter(
                    project_root / APP_DIRECTORY / TELEMETRY_DATABASE_FILENAME,
                    busy_timeout_ms=settings.telemetry.sqlite_busy_timeout_ms,
                ),
                queue_capacity=settings.telemetry.queue_capacity,
                batch_size=settings.telemetry.batch_size,
                flush_seconds=settings.telemetry.flush_seconds,
            )
            loop = Loop.create_default(
                backend,
                interaction=interaction,
                tool_registry=create_default_tool_registry(
                    interaction=interaction,
                    settings=ToolRuntimeSettings(user_agent=settings.web.user_agent),
                ),
                working_directory=working_directory,
                session_manager=SessionManager(
                    interaction=interaction,
                    session_store=SQLiteSessionStore(
                        project_root / APP_DIRECTORY / SESSION_DATABASE_FILENAME
                    ),
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
        set_telemetry(telemetry)
        telemetry_activity("application.started", severity="info", component="main")
        return runtime

    def run(self) -> None:
        """Run the composed interactive loop."""
        self._loop.run()

    def stop(self) -> None:
        """Record an interrupted application shutdown."""
        telemetry_activity("application.stopping", severity="info", reason="interrupted")

    def close(self) -> None:
        """Record shutdown and close process-wide telemetry."""
        telemetry_activity("application.stopped", severity="info", component="main")
        set_telemetry(None)
        self._telemetry.close(timeout=self._shutdown_timeout)
