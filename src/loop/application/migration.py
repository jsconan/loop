"""Coordinate legacy artifact imports through their owning components."""

from __future__ import annotations

import shutil
from pathlib import Path

from .. import constants
from ..permissions import SQLitePermissionAudit
from ..session import SQLiteSessionStore
from ..telemetry import SQLiteTelemetryAdapter, import_legacy_operational_log
from ..workspace import Workspace
from .paths import ApplicationPaths, WorkspacePaths


class ApplicationMigration:
    """Import legacy artifacts for one initialized workspace.

    Args:
        workspace (Workspace): Initialized workspace whose legacy directory is inspected.
        application_paths (ApplicationPaths): Central application destinations.
        workspace_paths (WorkspacePaths): Central workspace destinations.
        busy_timeout_ms (int): Maximum milliseconds to wait for centralized database locks.

    Raises:
        ValueError: If the workspace is not initialized or the timeout is not positive.
    """

    _application_paths: ApplicationPaths
    _legacy_root: Path
    _workspace: Workspace
    _workspace_paths: WorkspacePaths
    _busy_timeout_ms: int

    def __init__(
        self,
        workspace: Workspace,
        application_paths: ApplicationPaths,
        workspace_paths: WorkspacePaths,
        *,
        busy_timeout_ms: int = constants.DEFAULT_STORAGE_SQLITE_BUSY_TIMEOUT_MS,
    ) -> None:
        if workspace.id is None:
            raise ValueError("Application migration requires an initialized workspace.")
        if busy_timeout_ms <= 0:
            raise ValueError("SQLite busy timeout must be positive.")
        self._workspace = workspace
        self._legacy_root = workspace.root / constants.APP_DIRECTORY
        self._application_paths = application_paths
        self._workspace_paths = workspace_paths
        self._busy_timeout_ms = busy_timeout_ms

    def run(self) -> None:
        """Import supported legacy files without overwriting centralized files."""
        SQLiteSessionStore(
            self._workspace_paths.sessions, workspace_id=self._workspace.id
        ).import_legacy(self._legacy_root / constants.SESSION_DATABASE_FILENAME)
        self._copy_policy()
        telemetry_source = self._legacy_root / constants.TELEMETRY_DATABASE_FILENAME
        if telemetry_source.is_file():
            telemetry = SQLiteTelemetryAdapter(
                self._application_paths.telemetry,
                workspace_id=self._workspace.id,
                busy_timeout_ms=self._busy_timeout_ms,
            )
            try:
                telemetry.import_legacy(telemetry_source)
                telemetry.flush()
            finally:
                telemetry.close()
        SQLitePermissionAudit(
            self._application_paths.permissions_audit,
            busy_timeout_ms=self._busy_timeout_ms,
        ).import_legacy_jsonl(
            self._legacy_root / "permissions-audit.jsonl",
            workspace_id=self._workspace.id,
        )
        import_legacy_operational_log(
            self._legacy_root / constants.OPERATIONAL_LOG_FILENAME,
            self._application_paths.operational_log,
        )

    def _copy_policy(self) -> None:
        """Copy the opaque legacy permission policy when no central policy exists."""
        source = self._legacy_root / constants.PERMISSIONS_FILENAME
        destination = self._workspace_paths.permissions
        if not source.is_file() or destination.exists():
            return
        destination.parent.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
        with source.open("rb") as input_file, destination.open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file)
        destination.chmod(constants.PRIVATE_FILE_MODE)
