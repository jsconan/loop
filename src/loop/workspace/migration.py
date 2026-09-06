"""Migrate legacy workspace artifacts into centralized storage."""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from .. import constants
from .workspace import Workspace


class WorkspaceMigration:
    """Migrate legacy state for one initialized workspace.

    Args:
        workspace (Workspace): Initialized workspace whose legacy directory is inspected.
        sessions (Path): Central session database destination.
        permissions (Path): Central permission policy destination.
    """

    _legacy_root: Path
    _sessions: Path
    _permissions: Path

    def __init__(self, workspace: Workspace, sessions: Path, permissions: Path) -> None:
        if workspace.id is None:
            raise ValueError("Workspace migration requires an initialized workspace.")
        self._legacy_root = workspace.root / constants.APP_DIRECTORY
        self._sessions = sessions
        self._permissions = permissions

    def run(self) -> None:
        """Migrate supported files without overwriting centralized state."""
        self._migrate_sessions()
        self._copy_file(constants.PERMISSIONS_FILENAME, self._permissions)

    def _migrate_sessions(self) -> None:
        source = self._legacy_root / constants.SESSION_DATABASE_FILENAME
        destination = self._sessions
        if not source.is_file() or destination.exists():
            return
        self._ensure_parent(destination)
        temporary = destination.with_name(f".{destination.name}.migration.tmp")
        try:
            with (
                closing(sqlite3.connect(source)) as source_connection,
                closing(sqlite3.connect(temporary)) as destination_connection,
            ):
                source_connection.backup(destination_connection)
            temporary.chmod(constants.PRIVATE_FILE_MODE)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _copy_file(self, filename: str, destination: Path) -> None:
        source = self._legacy_root / filename
        if not source.is_file() or destination.exists():
            return
        self._ensure_parent(destination)
        with source.open("rb") as input_file, destination.open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file)
        destination.chmod(constants.PRIVATE_FILE_MODE)

    @staticmethod
    def _ensure_parent(path: Path) -> None:
        if not path.parent.exists():
            path.parent.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True)
