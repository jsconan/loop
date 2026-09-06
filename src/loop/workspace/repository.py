"""Persist workspace identity independently from application path policy."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from .. import constants
from .workspace import Workspace, WorkspaceNameSource


class WorkspaceRepository:
    """Store durable workspace identity in an injected catalog.

    Args:
        catalog (Path | str): Immutable application workspace-catalog path.
    """

    _catalog: Path

    def __init__(self, catalog: Path | str) -> None:
        self._catalog = Path(catalog).resolve()

    def initialize(
        self,
        workspace: Workspace,
        *,
        busy_timeout_ms: int = constants.DEFAULT_STORAGE_SQLITE_BUSY_TIMEOUT_MS,
    ) -> Workspace:
        """Load or create durable identity for a discovered workspace.

        Args:
            workspace (Workspace): Discovered workspace to initialize.
            busy_timeout_ms (int): Maximum milliseconds to wait for a catalog database lock.

        Returns:
            Workspace: Workspace carrying complete durable identity metadata.

        Raises:
            ValueError: If the timeout is not positive or a copied identity is still active.
        """
        if workspace.id is not None:
            return workspace
        if busy_timeout_ms <= 0:
            raise ValueError("SQLite busy timeout must be positive.")
        name = workspace.root.name or "Untitled workspace"
        name_source: WorkspaceNameSource = "directory" if workspace.root.name else "default"
        legacy = self._legacy_identity(workspace.root)
        connection = self._connect(busy_timeout_ms)
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT w.workspace_id, w.name, w.name_source, w.created_at_ns, w.updated_at_ns "
                "FROM workspaces AS w JOIN workspace_locations AS l "
                "ON l.workspace_id = w.workspace_id WHERE l.canonical_path = ?",
                (str(workspace.root),),
            ).fetchone()
            if row is None:
                now = time.time_ns()
                row = None
                row = legacy or (str(uuid4()), name, name_source, now, now)
                existing_location = connection.execute(
                    "SELECT l.canonical_path FROM workspace_locations AS l "
                    "WHERE l.workspace_id = ? AND l.status = 'active'",
                    (row[0],),
                ).fetchone()
                if existing_location is not None:
                    previous = Path(existing_location[0])
                    if previous.exists():
                        raise ValueError(
                            "Copied workspace shares an existing identity; rekey it before use."
                        )
                    connection.execute(
                        "UPDATE workspace_locations SET canonical_path = ?, last_seen_at_ns = ? "
                        "WHERE workspace_id = ? AND canonical_path = ?",
                        (str(workspace.root), now, row[0], str(previous)),
                    )
                else:
                    connection.execute(
                        "INSERT OR IGNORE INTO workspaces("
                        "workspace_id, name, name_source, created_at_ns, updated_at_ns"
                        ") VALUES (?, ?, ?, ?, ?)",
                        row,
                    )
                    connection.execute(
                        "INSERT INTO workspace_locations("
                        "location_id, workspace_id, canonical_path, first_seen_at_ns, "
                        "last_seen_at_ns, status) VALUES (?, ?, ?, ?, ?, ?)",
                        (str(uuid4()), row[0], str(workspace.root), now, now, "active"),
                    )
            else:
                connection.execute(
                    "UPDATE workspace_locations SET last_seen_at_ns = ? WHERE canonical_path = ?",
                    (time.time_ns(), str(workspace.root)),
                )
            connection.commit()
            initialized = replace(
                workspace,
                id=row[0],
                name=row[1],
                name_source=row[2],
                created_at_ns=row[3],
                updated_at_ns=row[4],
            )
            return initialized
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def list(self, workspace_id: str | None = None) -> tuple[Workspace, ...]:
        """Return registered workspaces in display order.

        Args:
            workspace_id (str | None): Optional identifier used to select one workspace.

        Returns:
            tuple[Workspace, ...]: Registered workspaces ordered by name and identifier.
        """
        query = (
            "SELECT w.workspace_id, w.name, w.name_source, w.created_at_ns, w.updated_at_ns, "
            "l.canonical_path FROM workspaces AS w JOIN workspace_locations AS l "
            "ON l.workspace_id = w.workspace_id"
        )
        parameters: tuple[str, ...] = ()
        if workspace_id is not None:
            query += " WHERE w.workspace_id = ?"
            parameters = (workspace_id,)
        query += " ORDER BY w.name, w.workspace_id"
        with closing(self._connect(constants.DEFAULT_STORAGE_SQLITE_BUSY_TIMEOUT_MS)) as connection:
            return tuple(
                Workspace(
                    root=Path(row[5]),
                    working_directory=Path(row[5]),
                    id=row[0],
                    name=row[1],
                    name_source=row[2],
                    created_at_ns=row[3],
                    updated_at_ns=row[4],
                )
                for row in connection.execute(query, parameters)
            )

    def rename(self, workspace_id: str, name: str) -> None:
        """Persist a user-authored workspace name.

        Args:
            workspace_id (str): Durable identifier of the workspace to rename.
            name (str): Validated non-empty replacement name.
        """
        with (
            closing(self._connect(constants.DEFAULT_STORAGE_SQLITE_BUSY_TIMEOUT_MS)) as connection,
            connection,
        ):
            connection.execute(
                "UPDATE workspaces SET name = ?, name_source = 'user', updated_at_ns = ? "
                "WHERE workspace_id = ?",
                (name, time.time_ns(), workspace_id),
            )

    def _connect(self, busy_timeout_ms: int) -> sqlite3.Connection:
        """Open the workspace catalog and ensure its schema exists."""
        path = self._catalog
        if not path.parent.exists():
            path.parent.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True)
        connection = sqlite3.connect(path)
        try:
            path.chmod(constants.PRIVATE_FILE_MODE)
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    name_source TEXT NOT NULL,
                    created_at_ns INTEGER NOT NULL,
                    updated_at_ns INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_locations (
                    location_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
                    canonical_path TEXT NOT NULL UNIQUE,
                    locator_kind TEXT,
                    locator BLOB,
                    first_seen_at_ns INTEGER NOT NULL,
                    last_seen_at_ns INTEGER NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
        except BaseException:
            connection.close()
            raise
        return connection

    def _legacy_identity(self, root: Path) -> tuple[str, str, WorkspaceNameSource, int, int] | None:
        """Return identity metadata from the former project-local catalog when available."""
        path = root / constants.APP_DIRECTORY / constants.WORKSPACE_DATABASE_FILENAME
        if not path.is_file() or path.resolve() == self._catalog:
            return None
        try:
            with closing(sqlite3.connect(path)) as connection:
                return connection.execute(
                    "SELECT workspace_id, name, name_source, created_at_ns, updated_at_ns "
                    "FROM workspaces ORDER BY created_at_ns LIMIT 1"
                ).fetchone()
        except sqlite3.Error:
            return None
