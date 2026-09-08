"""Persist workspace identity independently from application path policy."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from .. import constants
from .models import Workspace, WorkspaceNameSource


class WorkspaceRepository:
    """Store durable workspace identity in an injected catalog.

    Args:
        catalog (Path | str): Immutable application workspace-catalog path.
        workspace_data_root (Path | str | None): UUID-scoped data root. Defaults to the
            ``workspaces`` directory beside the catalog.
    """

    _catalog: Path
    _workspace_data_root: Path

    def __init__(
        self,
        catalog: Path | str,
        workspace_data_root: Path | str | None = None,
    ) -> None:
        self._catalog = Path(catalog).resolve()
        self._workspace_data_root = (
            Path(workspace_data_root).resolve()
            if workspace_data_root is not None
            else self._catalog.parent / "workspaces"
        )

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
        locator_kind, locator = self._locator(workspace.root)
        connection = self._connect(busy_timeout_ms)
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT w.workspace_id, w.name, w.name_source, w.created_at_ns, w.updated_at_ns "
                "FROM workspaces AS w JOIN workspace_locations AS l "
                "ON l.workspace_id = w.workspace_id "
                "WHERE l.canonical_path = ? AND l.status = 'active'",
                (str(workspace.root),),
            ).fetchone()
            relocated = False
            if row is None and locator is not None:
                matches = connection.execute(
                    "SELECT w.workspace_id, w.name, w.name_source, w.created_at_ns, "
                    "w.updated_at_ns, l.canonical_path FROM workspaces AS w "
                    "JOIN workspace_locations AS l ON l.workspace_id = w.workspace_id "
                    "WHERE l.locator_kind = ? AND l.locator = ? AND l.status = 'active'",
                    (locator_kind, locator),
                ).fetchall()
                if len(matches) == 1:
                    match = matches[0]
                    previous = Path(match[5])
                    if not previous.exists():
                        row = match[:5]
                        relocated = True
                        now = time.time_ns()
                        connection.execute(
                            "UPDATE workspace_locations SET canonical_path = ?, "
                            "last_seen_at_ns = ? WHERE workspace_id = ? AND canonical_path = ?",
                            (str(workspace.root), now, row[0], str(previous)),
                        )
                        row = self._refresh_directory_name(connection, row, workspace.root, now)
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
                        "last_seen_at_ns, status, locator_kind, locator) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(canonical_path) DO UPDATE SET "
                        "workspace_id=excluded.workspace_id,locator_kind=excluded.locator_kind,"
                        "locator=excluded.locator,first_seen_at_ns=excluded.first_seen_at_ns,"
                        "last_seen_at_ns=excluded.last_seen_at_ns,status='active'",
                        (
                            str(uuid4()),
                            row[0],
                            str(workspace.root),
                            now,
                            now,
                            "active",
                            locator_kind,
                            locator,
                        ),
                    )
            else:
                if not relocated:
                    connection.execute(
                        "UPDATE workspace_locations SET last_seen_at_ns = ?, locator_kind = ?, "
                        "locator = ? WHERE canonical_path = ?",
                        (time.time_ns(), locator_kind, locator, str(workspace.root)),
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
            "ON l.workspace_id = w.workspace_id WHERE l.status = 'active'"
        )
        parameters: tuple[str, ...] = ()
        if workspace_id is not None:
            query += " AND w.workspace_id = ?"
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
        with (  # pylint: disable=confusing-with-statement
            closing(self._connect(constants.DEFAULT_STORAGE_SQLITE_BUSY_TIMEOUT_MS)) as connection,
            connection,
        ):
            connection.execute(
                "UPDATE workspaces SET name = ?, name_source = 'user', updated_at_ns = ? "
                "WHERE workspace_id = ?",
                (name, time.time_ns(), workspace_id),
            )

    def refresh_name(
        self,
        workspace_id: str,
        name: str,
        source: WorkspaceNameSource,
        *,
        relocation_confirmed: bool = False,
    ) -> bool:
        """Conservatively refresh a workspace name from a trusted source.

        Args:
            workspace_id (str): Durable workspace identity.
            name (str): Non-empty candidate name.
            source (WorkspaceNameSource): Source supplying the candidate.
            relocation_confirmed (bool): Whether locator resolution confirmed a directory move.

        Returns:
            bool: Whether the durable name changed.

        Raises:
            ValueError: If the candidate is empty or the workspace is unknown.
        """
        candidate = name.strip()
        if not candidate:
            raise ValueError("Workspace name must not be empty.")
        with (  # pylint: disable=confusing-with-statement
            closing(self._connect(constants.DEFAULT_STORAGE_SQLITE_BUSY_TIMEOUT_MS)) as connection,
            connection,
        ):
            row = connection.execute(
                "SELECT name,name_source FROM workspaces WHERE workspace_id=?", (workspace_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown workspace identity '{workspace_id}'.")
            current_name, current_source = row
            allowed = current_source == "default" and source != "default"
            allowed = allowed or (source in {"provider", "remote"} and current_source == source)
            allowed = allowed or (
                source == "directory" and current_source == "directory" and relocation_confirmed
            )
            if not allowed or (current_name == candidate and current_source == source):
                return False
            connection.execute(
                "UPDATE workspaces SET name=?,name_source=?,updated_at_ns=? WHERE workspace_id=?",
                (candidate, source, time.time_ns(), workspace_id),
            )
        return True

    def attach(self, path: Path | str, workspace_id: str | None = None) -> Workspace:
        """Attach a path to a new or existing durable workspace identity.

        Args:
            path (Path | str): Existing workspace path to associate.
            workspace_id (str | None): Existing identity, or ``None`` to create one.

        Returns:
            Workspace: Initialized attached workspace.

        Raises:
            FileNotFoundError: If the path is not an existing directory.
            ValueError: If the identity is unknown or either side is already active elsewhere.
        """
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Workspace path is not a directory: {root}")
        if workspace_id is None:
            return self.initialize(Workspace.discover(root))
        kind, locator = self._locator(root)
        with closing(self._connect(constants.DEFAULT_STORAGE_SQLITE_BUSY_TIMEOUT_MS)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT workspace_id, name, name_source, created_at_ns, updated_at_ns "
                    "FROM workspaces WHERE workspace_id = ?",
                    (workspace_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"Unknown workspace identity '{workspace_id}'.")
                conflict = connection.execute(
                    "SELECT workspace_id FROM workspace_locations WHERE canonical_path = ? "
                    "AND status = 'active'",
                    (str(root),),
                ).fetchone()
                if conflict is not None and conflict[0] != workspace_id:
                    raise ValueError("Workspace path is already attached to another identity.")
                live = connection.execute(
                    "SELECT canonical_path FROM workspace_locations WHERE workspace_id = ? "
                    "AND status = 'active' AND canonical_path != ?",
                    (workspace_id, str(root)),
                ).fetchall()
                if any(Path(item[0]).exists() for item in live):
                    raise ValueError("Workspace identity is already attached to an active path.")
                now = time.time_ns()
                connection.execute(
                    "UPDATE workspace_locations SET status = 'inactive' WHERE workspace_id = ?",
                    (workspace_id,),
                )
                connection.execute(
                    "INSERT INTO workspace_locations(location_id, workspace_id, canonical_path, "
                    "locator_kind, locator, first_seen_at_ns, last_seen_at_ns, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'active') ON CONFLICT(canonical_path) DO UPDATE "
                    "SET workspace_id=excluded.workspace_id, locator_kind=excluded.locator_kind, "
                    "locator=excluded.locator, last_seen_at_ns=excluded.last_seen_at_ns, "
                    "status='active'",
                    (str(uuid4()), workspace_id, str(root), kind, locator, now, now),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return Workspace(root, root, *row)

    def forget(self, identity_or_path: str | Path) -> bool:
        """Deactivate location mappings without deleting workspace-owned data.

        Args:
            identity_or_path (str | Path): Workspace UUID or canonical location.

        Returns:
            bool: Whether an active mapping was deactivated.
        """
        value = str(identity_or_path)
        candidate = str(Path(value).expanduser().resolve())
        with (  # pylint: disable=confusing-with-statement
            closing(self._connect(constants.DEFAULT_STORAGE_SQLITE_BUSY_TIMEOUT_MS)) as connection,
            connection,
        ):
            cursor = connection.execute(
                "UPDATE workspace_locations SET status = 'inactive' WHERE status = 'active' "
                "AND (workspace_id = ? OR canonical_path = ?)",
                (value, candidate),
            )
        return cursor.rowcount > 0

    def rekey(self, path: Path | str) -> Workspace:
        """Assign a fresh UUID to an attached copied workspace path.

        Args:
            path (Path | str): Existing copied workspace path.

        Returns:
            Workspace: Workspace carrying the new identity.
        """
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Workspace path is not a directory: {root}")
        kind, locator = self._locator(root)
        now = time.time_ns()
        workspace_id = str(uuid4())
        name = root.name or "Untitled workspace"
        source: WorkspaceNameSource = "directory" if root.name else "default"
        previous = self.list_by_path(root)
        if previous is None:
            legacy = self._legacy_identity(root)
            previous_id = legacy[0] if legacy is not None else None
        else:
            previous_id = previous.id
        copied_data = self._copy_rekey_data(previous_id, workspace_id)
        try:
            with closing(
                self._connect(constants.DEFAULT_STORAGE_SQLITE_BUSY_TIMEOUT_MS)
            ) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(
                        "INSERT INTO workspaces("
                        "workspace_id,name,name_source,created_at_ns,updated_at_ns) "
                        "VALUES (?,?,?,?,?)",
                        (workspace_id, name, source, now, now),
                    )
                    cursor = connection.execute(
                        "UPDATE workspace_locations SET workspace_id=?,locator_kind=?,locator=?,"
                        "last_seen_at_ns=?,status='active' WHERE canonical_path=?",
                        (workspace_id, kind, locator, now, str(root)),
                    )
                    if cursor.rowcount == 0:
                        connection.execute(
                            "INSERT INTO workspace_locations("
                            "location_id,workspace_id,canonical_path,"
                            "locator_kind,locator,first_seen_at_ns,last_seen_at_ns,status) "
                            "VALUES (?,?,?,?,?,?,?,'active')",
                            (str(uuid4()), workspace_id, str(root), kind, locator, now, now),
                        )
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise
        except BaseException:
            if copied_data is not None:
                shutil.rmtree(copied_data)
            raise
        return Workspace(root, root, workspace_id, name, source, now, now)

    def list_by_path(self, path: Path | str) -> Workspace | None:
        """Return the active workspace at a canonical path.

        Args:
            path (Path | str): Location to query.

        Returns:
            Workspace | None: Matching workspace when registered.
        """
        root = Path(path).expanduser().resolve()
        with closing(self._connect(constants.DEFAULT_STORAGE_SQLITE_BUSY_TIMEOUT_MS)) as connection:
            row = connection.execute(
                "SELECT w.workspace_id, w.name, w.name_source, w.created_at_ns, w.updated_at_ns "
                "FROM workspaces AS w JOIN workspace_locations AS l "
                "ON l.workspace_id=w.workspace_id WHERE l.canonical_path=? AND l.status='active'",
                (str(root),),
            ).fetchone()
        return None if row is None else Workspace(root, root, *row)

    def resolve(self, identity_or_path: str | Path) -> Workspace:
        """Resolve an active workspace by UUID or path, attaching a new path when needed.

        Args:
            identity_or_path (str | Path): Registered UUID or existing workspace path.

        Returns:
            Workspace: Unambiguously resolved initialized workspace.

        Raises:
            ValueError: If an identifier is unknown.
        """
        value = str(identity_or_path)
        matches = self.list(value)
        if len(matches) == 1:
            return matches[0]
        path = Path(value).expanduser()
        if path.is_dir():
            return self.initialize(Workspace.discover(path))
        raise ValueError(f"Unknown workspace identity or path '{value}'.")

    @staticmethod
    def _locator(path: Path) -> tuple[str | None, bytes | None]:
        """Return an advisory platform locator for an existing directory."""
        try:
            stat = path.stat()
        except OSError:
            return None, None
        kind = "windows_volume_file" if sys.platform == "win32" else "posix_inode"
        return kind, f"{stat.st_dev}:{stat.st_ino}".encode("ascii")

    def _copy_rekey_data(self, previous_id: str | None, workspace_id: str) -> Path | None:
        """Copy UUID-scoped data and rewrite embedded session ownership."""
        if previous_id is None:
            return None
        source = self._workspace_data_root / previous_id
        if not source.is_dir():
            return None
        destination = self._workspace_data_root / workspace_id
        temporary = destination.with_name(f".{workspace_id}.rekey.tmp")
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"Rekey destination already exists: {destination}")
        shutil.copytree(source, temporary)
        try:
            sessions = temporary / constants.SESSION_DATABASE_FILENAME
            source_sessions = source / constants.SESSION_DATABASE_FILENAME
            if source_sessions.is_file():
                sessions.unlink(missing_ok=True)
                sessions.with_name(f"{sessions.name}-wal").unlink(missing_ok=True)
                sessions.with_name(f"{sessions.name}-shm").unlink(missing_ok=True)
                with (
                    closing(sqlite3.connect(source_sessions)) as source_connection,
                    closing(sqlite3.connect(sessions)) as destination_connection,
                ):
                    source_connection.backup(destination_connection)
                with (  # pylint: disable=confusing-with-statement
                    closing(sqlite3.connect(sessions)) as connection,
                    connection,
                ):
                    tables = {
                        row[0]
                        for row in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )
                    }
                    if "sessions" in tables:
                        for session_id, payload in connection.execute(
                            "SELECT id, session FROM sessions"
                        ).fetchall():
                            value = json.loads(payload)
                            value["workspace_id"] = workspace_id
                            connection.execute(
                                "UPDATE sessions SET session=? WHERE id=?",
                                (json.dumps(value, separators=(",", ":")), session_id),
                            )
            temporary.replace(destination)
            return destination
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    @staticmethod
    def _refresh_directory_name(
        connection: sqlite3.Connection,
        row: tuple[str, str, WorkspaceNameSource, int, int],
        root: Path,
        now: int,
    ) -> tuple[str, str, WorkspaceNameSource, int, int]:
        """Refresh only directory/default names after confirmed relocation."""
        workspace_id, _name, source, created, _updated = row
        replacement = root.name or "Untitled workspace"
        replacement_source: WorkspaceNameSource = "directory" if root.name else "default"
        if source == "directory" or (source == "default" and replacement_source != "default"):
            connection.execute(
                "UPDATE workspaces SET name=?, name_source=?, updated_at_ns=? WHERE workspace_id=?",
                (replacement, replacement_source, now, workspace_id),
            )
            return workspace_id, replacement, replacement_source, created, now
        return row

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
