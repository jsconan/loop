"""Discover workspaces and expose their durable artifact locations."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .. import constants
from ..utils import find_project_root

WorkspaceNameSource = Literal["user", "provider", "remote", "directory", "default"]


@dataclass(frozen=True, slots=True)
class WorkspaceStorage:
    """Expose every durable artifact owned by one workspace.

    Args:
        root (Path): Directory containing the workspace's application artifacts.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())

    @property
    def configuration(self) -> Path:
        """Return the workspace configuration path.

        Returns:
            Path: Workspace TOML configuration path.
        """
        return self.root / constants.APP_CONFIGURATION_FILENAME

    @property
    def sessions(self) -> Path:
        """Return the workspace session database path.

        Returns:
            Path: Workspace session database path.
        """
        return self.root / constants.SESSION_DATABASE_FILENAME

    @property
    def workspaces(self) -> Path:
        """Return the workspace catalog database path.

        Returns:
            Path: Workspace catalog database path.
        """
        return self.root / constants.WORKSPACE_DATABASE_FILENAME

    @property
    def telemetry(self) -> Path:
        """Return the workspace telemetry database path.

        Returns:
            Path: Workspace telemetry database path.
        """
        return self.root / constants.TELEMETRY_DATABASE_FILENAME

    @property
    def operational_log(self) -> Path:
        """Return the workspace operational log path.

        Returns:
            Path: Workspace operational log path.
        """
        return self.root / constants.OPERATIONAL_LOG_FILENAME

    @property
    def permissions(self) -> Path:
        """Return the workspace permission policy path.

        Returns:
            Path: Workspace permission policy path.
        """
        return self.root / constants.PERMISSIONS_FILENAME

    @property
    def permissions_audit(self) -> Path:
        """Return the workspace permission audit path.

        Returns:
            Path: Workspace permission audit path.
        """
        return self.root / constants.PERMISSIONS_AUDIT_FILENAME


@dataclass(frozen=True, slots=True)
class Workspace:
    """Represent one worktree and its active and durable locations.

    Args:
        root (Path): Canonical workspace or Git worktree root.
        working_directory (Path): Canonical active directory within the workspace.
        storage (WorkspaceStorage): Durable artifact locations owned by the workspace.
        id (str | None): Durable workspace identifier after initialization.
        name (str | None): Human-readable workspace name after initialization.
        name_source (WorkspaceNameSource | None): Origin of the name after initialization.
        created_at_ns (int | None): Workspace creation time after initialization.
        updated_at_ns (int | None): Last workspace metadata update after initialization.
    """

    root: Path
    working_directory: Path
    storage: WorkspaceStorage
    id: str | None = None
    name: str | None = None
    name_source: WorkspaceNameSource | None = None
    created_at_ns: int | None = None
    updated_at_ns: int | None = None

    def __post_init__(self) -> None:
        root = self.root.resolve()
        working_directory = self.working_directory.resolve()
        if working_directory != root and root not in working_directory.parents:
            raise ValueError("Workspace working directory must be within its root.")
        identity = (self.id, self.name, self.name_source, self.created_at_ns, self.updated_at_ns)
        if any(value is not None for value in identity) and any(
            value is None for value in identity
        ):
            raise ValueError("Workspace identity metadata must be complete when initialized.")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "working_directory", working_directory)

    def initialize(
        self,
        *,
        busy_timeout_ms: int = constants.DEFAULT_TELEMETRY_SQLITE_BUSY_TIMEOUT_MS,
    ) -> Workspace:
        """Load or create this workspace's durable identity and name.

        Args:
            busy_timeout_ms (int): Maximum milliseconds to wait for a catalog database lock.

        Returns:
            Workspace: Workspace carrying its durable identity and name.

        Raises:
            ValueError: If the SQLite busy timeout is not positive.
        """
        if self.id is not None:
            return self
        if busy_timeout_ms <= 0:
            raise ValueError("SQLite busy timeout must be positive.")
        name = self.root.name or "Untitled workspace"
        name_source: WorkspaceNameSource = "directory" if self.root.name else "default"
        connection = self._connect_catalog(busy_timeout_ms)
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT workspace_id, name, name_source, created_at_ns, updated_at_ns "
                "FROM workspaces ORDER BY created_at_ns LIMIT 1"
            ).fetchone()
            if row is None:
                now = time.time_ns()
                row = (str(uuid4()), name, name_source, now, now)
                connection.execute(
                    "INSERT INTO workspaces("
                    "workspace_id, name, name_source, created_at_ns, updated_at_ns"
                    ") VALUES (?, ?, ?, ?, ?)",
                    row,
                )
            connection.commit()
            return replace(
                self,
                id=row[0],
                name=row[1],
                name_source=row[2],
                created_at_ns=row[3],
                updated_at_ns=row[4],
            )
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _connect_catalog(self, busy_timeout_ms: int) -> sqlite3.Connection:
        """Open the workspace catalog and ensure its schema exists."""
        path = self.storage.workspaces
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(constants.PRIVATE_DIRECTORY_MODE)
        connection = sqlite3.connect(path)
        path.chmod(constants.PRIVATE_FILE_MODE)
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
        return connection

    @classmethod
    def discover(cls, working_directory: Path | str) -> Workspace:
        """Discover the workspace containing an active directory.

        Args:
            working_directory (Path | str): Directory from which the application was started.

        Returns:
            Workspace: Discovered worktree and its local artifact locations.
        """
        directory = Path(working_directory).resolve()
        root = find_project_root(directory) or directory
        return cls(
            root=root,
            working_directory=directory,
            storage=WorkspaceStorage(root / constants.APP_DIRECTORY),
        )
