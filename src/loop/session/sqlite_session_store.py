"""Persist complete loop contexts in a local SQLite database."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid7

from ..context import LoopContext
from .base import SessionInfo, SessionNotFoundError


class SQLiteSessionStore:
    """Store complete context snapshots in SQLite.

    Args:
        path (Path | str): SQLite database path. Its parent is created on the first save.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Return the configured database path.

        Returns:
            Path: SQLite database path.
        """
        return self._path

    def save(self, session_id: str | None, context: LoopContext) -> str:
        """Persist a complete context snapshot atomically.

        Args:
            session_id (str | None): Existing identifier, or ``None`` for a new session.
            context (LoopContext): Complete context snapshot to persist.

        Returns:
            str: Existing or newly assigned persistent identifier.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        session_id = session_id or str(uuid7())
        now = datetime.now(UTC).isoformat()
        payload = context.serialize()
        with closing(sqlite3.connect(self._path)) as connection:
            with connection:
                self._create_schema(connection)
                connection.execute(
                    """
                    INSERT INTO sessions (id, created_at, updated_at, message_count, context)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        message_count = excluded.message_count,
                        context = excluded.context
                    """,
                    (session_id, now, now, len(context.messages), payload),
                )
        return session_id

    def load(self, session_id: str) -> LoopContext:
        """Load a complete persisted context.

        Args:
            session_id (str): Identifier of the session to load.

        Returns:
            LoopContext: Reconstructed context state.

        Raises:
            SessionNotFoundError: If the database or requested session does not exist.
            ValueError: If the persisted session has an unsupported or invalid format.
        """
        if not self._path.is_file():
            raise SessionNotFoundError(f"Session '{session_id}' was not found.")
        with closing(sqlite3.connect(self._path)) as connection:
            self._create_schema(connection)
            row = connection.execute(
                "SELECT context FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"Session '{session_id}' was not found.")
        return LoopContext.deserialize(row[0])

    def list(self) -> list[SessionInfo]:
        """List persisted sessions from most to least recently updated.

        Returns:
            list[SessionInfo]: Lightweight persisted-session descriptions.
        """
        if not self._path.is_file():
            return []
        with closing(sqlite3.connect(self._path)) as connection:
            self._create_schema(connection)
            rows = connection.execute(
                """
                SELECT id, updated_at, message_count
                FROM sessions
                ORDER BY updated_at DESC, id DESC
                """
            ).fetchall()
        return [
            SessionInfo(id=row[0], updated_at=datetime.fromisoformat(row[1]), message_count=row[2])
            for row in rows
        ]

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        """Create the session table when it does not exist."""
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                message_count INTEGER NOT NULL,
                context TEXT NOT NULL
            )
            """
        )
