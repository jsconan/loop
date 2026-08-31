"""Persist session snapshots in a local SQLite database."""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from ...models import Message
from ...utils import utc_now
from ..models import (
    SESSION_NAME_SOURCE_INITIAL,
    SessionInfo,
    SessionNotFoundError,
    SessionWorkspaceMismatchError,
)
from ..naming import initial_session_name
from ..session import Session


class SQLiteSessionStore:
    """Store complete session snapshots in SQLite.

    Args:
        path (Path | str): SQLite database path. Its parent is created on the first save.
        workspace_id (str): Durable identity of the workspace owning the database.

    Raises:
        ValueError: If the workspace identifier is empty.
    """

    _path: Path
    _workspace_id: str

    def __init__(self, path: Path | str, *, workspace_id: str) -> None:
        self._path = Path(path)
        if not workspace_id:
            raise ValueError("Workspace identifier must not be empty.")
        self._workspace_id = workspace_id

    @property
    def path(self) -> Path:
        """Return the configured database path.

        Returns:
            Path: SQLite database path.
        """
        return self._path

    def save(self, session: Session) -> str:
        """Persist a session snapshot atomically.

        Args:
            session (Session): Session to persist.

        Returns:
            str: The session's stable identifier.

        Raises:
            SessionWorkspaceMismatchError: If the session belongs to another workspace.
        """
        if session.workspace_id is None:
            session.workspace_id = self._workspace_id
        self._validate_workspace(session)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if session.name is None:
            session.name = initial_session_name()
            session.name_source = SESSION_NAME_SOURCE_INITIAL

        now = utc_now().isoformat()
        payload = session.serialize()
        with closing(sqlite3.connect(self._path)) as connection:  # noqa: SIM117
            with connection:
                self._create_schema(connection)
                connection.execute(
                    """
                    INSERT INTO sessions
                        (id, name, name_source, created_at, updated_at, message_count, session)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        name_source = excluded.name_source,
                        updated_at = excluded.updated_at,
                        message_count = excluded.message_count,
                        session = excluded.session
                    """,
                    (
                        session.id,
                        session.name,
                        session.name_source or SESSION_NAME_SOURCE_INITIAL,
                        now,
                        now,
                        len(session.messages),
                        payload,
                    ),
                )
        return session.id

    def load(self, session_id: str) -> Session:
        """Load a persisted session snapshot.

        Args:
            session_id (str): Identifier of the session to load.

        Returns:
            Session: Reconstructed session state.

        Raises:
            SessionNotFoundError: If the database or requested session does not exist.
            SessionWorkspaceMismatchError: If the session belongs to another workspace.
            UnsupportedConversationItemError: If a serialized conversation item type is not
                supported.
            ValueError: If the persisted session has an unsupported or invalid format.
        """
        if not self._path.is_file():
            raise SessionNotFoundError(f"Session '{session_id}' was not found.")

        with closing(sqlite3.connect(self._path)) as connection:
            self._create_schema(connection)
            row = connection.execute(
                "SELECT name, name_source, session FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise SessionNotFoundError(f"Session '{session_id}' was not found.")
            session = Session.deserialize(row[2])
            session.id = session_id
            session.name = row[0]
            session.name_source = row[1]
            if session.workspace_id is None:
                session.workspace_id = self._workspace_id
                connection.execute(
                    "UPDATE sessions SET session = ? WHERE id = ?",
                    (session.serialize(), session_id),
                )
                connection.commit()
            self._validate_workspace(session)
        return session

    def _validate_workspace(self, session: Session) -> None:
        """Reject a session not owned by this workspace's storage."""
        if session.workspace_id != self._workspace_id:
            raise SessionWorkspaceMismatchError(
                f"Session '{session.id}' belongs to workspace '{session.workspace_id}', "
                f"not '{self._workspace_id}'."
            )

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
                SELECT id, name, updated_at, message_count
                FROM sessions
                ORDER BY updated_at DESC, id DESC
                """
            ).fetchall()
        return [
            SessionInfo(
                id=row[0],
                name=row[1],
                updated_at=datetime.fromisoformat(row[2]),
                message_count=row[3],
            )
            for row in rows
        ]

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        """Create the session table when it does not exist."""
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                name_source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                message_count INTEGER NOT NULL,
                session TEXT NOT NULL
            )
            """
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()}
        if "name" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN name TEXT")
        if "name_source" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN name_source TEXT")
        rows = connection.execute(
            "SELECT id, session FROM sessions WHERE name IS NULL OR name_source IS NULL"
        ).fetchall()
        for session_id, payload in rows:
            session = Session.deserialize(payload)
            first_message = next(
                (
                    message.content
                    for message in session.messages
                    if isinstance(message, Message) and message.role == "user"
                ),
                "",
            )
            name = session.name or initial_session_name(first_message)
            connection.execute(
                "UPDATE sessions SET name = ?, name_source = ? WHERE id = ?",
                (name, session.name_source or SESSION_NAME_SOURCE_INITIAL, session_id),
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS sessions_name_idx ON sessions(name COLLATE NOCASE)"
        )
        connection.commit()
