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
    SessionRevisionConflictError,
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
        self._path = Path(path).resolve()
        if not workspace_id:
            raise ValueError("Workspace identifier must not be empty.")
        self._workspace_id = workspace_id

    def import_legacy(self, source: Path | str) -> bool:
        """Import a legacy database when this store does not yet exist.

        Args:
            source (Path | str): Legacy session database copied through SQLite backup.

        Returns:
            bool: Whether the legacy database was imported.
        """
        legacy = Path(source).resolve()
        if not legacy.is_file() or self.path.exists():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.migration.tmp")
        try:
            with (
                closing(sqlite3.connect(legacy)) as source_connection,
                closing(sqlite3.connect(temporary)) as destination_connection,
            ):
                source_connection.backup(destination_connection)
            temporary.chmod(0o600)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return True

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
            SessionRevisionConflictError: If the snapshot is based on a stale revision.
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
        new_revision = session.revision + 1
        with closing(sqlite3.connect(self._path)) as connection:  # noqa: SIM117
            with connection:
                self._create_schema(connection)
                if session.revision == 0:
                    with closing(
                        connection.execute(
                            """
                            INSERT INTO sessions
                                (id, name, name_source, created_at, updated_at,
                                 message_count, session, revision)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(id) DO NOTHING
                            """,
                            (
                                session.id,
                                session.name,
                                session.name_source or SESSION_NAME_SOURCE_INITIAL,
                                now,
                                now,
                                len(session.messages),
                                payload,
                                new_revision,
                            ),
                        )
                    ) as cursor:
                        inserted = cursor.rowcount
                    if inserted != 1:
                        current_revision = self._current_revision(connection, session.id)
                        raise SessionRevisionConflictError(
                            session.id,
                            session.revision,
                            current_revision if current_revision is not None else 0,
                        )
                else:
                    with closing(
                        connection.execute(
                            """
                            UPDATE sessions SET
                                name = ?, name_source = ?, updated_at = ?, message_count = ?,
                                session = ?, revision = ?
                            WHERE id = ? AND revision = ?
                            """,
                            (
                                session.name,
                                session.name_source or SESSION_NAME_SOURCE_INITIAL,
                                now,
                                len(session.messages),
                                payload,
                                new_revision,
                                session.id,
                                session.revision,
                            ),
                        )
                    ) as cursor:
                        updated = cursor.rowcount
                    if updated != 1:
                        current_revision = self._current_revision(connection, session.id)
                        raise SessionRevisionConflictError(
                            session.id,
                            session.revision,
                            current_revision if current_revision is not None else 0,
                        )
        session.revision = new_revision
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
            with closing(
                connection.execute(
                    "SELECT name, name_source, session, revision FROM sessions WHERE id = ?",
                    (session_id,),
                )
            ) as cursor:
                row = cursor.fetchone()
            if row is None:
                raise SessionNotFoundError(f"Session '{session_id}' was not found.")
            session = Session.deserialize(row[2])
            session.id = session_id
            session.name = row[0]
            session.name_source = row[1]
            session.revision = row[3]
            if session.workspace_id is None:
                session.workspace_id = self._workspace_id
                with closing(
                    connection.execute(
                        """
                        UPDATE sessions SET session = ?, revision = revision + 1
                        WHERE id = ? AND revision = ?
                        """,
                        (session.serialize(), session_id, session.revision),
                    )
                ) as cursor:
                    migrated = cursor.rowcount
                if migrated != 1:
                    current_revision = self._current_revision(connection, session_id)
                    raise SessionRevisionConflictError(
                        session_id,
                        session.revision,
                        current_revision if current_revision is not None else 0,
                    )
                connection.commit()
                session.revision += 1
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
            with closing(
                connection.execute(
                    """
                    SELECT id, name, updated_at, message_count
                    FROM sessions
                    ORDER BY updated_at DESC, id DESC
                    """
                )
            ) as cursor:
                rows = cursor.fetchall()
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
        with closing(connection.execute("BEGIN IMMEDIATE")):
            pass
        with closing(
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    name_source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    session TEXT NOT NULL,
                    revision INTEGER NOT NULL
                )
                """
            )
        ):
            pass
        with closing(connection.execute("PRAGMA table_info(sessions)")) as cursor:
            columns = {row[1] for row in cursor.fetchall()}
        if "name" not in columns:
            with closing(connection.execute("ALTER TABLE sessions ADD COLUMN name TEXT")):
                pass
        if "name_source" not in columns:
            with closing(connection.execute("ALTER TABLE sessions ADD COLUMN name_source TEXT")):
                pass
        if "revision" not in columns:
            with closing(
                connection.execute(
                    "ALTER TABLE sessions ADD COLUMN revision INTEGER NOT NULL DEFAULT 1"
                )
            ):
                pass
        with closing(
            connection.execute(
                "SELECT id, session FROM sessions WHERE name IS NULL OR name_source IS NULL"
            )
        ) as cursor:
            rows = cursor.fetchall()
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
            with closing(
                connection.execute(
                    "UPDATE sessions SET name = ?, name_source = ? WHERE id = ?",
                    (name, session.name_source or SESSION_NAME_SOURCE_INITIAL, session_id),
                )
            ):
                pass
        with closing(
            connection.execute(
                "CREATE INDEX IF NOT EXISTS sessions_name_idx ON sessions(name COLLATE NOCASE)"
            )
        ):
            pass
        connection.commit()

    @staticmethod
    def _current_revision(connection: sqlite3.Connection, session_id: str) -> int | None:
        """Return the current persisted revision without opening another connection."""
        with closing(
            connection.execute("SELECT revision FROM sessions WHERE id = ?", (session_id,))
        ) as cursor:
            row = cursor.fetchone()
        return row[0] if row is not None else None
