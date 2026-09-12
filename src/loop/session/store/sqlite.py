"""Persist session snapshots in a local SQLite database."""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from ...models import Message
from ...utils import content_digest, utc_now, validate_content_handle
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
        self.path.parent.chmod(0o700)
        temporary = self.path.with_name(f".{self.path.name}.migration.tmp")
        try:
            with (
                closing(sqlite3.connect(legacy)) as source_connection,
                closing(sqlite3.connect(temporary)) as destination_connection,
            ):
                source_connection.backup(destination_connection)
            temporary.chmod(0o600)
            temporary.replace(self.path)
            self._enforce_private_modes()
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
            ValueError: If an artifact is invalid or the session references missing content.
        """
        if session.workspace_id is None:
            session.workspace_id = self._workspace_id
        self._validate_workspace(session)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.parent.chmod(0o700)
        if session.name is None:
            session.name = initial_session_name()
            session.name_source = SESSION_NAME_SOURCE_INITIAL

        now = utc_now().isoformat()
        snapshot = session.persistence_snapshot()
        new_revision = session.revision + 1
        with closing(sqlite3.connect(self._path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            self._enforce_private_modes()
            with connection:
                self._create_schema(connection)
                for artifact in snapshot.reference_artifacts:
                    self._store_reference(
                        connection,
                        artifact.content,
                        handle=artifact.handle,
                        digest=artifact.digest,
                    )
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
                                snapshot.payload,
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
                                snapshot.payload,
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
                references = set()
                for message in session.messages:
                    if not isinstance(message, Message):
                        continue
                    for reference in message.context:
                        references.add((reference.version, reference.handle, reference.size_bytes))
                digests = {version for version, _, _ in references}
                for digest, handle, size_bytes in references:
                    stored = connection.execute(
                        "SELECT handle, size_bytes, content FROM reference_artifacts "
                        "WHERE digest = ?",
                        (digest,),
                    ).fetchone()
                    if stored is None:
                        raise ValueError("Session references an unavailable artifact.")
                    _, stored_size, stored_content = stored
                    if stored_size != size_bytes or content_digest(stored_content) != digest:
                        raise ValueError("Reference artifact failed integrity validation.")
                connection.execute(
                    "DELETE FROM session_reference_artifacts WHERE session_id = ?",
                    (session.id,),
                )
                connection.executemany(
                    """
                    INSERT INTO session_reference_artifacts (session_id, digest)
                    VALUES (?, ?)
                    """,
                    ((session.id, digest) for digest in digests),
                )
        self._enforce_private_modes()
        session.revision = new_revision
        return session.id

    def load_reference(self, digest: str) -> bytes | None:
        """Load immutable referenced content when available.

        Args:
            digest (str): Algorithm-qualified full content digest.

        Returns:
            bytes | None: Complete immutable content, or ``None``.
        """
        if not self._path.is_file():
            return None
        self._enforce_private_modes()
        with closing(sqlite3.connect(self._path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            self._create_schema(connection)
            row = connection.execute(
                "SELECT content FROM reference_artifacts WHERE digest = ?",
                (digest,),
            ).fetchone()
        return bytes(row[0]) if row is not None else None

    def store_reference(self, content: bytes, *, handle: str, digest: str) -> None:
        """Persist one verified immutable reference without changing a session snapshot.

        Args:
            content (bytes): Complete immutable reference content.
            handle (str): Opaque content-read capability associated with the reference.
            digest (str): Algorithm-qualified digest of ``content``.

        Raises:
            ValueError: If the supplied identity does not match the content or conflicts with an
                existing artifact.
        """
        try:
            validate_content_handle(handle)
        except ValueError as error:
            raise ValueError("Reference artifact failed integrity validation.") from error
        if content_digest(content) != digest:
            raise ValueError("Reference artifact failed integrity validation.")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.parent.chmod(0o700)
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            self._create_schema(connection)
            self._store_reference(connection, content, handle=handle, digest=digest)
        self._enforce_private_modes()

    def collect_unreferenced(self) -> int:
        """Delete artifacts not owned by any persisted session.

        Returns:
            int: Number of artifacts deleted.
        """
        if not self._path.is_file():
            return 0
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            self._create_schema(connection)
            cursor = connection.execute(
                """
                DELETE FROM reference_artifacts
                WHERE NOT EXISTS (
                    SELECT 1 FROM session_reference_artifacts
                    WHERE session_reference_artifacts.digest = reference_artifacts.digest
                )
                """
            )
            deleted = cursor.rowcount
        self._enforce_private_modes()
        return deleted

    @staticmethod
    def _store_reference(
        connection: sqlite3.Connection,
        content: bytes,
        *,
        handle: str,
        digest: str,
    ) -> None:
        """Insert one artifact and verify any pre-existing durable identity."""
        try:
            connection.execute(
                """
                INSERT INTO reference_artifacts (digest, handle, size_bytes, content)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(digest) DO NOTHING
                """,
                (digest, handle, len(content), content),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError("Reference artifact failed integrity validation.") from error
        stored = connection.execute(
            "SELECT handle, size_bytes, content FROM reference_artifacts WHERE digest = ?",
            (digest,),
        ).fetchone()
        if stored is None or stored[1:] != (len(content), content):
            raise ValueError("Reference artifact failed integrity validation.")

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

        self._enforce_private_modes()
        with closing(sqlite3.connect(self._path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
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
            needs_migration = Session.requires_migration(row[2])
            session = Session.deserialize(row[2])
            session.id = session_id
            session.name = row[0]
            session.name_source = row[1]
            session.revision = row[3]
        needs_workspace = session.workspace_id is None
        if needs_workspace:
            session.workspace_id = self._workspace_id
        self._validate_workspace(session)
        if needs_migration or needs_workspace:
            self.save(session)
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
        self._enforce_private_modes()
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

    def _enforce_private_modes(self) -> None:
        """Repair the database directory and every SQLite sidecar to owner-only access."""
        self._path.parent.chmod(0o700)
        for path in (
            self._path,
            self._path.with_name(f"{self._path.name}-wal"),
            self._path.with_name(f"{self._path.name}-shm"),
            self._path.with_name(f"{self._path.name}-journal"),
        ):
            if path.exists():
                path.chmod(0o600)

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
        with closing(
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reference_artifacts (
                    digest TEXT PRIMARY KEY,
                    handle TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                    content BLOB NOT NULL
                )
                """
            )
        ):
            pass
        with closing(
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_reference_artifacts (
                    session_id TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    PRIMARY KEY (session_id, digest),
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (digest) REFERENCES reference_artifacts(digest) ON DELETE RESTRICT
                )
                """
            )
        ):
            pass
        connection.execute(
            "CREATE INDEX IF NOT EXISTS session_reference_artifacts_digest_idx "
            "ON session_reference_artifacts(digest)"
        )
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
