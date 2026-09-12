"""Persist session snapshots in memory."""

from threading import Lock

from ...models import Message
from ...utils import content_digest, utc_now, validate_content_handle
from ..models import (
    SESSION_NAME_SOURCE_INITIAL,
    SessionInfo,
    SessionNotFoundError,
    SessionRevisionConflictError,
    StoredSession,
)
from ..naming import initial_session_name
from ..session import Session


class MemorySessionStore:
    """Store session snapshots in an instance-local list."""

    _sessions: list[StoredSession]
    _lock: Lock
    _references: dict[str, bytes]
    _reference_handles: dict[str, str]
    _session_references: dict[str, set[str]]

    def __init__(self) -> None:
        self._sessions = []
        self._lock = Lock()
        self._references = {}
        self._reference_handles = {}
        self._session_references = {}

    def load_reference(self, digest: str) -> bytes | None:
        """Load immutable referenced content when available.

        Args:
            digest (str): Algorithm-qualified full content digest.

        Returns:
            bytes | None: Complete immutable content, or ``None``.
        """
        with self._lock:
            return self._references.get(digest)

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
        with self._lock:
            existing = self._reference_handles.setdefault(handle, digest)
            if existing != digest:
                raise ValueError("Reference artifact failed integrity validation.")
            self._references[digest] = content

    def collect_unreferenced(self) -> int:
        """Delete artifacts not owned by any persisted session.

        Returns:
            int: Number of artifacts deleted.
        """
        with self._lock:
            referenced = (
                set().union(*self._session_references.values())
                if self._session_references
                else set()
            )
            unused = self._references.keys() - referenced
            for digest in unused:
                del self._references[digest]
            return len(unused)

    def _find_session(self, session_id: str) -> StoredSession | None:
        return next(filter(lambda s: s["id"] == session_id, self._sessions), None)

    def save(self, session: Session) -> str:
        """Persist a session snapshot in memory.

        Args:
            session (Session): Session to persist.

        Returns:
            str: The session's stable identifier.

        Raises:
            SessionRevisionConflictError: If the snapshot is based on a stale revision.
            ValueError: If an artifact is invalid or the session references missing content.
        """
        if session.name is None:
            session.name = initial_session_name()
            session.name_source = SESSION_NAME_SOURCE_INITIAL
        now = utc_now()
        snapshot = session.persistence_snapshot()
        new_revision = session.revision + 1

        with self._lock:
            stored_session = self._find_session(session.id)
            current_revision = stored_session["revision"] if stored_session else 0
            if current_revision != session.revision:
                raise SessionRevisionConflictError(session.id, session.revision, current_revision)
            references = dict(self._references)
            handles = dict(self._reference_handles)
            for artifact in snapshot.reference_artifacts:
                existing_content = references.setdefault(artifact.digest, artifact.content)
                existing_digest = handles.setdefault(artifact.handle, artifact.digest)
                if existing_content != artifact.content or existing_digest != artifact.digest:
                    raise ValueError("Reference artifact failed integrity validation.")
            digests = set()
            for message in session.messages:
                if not isinstance(message, Message):
                    continue
                for reference in message.context:
                    stored = references.get(reference.version)
                    if stored is None:
                        raise ValueError("Session references an unavailable artifact.")
                    digests.add(reference.version)
            self._references = references
            self._reference_handles = handles
            if stored_session:
                stored_session["updated_at"] = now
                stored_session["message_count"] = len(session.messages)
                stored_session["name"] = session.name
                stored_session["name_source"] = session.name_source or SESSION_NAME_SOURCE_INITIAL
                stored_session["session"] = snapshot.payload
                stored_session["revision"] = new_revision
            else:
                self._sessions.append(
                    StoredSession(
                        id=session.id,
                        name=session.name,
                        name_source=session.name_source or SESSION_NAME_SOURCE_INITIAL,
                        created_at=now,
                        updated_at=now,
                        message_count=len(session.messages),
                        session=snapshot.payload,
                        revision=new_revision,
                    )
                )
            self._session_references[session.id] = digests
        session.revision = new_revision
        return session.id

    def load(self, session_id: str) -> Session:
        """Load a session snapshot from memory.

        Args:
            session_id (str): Identifier of the session to load.

        Returns:
            Session: Reconstructed session state.

        Raises:
            SessionNotFoundError: If the requested session does not exist.
            UnsupportedConversationItemError: If a serialized conversation item type is not
                supported.
            ValueError: If the persisted session has an unsupported or invalid format.
        """
        with self._lock:
            stored_session = self._find_session(session_id)
            if stored_session is None:
                raise SessionNotFoundError(f"Session '{session_id}' was not found.")
            payload = stored_session["session"]
            needs_migration = Session.requires_migration(payload)
            session = Session.deserialize(payload)
            session.id = session_id
            session.name = stored_session["name"]
            session.name_source = stored_session["name_source"]
            session.revision = stored_session["revision"]
        if needs_migration:
            self.save(session)
        return session

    def list(self) -> list[SessionInfo]:
        """List in-memory sessions from most to least recently updated.

        Returns:
            list[SessionInfo]: Lightweight persisted-session descriptions.
        """
        with self._lock:
            sessions = sorted(
                self._sessions, key=lambda session: (session["updated_at"], session["id"])
            )
        return [
            SessionInfo(
                id=session["id"],
                name=session["name"],
                updated_at=session["updated_at"],
                message_count=session["message_count"],
            )
            for session in reversed(sessions)
        ]
