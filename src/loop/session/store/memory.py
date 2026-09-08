"""Persist session snapshots in memory."""

from threading import Lock

from ...utils import utc_now
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

    def __init__(self) -> None:
        self._sessions = []
        self._lock = Lock()

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
        """
        if session.name is None:
            session.name = initial_session_name()
            session.name_source = SESSION_NAME_SOURCE_INITIAL
        now = utc_now()
        payload = session.serialize()
        new_revision = session.revision + 1

        with self._lock:
            stored_session = self._find_session(session.id)
            if stored_session:
                if stored_session["revision"] != session.revision:
                    raise SessionRevisionConflictError(
                        session.id, session.revision, stored_session["revision"]
                    )
                stored_session["updated_at"] = now
                stored_session["message_count"] = len(session.messages)
                stored_session["name"] = session.name
                stored_session["name_source"] = session.name_source or SESSION_NAME_SOURCE_INITIAL
                stored_session["session"] = payload
                stored_session["revision"] = new_revision
            else:
                if session.revision != 0:
                    raise SessionRevisionConflictError(session.id, session.revision, 0)
                self._sessions.append(
                    StoredSession(
                        id=session.id,
                        name=session.name,
                        name_source=session.name_source or SESSION_NAME_SOURCE_INITIAL,
                        created_at=now,
                        updated_at=now,
                        message_count=len(session.messages),
                        session=payload,
                        revision=new_revision,
                    )
                )
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
            session = Session.deserialize(stored_session["session"])
            session.id = session_id
            session.name = stored_session["name"]
            session.name_source = stored_session["name_source"]
            session.revision = stored_session["revision"]
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
