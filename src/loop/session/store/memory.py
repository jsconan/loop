"""Persist session snapshots in memory."""

from datetime import UTC, datetime
from uuid import uuid7

from ..models import SessionInfo, StoredSession
from ..session import Session, SessionNotFoundError


class MemorySessionStore:
    """Store session snapshots in an instance-local list."""

    _sessions: list[StoredSession]

    def __init__(self) -> None:
        self._sessions = []

    def _find_session(self, session_id: str) -> StoredSession | None:
        return next(filter(lambda s: s["id"] == session_id, self._sessions), None)

    def save(self, session: Session) -> str:
        """Persist a session snapshot in memory.

        Args:
            session (Session): Session to persist.

        Returns:
            str: Existing or newly assigned persistent identifier.
        """
        if session.id is None:
            session.id = str(uuid7())
        now = datetime.now(UTC)

        stored_session = self._find_session(session.id)
        if stored_session:
            stored_session["updated_at"] = now
            stored_session["message_count"] = len(session.messages)
            stored_session["session"] = session.serialize()
        else:
            self._sessions.append(
                StoredSession(
                    id=session.id,
                    created_at=now,
                    updated_at=now,
                    message_count=len(session.messages),
                    session=session.serialize(),
                )
            )
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
        stored_session = self._find_session(session_id)
        if stored_session is None:
            raise SessionNotFoundError(f"Session '{session_id}' was not found.")
        session = Session.deserialize(stored_session["session"])
        session.id = session_id
        return session

    def list(self) -> list[SessionInfo]:
        """List in-memory sessions from most to least recently updated.

        Returns:
            list[SessionInfo]: Lightweight persisted-session descriptions.
        """
        sessions = sorted(
            self._sessions, key=lambda session: (session["updated_at"], session["id"])
        )
        return [
            SessionInfo(
                id=session["id"],
                updated_at=session["updated_at"],
                message_count=session["message_count"],
            )
            for session in reversed(sessions)
        ]
