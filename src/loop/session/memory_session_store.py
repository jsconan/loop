"""Persist complete loop contexts in memory."""

from datetime import UTC, datetime
from uuid import uuid7

from ..context import LoopContext
from .base import Session, SessionInfo, SessionNotFoundError


class MemorySessionStore:
    """Store complete context snapshots in an instance-local list."""

    _sessions: list[Session]

    def __init__(self) -> None:
        self._sessions = []

    def save(self, session_id: str | None, context: LoopContext) -> str:
        """Persist a complete context snapshot in memory.

        Args:
            session_id (str | None): Existing identifier, or ``None`` for a new session.
            context (LoopContext): Complete context snapshot to persist.

        Returns:
            str: Existing or newly assigned persistent identifier.
        """
        session_id = session_id or str(uuid7())
        session = Session(
            id=session_id,
            updated_at=datetime.now(UTC),
            message_count=len(context.messages),
            context=context.serialize(),
        )
        for index, existing in enumerate(self._sessions):
            if existing.id == session_id:
                self._sessions[index] = session
                break
        else:
            self._sessions.append(session)
        return session_id

    def load(self, session_id: str) -> LoopContext:
        """Load a complete in-memory context snapshot.

        Args:
            session_id (str): Identifier of the session to load.

        Returns:
            LoopContext: Reconstructed context state.

        Raises:
            SessionNotFoundError: If the requested session does not exist.
            ValueError: If the persisted session has an unsupported or invalid format.
        """
        for session in self._sessions:
            if session.id == session_id:
                return LoopContext.deserialize(session.context)
        raise SessionNotFoundError(f"Session '{session_id}' was not found.")

    def list(self) -> list[SessionInfo]:
        """List in-memory sessions from most to least recently updated.

        Returns:
            list[SessionInfo]: Lightweight persisted-session descriptions.
        """
        sessions = sorted(self._sessions, key=lambda session: (session.updated_at, session.id))
        return [
            SessionInfo(
                id=session.id,
                updated_at=session.updated_at,
                message_count=session.message_count,
            )
            for session in reversed(sessions)
        ]
