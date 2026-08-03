"""Define the session persistence contract."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..context import LoopContext


class SessionNotFoundError(ValueError):
    """Report that a requested persisted session does not exist."""


@dataclass(frozen=True)
class SessionInfo:
    """Describe a persisted session without loading its complete context.

    Args:
        id (str): Persistent session identifier.
        updated_at (datetime): Time of the latest persisted update.
        message_count (int): Number of conversation items in the session.
    """

    id: str
    updated_at: datetime
    message_count: int


class SessionStore(Protocol):
    """Persist and retrieve complete loop contexts by session identifier."""

    def save(self, session_id: str | None, context: LoopContext) -> str:
        """Persist a complete context, creating an identifier when needed.

        Args:
            session_id (str | None): Existing identifier, or ``None`` for a new session.
            context (LoopContext): Complete context snapshot to persist.

        Returns:
            str: Existing or newly assigned persistent identifier.
        """

    def load(self, session_id: str) -> LoopContext:
        """Load a complete persisted context.

        Args:
            session_id (str): Identifier of the session to load.

        Returns:
            LoopContext: Reconstructed context state.

        Raises:
            SessionNotFoundError: If the requested session does not exist.
            ValueError: If its persisted format is invalid or unsupported.
        """

    def list(self) -> list[SessionInfo]:
        """List persisted sessions from most to least recently updated.

        Returns:
            list[SessionInfo]: Lightweight persisted-session descriptions.
        """
