"""Define the session persistence adapter contract."""

from typing import Protocol

from ..models import SessionInfo
from ..session import Session


class SessionStore(Protocol):
    """Persist and retrieve sessions by their identifier."""

    def save(self, session: Session) -> str:
        """Persist a session, creating an identifier when needed.

        Args:
            session (Session): Session to persist.

        Returns:
            str: Existing or newly assigned persistent identifier.
        """

    def load(self, session_id: str) -> Session:
        """Load a persisted session.

        Args:
            session_id (str): Identifier of the session to load.

        Returns:
            Session: Reconstructed session state.

        Raises:
            SessionNotFoundError: If the requested session does not exist.
            UnsupportedConversationItemError: If a serialized conversation item type is not
                supported.
            ValueError: If its persisted format is invalid or unsupported.
        """

    def list(self) -> list[SessionInfo]:
        """List persisted sessions from most to least recently updated.

        Returns:
            list[SessionInfo]: Lightweight persisted-session descriptions.
        """
