"""Session manager for handling user sessions."""

from typing import Iterable

from ..interaction import ConsoleInteraction, Interaction
from ..models import (
    ConversationItem,
    Message,
    Response,
    ToolResult,
)
from .session import Session, SessionStore
from .store import MemorySessionStore


class SessionManager:
    """Manage user sessions, including persistence and interaction.

    Args:
        interaction (Interaction | None): Service used for user input and output. Defaults to a
            console interaction.
        session (Session | str | None): Active session or persisted session identifier to load.
            Defaults to a fresh session.
        session_store (SessionStore | None): Store used to persist and retrieve sessions. Defaults
            to an instance-local memory store.

    Raises:
        SessionNotFoundError: If the requested persisted session does not exist.
        UnsupportedConversationItemError: If a serialized conversation item type is unsupported.
        ValueError: If the session or its persisted format is invalid or unsupported.
    """

    _interaction: Interaction
    _session: Session
    _session_store: SessionStore

    def __init__(
        self,
        interaction: Interaction | None = None,
        session: Session | str | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self._interaction = interaction or ConsoleInteraction()
        self._session_store = session_store or MemorySessionStore()

        if session and isinstance(session, (str, Session)):
            self.load_session(session)
        else:
            self._session = Session()

    @property
    def session(self) -> Session:
        """Return the active session.

        Returns:
            Session: The mutable session used by the loop.
        """
        return self._session

    @property
    def interaction(self) -> Interaction:
        """Return the service used for user input and output.

        Returns:
            Interaction: The configured interaction service.
        """
        return self._interaction

    @property
    def store(self) -> SessionStore:
        """Return the session store used for persistence.

        Returns:
            SessionStore: The configured session store.
        """
        return self._session_store

    @property
    def messages(self) -> list[ConversationItem]:
        """Return the current conversation history.

        Returns:
            list[ConversationItem]: Items accumulated during the conversation.
        """
        return self._session.messages

    @property
    def model(self) -> str | None:
        """Return the model explicitly selected for requests.

        Returns:
            str | None: The selected model, or ``None`` when the backend default is used.
        """
        return self._session.model

    @model.setter
    def model(self, value: str | None) -> None:
        """Set the model explicitly selected for requests.

        Args:
            value (str | None): The model to use, or ``None`` to use the backend default.
        """
        self._session.model = value

    @property
    def tokens(self) -> int:
        """Return the number of tokens used in the current session.

        Returns:
            int: The number of tokens used in the current session.
        """
        return self._session.tokens

    def load_session(self, session: Session | str) -> None:
        """Load a session from the store.

        Args:
            session (Session | str): The session to load, or its identifier.

        Raises:
            SessionNotFoundError: If the requested session does not exist.
            UnsupportedConversationItemError: If a serialized conversation item type is not
                supported.
            ValueError: If its persisted format is invalid or unsupported.
        """
        if isinstance(session, str):
            session = self._session_store.load(session)
        if not isinstance(session, Session):
            raise ValueError("Invalid session type.")
        self._session = session

    def add_message(self, message: ConversationItem | Response) -> None:
        """Add conversation items and persist the resulting complete session.

        Args:
            message (ConversationItem | Response): Conversation item to add.

        Raises:
            ValueError: If the value is not a supported conversation item.
        """
        self._session.add_message(message)
        self._session_store.save(self._session)

    def add_messages(self, messages: Iterable[ConversationItem]) -> None:
        """Add messages to the conversation history.

        Args:
            messages (Iterable[ConversationItem]): Conversation items to add.

        Raises:
            ValueError: If any value is not a supported conversation item.
        """
        self._session.add_messages(messages)
        self._session_store.save(self._session)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the session.

        Args:
            content (str): The content of the user message.
        """
        self.add_message(Message(role="user", content=content))

    def add_tool_call(self, call_id: str, output: str) -> None:
        """Add a tool result to the session.

        Args:
            call_id (str): Identifier used to associate the call with its result.
            output (str): The content of the tool result.
        """
        self.add_message(ToolResult(call_id=call_id, output=output))

    def add_response(self, response: Response) -> None:
        """Add a response to the session.

        Args:
            response (Response): The response to add.
        """
        self.add_message(response)
