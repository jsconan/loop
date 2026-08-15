"""Session manager for handling user sessions."""

import json
from typing import Iterable

from ..interaction import ConsoleInteraction, Interaction
from ..models import (
    ContentArtifact,
    ContextReference,
    ConversationItem,
    Message,
    Response,
    ToolResult,
)
from ..utils import bound_tool_result, cached_metadata, register_cached_metadata
from .models import SESSION_NAME_SOURCE_GENERATED, SESSION_NAME_SOURCE_INITIAL, SessionNameGenerator
from .naming import initial_session_name
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
        for message in session.messages:
            if isinstance(message, ToolResult):
                for artifact in message.artifacts:
                    register_cached_metadata(
                        artifact.handle,
                        artifact.source,
                        artifact.reloadable,
                    )

    def new_session(self) -> None:
        """Replace the active session with a fresh unpersisted session."""
        self._session = Session()

    def rename_session(self, name: str) -> None:
        """Assign and persist a user-controlled name to the active session.

        Args:
            name (str): New non-empty session name.

        Raises:
            ValueError: If the name is empty after normalization.
        """
        self._session.rename(name)
        self._session_store.save(self._session)

    def generate_session_name(
        self,
        generator: SessionNameGenerator,
    ) -> None:
        """Generate a session name from the first completed exchange.

        Args:
            generator (SessionNameGenerator): Auxiliary title generation service.
        """
        user_message = next(
            (
                message.content
                for message in self._session.messages
                if isinstance(message, Message) and message.role == "user"
            ),
            "",
        )
        assistant_message = next(
            (
                message.content
                for message in self._session.messages
                if isinstance(message, Message) and message.role == "assistant"
            ),
            "",
        )
        if not user_message or not assistant_message.strip():
            return
        name = generator.generate(user_message, assistant_message, self._session.model)
        if name:
            self._session.rename(name, source=SESSION_NAME_SOURCE_GENERATED)
            self._session_store.save(self._session)

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

    def add_user_message(
        self,
        content: str,
        context: Iterable[ContextReference] = (),
    ) -> None:
        """Construct and persist one complete user message.

        Args:
            content (str): Submitted user-message text.
            context (Iterable[ContextReference]): Resolved context snapshots attached to the
                message. Defaults to no explicit context.
        """
        if self._session.name is None:
            self._session.rename(initial_session_name(content), source=SESSION_NAME_SOURCE_INITIAL)
        self.add_message(Message(role="user", content=content, context=tuple(context)))

    def add_tool_call(
        self,
        call_id: str,
        output: str,
        working_directory: str,
        active_skills: Iterable[tuple[str, str]],
    ) -> None:
        """Add a tool result and its instruction state to the session.

        Args:
            call_id (str): Identifier used to associate the call with its result.
            output (str): The content of the tool result.
            working_directory (str): Effective instruction directory after the tool call.
            active_skills (Iterable[tuple[str, str]]): Active skill names and canonical locations
                after the tool call.
        """
        self.update_instruction_state(working_directory, active_skills)
        output, handle = bound_tool_result(output, f"tool result {call_id}")
        if handle is not None:
            self._interaction.info(
                f"Tool result '{call_id}' exceeded the context limit and was cached as '{handle}'."
            )
        self.add_message(
            ToolResult(
                call_id=call_id,
                output=output,
                artifacts=self._content_artifacts(output),
            )
        )

    @staticmethod
    def _content_artifacts(output: str) -> tuple[ContentArtifact, ...]:
        """Return registered artifact metadata referenced by one serialized result."""
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            return ()
        if not isinstance(payload, dict) or not isinstance(payload.get("handle"), str):
            return ()
        handle = payload["handle"]
        metadata = cached_metadata(handle)
        if metadata is None:
            return ()
        return (
            ContentArtifact(
                handle=handle,
                source=metadata["source"],
                reloadable=metadata["reloadable"],
            ),
        )

    def update_instruction_state(
        self,
        working_directory: str,
        active_skills: Iterable[tuple[str, str]],
    ) -> None:
        """Update the instruction state associated with the session.

        Args:
            working_directory (str): Effective instruction directory.
            active_skills (Iterable[tuple[str, str]]): Active skill names and canonical locations.
        """
        self._session.update_instruction_state(working_directory, active_skills)

    def add_response(self, response: Response) -> None:
        """Add a response to the session.

        Args:
            response (Response): The response to add.
        """
        self.add_message(response)
