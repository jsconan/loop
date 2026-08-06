"""Define the session persistence contract."""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Self, TypedDict

from pydantic import ValidationError

from ..models import ConversationItem, Message, Reasoning, ToolCall, ToolResult

_SCHEMA_VERSION = 1
_ITEM_TYPES = {
    "message": Message,
    "reasoning": Reasoning,
    "tool_call": ToolCall,
    "tool_result": ToolResult,
}
_TYPE_NAMES = {model: name for name, model in _ITEM_TYPES.items()}


class SessionNotFoundError(ValueError):
    """Report that a requested persisted session does not exist."""


class UnsupportedConversationItemError(ValueError):
    """Report an unsupported conversation item type in a serialized context."""


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


class SerializedMessage(TypedDict):
    """Define the JSON format for a serialized conversation item.

    Args:
        type (str): Type of the conversation item.
        data (dict): Serialized conversation item data.
    """

    type: str
    data: dict


class SerializedSession(TypedDict):
    """Define the JSON format for a persisted session.

    Args:
        version (int): Version of the serialized session format.
        messages (list[SerializedMessage]): Serialized conversation items.
        tokens (int): Total tokens in the context after the latest response.
        model (str | None): Model identifier reported by the latest response,
            or ``None`` when unknown.
    """

    version: int
    messages: list[SerializedMessage]
    tokens: int
    model: str | None


class StoredSession(TypedDict):
    """Define the JSON format for a persisted session snapshot.

    Args:
        id (str): Persistent session identifier.
        created_at (datetime): Time of the initial persisted creation.
        updated_at (datetime): Time of the latest persisted update.
        message_count (int): Number of conversation items in the session.
        session (str): Serialized session snapshot.
    """

    id: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    session: str


@dataclass
class Session:
    """Describe a living session.

    Args:
        id (str): Persistent session identifier.
        messages (list[ConversationItem]): Conversation items.
            Defaults to an empty list.
        tokens (int): Total tokens in the context after the latest response.
            Defaults to ``0``.
        model (str | None): Model identifier reported by the latest response,
            or ``None`` when unknown. Defaults to ``None``.
    """

    id: str | None = None
    messages: list[ConversationItem] = field(default_factory=list)
    tokens: int = 0
    model: str | None = None

    def add_message(self, message: ConversationItem) -> None:
        """Add one message to the conversation history.

        Args:
            message (ConversationItem): Conversation item to add.

        Raises:
            ValueError: If the value is not a supported conversation item.
        """
        self.messages.append(self._get_message(message))

    def add_messages(self, messages: Iterable[ConversationItem]) -> None:
        """Add messages to the conversation history.

        Args:
            messages (Iterable[ConversationItem]): Conversation items to add.

        Raises:
            ValueError: If any value is not a supported conversation item.
        """
        self.messages.extend([self._get_message(message) for message in messages])

    @staticmethod
    def _get_message(message: ConversationItem) -> ConversationItem:
        """Return a validated conversation item for storage."""
        if not isinstance(message, ConversationItem):
            raise ValueError(f"Expected a conversation item, got {type(message)}")
        return message

    def serialize(self) -> str:
        """Serialize the session into its versioned JSON format.

        Returns:
            str: Compact JSON representation of the session.

        Raises:
            UnsupportedConversationItemError: If a conversation item type is not supported.
        """
        messages = []
        for message in self.messages:
            item_type = type(message)
            if item_type not in _TYPE_NAMES:
                raise UnsupportedConversationItemError(
                    f"Unsupported conversation item type: {item_type.__name__}."
                )
            messages.append(
                SerializedMessage(
                    type=_TYPE_NAMES[item_type],
                    data=message.model_dump(mode="json"),
                )
            )

        return json.dumps(
            SerializedSession(
                version=_SCHEMA_VERSION,
                messages=messages,
                tokens=self.tokens,
                model=self.model,
            ),
            separators=(",", ":"),
        )

    @classmethod
    def deserialize(cls, value: str) -> Self:
        """Deserialize and validate a session from its versioned JSON format.

        Args:
            value (str): JSON representation of a complete session.

        Returns:
            Session: Reconstructed session state.

        Raises:
            UnsupportedConversationItemError: If a serialized conversation item type is not
                supported.
            ValueError: If the serialized session is invalid or uses an unsupported version.
        """
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("Invalid serialized session.") from error

        if not isinstance(payload, dict):
            raise ValueError("Invalid serialized session.")

        version = payload.get("version")
        if version != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported session version {version}.")

        try:
            messages = []
            for item in payload["messages"]:
                item_type = item["type"]
                if item_type not in _ITEM_TYPES:
                    raise UnsupportedConversationItemError(
                        f"Unsupported conversation item type: {item_type!r}."
                    )
                model = _ITEM_TYPES[item_type]
                messages.append(model.model_validate(item["data"]))

            tokens = payload["tokens"]
            model = payload["model"]

            if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
                raise TypeError("Invalid tokens count.")
            if model is not None and not isinstance(model, str):
                raise TypeError("Invalid serialized model name.")

        except (KeyError, TypeError, ValidationError) as error:
            raise ValueError("Invalid serialized session.") from error

        return cls(messages=messages, tokens=tokens, model=model)


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
            ValueError: If its persisted format is invalid or unsupported.
        """

    def list(self) -> list[SessionInfo]:
        """List persisted sessions from most to least recently updated.

        Returns:
            list[SessionInfo]: Lightweight persisted-session descriptions.
        """
