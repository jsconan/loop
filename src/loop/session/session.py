"""Define the session persistence contract."""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, Self

from pydantic import ValidationError

from ..models import ConversationItem, Message, Reasoning, Response, ToolCall, ToolResult
from .models import SerializedMessage, SerializedSession, SessionInfo

_SCHEMA_VERSION = 2
_SUPPORTED_VERSIONS = (1, _SCHEMA_VERSION)
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
        instruction_working_directory (str | None): Last effective instruction directory.
            Defaults to ``None``.
        active_skills (list[tuple[str, str]]): Active skill names and canonical locations.
            Defaults to an empty list.
    """

    id: str | None = None
    messages: list[ConversationItem] = field(default_factory=list)
    tokens: int = 0
    model: str | None = None
    instruction_working_directory: str | None = None
    active_skills: list[tuple[str, str]] = field(default_factory=list)

    def update_instruction_state(
        self,
        working_directory: str,
        active_skills: Iterable[tuple[str, str]],
    ) -> None:
        """Update the effective instruction state.

        Args:
            working_directory (str): Effective instruction directory.
            active_skills (Iterable[tuple[str, str]]): Active skill names and canonical locations.
        """
        self.instruction_working_directory = working_directory
        self.active_skills = list(active_skills)

    def add_message(self, message: ConversationItem | Response) -> None:
        """Add one message to the conversation history.

        Args:
            message (ConversationItem | Response): Conversation item to add.

        Raises:
            ValueError: If the value is not a supported conversation item.
        """
        if isinstance(message, Response):
            self.add_messages(message.items)
            if message.usage.total_tokens is not None:
                self.tokens = message.usage.total_tokens
            if isinstance(message.model, str):
                self.model = message.model
        else:
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
                instruction_working_directory=self.instruction_working_directory,
                active_skills=[list(identity) for identity in self.active_skills],
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
        if version not in _SUPPORTED_VERSIONS:
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
            if version == 1:
                instruction_working_directory = None
                active_skills = []
            else:
                instruction_working_directory = payload["instruction_working_directory"]
                active_skills = payload["active_skills"]

            if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
                raise TypeError("Invalid tokens count.")
            if model is not None and not isinstance(model, str):
                raise TypeError("Invalid serialized model name.")
            if instruction_working_directory is not None and not isinstance(
                instruction_working_directory, str
            ):
                raise TypeError("Invalid serialized instruction working directory.")
            if not isinstance(active_skills, list) or any(
                not isinstance(identity, list)
                or len(identity) != 2
                or not all(isinstance(value, str) for value in identity)
                for identity in active_skills
            ):
                raise TypeError("Invalid serialized active skills.")

        except (KeyError, TypeError, ValidationError) as error:
            raise ValueError("Invalid serialized session.") from error

        return cls(
            messages=messages,
            tokens=tokens,
            model=model,
            instruction_working_directory=instruction_working_directory,
            active_skills=[tuple(identity) for identity in active_skills],
        )


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
