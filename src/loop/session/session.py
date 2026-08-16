"""Define the session persistence contract."""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, Self

from pydantic import ValidationError

from ..models import (
    ConversationItem,
    Message,
    ModelContextItem,
    Reasoning,
    Response,
    ToolCall,
    ToolResult,
)
from .models import (
    SESSION_NAME_SOURCE_INITIAL,
    SESSION_NAME_SOURCE_USER,
    Compaction,
    SerializedMessage,
    SerializedSession,
    SessionInfo,
    SessionNameSource,
)
from .naming import initial_session_name, normalize_session_name, validate_session_source

_SCHEMA_VERSION = 4
_SUPPORTED_VERSIONS = (1, 2, 3, _SCHEMA_VERSION)
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
        name (str | None): Human-readable display name, or ``None`` before the first message.
        name_source (SessionNameSource | None): Origin controlling automatic replacement.
        messages (list[ConversationItem]): Conversation items.
            Defaults to an empty list.
        compactions (list[Compaction]): Durable replacement-context checkpoints.
            Defaults to an empty list.
        tokens (int): Total tokens in the context after the latest response.
            Defaults to ``0``.
        model (str | None): Model identifier reported by the latest response,
            or ``None`` when unknown. Defaults to ``None``.
        context_window (int | None): Context-window size for the latest selected model,
            or ``None`` when unknown. Defaults to ``None``.
        instruction_working_directory (str | None): Last effective instruction directory.
            Defaults to ``None``.
        active_skills (list[tuple[str, str]]): Active skill names and canonical locations.
            Defaults to an empty list.
    """

    id: str | None = None
    name: str | None = None
    name_source: SessionNameSource | None = None
    messages: list[ConversationItem] = field(default_factory=list)
    compactions: list[Compaction] = field(default_factory=list)
    tokens: int = 0
    model: str | None = None
    context_window: int | None = None
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

    def has_name(self) -> bool:
        """Return whether the session has a human-readable name.

        Returns:
            bool: ``True`` if the session has a name, ``False`` otherwise.
        """
        return self.name is not None and self.name_source is not None

    def has_initial_name(self) -> bool:
        """Return whether the session has a provisional name from the first user message.

        Returns:
            bool: ``True`` if the session name is provisional, ``False`` otherwise.
        """
        return self.name_source == SESSION_NAME_SOURCE_INITIAL

    def rename(self, name: str, source: SessionNameSource = SESSION_NAME_SOURCE_USER) -> None:
        """Replace the human-readable session name.

        Args:
            name (str): New non-empty name.
            source (SessionNameSource): Origin of the new name. Defaults to ``"user"``.

        Raises:
            ValueError: If the name is empty after normalization or the source is invalid.
        """
        normalized = normalize_session_name(name)
        if not normalized:
            raise ValueError("Session name cannot be empty.")
        validate_session_source(source)
        self.name = normalized
        self.name_source = source

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

    def add_compaction(self, compaction: Compaction) -> None:
        """Append a validated compaction checkpoint.

        Args:
            compaction (Compaction): Checkpoint covering a prefix of the complete history.

        Raises:
            ValueError: If the checkpoint is empty, exceeds history, or moves backward.
        """
        if not compaction.context:
            raise ValueError("Compaction context cannot be empty.")
        if compaction.boundary > len(self.messages):
            raise ValueError("Compaction boundary exceeds the complete history.")
        if self.compactions and compaction.boundary <= self.compactions[-1].boundary:
            raise ValueError("Compaction boundary must advance beyond the previous checkpoint.")
        self.compactions.append(compaction)

    def model_context(self) -> list[ModelContextItem]:
        """Return bounded context for the next model request.

        Returns:
            list[ModelContextItem]: Complete history, or the latest replacement context followed
                by items added after its boundary.
        """
        if not self.compactions:
            return self.messages
        checkpoint = self.compactions[-1]
        return [*checkpoint.context, *self.messages[checkpoint.boundary :]]

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
                name=self.name,
                name_source=self.name_source,
                messages=messages,
                compactions=[compaction.model_dump(mode="json") for compaction in self.compactions],
                tokens=self.tokens,
                model=self.model,
                context_window=self.context_window,
                instruction_working_directory=self.instruction_working_directory,
                active_skills=[list(identity) for identity in self.active_skills],
            ),
            separators=(",", ":"),
        )

    @classmethod
    def _deserialize_name(
        cls,
        version: int,
        payload: dict,
        messages: list[ConversationItem],
    ) -> tuple[str | None, SessionNameSource | None]:
        """Return validated name metadata from one supported snapshot."""
        if version < 3:
            first_user_message = next(
                (
                    message.content
                    for message in messages
                    if isinstance(message, Message) and message.role == "user"
                ),
                "",
            )
            name = initial_session_name(first_user_message) if first_user_message else None
            name_source = SESSION_NAME_SOURCE_INITIAL if name is not None else None
            return name, name_source
        name = payload["name"]
        name_source = payload["name_source"]
        if name is not None and (not isinstance(name, str) or not normalize_session_name(name)):
            raise TypeError("Invalid serialized session name.")
        validate_session_source(name_source, allow_none=True)
        return name, name_source

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
            context_window = payload["context_window"] if version >= 4 else None
            compactions = (
                [Compaction.model_validate(item) for item in payload["compactions"]]
                if version >= 4
                else []
            )
            if version == 1:
                instruction_working_directory = None
                active_skills = []
            else:
                instruction_working_directory = payload["instruction_working_directory"]
                active_skills = payload["active_skills"]
            name, name_source = cls._deserialize_name(version, payload, messages)

            if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
                raise TypeError("Invalid tokens count.")
            if model is not None and not isinstance(model, str):
                raise TypeError("Invalid serialized model name.")
            if context_window is not None and (
                not isinstance(context_window, int)
                or isinstance(context_window, bool)
                or context_window <= 0
            ):
                raise TypeError("Invalid serialized context window.")
            previous_boundary = -1
            for compaction in compactions:
                if not compaction.context or compaction.boundary > len(messages):
                    raise TypeError("Invalid serialized compaction.")
                if compaction.boundary <= previous_boundary:
                    raise TypeError("Invalid serialized compaction order.")
                previous_boundary = compaction.boundary
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
            name=name,
            name_source=name_source,
            messages=messages,
            compactions=compactions,
            tokens=tokens,
            model=model,
            context_window=context_window,
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
