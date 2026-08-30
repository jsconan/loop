"""Define the session persistence contract."""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Self
from uuid import uuid7

from pydantic import TypeAdapter, ValidationError

from ..models import (
    ConversationItem,
    Message,
    ModelAssignment,
    ModelContextItem,
    Reasoning,
    Response,
    ToolCall,
    ToolResult,
)
from ..utils import utc_now
from .models import (
    SESSION_NAME_SOURCE_INITIAL,
    SESSION_NAME_SOURCE_USER,
    Compaction,
    CompactionEvent,
    ConversationItemEvent,
    PendingToolCall,
    RunCompletedEvent,
    SerializedMessage,
    SerializedSession,
    SessionEvent,
    SessionNameSource,
    SessionRecoveryState,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    UnsupportedConversationItemError,
)
from .naming import initial_session_name, normalize_session_name, validate_session_source

_SCHEMA_VERSION = 8
_EVENT_ADAPTER = TypeAdapter(SessionEvent)
_ITEM_TYPES = {
    "message": Message,
    "reasoning": Reasoning,
    "tool_call": ToolCall,
    "tool_result": ToolResult,
}
_TYPE_NAMES = {model: name for name, model in _ITEM_TYPES.items()}


@dataclass
class Session:
    """Describe a living session.

    Args:
        id (str): Stable session identifier, generated when omitted.
        name (str | None): Human-readable display name, or ``None`` before the first message.
        name_source (SessionNameSource | None): Origin controlling automatic replacement.
        messages (list[ConversationItem]): Conversation items.
            Defaults to an empty list.
        compactions (list[Compaction]): Durable replacement-context checkpoints.
            Defaults to an empty list.
        tokens (int): Total tokens in the context after the latest response.
            Defaults to ``0``.
        model (str | None): Model identifier in the latest successfully used assignment,
            or ``None`` when unknown. Defaults to ``None``.
        context_window (int | None): Context-window size in the latest successfully used
            assignment, or ``None`` when unknown. Defaults to ``None``.
        instruction_working_directory (str | None): Last effective instruction directory.
            Defaults to ``None``.
        active_skills (list[tuple[str, str]]): Active skill names and canonical locations.
            Defaults to an empty list.
        events (list[SessionEvent]): Ordered durable replay and observability events.
            Defaults to an empty list.
    """

    id: str = field(default_factory=lambda: str(uuid7()))
    name: str | None = None
    name_source: SessionNameSource | None = None
    messages: list[ConversationItem] = field(default_factory=list)
    compactions: list[Compaction] = field(default_factory=list)
    tokens: int = 0
    model: str | None = None
    context_window: int | None = None
    instruction_working_directory: str | None = None
    active_skills: list[tuple[str, str]] = field(default_factory=list)
    events: list[SessionEvent] = field(default_factory=list)

    @property
    def assignment(self) -> ModelAssignment | None:
        """Return the durable model assignment for this session.

        Returns:
            ModelAssignment | None: Last resolved model and capacity, or ``None`` when unknown.
        """
        if self.model is None:
            return None
        return ModelAssignment(model=self.model, context_window=self.context_window)

    @assignment.setter
    def assignment(self, value: ModelAssignment | None) -> None:
        """Set the durable model assignment for this session.

        Args:
            value (ModelAssignment | None): Assignment to retain, or ``None`` to clear it.
        """
        self.model = value.model if value is not None else None
        self.context_window = value.context_window if value is not None else None

    def __post_init__(self) -> None:
        if self.events:
            return
        created_at = utc_now()
        checkpoints = {
            compaction.boundary: index for index, compaction in enumerate(self.compactions)
        }
        for index in range(len(self.messages) + 1):
            if index in checkpoints:
                self.events.append(
                    CompactionEvent(
                        id=str(uuid7()),
                        created_at=created_at,
                        compaction_index=checkpoints[index],
                    )
                )
            if index < len(self.messages):
                self.events.append(
                    ConversationItemEvent(
                        id=str(uuid7()),
                        created_at=created_at,
                        item_index=index,
                    )
                )

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
            validated = [self._get_message(item) for item in message.items]
            for item in validated:
                self.messages.append(item)
                if not isinstance(item, ToolCall):
                    self._add_conversation_event(len(self.messages) - 1)
            if message.usage.total_tokens is not None:
                self.tokens = message.usage.total_tokens
        else:
            self.add_messages((message,))

    def add_messages(self, messages: Iterable[ConversationItem]) -> None:
        """Add messages to the conversation history.

        Args:
            messages (Iterable[ConversationItem]): Conversation items to add.

        Raises:
            ValueError: If any value is not a supported conversation item.
        """
        validated = [self._get_message(message) for message in messages]
        for message in validated:
            self.messages.append(message)
            self._add_conversation_event(len(self.messages) - 1)

    def add_tool_call_event(self, call_id: str) -> None:
        """Place one persisted model tool call in the replay timeline.

        Args:
            call_id (str): Identifier of the canonical tool call to place.

        Raises:
            ValueError: If the call is unknown or already present in the timeline.
        """
        item_index = next(
            (
                index
                for index in range(len(self.messages) - 1, -1, -1)
                if isinstance(self.messages[index], ToolCall)
                and self.messages[index].call_id == call_id
            ),
            None,
        )
        recorded = {
            event.item_index for event in self.events if isinstance(event, ConversationItemEvent)
        }
        if item_index is None:
            raise ValueError(f"Unknown tool call '{call_id}'.")
        if item_index in recorded:
            raise ValueError(f"Tool call '{call_id}' is already in the session timeline.")
        self._add_conversation_event(item_index)

    def _add_conversation_event(self, item_index: int) -> None:
        """Append one canonical conversation item to the replay timeline."""
        self.events.append(
            ConversationItemEvent(
                id=str(uuid7()),
                created_at=utc_now(),
                item_index=item_index,
            )
        )

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
        self.events.append(
            CompactionEvent(
                id=str(uuid7()),
                created_at=utc_now(),
                compaction_index=len(self.compactions) - 1,
            )
        )

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

    def recovery_state(self) -> SessionRecoveryState | None:
        """Return the safest continuation for an interrupted agent run.

        Returns:
            SessionRecoveryState | None: Required continuation derived from durable history, or
                ``None`` when the latest run ended cleanly.
        """
        completed_items = 0
        for event in self.events:
            if isinstance(event, RunCompletedEvent):
                completed_items = event.metrics.item_count
        if completed_items >= len(self.messages):
            return None

        suffix = self.messages[completed_items:]
        calls = [item for item in suffix if isinstance(item, ToolCall)]
        results = {item.call_id for item in suffix if isinstance(item, ToolResult)}
        if not calls:
            if any(isinstance(item, Message) and item.role == "assistant" for item in suffix):
                return SessionRecoveryState(action="finalize_run")
            return SessionRecoveryState(action="query_model")

        presented_indices = {
            event.item_index for event in self.events if isinstance(event, ConversationItemEvent)
        }
        started = {
            event.call_id for event in self.events if isinstance(event, ToolExecutionStartedEvent)
        }
        pending = []
        for offset, item in enumerate(suffix):
            if not isinstance(item, ToolCall):
                continue
            call = item
            index = completed_items + offset
            if call.call_id in results:
                status = "result_available"
            elif call.call_id in started or index in presented_indices:
                status = "outcome_unknown"
            else:
                status = "not_started"
            pending.append(PendingToolCall(call=call, status=status))

        statuses = {item.status for item in pending}
        if "outcome_unknown" in statuses:
            action = "resolve_uncertain_tools"
        elif "not_started" in statuses:
            action = "execute_tools"
        elif isinstance(suffix[-1], (Message, Reasoning)):
            action = "finalize_run"
        else:
            action = "query_model"
        return SessionRecoveryState(action=action, pending_calls=tuple(pending))

    @staticmethod
    def _get_message(message: ConversationItem) -> ConversationItem:
        """Return a validated conversation item for storage."""
        if not isinstance(message, ConversationItem):
            raise ValueError(  # noqa: TRY004 - preserve the public deserialization contract.
                f"Expected a conversation item, got {type(message)}"
            )
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
                id=self.id,
                name=self.name,
                name_source=self.name_source,
                messages=messages,
                compactions=[compaction.model_dump(mode="json") for compaction in self.compactions],
                tokens=self.tokens,
                model=self.model,
                context_window=self.context_window,
                instruction_working_directory=self.instruction_working_directory,
                active_skills=[list(identity) for identity in self.active_skills],
                events=[event.model_dump(mode="json") for event in self.events],
            ),
            separators=(",", ":"),
        )

    @staticmethod
    def _validate_snapshot_metadata(
        *,
        id: object,
        name: object,
        name_source: object,
        tokens: object,
        model: object,
        context_window: object,
        instruction_working_directory: object,
        active_skills: object,
    ) -> None:
        """Validate scalar and instruction metadata from a current snapshot."""
        if not isinstance(id, str) or not id:
            raise TypeError("Invalid session identifier.")
        if name is not None and (not isinstance(name, str) or not normalize_session_name(name)):
            raise TypeError("Invalid serialized session name.")
        validate_session_source(name_source, allow_none=True)
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

    @staticmethod
    def _validate_timeline(
        messages: list[ConversationItem],
        compactions: list[Compaction],
        events: list[SessionEvent],
    ) -> None:
        """Validate canonical checkpoint and event references."""
        previous_boundary = -1
        for compaction in compactions:
            if not compaction.context or compaction.boundary > len(messages):
                raise TypeError("Invalid serialized compaction.")
            if compaction.boundary <= previous_boundary:
                raise TypeError("Invalid serialized compaction order.")
            previous_boundary = compaction.boundary
        for event in events:
            if isinstance(event, ConversationItemEvent) and event.item_index >= len(messages):
                raise TypeError("Invalid serialized conversation event.")
            if isinstance(event, CompactionEvent) and event.compaction_index >= len(compactions):
                raise TypeError("Invalid serialized compaction event.")
            if isinstance(
                event, (ToolExecutionStartedEvent, ToolExecutionCompletedEvent)
            ) and not any(
                isinstance(item, ToolCall) and item.call_id == event.call_id for item in messages
            ):
                raise TypeError("Invalid serialized tool execution event.")
        item_event_indices = [
            event.item_index for event in events if isinstance(event, ConversationItemEvent)
        ]
        compaction_event_indices = [
            event.compaction_index for event in events if isinstance(event, CompactionEvent)
        ]
        required_item_indices = {
            index for index, item in enumerate(messages) if not isinstance(item, ToolCall)
        }
        duplicate_items = len(item_event_indices) != len(set(item_event_indices))
        missing_items = not required_item_indices.issubset(item_event_indices)
        if duplicate_items or missing_items:
            raise TypeError("Invalid serialized conversation event coverage.")
        if sorted(compaction_event_indices) != list(range(len(compactions))):
            raise TypeError("Serialized compaction events do not cover every checkpoint once.")

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
            raise ValueError(  # noqa: TRY004 - all malformed snapshots share one error contract.
                "Invalid serialized session."
            )

        version = payload.get("version")
        if version in {1, 2, 3, 4, 5, 6, 7}:
            try:
                payload = cls._upcast_payload(payload)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Invalid serialized session.") from error
            version = payload["version"]
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
            context_window = payload["context_window"]
            compactions = [Compaction.model_validate(item) for item in payload["compactions"]]
            instruction_working_directory = payload["instruction_working_directory"]
            active_skills = payload["active_skills"]
            events = [_EVENT_ADAPTER.validate_python(item) for item in payload["events"]]
            id = payload["id"]
            name = payload["name"]
            name_source = payload["name_source"]
            cls._validate_snapshot_metadata(
                id=id,
                name=name,
                name_source=name_source,
                tokens=tokens,
                model=model,
                context_window=context_window,
                instruction_working_directory=instruction_working_directory,
                active_skills=active_skills,
            )
            cls._validate_timeline(messages, compactions, events)

        except (KeyError, TypeError, ValidationError) as error:
            raise ValueError("Invalid serialized session.") from error

        return cls(
            id=id,
            name=name,
            name_source=name_source,
            messages=messages,
            compactions=compactions,
            tokens=tokens,
            model=model,
            context_window=context_window,
            instruction_working_directory=instruction_working_directory,
            active_skills=[tuple(identity) for identity in active_skills],
            events=events,
        )

    @staticmethod
    def _upcast_payload(payload: dict) -> dict:
        """Return a current in-memory representation of one legacy snapshot."""
        value = dict(payload)
        version = value["version"]
        messages = value.get("messages", [])
        compactions = value.get("compactions", []) if version >= 4 else []
        if (
            not isinstance(messages, list)
            or not isinstance(compactions, list)
            or any(not isinstance(item, dict) for item in messages)
            or any(not isinstance(item, dict) for item in compactions)
        ):
            raise ValueError("Invalid serialized session.")
        if version in {5, 6}:
            stored_events = value.get("events")
            if not isinstance(stored_events, list) or any(
                not isinstance(event, dict) for event in stored_events
            ):
                raise ValueError("Invalid serialized session.")
            events = []
            for stored_event in stored_events:
                event = dict(stored_event)
                if version == 5 and event.get("type") == "run_completed":
                    model_duration = event.pop("model_duration_seconds", 0)
                    tool_duration = event.pop("tool_duration_seconds", 0)
                    elapsed_duration = event.pop("duration_seconds", None)
                    event["metrics"] = {
                        "active_duration_seconds": model_duration + tool_duration,
                        "elapsed_duration_seconds": elapsed_duration,
                        "model_duration_seconds": model_duration,
                        "tool_duration_seconds": tool_duration,
                        "model_calls": event.pop("model_calls", []),
                        "tools": event.pop("tools", []),
                        "message_count": event.pop("message_count", 0),
                        "item_count": event.pop("item_count", 0),
                        "usage": event.pop("usage", {}),
                        "model": event.pop("model", None),
                        "context_tokens": event.pop("context_tokens", None),
                        "context_window": event.pop("context_window", None),
                    }
                events.append(event)
        else:
            created_at = datetime.fromtimestamp(0, UTC)
            events = []
            checkpoints = {item.get("boundary"): index for index, item in enumerate(compactions)}
            for index in range(len(messages) + 1):
                if index in checkpoints:
                    events.append(
                        CompactionEvent(
                            id=str(uuid7()),
                            created_at=created_at,
                            compaction_index=checkpoints[index],
                        ).model_dump(mode="json")
                    )
                if index < len(messages):
                    events.append(
                        ConversationItemEvent(
                            id=str(uuid7()), created_at=created_at, item_index=index
                        ).model_dump(mode="json")
                    )
        first_user = next(
            (
                data.get("content", "")
                for item in messages
                if isinstance((data := item.get("data")), dict)
                if item.get("type") == "message" and data.get("role") == "user"
            ),
            "",
        )
        value.update(
            version=_SCHEMA_VERSION,
            id=value.get("id", str(uuid7())),
            name=value.get("name", initial_session_name(first_user) if first_user else None),
            name_source=value.get(
                "name_source", SESSION_NAME_SOURCE_INITIAL if first_user else None
            ),
            compactions=compactions,
            context_window=value.get("context_window"),
            instruction_working_directory=value.get("instruction_working_directory"),
            active_skills=value.get("active_skills", []),
            events=events,
        )
        return value
