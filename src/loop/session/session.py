"""Define the session persistence contract."""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from mimetypes import guess_type
from typing import Self

from pydantic import TypeAdapter, ValidationError

from ..models import (
    AgentRunStopReason,
    ConversationItem,
    Message,
    ModelAssignment,
    ModelContextItem,
    Reasoning,
    Response,
    ToolCall,
    ToolResult,
)
from ..utils import (
    content_digest,
    content_identity,
    encode_content_cursor,
    get_binary,
    new_uuid,
    utc_now,
    validate_content_handle,
)
from .models import (
    SESSION_NAME_SOURCE_INITIAL,
    SESSION_NAME_SOURCE_USER,
    Compaction,
    CompactionEvent,
    ConversationItemEvent,
    InputConsumedEvent,
    InputDeliveredEvent,
    InputPriorityChangedEvent,
    InputQueuedEvent,
    InputRecalledEvent,
    InputRejectedEvent,
    InteractionRequestedEvent,
    InteractionSettledEvent,
    PendingToolCall,
    ReferenceArtifact,
    RunCompletedEvent,
    RunFailedEvent,
    RunRetryStartedEvent,
    RunStartedEvent,
    SerializedMessage,
    SerializedSession,
    SessionEvent,
    SessionNameSource,
    SessionPersistenceSnapshot,
    SessionRecoveryState,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    UnsupportedConversationItemError,
)
from .naming import initial_session_name, normalize_session_name, validate_session_source

_SCHEMA_VERSION = 16
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
        workspace_id (str | None): Durable identity of the owning workspace. Defaults to ``None``
            for sessions constructed outside a workspace-aware manager.
        instruction_working_directory (str | None): Last effective instruction directory.
            Defaults to ``None``.
        active_skills (list[tuple[str, str]]): Active skill names and canonical locations.
            Defaults to an empty list.
        events (list[SessionEvent]): Ordered durable replay and observability events.
            Defaults to an empty list.
        revision (int): Store-managed persisted snapshot revision. New sessions start at zero.
    """

    id: str = field(default_factory=lambda: str(new_uuid()))
    name: str | None = None
    name_source: SessionNameSource | None = None
    messages: list[ConversationItem] = field(default_factory=list)
    compactions: list[Compaction] = field(default_factory=list)
    tokens: int = 0
    model: str | None = None
    context_window: int | None = None
    workspace_id: str | None = None
    instruction_working_directory: str | None = None
    active_skills: list[tuple[str, str]] = field(default_factory=list)
    events: list[SessionEvent] = field(default_factory=list)
    revision: int = field(default=0, compare=False, repr=False)

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
                        id=str(new_uuid()),
                        created_at=created_at,
                        compaction_index=checkpoints[index],
                    )
                )
            if index < len(self.messages):
                self.events.append(
                    ConversationItemEvent(
                        id=str(new_uuid()),
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
                id=str(new_uuid()),
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
                id=str(new_uuid()),
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
        last_completion = None
        for event in self.events:
            if isinstance(event, RunCompletedEvent):
                completed_items = event.metrics.item_count
                last_completion = event
        if completed_items >= len(self.messages):
            if self.active_run is not None or (
                last_completion is not None
                and last_completion.stop_reason == AgentRunStopReason.MAX_TURNS
            ):
                return SessionRecoveryState(action="query_model")
            return None

        suffix = self.messages[completed_items:]
        calls = [item for item in suffix if isinstance(item, ToolCall)]
        results = self._resolved_call_indices(suffix)
        if not calls:
            if any(isinstance(item, Message) and item.role == "assistant" for item in suffix):
                return SessionRecoveryState(action="finalize_run")
            return SessionRecoveryState(action="query_model")

        presented_indices = {
            event.item_index for event in self.events if isinstance(event, ConversationItemEvent)
        }
        started = self._unresolved_execution_ids()
        pending = []
        for offset, item in enumerate(suffix):
            if not isinstance(item, ToolCall):
                continue
            call = item
            index = completed_items + offset
            if offset in results:
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

    @property
    def retry_pending(self) -> bool:
        """Return whether the active run's latest recovery event is a recorded failure.

        Returns:
            bool: Whether ``/retry`` is allowed for the active durable run.
        """
        run_id = self.active_run_id
        if run_id is None:
            return False
        failed = False
        for event in self.events:
            if isinstance(event, (RunFailedEvent, RunRetryStartedEvent)) and event.run_id == run_id:
                failed = isinstance(event, RunFailedEvent)
        return failed

    def record_run_failure(self) -> None:
        """Record a recoverable failure for the current durable run.

        Raises:
            RuntimeError: If there is no active run or it already has a pending failure.
        """
        run_id = self.active_run_id
        if run_id is None or self.retry_pending:
            raise RuntimeError("There is no active run awaiting a failure record.")
        self.events.append(
            RunFailedEvent(
                id=str(new_uuid()),
                created_at=utc_now(),
                run_id=run_id,
            )
        )

    def begin_run_retry(self) -> None:
        """Record explicit retry authorization for the failed active run.

        Raises:
            RuntimeError: If the active run has no recorded failure to retry.
        """
        run_id = self.active_run_id
        if run_id is None or not self.retry_pending:
            raise RuntimeError("There is no failed run to retry.")
        self.events.append(
            RunRetryStartedEvent(
                id=str(new_uuid()),
                created_at=utc_now(),
                run_id=run_id,
            )
        )

    @property
    def active_run(self) -> RunStartedEvent | None:
        """Return the durable run that has not settled yet.

        Returns:
            RunStartedEvent | None: Active run metadata, or ``None`` when no run is pending.
        """
        active_run = None
        for event in self.events:
            if isinstance(event, RunStartedEvent):
                active_run = event
            elif isinstance(event, RunCompletedEvent) and (
                event.run_id is None
                or (active_run is not None and event.run_id == active_run.run_id)
            ):
                active_run = None
        return active_run

    @property
    def active_run_id(self) -> str | None:
        """Return the identity of the durable run that has not settled yet.

        Returns:
            str | None: Active run identity, or ``None`` when the latest run is complete.
        """
        active_run = self.active_run
        return active_run.run_id if active_run is not None else None

    @property
    def pending_inputs(self) -> tuple[InputQueuedEvent, ...]:
        """Return accepted inputs that have not reached a durable terminal state.

        Returns:
            tuple[InputQueuedEvent, ...]: Pending input records ordered by mailbox priority and
                admission sequence.
        """
        queued = {}
        settled = set()
        for event in self.events:
            if isinstance(event, InputQueuedEvent):
                queued[event.input_id] = event
            elif isinstance(event, InputPriorityChangedEvent):
                queued[event.input_id] = queued[event.input_id].model_copy(
                    update={"priority": event.priority, "sequence": event.sequence}
                )
            elif isinstance(event, (InputConsumedEvent, InputRejectedEvent, InputRecalledEvent)):
                settled.add(event.input_id)
        return tuple(
            sorted(
                (event for input_id, event in queued.items() if input_id not in settled),
                key=lambda event: (event.priority, event.sequence),
            )
        )

    def change_input_priority(self, input_id: str, priority: int) -> None:
        """Persist a changed priority for an input that is still waiting in the mailbox.

        Args:
            input_id (str): Identity of a pending accepted input.
            priority (int): Replacement mailbox priority.

        Returns:
            None: After the priority-change event is appended, or if the priority is unchanged.

        Raises:
            ValueError: If the input is unknown, delivered, consumed, or rejected.
        """
        pending = next((item for item in self.pending_inputs if item.input_id == input_id), None)
        delivered = any(
            isinstance(event, InputDeliveredEvent) and event.input_id == input_id
            for event in self.events
        )
        if pending is None or delivered:
            raise ValueError(f"Input '{input_id}' is not waiting in the mailbox.")
        if pending.priority == priority:
            return
        self.events.append(
            InputPriorityChangedEvent(
                id=str(new_uuid()),
                created_at=utc_now(),
                input_id=input_id,
                priority=priority,
                sequence=pending.sequence,
            )
        )

    def recall_input(self, input_id: str) -> None:
        """Close an input that is still waiting in the mailbox.

        Args:
            input_id (str): Identity of the accepted input to return to the editor.

        Raises:
            ValueError: If the input is unknown, delivered, or already settled.
        """
        pending = next((item for item in self.pending_inputs if item.input_id == input_id), None)
        delivered = any(
            isinstance(event, InputDeliveredEvent) and event.input_id == input_id
            for event in self.events
        )
        if pending is None or delivered:
            raise ValueError(f"Input '{input_id}' is not waiting in the mailbox.")
        self.events.append(
            InputRecalledEvent(id=str(new_uuid()), created_at=utc_now(), input_id=input_id)
        )

    @property
    def pending_interactions(self) -> tuple[InteractionRequestedEvent, ...]:
        """Return interaction requests that have not been answered or revoked.

        Returns:
            tuple[InteractionRequestedEvent, ...]: Unsettled requests in durable event order.
        """
        requests = {}
        settled = set()
        for event in self.events:
            if isinstance(event, InteractionRequestedEvent):
                requests[event.request_id] = event
            elif isinstance(event, InteractionSettledEvent):
                settled.add(event.request_id)
        return tuple(request for key, request in requests.items() if key not in settled)

    def mark_input_delivered(self, input_id: str) -> None:
        """Record delivery without consuming the durable input.

        Args:
            input_id (str): Identity of a previously accepted input.

        Raises:
            ValueError: If the input is unknown or already consumed or rejected.
        """
        if not any(item.input_id == input_id for item in self.pending_inputs):
            raise ValueError(f"Input '{input_id}' is not pending.")
        self.events.append(
            InputDeliveredEvent(id=str(new_uuid()), created_at=utc_now(), input_id=input_id)
        )

    def consume_input(self, input_id: str, item_index: int) -> None:
        """Atomically bind a pending input to its canonical user message.

        Args:
            input_id (str): Identity of a previously accepted input.
            item_index (int): Canonical message index created by consuming the input.

        Raises:
            ValueError: If the input is unknown, settled, or does not match the message.
        """
        pending = next((item for item in self.pending_inputs if item.input_id == input_id), None)
        if pending is None:
            raise ValueError(f"Input '{input_id}' is not pending.")
        if item_index >= len(self.messages):
            raise ValueError(f"Input '{input_id}' has no canonical message.")
        message = self.messages[item_index]
        if not isinstance(message, Message) or message.role != "user":
            raise ValueError(f"Input '{input_id}' is not bound to a user message.")
        if message.content != pending.content:
            raise ValueError(f"Input '{input_id}' does not match its canonical message.")
        self.events.append(
            InputConsumedEvent(
                id=str(new_uuid()),
                created_at=utc_now(),
                input_id=input_id,
                item_index=item_index,
            )
        )

    def reject_input(self, input_id: str) -> None:
        """Close a pending input that cannot be admitted or resolved.

        Args:
            input_id (str): Identity of a previously accepted input.

        Raises:
            ValueError: If the input is unknown or already settled.
        """
        if not any(item.input_id == input_id for item in self.pending_inputs):
            raise ValueError(f"Input '{input_id}' is not pending.")
        self.events.append(
            InputRejectedEvent(id=str(new_uuid()), created_at=utc_now(), input_id=input_id)
        )

    def request_interaction(self, event: InteractionRequestedEvent) -> None:
        """Persist a correlated interaction request without storing its arguments or answer.

        Args:
            event (InteractionRequestedEvent): Safe request identity and lifecycle metadata.

        Raises:
            ValueError: If the request belongs to another session or its ID was used before.
        """
        if event.session_id != self.id:
            raise ValueError("Interaction request belongs to another session.")
        if any(request.request_id == event.request_id for request in self.pending_interactions):
            raise ValueError(f"Interaction request '{event.request_id}' is already pending.")
        if any(
            isinstance(current, InteractionRequestedEvent)
            and current.request_id == event.request_id
            for current in self.events
        ):
            raise ValueError(f"Interaction request '{event.request_id}' was already recorded.")
        self.events.append(event)

    def settle_interaction(self, request_id: str, outcome: str) -> None:
        """Close one pending interaction without retaining its response value.

        Args:
            request_id (str): Identity of the pending request.
            outcome (str): Answered, revoked, or failed lifecycle outcome.

        Raises:
            ValueError: If the request is unknown, already settled, or has an invalid outcome.
        """
        if outcome not in {"answered", "revoked", "failed"}:
            raise ValueError("Invalid interaction outcome.")
        if not any(request.request_id == request_id for request in self.pending_interactions):
            raise ValueError(f"Interaction request '{request_id}' is not pending.")
        self.events.append(
            InteractionSettledEvent(
                id=str(new_uuid()),
                created_at=utc_now(),
                request_id=request_id,
                outcome=outcome,
            )
        )

    @staticmethod
    def _resolved_call_indices(items: list[ConversationItem]) -> set[int]:
        """Match each result to its preceding call occurrence, allowing reused provider IDs."""
        results = set()
        unmatched_results = {}
        for offset in range(len(items) - 1, -1, -1):
            item = items[offset]
            if isinstance(item, ToolResult):
                unmatched_results[item.call_id] = unmatched_results.get(item.call_id, 0) + 1
            elif isinstance(item, ToolCall) and unmatched_results.get(item.call_id, 0):
                results.add(offset)
                unmatched_results[item.call_id] -= 1
        return results

    def _unresolved_execution_ids(self) -> set[str]:
        """Keep execution markers only until their tool or run is durably resolved."""
        started = set()
        for event in self.events:
            if isinstance(event, ToolExecutionStartedEvent):
                started.add(event.call_id)
            elif isinstance(event, ToolExecutionCompletedEvent):
                started.discard(event.call_id)
            elif isinstance(event, RunCompletedEvent):
                started.clear()
        return started

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
            ValueError: If transient reference content conflicts with its durable identity.
        """
        return self.persistence_snapshot().payload

    def persistence_snapshot(self) -> SessionPersistenceSnapshot:
        """Project the session payload and immutable artifacts for one atomic save.

        Returns:
            SessionPersistenceSnapshot: JSON payload and deduplicated reference artifacts.

        Raises:
            UnsupportedConversationItemError: If a conversation item type is not supported.
            ValueError: If reference occurrence metadata or transient content conflicts with its
                durable identity.
        """
        try:
            self._validate_reference_metadata(self.messages)
        except TypeError as error:
            raise ValueError(str(error)) from error

        messages = []
        artifacts = {}
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
            if not isinstance(message, Message):
                continue
            for reference in message.context:
                if not reference.content and reference.size_bytes != 0:
                    continue
                content = get_binary(reference.content)
                digest = content_digest(content)
                if reference.version != digest or reference.size_bytes != len(content):
                    raise ValueError(
                        f"Referenced snapshot '{reference.path}' failed integrity validation."
                    )
                artifacts.setdefault(
                    digest,
                    ReferenceArtifact(handle=reference.handle, digest=digest, content=content),
                )

        payload = json.dumps(
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
                workspace_id=self.workspace_id,
                instruction_working_directory=self.instruction_working_directory,
                active_skills=[list(identity) for identity in self.active_skills],
                events=[event.model_dump(mode="json") for event in self.events],
            ),
            separators=(",", ":"),
        )
        return SessionPersistenceSnapshot(
            payload=payload,
            reference_artifacts=tuple(artifacts.values()),
        )

    @staticmethod
    def requires_migration(value: str) -> bool:
        """Return whether a serialized session needs normalization before use.

        Args:
            value (str): JSON representation of a session.

        Returns:
            bool: Whether the representation uses a supported legacy schema.

        Raises:
            ValueError: If the serialized session is invalid or uses an unsupported version.
        """
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("Invalid serialized session.") from error
        if not isinstance(payload, dict):
            raise ValueError(  # noqa: TRY004 - malformed snapshots share one public contract.
                "Invalid serialized session."
            )
        version = payload.get("version")
        if version == _SCHEMA_VERSION:
            return False
        if version in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
            return True
        raise ValueError(f"Unsupported session version {version}.")

    @staticmethod
    def _validate_snapshot_metadata(
        *,
        session_id: object,
        name: object,
        name_source: object,
        tokens: object,
        model: object,
        context_window: object,
        workspace_id: object,
        instruction_working_directory: object,
        active_skills: object,
    ) -> None:
        """Validate scalar and instruction metadata from a current snapshot."""
        if not isinstance(session_id, str) or not session_id:
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
        if workspace_id is not None and (not isinstance(workspace_id, str) or not workspace_id):
            raise TypeError("Invalid serialized workspace identifier.")
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
        session_id: str,
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
        queued_inputs = {}
        settled_inputs = set()
        delivered_inputs = set()
        requested_interactions = set()
        settled_interactions = set()
        active_run_id = None
        started_runs = set()
        failed_run_id = None
        for event in events:
            if isinstance(event, InputQueuedEvent):
                if event.session_id != session_id or event.input_id in queued_inputs:
                    raise TypeError("Invalid serialized input admission.")
                queued_inputs[event.input_id] = event
            elif isinstance(event, InputPriorityChangedEvent):
                current = queued_inputs.get(event.input_id)
                if (
                    current is None
                    or event.input_id in settled_inputs
                    or event.input_id in delivered_inputs
                    or event.sequence != current.sequence
                ):
                    raise TypeError("Invalid serialized input priority change.")
                queued_inputs[event.input_id] = current.model_copy(
                    update={"priority": event.priority, "sequence": event.sequence}
                )
            elif isinstance(event, InputDeliveredEvent):
                if event.input_id not in queued_inputs or event.input_id in settled_inputs:
                    raise TypeError("Invalid serialized input delivery.")
                delivered_inputs.add(event.input_id)
            elif isinstance(event, InputConsumedEvent):
                if event.input_id not in queued_inputs or event.input_id in settled_inputs:
                    raise TypeError("Invalid serialized input consumption.")
                if event.item_index >= len(messages):
                    raise TypeError("Invalid serialized input message reference.")
                message = messages[event.item_index]
                if (
                    not isinstance(message, Message)
                    or message.role != "user"
                    or message.content != queued_inputs[event.input_id].content
                ):
                    raise TypeError("Invalid serialized input message binding.")
                settled_inputs.add(event.input_id)
            elif isinstance(event, InputRejectedEvent):
                if event.input_id not in queued_inputs or event.input_id in settled_inputs:
                    raise TypeError("Invalid serialized input rejection.")
                settled_inputs.add(event.input_id)
            elif isinstance(event, InputRecalledEvent):
                if (
                    event.input_id not in queued_inputs
                    or event.input_id in settled_inputs
                    or event.input_id in delivered_inputs
                ):
                    raise TypeError("Invalid serialized input recall.")
                settled_inputs.add(event.input_id)
            elif isinstance(event, InteractionRequestedEvent):
                if event.session_id != session_id or event.request_id in requested_interactions:
                    raise TypeError("Invalid serialized interaction request.")
                requested_interactions.add(event.request_id)
            elif isinstance(event, InteractionSettledEvent):
                if (
                    event.request_id not in requested_interactions
                    or event.request_id in settled_interactions
                ):
                    raise TypeError("Invalid serialized interaction settlement.")
                settled_interactions.add(event.request_id)
            elif isinstance(event, RunStartedEvent):
                if active_run_id is not None or event.run_id in started_runs:
                    raise TypeError("Invalid serialized run start.")
                active_run_id = event.run_id
                started_runs.add(event.run_id)
                failed_run_id = None
            elif isinstance(event, RunFailedEvent):
                if event.run_id != active_run_id or failed_run_id is not None:
                    raise TypeError("Invalid serialized run failure.")
                failed_run_id = event.run_id
            elif isinstance(event, RunRetryStartedEvent):
                if event.run_id != active_run_id or failed_run_id != event.run_id:
                    raise TypeError("Invalid serialized run retry.")
                failed_run_id = None
            elif isinstance(event, RunCompletedEvent):
                if (
                    event.run_id is not None and event.run_id != active_run_id
                ) or failed_run_id is not None:
                    raise TypeError("Invalid serialized run completion.")
                active_run_id = None
                failed_run_id = None
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
        legacy_reasoning = version in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        if version in {4, 5, 6, 7, 8, 9, 10, 11}:
            try:
                payload = cls._externalize_legacy_references(payload)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Invalid serialized session.") from error
        if version == 11:
            payload["version"] = _SCHEMA_VERSION
            version = _SCHEMA_VERSION
        elif version in {12, 13, 14}:
            payload = dict(payload)
            payload["version"] = _SCHEMA_VERSION
            version = _SCHEMA_VERSION
        elif version in {9, 10}:
            payload = dict(payload)
            if version == 9:
                payload.pop("workspace_root", None)
                payload["workspace_id"] = None
            payload["version"] = _SCHEMA_VERSION
            version = payload["version"]
        elif version in {1, 2, 3, 4, 5, 6, 7, 8}:
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
                data = item["data"]
                if legacy_reasoning and item_type == "reasoning" and "summary" not in data:
                    data = dict(data)
                    data.update(content="", summary=data.get("content", ""))
                messages.append(model.model_validate(data))

            tokens = payload["tokens"]
            model = payload["model"]
            context_window = payload["context_window"]
            workspace_id = payload["workspace_id"]
            compactions = [Compaction.model_validate(item) for item in payload["compactions"]]
            instruction_working_directory = payload["instruction_working_directory"]
            active_skills = payload["active_skills"]
            events = [_EVENT_ADAPTER.validate_python(item) for item in payload["events"]]
            session_id = payload["id"]
            name = payload["name"]
            name_source = payload["name_source"]
            cls._validate_snapshot_metadata(
                session_id=session_id,
                name=name,
                name_source=name_source,
                tokens=tokens,
                model=model,
                context_window=context_window,
                workspace_id=workspace_id,
                instruction_working_directory=instruction_working_directory,
                active_skills=active_skills,
            )
            cls._validate_reference_metadata(messages)
            cls._validate_timeline(messages, compactions, events, session_id)

        except (KeyError, TypeError, ValidationError) as error:
            raise ValueError("Invalid serialized session.") from error

        return cls(
            id=session_id,
            name=name,
            name_source=name_source,
            messages=messages,
            compactions=compactions,
            tokens=tokens,
            model=model,
            context_window=context_window,
            workspace_id=workspace_id,
            instruction_working_directory=instruction_working_directory,
            active_skills=[tuple(identity) for identity in active_skills],
            events=events,
        )

    @staticmethod
    def _validate_reference_metadata(
        messages: Iterable[ConversationItem],
    ) -> None:
        """Reject reference occurrences whose metadata cannot describe their payload."""
        handles = {}
        for message in messages:
            if not isinstance(message, Message):
                continue
            for reference in message.context:
                if not 0 <= reference.included_bytes <= reference.size_bytes:
                    raise TypeError("Invalid serialized reference included bytes.")
                if reference.truncated != (reference.included_bytes < reference.size_bytes):
                    raise TypeError("Invalid serialized reference truncation metadata.")
                if reference.handle is None or reference.version is None:
                    raise TypeError("Invalid serialized reference artifact identity.")
                validate_content_handle(reference.handle)
                existing = handles.setdefault(reference.handle, reference.version)
                if existing != reference.version:
                    raise TypeError("Invalid serialized reference artifact handle binding.")
                expected_cursor = (
                    encode_content_cursor(reference.handle, reference.included_bytes)
                    if reference.truncated
                    else None
                )
                if reference.next_cursor != expected_cursor:
                    raise TypeError("Invalid serialized reference continuation cursor.")

    @staticmethod
    def _externalize_legacy_references(payload: dict) -> dict:
        """Convert legacy inline snapshots into canonical contentless occurrences."""
        value = dict(payload)
        messages = value.get("messages")
        if not isinstance(messages, list):
            raise TypeError
        normalized_messages = []
        for item in messages:
            if not isinstance(item, dict):
                raise TypeError
            copied_item = dict(item)
            if copied_item.get("type") != "message":
                normalized_messages.append(copied_item)
                continue
            data = copied_item.get("data")
            if not isinstance(data, dict):
                raise TypeError
            copied_data = dict(data)
            references = copied_data.get("context", [])
            if not isinstance(references, list):
                raise TypeError
            normalized_references = []
            for candidate in references:
                if not isinstance(candidate, dict):
                    raise TypeError
                reference = dict(candidate)
                preview = reference.pop("content")
                complete = reference.pop("snapshot_content", None)
                truncated = reference["truncated"]
                size_bytes = reference["size_bytes"]
                included_bytes = reference["included_bytes"]
                path = reference["path"]
                if not truncated and complete is None:
                    complete = preview
                if (
                    not isinstance(preview, str)
                    or not isinstance(complete, str)
                    or not isinstance(truncated, bool)
                    or not isinstance(path, str)
                    or not isinstance(size_bytes, int)
                    or isinstance(size_bytes, bool)
                    or not isinstance(included_bytes, int)
                    or isinstance(included_bytes, bool)
                ):
                    raise TypeError
                encoded = complete.encode("utf-8")
                preview_bytes = preview.encode("utf-8")
                if (
                    len(encoded) != size_bytes
                    or len(preview_bytes) != included_bytes
                    or not encoded.startswith(preview_bytes)
                    or truncated != (included_bytes < size_bytes)
                ):
                    raise ValueError("Invalid shipped reference snapshot metadata.")
                handle, digest = content_identity(encoded)
                reference.update(
                    content=complete,
                    handle=handle,
                    next_cursor=(
                        encode_content_cursor(handle, included_bytes) if truncated else None
                    ),
                    version=digest,
                    media_type=guess_type(path)[0] or "text/plain",
                )
                normalized_references.append(reference)
            copied_data["context"] = normalized_references
            copied_item["data"] = copied_data
            normalized_messages.append(copied_item)
        value["messages"] = normalized_messages
        return value

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
        elif version == 8:
            events = value.get("events")
            if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
                raise ValueError("Invalid serialized session.")
        else:
            created_at = datetime.fromtimestamp(0, UTC)
            events = []
            checkpoints = {item.get("boundary"): index for index, item in enumerate(compactions)}
            for index in range(len(messages) + 1):
                if index in checkpoints:
                    events.append(
                        CompactionEvent(
                            id=str(new_uuid()),
                            created_at=created_at,
                            compaction_index=checkpoints[index],
                        ).model_dump(mode="json")
                    )
                if index < len(messages):
                    events.append(
                        ConversationItemEvent(
                            id=str(new_uuid()), created_at=created_at, item_index=index
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
            id=value.get("id", str(new_uuid())),
            name=value.get("name", initial_session_name(first_user) if first_user else None),
            name_source=value.get(
                "name_source", SESSION_NAME_SOURCE_INITIAL if first_user else None
            ),
            compactions=compactions,
            context_window=value.get("context_window"),
            workspace_id=None,
            instruction_working_directory=value.get("instruction_working_directory"),
            active_skills=value.get("active_skills", []),
            events=events,
        )
        return value
