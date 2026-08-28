"""Define passive session persistence models."""

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Protocol, TypedDict

from pydantic import BaseModel, Field, field_validator

from .. import constants
from ..models import AgentRunStopReason, CompactionContextItem, RunMetrics, ToolCall
from ..permissions import AuthorizationResult
from ..utils import as_utc

SessionNameSource = Literal["initial", "generated", "user"]
SESSION_NAME_SOURCE_INITIAL: SessionNameSource = "initial"
SESSION_NAME_SOURCE_GENERATED: SessionNameSource = "generated"
SESSION_NAME_SOURCE_USER: SessionNameSource = "user"
SESSION_NAME_SOURCES: set[SessionNameSource] = {
    SESSION_NAME_SOURCE_INITIAL,
    SESSION_NAME_SOURCE_GENERATED,
    SESSION_NAME_SOURCE_USER,
}


class SessionNotFoundError(ValueError):
    """Report that a requested persisted session does not exist."""


class UnsupportedConversationItemError(ValueError):
    """Report an unsupported conversation item type in a serialized context."""


class GeneratedSessionName(BaseModel):
    """Validate structured output from the auxiliary title request."""

    title: str = Field(
        description=(f"Concise session title of at most {constants.SESSION_TITLE_MAX_WORDS} words.")
    )


class InstructionSnapshot(BaseModel):
    """Preserve the effective instruction state at a compaction boundary.

    Args:
        working_directory (str): Effective instruction-discovery directory.
        content (str | None): Complete instructions supplied to the backend.
        digest (str): SHA-256 digest of the complete instruction content.
        active_skills (tuple[tuple[str, str], ...]): Active skill names and canonical locations.
    """

    working_directory: str
    content: str | None
    digest: str
    active_skills: tuple[tuple[str, str], ...] = ()


class Compaction(BaseModel):
    """Record one durable replacement-context checkpoint.

    Args:
        id (str): Stable compaction identifier.
        boundary (int): Exclusive full-history index represented by the checkpoint.
        created_at (datetime): Time at which the checkpoint was created.
        provider (str): Backend namespace that produced the replacement items.
        model (str): Model used to create the checkpoint.
        context (tuple[CompactionContextItem, ...]): Exact replacement context.
        instructions (InstructionSnapshot): Instruction state used for compaction.
        input_tokens_before (int | None): Last known context usage before compaction.
        input_tokens_after (int | None): Context usage reported by compaction.
    """

    id: str
    boundary: int = Field(ge=0)
    created_at: datetime
    provider: str
    model: str
    context: tuple[CompactionContextItem, ...]
    instructions: InstructionSnapshot
    input_tokens_before: int | None = Field(default=None, ge=0)
    input_tokens_after: int | None = Field(default=None, ge=0)

    _normalize_created_at = field_validator("created_at")(as_utc)


class SessionEventModel(BaseModel):
    """Provide identity and time shared by durable session events.

    Args:
        id (str): Stable event identifier.
        created_at (datetime): UTC time at which the event occurred.
    """

    id: str
    created_at: datetime

    _normalize_created_at = field_validator("created_at")(as_utc)


class ConversationItemEvent(SessionEventModel):
    """Place one canonical conversation item in the replay timeline.

    Args:
        type (Literal["conversation_item"]): Event discriminator.
        item_index (int): Index into the session's canonical conversation items.
    """

    type: Literal["conversation_item"] = "conversation_item"
    item_index: int = Field(ge=0)


class PermissionEvent(SessionEventModel):
    """Record one atomic operation authorization.

    Args:
        type (Literal["permission"]): Event discriminator.
        result (AuthorizationResult): Policy, approval, and effective outcome.
    """

    type: Literal["permission"] = "permission"
    result: AuthorizationResult


class RunCompletedEvent(SessionEventModel):
    """Record aggregate and operation-level statistics for one agent run.

    Args:
        type (Literal["run_completed"]): Event discriminator.
        stop_reason (AgentRunStopReason): Reason the run returned control.
        started_at (datetime): UTC time at which execution began.
        metrics (RunMetrics): Complete durable statistics and current context occupancy.
    """

    type: Literal["run_completed"] = "run_completed"
    stop_reason: AgentRunStopReason
    started_at: datetime
    metrics: RunMetrics

    _normalize_started_at = field_validator("started_at")(as_utc)


class ToolExecutionStartedEvent(SessionEventModel):
    """Record that one local tool invocation crossed its execution boundary.

    Args:
        type (Literal["tool_execution_started"]): Event discriminator.
        call_id (str): Identifier of the model tool call being executed.
    """

    type: Literal["tool_execution_started"] = "tool_execution_started"
    call_id: str


class ToolExecutionCompletedEvent(SessionEventModel):
    """Record one durably resolved local tool request.

    Args:
        type (Literal["tool_execution_completed"]): Event discriminator.
        call_id (str): Identifier of the resolved model tool call.
        succeeded (bool): Whether the serialized result reports success.
        duration_seconds (float): Tool-function execution duration in seconds.
    """

    type: Literal["tool_execution_completed"] = "tool_execution_completed"
    call_id: str
    succeeded: bool
    duration_seconds: float = Field(ge=0)


class CompactionEvent(SessionEventModel):
    """Place one compaction checkpoint in the replay timeline.

    Args:
        type (Literal["compaction"]): Event discriminator.
        compaction_index (int): Index into the session's canonical compactions.
    """

    type: Literal["compaction"] = "compaction"
    compaction_index: int = Field(ge=0)


SessionEvent = Annotated[
    ConversationItemEvent
    | PermissionEvent
    | RunCompletedEvent
    | ToolExecutionStartedEvent
    | ToolExecutionCompletedEvent
    | CompactionEvent,
    Field(discriminator="type"),
]


ToolRecoveryStatus = Literal["not_started", "outcome_unknown", "result_available"]
SessionRecoveryAction = Literal[
    "query_model", "execute_tools", "resolve_uncertain_tools", "finalize_run"
]


@dataclass(frozen=True)
class PendingToolCall:
    """Describe the durable recovery status of one model tool request.

    Args:
        call (ToolCall): Canonical model tool request.
        status (ToolRecoveryStatus): Safest conclusion supported by durable history.
    """

    call: ToolCall
    status: ToolRecoveryStatus


@dataclass(frozen=True)
class SessionRecoveryState:
    """Describe how an interrupted session can safely continue.

    Args:
        action (SessionRecoveryAction): Next durable agent-loop boundary to enter.
        pending_calls (tuple[PendingToolCall, ...]): Tool requests belonging to the unfinished
            exchange, in model order.
    """

    action: SessionRecoveryAction
    pending_calls: tuple[PendingToolCall, ...] = ()


class SessionNameGenerator(Protocol):
    """Generate an improved name from the first conversation exchange."""

    def generate(self, user_message: str, assistant_message: str, model: str | None) -> str | None:
        """Generate a session name for one exchange.

        Args:
            user_message (str): First user message.
            assistant_message (str): First completed assistant answer.
            model (str | None): Conversation model, or ``None`` to use the backend default.

        Returns:
            str | None: Generated name, or ``None`` when no valid name was produced.
        """


@dataclass(frozen=True)
class SessionInfo:
    """Describe a persisted session without loading its complete context.

    Args:
        id (str): Persistent session identifier.
        name (str): Human-readable session name.
        updated_at (datetime): Time of the latest persisted update.
        message_count (int): Number of conversation items in the session.
    """

    id: str
    name: str
    updated_at: datetime
    message_count: int


class SerializedMessage(TypedDict):
    """Define the JSON format for a serialized conversation item."""

    type: str
    data: dict


class SerializedSession(TypedDict):
    """Define the JSON format for a persisted session."""

    version: int
    name: str | None
    name_source: SessionNameSource | None
    messages: list[SerializedMessage]
    compactions: list[dict]
    tokens: int
    model: str | None
    context_window: int | None
    instruction_working_directory: str | None
    active_skills: list[list[str]]
    events: list[dict]


class StoredSession(TypedDict):
    """Define the JSON format for a persisted session snapshot."""

    id: str
    name: str
    name_source: SessionNameSource
    created_at: datetime
    updated_at: datetime
    message_count: int
    session: str
