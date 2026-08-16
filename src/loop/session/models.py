"""Define passive session persistence models."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, TypedDict

from pydantic import BaseModel, Field

from .. import constants
from ..models import CompactionContextItem

SessionNameSource = Literal["initial", "generated", "user"]

SESSION_NAME_SOURCE_INITIAL: SessionNameSource = "initial"
SESSION_NAME_SOURCE_GENERATED: SessionNameSource = "generated"
SESSION_NAME_SOURCE_USER: SessionNameSource = "user"
SESSION_NAME_SOURCES: set[SessionNameSource] = {
    SESSION_NAME_SOURCE_INITIAL,
    SESSION_NAME_SOURCE_GENERATED,
    SESSION_NAME_SOURCE_USER,
}


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


class StoredSession(TypedDict):
    """Define the JSON format for a persisted session snapshot."""

    id: str
    name: str
    name_source: SessionNameSource
    created_at: datetime
    updated_at: datetime
    message_count: int
    session: str
