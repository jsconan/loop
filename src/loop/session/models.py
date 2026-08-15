"""Define passive session persistence models."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, TypedDict

from pydantic import BaseModel, Field

from .. import constants

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
    tokens: int
    model: str | None
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
