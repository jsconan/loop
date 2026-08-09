"""Define passive session persistence models."""

from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict


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
    """Define the JSON format for a serialized conversation item."""

    type: str
    data: dict


class SerializedSession(TypedDict):
    """Define the JSON format for a persisted session."""

    version: int
    messages: list[SerializedMessage]
    tokens: int
    model: str | None
    instruction_working_directory: str | None
    active_skills: list[list[str]]


class StoredSession(TypedDict):
    """Define the JSON format for a persisted session snapshot."""

    id: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    session: str
