"""Define sanitized, ordered execution lifecycle events."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ExecutionEventType(StrEnum):
    """Identify a lifecycle event emitted by the execution boundary."""

    REQUEST_NORMALIZED = "request_normalized"
    BACKEND_ATTESTED = "backend_attested"
    PROCESS_STARTED = "process_started"
    TERMINAL = "terminal"


class ExecutionEvent(BaseModel):
    """Record one sanitized and monotonically ordered lifecycle event.

    Args:
        request_id (str): Owning execution request identifier.
        sequence (int): Monotonic sequence within the request.
        type (ExecutionEventType): Event category.
        detail (str | None): Sanitized model-safe event detail.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    type: ExecutionEventType
    detail: str | None = None
