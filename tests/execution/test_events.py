"""Tests for sanitized execution lifecycle events."""

import pytest
from pydantic import ValidationError

from loop.execution.events import ExecutionEvent, ExecutionEventType


def test_execution_event_is_immutable_and_orders_sanitized_lifecycle_data():
    """Events retain the request sequence and reject invalid sequence values."""
    event = ExecutionEvent(
        request_id="request",
        sequence=0,
        type=ExecutionEventType.BACKEND_ATTESTED,
        detail="attested",
    )

    assert event.type is ExecutionEventType.BACKEND_ATTESTED
    with pytest.raises(ValidationError):
        ExecutionEvent(request_id="request", sequence=-1, type=ExecutionEventType.TERMINAL)
    with pytest.raises(ValidationError):
        event.detail = "changed"
