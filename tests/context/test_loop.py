"""Tests for conversation context state."""

import pytest

from loop import Message, ToolCall
from loop.context import LoopContext


def function_call() -> ToolCall:
    """Build a completed function-tool call."""
    return ToolCall(
        call_id="call_123",
        name="get_current_datetime",
        arguments="{}",
        id="fc_123",
    )


def test_adds_one_or_multiple_messages():
    """Context methods retain one or multiple local conversation items."""
    context = LoopContext()
    user_message = Message(role="user", content="hello")
    assistant_message = Message(role="assistant", content="answer")
    call = function_call()

    context.add_message(user_message)
    context.add_message(call)
    context.add_messages(message for message in (assistant_message, call))

    assert context.messages == [user_message, call, assistant_message, call]


@pytest.mark.parametrize(
    ("method", "argument", "invalid_type"),
    [
        ("add_message", "invalid", "<class 'str'>"),
        ("add_messages", [Message(role="user", content="hello"), 42], "<class 'int'>"),
    ],
)
def test_rejects_invalid_message_types(method, argument, invalid_type):
    """Context additions identify the invalid type without changing history."""
    context = LoopContext()

    with pytest.raises(
        ValueError,
        match=rf"Expected a conversation item, got {invalid_type}",
    ):
        getattr(context, method)(argument)

    assert context.messages == []
