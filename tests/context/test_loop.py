"""Tests for conversation context state."""

import pytest
from openai.types.responses import ResponseFunctionToolCall

from loop.context import LoopContext


def function_call() -> ResponseFunctionToolCall:
    """Build a completed Responses API function-call item."""
    return ResponseFunctionToolCall(
        id="fc_123",
        call_id="call_123",
        name="get_current_datetime",
        arguments="{}",
        type="function_call",
        status="completed",
    )


def test_adds_one_or_multiple_messages():
    """Context methods add dictionaries and dump models to conversation history."""
    context = LoopContext()
    user_message = {"role": "user", "content": "hello"}
    assistant_message = {"role": "assistant", "content": "answer"}
    call = function_call()
    dumped_call = call.model_dump(exclude_none=True)

    context.add_message(user_message)
    context.add_message(call)
    context.add_messages(message for message in (assistant_message, call))

    assert context.messages == [user_message, dumped_call, assistant_message, dumped_call]


@pytest.mark.parametrize(
    ("method", "argument"),
    [
        ("add_message", "invalid"),
        ("add_messages", [{"role": "user", "content": "hello"}, "invalid"]),
    ],
)
def test_rejects_invalid_message_types(method, argument):
    """Context additions reject unsupported message types without changing history."""
    context = LoopContext()

    with pytest.raises(ValueError, match="Expected message to be a dict or BaseModel"):
        getattr(context, method)(argument)

    assert context.messages == []
