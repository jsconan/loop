"""Tests for conversation context state."""

import pytest

from loop import (
    LoopContext,
    Message,
    Reasoning,
    ToolCall,
    ToolResult,
    UnsupportedConversationItemError,
)


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


def test_serializes_and_deserializes_complete_typed_contexts():
    """Context JSON preserves every item type, token count, and model identifier."""
    context = LoopContext(
        messages=[
            Message(role="user", content="hello"),
            Reasoning(content="thinking", id="reasoning"),
            function_call(),
            ToolResult(call_id="call_123", output="done"),
        ],
        tokens=12,
        model="model-a",
    )

    assert LoopContext.deserialize(context.serialize()) == context


def test_serialize_identifies_unsupported_conversation_item_types():
    """Context serialization reports the unsupported Python item type."""
    context = LoopContext(messages=[object()])

    with pytest.raises(
        UnsupportedConversationItemError,
        match="Unsupported conversation item type: object\\.",
    ):
        context.serialize()


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        '{"version":2,"messages":[],"tokens":0,"model":null}',
        (
            '{"version":1,"messages":[{"type":"message","data":{}}],'
            '"tokens":0,"model":null}'
        ),
        '{"version":1,"messages":[],"tokens":true,"model":null}',
        '{"version":1,"messages":[],"tokens":0,"model":42}',
    ],
)
def test_deserialize_rejects_invalid_or_unsupported_data(payload):
    """Context deserialization rejects corrupt and incorrectly typed data."""
    with pytest.raises(ValueError):
        LoopContext.deserialize(payload)


def test_deserialize_identifies_unsupported_conversation_item_types():
    """Context deserialization reports the unsupported serialized item type."""
    payload = (
        '{"version":1,"messages":[{"type":"unknown","data":{}}],'
        '"tokens":0,"model":null}'
    )

    with pytest.raises(
        UnsupportedConversationItemError,
        match="Unsupported conversation item type: 'unknown'\\.",
    ):
        LoopContext.deserialize(payload)
