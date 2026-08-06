"""Tests for session state, serialization, and persistence contracts."""

from datetime import UTC, datetime

import pytest

from loop import (
    Message,
    Reasoning,
    Session,
    SessionInfo,
    ToolCall,
    ToolResult,
    UnsupportedConversationItemError,
)


def function_call() -> ToolCall:
    """Build a completed local function-tool call."""
    return ToolCall(
        call_id="call_123",
        name="get_current_datetime",
        arguments="{}",
        id="fc_123",
    )


def test_session_info_describes_a_persisted_session():
    """Session summaries expose stable persistence metadata."""
    updated_at = datetime(2026, 8, 4, tzinfo=UTC)

    info = SessionInfo(id="session-id", updated_at=updated_at, message_count=2)

    assert info == SessionInfo("session-id", updated_at, 2)


def test_session_adds_one_or_multiple_messages():
    """Sessions append individual and iterable conversation items in order."""
    session = Session()
    user = Message(role="user", content="hello")
    answer = Message(role="assistant", content="hi")

    session.add_message(user)
    session.add_messages(message for message in [function_call(), answer])

    assert session.messages == [user, function_call(), answer]


@pytest.mark.parametrize("method", ["add_message", "add_messages"])
def test_session_rejects_invalid_message_types(method):
    """Both message insertion interfaces reject non-conversation values atomically."""
    session = Session(messages=[Message(role="user", content="existing")])
    value = object() if method == "add_message" else [Message(role="user", content="new"), object()]

    with pytest.raises(ValueError, match="Expected a conversation item"):
        getattr(session, method)(value)

    assert session.messages == [Message(role="user", content="existing")]


def test_session_serializes_and_deserializes_all_conversation_items():
    """Session snapshots round-trip every supported item type and response metadata."""
    session = Session(
        messages=[
            Message(role="user", content="hello"),
            Reasoning(content="thinking", id="reasoning"),
            function_call(),
            ToolResult(call_id="call_123", output="done"),
        ],
        tokens=42,
        model="model-a",
    )

    assert Session.deserialize(session.serialize()) == session


def test_session_serialization_identifies_unsupported_item_types():
    """Serialization reports the unsupported Python conversation item type."""
    session = Session(messages=[object()])

    with pytest.raises(
        UnsupportedConversationItemError,
        match="Unsupported conversation item type: object\\.",
    ):
        session.serialize()


@pytest.mark.parametrize(
    "payload, message",
    [
        ("not-json", "Invalid serialized session"),
        ("[]", "Invalid serialized session"),
        ('{"version":2,"messages":[],"tokens":0,"model":null}', "Unsupported session version 2"),
        ('{"messages":[],"tokens":0,"model":null}', "Unsupported session version None"),
        (
            '{"version":1,"messages":[{"type":"message","data":{}}],"tokens":0,"model":null}',
            "Invalid serialized session",
        ),
        ('{"version":1,"messages":[],"tokens":true,"model":null}', "Invalid serialized session"),
        ('{"version":1,"messages":[],"tokens":-1,"model":null}', "Invalid serialized session"),
        ('{"version":1,"messages":[],"tokens":0,"model":42}', "Invalid serialized session"),
        ('{"version":1,"messages":null,"tokens":0,"model":null}', "Invalid serialized session"),
    ],
)
def test_session_deserialization_rejects_invalid_data(payload, message):
    """Deserialization rejects malformed, incomplete, and incorrectly typed snapshots."""
    with pytest.raises(ValueError, match=message):
        Session.deserialize(payload)


def test_session_deserialization_identifies_unsupported_item_types():
    """Deserialization reports the unsupported serialized conversation item type."""
    payload = '{"version":1,"messages":[{"type":"unknown","data":{}}],"tokens":0,"model":null}'

    with pytest.raises(
        UnsupportedConversationItemError,
        match="Unsupported conversation item type: 'unknown'\\.",
    ):
        Session.deserialize(payload)
