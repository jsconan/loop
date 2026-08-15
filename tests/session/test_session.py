"""Tests for session state, serialization, and persistence contracts."""

from datetime import UTC, datetime

import pytest

from loop import (
    ContentArtifact,
    ContextReference,
    Message,
    Reasoning,
    Response,
    ResponseMetadata,
    Session,
    SessionInfo,
    ToolCall,
    ToolResult,
    UnsupportedConversationItemError,
    Usage,
)
from loop.session.models import SESSION_NAME_SOURCE_INITIAL, SESSION_NAME_SOURCE_USER


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

    info = SessionInfo(id="session-id", name="Useful title", updated_at=updated_at, message_count=2)

    assert info == SessionInfo("session-id", "Useful title", updated_at, 2)


def test_session_adds_one_or_multiple_messages():
    """Sessions append individual and iterable conversation items in order."""
    session = Session()
    user = Message(role="user", content="hello")
    answer = Message(role="assistant", content="hi")

    session.add_message(user)
    session.add_messages(message for message in [function_call(), answer])

    assert session.messages == [user, function_call(), answer]


def test_session_rename_validates_name_and_source():
    """Session renaming rejects empty names and unknown provenance values."""
    session = Session()

    with pytest.raises(ValueError, match="cannot be empty"):
        session.rename("  ")
    with pytest.raises(ValueError, match="Invalid session name source"):
        session.rename("Valid", source="unknown")


def test_session_has_name_returns_false_when_no_name_or_source():
    """A freshly created session has no name and no source."""
    session = Session()

    assert session.has_name() is False


def test_session_has_name_returns_true_when_name_and_source_set():
    """A renamed session has both a name and a source, so has_name is true."""
    session = Session()
    session.rename("My Session", source=SESSION_NAME_SOURCE_USER)

    assert session.has_name() is True


def test_session_has_initial_name_returns_false_when_no_source():
    """A session with no name source does not have an initial name."""
    session = Session()

    assert session.has_initial_name() is False


def test_session_has_initial_name_returns_true_after_initial_rename():
    """A session renamed with the initial source reports has_initial_name as true."""
    session = Session()
    session.rename("Draft", source=SESSION_NAME_SOURCE_INITIAL)

    assert session.has_initial_name() is True


def test_session_has_initial_name_returns_false_for_non_initial_sources():
    """A session renamed with a non-initial source reports has_initial_name as false."""
    session = Session()
    session.rename("My Session", source=SESSION_NAME_SOURCE_USER)

    assert session.has_initial_name() is False


def test_session_updates_instruction_state():
    """Sessions replace their effective directory and materialize active skill identities."""
    session = Session()
    active_skills = iter([("review", "/skills/review/SKILL.md")])

    session.update_instruction_state("/project", active_skills)

    assert session.instruction_working_directory == "/project"
    assert session.active_skills == [("review", "/skills/review/SKILL.md")]


def test_session_adds_response_items_and_updates_reported_metadata():
    """Completed responses append their items and replace reported session metadata."""
    session = Session(tokens=10, model="model-a")
    answer = Message(role="assistant", content="done")
    response = Response(
        answer="done",
        reasoning="",
        items=(function_call(), answer),
        usage=Usage(total_tokens=42),
        model="model-b",
    )

    session.add_message(response)

    assert session.messages == [function_call(), answer]
    assert session.tokens == 42
    assert session.model == "model-b"


def test_session_preserves_metadata_omitted_from_a_response():
    """Completed responses retain existing metadata when replacements are unavailable."""
    session = Session(tokens=10, model="model-a")
    response = Response(answer="", reasoning="")

    session.add_message(response)

    assert session.messages == []
    assert session.tokens == 10
    assert session.model == "model-a"


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
    metadata = ResponseMetadata(
        response_id="response_1",
        model="model-a",
        usage=Usage(input_tokens=30, output_tokens=12, total_tokens=42),
    )
    session = Session(
        messages=[
            Message(
                role="user",
                content="hello",
                context=(
                    ContextReference(
                        kind="file",
                        path="src/app.py",
                        content="print('hello')\n",
                        size_bytes=15,
                        included_bytes=15,
                        truncated=False,
                    ),
                ),
            ),
            Reasoning(content="thinking", id="reasoning", metadata=metadata),
            function_call(),
            ToolResult(
                call_id="call_123",
                output="done",
                artifacts=(
                    ContentArtifact(
                        handle="0123456789abcdef0123456789abcdef",
                        source="https://example.com/source.txt",
                        reloadable=True,
                    ),
                ),
            ),
        ],
        tokens=42,
        model="model-a",
    )

    assert Session.deserialize(session.serialize()) == session


def test_session_round_trips_instruction_context_and_reads_version_one_defaults():
    """Persistence retains refresh intent while older snapshots receive safe defaults."""
    session = Session(
        instruction_working_directory="/project/src",
        active_skills=[("review", "/project/.agents/skills/review/SKILL.md")],
    )

    assert Session.deserialize(session.serialize()) == session
    restored = Session.deserialize('{"version":1,"messages":[],"tokens":0,"model":null}')
    assert restored.instruction_working_directory is None
    assert not restored.active_skills


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
        ('{"version":4,"messages":[],"tokens":0,"model":null}', "Unsupported session version 4"),
        ('{"messages":[],"tokens":0,"model":null}', "Unsupported session version None"),
        (
            '{"version":1,"messages":[{"type":"message","data":{}}],"tokens":0,"model":null}',
            "Invalid serialized session",
        ),
        ('{"version":1,"messages":[],"tokens":true,"model":null}', "Invalid serialized session"),
        ('{"version":1,"messages":[],"tokens":-1,"model":null}', "Invalid serialized session"),
        ('{"version":1,"messages":[],"tokens":0,"model":42}', "Invalid serialized session"),
        ('{"version":1,"messages":null,"tokens":0,"model":null}', "Invalid serialized session"),
        (
            '{"version":3,"name":"","name_source":"initial","messages":[],"tokens":0,'
            '"model":null,"instruction_working_directory":null,"active_skills":[]}',
            "Invalid serialized session",
        ),
        (
            '{"version":3,"name":"name","name_source":"unknown","messages":[],"tokens":0,'
            '"model":null,"instruction_working_directory":null,"active_skills":[]}',
            r"Invalid session name source ''unknown''\.",
        ),
        (
            '{"version":2,"messages":[],"tokens":0,"model":null,'
            '"instruction_working_directory":42,"active_skills":[]}',
            "Invalid serialized session",
        ),
        (
            '{"version":2,"messages":[],"tokens":0,"model":null,'
            '"instruction_working_directory":null,"active_skills":null}',
            "Invalid serialized session",
        ),
        (
            '{"version":2,"messages":[],"tokens":0,"model":null,'
            '"instruction_working_directory":null,"active_skills":[["name"]]}',
            "Invalid serialized session",
        ),
        (
            '{"version":2,"messages":[],"tokens":0,"model":null,'
            '"instruction_working_directory":null,"active_skills":[["name",1]]}',
            "Invalid serialized session",
        ),
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
