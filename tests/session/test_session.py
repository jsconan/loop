"""Tests for session state, serialization, and persistence contracts."""

import json
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest

from loop import (
    Compaction,
    CompactionContextItem,
    ContentArtifact,
    ContextReference,
    InstructionSnapshot,
    Message,
    ModelAssignment,
    Reasoning,
    Response,
    ResponseMetadata,
    RunMetrics,
    Session,
    SessionInfo,
    ToolCall,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    ToolResult,
    UnsupportedConversationItemError,
    Usage,
)
from loop.session.models import (
    SESSION_NAME_SOURCE_INITIAL,
    SESSION_NAME_SOURCE_USER,
    RunCompletedEvent,
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

    info = SessionInfo(id="session-id", name="Useful title", updated_at=updated_at, message_count=2)

    assert info == SessionInfo("session-id", "Useful title", updated_at, 2)


def test_session_generates_a_stable_uuidv7_identifier():
    """Fresh sessions receive an identifier before persistence begins."""
    session = Session()

    assert session.id is not None
    assert UUID(session.id).version == 7


def test_session_deserialization_rejects_an_empty_identifier():
    """Current snapshots require a non-empty session identifier."""
    payload = json.loads(Session().serialize())
    payload["id"] = ""

    with pytest.raises(ValueError, match="Invalid serialized session"):
        Session.deserialize(json.dumps(payload))


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


def test_session_adds_response_items_without_managing_model_assignment():
    """Completed responses append history and usage without changing durable assignment state."""
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
    assert session.model == "model-a"


def test_session_groups_model_metadata_as_one_assignment():
    """The session exposes and replaces its durable model metadata as one assignment."""
    session = Session()

    assert session.assignment is None
    session.assignment = ModelAssignment(model="model", context_window=8192)

    assert session.assignment == ModelAssignment(model="model", context_window=8192)


def test_session_places_response_tool_calls_when_they_are_executed():
    """Multiple model tool calls replay beside their sequential local results."""
    first = ToolCall(call_id="first", name="one", arguments="{}")
    second = ToolCall(call_id="second", name="two", arguments="{}")
    session = Session()
    session.add_message(
        Response(answer="", reasoning="", tool_calls=(first, second), items=(first, second))
    )

    assert session.events == []
    session.add_tool_call_event("first")
    session.add_message(ToolResult(call_id="first", output="one"))
    session.add_tool_call_event("second")
    session.add_message(ToolResult(call_id="second", output="two"))

    assert [session.messages[event.item_index] for event in session.events] == [
        first,
        ToolResult(call_id="first", output="one"),
        second,
        ToolResult(call_id="second", output="two"),
    ]
    with pytest.raises(ValueError, match="already"):
        session.add_tool_call_event("first")
    with pytest.raises(ValueError, match="Unknown"):
        session.add_tool_call_event("missing")


@pytest.mark.parametrize(
    ("messages", "events", "action", "statuses"),
    [
        ([Message(role="user", content="question")], [], "query_model", []),
        (
            [
                Message(role="user", content="question"),
                ToolCall(call_id="call", name="demo", arguments="{}"),
            ],
            [],
            "execute_tools",
            ["not_started"],
        ),
        (
            [
                Message(role="user", content="question"),
                ToolCall(call_id="call", name="demo", arguments="{}"),
            ],
            [
                ToolExecutionStartedEvent(
                    id="started", created_at=datetime(2026, 8, 26, tzinfo=UTC), call_id="call"
                )
            ],
            "resolve_uncertain_tools",
            ["outcome_unknown"],
        ),
        (
            [
                Message(role="user", content="question"),
                ToolCall(call_id="call", name="demo", arguments="{}"),
                ToolResult(call_id="call", output="done"),
            ],
            [],
            "query_model",
            ["result_available"],
        ),
        (
            [
                Message(role="user", content="question"),
                Message(role="assistant", content="answer"),
            ],
            [],
            "finalize_run",
            [],
        ),
    ],
)
def test_session_classifies_every_interrupted_agent_boundary(messages, events, action, statuses):
    """Recovery derives the safest continuation from canonical items and lifecycle events."""
    calls = tuple(item for item in messages if isinstance(item, ToolCall))
    if calls:
        session = Session()
        for item in messages:
            if isinstance(item, Message):
                session.add_message(item)
        session.add_message(Response(answer="", reasoning="", tool_calls=calls, items=calls))
        for item in messages:
            if isinstance(item, ToolResult):
                session.add_message(item)
        session.events.extend(events)
    else:
        session = Session(messages=messages, events=events)

    state = session.recovery_state()

    assert state is not None
    assert state.action == action
    assert [pending.status for pending in state.pending_calls] == statuses


def test_session_treats_presented_legacy_calls_as_uncertain_and_completed_runs_as_clean():
    """Legacy execution markers remain conservative while terminal runs need no recovery."""
    call = ToolCall(call_id="call", name="demo", arguments="{}")
    session = Session(messages=[Message(role="user", content="question"), call])

    assert session.recovery_state().pending_calls[0].status == "outcome_unknown"

    session.events.append(
        RunCompletedEvent(
            id="run",
            created_at=datetime(2026, 8, 26, tzinfo=UTC),
            stop_reason="cancelled",
            started_at=datetime(2026, 8, 26, tzinfo=UTC),
            metrics=RunMetrics(
                active_duration_seconds=0,
                model_duration_seconds=0,
                tool_duration_seconds=0,
                message_count=1,
                item_count=2,
            ),
        )
    )

    assert session.recovery_state() is None


def test_session_finalizes_a_response_that_followed_durable_tool_results():
    """A final assistant item after consumed tool results needs only run bookkeeping."""
    call = ToolCall(call_id="call", name="demo", arguments="{}")
    session = Session()
    session.add_message(Message(role="user", content="question"))
    session.add_message(Response(answer="", reasoning="", tool_calls=(call,), items=(call,)))
    session.add_message(ToolResult(call_id="call", output="done"))
    session.add_message(Message(role="assistant", content="answer"))

    assert session.recovery_state().action == "finalize_run"


def test_session_recovery_uses_complete_history_after_a_compaction():
    """Compaction does not hide an unfinished tool request from recovery classification."""
    call = ToolCall(call_id="call", name="demo", arguments="{}")
    session = Session()
    session.add_message(Message(role="user", content="question"))
    session.add_compaction(
        Compaction(
            id="checkpoint",
            boundary=1,
            created_at=datetime(2026, 8, 26, tzinfo=UTC),
            provider="test",
            model="model",
            context=(CompactionContextItem(provider="test", data={}),),
            instructions=InstructionSnapshot(
                working_directory="/project", content=None, digest="digest"
            ),
        )
    )
    session.add_message(Response(answer="", reasoning="", tool_calls=(call,), items=(call,)))

    state = session.recovery_state()

    assert state.action == "execute_tools"
    assert state.pending_calls[0].call == call


def test_session_round_trips_and_validates_tool_execution_lifecycle_events():
    """Tool lifecycle checkpoints survive persistence and must reference known calls."""
    call = ToolCall(call_id="call", name="demo", arguments="{}")
    session = Session(messages=[call, ToolResult(call_id="call", output="done")])
    session.events.extend(
        [
            ToolExecutionStartedEvent(
                id="start", created_at=datetime(2026, 8, 26, tzinfo=UTC), call_id="call"
            ),
            ToolExecutionCompletedEvent(
                id="end",
                created_at=datetime(2026, 8, 26, tzinfo=UTC),
                call_id="call",
                succeeded=True,
                duration_seconds=0.5,
            ),
        ]
    )

    assert Session.deserialize(session.serialize()) == session

    payload = json.loads(session.serialize())
    lifecycle = next(
        event for event in payload["events"] if event["type"] == "tool_execution_started"
    )
    lifecycle["call_id"] = "missing"
    with pytest.raises(ValueError, match="Invalid serialized session"):
        Session.deserialize(json.dumps(payload))


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
                        source="https://my-host.local/source.txt",
                        reloadable=True,
                    ),
                ),
            ),
        ],
        tokens=42,
        model="model-a",
        context_window=128000,
    )

    assert Session.deserialize(session.serialize()) == session


def test_session_round_trips_instruction_context_and_events():
    """Persistence retains instruction state and the durable replay timeline."""
    session = Session(
        instruction_working_directory="/project/src",
        active_skills=[("review", "/project/.agents/skills/review/SKILL.md")],
    )

    assert Session.deserialize(session.serialize()) == session


def test_session_upcasts_version_six_events_without_rewriting_them():
    """The immediately previous schema retains its typed timeline during in-memory migration."""
    session = Session(messages=[Message(role="user", content="question")])
    payload = json.loads(session.serialize())
    payload["version"] = 6

    assert Session.deserialize(json.dumps(payload)) == session


def test_session_upcasts_version_five_run_metrics_without_rewriting_history():
    """Legacy wall duration remains distinct from reconstructed active checkpoint time."""
    now = datetime(2026, 8, 20, tzinfo=UTC)
    metrics = RunMetrics(
        active_duration_seconds=3,
        model_duration_seconds=2,
        tool_duration_seconds=1,
        message_count=0,
        item_count=0,
    )
    session = Session(messages=[Message(role="user", content="question")])
    session.events.append(
        RunCompletedEvent(
            id="run",
            created_at=now,
            started_at=now,
            stop_reason="completed",
            metrics=metrics,
        )
    )
    payload = json.loads(session.serialize())
    payload["version"] = 5
    event = payload["events"][1]
    legacy_metrics = event.pop("metrics")
    event.update(legacy_metrics, duration_seconds=8)
    event.pop("active_duration_seconds")
    event.pop("elapsed_duration_seconds")

    restored = Session.deserialize(json.dumps(payload))

    restored_metrics = restored.events[1].metrics
    assert restored_metrics.active_duration_seconds == 3
    assert restored_metrics.elapsed_duration_seconds == 8


def test_session_serialization_normalizes_durable_event_timestamps_to_utc():
    """Durable event timestamps are persisted in UTC regardless of their input offset."""
    offset_time = datetime(2026, 8, 20, 12, tzinfo=timezone(timedelta(hours=2)))
    session = Session(messages=[Message(role="user", content="question")])
    session.events.append(
        RunCompletedEvent(
            id="run",
            created_at=offset_time,
            started_at=offset_time,
            stop_reason="completed",
            metrics=RunMetrics(
                active_duration_seconds=0,
                model_duration_seconds=0,
                tool_duration_seconds=0,
                message_count=0,
                item_count=0,
            ),
        )
    )

    event = json.loads(session.serialize())["events"][1]

    assert event["created_at"] == "2026-08-20T10:00:00Z"
    assert event["started_at"] == "2026-08-20T10:00:00Z"


def test_session_rejects_malformed_version_five_run_metrics():
    """Legacy run upcasting normalizes invalid arithmetic fields to the session error contract."""
    payload = json.loads(Session().serialize())
    payload["version"] = 5
    payload["events"] = [
        {
            "id": "run",
            "created_at": "2026-08-20T00:00:00Z",
            "type": "run_completed",
            "stop_reason": "completed",
            "started_at": "2026-08-20T00:00:00Z",
            "duration_seconds": 1,
            "model_duration_seconds": "invalid",
            "tool_duration_seconds": 0,
            "message_count": 0,
            "item_count": 0,
        }
    ]

    with pytest.raises(ValueError, match="Invalid serialized session"):
        Session.deserialize(json.dumps(payload))


def test_session_retains_full_history_while_latest_compaction_bounds_model_context():
    """Multiple checkpoints persist while only the latest seeds subsequent model input."""
    messages = [Message(role="user", content=str(index)) for index in range(4)]
    session = Session(messages=messages)
    instructions = InstructionSnapshot(
        working_directory="/project",
        content="rules",
        digest="digest",
        active_skills=(("review", "/skills/review/SKILL.md"),),
    )
    first = Compaction(
        id="first",
        boundary=2,
        created_at=datetime(2026, 8, 16, tzinfo=UTC),
        provider="openai",
        model="model",
        context=(CompactionContextItem(provider="openai", data={"type": "compaction"}),),
        instructions=instructions,
    )
    second = first.model_copy(
        update={
            "id": "second",
            "boundary": 3,
            "context": (
                CompactionContextItem(provider="openai", data={"type": "compaction", "id": "2"}),
            ),
        }
    )

    session.add_compaction(first)
    session.add_compaction(second)
    restored = Session.deserialize(session.serialize())

    assert restored.messages == messages
    assert restored.compactions == [first, second]
    assert restored.model_context() == [*second.context, messages[3]]


@pytest.mark.parametrize(
    "compaction, message",
    [
        (
            Compaction(
                id="empty",
                boundary=0,
                created_at=datetime(2026, 8, 16, tzinfo=UTC),
                provider="openai",
                model="model",
                context=(),
                instructions=InstructionSnapshot(
                    working_directory="/project", content=None, digest="digest"
                ),
            ),
            "cannot be empty",
        ),
        (
            Compaction(
                id="past-end",
                boundary=2,
                created_at=datetime(2026, 8, 16, tzinfo=UTC),
                provider="openai",
                model="model",
                context=(CompactionContextItem(provider="openai", data={}),),
                instructions=InstructionSnapshot(
                    working_directory="/project", content=None, digest="digest"
                ),
            ),
            "exceeds",
        ),
    ],
)
def test_session_rejects_invalid_compaction_boundaries(compaction, message):
    """Checkpoints must contain replacement context and cover only stored history."""
    session = Session(messages=[Message(role="user", content="one")])

    with pytest.raises(ValueError, match=message):
        session.add_compaction(compaction)


def test_session_rejects_a_checkpoint_that_does_not_advance():
    """A later checkpoint must cover new full-history items."""
    session = Session(messages=[Message(role="user", content="one")])
    checkpoint = Compaction(
        id="first",
        boundary=1,
        created_at=datetime(2026, 8, 16, tzinfo=UTC),
        provider="openai",
        model="model",
        context=(CompactionContextItem(provider="openai", data={}),),
        instructions=InstructionSnapshot(working_directory="/project", content=None, digest="d"),
    )
    session.add_compaction(checkpoint)

    with pytest.raises(ValueError, match="must advance"):
        session.add_compaction(checkpoint.model_copy(update={"id": "second"}))


@pytest.mark.parametrize("mutation", ["window", "empty", "boundary", "order"])
def test_session_deserialization_validates_compaction_metadata(mutation):
    """Version-four snapshots reject invalid capacity, boundaries, and checkpoint ordering."""
    session = Session(messages=[Message(role="user", content="one")], context_window=100)
    checkpoint = Compaction(
        id="first",
        boundary=1,
        created_at=datetime(2026, 8, 16, tzinfo=UTC),
        provider="openai",
        model="model",
        context=(CompactionContextItem(provider="openai", data={}),),
        instructions=InstructionSnapshot(working_directory="/project", content=None, digest="d"),
    )
    session.add_compaction(checkpoint)
    payload = json.loads(session.serialize())
    if mutation == "window":
        payload["context_window"] = 0
    elif mutation == "empty":
        payload["compactions"][0]["context"] = []
    elif mutation == "boundary":
        payload["compactions"][0]["boundary"] = 2
    else:
        payload["compactions"].append(payload["compactions"][0] | {"id": "second"})

    with pytest.raises(ValueError, match="Invalid serialized session"):
        Session.deserialize(json.dumps(payload))


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
        ('{"version":9,"messages":[],"tokens":0,"model":null}', "Unsupported session version 9"),
        ('{"messages":[],"tokens":0,"model":null}', "Unsupported session version None"),
        ('{"version":5,"messages":[]}', "Invalid serialized session"),
    ],
)
def test_session_deserialization_rejects_invalid_data(payload, message):
    """Deserialization rejects malformed, incomplete, and incorrectly typed snapshots."""
    with pytest.raises(ValueError, match=message):
        Session.deserialize(payload)


def test_session_deserialization_identifies_unsupported_item_types():
    """Deserialization reports the unsupported serialized conversation item type."""
    payload = json.loads(Session().serialize())
    payload["messages"] = [{"type": "unknown", "data": {}}]

    with pytest.raises(
        UnsupportedConversationItemError,
        match="Unsupported conversation item type: 'unknown'\\.",
    ):
        Session.deserialize(json.dumps(payload))


@pytest.mark.parametrize(
    "mutation",
    [
        "name",
        "tokens",
        "model",
        "window",
        "directory",
        "skills",
        "item_range",
        "item_coverage",
        "compaction_range",
        "compaction_coverage",
    ],
)
def test_session_deserialization_rejects_invalid_event_and_snapshot_metadata(mutation):
    """Current snapshots reject malformed metadata and incomplete event references."""
    session = Session(messages=[Message(role="user", content="question")])
    checkpoint = Compaction(
        id="checkpoint",
        boundary=1,
        created_at=datetime(2026, 8, 20, tzinfo=UTC),
        provider="test",
        model="model",
        context=(CompactionContextItem(provider="test", data={}),),
        instructions=InstructionSnapshot(working_directory="/project", content=None, digest="d"),
    )
    session.add_compaction(checkpoint)
    payload = json.loads(session.serialize())
    if mutation == "name":
        payload["name"] = ""
    elif mutation == "tokens":
        payload["tokens"] = True
    elif mutation == "model":
        payload["model"] = 42
    elif mutation == "window":
        payload["context_window"] = 0
    elif mutation == "directory":
        payload["instruction_working_directory"] = 42
    elif mutation == "skills":
        payload["active_skills"] = [["incomplete"]]
    elif mutation == "item_range":
        payload["events"][0]["item_index"] = 2
    elif mutation == "item_coverage":
        payload["events"] = payload["events"][1:]
    elif mutation == "compaction_range":
        payload["events"][1]["compaction_index"] = 2
    else:
        payload["events"] = payload["events"][:1]

    with pytest.raises(ValueError, match="Invalid serialized session"):
        Session.deserialize(json.dumps(payload))
