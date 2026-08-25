"""Tests for session coordination and persistence."""

import json
from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call
from uuid import uuid4

import pytest

from loop import (
    Action,
    AnswerCompleted,
    AnswerDelta,
    AuthorizationResult,
    Compaction,
    CompactionContextItem,
    CompactionResult,
    ConsoleInteraction,
    ContentArtifact,
    ContextReference,
    Decision,
    FileTarget,
    InstructionSnapshot,
    MemorySessionStore,
    Message,
    ModelAssignment,
    Operation,
    PolicyDecision,
    Reasoning,
    ReasoningCompleted,
    ReasoningDelta,
    Response,
    ResponseCompleted,
    RunMetrics,
    Session,
    SessionManager,
    ToolCall,
    ToolCallCompleted,
    ToolResult,
    Usage,
)
from loop.interaction import Interaction
from loop.session import PermissionEvent, RunCompletedEvent, SessionStore
from loop.utils import cached_metadata, cached_path, store_content


def response_interaction() -> MagicMock:
    """Build an interaction mock backed by a no-op response scope."""
    interaction = MagicMock(spec=Interaction)
    interaction.response_context.return_value = nullcontext()
    return interaction


def test_manager_creates_default_services_and_an_empty_session():
    """Managers provide usable defaults when no collaborators are supplied."""
    manager = SessionManager()

    assert isinstance(manager.interaction, ConsoleInteraction)
    assert isinstance(manager.store, MemorySessionStore)
    assert manager.session == Session()
    assert manager.messages == []
    assert manager.model is None


def test_manager_uses_injected_services_and_session():
    """Managers expose the exact interaction, store, and session supplied by callers."""
    interaction = Mock(spec=Interaction)
    store = Mock(spec=SessionStore)
    session = Session(messages=[Message(role="user", content="hello")], model="model-a")

    manager = SessionManager(interaction=interaction, session=session, session_store=store)

    assert manager.interaction is interaction
    assert manager.store is store
    assert manager.session is session
    assert manager.messages is session.messages
    assert manager.model == "model-a"


def test_manager_response_uses_terminal_text_and_an_interaction_override():
    """Response collection renders events and returns authoritative terminal metadata."""
    configured = response_interaction()
    interaction = response_interaction()
    manager = SessionManager(interaction=configured)
    tool_call = ToolCall(call_id="call", name="tool", arguments="{}", id="fc")
    items = (
        Reasoning(content="think again", id="r"),
        Message(role="assistant", content="hello world"),
    )
    events = [
        ReasoningDelta(text="incomplete "),
        ReasoningDelta(text="thought"),
        AnswerDelta(text="incomplete "),
        AnswerDelta(text="answer"),
        ReasoningCompleted(text="incomplete thought"),
        AnswerCompleted(text="incomplete answer"),
        ToolCallCompleted(call=tool_call),
        SimpleNamespace(ignored=True),
        ResponseCompleted(
            items=items,
            usage=Usage(total_tokens=230),
            model="served-model",
            answer="  hello world  ",
            reasoning="  think again  ",
            structured_output={"message": "hello world"},
        ),
    ]

    response = manager.response(events, debug=True, interaction=interaction)

    assert response == Response(
        answer="  hello world  ",
        reasoning="  think again  ",
        tool_calls=(tool_call,),
        items=items,
        usage=Usage(total_tokens=230),
        model="served-model",
        structured_output={"message": "hello world"},
    )
    assert [call.args for call in interaction.reasoning_delta.call_args_list] == [
        ("incomplete ",),
        ("thought",),
    ]
    assert [call.args for call in interaction.answer_delta.call_args_list] == [
        ("incomplete ",),
        ("answer",),
    ]
    interaction.reasoning.assert_not_called()
    interaction.answer.assert_not_called()
    interaction.response_context.assert_called_once_with()
    assert interaction.debug.call_count == len(events)
    configured.response_context.assert_not_called()


def test_manager_response_displays_completed_text_with_its_configured_interaction():
    """Completed response text is rendered through the manager's interaction."""
    interaction = response_interaction()
    manager = SessionManager(interaction=interaction)

    response = manager.response(
        [
            ReasoningCompleted(text="think"),
            AnswerCompleted(text="answer"),
            ResponseCompleted(answer="answer", reasoning="think"),
        ]
    )

    assert response == Response(answer="answer", reasoning="think")
    interaction.reasoning.assert_called_once_with("think")
    interaction.answer.assert_called_once_with("answer")
    interaction.response_context.assert_called_once_with()


def test_manager_response_defaults_missing_metadata():
    """A completion without metadata returns default response values."""
    interaction = response_interaction()
    manager = SessionManager(interaction=interaction)

    response = manager.response([ResponseCompleted()])

    assert response == Response(answer="", reasoning="")


def test_manager_replays_visible_session_items_in_durable_order():
    """Replay mirrors live output, compactions, and intentionally hidden tool results."""
    interaction = MagicMock(spec=Interaction)
    items = (
        Message(role="user", content="question"),
        Reasoning(content="thought"),
        ToolCall(call_id="known", name="search", arguments='{"query":"term"}'),
        ToolResult(call_id="known", output="result"),
        Message(role="assistant", content="answer"),
    )
    instructions = InstructionSnapshot(working_directory="/project", content=None, digest="digest")
    compactions = (
        Compaction(
            id="first",
            boundary=0,
            created_at=datetime(2026, 8, 16, tzinfo=UTC),
            provider="test",
            model="model",
            context=(CompactionContextItem(provider="test", data={}),),
            instructions=instructions,
        ),
        Compaction(
            id="second",
            boundary=3,
            created_at=datetime(2026, 8, 16, tzinfo=UTC),
            provider="test",
            model="model",
            context=(CompactionContextItem(provider="test", data={}),),
            instructions=instructions,
            input_tokens_before=12_345,
            input_tokens_after=678,
        ),
    )
    manager = SessionManager(session=Session(messages=list(items), compactions=list(compactions)))

    manager.replay(interaction=interaction)

    assert interaction.method_calls == [
        call.info("Compacted session context."),
        call.user("question"),
        call.reasoning("thought"),
        call.tool_call("search", '{"query":"term"}'),
        call.info("Compacted session context from 12,345 to 678 tokens."),
        call.answer("answer"),
    ]


def test_manager_replays_permissions_and_run_statistics():
    """Replay includes prompted approvals and run summaries while hiding automatic decisions."""
    interaction = MagicMock(spec=Interaction)
    now = datetime(2026, 8, 20, tzinfo=UTC)
    operation = Operation(
        tool_id="read",
        action=Action.FILESYSTEM_READ,
        target=FileTarget(path="/workspace/file.txt"),
    )
    result = AuthorizationResult(
        operations=(operation,),
        policy=PolicyDecision(
            decision=Decision.ASK,
            reason="approval required",
            sources=("default:filesystem.read",),
        ),
        decision=Decision.ALLOW,
        prompted=True,
        reason="allowed",
        source="user",
    )
    prompted = PermissionEvent(
        id="permission",
        created_at=now,
        result=result,
    )
    metrics = RunMetrics(
        active_duration_seconds=1,
        model_duration_seconds=1,
        tool_duration_seconds=0,
        message_count=0,
        item_count=0,
    )
    manager = SessionManager(
        interaction=interaction,
        session=Session(
            events=[
                prompted,
                prompted.model_copy(
                    update={
                        "id": "automatic",
                        "result": result.model_copy(update={"prompted": False}),
                    }
                ),
                RunCompletedEvent(
                    id="run",
                    created_at=now,
                    started_at=now,
                    stop_reason="completed",
                    metrics=metrics,
                ),
            ]
        ),
    )

    manager.replay()

    interaction.permission.assert_called_once_with("Permission requested.", "allow")
    interaction.run_metrics.assert_called_once_with(metrics)


def test_manager_loads_a_session_identifier_during_initialization():
    """A session identifier is resolved through the configured store at construction."""
    store = Mock(spec=SessionStore)
    loaded = Session(id="session-id")
    store.load.return_value = loaded

    manager = SessionManager(session="session-id", session_store=store)

    assert manager.session is loaded
    store.load.assert_called_once_with("session-id")


def test_manager_replaces_the_active_session_when_loading():
    """Explicit loading replaces the active session with the stored snapshot."""
    store = Mock(spec=SessionStore)
    loaded = Session(id="stored-id")
    store.load.return_value = loaded
    manager = SessionManager(session=Session(id="original-id"), session_store=store)

    manager.load_session("stored-id")

    assert manager.session is loaded
    store.load.assert_called_once_with("stored-id")


def test_manager_accepts_a_session_object_when_loading():
    """Loading a session object replaces the active session without consulting the store."""
    store = Mock(spec=SessionStore)
    loaded = Session(id="replacement-id")
    manager = SessionManager(session_store=store)

    manager.load_session(loaded)

    assert manager.session is loaded
    store.load.assert_not_called()


def test_manager_rejects_an_invalid_session_type():
    """Loading rejects values that are neither sessions nor persisted identifiers."""
    manager = SessionManager()

    with pytest.raises(ValueError, match="Invalid session type"):
        manager.load_session(object())


def test_manager_adds_and_persists_one_conversation_item():
    """Adding one item updates the active session before persisting it."""
    store = Mock(spec=SessionStore)
    session = Session()
    manager = SessionManager(session=session, session_store=store)
    message = Message(role="assistant", content="answer")

    manager.add_message(message)

    assert manager.messages == [message]
    store.save.assert_called_once_with(session)


def test_manager_adds_and_persists_an_iterable_of_items():
    """Adding an iterable consumes it in order and persists the completed session once."""
    store = Mock(spec=SessionStore)
    session = Session()
    manager = SessionManager(session=session, session_store=store)
    messages = (
        item
        for item in [
            Message(role="user", content="question"),
            Message(role="assistant", content="answer"),
        ]
    )

    manager.add_messages(messages)

    assert manager.messages == [
        Message(role="user", content="question"),
        Message(role="assistant", content="answer"),
    ]
    store.save.assert_called_once_with(session)


def test_manager_constructs_and_persists_complete_user_messages():
    """User-message creation remains owned by the session boundary."""
    store = Mock(spec=SessionStore)
    session = Session()
    manager = SessionManager(session=session, session_store=store)
    reference = ContextReference(
        kind="file",
        path="app.py",
        content="pass\n",
        size_bytes=5,
        included_bytes=5,
        truncated=False,
    )

    manager.add_user_message("Review @app.py", context=iter([reference]))

    assert manager.messages == [
        Message(role="user", content="Review @app.py", context=(reference,))
    ]
    assert session.name == "Review @app.py"
    assert session.name_source == "initial"
    store.save.assert_called_once_with(session)


def test_manager_generates_names_regardless_of_the_current_name_source():
    """A completed exchange may replace both provisional and user-controlled names."""
    store = Mock(spec=SessionStore)
    generator = Mock()
    generator.generate.return_value = "Review application architecture"
    session = Session()
    manager = SessionManager(session=session, session_store=store)
    manager.add_user_message("Please review this app")
    manager.add_message(Message(role="assistant", content="The architecture is sound."))

    manager.generate_session_name(generator)

    assert session.name == "Review application architecture"
    assert session.name_source == "generated"
    generator.generate.assert_called_once_with(
        "Please review this app", "The architecture is sound.", None
    )
    manager.rename_session("My review")
    manager.generate_session_name(generator)
    assert session.name == "Review application architecture"
    assert session.name_source == "generated"
    assert generator.generate.call_count == 2
    assert store.save.call_count == 5


def test_manager_keeps_provisional_name_without_a_usable_exchange_or_generated_name():
    """Missing exchange text and empty generation results do not cause extra persistence."""
    store = Mock(spec=SessionStore)
    generator = Mock()
    generator.generate.return_value = None
    session = Session(name="Initial", name_source="initial")
    manager = SessionManager(session=session, session_store=store)

    manager.add_user_message("question")
    manager.generate_session_name(generator)
    manager.add_message(Message(role="assistant", content="answer"))
    manager.generate_session_name(generator)

    assert session.name == "Initial"
    generator.generate.assert_called_once_with("question", "answer", None)
    assert store.save.call_count == 2

    empty = SessionManager(
        session=Session(name="Initial", name_source="initial"), session_store=store
    )
    empty.generate_session_name(generator)


def test_manager_starts_a_fresh_unpersisted_session():
    """Starting over replaces active state without creating a stored record."""
    store = Mock(spec=SessionStore)
    manager = SessionManager(
        session=Session(id="old", name="Old", name_source="user"), session_store=store
    )

    manager.new_session()

    assert manager.session == Session()
    store.save.assert_not_called()


def test_manager_new_session_forwards_model():
    """Starting over forwards the current model to the new session."""
    store = Mock(spec=SessionStore)
    original = Session(model="original-model")
    manager = SessionManager(session=original, session_store=store)

    manager.new_session()

    assert manager.session.model == "original-model"
    store.save.assert_not_called()


def test_manager_new_session_forwards_none_model():
    """Starting over forwards None model when none was set."""
    store = Mock(spec=SessionStore)
    original = Session()  # model defaults to None
    manager = SessionManager(session=original, session_store=store)

    manager.new_session()

    assert manager.session.model is None
    store.save.assert_not_called()


def test_manager_adds_a_tool_result_with_its_instruction_state():
    """Tool results and their effective instruction state are persisted together."""
    store = Mock(spec=SessionStore)
    session = Session()
    manager = SessionManager(session=session, session_store=store)
    active_skills = iter([("review", "/skills/review/SKILL.md")])

    manager.add_tool_call("call-id", "result", "/project", active_skills)

    assert manager.messages == [ToolResult(call_id="call-id", output="result")]
    assert session.instruction_working_directory == "/project"
    assert session.active_skills == [("review", "/skills/review/SKILL.md")]
    store.save.assert_called_once_with(session)


def test_manager_caches_oversized_tool_results_before_persistence():
    """Oversized results persist only bounded previews with resumable artifact handles."""
    store = Mock(spec=SessionStore)
    interaction = Mock(spec=Interaction)
    session = Session()
    manager = SessionManager(interaction=interaction, session=session, session_store=store)

    manager.add_tool_call("large-call", "x" * (20 * 1024 + 1), "/project", [])

    output = manager.messages[0].output
    result = json.loads(output)
    assert len(output.encode("utf-8")) <= 20 * 1024
    assert result["size_bytes"] == 20 * 1024 + 1
    assert result["truncated"] is True
    assert result["handle"]
    assert manager.messages[0].artifacts == (
        ContentArtifact(
            handle=result["handle"],
            source="tool result large-call",
            reloadable=False,
        ),
    )
    interaction.info.assert_called_once()
    store.save.assert_called_once_with(session)


def test_manager_restores_artifact_metadata_from_a_loaded_session():
    """Loading a session restores reload sources persisted on prior tool results."""
    handle = uuid4().hex
    artifact = ContentArtifact(
        handle=handle,
        source="https://example.com/source.txt",
        reloadable=True,
    )
    session = Session(messages=[ToolResult(call_id="call", output="result", artifacts=(artifact,))])

    SessionManager(session=session)

    assert cached_metadata(handle) == {
        "source": "https://example.com/source.txt",
        "reloadable": True,
    }


def test_manager_restores_immutable_mention_content_into_an_expired_cache():
    """Loading a session makes a truncated attachment continuation readable again."""
    handle = store_content("old", "old source")
    cached_path(handle)[0].unlink()
    session = Session(
        messages=[
            Message(
                role="user",
                content="Review @large.txt",
                context=(
                    ContextReference(
                        kind="file",
                        path="large.txt",
                        content="snap",
                        size_bytes=8,
                        included_bytes=4,
                        truncated=True,
                        handle=handle,
                        next_cursor="cursor",
                        snapshot_content="snapshot",
                    ),
                ),
            )
        ]
    )

    SessionManager(session=session).load_session(session)

    assert cached_path(handle)[0].read_text(encoding="utf-8") == "snapshot"


def test_manager_ignores_unregistered_handles_in_tool_output():
    """Untrusted output cannot invent session artifact metadata without registration."""
    manager = SessionManager()

    manager.add_tool_call(
        "call",
        json.dumps({"handle": uuid4().hex, "source": "private"}),
        "/project",
        [],
    )

    assert manager.messages[0].artifacts == ()


def test_manager_updates_instruction_state_without_persisting_an_incomplete_query():
    """Instruction state remains in memory until the query response completes."""
    store = Mock(spec=SessionStore)
    session = Session()
    manager = SessionManager(session=session, session_store=store)
    active_skills = iter([("review", "/skills/review/SKILL.md")])

    manager.update_instruction_state("/project", active_skills)

    assert session.instruction_working_directory == "/project"
    assert session.active_skills == [("review", "/skills/review/SKILL.md")]
    store.save.assert_not_called()


def test_manager_adds_a_response_without_reconciling_model_assignment():
    """Adding a response persists its items without independently managing the model."""
    store = Mock(spec=SessionStore)
    session = Session()
    manager = SessionManager(session=session, session_store=store)
    answer = Message(role="assistant", content="done")
    response = Response(answer="done", reasoning="", items=(answer,), model="model-b")

    manager.add_response(response)

    assert manager.messages == [answer]
    assert manager.model is None
    store.save.assert_called_once_with(session)


def test_manager_exposes_durable_assignment_metadata():
    """Session usage and last-used assignment metadata remain available."""
    session = Session(tokens=17)
    manager = SessionManager(session=session)

    manager.assignment = ModelAssignment(model="selected-model", context_window=128000)

    assert manager.assignment == ModelAssignment(model="selected-model", context_window=128000)
    assert manager.model == "selected-model"
    assert manager.tokens == 17
    assert manager.context_window == 128000


def test_manager_exposes_context_metadata_and_validates_window_selection():
    """Session usage, model, and context-window metadata remain available and validated."""
    session = Session(tokens=17)
    manager = SessionManager(session=session)

    manager.model = "selected-model"
    manager.context_window = 128000

    assert manager.model == "selected-model"
    assert manager.tokens == 17
    assert manager.context_window == 128000
    with pytest.raises(ValueError, match="must be positive"):
        manager.context_window = 0


def test_manager_persists_compaction_with_instruction_and_skill_snapshot():
    """Compaction atomically records replacement context and exact instruction state."""
    store = Mock(spec=SessionStore)
    session = Session(messages=[Message(role="user", content="hello")], tokens=90)
    manager = SessionManager(session=session, session_store=store)
    result = CompactionResult(
        items=(
            CompactionContextItem(
                provider="openai",
                data={"type": "compaction", "encrypted_content": "opaque"},
            ),
        ),
        usage=Usage(input_tokens=90, output_tokens=20, total_tokens=110),
        context_tokens=20,
    )

    manager.add_compaction(
        result,
        model="model",
        instructions="project rules",
        working_directory="/project",
        active_skills=iter([("review", "/skills/review/SKILL.md")]),
    )

    checkpoint = session.compactions[0]
    assert manager.model_context == [*checkpoint.context]
    assert checkpoint.boundary == 1
    assert checkpoint.instructions.content == "project rules"
    assert checkpoint.instructions.active_skills == (("review", "/skills/review/SKILL.md"),)
    assert checkpoint.input_tokens_before == 90
    assert checkpoint.input_tokens_after == 20
    assert session.tokens == 20
    store.save.assert_called_once_with(session)


def test_manager_preserves_usage_when_compaction_omits_token_counts():
    """A compactor without usage metadata does not erase the latest known context usage."""
    session = Session(messages=[Message(role="user", content="hello")], tokens=90)
    manager = SessionManager(session=session)

    manager.add_compaction(
        CompactionResult(
            items=(CompactionContextItem(provider="openai", data={"type": "compaction"}),)
        ),
        model="model",
        instructions=None,
        working_directory="/project",
        active_skills=(),
    )

    assert session.tokens == 90
    assert session.compactions[0].input_tokens_after is None


def test_manager_rolls_back_compaction_when_persistence_fails():
    """A failed checkpoint write restores all prior in-memory session state."""
    store = Mock(spec=SessionStore)
    store.save.side_effect = OSError("disk full")
    session = Session(
        messages=[Message(role="user", content="hello")],
        tokens=90,
        instruction_working_directory="/old",
        active_skills=[("old", "/skills/old/SKILL.md")],
    )
    manager = SessionManager(session=session, session_store=store)

    with pytest.raises(OSError, match="disk full"):
        manager.add_compaction(
            CompactionResult(
                items=(CompactionContextItem(provider="openai", data={"type": "compaction"}),),
                usage=Usage(total_tokens=110),
                context_tokens=20,
            ),
            model="model",
            instructions="new rules",
            working_directory="/new",
            active_skills=(("new", "/skills/new/SKILL.md"),),
        )

    assert session.compactions == []
    assert session.tokens == 90
    assert session.instruction_working_directory == "/old"
    assert session.active_skills == [("old", "/skills/old/SKILL.md")]


@pytest.mark.parametrize("method", ["add_message", "add_messages"])
def test_manager_does_not_persist_rejected_items(method):
    """Validation failures leave persistence untouched for both insertion interfaces."""
    store = Mock(spec=SessionStore)
    manager = SessionManager(session_store=store)
    value = object() if method == "add_message" else [object()]

    with pytest.raises(ValueError, match="Expected a conversation item"):
        getattr(manager, method)(value)

    store.save.assert_not_called()


@pytest.mark.parametrize("kind", ["run", "tool_call", "message"])
def test_manager_rolls_back_every_mutation_when_persistence_fails(kind):
    """Failed writes restore identical in-memory state for messages and timeline events."""
    store = Mock(spec=SessionStore)
    store.save.side_effect = OSError("disk full")
    session = Session(messages=[ToolCall(call_id="call", name="demo", arguments="{}")])
    session.events.clear()
    manager = SessionManager(session=session, session_store=store)

    with pytest.raises(OSError, match="disk full"):
        if kind == "run":
            manager.record_run(
                "completed",
                datetime(2026, 8, 20, tzinfo=UTC),
                RunMetrics(
                    active_duration_seconds=0,
                    model_duration_seconds=0,
                    tool_duration_seconds=0,
                    message_count=0,
                    item_count=1,
                ),
            )
        elif kind == "tool_call":
            manager.add_tool_call_event("call")
        else:
            manager.add_message(Message(role="assistant", content="answer"))

    assert session.events == []
    assert session.messages == [ToolCall(call_id="call", name="demo", arguments="{}")]
