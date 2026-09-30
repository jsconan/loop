"""Tests for session coordination and persistence."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import UTC, datetime
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch
from uuid import uuid4

import pytest

from loop import (
    Action,
    AnswerCompleted,
    AnswerDelta,
    ApprovalChoice,
    AuthorizationResult,
    BackendResponseError,
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
    SessionExecutionConflictError,
    SessionManager,
    SessionWorkspaceMismatchError,
    ToolCall,
    ToolCallCompleted,
    ToolResult,
    Usage,
)
from loop.instructions import CapturedInstruction
from loop.interaction import Interaction
from loop.session import (
    PermissionEvent,
    RunCompletedEvent,
    SessionStore,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
)
from loop.session import session_manager as session_manager_module
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry
from loop.utils import (
    cached_metadata,
    cached_path,
    content_identity,
    encode_content_cursor,
    sha256_digest,
    store_content,
)


def response_interaction() -> MagicMock:
    """Build an interaction mock backed by a no-op response scope."""
    interaction = MagicMock(spec=Interaction)
    interaction.response_context.return_value = nullcontext()
    return interaction


def persist_reference(store, content: bytes, path: str = "large.txt"):
    """Persist one reference through the atomic session boundary."""
    handle, version = content_identity(content)
    reference = ContextReference(
        kind="file",
        path=path,
        content=content.decode(),
        size_bytes=len(content),
        included_bytes=0,
        truncated=True,
        handle=handle,
        next_cursor=encode_content_cursor(handle, 0),
        version=version,
        media_type="text/plain",
    )
    manager = SessionManager(session_store=store)
    manager.add_user_message(f"Review @{path}", context=(reference,))
    store_content(content, f"mentioned file {path}", handle=handle)
    return manager.session, handle, version


def test_manager_creates_default_services_and_an_empty_session():
    """Managers provide usable defaults when no collaborators are supplied."""
    manager = SessionManager()

    assert isinstance(manager.interaction, ConsoleInteraction)
    assert isinstance(manager.store, MemorySessionStore)
    assert manager.session.id is not None
    assert manager.session.messages == []
    assert manager.session.name is None
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


def test_manager_seeds_references_from_a_supplied_in_memory_session():
    """Managers persist transient reference bytes supplied by an active session."""
    content = b"snapshot"
    handle, version = content_identity(content)
    session = Session(
        messages=[
            Message(
                role="user",
                content="Review @source.txt",
                context=(
                    ContextReference(
                        kind="file",
                        path="source.txt",
                        content=content.decode(),
                        size_bytes=len(content),
                        included_bytes=len(content),
                        truncated=False,
                        handle=handle,
                        version=version,
                    ),
                ),
            )
        ]
    )
    store = MemorySessionStore()

    manager = SessionManager(session=session, session_store=store)

    assert manager.session is session
    assert store.load_reference(version) == content
    assert manager.model_context[0].context[0].content == content.decode()
    assert session.revision == 0
    assert store.list() == []


def test_manager_rejects_corrupt_reference_bytes_from_a_supplied_session():
    """Managers do not seed or save an in-memory session with mismatched reference bytes."""
    handle, version = content_identity(b"expected")
    session = Session(
        messages=[
            Message(
                role="user",
                content="Review @source.txt",
                context=(
                    ContextReference(
                        kind="file",
                        path="source.txt",
                        content="changed",
                        size_bytes=len(b"changed"),
                        included_bytes=len(b"changed"),
                        truncated=False,
                        handle=handle,
                        version=version,
                    ),
                ),
            )
        ]
    )
    store = MemorySessionStore()

    with pytest.raises(ValueError, match="integrity validation"):
        SessionManager(session=session, session_store=store)

    assert store.list() == []
    assert store.load_reference(version) is None


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


def test_manager_response_rejects_missing_and_duplicate_completion_events():
    """The shared collector accepts exactly one explicit response completion."""
    manager = SessionManager(interaction=response_interaction())

    with pytest.raises(BackendResponseError, match="without an explicit completion"):
        manager.response([AnswerDelta(text="partial")])
    with pytest.raises(BackendResponseError, match="more than one"):
        manager.response([ResponseCompleted(), ResponseCompleted()])
    with pytest.raises(BackendResponseError, match="event after response completion"):
        manager.response([ResponseCompleted(), AnswerDelta(text="late")])


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
    session = Session(messages=list(items), compactions=list(compactions))
    session.events.extend(
        [
            ToolExecutionStartedEvent(
                id="start", created_at=datetime(2026, 8, 16, tzinfo=UTC), call_id="known"
            ),
            ToolExecutionCompletedEvent(
                id="end",
                created_at=datetime(2026, 8, 16, tzinfo=UTC),
                call_id="known",
                succeeded=True,
                duration_seconds=0.1,
            ),
        ]
    )
    manager = SessionManager(session=session)

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
        approval_choice=ApprovalChoice.SESSION,
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
                        "id": "legacy",
                        "result": result.model_copy(update={"approval_choice": None}),
                    }
                ),
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

    assert interaction.permission.call_args_list == [
        call("Permission requested.", "allow (session)"),
        call("Permission requested.", "allow"),
    ]
    interaction.run_metrics.assert_called_once_with(metrics)


def test_manager_loads_a_session_identifier_during_initialization():
    """A session identifier is resolved through the configured store at construction."""
    store = Mock(spec=SessionStore)
    loaded = Session(id="session-id")
    store.load.return_value = loaded

    manager = SessionManager(session="session-id", session_store=store)

    assert manager.session is loaded
    store.load.assert_called_once_with("session-id")


def test_manager_rejects_an_empty_workspace_identifier():
    """A configured workspace identity must be non-empty."""
    with pytest.raises(ValueError, match="must not be empty"):
        SessionManager(workspace_id="")


def test_manager_binds_new_and_unowned_sessions_to_its_workspace():
    """A workspace-aware manager assigns its durable ID to every accepted session."""
    manager = SessionManager(workspace_id="workspace")
    assert manager.session.workspace_id == "workspace"

    loaded = Session(id="loaded")
    manager.load_session(loaded)
    assert loaded.workspace_id == "workspace"

    owned = Session(id="owned", workspace_id="workspace")
    manager.load_session(owned)
    assert manager.session is owned

    manager.new_session()
    assert manager.session.workspace_id == "workspace"


def test_manager_rejects_sessions_owned_by_another_workspace():
    """Cross-workspace session loading fails before replacing the active session."""
    manager = SessionManager(workspace_id="workspace")
    original = manager.session

    with pytest.raises(SessionWorkspaceMismatchError, match="belongs to workspace"):
        manager.load_session(Session(workspace_id="another-workspace"))

    assert manager.session is original


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


def test_manager_exposes_durable_recovery_classification():
    """The manager exposes session-owned recovery state without applying execution policy."""
    session = Session(messages=[Message(role="user", content="unfinished")])
    manager = SessionManager(session=session)

    assert manager.recovery_state.action == "query_model"
    manager.new_session()
    assert manager.recovery_state is None


def test_manager_rejects_overlapping_execution_and_releases_normal_ownership():
    """One process cannot own the same logical session twice and releases a completed lease."""
    first = SessionManager(session=Session(id="shared", workspace_id="workspace"))
    second = SessionManager(session=Session(id="shared", workspace_id="workspace"))

    with (
        first.execution(),
        pytest.raises(SessionExecutionConflictError, match="already executing"),
        second.execution(),
    ):
        pass

    with second.execution():
        pass


@pytest.mark.parametrize("error", [RuntimeError("failed"), KeyboardInterrupt()])
def test_manager_releases_execution_ownership_after_failure_or_cancellation(error):
    """Exceptions and cancellation cannot strand a process-local session lease."""
    first = SessionManager(session=Session(id="shared", workspace_id="workspace"))
    second = SessionManager(session=Session(id="shared", workspace_id="workspace"))

    with pytest.raises(type(error)), first.execution():
        raise error

    with second.execution():
        pass


def test_manager_allows_separate_sessions_to_execute_concurrently():
    """Ownership isolation does not serialize executions of different logical sessions."""
    first = SessionManager(session=Session(id="first", workspace_id="workspace"))
    second = SessionManager(session=Session(id="second", workspace_id="workspace"))

    with first.execution(), second.execution():
        pass


def test_manager_does_not_conflate_sessions_from_independent_memory_stores():
    """Equal IDs without workspace identity remain independent across isolated stores."""
    first = SessionManager(session=Session(id="shared"))
    second = SessionManager(session=Session(id="shared"))

    with first.execution(), second.execution():
        pass


def test_manager_rejects_same_session_execution_across_threads():
    """A second thread cannot inherit or bypass an active session owner."""
    first = SessionManager(session=Session(id="shared", workspace_id="workspace"))
    second = SessionManager(session=Session(id="shared", workspace_id="workspace"))
    acquired = Event()
    release = Event()

    def hold_execution() -> None:
        """Hold ownership until the competing execution has been attempted."""
        with first.execution():
            acquired.set()
            release.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(hold_execution)
        assert acquired.wait(timeout=5)
        with pytest.raises(SessionExecutionConflictError), second.execution():
            pass
        release.set()
        future.result(timeout=5)


def test_manager_allows_different_sessions_across_threads():
    """Distinct session ownership does not suppress actual threaded concurrency."""
    barrier = Barrier(2)
    managers = [
        SessionManager(session=Session(id=session_id, workspace_id="workspace"))
        for session_id in ("first", "second")
    ]

    def overlap(manager: SessionManager) -> None:
        """Reach a shared boundary while retaining the session lease."""
        with manager.execution():
            barrier.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(overlap, manager) for manager in managers]
        for future in futures:
            future.result(timeout=5)


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

    _, version = content_identity("pass\n")
    assert manager.messages == [
        Message(
            role="user",
            content="Review @app.py",
            context=(
                reference.model_copy(
                    update={"handle": manager.messages[0].context[0].handle, "version": version}
                ),
            ),
        )
    ]
    assert session.name == "Review @app.py"
    assert session.name_source == "initial"
    store.save.assert_called_once()


def test_manager_does_not_commit_when_reference_cache_activation_fails():
    """A cache failure leaves both the active session and durable store unchanged."""
    store = Mock(spec=SessionStore)
    manager = SessionManager(session_store=store)
    reference = ContextReference(
        kind="file",
        path="app.py",
        content="pass\n",
        size_bytes=5,
        included_bytes=5,
        truncated=False,
    )

    with (
        patch("loop.session.session_manager.store_content", side_effect=OSError("cache failed")),
        pytest.raises(OSError, match="cache failed"),
    ):
        manager.add_user_message("Review @app.py", context=(reference,))

    assert manager.messages == []
    store.save.assert_not_called()


def test_manager_rejects_mismatched_reference_before_overwriting_its_cache():
    """Rejected reference bytes leave an existing immutable cache entry untouched."""
    trusted = b"trusted"
    handle, version = content_identity(trusted)
    store_content(trusted, "mentioned file source.txt", handle=handle)
    store = Mock(spec=SessionStore)
    manager = SessionManager(session_store=store)
    reference = ContextReference(
        kind="file",
        path="source.txt",
        content="changed",
        size_bytes=len(trusted),
        included_bytes=len(trusted),
        truncated=False,
        handle=handle,
        version=version,
    )

    with pytest.raises(ValueError, match="integrity validation"):
        manager.add_user_message("Review @source.txt", context=(reference,))

    assert cached_path(handle)[0].read_bytes() == trusted
    assert manager.messages == []
    store.save.assert_not_called()


def test_manager_rejects_a_handle_rebound_to_different_content_before_cache_activation():
    """A duplicate capability cannot overwrite another reference's cached content."""
    store = Mock(spec=SessionStore)
    manager = SessionManager(session_store=store)
    handle, first_version = content_identity("first")
    _, second_version = content_identity("second")
    first = ContextReference(
        kind="file",
        path="first.txt",
        content="first",
        size_bytes=5,
        included_bytes=5,
        truncated=False,
        handle=handle,
        version=first_version,
    )
    second = first.model_copy(
        update={
            "path": "second.txt",
            "content": "second",
            "size_bytes": 6,
            "included_bytes": 6,
            "version": second_version,
        }
    )

    with pytest.raises(ValueError, match="reuses an artifact handle"):
        manager.add_user_message("Review", context=(first, second))

    assert manager.messages == []
    store.save.assert_not_called()


def test_manager_generates_names_regardless_of_the_current_name_source():
    """A completed exchange may replace both provisional and user-controlled names."""
    interaction = Mock(spec=Interaction)
    store = Mock(spec=SessionStore)
    generator = Mock()
    generator.generate.return_value = "Review application architecture"
    session = Session()
    manager = SessionManager(interaction=interaction, session=session, session_store=store)
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
    assert interaction.info.call_args_list == [
        call("Generating a session name..."),
        call("Session name: Review application architecture"),
        call("Generating a session name..."),
        call("Session name: Review application architecture"),
    ]


def test_manager_renames_persisted_sessions_without_switching_the_active_session():
    """Renaming stored sessions preserves the active session unless it is the target."""
    store = MemorySessionStore()
    active = Session(name="Active", name_source="user")
    stored = Session(name="Stored", name_source="user")
    store.save(active)
    stored_id = store.save(stored)
    manager = SessionManager(session=active, session_store=store)

    manager.rename_persisted_session(stored_id, "Renamed stored")
    manager.rename_persisted_session(active.id, "Renamed active")

    assert manager.session is active
    assert manager.session.name == "Renamed active"
    assert store.load(stored_id).name == "Renamed stored"


def test_manager_reports_name_generation_failures(monkeypatch):
    """Name generation reports a retryable problem and preserves the current name on failure."""
    interaction = Mock(spec=Interaction)
    generator = Mock()
    error = RuntimeError("Generator is unavailable")
    generator.generate.side_effect = error
    logger = Mock()
    monkeypatch.setattr(session_manager_module, "_LOGGER", logger)
    session = Session()
    manager = SessionManager(interaction=interaction, session=session)
    manager.add_user_message("Please review this app")
    manager.add_message(Message(role="assistant", content="The architecture is sound."))

    manager.generate_session_name(generator)

    problem = interaction.report.call_args.args[0]
    assert problem.code == "session.name_generation_failed"
    assert problem.title == "Could not generate session name"
    assert problem.detail == "Could not generate the session name."
    assert problem.severity == "warning"
    assert problem.retryable is True
    assert problem.operation == "generate_session_name"
    assert session.name == "Please review this app"
    interaction.info.assert_called_once_with("Generating a session name...")
    assert logger.log.call_args.kwargs["extra"]["exception.type"] == "builtins.RuntimeError"
    assert "exc_info" not in logger.log.call_args.kwargs


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

    assert manager.session.id != "old"
    assert manager.session.messages == []
    assert manager.session.name is None
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
    call = ToolCall(call_id="call-id", name="demo", arguments="{}")
    session = Session(messages=[call])
    manager = SessionManager(session=session, session_store=store)
    active_skills = iter([("review", "/skills/review/SKILL.md")])

    manager.add_tool_call(
        "call-id",
        "result",
        "/project",
        active_skills,
        succeeded=True,
        duration_seconds=0.25,
    )

    assert manager.messages == [call, ToolResult(call_id="call-id", output="result")]
    assert session.instruction_working_directory == "/project"
    assert session.active_skills == [("review", "/skills/review/SKILL.md")]
    completed = session.events[-1]
    assert isinstance(completed, ToolExecutionCompletedEvent)
    assert completed.call_id == "call-id"
    assert completed.succeeded is True
    assert completed.duration_seconds == 0.25
    store.save.assert_called_once_with(session)


def test_manager_records_tool_execution_start_only_for_known_calls():
    """Execution-start checkpoints are durable and reject unknown model requests."""
    store = Mock(spec=SessionStore)
    call = ToolCall(call_id="call", name="demo", arguments="{}")
    session = Session(messages=[call])
    manager = SessionManager(session=session, session_store=store)

    manager.record_tool_execution_started("call")

    assert isinstance(session.events[-1], ToolExecutionStartedEvent)
    assert session.events[-1].call_id == "call"
    store.save.assert_called_once_with(session)
    with pytest.raises(ValueError, match="Unknown tool call"):
        manager.record_tool_execution_started("missing")


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
        source="https://my-host.local/source.txt",
        reloadable=True,
    )
    session = Session(messages=[ToolResult(call_id="call", output="result", artifacts=(artifact,))])

    SessionManager(session=session)

    assert cached_metadata(handle) == {
        "source": "https://my-host.local/source.txt",
        "reloadable": True,
    }


def test_manager_restores_digest_backed_mention_content_from_its_store():
    """Loading repairs a missing cache file from its durable store artifact."""
    store = MemorySessionStore()
    session, handle, _version = persist_reference(store, b"snapshot")
    cached_path(handle)[0].unlink()

    SessionManager(session=session, session_store=store).load_session(session)

    assert cached_path(handle)[0].read_text(encoding="utf-8") == "snapshot"


def test_manager_repairs_tampered_cached_mention_content_from_its_store():
    """Loading replaces cached bytes that fail the reference integrity check."""
    store = MemorySessionStore()
    session, handle, _version = persist_reference(store, b"snapshot")
    cached_path(handle)[0].write_bytes(b"tampered")

    SessionManager(session=session, session_store=store).load_session(session)

    assert cached_path(handle)[0].read_bytes() == b"snapshot"


def test_manager_repairs_cached_mention_content_after_a_cache_read_failure():
    """Loading repairs a durable reference when cache verification cannot read its file."""
    store = MemorySessionStore()
    session, handle, _version = persist_reference(store, b"snapshot")

    with patch.object(type(cached_path(handle)[0]), "read_bytes", side_effect=OSError):
        SessionManager(session=session, session_store=store).load_session(session)

    assert cached_path(handle)[0].read_bytes() == b"snapshot"


def test_manager_rejects_missing_durable_content_despite_a_valid_cache():
    """Session activation requires durable content even when a matching cache file remains."""
    content = b"snapshot"
    handle, version = content_identity(content)
    store_content(content, "mentioned file source.txt", handle=handle)
    store = MagicMock(spec=SessionStore)
    store.load_reference.return_value = None
    session = Session(
        messages=[
            Message(
                role="user",
                content="Review @source.txt",
                context=(
                    ContextReference(
                        kind="file",
                        path="source.txt",
                        size_bytes=len(content),
                        included_bytes=0,
                        truncated=True,
                        handle=handle,
                        next_cursor=encode_content_cursor(handle, 0),
                        version=version,
                    ),
                ),
            )
        ]
    )

    with pytest.raises(ValueError, match="unavailable"):
        SessionManager(session_store=store).load_session(session)


def test_manager_rejects_incomplete_artifact_identity_while_loading():
    """Session activation refuses a reference without both durable identity fields."""
    session = Session(
        messages=[
            Message(
                role="user",
                content="Review @source.txt",
                context=(
                    ContextReference(
                        kind="file",
                        path="source.txt",
                        size_bytes=0,
                        included_bytes=0,
                        truncated=False,
                    ),
                ),
            )
        ]
    )

    with pytest.raises(ValueError, match="incomplete artifact identity"):
        SessionManager().load_session(session)


def test_manager_stores_and_activates_content_addressed_references():
    """Capturing a reference persists it and makes its immutable bytes immediately readable."""
    store = MemorySessionStore()
    _session, handle, version = persist_reference(store, b"snapshot", "source.txt")

    assert version == f"sha256:{sha256_digest(b'snapshot')}"
    assert handle and handle != version[7:39]
    assert store.load_reference(version) == b"snapshot"
    assert cached_path(handle)[0].read_bytes() == b"snapshot"


def test_manager_does_not_save_current_sessions_while_loading():
    """Loading current snapshots performs no migration save or version scan."""
    session = Session()
    store = Mock(spec=SessionStore)
    store.load.return_value = session

    SessionManager(session_store=store).load_session(session.id)

    store.save.assert_not_called()


def test_manager_persists_contentless_occurrences_and_hydrates_each_version_once():
    """Messages store only artifact metadata while active context supplies one verified payload."""
    store = MemorySessionStore()
    manager = SessionManager(session_store=store)
    content = b"shared snapshot"
    handle, version = content_identity(content)
    reference = ContextReference(
        kind="file",
        path="source.txt",
        content=content.decode(),
        size_bytes=len(content),
        included_bytes=len(content),
        truncated=False,
        handle=handle,
        version=version,
        media_type="text/plain",
    )

    manager.add_user_message("Review @source.txt", context=(reference,))
    manager.add_message(Reasoning(content="considering"))
    manager.add_user_message("Revisit @source.txt", context=(reference,))

    persisted = store.load(manager.session.id)
    assert persisted.messages[0].context[0].content == ""
    assert "shared snapshot" not in persisted.serialize()
    first, reasoning, second = manager.model_context
    assert first.context[0].content == "shared snapshot"
    assert first.context[0].included_bytes == len(content)
    assert first.context[0].reused is False
    assert second.context[0].content == ""
    assert second.context[0].included_bytes == len(content)
    assert second.context[0].payload_start_bytes == len(content)
    assert second.context[0].reused is True
    assert reasoning == Reasoning(content="considering")


def test_manager_model_context_rejects_content_without_artifact_identity():
    """Hydration refuses a canonical occurrence without durable artifact identity."""
    manager = SessionManager()
    reference = ContextReference(
        kind="file",
        path="legacy.txt",
        content="legacy",
        size_bytes=6,
        included_bytes=6,
        truncated=False,
        version="a" * 64,
    )
    manager.session.add_message(Message(role="user", content="Review", context=(reference,)))
    with pytest.raises(ValueError, match="artifact handle"):
        _ = manager.model_context


def test_manager_hydrates_a_payload_after_an_earlier_metadata_only_occurrence():
    """A lazy occurrence does not suppress a later requested payload of the same version."""
    store = MemorySessionStore()
    manager = SessionManager(session_store=store)
    content = b"snapshot"
    handle, version = content_identity(content)
    lazy = ContextReference(
        kind="file",
        path="source.txt",
        content=content.decode(),
        size_bytes=len(content),
        included_bytes=0,
        truncated=True,
        handle=handle,
        next_cursor=encode_content_cursor(handle, 0),
        version=version,
    )
    eager = lazy.model_copy(
        update={"included_bytes": len(content), "truncated": False, "next_cursor": None}
    )
    manager.add_user_message("Note @source.txt", context=(lazy,))
    manager.add_user_message("Read @source.txt", context=(eager,))

    first, second = manager.model_context
    assert first.context[0].content == ""
    assert first.context[0].reused is False
    assert second.context[0].content == "snapshot"
    assert second.context[0].reused is False


def test_manager_hydrates_only_the_unseen_suffix_of_an_overlapping_reference():
    """A larger later prefix supplies only bytes not present in the earlier occurrence."""
    store = MemorySessionStore()
    manager = SessionManager(session_store=store)
    content = b"abcdefgh"
    handle, version = content_identity(content)
    first = ContextReference(
        kind="file",
        path="source.txt",
        content=content.decode(),
        size_bytes=len(content),
        included_bytes=4,
        truncated=True,
        handle=handle,
        next_cursor=encode_content_cursor(handle, 4),
        version=version,
    )
    second = first.model_copy(
        update={
            "included_bytes": len(content),
            "truncated": False,
            "next_cursor": None,
        }
    )
    manager.add_user_message("Review @source.txt", context=(first,))
    manager.add_user_message("Read more @source.txt", context=(second,))

    first_message, second_message = manager.model_context
    first_reference = first_message.context[0]
    second_reference = second_message.context[0]
    assert first_reference.content == "abcd"
    assert first_reference.payload_start_bytes == 0
    assert second_reference.content == "efgh"
    assert second_reference.included_bytes == len(content)
    assert second_reference.payload_start_bytes == 4
    assert second_reference.reused is False


def test_manager_normalizes_occurrence_cursors_and_rejects_invalid_metadata():
    """Capture canonicalizes cursors and rejects bad identities, sizes, and UTF-8 boundaries."""
    manager = SessionManager()
    handle, version = content_identity("é")
    base = ContextReference(
        kind="file",
        path="source.txt",
        content="é",
        size_bytes=2,
        included_bytes=1,
        truncated=True,
        handle=handle,
        next_cursor=encode_content_cursor(handle, 1),
        version=version,
    )
    for reference, message in (
        (base.model_copy(update={"version": None}), "provided together"),
        (base.model_copy(update={"size_bytes": 3}), "size"),
        (base.model_copy(update={"included_bytes": 3}), "included bytes"),
        (base.model_copy(update={"truncated": False}), "truncation"),
    ):
        with pytest.raises(ValueError, match=message):
            manager.add_user_message("Review", context=(reference,))

    manager.add_user_message("Review", context=(base,))
    with pytest.raises(ValueError, match="UTF-8 boundary"):
        _ = manager.model_context

    normalized = SessionManager()
    normalized.add_user_message("Review", context=(base.model_copy(update={"next_cursor": None}),))
    assert normalized.messages[0].context[0].next_cursor == encode_content_cursor(handle, 1)


def test_manager_hydrates_binary_reference_payloads_as_bytes():
    """A non-text artifact preserves its exact bytes through active-context hydration."""
    manager = SessionManager(session_store=MemorySessionStore())
    content = b"\xff"
    handle, version = content_identity(content)
    reference = ContextReference(
        kind="file",
        path="binary.bin",
        content=content,
        size_bytes=len(content),
        included_bytes=len(content),
        truncated=False,
        handle=handle,
        version=version,
        media_type="application/octet-stream",
    )

    manager.add_user_message("Review @binary.bin", context=(reference,))

    assert manager.model_context[0].context[0].content == content


def test_manager_model_context_requires_a_complete_artifact_identity():
    """Hydration rejects a handle without its corresponding digest."""
    manager = SessionManager()
    manager.session.add_message(
        Message(
            role="user",
            content="Review",
            context=(
                ContextReference(
                    kind="file",
                    path="source.txt",
                    size_bytes=0,
                    included_bytes=0,
                    truncated=False,
                    handle=uuid4().hex,
                ),
            ),
        )
    )
    with pytest.raises(ValueError, match="artifact identity"):
        _ = manager.model_context


def test_manager_rejects_noncapability_reference_handles():
    """Hydration and capture reject malformed content-read capabilities."""
    _, version = content_identity("source")
    reference = ContextReference(
        kind="file",
        path="source.txt",
        content="source",
        size_bytes=6,
        included_bytes=6,
        truncated=False,
        handle="invalid",
        version=version,
    )
    manager = SessionManager()
    manager.session.add_message(Message(role="user", content="Review", context=(reference,)))

    with pytest.raises(ValueError, match="invalid artifact handle"):
        _ = manager.model_context
    with pytest.raises(ValueError, match="invalid artifact handle"):
        manager.add_user_message("Review", context=(reference,))


@pytest.mark.parametrize("failure", ["missing", "wrong-size"])
def test_manager_model_context_rejects_unusable_artifacts(failure):
    """Hydration fails closed when durable occurrence content is absent or inconsistent."""
    content = b"snapshot"
    handle, version = content_identity(content)
    store = MagicMock(spec=SessionStore)
    store.load_reference.return_value = None if failure == "missing" else content
    manager = SessionManager(session_store=store)
    manager.session.add_message(
        Message(
            role="user",
            content="Review",
            context=(
                ContextReference(
                    kind="file",
                    path="source.txt",
                    size_bytes=len(content) + (failure == "wrong-size"),
                    included_bytes=len(content),
                    truncated=False,
                    handle=handle,
                    version=version,
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="unavailable|integrity validation"):
        _ = manager.model_context


def test_manager_rejects_a_restored_reference_with_the_wrong_digest():
    """Session loading refuses durable reference bytes that do not match their manifest."""
    handle = "0123456789abcdef0123456789abcdef"
    store = MagicMock(spec=SessionStore)
    store.load_reference.return_value = b"tampered"
    session = Session(
        messages=[
            Message(
                role="user",
                content="Review @source.txt",
                context=(
                    ContextReference(
                        kind="file",
                        path="source.txt",
                        content="",
                        size_bytes=8,
                        included_bytes=0,
                        truncated=True,
                        handle=handle,
                        version=f"sha256:{'0' * 64}",
                    ),
                ),
            )
        ]
    )

    with pytest.raises(ValueError, match="integrity validation"):
        SessionManager(session_store=store).load_session(session)


def test_manager_preserves_the_active_session_when_reference_restoration_fails():
    """A failed candidate load leaves the previously active session available."""
    handle = "0123456789abcdef0123456789abcdef"
    store = MagicMock(spec=SessionStore)
    store.load_reference.return_value = b"tampered"
    active = Session(messages=[Message(role="user", content="continue current work")])
    candidate = Session(
        messages=[
            Message(
                role="user",
                content="Review @source.txt",
                context=(
                    ContextReference(
                        kind="file",
                        path="source.txt",
                        size_bytes=8,
                        included_bytes=0,
                        truncated=True,
                        handle=handle,
                        version=f"sha256:{'0' * 64}",
                    ),
                ),
            )
        ]
    )
    manager = SessionManager(session=active, session_store=store)

    with pytest.raises(ValueError, match="integrity validation"):
        manager.load_session(candidate)

    assert manager.session is active
    assert manager.messages == [Message(role="user", content="continue current work")]


def test_manager_rejects_a_restored_reference_with_an_invalid_store_digest():
    """Session loading verifies the durable digest, independent of the session manifest."""
    handle = "0123456789abcdef0123456789abcdef"
    content = b"snapshot"
    store = MagicMock(spec=SessionStore)
    store.load_reference.return_value = b"tampered"
    session = Session(
        messages=[
            Message(
                role="user",
                content="Review @source.txt",
                context=(
                    ContextReference(
                        kind="file",
                        path="source.txt",
                        content="",
                        size_bytes=len(content),
                        included_bytes=0,
                        truncated=True,
                        handle=handle,
                        version=f"sha256:{sha256_digest(content)}",
                    ),
                ),
            )
        ]
    )

    with pytest.raises(ValueError, match="integrity validation"):
        SessionManager(session_store=store).load_session(session)


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
    reference = CapturedInstruction.capture(
        "/project/AGENTS.md",
        "project rules",
        workspace_id="workspace-id",
        workspace_root="/project",
    )

    manager.add_compaction(
        result,
        model="model",
        instructions="project rules",
        working_directory="/project",
        active_skills=iter([("review", "/skills/review/SKILL.md")]),
        references=(reference,),
    )

    checkpoint = session.compactions[0]
    assert manager.model_context == [*checkpoint.context]
    assert checkpoint.boundary == 1
    assert checkpoint.instructions.content == "project rules"
    assert checkpoint.instructions.active_skills == (("review", "/skills/review/SKILL.md"),)
    assert checkpoint.instructions.references == (reference,)
    assert Session.deserialize(session.serialize()).compactions[0].instructions.references == (
        reference,
    )
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


def test_manager_records_persistence_failure_without_changing_atomicity():
    """Session persistence emits isolated failure diagnostics without changing atomicity."""
    store = Mock(spec=SessionStore)
    manager = SessionManager(session_store=store)
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    try:
        manager.add_message(Message(role="assistant", content="complete"))
        store.save.side_effect = OSError("disk full")
        with pytest.raises(OSError, match="disk full"):
            manager.add_message(Message(role="assistant", content="not persisted"))
        assert telemetry.close(1)
    finally:
        set_telemetry(None)

    assert [record.event_name for record in adapter.records] == [
        "session.persistence.started",
        "session.persistence.completed",
        "session.persistence.started",
        "session.persistence.failed",
    ]
    assert manager.messages == [Message(role="assistant", content="complete")]
