"""Tests for session coordination and persistence."""

import json
from unittest.mock import Mock
from uuid import uuid4

import pytest

from loop import (
    ConsoleInteraction,
    ContentArtifact,
    MemorySessionStore,
    Message,
    Response,
    Session,
    SessionManager,
    ToolResult,
)
from loop.interaction import Interaction
from loop.session import SessionStore
from loop.utils import cached_metadata


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


def test_manager_builds_and_persists_a_user_message():
    """The user-message convenience method creates and persists its conversation item."""
    store = Mock(spec=SessionStore)
    session = Session()
    manager = SessionManager(session=session, session_store=store)

    manager.add_user_message("hello")

    assert manager.messages == [Message(role="user", content="hello")]
    store.save.assert_called_once_with(session)


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
    session = Session(
        messages=[ToolResult(call_id="call", output="result", artifacts=(artifact,))]
    )

    SessionManager(session=session)

    assert cached_metadata(handle) == {
        "source": "https://example.com/source.txt",
        "reloadable": True,
    }


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


def test_manager_adds_a_response_and_persists_its_session_updates():
    """Adding a response records its items and metadata before persistence."""
    store = Mock(spec=SessionStore)
    session = Session()
    manager = SessionManager(session=session, session_store=store)
    answer = Message(role="assistant", content="done")
    response = Response(answer="done", reasoning="", items=(answer,), model="model-b")

    manager.add_response(response)

    assert manager.messages == [answer]
    assert manager.model == "model-b"
    store.save.assert_called_once_with(session)


def test_manager_exposes_token_usage_and_allows_model_selection():
    """Session metadata remains available and model selection updates the active session."""
    session = Session(tokens=17)
    manager = SessionManager(session=session)

    manager.model = "selected-model"

    assert manager.model == "selected-model"
    assert manager.tokens == 17


@pytest.mark.parametrize("method", ["add_message", "add_messages"])
def test_manager_does_not_persist_rejected_items(method):
    """Validation failures leave persistence untouched for both insertion interfaces."""
    store = Mock(spec=SessionStore)
    manager = SessionManager(session_store=store)
    value = object() if method == "add_message" else [object()]

    with pytest.raises(ValueError, match="Expected a conversation item"):
        getattr(manager, method)(value)

    store.save.assert_not_called()
