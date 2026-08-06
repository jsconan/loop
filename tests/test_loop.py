"""Tests for normalized response handling and conversation orchestration."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop import (
    AnswerCompleted,
    AnswerDelta,
    Interaction,
    Loop,
    Message,
    Reasoning,
    ReasoningCompleted,
    ReasoningDelta,
    Response,
    ResponseCompleted,
    Session,
    SQLiteSessionStore,
    ToolCall,
    ToolCallCompleted,
    ToolRegistry,
    ToolResult,
    Usage,
)
from loop import tool_registry as default_tool_registry


def function_call() -> ToolCall:
    """Build a completed local function-tool call."""
    return ToolCall(call_id="call_123", name="get_current_datetime", arguments="{}", id="fc_123")


def loop_backend(**attributes):
    """Build a minimal backend satisfying the loop contract."""
    defaults = {
        "tool_registry": default_tool_registry,
        "default_model": "default-model",
        "get_context_window": lambda _model: None,
    }
    return SimpleNamespace(**(defaults | attributes))


def test_loop_exposes_its_configured_state(tmp_path):
    """Loop accessors expose configured dependencies and mutable state."""
    backend = loop_backend()
    interaction = Mock(spec=Interaction)
    loop = Loop(
        backend=backend,
        model="requested-model",
        debug=True,
        interaction=interaction,
        working_directory=tmp_path,
    )

    assert loop.backend is backend
    assert loop.messages == []
    assert loop.debug is True
    assert loop.stream is False
    assert loop.interaction is interaction
    assert loop.working_directory == tmp_path.resolve()
    assert loop.instructions is None
    assert loop.session == Session()
    assert loop.model == "requested-model"
    assert loop.skill_manager is not None

    loop.debug = False
    assert loop.debug is False


def test_loops_share_local_conversation_context(tmp_path):
    """Injected context carries local history and metadata between loop modes."""
    session = Session(
        messages=[Message(role="user", content="hello")], tokens=12, model="served-model"
    )
    first = Loop(backend=loop_backend(), session=session, working_directory=tmp_path)
    second_backend = Mock(default_model="other-model")
    second_backend.get_response.return_value = []
    second = Loop(backend=second_backend, session=session, working_directory=tmp_path, stream=True)

    assert first.session is second.session is session
    assert second.messages == [Message(role="user", content="hello")]
    assert second.session.tokens == 12
    assert second.session.model == "served-model"
    assert list(second.query()) == []
    second_backend.get_response.assert_called_once_with(
        input=session.messages,
        instructions=None,
        stream=True,
        model="other-model",
    )


def test_new_session_is_not_persisted_until_its_first_completed_query(tmp_path):
    """A fresh session creates storage only after a query result forms a complete snapshot."""
    backend = loop_backend(
        get_response=Mock(
            return_value=[
                ResponseCompleted(
                    items=(Message(role="assistant", content="answer"),),
                    usage=Usage(total_tokens=9),
                    model="served-model",
                )
            ]
        )
    )
    interaction = Mock(spec=Interaction)
    interaction.response.return_value = nullcontext()
    interaction.input.side_effect = ["hello", False]
    store = SQLiteSessionStore(tmp_path / ".loop" / "sessions.db")
    loop = Loop(
        backend=backend,
        working_directory=tmp_path,
        interaction=interaction,
        session_store=store,
    )

    assert not (tmp_path / ".loop").exists()

    loop.run()

    session_info = store.list()[0]
    assert store.load(session_info.id) == loop.session
    assert loop.session.messages == [
        Message(role="user", content="hello"),
        Message(role="assistant", content="answer"),
    ]
    assert loop.session.tokens == 9
    assert loop.session.model == "served-model"


def test_loop_without_a_session_store_never_creates_session_files(tmp_path):
    """A caller that omits persistence keeps completed queries entirely in memory."""
    interaction = Mock(spec=Interaction)
    interaction.response.return_value = nullcontext()
    interaction.input.side_effect = ["hello", False]
    loop = Loop(
        backend=loop_backend(get_response=Mock(return_value=[ResponseCompleted()])),
        working_directory=tmp_path,
        interaction=interaction,
    )

    loop.run()

    assert not (tmp_path / ".loop").exists()


def test_persisted_session_identifier_uses_the_default_memory_store(monkeypatch):
    """A persisted identifier is resolved through an instance-local memory store by default."""
    stored = Session(messages=[Message(role="user", content="saved")])
    store = Mock()
    store.load.return_value = stored
    store_factory = Mock(return_value=store)
    monkeypatch.setattr("loop.loop.MemorySessionStore", store_factory)

    loop = Loop(backend=loop_backend(), session="session-id")

    assert loop.session is stored
    store_factory.assert_called_once_with()
    store.load.assert_called_once_with("session-id")


def test_loop_loads_a_persisted_session_identifier(tmp_path):
    """The constructor accepts a session identifier and resumes its complete state."""
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    stored = Session(
        messages=[Message(role="user", content="saved")], tokens=4, model="saved-model"
    )
    session_id = store.save(stored)

    loop = Loop(backend=loop_backend(), session=session_id, session_store=store)

    assert loop.session == stored


def test_loop_loads_project_instructions_once(monkeypatch, tmp_path):
    """A loop retains instructions loaded for its normalized working directory."""
    loader = Mock(return_value="project rules")
    monkeypatch.setattr("loop.loop.load_agents_instructions", loader)

    loop = Loop(backend=loop_backend(), working_directory=str(tmp_path))

    assert loop.instructions == "project rules"
    loader.assert_called_once_with(tmp_path.resolve())


def test_run_requeries_after_a_tool_call_and_records_local_items(tmp_path):
    """The runner records a tool result, requeries, reports usage, and exits."""
    registry = ToolRegistry()

    @registry.tool
    def echo(text: str) -> str:
        """Echo text."""
        return text

    call = ToolCall(call_id="call", name="echo", arguments='{"text":"done"}', id="fc")
    backend = Mock(tool_registry=registry, default_model="requested-model")
    backend.get_context_window.return_value = 1000
    backend.get_response.side_effect = [
        [
            ToolCallCompleted(call=call),
            ResponseCompleted(items=(call,), usage=Usage(total_tokens=10)),
        ],
        [
            AnswerDelta(text="done"),
            AnswerCompleted(text="done"),
            ResponseCompleted(
                items=(Message(role="assistant", content="done"),),
                usage=Usage(total_tokens=12),
                answer="done",
            ),
        ],
    ]
    interaction = Mock(spec=Interaction)
    interaction.response.return_value = nullcontext()
    interaction.input.side_effect = ["hello", False]

    Loop(backend=backend, interaction=interaction, working_directory=tmp_path).run()

    second_input = backend.get_response.call_args_list[1].kwargs["input"]
    assert second_input[:3] == [
        Message(role="user", content="hello"),
        call,
        ToolResult(call_id="call", output="done"),
    ]
    assert second_input[-1] == Message(role="assistant", content="done")
    interaction.answer_delta.assert_called_once_with("done", start=True)
    assert interaction.response.call_count == 2
    interaction.token_usage.assert_called_once_with("requested-model", 12, 1000)
    interaction.conversation_ended.assert_called_once_with()


def test_run_keeps_handled_commands_out_of_model_history():
    """The runner skips every command consumed by its command manager."""
    backend = Mock(tool_registry=ToolRegistry(), default_model="model")
    interaction = Mock(spec=Interaction)
    interaction.input.side_effect = ["/help", "/missing", False]

    loop = Loop(backend=backend, interaction=interaction)
    loop.run()

    assert loop.messages == []
    backend.get_response.assert_not_called()


@pytest.mark.parametrize("command", ["/exit", "/quit"])
def test_run_exit_commands_end_the_conversation(command):
    """Predefined slash exit commands terminate without a backend request."""
    backend = Mock(tool_registry=ToolRegistry(), default_model="model")
    interaction = Mock(spec=Interaction)
    interaction.input.return_value = command

    loop = Loop(backend=backend, interaction=interaction)
    loop.run()

    assert loop.messages == []
    backend.get_response.assert_not_called()
    interaction.conversation_ended.assert_called_once_with()


def test_handle_tool_calls_uses_local_objects():
    """Tool results remain typed in context."""
    registry = Mock()
    registry.call.return_value = "tool result"
    backend = loop_backend(tool_registry=registry, get_response=Mock(return_value=[]))
    loop = Loop(backend=backend)
    call = function_call()
    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))

    assert loop.handle_tool_calls(response) is True

    assert loop.messages == [ToolResult(call_id="call_123", output="tool result")]
    registry.call.assert_called_once_with(
        call.name,
        call.arguments,
        interaction=loop.interaction,
        skill_manager=loop.skill_manager,
    )
    assert loop.handle_tool_calls(Response(answer="", reasoning="")) is False


def test_query_selects_only_the_event_production_mode():
    """Both loop modes forward identical history with only the stream flag differing."""
    backend = loop_backend(get_response=Mock(return_value=[]))
    session = Session(messages=[Message(role="user", content="hello")])

    list(Loop(backend=backend, session=session).query())
    list(Loop(backend=backend, session=session, stream=True).query())

    assert backend.get_response.call_args_list[0].kwargs["stream"] is False
    assert backend.get_response.call_args_list[1].kwargs["stream"] is True


def test_query_prefers_the_explicit_model_over_response_metadata():
    """Request selection stays independent of a model reported by an earlier response."""
    backend = loop_backend(get_response=Mock(return_value=[]))
    session = Session(model="served-model")

    list(Loop(backend=backend, model="requested-model", session=session).query())

    assert backend.get_response.call_args.kwargs["model"] == "requested-model"


def test_query_rejects_missing_model_selection():
    """A query fails clearly when neither the loop nor backend selects a model."""
    backend = loop_backend(default_model=None, get_response=Mock())

    with pytest.raises(ValueError, match="No model was selected"):
        Loop(backend=backend).query()

    backend.get_response.assert_not_called()


def test_one_output_loop_uses_terminal_response_text(capsys):
    """Streaming output displays deltas but returns authoritative terminal text."""
    interaction = Mock(spec=Interaction)
    interaction.response.return_value = nullcontext()
    call = function_call()
    items = (
        Reasoning(content="think again", id="r"),
        Message(role="assistant", content="hello world"),
        call,
    )
    events = [
        ReasoningDelta(text="incomplete "),
        ReasoningDelta(text="thought"),
        AnswerDelta(text="incomplete "),
        AnswerDelta(text="answer"),
        ToolCallCompleted(call=call),
        SimpleNamespace(ignored=True),
        ResponseCompleted(
            items=items,
            usage=Usage(total_tokens=230),
            model="served-model",
            answer="  hello world  ",
            reasoning="  think again  ",
        ),
    ]
    loop = Loop(backend=loop_backend(), interaction=interaction, debug=True)

    response = loop.output(events)

    assert response == Response(
        answer="  hello world  ",
        reasoning="  think again  ",
        tool_calls=(call,),
        items=items,
        usage=Usage(total_tokens=230),
        model="served-model",
    )
    assert loop.session == Session()
    assert interaction.reasoning_delta.call_args_list[0].kwargs == {"start": True}
    assert interaction.reasoning_delta.call_args_list[1].kwargs == {"start": False}
    assert interaction.answer_delta.call_args_list[0].kwargs == {"start": True}
    assert interaction.answer_delta.call_args_list[1].kwargs == {"start": False}
    interaction.response.assert_called_once_with()
    assert interaction.debug.call_count == len(events)
    capsys.readouterr()


def test_output_displays_non_streaming_completed_text():
    """Completed text is displayed directly when no streaming deltas were received."""
    interaction = Mock(spec=Interaction)
    interaction.response.return_value = nullcontext()

    response = Loop(backend=loop_backend(), interaction=interaction).output(
        [
            ReasoningCompleted(text="think"),
            AnswerCompleted(text="answer"),
            ResponseCompleted(answer="answer", reasoning="think"),
        ]
    )

    assert response == Response(answer="answer", reasoning="think")
    interaction.reasoning.assert_called_once_with("think")
    interaction.answer.assert_called_once_with("answer")
    interaction.response.assert_called_once_with()


def test_empty_output_preserves_existing_context_metadata():
    """A completion without reported metadata leaves existing context values unchanged."""
    session = Session(tokens=7, model="existing")
    interaction = Mock(spec=Interaction)
    interaction.response.return_value = nullcontext()

    response = Loop(backend=loop_backend(), session=session, interaction=interaction).output(
        [ResponseCompleted()]
    )

    assert response == Response(answer="", reasoning="")
    assert session.tokens == 7
    assert session.model == "existing"
    interaction.response.assert_called_once_with()


def test_end_uses_the_injected_interaction():
    """Conversation termination is delegated to the configured interaction."""
    interaction = Mock(spec=Interaction)
    Loop(backend=loop_backend(), interaction=interaction).end()
    interaction.conversation_ended.assert_called_once_with()
