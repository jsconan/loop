"""Tests for normalized response handling and conversation orchestration."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop import (
    AnswerCompleted,
    AnswerDelta,
    Interaction,
    LoopContext,
    Loop,
    Message,
    Reasoning,
    ReasoningCompleted,
    ReasoningDelta,
    Response,
    ResponseCompleted,
    ToolCall,
    ToolCallCompleted,
    ToolRegistry,
    ToolResult,
    Usage,
)
from loop import tool_registry as default_tool_registry


def function_call() -> ToolCall:
    """Build a completed local function-tool call."""
    return ToolCall(
        call_id="call_123", name="get_current_datetime", arguments="{}", id="fc_123"
    )


def loop_backend(**attributes):
    """Build a minimal backend satisfying the loop contract."""
    defaults = {
        "tool_registry": default_tool_registry,
        "default_model": "default-model",
        "get_context_window": lambda _model: None,
    }
    return SimpleNamespace(**(defaults | attributes))


def test_default_backend_receives_custom_tool_registry(monkeypatch):
    """A loop-created backend uses the registry supplied to the loop."""
    registry = ToolRegistry()
    created_backend = Mock()
    created_backend.get_response.return_value = [ResponseCompleted()]
    backend_factory = Mock(return_value=created_backend)
    monkeypatch.setattr("loop.loop.OpenAIBackend", backend_factory)

    loop = Loop(tool_registry=registry, stream=True)

    assert list(loop.query()) == [ResponseCompleted()]
    backend_factory.assert_called_once_with(tool_registry=registry)


def test_default_loop_uses_the_default_tool_registry(monkeypatch):
    """A default loop passes the populated shared registry to its backend."""
    created_backend = Mock(tool_registry=default_tool_registry)
    backend_factory = Mock(return_value=created_backend)
    monkeypatch.setattr("loop.loop.OpenAIBackend", backend_factory)

    Loop()

    backend_factory.assert_called_once_with(tool_registry=default_tool_registry)
    assert default_tool_registry.definitions()


def test_backend_and_tool_registry_cannot_both_be_supplied():
    """Ambiguous dependency injection is rejected explicitly."""
    with pytest.raises(ValueError, match="either backend or tool_registry"):
        Loop(backend=SimpleNamespace(), tool_registry=ToolRegistry())


def test_loop_exposes_its_configured_state(tmp_path):
    """Loop accessors expose configured dependencies and mutable state."""
    backend = loop_backend()
    interaction = Mock(spec=Interaction)
    loop = Loop(backend=backend, debug=True, interaction=interaction, working_directory=tmp_path)

    assert loop.backend is backend
    assert loop.messages == []
    assert loop.debug is True
    assert loop.stream is False
    assert loop.interaction is interaction
    assert loop.working_directory == tmp_path.resolve()
    assert loop.instructions is None
    assert loop.context == LoopContext(model="default-model")
    assert loop.skill_manager is not None

    loop.debug = False
    assert loop.debug is False


def test_loops_share_local_conversation_context(tmp_path):
    """Injected context carries local history and metadata between loop modes."""
    context = LoopContext(messages=[Message(role="user", content="hello")])
    first = Loop(backend=loop_backend(), context=context, working_directory=tmp_path)
    first.output([ResponseCompleted(usage=Usage(total_tokens=12), model="served-model")])
    second_backend = Mock(default_model="other-model")
    second_backend.get_response.return_value = []
    second = Loop(backend=second_backend, context=context, working_directory=tmp_path, stream=True)

    assert first.context is second.context is context
    assert second.messages == [Message(role="user", content="hello")]
    assert second.context.tokens == 12
    assert second.context.model == "served-model"
    assert list(second.query()) == []
    second_backend.get_response.assert_called_once_with(
        input=context.messages,
        instructions=None,
        stream=True,
    )


def test_loop_loads_project_instructions_once(monkeypatch, tmp_path):
    """A loop retains instructions loaded for its normalized working directory."""
    loader = Mock(return_value="project rules")
    monkeypatch.setattr("loop.loop.load_agents_instructions", loader)

    loop = Loop(backend=loop_backend(), working_directory=str(tmp_path))

    assert loop.instructions == "project rules"
    loader.assert_called_once_with(tmp_path.resolve())


def test_run_requeries_after_a_tool_call_and_records_local_items():
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
    interaction.input.side_effect = ["hello", False]

    Loop(backend=backend, interaction=interaction).run()

    second_input = backend.get_response.call_args_list[1].kwargs["input"]
    assert second_input[:3] == [
        Message(role="user", content="hello"),
        call,
        ToolResult(call_id="call", output="done"),
    ]
    assert second_input[-1] == Message(role="assistant", content="done")
    interaction.answer_delta.assert_called_once_with("done", start=True)
    interaction.response_finished.assert_called_once_with()
    interaction.token_usage.assert_called_once_with("requested-model", 12, 1000)
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
    context = LoopContext([Message(role="user", content="hello")])

    list(Loop(backend=backend, context=context).query())
    list(Loop(backend=backend, context=context, stream=True).query())

    assert backend.get_response.call_args_list[0].kwargs["stream"] is False
    assert backend.get_response.call_args_list[1].kwargs["stream"] is True


def test_one_output_loop_uses_terminal_response_text(capsys):
    """Streaming output displays deltas but returns authoritative terminal text."""
    interaction = Mock(spec=Interaction)
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
    assert loop.context.tokens == 230
    assert loop.context.model == "served-model"
    assert interaction.reasoning_delta.call_args_list[0].kwargs == {"start": True}
    assert interaction.reasoning_delta.call_args_list[1].kwargs == {"start": False}
    assert interaction.answer_delta.call_args_list[0].kwargs == {"start": True}
    assert interaction.answer_delta.call_args_list[1].kwargs == {"start": False}
    interaction.response_finished.assert_called_once_with()
    assert interaction.debug.call_count == len(events)
    capsys.readouterr()


def test_output_displays_non_streaming_completed_text():
    """Completed text is displayed directly when no streaming deltas were received."""
    interaction = Mock(spec=Interaction)

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
    interaction.response_finished.assert_not_called()


def test_empty_output_preserves_existing_context_metadata():
    """A completion without reported metadata leaves existing context values unchanged."""
    context = LoopContext(tokens=7, model="existing")
    interaction = Mock(spec=Interaction)

    response = Loop(backend=loop_backend(), context=context, interaction=interaction).output(
        [ResponseCompleted()]
    )

    assert response == Response(answer="", reasoning="")
    assert context.tokens == 7
    assert context.model == "existing"
    interaction.response_finished.assert_not_called()


def test_end_uses_the_injected_interaction():
    """Conversation termination is delegated to the configured interaction."""
    interaction = Mock(spec=Interaction)
    Loop(backend=loop_backend(), interaction=interaction).end()
    interaction.conversation_ended.assert_called_once_with()
