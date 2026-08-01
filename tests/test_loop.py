"""Tests for response collection and tool-call history."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseCompletedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
)

from loop.interaction import Interaction
from loop.loop import BaseLoop, Response, StreamingLoop
from loop.tooling import ToolRegistry
from loop.tooling import tool_registry as default_tool_registry


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


def loop_client(**attributes):
    """Build a minimal test client that satisfies the loop's client contract."""
    defaults = {
        "tool_registry": default_tool_registry,
        "default_model": "default-model",
        "get_context_window": lambda _model: None,
    }
    return SimpleNamespace(**(defaults | attributes))


def test_default_client_receives_custom_tool_registry(monkeypatch):
    """A loop-created client uses the registry supplied to the loop."""
    registry = ToolRegistry()
    created_client = Mock()
    created_client.get_response.return_value = "response"
    client_factory = Mock(return_value=created_client)
    monkeypatch.setattr("loop.loop.Client", client_factory)

    loop = StreamingLoop(tool_registry=registry)

    assert loop.query() == "response"
    client_factory.assert_called_once_with(tool_registry=registry)


def test_default_loop_uses_the_default_tool_registry(monkeypatch):
    """A default loop passes the populated shared registry to its client."""
    created_client = Mock(tool_registry=default_tool_registry)
    client_factory = Mock(return_value=created_client)
    monkeypatch.setattr("loop.loop.Client", client_factory)

    BaseLoop()

    client_factory.assert_called_once_with(tool_registry=default_tool_registry)
    assert default_tool_registry.schemas()


def test_client_and_tool_registry_cannot_both_be_supplied():
    """Ambiguous dependency injection is rejected explicitly."""
    with pytest.raises(ValueError, match="either client or tool_registry"):
        StreamingLoop(client=SimpleNamespace(), tool_registry=ToolRegistry())


def test_loop_exposes_its_configured_state(tmp_path):
    """Loop accessors expose the configured dependencies and conversation state."""
    client = loop_client()
    interaction = Mock(spec=Interaction)
    loop = BaseLoop(
        client=client,
        debug=True,
        interaction=interaction,
        working_directory=tmp_path,
    )

    assert loop.client is client
    assert loop.messages == []
    assert loop.debug is True
    assert loop.interaction is interaction
    assert loop.working_directory == tmp_path.resolve()
    assert loop.instructions is None

    loop.debug = False

    assert loop.debug is False


def test_loop_loads_project_instructions_once_at_initialization(monkeypatch, tmp_path):
    """A loop retains the instructions returned for its configured working directory."""
    loader = Mock(return_value="project rules")
    monkeypatch.setattr("loop.loop.load_agents_instructions", loader)

    loop = BaseLoop(client=loop_client(), working_directory=tmp_path)

    assert loop.instructions == "project rules"
    loader.assert_called_once_with(tmp_path.resolve())


def test_loop_accepts_a_string_working_directory(tmp_path):
    """A string working directory is normalized to an absolute path."""
    loop = BaseLoop(client=loop_client(), working_directory=str(tmp_path))

    assert loop.working_directory == tmp_path.resolve()


def test_run_requeries_after_a_tool_call_and_ends():
    """The public runner records a tool result, requeries, then exits on the next input."""
    registry = ToolRegistry()

    @registry.tool
    def echo(text: str) -> str:
        """Echo text."""
        return text

    call = ResponseFunctionToolCall(
        id="fc",
        call_id="call",
        name="echo",
        arguments='{"text":"done"}',
        type="function_call",
        status="completed",
    )
    client = Mock(tool_registry=registry)
    client.get_context_window.return_value = None
    client.get_response.side_effect = [
        SimpleNamespace(output=[call]),
        SimpleNamespace(output=[]),
    ]
    interaction = Mock(spec=Interaction)
    interaction.input.side_effect = ["hello", False]
    loop = BaseLoop(client=client, interaction=interaction)

    loop.run()

    assert client.get_response.call_count == 2
    second_payload = client.get_response.call_args_list[1].kwargs["input"]
    assert second_payload[-1] == {
        "type": "function_call_output",
        "call_id": "call",
        "output": "done",
    }
    interaction.conversation_ended.assert_called_once_with()


def test_run_displays_token_usage_after_output():
    """The public runner reports updated usage only after displaying the response."""
    events = []
    interaction = Mock(spec=Interaction)
    interaction.input.side_effect = ["hello", False]
    interaction.answer.side_effect = lambda _content: events.append("output")
    interaction.token_usage.side_effect = lambda *_args: events.append("usage")
    client = Mock(default_model="requested-model")
    client.get_context_window.return_value = 1000
    client.get_response.return_value = SimpleNamespace(
        output=[
            ResponseOutputMessage.model_validate(
                {
                    "id": "message_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "answer", "annotations": []}
                    ],
                }
            )
        ],
        usage=ResponseUsage(
            input_tokens=10,
            input_tokens_details={"cache_write_tokens": 0, "cached_tokens": 0},
            output_tokens=2,
            output_tokens_details={"reasoning_tokens": 0},
            total_tokens=12,
        ),
        model="served-model",
    )

    BaseLoop(client=client, interaction=interaction).run()

    assert events == ["output", "usage"]
    interaction.token_usage.assert_called_once_with("served-model", 12, 1000)


def test_response_output_and_tool_results_use_responses_api_input_items():
    """Completed calls and their registry results use Responses API input items."""
    tool_registry = Mock()
    tool_registry.call.return_value = "tool result"
    client = Mock(tool_registry=tool_registry)
    client.get_response.return_value = "next response"
    loop = StreamingLoop(client=client)
    call = function_call()
    response = Response("", "", [call], [call])

    loop.record_output(response)
    assert loop.handle_tool_calls(response)

    assert loop.query() == "next response"
    assert client.get_response.call_args.kwargs["input"] == [
        call.model_dump(exclude_none=True),
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "tool result",
        },
    ]
    assert tool_registry.call.call_count == 1
    tool_registry.call.assert_called_once_with(
        "get_current_datetime",
        "{}",
        interaction=loop.interaction,
        skill_manager=loop.skill_manager,
    )


def test_latest_model_request_sets_current_context():
    """A tool follow-up replaces context usage with its resulting context size."""
    interaction = Mock(spec=Interaction)
    interaction.input.side_effect = ["hello", False]
    client = Mock()
    client.get_context_window.return_value = 1000
    loop = BaseLoop(client=client, interaction=interaction)
    reasoning_one = ResponseReasoningItem.model_validate(
        {
            "id": "reasoning_1",
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "think one"}],
        }
    )
    reasoning_two = ResponseReasoningItem.model_validate(
        {
            "id": "reasoning_2",
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "think two"}],
        }
    )
    answer_one = ResponseOutputMessage.model_validate(
        {
            "id": "message_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "answer one", "annotations": []}],
        }
    )
    answer_two = ResponseOutputMessage.model_validate(
        {
            "id": "message_2",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "answer two", "annotations": []}],
        }
    )

    def raw_usage(input_tokens, output_tokens, total_tokens):
        """Build response usage for one model request."""
        return ResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details={"cache_write_tokens": 0, "cached_tokens": 0},
            output_tokens=output_tokens,
            output_tokens_details={"reasoning_tokens": 0},
            total_tokens=total_tokens,
        )

    assert interaction.input() == "hello"
    loop.output(SimpleNamespace(output=[reasoning_one, answer_one], usage=raw_usage(100, 5, 105)))
    loop.output(SimpleNamespace(output=[reasoning_two, answer_two], usage=raw_usage(110, 9, 119)))
    assert interaction.input() is False

    interaction.token_usage.assert_not_called()


def test_no_tool_calls_are_reported_without_dispatch():
    """A response without function calls ends the tool-processing cycle."""
    loop = BaseLoop(client=loop_client())
    assert loop.handle_tool_calls(Response("answer", "reasoning")) is False


def test_non_streaming_query_and_streaming_query_forward_history_and_instructions(tmp_path):
    """Each loop mode forwards its history, instructions, and streaming behavior."""
    (tmp_path / "AGENTS.md").write_text("project rules", encoding="utf-8")
    client = Mock()
    client.get_response.side_effect = ["plain", "stream"]
    plain_loop = BaseLoop(client=client, working_directory=tmp_path)
    stream_loop = StreamingLoop(client=client, working_directory=tmp_path)

    assert plain_loop.query() == "plain"
    assert stream_loop.query() == "stream"
    assert client.get_response.call_args_list[0].kwargs == {
        "input": [],
        "instructions": "project rules",
    }
    assert client.get_response.call_args_list[1].kwargs == {
        "input": [],
        "instructions": "project rules",
        "stream": True,
    }


def test_end_uses_the_injected_interaction():
    """Termination output uses the injected interaction."""
    interaction = Mock(spec=Interaction)
    registry = ToolRegistry()
    loop = BaseLoop(client=loop_client(tool_registry=registry), interaction=interaction)

    loop.end()

    interaction.conversation_ended.assert_called_once_with()
    interaction.token_usage.assert_not_called()


def test_output_tracks_current_context_for_each_step():
    """Each model response replaces the tracked context token count."""
    interaction = Mock(spec=Interaction)
    interaction.input.side_effect = ["first", "second", "third", False]
    client = Mock()
    client.get_context_window.return_value = 262144
    loop = BaseLoop(client=client, interaction=interaction)
    raw_usage = [
        ResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details={"cache_write_tokens": 0, "cached_tokens": 0},
            output_tokens=output_tokens,
            output_tokens_details={"reasoning_tokens": reasoning_tokens},
            total_tokens=total_tokens,
        )
        for input_tokens, output_tokens, reasoning_tokens, total_tokens in [
            (2281, 159, 59, 2440),
            (2942, 171, 71, 3113),
            (2728, 292, 92, 3020),
        ]
    ]

    assert interaction.input() == "first"
    first = loop.output(SimpleNamespace(output=[], usage=raw_usage[0])).usage
    assert interaction.input() == "second"
    second = loop.output(SimpleNamespace(output=[], usage=raw_usage[1])).usage
    assert interaction.input() == "third"
    third = loop.output(SimpleNamespace(output=[], usage=raw_usage[2])).usage
    assert interaction.input() is False

    assert (first, second, third) == (2440, 3113, 3020)

    interaction.token_usage.assert_not_called()


def test_output_tracks_the_model_reported_by_the_backend():
    """Response output tracks the model reported by the backend."""
    interaction = Mock(spec=Interaction)
    interaction.input.side_effect = ["hello", False]
    client = Mock(default_model="requested-model")
    client.get_context_window.return_value = 2000
    loop = BaseLoop(client=client, interaction=interaction)
    usage = ResponseUsage(
        input_tokens=10,
        input_tokens_details={"cache_write_tokens": 0, "cached_tokens": 0},
        output_tokens=2,
        output_tokens_details={"reasoning_tokens": 0},
        total_tokens=12,
    )

    assert interaction.input() == "hello"
    loop.output(SimpleNamespace(output=[], usage=usage, model="served-model"))
    assert interaction.input() is False

    client.get_context_window.assert_not_called()
    interaction.token_usage.assert_not_called()


def test_non_streaming_output_collects_reasoning_message_call_and_ignores_unknown(capsys):
    """Completed output is classified through its public Responses API item types."""
    reasoning = ResponseReasoningItem.model_validate(
        {
            "id": "reasoning_1",
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "consider this"}],
        }
    )
    message = ResponseOutputMessage.model_validate(
        {
            "id": "message_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "the answer", "annotations": []}],
        }
    )
    call = function_call()
    unknown = SimpleNamespace(kind="unknown")
    loop = BaseLoop(client=loop_client(), debug=True)

    usage = ResponseUsage(
        input_tokens=100,
        input_tokens_details={"cache_write_tokens": 0, "cached_tokens": 0},
        output_tokens=20,
        output_tokens_details={"reasoning_tokens": 0},
        total_tokens=120,
    )
    result = loop.output(SimpleNamespace(output=[reasoning, message, call, unknown], usage=usage))

    assert result.answer == "the answer"
    assert result.reasoning == "consider this"
    assert result.tool_calls == [call]
    assert result.output_items == [reasoning, message, call, unknown]
    assert result.usage == 120
    assert "[DEBUG EVENT]" in capsys.readouterr().out


def test_non_streaming_output_handles_empty_content(capsys):
    """Reasoning and messages with no content contribute empty text."""
    reasoning = ResponseReasoningItem(id="r", type="reasoning", summary=[], content=[])
    message = ResponseOutputMessage(
        id="m", type="message", role="assistant", status="completed", content=[]
    )
    result = BaseLoop(client=loop_client()).output(SimpleNamespace(output=[reasoning, message]))
    assert result.answer == result.reasoning == ""
    capsys.readouterr()


def test_streaming_output_collects_done_text_and_completed_tool_call(capsys):
    """Streaming output collects completed text and function calls."""
    loop = StreamingLoop(client=loop_client())
    call = function_call()
    events = [
        ResponseReasoningTextDeltaEvent(
            delta="think ",
            item_id="rs_123",
            output_index=0,
            content_index=0,
            sequence_number=1,
            type="response.reasoning_text.delta",
        ),
        ResponseReasoningTextDeltaEvent(
            delta="again",
            item_id="rs_123",
            output_index=0,
            content_index=0,
            sequence_number=2,
            type="response.reasoning_text.delta",
        ),
        ResponseReasoningTextDoneEvent(
            text="think again",
            item_id="rs_123",
            output_index=0,
            content_index=0,
            sequence_number=3,
            type="response.reasoning_text.done",
        ),
        ResponseTextDeltaEvent(
            delta="hello ",
            item_id="msg_123",
            output_index=1,
            content_index=0,
            sequence_number=4,
            type="response.output_text.delta",
            logprobs=[],
        ),
        ResponseTextDeltaEvent(
            delta="world",
            item_id="msg_123",
            output_index=1,
            content_index=0,
            sequence_number=5,
            type="response.output_text.delta",
            logprobs=[],
        ),
        ResponseTextDoneEvent(
            text="hello world",
            item_id="msg_123",
            output_index=1,
            content_index=0,
            sequence_number=6,
            type="response.output_text.done",
            logprobs=[],
        ),
        ResponseOutputItemDoneEvent(
            item=call,
            output_index=2,
            sequence_number=7,
            type="response.output_item.done",
        ),
        ResponseCompletedEvent.model_validate(
            {
                "type": "response.completed",
                "sequence_number": 8,
                "response": {
                    "id": "response_1",
                    "created_at": 0,
                    "model": "gpt-4o",
                    "object": "response",
                    "output": [],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                        "usage": {
                            "input_tokens": 200,
                            "input_tokens_details": {
                                "cache_write_tokens": 0,
                                "cached_tokens": 0,
                            },
                            "output_tokens": 30,
                            "output_tokens_details": {"reasoning_tokens": 7},
                        "total_tokens": 230,
                    },
                },
            }
        ),
    ]

    response = loop.output(events)

    assert response.reasoning == "think again"
    assert response.answer == "hello world"
    assert response.tool_calls == [call]
    assert response.output_items == [call]
    assert response.usage == 230
    assert response.model == "gpt-4o"
    capsys.readouterr()


def test_streaming_debug_and_completed_non_tool_items(capsys):
    """Debug streaming records every completed item, including ordinary messages."""
    message = ResponseOutputMessage.model_validate(
        {
            "id": "message_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [],
        }
    )
    event = ResponseOutputItemDoneEvent(
        item=message, output_index=0, sequence_number=1, type="response.output_item.done"
    )
    result = StreamingLoop(client=loop_client(), debug=True).output([event])
    assert result.output_items == [message]
    assert result.tool_calls == []
    assert "[DEBUG EVENT]" in capsys.readouterr().out
