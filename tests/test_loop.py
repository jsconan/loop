"""Tests for response collection and tool-call history."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseTextDeltaEvent,
)

from loop.loop import BaseLoop, Response, StreamingLoop
from loop.interaction import Interaction
from loop.tooling import ToolRegistry, tool_registry as default_tool_registry


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


def test_loop_uses_its_injected_interaction():
    """A loop uses its injected interaction independently of the tool registry."""
    interaction = Mock(spec=Interaction)
    registry = ToolRegistry()
    interaction.input.return_value = "hello"

    loop = BaseLoop(tool_registry=registry, interaction=interaction)

    assert loop.input() == "hello"
    interaction.input.assert_called_once_with()


def test_run_requeries_after_a_tool_call_and_ends(capsys):
    """The public runner records a tool result, requeries, then exits on the next input."""
    registry = ToolRegistry()

    @registry.tool
    def echo(text: str) -> str:
        """Echo text."""
        return text

    call = ResponseFunctionToolCall(
        id="fc", call_id="call", name="echo", arguments='{"text":"done"}',
        type="function_call", status="completed",
    )
    client = Mock(tool_registry=registry)
    client.get_response.side_effect = [
        SimpleNamespace(output=[call]),
        SimpleNamespace(output=[]),
    ]
    loop = BaseLoop(client=client)
    entries = iter(["hello", False])
    loop.input = lambda: next(entries)

    loop.run()

    assert client.get_response.call_count == 2
    second_payload = client.get_response.call_args_list[1].kwargs["input"]
    assert second_payload[-1] == {
        "type": "function_call_output", "call_id": "call", "output": "done"
    }
    assert "Conversation ended." in capsys.readouterr().out


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
        interaction=loop._interaction,
    )


def test_no_tool_calls_are_reported_without_dispatch():
    """A response without function calls ends the tool-processing cycle."""
    loop = BaseLoop(client=SimpleNamespace())
    assert loop.handle_tool_calls(Response("answer", "reasoning")) is False


def test_non_streaming_query_and_streaming_query_forward_conversation_history():
    """Each loop mode selects the expected request streaming behavior."""
    client = Mock()
    client.get_response.side_effect = ["plain", "stream"]
    plain_loop = BaseLoop(client=client)
    stream_loop = StreamingLoop(client=client)

    assert plain_loop.query() == "plain"
    assert stream_loop.query() == "stream"
    assert client.get_response.call_args_list[0].kwargs == {"input": []}
    assert client.get_response.call_args_list[1].kwargs == {"input": [], "stream": True}


def test_input_reprompts_for_blank_and_recognizes_exit(monkeypatch, capsys):
    """Interactive input rejects blanks and accepts case-insensitive exit commands."""
    values = iter(["   ", " EXIT "])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(values))
    assert BaseLoop(client=SimpleNamespace()).input() is False
    assert "Please enter a message!" in capsys.readouterr().out


def test_input_returns_trimmed_message(monkeypatch):
    """Interactive input returns a non-command message without surrounding whitespace."""
    monkeypatch.setattr("builtins.input", lambda _prompt: " hello ")
    assert BaseLoop(client=SimpleNamespace()).input() == "hello"


def test_input_and_end_use_the_injected_interaction():
    """Input validation and termination output avoid process-global terminal functions."""
    interaction = Mock(spec=Interaction)
    interaction.input.side_effect = ["", "q"]
    registry = ToolRegistry()
    loop = BaseLoop(client=SimpleNamespace(tool_registry=registry), interaction=interaction)

    assert loop.input() is False
    loop.end()

    assert interaction.input.call_count == 2
    interaction.invalid_input.assert_called_once_with()
    interaction.conversation_ended.assert_called_once_with()


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
    loop = BaseLoop(client=SimpleNamespace(), debug=True)

    result = loop.output(SimpleNamespace(output=[reasoning, message, call, unknown]))

    assert result.answer == "the answer"
    assert result.reasoning == "consider this"
    assert result.tool_calls == [call]
    assert result.output_items == [reasoning, message, call, unknown]
    assert "[DEBUG EVENT]" in capsys.readouterr().out


def test_non_streaming_output_handles_empty_content(capsys):
    """Reasoning and messages with no content contribute empty text."""
    reasoning = ResponseReasoningItem(id="r", type="reasoning", summary=[], content=[])
    message = ResponseOutputMessage(
        id="m", type="message", role="assistant", status="completed", content=[]
    )
    result = BaseLoop(client=SimpleNamespace()).output(
        SimpleNamespace(output=[reasoning, message])
    )
    assert result.answer == result.reasoning == ""
    capsys.readouterr()


def test_streaming_output_collects_all_deltas_and_completed_tool_call(capsys):
    """Streaming output collects deltas and completed function calls."""
    loop = StreamingLoop(client=SimpleNamespace())
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
        ResponseTextDeltaEvent(
            delta="hello ",
            item_id="msg_123",
            output_index=1,
            content_index=0,
            sequence_number=3,
            type="response.output_text.delta",
            logprobs=[],
        ),
        ResponseTextDeltaEvent(
            delta="world",
            item_id="msg_123",
            output_index=1,
            content_index=0,
            sequence_number=4,
            type="response.output_text.delta",
            logprobs=[],
        ),
        ResponseOutputItemDoneEvent(
            item=call,
            output_index=2,
            sequence_number=5,
            type="response.output_item.done",
        ),
    ]

    response = loop.output(events)

    assert response.reasoning == "think again"
    assert response.answer == "hello world"
    assert response.tool_calls == [call]
    assert response.output_items == [call]
    capsys.readouterr()


def test_streaming_debug_and_completed_non_tool_items(capsys):
    """Debug streaming records every completed item, including ordinary messages."""
    message = ResponseOutputMessage.model_validate(
        {
            "id": "message_1", "type": "message", "role": "assistant",
            "status": "completed", "content": [],
        }
    )
    event = ResponseOutputItemDoneEvent(
        item=message, output_index=0, sequence_number=1, type="response.output_item.done"
    )
    result = StreamingLoop(client=SimpleNamespace(), debug=True).output([event])
    assert result.output_items == [message]
    assert result.tool_calls == []
    assert "[DEBUG EVENT]" in capsys.readouterr().out
