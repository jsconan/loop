"""Tests for the interaction base class."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from loop import (
    AnswerCompleted,
    AnswerDelta,
    Interaction,
    Message,
    Reasoning,
    ReasoningCompleted,
    ReasoningDelta,
    Response,
    ResponseCompleted,
    ToolCall,
    ToolCallCompleted,
    Usage,
)


def interaction_mock() -> Mock:
    """Build an interaction mock backed by a no-op response scope."""
    interaction = Mock(spec=Interaction)
    interaction.response.return_value = nullcontext()
    return interaction


def test_output_uses_terminal_response_text():
    """Streaming output displays deltas but returns authoritative terminal text."""
    interaction = interaction_mock()
    call = ToolCall(call_id="call", name="tool", arguments="{}", id="fc")
    items = (
        Reasoning(content="think again", id="r"),
        Message(role="assistant", content="hello world"),
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
            structured_output={"message": "hello world"},
        ),
    ]

    response = Interaction.output(interaction, events, debug=True)

    assert response == Response(
        answer="  hello world  ",
        reasoning="  think again  ",
        tool_calls=(call,),
        items=items,
        usage=Usage(total_tokens=230),
        model="served-model",
        structured_output={"message": "hello world"},
    )
    assert interaction.reasoning_delta.call_args_list[0].kwargs == {"start": True}
    assert interaction.reasoning_delta.call_args_list[1].kwargs == {"start": False}
    assert interaction.answer_delta.call_args_list[0].kwargs == {"start": True}
    assert interaction.answer_delta.call_args_list[1].kwargs == {"start": False}
    interaction.response.assert_called_once_with()
    assert interaction.debug.call_count == len(events)


def test_output_displays_non_streaming_completed_text():
    """Completed text is displayed directly when no streaming deltas were received."""
    interaction = interaction_mock()

    response = Interaction.output(
        interaction,
        [
            ReasoningCompleted(text="think"),
            AnswerCompleted(text="answer"),
            ResponseCompleted(answer="answer", reasoning="think"),
        ],
    )

    assert response == Response(answer="answer", reasoning="think")
    interaction.reasoning.assert_called_once_with("think")
    interaction.answer.assert_called_once_with("answer")
    interaction.response.assert_called_once_with()


def test_empty_output_returns_an_empty_response():
    """A completion without reported metadata returns default response values."""
    interaction = interaction_mock()

    response = Interaction.output(interaction, [ResponseCompleted()])

    assert response == Response(answer="", reasoning="")
    interaction.response.assert_called_once_with()
