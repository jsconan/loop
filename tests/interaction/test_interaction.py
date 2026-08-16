"""Tests for the interaction base class."""

from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from loop import (
    AnswerCompleted,
    AnswerDelta,
    Compaction,
    CompactionContextItem,
    InstructionSnapshot,
    Interaction,
    Message,
    Reasoning,
    ReasoningCompleted,
    ReasoningDelta,
    Response,
    ResponseCompleted,
    ToolCall,
    ToolCallCompleted,
    ToolResult,
    Usage,
)


def interaction_mock() -> MagicMock:
    """Build an interaction mock backed by a no-op response scope."""
    interaction = MagicMock(spec=Interaction)
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

    response = Interaction.response(interaction, events, debug=True)

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
    interaction.response_context.assert_called_once_with()
    assert interaction.debug.call_count == len(events)


def test_output_displays_non_streaming_completed_text():
    """Completed text is displayed directly when no streaming deltas were received."""
    interaction = interaction_mock()

    response = Interaction.response(
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
    interaction.response_context.assert_called_once_with()


def test_empty_output_returns_an_empty_response():
    """A completion without reported metadata returns default response values."""
    interaction = interaction_mock()

    response = Interaction.response(interaction, [ResponseCompleted()])

    assert response == Response(answer="", reasoning="")
    interaction.response_context.assert_called_once_with()


def test_history_replays_visible_conversation_items_in_order():
    """Persisted history mirrors live output, including compaction checkpoints."""
    interaction = interaction_mock()
    items = (
        Message(role="user", content="question"),
        Reasoning(content="thought"),
        ToolCall(call_id="known", name="search", arguments='{"query":"term"}'),
        ToolResult(call_id="known", output="result"),
        ToolResult(call_id="missing", output="orphaned"),
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
        Compaction(
            id="third",
            boundary=len(items),
            created_at=datetime(2026, 8, 16, tzinfo=UTC),
            provider="test",
            model="model",
            context=(CompactionContextItem(provider="test", data={}),),
            instructions=instructions,
            input_tokens_before=678,
            input_tokens_after=678,
        ),
    )

    Interaction.history(interaction, items, compactions)

    assert interaction.method_calls == [
        call.info("Compacted session context."),
        call.user("question"),
        call.reasoning("thought"),
        call.tool_call("search", '{"query":"term"}'),
        call.info("Compacted session context from 12,345 to 678 tokens."),
        call.answer("answer"),
        call.info("Compacted session context."),
    ]
