"""Verify UI-independent response collection and terminal-event validation."""

from types import SimpleNamespace

import pytest

from loop import (
    AnswerCompleted,
    AnswerDelta,
    BackendResponseError,
    Message,
    ReasoningCompleted,
    ReasoningDelta,
    Response,
    ResponseCollector,
    ResponseCompleted,
    ToolCall,
    ToolCallCompleted,
    Usage,
)


def test_collector_uses_terminal_content_and_keeps_tool_call_order():
    """Final metadata overrides partial text without changing ordered canonical items."""
    collector = ResponseCollector()
    calls = tuple(ToolCall(call_id=str(index), name="tool", arguments="{}") for index in range(2))
    items = (Message(role="assistant", content="canonical answer"),)
    for event in (
        ReasoningDelta(text="partial thought"),
        AnswerDelta(text="partial answer"),
        ReasoningCompleted(text="provisional thought"),
        AnswerCompleted(text="provisional answer"),
        *(ToolCallCompleted(call=call) for call in calls),
        SimpleNamespace(extension=True),
        ResponseCompleted(
            answer="  final answer  ",
            reasoning="  final thought  ",
            items=items,
            model="served-model",
            usage=Usage(total_tokens=42),
            structured_output={"answer": True},
        ),
    ):
        collector.add(event)

    assert collector.finish() == Response(
        answer="  final answer  ",
        reasoning="  final thought  ",
        tool_calls=calls,
        items=items,
        model="served-model",
        usage=Usage(total_tokens=42),
        structured_output={"answer": True},
    )


def test_collector_accepts_an_empty_terminal_response():
    """A terminal event alone supplies a complete response with default metadata."""
    collector = ResponseCollector()
    collector.add(ResponseCompleted())

    assert collector.finish() == Response(answer="", reasoning="")


def test_collector_captures_terminal_fields_when_the_event_is_accepted():
    """Later producer reuse of a terminal event cannot replace its accepted response fields."""
    collector = ResponseCollector()
    completion = ResponseCompleted(answer="accepted", reasoning="thought", model="original")
    collector.add(completion)
    completion.answer = "replaced"
    completion.reasoning = "replaced"
    completion.model = "replaced"

    assert collector.finish() == Response(answer="accepted", reasoning="thought", model="original")


@pytest.mark.parametrize(
    ("events", "started"),
    [
        ([], False),
        ([SimpleNamespace(extension=True)], False),
        ([AnswerDelta(text="partial")], True),
        ([ReasoningDelta(text="partial")], True),
        ([AnswerCompleted(text="answer")], True),
        ([ReasoningCompleted(text="thought")], True),
        ([ToolCallCompleted(call=ToolCall(call_id="c", name="tool", arguments="{}"))], True),
    ],
)
def test_collector_requires_explicit_completion_and_reports_partial_output(events, started):
    """Incomplete streams report exactly whether recognizable response output was observed."""
    collector = ResponseCollector()
    for event in events:
        collector.add(event)

    with pytest.raises(BackendResponseError, match="without an explicit completion") as raised:
        collector.finish()

    assert raised.value.response_started is started
    assert raised.value.provider == "backend"
    assert raised.value.operation == "collect_response"


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (ResponseCompleted(), "more than one response completion"),
        (AnswerDelta(text="late"), "an event after response completion"),
        (SimpleNamespace(extension=True), "an event after response completion"),
    ],
)
def test_collector_rejects_every_event_after_completion(event, message):
    """No response or extension event can be silently accepted after the terminal event."""
    collector = ResponseCollector()
    collector.add(ResponseCompleted())

    with pytest.raises(BackendResponseError, match=message) as raised:
        collector.add(event)

    assert raised.value.response_started is True


def test_collectors_do_not_share_response_state():
    """Interleaved independent collectors preserve their own terminal content and calls."""
    first, second = ResponseCollector(), ResponseCollector()
    first.add(ToolCallCompleted(call=ToolCall(call_id="c", name="tool", arguments="{}")))
    second.add(ResponseCompleted(answer="second"))
    first.add(ResponseCompleted(answer="first"))

    assert first.finish().answer == "first"
    assert len(first.finish().tool_calls) == 1
    assert second.finish() == Response(answer="second", reasoning="")
