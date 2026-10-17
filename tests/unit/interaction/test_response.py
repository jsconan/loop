"""Verify response presentation independently of input, collection, and persistence."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from loop import (
    AnswerCompleted,
    AnswerDelta,
    InteractionOutput,
    ReasoningCompleted,
    ReasoningDelta,
    ResponseCompleted,
    ResponsePresenter,
    ToolCall,
    ToolCallCompleted,
)


@pytest.mark.parametrize("channel", ["answer", "reasoning"])
def test_presenter_displays_streams_without_repeating_completed_text(channel):
    """Each streamed channel suppresses its own completed text after any delta."""
    output = Mock(spec=InteractionOutput)
    presenter = ResponsePresenter(output)
    delta, completed = (
        (AnswerDelta, AnswerCompleted)
        if channel == "answer"
        else (ReasoningDelta, ReasoningCompleted)
    )

    presenter.present(delta(text=""))
    presenter.present(delta(text="partial"))
    presenter.present(completed(text="complete"))

    assert output.method_calls == [
        getattr(call, f"{channel}_delta")(""),
        getattr(call, f"{channel}_delta")("partial"),
    ]


def test_presenter_keeps_answer_and_reasoning_stream_state_separate():
    """Streaming one channel does not suppress completed text in the other channel."""
    output = Mock(spec=InteractionOutput)
    presenter = ResponsePresenter(output)

    presenter.present(ReasoningDelta(text="thought"))
    presenter.present(AnswerCompleted(text="answer"))

    assert output.method_calls == [call.reasoning_delta("thought"), call.answer("answer")]


def test_presenter_displays_completed_channels_without_deltas():
    """Nonstreamed reasoning and answer retain their original presentation order."""
    output = Mock(spec=InteractionOutput)
    presenter = ResponsePresenter(output)

    presenter.present(ReasoningCompleted(text="thought"))
    presenter.present(AnswerCompleted(text="answer"))

    assert output.method_calls == [call.reasoning("thought"), call.answer("answer")]


def test_presenter_ignores_terminal_metadata_tools_and_extension_events():
    """Response metadata and tool declarations do not cause visible execution or output."""
    output = Mock(spec=InteractionOutput)
    presenter = ResponsePresenter(output)

    for event in (
        ToolCallCompleted(call=ToolCall(call_id="c", name="tool", arguments="{}")),
        ResponseCompleted(answer="terminal", reasoning="terminal"),
        SimpleNamespace(extension=True),
    ):
        presenter.present(event)

    assert output.method_calls == []


def test_presenters_do_not_share_stream_state():
    """A fresh response still renders completed text after another response streamed."""
    output = Mock(spec=InteractionOutput)
    first, second = ResponsePresenter(output), ResponsePresenter(output)

    first.present(AnswerDelta(text="first"))
    second.present(AnswerCompleted(text="second"))

    assert output.method_calls == [call.answer_delta("first"), call.answer("second")]
