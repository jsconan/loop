"""Test turn retention and bounded semantic summary batches."""

import pytest

from loop import Message, Session, ToolCall, ToolResult
from loop.backend import estimate_context
from loop.compaction.planning import CompactionLimitError, RetainedContext, portable_batches


def test_long_active_turn_splits_only_after_a_completed_parallel_round():
    """Keep the active request and whole tool rounds when the earlier round is too large."""
    messages = [
        Message(role="user", content="Opening request"),
        Message(role="assistant", content="Opening answer"),
        Message(role="user", content="Active request"),
        ToolCall(call_id="reused", name="left", arguments="{}"),
        ToolCall(call_id="right", name="right", arguments="{}"),
        ToolResult(call_id="reused", output="x" * 5000),
        ToolResult(call_id="right", output="result for right"),
        ToolCall(call_id="reused", name="continue", arguments="{}"),
        ToolResult(call_id="reused", output="result from the later call"),
    ]
    session = Session(messages=messages)
    retained = RetainedContext.select(session, 500)
    assert retained.coverage.opening == (0,)
    assert retained.coverage.tail == (2, 7, 8)
    assert retained.middle == (1, 3, 4, 5, 6)
    assert retained.end == len(messages)
    with pytest.raises(CompactionLimitError, match="active request and its tool round"):
        RetainedContext.select(session, 10)


def test_opening_is_deduplicated_when_it_also_belongs_to_recent_tail():
    """Use canonical indices to retain first-user additions exactly once."""
    session = Session(
        messages=[
            Message(role="user", content="First opening"),
            Message(role="user", content="First addition"),
            Message(role="assistant", content="Settled"),
            Message(role="user", content="Active"),
        ]
    )
    retained = RetainedContext.select(session, 1000)
    assert retained.coverage.opening == ()
    assert retained.middle == ()
    assert retained.coverage.tail == (0, 1, 2, 3)


def test_summary_batches_keep_each_parallel_tool_round_whole():
    """Large histories use bounded batches while keeping paired tool results together."""
    beginning = Message(role="user", content="opening")
    group = [
        ToolCall(call_id="a", name="first", arguments="{}"),
        ToolCall(call_id="b", name="second", arguments="{}"),
        ToolResult(call_id="a", output="a result"),
        ToolResult(call_id="b", output="b result"),
    ]
    end = Message(role="assistant", content="result recorded")
    budget = estimate_context(group)
    assert list(portable_batches([beginning, *group, end], budget)) == [[beginning], group, [end]]
    with pytest.raises(CompactionLimitError, match="exceeds the summary limit"):
        list(portable_batches(group, budget - 1))
    with pytest.raises(CompactionLimitError, match="unresolved tool round"):
        list(portable_batches(group[:-1], budget))


def test_opening_excess_remains_in_the_middle_when_recent_content_needs_room():
    """An over-budget opening is summarized while the active user request survives."""
    session = Session(
        messages=[
            Message(role="user", content="x" * 5000),
            Message(role="assistant", content="settled"),
            Message(role="user", content="active"),
        ]
    )
    retained = RetainedContext.select(session, 100)
    assert retained.coverage.opening == ()
    assert retained.coverage.tail == (2,)
    assert retained.middle == (0, 1)


def test_long_first_turn_pins_consecutive_user_additions_before_remaining_rounds():
    """Opening additions in the active turn are retained exactly once."""
    session = Session(
        messages=[
            Message(role="user", content="first request"),
            Message(role="user", content="clarification"),
            ToolCall(call_id="same", name="first", arguments="{}"),
            ToolCall(call_id="same", name="parallel", arguments="{}"),
            ToolResult(call_id="same", output="x" * 6000),
            ToolResult(call_id="same", output="paired output"),
            ToolCall(call_id="next", name="next", arguments="{}"),
            ToolResult(call_id="next", output="later output"),
        ]
    )
    retained = RetainedContext.select(session, 500)
    assert retained.coverage.opening == ()
    assert retained.coverage.tail == (0, 1, 6, 7)
    assert retained.middle == (2, 3, 4, 5)


def test_no_user_turn_is_needed_to_summarize_legacy_assistant_items():
    """Assistant-only legacy history is eligible middle without a synthetic user task."""
    legacy = Session(messages=[Message(role="assistant", content="legacy answer")])
    retained = RetainedContext.select(legacy, 200)
    assert retained.middle == (0,)
    assert retained.coverage.tail == ()
    assert list(portable_batches([], 100)) == []


def test_recent_turn_boundary_cannot_drop_the_call_for_a_later_result():
    """Cross-turn legacy tool results keep the full paired range on a safe boundary."""
    session = Session(
        messages=[
            Message(role="user", content="first request"),
            ToolCall(call_id="first", name="work", arguments="{}"),
            Message(role="user", content="addition while tool was pending"),
            ToolResult(call_id="first", output="done"),
            Message(role="assistant", content="finished"),
            Message(role="user", content="active"),
        ]
    )
    retained = RetainedContext.select(session, 600)
    assert retained.coverage.tail == tuple(range(len(session.messages)))
    assert retained.middle == ()


def test_active_request_retains_a_tool_round_that_crosses_its_boundary():
    """A recently delivered user message can sit inside an active parallel tool round."""
    messages = [
        Message(role="user", content="x" * 5000),
        ToolCall(call_id="same", name="first", arguments="{}"),
        ToolCall(call_id="same", name="parallel", arguments="{}"),
        Message(role="user", content="Addition during tool execution"),
        ToolResult(call_id="same", output="first output"),
        ToolResult(call_id="same", output="second output"),
    ]
    session = Session(messages=messages)

    retained = RetainedContext.select(session, 200)

    assert retained.middle == (0,)
    assert retained.coverage.tail == (1, 2, 3, 4, 5)
