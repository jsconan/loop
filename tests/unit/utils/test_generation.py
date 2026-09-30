"""Tests for provider-neutral streamed-generation protection."""

from loop import AnswerDelta, ReasoningDelta
from loop.utils import GenerationWatchdog


def test_watchdog_ignores_whitespace_and_separates_channels():
    """Whitespace and alternating answer or reasoning text do not form a repeated suffix."""
    watchdog = GenerationWatchdog(10, 1000, 4, 16, 3)

    assert watchdog.observe([AnswerDelta(text="  \n\t")]) is None
    assert watchdog.observe([AnswerDelta(text="same"), ReasoningDelta(text="same")]) is None
    assert watchdog.observe([AnswerDelta(text=" prose"), AnswerDelta(text=" continues")]) is None


def test_watchdog_stops_a_configured_repeated_suffix():
    """A bounded exact answer suffix reports only safe repetition diagnostics."""
    watchdog = GenerationWatchdog(10, 1000, 4, 16, 3)

    stopped = watchdog.observe([AnswerDelta(text="abcdabcdabcd")])

    assert stopped is not None
    assert stopped.reason == "repetition"
    assert stopped.channel == "answer"
    assert stopped.pattern_size == 4
    assert stopped.repetition_count == 3


def test_watchdog_with_zero_min_count_reports_no_repetition():
    """A disabled repetition count is inert instead of raising on division."""
    watchdog = GenerationWatchdog(10, 1000, 4, 16, 0)

    assert watchdog.observe([AnswerDelta(text="abcd abcd abcd")]) is None


def test_watchdog_ignores_disabled_empty_and_already_reported_patterns():
    """Disabled, empty, and repeated observations do not produce duplicate stops."""
    disabled = GenerationWatchdog(None, None, 0, 0, 3)
    watchdog = GenerationWatchdog(None, None, 4, 16, 3)

    assert disabled.observe([AnswerDelta(text="abcdabcdabcd")]) is None
    assert watchdog.observe([AnswerDelta(text="")]) is None
    assert watchdog.observe([AnswerDelta(text="abcdabcdabcd")]) is not None
    assert watchdog.observe([AnswerDelta(text="abcd")]) is None
