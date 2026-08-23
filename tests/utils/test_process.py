"""Tests for safe process invocation utilities."""

from unittest.mock import MagicMock

import pytest

from loop.utils.process import kill_process_group, parse_command_line, read_bounded_stream


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git status", ("git", "status")),
        ("printf '%s' 'a|b'", ("printf", "%s", "a|b")),
        (r"printf '%s' a\|b", ("printf", "%s", "a|b")),
        ('printf "%s" "a\\\"b"', ("printf", "%s", 'a"b')),
        ("command ''", ("command", "")),
    ],
)
def test_parse_command_line_returns_exact_quoted_and_escaped_arguments(command, expected):
    """Restricted parsing preserves literals while producing an exact argument vector."""
    assert parse_command_line(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        "",
        "   ",
        'echo "unterminated',
        "echo trailing\\",
        'echo "value\\',
        "echo ok | grep ok",
        "echo ok > output",
        "echo ok && date",
        "echo $(date)",
        "echo `date`",
        "echo (value)",
    ],
)
def test_parse_command_line_rejects_incomplete_or_unquoted_shell_syntax(command):
    """Restricted parsing fails before execution for invalid or shell-language command text."""
    with pytest.raises(ValueError):
        parse_command_line(command)


def test_read_bounded_stream_drains_every_chunk_but_retains_the_configured_limit():
    """Output draining continues past the retained budget to avoid blocked child processes."""
    stream = MagicMock()
    stream.read.side_effect = ["abcd", "efgh", "ignored", ""]
    chunks = []

    read_bounded_stream(stream, chunks, 6)

    assert chunks == ["abcd", "ef"]
    assert stream.read.call_count == 4
    stream.read.assert_called_with(8192)


def test_read_bounded_stream_can_drain_without_retaining_output():
    """A zero output budget still consumes every stream chunk."""
    stream = MagicMock()
    stream.read.side_effect = ["content", ""]
    chunks = []

    read_bounded_stream(stream, chunks, 0)

    assert chunks == []
    assert stream.read.call_count == 2


def test_kill_process_group_terminates_the_complete_posix_group(monkeypatch):
    """POSIX process cleanup targets the leader's group with an uncatchable signal."""
    process = MagicMock(pid=123)
    killpg = MagicMock()
    monkeypatch.setattr("loop.utils.process.os.name", "posix")
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)

    kill_process_group(process)

    killpg.assert_called_once_with(123, 9)
    process.kill.assert_not_called()


def test_kill_process_group_ignores_a_posix_lookup_race(monkeypatch):
    """A process that exits during cleanup does not surface a spurious failure."""
    process = MagicMock(pid=123)
    monkeypatch.setattr("loop.utils.process.os.name", "posix")
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock(side_effect=ProcessLookupError))

    kill_process_group(process)

    process.kill.assert_not_called()


def test_kill_process_group_uses_the_portable_single_process_fallback(monkeypatch):
    """Non-POSIX cleanup uses the process implementation's portable kill operation."""
    process = MagicMock()
    monkeypatch.setattr("loop.utils.process.os.name", "nt")

    kill_process_group(process)

    process.kill.assert_called_once_with()
