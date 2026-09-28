"""Tests for safe process invocation utilities."""

import errno
import logging
import subprocess
import time
from unittest.mock import MagicMock

import pytest

from loop.utils.process import (
    ProcessCaptureStatus,
    kill_process_group,
    parse_command_line,
    read_bounded_stream,
    supervise_process,
)


def test_supervise_process_bounds_both_streams_and_preserves_exit(caplog):
    """The shared supervisor drains real child pipes and reports discarded output."""
    caplog.set_level(logging.WARNING, logger="loop.utils.process")
    process = subprocess.Popen(
        ["/bin/sh", "-c", "printf abcdef; printf xy >&2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    capture = supervise_process(process, time.monotonic() + 2, stdout_limit=3)

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert capture.exit_code == 0
    assert capture.stdout == "abc"
    assert capture.stdout_discarded == 3
    assert capture.stderr == "xy"


def test_supervise_process_kills_a_child_after_deadline(caplog):
    """Timeout cleanup terminates the owned process group and bounds wall time."""
    caplog.set_level(logging.WARNING, logger="loop.utils.process")
    process = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    capture = supervise_process(process, time.monotonic() + 0.1)

    assert capture.status is ProcessCaptureStatus.TIMED_OUT
    assert process.poll() is not None


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git status", ("git", "status")),
        ("printf '%s' 'a|b'", ("printf", "%s", "a|b")),
        (r"printf '%s' a\|b", ("printf", "%s", "a|b")),
        ('printf "%s" "a\\"b"', ("printf", "%s", 'a"b')),
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

    discarded = read_bounded_stream(stream, chunks, maximum=6)

    assert chunks == ["abcd", "ef"]
    assert discarded == 9
    assert stream.read.call_count == 4
    stream.read.assert_called_with(8192)


def test_read_bounded_stream_can_drain_without_retaining_output():
    """A zero output budget still consumes every stream chunk."""
    stream = MagicMock()
    stream.read.side_effect = ["content", ""]
    chunks = []

    discarded = read_bounded_stream(stream, chunks, maximum=0)

    assert chunks == []
    assert discarded == 7
    assert stream.read.call_count == 2


def test_read_bounded_stream_reports_no_discarded_characters_within_limit():
    """A stream shorter than the output budget reports no discarded characters."""
    stream = MagicMock()
    stream.read.side_effect = ["content", ""]
    chunks = []

    discarded = read_bounded_stream(stream, chunks, maximum=len("content"))

    assert chunks == ["content"]
    assert discarded == 0


def test_read_bounded_stream_tolerates_supervisor_pipe_close():
    """An intentional descriptor close during timeout cleanup ends the reader quietly."""
    stream = MagicMock()
    stream.read.side_effect = ["partial", OSError(errno.EBADF, "Bad file descriptor")]
    chunks = []

    assert read_bounded_stream(stream, chunks, maximum=20) == 0
    assert chunks == ["partial"]


def test_read_bounded_stream_preserves_unrelated_read_failures():
    """An unrelated stream error still reaches its owning supervisor."""
    stream = MagicMock()
    stream.read.side_effect = OSError(errno.EIO, "I/O error")

    with pytest.raises(OSError, match="I/O error"):
        read_bounded_stream(stream, [], maximum=20)


def test_read_bounded_stream_preserves_unrelated_close_failures():
    """A stream-close failure other than an intentional descriptor close is surfaced."""
    stream = MagicMock()
    stream.read.return_value = ""
    stream.close.side_effect = OSError(errno.EIO, "close failed")

    with pytest.raises(OSError, match="close failed"):
        read_bounded_stream(stream, [], maximum=20)


def test_kill_process_group_terminates_the_complete_posix_group(monkeypatch):
    """POSIX process cleanup targets the leader's group with an uncatchable signal."""
    process = MagicMock(pid=123)
    killpg = MagicMock()
    monkeypatch.setattr("loop.utils.process.os.name", "posix")
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)

    kill_process_group(process)

    killpg.assert_called_once_with(123, 9)
    process.kill.assert_not_called()


def test_kill_process_group_logs_a_posix_lookup_race(monkeypatch, caplog):
    """A process-group lookup race is warned about without surfacing the failure."""
    process = MagicMock(pid=123)
    monkeypatch.setattr("loop.utils.process.os.name", "posix")
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock(side_effect=ProcessLookupError))

    with caplog.at_level(logging.WARNING, logger="loop.utils.process"):
        kill_process_group(process)

    assert [record.getMessage() for record in caplog.records] == [
        "Process group 123 was already gone during termination."
    ]
    assert all(record.levelno == logging.WARNING for record in caplog.records)
    process.kill.assert_not_called()


def test_supervise_process_logs_failed_reader_stream_close(caplog):
    """A failed forced pipe close is captured as a warning during cleanup."""
    process = MagicMock(stdout=MagicMock(), stderr=MagicMock(), returncode=0)
    process.wait.return_value = 0
    process.stdout.fileno.side_effect = ValueError("private stream detail")
    process.stderr.fileno.side_effect = ValueError("private stream detail")
    thread = MagicMock()
    thread.is_alive.return_value = True

    with caplog.at_level(logging.WARNING, logger="loop.utils.process"):
        capture = supervise_process(
            process,
            time.monotonic() + 1,
            terminate=MagicMock(),
            thread_factory=lambda **kwargs: thread,
        )

    assert capture.status is ProcessCaptureStatus.TIMED_OUT
    assert [record.getMessage() for record in caplog.records] == [
        "Could not close a process output stream during cleanup.",
        "Could not close a process output stream during cleanup.",
    ]
    assert all(record.levelno == logging.WARNING for record in caplog.records)
    assert "private stream detail" not in caplog.text


def test_supervise_process_logs_cleanup_wait_timeout(caplog):
    """A child that outlives the cleanup wait is captured as a warning."""
    process = MagicMock(stdout=MagicMock(), stderr=MagicMock(), pid=123, returncode=None)
    process.wait.side_effect = [
        None,
        subprocess.TimeoutExpired("private command", 1),
    ]
    thread = MagicMock()
    thread.is_alive.return_value = False

    with caplog.at_level(logging.WARNING, logger="loop.utils.process"):
        capture = supervise_process(
            process,
            time.monotonic() + 1,
            terminate=MagicMock(),
            thread_factory=lambda **kwargs: thread,
        )

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert [record.getMessage() for record in caplog.records] == [
        "Process 123 did not exit within the cleanup wait time."
    ]
    assert all(record.levelno == logging.WARNING for record in caplog.records)
    assert "private command" not in caplog.text


def test_kill_process_group_uses_the_portable_single_process_fallback(monkeypatch):
    """Non-POSIX cleanup uses the process implementation's portable kill operation."""
    process = MagicMock()
    monkeypatch.setattr("loop.utils.process.os.name", "nt")

    kill_process_group(process)

    process.kill.assert_called_once_with()
