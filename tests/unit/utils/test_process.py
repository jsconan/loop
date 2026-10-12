"""Tests for safe process invocation utilities."""

import ctypes
import errno
import logging
import os
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from loop.utils.process import (
    ProcessCaptureStatus,
    kill_process_group,
    parse_command_line,
    read_bounded_stream,
    supervise_process,
)


@pytest.fixture
def process_os(monkeypatch):
    """Provide a waitable POSIX child boundary for supervisor unit cases."""
    boundary = SimpleNamespace(
        name="posix",
        P_PID=1,
        WEXITED=1,
        WNOHANG=2,
        WNOWAIT=4,
        close=os.close,
        killpg=MagicMock(),
        waitid=MagicMock(),
    )
    boundary.waitid.side_effect = lambda _idtype, pid, _options: SimpleNamespace(si_pid=pid)
    monkeypatch.setattr("loop.utils.process.os", boundary)
    return boundary


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


def test_kill_process_group_ignores_an_already_exited_posix_group(monkeypatch, caplog):
    """A vanished process group is normal cleanup and does not emit a warning."""
    process = MagicMock(pid=123)
    monkeypatch.setattr("loop.utils.process.os.name", "posix")
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock(side_effect=ProcessLookupError))

    with caplog.at_level(logging.WARNING, logger="loop.utils.process"):
        kill_process_group(process)

    assert not caplog.records
    process.kill.assert_not_called()


def _install_macos_process_api(monkeypatch, snapshots, process_states):
    """Stub libproc snapshots and process states for Darwin signal-error tests."""
    from loop.utils import process as process_utils

    snapshot_iter = iter(snapshots)
    list_group = MagicMock()

    def populate_group(_process_group_id, buffer, _buffer_size):
        """Fill the requested PID array from the next configured snapshot."""
        snapshot = next(snapshot_iter)
        if isinstance(snapshot, OSError):
            ctypes.set_errno(snapshot.errno)
            return -1
        for index, process_id in enumerate(snapshot):
            buffer[index] = process_id
        return len(snapshot)

    process_info = MagicMock()

    def fill_process_info(process_id, _flavor, _argument, buffer, _buffer_size):
        """Return the configured BSD status and errno for a process identifier."""
        result, error_number, status = process_states[process_id]
        ctypes.set_errno(error_number)
        if result > 0 and status is not None:
            ctypes.c_uint32.from_buffer(buffer, 4).value = status
        return result

    list_group.side_effect = populate_group
    process_info.side_effect = fill_process_info
    library = SimpleNamespace(proc_listpgrppids=list_group, proc_pidinfo=process_info)
    monkeypatch.setattr(process_utils.ctypes, "CDLL", MagicMock(return_value=library))
    return list_group, process_info


def test_kill_process_group_ignores_darwin_eprem_for_an_empty_zombie_group(monkeypatch):
    """A macOS zombie-only group is already stopped when libproc confirms no live member."""
    process = MagicMock(pid=123)
    killpg = MagicMock(side_effect=PermissionError(errno.EPERM, "operation not permitted"))
    monkeypatch.setattr("loop.utils.process.os.name", "posix")
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)
    monkeypatch.setattr("loop.utils.process.sys.platform", "darwin")
    _install_macos_process_api(
        monkeypatch,
        [[123], [123]],
        {123: (0, errno.ESRCH, None)},
    )

    kill_process_group(process)

    killpg.assert_called_once_with(123, 9)


def test_kill_process_group_preserves_darwin_eprem_when_a_group_member_is_live(monkeypatch):
    """A live macOS group member keeps the native signaling error visible."""
    process = MagicMock(pid=123)
    killpg = MagicMock(side_effect=PermissionError(errno.EPERM, "operation not permitted"))
    monkeypatch.setattr("loop.utils.process.os.name", "posix")
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)
    monkeypatch.setattr("loop.utils.process.sys.platform", "darwin")
    _install_macos_process_api(monkeypatch, [[123]], {123: (8, 0, 1)})

    with pytest.raises(PermissionError) as raised:
        kill_process_group(process)

    assert raised.value.errno == errno.EPERM
    killpg.assert_called_once_with(123, 9)


def test_kill_process_group_keeps_darwin_inspection_failures_closed(monkeypatch):
    """An unavailable macOS process listing cannot turn a denied signal into success."""
    process = MagicMock(pid=123)
    killpg = MagicMock(side_effect=PermissionError(errno.EPERM, "operation not permitted"))
    monkeypatch.setattr("loop.utils.process.os.name", "posix")
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)
    monkeypatch.setattr("loop.utils.process.sys.platform", "darwin")
    _install_macos_process_api(
        monkeypatch,
        [OSError(errno.EACCES, "listing denied")],
        {},
    )

    with pytest.raises(PermissionError) as raised:
        kill_process_group(process)

    assert raised.value.errno == errno.EPERM
    assert isinstance(raised.value.__cause__, PermissionError)
    assert raised.value.__cause__.errno == errno.EACCES


def test_kill_process_group_rechecks_members_that_appear_during_darwin_inspection(monkeypatch):
    """A newly observed macOS group member is inspected before cleanup treats it as empty."""
    process = MagicMock(pid=123)
    killpg = MagicMock(side_effect=PermissionError(errno.EPERM, "operation not permitted"))
    monkeypatch.setattr("loop.utils.process.os.name", "posix")
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)
    monkeypatch.setattr("loop.utils.process.sys.platform", "darwin")
    _install_macos_process_api(
        monkeypatch,
        [[123], [123, 124], [123, 124], [123, 124]],
        {123: (0, errno.ESRCH, None), 124: (0, errno.ESRCH, None)},
    )

    kill_process_group(process)

    killpg.assert_called_once_with(123, 9)


def test_kill_process_group_preserves_non_darwin_eprem(monkeypatch):
    """Only Darwin's zombie-only group result receives the libproc compatibility check."""
    process = MagicMock(pid=123)
    killpg = MagicMock(side_effect=PermissionError(errno.EPERM, "operation not permitted"))
    monkeypatch.setattr("loop.utils.process.os.name", "posix")
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)
    monkeypatch.setattr("loop.utils.process.sys.platform", "linux")

    with pytest.raises(PermissionError) as raised:
        kill_process_group(process)

    assert raised.value.errno == errno.EPERM


@pytest.mark.parametrize(
    ("snapshots", "process_states", "capacity", "passes", "expected_error"),
    [
        ([[123, 124]], {}, 2, 8, errno.EOVERFLOW),
        ([[123]], {123: (0, errno.EACCES, None)}, 65536, 8, errno.EACCES),
        ([[123]], {123: (0, 0, None)}, 65536, 8, errno.EIO),
        ([[123]], {123: (4, 0, None)}, 65536, 8, errno.EIO),
        (
            [[123], OSError(errno.EACCES, "listing denied")],
            {123: (8, 0, 5)},
            65536,
            8,
            errno.EACCES,
        ),
        ([[123], [123, 124]], {123: (8, 0, 5)}, 2, 8, errno.EOVERFLOW),
        ([[123], [123, 124]], {123: (8, 0, 5)}, 65536, 1, errno.EAGAIN),
    ],
)
def test_kill_process_group_preserves_darwin_eprem_when_inspection_is_incomplete(
    monkeypatch, snapshots, process_states, capacity, passes, expected_error
):
    """Bounded or failed macOS inspection never classifies an uncertain group as empty."""
    from loop.utils import process as process_utils

    process = MagicMock(pid=123)
    killpg = MagicMock(side_effect=PermissionError(errno.EPERM, "operation not permitted"))
    monkeypatch.setattr("loop.utils.process.os.name", "posix")
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)
    monkeypatch.setattr("loop.utils.process.sys.platform", "darwin")
    monkeypatch.setattr(process_utils, "_MACOS_PROCESS_GROUP_PID_CAPACITY", capacity)
    monkeypatch.setattr(process_utils, "_MACOS_PROCESS_GROUP_INSPECTION_PASSES", passes)
    _install_macos_process_api(monkeypatch, snapshots, process_states)

    with pytest.raises(PermissionError) as raised:
        kill_process_group(process)

    assert raised.value.errno == errno.EPERM
    assert raised.value.__cause__.errno == expected_error
    killpg.assert_called_once_with(123, 9)


def test_supervise_process_logs_failed_reader_stream_close(caplog, process_os):
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


def test_supervise_process_logs_cleanup_wait_timeout(caplog, process_os):
    """A child that outlives the cleanup wait is captured as a warning."""
    process = MagicMock(stdout=MagicMock(), stderr=MagicMock(), pid=123, returncode=None)
    process.wait.side_effect = subprocess.TimeoutExpired("private command", 1)
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


def test_read_bounded_stream_tolerates_already_closed_descriptor():
    """Cleanup ignores EBADF when a supervisor has already closed the reader descriptor."""
    stream = MagicMock()
    stream.read.return_value = ""
    stream.close.side_effect = OSError(errno.EBADF, "already closed")

    assert read_bounded_stream(stream, [], maximum=20) == 0


@pytest.mark.parametrize("missing", ["stdout", "stderr"])
def test_supervise_process_refuses_missing_capture_pipe(missing):
    """A missing output pipe terminates the owned process and returns unavailable."""
    process = MagicMock(returncode=-9)
    setattr(process, missing, None)
    terminate = MagicMock()
    result = supervise_process(process, 100, terminate=terminate)
    assert result.status is ProcessCaptureStatus.UNAVAILABLE
    terminate.assert_called_once_with(process)
    process.wait.assert_called_once()


@pytest.mark.parametrize("missing", ["stdout", "stderr"])
def test_supervise_process_reaps_child_when_missing_pipe_termination_fails(missing):
    """A denied group signal does not skip reaping a child with incomplete capture pipes."""
    process = MagicMock(returncode=-9)
    setattr(process, missing, None)
    terminate = MagicMock(side_effect=PermissionError("signal denied"))

    with pytest.raises(PermissionError, match="signal denied"):
        supervise_process(process, 100, terminate=terminate)

    terminate.assert_called_once_with(process)
    process.wait.assert_called_once()


def test_supervise_process_drains_fake_child_and_bounds_output(process_os):
    """Both streams are drained and truncated without launching a real child."""
    process = MagicMock(returncode=7)
    process.stdout.read.side_effect = ["abcdef", ""]
    process.stderr.read.side_effect = ["xy", ""]
    terminate = MagicMock(side_effect=lambda owned: owned.wait.assert_not_called())

    capture = supervise_process(process, time.monotonic() + 1, stdout_limit=3, terminate=terminate)

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert capture.exit_code == 7
    assert capture.stdout == "abc"
    assert capture.stdout_discarded == 3
    assert capture.stderr == "xy"
    terminate.assert_called_once_with(process)
    assert process.wait.call_count == 1
    process_os.waitid.assert_called_once_with(1, process.pid, 1 | 2 | 4)


def test_supervise_process_reaps_child_after_group_signal_failure(process_os):
    """A failed group signal remains visible after the child and reader pipes are cleaned up."""
    process = MagicMock(returncode=0)
    process.stdout.read.return_value = ""
    process.stderr.read.return_value = ""
    terminate = MagicMock(side_effect=PermissionError("signal denied"))

    with pytest.raises(PermissionError, match="signal denied"):
        supervise_process(process, time.monotonic() + 1, terminate=terminate)

    terminate.assert_called_once_with(process)
    process.wait.assert_called_once()
    assert process.stdout.close.called
    assert process.stderr.close.called


@pytest.mark.parametrize(
    ("signal_error", "expected_error"),
    [(None, OSError), (PermissionError("signal denied"), PermissionError)],
)
def test_supervise_process_surfaces_cleanup_wait_errors_after_reader_cleanup(
    process_os, signal_error, expected_error
):
    """A failed reap does not strand reader pipes or replace an earlier signal error."""
    process = MagicMock(returncode=0)
    process.stdout.read.return_value = ""
    process.stderr.read.return_value = ""
    process.wait.side_effect = OSError("wait failed")
    terminate = MagicMock(side_effect=signal_error)

    with pytest.raises(expected_error):
        supervise_process(process, time.monotonic() + 1, terminate=terminate)

    terminate.assert_called_once_with(process)
    process.wait.assert_called_once()
    assert process.stdout.close.called
    assert process.stderr.close.called


def test_supervise_process_times_out_and_reaps_fake_child(process_os):
    """Deadline expiry still terminates and reaps the owned child."""
    process = MagicMock(returncode=-9)
    process.stdout.read.return_value = ""
    process.stderr.read.return_value = ""
    process_os.waitid.side_effect = None
    process_os.waitid.return_value = None
    terminate = MagicMock()

    capture = supervise_process(process, time.monotonic() - 1, terminate=terminate)

    assert capture.status is ProcessCaptureStatus.TIMED_OUT
    terminate.assert_called_once_with(process)
    assert process.wait.call_count == 1


def test_supervise_process_cancels_before_reaping_the_group_leader(process_os):
    """An explicit cancellation signal reaches group cleanup while the leader remains waitable."""
    process = MagicMock(pid=123, stdout=MagicMock(), stderr=MagicMock(), returncode=-9)
    process.stdout.read.return_value = ""
    process.stderr.read.return_value = ""
    cancelled = threading.Event()
    cancelled.set()
    terminate = MagicMock(side_effect=lambda owned: owned.wait.assert_not_called())

    capture = supervise_process(
        process,
        time.monotonic() + 1,
        terminate=terminate,
        cancel_event=cancelled,
    )

    assert capture.status is ProcessCaptureStatus.CANCELLED
    process_os.waitid.assert_not_called()
    terminate.assert_called_once_with(process)
    process.wait.assert_called_once()


@pytest.mark.parametrize(
    ("cancelled", "expected"),
    [
        (False, ProcessCaptureStatus.TIMED_OUT),
        (True, ProcessCaptureStatus.CANCELLED),
    ],
)
def test_supervise_process_uses_kqueue_when_waitid_is_unavailable(
    monkeypatch, process_os, cancelled, expected
):
    """The kqueue fallback honors the same deadline and cancellation outcomes as waitid."""
    from loop.utils import process as process_utils

    process_os.waitid = None
    queue = MagicMock()
    event = object()
    selector = SimpleNamespace(
        KQ_FILTER_PROC=1,
        KQ_EV_ADD=2,
        KQ_EV_ENABLE=4,
        KQ_NOTE_EXIT=8,
        kevent=MagicMock(return_value=event),
        kqueue=MagicMock(return_value=queue),
    )
    monkeypatch.setattr(process_utils, "select", selector)
    process = MagicMock(pid=123, stdout=MagicMock(), stderr=MagicMock(), returncode=-9)
    process.stdout.read.return_value = ""
    process.stderr.read.return_value = ""
    cancellation = threading.Event()
    if cancelled:
        cancellation.set()
    deadline = time.monotonic() + 1 if cancelled else time.monotonic() - 1

    capture = supervise_process(
        process,
        deadline,
        terminate=MagicMock(),
        cancel_event=cancellation,
    )

    assert capture.status is expected
    selector.kevent.assert_called_once_with(
        123,
        filter=selector.KQ_FILTER_PROC,
        flags=selector.KQ_EV_ADD | selector.KQ_EV_ENABLE,
        fflags=selector.KQ_NOTE_EXIT,
    )
    queue.control.assert_called_once_with([event], 0, 0)
    queue.close.assert_called_once_with()


def test_supervise_process_fails_closed_without_a_non_reaping_posix_observer(
    monkeypatch, process_os
):
    """An unsupported POSIX host never falls back to reaping before group cleanup."""
    from loop.utils import process as process_utils

    process_os.waitid = None
    monkeypatch.setattr(process_utils, "select", SimpleNamespace())
    process = MagicMock(pid=123, stdout=MagicMock(), stderr=MagicMock(), returncode=-9)
    process.stdout.read.return_value = ""
    process.stderr.read.return_value = ""
    terminate = MagicMock()

    with pytest.raises(OSError, match="cannot be observed without reaping"):
        supervise_process(process, time.monotonic() + 1, terminate=terminate)

    terminate.assert_called_once_with(process)
    process.wait.assert_called_once()


@pytest.mark.parametrize("lookup_error", [False, True])
def test_supervise_process_completes_through_kqueue_exit_or_vanished_child(
    monkeypatch, process_os, lookup_error
):
    """The macOS observer accepts a queued exit event or an already exited child."""
    from loop.utils import process as process_utils

    process_os.waitid = None
    queue = MagicMock()
    event = object()
    queue.control.side_effect = ProcessLookupError if lookup_error else [None, [], [event]]
    selector = SimpleNamespace(
        KQ_FILTER_PROC=1,
        KQ_EV_ADD=2,
        KQ_EV_ENABLE=4,
        KQ_NOTE_EXIT=8,
        kevent=MagicMock(return_value=event),
        kqueue=MagicMock(return_value=queue),
    )
    monkeypatch.setattr(process_utils, "select", selector)
    process = MagicMock(pid=123, stdout=MagicMock(), stderr=MagicMock(), returncode=0)
    process.stdout.read.return_value = ""
    process.stderr.read.return_value = ""

    capture = supervise_process(process, time.monotonic() + 1, terminate=MagicMock())

    assert capture.status is ProcessCaptureStatus.COMPLETED
    queue.close.assert_called_once_with()
    assert queue.control.call_count == (1 if lookup_error else 3)


def test_supervise_process_keeps_the_portable_process_wait_path(monkeypatch, process_os):
    """Non-POSIX completion and timeout retain the existing direct-process behavior."""
    process_os.name = "nt"
    process = MagicMock(pid=123, stdout=MagicMock(), stderr=MagicMock(), returncode=0)
    process.stdout.read.return_value = ""
    process.stderr.read.return_value = ""
    process.wait.side_effect = [subprocess.TimeoutExpired("fake", 0.01), 0, 0]
    terminate = MagicMock()

    completed = supervise_process(process, time.monotonic() + 1, terminate=terminate)

    assert completed.status is ProcessCaptureStatus.COMPLETED
    assert completed.exit_code == 0
    terminate.assert_not_called()

    process.wait.reset_mock()
    process.wait.side_effect = None
    process.wait.return_value = -9
    timed_out = supervise_process(process, time.monotonic() - 1, terminate=terminate)

    assert timed_out.status is ProcessCaptureStatus.TIMED_OUT
    terminate.assert_called_once_with(process)

    process.wait.reset_mock()
    process.wait.side_effect = None
    cancelled = threading.Event()
    cancelled.set()
    cancellation = supervise_process(
        process,
        time.monotonic() + 1,
        terminate=terminate,
        cancel_event=cancelled,
    )

    assert cancellation.status is ProcessCaptureStatus.CANCELLED
    assert process.wait.call_count == 1
    assert terminate.call_count == 2


def test_supervise_process_polls_waitid_without_reaping_before_exit(monkeypatch, process_os):
    """A non-exit observation waits briefly and retries before group cleanup."""
    from loop.utils import process as process_utils

    process_os.waitid.side_effect = [None, SimpleNamespace(si_pid=123)]
    sleep = MagicMock()
    monkeypatch.setattr(process_utils.time, "sleep", sleep)
    process = MagicMock(pid=123, stdout=MagicMock(), stderr=MagicMock(), returncode=0)
    process.stdout.read.return_value = ""
    process.stderr.read.return_value = ""

    capture = supervise_process(process, time.monotonic() + 1, terminate=MagicMock())

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert process_os.waitid.call_count == 2
    sleep.assert_called_once()


@pytest.mark.parametrize(
    "failure,status",
    [
        (subprocess.TimeoutExpired("owned", 1), ProcessCaptureStatus.TIMED_OUT),
        (KeyboardInterrupt(), ProcessCaptureStatus.CANCELLED),
    ],
)
def test_supervise_process_captures_buffered_output_after_termination(failure, status, process_os):
    """Timeout and cancellation retain bounded output that becomes readable only at cleanup EOF."""
    process = MagicMock(returncode=-9)
    process.stdout.read.side_effect = ["ready", ""]
    process.stderr.read.side_effect = ["error", ""]
    if isinstance(failure, KeyboardInterrupt):
        process_os.waitid.side_effect = failure
    else:
        process_os.waitid.side_effect = None
        process_os.waitid.return_value = None
    terminate = MagicMock()

    def thread_factory(*, target, daemon):
        """Complete a controlled reader only when cleanup joins after owned termination."""
        thread = MagicMock()
        thread.is_alive.return_value = True

        def join(timeout):
            """Release buffered data at the modeled post-termination pipe EOF."""
            terminate.assert_called_once_with(process)
            target()
            thread.is_alive.return_value = False

        thread.join.side_effect = join
        return thread

    result = supervise_process(
        process,
        100,
        stdout_limit=3,
        stderr_limit=2,
        terminate=terminate,
        thread_factory=thread_factory,
    )
    assert result.status is status
    assert result.stdout == "rea" and result.stderr == "er"
    assert result.stdout_discarded == 2 and result.stderr_discarded == 3
    assert result.exit_code is None
