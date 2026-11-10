"""Tests for safe process invocation utilities."""

import ctypes
import errno
import io
import logging
import os
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from loop.utils.process import (
    CapturedChunks,
    ProcessCaptureStatus,
    is_finite_deadline,
    kill_process_group,
    parse_command_line,
    read_bounded_stream,
    render_terminal_text,
    supervise_process,
)


@pytest.mark.parametrize(
    ("value", "valid"),
    [(0, True), (1.5, True), (True, False), ("1", False), (10**1000, False)],
)
def test_deadline_validation_accepts_only_representable_numeric_values(value, valid):
    """Validation rejects booleans, nonnumeric values, and unrepresentable integers."""
    assert is_finite_deadline(value) is valid


def test_terminal_output_renders_controls_without_executing_ansi_or_clipboard_sequences():
    """Untrusted cursor, clipboard and hyperlink controls become visible text only."""
    source = "hello\x1b[31m\x1b]52;c;secret\x07\r\n\t\x7f\x9b"
    assert (
        render_terminal_text(source)
        == "hello\u241b[31m\u241b]52;c;secret\u2407\u240d\n\u2409\u2421\ufffd"
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


@pytest.mark.parametrize("preserve", [False, True])
def test_supervised_reader_preserves_requested_newline_semantics(monkeypatch, process_os, preserve):
    """Cooperative pipe polling retains the text stream's newline behavior."""

    class PollableBytes(io.BytesIO):
        """Represent already-available bytes behind a selectable descriptor."""

        def fileno(self):
            return 42

    poller = MagicMock()
    poller.poll.side_effect = [[], [(42, 1)], [(42, 1)], [(42, 1)]]
    monkeypatch.setattr("loop.utils.process.select.poll", lambda: poller)
    stream = io.TextIOWrapper(PollableBytes(b"a\r\nb\r"), encoding="utf-8")
    chunks = CapturedChunks("stdout", None, threading.Event(), preserve)
    assert read_bounded_stream(stream, chunks) == 0
    assert "".join(chunks) == ("a\r\nb\r" if preserve else "a\nb\n")
    poller.register.assert_called_once()


def test_supervised_reader_stops_when_no_pipe_data_arrives(monkeypatch, process_os):
    """An idle inherited pipe stops promptly without pretending its output was complete."""

    class PollableBytes(io.BytesIO):
        """Represent a selectable pipe whose writer remains open."""

        def fileno(self):
            return 42

    stop = threading.Event()
    poller = MagicMock()

    def stop_waiting(_timeout):
        """Request cleanup while the mocked pipe is idle."""
        stop.set()
        return []

    poller.poll.side_effect = stop_waiting
    monkeypatch.setattr("loop.utils.process.select.poll", lambda: poller)
    stream = io.TextIOWrapper(PollableBytes(), encoding="utf-8")
    chunks = CapturedChunks("stdout", None, stop)
    assert read_bounded_stream(stream, chunks) is None
    assert stream.closed
    assert not chunks


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
    """A closed descriptor reports unknown drain completeness without losing captured text."""
    stream = MagicMock()
    stream.read.side_effect = ["partial", OSError(errno.EBADF, "Bad file descriptor")]
    chunks = []

    assert read_bounded_stream(stream, chunks, maximum=20) is None
    assert chunks == ["partial"]


def test_read_bounded_stream_preserves_unrelated_read_failures():
    """An unrelated stream error still reaches its owning supervisor."""
    stream = MagicMock()
    stream.read.side_effect = OSError(errno.EIO, "I/O error")

    with pytest.raises(OSError, match="I/O error"):
        read_bounded_stream(stream, [], maximum=20)


def test_read_bounded_stream_accepts_only_an_explicit_pty_eof():
    """A PTY's EIO end condition completes capture without weakening pipe error handling."""
    stream = MagicMock()
    stream.read.side_effect = ["terminal", OSError(errno.EIO, "slave closed")]
    chunks = []

    assert read_bounded_stream(stream, chunks, maximum=4, eof_errno=errno.EIO) == 4
    assert chunks == ["term"]


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
    """A descriptor closed during cleanup leaves drain completeness unknown."""
    stream = MagicMock()
    stream.read.return_value = ""
    stream.close.side_effect = OSError(errno.EBADF, "already closed")

    assert read_bounded_stream(stream, [], maximum=20) is None


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


def test_supervise_process_rejects_invalid_deadline_with_missing_pipe():
    """Invalid deadlines still terminate and reap a child with a missing capture pipe."""
    process = MagicMock(returncode=-9, stdout=None)
    terminate = MagicMock()
    with pytest.raises(ValueError, match="Process deadline must be finite"):
        supervise_process(process, float("nan"), terminate=terminate)
    terminate.assert_called_once_with(process)
    process.wait.assert_called_once()


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), float("-inf")])
def test_supervise_process_reaps_child_when_deadline_is_nonfinite(process_os, deadline):
    """An invalid deadline still terminates and reaps the already-started child."""
    process = MagicMock(returncode=-9)
    terminate = MagicMock()
    with pytest.raises(ValueError, match="Process deadline must be finite"):
        supervise_process(process, deadline, terminate=terminate)
    terminate.assert_called_once_with(process)
    process.wait.assert_called_once()
    process.stdout.close.assert_called_once()
    process.stderr.close.assert_called_once()


@pytest.mark.parametrize(
    ("failure", "index", "alive", "exception"),
    [
        ("construct", 0, False, RuntimeError("reader setup failed")),
        ("construct", 1, False, RuntimeError("reader setup failed")),
        ("start", 0, False, RuntimeError("reader setup failed")),
        ("start", 1, False, RuntimeError("reader setup failed")),
        ("start", 1, True, RuntimeError("reader setup failed")),
        ("start", 0, False, KeyboardInterrupt()),
    ],
)
def test_supervise_process_cleans_up_when_reader_setup_fails(
    process_os, failure, index, alive, exception
):
    """A reader setup error cannot leave the already spawned process unsupervised."""
    process = MagicMock(returncode=-9)
    terminate = MagicMock()
    threads = [MagicMock(), MagicMock()]
    for thread in threads:
        thread.is_alive.return_value = False
    if alive:
        threads[index].is_alive.side_effect = [True, False]
    if failure == "start":
        threads[index].start.side_effect = exception
    created = []

    def make_thread(**_kwargs):
        """Return controlled reader threads or fail while constructing one."""
        current = len(created)
        if failure == "construct" and current == index:
            raise exception
        created.append(threads[current])
        return threads[current]

    with pytest.raises(type(exception)):
        supervise_process(process, 100, terminate=terminate, thread_factory=make_thread)

    terminate.assert_called_once_with(process)
    process.wait.assert_called_once()
    if failure == "start" and (index == 1 or alive):
        threads[0].join.assert_called_once()
    if failure == "start" and alive:
        threads[index].join.assert_called_once()
    process.stdout.close.assert_called_once()
    process.stderr.close.assert_called_once()


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


def test_supervise_process_observes_output_beyond_final_capture(process_os):
    """A live observer receives later stdout and stderr while final capture stays bounded."""
    process = MagicMock(returncode=0)
    process.stdout.read.side_effect = ["abcdef", "later", ""]
    process.stderr.read.side_effect = ["xy", ""]
    observed = []

    capture = supervise_process(
        process,
        time.monotonic() + 1,
        stdout_limit=3,
        output_observer=lambda stream, chunk: observed.append((stream, chunk)),
    )

    assert [(stream, chunk) for stream, chunk in observed if stream == "stdout"] == [
        ("stdout", "abcdef"),
        ("stdout", "later"),
    ]
    assert ("stderr", "xy") in observed
    assert capture.stdout == "abc"
    assert capture.stderr == "xy"
    assert capture.stdout_discarded == 8


def test_supervise_process_preserves_capture_when_live_observer_fails(process_os):
    """A failed live sink retains the bounded chunk and reports incomplete capture."""
    process = MagicMock(returncode=0)
    process.stdout.read.side_effect = ["abcdef", ""]
    process.stderr.read.return_value = ""

    def fail_observer(_stream, _chunk):
        """Model a failed live output consumer."""
        raise OSError("live sink failed")

    capture = supervise_process(
        process, time.monotonic() + 1, stdout_limit=3, output_observer=fail_observer
    )

    assert capture.status is ProcessCaptureStatus.UNAVAILABLE
    assert capture.stdout == "abc"
    assert capture.stdout_discarded is None


def test_supervise_process_decodes_available_live_bytes_before_pipe_eof(process_os):
    """Live capture decodes split UTF-8 and preserves newlines from available pipe chunks."""

    class AvailableBytes(io.BytesIO):
        """Supply one raw byte per read without waiting for the whole text stream."""

        def read1(self, size):
            """Return the next available raw byte even when a larger read was requested."""
            return super().read(min(size, 1))

    process = MagicMock(returncode=0)
    process.stdout = io.TextIOWrapper(AvailableBytes("é\r\nnext".encode()), encoding="utf-8")
    process.stderr = io.TextIOWrapper(AvailableBytes(b"err"), encoding="utf-8")
    observed = []

    capture = supervise_process(
        process,
        time.monotonic() + 1,
        preserve_newlines=True,
        output_observer=lambda stream, chunk: observed.append((stream, chunk)),
    )

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert capture.stdout == "é\r\nnext"
    assert "".join(chunk for stream, chunk in observed if stream == "stdout") == capture.stdout
    assert "".join(chunk for stream, chunk in observed if stream == "stderr") == "err"


@pytest.mark.parametrize(
    ("data", "limit", "expected", "discarded"),
    [(b"\xc3", 0, "", 1), (b"\xc3", 1, "\ufffd", 0), (b"ok", 2, "ok", 0)],
)
def test_supervise_process_flushes_partial_utf8_at_pty_eof(
    process_os, data, limit, expected, discarded
):
    """A closing PTY counts incomplete UTF-8 and handles a fully decoded final chunk."""

    class PtyBytes(io.BytesIO):
        """Raise the native PTY EOF error after the supplied bytes are decoded."""

        def read1(self, size):
            """Return a byte once and then report the closed slave."""
            value = super().read(min(size, 1))
            if value:
                return value
            raise OSError(errno.EIO, "slave closed")

    process = MagicMock(returncode=0)
    process.stdout = io.TextIOWrapper(PtyBytes(data), encoding="utf-8", errors="replace")
    process.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    observed = []

    def read(stream, chunks, *, maximum):
        """Apply terminal EOF semantics to only the owned PTY stream."""
        return read_bounded_stream(
            stream,
            chunks,
            maximum=maximum,
            eof_errno=errno.EIO if stream is process.stdout else None,
        )

    capture = supervise_process(
        process,
        time.monotonic() + 1,
        stdout_limit=limit,
        reader=read,
        output_observer=lambda stream, chunk: observed.append((stream, chunk)),
    )

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert capture.stdout == expected
    assert capture.stdout_discarded == discarded
    assert "".join(content for stream, content in observed if stream == "stdout") == data.decode(
        "utf-8", errors="replace"
    )


@pytest.mark.parametrize("failed_stream", ["stdout", "stderr"])
@pytest.mark.parametrize("failure", ["exception", "incomplete"])
def test_supervise_process_does_not_report_success_after_reader_failure(
    process_os, failed_stream, failure
):
    """A failed reader preserves partial output and cannot report a complete successful capture."""
    process = MagicMock(returncode=0)
    uncaught = []

    def reader(stream, chunks, *, maximum):
        """Produce controlled output and fail only for the selected pipe."""
        chunks.append("partial" if stream is getattr(process, failed_stream) else "complete")
        if stream is getattr(process, failed_stream):
            if failure == "exception":
                raise OSError(errno.EIO, "synthetic reader failure")
            return None
        return 0

    def thread_factory(*, target, daemon):
        """Model a finished reader thread without propagating its exception to start()."""
        thread = MagicMock()

        def start():
            """Record an exception that would otherwise escape inside the reader thread."""
            try:
                target()
            except OSError as error:
                uncaught.append(error)

        thread.start.side_effect = start
        thread.is_alive.return_value = False
        return thread

    terminate = MagicMock()
    capture = supervise_process(
        process,
        time.monotonic() + 1,
        terminate=terminate,
        reader=reader,
        thread_factory=thread_factory,
    )

    assert capture.status is ProcessCaptureStatus.UNAVAILABLE
    assert capture.exit_code is None
    assert getattr(capture, failed_stream) == "partial"
    assert getattr(capture, f"{failed_stream}_discarded") is None
    assert not uncaught
    terminate.assert_called_once_with(process)
    process.wait.assert_called_once()


@pytest.mark.parametrize("cancelled", [False, True])
def test_supervise_process_keeps_interruption_outcome_when_reader_fails(process_os, cancelled):
    """Timeout and cancellation remain distinct while a failed drain stays incomplete."""
    process = MagicMock(returncode=-9)
    process.stdout.read.side_effect = ["partial", OSError(errno.EIO, "reader failed")]
    process.stderr.read.return_value = ""
    event = threading.Event()
    if cancelled:
        event.set()
    else:
        process_os.waitid.side_effect = None
        process_os.waitid.return_value = None

    capture = supervise_process(
        process,
        time.monotonic() + 1 if cancelled else time.monotonic() - 1,
        terminate=MagicMock(),
        cancel_event=event,
    )

    assert capture.status is (
        ProcessCaptureStatus.CANCELLED if cancelled else ProcessCaptureStatus.TIMED_OUT
    )
    assert capture.stdout == "partial" and capture.stdout_discarded is None
    assert capture.stderr_discarded == 0


def test_supervise_process_configures_payload_pipes_before_reading(process_os):
    """Payload readers preserve carriage returns and existing decoded-character bounds."""
    process = MagicMock(returncode=0)
    process.stdout.read.side_effect = ["a\rb\r\nc\n", ""]
    process.stderr.read.side_effect = ["x\ry", ""]

    capture = supervise_process(
        process, time.monotonic() + 1, terminate=MagicMock(), preserve_newlines=True
    )

    assert capture.stdout == "a\rb\r\nc\n"
    assert capture.stderr == "x\ry"
    process.stdout.reconfigure.assert_called_once_with(newline="")
    process.stderr.reconfigure.assert_called_once_with(newline="")


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_supervise_process_reaps_child_when_pipe_configuration_fails(process_os, stream_name):
    """Failed text-pipe configuration still terminates and reaps the owned child."""
    process = MagicMock(returncode=-9)
    getattr(process, stream_name).reconfigure.side_effect = OSError("pipe setup failed")
    terminate = MagicMock()

    with pytest.raises(OSError, match="pipe setup failed"):
        supervise_process(process, 100, terminate=terminate, preserve_newlines=True)

    terminate.assert_called_once_with(process)
    process.wait.assert_called_once()
    process.stdout.close.assert_called_once()
    process.stderr.close.assert_called_once()


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


@pytest.mark.parametrize("cancel_reader", [0, 1])
def test_supervise_process_cancels_while_descendants_retain_pipes(
    monkeypatch, process_os, cancel_reader
):
    """Cancellation interrupts pipe draining after leader exit and retains cleanup output."""
    clock = SimpleNamespace(current=10.0)
    monkeypatch.setattr("loop.utils.process.time.monotonic", lambda: clock.current)
    process = MagicMock(pid=123, returncode=0)
    process.stdout.read.side_effect = ["ready", ""]
    process.stderr.read.side_effect = ["error", ""]
    cancelled = threading.Event()
    terminate = MagicMock(side_effect=lambda owned: owned.wait.assert_not_called())
    readers = []

    def thread_factory(*, target, daemon):
        """Keep the selected pipe open until group cleanup releases its EOF."""
        index = len(readers)
        thread = MagicMock()
        thread.is_alive.return_value = True

        def join(timeout):
            """Request cancellation during a bounded join, then drain after termination."""
            if not thread.is_alive():
                return
            if index < cancel_reader or terminate.called:
                target()
                thread.is_alive.return_value = False
            else:
                clock.current += timeout
                cancelled.set()

        thread.join.side_effect = join
        readers.append(thread)
        return thread

    capture = supervise_process(
        process,
        100,
        terminate=terminate,
        thread_factory=thread_factory,
        cancel_event=cancelled,
    )

    assert capture.status is ProcessCaptureStatus.CANCELLED
    assert clock.current < 11
    assert capture.stdout == "ready" and capture.stderr == "error"
    assert capture.stdout_discarded == capture.stderr_discarded == 0
    assert capture.exit_code is None
    terminate.assert_called_once_with(process)
    process.wait.assert_called_once()
    process_os.waitid.assert_called_once()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("complete", ProcessCaptureStatus.COMPLETED),
        ("deadline", ProcessCaptureStatus.TIMED_OUT),
        ("cancel-on-exit", ProcessCaptureStatus.CANCELLED),
    ],
)
def test_supervise_process_bounds_cancellable_draining_after_leader_exit(
    monkeypatch, process_os, mode, expected
):
    """Cancellable draining preserves completion, deadlines and cancellation at leader exit."""
    clock = SimpleNamespace(current=10.0)
    monkeypatch.setattr("loop.utils.process.time.monotonic", lambda: clock.current)
    process = MagicMock(pid=123, returncode=0)
    process.stdout.read.side_effect = ["ready", ""]
    process.stderr.read.side_effect = ["error", ""]
    cancelled = threading.Event()
    terminate = MagicMock()
    if mode == "cancel-on-exit":
        process_os.waitid.side_effect = lambda *args: cancelled.set() or SimpleNamespace(si_pid=123)

    def thread_factory(*, target, daemon):
        """Model reader completion or a retained pipe under a controlled clock."""
        thread = MagicMock()
        thread.is_alive.return_value = True

        def join(timeout):
            """Advance a retained pipe's wait or finish reading available output once."""
            if thread.is_alive():
                if mode == "complete" or terminate.called:
                    target()
                    thread.is_alive.return_value = False
                else:
                    clock.current += timeout

        thread.join.side_effect = join
        return thread

    capture = supervise_process(
        process,
        10.025,
        terminate=terminate,
        thread_factory=thread_factory,
        cancel_event=cancelled,
    )

    assert capture.status is expected
    assert capture.stdout == "ready" and capture.stderr == "error"
    assert capture.stdout_discarded == capture.stderr_discarded == 0
    assert capture.exit_code == (0 if mode == "complete" else None)
    assert clock.current <= 10.025
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
