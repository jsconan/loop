"""Exercise owned native processes in disposable workspaces."""

import errno
import io
import logging
import os
import select
import signal
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from loop import constants
from loop.utils.process import (
    ProcessCaptureStatus,
    read_bounded_stream,
    supervise_process,
)

pytestmark = pytest.mark.integration


@contextmanager
def owned_pipe():
    """Own both pipe ends even if wrapping either descriptor fails."""
    read_fd, write_fd = os.pipe()
    with ExitStack() as cleanup:
        try:
            reader = os.fdopen(read_fd, "rb", buffering=0)
        except BaseException:
            os.close(read_fd)
            os.close(write_fd)
            raise
        cleanup.enter_context(reader)
        try:
            writer = os.fdopen(write_fd, "wb", buffering=0)
        except BaseException:
            os.close(write_fd)
            raise
        cleanup.enter_context(writer)
        yield reader, writer


@contextmanager
def detached_pipe_processes():
    """Own a leader, detached writer, and their pipe descriptors through setup failures."""

    def stop_process(process):
        """Stop and reap only the process created by this fixture."""
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        process.wait(timeout=5)

    with ExitStack() as cleanup:
        read_pipe, write_pipe = cleanup.enter_context(owned_pipe())
        wake_read, wake_write = cleanup.enter_context(owned_pipe())
        writer = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                (
                    "import os,sys; os.read(int(sys.argv[1]),1)\n"
                    "try: os.write(1,b'late')\nexcept BrokenPipeError: pass"
                ),
                str(wake_read.fileno()),
            ],
            stdout=write_pipe,
            stderr=subprocess.DEVNULL,
            pass_fds=(wake_read.fileno(),),
            start_new_session=True,
        )
        cleanup.callback(stop_process, writer)
        wake_read.close()
        process = subprocess.Popen(
            [sys.executable, "-I", "-c", "import os; os.write(1,b'before')"],
            stdout=write_pipe,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        cleanup.callback(stop_process, process)
        if process.stderr is not None:
            cleanup.enter_context(process.stderr)
        write_pipe.close()
        process.stdout = cleanup.enter_context(
            io.TextIOWrapper(io.BufferedReader(read_pipe), encoding="utf-8")
        )
        yield process, writer, wake_write


@pytest.fixture
def process_clock(monkeypatch):
    """Provide a fixed process clock while yielding to native child scheduling."""
    monkeypatch.setattr(
        "loop.utils.process.time",
        SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda _delay: os.sched_yield()),
    )


def test_supervise_process_bounds_both_streams_and_preserves_exit(caplog, process_clock):
    """The shared supervisor drains real child pipes and reports discarded output."""
    caplog.set_level(logging.WARNING, logger="loop.utils.process")
    process = subprocess.Popen(
        ["/bin/sh", "-c", "printf abcdef; printf xy >&2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    capture = supervise_process(process, 110.0, stdout_limit=3)

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert capture.exit_code == 0
    assert capture.stdout == "abc"
    assert capture.stdout_discarded == 3
    assert capture.stderr == "xy"
    assert not caplog.records


def test_supervise_process_streams_after_native_capture_limit(process_clock):
    """An owned child continues to publish output after its final capture is full."""
    process = subprocess.Popen(
        ["/bin/sh", "-c", "printf abcdef; printf later"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    observed = []

    capture = supervise_process(
        process, 110.0, stdout_limit=3, output_observer=lambda stream, chunk: observed.append(chunk)
    )

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert capture.stdout == "abc"
    assert capture.stdout_discarded == 8
    assert "".join(observed) == "abcdeflater"


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX inherited pipes and sessions")
def test_supervisor_stops_readers_when_detached_writer_retains_a_pipe(monkeypatch):
    """A separately owned writer cannot retain a completed capture's reader threads."""
    monkeypatch.setattr(constants, "COMMAND_CLEANUP_WAIT_SECONDS", 0.5)
    threads = []

    def create_reader(**kwargs):
        """Keep the actual reader handles for the post-capture check."""
        thread = threading.Thread(**kwargs)
        threads.append(thread)
        return thread

    output = []
    with detached_pipe_processes() as (process, writer, wake_write):
        capture = supervise_process(
            process,
            time.monotonic() + 0.4,
            thread_factory=create_reader,
            output_observer=lambda _stream, content: output.append(content),
        )
        assert capture.status is ProcessCaptureStatus.TIMED_OUT
        assert capture.stdout == "before"
        assert all(not thread.is_alive() for thread in threads)
        before = list(output)
        wake_write.write(b"x")
        writer.wait(timeout=5)
        assert output == before


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX inherited pipes and sessions")
@pytest.mark.parametrize("failed_wrap", [1, 2])
def test_owned_pipe_closes_descriptors_when_wrapping_fails(monkeypatch, failed_wrap):
    """Neither end survives failure to wrap either newly created pipe descriptor."""
    make_pipe, wrap = os.pipe, os.fdopen
    descriptors = []
    calls = 0

    def recorded_pipe():
        """Retain the exact descriptors for the cleanup assertion."""
        pair = make_pipe()
        descriptors.extend(pair)
        return pair

    def fail_wrap(*args, **kwargs):
        """Fail at the selected descriptor without taking ownership of it."""
        nonlocal calls
        calls += 1
        if calls == failed_wrap:
            raise OSError("pipe wrapping failed")
        return wrap(*args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "pipe", recorded_pipe)
        patcher.setattr(os, "fdopen", fail_wrap)
        with pytest.raises(OSError, match="pipe wrapping failed"), owned_pipe():
            pytest.fail("The pipe wrapping unexpectedly succeeded")
    assert len(descriptors) == 2
    for descriptor in descriptors:
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX inherited pipes and sessions")
def test_detached_writer_is_reaped_when_leader_setup_fails(monkeypatch):
    """A failed second launch still stops and reaps the already-started writer."""
    start = subprocess.Popen
    created = []

    def fail_leader(*args, **kwargs):
        """Start the writer and reject the later leader launch."""
        if created:
            raise OSError("leader launch failed")
        process = start(*args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", fail_leader)
    with pytest.raises(OSError, match="leader launch failed"), detached_pipe_processes():
        pytest.fail("The leader launch unexpectedly succeeded")
    assert len(created) == 1
    assert created[0].poll() is not None


def test_supervise_process_reports_reader_failure_after_native_child_exit(process_clock):
    """A reader failure preserves partial output after the owned native child is reaped."""
    process = subprocess.Popen(
        ["/bin/sh", "-c", "printf partial; printf error >&2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    def reader(stream, chunks, *, maximum):
        """Fail the stdout reader after capturing the child's real pipe output."""
        if stream is process.stdout:
            chunks.append(stream.read(len("partial")))
            raise OSError("synthetic capture failure")
        return read_bounded_stream(stream, chunks, maximum=maximum)

    capture = supervise_process(process, 110.0, reader=reader)

    assert capture.status is ProcessCaptureStatus.UNAVAILABLE
    assert capture.exit_code is None
    assert capture.stdout == "partial" and capture.stdout_discarded is None
    assert capture.stderr == "error" and capture.stderr_discarded == 0
    assert process.returncode == 0
    assert process.stdout.closed and process.stderr.closed


def test_supervise_process_reaps_native_child_when_reader_start_fails():
    """Reader setup failure terminates and reaps the real owned process group."""
    blocked_read, blocked_write = os.pipe()
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            "import os,sys; os.read(int(sys.argv[1]),1)",
            str(blocked_read),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        pass_fds=(blocked_read,),
    )
    os.close(blocked_read)
    created = 0

    def thread_factory(**kwargs):
        """Start one real reader and fail before starting the second."""
        nonlocal created
        created += 1
        thread = threading.Thread(**kwargs)
        if created == 2:
            thread.start = MagicMock(side_effect=RuntimeError("reader start failed"))
        return thread

    try:
        with pytest.raises(RuntimeError, match="reader start failed"):
            supervise_process(process, 100.0, thread_factory=thread_factory)

        assert process.returncode == -signal.SIGKILL
        assert process.stdout.closed and process.stderr.closed
    finally:
        os.close(blocked_write)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_supervise_process_kills_a_child_after_deadline(process_boundary, caplog):
    """A deterministic deadline failure terminates and reaps the complete owned group."""
    boundary = process_boundary
    boundary.waitid.side_effect = None
    boundary.waitid.return_value = None
    caplog.set_level(logging.WARNING, logger="loop.utils.process")

    capture = supervise_process(boundary.process, 100, thread_factory=boundary.thread_factory)

    assert capture.status is ProcessCaptureStatus.TIMED_OUT
    boundary.kill.assert_called_once_with(123, 9)
    assert boundary.process.wait.call_count == 1
    assert not caplog.records


def test_supervise_process_retains_flushed_short_output_at_timeout(caplog, monkeypatch):
    """A fake deadline retains real flushed pipe output and reaps the owned child."""
    ready_read, ready_write = os.pipe()
    blocked_read, blocked_write = os.pipe()
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import os,sys; print('STARTED',flush=True); "
                "print('DIAGNOSTIC',file=sys.stderr,flush=True); "
                "os.write(int(sys.argv[1]),b'r'); os.close(int(sys.argv[1])); "
                "os.read(int(sys.argv[2]),1)"
            ),
            str(ready_write),
            str(blocked_read),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        pass_fds=(ready_write, blocked_read),
    )
    os.close(ready_write)
    os.close(blocked_read)
    observed = False

    def expired_after_entry():
        """Advance the fake deadline only after the child flushes both streams."""
        nonlocal observed
        if not observed:
            assert os.read(ready_read, 1) == b"r"
            observed = True
        return 101.0

    monkeypatch.setattr("loop.utils.process.time", SimpleNamespace(monotonic=expired_after_entry))
    try:
        capture = supervise_process(process, 100.0)
    finally:
        os.close(ready_read)
        os.close(blocked_write)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    assert capture.status is ProcessCaptureStatus.TIMED_OUT
    assert capture.stdout == "STARTED\n"
    assert capture.stderr == "DIAGNOSTIC\n"
    assert capture.stdout_discarded == 0 and capture.stderr_discarded == 0
    assert process.poll() is not None
    assert not caplog.records


def test_supervise_process_cancels_retained_pipes_after_leader_exit(caplog, monkeypatch):
    """Real inherited pipes are drained after prompt cancellation and owned group cleanup."""
    cancelled = threading.Event()
    blocked_read, blocked_write = os.pipe()
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import os,sys\n"
                "if os.fork() == 0:\n"
                " os.read(int(sys.argv[1]),1)\n"
                " os._exit(0)\n"
                "print('ready',end='',flush=True)\n"
                "print('error',end='',file=sys.stderr,flush=True)\n"
            ),
            str(blocked_read),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        pass_fds=(blocked_read,),
    )
    os.close(blocked_read)
    monkeypatch.setattr(
        "loop.utils.process.time",
        SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda _delay: os.sched_yield()),
    )

    def thread_factory(**kwargs):
        """Use real readers and request cancellation at their first post-exit join."""
        thread = threading.Thread(**kwargs)
        original_join = thread.join

        def join(timeout):
            """Cancel after the leader is observed exiting, before a retained pipe drains."""
            cancelled.set()
            original_join(timeout)

        thread.join = join
        return thread

    try:
        capture = supervise_process(
            process,
            110.0,
            thread_factory=thread_factory,
            cancel_event=cancelled,
        )
    finally:
        os.close(blocked_write)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    assert capture.status is ProcessCaptureStatus.CANCELLED
    assert capture.stdout == "ready" and capture.stderr == "error"
    assert capture.stdout_discarded == capture.stderr_discarded == 0
    assert capture.exit_code is None and process.returncode is not None
    assert process.stdout.closed and process.stderr.closed
    assert not caplog.records


def test_supervise_process_kills_same_group_descendant_after_parent_exit(process_boundary):
    """Retained descendant pipes trigger group termination and forced descriptor cleanup."""
    boundary = process_boundary

    def retained_thread(**kwargs):
        """Represent a reader held open by a descendant without creating a real thread."""
        thread = boundary.thread_factory(**kwargs)
        thread.is_alive.return_value = True
        return thread

    capture = supervise_process(boundary.process, 101, thread_factory=retained_thread)

    assert capture.status is ProcessCaptureStatus.TIMED_OUT
    boundary.kill.assert_called_once_with(123, 9)
    assert [call.args for call in boundary.close.call_args_list] == [(101,), (102,)]
    assert boundary.process.wait.call_count == 1
    assert all(thread.join.call_count == 2 for thread in boundary.threads)


def test_native_child_receives_only_the_isolated_environment(tmp_path, process_clock):
    """A real child receives only the explicit fixture environment and no ambient settings."""
    process = subprocess.Popen(
        ["/usr/bin/env"],
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    terminate = MagicMock()
    capture = supervise_process(process, 110.0, terminate=terminate)

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert capture.exit_code == 0
    inherited = dict(line.split("=", 1) for line in capture.stdout.splitlines())
    assert inherited == dict(os.environ)
    assert inherited["HOME"] == str(tmp_path.parent / "home")
    assert "OPENAI_API_KEY" not in inherited
    assert "SSH_AUTH_SOCK" not in inherited


def test_supervisor_terminates_same_group_descendant_after_parent_exit(tmp_path, monkeypatch):
    """An exited leader retains its group identity until a pipe-holding child is terminated."""
    marker = tmp_path / "descendant-survived"
    gate_read, gate_write = os.pipe()
    ready_read, ready_write = os.pipe()
    exited_read, exited_write = os.pipe()
    source = f"""import os
pid=os.fork()
if pid == 0:
 os.close({ready_read})
 os.write(1,b'descendant\\n')
 os.write({ready_write},b'rr')
 os.close({ready_write})
 os.read({gate_read},1)
 open({str(marker)!r},'w').write('survived')
else:
 os.close({ready_write})
 os.read({ready_read},1)
os._exit(0)
"""
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        pass_fds=(gate_read, ready_read, ready_write, exited_write),
    )
    for descriptor in (gate_read, ready_write, exited_write):
        os.close(descriptor)
    clock = [100.0]
    monkeypatch.setattr(
        "loop.utils.process.time",
        SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda _delay: os.sched_yield()),
    )

    def thread_factory(**kwargs):
        """Advance the fake deadline when output draining begins after leader exit."""
        thread = threading.Thread(**kwargs)
        original_join = thread.join

        def join(timeout):
            """Allow the timeout path to inspect a still-open inherited pipe."""
            if clock[0] == 100.0:
                clock[0] = 102.0
                original_join(0)
            else:
                original_join(timeout)

        thread.join = join
        return thread

    try:
        assert os.read(ready_read, 1) == b"r"
        capture = supervise_process(process, 101.0, thread_factory=thread_factory)
        try:
            os.write(gate_write, b"x")
        except BrokenPipeError:
            pass
        os.close(gate_write)
        gate_write = -1
        assert os.read(exited_read, 1) == b""
    finally:
        if gate_write >= 0:
            os.close(gate_write)
        os.close(ready_read)
        os.close(exited_read)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    assert capture.status is ProcessCaptureStatus.TIMED_OUT
    assert capture.stdout == "descendant\n"
    assert capture.stdout_discarded == 0
    assert process.poll() is not None
    assert not marker.exists()


def test_supervisor_kills_same_group_descendant_after_output_pipes_close(tmp_path, monkeypatch):
    """A live descendant is terminated even after its leader and output pipes finish."""
    marker = tmp_path / "descendant-survived"
    gate_read, gate_write = os.pipe()
    ready_read, ready_write = os.pipe()
    exited_read, exited_write = os.pipe()
    source = f"""import os
pid=os.fork()
if pid == 0:
 os.close({ready_read})
 os.close(1)
 os.close(2)
 os.write({ready_write},b'rr')
 os.close({ready_write})
 os.read({gate_read},1)
 open({str(marker)!r},'w').write('survived')
else:
 os.close({ready_write})
 os.read({ready_read},1)
os._exit(0)
"""
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        pass_fds=(gate_read, ready_read, ready_write, exited_write),
    )
    for descriptor in (gate_read, ready_write, exited_write):
        os.close(descriptor)
    monkeypatch.setattr(
        "loop.utils.process.time",
        SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda _delay: os.sched_yield()),
    )
    try:
        assert os.read(ready_read, 1) == b"r"
        capture = supervise_process(process, 110.0)
        try:
            os.write(gate_write, b"x")
        except BrokenPipeError:
            pass
        os.close(gate_write)
        gate_write = -1
        assert os.read(exited_read, 1) == b""
    finally:
        if gate_write >= 0:
            os.close(gate_write)
        os.close(ready_read)
        os.close(exited_read)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert capture.exit_code == 0
    assert process.poll() == 0
    assert not marker.exists()


@pytest.mark.skipif(not hasattr(select, "kqueue"), reason="requires the macOS kqueue observer")
def test_supervisor_observes_an_unpolled_child_when_waitid_is_unavailable(monkeypatch):
    """The kqueue compatibility path keeps an already-exited Popen child waitable."""
    from loop.utils import process as process_utils

    monkeypatch.setattr(process_utils.os, "waitid", None, raising=False)
    process = subprocess.Popen(
        ["/bin/sh", "-c", "printf kqueue-ok"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    observer = select.kqueue()
    try:
        event = select.kevent(
            process.pid,
            filter=select.KQ_FILTER_PROC,
            flags=select.KQ_EV_ADD | select.KQ_EV_ENABLE,
            fflags=select.KQ_NOTE_EXIT,
        )
        try:
            observer.control([event], 0, 0)
            assert observer.control(None, 1, None)
        except ProcessLookupError:
            pass  # An exited but unreaped child may disappear before kqueue registration.
    finally:
        observer.close()
    assert process.returncode is None

    terminate = MagicMock()
    monkeypatch.setattr(process_utils, "time", SimpleNamespace(monotonic=lambda: 100.0))
    capture = supervise_process(process, 110.0, terminate=terminate)

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert capture.exit_code == 0
    assert capture.stdout == "kqueue-ok"
    terminate.assert_called_once_with(process)


def test_stale_group_identifier_reuse_cannot_signal_an_unrelated_sentinel(monkeypatch):
    """A simulated PGID reuse after leader reaping leaves the unrelated sentinel alive."""
    from loop.utils import process as process_utils

    blocked_read, blocked_write = os.pipe()
    sentinel = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            "import os,sys; os.read(int(sys.argv[1]),1)",
            str(blocked_read),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        pass_fds=(blocked_read,),
    )
    os.close(blocked_read)
    reaped = False
    child = MagicMock(
        pid=sentinel.pid,
        stdout=MagicMock(),
        stderr=MagicMock(),
        returncode=0,
    )
    child.stdout.read.return_value = ""
    child.stderr.read.return_value = ""

    def waitid(_kind, pid, _options):
        """Keep the simulated leader waitable until the supervisor finishes group cleanup."""
        return type("WaitInfo", (), {"si_pid": pid})()

    def reap(*, timeout=None):
        """Model the point after which the leader's old numeric group ID may be reused."""
        nonlocal reaped
        reaped = True
        return 0

    def terminate(process):
        """Map a stale signal to the new group that now owns the same numeric identifier."""
        if reaped:
            os.killpg(process.pid, signal.SIGKILL)

    child.wait.side_effect = reap
    monkeypatch.setattr(process_utils.os, "waitid", waitid, raising=False)
    monkeypatch.setattr(process_utils, "time", SimpleNamespace(monotonic=lambda: 100.0))
    for name, value in (("P_PID", 1), ("WEXITED", 1), ("WNOHANG", 2), ("WNOWAIT", 4)):
        monkeypatch.setattr(process_utils.os, name, value, raising=False)
    try:
        capture = supervise_process(
            child,
            110.0,
            terminate=terminate,
        )

        assert capture.status is ProcessCaptureStatus.COMPLETED
        assert child.wait.call_count == 1
        assert sentinel.poll() is None
        os.killpg(sentinel.pid, 0)
    finally:
        sentinel.kill()
        sentinel.wait()
        os.close(blocked_write)
