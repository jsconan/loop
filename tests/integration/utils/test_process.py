"""Exercise owned native processes in disposable workspaces."""

import logging
import os
import select
import signal
import subprocess
import sys
import time
from unittest.mock import MagicMock

import pytest

from loop.utils.process import (
    ProcessCaptureStatus,
    supervise_process,
)

pytestmark = pytest.mark.integration


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
    assert not caplog.records


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


def test_supervise_process_retains_flushed_short_output_at_timeout(caplog):
    """A real blocking pipe reader delivers flushed short output during bounded timeout cleanup."""
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys,time; print('STARTED',flush=True); "
                "print('DIAGNOSTIC',file=sys.stderr,flush=True); time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    capture = supervise_process(process, time.monotonic() + 1)
    assert capture.status is ProcessCaptureStatus.TIMED_OUT
    assert capture.stdout == "STARTED\n"
    assert capture.stderr == "DIAGNOSTIC\n"
    assert capture.stdout_discarded == 0 and capture.stderr_discarded == 0
    assert process.poll() is not None
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


def test_native_child_inherits_only_the_isolated_environment(tmp_path):
    """Real children receive fixture roots without ambient credentials or user settings."""
    process = subprocess.Popen(
        ["/usr/bin/env"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    terminate = MagicMock()
    capture = supervise_process(process, time.monotonic() + 2, terminate=terminate)

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert capture.exit_code == 0
    inherited = dict(line.split("=", 1) for line in capture.stdout.splitlines())
    assert inherited == dict(os.environ)
    assert inherited["HOME"] == str(tmp_path.parent / "home")
    assert "OPENAI_API_KEY" not in inherited
    assert "SSH_AUTH_SOCK" not in inherited


def test_supervisor_terminates_same_group_descendant_after_parent_exit(tmp_path):
    """An exited leader retains its group identity until a pipe-holding child is terminated."""
    marker = tmp_path / "descendant-survived"
    source = (
        "import os,time; pid=os.fork(); "
        "if_child = pid == 0; "
        "os.write(1,b'descendant\\n') if if_child else None; "
        f"time.sleep(0.7) if if_child else None; "
        f"open({str(marker)!r},'w').write('survived') if if_child else None; "
        "os._exit(0)"
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    capture = supervise_process(process, time.monotonic() + 0.15)

    assert capture.status is ProcessCaptureStatus.TIMED_OUT
    assert capture.stdout == "descendant\n"
    assert capture.stdout_discarded == 0
    assert process.poll() is not None
    time.sleep(0.8)
    assert not marker.exists()


def test_supervisor_kills_same_group_descendant_after_output_pipes_close(tmp_path):
    """A live descendant is terminated even after its leader and output pipes finish."""
    ready = tmp_path / "descendant-ready"
    marker = tmp_path / "descendant-survived"
    source = f"""import os,time
pid=os.fork()
if pid == 0:
 os.close(1)
 os.close(2)
 open({str(ready)!r},'w').close()
 time.sleep(0.7)
 open({str(marker)!r},'w').write('survived')
else:
 while not os.path.exists({str(ready)!r}): time.sleep(0.005)
os._exit(0)
"""
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    capture = supervise_process(process, time.monotonic() + 2)

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert capture.exit_code == 0
    assert process.poll() == 0
    time.sleep(0.8)
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
    time.sleep(0.05)
    assert process.returncode is None

    terminate = MagicMock()
    capture = supervise_process(process, time.monotonic() + 2, terminate=terminate)

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert capture.exit_code == 0
    assert capture.stdout == "kqueue-ok"
    terminate.assert_called_once_with(process)


def test_stale_group_identifier_reuse_cannot_signal_an_unrelated_sentinel(monkeypatch):
    """A simulated PGID reuse after leader reaping leaves the unrelated sentinel alive."""
    from loop.utils import process as process_utils

    sentinel = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
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
    for name, value in (("P_PID", 1), ("WEXITED", 1), ("WNOHANG", 2), ("WNOWAIT", 4)):
        monkeypatch.setattr(process_utils.os, name, value, raising=False)
    try:
        capture = supervise_process(
            child,
            time.monotonic() + 2,
            terminate=terminate,
        )

        assert capture.status is ProcessCaptureStatus.COMPLETED
        assert child.wait.call_count == 1
        assert sentinel.poll() is None
        os.killpg(sentinel.pid, 0)
    finally:
        sentinel.kill()
        sentinel.wait(timeout=2)
