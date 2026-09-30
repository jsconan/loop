"""Exercise owned native processes in disposable workspaces."""

import logging
import os
import subprocess
import time

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
    boundary.process.wait.side_effect = [subprocess.TimeoutExpired("stub", 1), -9]
    caplog.set_level(logging.WARNING, logger="loop.utils.process")

    capture = supervise_process(boundary.process, 101, thread_factory=boundary.thread_factory)

    assert capture.status is ProcessCaptureStatus.TIMED_OUT
    boundary.kill.assert_called_once_with(123, 9)
    assert boundary.process.wait.call_count == 2
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
    assert boundary.process.wait.call_count == 2
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
    capture = supervise_process(process, time.monotonic() + 2)

    assert capture.status is ProcessCaptureStatus.COMPLETED
    assert capture.exit_code == 0
    inherited = dict(line.split("=", 1) for line in capture.stdout.splitlines())
    assert inherited == dict(os.environ)
    assert inherited["HOME"] == str(tmp_path.parent / "home")
    assert "OPENAI_API_KEY" not in inherited
    assert "SSH_AUTH_SOCK" not in inherited
