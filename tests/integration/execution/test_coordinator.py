"""Exercise owned native processes in disposable workspaces."""

import subprocess
import time

import pytest

from loop.execution.coordinator import (
    HostCommandRequest,
    LocalHostCommandExecutor,
)
from loop.execution.sandbox import SandboxOutcome, SandboxRequest

pytestmark = pytest.mark.integration


def request(tmp_path, **changes):
    """Build a bound sandbox command in the disposable workspace."""
    values = {
        "source": "printf ok",
        "cwd": tmp_path,
        "workspace": tmp_path,
        "read_roots": (tmp_path.parent,),
        "write_roots": (tmp_path,),
        "network": False,
        "environment": {"PATH": "/usr/bin:/bin"},
        "policy_version": "macos-v1",
        "deadline": time.monotonic() + 60,
        "workspace_id": "workspace-1",
    }
    values.update(changes)
    return SandboxRequest.create(**values)


def test_local_host_executor_captures_approved_shell_output(tmp_path):
    """The separate host supervisor returns bounded output and the shell exit code."""
    approved = HostCommandRequest.from_sandbox(request(tmp_path, source="printf host; exit 7"))

    result = LocalHostCommandExecutor().run_host_command(approved)

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 7
    assert result.stdout == "host"


def test_local_host_executor_times_out_and_cleans_up(tmp_path, process_boundary):
    """Executor and supervisor compose deadline handling with owned group cleanup."""
    boundary = process_boundary
    boundary.process.wait.side_effect = [subprocess.TimeoutExpired("stub", 1), -9]
    approved = HostCommandRequest.from_sandbox(request(tmp_path, deadline=101))

    result = LocalHostCommandExecutor().run_host_command(approved)

    assert result.outcome is SandboxOutcome.TIMED_OUT
    assert result.possible_effects
    boundary.kill.assert_called_once_with(123, 9)
    assert boundary.process.wait.call_count == 2


def test_local_host_executor_completes_within_its_budget(tmp_path):
    """A real host child returns its output within a generous scheduling budget."""
    approved = HostCommandRequest.from_sandbox(
        request(tmp_path, source="printf finished", deadline=time.monotonic() + 2)
    )

    result = LocalHostCommandExecutor().run_host_command(approved)

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.stdout == "finished"


def test_local_host_executor_timeout_closes_retained_pipe_without_thread_error(
    tmp_path, process_boundary, monkeypatch
):
    """Executor cleanup closes descendant-held pipes without scheduling real readers."""
    boundary = process_boundary

    def retained_thread(**kwargs):
        """Represent a descendant-held reader at the operating-system boundary."""
        thread = boundary.thread_factory(**kwargs)
        thread.is_alive.return_value = True
        return thread

    monkeypatch.setattr("loop.execution.coordinator.threading.Thread", retained_thread)
    approved = HostCommandRequest.from_sandbox(request(tmp_path, deadline=101))

    result = LocalHostCommandExecutor().run_host_command(approved)

    assert result.outcome is SandboxOutcome.TIMED_OUT
    boundary.kill.assert_called_once_with(123, 9)
    assert [call.args for call in boundary.close.call_args_list] == [(101,), (102,)]
    assert all(thread.join.call_count == 2 for thread in boundary.threads)
