"""Exercise command continuation dispatch across the execution and tool boundaries."""

import asyncio
import json
from contextvars import ContextVar
from threading import Event, Thread
from unittest.mock import AsyncMock, Mock

import pytest

from loop.execution import CommandExecutionService
from loop.execution.sandbox import CommandProcessResult, SandboxCapability, SandboxOutcome
from loop.instructions import InstructionsManager, RuntimeEnvironment
from loop.models import ToolCall
from loop.permissions import (
    Action,
    Decision,
    Operation,
    PermissionManager,
    ProcessBoundary,
    ProcessTarget,
)
from loop.session import SessionManager
from loop.tooling import ToolContext, ToolRegistry
from loop.tools.system import poll_command_job, read_command_output, run_command

pytestmark = pytest.mark.integration


def test_fast_command_returns_final_result_without_polling(tmp_path):
    """A command that finishes during the initial wait returns its captured result."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="finished"
    )
    sessions = SessionManager()
    permissions = PermissionManager(tmp_path)
    permissions.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    service = CommandExecutionService(backend, Mock())
    registry = ToolRegistry(execution_service=service, session_manager=sessions)
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"), cwd=str(tmp_path), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(operation,),
        permission_manager=permissions,
        session_manager=sessions,
    )
    try:
        result = service.run_command(context, ":", wait_seconds=5)
        assert result["exit_code"] == 0
        assert result["stdout"]["content"] == "finished"
        assert backend.run.call_count == 1
        assert len(service.inspect_command_jobs(context)["jobs"]) == 1
        registry.bind_session_manager(SessionManager())
    finally:
        registry.close()
        permissions.close()


def test_command_snapshots_hard_limit_before_approval(tmp_path, monkeypatch):
    """An in-flight settings change cannot alter an admitted job's execution limit."""
    approval_entered = Event()
    release_approval = Event()
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    sessions = SessionManager()
    permissions = PermissionManager(tmp_path)
    permissions.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    authorize = permissions.authorize_sandboxed_command

    def wait_for_approval(*args, **kwargs):
        """Delay admission until the invoking test changes the live settings."""
        approval_entered.set()
        if not release_approval.wait(5):
            raise TimeoutError("Synthetic approval was not released.")
        return authorize(*args, **kwargs)

    monkeypatch.setattr(permissions, "authorize_sandboxed_command", wait_for_approval)
    service = CommandExecutionService(backend, Mock())
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"), cwd=str(tmp_path), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(operation,),
        permission_manager=permissions,
        session_manager=sessions,
    )
    context.settings.command_job_runtime_limit_seconds = 42
    result = []
    thread = Thread(target=lambda: result.append(service.run_command(context, ":")))
    try:
        thread.start()
        assert approval_entered.wait(5)
        context.settings.command_job_runtime_limit_seconds = 1
        release_approval.set()
        thread.join(5)
        assert not thread.is_alive()
        assert result[0]["exit_code"] == 0
        assert backend.run.call_args.args[0].runtime_timeout == 42
        assert service.inspect_command_jobs(context)["jobs"][0]["runtime_limit_seconds"] == 42
    finally:
        release_approval.set()
        thread.join(5)
        service.close()
        permissions.close()


@pytest.mark.parametrize("mode", ["completed", "host_offer", "mediated"])
def test_async_command_survives_owner_loop_closure(tmp_path, monkeypatch, mode):
    """A continued job settles through a new loop after its original owner loop closes."""
    release = Event()
    lease = Mock()
    revisions = []
    backend = Mock(capabilities=frozenset({SandboxCapability.MEDIATED_EGRESS}))

    def run_after_handoff(*_args, **kwargs):
        """Publish and finish only after the first event loop has closed."""
        if not release.wait(timeout=5):
            raise TimeoutError("Synthetic command was not released.")
        if mode == "host_offer":
            return CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="no sandbox")
        if mode == "mediated":
            authority = kwargs["policy_authority"]
            revision = kwargs["approved_revision"]
            authority.bind_egress(revision, lease)
            try:
                revisions.append(authority.try_egress_policy_revision())
            finally:
                authority.unbind_egress(lease)
        kwargs["output_observer"]("stdout", "continued output")
        return CommandProcessResult(
            SandboxOutcome.COMPLETED, exit_code=0, stdout="continued output"
        )

    backend.run.side_effect = run_after_handoff
    sessions = SessionManager()
    permissions = PermissionManager(tmp_path)
    permissions.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    if mode == "mediated":
        permissions.set_yolo(True)
    interaction = Mock(supports_fresh_host_approval=True)
    interaction.prompt_async = AsyncMock(return_value="approve")
    host = Mock()
    if mode == "host_offer":
        permissions.set_direct_host_execution(True)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, sessions.temporary_directory)
    )
    invalidate = Mock(wraps=instructions.invalidate)
    monkeypatch.setattr(instructions, "invalidate", invalidate)
    registry = ToolRegistry(
        [run_command, poll_command_job, read_command_output],
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, host),
        session_manager=sessions,
    )

    async def start():
        """Start one command and retain its original event loop for verification."""
        loop = asyncio.get_running_loop()
        arguments = {"command": ":", "cwd": str(tmp_path), "wait_seconds": 0}
        if mode == "mediated":
            arguments["egress_origins"] = ["https://example.com"]
        result = await registry.call_async(
            "run_command",
            json.dumps(arguments),
            interaction=interaction,
            instructions_manager=instructions,
        )
        return json.loads(result), loop

    async def finish(handle):
        """Poll for completion and take command output from the same result."""
        cursors = {"stdout_cursor": None, "stderr_cursor": None}
        for _ in range(10):
            result = json.loads(
                await registry.call_async(
                    "poll_command_job",
                    json.dumps({"handle": handle, "wait_seconds": 5, **cursors}),
                    instructions_manager=instructions,
                )
            )
            assert result["ok"], result
            status = result["result"]
            cursors = {
                f"{stream}_cursor": status[stream].get("next_cursor")
                for stream in ("stdout", "stderr")
            }
            if status["status"] == "complete":
                return status, status["stdout"]
        pytest.fail("The continued command did not settle.")

    try:
        first, original_loop = asyncio.run(start())
        assert original_loop.is_closed()
        assert first["ok"], first
        assert first["result"]["status"] == "running"
        invalidate.reset_mock()
        release.set()
        final, output = asyncio.run(finish(first["result"]["handle"]))
        if mode == "host_offer":
            assert final["result"]["code"] == "sandbox.unavailable"
            assert "host_offer" in final["result"]["metadata"]
            assert output["content"] == ""
            invalidate.assert_not_called()
            interaction.prompt_async.assert_not_awaited()
            host.run_host_command.assert_not_called()
        else:
            assert final["exit_code"] == 0
            assert output["content"] == "continued output"
            invalidate.assert_called_once_with(None)
        if mode == "mediated":
            assert revisions == [permissions.egress_policy_revision()]
        backend.run.assert_called_once()
    finally:
        release.set()
        registry.close()
        permissions.close()


def test_command_yields_once_and_can_be_cancelled_without_replay(tmp_path):
    """A yielded invocation retains its context and controls one cancellable native attempt."""
    started = Event()
    captured = []
    correlation = ContextVar("command_test_correlation", default=None)
    backend = Mock()

    def run_command(*_args, **kwargs):
        """Keep the admitted command alive until its owner requests cancellation."""
        captured.append(correlation.get())
        started.set()
        if not kwargs["cancellation_event"].wait(timeout=5):
            raise TimeoutError("Synthetic command was not cancelled.")
        return CommandProcessResult(SandboxOutcome.CANCELLED)

    backend.run.side_effect = run_command
    sessions = SessionManager()
    permissions = PermissionManager(tmp_path)
    permissions.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    service = CommandExecutionService(backend, Mock())
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"), cwd=str(tmp_path), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(operation,),
        permission_manager=permissions,
        session_manager=sessions,
        call_id="owned-command",
    )
    sessions.add_message(ToolCall(call_id="owned-command", name="run_command", arguments="{}"))
    token = correlation.set("original-context")
    try:
        first = service.run_command(context, ":", wait_seconds=0)
        assert started.wait(timeout=5)
        assert first["status"] == "running"
        assert first["elapsed_seconds"] >= 0
        status = service.poll_command_job(context, first["handle"], wait_seconds=0)
        assert status["handle"] == first["handle"]
        assert status["status"] == "running"
        assert status["stdout"]["content"] == ""
        assert backend.run.call_count == 1
        assert captured == ["original-context"]

        cancellation = service.cancel_command_job(context, first["handle"])
        assert cancellation["cancel_requested"]
        final = service.poll_command_job(context, first["handle"], wait_seconds=5)
        assert final["status"] == "complete"
        assert final["result"]["code"] == "sandbox.cancelled"
        assert final["elapsed_seconds"] >= first["elapsed_seconds"]
        assert backend.run.call_count == 1
    finally:
        correlation.reset(token)
        service.close()
        permissions.close()


def test_host_approval_is_not_requested_after_handoff(tmp_path):
    """A no-effects sandbox failure after handoff cannot open a stale host prompt."""
    release_sandbox = Event()
    interaction = Mock(supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    backend = Mock()

    def fail_after_handoff(*_args, **_kwargs):
        """Report an eligible host offer only after the original caller has returned."""
        if not release_sandbox.wait(timeout=5):
            raise TimeoutError("Synthetic sandbox attempt was not released.")
        return CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="no sandbox")

    backend.run.side_effect = fail_after_handoff
    host = Mock()
    sessions = SessionManager()
    permissions = PermissionManager(tmp_path)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    permissions.set_direct_host_execution(True)
    service = CommandExecutionService(backend, host)
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"), cwd=str(tmp_path), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=permissions,
        session_manager=sessions,
    )
    try:
        first = service.run_command(context, ":", wait_seconds=0)
        assert first["status"] == "running"
        release_sandbox.set()
        final = service.poll_command_job(context, first["handle"], wait_seconds=5)
        assert final["status"] == "complete"
        assert final["result"]["code"] == "sandbox.unavailable"
        assert "host_offer" in final["result"]["metadata"]
        assert all(
            "without the OS sandbox" not in call.args[0]
            for call in interaction.prompt.call_args_list
        )
        host.run_host_command.assert_not_called()
    finally:
        release_sandbox.set()
        service.close()
        permissions.close()


def test_cancelled_async_poll_preserves_the_command(tmp_path, monkeypatch):
    """Output does not end a poll, and cancelling a poll leaves the command running."""
    release = Event()
    emit = Event()
    output_published = Event()
    polling = Event()
    backend = Mock()

    def run_command(*_args, **kwargs):
        """Publish a partial stream, then remain active until the test releases the job."""
        if not emit.wait(timeout=5):
            raise TimeoutError("Synthetic command was not allowed to emit output.")
        kwargs["output_observer"]("stdout", "ready")
        output_published.set()
        if not release.wait(timeout=5):
            raise TimeoutError("Synthetic command was not released.")
        kwargs["output_observer"]("stdout", "done")
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0, stdout="readydone")

    backend.run.side_effect = run_command
    sessions = SessionManager()
    permissions = PermissionManager(tmp_path)
    permissions.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    service = CommandExecutionService(backend, Mock())
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"), cwd=str(tmp_path), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(operation,),
        permission_manager=permissions,
        session_manager=sessions,
    )
    registry = ToolRegistry(
        [poll_command_job, read_command_output],
        permission_manager=permissions,
        execution_service=service,
        session_manager=sessions,
    )
    original_poll = service.poll_command_job

    def observe_poll(*args, **kwargs):
        """Expose that the async poll entered its completion wait."""
        polling.set()
        return original_poll(*args, **kwargs)

    monkeypatch.setattr(service, "poll_command_job", observe_poll)

    async def scenario():
        """Poll returns current output while the separate completion wait remains pending."""
        handle = service.run_command(context, ":", wait_seconds=0)["handle"]
        emit.set()
        assert await asyncio.to_thread(output_published.wait, 5), "Output was not published."
        immediate = json.loads(
            await registry.call_async(
                "poll_command_job",
                json.dumps({"handle": handle, "wait_seconds": 0}),
            )
        )["result"]
        assert immediate["stdout"]["content"] == "ready"
        assert immediate["status"] == "running"
        cursor = immediate["stdout"]["next_cursor"]

        arguments = json.dumps({"handle": handle, "wait_seconds": 30, "stdout_cursor": cursor})
        task = asyncio.create_task(registry.call_async("poll_command_job", arguments))
        assert await asyncio.to_thread(polling.wait, 5), "Poll was not admitted."
        assert not task.done(), "New output must not complete a poll."
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

        waiting = service.poll_command_job(context, handle, wait_seconds=0, stdout_cursor=cursor)
        assert waiting["status"] == "running"
        assert waiting["stdout"]["content"] == ""
        assert not waiting["cancel_requested"]
        release.set()
        final = original_poll(context, handle, wait_seconds=5, stdout_cursor=cursor)
        assert final["status"] == "complete"
        assert final["exit_code"] == 0
        assert final["stdout"]["content"] == "done"

    try:
        asyncio.run(scenario())
    finally:
        emit.set()
        release.set()
        registry.close()
        permissions.close()
