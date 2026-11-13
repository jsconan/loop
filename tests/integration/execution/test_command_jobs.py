"""Exercise command continuation dispatch across the execution and tool boundaries."""

import asyncio
import json
from contextvars import ContextVar
from dataclasses import replace
from threading import Event, Thread
from unittest.mock import AsyncMock, Mock

import pytest

from loop.execution import CommandExecutionService
from loop.execution.jobs import CommandJob
from loop.execution.sandbox import CommandProcessResult, SandboxCapability, SandboxOutcome
from loop.instructions import InstructionsManager, RuntimeEnvironment
from loop.models import ToolCall
from loop.permissions import (
    Action,
    Operation,
    PermissionManager,
    PolicyScope,
    ProcessBoundary,
    ProcessTarget,
)
from loop.session import SessionManager
from loop.tooling import ToolContext, ToolRegistry
from loop.tools.system import read_command_job, run_command

pytestmark = pytest.mark.integration


def test_fast_command_returns_final_result_without_polling(tmp_path):
    """A command that finishes during the initial wait returns its captured result."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="finished"
    )
    sessions = SessionManager()
    permissions = PermissionManager(tmp_path)
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
    context.settings.command_hard_timeout = 42
    result = []
    thread = Thread(target=lambda: result.append(service.run_command(context, ":")))
    try:
        thread.start()
        assert approval_entered.wait(5)
        context.settings.command_hard_timeout = 1
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
    permissions.bind_temporary_directory(sessions.temporary_directory)
    if mode == "mediated":
        permissions.set_yolo(True)
    interaction = Mock(supports_fresh_host_approval=True)
    interaction.prompt_async = AsyncMock(return_value="approve")
    host = Mock()
    if mode == "host_offer":
        permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, sessions.temporary_directory)
    )
    invalidate = Mock(wraps=instructions.invalidate)
    monkeypatch.setattr(instructions, "invalidate", invalidate)
    registry = ToolRegistry(
        [run_command, read_command_job],
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

    async def finish(handle, cursor):
        """Read all remaining pages through a new event loop."""
        events = []
        for _ in range(10):
            result = json.loads(
                await registry.call_async(
                    "read_command_job",
                    json.dumps(
                        {
                            "handle": handle,
                            "cursor": cursor,
                            "wait_seconds": 5,
                            "include_output": True,
                        }
                    ),
                    instructions_manager=instructions,
                )
            )
            assert result["ok"], result
            page = result["result"]
            events.extend(page["events"])
            cursor = page["next_cursor"]
            if page["result"] is not None:
                return page["result"], events
        pytest.fail("The continued command did not settle.")

    try:
        first, original_loop = asyncio.run(start())
        assert original_loop.is_closed()
        assert first["ok"], first
        assert first["result"]["status"] == "running"
        invalidate.reset_mock()
        release.set()
        final, events = asyncio.run(
            finish(first["result"]["handle"], first["result"]["next_cursor"])
        )
        if mode == "host_offer":
            assert final["code"] == "sandbox.unavailable"
            assert "host_offer" in final["metadata"]
            assert events == []
            invalidate.assert_not_called()
            interaction.prompt_async.assert_not_awaited()
            host.run_host_command.assert_not_called()
        else:
            assert "exit_code" in final, final
            assert final["exit_code"] == 0
            assert final["stdout"]["content"] == "continued output"
            assert "".join(event["content"] for event in events) == "continued output"
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
        page = service.read_command_job(context, first["handle"], wait_seconds=0)
        assert page["handle"] == first["handle"]
        assert page["status"] == "running"
        assert backend.run.call_count == 1
        assert captured == ["original-context"]

        cancellation = service.cancel_command_job(context, first["handle"])
        assert cancellation["cancel_requested"]
        final = service.read_command_job(
            context, first["handle"], cursor=first["next_cursor"], wait_seconds=5
        )
        assert final["status"] == "complete"
        assert final["result"]["code"] == "sandbox.cancelled"
        assert final["elapsed_seconds"] >= first["elapsed_seconds"]
        assert backend.run.call_count == 1
    finally:
        correlation.reset(token)
        service.close()
        permissions.close()


@pytest.mark.parametrize("cancel", [False, True])
def test_approved_host_retry_remains_owned_after_handoff(tmp_path, monkeypatch, cancel):
    """An approved host retry can be read or cancelled after the original call returns."""
    host_started = Event()
    release_host = Event()
    interaction = Mock(supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="no sandbox")
    host = Mock()

    def run_host(_request, *, cancellation_event):
        """Wait for the test to settle or cancel the approved host attempt."""
        host_started.set()
        if not release_host.wait(timeout=5):
            raise TimeoutError("Synthetic host command was not released.")
        if cancellation_event.is_set():
            return CommandProcessResult(SandboxOutcome.CANCELLED, possible_effects=True)
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0, stdout="host output")

    def wait_for_host(_job, _seconds, _cancellation_event):
        """Make the initial wait expire only after the authorized host process starts."""
        if not host_started.wait(timeout=5):
            raise TimeoutError("The approved host attempt did not start.")
        return False

    host.run_host_command.side_effect = run_host
    monkeypatch.setattr(CommandJob, "wait", wait_for_host)
    sessions = SessionManager()
    permissions = PermissionManager(tmp_path)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
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
        first = service.run_command(context, ":", wait_seconds=1)
        assert first["status"] == "running"
        assert host_started.is_set()
        assert any(
            "without the OS sandbox" in call.args[0] for call in interaction.prompt.call_args_list
        )
        if cancel:
            assert service.cancel_command_job(context, first["handle"])["cancel_requested"]
        release_host.set()
        final = service.read_command_job(
            context, first["handle"], cursor=first["next_cursor"], wait_seconds=5
        )
        assert final["status"] == "complete"
        if cancel:
            assert final["result"]["code"] == "host_command.cancelled"
            assert final["cancel_requested"]
        else:
            assert final["result"]["boundary"] == "host"
            assert final["result"]["stdout"]["content"] == "host output"
        backend.run.assert_called_once()
        host.run_host_command.assert_called_once()
    finally:
        release_host.set()
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
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
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
        final = service.read_command_job(
            context, first["handle"], cursor=first["next_cursor"], wait_seconds=5
        )
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


def test_origin_cancellation_during_host_approval_prevents_launch(tmp_path, monkeypatch):
    """An interrupted origin cannot start a host command after a blocked approval returns."""
    entered_prompt = Event()
    release_prompt = Event()
    handoff_attempted = Event()
    stopped = Event()
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="no sandbox")
    host = Mock()
    interaction = Mock(supports_fresh_host_approval=True)

    def prompt(message, **_kwargs):
        """Block only the separate host decision until cancellation is requested."""
        if "without the OS sandbox" in message:
            entered_prompt.set()
            if not release_prompt.wait(timeout=5):
                raise TimeoutError("Synthetic host approval was not released.")
        return "approve"

    interaction.prompt.side_effect = prompt
    original_handoff = CommandJob.handoff

    def wait_for_prompt(_job, _seconds, _cancellation_event):
        """Hold the initial wait until the separate approval is active."""
        if not entered_prompt.wait(timeout=5):
            raise TimeoutError("Host approval was not entered.")

    def observe_handoff(job, cancellation_event=None):
        """Expose the caller's attempt to transfer ownership while approval is active."""
        handoff_attempted.set()
        return original_handoff(job, cancellation_event)

    monkeypatch.setattr(CommandJob, "wait", wait_for_prompt)
    monkeypatch.setattr(CommandJob, "handoff", observe_handoff)
    sessions = SessionManager()
    permissions = PermissionManager(tmp_path)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
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
        cancellation_event=stopped,
    )
    outcomes = []
    caller = Thread(
        target=lambda: outcomes.append(service.run_command(context, ":", wait_seconds=1))
    )
    caller.start()
    try:
        assert entered_prompt.wait(timeout=5), (outcomes, service.inspect_command_jobs(context))
        assert handoff_attempted.wait(timeout=5)
        stopped.set()
        release_prompt.set()
        caller.join(timeout=5)
        assert not caller.is_alive()
        assert len(outcomes) == 1
        owned = service.inspect_command_jobs(context)["jobs"]
        assert len(owned) == 1
        final = service.read_command_job(
            replace(context, cancellation_event=None), owned[0]["handle"], wait_seconds=5
        )
        assert final["status"] == "complete"
        assert final["cancel_requested"]
        host.run_host_command.assert_not_called()
        assert final["result"]["code"] == "sandbox.cancelled"
        backend.run.assert_called_once()
    finally:
        stopped.set()
        release_prompt.set()
        caller.join(timeout=5)
        service.close()
        permissions.close()


def test_cancelled_async_read_preserves_the_command(tmp_path, monkeypatch):
    """Read live output, interrupt a later read, and retrieve the command's final result."""
    release = Event()
    emit = Event()
    reading = Event()
    backend = Mock()

    def run_command(*_args, **kwargs):
        """Publish live output and remain active until the test releases the command."""
        if not emit.wait(timeout=5):
            raise TimeoutError("Synthetic command was not allowed to emit output.")
        kwargs["output_observer"]("stdout", "ready")
        if not release.wait(timeout=5):
            raise TimeoutError("Synthetic command was not released.")
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0, stdout="done")

    backend.run.side_effect = run_command
    sessions = SessionManager()
    permissions = PermissionManager(tmp_path)
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
        [read_command_job],
        permission_manager=permissions,
        execution_service=service,
        session_manager=sessions,
    )
    original_read = service.read_command_job

    def observe_read(*args, **kwargs):
        """Expose admission to the actual execution service read."""
        reading.set()
        return original_read(*args, **kwargs)

    monkeypatch.setattr(service, "read_command_job", observe_read)

    async def scenario():
        """Read live output, cancel a subsequent wait, and finish the same command."""
        handle = service.run_command(context, ":", wait_seconds=0)["handle"]
        arguments = json.dumps({"handle": handle, "cursor": 0, "wait_seconds": 30})
        task = asyncio.create_task(registry.call_async("read_command_job", arguments))
        assert await asyncio.to_thread(reading.wait, 5), "Read was not admitted."
        emit.set()
        page = json.loads(await asyncio.wait_for(task, timeout=2))["result"]
        assert page["events"][0]["content"] == "ready"
        assert page["status"] == "running"
        cursor = page["next_cursor"]
        assert cursor == 1
        reading.clear()
        arguments = json.dumps({"handle": handle, "cursor": cursor, "wait_seconds": 30})
        task = asyncio.create_task(registry.call_async("read_command_job", arguments))
        assert await asyncio.to_thread(reading.wait, 5), "Second read was not admitted."
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        immediate = json.loads(
            await registry.call_async(
                "read_command_job",
                json.dumps({"handle": handle, "cursor": cursor, "wait_seconds": 0}),
            )
        )["result"]
        assert immediate["status"] == "running"
        assert immediate["events"] == []
        assert immediate["next_cursor"] == cursor
        assert not immediate["cancel_requested"]
        release.set()
        final = json.loads(await registry.call_async("read_command_job", arguments))["result"]
        assert final["result"]["exit_code"] == 0

    try:
        asyncio.run(scenario())
    finally:
        emit.set()
        release.set()
        registry.close()
        permissions.close()
