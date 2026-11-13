"""Exercise command continuation dispatch across the execution and tool boundaries."""

import asyncio
import json
from threading import Event
from unittest.mock import Mock

import pytest

from loop.execution import CommandExecutionService
from loop.execution.sandbox import CommandProcessResult, SandboxOutcome
from loop.permissions import Action, Operation, PermissionManager, ProcessBoundary, ProcessTarget
from loop.session import SessionManager
from loop.tooling import ToolContext, ToolRegistry
from loop.tools.system import read_command_job

pytestmark = pytest.mark.integration


def test_cancelled_async_read_preserves_the_background_command(tmp_path, monkeypatch):
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
        handle = service.run_command(context, ":", background=True)["handle"]
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
