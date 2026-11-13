"""Exercise command jobs sharing session and execution authority across registries."""

import asyncio
import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop import Loop
from loop.command_authorities import command_authorities
from loop.execution import CommandExecutionService
from loop.execution.sandbox import CommandProcessResult, SandboxOutcome
from loop.instructions import InstructionsManager, RuntimeEnvironment
from loop.permissions import PermissionManager
from loop.session import SessionManager
from loop.tooling import ToolRegistry
from loop.tools.system import cancel_command_job, read_command_job, run_command

pytestmark = pytest.mark.integration


def make_backend(run):
    """Build a native-boundary double with one controlled command execution."""
    backend = Mock()
    backend.capabilities = frozenset()
    backend.system_read_roots = ()
    backend.capability_failure.return_value = None
    backend.run.side_effect = run
    return backend


def model_backend():
    """Provide the model metadata needed to construct an application loop."""
    return SimpleNamespace(default_model="fixture", get_context_window=lambda _: None)


@pytest.mark.parametrize("shared", ["service", "permissions", "instructions"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_shared_authority_prevents_recomposition_until_job_settles(
    tmp_path, monkeypatch, shared, asynchronous
):
    """Shared objects stay bound while a sync or async detached job owns them."""
    entered, release, settled = Event(), Event(), Event()
    original_release = command_authorities.release

    def release_authorities(lease):
        """Signal only after the real authority reservation has been released."""
        original_release(lease)
        settled.set()

    monkeypatch.setattr(command_authorities, "release", release_authorities)

    def run(_request, **_kwargs):
        """Keep the admitted worker active until the test checks a second loop."""
        entered.set()
        if not release.wait(10):
            raise TimeoutError("Fixture command was not released")
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)

    first_session, second_session = SessionManager(), SessionManager()
    first_permissions = PermissionManager(tmp_path, yolo=True)
    first_permissions.bind_temporary_directory(first_session.temporary_directory)
    first_instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, first_session.temporary_directory)
    )
    first_service = CommandExecutionService(make_backend(run), Mock())
    first = ToolRegistry(
        [run_command, read_command_job, cancel_command_job],
        permission_manager=first_permissions,
        execution_service=first_service,
        session_manager=first_session,
    )
    second_permissions = (
        first_permissions if shared == "permissions" else PermissionManager(tmp_path, yolo=True)
    )
    second_instructions = first_instructions if shared == "instructions" else InstructionsManager()
    second_service = (
        first_service if shared == "service" else CommandExecutionService(make_backend(run), Mock())
    )
    second = ToolRegistry(
        [run_command],
        permission_manager=second_permissions,
        execution_service=second_service,
        session_manager=second_session,
    )
    args = json.dumps({"command": ":", "cwd": "$WORKSPACE_ROOT", "wait_seconds": 0})
    try:
        if asynchronous:
            result = json.loads(
                asyncio.run(
                    first.call_async("run_command", args, instructions_manager=first_instructions)
                )
            )
        else:
            result = json.loads(
                first.call("run_command", args, instructions_manager=first_instructions)
            )
        assert result["ok"], result
        assert entered.wait(5)
        cancelled = json.loads(
            first.call(
                "cancel_command_job",
                json.dumps({"handle": result["result"]["handle"]}),
                instructions_manager=first_instructions,
            )
        )
        assert cancelled["ok"] and cancelled["result"]["cancel_requested"]
        assert not settled.is_set()
        original_scratch = first_permissions.temporary_directory
        original_roots = dict(first_instructions.virtual_paths.command_roots())

        with pytest.raises(RuntimeError, match="commands are active"):
            Loop.create_default(
                model_backend(),
                working_directory=tmp_path,
                permission_manager=second_permissions,
                instructions_manager=second_instructions,
                session_manager=second_session,
                tool_registry=second,
            )
        assert first_permissions.temporary_directory == original_scratch
        assert dict(first_instructions.virtual_paths.command_roots()) == original_roots
        with pytest.raises(OSError, match="Command execution still owns temporary storage"):
            first_permissions.bind_temporary_directory(second_session.temporary_directory)

        release.set()
        assert settled.wait(5)
        created = Loop.create_default(
            model_backend(),
            working_directory=tmp_path,
            permission_manager=second_permissions,
            instructions_manager=second_instructions,
            session_manager=second_session,
            tool_registry=second,
        )
        assert created.tool_registry is second
    finally:
        release.set()
        first.close()
        second.close()
        first_permissions.close()
        if second_permissions is not first_permissions:
            second_permissions.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_command_keeps_planned_roots_when_working_directory_changes(tmp_path, asynchronous):
    """The native request uses the root authorized before the execution callback ran."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = SessionManager()
    permissions = PermissionManager(workspace, yolo=True)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(workspace, sessions.temporary_directory)
    )
    seen = []

    def execute(request, **_kwargs):
        """Record the exact bound native workspace and scratch roots."""
        seen.append(request)
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)

    service = CommandExecutionService(make_backend(execute), Mock())
    registry = ToolRegistry(
        [run_command],
        permission_manager=permissions,
        execution_service=service,
        session_manager=sessions,
    )
    args = json.dumps({"command": ":", "cwd": "$WORKSPACE_ROOT", "wait_seconds": 5})

    def change_directory():
        """Move the live instruction root after planning but before command dispatch."""
        instructions.set_working_directory(tmp_path)

    async def change_directory_async():
        """Move the same live root at the asynchronous execution boundary."""
        change_directory()

    try:
        if asynchronous:
            result = json.loads(
                asyncio.run(
                    registry.call_async(
                        "run_command",
                        args,
                        instructions_manager=instructions,
                        execution_started=change_directory_async,
                    )
                )
            )
        else:
            result = json.loads(
                registry.call(
                    "run_command",
                    args,
                    instructions_manager=instructions,
                    execution_started=change_directory,
                )
            )
        assert result["ok"], result
        assert len(seen) == 1
        assert seen[0].workspace == workspace
        assert seen[0].cwd == workspace
        assert dict(seen[0].environment)["TMPDIR"] == str(sessions.temporary_directory)
    finally:
        registry.close()
        permissions.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_stale_permission_scratch_rejects_command_before_native_launch(tmp_path, asynchronous):
    """A registry bound to an old session cannot use another session's permission scratch."""
    first, second = SessionManager(), SessionManager()
    permissions = PermissionManager(tmp_path, yolo=True)
    permissions.bind_temporary_directory(first.temporary_directory)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, first.temporary_directory)
    )
    backend = make_backend(lambda _request, **_kwargs: None)
    registry = ToolRegistry(
        [run_command],
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, Mock()),
        session_manager=first,
    )
    permissions.bind_temporary_directory(second.temporary_directory)
    args = json.dumps({"command": ":", "cwd": "$WORKSPACE_ROOT", "wait_seconds": 0})
    try:
        if asynchronous:
            result = json.loads(
                asyncio.run(
                    registry.call_async("run_command", args, instructions_manager=instructions)
                )
            )
        else:
            result = json.loads(
                registry.call("run_command", args, instructions_manager=instructions)
            )
        assert not result["ok"]
        assert result["problem"]["code"] == "command_job.unavailable"
        backend.run.assert_not_called()
    finally:
        registry.close()
        permissions.close()
