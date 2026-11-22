"""Exercise saved task approval, outputs and scope through the native command path."""

import json
import os
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from loop.application.verification import VerificationCommands
from loop.commands import CommandManager
from loop.execution import CommandExecutionService
from loop.execution.sandbox import UnavailableSandboxBackend, select_sandbox_backend
from loop.instructions import InstructionsManager, RuntimeEnvironment
from loop.interaction import Interaction
from loop.permissions import PermissionManager
from loop.session import SessionManager
from loop.tooling import ToolRegistry
from loop.tools import BUILTIN_TOOLS

pytestmark = pytest.mark.integration
_PARENT, _MODE, _EXPECTED = (
    ("/private/tmp", "LOOP_NATIVE_MACOS_TEST", "owned-volume")
    if sys.platform == "darwin"
    else ("/tmp", "LOOP_NATIVE_LINUX_TEST", "installed")
)
_NATIVE_ENABLED = os.environ.get(_MODE) == _EXPECTED


@pytest.mark.skipif(not _NATIVE_ENABLED, reason=f"Set {_MODE}={_EXPECTED} for native qualification")
def test_native_verification_finishes_owned_job_and_preserves_undeclared_outputs(native_directory):
    """An opted-in workspace still confines a saved task to its explicitly approved outputs."""
    root = native_directory(_PARENT)
    checkout = root / "checkout"
    checkout.mkdir()
    (checkout / "out").mkdir()
    (checkout / ".coverage").write_text("existing coverage")
    (checkout / ".env").write_text("hard canary")
    data = root / "owned data"
    data.mkdir(mode=0o700)
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "approve"
    permissions = PermissionManager(checkout, protected_host_roots=(data,))
    sessions = SessionManager()
    permissions.bind_temporary_directory(sessions.temporary_directory)
    permissions.set_workspace_command_writes(True)
    permissions.set_limit("deny_local_networks", True)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(checkout, permissions.temporary_directory)
    )
    backend = select_sandbox_backend()
    assert not isinstance(backend, UnavailableSandboxBackend), backend.capability_failure()
    host = Mock()
    host.run_host_command.side_effect = AssertionError("Unexpected host execution")
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, host),
        session_manager=sessions,
    )
    commands = CommandManager(interaction=interaction)
    commands.register_provider(VerificationCommands(data, registry, instructions))
    definition = {
        "command": (
            "remaining=200; until test -f release; do remaining=$((remaining - 1)); "
            'test "$remaining" -gt 0 || exit 33; sleep 0.05; done; '
            "printf worker1 > out/coverage.1 & a=$!; "
            "printf worker2 > out/coverage.2 & b=$!; "
            'wait "$a" && wait "$b" && printf \'%s\' "$REPORT" > summary '
            '&& printf cache > "$LOOP_OUTPUT_DIR/cache" || exit; '
            "if printf forbidden > .coverage || cat .env || printf forbidden > .env; "
            "then exit 42; fi"
        ),
        "environment": {"REPORT": "approved"},
        "write_roots": ["$WORKSPACE_ROOT/out"],
        "write_files": ["summary"],
    }
    try:
        commands.call("verify", f"save check {shlex.quote(json.dumps(definition))}")
        try:
            with ThreadPoolExecutor(max_workers=1) as worker:
                worker.submit(commands.call, "verify", "run check wait_seconds=0").result(
                    timeout=20
                )
            first = json.loads(interaction.tool_result.call_args.args[0])
            assert first["ok"] and first["result"]["status"] == "running", first
            with sessions.execution():
                poll = registry.command(
                    "poll_command_job",
                    (first["result"]["handle"], "wait_seconds=0"),
                    interaction=interaction,
                    instructions_manager=instructions,
                )
                assert json.loads(poll.output)["result"]["status"] == "running"
        finally:
            (checkout / "release").touch()
        interaction.prompt.assert_called_once()
        assert '"workspace_write": false' in interaction.info.call_args.args[0]
        final = registry.command(
            "poll_command_job",
            (first["result"]["handle"], "wait_seconds=10"),
            interaction=interaction,
            instructions_manager=instructions,
        )
        settled = json.loads(final.output)["result"]
        assert settled["status"] == "complete", settled
        assert settled["exit_code"] == 0, settled
        assert (checkout / "summary").read_text() == "approved"
        assert (checkout / "out/coverage.1").read_text() == "worker1"
        assert (checkout / "out/coverage.2").read_text() == "worker2"

        definition["environment"]["REPORT"] = "replacement"
        commands.call("verify", f"save check {shlex.quote(json.dumps(definition))}")
        commands.call("verify", "run check wait_seconds=10")
        second = json.loads(interaction.tool_result.call_args.args[0])
        assert second["ok"], second
        assert second["result"]["exit_code"] == 0, second
        assert interaction.prompt.call_count == 2
        assert (checkout / "summary").read_text() == "replacement"
        assert (checkout / ".coverage").read_text() == "existing coverage"
        assert (checkout / ".env").read_text() == "hard canary"
        artifact = registry.command(
            "read_text_file",
            (f"{second['result']['scratch']['output_path']}/cache",),
            interaction=interaction,
            instructions_manager=instructions,
        )
        content = json.loads(artifact.output)
        assert content["ok"], content
        assert content["result"]["content"] == "cache"
    finally:
        commands.close()
        registry.close()
        permissions.close()
