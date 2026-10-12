"""Verify unavailable or invalid Linux authority fails closed without a host launch."""

import json
from unittest.mock import Mock

import pytest

from loop import PermissionManager, ToolRegistry
from loop.execution import CommandExecutionService
from loop.execution.sandbox import CommandProcessResult, SandboxOutcome, select_sandbox_backend
from loop.tools.system import run_command

pytestmark = [pytest.mark.integration, pytest.mark.linux]


@pytest.fixture
def unavailable_backend():
    """Select actual failed host capability; installed enforcement has its own native corpus."""
    backend = select_sandbox_backend()
    failure = backend.capability_failure() if hasattr(backend, "capability_failure") else None
    if failure is None:
        pytest.skip(
            "Requires unavailable native authority; installed execution has separate checks"
        )
    return backend, failure


def test_linux_selection_retains_failure_without_implicit_host_execution(unavailable_backend):
    """Absent or invalid installation retains its exact typed failure without launching a child."""
    backend, failure = unavailable_backend
    result = backend.run(None)
    assert result.outcome is (
        failure.outcome if isinstance(failure, CommandProcessResult) else SandboxOutcome.UNAVAILABLE
    )


@pytest.mark.e2e
def test_public_linux_command_cannot_write_without_separate_host_approval(
    tmp_path, unavailable_backend
):
    """A public request cannot create data when actual native authority is absent or invalid."""
    target = tmp_path / "must-not-exist"
    interaction = Mock()
    interaction.prompt.side_effect = lambda message, **kwargs: (
        "deny" if "without the OS sandbox" in message else "approve"
    )
    backend, failure = unavailable_backend
    host = Mock()
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(backend, host),
    )
    try:
        output = json.loads(
            registry.call(
                "run_command",
                json.dumps({"command": "printf forbidden > must-not-exist", "cwd": str(tmp_path)}),
                interaction=interaction,
            )
        )
        outcome = (
            failure.outcome
            if isinstance(failure, CommandProcessResult)
            else SandboxOutcome.UNAVAILABLE
        )
        assert output["problem"]["code"] == "sandbox." + outcome.value
        if outcome in (SandboxOutcome.INVALID, SandboxOutcome.STALE):
            assert "host_offer" not in output["problem"].get("metadata", {})
            assert not any(
                "without the OS sandbox" in str(call) for call in interaction.prompt.call_args_list
            )
        assert not target.exists()
        host.run_host_command.assert_not_called()
    finally:
        registry.close()
