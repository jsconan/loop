"""Verify unsupported Linux native execution fails closed without a host fallback."""

import json
from unittest.mock import Mock

import pytest

from loop import PermissionManager, ToolRegistry
from loop.execution import CommandExecutionService
from loop.execution.sandbox import SandboxOutcome, select_sandbox_backend
from loop.tools.system import run_command

pytestmark = [pytest.mark.integration, pytest.mark.linux]


def test_linux_selection_is_unavailable_without_implicit_host_execution(tmp_path):
    """The current Linux backend reports its missing native boundary without launching."""
    backend = select_sandbox_backend()
    assert backend.capability_failure()
    assert backend.run(None).outcome is SandboxOutcome.UNAVAILABLE


@pytest.mark.e2e
def test_public_linux_command_cannot_write_without_separate_host_approval(tmp_path):
    """A public request cannot create data when Linux sandbox qualification is unavailable."""
    target = tmp_path / "must-not-exist"
    interaction = Mock()
    interaction.prompt.side_effect = lambda message, **kwargs: (
        "deny" if "without the OS sandbox" in message else "approve"
    )
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService.for_host(),
    )
    try:
        output = json.loads(
            registry.call(
                "run_command",
                json.dumps({"command": "printf forbidden > must-not-exist", "cwd": str(tmp_path)}),
                interaction=interaction,
            )
        )
        assert output["problem"]["code"] == "sandbox.unavailable"
        assert not target.exists()
    finally:
        registry.close()
