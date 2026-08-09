"""Tests for tool invocation context."""

from unittest.mock import Mock

from loop.context import ToolContext
from loop.interaction import Interaction
from loop.permissions import (
    Capability,
    PermissionConfiguration,
    PermissionManager,
    PermissionMode,
    PermissionRequest,
)
from loop.skills import InstructionsManager


def test_confirm_delegates_to_the_interaction():
    """Confirmation forwards the question and default through the interaction boundary."""
    interaction = Mock(spec=Interaction)
    interaction.confirm.return_value = True
    context = ToolContext(interaction=interaction, tool_name="guarded")

    assert context.confirm("Continue?", default=True) is True
    interaction.confirm.assert_called_once_with("Continue?", default=True)


def test_authorize_reuses_dispatch_grants_and_fails_closed_without_policy():
    """Additional authorization reuses exact grants and denies unavailable policy services."""
    interaction = Mock(spec=Interaction)
    grant = PermissionRequest(
        tool_name="demo", capability=Capability.FILESYSTEM_READ, resource="/project/file"
    )
    context = ToolContext(interaction, "demo", grants=frozenset({grant}))

    assert context.authorize(Capability.FILESYSTEM_READ, resource="/project/file") is True
    assert context.authorize(Capability.FILESYSTEM_WRITE, resource="/project/file") is False


def test_authorize_delegates_additional_requests_to_the_permission_manager():
    """A tool can request stricter authority discovered during execution."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(
        interaction=interaction,
        configuration=PermissionConfiguration(mode=PermissionMode.UNRESTRICTED),
    )
    context = ToolContext(interaction, "demo", permission_manager=manager)

    assert context.authorize(Capability.NETWORK_WRITE, reason="Publish result.") is True


def test_instruction_observations_delegate_only_when_a_manager_is_available(tmp_path):
    """Context observations remain optional and forward each supported event."""
    interaction = Mock(spec=Interaction)
    manager = Mock(spec=InstructionsManager)
    context = ToolContext(interaction, "files", manager)

    context.observe_file(tmp_path / "file.txt")
    context.observe_directory(tmp_path)
    context.invalidate_instructions(tmp_path / "AGENTS.md")

    assert manager.observe_path.call_args_list[0].args == (tmp_path / "file.txt",)
    assert manager.observe_path.call_args_list[1].kwargs == {"directory": True}
    manager.invalidate.assert_called_once_with(tmp_path / "AGENTS.md")

    unavailable = ToolContext(interaction, "files")
    unavailable.observe_file(tmp_path / "file.txt")
    unavailable.observe_directory(tmp_path)
    unavailable.invalidate_instructions()
