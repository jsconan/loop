"""Tests for tool invocation context."""

from unittest.mock import Mock

from loop import InstructionsManager, ToolContext
from loop.interaction import Interaction


def test_confirm_delegates_to_the_interaction():
    """Confirmation forwards the question and default through the interaction boundary."""
    interaction = Mock(spec=Interaction)
    interaction.confirm.return_value = True
    context = ToolContext(interaction=interaction, tool_name="guarded")

    assert context.confirm("Continue?", default=True) is True
    interaction.confirm.assert_called_once_with("Continue?", default=True)


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
