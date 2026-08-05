"""Tests for commands available in every conversation."""

from unittest.mock import Mock

import pytest

from loop import CommandContext, CommandManager, Interaction
from loop.commands import exit as exit_command
from loop.commands import help as help_command
from loop.commands import quit as quit_command


def test_help_displays_slash_prefixed_command_catalog():
    """Help adds the presentation-only slash to slash-free command metadata."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)

    @manager.register(name="long-command", description="Run a longer command.")
    def long_command() -> None:
        pass

    help_command(CommandContext("help", interaction, manager))

    help_text = interaction.info.call_args.args[0]
    assert "  /help          Show the available commands." in help_text
    assert "  /long-command  Run a longer command." in help_text


@pytest.mark.parametrize("function,name", [(help_command, "help"), (exit_command, "exit")])
def test_builtins_require_manager_context(function, name):
    """Built-ins reject independent invocation because they operate on manager state."""
    context = CommandContext(name, Mock(spec=Interaction))
    with pytest.raises(ValueError, match="requires a CommandManager"):
        function(context)


@pytest.mark.parametrize("function,name", [(exit_command, "exit"), (quit_command, "quit")])
def test_exit_commands_request_termination(function, name):
    """Exit handlers request manager termination through their context."""
    manager = CommandManager()
    function(CommandContext(name, Mock(spec=Interaction), manager))
    assert manager.exit_requested is True
