"""Tests for commands available in every conversation."""

from unittest.mock import Mock

from loop import Command, CommandManager, Interaction
from loop.commands import exit_command, help_command


def test_help_displays_all_commands_from_manager_metadata():
    """Help renders the authoritative command catalog through the interaction."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager((Command("/long-command", "Run a longer command.", Mock()),))

    help_command(manager, interaction, "")

    help_text = interaction.info.call_args.args[0]
    assert "  /help          Show the available commands." in help_text
    assert "  /exit          End the conversation." in help_text
    assert "  /quit          End the conversation." in help_text
    assert "  /long-command  Run a longer command." in help_text


def test_help_reports_unsupported_arguments():
    """Help owns presentation of its argument error."""
    interaction = Mock(spec=Interaction)

    help_command(CommandManager(), interaction, "extra")

    interaction.warning.assert_called_once_with("/help does not accept arguments.")


def test_exit_command_requests_termination():
    """The exit handler requests manager termination."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager()

    exit_command(manager, interaction, "")

    assert manager.exit_requested is True


def test_exit_reports_unsupported_arguments_without_terminating():
    """Invalid exit usage remains handled without ending the conversation."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager()

    exit_command(manager, interaction, "later")

    interaction.warning.assert_called_once_with("Exit commands do not accept arguments.")
    assert manager.exit_requested is False
