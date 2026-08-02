"""Tests for user-command registration and dispatch."""

from unittest.mock import Mock

import pytest

from loop import Command, CommandManager, Interaction


def command(name: str = "/test") -> Command:
    """Build a command with a mock handler."""
    return Command(name, "Test command.", Mock())


def test_manager_registers_builtins_before_additional_commands():
    """Built-ins remain first while extra definitions preserve registration order."""
    first = command("/first")
    second = command("/second")

    manager = CommandManager((first, second))

    assert [item.name for item in manager.commands] == [
        "/help",
        "/exit",
        "/quit",
        "/first",
        "/second",
    ]
    assert manager.commands[-2:] == (first, second)
    assert manager.exit_requested is False


@pytest.mark.parametrize("name", ["help", "/", "/bad name", "/bad\tname"])
def test_register_rejects_invalid_command_names(name):
    """Command names require a useful slash-prefixed token without whitespace."""
    with pytest.raises(ValueError, match="Invalid command name"):
        CommandManager((command(name),))


def test_register_rejects_duplicate_command_names():
    """One public command name cannot route to multiple handlers."""
    manager = CommandManager()

    with pytest.raises(ValueError, match="already registered"):
        manager.register(command("/help"))


def test_handle_user_command_leaves_normal_messages_unhandled():
    """Ordinary model prompts bypass local command routing."""
    registered = command()
    interaction = Mock(spec=Interaction)
    manager = CommandManager((registered,))

    assert manager.handle_user_command("explain /test", interaction) is False
    registered.handler.assert_not_called()


def test_handle_user_command_passes_interaction_and_stripped_arguments():
    """Recognized commands receive the active interaction and normalized arguments."""
    registered = command()
    interaction = Mock(spec=Interaction)
    manager = CommandManager((registered,))

    assert manager.handle_user_command("/test\t  some arguments  ", interaction) is True
    registered.handler.assert_called_once_with(manager, interaction, "some arguments")
    assert manager.exit_requested is False


def test_handle_user_command_consumes_and_displays_unknown_commands():
    """Unknown slash commands are reported locally without requesting termination."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager()

    assert manager.handle_user_command("/missing argument", interaction) is True
    interaction.warning.assert_called_once_with(
        "Unknown command '/missing'. Type /help for available commands."
    )
    assert manager.exit_requested is False


def test_request_exit_changes_manager_state_without_a_dispatch_result():
    """Termination is explicit manager state independent of command handling results."""
    manager = CommandManager()

    manager.request_exit()

    assert manager.exit_requested is True
