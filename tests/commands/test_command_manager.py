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

    manager = CommandManager((first, second), Mock(spec=Interaction))

    assert [item.name for item in manager.commands] == [
        "/help",
        "/exit",
        "/quit",
        "/first",
        "/second",
    ]
    assert manager.commands[-2:] == (first, second)
    assert manager.exit_requested is False


def test_interaction_property_can_be_replaced_and_cleared():
    """The default interaction can be read, replaced, and cleared."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager()

    assert manager.interaction is None
    manager.interaction = interaction
    assert manager.interaction is interaction
    manager.interaction = None
    assert manager.interaction is None


@pytest.mark.parametrize("name", ["help", "/", "/bad name", "/bad\tname"])
def test_register_rejects_invalid_command_names(name):
    """Command names require a useful slash-prefixed token without whitespace."""
    with pytest.raises(ValueError, match="Invalid command name"):
        CommandManager((command(name),), Mock(spec=Interaction))


def test_register_rejects_duplicate_command_names():
    """One public command name cannot route to multiple handlers."""
    manager = CommandManager(interaction=Mock(spec=Interaction))

    with pytest.raises(ValueError, match="already registered"):
        manager.register(command("/help"))


def test_handle_user_command_leaves_normal_messages_unhandled():
    """Ordinary model prompts bypass local command routing."""
    registered = command()
    interaction = Mock(spec=Interaction)
    manager = CommandManager((registered,), interaction)

    assert manager.handle_user_command("explain /test") is False
    registered.handler.assert_not_called()


def test_handle_user_command_uses_instance_interaction_and_stripped_arguments():
    """Recognized commands use the stored interaction and normalized arguments by default."""
    registered = command()
    interaction = Mock(spec=Interaction)
    manager = CommandManager((registered,), interaction)

    assert manager.handle_user_command("/test\t  some arguments  ") is True
    registered.handler.assert_called_once_with(manager, interaction, "some arguments")
    assert manager.exit_requested is False


def test_handle_user_command_accepts_call_interaction_without_stored_interaction():
    """A call-specific interaction works when the manager has no stored interaction."""
    registered = command()
    call_interaction = Mock(spec=Interaction)
    manager = CommandManager((registered,))

    assert manager.handle_user_command("/test", call_interaction) is True
    registered.handler.assert_called_once_with(manager, call_interaction, "")


def test_handle_user_command_consumes_and_displays_unknown_commands():
    """Unknown slash commands are reported locally without requesting termination."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)

    assert manager.handle_user_command("/missing argument") is True
    interaction.warning.assert_called_once_with(
        "Unknown command '/missing'. Type /help for available commands."
    )
    assert manager.exit_requested is False


def test_call_dispatches_registered_command_with_explicit_interaction():
    """Direct command calls route supplied arguments and invocation interaction."""
    registered = command()
    interaction = Mock(spec=Interaction)
    manager = CommandManager((registered,))

    manager.call("/test", "some arguments", interaction=interaction)

    registered.handler.assert_called_once_with(manager, interaction, "some arguments")


def test_request_exit_changes_manager_state_without_a_dispatch_result():
    """Termination is explicit manager state independent of command handling results."""
    manager = CommandManager()

    manager.request_exit()

    assert manager.exit_requested is True
