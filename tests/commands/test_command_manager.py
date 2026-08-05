"""Tests for user-command registration and dispatch."""

from unittest.mock import Mock

import pytest

from loop import Command, CommandContext, CommandManager, CommandRegistrationError, Interaction
from loop.commands.utils import get_command_arguments_model


def declared_command(name: str = "test") -> Command:
    """Build a parameterless command with explicit metadata."""
    def function() -> None:
        pass

    return Command(name, "Test command.", function, get_command_arguments_model(function, name))


def test_manager_registers_builtins_before_declared_and_discovered_commands():
    """Built-ins and both registration forms preserve catalog order."""
    explicit = declared_command("explicit")

    def discovered() -> None:
        """Run a discovered command."""

    manager = CommandManager((explicit, discovered), Mock(spec=Interaction))

    assert [item.name for item in manager.commands] == [
        "help", "exit", "quit", "explicit", "discovered"
    ]
    assert manager.commands[3] is explicit
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


def test_register_supports_decorators_and_declared_metadata():
    """Decorators may discover metadata or receive explicit slash-free values."""
    manager = CommandManager()

    @manager.register
    def discovered(enabled: bool = False) -> None:
        """Discover this description."""

    @manager.register(name="other", description="Declared description.")
    def target() -> None:
        pass

    assert discovered is manager.commands[-2].function
    assert manager.commands[-2].description == "Discover this description."
    assert target is manager.commands[-1].function
    assert manager.commands[-1].name == "other"
    assert manager.commands[-1].description == "Declared description."


@pytest.mark.parametrize("name", ["", "/test", "bad name", "bad\tname"])
def test_register_rejects_invalid_command_names(name):
    """Declared command names must be useful slash-free tokens without whitespace."""
    manager = CommandManager()

    with pytest.raises(ValueError, match="Invalid command name"):
        manager.register(declared_command(name))


def test_register_rejects_duplicates_and_conflicting_command_metadata():
    """Registration rejects duplicate names and overrides on complete declarations."""
    manager = CommandManager()

    with pytest.raises(ValueError, match="already registered"):
        manager.register(declared_command("help"))
    with pytest.raises(ValueError, match="cannot override"):
        manager.register(declared_command(), name="renamed")


def test_register_rejects_missing_descriptions_and_invalid_parameters():
    """Discovery requires documentation and schema-compatible annotated parameters."""
    manager = CommandManager()

    def undocumented() -> None:
        pass

    def untyped(value) -> None:
        """Use an untyped value."""

    with pytest.raises(CommandRegistrationError, match="must have a docstring"):
        manager.register(undocumented)
    with pytest.raises(CommandRegistrationError, match="needs a type annotation"):
        manager.register(untyped)


def test_handle_user_command_deserializes_arguments_and_injects_context():
    """Slash input detects a command while dispatch uses its slash-free name and typed values."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)

    calls = []

    @manager.register
    def inspect_context(context: CommandContext, count: int) -> None:
        """Inspect one invocation."""
        assert context.interaction is interaction
        assert context.manager is manager
        calls.append((context.name, count))

    assert manager.handle_user_command("ordinary /inspect_context") is False
    assert manager.handle_user_command("/inspect_context 3") is True
    assert manager.call("inspect_context", '{"count": 4}') is None
    assert calls == [("inspect_context", 3), ("inspect_context", 4)]


def test_handle_user_command_reports_invalid_and_unknown_commands():
    """Invalid arguments and unknown slash commands are consumed and reported."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)

    @manager.register
    def pair(first: int, second: int) -> None:
        """Add a pair."""

    assert manager.handle_user_command("/pair invalid") is True
    assert interaction.warning.call_args.args[0].startswith("Invalid arguments for command '/pair'")
    assert manager.handle_user_command("/missing argument") is True
    interaction.warning.assert_called_with(
        "Unknown command '/missing'. Type /help for available commands."
    )


def test_call_requires_interaction_only_when_dispatch_needs_it():
    """Context-free calls work alone while contextual, unknown, and invalid calls need output."""
    manager = CommandManager()

    plain_called = []

    @manager.register
    def plain() -> None:
        """Perform a plain side effect."""
        plain_called.append(True)

    @manager.register
    def contextual(context: CommandContext) -> None:
        """Use context."""

    @manager.register
    def count(value: int) -> None:
        """Parse a count."""

    assert manager.call("plain") is None
    assert plain_called == [True]
    with pytest.raises(ValueError, match="requires an Interaction"):
        manager.call("contextual")
    with pytest.raises(ValueError, match="dispatch requires an Interaction"):
        manager.call("missing")
    with pytest.raises(Exception):  # Pydantic exposes its concrete validation exception.
        manager.call("count", "invalid")
    with pytest.raises(ValueError, match="must not start"):
        manager.call("/plain")


def test_request_exit_changes_manager_state():
    """Termination is explicit manager state independent of command return values."""
    manager = CommandManager()
    manager.request_exit()
    assert manager.exit_requested is True
