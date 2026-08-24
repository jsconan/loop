"""Tests for user-command registration and dispatch."""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from loop import (
    Command,
    CommandCompletion,
    CommandContext,
    CommandManager,
    CommandRegistration,
    CommandRegistrationError,
    CommandsProvider,
    CompletionValue,
    Interaction,
    command,
)
from loop.commands.utils import get_command_arguments_model


def declared_command(name: str = "test") -> Command:
    """Build a parameterless command with explicit metadata."""

    def function() -> None:
        pass

    return Command(
        function,
        name=name,
        description="Test command.",
        arguments_model=get_command_arguments_model(function, name),
    )


def test_manager_registers_explicit_discovered_and_configured_commands():
    """Every registration form preserves explicit catalog order."""
    explicit = declared_command("explicit")

    def discovered() -> None:
        """Run a discovered command."""

    configured = CommandRegistration(discovered, name="configured")
    manager = CommandManager(
        (
            CommandRegistration(
                explicit.function, name=explicit.name, description=explicit.description
            ),
            discovered,
            configured,
        ),
        Mock(spec=Interaction),
    )

    assert [item.name for item in manager.commands] == [
        "help",
        "exit",
        "quit",
        "explicit",
        "discovered",
        "configured",
    ]
    assert manager.commands[3].name == explicit.name
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


def test_manager_registers_structural_command_providers():
    """Providers contribute registrations without inheriting a framework base class."""

    class Provider:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_commands(self):
            """Return one configured command."""

            def supplied() -> None:
                """Run a supplied command."""

            return (CommandRegistration(supplied, name=self.name),)

    provider: CommandsProvider = Provider("first")
    manager = CommandManager(providers=(provider,))
    manager.register_providers((Provider("second"),))

    assert [command.name for command in manager.commands] == [
        "help",
        "exit",
        "quit",
        "first",
        "second",
    ]


def test_register_resolves_declared_metadata_and_registration_options():
    """Registration resolves passive declarations and explicit options."""
    manager = CommandManager()

    @command
    def discovered(enabled: bool = False) -> None:
        """Discover this description."""

    def target() -> None:
        """Run the target command."""

    manager.register(discovered)
    manager.register(target, name="other", description="Declared description.")

    assert discovered is manager.commands[-2].function
    assert manager.commands[-2].description == "Discover this description."
    assert target is manager.commands[-1].function
    assert manager.commands[-1].name == "other"
    assert manager.commands[-1].description == "Declared description."


def test_register_retains_explicit_and_function_declared_completion_metadata():
    """Registration carries completion grammars from options or decorated functions."""
    manager = CommandManager()
    explicit = CommandCompletion(values=(CompletionValue("explicit"),))
    inherited = CommandCompletion(values=(CompletionValue("inherited"),))

    @command(completion=explicit)
    def first(value: str) -> None:
        """Select the first value."""

    def second(value: str) -> None:
        """Select the second value."""

    CommandCompletion.set_completion(second, inherited)
    manager.register(first)
    manager.register(second)

    assert manager.commands[-2].completion is explicit
    assert manager.commands[-1].completion is inherited


@pytest.mark.parametrize("name", ["", "/test", "bad name", "bad\tname"])
def test_register_rejects_invalid_command_names(name):
    """Declared command names must be useful slash-free tokens without whitespace."""
    manager = CommandManager()

    with pytest.raises(ValueError, match="Invalid command name"):
        manager.register(
            CommandRegistration(
                declared_command(name).function,
                name=name,
                description="Test command.",
            )
        )


def test_register_rejects_duplicates_and_allows_registration_metadata_overrides():
    """Registration rejects duplicates and allows metadata overrides on registrations."""
    manager = CommandManager()
    manager.register(
        CommandRegistration(
            declared_command("test").function, name="test", description="Test command."
        )
    )

    with pytest.raises(ValueError, match="already registered"):
        manager.register(
            CommandRegistration(
                declared_command("test").function, name="test", description="Test command."
            )
        )
    manager.register(
        CommandRegistration(
            declared_command().function,
            name="renamed",
            description="Test command.",
        )
    )

    def configured() -> None:
        """Run the configured command."""

    completion = CommandCompletion(values=(CompletionValue("configured"),))
    manager.register(
        CommandRegistration(configured, name="configured", description="Run."),
        name="other",
        description="Run another command.",
        completion=completion,
    )
    assert manager.commands[-1].name == "other"
    assert manager.commands[-1].description == "Run another command."
    assert manager.commands[-1].completion is completion


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

    def inspect_context(context: CommandContext, count: int) -> None:
        """Inspect one invocation."""
        assert context.interaction is interaction
        calls.append((context.name, count))

    manager.register(inspect_context)

    assert manager.handle_user_command("ordinary /inspect_context") is False
    assert manager.handle_user_command("/inspect_context 3") is True
    assert manager.call("inspect_context", "count=4") is None
    assert calls == [("inspect_context", 3), ("inspect_context", 4)]


def test_handle_user_command_reports_invalid_and_unknown_commands():
    """Invalid arguments and unknown slash commands are consumed and reported."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)

    def pair(first: int, second: int) -> None:
        """Add a pair."""

    manager.register(pair)

    assert manager.handle_user_command("/pair invalid") is True
    assert interaction.report.call_args.args[0].detail.startswith(
        "Invalid arguments for command '/pair'"
    )
    assert manager.handle_user_command("/pair unknown=1") is True
    assert "Unknown parameter 'unknown'" in interaction.report.call_args.args[0].detail
    assert manager.handle_user_command("/missing argument") is True
    assert interaction.report.call_args.args[0].detail == (
        "Unknown command '/missing'. Type /help for available commands."
    )


def test_call_requires_interaction_only_when_dispatch_needs_it():
    """Context-free calls work alone while contextual, unknown, and invalid calls need output."""
    manager = CommandManager()

    plain_called = []

    def plain() -> None:
        """Perform a plain side effect."""
        plain_called.append(True)

    def contextual(context: CommandContext) -> None:
        """Use context."""

    def count(value: int) -> None:
        """Parse a count."""

    manager.register(plain)
    manager.register(contextual)
    manager.register(count)

    assert manager.call("plain") is None
    assert plain_called == [True]
    with pytest.raises(ValueError, match="requires an Interaction"):
        manager.call("contextual")
    with pytest.raises(ValueError, match="dispatch requires an Interaction"):
        manager.call("missing")
    with pytest.raises(ValidationError):
        manager.call("count", "invalid")
    with pytest.raises(ValueError, match="must not start"):
        manager.call("/plain")


def test_help_is_mandatory_and_displays_the_complete_sorted_catalog():
    """Every manager exposes discovery and renders all registered commands alphabetically."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction, exit_command_names=None)

    def zebra() -> None:
        """Run the zebra command."""

    manager.register(zebra)

    manager.call("help")

    rows = [command.name for command in interaction.table.call_args.args[0]]
    assert rows == ["help", "zebra"]
    assert interaction.table.call_args.kwargs == {
        "title": "Available commands:",
        "prefix": "  /",
    }


def test_exit_command_names_control_termination_aliases():
    """Configured aliases request exit while None adds no termination command."""
    manager = CommandManager(exit_command_names=("stop", "leave"))

    assert [command.name for command in manager.commands] == ["help", "stop", "leave"]
    manager.call("stop")
    assert manager.exit_requested is True

    without_exit = CommandManager(exit_command_names=None)
    assert [command.name for command in without_exit.commands] == ["help"]


def test_request_exit_changes_manager_state():
    """Termination remains an explicit manager operation independent of command aliases."""
    manager = CommandManager(exit_command_names=None)
    manager.request_exit()
    assert manager.exit_requested is True
