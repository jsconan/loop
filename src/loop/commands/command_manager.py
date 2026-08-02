"""Register and dispatch user commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .builtins import exit_command, help_command
from .command import Command

if TYPE_CHECKING:
    from ..interaction import Interaction


BUILTIN_COMMANDS = (
    Command("/help", "Show the available commands.", help_command),
    Command("/exit", "End the conversation.", exit_command),
    Command("/quit", "End the conversation.", exit_command),
)


class CommandManager:
    """Collect predefined commands and handle command input.

    Args:
        commands (tuple[Command, ...]): Additional commands registered after the built-ins.

    Raises:
        ValueError: If a command name is invalid or registered more than once.
    """

    _commands: dict[str, Command]
    _exit_requested: bool

    def __init__(self, commands: tuple[Command, ...] = ()) -> None:
        self._commands = {}
        self._exit_requested = False
        for command in (*BUILTIN_COMMANDS, *commands):
            self.register(command)

    @property
    def commands(self) -> tuple[Command, ...]:
        """Return registered commands in display order.

        Returns:
            tuple[Command, ...]: Registered command definitions.
        """
        return tuple(self._commands.values())

    @property
    def exit_requested(self) -> bool:
        """Return whether a command requested conversation termination.

        Returns:
            bool: Whether the conversation should terminate.
        """
        return self._exit_requested

    def register(self, command: Command) -> None:
        """Register one predefined command.

        Args:
            command (Command): Command definition to register.

        Raises:
            ValueError: If the name does not start with a slash, contains whitespace, or is
                already registered.
        """
        if (
            not command.name.startswith("/")
            or command.name == "/"
            or any(character.isspace() for character in command.name)
        ):
            raise ValueError(f"Invalid command name '{command.name}'.")
        if command.name in self._commands:
            raise ValueError(f"Command '{command.name}' is already registered.")
        self._commands[command.name] = command

    def handle_user_command(self, user_input: str, interaction: Interaction) -> bool:
        """Handle slash-prefixed input and related display.

        Args:
            user_input (str): Stripped user input to classify and dispatch.
            interaction (Interaction): Active interaction used by commands and error reporting.

        Returns:
            bool: ``True`` when the input was consumed as a command; otherwise ``False``.
        """
        if not user_input.startswith("/"):
            return False

        parts = user_input.split(maxsplit=1)
        name = parts[0]
        arguments = parts[1] if len(parts) == 2 else ""
        command = self._commands.get(name)
        if command is None:
            interaction.warning(f"Unknown command '{name}'. Type /help for available commands.")
            return True
        command.handler(self, interaction, arguments.strip())
        return True

    def request_exit(self) -> None:
        """Request termination of the active conversation loop."""
        self._exit_requested = True
