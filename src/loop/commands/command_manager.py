"""Register and dispatch schema-backed user commands."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..completion import COMPLETION_ATTRIBUTE, CommandCompletion
from ..context import CommandContext
from .builtins import exit as exit_command
from .builtins import help as help_command
from .builtins import permissions as permissions_command
from .builtins import quit as quit_command
from .command import Command
from .utils import CommandRegistrationError, get_command_arguments_model, takes_command_context

if TYPE_CHECKING:
    from ..interaction import Interaction
    from ..permissions import PermissionManager


BUILTIN_COMMANDS = (help_command, permissions_command, exit_command, quit_command)


class CommandManager:
    """Collect command declarations and route user input to their functions.

    Args:
        commands (Iterable[Command | Callable[..., None]] | None): Additional command declarations
            registered after the built-ins, or ``None`` to register only the built-ins.
        interaction (Interaction | None): Default interaction used during dispatch, or ``None``
            when callers will provide one for each invocation.
        permission_manager (PermissionManager | None): Tool policy manager exposed to permission
            management commands.

    Raises:
        ValueError: If a command name is invalid or registered more than once.
        CommandRegistrationError: If a function cannot be represented by an argument schema.
    """

    _commands: dict[str, Command]
    _exit_requested: bool
    _interaction: Interaction | None

    def __init__(
        self,
        commands: Iterable[Command | Callable[..., None]] | None = None,
        interaction: Interaction | None = None,
        permission_manager: PermissionManager | None = None,
    ) -> None:
        self._commands = {}
        self._exit_requested = False
        self._interaction = interaction
        self._permission_manager = permission_manager
        for command in (*BUILTIN_COMMANDS, *(commands or ())):
            self.register(command)

    @property
    def interaction(self) -> Interaction | None:
        """Return the default interaction used during command dispatch.

        Returns:
            Interaction | None: The default interaction, or ``None`` when none is configured.
        """
        return self._interaction

    @interaction.setter
    def interaction(self, interaction: Interaction | None) -> None:
        """Set or clear the default interaction used during command dispatch.

        Args:
            interaction (Interaction | None): Default interaction to use, or ``None`` to clear it.
        """
        self._interaction = interaction

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

    def register(
        self,
        function: Command | Callable[..., None] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        completion: CommandCompletion | None = None,
    ) -> Callable[..., None]:
        """Register a command declaration or function, directly or as a decorator.

        Args:
            function (Command | Callable[..., None] | None): Command or function to register when
                called directly. Omit it when using registration options as a decorator.
            name (str | None): Slash-free command name. Defaults to the function name.
            description (str | None): Display description. Defaults to the docstring summary.
            completion (CommandCompletion | None): Optional shell-like argument completion grammar.

        Returns:
            Callable[..., None]: The registered function, or a decorator when no target is given.

        Raises:
            ValueError: If the command name is invalid, duplicated, or conflicts with explicit
                metadata supplied for a ``Command`` instance.
            CommandRegistrationError: If metadata or parameters cannot produce a schema.
        """

        def _register(target: Command | Callable[..., None]) -> Callable[..., None]:
            if isinstance(target, Command):
                if name is not None or description is not None or completion is not None:
                    raise ValueError("Explicit metadata cannot override a Command declaration.")
                command = target
            else:
                command_name = name or target.__name__
                command = Command(
                    name=command_name,
                    description=description or self._description_for(target),
                    function=target,
                    arguments_model=get_command_arguments_model(target, command_name),
                    completion=completion or getattr(target, COMPLETION_ATTRIBUTE, None),
                )
            if (
                not command.name
                or command.name.startswith("/")
                or any(character.isspace() for character in command.name)
            ):
                raise ValueError(f"Invalid command name '{command.name}'.")
            if command.name in self._commands:
                raise ValueError(f"Command '{command.name}' is already registered.")
            self._commands[command.name] = command
            return command.function

        return _register(function) if function is not None else _register

    def handle_user_command(self, user_input: str, interaction: Interaction | None = None) -> bool:
        """Classify and dispatch slash-prefixed user input.

        Args:
            user_input (str): Stripped user input to classify and dispatch.
            interaction (Interaction | None): Invocation interaction overriding the default.

        Returns:
            bool: ``True`` when the input was consumed as a command; otherwise ``False``.
        """
        if not user_input.startswith("/"):
            return False

        parts = user_input[1:].split(maxsplit=1)
        name = parts[0]
        arguments = parts[1] if len(parts) == 2 else ""
        self.call(name, arguments.strip(), interaction=interaction)
        return True

    def call(
        self,
        name: str,
        arguments: str = "",
        *,
        interaction: Interaction | None = None,
    ) -> None:
        """Dispatch a command call by its slash-free registered name.

        Args:
            name (str): Slash-free registered command name.
            arguments (str): Raw argument text, or a JSON object for multiple parameters.
            interaction (Interaction | None): Invocation interaction overriding the default.

        Raises:
            ValueError: If a slash-prefixed name is supplied or an interaction required for
                dispatch is unavailable.
        """
        if name.startswith("/"):
            raise ValueError("Command names passed to call() must not start with '/'.")
        active_interaction = interaction if interaction is not None else self._interaction
        command = self._commands.get(name)
        if command is None:
            if active_interaction is None:
                raise ValueError("Command dispatch requires an Interaction.")
            active_interaction.warning(
                f"Unknown command '/{name}'. Type /help for available commands."
            )
            return

        context = None
        if takes_command_context(command.function):
            if active_interaction is None:
                raise ValueError(f"Command '{name}' requires an Interaction.")
            context = CommandContext(
                name=name,
                interaction=active_interaction,
                manager=self,
                permission_manager=self._permission_manager,
            )
        try:
            command.call(arguments, context)
        except ValidationError as exc:
            if active_interaction is None:
                raise
            active_interaction.warning(f"Invalid arguments for command '/{name}': {exc}")

    def request_exit(self) -> None:
        """Request termination of the active conversation loop."""
        self._exit_requested = True

    @staticmethod
    def _description_for(function: Callable[..., None]) -> str:
        """Return a command description from its docstring summary."""
        docstring = inspect.getdoc(function)
        if not docstring:
            raise CommandRegistrationError(f"Command '{function.__name__}' must have a docstring.")
        return docstring.split("\n\n", maxsplit=1)[0].replace("\n", " ")
