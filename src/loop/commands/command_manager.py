"""Register and dispatch schema-backed user commands."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..completion import CommandCompletion
from ..errors import Problem
from ..telemetry import telemetry_activity, telemetry_error
from .command import Command, CommandRegistration, CommandsProvider
from .context import CommandContext
from .models import CommandArgumentError
from .utils import takes_command_context

if TYPE_CHECKING:
    from ..interaction import Interaction


class CommandManager:
    """Collect command declarations and route user input to their functions.

    Args:
        commands (Iterable[CommandRegistration | Callable[..., None]] | None): Commands
            registered after provider commands in iteration order, or ``None``.
        interaction (Interaction | None): Default interaction used during dispatch, or ``None``
            when callers will provide one for each invocation.
        exit_command_names (tuple[str, ...] | None): Names that request conversation termination.
            Defaults to ``("exit", "quit")``. Pass ``None`` to expose no exit command. The help
            command is always registered.
        providers (Iterable[CommandsProvider] | None): Capability providers registered after
            built-ins and before individual commands, or ``None``.

    Raises:
        ValueError: If a command name is invalid or registered more than once.
        CommandRegistrationError: If a function cannot be represented by an argument schema.
    """

    _commands: dict[str, Command]
    _exit_requested: bool
    _interaction: Interaction | None

    def __init__(
        self,
        commands: Iterable[CommandRegistration | Callable[..., None]] | None = None,
        interaction: Interaction | None = None,
        exit_command_names: tuple[str, ...] | None = ("exit", "quit"),
        providers: Iterable[CommandsProvider] | None = None,
    ) -> None:
        self._commands = {}
        self._exit_requested = False
        self._interaction = interaction
        self.register(
            CommandRegistration(
                self.help,
                name="help",
                description="Show the available commands.",
            )
        )
        for name in exit_command_names or ():
            self.register(
                CommandRegistration(
                    self.request_exit,
                    name=name,
                    description="End the conversation.",
                )
            )
        self.register_providers(providers or ())
        self.register_all(commands or ())

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
        function: CommandRegistration | Callable[..., None],
        *,
        name: str | None = None,
        description: str | None = None,
        completion: CommandCompletion | None = None,
    ) -> Callable[..., None]:
        """Register a command declaration or function, directly or as a decorator.

        Args:
            function (Command | CommandRegistration | Callable[..., None] | None): Command,
                registration, or function to register. Omit it when using options as a decorator.
            name (str | None): Slash-free command name. Defaults to the function name.
            description (str | None): Display description. Defaults to the docstring summary.
            completion (CommandCompletion | None): Optional shell-like argument completion grammar.

        Raises:
            ValueError: If the command name is invalid or duplicated.
            CommandRegistrationError: If metadata or parameters cannot produce a schema.
        """
        if isinstance(function, CommandRegistration):
            registration = function
            function = registration.function
            name = name if name is not None else registration.name
            description = description if description is not None else registration.description
            completion = completion if completion is not None else registration.completion
        declaration = Command.get_declaration(function) or Command(function)
        command = declaration.registered(
            name=name,
            description=description,
            completion=completion,
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

    def register_all(
        self,
        commands: Iterable[CommandRegistration | Callable[..., None]],
    ) -> None:
        """Register commands in iteration order.

        Args:
            commands (Iterable[Command | CommandRegistration | Callable[..., None]]): Commands to
                register.
        """
        for command in commands:
            self.register(command)

    def register_provider(self, provider: CommandsProvider) -> None:
        """Register all commands exposed by one provider.

        Args:
            provider (CommandsProvider): Provider whose registrations should be added.
        """
        self.register_all(provider.get_commands())

    def register_providers(self, providers: Iterable[CommandsProvider]) -> None:
        """Register commands exposed by multiple providers in order.

        Args:
            providers (Iterable[CommandsProvider]): Providers to register.
        """
        for provider in providers:
            self.register_provider(provider)

    def handle_user_command(self, user_input: str, interaction: Interaction | None = None) -> bool:
        """Classify and dispatch slash-prefixed user input.

        Args:
            user_input (str): Stripped user input to classify and dispatch.
            interaction (Interaction | None): Invocation interaction overriding the default.

        Returns:
            bool: ``True`` when the input was consumed as a command; otherwise ``False``.

        Raises:
            CommandArgumentError: If argument syntax or binding is invalid and no interaction is
                available to report it.
            ValidationError: If arguments fail schema validation and no interaction is available
                to report it.
            ValueError: If dispatch requires an unavailable interaction.
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
            arguments (str): Shell-like positional and ``name=value`` argument text.
            interaction (Interaction | None): Invocation interaction overriding the default.

        Raises:
            CommandArgumentError: If argument syntax or binding is invalid and no
                interaction is available to report it.
            ValidationError: If arguments fail schema validation and no interaction is available
                to report it.
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
            active_interaction.report(
                Problem(
                    code="command.unknown",
                    title="Unknown command",
                    detail=f"Unknown command '/{name}'. Type /help for available commands.",
                    severity="warning",
                    operation=name,
                )
            )
            return

        context = None
        if takes_command_context(command.function):
            if active_interaction is None:
                raise ValueError(f"Command '{name}' requires an Interaction.")
            context = CommandContext(
                name=name,
                interaction=active_interaction,
            )
        try:
            telemetry_activity("command.started", command=name)
            command.call(arguments, context)
            telemetry_activity("command.completed", command=name)
        except (CommandArgumentError, ValidationError) as exc:
            telemetry_error(
                "command.failed",
                error_type="command.invalid_arguments",
                exception=exc,
                command=name,
            )
            if active_interaction is None:
                raise
            active_interaction.report(
                Problem.from_exception(
                    exc,
                    code="command.invalid_arguments",
                    title="Invalid command arguments",
                    detail=f"Invalid arguments for command '/{name}': {exc}",
                    severity="warning",
                    operation=name,
                )
            )

    def request_exit(self) -> None:
        """Request termination of the active conversation loop."""
        self._exit_requested = True

    def help(self, context: CommandContext) -> None:
        """Display the complete command catalog in alphabetical order."""
        commands = sorted(self.commands, key=lambda command: command.name.casefold())
        context.interaction.table(commands, title="Available commands:", prefix="  /")
