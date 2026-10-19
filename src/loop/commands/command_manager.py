"""Register and dispatch schema-backed user commands."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from contextvars import copy_context
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..completion import CommandCompletion, CommandCompletionAdapter
from ..errors import Problem, ProblemException, log_problem
from ..models import CommandCapability, CommandCapabilityProvider
from ..telemetry import telemetry_activity, telemetry_error
from .command import Command, CommandRegistration, CommandsProvider
from .context import CommandContext
from .models import CommandArgumentError
from .utils import takes_command_context

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..interaction import Interaction


class CommandManager:
    """Collect command declarations and route user input to their functions.

    Commands declare the runtime capabilities they need. When a runtime context is supplied, the
    same capability check controls dispatch, help, and completion; unavailable commands are
    consumed with an explanation instead of being treated as conversation input.

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
    _capability_provider: CommandCapabilityProvider | None
    _completion_adapter: CommandCompletionAdapter

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
        self._capability_provider = None
        self._completion_adapter = CommandCompletionAdapter(
            lambda: self.active_commands,
            capabilities=lambda: self.capabilities,
        )
        self.register(
            CommandRegistration(
                self.help,
                name="help",
                description="Show the available commands.",
                capabilities=frozenset({CommandCapability.CONTEXT_READ}),
            )
        )
        for name in exit_command_names or ():
            self.register(
                CommandRegistration(
                    self.request_exit,
                    name=name,
                    description="End the conversation.",
                    capabilities=frozenset({CommandCapability.CONVERSATION_CONTROL}),
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
    def capabilities(self) -> frozenset[CommandCapability] | None:
        """Return capabilities provided by the current runtime context.

        Returns:
            frozenset[CommandCapability] | None: Active capabilities, or ``None`` when no runtime
                context restricts command availability.
        """
        return self._capability_provider() if self._capability_provider is not None else None

    @property
    def active_commands(self) -> tuple[Command, ...]:
        """Return commands whose required capabilities are present in the current context.

        Returns:
            tuple[Command, ...]: Available commands in registration order. Without a capability
                provider, the complete registry is available.
        """
        return self._commands_for(self.capabilities)

    @property
    def completion_adapter(self) -> CommandCompletionAdapter:
        """Return the completion source for registered commands and their providers.

        Returns:
            CommandCompletionAdapter: Adapter backed by this manager's active command catalog.
        """
        return self._completion_adapter

    def _commands_for(
        self,
        capabilities: frozenset[CommandCapability] | None,
    ) -> tuple[Command, ...]:
        """Select commands from one capability snapshot."""
        if capabilities is None:
            return self.commands
        return tuple(
            command for command in self._commands.values() if command.capabilities <= capabilities
        )

    def set_capability_provider(
        self,
        provider: CommandCapabilityProvider | None,
    ) -> None:
        """Set the live context source used to select available commands.

        Args:
            provider (CommandCapabilityProvider | None): Source of current capabilities, or
                ``None`` to expose every registered command.
        """
        self._capability_provider = provider

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
        capabilities: frozenset[CommandCapability] | None = None,
    ) -> Callable[..., None]:
        """Register a command declaration or function, directly or as a decorator.

        Args:
            function (Command | CommandRegistration | Callable[..., None] | None): Command,
                registration, or function to register. Omit it when using options as a decorator.
            name (str | None): Slash-free command name. Defaults to the function name.
            description (str | None): Display description. Defaults to the docstring summary.
            completion (CommandCompletion | None): Optional shell-like argument completion grammar.
            capabilities (frozenset[CommandCapability] | None): Required runtime capabilities, or
                ``None`` to inherit the declaration. Defaults to ``None``.

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
            capabilities = capabilities if capabilities is not None else registration.capabilities
        declaration = Command.get_declaration(function) or Command(function)
        command = declaration.registered(
            name=name,
            description=description,
            completion=completion,
            capabilities=capabilities,
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
        """Register commands and completion sources exposed by one provider.

        Args:
            provider (CommandsProvider): Provider whose command and completion declarations are
                added.
        """
        self.register_all(provider.get_commands())
        self._completion_adapter.register_provider(provider)

    def register_providers(self, providers: Iterable[CommandsProvider]) -> None:
        """Register commands exposed by multiple providers in order.

        Args:
            providers (Iterable[CommandsProvider]): Providers to register.
        """
        for provider in providers:
            self.register_provider(provider)

    def handle_user_command(
        self,
        user_input: str,
        interaction: Interaction | None = None,
    ) -> bool:
        """Classify and dispatch slash-prefixed user input.

        Args:
            user_input (str): Stripped user input to classify and dispatch.
            interaction (Interaction | None): Invocation interaction overriding the default.

        Returns:
            bool: ``True`` when the input was consumed as a command; otherwise ``False``.

        Raises:
            CommandArgumentError: If argument syntax or binding is invalid and no interaction is
                available to report it.
            ProblemException: If a command reports a structured operational problem and no
                interaction is available to display it.
            ValidationError: If arguments fail schema validation and no interaction is available
                to report it.
            ValueError: If dispatch requires an unavailable interaction.
        """
        command = self._parse_user_command(user_input)
        if command is None:
            return False
        name, arguments = command
        self.call(name, arguments, interaction=interaction)
        return True

    async def handle_user_command_async(
        self,
        user_input: str,
        interaction: Interaction | None = None,
    ) -> bool:
        """Await synchronous command dispatch without blocking its async interaction owner.

        Dispatch one command in a worker while its caller awaits completion. Cancellation waits
        for the command to settle before returning ownership, including repeated cancellation.
        Ordinary conversation input returns immediately without starting a worker.

        Args:
            user_input (str): Stripped user input to classify and dispatch.
            interaction (Interaction | None): Invocation interaction overriding the default.

        Returns:
            bool: Whether the input was consumed as a command.

        Raises:
            asyncio.CancelledError: After owned command cleanup when the caller is cancelled.
            KeyboardInterrupt: A synchronous command interrupts its execution.
            CommandArgumentError: Argument syntax or binding is invalid without an interaction.
            ProblemException: A command reports a problem without an interaction to display it.
            ValidationError: Arguments fail schema validation without a reporting interaction.
            ValueError: Dispatch requires an unavailable interaction.
        """
        if not self.is_user_command(user_input):
            return False
        pending = asyncio.get_running_loop().run_in_executor(
            None,
            copy_context().run,
            self.handle_user_command,
            user_input,
            interaction,
        )
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError as error:
            while not pending.done():
                try:
                    await asyncio.wait({pending})
                except asyncio.CancelledError:
                    continue
            failure = pending.exception()
            if failure is not None:
                error.add_note(f"Command also stopped with {type(failure).__name__}.")
            raise

    def is_user_command(self, user_input: str) -> bool:
        """Return whether input uses the registered command syntax.

        Args:
            user_input (str): User input to classify.

        Returns:
            bool: Whether ``user_input`` is a command invocation.
        """
        return self._parse_user_command(user_input) is not None

    @staticmethod
    def _parse_user_command(user_input: str) -> tuple[str, str] | None:
        """Parse command input once using the manager's command prefix."""
        if not user_input.startswith("/"):
            return None
        parts = user_input[1:].split(maxsplit=1)
        name = parts[0] if parts else ""
        arguments = parts[1].strip() if len(parts) == 2 else ""
        return name, arguments

    def call(
        self,
        name: str,
        arguments: str = "",
        *,
        interaction: Interaction | None = None,
        available_commands: Iterable[Command] | None = None,
    ) -> None:
        """Dispatch a command call by its slash-free registered name.

        Args:
            name (str): Slash-free registered command name.
            arguments (str): Shell-like positional and ``name=value`` argument text.
            interaction (Interaction | None): Invocation interaction overriding the default.
            available_commands (Iterable[Command] | None): Context-filtered catalog shown by a help
                command invocation, or ``None`` to show the full registry.

        Raises:
            CommandArgumentError: If argument syntax or binding is invalid and no
                interaction is available to report it.
            ProblemException: If a command reports a structured operational problem and no
                interaction is available to display it.
            ValidationError: If arguments fail schema validation and no interaction is available
                to report it.
            ValueError: If a slash-prefixed name is supplied or an interaction required for
                dispatch is unavailable.
        """
        if name.startswith("/"):
            raise ValueError("Command names passed to call() must not start with '/'.")
        active_interaction = interaction if interaction is not None else self._interaction
        capabilities = self.capabilities
        available = self._commands_for(capabilities)
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

        if command not in available:
            self._report_unavailable(name, active_interaction)
            return

        context = None
        if takes_command_context(command.function):
            if active_interaction is None:
                raise ValueError(f"Command '{name}' requires an Interaction.")
            context = CommandContext(
                name=name,
                interaction=active_interaction,
                available_commands=(
                    tuple(available_commands) if available_commands is not None else available
                ),
                capabilities=capabilities,
            )
        try:
            telemetry_activity("command.started", command=name)
            command.call(arguments, context)
            telemetry_activity("command.completed", command=name)
        except ProblemException as exc:
            log_problem(_LOGGER, exc.problem, exc)
            if active_interaction is None:
                raise
            active_interaction.report(exc.problem)
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

    def _report_unavailable(
        self,
        name: str,
        interaction: Interaction | None,
    ) -> None:
        """Explain that a registered command is inactive in the current context."""
        if interaction is None:
            raise ValueError("Command dispatch requires an Interaction.")
        interaction.warning(
            f"Command '/{name}' is unavailable in the current conversation state. "
            "Type /help to see commands that can run now."
        )

    def help(self, context: CommandContext) -> None:
        """Display the command catalog available in the current dispatch context."""
        available_commands = (
            context.available_commands if context.available_commands is not None else self.commands
        )
        commands = sorted(
            available_commands,
            key=lambda command: command.name.casefold(),
        )
        context.interaction.table(commands, title="Available commands:", prefix="  /")
