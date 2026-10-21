"""Register and dispatch schema-backed user commands."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from functools import partial
from threading import BoundedSemaphore, Lock, current_thread, get_ident
from typing import Any

from pydantic import ValidationError

from ..completion import CommandCompletion, CommandCompletionAdapter
from ..errors import Problem, ProblemException, log_problem
from ..interaction import AsyncInteractionInput, Interaction
from ..models import CommandCapability, CommandCapabilityProvider
from ..telemetry import telemetry_activity, telemetry_error
from ..utils import close_awaitable, has_running_event_loop, is_async_callable
from .command import Command, CommandRegistration, CommandsProvider
from .context import CommandContext
from .models import CommandArgumentError
from .utils import takes_command_context

_LOGGER = logging.getLogger(__name__)
_COMMAND_WORKER_COUNT = 2
_COMMAND_WORKER_QUEUE_SIZE = 2


class SyncCommandInteraction(AsyncInteractionInput):
    """Expose synchronous input through an async command on a sync-owned runner."""

    _source: Interaction

    def __init__(self, source: Interaction) -> None:
        self._source = source

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    async def prompt_async(self, *args: Any, **kwargs: Any) -> object | bool:
        """Forward a command prompt to its synchronous owner thread.

        Args:
            args (tuple[Any, ...]): Positional prompt arguments forwarded to ``prompt``.
            kwargs (dict[str, Any]): Keyword prompt arguments forwarded to ``prompt``.

        Returns:
            object | bool: The selected value, entered text, or exit result.

        Raises:
            ProblemException: If the async prompt requests initial editor text.
            ValueError: If the underlying synchronous prompt rejects its arguments.
        """
        initial = kwargs.pop("initial", None)
        if initial is not None:
            raise ProblemException(
                Problem(
                    code="interaction.async_required",
                    title="Asynchronous interaction required",
                    detail="Initial editor text is unavailable through a synchronous prompt.",
                    operation="command input",
                )
            )
        return self._source.prompt(*args, **kwargs)

    async def confirm_async(self, message: str, *, default: bool = False) -> bool:
        """Forward command confirmation to its synchronous owner thread.

        Args:
            message (str): Confirmation question to display.
            default (bool): Answer to use when the user enters no response.

        Returns:
            bool: Whether the user approved the operation.

        Raises:
            ValueError: If the underlying synchronous confirmation rejects its arguments.
        """
        return self._source.confirm(message, default=default)


class CommandManager:
    """Collect command declarations and route user input to their functions.

    Commands declare the runtime capabilities they need. When a runtime context is supplied, the
    same capability check controls dispatch, help, and completion; unavailable commands are
    consumed with an explanation instead of being treated as conversation input.

    Args:
        commands (Iterable[CommandRegistration | Callable[..., object]] | None): Commands
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
    _async_runner: asyncio.Runner | None
    _runner_thread: int | None
    _runner_lock: Lock
    _executor: ThreadPoolExecutor
    _worker_thread_prefix: str
    _worker_slots: BoundedSemaphore
    _closed: bool

    def __init__(
        self,
        commands: Iterable[CommandRegistration | Callable[..., object]] | None = None,
        interaction: Interaction | None = None,
        exit_command_names: tuple[str, ...] | None = ("exit", "quit"),
        providers: Iterable[CommandsProvider] | None = None,
    ) -> None:
        self._commands = {}
        self._exit_requested = False
        self._interaction = interaction
        self._capability_provider = None
        self._async_runner = None
        self._runner_thread = None
        self._runner_lock = Lock()
        self._worker_thread_prefix = f"loop-command-{id(self):x}"
        self._executor = ThreadPoolExecutor(
            max_workers=_COMMAND_WORKER_COUNT,
            thread_name_prefix=self._worker_thread_prefix,
        )
        self._worker_slots = BoundedSemaphore(_COMMAND_WORKER_COUNT + _COMMAND_WORKER_QUEUE_SIZE)
        self._closed = False
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

    def close(self) -> None:
        """Settle managed command workers and release the registry-owned async runner."""
        with self._runner_lock:
            if self._closed:
                return
            self._closed = True
            async_runner = self._async_runner
        self._executor.shutdown(wait=True, cancel_futures=False)
        if async_runner is not None:
            async_runner.close()

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
        function: CommandRegistration | Callable[..., object],
        *,
        name: str | None = None,
        description: str | None = None,
        completion: CommandCompletion | None = None,
        capabilities: frozenset[CommandCapability] | None = None,
    ) -> Callable[..., object]:
        """Register a command declaration or function, directly or as a decorator.

        Args:
            function (CommandRegistration | Callable[..., object] | None): Command,
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
        commands: Iterable[CommandRegistration | Callable[..., object]],
    ) -> None:
        """Register commands in iteration order.

        Args:
            commands (Iterable[CommandRegistration | Callable[..., object]]): Commands to
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
                interaction is available to display it, or async dispatch is required inside a
                running event loop.
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
        """Dispatch async callbacks on the owner and sync callbacks through bounded workers.

        Cancellation waits for worker callbacks to settle before returning ownership. Ordinary
        conversation input returns immediately.

        Args:
            user_input (str): Stripped user input to classify and dispatch.
            interaction (Interaction | None): Invocation interaction overriding the default.

        Returns:
            bool: Whether the input was consumed as a command.

        Raises:
            asyncio.CancelledError: If async work is cancelled, after sync workers settle.
            KeyboardInterrupt: If a command interrupts its execution.
            CommandArgumentError: Argument syntax or binding is invalid without an interaction.
            ProblemException: A command reports a problem without an interaction to display it,
                or synchronous work cannot be admitted safely.
            ValidationError: Arguments fail schema validation without a reporting interaction.
            ValueError: Dispatch requires an unavailable interaction.
        """
        command = self._parse_user_command(user_input)
        if command is None:
            return False
        name, arguments = command
        await self.call_async(name, arguments, interaction=interaction)
        return True

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
                interaction is available to display it, or async dispatch is required because
                this caller cannot safely use the owned runner.
            ValidationError: If arguments fail schema validation and no interaction is available
                to report it.
            ValueError: If a slash-prefixed name is supplied or an interaction required for
                dispatch is unavailable.
        """
        command, active_interaction, context = self._prepare_call(
            name,
            interaction=interaction,
            available_commands=available_commands,
            async_dispatch=False,
        )
        if command is None:
            return
        try:
            telemetry_activity("command.started", command=name)
            if is_async_callable(command.function):
                runner = self._sync_runner()
                if runner is None:
                    raise self._async_required(name)
                if (
                    context is not None
                    and active_interaction is not None
                    and not isinstance(active_interaction, AsyncInteractionInput)
                ):
                    context = replace(
                        context,
                        interaction=SyncCommandInteraction(active_interaction),
                    )
                runner.run(command.call_async(arguments, context))
            else:
                command.call(arguments, context)
            telemetry_activity("command.completed", command=name)
        except (ProblemException, CommandArgumentError, ValidationError) as error:
            self._report_call_failure(name, active_interaction, error)

    async def call_async(
        self,
        name: str,
        arguments: str = "",
        *,
        interaction: Interaction | None = None,
        available_commands: Iterable[Command] | None = None,
    ) -> None:
        """Dispatch async callbacks on the owner and sync callbacks through bounded workers.

        Args:
            name (str): Slash-free registered command name.
            arguments (str): Shell-like positional and ``name=value`` argument text.
            interaction (Interaction | None): Invocation interaction overriding the default.
            available_commands (Iterable[Command] | None): Context-filtered catalog shown by a
                help command, or ``None`` to show the full registry.

        Raises:
            asyncio.CancelledError: If async work is cancelled, after sync workers
                settle.
            KeyboardInterrupt: If a command interrupts its execution.
            CommandArgumentError: If argument syntax or binding is invalid and no interaction is
                available to report it.
            ProblemException: If a command reports a problem and no interaction is available to
                display it, worker admission is full, or a returned future belongs to another
                event loop. A command context without an owner-routed interaction is reported
                through the interaction before callback execution.
            ValidationError: If arguments fail schema validation without an interaction to report
                it.
            ValueError: If a slash-prefixed name is supplied or required interaction is missing.
        """
        command, active_interaction, context = self._prepare_call(
            name,
            interaction=interaction,
            available_commands=available_commands,
            async_dispatch=True,
        )
        if command is None:
            return
        try:
            telemetry_activity("command.started", command=name)
            if is_async_callable(command.function):
                await command.call_async(arguments, context)
            else:
                result = await self._run_sync(command, arguments, context)
                if (
                    isinstance(result, asyncio.Future)
                    and result.get_loop() is not asyncio.get_running_loop()
                ):
                    close_awaitable(result)
                    raise ProblemException(
                        Problem(
                            code="command.loop_mismatch",
                            title="Command event loop mismatch",
                            detail=(
                                f"Command '/{command.name}' returned a future attached to a "
                                "different event loop. Return a coroutine that can run on the "
                                "command manager's loop."
                            ),
                            operation=command.name,
                        )
                    )
                if inspect.isawaitable(result):
                    await result
            telemetry_activity("command.completed", command=name)
        except (ProblemException, CommandArgumentError, ValidationError) as error:
            self._report_call_failure(name, active_interaction, error)

    def _prepare_call(
        self,
        name: str,
        *,
        interaction: Interaction | None,
        available_commands: Iterable[Command] | None,
        async_dispatch: bool,
    ) -> tuple[Command | None, Interaction | None, CommandContext | None]:
        """Resolve command visibility and build the invocation context on the owner."""
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
            return None, active_interaction, None
        if command not in available:
            self._report_unavailable(name, active_interaction)
            return None, active_interaction, None
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
                execution_mode="async" if async_dispatch else "sync",
                _blocking_runner=self.run_blocking if async_dispatch else None,
            )
        return command, active_interaction, context

    def _report_call_failure(
        self,
        name: str,
        interaction: Interaction | None,
        error: ProblemException | CommandArgumentError | ValidationError,
    ) -> None:
        """Report one command failure through its interaction or preserve it for the caller."""
        if isinstance(error, ProblemException):
            if error.problem.code == "command.async_required":
                raise error
            log_problem(_LOGGER, error.problem, error)
            if interaction is None:
                raise error
            interaction.report(error.problem)
            return
        telemetry_error(
            "command.failed",
            error_type="command.invalid_arguments",
            exception=error,
            command=name,
        )
        if interaction is None:
            raise error
        interaction.report(
            Problem.from_exception(
                error,
                code="command.invalid_arguments",
                title="Invalid command arguments",
                detail=f"Invalid arguments for command '/{name}': {error}",
                severity="warning",
                operation=name,
            )
        )

    def _sync_runner(self) -> asyncio.Runner | None:
        """Return the manager-owned async runner outside an already running event loop."""
        if has_running_event_loop():
            return None
        with self._runner_lock:
            if self._closed:
                return None
            if self._async_runner is None:
                self._async_runner = asyncio.Runner()
                self._runner_thread = get_ident()
            elif self._runner_thread != get_ident():
                return None
            return self._async_runner

    def _async_required(self, name: str) -> ProblemException:
        """Build guidance for async callbacks that cannot use this sync caller's runner."""
        return ProblemException(
            Problem(
                code="command.async_required",
                title="Asynchronous command invocation required",
                detail=(
                    f"Command '/{name}' cannot run through this synchronous call. Use "
                    "await manager.call_async(...) or "
                    "await manager.handle_user_command_async(...) from the running event loop."
                ),
                operation=name,
            )
        )

    async def _run_sync(
        self,
        command: Command,
        arguments: str,
        context: CommandContext | None,
    ) -> Any:
        """Invoke a sync callback in a bounded worker and settle it after cancellation."""
        if context is not None and not context.interaction.worker_calls_routed:
            raise ProblemException(
                Problem(
                    code="command.context_relay_unavailable",
                    title="Command context relay unavailable",
                    detail=(
                        f"Synchronous command '/{command.name}' uses an interaction that does not "
                        "relay worker calls. Dispatch it through SessionController or provide an "
                        "owner-routed interaction."
                    ),
                    operation=command.name,
                )
            )
        return await self.run_blocking(
            command.invoke,
            arguments,
            context,
            _operation=f"command '/{command.name}'",
        )

    async def run_blocking(
        self,
        function: Callable[..., Any],
        *args: Any,
        _operation: str = "command operation",
        **kwargs: Any,
    ) -> Any:
        """Run synchronous work with this manager's bounded worker capacity.

        Args:
            function (Callable[..., Any]): Synchronous operation to invoke.
            *args (Any): Positional arguments passed to the operation.
            _operation (str): Operation label used in worker admission errors.
            **kwargs (Any): Keyword arguments passed to the operation.

        Returns:
            Any: Value returned by the synchronous operation.

        Raises:
            ProblemException: If the worker pool is closed or at capacity.
            asyncio.CancelledError: If the caller is cancelled after the worker settles.
            Exception: If the synchronous operation raises an exception.
        """
        if self._closed or not self._worker_slots.acquire(blocking=False):
            raise ProblemException(
                Problem(
                    code="command.dispatch_busy",
                    title="Command dispatch busy",
                    detail=(
                        "Synchronous command capacity is full; retry after active work settles."
                    ),
                    operation=_operation,
                )
            )
        call_context = copy_context()
        callback = partial(function, *args, **kwargs)
        try:
            pending = asyncio.get_running_loop().run_in_executor(
                self._executor,
                call_context.run,
                callback,
            )
        except Exception:
            self._worker_slots.release()
            raise
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError as error:
            settled = asyncio.create_task(asyncio.wait({pending}))
            while not settled.done():
                try:
                    await asyncio.shield(settled)
                except asyncio.CancelledError:
                    continue
            settled.result()
            if not pending.cancelled():
                failure = pending.exception()
                if failure is not None:
                    error.add_note(f"Command also stopped with {type(failure).__name__}.")
                else:
                    close_awaitable(pending.result())
            raise
        finally:
            self._worker_slots.release()

    def run_blocking_sync(
        self,
        function: Callable[..., Any],
        *args: Any,
        _operation: str = "command operation",
        **kwargs: Any,
    ) -> Any:
        """Run synchronous work with bounded admission for synchronous UI callbacks.

        Args:
            function (Callable[..., Any]): Synchronous operation to invoke.
            *args (Any): Positional arguments passed to the operation.
            _operation (str): Operation label used in worker admission errors.
            **kwargs (Any): Keyword arguments passed to the operation.

        Returns:
            Any: Value returned by the synchronous operation.

        Raises:
            ProblemException: If called from an event loop or the worker pool is closed or full.
            Exception: If the synchronous operation raises an exception.
        """
        if has_running_event_loop():
            raise ProblemException(
                Problem(
                    code="command.async_required",
                    title="Asynchronous operation required",
                    detail=(
                        "Synchronous file completion cannot run on an active event loop. "
                        "Use the asynchronous completion path."
                    ),
                    operation=_operation,
                )
            )
        if current_thread().name.startswith(f"{self._worker_thread_prefix}_"):
            return function(*args, **kwargs)
        if self._closed or not self._worker_slots.acquire(blocking=False):
            raise ProblemException(
                Problem(
                    code="command.dispatch_busy",
                    title="Command dispatch busy",
                    detail=(
                        "Synchronous command capacity is full; retry after active work settles."
                    ),
                    operation=_operation,
                )
            )
        call_context = copy_context()
        callback = partial(function, *args, **kwargs)
        try:
            pending = self._executor.submit(call_context.run, callback)
        except Exception:
            self._worker_slots.release()
            raise
        try:
            return pending.result()
        finally:
            self._worker_slots.release()

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
