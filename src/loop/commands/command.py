"""Define schema-backed user commands handled by the conversation loop."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any, Protocol, overload

from pydantic import BaseModel

from ..completion import CommandCompletion
from ..errors import Problem, ProblemException
from ..models import CommandCapability
from ..utils import callable_name, close_awaitable, is_async_callable, run_in_worker_async
from .context import CommandContext
from .models import CommandRegistrationError
from .utils import (
    get_command_arguments_model,
    parse_model_arguments,
    takes_command_context,
)

_COMMAND_ATTR = "__loop_command__"


@dataclass(frozen=True)
class Command:
    """Describe and, once registered, invoke a user command.

    Args:
        function (Callable[..., Any]): Python function invoked for the command.
        name (str | None): Slash-free public name, or ``None`` to derive it at registration.
        description (str | None): Display description, or ``None`` to derive it from the docstring.
        arguments_model (type[BaseModel] | None): Pydantic model used to validate arguments after
            registration, or ``None`` for a passive declaration.
        completion (CommandCompletion | None): Optional shell-like argument completion grammar.
        capabilities (frozenset[CommandCapability]): Runtime operations required to expose this
            command. Defaults to ``SESSION_IDLE`` so commands without an explicit live
            capability remain limited to an idle session.
    """

    function: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    arguments_model: type[BaseModel] | None = None
    completion: CommandCompletion | None = None
    capabilities: frozenset[CommandCapability] = frozenset({CommandCapability.SESSION_IDLE})

    def registered(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        completion: CommandCompletion | None = None,
        capabilities: frozenset[CommandCapability] | None = None,
    ) -> Command:
        """Return a registry-ready copy with resolved metadata and validation.

        Args:
            name (str | None): Container-specific public name, or ``None`` to inherit or derive it.
            description (str | None): Container-specific description, or ``None`` to inherit or
                derive it.
            completion (CommandCompletion | None): Container-specific completion grammar, or
                ``None`` to inherit or discover one.
            capabilities (frozenset[CommandCapability] | None): Context capabilities required by
                this registration, or ``None`` to inherit the declaration.

        Returns:
            Command: Immutable, fully resolved command for one manager.

        Raises:
            CommandRegistrationError: If the function lacks a description, its parameters cannot
                produce an argument schema.
        """
        resolved_name = self.name if name is None else name
        if resolved_name is None:
            resolved_name = callable_name(self.function)
        return replace(
            self,
            name=resolved_name,
            description=description or self.description or self._description_for(self.function),
            completion=(
                completion or self.completion or CommandCompletion.get_completion(self.function)
            ),
            capabilities=self.capabilities if capabilities is None else capabilities,
            arguments_model=get_command_arguments_model(self.function, resolved_name),
        )

    def call(self, arguments: str, context: CommandContext | None = None) -> Any:
        """Parse arguments and invoke the command function.

        Args:
            arguments (str): Shell-like positional and ``name=value`` argument text.
            context (CommandContext | None): Runtime context supplied to a context-aware command.

        Returns:
            Any: Value returned by the command function.

        Raises:
            ValidationError: If the supplied arguments do not match the command parameters.
            CommandArgumentError: If the argument syntax or parameter binding is invalid.
            ProblemException: If the registered function is asynchronous or returns an awaitable.
            ValueError: If the command is passive, or declares a context but none is provided.
        """
        if is_async_callable(self.function):
            raise ProblemException(
                Problem(
                    code="command.async_required",
                    title="Asynchronous command required",
                    detail=(
                        f"Command '{self.name}' is asynchronous; use "
                        "CommandManager.call_async() or "
                        "CommandManager.handle_user_command_async()."
                    ),
                    operation=self.name,
                )
            )
        result = self.invoke(arguments, context)
        if inspect.isawaitable(result):
            close_awaitable(result)
            raise ProblemException(
                Problem(
                    code="command.async_required",
                    title="Asynchronous command required",
                    detail=(
                        f"Command '{self.name}' returned an awaitable during sync dispatch; use "
                        "CommandManager.call_async() or "
                        "CommandManager.handle_user_command_async()."
                    ),
                    operation=self.name,
                )
            )
        return result

    async def call_async(self, arguments: str, context: CommandContext | None = None) -> Any:
        """Invoke a command asynchronously and await an asynchronous result.

        Context-free synchronous callbacks use the shared bounded worker. Context-aware
        synchronous callbacks require ``CommandManager`` so their interaction can be relayed to
        its owner.

        Args:
            arguments (str): Shell-like positional and ``name=value`` argument text.
            context (CommandContext | None): Runtime context supplied to a context-aware command.

        Returns:
            Any: Value returned by the command function after it settles.

        Raises:
            ValidationError: If supplied arguments do not match the command parameters.
            CommandArgumentError: If argument syntax or parameter binding is invalid.
            ProblemException: If a synchronous callback has a context that cannot be relayed, or
                returns a future attached to a different event loop.
            asyncio.CancelledError: If cancellation is requested while a synchronous callback is
                running, after that callback settles.
            RuntimeError: If bounded worker admission is full or unavailable.
            ValueError: If the command is passive, or declares a context but none is provided.
        """
        if is_async_callable(self.function):
            result = self.invoke(arguments, context)
        elif context is not None:
            raise ProblemException(
                Problem(
                    code="command.context_relay_unavailable",
                    title="Command context relay unavailable",
                    detail=(
                        f"Synchronous command '/{self.name}' requires CommandManager's "
                        "owner-routed context when called asynchronously."
                    ),
                    operation=self.name,
                )
            )
        else:
            result = await run_in_worker_async(self.invoke, arguments, context)
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
                        f"Command '/{self.name}' returned a future attached to a different "
                        "event loop."
                    ),
                    operation=self.name,
                )
            )
        return await result if inspect.isawaitable(result) else result

    def invoke(self, arguments: str, context: CommandContext | None = None) -> Any:
        """Validate and invoke the registered function without adapting its result.

        Use this method when a command manager needs the raw callback result to choose its own
        execution boundary. Most callers should use :meth:`call` or :meth:`call_async`.

        Args:
            arguments (str): Shell-like positional and ``name=value`` argument text.
            context (CommandContext | None): Runtime context supplied to a context-aware command.

        Returns:
            Any: Value returned by the registered function, possibly an awaitable.

        Raises:
            ValidationError: If the supplied arguments do not match the command parameters.
            CommandArgumentError: If the argument syntax or parameter binding is invalid.
            ValueError: If the command is passive, or declares a context but none is provided.
        """
        if self.arguments_model is None or self.name is None or self.description is None:
            raise ValueError("A command must be registered before it can be called.")
        values = parse_model_arguments(self.arguments_model, arguments).model_dump()
        if takes_command_context(self.function):
            if context is None:
                raise ValueError(f"Command '{self.name}' requires a CommandContext.")
            return self.function(context, **values)
        return self.function(**values)

    @staticmethod
    def _description_for(function: Callable[..., Any]) -> str:
        """Return a command description from its docstring summary."""
        docstring = inspect.getdoc(function)
        if not docstring:
            raise CommandRegistrationError(
                f"Command '{callable_name(function)}' must have a docstring."
            )
        return docstring.split("\n\n", maxsplit=1)[0].replace("\n", " ")

    @staticmethod
    def get_declaration(function: Callable[..., Any]) -> Command | None:
        """Return the passive command declaration attached to a function.

        Args:
            function (Callable[..., Any]): Function that may carry command metadata.

        Returns:
            Command | None: Attached passive command, or ``None``.
        """
        declaration = getattr(function, _COMMAND_ATTR, None)
        return declaration if isinstance(declaration, Command) else None

    @staticmethod
    def set_declaration(function: Callable[..., Any], declaration: Command) -> None:
        """Attach a passive command declaration to a function.

        Args:
            function (Callable[..., Any]): Function to attach command metadata to.
            declaration (Command): Command metadata to attach.
        """
        setattr(function, _COMMAND_ATTR, declaration)


@dataclass(frozen=True)
class CommandRegistration:
    """Configure one callable for registration in a command manager.

    Args:
        function (Callable[..., Any]): Callable to expose as a command.
        name (str | None): Container-specific public name, or ``None`` to inherit or derive it.
        description (str | None): Container-specific description, or ``None`` to inherit or derive
            it.
        completion (CommandCompletion | None): Container-specific completion grammar, or ``None``
            to inherit one declared on the function.
        capabilities (frozenset[CommandCapability] | None): Required runtime capabilities, or
            ``None`` to inherit those declared on the function.
    """

    function: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    completion: CommandCompletion | None = None
    capabilities: frozenset[CommandCapability] | None = None


class CommandsProvider(Protocol):
    """Provide command registrations for one application capability."""

    def get_commands(self) -> Iterable[CommandRegistration]:
        """Return command registrations contributed by this provider.

        Returns:
            Iterable[CommandRegistration]: Provider-owned command registrations.
        """


@overload
def command[CommandFunction: Callable[..., Any]](
    function: CommandFunction, /
) -> CommandFunction: ...


@overload
def command[CommandFunction: Callable[..., Any]](
    function: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    completion: CommandCompletion | None = None,
    capabilities: frozenset[CommandCapability] | None = None,
) -> Callable[[CommandFunction], CommandFunction]: ...


def command[CommandFunction: Callable[..., Any]](
    function: CommandFunction | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    completion: CommandCompletion | None = None,
    capabilities: frozenset[CommandCapability] | None = None,
) -> CommandFunction | Callable[[CommandFunction], CommandFunction]:
    """Declare a function as a user command without registering it.

    Args:
        function (CommandFunction | None): Function to declare, or ``None`` when decorating with
            options.
        name (str | None): Public name, defaulting to the function name at registration.
        description (str | None): Display description, defaulting to the docstring summary.
        completion (CommandCompletion | None): Optional argument completion grammar.
        capabilities (frozenset[CommandCapability] | None): Required runtime capabilities, or
            ``None`` to keep the command limited to an idle session.

    Returns:
        CommandFunction | Callable[[CommandFunction], CommandFunction]: Unchanged declared function
            or a decorator.

    Raises:
        ValueError: If the function already has a command declaration.
    """

    def _declare(target: CommandFunction) -> CommandFunction:
        if Command.get_declaration(target) is not None:
            raise ValueError("The function is already declared as a command.")
        Command.set_declaration(
            target,
            Command(
                target,
                name=name,
                description=description,
                completion=completion,
                capabilities=(
                    frozenset({CommandCapability.SESSION_IDLE})
                    if capabilities is None
                    else capabilities
                ),
            ),
        )
        return target

    return _declare(function) if function is not None else _declare
