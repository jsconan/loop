"""Define schema-backed user commands handled by the conversation loop."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Protocol, overload

from pydantic import BaseModel

from ..completion import CommandCompletion
from ..utils import callable_name
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
        function (Callable[..., None]): Python function invoked for the command.
        name (str | None): Slash-free public name, or ``None`` to derive it at registration.
        description (str | None): Display description, or ``None`` to derive it from the docstring.
        arguments_model (type[BaseModel] | None): Pydantic model used to validate arguments after
            registration, or ``None`` for a passive declaration.
        completion (CommandCompletion | None): Optional shell-like argument completion grammar.
    """

    function: Callable[..., None]
    name: str | None = None
    description: str | None = None
    arguments_model: type[BaseModel] | None = None
    completion: CommandCompletion | None = None

    def registered(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        completion: CommandCompletion | None = None,
    ) -> Command:
        """Return a registry-ready copy with resolved metadata and validation.

        Args:
            name (str | None): Container-specific public name, or ``None`` to inherit or derive it.
            description (str | None): Container-specific description, or ``None`` to inherit or
                derive it.
            completion (CommandCompletion | None): Container-specific completion grammar, or
                ``None`` to inherit or discover one.

        Returns:
            Command: Immutable, fully resolved command for one manager.

        Raises:
            CommandRegistrationError: If the function lacks a description or its parameters
                cannot produce an argument schema.
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
            arguments_model=get_command_arguments_model(self.function, resolved_name),
        )

    def call(self, arguments: str, context: CommandContext | None = None) -> None:
        """Parse arguments and invoke the command function.

        Args:
            arguments (str): Shell-like positional and ``name=value`` argument text.
            context (CommandContext | None): Runtime context supplied to a context-aware command.

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
            self.function(context, **values)
        else:
            self.function(**values)

    @staticmethod
    def _description_for(function: Callable[..., None]) -> str:
        """Return a command description from its docstring summary."""
        docstring = inspect.getdoc(function)
        if not docstring:
            raise CommandRegistrationError(
                f"Command '{callable_name(function)}' must have a docstring."
            )
        return docstring.split("\n\n", maxsplit=1)[0].replace("\n", " ")

    @staticmethod
    def get_declaration(function: Callable[..., None]) -> Command | None:
        """Return the passive command declaration attached to a function.

        Args:
            function (Callable[..., None]): Function that may carry command metadata.

        Returns:
            Command | None: Attached passive command, or ``None``.
        """
        declaration = getattr(function, _COMMAND_ATTR, None)
        return declaration if isinstance(declaration, Command) else None

    @staticmethod
    def set_declaration(function: Callable[..., None], declaration: Command) -> None:
        """Attach a passive command declaration to a function.

        Args:
            function (Callable[..., None]): Function to attach command metadata to.
            declaration (Command): Command metadata to attach.
        """
        setattr(function, _COMMAND_ATTR, declaration)


@dataclass(frozen=True)
class CommandRegistration:
    """Configure one callable for registration in a command manager.

    Args:
        function (Callable[..., None]): Callable to expose as a command.
        name (str | None): Container-specific public name, or ``None`` to inherit or derive it.
        description (str | None): Container-specific description, or ``None`` to inherit or derive
            it.
        completion (CommandCompletion | None): Container-specific completion grammar, or ``None``
            to inherit one declared on the function.
    """

    function: Callable[..., None]
    name: str | None = None
    description: str | None = None
    completion: CommandCompletion | None = None


class CommandsProvider(Protocol):
    """Provide command registrations for one application capability."""

    def get_commands(self) -> Iterable[CommandRegistration]:
        """Return command registrations contributed by this provider.

        Returns:
            Iterable[CommandRegistration]: Provider-owned command registrations.
        """


@overload
def command[CommandFunction: Callable[..., None]](
    function: CommandFunction, /
) -> CommandFunction: ...


@overload
def command[CommandFunction: Callable[..., None]](
    function: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    completion: CommandCompletion | None = None,
) -> Callable[[CommandFunction], CommandFunction]: ...


def command[CommandFunction: Callable[..., None]](
    function: CommandFunction | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    completion: CommandCompletion | None = None,
) -> CommandFunction | Callable[[CommandFunction], CommandFunction]:
    """Declare a function as a user command without registering it.

    Args:
        function (CommandFunction | None): Function to declare, or ``None`` when decorating with
            options.
        name (str | None): Public name, defaulting to the function name at registration.
        description (str | None): Display description, defaulting to the docstring summary.
        completion (CommandCompletion | None): Optional argument completion grammar.

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
            Command(target, name=name, description=description, completion=completion),
        )
        return target

    return _declare(function) if function is not None else _declare
