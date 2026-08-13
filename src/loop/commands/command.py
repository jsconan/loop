"""Define schema-backed user commands handled by the conversation loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from ..completion import CommandCompletion
from ..context import CommandContext
from .utils import parse_model_arguments, takes_command_context


@dataclass(frozen=True)
class Command:
    """Represent a command with its display metadata and argument validator.

    Args:
        name (str): Slash-free public command name.
        description (str): Short description shown by command discovery.
        function (Callable[..., None]): Python function invoked for the command.
        arguments_model (type[BaseModel]): Pydantic model used to validate arguments.
        completion (CommandCompletion | None): Optional shell-like argument completion grammar.
    """

    name: str
    description: str
    function: Callable[..., None]
    arguments_model: type[BaseModel]
    completion: CommandCompletion | None = None

    def call(self, arguments: str, context: CommandContext | None = None) -> None:
        """Parse arguments and invoke the command function.

        Args:
            arguments (str): Shell-like positional and ``name=value`` argument text.
            context (CommandContext | None): Runtime context supplied to a context-aware command.

        Raises:
            ValidationError: If the supplied arguments do not match the command parameters.
            CommandArgumentError: If the argument syntax or parameter binding is invalid.
            ValueError: If the command declares a context but none is provided.
        """
        values = parse_model_arguments(self.arguments_model, arguments).model_dump()
        if takes_command_context(self.function):
            if context is None:
                raise ValueError(f"Command '{self.name}' requires a CommandContext.")
            self.function(context, **values)
        else:
            self.function(**values)
