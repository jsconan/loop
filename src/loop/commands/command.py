"""Define schema-backed user commands handled by the conversation loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from ..completion import CommandCompletion
from ..context import CommandContext
from .utils import takes_command_context


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
        """Deserialize arguments and invoke the command function.

        Args:
            arguments (str): Raw argument text, or a JSON object for multiple parameters.
            context (CommandContext | None): Runtime context supplied to a context-aware command.

        Raises:
            ValidationError: If the supplied arguments do not match the command parameters.
            ValueError: If the command declares a context but none is provided.
        """
        values = self._validate_arguments(arguments).model_dump()
        if takes_command_context(self.function):
            if context is None:
                raise ValueError(f"Command '{self.name}' requires a CommandContext.")
            self.function(context, **values)
        else:
            self.function(**values)

    def _validate_arguments(self, arguments: str) -> BaseModel:
        """Validate JSON arguments or coerce raw text for one declared argument."""
        try:
            return self.arguments_model.model_validate_json(arguments or "{}")
        except ValidationError:
            fields = self.arguments_model.model_fields
            if len(fields) != 1:
                raise
            argument_name = next(iter(fields))
            return self.arguments_model.model_validate({argument_name: arguments})
