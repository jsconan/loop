"""Provide utility functions for command registration and dispatch."""

import inspect
from collections.abc import Callable
from typing import get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

from ..context import CommandContext


class CommandRegistrationError(ValueError):
    """Indicate that a Python function cannot be registered as a command."""


def get_command_arguments_model(
    function: Callable[..., None],
    command_name: str,
) -> type[BaseModel]:
    """Build a validating arguments model from a command signature.

    Args:
        function (Callable[..., None]): Function whose parameters define the command schema.
        command_name (str): Slash-free command name used in model and error messages.

    Returns:
        type[BaseModel]: Pydantic model that validates deserialized command arguments.

    Raises:
        CommandRegistrationError: If a parameter cannot be represented by a schema or
            ``CommandContext`` is declared anywhere except first.
    """
    hints = get_type_hints(function, include_extras=True)
    parameters = list(inspect.signature(function).parameters.values())
    if parameters and hints.get(parameters[0].name) is CommandContext:
        parameters = parameters[1:]
    if any(hints.get(parameter.name) is CommandContext for parameter in parameters):
        raise CommandRegistrationError(
            f"Command '{command_name}' must declare CommandContext as its first parameter."
        )

    fields = {}
    for parameter in parameters:
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise CommandRegistrationError(
                f"Command '{command_name}' has unsupported parameter '{parameter.name}'."
            )
        if parameter.name not in hints:
            raise CommandRegistrationError(
                f"Command '{command_name}' parameter '{parameter.name}' needs a type annotation."
            )
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (hints[parameter.name], default)

    model_name = "".join(part.title() for part in command_name.split("_"))
    return create_model(f"{model_name}Arguments", __config__=ConfigDict(extra="forbid"), **fields)


def takes_command_context(function: Callable[..., None]) -> bool:
    """Determine whether a function requests an injected command context.

    Args:
        function (Callable[..., None]): Function whose first parameter may request context.

    Returns:
        bool: Whether the first parameter is annotated as ``CommandContext``.
    """
    parameters = list(inspect.signature(function).parameters.values())
    if not parameters:
        return False
    hints = get_type_hints(function, include_extras=True)
    return hints.get(parameters[0].name) is CommandContext
