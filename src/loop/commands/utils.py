"""Provide utility functions for command registration and dispatch."""

import inspect
import json
import shlex
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model

from ..utils import callable_hints
from .context import CommandContext
from .models import CommandArgumentError, CommandRegistrationError, CommandRemainder


def parse_model_arguments(
    model: type[BaseModel],
    arguments: str | list[str] | tuple[str, ...],
) -> BaseModel:
    """Bind shell-like arguments to a Pydantic model and validate them.

    Args:
        model (type[BaseModel]): Model whose ordered fields define positional and named arguments.
        arguments (str | list[str] | tuple[str, ...]): Shell-like text to tokenize, or tokens
            already parsed by an outer command.

    Returns:
        BaseModel: Validated instance of ``model``.

    Raises:
        CommandArgumentError: If tokenization or field binding fails.
        ValidationError: If bound values fail model validation.
    """
    fields = model.model_fields
    remainder = next(
        (
            name
            for name, field in fields.items()
            if any(isinstance(item, CommandRemainder) for item in field.metadata)
        ),
        None,
    )
    values = {}
    remainder_values = []
    for token in _get_argument_tokens(arguments):
        explicit_name, separator, raw_value = token.partition("=")
        if separator and explicit_name.isidentifier():
            if explicit_name in fields and explicit_name != remainder:
                name = explicit_name
                if name in values:
                    raise CommandArgumentError(f"Parameter '{name}' was supplied more than once.")
            elif remainder is not None:
                remainder_values.append(token)
                continue
            else:
                raise CommandArgumentError(f"Unknown parameter '{explicit_name}'.")
        else:
            raw_value = token
            name = next(
                (
                    field_name
                    for field_name in fields
                    if field_name not in values and field_name != remainder
                ),
                remainder or "",
            )
            if name == remainder:
                remainder_values.append(token)
                continue
            if not name:
                raise CommandArgumentError("Too many positional arguments.")
        values[name] = _decode_model_argument(fields[name], raw_value)
    if remainder is not None:
        values[remainder] = remainder_values
    return model.model_validate(values)


def _get_argument_tokens(arguments: str | list[str] | tuple[str, ...]) -> list[str]:
    """Tokenize shell-like argument text or return pre-parsed tokens."""
    if isinstance(arguments, str):
        try:
            return shlex.split(arguments)
        except ValueError as exc:
            raise CommandArgumentError(f"Could not parse arguments: {exc}") from exc

    return arguments


def _decode_model_argument(field, raw_value: str) -> object:
    """Decode one JSON-shaped value while preserving valid string input."""
    try:
        decoded = json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value

    adapter = TypeAdapter(field.rebuild_annotation())
    try:
        adapter.validate_python(decoded)
    except ValidationError:
        try:
            adapter.validate_python(raw_value)
        except ValidationError:
            return decoded
        return raw_value
    return decoded


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
    hints = callable_hints(function)
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
    model = create_model(f"{model_name}Arguments", __config__=ConfigDict(extra="forbid"), **fields)
    remainder_fields = [
        name
        for name, field in model.model_fields.items()
        if any(isinstance(item, CommandRemainder) for item in field.metadata)
    ]
    if remainder_fields and (
        len(remainder_fields) != 1
        or remainder_fields[0] != next(reversed(model.model_fields))
        or model.model_fields[remainder_fields[0]].annotation != tuple[str, ...]
    ):
        raise CommandRegistrationError(
            f"Command '{command_name}' remainder must be its final tuple[str, ...] parameter."
        )
    return model


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
    hints = callable_hints(function)
    return hints.get(parameters[0].name) is CommandContext
