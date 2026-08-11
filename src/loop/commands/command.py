"""Define schema-backed user commands handled by the conversation loop."""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, TypeAdapter, ValidationError

from ..completion import CommandCompletion
from ..context import CommandContext
from .utils import takes_command_context


class CommandArgumentError(ValueError):
    """Indicate invalid slash-command argument syntax or binding."""


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
        values = self._validate_arguments(arguments).model_dump()
        if takes_command_context(self.function):
            if context is None:
                raise ValueError(f"Command '{self.name}' requires a CommandContext.")
            self.function(context, **values)
        else:
            self.function(**values)

    def _validate_arguments(self, arguments: str) -> BaseModel:
        """Bind shell-like tokens and validate their independently decoded values."""
        try:
            tokens = shlex.split(arguments)
        except ValueError as exc:
            raise CommandArgumentError(f"Could not parse arguments: {exc}") from exc

        fields = self.arguments_model.model_fields
        values = {}
        for token in tokens:
            name, separator, raw_value = token.partition("=")
            if separator and name.isidentifier():
                if name not in fields:
                    raise CommandArgumentError(f"Unknown parameter '{name}'.")
                if name in values:
                    raise CommandArgumentError(f"Parameter '{name}' was supplied more than once.")
            else:
                unbound = next(
                    (field_name for field_name in fields if field_name not in values),
                    None,
                )
                if unbound is None:
                    raise CommandArgumentError("Too many positional arguments.")
                name = unbound
                raw_value = token
            values[name] = self._decode_argument(name, raw_value)
        return self.arguments_model.model_validate(values)

    def _decode_argument(self, name: str, raw_value: str) -> object:
        """Decode one JSON-shaped value while preserving valid string input."""
        try:
            decoded = json.loads(raw_value)
        except json.JSONDecodeError:
            return raw_value

        adapter = TypeAdapter(self.arguments_model.model_fields[name].rebuild_annotation())
        try:
            adapter.validate_python(decoded)
        except ValidationError:
            try:
                adapter.validate_python(raw_value)
            except ValidationError:
                return decoded
            return raw_value
        return decoded
