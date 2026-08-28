"""Complete registered commands and their declarative arguments."""

from __future__ import annotations

import shlex
from collections.abc import Callable, Iterable
from enum import Enum
from types import NoneType, UnionType
from typing import TYPE_CHECKING, Literal, Union, get_args, get_origin

from prompt_toolkit.document import Document

from ..models import (
    CommandCompletion,
    CompletionMatch,
    CompletionProvider,
    CompletionProviderRegistration,
    CompletionValue,
    SchemaCompletionProvider,
    SchemaCompletionProviderRegistration,
    SchemaCompletionState,
)
from .adapter import CompletionAdapter

if TYPE_CHECKING:
    from ...commands.command import Command


class CommandCompletionAdapter(CompletionAdapter):
    """Complete registered slash commands and their declarative argument grammars.

    Args:
        commands (Callable[[], Iterable[Command]]): Lazy source of registered commands.
        marker (str): Symbol introducing command names. Defaults to ``/``.
        providers (Iterable[object] | None): Capability providers inspected for an optional
            ``get_completion_providers()`` method during construction.

    Raises:
        ValueError: If ``marker`` is not one non-alphanumeric, non-whitespace character.
    """

    _commands: Callable[[], Iterable[Command]]
    _marker: str
    _providers: dict[str, CompletionProvider]
    _schema_providers: dict[str, SchemaCompletionProvider]

    def __init__(
        self,
        commands: Callable[[], Iterable[Command]],
        marker: str = "/",
        providers: Iterable[object] | None = None,
    ) -> None:
        if len(marker) != 1 or marker.isalnum() or marker.isspace():
            raise ValueError("A completion marker must be one non-alphanumeric character.")
        self._commands = commands
        self._marker = marker
        self._providers = {}
        self._schema_providers = {}
        self.register_providers(providers or ())

    def register(
        self,
        registration: CompletionProviderRegistration | SchemaCompletionProviderRegistration,
    ) -> None:
        """Register one named value or schema completion source.

        Args:
            registration (CompletionProviderRegistration | SchemaCompletionProviderRegistration):
                Named completion source to register.

        Raises:
            ValueError: If the provider name is empty or already registered for its source kind.
        """
        if not registration.name:
            raise ValueError("A completion provider name must not be empty.")
        registry = (
            self._providers
            if isinstance(registration, CompletionProviderRegistration)
            else self._schema_providers
        )
        if registration.name in registry:
            raise ValueError(f"Completion provider '{registration.name}' is already registered.")
        registry[registration.name] = registration.provider

    def register_provider(self, provider: object) -> None:
        """Register named completion sources exposed by one provider, if any.

        Args:
            provider (object): Capability provider optionally implementing
                ``get_completion_providers()``.
        """
        get_completion_providers = getattr(provider, "get_completion_providers", None)
        if get_completion_providers is None:
            return
        for registration in get_completion_providers():
            self.register(registration)

    def register_providers(self, providers: Iterable[object]) -> None:
        """Register completion sources exposed by multiple providers in order.

        Args:
            providers (Iterable[object]): Capability providers to inspect and register.
        """
        for provider in providers:
            self.register_provider(provider)

    @property
    def front_markers(self) -> tuple[str, ...]:
        """Return the command activation marker.

        Returns:
            tuple[str, ...]: Single configured marker.
        """
        return (self._marker,)

    @property
    def keywords(self) -> tuple[str, ...]:
        """Return the currently registered command names.

        Returns:
            tuple[str, ...]: Current command activation keywords.
        """
        return tuple(command.name for command in self._commands())

    def match(self, document: Document) -> CompletionMatch | None:
        """Match a command name or traverse its argument completion grammar.

        Args:
            document (Document): Current editable input and cursor position.

        Returns:
            CompletionMatch | None: Command fragment and grammar state, or ``None``.
        """
        before = document.text_before_cursor
        if not before.startswith(self._marker):
            return None
        body = before[len(self._marker) :]
        commands = tuple(self._commands())
        if not any(character.isspace() for character in body):
            values = tuple(
                CompletionValue(command.name, command.description) for command in commands
            )
            return CompletionMatch(body, before, self._marker, values)

        parts = body.split(maxsplit=1)
        command = next((item for item in commands if item.name == parts[0]), None)
        if command is None:
            return None
        arguments = parts[1] if len(parts) == 2 else ""
        if command.completion is None:
            return self._match_model_arguments(command.arguments_model, arguments, before)
        grammar = command.completion
        try:
            tokens = shlex.split(arguments)
        except ValueError:
            return None
        fragment = "" if before[-1].isspace() else (tokens.pop() if tokens else "")
        node = grammar
        consumed = []
        for token in tokens:
            consumed.append(token)
            node = node.children.get(token) or node.next
            if node is None:
                return None
            if node.schema_provider is not None:
                provider = node.schema_provider
                if isinstance(provider, str):
                    provider = self._schema_providers[provider]
                model = provider(tuple(consumed))
                if model is None:
                    return None
                parts = arguments.split(maxsplit=1)
                remaining = parts[1] if len(consumed) == 1 and len(parts) == 2 else ""
                return self._match_model_arguments(model, remaining, before)
        return CompletionMatch(fragment, fragment, state=node)

    def complete(self, match: CompletionMatch) -> Iterable[CompletionValue]:
        """Return static and dynamic values from the matched grammar state.

        Args:
            match (CompletionMatch): Active command or argument match.

        Returns:
            Iterable[CompletionValue]: Available command names or argument values.

        Raises:
            KeyError: If a grammar references an unregistered named provider.
        """
        if isinstance(match.state, tuple):
            return match.state
        if isinstance(match.state, SchemaCompletionState):
            return tuple(
                CompletionValue(f"{match.state.prefix}{value.value}", value.description)
                for value in self._grammar_values(match.state.grammar)
            )
        if not isinstance(match.state, CommandCompletion):
            return ()
        return self._grammar_values(match.state)

    def _grammar_values(self, grammar: CommandCompletion) -> tuple[CompletionValue, ...]:
        """Resolve static and dynamic values for one grammar node."""
        values = [*grammar.values]
        provider = grammar.provider
        if isinstance(provider, str):
            provider = self._providers[provider]
        if provider is not None:
            values.extend(provider())
        return tuple(values)

    def _match_model_arguments(self, model, arguments: str, before: str) -> CompletionMatch | None:
        """Match positional or named input against a Pydantic model schema."""
        try:
            tokens = shlex.split(arguments)
        except ValueError:
            return None
        fragment = "" if before[-1].isspace() else (tokens.pop() if tokens else "")
        fields = model.model_fields
        assigned = set()
        for token in tokens:
            name, separator, _ = token.partition("=")
            if separator and name.isidentifier():
                if name not in fields or name in assigned:
                    return None
            else:
                name = next((field_name for field_name in fields if field_name not in assigned), "")
                if not name:
                    return None
            assigned.add(name)

        name, separator, value_fragment = fragment.partition("=")
        if separator and name.isidentifier():
            if name not in fields or name in assigned:
                return None
            grammar = self._field_completion(fields[name])
            if grammar is None:
                return None
            return CompletionMatch(
                value_fragment, fragment, state=SchemaCompletionState(grammar, f"{name}=")
            )

        remaining = [field_name for field_name in fields if field_name not in assigned]
        if not remaining:
            return None
        values = [
            CompletionValue(f"{field_name}=", fields[field_name].description or "parameter")
            for field_name in remaining
        ]
        grammar = self._field_completion(fields[remaining[0]])
        if grammar is not None:
            values.extend(self._grammar_values(grammar))
        return CompletionMatch(fragment, fragment, state=tuple(values))

    @staticmethod
    def _field_completion(field) -> CommandCompletion | None:
        """Return declared or inferred finite completion for one model field."""
        declared = next(
            (metadata for metadata in field.metadata if isinstance(metadata, CommandCompletion)),
            None,
        )
        if declared is not None:
            return declared
        values = CommandCompletionAdapter._annotation_values(field.annotation)
        return CommandCompletion(values=values) if values else None

    @staticmethod
    def _annotation_values(annotation: object) -> tuple[CompletionValue, ...]:
        """Extract enum and literal strings from a field annotation."""
        origin = get_origin(annotation)
        if origin is Literal:
            return tuple(CompletionValue(str(value)) for value in get_args(annotation))
        if origin in {UnionType, Union}:
            return tuple(
                value
                for member in get_args(annotation)
                if member is not NoneType
                for value in CommandCompletionAdapter._annotation_values(member)
            )
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return tuple(CompletionValue(str(member.value)) for member in annotation)
        if annotation is bool:
            return (CompletionValue("true"), CompletionValue("false"))
        return ()
