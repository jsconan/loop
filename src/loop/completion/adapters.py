"""Define independently registered interactive completion adapters."""

from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import TYPE_CHECKING, Literal, Union, get_args, get_origin

from prompt_toolkit.document import Document

from ..utils.path import iter_visible_paths
from .models import CommandCompletion, CompletionMatch, CompletionProvider, CompletionValue

if TYPE_CHECKING:
    from ..commands.command import Command


class CompletionAdapter(ABC):
    """Provide candidates for one independently registered completion capability."""

    @property
    def front_markers(self) -> tuple[str, ...]:
        """Return symbols that can activate the adapter.

        Returns:
            tuple[str, ...]: Declared front-marker symbols.
        """
        return ()

    @property
    def keywords(self) -> tuple[str, ...]:
        """Return leading keywords that can activate the adapter.

        Returns:
            tuple[str, ...]: Declared activation keywords.
        """
        return ()

    @abstractmethod
    def match(self, document: Document) -> CompletionMatch | None:
        """Match the input at the cursor and describe its replacement region.

        Args:
            document (Document): Current editable input and cursor position.

        Returns:
            CompletionMatch | None: Active match, or ``None`` when the adapter is inactive.
        """

    @abstractmethod
    def complete(self, match: CompletionMatch) -> Iterable[CompletionValue]:
        """Return candidates for an active match.

        Args:
            match (CompletionMatch): Activation returned by ``match``.

        Returns:
            Iterable[CompletionValue]: Candidates for the active fragment.
        """


class MarkerCompletionAdapter(CompletionAdapter):
    """Complete token fragments introduced by a configurable marker.

    Args:
        marker (str): Single non-alphanumeric symbol that activates completion.
        provider (CompletionProvider): Lazy source of completion candidates.

    Raises:
        ValueError: If ``marker`` is not one non-alphanumeric, non-whitespace character.
    """

    _marker: str
    _provider: CompletionProvider

    def __init__(self, marker: str, provider: CompletionProvider) -> None:
        if len(marker) != 1 or marker.isalnum() or marker.isspace():
            raise ValueError("A completion marker must be one non-alphanumeric character.")
        self._marker = marker
        self._provider = provider

    @property
    def front_markers(self) -> tuple[str, ...]:
        """Return the configured activation marker.

        Returns:
            tuple[str, ...]: Single configured marker.
        """
        return (self._marker,)

    def match(self, document: Document) -> CompletionMatch | None:
        """Match a bounded marker token before the cursor.

        Args:
            document (Document): Current editable input and cursor position.

        Returns:
            CompletionMatch | None: Active marker fragment, or ``None`` outside a marker token.
        """
        before = document.text_before_cursor
        for index in range(len(before) - 1, -1, -1):
            character = before[index]
            if character.isspace():
                return None
            if character == self._marker:
                if index and before[index - 1].isalnum():
                    return None
                fragment = before[index + 1 :]
                return CompletionMatch(fragment, f"{self._marker}{fragment}", self._marker)
        return None

    def complete(self, match: CompletionMatch) -> Iterable[CompletionValue]:
        """Return values from the bound provider.

        Args:
            match (CompletionMatch): Active marker match.

        Returns:
            Iterable[CompletionValue]: Values currently exposed by the provider.
        """
        del match
        return self._provider()


class ProjectPathCompletionAdapter(MarkerCompletionAdapter):
    """Complete visible project paths after a marker and cache one prompt snapshot.

    Args:
        marker (str): Single symbol that activates path completion.
        working_directory (Path): Root used for project-relative paths.
    """

    _working_directory: Path
    _paths: tuple[CompletionValue, ...] | None

    def __init__(self, marker: str, working_directory: Path) -> None:
        self._working_directory = working_directory
        self._paths = None
        super().__init__(marker, self._path_values)

    def _path_values(self) -> tuple[CompletionValue, ...]:
        """Return a lazily indexed snapshot of visible project paths."""
        if self._paths is not None:
            return self._paths
        root = self._working_directory
        if not root.is_dir():
            self._paths = ()
            return self._paths
        values = []
        for path in iter_visible_paths(root, recursive=True):
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                relative += "/"
            values.append(CompletionValue(relative, "directory" if path.is_dir() else "file"))
        self._paths = tuple(values)
        return self._paths


class CommandCompletionAdapter(CompletionAdapter):
    """Complete registered slash commands and their declarative argument grammars.

    Args:
        commands (Callable[[], Iterable[Command]]): Lazy source of registered commands.
        marker (str): Symbol introducing command names. Defaults to ``/``.
        providers (Mapping[str, CompletionProvider] | None): Dynamic providers referenced by name
            from command completion grammars.

    Raises:
        ValueError: If ``marker`` is not one non-alphanumeric, non-whitespace character.
    """

    _commands: Callable[[], Iterable[Command]]
    _marker: str
    _providers: Mapping[str, CompletionProvider]

    def __init__(
        self,
        commands: Callable[[], Iterable[Command]],
        marker: str = "/",
        providers: Mapping[str, CompletionProvider] | None = None,
    ) -> None:
        if len(marker) != 1 or marker.isalnum() or marker.isspace():
            raise ValueError("A completion marker must be one non-alphanumeric character.")
        self._commands = commands
        self._marker = marker
        self._providers = providers or {}

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
        grammar = command.completion or self._schema_completion(command)
        if grammar is None:
            return None
        arguments = parts[1] if len(parts) == 2 else ""
        try:
            tokens = shlex.split(arguments)
        except ValueError:
            return None
        fragment = "" if before[-1].isspace() else (tokens.pop() if tokens else "")
        node = grammar
        for token in tokens:
            node = node.children.get(token) or node.next
            if node is None:
                return None
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
        if not isinstance(match.state, CommandCompletion):
            return ()
        values = [*match.state.values]
        provider = match.state.provider
        if isinstance(provider, str):
            provider = self._providers[provider]
        if provider is not None:
            values.extend(provider())
        return values

    @staticmethod
    def _schema_completion(command: Command) -> CommandCompletion | None:
        """Infer finite values when a command has exactly one argument."""
        fields = command.arguments_model.model_fields
        if len(fields) != 1:
            return None
        field = next(iter(fields.values()))
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
