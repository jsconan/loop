"""Define interactive completion models and command grammar metadata."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

COMPLETION_ATTRIBUTE = "__completion__"


@dataclass(frozen=True)
class CompletionValue:
    """Describe one insertable completion value.

    Args:
        value (str): Text inserted into the input.
        description (str): Optional explanation displayed beside the value.
    """

    value: str
    description: str = ""


type CompletionProvider = Callable[[], Iterable[CompletionValue]]


@dataclass(frozen=True)
class CompletionMatch:
    """Describe an adapter activation at the current cursor.

    Args:
        fragment (str): Candidate fragment used for matching and ranking.
        replaced (str): Text before the cursor replaced by a selected candidate.
        prefix (str): Text prepended to each inserted and displayed candidate.
        state (object | None): Adapter-private state captured while matching.
    """

    fragment: str
    replaced: str
    prefix: str = ""
    state: object | None = None


@dataclass(frozen=True)
class CommandCompletion:
    """Describe one level of a shell-like command completion grammar.

    Args:
        values (tuple[CompletionValue, ...]): Static values accepted at this level.
        provider (CompletionProvider | str | None): Runtime value provider or adapter-local
            provider name for this level.
        children (Mapping[str, CommandCompletion]): Next completion level selected by a value.
        next (CommandCompletion | None): Unconditional next completion level.
    """

    values: tuple[CompletionValue, ...] = ()
    provider: CompletionProvider | str | None = None
    children: Mapping[str, CommandCompletion] = field(default_factory=dict)
    next: CommandCompletion | None = None


@dataclass(frozen=True)
class SchemaCompletionState:
    """Describe schema-derived completion state for one command field.

    Args:
        grammar (CommandCompletion): Completion grammar for the active field.
        prefix (str): Named-argument prefix prepended to completed values.
    """

    grammar: CommandCompletion
    prefix: str = ""
