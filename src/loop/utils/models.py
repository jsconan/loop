"""Define passive path utility models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from pathspec import GitIgnoreSpec

type IgnoreRule = tuple[Path, GitIgnoreSpec]
type IgnoreRules = dict[str, list[IgnoreRule]]


@dataclass(frozen=True)
class ChoiceItem:
    """Describe one selectable prompt option.

    Args:
        index (str): Text entered to select the option.
        value (object): Value returned when the option is selected.
        name (str): Human-readable option name accepted as direct input.
        description (str | None): Optional supplementary description of the option.
            Defaults to ``None``.
    """

    index: str
    value: object
    name: str
    description: str | None = None

    @property
    def label(self) -> str:
        """Return a human-readable label for the option."""
        return f"({self.index}) {self.name}"


class BoundedTextContent(TypedDict):
    """Describe one bounded, resumable portion of textual content."""

    content: str
    size_bytes: int
    start_byte: int
    end_byte: int
    included_bytes: int
    truncated: bool
    truncation_reason: NotRequired[Literal["bytes", "lines", "line_too_long"]]
    start_line: NotRequired[int]
    end_line: NotRequired[int]
    next_start_byte: NotRequired[int]
    next_start_line: NotRequired[int]


class CachedContentMetadata(TypedDict):
    """Describe persisted information used to recover an expired artifact."""

    source: str
    reloadable: bool


class TextSearchContext(TypedDict):
    """Describe one neighboring source line around a text match."""

    line: int
    text: str


class TextSearchMatch(TypedDict):
    """Describe one matching source line returned by text search."""

    path: str
    line: int
    column: int
    text: str
    context: NotRequired[list[TextSearchContext]]
