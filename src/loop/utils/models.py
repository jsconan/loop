"""Define passive path utility models."""

from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from pathspec import GitIgnoreSpec

type IgnoreRule = tuple[Path, GitIgnoreSpec]
type IgnoreRules = dict[str, list[IgnoreRule]]


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
