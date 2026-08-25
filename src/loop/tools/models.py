"""Define passive file-tool models."""

from typing import Literal, NotRequired, TypedDict

from ..utils import BoundedTextContent, TextSearchMatch


class FolderEntry(TypedDict):
    """Describe a listed folder entry."""

    path: str
    type: Literal["file", "folder"]


class FileContentResult(BoundedTextContent):
    """Describe bounded text loaded from a local file."""

    path: str


class TextSearchResult(TypedDict):
    """Describe bounded matches from a local text search."""

    matches: list[TextSearchMatch]
    truncated: bool


class CachedContentResult(BoundedTextContent):
    """Describe bounded text loaded from a cached artifact."""

    handle: str
    source: str
    next_cursor: NotRequired[str]
