"""Define passive file-tool models."""

from typing import Literal, TypedDict

from ..utils import BoundedTextContent


class FolderEntry(TypedDict):
    """Describe a listed folder entry."""

    path: str
    type: Literal["file", "folder"]


class FileContentResult(BoundedTextContent):
    """Describe bounded text loaded from a local file."""

    path: str


class CachedContentResult(BoundedTextContent):
    """Describe bounded text loaded from a cached artifact."""

    handle: str
    source: str
