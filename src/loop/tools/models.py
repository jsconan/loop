"""Define passive file-tool models."""

from typing import Literal, TypedDict


class FolderEntry(TypedDict):
    """Describe a listed folder entry."""

    path: str
    type: Literal["file", "folder"]
