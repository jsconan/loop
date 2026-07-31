"""Provide tools for accessing files and folders on the local disk."""

from pathlib import Path
from typing import Annotated, Literal, TypedDict

from pydantic import Field

from ..interaction import ToolContext
from ..tooling import tool_registry
from ..utils.path import is_path_ignored, iter_visible_paths


class FolderEntry(TypedDict):
    """Describe a listed folder entry."""

    path: str
    type: Literal["file", "folder"]


@tool_registry.tool
def list_folder(
    path: Annotated[str, Field(description="Path to the folder whose entries should be listed.")],
    entry_type: Annotated[
        Literal["files", "folders", "all"],
        Field(description="Type of entries to list."),
    ] = "all",
    recursive: Annotated[
        bool,
        Field(description="Whether to include entries in nested folders."),
    ] = False,
) -> list[FolderEntry] | str:
    """List selected, non-ignored entries in a folder on the local disk."""
    try:
        folder = Path(path).resolve()
        entries = iter_visible_paths(folder, recursive)
        return sorted(
            (
                {
                    "path": str(entry.relative_to(folder)) if recursive else entry.name,
                    "type": "folder" if entry.is_dir() else "file",
                }
                for entry in entries
                if (entry_type in ("all", "files") and entry.is_file())
                or (entry_type in ("all", "folders") and entry.is_dir())
            ),
            key=lambda entry: entry["path"],
        )
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error listing folder: {exc}"


@tool_registry.tool
def read_text_file(
    context: ToolContext,
    path: Annotated[str, Field(description="Path to the text file to read.")],
) -> str:
    """Read the contents of a text file from the local disk."""
    try:
        if is_path_ignored(path) and not context.confirm(
            f"Agent wants to read ignored file '{path}'. Proceed?"
        ):
            return "Read operation cancelled by user."

        content = Path(path).read_bytes()
        if not content:
            return f"File '{path}' is empty."
        if b"\0" in content:
            return f"Error reading file: File '{path}' appears to be binary."
        return content.decode("utf-8")
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error reading file: {exc}"


@tool_registry.tool
def write_text_file(
    context: ToolContext,
    path: Annotated[str, Field(description="Path to the text file to write.")],
    content: Annotated[str, Field(description="Content to write to the file.")],
) -> str:
    """Write content to a text file on the local disk."""
    if not context.confirm(f"Agent wants to write to file '{path}'. Proceed?"):
        return "Write operation cancelled by user."

    try:
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        return f"Successfully wrote to file '{path}'."
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error writing to file: {exc}"
