"""Provide tools for accessing files and folders on the local disk."""

from pathlib import Path
from typing import Annotated

from pydantic import Field

from ..interaction import ToolContext
from ..tooling import tool_registry


@tool_registry.tool
def list_files(
    path: Annotated[str, Field(description="Path to the folder whose files should be listed.")],
) -> list[str] | str:
    """List the files directly contained in a folder on the local disk."""
    try:
        return sorted(entry.name for entry in Path(path).iterdir() if entry.is_file())
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error listing files: {exc}"


@tool_registry.tool
def list_folders(
    path: Annotated[
        str, Field(description="Path to the folder whose subfolders should be listed.")
    ],
) -> list[str] | str:
    """List the folders directly contained in a folder on the local disk."""
    try:
        return sorted(entry.name for entry in Path(path).iterdir() if entry.is_dir())
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error listing folders: {exc}"


@tool_registry.tool
def read_text_file(
    path: Annotated[str, Field(description="Path to the text file to read.")],
) -> str:
    """Read the contents of a text file from the local disk."""
    try:
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
        return "Write operation cancelled."

    try:
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        return f"Successfully wrote to file '{path}'."
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error writing to file: {exc}"
