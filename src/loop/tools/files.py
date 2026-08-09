"""Provide tools for accessing files and folders on the local disk."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from ..context import ToolContext
from ..permissions import Capability, PermissionRequest
from .models import FolderEntry
from ..tooling import tool_registry
from ..utils import format_content_preview, is_path_ignored, iter_visible_paths


def _file_permission(capability: Capability):
    """Return a resolver for one normalized filesystem resource."""

    def _resolve(arguments: dict[str, object]) -> tuple[PermissionRequest, ...]:
        path = Path(str(arguments["path"]))
        if capability is Capability.FILESYSTEM_WRITE and not path.exists():
            parent = path.parent.resolve()
            resource = str(parent / path.name)
        else:
            resource = str(path.resolve())
        return (PermissionRequest(tool_name="", capability=capability, resource=resource),)

    return _resolve


@tool_registry.tool(
    capabilities={Capability.FILESYSTEM_READ},
    permission_resolver=_file_permission(Capability.FILESYSTEM_READ),
)
def list_folder(
    context: ToolContext,
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
        if is_path_ignored(folder):
            return f"Error listing folder: Path '{path}' is ignored."
        entries = iter_visible_paths(folder, recursive)
        result = sorted(
            (
                FolderEntry(
                    path=str(entry.relative_to(folder)) if recursive else entry.name,
                    type="folder" if entry.is_dir() else "file",
                )
                for entry in entries
                if (entry_type in ("all", "files") and entry.is_file())
                or (entry_type in ("all", "folders") and entry.is_dir())
            ),
            key=lambda entry: entry["path"],
        )
        context.observe_directory(folder)
        return result
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error listing folder: {exc}"


@tool_registry.tool(
    capabilities={Capability.FILESYSTEM_READ},
    permission_resolver=_file_permission(Capability.FILESYSTEM_READ),
)
def read_text_file(
    context: ToolContext,
    path: Annotated[str, Field(description="Path to the text file to read.")],
) -> str:
    """Read the contents of a text file from the local disk."""
    try:
        content = Path(path).read_bytes()
        context.observe_file(path)
        if not content:
            return f"File '{path}' is empty."
        if b"\0" in content:
            return f"Error reading file: File '{path}' appears to be binary."
        return content.decode("utf-8")
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error reading file: {exc}"


@tool_registry.tool(
    capabilities={Capability.FILESYSTEM_WRITE},
    permission_resolver=_file_permission(Capability.FILESYSTEM_WRITE),
)
def write_text_file(
    context: ToolContext,
    path: Annotated[str, Field(description="Path to the text file to write.")],
    content: Annotated[str, Field(description="Content to write to the file.")],
) -> str:
    """Write content to a file on the local disk."""
    preview = format_content_preview(content)

    context.interaction.info(f"Content to write to '{path}':\n{preview}")

    try:
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        context.observe_file(path)
        context.invalidate_instructions(path)
        return f"Successfully wrote to file '{path}'."
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error writing to file: {exc}"
