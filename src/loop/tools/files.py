"""Provide tools for accessing files and folders on the local disk."""

import shutil
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from .. import constants
from ..context import ToolContext
from ..permissions import Capability, PermissionRequest
from ..tooling import tool_registry
from ..utils import (
    format_content_diff,
    format_content_preview,
    is_path_ignored,
    iter_visible_paths,
    read_bounded_text,
)
from .models import FileContentResult, FolderEntry


def _write_preview(path: Path, content: str) -> str:
    """Return the write preview displayed in a permission prompt."""
    if not path.exists() or path.stat().st_size == 0:
        return f"Proposed content:\n{format_content_preview(content)}"
    try:
        existing = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return (
            "Existing content could not be previewed; proposed content:\n"
            f"{format_content_preview(content)}"
        )
    return f"Proposed changes:\n{format_content_diff(existing, content, str(path))}"


def _file_permission(capability: Capability):
    """Return a resolver for one normalized filesystem resource."""

    def _resolve(arguments: dict[str, object]) -> tuple[PermissionRequest, ...]:
        path = Path(str(arguments["path"]))
        if capability is Capability.FILESYSTEM_WRITE and not path.exists():
            parent = path.parent.resolve()
            resource = str(parent / path.name)
        else:
            resource = str(path.resolve())
        reason = None
        if capability is Capability.FILESYSTEM_WRITE:
            reason = _write_preview(path, str(arguments["content"]))
        return (
            PermissionRequest(
                tool_name="", capability=capability, resource=resource, reason=reason
            ),
        )

    return _resolve


def _delete_permission(arguments: dict[str, object]) -> tuple[PermissionRequest, ...]:
    """Return the permanent-deletion request for the supplied path."""
    path = Path(str(arguments["path"]))
    kind = _deletion_kind(path)
    if kind == "symbolic link":
        reason = "Permanently delete this symbolic link; its target will not be deleted."
    elif kind == "folder":
        reason = "Permanently delete this folder and all of its contents."
    elif kind == "file":
        reason = "Permanently delete this file."
    else:
        reason = "Deletion supports files, symbolic links, and folders only."
    return (
        PermissionRequest(
            tool_name="",
            capability=Capability.FILESYSTEM_DELETE,
            resource=str(path.absolute()),
            reason=reason,
        ),
    )


def _deletion_kind(path: Path) -> Literal["file", "symbolic link", "folder"] | None:
    """Return the supported deletion kind for a path, when any."""
    if path.is_symlink():
        return "symbolic link"
    if path.is_dir():
        return "folder"
    if path.is_file():
        return "file"
    return None


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
    start_line: Annotated[
        int,
        Field(description="One-based starting line.", ge=1),
    ] = 1,
    max_lines: Annotated[
        int | None,
        Field(
            description="Optional line ceiling; the first reached line or byte limit wins.", ge=1
        ),
    ] = None,
    max_bytes: Annotated[
        int,
        Field(
            description="Maximum UTF-8 bytes returned, capped by the application hard limit.",
            ge=1,
            le=constants.MAX_TOOL_CONTENT_BYTES,
        ),
    ] = constants.MAX_TOOL_CONTENT_BYTES,
) -> FileContentResult | str:
    """Read a bounded, resumable portion of a UTF-8 text file."""
    try:
        file_path = Path(path)
        if file_path.stat().st_size == 0:
            context.observe_file(path)
            return f"File '{path}' is empty."
        result = FileContentResult(
            path=path,
            **read_bounded_text(
                file_path,
                start_line=start_line,
                max_lines=max_lines,
                max_bytes=max_bytes,
                preserve_line_boundaries=True,
            ),
        )
        context.observe_file(path)
        return result
    except Exception as exc:  # pylint: disable=broad-except
        if str(exc) == "Content appears to be binary.":
            return f"Error reading file: File '{path}' appears to be binary."
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
    try:
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        context.observe_file(path)
        context.invalidate_instructions(path)
        return f"Successfully wrote to file '{path}'."
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error writing to file: {exc}"


@tool_registry.tool(
    capabilities={Capability.FILESYSTEM_DELETE},
    permission_resolver=_delete_permission,
)
def delete_path(
    context: ToolContext,
    path: Annotated[
        str,
        Field(description="Path to the file, symbolic link, or folder to permanently delete."),
    ],
) -> str:
    """Permanently delete a file, symbolic link, or folder tree from the local disk."""
    try:
        target = Path(path)
        kind = _deletion_kind(target)
        if kind == "folder":
            shutil.rmtree(target)
        elif kind in {"file", "symbolic link"}:
            target.unlink()
        elif target.exists():
            return (
                f"Error deleting path: Path '{path}' is not a file, symbolic link, or folder."
            )
        else:
            return f"Error deleting path: Path '{path}' does not exist."
        context.invalidate_instructions(target)
        return f"Successfully deleted path '{path}'."
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error deleting path: {exc}"
