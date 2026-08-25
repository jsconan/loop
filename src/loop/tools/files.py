"""Provide tools for accessing files and folders on the local disk."""

import logging
import shutil
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from .. import constants
from ..errors import Problem, ProblemException, log_problem
from ..models import ToolResultPresentation, ToolResultPresentationSpec
from ..permissions import Action, FileTarget, Operation, OperationPlan
from ..tooling import ToolContext, tool
from ..utils import (
    canonical_path,
    filter_paths_by_globs,
    format_content_diff,
    format_content_preview,
    is_binary_file,
    is_path_ignored,
    iter_visible_paths,
    read_bounded_text,
    search_text_paths,
    sha256_digest,
    write_text_atomically,
)
from .models import FileContentResult, FolderEntry, TextSearchResult

_LOGGER = logging.getLogger(__name__)


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


def _file_plan(action: Action):
    """Return a planner for one normalized filesystem operation."""

    def _plan(arguments: dict[str, object]) -> OperationPlan:
        path = Path(str(arguments["path"]))
        if action is Action.FILESYSTEM_CREATE:
            expected_exists = path.exists()
            planned_action = (
                Action.FILESYSTEM_REPLACE if expected_exists else Action.FILESYSTEM_CREATE
            )
            resource = canonical_path(path)
        else:
            planned_action = action
            resource = (
                str(path.absolute())
                if action is Action.FILESYSTEM_DELETE and path.is_symlink()
                else canonical_path(path)
            )
        reason = None
        if action is Action.FILESYSTEM_CREATE:
            reason = _write_preview(path, str(arguments["content"]))
        elif action is Action.FILESYSTEM_DELETE:
            kind = _deletion_kind(path)
            reason = (
                "Permanently delete this symbolic link; its target will not be deleted."
                if kind == "symbolic link"
                else "Permanently delete this folder and all of its contents."
                if kind == "folder"
                else f"Permanently delete this {kind}."
                if kind is not None
                else "Deletion supports files, symbolic links, and folders only."
            )
        normalized = dict(arguments)
        normalized["path"] = resource
        return OperationPlan(
            arguments=normalized,
            operations=(
                Operation(
                    tool_id="",
                    action=planned_action,
                    target=FileTarget(
                        path=resource,
                        expected_exists=expected_exists
                        if action is Action.FILESYSTEM_CREATE
                        else None,
                        expected_digest=sha256_digest(path.read_bytes())
                        if action is Action.FILESYSTEM_CREATE and expected_exists
                        else None,
                    ),
                    reason=reason,
                ),
            ),
        )

    return _plan


def _edit_problem(code: str, detail: str) -> ProblemException:
    """Return a structured planning failure for an invalid text edit."""
    return ProblemException(
        Problem(
            code=code,
            title="Could not edit file",
            detail=detail,
            severity="warning",
            operation="edit_text_file",
        )
    )


def _edited_content(
    content: str,
    old_content: str,
    new_content: str,
    replace_all: bool,
) -> tuple[str, int]:
    """Return content with one validated exact replacement applied."""
    if not old_content:
        raise _edit_problem(
            "filesystem.empty_match",
            "old_content cannot be empty. Include exact existing content that anchors the edit.",
        )
    occurrences = content.count(old_content)
    if not occurrences:
        raise _edit_problem(
            "filesystem.content_not_found",
            "old_content was not found. Read the relevant file range again and retry with exact "
            "content.",
        )
    if occurrences > 1 and not replace_all:
        raise _edit_problem(
            "filesystem.content_ambiguous",
            f"Found {occurrences} matches for old_content. Include more surrounding content or "
            "set replace_all to true.",
        )
    if old_content == new_content:
        raise _edit_problem(
            "filesystem.no_content_change",
            "old_content and new_content are identical; no edit was requested.",
        )
    replacements = occurrences if replace_all else 1
    return content.replace(old_content, new_content, replacements), replacements


def _edit_plan(arguments: dict[str, object]) -> OperationPlan:
    """Plan one exact UTF-8 text replacement and its approved resulting content."""
    path = Path(str(arguments["path"]))
    resource = canonical_path(path)
    try:
        if not path.is_file():
            raise _edit_problem(
                "filesystem.path_not_file",
                f"Path '{path}' is not an existing regular file.",
            )
        original_bytes = path.read_bytes()
        try:
            original = original_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _edit_problem(
                "filesystem.binary_file",
                f"File '{path}' is not valid UTF-8 text and cannot be edited with this tool.",
            ) from exc
        updated, _ = _edited_content(
            original,
            str(arguments["old_content"]),
            str(arguments["new_content"]),
            bool(arguments["replace_all"]),
        )
    except OSError as exc:
        raise _edit_problem("filesystem.edit_failed", str(exc)) from exc

    normalized = dict(arguments)
    normalized["path"] = resource
    return OperationPlan(
        arguments=normalized,
        operations=(
            Operation(
                tool_id="",
                action=Action.FILESYSTEM_REPLACE,
                target=FileTarget(
                    path=resource,
                    expected_exists=True,
                    expected_digest=sha256_digest(original_bytes),
                ),
                reason=f"Proposed changes:\n{format_content_diff(original, updated, resource)}",
            ),
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


@tool(
    actions={Action.FILESYSTEM_LIST},
    operation_planner=_file_plan(Action.FILESYSTEM_LIST),
    result_presentation=ToolResultPresentationSpec(kind=ToolResultPresentation.TREE),
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
) -> list[FolderEntry] | Problem:
    """List selected, non-ignored entries in a folder on the local disk."""
    try:
        folder = Path(path).resolve()
        if is_path_ignored(folder):
            return Problem(
                code="filesystem.path_ignored",
                title="Folder cannot be listed",
                detail=f"Path '{path}' is ignored.",
                operation="list_folder",
            )
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
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
        problem = Problem.from_exception(
            exc,
            code="filesystem.list_failed",
            title="Could not list folder",
            operation="list_folder",
        )
        log_problem(_LOGGER, problem, exc)
        return problem


@tool(
    actions={Action.FILESYSTEM_READ},
    operation_planner=_file_plan(Action.FILESYSTEM_READ),
    result_presentation=ToolResultPresentationSpec(kind=ToolResultPresentation.TEXT),
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
) -> FileContentResult | str | Problem:
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
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
        if str(exc) == "Content appears to be binary.":
            return Problem(
                code="filesystem.binary_file",
                title="Could not read text file",
                detail=f"File '{path}' appears to be binary.",
                operation="read_text_file",
            )
        problem = Problem.from_exception(
            exc,
            code="filesystem.read_failed",
            title="Could not read file",
            operation="read_text_file",
        )
        log_problem(_LOGGER, problem, exc)
        return problem


@tool(
    actions={Action.FILESYSTEM_READ},
    operation_planner=_file_plan(Action.FILESYSTEM_READ),
)
def search_text(
    context: ToolContext,
    path: Annotated[str, Field(description="Path to a text file or folder to search.")],
    query: Annotated[
        str,
        Field(description="Non-empty literal text or regular expression to find.", min_length=1),
    ],
    regex: Annotated[
        bool,
        Field(description="Whether query is a regular expression instead of literal text."),
    ] = False,
    case: Annotated[
        Literal["smart", "sensitive", "insensitive"],
        Field(
            description="Case strategy; smart treats queries containing uppercase as sensitive."
        ),
    ] = "smart",
    include: Annotated[
        list[str] | None,
        Field(description="Optional inclusive Git-style file globs.", max_length=20),
    ] = None,
    context_lines: Annotated[
        int,
        Field(description="Neighboring lines to return around every match.", ge=0, le=10),
    ] = 0,
    max_results: Annotated[
        int,
        Field(description="Maximum matching lines to return.", ge=1, le=1000),
    ] = 100,
    max_bytes: Annotated[
        int,
        Field(
            description="Maximum approximate result bytes retained, capped by the application.",
            ge=1,
            le=constants.MAX_TOOL_CONTENT_BYTES,
        ),
    ] = constants.MAX_TOOL_CONTENT_BYTES,
) -> TextSearchResult | Problem:
    """Search bounded text matches in a file or folder on the local disk."""
    try:
        target = Path(path).resolve()
        if is_path_ignored(target):
            return Problem(
                code="filesystem.path_ignored",
                title="Path cannot be searched",
                detail=f"Path '{path}' is ignored.",
                operation="search_text",
            )
        if target.is_file():
            root = target.parent
            candidates = [] if is_binary_file(target) else [target]
        elif target.is_dir():
            root = target
            candidates = [
                entry
                for entry in iter_visible_paths(target, True)
                if entry.is_file() and not entry.is_symlink() and not is_binary_file(entry)
            ]
        else:
            return Problem(
                code="filesystem.path_not_searchable",
                title="Path cannot be searched",
                detail=f"Path '{path}' is not an existing file or folder.",
                operation="search_text",
            )
        candidates = list(filter_paths_by_globs(candidates, root, include))
        matches, truncated = search_text_paths(
            candidates,
            query,
            root=root,
            regex=regex,
            case=case,
            context_lines=context_lines,
            max_results=max_results,
            max_bytes=max_bytes,
        )
        if target.is_dir():
            context.observe_directory(target)
        else:
            context.observe_file(target)
        for match_path in {match["path"] for match in matches}:
            context.observe_file(root / match_path)
        return TextSearchResult(matches=matches, truncated=truncated)
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
        detail = str(exc)
        code = "filesystem.invalid_search_pattern" if regex and "regex parse error" in detail else (
            "filesystem.search_unavailable" if isinstance(exc, FileNotFoundError) else
            "filesystem.search_failed"
        )
        problem = Problem.from_exception(
            exc,
            code=code,
            title="Could not search text",
            operation="search_text",
        )
        log_problem(_LOGGER, problem, exc)
        return problem


@tool(
    actions={Action.FILESYSTEM_CREATE, Action.FILESYSTEM_REPLACE},
    operation_planner=_file_plan(Action.FILESYSTEM_CREATE),
)
def write_text_file(
    context: ToolContext,
    path: Annotated[str, Field(description="Path to the text file to write.")],
    content: Annotated[str, Field(description="Content to write to the file.")],
) -> str | Problem:
    """Write content to a file on the local disk."""
    try:
        target = Path(path)
        operation = context.operations[0] if context.operations else None
        planned = operation.target if operation is not None else None
        if not isinstance(planned, FileTarget) or planned.expected_exists is None:
            raise RuntimeError("Authorized file-state precondition is missing.")
        write_text_atomically(
            target,
            content,
            expected_digest=planned.expected_digest if planned.expected_exists else None,
        )
        context.observe_file(target)
        context.invalidate_instructions(target)
        return f"Successfully wrote to file '{path}'."
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
        problem = Problem.from_exception(
            exc,
            code="filesystem.write_failed",
            title="Could not write file",
            operation="write_text_file",
        )
        log_problem(_LOGGER, problem, exc)
        return problem


@tool(
    actions={Action.FILESYSTEM_REPLACE},
    operation_planner=_edit_plan,
)
def edit_text_file(
    context: ToolContext,
    path: Annotated[str, Field(description="Path to the existing UTF-8 text file to edit.")],
    old_content: Annotated[
        str,
        Field(
            description="Exact, non-empty existing content that uniquely anchors the edit."
        ),
    ],
    new_content: Annotated[
        str,
        Field(description="Replacement content; use an empty string to delete the matched text."),
    ],
    replace_all: Annotated[
        bool,
        Field(description="Whether to replace every exact match instead of requiring one match."),
    ] = False,
) -> str | Problem:
    """Replace exact content in an existing UTF-8 text file."""
    try:
        target = Path(path)
        operation = context.operations[0] if context.operations else None
        planned = operation.target if operation is not None else None
        if not isinstance(planned, FileTarget) or not planned.expected_exists:
            raise RuntimeError("Authorized file-state precondition is missing.")
        if planned.expected_digest is None:
            raise RuntimeError("Approved edit content is missing.")
        current_bytes = target.read_bytes()
        if sha256_digest(current_bytes) != planned.expected_digest:
            raise RuntimeError("The target changed after approval; replacement was cancelled.")
        updated, replacement_count = _edited_content(
            current_bytes.decode("utf-8"), old_content, new_content, replace_all
        )
        write_text_atomically(target, updated, expected_digest=planned.expected_digest)
        context.observe_file(target)
        context.invalidate_instructions(target)
        noun = "replacement" if replacement_count == 1 else "replacements"
        return f"Successfully edited file '{path}' ({replacement_count} {noun})."
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
        problem = Problem.from_exception(
            exc,
            code="filesystem.edit_failed",
            title="Could not edit file",
            operation="edit_text_file",
        )
        log_problem(_LOGGER, problem, exc)
        return problem


@tool(
    actions={Action.FILESYSTEM_DELETE},
    operation_planner=_file_plan(Action.FILESYSTEM_DELETE),
)
def delete_path(
    context: ToolContext,
    path: Annotated[
        str,
        Field(description="Path to the file, symbolic link, or folder to permanently delete."),
    ],
) -> str | Problem:
    """Permanently delete a file, symbolic link, or folder tree from the local disk."""
    try:
        target = Path(path)
        kind = _deletion_kind(target)
        if kind == "folder":
            shutil.rmtree(target)
        elif kind in {"file", "symbolic link"}:
            target.unlink()
        elif target.exists():
            return Problem(
                code="filesystem.unsupported_path",
                title="Could not delete path",
                detail=f"Path '{path}' is not a file, symbolic link, or folder.",
                operation="delete_path",
            )
        else:
            return Problem(
                code="filesystem.path_missing",
                title="Could not delete path",
                detail=f"Path '{path}' does not exist.",
                operation="delete_path",
            )
        context.invalidate_instructions(target)
        return f"Successfully deleted path '{path}'."
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
        problem = Problem.from_exception(
            exc,
            code="filesystem.delete_failed",
            title="Could not delete path",
            operation="delete_path",
        )
        log_problem(_LOGGER, problem, exc)
        return problem
