"""Provide tools for accessing files and folders on the local disk."""

import logging
import os
import shutil
import stat
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from .. import constants
from ..errors import Problem, ProblemException, log_problem
from ..models import ToolResultPresentation, ToolResultPresentationSpec
from ..permissions import Action, FileKind, FileManifestEntry, FileTarget, Operation, OperationPlan
from ..tooling import TOOL_READY, ToolContext, ToolPreflightResult, ToolStatus, tool
from ..utils import (
    TextSearchCase,
    canonical_path,
    filter_paths_by_globs,
    format_content_diff,
    format_content_preview,
    is_binary_file,
    is_path_ignored,
    iter_visible_paths,
    read_bounded_text,
    ripgrep_path,
    search_text_paths,
    sha256_digest,
    write_text_atomically,
)
from .models import FileContentResult, FolderEntry, TextSearchResult

_LOGGER = logging.getLogger(__name__)


def _check_text_search() -> ToolPreflightResult:
    """Return whether the external text-search capability is available."""
    try:
        ripgrep_path()
    except FileNotFoundError as exc:
        return ToolPreflightResult(status=ToolStatus.BROKEN, detail=str(exc))
    return TOOL_READY


def _write_preview(path: Path, content: str, existing: str | None) -> str:
    """Return the write preview displayed in a permission prompt."""
    if not existing:
        return f"Proposed content:\n{format_content_preview(content)}"
    return f"Proposed changes:\n{format_content_diff(existing, content, str(path))}"


def _file_plan(action: Action):
    """Return a planner for one non-mutating filesystem operation."""

    def _plan(arguments: dict[str, object]) -> OperationPlan:
        path = Path(str(arguments["path"]))
        resource = canonical_path(path)
        normalized = dict(arguments)
        normalized["path"] = resource
        return OperationPlan(
            arguments=normalized,
            operations=(
                Operation(
                    tool_id="",
                    action=action,
                    target=FileTarget(path=resource),
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


def _planning_problem(operation: str, code: str, title: str, detail: str) -> ProblemException:
    """Return one structured expected filesystem planning failure."""
    return ProblemException(
        Problem(code=code, title=title, detail=detail, severity="warning", operation=operation)
    )


def _mutation_path(path: Path) -> Path:
    """Resolve a mutation target's parent without following its final path component."""
    return path.parent.resolve() / path.name


def _lexical_mutation_path(path: Path) -> Path:
    """Return an absolute normalized target without inspecting the filesystem."""
    return Path(os.path.abspath(path))


def _path_kind(path: Path) -> FileKind | None:
    """Return the supported object kind without following symbolic links."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    return None


def _capture_entry(path: Path, relative_path: str, *, digest: bool) -> FileManifestEntry:
    """Capture one object state without following symbolic links."""
    status = path.lstat()
    kind = _path_kind(path)
    if kind is None:
        raise ValueError(f"Path '{path}' has an unsupported object type.")
    content_digest = None
    if digest and kind == "file":
        content_digest = sha256_digest(path.read_bytes())
    return FileManifestEntry(
        relative_path=relative_path,
        kind=kind,
        device=status.st_dev,
        inode=status.st_ino,
        size=status.st_size,
        mtime_ns=status.st_mtime_ns,
        mode=status.st_mode,
        digest=content_digest,
    )


def _read_regular_entry(path: Path, relative_path: str) -> tuple[FileManifestEntry, bytes]:
    """Read one regular file and bind its digest to stable surrounding metadata."""
    before = _capture_entry(path, relative_path, digest=False)
    if before.kind != "file":
        raise ValueError(f"Path '{path}' is not a regular file.")
    content = path.read_bytes()
    after = _capture_entry(path, relative_path, digest=False)
    if after != before:
        raise ValueError("The target changed during planning; operation was cancelled.")
    return after.model_copy(update={"digest": sha256_digest(content)}), content


def _target_from_entry(
    path: str,
    entry: FileManifestEntry,
    *,
    recursive: bool = False,
    manifest: tuple[FileManifestEntry, ...] = (),
) -> FileTarget:
    """Build an executable file capability from one captured state."""
    parent = Path(path).parent.lstat()
    return FileTarget(
        path=path,
        expected_exists=True,
        expected_digest=entry.digest,
        expected_kind=entry.kind,
        expected_device=entry.device,
        expected_inode=entry.inode,
        expected_size=entry.size,
        expected_mtime_ns=entry.mtime_ns,
        expected_mode=entry.mode,
        expected_parent_device=parent.st_dev,
        expected_parent_inode=parent.st_ino,
        recursive=recursive,
        manifest=manifest,
    )


def _mutation_operation(action: Action, target: FileTarget, reason: str | None = None) -> Operation:
    """Build one filesystem mutation operation before registry identity binding."""
    return Operation(tool_id="", action=action, target=target, reason=reason)


def _read_operation(action: Action, path: str) -> Operation:
    """Build one prerequisite filesystem inspection operation."""
    return Operation(tool_id="", action=action, target=FileTarget(path=path))


def _write_plan(arguments: dict[str, object]) -> OperationPlan:
    """Authorize target inspection before planning a create or replacement."""
    path = _lexical_mutation_path(Path(str(arguments["path"])))
    resource = str(path)
    normalized = {**arguments, "path": resource}
    boundaries = tuple(
        _mutation_operation(action, FileTarget(path=resource))
        for action in (Action.FILESYSTEM_CREATE, Action.FILESYSTEM_REPLACE)
    )
    return OperationPlan(
        arguments=normalized,
        operations=(_read_operation(Action.FILESYSTEM_READ, resource),),
        boundary_operations=boundaries,
        continuation=lambda: _finish_write_plan(normalized),
    )


def _finish_write_plan(arguments: dict[str, object]) -> OperationPlan:
    """Inspect an authorized write target and produce its mutation capability."""
    path = _mutation_path(Path(str(arguments["path"])))
    resource = str(path)
    normalized = {**arguments, "path": resource}
    kind = _path_kind(path)
    if kind is None:
        if path.exists() or path.is_symlink():
            raise _planning_problem(
                "write_text_file",
                "filesystem.unsupported_path",
                "Could not write file",
                f"Path '{path}' is not a regular file.",
            )
        try:
            parent = path.parent.lstat()
        except OSError as exc:
            raise _planning_problem(
                "write_text_file",
                "filesystem.write_failed",
                "Could not write file",
                str(exc),
            ) from exc
        operation = _mutation_operation(
            Action.FILESYSTEM_CREATE,
            FileTarget(
                path=resource,
                expected_exists=False,
                expected_parent_device=parent.st_dev,
                expected_parent_inode=parent.st_ino,
            ),
            _write_preview(path, str(arguments["content"]), None),
        )
        return OperationPlan(
            arguments=normalized, operations=(operation,), boundary_operations=(operation,)
        )
    if kind != "file":
        raise _planning_problem(
            "write_text_file",
            "filesystem.path_not_file",
            "Could not write file",
            f"Path '{path}' is not a regular file.",
        )
    entry, existing_bytes = _read_regular_entry(path, ".")
    try:
        existing = existing_bytes.decode("utf-8")
    except UnicodeDecodeError:
        existing = None
    target = _target_from_entry(str(path), entry)
    operation = _mutation_operation(
        Action.FILESYSTEM_REPLACE,
        target,
        (
            _write_preview(path, str(arguments["content"]), existing)
            if existing is not None
            else "Existing content could not be previewed; proposed content:\n"
            f"{format_content_preview(str(arguments['content']))}"
        ),
    )
    return OperationPlan(arguments=normalized, operations=(operation,))


def _edit_plan(arguments: dict[str, object]) -> OperationPlan:
    """Plan prerequisite authorization for one exact UTF-8 text replacement."""
    path = _lexical_mutation_path(Path(str(arguments["path"])))
    resource = str(path)
    normalized = {**arguments, "path": resource}
    boundary = _mutation_operation(Action.FILESYSTEM_REPLACE, FileTarget(path=resource))
    return OperationPlan(
        arguments=normalized,
        operations=(_read_operation(Action.FILESYSTEM_READ, resource),),
        boundary_operations=(boundary,),
        continuation=lambda: _finish_edit_plan(normalized),
    )


def _finish_edit_plan(arguments: dict[str, object]) -> OperationPlan:
    """Inspect an authorized edit target and produce its exact mutation capability."""
    path = _mutation_path(Path(str(arguments["path"])))
    resource = str(path)
    normalized = {**arguments, "path": resource}
    try:
        if _path_kind(path) != "file":
            raise _edit_problem(
                "filesystem.path_not_file", f"Path '{path}' is not an existing regular file."
            )
        entry, original_bytes = _read_regular_entry(path, ".")
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

    return OperationPlan(
        arguments=normalized,
        operations=(
            _mutation_operation(
                Action.FILESYSTEM_REPLACE,
                _target_from_entry(str(path), entry),
                reason=f"Proposed changes:\n{format_content_diff(original, updated, resource)}",
            ),
        ),
    )


def _delete_plan(arguments: dict[str, object]) -> OperationPlan:
    """Plan protected-boundary and prerequisite inspection for one deletion."""
    path = _lexical_mutation_path(Path(str(arguments["path"])))
    resource = str(path)
    normalized = {**arguments, "path": resource}
    boundary = _mutation_operation(
        Action.FILESYSTEM_DELETE, FileTarget(path=resource, recursive=True)
    )
    return OperationPlan(
        arguments=normalized,
        operations=(_read_operation(Action.FILESYSTEM_READ, resource),),
        boundary_operations=(boundary,),
        continuation=lambda: _inspect_delete_plan(normalized),
    )


def _inspect_delete_plan(arguments: dict[str, object]) -> OperationPlan:
    """Inspect an authorized deletion target and plan its exact capability."""
    path = _mutation_path(Path(str(arguments["path"])))
    resource = str(path)
    normalized = {**arguments, "path": resource}
    kind = _path_kind(path)
    if kind is None:
        if path.exists() or path.is_symlink():
            raise _planning_problem(
                "delete_path",
                "filesystem.unsupported_path",
                "Could not delete path",
                f"Path '{path}' is not a file, symbolic link, or folder.",
            )
        raise _planning_problem(
            "delete_path",
            "filesystem.path_missing",
            "Could not delete path",
            f"Path '{path}' does not exist.",
        )
    recursive = kind == "directory"
    reason = (
        "Permanently delete this symbolic link; its target will not be deleted."
        if kind == "symlink"
        else "Permanently delete this folder and all of its contents."
        if recursive
        else "Permanently delete this file."
    )
    boundary = _mutation_operation(
        Action.FILESYSTEM_DELETE, FileTarget(path=resource, recursive=recursive), reason
    )
    if kind == "symlink":
        entry = _capture_entry(path, ".", digest=False)
        final = _mutation_operation(
            Action.FILESYSTEM_DELETE, _target_from_entry(resource, entry), reason
        )
        return OperationPlan(
            arguments=normalized, operations=(final,), boundary_operations=(boundary,)
        )
    if not recursive:
        return _finish_file_delete_plan(normalized, reason)
    return OperationPlan(
        arguments=normalized,
        operations=(_read_operation(Action.FILESYSTEM_LIST, resource),),
        boundary_operations=(boundary,),
        continuation=lambda: _directory_metadata_plan(normalized, reason),
    )


def _finish_file_delete_plan(arguments: dict[str, object], reason: str) -> OperationPlan:
    """Capture an authorized regular-file deletion capability."""
    path = Path(str(arguments["path"]))
    entry = _capture_entry(path, ".", digest=True)
    if entry.kind != "file":
        raise _planning_problem(
            "delete_path",
            "filesystem.target_changed",
            "Could not delete path",
            "The target changed during planning; deletion was cancelled.",
        )
    return OperationPlan(
        arguments=arguments,
        operations=(
            _mutation_operation(
                Action.FILESYSTEM_DELETE, _target_from_entry(str(path), entry), reason
            ),
        ),
    )


def _directory_entries(path: Path) -> tuple[FileManifestEntry, ...]:
    """List a recursive tree and capture metadata without loading file contents."""
    entries = [_capture_entry(path, ".", digest=False)]
    if entries[0].kind != "directory":
        raise ValueError("The recursive target changed; deletion was cancelled.")
    for root, directories, files in os.walk(path, topdown=True, followlinks=False):
        root_path = Path(root)
        directories.sort()
        files.sort()
        for name in directories + files:
            child = root_path / name
            relative = child.relative_to(path).as_posix()
            entry = _capture_entry(child, relative, digest=False)
            entries.append(entry)
            if entry.kind == "symlink" and name in directories:
                directories.remove(name)
    return tuple(entries)


def _directory_metadata_plan(arguments: dict[str, object], reason: str) -> OperationPlan:
    """List an authorized tree and request reads needed for its exact manifest."""
    path = Path(str(arguments["path"]))
    metadata = _directory_entries(path)
    reads = tuple(
        _read_operation(Action.FILESYSTEM_READ, str(path / entry.relative_path))
        for entry in metadata
        if entry.kind == "file"
    )
    if not reads:
        return _finish_directory_delete_plan(arguments, reason, metadata)
    return OperationPlan(
        arguments=arguments,
        operations=reads,
        continuation=lambda: _finish_directory_delete_plan(arguments, reason, metadata),
    )


def _finish_directory_delete_plan(
    arguments: dict[str, object],
    reason: str,
    metadata: tuple[FileManifestEntry, ...],
) -> OperationPlan:
    """Capture file digests and produce one explicit recursive deletion contract."""
    path = Path(str(arguments["path"]))
    completed = []
    for expected in metadata:
        current = _capture_entry(
            path if expected.relative_path == "." else path / expected.relative_path,
            expected.relative_path,
            digest=expected.kind == "file",
        )
        if current.model_copy(update={"digest": None}) != expected:
            raise _planning_problem(
                "delete_path",
                "filesystem.target_changed",
                "Could not delete path",
                "The target changed during planning; deletion was cancelled.",
            )
        completed.append(current)
    manifest = tuple(completed)
    target = _target_from_entry(str(path), manifest[0], recursive=True, manifest=manifest)
    return OperationPlan(
        arguments=arguments,
        operations=(_mutation_operation(Action.FILESYSTEM_DELETE, target, reason),),
    )


def _matches_target(path: Path, target: FileTarget) -> bool:
    """Return whether a path still satisfies an authorized state capability."""
    if not _matches_parent(path, target):
        return False
    try:
        current = _capture_entry(path, ".", digest=target.expected_digest is not None)
    except (OSError, ValueError):
        return False
    return (
        current.kind == target.expected_kind
        and current.device == target.expected_device
        and current.inode == target.expected_inode
        and current.size == target.expected_size
        and current.mtime_ns == target.expected_mtime_ns
        and current.mode == target.expected_mode
        and current.digest == target.expected_digest
    )


def _matches_parent(path: Path, target: FileTarget) -> bool:
    """Return whether the canonical parent directory retains its approved identity."""
    try:
        parent = path.parent.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(parent.st_mode)
        and parent.st_dev == target.expected_parent_device
        and parent.st_ino == target.expected_parent_inode
    )


def _matches_manifest(path: Path, target: FileTarget) -> bool:
    """Return whether a recursive target still matches its approved manifest.

    This is an immediate optimistic precondition, not an atomic tree compare-and-delete. The
    executor additionally relies on the standard library's descriptor-based ``rmtree`` protection
    where available and refuses recursion on platforms without it.
    """
    if not _matches_parent(path, target):
        return False
    try:
        current_metadata = _directory_entries(path)
        if len(current_metadata) != len(target.manifest):
            return False
        completed = tuple(
            _capture_entry(
                path if entry.relative_path == "." else path / entry.relative_path,
                entry.relative_path,
                digest=entry.kind == "file",
            )
            for entry in current_metadata
        )
    except (OSError, ValueError):
        return False
    return completed == target.manifest


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
    preflight=_check_text_search,
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
        TextSearchCase,
        Field(description="Case strategy; smart treats queries containing uppercase as sensitive."),
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
        code = (
            "filesystem.invalid_search_pattern"
            if regex and "regex parse error" in detail
            else (
                "filesystem.search_unavailable"
                if isinstance(exc, FileNotFoundError)
                else "filesystem.search_failed"
            )
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
    actions={Action.FILESYSTEM_READ, Action.FILESYSTEM_CREATE, Action.FILESYSTEM_REPLACE},
    operation_planner=_write_plan,
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
            Path(planned.path),
            content,
            expected_digest=planned.expected_digest if planned.expected_exists else None,
            expected_device=planned.expected_device,
            expected_inode=planned.expected_inode,
            expected_mode=planned.expected_mode,
            expected_parent_device=planned.expected_parent_device,
            expected_parent_inode=planned.expected_parent_inode,
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
    actions={Action.FILESYSTEM_READ, Action.FILESYSTEM_REPLACE},
    operation_planner=_edit_plan,
)
def edit_text_file(
    context: ToolContext,
    path: Annotated[str, Field(description="Path to the existing UTF-8 text file to edit.")],
    old_content: Annotated[
        str,
        Field(description="Exact, non-empty existing content that uniquely anchors the edit."),
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
        target = Path(planned.path)
        if not _matches_target(target, planned):
            raise RuntimeError("The target changed after approval; replacement was cancelled.")
        current_bytes = target.read_bytes()
        updated, replacement_count = _edited_content(
            current_bytes.decode("utf-8"), old_content, new_content, replace_all
        )
        write_text_atomically(
            target,
            updated,
            expected_digest=planned.expected_digest,
            expected_device=planned.expected_device,
            expected_inode=planned.expected_inode,
            expected_mode=planned.expected_mode,
            expected_parent_device=planned.expected_parent_device,
            expected_parent_inode=planned.expected_parent_inode,
        )
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
    actions={Action.FILESYSTEM_LIST, Action.FILESYSTEM_READ, Action.FILESYSTEM_DELETE},
    operation_planner=_delete_plan,
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
        operation = context.operations[0] if context.operations else None
        planned = operation.target if operation is not None else None
        if (
            not isinstance(planned, FileTarget)
            or planned.expected_exists is not True
            or planned.expected_kind is None
        ):
            raise RuntimeError("Authorized deletion capability is missing.")
        target = Path(planned.path)
        if planned.recursive:
            if (
                not shutil.rmtree.avoids_symlink_attacks
                or planned.expected_kind != "directory"
                or not _matches_manifest(target, planned)
            ):
                raise RuntimeError("The target changed after approval; deletion was cancelled.")
            shutil.rmtree(target)
        elif planned.expected_kind in {"file", "symlink"}:
            if not _matches_target(target, planned):
                raise RuntimeError("The target changed after approval; deletion was cancelled.")
            target.unlink()
        else:
            raise RuntimeError("Authorized deletion capability has an unsupported target kind.")
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
