"""Provide tools for accessing files and folders on the local disk."""

import logging
import os
import stat
from functools import partial
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field

from .. import constants
from ..errors import Problem, ProblemException, log_problem
from ..models import ToolResultPresentation, ToolResultPresentationSpec
from ..permissions import Action, FileKind, FileManifestEntry, FileTarget, Operation, OperationPlan
from ..tooling import ToolContext, normalize_optional_null, tool
from ..utils import (
    TextSearchCase,
    VirtualPath,
    filter_paths_by_globs,
    format_content_diff,
    format_content_preview,
    is_path_ignored,
    iter_visible_paths,
    open_selected_file,
    open_selected_parent,
    read_bounded_text_stream,
    search_text_paths,
    sha256_digest,
    write_text_atomically,
)
from .models import FileContentResult, FolderEntry, TextSearchResult

_LOGGER = logging.getLogger(__name__)


def _write_preview(path: Path, content: str, existing: str | None) -> str:
    """Return the write preview displayed in a permission prompt."""
    if not existing:
        return f"Proposed content:\n{format_content_preview(content)}"
    return f"Proposed changes:\n{format_content_diff(existing, content, str(path))}"


def _file_plan(action: Action):
    """Return a planner for one non-mutating filesystem operation."""

    def _plan(arguments: dict[str, object]) -> OperationPlan:
        path = Path(str(arguments["path"]))
        resource = str(_lexical_mutation_path(path))
        normalized = dict(arguments)
        normalized["path"] = resource
        return OperationPlan(
            arguments=normalized,
            continuation=lambda: _bound_read_plan(normalized, action),
            operations=(
                Operation(
                    tool_id="",
                    action=action,
                    target=FileTarget(path=resource),
                ),
            ),
        )

    return _plan


def _bound_read_plan(arguments: dict[str, object], action: Action) -> OperationPlan:
    """Bind an authorized file identity before final permission evaluation and execution."""
    path = Path(str(arguments["path"]))
    try:
        details = path.stat()
    except FileNotFoundError:
        target = FileTarget(path=str(path))
    else:
        target = FileTarget(
            path=str(path),
            expected_kind="directory" if stat.S_ISDIR(details.st_mode) else None,
            expected_device=details.st_dev,
            expected_inode=details.st_ino,
            expected_ctime_ns=details.st_ctime_ns,
            expected_link_count=details.st_nlink,
        )
    return OperationPlan(
        arguments=arguments,
        operations=(Operation(tool_id="", action=action, target=target),),
    )


def _approved_read_root(context: ToolContext, action: Action) -> FileTarget | None:
    """Return the exact read or listing root admitted by this invocation's planner."""
    return next(
        (
            operation.target
            for operation in context.operations
            if operation.action is action and isinstance(operation.target, FileTarget)
        ),
        None,
    )


def _readable(
    context: ToolContext,
    path: Path,
    action: Action = Action.FILESYSTEM_READ,
    *,
    supporting: bool = False,
) -> bool:
    """Apply current policy before a tool discovers or inspects one path."""
    if context.permission_manager is None:
        raise RuntimeError("File permission authority is unavailable.")
    allowed = context.permission_manager.can_inspect_file(
        path,
        tool_id=context.tool_name,
        action=action,
        approved_root=None if supporting else _approved_read_root(context, action),
    )
    if allowed and not supporting:
        context.file_read_paths.add(str(path))
    return allowed


def _open_read(context: ToolContext, path: Path, *, supporting: bool = False) -> int:
    """Recheck current policy and open the authorized identity for one tool read."""
    if context.permission_manager is None:
        raise RuntimeError("File permission authority is unavailable.")
    identity = next(
        (
            (
                operation.target.expected_device,
                operation.target.expected_inode,
                operation.target.expected_ctime_ns,
                operation.target.expected_link_count,
            )
            for operation in context.operations
            if isinstance(operation.target, FileTarget)
            and (
                operation.target.path == str(path) or Path(operation.target.path).resolve() == path
            )
            and operation.target.expected_device is not None
            and operation.target.expected_inode is not None
            and operation.target.expected_ctime_ns is not None
            and operation.target.expected_link_count is not None
        ),
        None,
    )
    selected, descriptor = context.permission_manager.open_read_file(
        path,
        identity,
        tool_id=context.tool_name,
        approved_root=(
            None if supporting else _approved_read_root(context, Action.FILESYSTEM_READ)
        ),
    )
    context.file_read_paths.update((str(path), str(selected)))
    return descriptor


def _read_ignore(context: ToolContext, path: Path) -> str | None:
    """Load ignore policy only through an independent, descriptor-bound read."""
    if not _readable(context, path, supporting=True) or not path.is_file():
        return None
    with os.fdopen(_open_read(context, path, supporting=True), "r", encoding="utf-8") as stream:
        return stream.read()


def _searchable(context: ToolContext, path: Path) -> bool:
    """Filter binary data through the same authorized descriptor boundary as text reads."""
    with os.fdopen(_open_read(context, path), "rb") as stream:
        return b"\0" not in stream.read(constants.DEFAULT_STREAM_CHUNK_SIZE)


def _mutation_denial(context: ToolContext) -> Problem | None:
    """Recheck the current hard boundary before consuming an approved mutation capability."""
    if context.permission_manager is None:
        raise RuntimeError("File permission authority is unavailable.")
    context.permission_manager.require_policy_ready()
    denial = context.permission_manager.check_boundaries(
        context.operations + context.prerequisite_operations
    )
    if denial is None:
        return None
    return Problem(
        code="tool.denied",
        title="Tool denied",
        detail=denial.reason,
        severity="warning",
        operation=context.tool_name,
    )


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


def _capture_entry(
    path: Path,
    relative_path: str,
    *,
    digest: bool,
    identity: tuple[int, int, int, int] | None = None,
) -> FileManifestEntry:
    """Capture one object state without following symbolic links."""
    status = path.lstat()
    kind = _path_kind(path)
    if kind is None:
        raise ValueError(f"Path '{path}' has an unsupported object type.")
    content_digest = None
    if digest and kind == "file":
        _, content = _read_regular_entry(path, relative_path, identity)
        content_digest = sha256_digest(content)
    return FileManifestEntry(
        relative_path=relative_path,
        kind=kind,
        device=status.st_dev,
        inode=status.st_ino,
        size=status.st_size,
        mtime_ns=status.st_mtime_ns,
        ctime_ns=status.st_ctime_ns,
        link_count=status.st_nlink,
        mode=status.st_mode,
        digest=content_digest,
    )


def _read_regular_entry(
    path: Path, relative_path: str, identity: tuple[int, int, int, int] | None
) -> tuple[FileManifestEntry, bytes]:
    """Read one regular file and bind its digest to stable surrounding metadata."""
    before = _capture_entry(path, relative_path, digest=False)
    if before.kind != "file":
        raise ValueError(f"Path '{path}' is not a regular file.")
    descriptor = open_selected_file(Path(path.anchor), path, (before.device, before.inode))
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        current = (opened.st_dev, opened.st_ino, opened.st_ctime_ns, opened.st_nlink)
        if (
            current != (before.device, before.inode, before.ctime_ns, before.link_count)
            or current != identity
        ):
            raise PermissionError("Selected file or aliases changed before reading.")
        content = stream.read()
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
        expected_ctime_ns=entry.ctime_ns,
        expected_link_count=entry.link_count,
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
    """Bind metadata before authorizing the prerequisite content inspection."""
    try:
        details = Path(path).stat()
    except FileNotFoundError:
        target = FileTarget(path=path)
    else:
        target = FileTarget(
            path=path,
            expected_device=details.st_dev,
            expected_inode=details.st_ino,
            expected_ctime_ns=details.st_ctime_ns,
            expected_link_count=details.st_nlink,
        )
    return Operation(tool_id="", action=action, target=target)


def _write_plan(arguments: dict[str, object]) -> OperationPlan:
    """Authorize target inspection before planning a create or replacement."""
    path = _lexical_mutation_path(Path(str(arguments["path"])))
    resource = str(path)
    normalized = {**arguments, "path": resource}
    boundaries = tuple(
        _mutation_operation(action, FileTarget(path=resource))
        for action in (Action.FILESYSTEM_CREATE, Action.FILESYSTEM_REPLACE)
    )
    read = _read_operation(Action.FILESYSTEM_READ, resource)
    return OperationPlan(
        arguments=normalized,
        operations=(read,),
        boundary_operations=boundaries,
        continuation=lambda: _finish_write_plan(normalized, read.target),
    )


def _finish_write_plan(arguments: dict[str, object], selected: FileTarget) -> OperationPlan:
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
    entry, existing_bytes = _read_regular_entry(
        path,
        ".",
        (
            selected.expected_device,
            selected.expected_inode,
            selected.expected_ctime_ns,
            selected.expected_link_count,
        ),
    )
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
    read = _read_operation(Action.FILESYSTEM_READ, resource)
    return OperationPlan(
        arguments=normalized,
        operations=(read,),
        boundary_operations=(boundary,),
        continuation=lambda: _finish_edit_plan(normalized, read.target),
    )


def _finish_edit_plan(arguments: dict[str, object], selected: FileTarget) -> OperationPlan:
    """Capture the authorized edit source before validating model-supplied anchors."""
    path = _mutation_path(Path(str(arguments["path"])))
    resource = str(path)
    normalized = {**arguments, "path": resource}
    try:
        if _path_kind(path) != "file":
            raise _edit_problem(
                "filesystem.path_not_file", f"Path '{path}' is not an existing regular file."
            )
        entry, original_bytes = _read_regular_entry(
            path,
            ".",
            (
                selected.expected_device,
                selected.expected_inode,
                selected.expected_ctime_ns,
                selected.expected_link_count,
            ),
        )
    except OSError as exc:
        raise _edit_problem("filesystem.edit_failed", str(exc)) from exc

    target = _target_from_entry(resource, entry)
    return OperationPlan(
        arguments=normalized,
        boundary_operations=(_mutation_operation(Action.FILESYSTEM_REPLACE, target),),
        continuation=lambda: _edit_preview_plan(normalized, target, original_bytes),
    )


def _edit_preview_plan(
    arguments: dict[str, object],
    target: FileTarget,
    original_bytes: bytes,
) -> OperationPlan:
    """Validate an edit and prepare consent only after its source preconditions are checked."""
    try:
        original = original_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _edit_problem(
            "filesystem.binary_file",
            f"File '{target.path}' is not valid UTF-8 text and cannot be edited with this tool.",
        ) from exc
    updated, _ = _edited_content(
        original,
        str(arguments["old_content"]),
        str(arguments["new_content"]),
        bool(arguments["replace_all"]),
    )
    return OperationPlan(
        arguments=arguments,
        operations=(
            _mutation_operation(
                Action.FILESYSTEM_REPLACE,
                target,
                reason=f"Proposed changes:\n{format_content_diff(original, updated, target.path)}",
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
        read = _read_operation(Action.FILESYSTEM_READ, resource)
        return OperationPlan(
            arguments=normalized,
            operations=(read,),
            boundary_operations=(boundary,),
            continuation=lambda: _finish_file_delete_plan(normalized, reason, read.target),
        )
    return OperationPlan(
        arguments=normalized,
        operations=(_read_operation(Action.FILESYSTEM_LIST, resource),),
        boundary_operations=(boundary,),
        continuation=lambda: _directory_metadata_plan(normalized, reason),
    )


def _finish_file_delete_plan(
    arguments: dict[str, object], reason: str, selected: FileTarget
) -> OperationPlan:
    """Capture an authorized regular-file deletion capability."""
    path = Path(str(arguments["path"]))
    entry = _capture_entry(
        path,
        ".",
        digest=True,
        identity=(
            selected.expected_device,
            selected.expected_inode,
            selected.expected_ctime_ns,
            selected.expected_link_count,
        ),
    )
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
        selected = path if expected.relative_path == "." else path / expected.relative_path
        current = _capture_entry(
            selected,
            expected.relative_path,
            digest=False,
        )
        if current != expected:
            raise _planning_problem(
                "delete_path",
                "filesystem.target_changed",
                "Could not delete path",
                "The target changed during planning; deletion was cancelled.",
            )
        if expected.kind == "file":
            current = _capture_entry(
                selected,
                expected.relative_path,
                digest=True,
                identity=(expected.device, expected.inode, expected.ctime_ns, expected.link_count),
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
        current = _capture_entry(
            path,
            ".",
            digest=target.expected_digest is not None,
            identity=(
                target.expected_device,
                target.expected_inode,
                target.expected_ctime_ns,
                target.expected_link_count,
            ),
        )
    except (OSError, ValueError):
        return False
    return (
        current.kind == target.expected_kind
        and current.device == target.expected_device
        and current.inode == target.expected_inode
        and current.size == target.expected_size
        and current.mtime_ns == target.expected_mtime_ns
        and current.ctime_ns == target.expected_ctime_ns
        and current.link_count == target.expected_link_count
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
    executor additionally binds parent descriptors and deletes only approved manifest entries.
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
                identity=(entry.device, entry.inode, entry.ctime_ns, entry.link_count),
            )
            for entry in current_metadata
        )
    except (OSError, ValueError):
        return False
    return completed == target.manifest


def _delete_selected(path: Path, target: FileTarget) -> None:
    """Delete only manifest-selected names through identity-bound parent descriptors."""
    entries = (
        target.manifest
        if target.recursive
        else (
            FileManifestEntry(
                relative_path=".",
                kind=target.expected_kind,
                device=target.expected_device,
                inode=target.expected_inode,
                size=target.expected_size,
                mtime_ns=target.expected_mtime_ns,
                ctime_ns=target.expected_ctime_ns,
                link_count=target.expected_link_count,
                mode=target.expected_mode,
                digest=target.expected_digest,
            ),
        )
    )
    parents = {entry.relative_path: (entry.device, entry.inode) for entry in entries}
    removed_alias_states = {}
    for entry in reversed(entries):
        selected = path if entry.relative_path == "." else path / entry.relative_path
        parent_identity = (
            (target.expected_parent_device, target.expected_parent_inode)
            if entry.relative_path == "."
            else parents[Path(entry.relative_path).parent.as_posix()]
        )
        with open_selected_parent(selected, parent_identity) as directory:
            current = os.stat(selected.name, dir_fd=directory, follow_symlinks=False)
            if (current.st_dev, current.st_ino, current.st_mode) != (
                entry.device,
                entry.inode,
                entry.mode,
            ):
                raise RuntimeError("The target changed after approval; deletion was cancelled.")
            if entry.kind == "directory":
                os.rmdir(selected.name, dir_fd=directory)
            elif entry.kind == "file":
                identity = (entry.device, entry.inode)
                expected_state = removed_alias_states.get(
                    identity, (entry.ctime_ns, entry.link_count)
                )
                descriptor = open_selected_file(Path(selected.anchor), selected, identity)
                try:
                    opened = os.fstat(descriptor)
                    if (opened.st_ctime_ns, opened.st_nlink) != expected_state:
                        raise PermissionError("Selected file or aliases changed before deletion.")
                    os.unlink(selected.name, dir_fd=directory)
                    # Unlinking one approved name changes the shared inode's change time.
                    # Bind that known effect before removing its next approved alias.
                    remaining = os.fstat(descriptor)
                    removed_alias_states[identity] = (remaining.st_ctime_ns, remaining.st_nlink)
                finally:
                    os.close(descriptor)
            else:
                os.unlink(selected.name, dir_fd=directory)


@tool(
    actions={Action.FILESYSTEM_LIST},
    operation_planner=_file_plan(Action.FILESYSTEM_LIST),
    result_presentation=ToolResultPresentationSpec(kind=ToolResultPresentation.TREE),
)
def list_folder(
    context: ToolContext,
    path: Annotated[
        str,
        VirtualPath.METADATA_KEY,
        Field(description=f"Folder to list; use {VirtualPath.TEMPORARY}/ for session files."),
    ],
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
        requested = Path(path)
        if not _readable(context, requested, Action.FILESYSTEM_LIST):
            raise PermissionError("File policy no longer permits listing this folder.")
        folder = requested.resolve()
        read_ignore = partial(_read_ignore, context)
        if is_path_ignored(folder, read_ignore=read_ignore):
            return Problem(
                code="filesystem.path_ignored",
                title="Folder cannot be listed",
                detail=f"Path '{path}' is ignored.",
                operation="list_folder",
            )
        entries = iter_visible_paths(
            folder,
            recursive,
            read_ignore=read_ignore,
            allowed=lambda entry: _readable(context, entry, Action.FILESYSTEM_LIST),
        )
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
        if not _readable(context, requested, Action.FILESYSTEM_LIST):
            raise PermissionError("File policy no longer permits listing this folder.")
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
    result_path_fields=(("result", "path"),),
)
def read_text_file(
    context: ToolContext,
    path: Annotated[
        str,
        VirtualPath.METADATA_KEY,
        Field(
            description="Path to the text file to read. "
            f"Use {VirtualPath.TEMPORARY}/ for session files."
        ),
    ],
    start_line: Annotated[
        int | None,
        Field(
            description="One-based starting line. When continuing by byte offset, use JSON "
            "null (unquoted).",
            ge=1,
        ),
        BeforeValidator(normalize_optional_null),
    ] = 1,
    start_byte: Annotated[
        int | None,
        Field(
            description="Byte offset from next_start_byte when a line is split. "
            "For line-based reads, use JSON null (unquoted). With a later starting line, "
            "a zero byte offset selects the line.",
            ge=0,
        ),
        BeforeValidator(normalize_optional_null),
    ] = None,
    max_lines: Annotated[
        int | None,
        Field(
            description="Line ceiling; the first reached line or byte limit wins. "
            "For no line ceiling, use JSON null (unquoted).",
            ge=1,
        ),
        BeforeValidator(normalize_optional_null),
    ] = None,
    max_bytes: Annotated[
        int,
        Field(
            description="Requested UTF-8 byte limit; the first character or line may increase it "
            "up to the application ceiling.",
            ge=1,
            le=constants.MAX_TOOL_CONTENT_BYTES,
        ),
    ] = constants.MAX_TOOL_CONTENT_BYTES,
) -> FileContentResult | Problem:
    """Read a bounded UTF-8 text range with its complete source version."""
    try:
        file_path = Path(path)
        with os.fdopen(_open_read(context, file_path), "rb") as stream:
            result = FileContentResult(
                path=path,
                **read_bounded_text_stream(
                    stream,
                    start_byte=start_byte,
                    start_line=None if start_byte is not None and start_line == 1 else start_line,
                    max_lines=max_lines,
                    max_bytes=max_bytes,
                    preserve_line_boundaries=True,
                    include_version=True,
                ),
            )
        if not _readable(context, file_path):
            raise PermissionError("File policy changed while reading.")
        context.record_file_read(file_path, result["version"])
        context.observe_file(path)
        return result
    except FileNotFoundError:
        context.record_file_read(Path(path), None)
        return Problem(
            code="filesystem.path_missing",
            title="File does not exist",
            severity="warning",
            detail=f"File '{context.display_path(path)}' does not exist.",
            operation="read_text_file",
        )
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
    result_path_fields=(("result", "matches", "*", "path"),),
    requires_execution_service=True,
)
def search_text(
    context: ToolContext,
    path: Annotated[
        str,
        VirtualPath.METADATA_KEY,
        Field(
            description="Path to a text file or folder to search. "
            f"Use {VirtualPath.TEMPORARY}/ for session files."
        ),
    ],
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
        Field(
            description="Inclusive Git-style file globs. "
            "Use JSON null (unquoted) to search all visible files.",
            max_length=20,
        ),
        BeforeValidator(normalize_optional_null),
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
        requested = Path(path)
        if not _readable(context, requested):
            raise PermissionError("File policy no longer permits searching this path.")
        target = requested.resolve()
        read_ignore = partial(_read_ignore, context)
        if is_path_ignored(target, read_ignore=read_ignore):
            return Problem(
                code="filesystem.path_ignored",
                title="Path cannot be searched",
                detail=f"Path '{path}' is ignored.",
                operation="search_text",
            )
        if target.is_file():
            root = target.parent
            candidates = [target]
        elif target.is_dir():
            root = target
            candidates = [
                entry
                for entry in iter_visible_paths(
                    target,
                    True,
                    read_ignore=read_ignore,
                    allowed=lambda candidate: _readable(context, candidate),
                )
                if entry.is_file() and not entry.is_symlink()
            ]
        else:
            return Problem(
                code="filesystem.path_not_searchable",
                title="Path cannot be searched",
                detail=f"Path '{path}' is not an existing file or folder.",
                operation="search_text",
            )
        candidates = [
            entry
            for entry in filter_paths_by_globs(candidates, root, include)
            if _searchable(context, entry)
        ]
        matches, truncated = search_text_paths(
            candidates,
            query,
            root=root,
            regex=regex,
            case=case,
            context_lines=context_lines,
            max_results=max_results,
            max_bytes=max_bytes,
            opener=lambda path: _open_read(context, path),
            native_runner=(
                context.execution_service.run_trusted_search
                if context.execution_service is not None
                else None
            ),
        )
        if not _readable(context, requested):
            raise PermissionError("File policy no longer permits searching this path.")
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
    path: Annotated[
        str,
        VirtualPath.METADATA_KEY,
        Field(
            description="Path to the text file to write. "
            f"Use {VirtualPath.TEMPORARY}/ for session files."
        ),
    ],
    content: Annotated[str, Field(description="Content to write to the file.")],
) -> str | Problem:
    """Write content to a file on the local disk."""
    try:
        target = Path(path)
        operation = context.operations[0] if context.operations else None
        planned = operation.target if operation is not None else None
        if not isinstance(planned, FileTarget) or planned.expected_exists is None:
            raise RuntimeError("Authorized file-state precondition is missing.")
        if denial := _mutation_denial(context):
            return denial
        write_text_atomically(
            Path(planned.path),
            content,
            expected_digest=planned.expected_digest if planned.expected_exists else None,
            expected_device=planned.expected_device,
            expected_inode=planned.expected_inode,
            expected_ctime_ns=planned.expected_ctime_ns,
            expected_link_count=planned.expected_link_count,
            expected_mode=planned.expected_mode,
            expected_parent_device=planned.expected_parent_device,
            expected_parent_inode=planned.expected_parent_inode,
        )
        context.record_file_version(Path(planned.path), content)
        context.observe_file(target)
        context.invalidate_instructions(target)
        return f"Successfully wrote to file '{context.display_path(path)}'."
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
    path: Annotated[
        str,
        VirtualPath.METADATA_KEY,
        Field(description=f"UTF-8 file to edit; use {VirtualPath.TEMPORARY}/ for session files."),
    ],
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
        if denial := _mutation_denial(context):
            return denial
        target = Path(planned.path)
        if not _matches_target(target, planned):
            raise RuntimeError("The target changed after approval; replacement was cancelled.")
        _, current_bytes = _read_regular_entry(
            target,
            ".",
            (
                planned.expected_device,
                planned.expected_inode,
                planned.expected_ctime_ns,
                planned.expected_link_count,
            ),
        )
        updated, replacement_count = _edited_content(
            current_bytes.decode("utf-8"), old_content, new_content, replace_all
        )
        write_text_atomically(
            target,
            updated,
            expected_digest=planned.expected_digest,
            expected_device=planned.expected_device,
            expected_inode=planned.expected_inode,
            expected_ctime_ns=planned.expected_ctime_ns,
            expected_link_count=planned.expected_link_count,
            expected_mode=planned.expected_mode,
            expected_parent_device=planned.expected_parent_device,
            expected_parent_inode=planned.expected_parent_inode,
        )
        context.record_file_read(target, f"sha256:{planned.expected_digest}")
        context.record_file_version(target, updated)
        context.observe_file(target)
        context.invalidate_instructions(target)
        noun = "replacement" if replacement_count == 1 else "replacements"
        return (
            f"Successfully edited file '{context.display_path(path)}' ({replacement_count} {noun})."
        )
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
        VirtualPath.METADATA_KEY,
        Field(
            description="File, link, or folder to delete; use "
            f"{VirtualPath.TEMPORARY}/ for session files."
        ),
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
        if denial := _mutation_denial(context):
            return denial
        target = Path(planned.path)
        if planned.recursive:
            if planned.expected_kind != "directory" or not _matches_manifest(target, planned):
                raise RuntimeError("The target changed after approval; deletion was cancelled.")
            _delete_selected(target, planned)
        elif planned.expected_kind in {"file", "symlink"}:
            if not _matches_target(target, planned):
                raise RuntimeError("The target changed after approval; deletion was cancelled.")
            _delete_selected(target, planned)
            context.record_file_removal(target)
        else:
            raise RuntimeError("Authorized deletion capability has an unsupported target kind.")
        context.invalidate_instructions(target)
        return f"Successfully deleted path '{context.display_path(target)}'."
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
        problem = Problem.from_exception(
            exc,
            code="filesystem.delete_failed",
            title="Could not delete path",
            operation="delete_path",
        )
        log_problem(_LOGGER, problem, exc)
        return problem
