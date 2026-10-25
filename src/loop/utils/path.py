"""Provide private directory preparation and repository-aware path utilities."""

import ctypes
import errno
import os
import stat
import sys
from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

from pathspec import GitIgnoreSpec

from .. import constants
from .models import IgnoreRule, IgnoreRules

_DARWIN_EXTENDED_ACL_TYPE = 0x100
_DARWIN_FIRST_ACL_ENTRY = 0


def ensure_private_directory(
    path: Path,
    *,
    create: bool = True,
    anchor: tuple[Path, int] | None = None,
) -> None:
    """Create or validate a private directory without following symbolic path components.

    Create missing parents with mode ``0700`` using no-follow, directory-relative operations.
    Require the selected directory to have mode ``0700`` and belong to the current user.
    Refuse unsafe existing directories without changing them, and verify the final path still
    names the opened directory.

    Args:
        path (Path): Absolute canonical directory path to create or validate.
        create (bool): Create missing components when true; otherwise require existing storage.
        anchor (tuple[Path, int] | None): Approved directory pathname and open descriptor used
            instead of traversing from ``/``. The caller retains ownership of the descriptor.

    Raises:
        ValueError: The path is relative, aliased, not a directory, not private, or owned by
            another user.
        OSError: Directory creation or inspection fails.
    """
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("Path must be a private real directory.")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    components = path.parts[1:]
    if anchor is None:
        descriptor = os.open(path.anchor, flags)
    else:
        root, approved = anchor
        current = root.lstat()
        bound = os.fstat(approved)
        if (
            not root.is_absolute()
            or root.resolve(strict=True) != root
            or not path.is_relative_to(root)
            or not stat.S_ISDIR(bound.st_mode)
            or (bound.st_dev, bound.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise ValueError("Approved directory authority changed.")
        components = path.relative_to(root).parts
        descriptor = os.dup(approved)
    try:
        for component in components:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=constants.PRIVATE_DIRECTORY_MODE, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        bound = os.fstat(descriptor)
        details = path.lstat()
        if (
            path.resolve(strict=True) != path
            or not stat.S_ISDIR(details.st_mode)
            or details.st_uid != os.getuid()
            or stat.S_IMODE(details.st_mode) != constants.PRIVATE_DIRECTORY_MODE
            or (details.st_dev, details.st_ino) != (bound.st_dev, bound.st_ino)
            or _has_directory_acl(descriptor)
        ):
            raise ValueError("Path must be a private real directory.")
    finally:
        os.close(descriptor)


def _has_directory_acl(descriptor: int) -> bool:
    """Refuse extended access lists on harness-owned private directories."""
    if sys.platform == "darwin":
        native = ctypes.CDLL(None, use_errno=True)
        native.acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
        native.acl_get_fd_np.restype = ctypes.c_void_p
        native.acl_get_entry.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        native.acl_free.argtypes = [ctypes.c_void_p]
        acl = native.acl_get_fd_np(descriptor, _DARWIN_EXTENDED_ACL_TYPE)
        if not acl:
            if ctypes.get_errno() == errno.ENOENT:
                return False
            raise OSError(ctypes.get_errno(), "Cannot validate private directory access list.")
        try:
            entry = ctypes.c_void_p()
            result = native.acl_get_entry(acl, _DARWIN_FIRST_ACL_ENTRY, ctypes.byref(entry))
            if result < 0 and ctypes.get_errno() != errno.EINVAL:
                raise OSError(ctypes.get_errno(), "Cannot inspect private directory access list.")
            return result == 0
        finally:
            native.acl_free(acl)
    try:
        return bool(os.getxattr(descriptor, "system.posix_acl_access"))
    except OSError as error:
        if error.errno not in (errno.ENODATA, errno.ENOTSUP):
            raise
        return False


@contextmanager
def private_directory_lock(directory: Path, name: str) -> Generator[int, None, None]:
    """Hold a nonblocking native advisory lock outside an untrusted writable leaf.

    Args:
        directory (Path): Existing harness-owned private lock directory.
        name (str): Single lock filename controlled by the harness.

    Yields:
        int: Close-on-exec lock descriptor, retained until the context settles.

    Raises:
        BlockingIOError: Another cooperating owner holds this lock.
        ValueError: The directory, filename, or existing lock is unsafe.
        OSError: Native inspection or locking fails.
    """
    try:
        locking = import_module("fcntl")
    except ModuleNotFoundError as error:
        raise OSError(errno.ENOTSUP, "Native session ownership requires macOS or Linux.") from error
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Invalid ownership lock name.")
    ensure_private_directory(directory, create=False)
    parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        descriptor = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            constants.PRIVATE_FILE_MODE,
            dir_fd=parent,
        )
        try:
            bound = os.fstat(descriptor)
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if (
                not stat.S_ISREG(bound.st_mode)
                or bound.st_uid != os.getuid()
                or stat.S_IMODE(bound.st_mode) != constants.PRIVATE_FILE_MODE
                or bound.st_nlink != 1
                or (bound.st_dev, bound.st_ino) != (current.st_dev, current.st_ino)
            ):
                raise ValueError("Unsafe ownership lock.")
            locking.flock(descriptor, locking.LOCK_EX | locking.LOCK_NB)
            yield descriptor
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def canonical_path(path: Path | str) -> str:
    """Return an absolute normalized path without requiring it to exist.

    Args:
        path (Path | str): Filesystem path to normalize.

    Returns:
        str: Absolute canonical representation.
    """
    candidate = Path(path)
    if candidate.exists() or candidate.is_symlink():
        return str(candidate.resolve())
    return str(candidate.parent.resolve() / candidate.name)


def find_project_root(working_directory: Path | str) -> Path | None:
    """Return the closest Git project root containing the working directory.

    Args:
        working_directory (Path | str): Directory from which to search upward.

    Returns:
        Path | None: The closest directory containing a ``.git`` marker, or ``None`` when no
        project root is found.
    """
    working_directory = Path(working_directory)
    for directory in (working_directory, *working_directory.parents):
        if (directory / constants.GIT_DIRECTORY).exists():
            return directory
    return None


def _load_ignore_rules(directory: Path, rules: IgnoreRules) -> None:
    """Add ignore files in a directory to their active hierarchical rule sets."""
    for filename in constants.IGNORE_FILENAMES:
        ignore_file = directory / filename
        if ignore_file.is_file():
            rules[filename].append(
                (
                    directory,
                    GitIgnoreSpec.from_lines(ignore_file.read_text(encoding="utf-8").splitlines()),
                )
            )


def _initial_ignore_rules(folder: Path) -> IgnoreRules:
    """Load ignore rules in scope from the Git project root through a folder."""
    rules = {filename: [] for filename in constants.IGNORE_FILENAMES}
    root = find_project_root(folder) or folder
    relative_folder = folder.relative_to(root)
    directory = root
    _load_ignore_rules(directory, rules)
    for part in relative_folder.parts:
        directory /= part
        _load_ignore_rules(directory, rules)
    return rules


def _ignore_decision(path: Path, is_directory: bool, rules: list[IgnoreRule]) -> bool | None:
    """Return the last matching decision from one hierarchical ignore source."""
    decision = None
    for base, spec in rules:
        relative_path = path.relative_to(base).as_posix()
        if is_directory:
            relative_path += "/"
        result = spec.check_file(relative_path)
        if result.include is not None:
            decision = result.include
    return decision


def _is_ignored(path: Path, is_directory: bool, rules: IgnoreRules) -> bool:
    """Evaluate a path against already-loaded Git and agent ignore rules."""
    if is_directory and path.name in {
        constants.GIT_DIRECTORY.name,
        constants.APP_DIRECTORY.name,
    }:
        return True

    agent_decision = _ignore_decision(path, is_directory, rules[constants.AGENT_IGNORE_FILENAME])
    if agent_decision is not None:
        return agent_decision
    return _ignore_decision(path, is_directory, rules[constants.GIT_IGNORE_FILENAME]) is True


def is_path_ignored(path: Path | str, root: Path | str | None = None) -> bool:
    """Return whether an explicit path is excluded by scoped ignore files.

    Rules are loaded from the supplied root through the path's parent. When no
    root is supplied, the closest Git project root is used, falling back to the
    path's parent directory outside a repository. Ignored parent directories
    stop rule discovery, matching recursive traversal behavior.

    Args:
        path (Path | str): File or directory to evaluate.
        root (Path | str | None): Optional boundary for hierarchical ignore-file discovery.

    Returns:
        bool: Whether ``.gitignore``, higher-priority ``.agentignore``, or Git metadata
        exclusion hides the path.

    Raises:
        ValueError: If the path is outside the supplied root.
    """
    path = Path(path).resolve()
    root = Path(root).resolve() if root is not None else find_project_root(path.parent)
    root = root or path.parent
    relative_path = path.relative_to(root)
    rules = {filename: [] for filename in constants.IGNORE_FILENAMES}
    directory = root
    _load_ignore_rules(directory, rules)

    for index, part in enumerate(relative_path.parts):
        candidate = directory / part
        is_directory = candidate.is_dir()
        if _is_ignored(candidate, is_directory, rules):
            return True
        directory = candidate
        if index < len(relative_path.parts) - 1 and is_directory:
            _load_ignore_rules(directory, rules)
    return False


def _iter_visible_paths(folder: Path, recursive: bool, rules: IgnoreRules) -> Iterator[Path]:
    """Yield visible child paths, pruning ignored directories during recursion."""
    for entry in folder.iterdir():
        is_directory = entry.is_dir()
        if _is_ignored(entry, is_directory, rules):
            continue
        yield entry
        if recursive and is_directory and not entry.is_symlink():
            nested_rules = {name: active.copy() for name, active in rules.items()}
            _load_ignore_rules(entry, nested_rules)
            yield from _iter_visible_paths(entry, recursive=True, rules=nested_rules)


def iter_visible_paths(folder: Path | str, recursive: bool = False) -> Iterator[Path]:
    """Yield non-ignored paths immediately below a folder or throughout its tree.

    Hierarchical ``.gitignore`` and ``.agentignore`` files use Git pattern syntax,
    with agent-specific decisions taking precedence. Git metadata directories are
    always omitted, ignored directories are not traversed, and symbolic-link
    directories are listed without being followed.

    Args:
        folder (Path | str): Directory whose visible children should be discovered.
        recursive (bool): Whether to traverse visible child directories recursively.

    Yields:
        Path: Visible files and directories as absolute paths.
    """
    folder = Path(folder).resolve()
    yield from _iter_visible_paths(folder, recursive, _initial_ignore_rules(folder))


def filter_paths_by_globs(
    paths: Iterable[Path], root: Path | str, patterns: Sequence[str] | None
) -> Iterator[Path]:
    """Yield paths whose root-relative names match any supplied Git-style glob.

    Args:
        paths (Iterable[Path]): Candidate paths to filter.
        root (Path | str): Root used to derive portable relative names.
        patterns (Sequence[str] | None): Inclusive Git-style globs, or ``None`` to include all.

    Yields:
        Path: Candidate paths selected by at least one pattern, or every path without patterns.
    """
    if not patterns:
        yield from paths
        return
    spec = GitIgnoreSpec.from_lines(patterns)
    root = Path(root).resolve()
    for path in paths:
        if spec.match_file(path.resolve().relative_to(root).as_posix()):
            yield path
