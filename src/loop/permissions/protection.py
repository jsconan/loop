"""Define protected workspace paths shared by permission and native policy."""

from __future__ import annotations

import os
import stat
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .. import constants

PRIVATE_READ_COMPONENTS = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".config",
    constants.APP_DIRECTORY.as_posix(),
    "Library",
)
_PRIVATE_READ_FOLDED = frozenset(name.casefold() for name in PRIVATE_READ_COMPONENTS)
ENVIRONMENT_TEMPLATE_SUFFIXES = (".example", ".sample", ".template")
_MAX_ALIAS_ENTRIES = 100_000
_MAX_SCRATCH_RESCANS = 2


def is_environment_value(name: str) -> bool:
    """Identify environment-value names while allowing named public templates.

    Args:
        name (str): One path component, without directory separators.

    Returns:
        bool: Whether the name has the built-in environment-value hard denial.
    """
    folded = name.casefold()
    return folded == ".env" or (
        folded.startswith(".env.") and not folded.endswith(ENVIRONMENT_TEMPLATE_SUFFIXES)
    )


def protected_environment_suffixes(words: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Partition private environment names by the trailing characters of public suffixes.

    Args:
        words (Iterable[str]): Template suffix words in the consumer's filename encoding.

    Yields:
        tuple[str, str]: A literal tail and excluded preceding characters. An empty
            exclusion denotes an exact tail.
    """
    tree: dict = {}
    for word in words:
        node = tree
        for character in reversed(word):
            node = node.setdefault(character, {})
        node["exact"] = True
        node["."] = {"stop": True}

    def tails(node: dict, suffix: str) -> Iterator[tuple[str, str]]:
        """Visit all endings outside the public-template suffixes."""
        if node.get("stop"):
            return
        if not node.get("exact"):
            yield suffix, ""
        children = sorted(key for key in node if key != "exact")
        yield suffix, "".join(children)
        for character in children:
            yield from tails(node[character], character + suffix)

    yield from tails(tree, "")


def validate_hard_denied_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    """Validate bounded exact path designations without interpreting glob syntax.

    Args:
        paths (tuple[str, ...]): Absolute file or directory names from trusted policy.

    Returns:
        tuple[str, ...]: Validated designations, including future paths.

    Raises:
        ValueError: A designation is relative, non-normalized, patterned or exceeds the limit.
    """
    if len(paths) > 256 or any(
        not Path(path).is_absolute()
        or Path(path).as_posix() != path
        or ".." in Path(path).parts
        or any(character in path for character in "\x00\n\r*?[]{}\\")
        for path in paths
    ):
        raise ValueError("Hard-denied paths require at most 256 normalized absolute exact paths.")
    return paths


def is_hard_resource(
    path: Path,
    *,
    workspace: Path | None,
    scratch: Path,
    scratch_data: bool = False,
) -> bool:
    """Classify built-in private data relative to the admitted host-data scope.

    Args:
        path (Path): Absolute candidate path.
        workspace (Path | None): Active workspace, or None when no workspace is selected.
        scratch (Path): Root of private session scratch.
        scratch_data (bool): Whether the session owner permits ordinary data at private names.

    Returns:
        bool: Whether a private component or environment-value name protects this resource.
    """
    if path.is_relative_to(scratch):
        if scratch_data:
            return False
        relative = path.relative_to(scratch)
    else:
        relative = (
            path.relative_to(workspace) if workspace and path.is_relative_to(workspace) else path
        )
    return any(
        part.casefold() in _PRIVATE_READ_FOLDED or is_environment_value(part)
        for part in relative.parts
    )


class IncompleteHardlinkAliases(ValueError):
    """Report an inode with names outside the inspected data view.

    Args:
        detail (str): Optional receiver guidance for resolving this refused admission.
    """

    def __init__(self, detail: str = "") -> None:
        message = "Incomplete hardlink aliases in the permitted data view."
        super().__init__(f"{message} {detail}" if detail else message)


@contextmanager
def _open_peer_directory(path: Path, flags: int) -> Generator[int, None, None]:
    """Open each path component without following symlinks, keeping one parent pinned."""
    search_flags = flags | getattr(os, "O_SEARCH", getattr(os, "O_PATH", os.O_RDONLY))
    descriptor = os.open(path.anchor, search_flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, search_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        directory = os.open(".", flags, dir_fd=descriptor)
        try:
            yield directory
        finally:
            os.close(directory)
    finally:
        os.close(descriptor)


def inspect_hardlink_aliases(
    roots: tuple[Path, ...],
    deadline: float,
    *,
    max_entries: int = _MAX_ALIAS_ENTRIES,
    unaccounted: set[Path] | None = None,
    scratch: Path | None = None,
    hardlink_sources: tuple[Path, ...] = (),
    denied_peer: Callable[[Path], bool] | None = None,
) -> dict[Path, tuple[Path, ...]]:
    """Account for regular-file links within a bounded, stable no-follow root inventory.

    This snapshot does not prevent later topology changes. Native callers must enforce link
    and relocation rules; direct callers must bind the selected identity again at opening.

    Args:
        roots (tuple[Path, ...]): Canonical directories constituting the permitted data view.
        deadline (float): Absolute monotonic time at which inspection must stop.
        max_entries (int): Maximum total directory entries across all scans. Defaults to the
            safety ceiling.
        unaccounted (set[Path] | None): Collect all visible names of incomplete alias sets.
            The caller must mask those names before access. Omitting this option refuses them.
        scratch (Path | None): Bound private scratch directory. Concurrent changes to its
            membership can restart the entire inventory within the original deadline and entry
            limit. Other changes still refuse admission. Defaults to no resumable directory.
        hardlink_sources (tuple[Path, ...]): Approved external directories to search for missing
            names by metadata only. Their contents remain outside the permitted data view.
        denied_peer (Callable[[Path], bool] | None): Additional policy denials and native mount
            boundaries for external candidates. Built-in private names are always excluded.

    Returns:
        dict[Path, tuple[Path, ...]]: Every accounted multiply linked path mapped to all names,
            including external peers. An external name confers no command access.

    Raises:
        ValueError: A root or entry changes, aliases are incomplete, or the bound is exhausted.
        TimeoutError: The inspection deadline expires.
        OSError: A required directory or entry cannot be inspected without following a link.
    """
    selected = []
    for root in sorted(set(roots), key=lambda path: (len(path.parts), str(path))):
        if not root.is_absolute() or root.resolve(strict=True) != root:
            raise ValueError("Alias inventory requires canonical absolute roots.")
        if not any(root.is_relative_to(parent) for parent in selected):
            selected.append(root)

    def directory_identity(details: os.stat_result) -> tuple[int, ...]:
        """Pin the object, ownership and access mode independently of directory membership."""
        return details.st_dev, details.st_ino, details.st_uid, details.st_gid, details.st_mode

    scratch_identity = directory_identity(scratch.lstat()) if scratch is not None else None
    remaining = max_entries
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

    def consume_entry():
        """Share the original deadline and entry limit across every walk and retry."""
        nonlocal remaining
        if time.monotonic() >= deadline:
            raise TimeoutError("Alias inventory deadline expired.")
        remaining -= 1
        if remaining < 0:
            raise ValueError("Alias inventory entry bound exhausted.")

    def record(path, details, parent):
        """Count a physical directory entry once even when its parent has another spelling."""
        regular[path] = details
        if details.st_nlink > 1:
            names = identities.setdefault((details.st_dev, details.st_ino), {})
            key = parent.st_dev, parent.st_ino, path.name
            if key in names:
                raise ValueError("Alias inventory contains duplicate directory entries.")
            names[key] = path

    def inspect(directory, descriptor):
        """Traverse pinned directories and retain membership and regular-file identities."""
        before = os.fstat(descriptor)
        current = directory.lstat()
        if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("Alias inventory directory changed.")
        directories[directory] = before
        with os.scandir(descriptor) as entries:
            for entry in entries:
                consume_entry()
                path = directory / entry.name
                details = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(details.st_mode):
                    child = os.open(entry.name, flags, dir_fd=descriptor)
                    try:
                        opened = os.fstat(child)
                        if (opened.st_dev, opened.st_ino) != (details.st_dev, details.st_ino):
                            raise ValueError("Alias inventory directory changed.")
                        inspect(path, child)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(details.st_mode):
                    if details.st_nlink > 1 and (
                        details.st_dev != before.st_dev or details.st_ino != entry.inode()
                    ):
                        raise ValueError("Alias inventory requires physical directory entries.")
                    record(path, details, before)

    def find_peers():
        """Visit approved source directories breadth first until every visible inode is complete."""
        missing = {
            identity
            for identity, names in identities.items()
            if len(names) != regular[next(iter(names.values()))].st_nlink
        }
        seen = {(details.st_dev, details.st_ino) for details in directories.values()}
        pending = deque((path, None) for path in hardlink_sources)
        while missing and pending:
            if time.monotonic() >= deadline:
                raise TimeoutError("Alias inventory deadline expired.")
            directory, expected = pending.popleft()
            if not directory.is_absolute() or ".." in directory.parts:
                raise ValueError("Alias inventory requires canonical absolute sources.")
            if any(
                part.casefold() in _PRIVATE_READ_FOLDED or is_environment_value(part)
                for part in directory.parts
            ) or (denied_peer is not None and denied_peer(directory)):
                continue
            with _open_peer_directory(directory, flags) as descriptor:
                before = os.fstat(descriptor)
                if expected is not None and not os.path.samestat(expected, before):
                    raise ValueError("Alias inventory directory changed.")
                key = before.st_dev, before.st_ino
                if key in seen:
                    continue
                seen.add(key)
                directories[directory] = before
                with os.scandir(descriptor) as entries:
                    for entry in entries:
                        consume_entry()
                        path = directory / entry.name
                        if (
                            entry.name.casefold() in _PRIVATE_READ_FOLDED
                            or is_environment_value(entry.name)
                            or denied_peer is not None
                            and denied_peer(path)
                        ):
                            continue
                        details = entry.stat(follow_symlinks=False)
                        if details.st_dev != before.st_dev or details.st_ino != entry.inode():
                            continue
                        if stat.S_ISDIR(details.st_mode):
                            pending.append((path, details))
                        elif (details.st_dev, details.st_ino) in missing:
                            record(path, details, before)
                            identity = details.st_dev, details.st_ino
                            if len(identities[identity]) == details.st_nlink:
                                missing.remove(identity)
                                if not missing:
                                    break

    rescans = 0
    while True:
        identities = {}
        directories = {}
        regular = {}
        changed = False
        for root in selected:
            if time.monotonic() >= deadline:
                raise TimeoutError("Alias inventory deadline expired.")
            descriptor = os.open(root, flags)
            try:
                inspect(root, descriptor)
            finally:
                os.close(descriptor)
        visible = set(regular) if unaccounted is not None and hardlink_sources else None
        if hardlink_sources:
            find_peers()
        for path, before in (*directories.items(), *regular.items()):
            if time.monotonic() >= deadline:
                raise TimeoutError("Alias inventory deadline expired.")
            after = path.lstat()
            if path == scratch and (
                directory_identity(before) != scratch_identity
                or directory_identity(after) != scratch_identity
            ):
                raise ValueError("Alias inventory scratch identity changed.")
            if (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                if path != scratch or rescans == _MAX_SCRATCH_RESCANS:
                    raise ValueError("Alias inventory entry changed.")
                changed = True
        if scratch is not None and scratch not in directories:
            raise ValueError("Alias inventory did not traverse the bound scratch directory.")
        if not changed:
            break
        rescans += 1
    aliases = {}
    for entries in identities.values():
        paths = entries.values()
        if any(regular[path].st_nlink != len(paths) for path in paths):
            if unaccounted is None:
                raise IncompleteHardlinkAliases()
            unaccounted.update(paths if visible is None else visible.intersection(paths))
        else:
            names = tuple(sorted(paths))
            aliases.update((path, names) for path in names)
    return aliases


def is_protected_external_read(path: Path) -> bool:
    """Check whether an external read root contains protected private data.

    Args:
        path (Path): Canonical candidate read root.

    Returns:
        bool: Whether the root includes a protected path component.
    """
    return any(part.casefold() in _PRIVATE_READ_FOLDED for part in path.parts)


def write_file_state(path: Path) -> tuple[int, int] | None:
    """Inspect a literal output slot without following an existing leaf link.

    Args:
        path (Path): Exact absolute file path with an existing real parent.

    Returns:
        tuple[int, int] | None: The device and inode of a single-link file, or None if absent.

    Raises:
        OSError: The parent or the current entry cannot be inspected.
        ValueError: The path, parent, or existing entry is unsafe for an exact write grant.
    """
    parent = path.parent
    if (
        not path.is_absolute()
        or not path.name
        or ".." in path.parts
        or parent.resolve(strict=True) != parent
        or not parent.is_dir()
    ):
        raise ValueError("A write-file grant requires an exact path with an existing real parent.")
    try:
        details = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise ValueError(
            "A write-file grant requires an absent path or a single-link regular file."
        )
    return details.st_dev, details.st_ino


@dataclass(frozen=True)
class ProtectedWorkspacePaths:
    """Bind instruction filenames and control paths for command write protection.

    Args:
        instruction_names (tuple[str, ...]): Protected basenames at every workspace depth.
        directories (tuple[str, ...]): Protected workspace-relative directory paths.
        files (tuple[str, ...]): Protected workspace-relative file paths.
    """

    instruction_names: tuple[str, ...]
    directories: tuple[str, ...]
    files: tuple[str, ...]

    def protects_write_file(self, relative: Path) -> bool:
        """Check a workspace file against native control-path protections.

        Args:
            relative (Path): File path relative to the workspace.

        Returns:
            bool: Whether the exact file must remain read-only.
        """
        return self.protects_directory(relative) or relative.name in (
            *self.instruction_names,
            *self.files,
        )

    def protects_directory(self, relative: Path) -> bool:
        """Check whether a workspace-relative path lies in a protected directory.

        Args:
            relative (Path): Path relative to the selected workspace.

        Returns:
            bool: Whether a protected directory occurs at an applicable scope.
        """
        parts = relative.parts
        for directory in self.directories:
            protected = Path(directory).parts
            if len(protected) == 1:
                if protected[0] in parts:
                    return True
            elif any(
                parts[index : index + len(protected)] == protected
                for index in range(len(parts) - len(protected) + 1)
            ):
                return True
        return False


def protected_workspace_paths(configured: Iterable[str] = ()) -> ProtectedWorkspacePaths:
    """Build the active protected-path specification from trusted configuration.

    Args:
        configured (Iterable[str]): Active project instruction basenames.

    Returns:
        ProtectedWorkspacePaths: Names and relative paths denied to ordinary commands.

    Raises:
        ValueError: If an instruction name is not a plain filename.
    """
    names = tuple(
        dict.fromkeys(
            (constants.DEFAULT_AGENTS_FILENAME, constants.DEFAULT_SKILL_FILENAME, *configured)
        )
    )
    if any(
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        for name in names
    ):
        raise ValueError("Protected instruction names must be plain filenames.")
    return ProtectedWorkspacePaths(
        instruction_names=names,
        directories=(
            constants.GIT_DIRECTORY.as_posix(),
            constants.APP_DIRECTORY.as_posix(),
            constants.DEFAULT_SKILLS_DIRECTORY.as_posix(),
        ),
        files=(constants.GIT_IGNORE_FILENAME, constants.AGENT_IGNORE_FILENAME),
    )
