"""Define protected workspace paths shared by permission and native policy."""

from __future__ import annotations

import os
import stat
import time
from collections.abc import Callable, Collection, Generator, Iterable, Iterator
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
_MAX_ALIAS_RECORDS = 100_000
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


def _inventory_state(details: os.stat_result) -> tuple[int, ...]:
    """Compare identity, authority and change metadata while ignoring access time."""
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_uid,
        details.st_gid,
        details.st_nlink,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


@dataclass
class _LinkGroup:
    """Keep one metadata observation for an inode and its distinct physical names."""

    details: os.stat_result
    entries: dict[tuple[int, int, str], Path]
    visible: int = 0


class HardlinkInventory:
    """Gather and verify bounded, attempt-local evidence for a permitted filesystem view.

    Args:
        deadline (float): Absolute monotonic deadline for traversal and verification.
        scratch (Path | None): Bound scratch root whose membership alone can trigger a retry.
        max_records (int): Directory and hardlink record limit shared by scans and retries.
    """

    _deadline: float
    _scratch: Path | None
    _scratch_identity: tuple[int, ...] | None
    _remaining: int
    _rescans: int
    _directories: dict[Path, os.stat_result]
    _single_link_directories: set[Path]
    _groups: dict[tuple[int, int], _LinkGroup]

    _FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

    def __init__(
        self,
        deadline: float,
        scratch: Path | None = None,
        max_records: int = _MAX_ALIAS_RECORDS,
    ) -> None:
        self._deadline = deadline
        self._scratch = scratch
        self._scratch_identity = self._identity(scratch.lstat()) if scratch is not None else None
        self._remaining = max_records
        self._rescans = 0
        self._directories = {}
        self._single_link_directories = set()
        self._groups = {}

    @staticmethod
    def _identity(details: os.stat_result) -> tuple[int, ...]:
        """Pin scratch authority independently of directory membership."""
        return details.st_dev, details.st_ino, details.st_uid, details.st_gid, details.st_mode

    def _check_deadline(self) -> None:
        """Bound both admitted traversal and metadata-only discovery."""
        if time.monotonic() >= self._deadline:
            raise TimeoutError("Alias inventory deadline expired.")

    def _reserve_record(self) -> None:
        """Bound retained evidence across all scans."""
        self._remaining -= 1
        if self._remaining < 0:
            raise ValueError("Hardlink inventory metadata limit exceeded.")

    def _record(
        self,
        path: Path,
        details: os.stat_result,
        parent: os.stat_result,
        *,
        visible: bool,
    ) -> _LinkGroup:
        """Record one physical entry against its inode's common evidence."""
        self._reserve_record()
        identity = details.st_dev, details.st_ino
        group = self._groups.get(identity)
        if group is None:
            group = self._groups[identity] = _LinkGroup(details, {})
        elif _inventory_state(group.details) != _inventory_state(details):
            raise ValueError("Alias inventory entry changed.")
        key = parent.st_dev, parent.st_ino, path.name
        if key in group.entries:
            raise ValueError("Alias inventory contains duplicate directory entries.")
        group.entries[key] = path
        group.visible += visible
        return group

    def scan(
        self,
        roots: tuple[Path, ...],
        observe: Callable[[Path, Path, os.stat_result], None] | None = None,
    ) -> None:
        """Replace the current data view in one no-follow, depth-first walk.

        Args:
            roots (tuple[Path, ...]): Canonical admitted directories. Overlaps are traversed once.
            observe (Callable[[Path, Path, os.stat_result], None] | None): Optional observer
                receiving the traversal root, entry path and same no-follow stat used for proof.

        Raises:
            ValueError: A root or physical entry is unsafe, changed, or exceeds the record limit.
            TimeoutError: The shared deadline expires.
            OSError: A required directory or entry cannot be inspected.
        """
        selected = []
        for root in sorted(set(roots), key=lambda path: (len(path.parts), str(path))):
            if not root.is_absolute() or root.resolve(strict=True) != root:
                raise ValueError("Alias inventory requires canonical absolute roots.")
            if not any(root.is_relative_to(parent) for parent in selected):
                selected.append(root)
        self._directories.clear()
        self._single_link_directories.clear()
        self._groups.clear()

        def visit(root, directory, descriptor):
            """Walk pinned directories and share each admitted entry observation."""
            before = os.fstat(descriptor)
            if not os.path.samestat(before, directory.lstat()):
                raise ValueError("Alias inventory directory changed.")
            self._reserve_record()
            self._directories[directory] = before
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    self._check_deadline()
                    details = entry.stat(follow_symlinks=False)
                    if observe is not None:
                        observe(root, directory / entry.name, details)
                    if stat.S_ISDIR(details.st_mode):
                        child = os.open(entry.name, self._FLAGS, dir_fd=descriptor)
                        try:
                            if not os.path.samestat(os.fstat(child), details):
                                raise ValueError("Alias inventory directory changed.")
                            visit(root, directory / entry.name, child)
                        finally:
                            os.close(child)
                    elif stat.S_ISREG(details.st_mode):
                        if details.st_nlink > 1:
                            if details.st_dev != before.st_dev or details.st_ino != entry.inode():
                                raise ValueError(
                                    "Alias inventory requires physical directory entries."
                                )
                            self._record(directory / entry.name, details, before, visible=True)
                        else:
                            self._single_link_directories.add(directory)

        for root in selected:
            self._check_deadline()
            descriptor = os.open(root, self._FLAGS)
            try:
                visit(root, root, descriptor)
            finally:
                os.close(descriptor)

    def search(
        self,
        sources: tuple[Path, ...],
        denied_peer: Callable[[Path, os.stat_result], bool] | None = None,
        *,
        relevant: Callable[[Path], bool] | None = None,
    ) -> None:
        """Find missing names by metadata, keeping only matching entries and their ancestry.

        Args:
            sources (tuple[Path, ...]): Approved canonical source directories.
            denied_peer (Callable[[Path, os.stat_result], bool] | None): Additional path
                and mount exclusions using the pinned traversal metadata.
            relevant (Callable[[Path], bool] | None): Select visible names that need a complete
                group. Defaults to all visible names.

        Raises:
            ValueError: A source or matching entry changes, is ambiguous, or exceeds the record
                limit.
            TimeoutError: The shared deadline expires.
            OSError: A required directory or matching entry cannot be inspected.
        """
        missing = {
            identity
            for identity, group in self._groups.items()
            if len(group.entries) != group.details.st_nlink
            and (relevant is None or any(relevant(path) for path in group.entries.values()))
        }
        seen = {(details.st_dev, details.st_ino) for details in self._directories.values()}

        def visit(directory, descriptor, expected):
            """Walk source directories through pinned parents with bounded live state."""
            before = os.fstat(descriptor)
            if _inventory_state(before) != _inventory_state(expected):
                raise ValueError("Alias inventory directory changed.")
            key = before.st_dev, before.st_ino
            if key in seen:
                return False
            seen.add(key)
            matched = False
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    self._check_deadline()
                    identity = before.st_dev, entry.inode()
                    candidate = identity in missing
                    if not candidate and not entry.is_dir(follow_symlinks=False):
                        continue
                    path = directory / entry.name
                    if entry.name.casefold() in _PRIVATE_READ_FOLDED or is_environment_value(
                        entry.name
                    ):
                        continue
                    details = entry.stat(follow_symlinks=False)
                    if (details.st_dev, details.st_ino) != identity or (
                        denied_peer is not None and denied_peer(path, details)
                    ):
                        continue
                    if stat.S_ISDIR(details.st_mode):
                        child = os.open(entry.name, self._FLAGS, dir_fd=descriptor)
                        try:
                            matched |= visit(path, child, details)
                        finally:
                            os.close(child)
                    elif candidate:
                        group = self._record(path, details, before, visible=False)
                        matched = True
                        if len(group.entries) == group.details.st_nlink:
                            missing.remove(identity)
                    if not missing:
                        break
            if matched:
                self._reserve_record()
                self._directories[directory] = before
            else:
                seen.remove(key)
            return matched

        for directory in sources:
            if not missing:
                break
            self._check_deadline()
            if not directory.is_absolute() or ".." in directory.parts:
                raise ValueError("Alias inventory requires canonical absolute sources.")
            if any(
                part.casefold() in _PRIVATE_READ_FOLDED or is_environment_value(part)
                for part in directory.parts
            ):
                continue
            with _open_peer_directory(directory, self._FLAGS) as descriptor:
                details = os.fstat(descriptor)
                if denied_peer is None or not denied_peer(directory, details):
                    visit(directory, descriptor, details)

    def _has_new_links(self, directory: Path, before: os.stat_result) -> bool:
        """Recheck ordinary files without retaining a snapshot of each file."""
        descriptor = os.open(directory, self._FLAGS)
        try:
            if not os.path.samestat(before, os.fstat(descriptor)):
                raise ValueError("Alias inventory directory changed.")
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    self._check_deadline()
                    details = entry.stat(follow_symlinks=False)
                    if details.st_nlink > 1 and stat.S_ISREG(details.st_mode):
                        group = self._groups.get((details.st_dev, details.st_ino))
                        key = before.st_dev, before.st_ino, entry.name
                        if group is None or group.entries.get(key) != directory / entry.name:
                            return True
            return False
        finally:
            os.close(descriptor)

    def verify(self) -> bool:
        """Revalidate the evidence, allowing only a bounded scratch membership retry.

        Returns:
            bool: Whether this inventory is stable. False calls for a fresh scan and classification.

        Raises:
            ValueError: Directory or hardlink evidence changes outside the allowed scratch retry.
            TimeoutError: The shared deadline expires.
            OSError: A retained directory or linked entry cannot be inspected.
        """
        changed = False
        for path, before in self._directories.items():
            self._check_deadline()
            after = path.lstat()
            new_links = False
            state = _inventory_state(before)
            if path in self._single_link_directories and state == _inventory_state(after):
                new_links = self._has_new_links(path, before)
                after = path.lstat()
            if path == self._scratch and (
                self._identity(before) != self._scratch_identity
                or self._identity(after) != self._scratch_identity
            ):
                raise ValueError("Alias inventory scratch identity changed.")
            if state != _inventory_state(after):
                if path != self._scratch or self._rescans == _MAX_SCRATCH_RESCANS:
                    raise ValueError("Alias inventory entry changed.")
                changed = True
            elif new_links:
                raise ValueError("Alias inventory entry changed.")
        for group in self._groups.values():
            state = _inventory_state(group.details)
            for path in group.entries.values():
                self._check_deadline()
                if state != _inventory_state(path.lstat()):
                    raise ValueError("Alias inventory entry changed.")
        if self._scratch is not None and self._scratch not in self._directories:
            raise ValueError("Alias inventory did not traverse the bound scratch directory.")
        self._rescans += changed
        return not changed

    def groups(self) -> Iterator[tuple[Collection[Path], int, bool]]:
        """Iterate sparse inode groups for the admission policy.

        Yields:
            tuple[Collection[Path], int, bool]: A view of observed names in order, the number
                of leading names in the admitted view, and whether all names were accounted for.
        """
        for group in self._groups.values():
            self._check_deadline()
            yield (
                group.entries.values(),
                group.visible,
                len(group.entries) == (group.details.st_nlink),
            )


def inspect_hardlink_aliases(
    roots: tuple[Path, ...],
    deadline: float,
    *,
    max_records: int = _MAX_ALIAS_RECORDS,
    scratch: Path | None = None,
) -> dict[Path, tuple[Path, ...]]:
    """Account for regular-file links within a bounded, stable no-follow root inventory.

    This snapshot does not prevent later topology changes. Native callers must enforce link
    and relocation rules; direct callers must bind the selected identity again at opening.

    Args:
        roots (tuple[Path, ...]): Canonical directories constituting the permitted data view.
        deadline (float): Absolute monotonic time at which inspection must stop.
        max_records (int): Directory and hardlink record limit across scans and retries.
        scratch (Path | None): Bound private scratch directory eligible for membership retries.

    Returns:
        dict[Path, tuple[Path, ...]]: Each accounted multiply linked name mapped to its group.

    Raises:
        ValueError: Directory or hardlink evidence changes, aliases are incomplete, or the
            record limit is exhausted.
        TimeoutError: The inspection deadline expires.
        OSError: A required directory or entry cannot be inspected without following a link.
    """
    inventory = HardlinkInventory(deadline, scratch, max_records)
    while True:
        inventory.scan(roots)
        if inventory.verify():
            break
    aliases = {}
    for names, _, complete in inventory.groups():
        if not complete:
            raise IncompleteHardlinkAliases()
        group = tuple(sorted(names))
        aliases.update(dict.fromkeys(group, group))
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
