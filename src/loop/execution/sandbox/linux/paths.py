"""Inspect bound Linux roots before constructing protected mounts or executing command source.

This primitive supplies protected entries, private masks, writable ancestor mountpoints,
read-only hardlink mounts and exact identity-bound external executable files. Its trusted caller
must apply the qualified narrow namespace and filename policy and close host descriptors before
payload entry. It does not launch a backend.
"""

from __future__ import annotations

import errno
import fcntl
import os
import stat
import time
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from .... import constants
from ....constants import GIT_DIRECTORY
from ....permissions.protection import (
    PRIVATE_READ_COMPONENTS,
    IncompleteHardlinkAliases,
    inspect_hardlink_aliases,
    is_environment_value,
    is_protected_external_read,
)
from ....sandbox_catalog import hardlink_source_paths
from ..constants import TMPDIR_ENVIRONMENT_VARIABLE
from .constants import (
    NATIVE_ALIAS_PATH,
    NATIVE_CONVENTIONAL_TMP_PATH,
    NATIVE_SCRATCH_PATH,
    SESSION_CONVENTIONAL_TMP_NAME,
)
from .mountinfo import MINIMUM_MOUNTINFO_FIELDS, read_mountinfo

if TYPE_CHECKING:
    from ..contracts import SandboxRequest

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_PRIVATE_NAMES = frozenset(name.casefold() for name in PRIVATE_READ_COMPONENTS)
_MAX_HARDLINK_MOUNTS = 2_000


def _check_deadline(deadline: float) -> None:
    """Refuse continued preparation after the bound command deadline."""
    if time.monotonic() >= deadline:
        raise TimeoutError("Linux path preparation exceeded the command deadline.")


def _directory_state(details: os.stat_result) -> tuple[int, int, int, int]:
    """Bind a scanned directory to its inode and membership-change timestamps."""
    return details.st_dev, details.st_ino, details.st_mtime_ns, details.st_ctime_ns


def payload_path(path: Path, scratch: Path) -> Path:
    """Map a private scratch path into its request-local namespace.

    Args:
        path (Path): Bound host path.
        scratch (Path): This request's private host scratch directory.

    Returns:
        Path: Stable namespace path beneath scratch, or the unchanged external path.
    """
    return NATIVE_SCRATCH_PATH / path.relative_to(scratch) if path.is_relative_to(scratch) else path


def conventional_tmp_available(
    request: SandboxRequest,
    scratch: Path,
    *,
    runtime_roots: tuple[Path, ...] = (),
) -> bool:
    """Check whether a session-private conventional temporary mount avoids other authority.

    Args:
        request (SandboxRequest): Bound request and its approved path identities.
        scratch (Path): Bound physical session scratch directory.
        runtime_roots (tuple[Path, ...]): Authenticated system runtime roots.

    Returns:
        bool: Whether the separate session subtree can safely occupy the conventional path.
    """
    if not request.session_scratch or not request.scratch_data or request.scratch_root != scratch:
        return False
    temporary = NATIVE_CONVENTIONAL_TMP_PATH
    physical = temporary.resolve()
    candidates = (
        request.workspace,
        request.cwd,
        scratch,
        *request.read_roots,
        *request.write_roots,
        *request.read_aliases,
        *(path for pair in request.path_roots for path in pair),
        *(
            path
            for _, spelling, resolved, _, _ in request.executable_identities
            for path in (spelling, resolved)
        ),
        *runtime_roots,
    )
    return not any(
        path.is_relative_to(root) or root.is_relative_to(path)
        for path in candidates
        for root in (temporary, physical)
    )


@contextmanager
def _conventional_tmp_source(
    request: SandboxRequest,
    scratch: Path,
) -> Generator[tuple[Path, int, tuple[int, ...]], None, None]:
    """Create and pin the exact private session subtree without following links."""
    path = scratch / SESSION_CONVENTIONAL_TMP_NAME
    with ExitStack() as opened:
        parent = os.open(scratch, _DIRECTORY_FLAGS)
        opened.callback(os.close, parent)
        if not request.directory_matches_descriptor(scratch, parent):
            raise ValueError("Session scratch changed before temporary preparation.")
        try:
            os.mkdir(path.name, mode=constants.PRIVATE_DIRECTORY_MODE, dir_fd=parent)
        except FileExistsError:
            pass
        descriptor = os.open(path.name, _DIRECTORY_FLAGS, dir_fd=parent)
        opened.callback(os.close, descriptor)
        details = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISDIR(details.st_mode)
            or details.st_uid != os.getuid()
            or stat.S_IMODE(details.st_mode) != constants.PRIVATE_DIRECTORY_MODE
            or (details.st_dev, details.st_ino) != (current.st_dev, current.st_ino)
            or path.resolve(strict=True) != path
        ):
            raise ValueError("Session temporary subtree is not a private real directory.")
        for acl in ("system.posix_acl_access", "system.posix_acl_default"):
            try:
                value = os.getxattr(descriptor, acl)
            except OSError as error:
                if error.errno not in (errno.ENODATA, errno.ENOTSUP):
                    raise
            else:
                if value:
                    raise ValueError("Session temporary subtree has an extended access list.")
        state = _source_state(details)
        if any(_source_state(latest) != state for latest in (os.fstat(descriptor), path.lstat())):
            raise ValueError("Session temporary subtree changed during inspection.")
        yield path, descriptor, state


def _hardlink_mount_sources(
    hardlinked: dict[Path, tuple[int, int]],
    writable_roots: tuple[Path, ...],
    deadline: float,
) -> tuple[Path, ...]:
    """Choose bounded read-only file or containing-directory mounts for writable hardlinks."""
    paths = set()
    for path in hardlinked:
        _check_deadline(deadline)
        if any(path.is_relative_to(root) for root in writable_roots):
            paths.add(path)
    if len(paths) <= _MAX_HARDLINK_MOUNTS:
        return tuple(sorted(paths))

    by_parent: dict[Path, set[Path]] = {}
    for path in paths:
        _check_deadline(deadline)
        by_parent.setdefault(path.parent, set()).add(path)
    mounts = set()
    for parent, siblings in by_parent.items():
        _check_deadline(deadline)
        if len(siblings) > 1 and not any(root.is_relative_to(parent) for root in writable_roots):
            mounts.add(parent)
        else:
            mounts.update(siblings)

    # Index each ancestor once; update membership when a subtree becomes one mount.
    coverage: dict[Path, set[Path]] = {}
    for path in mounts:
        parent = path.parent
        while parent not in writable_roots:
            _check_deadline(deadline)
            coverage.setdefault(parent, set()).add(path)
            parent = parent.parent
    while len(mounts) > _MAX_HARDLINK_MOUNTS:
        candidates = {}
        for parent, covered in coverage.items():
            _check_deadline(deadline)
            if len(covered) > 1 and not any(root.is_relative_to(parent) for root in writable_roots):
                candidates[parent] = covered
        if not candidates:
            raise ValueError("Linux writable hardlinks exceed the bounded mount limit.")
        parent = min(
            candidates, key=lambda path: (len(candidates[path]), -len(path.parts), str(path))
        )
        removed = candidates[parent].copy()
        mounts.difference_update(removed)
        mounts.add(parent)
        for ancestor, covered in coverage.items():
            _check_deadline(deadline)
            covered.difference_update(removed)
            if parent != ancestor and parent.is_relative_to(ancestor):
                covered.add(parent)

    minimal = set()
    for path in sorted(mounts, key=lambda item: (len(item.parts), str(item))):
        _check_deadline(deadline)
        if not any(path.is_relative_to(parent) for parent in minimal if parent != path):
            minimal.add(path)
    _check_deadline(deadline)
    return tuple(sorted(minimal))


def _scan(
    root: Path,
    request: SandboxRequest,
    inspected: dict[Path, os.stat_result],
) -> tuple[set[Path], set[Path], dict[Path, tuple[int, int]], set[tuple[int, int]]]:
    """Collect protected, private and hardlinked entries while checking directory stability."""
    original = root.lstat()
    if not stat.S_ISDIR(original.st_mode) or root.resolve(strict=True) != root:
        raise ValueError("Linux path root is not a real directory.")
    if is_protected_external_read(root) and not (
        request.session_scratch
        and str(root) == request.shell_environment().get(TMPDIR_ENVIRONMENT_VARIABLE)
    ):
        raise ValueError("Linux path root includes private data.")
    if root != request.scratch_root and any(
        is_environment_value(part)
        for part in (
            root.relative_to(request.workspace) if root.is_relative_to(request.workspace) else root
        ).parts
    ):
        raise ValueError("Linux path root includes environment values.")
    protection = request.protected_paths
    git_granted = request.git_create or request.workspace / GIT_DIRECTORY in request.write_roots
    protected, private, hardlinked, sensitive_inodes = set(), set(), {}, set()
    pending = [(root, False, False)]
    while pending:
        _check_deadline(request.deadline)
        current, protected_parent, private_parent = pending.pop()
        descriptor = os.open(current, _DIRECTORY_FLAGS)
        try:
            before = os.fstat(descriptor)
            inspected.setdefault(current, before)
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    _check_deadline(request.deadline)
                    details = entry.stat(follow_symlinks=False)
                    path = current / entry.name
                    relative = path.relative_to(root)
                    if git_granted and relative.parts[0] == GIT_DIRECTORY.as_posix():
                        relative = Path(*relative.parts[1:])
                    is_protected = (
                        protected_parent
                        or root == request.workspace
                        and (
                            protection.protects_directory(relative)
                            or entry.name in (*protection.instruction_names, *protection.files)
                        )
                    )
                    is_private = (
                        private_parent
                        or request.denies_path(path)
                        or (
                            not (request.scratch_data and root == request.scratch_root)
                            and (
                                entry.name.casefold() in _PRIVATE_NAMES
                                or is_environment_value(entry.name)
                            )
                        )
                    )
                    inode = (details.st_dev, details.st_ino)
                    if stat.S_ISREG(details.st_mode) and (is_protected or is_private):
                        sensitive_inodes.add(inode)
                    if stat.S_ISREG(details.st_mode) and details.st_nlink > 1:
                        if request.scratch_data and root == request.scratch_root:
                            raise ValueError("Session scratch contains a shared inode.")
                        hardlinked[path] = inode
                    if (
                        (is_protected or is_private)
                        and stat.S_ISLNK(details.st_mode)
                        and not (private_parent and request.denies_path(current))
                    ):
                        raise ValueError("Linux protected or private entry is a symlink.")
                    if is_protected and not protected_parent:
                        protected.add(path)
                    if is_private and not private_parent:
                        if not (stat.S_ISDIR(details.st_mode) or stat.S_ISREG(details.st_mode)):
                            raise ValueError("Linux private entry has an unsafe type.")
                        private.add(path)
                    if stat.S_ISDIR(details.st_mode):
                        pending.append((path, is_protected, is_private))
            for latest in (os.fstat(descriptor), current.lstat()):
                if _directory_state(before) != _directory_state(latest):
                    raise ValueError("Linux scanned directory changed.")
        finally:
            os.close(descriptor)
    latest = root.lstat()
    if (original.st_dev, original.st_ino) != (latest.st_dev, latest.st_ino):
        raise ValueError("Linux path root changed during preparation.")
    return protected, private, hardlinked, sensitive_inodes


def _hardlink_mounts(roots: tuple[Path, ...], source: Path) -> frozenset[tuple[str, ...]]:
    """Identify nested mounts and refuse a cache rooted at a mount or aliased subtree."""
    mounts = set()
    for fields in read_mountinfo():
        if len(fields) < MINIMUM_MOUNTINFO_FIELDS or "-" not in fields:
            raise ValueError("Ambiguous Linux hardlink source mount authority.")
        mount = Path(fields[4])
        if mount == source or (
            mount != Path("/") and source.is_relative_to(mount) and fields[3] != "/"
        ):
            raise ValueError("Linux hardlink source has a mounted alias.")
        if any(mount.is_relative_to(root) for root in (*roots, source)):
            mounts.add(tuple(fields))
    return frozenset(mounts)


def _inspect_cache_aliases(
    request: SandboxRequest,
    roots: tuple[Path, ...],
    scratch: Path,
    unaccounted: set[Path],
) -> None:
    """Verify unresolved visible links against the owner-bound cache metadata."""
    sources = request.hardlink_sources
    (source,) = sources
    details = source.lstat()
    uid = os.getuid()

    def controlled(details: os.stat_result) -> bool:
        """Check owner and effective write access, including the POSIX ACL group mask."""
        return details.st_uid in (uid, 0) and not (
            details.st_mode & constants.GROUP_AND_OTHER_WRITE_MODE_MASK
        )

    if (
        source.resolve(strict=True) != source
        or not stat.S_ISDIR(details.st_mode)
        or details.st_uid != uid
        or not controlled(details)
        or source.parent.lstat().st_uid != uid
        or request.denies_path(source)
    ):
        raise ValueError("Linux hardlink source is not an owner-controlled account cache.")
    for directory in source.parents:
        parent = directory.lstat()
        if not controlled(parent) and not (
            directory != source.parent and parent.st_uid == 0 and parent.st_mode & stat.S_ISVTX
        ):
            raise ValueError("Linux hardlink source has unsafe account ancestry.")
    mounts = _hardlink_mounts(roots, source)
    mounted = {Path(fields[4]) for fields in mounts}
    protection = request.protected_paths
    owner_only = set()

    def denied_peer(path: Path) -> bool:
        """Apply policy and stop at native mount boundaries, including mounted files."""
        if (
            path in mounted
            or request.denies_path(path)
            or protection.protects_write_file(path.relative_to(source))
        ):
            return True
        details = path.lstat()
        if details.st_uid not in (0, uid) or (
            not owner_only.intersection(path.parents) and not controlled(details)
        ):
            return True
        if stat.S_ISDIR(details.st_mode) and not details.st_mode & (stat.S_IXGRP | stat.S_IXOTH):
            owner_only.add(path)
        return False

    aliases = inspect_hardlink_aliases(
        roots,
        request.deadline,
        unaccounted=unaccounted,
        scratch=scratch,
        hardlink_sources=sources,
        denied_peer=denied_peer,
    )
    for path, group in aliases.items():
        if (
            path.is_relative_to(source)
            and not any(path.is_relative_to(root) for root in roots)
            and any(name.is_relative_to(mount) for name in group for mount in mounted)
        ):
            raise ValueError("Linux shared dependency has a mounted alias.")
    if _hardlink_mounts(roots, source) != mounts:
        raise ValueError("Linux hardlink source mounts changed.")


def prepare_paths(
    request: SandboxRequest,
    scratch: Path,
    *,
    runtime_roots: tuple[Path, ...] = (),
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    """Inspect all approved roots and return deterministic protected mount preparation inputs.

    Scan the workspace, scratch and additional read/write roots, including protected subtrees.
    Collect multiply linked regular files for read-only bind mounts, preserving reads and
    execution while preventing writes through those paths. Small sets use individual file mounts;
    larger sets use containing directories to stay below Bubblewrap's argument ceiling, which also
    makes other files in those directories read-only. Refuse visible hardlinks that alias a
    protected or private file or whose alias set is incomplete. If a visible group is incomplete,
    the authorized account cache may supply missing names without becoming a mount. Conceal
    hardlinks already under private directory masks without denying an unrelated command. Skip
    ordinary symlinks while rejecting symlinks within protected/private entries. Return ancestor
    directories for self-bind mountpoints so
    protected entries cannot escape through relocation; their unrelated contents retain the
    approved write semantics. Future protected filenames still require the production filename
    policy. Explicit top-level Git authority does not exempt nested Git, instructions, controls or
    private data. No filesystem mutation occurs.

    Args:
        request (SandboxRequest): Approved command with current canonical and alias identities
            and its execution deadline. Opaque source is never inspected or executed.
        scratch (Path): Live private real directory bound as the request's TMPDIR.
        runtime_roots (tuple[Path, ...]): Already authenticated read-only runtime roots. Exact
            duplicate read grants retain their runtime treatment; workspace/write roots do not.

    Returns:
        tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]: Sorted
            existing protected entries for read-only binds, private entries for type-correct
            masks, ancestor directories for self-bind mountpoints, and multiply linked regular
            file or containing-directory paths for read-only binds. The caller must bind and
            revalidate every mount source before applying mounts; this result is not a launch
            authorization.

    Raises:
        ValueError: Request/scratch authority is stale or unsafe, a hardlink aliases a protected
            or private file, its mount count cannot be bounded, a private/protected symlink or
            unsafe private type is found, or a directory changes during scan.
        TimeoutError: Preparation reaches the approved execution deadline.
        OSError: A required root, entry or no-follow descriptor cannot be inspected.
    """
    _check_deadline(request.deadline)
    if not request.paths_are_current():
        raise ValueError("Linux approved path authority changed.")
    if any(path.is_relative_to(scratch) for path in request.write_files):
        raise ValueError("Linux exact file scopes cannot overlap the writable scratch view.")
    details = scratch.lstat()
    if (
        request.shell_environment().get(TMPDIR_ENVIRONMENT_VARIABLE) != str(scratch)
        or not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.getuid()
        or stat.S_IMODE(details.st_mode) != constants.PRIVATE_DIRECTORY_MODE
    ):
        raise ValueError("Linux scratch is not a bound private directory.")
    protected, private, anchors, hardlinked = set(), set(), set(), {}
    sensitive_inodes = set()
    inspected = {}
    roots = dict.fromkeys(
        (
            request.workspace,
            *(root for root in request.read_roots if root not in runtime_roots),
            *request.write_roots,
            scratch,
        )
    )
    unaccounted: set[Path] = set()
    if request.hardlink_sources and (
        request.permission_revision is None
        or request.hardlink_sources != hardlink_source_paths(("shared_dependencies",), "linux")
    ):
        raise ValueError("Unapproved Linux hardlink source.")
    inspect_hardlink_aliases(
        tuple(roots), request.deadline, unaccounted=unaccounted, scratch=scratch
    )
    for root in roots:
        root_protected, root_private, root_hardlinked, root_sensitive = _scan(
            root, request, inspected
        )
        protected.update(root_protected)
        private.update(root_private)
        hardlinked.update(root_hardlinked)
        sensitive_inodes.update(root_sensitive)
        for path in root_protected | root_private:
            anchor = path if path.is_dir() else path.parent
            while anchor != root:
                anchors.add(anchor)
                anchor = anchor.parent
    directory_masks = {path for path in private if path.is_dir()}

    def is_concealed(path: Path) -> bool:
        """Check whether a directory mask prevents every access to this descendant."""
        return not directory_masks.isdisjoint(path.parents)

    if request.hardlink_sources and any(not is_concealed(path) for path in unaccounted):
        unaccounted.clear()
        _inspect_cache_aliases(request, tuple(roots), scratch, unaccounted)
    if any(not is_concealed(path) for path in unaccounted):
        raise IncompleteHardlinkAliases(
            "Some links have names outside the verified scope or in protected data."
        )
    if any(
        inode in sensitive_inodes and not is_concealed(path) for path, inode in hardlinked.items()
    ):
        raise ValueError("Linux hardlinked file aliases protected or private data.")

    required_paths = (
        request.cwd,
        *roots,
        *request.write_files,
        *request.read_aliases,
        *request.read_alias_targets,
        *(
            path
            for _, spelling, resolved, _, _ in (
                *request.executable_identities,
                *request.workspace_executable_identities,
            )
            for path in (spelling, resolved)
        ),
    )
    if any(path in directory_masks or is_concealed(path) for path in required_paths):
        raise ValueError("A required Linux sandbox path is inside a denied directory.")

    protected = {path for path in protected - private if not is_concealed(path)}
    private = {path for path in private if not is_concealed(path)}
    anchors = {path for path in anchors if not is_concealed(path)}
    # An earlier subtree can change while another approved root is scanned. Keep the first
    # observation of overlapping directories and recheck the whole request before returning.
    for current, before in inspected.items():
        _check_deadline(request.deadline)
        if _directory_state(before) != _directory_state(current.lstat()):
            raise ValueError("Linux scanned directory changed.")
    if not request.paths_are_current():
        raise ValueError("Linux approved path authority changed.")
    _check_deadline(request.deadline)
    read_only_mounts = _hardlink_mount_sources(
        {path: identity for path, identity in hardlinked.items() if not is_concealed(path)},
        (*request.write_roots, scratch),
        request.deadline,
    )
    _check_deadline(request.deadline)
    return (
        tuple(sorted(protected)),
        tuple(sorted(private)),
        tuple(sorted(anchors)),
        read_only_mounts,
    )


def _source_state(details: os.stat_result) -> tuple[int, ...]:
    """Bind a mount source to its inode, authority and change timestamps, excluding atime."""
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


@contextmanager
def bind_paths(
    request: SandboxRequest,
    scratch: Path,
    *,
    runtime_roots: tuple[Path, ...] = (),
) -> Generator[
    tuple[
        tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]],
        dict[Path, int],
    ],
    None,
    None,
]:
    """Inspect approved paths and hold inode-bound mount descriptors for trusted native setup.

    Pin workspace/read/write/scratch roots, protected entries, private mask targets and ancestor
    mountpoints, and hardlink file or containing-directory sources using Linux O_PATH descriptors
    with no-follow and close-on-exec flags. Revalidate every pathname and descriptor plus shared
    request authority before yielding. The caller owns mount construction and must use
    descriptor-backed sources, apply type-correct masks, mount hardlinked paths read-only and
    enforce the filename policy, then close all setup descriptors before opaque source executes.
    This context performs no mounts, launch, policy loading or filesystem mutation. Its
    descriptors remain valid only inside the context and close on normal exit, caller failure or
    preparation failure.

    Args:
        request (SandboxRequest): Approved command authority and execution deadline.
        scratch (Path): Live private directory bound as the request's TMPDIR.
        runtime_roots (tuple[Path, ...]): Already authenticated read-only runtime roots, passed
            to data-root inspection without exempting workspace or write authority.

    Yields:
        tuple[tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]],
        dict[Path, int]]: Protected/private/ancestor/hardlink inventories from prepare_paths and
            their mount-source descriptor mapping, also including every workspace/read/write/
            scratch root. Private descriptors pin original target types; they are never sources
            for exposing private bytes. This result supplies no runtime, mask-control or helper
            authority.

    Raises:
        ValueError: Approved path authority is unsafe/stale, the bounded hardlink mount count is
            exceeded, a mount source changes or has an unsafe type, or path inspection finds a
            protected/private hardlink alias or drift.
        TimeoutError: Preparation reaches the approved command deadline.
        OSError: Linux O_PATH is unavailable or a required source cannot be inspected or opened.
    """
    path_flag = getattr(os, "O_PATH", None)
    if path_flag is None:
        raise OSError(errno.ENOTSUP, "Linux inode-bound mount preparation requires O_PATH.")
    inventories = prepare_paths(request, scratch, runtime_roots=runtime_roots)
    hardlinked = inventories[3]
    sources = dict.fromkeys(
        (
            request.workspace,
            *request.read_roots,
            *request.write_roots,
            *(path.parent for path in request.write_files),
            scratch,
            *(canonical for _, canonical in request.path_roots),
            *(alias.parent for alias in request.read_aliases),
            *(path for inventory in inventories for path in inventory),
        )
    )
    descriptors, identities = {}, {}
    with ExitStack() as opened:
        for path in sources:
            _check_deadline(request.deadline)
            before = path.lstat()
            if (
                not (stat.S_ISDIR(before.st_mode) or stat.S_ISREG(before.st_mode))
                or stat.S_ISREG(before.st_mode)
                and path not in hardlinked
                and before.st_nlink != 1
                or path.resolve(strict=True) != path
            ):
                raise ValueError("Linux mount source is not a real directory or single-link file.")
            descriptor = os.open(path, path_flag | os.O_NOFOLLOW | os.O_CLOEXEC)
            opened.callback(os.close, descriptor)
            if _source_state(os.fstat(descriptor)) != _source_state(before):
                raise ValueError("Linux mount source changed during binding.")
            descriptors[path] = descriptor
            identities[path] = before
        for path, descriptor in descriptors.items():
            _check_deadline(request.deadline)
            for latest in (path.lstat(), os.fstat(descriptor)):
                if _source_state(latest) != _source_state(identities[path]):
                    raise ValueError("Linux bound mount source changed.")
        if not request.paths_are_current():
            raise ValueError("Linux approved path authority changed.")
        _check_deadline(request.deadline)
        yield inventories, descriptors


@contextmanager
def mount_arguments(
    request: SandboxRequest,
    scratch: Path,
    *,
    empty_mask: int | None = None,
    skip_path_roots: tuple[Path, ...] = (),
    skip_executable_roots: tuple[Path, ...] = (),
    runtime_roots: tuple[Path, ...] = (),
) -> Generator[tuple[tuple[str, ...], tuple[int, ...], SandboxRequest], None, None]:
    """Prepare descriptor-backed request mounts without granting native launch authority.

    Bind approved roots at their canonical spellings, outer roots before nested roots. A path
    under any approved write root or private scratch remains writable. Exact-file parents also
    need writable mounts for creation and deletion; only the independently enforced filename
    layer grants writes within those parents. Other roots and ancestors are read-only. Bind
    the fresh host scratch at its original spelling for trusted identity
    checks and at a stable namespace path for payload access. Apply ancestor mountpoints before
    protected read-only binds, then hide private directories with empty tmpfs mounts and private
    regular files with the supplied empty inode. Mirror every restrictive scratch overlay;
    selected workspaces below scratch retain their read-only role. Refuse a whole filesystem or
    home bind and read aliases whose target is absent from this request view. For eligible fresh
    session scratch, create and pin a private conventional temporary subtree, with its contents
    retained as part of the session working directory. Existing grants
    that overlap the conventional destination keep their original view. The filename layer still
    protects future names.

    The caller must authenticate the helper and immutable mask-control inode, provide reviewed
    runtime/proc/device/control mounts and namespace/network setup, enforce the filename policy,
    and close every setup descriptor before source entry. Returned arguments are only a request
    filesystem fragment; they contain no shell source, capabilities or launch operation.

    Args:
        request (SandboxRequest): Current approved roots, read aliases and execution deadline.
        scratch (Path): Live private directory bound as the request's TMPDIR.
        empty_mask (int | None): Caller-authenticated read-only empty regular-file descriptor
            for private file masks. Required only when such targets exist. The original descriptor
            remains caller-owned; this context duplicates and owns its mount-source descriptor.
        skip_path_roots (tuple[Path, ...]): Request PATH spellings already supplied by an
            authenticated runtime mount, whose duplicate directory mounts and aliases are omitted.
        skip_executable_roots (tuple[Path, ...]): Authenticated runtime roots that already supply
            named executable files, so no duplicate exact-file mount is needed.
        runtime_roots (tuple[Path, ...]): Authenticated read-only runtime roots whose exact
            duplicates are omitted from the data census, and which must not overlap the
            optional conventional temporary mount. Workspace and write roots remain scanned.

    Yields:
        tuple[tuple[str, ...], tuple[int, ...], SandboxRequest]: Bubblewrap filesystem arguments,
            descriptors held only for this context, and the request with masked PATH roots
            removed. Private original-target descriptors are withheld from inheritance.

    Raises:
        ValueError: Request/source authority is unsafe or stale, a root covers the filesystem or
            home, a reserved namespace destination overlaps another grant, a skipped PATH root is
            not bound to the request, an alias target is outside the bound view, an executable
            source changes, a temporary subtree is unsafe, or a required mask is missing/unsafe.
        TimeoutError: Preparation reaches the approved command deadline.
        OSError: Linux O_PATH is unavailable or source/mask descriptors cannot be inspected.
    """
    file_parents = {path.parent for path in request.write_files}
    roots = {request.workspace, *request.read_roots, *request.write_roots, *file_parents, scratch}
    skipped_paths = set(skip_path_roots)
    path_roots = tuple(
        (spelling, canonical)
        for spelling, canonical in request.path_roots
        if spelling not in skipped_paths
    )
    if not skipped_paths <= {spelling for spelling, _ in request.path_roots}:
        raise ValueError("Linux skipped PATH roots are not bound to this request.")
    metadata = {canonical for _, canonical in request.path_roots} | {
        alias.parent for alias in request.read_aliases
    }
    mounted_metadata = {canonical for _, canonical in path_roots} | {
        alias.parent for alias in request.read_aliases
    }
    home = Path.home().resolve(strict=True)
    if any(root == Path("/") or home.is_relative_to(root) for root in roots | metadata):
        raise ValueError("Linux request mounts cannot expose the whole filesystem or home.")
    aliases = {
        **{alias: alias.resolve(strict=True) for alias in request.read_aliases},
    }
    if any(
        path == reserved or path.is_relative_to(reserved) or reserved.is_relative_to(path)
        for path in (*roots, *metadata, *aliases, *aliases.values())
        for reserved in (NATIVE_SCRATCH_PATH, NATIVE_ALIAS_PATH)
    ):
        raise ValueError("Reserved native namespace destination overlaps another mount.")
    if any(not any(target.is_relative_to(root) for root in roots) for target in aliases.values()):
        raise ValueError("Linux request alias target is outside the mounted view.")
    with ExitStack() as owned:
        if not request.paths_are_current():
            raise ValueError("Linux approved path authority changed.")
        conventional = (
            owned.enter_context(_conventional_tmp_source(request, scratch))
            if conventional_tmp_available(request, scratch, runtime_roots=runtime_roots)
            else None
        )
        temporary_state = conventional[2] if conventional else None
        inventories, descriptors = owned.enter_context(
            bind_paths(request, scratch, runtime_roots=runtime_roots)
        )
        protected, private, anchors, hardlinked = inventories
        arguments, scratch_mounts, inherited = [], [], set()

        def mount(option: str, descriptor: int, path: Path) -> None:
            """Preserve each pinned mount and its scratch mirror, including restrictive overlays."""
            arguments.extend((option, str(descriptor), str(path)))
            native = payload_path(path, scratch)
            if native != path:
                scratch_mounts.extend((option, str(descriptor), str(native)))

        writable = (*request.write_roots, scratch)
        for path in sorted(mounted_metadata - roots):
            mount("--ro-bind-fd", descriptors[path], path)
            inherited.add(descriptors[path])
        for path in sorted(
            roots | (set(anchors) - set(private)), key=lambda path: (len(path.parts), str(path))
        ):
            _check_deadline(request.deadline)
            option = (
                "--bind-fd"
                if (path in file_parents or any(path.is_relative_to(root) for root in writable))
                and not (
                    request.workspace.is_relative_to(scratch)
                    and path.is_relative_to(request.workspace)
                    and not any(path.is_relative_to(root) for root in request.write_roots)
                )
                else "--ro-bind-fd"
            )
            mount(option, descriptors[path], path)
            inherited.add(descriptors[path])
        if conventional is not None:
            temporary_path, temporary_descriptor, _ = conventional
            mount("--bind-fd", temporary_descriptor, temporary_path)
            inherited.add(temporary_descriptor)
        for path in hardlinked:
            _check_deadline(request.deadline)
            mount("--ro-bind-fd", descriptors[path], path)
            inherited.add(descriptors[path])
        for path in protected:
            _check_deadline(request.deadline)
            mount("--ro-bind-fd", descriptors[path], path)
            inherited.add(descriptors[path])
        mask_descriptor = None
        masked_directories = []
        for path in private:
            _check_deadline(request.deadline)
            if stat.S_ISDIR(os.fstat(descriptors[path]).st_mode):
                masked_directories.append(path)
                arguments.extend(("--tmpfs", str(path), "--remount-ro", str(path)))
                native = payload_path(path, scratch)
                if native != path:
                    scratch_mounts.extend(("--tmpfs", str(native), "--remount-ro", str(native)))
            else:
                if mask_descriptor is None:
                    if empty_mask is None:
                        raise ValueError("Linux private file masks require an empty control inode.")
                    mask_descriptor = os.dup(empty_mask)
                    owned.callback(os.close, mask_descriptor)
                    details = os.fstat(mask_descriptor)
                    if (
                        not stat.S_ISREG(details.st_mode)
                        or details.st_size != 0
                        or details.st_nlink != 1
                        or fcntl.fcntl(mask_descriptor, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
                    ):
                        raise ValueError("Linux private file mask control inode is unsafe.")
                mount("--ro-bind-fd", mask_descriptor, path)
                inherited.add(mask_descriptor)
        for spelling, canonical in path_roots:
            if spelling != canonical and not any(
                spelling.is_relative_to(root) for root in roots | metadata
            ):
                arguments.extend(("--symlink", str(canonical), str(spelling)))
        arguments.extend(("--dir", str(NATIVE_SCRATCH_PATH)))
        arguments.extend(scratch_mounts)
        inherited.add(descriptors[scratch])
        if conventional is not None:
            temporary_path, temporary_descriptor, _ = conventional
            if any(
                _source_state(details) != temporary_state
                for details in (os.fstat(temporary_descriptor), temporary_path.lstat())
            ):
                raise ValueError("Session temporary subtree changed during binding.")
            arguments.extend(
                (
                    "--dir",
                    str(NATIVE_CONVENTIONAL_TMP_PATH),
                    "--bind-fd",
                    str(temporary_descriptor),
                    str(NATIVE_CONVENTIONAL_TMP_PATH),
                )
            )
        executable_mounts = set()
        for _, spelling, resolved, device, inode in request.executable_identities:
            if resolved in executable_mounts:
                continue
            if any(resolved.is_relative_to(root) for root in (*roots, *skip_executable_roots)):
                continue
            path_root = next(
                (canonical for _, canonical in path_roots if resolved.is_relative_to(canonical)),
                None,
            )
            if path_root is None:
                raise ValueError("Linux executable target is outside the mounted PATH roots.")
            root_details = path_root.stat()
            if (
                not stat.S_ISDIR(root_details.st_mode)
                or root_details.st_uid not in (0, os.getuid())
                or root_details.st_mode & constants.OTHER_WRITE_MODE_BIT
                or root_details.st_mode & constants.GROUP_WRITE_MODE_BIT
                and root_details.st_gid != os.getgid()
            ):
                raise ValueError("Linux executable PATH root is unsafe.")
            path_flag = getattr(os, "O_PATH", None)
            if path_flag is None:
                raise OSError(errno.ENOTSUP, "Linux executable pinning requires O_PATH.")
            if (
                spelling.resolve(strict=True) != resolved
                or resolved.resolve(strict=True) != resolved
            ):
                raise ValueError("Linux executable path changed during binding.")
            before = resolved.lstat()
            descriptor = os.open(resolved, path_flag | os.O_NOFOLLOW | os.O_CLOEXEC)
            owned.callback(os.close, descriptor)
            bound = os.fstat(descriptor)
            latest = resolved.lstat()
            if (
                not stat.S_ISREG(bound.st_mode)
                or (bound.st_dev, bound.st_ino) != (device, inode)
                or bound.st_nlink != 1
                or bound.st_uid not in (0, os.getuid())
                or bound.st_mode & constants.GROUP_AND_OTHER_WRITE_MODE_MASK
                or _source_state(before) != _source_state(bound)
                or _source_state(latest) != _source_state(bound)
                or spelling.resolve(strict=True) != resolved
            ):
                raise ValueError("Linux executable changed during binding.")
            arguments.extend(("--ro-bind-fd", str(descriptor), str(resolved)))
            inherited.add(descriptor)
            executable_mounts.add(resolved)
        if mask_descriptor is not None and (
            _source_state(os.fstat(mask_descriptor)) != _source_state(details)
            or os.fstat(mask_descriptor).st_size != 0
        ):
            raise ValueError("Linux private file mask control inode changed.")
        if not request.paths_are_current():
            raise ValueError("Linux approved path authority changed.")
        _check_deadline(request.deadline)
        yield (
            tuple(arguments),
            tuple(sorted(inherited)),
            request.without_hidden_search_paths(tuple(masked_directories)),
        )
