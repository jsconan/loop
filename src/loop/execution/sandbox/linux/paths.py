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
import pwd
import re
import stat
import time
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from .... import constants
from ....constants import GIT_DIRECTORY
from ....permissions.protection import PRIVATE_READ_COMPONENTS, is_protected_external_read
from ..constants import TMPDIR_ENVIRONMENT_VARIABLE
from .constants import NATIVE_ALIAS_PATH, NATIVE_SCRATCH_PATH

if TYPE_CHECKING:
    from ..contracts import SandboxRequest

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_PRIVATE_NAMES = frozenset(name.casefold() for name in PRIVATE_READ_COMPONENTS)
_MAX_HARDLINK_MOUNTS = 2_000
_MAX_VENV_CONFIG_BYTES = 64 * 1024


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


def payload_aliases(request: SandboxRequest) -> tuple[tuple[Path, Path, Path], ...]:
    """Map bound virtual aliases to stable request-local symlinks.

    Args:
        request (SandboxRequest): Original bound virtual aliases and targets.

    Returns:
        tuple[tuple[Path, Path, Path], ...]: Host alias, namespace alias and original target.
    """
    return tuple(
        (alias, NATIVE_ALIAS_PATH / f"r{index}", target)
        for index, (_, alias, target) in enumerate(request.aliases)
    )


def python_runtime_mounts(
    workspace: Path,
    path_roots: tuple[tuple[Path, Path], ...],
    executable_identities: tuple[tuple[str, Path, Path, int, int], ...],
    *,
    system_roots: tuple[Path, ...] = (),
) -> tuple[tuple[Path, Path], ...]:
    """Bind a user-local CPython runtime referenced by an identity-bound uv tool.

    Args:
        workspace (Path): Canonical workspace containing the active virtual environment.
        path_roots (tuple[tuple[Path, Path], ...]): Bound PATH spellings and canonical roots.
        executable_identities (tuple[tuple[str, Path, Path, int, int], ...]): Exact executable
            names, spellings, canonical files, and device/inode identities for this command.
        system_roots (tuple[Path, ...]): Authenticated system runtime roots already mounted.

    Returns:
        tuple[tuple[Path, Path], ...]: Canonical user-local runtime roots and their configured
            path aliases, or an empty tuple when the venv uses a system or workspace runtime.

    Raises:
        ValueError: The user home, venv metadata, interpreter, or runtime path is unsafe.
        OSError: Required user or runtime filesystem metadata cannot be inspected.
    """
    if not any(name == "uv" for name, *_ in executable_identities):
        return ()
    try:
        home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    except (KeyError, OSError, RuntimeError) as error:
        raise ValueError(
            "Current account home is unavailable for Python runtime binding."
        ) from error
    home_details = home.stat()
    if (
        not home.is_dir()
        or home_details.st_uid != os.getuid()
        or home_details.st_mode & constants.OTHER_WRITE_MODE_BIT
        or home_details.st_mode & constants.GROUP_WRITE_MODE_BIT
        and home_details.st_gid != os.getgid()
    ):
        raise ValueError("Current account home is unsafe for Python runtime binding.")

    for _, path_root in path_roots:
        if not path_root.is_relative_to(workspace) or path_root.name != "bin":
            continue
        config = path_root.parent / "pyvenv.cfg"
        try:
            before = config.lstat()
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_mode & constants.OTHER_WRITE_MODE_BIT
            or before.st_mode & constants.GROUP_WRITE_MODE_BIT
            and before.st_gid != os.getgid()
            or before.st_size > _MAX_VENV_CONFIG_BYTES
        ):
            raise ValueError("Workspace Python environment metadata is unsafe.")
        descriptor = os.open(config, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            bound = os.fstat(descriptor)
            if (before.st_dev, before.st_ino) != (bound.st_dev, bound.st_ino):
                raise ValueError("Workspace Python environment metadata changed.")
            content = os.read(descriptor, _MAX_VENV_CONFIG_BYTES + 1)
            latest = os.fstat(descriptor)
            if len(content) > _MAX_VENV_CONFIG_BYTES or (
                bound.st_dev,
                bound.st_ino,
                bound.st_size,
                bound.st_mtime_ns,
            ) != (latest.st_dev, latest.st_ino, latest.st_size, latest.st_mtime_ns):
                raise ValueError("Workspace Python environment metadata changed.")
        finally:
            os.close(descriptor)
        try:
            lines = content.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError("Workspace Python environment metadata is invalid.") from error
        values = {}
        for line in lines:
            key, separator, value = line.partition("=")
            if not separator:
                continue
            key = key.strip().casefold()
            if key in {"home", "implementation", "version_info"}:
                if key in values:
                    raise ValueError("Workspace Python environment metadata is ambiguous.")
                values[key] = value.strip()
        version = re.match(r"^(\d+)\.(\d+)(?:\.|$)", values.get("version_info", ""))
        home_value = values.get("home", "")
        if (
            values.get("implementation", "").casefold() != "cpython"
            or version is None
            or not home_value
            or "\x00" in home_value
        ):
            continue
        configured_home = Path(home_value)
        if not configured_home.is_absolute() or configured_home.name != "bin":
            raise ValueError("Workspace Python runtime path is invalid.")
        try:
            canonical_home = configured_home.resolve(strict=True)
            runtime_root = canonical_home.parent
            interpreter = (path_root / "python").resolve(strict=True)
            stdlib = runtime_root / "lib" / f"python{version.group(1)}.{version.group(2)}"
        except (OSError, RuntimeError):
            continue
        if (
            not canonical_home.is_dir()
            or not interpreter.is_relative_to(canonical_home)
            or not stat.S_ISREG(interpreter.stat().st_mode)
            or not interpreter.stat().st_mode & constants.EXECUTE_MODE_MASK
            or not stdlib.is_dir()
            or not stdlib.resolve(strict=True).is_relative_to(runtime_root)
            or configured_home.parent.resolve(strict=True) != runtime_root
        ):
            continue
        if any(runtime_root.is_relative_to(root) for root in system_roots):
            return ()
        if (
            not runtime_root.is_relative_to(home)
            or runtime_root == home
            or is_protected_external_read(runtime_root)
            or is_protected_external_read(configured_home)
            or runtime_root.is_relative_to(workspace)
        ):
            raise ValueError("Workspace Python runtime root is outside the safe user-local scope.")

        current = runtime_root
        while current != home:
            details = current.lstat()
            if (
                not stat.S_ISDIR(details.st_mode)
                or details.st_uid not in (0, os.getuid())
                or details.st_mode & constants.OTHER_WRITE_MODE_BIT
                or details.st_mode & constants.GROUP_WRITE_MODE_BIT
                and details.st_gid != os.getgid()
            ):
                raise ValueError("Workspace Python runtime directory is unsafe.")
            current = current.parent
        interpreter_details = interpreter.stat()
        if (
            interpreter_details.st_uid not in (0, os.getuid())
            or interpreter_details.st_nlink != 1
            or interpreter_details.st_mode & constants.OTHER_WRITE_MODE_BIT
            or interpreter_details.st_mode & constants.GROUP_WRITE_MODE_BIT
            and interpreter_details.st_gid != os.getgid()
        ):
            raise ValueError("Workspace Python interpreter is unsafe.")
        alias_root = configured_home.parent
        return ((runtime_root, alias_root),)
    return ()


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
                    is_private = private_parent or entry.name.casefold() in _PRIVATE_NAMES
                    inode = (details.st_dev, details.st_ino)
                    if stat.S_ISREG(details.st_mode) and (is_protected or is_private):
                        sensitive_inodes.add(inode)
                    if stat.S_ISREG(details.st_mode) and details.st_nlink > 1:
                        hardlinked[path] = inode
                    if (is_protected or is_private) and stat.S_ISLNK(details.st_mode):
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


def prepare_paths(
    request: SandboxRequest,
    scratch: Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    """Inspect all approved roots and return deterministic protected mount preparation inputs.

    Scan the workspace, scratch and additional read/write roots, including protected subtrees.
    Collect multiply linked regular files for read-only bind mounts, preserving reads and
    execution while preventing writes through those paths. Small sets use individual file mounts;
    larger sets use containing directories to stay below Bubblewrap's argument ceiling, which also
    makes other files in those directories read-only. Refuse hardlinks that alias a protected or
    private file found in the approved roots. Skip ordinary symlinks while rejecting symlinks
    within protected/private entries. Return ancestor directories for self-bind mountpoints so
    protected entries cannot escape through relocation; their unrelated contents retain the
    approved write semantics. Future protected filenames still require the production filename
    policy. Explicit top-level Git authority does not exempt nested Git, instructions, controls or
    private data. No filesystem mutation occurs.

    Args:
        request (SandboxRequest): Approved command with current canonical and alias identities
            and its execution deadline. Opaque source is never inspected or executed.
        scratch (Path): Live private real directory bound as the request's TMPDIR.

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
    roots = dict.fromkeys((request.workspace, *request.read_roots, *request.write_roots, scratch))
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
    if set(hardlinked.values()) & sensitive_inodes:
        raise ValueError("Linux hardlinked file aliases protected or private data.")
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
        hardlinked, (*request.write_roots, scratch), request.deadline
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
    inventories = prepare_paths(request, scratch)
    hardlinked = inventories[3]
    sources = dict.fromkeys(
        (
            request.workspace,
            *request.read_roots,
            *request.write_roots,
            scratch,
            *(canonical for _, canonical in request.path_roots),
            *(alias.parent for alias in request.read_aliases),
            *(alias.parent for _, alias, _ in request.aliases),
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
) -> Generator[tuple[tuple[str, ...], tuple[int, ...]], None, None]:
    """Prepare descriptor-backed request mounts without granting native launch authority.

    Bind approved roots at their canonical spellings, outer roots before nested roots. A path
    under any approved write root or private scratch remains writable; other roots and ancestors
    are read-only. Bind the fresh host scratch at its original spelling for trusted identity
    checks and at a stable namespace path for payload access. Apply ancestor mountpoints before
    protected read-only binds, then hide private directories with empty tmpfs mounts and private
    regular files with the supplied empty inode. Mirror every restrictive scratch overlay;
    selected workspaces below scratch retain their read-only role. Preserve original alias
    metadata for validation and recreate virtual aliases at stable, read-only namespace paths.
    Refuse a whole filesystem or home bind and aliases whose target is absent from this request
    view. The
    filename layer still protects future names.

    The caller must authenticate the helper and immutable mask-control inode, provide reviewed
    runtime/proc/device/control mounts and namespace/network setup, enforce the filename policy,
    and close every setup descriptor before source entry. Returned arguments are only a request
    filesystem fragment; they contain no shell source, capabilities or launch operation.

    Args:
        request (SandboxRequest): Current approved roots, aliases and execution deadline.
        scratch (Path): Live private directory bound as the request's TMPDIR.
        empty_mask (int | None): Caller-authenticated read-only empty regular-file descriptor
            for private file masks. Required only when such targets exist. The original descriptor
            remains caller-owned; this context duplicates and owns its mount-source descriptor.
        skip_path_roots (tuple[Path, ...]): Request PATH spellings already supplied by an
            authenticated runtime mount, whose duplicate directory mounts and aliases are omitted.
        skip_executable_roots (tuple[Path, ...]): Authenticated runtime roots that already supply
            named executable files, so no duplicate exact-file mount is needed.

    Yields:
        tuple[tuple[str, ...], tuple[int, ...]]: Bubblewrap filesystem argument fragment and
            descriptors held only for this context, to pass to trusted native setup. Private
            original-target descriptors are withheld from the returned inheritance list.

    Raises:
        ValueError: Request/source authority is unsafe or stale, a root covers the filesystem or
            home, a reserved namespace destination overlaps another grant, a skipped PATH root is
            not bound to the request, an alias target is outside the bound view, an executable
            source changes, or a required mask is missing/unsafe.
        TimeoutError: Preparation reaches the approved command deadline.
        OSError: Linux O_PATH is unavailable or source/mask descriptors cannot be inspected.
    """
    roots = {request.workspace, *request.read_roots, *request.write_roots, scratch}
    skipped_paths = set(skip_path_roots)
    path_roots = tuple(
        (spelling, canonical)
        for spelling, canonical in request.path_roots
        if spelling not in skipped_paths
    )
    if not skipped_paths <= {spelling for spelling, _ in request.path_roots}:
        raise ValueError("Linux skipped PATH roots are not bound to this request.")
    metadata = (
        {canonical for _, canonical in request.path_roots}
        | {alias.parent for alias in request.read_aliases}
        | {alias.parent for _, alias, _ in request.aliases}
    )
    mounted_metadata = (
        {canonical for _, canonical in path_roots}
        | {alias.parent for alias in request.read_aliases}
        | {alias.parent for _, alias, _ in request.aliases}
    )
    home = Path.home().resolve(strict=True)
    if any(root == Path("/") or home.is_relative_to(root) for root in roots | metadata):
        raise ValueError("Linux request mounts cannot expose the whole filesystem or home.")
    aliases = {
        **{alias: alias.resolve(strict=True) for alias in request.read_aliases},
        **{alias: target for _, alias, target in request.aliases},
    }
    if any(
        path == reserved or path.is_relative_to(reserved) or reserved.is_relative_to(path)
        for path in (*roots, *metadata, *aliases, *aliases.values())
        for reserved in (NATIVE_SCRATCH_PATH, NATIVE_ALIAS_PATH)
    ):
        raise ValueError("Reserved native namespace destination overlaps another mount.")
    if any(not any(target.is_relative_to(root) for root in roots) for target in aliases.values()):
        raise ValueError("Linux request alias target is outside the mounted view.")
    with bind_paths(request, scratch) as (inventories, descriptors), ExitStack() as owned:
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
                if any(path.is_relative_to(root) for root in writable)
                and not (
                    request.workspace.is_relative_to(scratch)
                    and path.is_relative_to(request.workspace)
                    and not any(path.is_relative_to(root) for root in request.write_roots)
                )
                else "--ro-bind-fd"
            )
            mount(option, descriptors[path], path)
            inherited.add(descriptors[path])
        for path in hardlinked:
            _check_deadline(request.deadline)
            mount("--ro-bind-fd", descriptors[path], path)
            inherited.add(descriptors[path])
        for path in protected:
            _check_deadline(request.deadline)
            if not any(path.is_relative_to(mask) for mask in private):
                mount("--ro-bind-fd", descriptors[path], path)
                inherited.add(descriptors[path])
        mask_descriptor = None
        for path in private:
            _check_deadline(request.deadline)
            if stat.S_ISDIR(os.fstat(descriptors[path]).st_mode):
                arguments.extend(("--tmpfs", str(path)))
                native = payload_path(path, scratch)
                if native != path:
                    scratch_mounts.extend(("--tmpfs", str(native)))
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
        if request.aliases:
            arguments.extend(("--dir", str(NATIVE_ALIAS_PATH)))
            for _, alias, target in payload_aliases(request):
                arguments.extend(("--symlink", str(payload_path(target, scratch)), str(alias)))
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
        yield tuple(arguments), tuple(sorted(inherited))
