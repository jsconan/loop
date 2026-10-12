"""Own fixed namespace and descriptor-filter arguments while keeping sources alive.

This owner starts no process and grants no command authority. The trusted launcher must supply
original authenticated helper and control mounts and the fixed bootstrap entry.
"""

from __future__ import annotations

import os
import stat
import tempfile
import time
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from .host import protected_path, runtime_read_files, runtime_read_roots
from .paths import mount_arguments, python_runtime_mounts
from .seccomp import descriptor_filter

if TYPE_CHECKING:
    from ..contracts import SandboxRequest


def _runtime_path_alias_mounts(
    request: SandboxRequest,
    runtime_roots: tuple[tuple[Path, Path], ...],
    runtime_files: tuple[Path, ...],
    scratch: Path,
    extra_aliases: tuple[tuple[Path, Path], ...] = (),
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Recreate aliases for authenticated runtime PATHs and approved Python installations."""
    runtime_paths = tuple(dict.fromkeys(path for pair in runtime_roots for path in pair))

    def inside_runtime(path: Path) -> bool:
        """Return whether a path is already present in an authenticated runtime tree."""
        return any(path.is_relative_to(root) for root in runtime_paths)

    candidates = tuple(
        (spelling, canonical)
        for spelling, canonical in request.path_roots
        if inside_runtime(spelling) and inside_runtime(canonical)
    )
    if not candidates and not extra_aliases:
        return (), ()

    request_directories = {request.workspace, *request.read_roots, *request.write_roots, scratch}
    aliases: dict[Path, Path] = {}
    covered = []
    for spelling, canonical in candidates:
        path_aliases = {}
        eligible = True
        for link in reversed((spelling, *spelling.parents)):
            if not link.is_symlink():
                continue
            value = link.readlink()
            target = Path(
                os.path.normpath(str(value if value.is_absolute() else link.parent / value))
            )
            resolved = link.resolve(strict=True)
            if target.resolve(strict=True) != resolved:
                raise ValueError("Native runtime PATH link changed during binding.")
            if not inside_runtime(resolved):
                eligible = False
                break
            if inside_runtime(target):
                continue
            authenticated = protected_path(target, directory=True)
            if authenticated != resolved:
                raise ValueError("Native runtime PATH target authority differs.")
            if any(
                target == mounted
                or target.is_relative_to(mounted)
                or mounted.is_relative_to(target)
                for mounted in (
                    *runtime_files,
                    *request_directories,
                    *(alias.parent for alias in request.read_aliases),
                    *(alias.parent for _, alias, _ in request.aliases),
                    *(target for _, _, target in request.aliases),
                    *(
                        path
                        for other_spelling, path in request.path_roots
                        if other_spelling != spelling
                    ),
                )
            ):
                raise ValueError("Native runtime PATH alias overlaps another mount.")
            path_aliases[target] = authenticated
        if not eligible:
            continue
        covered.append((spelling, canonical))
        for target, resolved in path_aliases.items():
            previous = aliases.get(target)
            if previous is not None and previous != resolved:
                raise ValueError("Native runtime PATH aliases have conflicting targets.")
            aliases[target] = resolved

    for alias, target in extra_aliases:
        if alias == target:
            continue
        if any(
            alias == mounted or alias.is_relative_to(mounted) or mounted.is_relative_to(alias)
            for mounted in (
                *runtime_files,
                *request_directories,
                *(alias_path.parent for alias_path in request.read_aliases),
                *(alias_path.parent for _, alias_path, _ in request.aliases),
                *(alias_target for _, _, alias_target in request.aliases),
                *(path for _, path in request.path_roots),
            )
        ):
            raise ValueError("Native Python runtime alias overlaps another mount.")
        aliases[alias] = target

    if not covered and not aliases:
        return (), ()
    covered_spellings = {spelling for spelling, _ in covered}
    request_directories.update(
        canonical for spelling, canonical in request.path_roots if spelling not in covered_spellings
    )
    request_directories.update(alias.parent for alias in request.read_aliases)
    request_directories.update(alias.parent for _, alias, _ in request.aliases)
    request_directories.update(target for _, _, target in request.aliases)

    if any(
        left != right and (left.is_relative_to(right) or right.is_relative_to(left))
        for index, left in enumerate(aliases)
        for right in tuple(aliases)[index + 1 :]
    ):
        raise ValueError("Native runtime PATH aliases overlap.")

    available_directories = {Path("/")}
    directory_mounts = (*runtime_paths, *request_directories)
    for directory in directory_mounts:
        available_directories.update(directory.parents)
        available_directories.add(directory)
    for file in runtime_files:
        available_directories.update(file.parents)

    arguments = []
    for target, resolved in sorted(
        aliases.items(), key=lambda item: (len(item[0].parts), str(item[0]))
    ):
        for parent in reversed(target.parents):
            if parent in available_directories or inside_runtime(parent):
                continue
            arguments.extend(("--dir", str(parent)))
            available_directories.add(parent)
        arguments.extend(("--symlink", str(resolved), str(target)))
    return tuple(covered_spellings), tuple(arguments)


@contextmanager
def namespace_arguments(
    request: SandboxRequest,
    scratch: Path,
    *,
    empty_mask: int,
) -> Generator[tuple[tuple[str, ...], tuple[int, ...]], None, None]:
    """Keep fixed isolated setup arguments and inherited source/filter descriptors owned.

    Reuse the filesystem owner before generating the production filter. Resolve live protected
    symlink targets needed by PATH roots that are already covered by native runtime mounts. Isolate
    user, mount, PID and IPC views, map the ordinary caller to namespace zero, create a new session,
    die with the parent and drop all capabilities. Isolate network unless explicitly granted.
    Pin protected runtime spellings and canonical destinations with read-only O_PATH mounts, and
    pin named external executables by identity at their exact PATH target. Then add proc/dev and
    the bound cwd without exposing the host root. Sources and filter remain alive through the
    trusted launch context and close on every exit. Bootstrap capability
    additions and control mounts require their separately authenticated trusted owner.
    A command may start only after enforcing policy entry and closure of setup descriptors.

    Args:
        request (SandboxRequest): Already authorized, current shared request and original deadline.
        scratch (Path): Private real scratch bound as the request's TMPDIR.
        empty_mask (int): Original authenticated read-only empty control inode descriptor.

    Yields:
        tuple[tuple[str, ...], tuple[int, ...]]: Fixed Bubblewrap argument fragment and descriptors
            to inherit into trusted setup, valid only while this context remains open.

    Raises:
        ValueError: Filesystem/request authority or native filter generation is invalid.
        TimeoutError: The original preparation deadline expires.
        OSError: Source/filter authority, descriptor I/O or resource preparation fails.
    """
    with ExitStack() as runtime, tempfile.TemporaryFile() as stream:
        runtime_mounts, runtime_descriptors = [], []
        runtime_bindings = []
        path_flag = getattr(os, "O_PATH", None)
        if path_flag is None:
            raise OSError("Runtime mount preparation requires Linux O_PATH.")
        for spelling in runtime_read_roots():
            canonical = protected_path(spelling, directory=True)
            before = canonical.stat()
            descriptor = os.open(canonical, path_flag | os.O_NOFOLLOW | os.O_CLOEXEC)
            runtime.callback(os.close, descriptor)
            bound = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(bound.st_mode)
                or (bound.st_dev, bound.st_ino, bound.st_mode, bound.st_uid, bound.st_gid)
                != (before.st_dev, before.st_ino, before.st_mode, before.st_uid, before.st_gid)
                or protected_path(spelling, directory=True) != canonical
                or canonical.stat() != before
            ):
                raise ValueError("Native runtime directory changed during binding.")
            runtime_mounts.extend(("--ro-bind-fd", str(descriptor), str(canonical)))
            if spelling != canonical:
                runtime_mounts.extend(("--symlink", str(canonical), str(spelling)))
            runtime_descriptors.append(descriptor)
            runtime_bindings.append((spelling, canonical))
        file_bindings = runtime_read_files(deadline=request.deadline)
        for spelling, authority in file_bindings:
            descriptor = runtime.enter_context(authority.open_verified(deadline=request.deadline))
            runtime_mounts.extend(("--ro-bind-fd", str(descriptor), str(spelling)))
            runtime_descriptors.append(descriptor)
        executable_identities = getattr(request, "executable_identities", ())
        if not isinstance(executable_identities, tuple):
            executable_identities = ()
        python_mounts = python_runtime_mounts(
            request.workspace,
            request.path_roots,
            executable_identities,
            system_roots=tuple(path for _, path in runtime_bindings),
        )
        if any(
            root not in request.read_roots or root not in request.automatic_tool_reads
            for root, _ in python_mounts
        ):
            raise ValueError("Native Python runtime authority changed during preparation.")
        covered_paths, runtime_aliases = _runtime_path_alias_mounts(
            request,
            tuple(runtime_bindings),
            tuple(path for path, _ in file_bindings),
            scratch,
            tuple((alias, root) for root, alias in python_mounts),
        )
        runtime_mounts.extend(runtime_aliases)
        with mount_arguments(
            request,
            scratch,
            empty_mask=empty_mask,
            skip_path_roots=covered_paths,
            skip_executable_roots=tuple(canonical for _, canonical in runtime_bindings),
        ) as (mounts, sources):
            data = descriptor_filter(deadline=request.deadline)
            if stream.write(data) != len(data):
                raise OSError("Namespace descriptor filter write was incomplete.")
            stream.seek(0)
            if time.monotonic() >= request.deadline:
                raise TimeoutError("Namespace preparation request deadline expired.")
            arguments = (
                "--unshare-user",
                "--uid",
                "0",
                "--gid",
                "0",
                "--unshare-pid",
                "--unshare-ipc",
                "--new-session",
                "--die-with-parent",
                "--cap-drop",
                "ALL",
                *(("--unshare-net",) if not request.network else ()),
                *runtime_mounts,
                *mounts,
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--chdir",
                str(request.cwd),
                "--seccomp",
                str(stream.fileno()),
            )
            yield arguments, (*runtime_descriptors, *sources, stream.fileno())
