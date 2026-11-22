"""Render the AppArmor filename layer for an already bound native command request.

This layer must be combined with authenticated protected admission, the reviewed mount/PID/
network boundary, descriptor filtering and irreversible bootstrap privilege removal. Rendering
does not compile or load policy, validate writable hardlinks, or make a Linux backend available.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from ....constants import GIT_DIRECTORY, NATIVE_UNIT_PREFIX
from ....permissions.protection import (
    ENVIRONMENT_TEMPLATE_SUFFIXES,
    PRIVATE_READ_COMPONENTS,
    is_protected_external_read,
    protected_environment_suffixes,
)
from ....utils.hashing import sha256_digest
from ..constants import TMPDIR_ENVIRONMENT_VARIABLE
from .constants import (
    NATIVE_CONVENTIONAL_TMP_PATH,
    NATIVE_SCRATCH_PATH,
    NATIVE_TRACKING_PATH,
    SESSION_CONVENTIONAL_TMP_NAME,
)
from .host import RUNTIME_READ_FILES, SYSTEM_CA_DIRECTORY
from .paths import conventional_tmp_available, payload_path

if TYPE_CHECKING:
    from ..contracts import SandboxRequest

_PROFILE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}")
_SYSTEM_READ_ROOTS = (Path("/usr"), Path("/bin"), Path("/lib"))
_PROFILE_PLACEHOLDER = "payload_" + "0" * 64


def environment_value_patterns() -> tuple[str, ...]:
    """Render protected environment basenames as byte-oriented AppArmor globs.

    Returns:
        tuple[str, ...]: Unanchored globs for the native file rules.
    """

    def variants(char: str) -> str:
        return char.lower() + char.upper() if char.isascii() else f"\\x{ord(char):02x}"

    def literal(value: str) -> str:
        return "".join(
            f"[{variants(char)}]"
            if char.isascii() and char.isalpha()
            else char
            if char.isascii()
            else variants(char)
            for char in value
        )

    words = [suffix[1:] for suffix in ENVIRONMENT_TEMPLATE_SUFFIXES]
    # The native byte matcher needs the UTF-8 spelling of the long-s fold explicitly.
    words.append("ſample".encode().decode("latin1"))
    patterns = [literal(".env")]
    for tail, excluded in protected_environment_suffixes(words):
        prefix = "*[^" + "".join(map(variants, excluded)) + "/]" if excluded else ""
        patterns.append(literal(".env.") + prefix + literal(tail))
    return tuple(patterns)


def _literal(value: str) -> str:
    """Encode UTF-8 bytes literally, preserving only path separators as syntax."""
    if "\x00" in value:
        raise ValueError("Policy paths cannot contain NUL.")
    return "".join("/" if byte == ord("/") else f"\\x{byte:02x}" for byte in value.encode())


def _tree(path: Path) -> str:
    """Represent a literal directory and its descendants without widening its spelling."""
    return _literal(str(path).rstrip("/")) + "/{,**}"


def payload_policy(
    request: SandboxRequest,
    scratch: Path,
    cache_lock: Path,
    profile: str,
    *,
    runtime_roots: tuple[Path, ...] = _SYSTEM_READ_ROOTS,
) -> str:
    """Render one enforcing payload profile from current, approved request authority.

    The trusted caller must authenticate the cache lock inode and combine this filename layer
    with the qualified native namespaces, protected-ancestor mounts, hardlink scan and filter.
    Network/process permissions here rely on that surrounding boundary; they grant no host
    network namespace or host process visibility. No helper, policy loading or lifecycle action
    is performed. Strict parser compilation and native composition qualification are required
    before using the result for command execution.

    Args:
        request (SandboxRequest): Authorized request with current directory/executable identities.
        scratch (Path): Private scratch already bound as the request's TMPDIR.
        cache_lock (Path): Absolute authenticated namespace-cache lock denied to payload code.
        profile (str): Simple profile name, at most 128 ASCII characters, excluding bootstrap.
        runtime_roots (tuple[Path, ...]): Host-authenticated runtime spellings and destinations.
            The default retains the standalone fixture layout; production preparation must use
            runtime_read_roots in the initial host view, never caller/model-selected roots.

    Returns:
        str: Complete single-profile AppArmor source with literal UTF-8 path encoding.

    Raises:
        ValueError: A profile/path is malformed, request identities are stale, scratch is not
            bound, or a read/write root includes protected private data.
    """
    if _PROFILE_NAME.fullmatch(profile) is None or profile == "bootstrap":
        raise ValueError("Invalid payload profile name.")
    if not runtime_roots or any(
        not root.is_absolute()
        or ".." in root.parts
        or root == Path("/")
        or is_protected_external_read(root)
        for root in runtime_roots
    ):
        raise ValueError("Invalid native runtime roots.")
    if not request.paths_are_current():
        raise ValueError("Policy request authority changed.")
    if request.shell_environment().get(TMPDIR_ENVIRONMENT_VARIABLE) != str(scratch):
        raise ValueError("Policy scratch is not bound to the request.")
    if not cache_lock.is_absolute() or ".." in cache_lock.parts or not cache_lock.name:
        raise ValueError("Policy cache lock must be absolute.")
    conventional = conventional_tmp_available(request, scratch, runtime_roots=runtime_roots)
    reads = tuple(
        dict.fromkeys(
            (
                payload_path(request.workspace, scratch),
                payload_path(NATIVE_SCRATCH_PATH, scratch),
                *((payload_path(NATIVE_CONVENTIONAL_TMP_PATH, scratch),) if conventional else ()),
                *sorted(
                    payload_path(path, scratch)
                    for path in (
                        *request.read_roots,
                        *request.read_aliases,
                    )
                ),
            )
        )
    )
    writes = tuple(
        dict.fromkeys(
            (
                *sorted(payload_path(path, scratch) for path in request.write_roots),
                payload_path(NATIVE_SCRATCH_PATH, scratch),
                *((payload_path(NATIVE_CONVENTIONAL_TMP_PATH, scratch),) if conventional else ()),
            )
        )
    )
    if any(is_protected_external_read(root) for root in (*reads, *writes)):
        raise ValueError("Protected private data cannot be a policy root.")
    executable_files = set()
    for _, spelling, resolved, _, _ in request.executable_identities:
        if is_protected_external_read(spelling) or is_protected_external_read(resolved):
            raise ValueError("Protected private data cannot be an executable.")
        if not any(resolved.is_relative_to(root) for root in (*runtime_roots, *reads, *writes)):
            executable_files.update((spelling, resolved))
    if any(
        denied.is_relative_to(root) or root.is_relative_to(denied)
        for denied in (*request.hard_denied_paths, *request.protected_host_roots)
        for root in runtime_roots
    ):
        raise ValueError("Hard denial requires refusing an encompassing runtime root.")
    lines = [
        "abi <abi/4.0>,",
        f"profile {profile} flags=(attach_disconnected) {{",
        *(f'  allow rmix "{_tree(root)}",' for root in dict.fromkeys(runtime_roots)),
        "  allow r /proc/{,**},",
        "  allow rw /dev/{,**},",
        # glibc publishes each semaphore from a staging name inside the private tmpfs.
        '  link subset "/dev/shm/sem.*" -> "/dev/shm/sem.*",',
        *(f'  allow r "{_literal(str(path))}",' for path in RUNTIME_READ_FILES),
        # TLS clients enumerate this synthetic directory when checking for hashed anchors.
        f'  allow r "{_literal(str(SYSTEM_CA_DIRECTORY))}/",',
        *(f'  allow rmix "{_tree(root)}",' for root in reads),
        *(f'  allow rmix "{_literal(str(path))}",' for path in sorted(executable_files)),
        *(f'  allow rwlkmix "{_tree(root)}",' for root in writes),
        *(f'  allow rwlkmix "{_literal(str(root).rstrip("/"))}/{{,**/}}*",' for root in writes),
        *(f'  allow rwk "{_literal(str(path))}",' for path in sorted(set(request.write_files))),
        "  allow network,",
        "  allow unix,",
        "  allow signal,",
        "  allow ptrace,",
        "  allow mqueue,",
        "  allow dbus,",
        "  audit deny capability,",
        "  audit deny userns,",
        "  deny mount,",
        "  deny umount,",
        "  deny /sys/kernel/security/{,**} rwkl,",
        f'  deny "{_tree(NATIVE_TRACKING_PATH)}" rwklmx,',
        f'  deny "{_literal(str(cache_lock))}" rwkl,',
    ]
    workspace = _literal(str(payload_path(request.workspace, scratch)).rstrip("/"))
    protection = request.protected_paths
    for name in (*protection.instruction_names, *protection.files):
        lines.append(f'  deny "{workspace}/{{,**/}}{_literal(name)}" wkl,')
    git_granted = request.git_create or request.workspace / GIT_DIRECTORY in writes
    for name in protection.directories:
        prefix = "*/{,**/}" if git_granted and name == GIT_DIRECTORY.as_posix() else "{,**/}"
        pattern = workspace + "/" + prefix + _literal(name)
        lines.extend((f'  deny "{pattern}" wkl,', f'  deny "{pattern}/{{,**}}" wkl,'))
    # Reused session scratch also needs filename guards for future private entries.
    private_roots = tuple(
        dict.fromkeys(
            root
            for root in (*runtime_roots, *reads, *writes)
            if (root != NATIVE_SCRATCH_PATH or request.session_scratch and not request.scratch_data)
            and (root != NATIVE_CONVENTIONAL_TMP_PATH or not conventional)
        )
    )
    for name in PRIVATE_READ_COMPONENTS:
        pattern = "".join(
            r"{[sS],\xc5\xbf}"
            if char.lower() == "s"
            else f"[{char.lower()}{char.upper()}]"
            if char.isalpha()
            else char
            for char in name
        )
        for root in private_roots:
            prefix = _literal(str(root).rstrip("/"))
            path = (prefix if prefix else "") + "/{,**/}" + pattern
            lines.extend((f'  deny "{path}" rwklmx,', f'  deny "{path}/{{,**}}" rwklmx,'))
    # Factor the Cartesian product as AARE alternatives instead of repeating long UTF-8
    # root spellings for every suffix. Retain the compiler's existing source-size ceiling.
    prefixes = ",".join(_literal(str(root).strip("/")) for root in private_roots)
    if len(private_roots) > 1:
        prefixes = "{" + prefixes + "}"
    values = ",".join(environment_value_patterns())
    lines.append(f'  deny "/{prefixes}/{{,**/}}{{{values}}}{{,/**}}" rwklmx,')
    # Scratch is mounted at a disjoint namespace path, so protecting its physical
    # application parent does not cover the active agent-data mount.
    for path in request.protected_host_roots:
        if request.scratch_root is not None and path.is_relative_to(request.scratch_root):
            continue
        lines.append(f'  deny "{_literal(str(path))}{{,/**}}" rwklmx,')
    for path in request.hard_denied_paths:
        denied = (
            NATIVE_SCRATCH_PATH if scratch.is_relative_to(path) else payload_path(path, scratch)
        )
        lines.append(f'  deny "{_literal(str(denied))}{{,/**}}" rwklmx,')
        conventional_root = scratch / SESSION_CONVENTIONAL_TMP_NAME
        if conventional and (
            path.is_relative_to(conventional_root) or conventional_root.is_relative_to(path)
        ):
            alternate = (
                NATIVE_CONVENTIONAL_TMP_PATH / path.relative_to(conventional_root)
                if path.is_relative_to(conventional_root)
                else NATIVE_CONVENTIONAL_TMP_PATH
            )
            lines.append(f'  deny "{_literal(str(alternate))}{{,/**}}" rwklmx,')
    return "\n".join((*lines, "}", ""))


def payload_profile(
    request: SandboxRequest,
    scratch: Path,
    cache_lock: Path,
    runtime_roots: tuple[Path, ...],
) -> str:
    """Name a profile from its exact policy while excluding its self-referential name.

    Args:
        request (SandboxRequest): Current authorized request and host scratch binding.
        scratch (Path): Private host scratch directory mounted at the stable namespace path.
        cache_lock (Path): Authenticated cache lock path denied by the generated policy.
        runtime_roots (tuple[Path, ...]): Authenticated protected runtime directories.

    Returns:
        str: Deterministic profile name for the rendered policy authority.

    Raises:
        ValueError: The request, scratch binding, runtime roots, or canonical profile template
            cannot be rendered safely.
    """
    source = payload_policy(
        request,
        scratch,
        cache_lock,
        _PROFILE_PLACEHOLDER,
        runtime_roots=runtime_roots,
    )
    header = f"profile {_PROFILE_PLACEHOLDER} flags=(attach_disconnected) {{"
    canonical_header = "profile <native-payload-profile> flags=(attach_disconnected) {"
    if source.count(header) != 1:
        raise ValueError("Payload profile template does not contain one exact profile name.")
    return "payload_" + sha256_digest(source.replace(header, canonical_header, 1))


def runtime_policy_sources(
    directory: Path,
    generation: str,
) -> tuple[dict[str, dict[str, bytes]], bytes]:
    """Render fixed trusted setup policies for an administrator-reviewed installed runtime.

    Setup grants are usable only by the protected trusted receiver's isolated helper. Opaque
    source enters a separately compiled filename policy after capability and descriptor removal.
    Rendering grants no setup consent or public readiness and never loads kernel policy.

    Args:
        directory (Path): Literal protected private runtime installation directory.
        generation (str): Fixed simple namespace identity selected by trusted setup.

    Returns:
        tuple[dict[str, dict[str, bytes]], bytes]: Initial namespace sources and global attachment.

    Raises:
        ValueError: Installation spelling or generation identity is unsafe.
    """
    if (
        re.fullmatch(r"/[A-Za-z0-9_./-]+", str(directory)) is None
        or directory == Path("/")
        or ".." in directory.parts
        or _PROFILE_NAME.fullmatch(generation) is None
    ):
        raise ValueError("Invalid trusted runtime policy identity.")
    bootstrap = b"""abi <abi/4.0>,
profile bootstrap flags=(attach_disconnected) {
  allow file rwlkmix /{**,},
  allow capability mac_admin,
  allow capability setpcap,
  allow network,
  allow unix,
  allow signal,
  allow ptrace,
  allow mqueue,
  allow dbus,
  change_profile -> &payload_*,
}
"""
    attachment = f"""abi <abi/4.0>,
profile {NATIVE_UNIT_PREFIX}{generation} "{directory}/helper" flags=(attach_disconnected) {{
  allow file rwlkmix /{{**,}},
  allow capability,
  allow userns,
  allow mount,
  allow umount,
  allow pivot_root,
  allow network,
  allow unix,
  allow signal,
  allow ptrace,
  allow mqueue,
  allow dbus,
  change_profile -> &:{generation}:bootstrap,
}}
""".encode("ascii")
    return {"": {"bootstrap": bootstrap}}, attachment
