"""Remove namespace-local setup authority before entering an already loaded payload policy.

This module is for an isolated, trusted bootstrap child, not the application process. Its caller
must authenticate the helper and loaded bootstrap, compile and load the bound payload policy,
and terminate the child on any failure. It does not select a backend or authorize a policy.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import re
import shlex
import socket
import stat
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .... import constants
from ....constants import MAX_NATIVE_POLICY_MODE_BYTES, MAX_NATIVE_PROFILE_NAME_BYTES
from ....utils.hashing import sha256_digest
from ..constants import TMPDIR_ENVIRONMENT_VARIABLE
from .constants import (
    NATIVE_PAYLOAD_MARKER_PREFIX,
    NATIVE_PROFILE_GROUP_HASH_CHARS,
    NATIVE_SCRATCH_PATH,
    NATIVE_TRACKING_PATH,
    PROC_STATUS_PATH,
    PROC_UID_MAP_PATH,
)
from .egress import start_namespace_proxy
from .host import MAX_COMPILED_BYTES, HostFile, read_policy_bytes
from .installation import MAX_RETAINED_BYTES, generation_limits
from .seccomp import install_local_network_filter

if TYPE_CHECKING:
    from ..contracts import SandboxRequest

_DIRECT_NAMESPACE_LEVEL = 1
_NESTED_NAMESPACE_LEVEL = 2
_MAX_BOOTSTRAP_DESCRIPTORS = 256
_CAP_MAC_ADMIN = 33
_CAP_SETPCAP = 8
_MAX_CAPABILITY_NUMBER = 63
_CAPABILITY_HEADER_VERSION = 0x20080522
_PR_CAPBSET_DROP = 24
_PR_CAP_AMBIENT = 47
_PR_CAP_AMBIENT_CLEAR_ALL = 4
_PR_SET_NO_NEW_PRIVS = 38
_LOCK_RETRY_SECONDS = 0.01


def _payload_environment(request: SandboxRequest) -> dict[str, str]:
    """Map generated host scratch paths to the request-private namespace spelling."""
    environment = request.shell_environment()
    scratch_value = environment.get(TMPDIR_ENVIRONMENT_VARIABLE)
    if scratch_value is None:
        if request.generated_environment:
            raise ValueError("Generated scratch environment has no TMPDIR.")
        return environment
    host_scratch = Path(scratch_value)
    for name in dict.fromkeys((TMPDIR_ENVIRONMENT_VARIABLE, *request.generated_environment)):
        value = environment.get(name)
        if value is None:
            raise ValueError("Generated scratch environment is incomplete.")
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Generated scratch environment path is unsafe.")
        try:
            relative = path.relative_to(host_scratch)
        except ValueError as error:
            raise ValueError("Generated environment escaped private scratch.") from error
        environment[name] = str(NATIVE_SCRATCH_PATH / relative)
    return environment


class CapabilityHeader(ctypes.Structure):
    """Represent the Linux version-three capability header."""

    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class CapabilityWord(ctypes.Structure):
    """Represent one effective, permitted and inheritable capability word."""

    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


def _status() -> dict[str, int]:
    """Require all capability sets and NoNewPrivs from the current process status."""
    fields = {}
    for line in PROC_STATUS_PATH.read_text(encoding="ascii").splitlines():
        key, _, value = line.partition(":")
        if key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb", "NoNewPrivs"):
            if key in fields:
                raise ValueError("Duplicate bootstrap status field.")
            fields[key] = int(value.strip(), 10 if key == "NoNewPrivs" else 16)

    if len(fields) != 6:
        raise ValueError("Incomplete bootstrap status.")
    return fields


def _require_native(result: int, operation: str) -> None:
    """Stop on a failed native operation without continuing toward payload execution."""
    if result != 0:
        raise OSError(ctypes.get_errno(), operation)


def _transition_identity(profile: str, namespace: str, host_uid: int, namespace_level: int) -> None:
    """Reject malformed stack names, host mappings and unsupported namespace depths."""
    if (
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", profile)
        or profile == "bootstrap"
        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", namespace)
        or not isinstance(host_uid, int)
        or isinstance(host_uid, bool)
        or host_uid <= 0
        or not isinstance(namespace_level, int)
        or isinstance(namespace_level, bool)
        or namespace_level not in (_DIRECT_NAMESPACE_LEVEL, _NESTED_NAMESPACE_LEVEL)
    ):
        raise ValueError("Invalid payload transition identity.")


def enter_bootstrap(
    apparmor: HostFile,
    namespace: str,
    host_uid: int,
    *,
    namespace_level: int = _DIRECT_NAMESPACE_LEVEL,
) -> None:
    """Stack the fixed enforcing setup profile through authenticated native library authority.

    Call only from the trusted isolated bootstrap before executing any request source. The
    initial host owner must authenticate the helper, loaded generation and this library, and
    supply the original generation namespace and ordinary UID. Require its exact one-ID mapped
    user view before the native transition, then verify enforcing bootstrap identity, namespace
    stack/depth, NoNewPrivs and only the two namespace-local setup capabilities. This operation
    authorizes no command, loads no policy and leaves setup capabilities for trusted preparation;
    activate_payload must revoke them and close setup descriptors before command entry.
    Any failure requires the owning launcher to terminate its confined child.

    Args:
        apparmor (HostFile): Initial-host-authenticated libapparmor inode and original bytes.
        namespace (str): Original installed generation namespace, never request-selected.
        host_uid (int): Positive ordinary host UID mapped alone to namespace UID zero.
        namespace_level (int): Expected direct or nested namespace depth; defaults to direct.

    Raises:
        ValueError: Transition identity, mapped UID or post-transition enforcing setup differs.
        OSError: Native library authority, required API or profile transition fails.
    """
    _transition_identity("payload", namespace, host_uid, namespace_level)
    if os.getuid() != 0 or PROC_UID_MAP_PATH.read_text(encoding="ascii").split() != [
        "0",
        str(host_uid),
        "1",
    ]:
        raise ValueError("Unverified bootstrap user mapping.")
    with apparmor.open_verified() as descriptor:
        library = ctypes.CDLL(f"/proc/self/fd/{descriptor}", use_errno=True)
        try:
            transition = library.aa_change_profile
        except AttributeError as error:
            raise OSError("Required libapparmor API is unavailable.") from error

        transition.argtypes = [ctypes.c_char_p]
        transition.restype = ctypes.c_int
        _require_native(
            transition(("&:" + namespace + ":bootstrap").encode("ascii")),
            "Bootstrap profile stack failed.",
        )

    _require_bootstrap("payload", namespace, host_uid, namespace_level)


def _require_bootstrap(profile: str, namespace: str, host_uid: int, namespace_level: int) -> Path:
    """Require the mapped, enforcing setup context before inspecting or changing policy."""
    _transition_identity(profile, namespace, host_uid, namespace_level)
    security = Path("/sys/kernel/security/apparmor")
    if (
        os.getuid() != 0
        or PROC_UID_MAP_PATH.read_text(encoding="ascii").split() != ["0", str(host_uid), "1"]
        or Path("/proc/self/attr/current").read_text(encoding="ascii").strip()
        != "bootstrap (enforce)"
        or (security / ".ns_name").read_text(encoding="ascii").strip() != namespace
        or (security / ".ns_level").read_text(encoding="ascii").strip() != str(namespace_level)
        or (security / ".ns_stacked").read_text(encoding="ascii").strip() != "yes"
    ):
        raise ValueError("Unverified bootstrap confinement.")

    before = _status()
    # MAC_ADMIN and SETPCAP, in this user namespace only.
    setup_caps = (1 << _CAP_MAC_ADMIN) | (1 << _CAP_SETPCAP)
    if (
        before["NoNewPrivs"] != 1
        or before["CapEff"] != setup_caps
        or any(value & ~setup_caps for key, value in before.items() if key != "NoNewPrivs")
    ):
        raise ValueError("Unexpected bootstrap capabilities.")

    return security


def _cached_profiles(security: Path, capacity: int) -> dict[str, Path]:
    """Require a bounded flat enforcing inventory in the current policy namespace."""
    profiles = {}
    for directory in (security / "policy/profiles").iterdir():
        if len(profiles) >= capacity + 1:
            raise ValueError("Policy cache exceeds its capacity.")

        name = (
            read_policy_bytes(directory / "name", MAX_NATIVE_PROFILE_NAME_BYTES)
            .decode("ascii")
            .strip()
        )
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", name) or name in profiles:
            raise ValueError("Ambiguous policy cache inventory.")

        if (
            read_policy_bytes(directory / "mode", MAX_NATIVE_POLICY_MODE_BYTES).strip()
            != b"enforce"
        ):
            raise ValueError("Policy cache is not enforcing.")

        children = directory / "profiles"
        if children.exists() and next(children.iterdir(), None) is not None:
            raise ValueError("Nested profiles are not cache entries.")

        profiles[name] = directory

    if "bootstrap" not in profiles:
        raise ValueError("Policy cache has no bootstrap profile.")

    return profiles


def _reclaim_profiles(
    security: Path,
    profiles: dict[str, Path],
    profile: str,
    namespace: str,
    tracking: tuple[Path, tuple[int, int], str],
) -> dict[str, Path]:
    """Retire request profiles only after their protected service cgroup has disappeared."""
    root, identity, own_group = tracking
    info = root.stat(follow_symlinks=False)
    if (
        root != NATIVE_TRACKING_PATH
        or not stat.S_ISDIR(info.st_mode)
        or (info.st_dev, info.st_ino) != identity
    ):
        raise ValueError("Generation tracking authority changed.")
    group_pattern = re.compile(
        re.escape(constants.NATIVE_UNIT_PREFIX + namespace) + r"@[A-Za-z0-9_.:\\-]{1,128}\.service"
    )
    active = set()
    own_found = False
    for entry in root.iterdir():
        if entry.is_symlink():
            raise ValueError("Aliased generation tracking entry.")
        if not entry.is_dir():
            continue
        if group_pattern.fullmatch(entry.name) is None:
            raise ValueError("Unexpected generation tracking entry.")
        own_found |= entry.name == own_group
        active.add(sha256_digest(entry.name)[:NATIVE_PROFILE_GROUP_HASH_CHARS])
    own_suffix = sha256_digest(own_group)[:NATIVE_PROFILE_GROUP_HASH_CHARS]
    if not own_found or not profile.endswith("_" + own_suffix):
        raise ValueError("Current request differs from protected generation tracking.")
    pattern = re.compile(rf"payload_[a-f0-9]{{64}}_([a-f0-9]{{{NATIVE_PROFILE_GROUP_HASH_CHARS}}})")
    retained = dict(profiles)
    for name in profiles:
        if name == "bootstrap":
            continue
        match = pattern.fullmatch(name)
        if match is None:
            raise ValueError("Unrecognized retained payload profile.")
        if match.group(1) in active:
            continue
        with (security / ".remove").open("wb", buffering=0) as stream:
            if stream.write(name.encode("ascii")) != len(name):
                raise OSError(errno.EIO, "Incomplete payload policy removal.")
        del retained[name]
    return retained


def load_cached_payload(
    profile: str,
    compiled_policy: bytes,
    namespace: str,
    host_uid: int,
    lock: Path,
    capacity: int,
    *,
    namespace_level: int = _DIRECT_NAMESPACE_LEVEL,
    aggregate_limit: int = MAX_RETAINED_BYTES,
    deadline: float | None = None,
    tracking: tuple[Path, tuple[int, int], str] | None = None,
    bootstrap_policy: bytes | None = None,
) -> bool:
    """Add or exactly reuse one trusted compiled payload policy without replacement.

    Call only in the trusted isolated bootstrap, before capability removal. The caller must
    authenticate the helper, bootstrap and securityfs, bind a root-owned read-only lock inode
    shared by every bootstrap in this namespace, and supply a strictly compiled single-profile
    policy bound to the authorized request. This function does not validate policy semantics.
    AppArmor must grant lock and write permission for exclusive locking of the lock inode;
    its mount and ownership must still prevent content writes by the bootstrap.
    Every cooperating loader opens the lock independently; a supplied deadline bounds waiting for
    a held lock or active profiles occupying the reserved capacity. Without a deadline, a held lock
    or exhausted capacity fails immediately. No payload source may run on
    failure. Kernel exports must be available, and
    existing entries are reused only when enforcing and byte-identical to the compiled input.
    Existing entries are never replaced. When supplied authenticated protected service tracking,
    remove only request profiles whose owning service cgroup no longer exists.
    Under that same lock, bound aggregate exported policy bytes (including bootstrap) to
    sixteen MiB before admitting new bytes. This is a namespace-local bound; the production
    admission owner must additionally reserve installation-wide capacity across generations.

    Args:
        profile (str): Simple, unqualified name of the one compiled payload profile.
        compiled_policy (bytes): Trusted single-profile parser output, at most one MiB.
        namespace (str): Expected immediate AppArmor namespace created by reviewed setup.
        host_uid (int): Non-root ordinary caller UID mapped to isolated UID zero in the
            immediate parent namespace. Nested callers must authenticate that parent's sole
            UID/GID mapping to the same non-root host identity before creating this namespace.
        lock (Path): Authenticated absolute regular lock path, read-only with one link and
            contents equal to the namespace name followed by a newline.
        capacity (int): Positive bound on payload variants, excluding the bootstrap profile.
            The production owner must select and qualify this bound, not accept model input.
        aggregate_limit (int): Trusted reserved namespace byte ceiling, positive and at most
            sixteen MiB, including its bootstrap. Not caller/model-selected authority.
        namespace_level (int): Exact expected AppArmor namespace depth, one by default. Two
            requires a trusted caller that authenticated the root-created outer namespace,
            protected generation membership and complete target namespace binding before
            establishing the inner mapping. Other depths and non-integer values are refused.
        deadline (float | None): Absolute monotonic deadline for lock contention; None refuses a
            busy lock immediately.
        tracking (tuple[Path, tuple[int, int], str] | None): Trusted read-only generation cgroup
            mount, original device/inode and the current service name, or None for add-only use.
        bootstrap_policy (bytes | None): Original compiled bootstrap bytes to verify while the
            cache lock is held, or None when the caller does not require that attestation here.

    Returns:
        bool: True when added and verified, or False when exactly reused without a load.

    Raises:
        ValueError: Input, setup confinement, lock, inventory or loaded bytes are invalid.
        TimeoutError: The supplied deadline expires while waiting for the lock.
        OSError: Required native reads, locking or add-only loading fail, including a busy lock
            or a capacity refusal (ENOSPC). A failed verification may leave an unusable retained
            profile; callers must fail closed and request administrator recovery.
    """
    if (
        not isinstance(compiled_policy, bytes)
        or not 0 < len(compiled_policy) <= MAX_COMPILED_BYTES
        or not isinstance(capacity, int)
        or isinstance(capacity, bool)
        or capacity <= 0
        or not isinstance(aggregate_limit, int)
        or isinstance(aggregate_limit, bool)
        or not 0 < aggregate_limit <= MAX_RETAINED_BYTES
        or not lock.is_absolute()
    ):
        raise ValueError("Invalid policy cache input.")

    security = _require_bootstrap(profile, namespace, host_uid, namespace_level)
    descriptor = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)

    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & constants.WRITE_MODE_MASK
            or metadata.st_size != len(namespace) + 1
            or os.read(descriptor, MAX_NATIVE_PROFILE_NAME_BYTES + 1)
            != (namespace + "\n").encode("ascii")
        ):
            raise ValueError("Invalid policy cache lock.")

        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                if deadline is None:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Policy admission deadline expired.") from error
                time.sleep(min(_LOCK_RETRY_SECONDS, remaining))
                continue
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Policy admission deadline expired.")
            profiles = _cached_profiles(security, capacity)
            if bootstrap_policy is not None and (
                read_policy_bytes(profiles["bootstrap"] / "raw_data", MAX_COMPILED_BYTES)
                != bootstrap_policy
            ):
                raise ValueError("Loaded bootstrap differs from installed authority.")
            if tracking is not None:
                expected = _reclaim_profiles(security, profiles, profile, namespace, tracking)
                profiles = _cached_profiles(security, capacity)
                if profiles != expected:
                    raise ValueError("Unexpected policy cache change during reclamation.")
            sizes = {
                name: len(read_policy_bytes(directory / "raw_data", MAX_COMPILED_BYTES))
                for name, directory in profiles.items()
            }
            existing = profiles.get(profile)
            if existing is not None:
                if sum(sizes.values()) > aggregate_limit:
                    raise OSError(errno.ENOSPC, "Retained generation policy bytes exceed capacity.")
                if read_policy_bytes(existing / "raw_data", MAX_COMPILED_BYTES) != compiled_policy:
                    raise ValueError("Loaded policy differs from the bound input.")
                return False
            byte_full = sum(sizes.values()) + len(compiled_policy) > aggregate_limit
            count_full = len(profiles) - 1 >= capacity
            if not byte_full and not count_full:
                break
            if (
                tracking is None
                or deadline is None
                or sizes["bootstrap"] + len(compiled_policy) > aggregate_limit
            ):
                message = (
                    "Retained generation policy bytes exceed capacity."
                    if byte_full
                    else "Policy cache full; administrator refresh required."
                )
                raise OSError(errno.ENOSPC, message)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Policy admission deadline expired.")
            time.sleep(min(_LOCK_RETRY_SECONDS, remaining))

        with (security / ".load").open("wb", buffering=0) as stream:
            if stream.write(compiled_policy) != len(compiled_policy):
                raise OSError(errno.EIO, "Incomplete add-only policy load.")

        loaded = _cached_profiles(security, capacity)
        if set(loaded) != set(profiles) | {profile}:
            raise ValueError("Unexpected policy cache change during admission.")

        if read_policy_bytes(loaded[profile] / "raw_data", MAX_COMPILED_BYTES) != compiled_policy:
            raise ValueError("Loaded policy differs from the bound input.")

        if bootstrap_policy is not None and (
            read_policy_bytes(loaded["bootstrap"] / "raw_data", MAX_COMPILED_BYTES)
            != bootstrap_policy
        ):
            raise ValueError("Loaded bootstrap differs from installed authority.")

        return True
    finally:
        os.close(descriptor)


def enter_payload(
    apparmor: HostFile,
    profile: str,
    namespace: str,
    host_uid: int,
    *,
    namespace_level: int = _DIRECT_NAMESPACE_LEVEL,
) -> None:
    """Irreversibly remove bootstrap capabilities and stack a trusted payload profile.

    Call only in the isolated bootstrap child after loading its authenticated, bound policy.
    Prepare the AppArmor library through apparmor_library in the initial host view first.
    The bootstrap loads only its revalidated inode and closes that descriptor before revocation.
    A successful return establishes zero capability sets, NoNewPrivs and the exact visible
    bootstrap/payload stack. Never catch failure and execute a command in the setup context.

    Args:
        apparmor (HostFile): Trusted host-prepared AppArmor library authority, never request input.
        profile (str): Already loaded payload profile's simple, unqualified name.
        namespace (str): Expected immediate AppArmor namespace created by reviewed setup.
        host_uid (int): Non-root ordinary caller UID mapped to isolated UID zero in the
            immediate parent namespace. Nested callers must authenticate that parent's sole
            UID/GID mapping to the same non-root host identity before creating this namespace.
        namespace_level (int): Exact expected AppArmor namespace depth, one by default. Two
            requires the authenticated root-created outer namespace and protected generation
            context described by load_cached_payload. Other depths and non-integers are refused.

    Raises:
        ValueError: An argument, namespace, mapping, capability or final-state invariant fails.
        OSError: A required native interface or irreversible operation fails.
    """
    _require_bootstrap(profile, namespace, host_uid, namespace_level)
    last_capability = int(Path("/proc/sys/kernel/cap_last_cap").read_text(encoding="ascii"))
    if not _CAP_MAC_ADMIN <= last_capability <= _MAX_CAPABILITY_NUMBER:
        raise ValueError("Unsupported native capability range.")

    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    libc.prctl.restype = ctypes.c_int
    libc.capset.argtypes = [ctypes.POINTER(CapabilityHeader), ctypes.POINTER(CapabilityWord)]
    libc.capset.restype = ctypes.c_int
    with apparmor.open_verified() as descriptor:
        library = ctypes.CDLL(f"/proc/self/fd/{descriptor}", use_errno=True)
        try:
            transition = library.aa_change_profile
        except AttributeError as error:
            raise OSError("Required libapparmor API is unavailable.") from error

        transition.argtypes = [ctypes.c_char_p]
        transition.restype = ctypes.c_int
    for capability in range(last_capability + 1):
        _require_native(
            libc.prctl(_PR_CAPBSET_DROP, capability, 0, 0, 0),
            "Capability bounding drop failed.",
        )

    _require_native(
        libc.prctl(_PR_CAP_AMBIENT, _PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0),
        "Ambient capability clear failed.",
    )
    header = CapabilityHeader(_CAPABILITY_HEADER_VERSION, 0)
    words = (CapabilityWord * 2)()
    _require_native(libc.capset(ctypes.byref(header), words), "Capability clear failed.")
    _require_native(libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "NoNewPrivs failed.")
    revoked = _status()

    if revoked["NoNewPrivs"] != 1 or any(
        value for key, value in revoked.items() if key != "NoNewPrivs"
    ):
        raise ValueError("Bootstrap authority was not revoked.")

    _require_native(
        transition(("&" + profile).encode("ascii")),
        "Payload stacking failed.",
    )
    if (
        Path("/proc/self/attr/current").read_text(encoding="ascii").strip()
        != f"bootstrap//&{profile} (enforce)"
        or _status() != revoked
    ):
        raise ValueError("Unexpected payload confinement.")


def activate_payload(
    apparmor: HostFile,
    bootstrap: HostFile,
    lock: HostFile,
    profile: str,
    compiled_policy: bytes,
    namespace: str,
    host_uid: int,
    *,
    reservation: HostFile,
    namespace_level: int = _DIRECT_NAMESPACE_LEVEL,
    deadline: float | None = None,
    tracking: tuple[Path, tuple[int, int], str] | None = None,
    egress_descriptor: int | None = None,
) -> None:
    """Authenticate loaded setup policy, admit exact bytes and irreversibly enter payload.

    Call only in the single-threaded isolated trusted setup child with no command code running.
    Host preparation must reserve installation-wide capacity and bind the generation reservation,
    installed helper/receipt/unit authority, bootstrap compiled bytes,
    library and lock before UID mapping, and compile the authorized request with compile_payload.
    The surrounding owner must establish reviewed mounts, descriptor filter and protected
    generation admission. This operation does not create those prerequisites or enable Linux.
    It checks the actual exported enforcing bootstrap against installed bytes before and after
    add-only admission. All nonstandard descriptors except one validated attempt-private bridge
    are closed before removing capabilities; standard descriptors must already refer only to
    authorized payload I/O. Any failure must
    terminate the setup child without executing source or offering unrestricted execution.

    Args:
        apparmor (HostFile): Host-bound native transition library.
        bootstrap (HostFile): Host-bound installed compiled bootstrap policy, at most one MiB.
        lock (HostFile): Host-bound read-only namespace lock shared by all loaders.
        profile (str): Simple identity of the compiled authorized payload.
        compiled_policy (bytes): Trusted compile_payload output for exactly this request.
        namespace (str): Authenticated immediate namespace identity.
        host_uid (int): Non-root host caller UID represented as namespace UID zero.
        reservation (HostFile): Host-bound fixed installation-wide generation budget.
            Its namespace binding and bytes are checked before policy admission.
        namespace_level (int): Authenticated immediate namespace depth, one or two.
        deadline (float | None): Absolute monotonic deadline for policy-lock contention; None
            refuses a busy lock immediately.
        tracking (tuple[Path, tuple[int, int], str] | None): Authenticated generation tracking
            mount, original device/inode and current service name for request profile retirement.
        egress_descriptor (int | None): Sole inherited attempt bridge socket retained after
            setup descriptor cleanup; absent for ordinary commands.

    Raises:
        ValueError: Installed/loaded authority, inventory or transition invariants fail.
        TimeoutError: The supplied deadline expires while waiting for policy admission or capacity.
        OSError: Native authentication, admission, descriptor closure or transition fails.
    """
    _require_bootstrap(profile, namespace, host_uid, namespace_level)
    if bootstrap.identity[4] > MAX_COMPILED_BYTES or lock.identity[2] & constants.WRITE_MODE_MASK:
        raise ValueError("Invalid installed bootstrap authority.")

    with bootstrap.open_verified() as descriptor:
        expected = os.pread(descriptor, MAX_COMPILED_BYTES + 1, 0)

    if not expected or len(expected) > MAX_COMPILED_BYTES:
        raise ValueError("Invalid installed bootstrap policy.")

    capacity, byte_limit = generation_limits(reservation, namespace)
    with lock.open_verified():
        load_cached_payload(
            profile,
            compiled_policy,
            namespace,
            host_uid,
            lock.path,
            capacity,
            namespace_level=namespace_level,
            aggregate_limit=byte_limit,
            deadline=deadline,
            bootstrap_policy=expected,
            **({"tracking": tracking} if tracking is not None else {}),
        )
    with lock.open_verified():
        pass

    if egress_descriptor is not None:
        if not isinstance(egress_descriptor, int) or egress_descriptor <= 2:
            raise ValueError("Invalid mediated egress descriptor.")
        channel = socket.socket(fileno=egress_descriptor)
        try:
            if (
                channel.family != socket.AF_UNIX
                or channel.type != socket.SOCK_SEQPACKET
                or channel.getpeername() != ""
            ):
                raise ValueError("Mediated egress bridge is not a private socket pair.")
        finally:
            channel.detach()
    descriptors = os.listdir("/proc/self/fd")
    if len(descriptors) > _MAX_BOOTSTRAP_DESCRIPTORS or any(
        not name.isdecimal() for name in descriptors
    ):
        raise ValueError("Bootstrap descriptor inventory exceeds its bound.")

    for name in descriptors:
        descriptor = int(name)
        if descriptor > 2 and descriptor != egress_descriptor:
            try:
                os.close(descriptor)
            except OSError as error:
                # The directory descriptor used by listdir has already closed.
                if error.errno != errno.EBADF:
                    raise
    enter_payload(apparmor, profile, namespace, host_uid, namespace_level=namespace_level)


def execute_payload(
    request: SandboxRequest,
    apparmor: HostFile,
    bootstrap: HostFile,
    lock: HostFile,
    compiled_policy: bytes,
    profile: str,
    namespace: str,
    host_uid: int,
    *,
    reservation: HostFile,
    namespace_level: int = _DIRECT_NAMESPACE_LEVEL,
    tracking: tuple[Path, tuple[int, int], str] | None = None,
    egress_descriptor: int | None = None,
    local_network_filter: bytes | None = None,
) -> None:
    """Irreversibly activate the bound payload and replace trusted setup with its shell.

    Call only in the isolated, single-threaded trusted bootstrap after enter_bootstrap. Its
    owner must restore and authorize the shared request in the initial host view, compile its
    exact runtime profile and preserve authenticated mounts and standard payload I/O. Require
    current request paths and the original cwd before dropping setup authority; preserve the
    original deadline and caller file-creation mask across activation. Never execute source when
    preparation, activation or deadline checks fail. The receiver must supervise this owned child
    and verify descendant cleanup; this operation alone does not enable the public backend.

    Args:
        request (SandboxRequest): Original authorized shared request, never recaptured grants.
        apparmor (HostFile): Host-authenticated native library authority.
        bootstrap (HostFile): Original compiled enforcing bootstrap authority.
        lock (HostFile): Original immutable shared cache lock.
        compiled_policy (bytes): Strict compilation for this request's derived runtime profile.
        profile (str): Parent-derived identity of the exact canonical policy source.
        namespace (str): Original generation namespace identity.
        host_uid (int): Original ordinary host UID mapped alone to namespace zero.
        reservation (HostFile): Original installation-wide generation capacity authority.
        namespace_level (int): Expected direct or nested policy namespace depth, direct by default.
        tracking (tuple[Path, tuple[int, int], str] | None): Authenticated generation tracking
            mount, original device/inode and current service name for request profile retirement.
        egress_descriptor (int | None): Receiver-owned private socket handed to the scoped
            network-namespace bridge; absent for ordinary commands.
        local_network_filter (bytes | None): Host-prepared filter for a local-network deny.

    Raises:
        ValueError: Bound paths, cwd or activation authority differ.
        TimeoutError: The original request execution deadline expires.
        OSError: Native activation or fixed shell replacement fails.
        RuntimeError: The native exec interface unexpectedly returns without replacement.
    """
    if time.monotonic() >= request.deadline:
        raise TimeoutError("Payload execution deadline expired.")

    if (local_network_filter is None) != request.local_network:
        raise ValueError("Local-network filter and request authority differ.")

    if not request.paths_are_current() or Path.cwd() != request.cwd:
        raise ValueError("Payload execution paths or cwd changed.")

    environment = _payload_environment(request)
    activate_payload(
        apparmor,
        bootstrap,
        lock,
        profile,
        compiled_policy,
        namespace,
        host_uid,
        reservation=reservation,
        namespace_level=namespace_level,
        deadline=request.deadline,
        **({"tracking": tracking} if tracking is not None else {}),
        **({"egress_descriptor": egress_descriptor} if egress_descriptor is not None else {}),
    )

    if time.monotonic() >= request.deadline:
        raise TimeoutError("Payload execution deadline expired.")

    if isinstance(request.egress_origins, tuple) and request.egress_origins:
        if egress_descriptor is None:
            raise ValueError("Mediated egress requires an inherited private channel.")
        proxy_port = start_namespace_proxy(egress_descriptor)
        proxy_url = f"http://127.0.0.1:{proxy_port}"
        environment.update(
            {
                "http_proxy": proxy_url,
                "https_proxy": proxy_url,
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
                "ALL_PROXY": proxy_url,
                "all_proxy": proxy_url,
                "NO_PROXY": "",
                "no_proxy": "",
            }
        )
    elif egress_descriptor is not None:
        raise ValueError("Unexpected mediated egress channel.")

    if local_network_filter is not None:
        install_local_network_filter(local_network_filter)

    os.umask(request.file_creation_mask)
    os.execve(
        constants.SHELL_COMMAND_PATH,
        [
            constants.SHELL_COMMAND_PATH,
            constants.SHELL_COMMAND_OPTION,
            (
                f'{constants.SHELL_PRINT_BUILTIN} "%s\\n" "$1" && exec '
                f"{shlex.quote(constants.SHELL_COMMAND_PATH)} "
                f'{shlex.quote(constants.SHELL_COMMAND_OPTION)} "$2"'
            ),
            "loop-payload",
            NATIVE_PAYLOAD_MARKER_PREFIX + request.attempt_id,
            request.execution_source,
        ],
        environment,
    )
    raise RuntimeError("Payload shell replacement unexpectedly returned.")
