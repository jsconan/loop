"""Provision protected receipts and reserve finite policy capacity across generations.

These private preparation primitives require explicit administrator authorization at their caller.
They never elevate or remove live reservations. Activation publishes only checked admission.
"""

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import time
from collections.abc import Generator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

from ....constants import (
    MAX_NATIVE_INSTALLATION_ENTRIES,
    MAX_NATIVE_POLICY_MODE_BYTES,
    MAX_NATIVE_POLICY_NAME_BYTES,
    MAX_NATIVE_PROFILE_NAME_BYTES,
    MAX_NATIVE_UNIT_BYTES,
    NATIVE_MANAGEMENT_TIMEOUT_SECONDS,
    NATIVE_UNIT_PREFIX,
)
from ....utils.process import ProcessCaptureStatus, supervise_process
from .constants import (
    CGROUP_ROOT,
    SYSTEM_COMMAND_PATH,
    SYSTEM_LOCALE,
    SYSTEMCTL_PATH,
    UNSAFE_HOST_PATH_MODE_BITS,
)
from .host import (
    MAX_COMPILED_BYTES,
    MAX_SOURCE_BYTES,
    HostFile,
    bind_host_file,
    compile_policy_source,
    installation_files,
    protected_path,
    read_policy_bytes,
)
from .mountinfo import MINIMUM_MOUNTINFO_FIELDS, read_descriptor_mount_ids, read_mountinfo

if TYPE_CHECKING:
    from .admission import GenerationAdmission

MAX_PROFILES = 32
MAX_RETAINED_BYTES = 16_777_216
_INSTALLATION_RESERVED_PROFILE_COUNT = 2
_INSTALLATION_RESERVED_POLICY_BYTES = 2 * MAX_COMPILED_BYTES
_GENERATION_BOOTSTRAP_PROFILE_COUNT = 1
_GENERATION_BOOTSTRAP_POLICY_BYTES = MAX_COMPILED_BYTES
_MAX_GENERATION_NAMESPACES = 2
_MAX_MANAGER_STDOUT_BYTES = 8192
_MAX_MANAGER_STDERR_BYTES = 4096
_MAX_CGROUP_EVENTS_BYTES = 4096
_MAX_GLOBAL_POLICY_INVENTORY_ENTRIES = 4096
_MAX_POLICY_ATTACHMENT_BYTES = 4096
_MAX_RESERVATION_RECORD_BYTES = 256


class RetirementState(IntEnum):
    """Name the fixed order of checked native and file removal steps."""

    COMPLETE = 0
    NAMESPACE_REMOVED = 1
    ATTACHMENT_REMOVED = 2
    HELPER_REMOVED = 3
    RECEIVER_REMOVED = 4
    TRACKING_REMOVED = 5
    SLICE_REMOVED = 6
    SOCKET_REMOVED = 7
    TEMPLATE_REMOVED = 8


@dataclass(frozen=True)
class GenerationLifecycle:
    """Own authenticated generation authority for exact checked retirement.

    Args:
        config (GenerationAdmission): Bound fixed admission and reservation configuration.
        files (dict[str, HostFile]): Original receiver and runtime-unit bindings.
        lock (HostFile): Read-only installation-wide management lock.
        compiler_lock (HostFile): Distinct read-only compiler lock.
        artifacts (dict[str, HostFile]): Complete original generation artifact inventory.
        namespace (tuple[int, int]): Original kernel policy namespace device and inode.
        policies (dict[str, dict[str, bytes]]): Exact loaded namespace policy bytes.
        helper (HostFile): Original helper authority, including a known withdrawn binding.
        compiled (bytes): Exact global compiled attachment policy.
        source (bytes): Original literal global attachment source.
    """

    config: "GenerationAdmission"
    files: dict[str, HostFile]
    lock: HostFile
    compiler_lock: HostFile
    artifacts: dict[str, HostFile]
    namespace: tuple[int, int]
    policies: dict[str, dict[str, bytes]]
    helper: HostFile
    compiled: bytes
    source: bytes

    def retire(self, record: HostFile, deadline: float) -> None:
        """Remove original idle authority using observed native state and fixed file bindings.

        Args:
            record (HostFile): Original protected management record.
            deadline (float): Absolute monotonic removal deadline.

        Raises:
            PermissionError: Initial-host administrator authority is absent.
            ValueError: Original authority or native state differs from the fixed owner.
            TimeoutError: The checked removal deadline expires.
            OSError: Live membership, contention or native removal prevents completion.
        """
        self._retire_observed(record, deadline)

    def _retire_observed(self, record: HostFile, deadline: float) -> None:
        """Retry checked removal from the original record and current native state."""
        root = self.config.receiver.parent
        if record.path != root / "management.json":
            raise ValueError("Invalid original management record location.")

        original = {**self.artifacts, "management.json": record}
        order = sorted(
            (name for name in self.artifacts),
            key=lambda name: (
                self.artifacts[name].path in {self.helper.path, self.config.receiver},
                self.artifacts[name].path == self.config.receiver,
                str(self.artifacts[name].path),
            ),
        )
        missing = [name for name in order if not original[name].path.exists()]
        if missing and order[: len(missing)] != missing:
            raise ValueError("Partial generation artifact removal differs from the fixed order.")

        remaining = {name: bound for name, bound in original.items() if bound.path.exists()}
        policy, device = _policy_root()
        namespace = policy / "namespaces" / self.config.generation
        attachment = _attachment_entry(policy, device, NATIVE_UNIT_PREFIX + self.config.generation)
        policies = dict(self.policies)
        if namespace.exists():
            children = set(policies) - {""}
            if any((namespace / "namespaces" / child).exists() for child in children):
                if any(not (namespace / "namespaces" / child).exists() for child in children):
                    raise ValueError("Partial namespace inventory differs from the fixed policy.")
            else:
                policies = {"": policies[""]}
            if missing or attachment is None:
                raise ValueError("Artifact or attachment removed while namespace remains.")
            recovery = RetirementState.COMPLETE
        else:
            if namespace.is_symlink():
                raise ValueError("Generation namespace was replaced with an alias.")
            recovery = RetirementState.NAMESPACE_REMOVED
        helper = self.helper
        if helper.path.exists():
            current = bind_host_file(helper.path)
            if current != helper:
                if (
                    current.identity[:2] != helper.identity[:2]
                    or current.identity[3:6] != helper.identity[3:6]
                    or current.digest != helper.digest
                    or stat.S_IMODE(current.identity[2]) != 0o444
                    or current.identity[6] < helper.identity[6]
                ):
                    raise ValueError("Withdrawn helper differs from original authority.")
                helper = current
                remaining["helper"] = current
        elif "helper" in self.artifacts:
            helper = replace(
                helper,
                identity=(
                    *helper.identity[:2],
                    helper.identity[2] & ~0o777 | 0o444,
                    *helper.identity[3:],
                ),
            )
        if attachment is None:
            if helper.path.exists() and stat.S_IMODE(helper.identity[2]) != 0o444:
                raise ValueError("Attachment absent while helper remains executable.")
            recovery = RetirementState.ATTACHMENT_REMOVED
        elif missing:
            raise ValueError("Artifacts removed while attachment remains.")
        if "helper" not in remaining:
            recovery = RetirementState.HELPER_REMOVED
        if not self.config.receiver.exists():
            if "helper" in remaining:
                raise ValueError("Receiver absent while helper remains.")
            recovery = RetirementState.RECEIVER_REMOVED
        group = CGROUP_ROOT / self.config.slice_unit.path.name
        if not group.exists() and recovery >= RetirementState.RECEIVER_REMOVED:
            recovery = RetirementState.TRACKING_REMOVED
        for phase, bound in (
            (RetirementState.SLICE_REMOVED, self.config.slice_unit),
            (RetirementState.SOCKET_REMOVED, self.files["socket_unit"]),
            (RetirementState.TEMPLATE_REMOVED, self.files["service_unit"]),
        ):
            if not bound.path.exists():
                if recovery != RetirementState(phase - 1):
                    raise ValueError("Unit removal differs from the fixed order.")
                recovery = phase
        if recovery >= RetirementState.TRACKING_REMOVED and set(remaining) != {"management.json"}:
            raise ValueError("Artifacts remain after tracking removal.")
        if recovery >= RetirementState.SLICE_REMOVED:
            _recursive_idle(group, absent=True)
            tool = bind_host_file(SYSTEMCTL_PATH)
            with _administration(self.lock):
                _manager(tool, ["daemon-reload"], deadline)
        if not self.config.reservation.path.exists():
            if recovery is not RetirementState.TEMPLATE_REMOVED:
                raise ValueError("Capacity absent before native retirement completed.")
            with (
                _administration(self.lock),
                record.open_verified(),
                self.compiler_lock.open_verified(),
            ):
                tool = bind_host_file(SYSTEMCTL_PATH)
                prefix = NATIVE_UNIT_PREFIX + self.config.generation
                _recursive_idle(group, absent=True)
                for bound in (
                    self.config.slice_unit,
                    self.files["socket_unit"],
                ):
                    _unit_state(tool, bound, deadline, removed=True)
                _retired_instances(
                    tool, self.config, self.files["service_unit"], deadline, removed=True
                )
                inert = set()
                for line in _manager(
                    tool,
                    ["list-units", "--all", "--plain", "--no-legend", "--", prefix + "*"],
                    deadline,
                ).splitlines():
                    name = line.split()[0] if line.split() else ""
                    if (
                        re.fullmatch(
                            re.escape(prefix) + r"@[A-Za-z0-9_.:\\-]{1,128}\.service", name
                        )
                        is None
                    ):
                        raise ValueError("Unknown retired manager record remains.")
                    inert.add(name)
                _retired_records_absent(
                    tool,
                    prefix,
                    {
                        self.config.slice_unit.path.name,
                        self.files["socket_unit"].path.name,
                        self.files["service_unit"].path.name,
                        *inert,
                    },
                    deadline,
                )
                if (
                    _manager(
                        tool, ["list-jobs", "--no-legend", "--", prefix + "*"], deadline
                    ).strip()
                    or (Path("/run") / (prefix + ".socket")).exists()
                    or (Path("/run") / (prefix + ".socket")).is_symlink()
                    or list(self.config.reservation.path.parent.iterdir())
                ):
                    raise ValueError("Retired native authority remains after capacity release.")
                expected = {
                    record.path,
                    self.lock.path,
                    self.compiler_lock.path,
                    self.config.reservation.path.parent,
                }
                if set(root.rglob("*")) != expected:
                    raise ValueError("Unexpected private files after capacity release.")
                if bind_host_file(record.path) != record:
                    raise ValueError("Original management record changed before removal.")
                record.path.unlink()
            return
        empty_directories = {
            path.relative_to(root).as_posix(): (path.lstat().st_dev, path.lstat().st_ino)
            for path in root.rglob("*")
            if path.is_dir() and not any(path.iterdir())
        }
        retire_generation(
            self.config,
            self.files,
            self.lock,
            self.namespace,
            policies,
            helper,
            self.compiled,
            self.source,
            self.compiler_lock,
            remaining,
            deadline,
            recovery=recovery,
            empty_directories=empty_directories or None,
            cached_payloads=recovery is RetirementState.COMPLETE,
        )


_GENERATION_PROFILES = MAX_PROFILES - _INSTALLATION_RESERVED_PROFILE_COUNT
_GENERATION_BYTES = MAX_RETAINED_BYTES - _INSTALLATION_RESERVED_POLICY_BYTES
_UNIT_DIRECTORY = Path("/run/systemd/system")
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}")


@contextmanager
def _administration(lock: HostFile) -> Generator[None, None, None]:
    """Serialize explicitly authorized initial-host-root installation operations."""
    if os.getuid() != 0 or Path("/proc/self/uid_map").read_text(encoding="ascii").split() != [
        "0",
        "0",
        "4294967295",
    ]:
        raise PermissionError(
            "Installation preparation requires initial host administrator authority."
        )
    if lock.identity[2] & 0o222:
        raise ValueError("Installation lock must be read-only.")
    with lock.open_verified() as descriptor:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def provision_receipt(receipt: Path, files: dict[str, Path], lock: HostFile) -> dict[str, HostFile]:
    """Publish one complete protected receipt without replacing existing authority.

    Bind installed artifacts and use installation_files to validate the full schema before
    publication. The administrator caller supplies fixed reviewed installation paths, never
    client requests. Hold the same installation lock used for retained-policy reservations.
    Sync complete bytes and the publication directory before returning authenticated authority.
    Interrupted publication or failed synchronization retains any published receipt; no artifacts
    are replaced or deleted. Receipt authentication alone establishes no operational readiness.

    Args:
        receipt (Path): New fixed receipt path beneath existing protected installation ancestry.
        files (dict[str, Path]): Exact required artifact roles and installed protected paths.
        lock (HostFile): Persistent installation-wide read-only management lock authority.

    Returns:
        dict[str, HostFile]: Published receipt and fully authenticated installed artifacts.

    Raises:
        PermissionError: Caller lacks initial-host administrator authority.
        ValueError: Receipt location, artifact roles, identities or permissions are unsafe.
        OSError: Contention, an existing receipt or filesystem failure prevents publication.
    """
    if not receipt.is_absolute() or ".." in receipt.parts or not receipt.name:
        raise ValueError("Invalid installation receipt location.")
    with _administration(lock), ExitStack() as pins:
        parent = protected_path(receipt.parent, directory=True)
        if parent != receipt.parent:
            raise ValueError("Receipt directory must use its canonical protected spelling.")
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        pins.callback(os.close, directory)
        record = {
            "schema": 1,
            "files": {
                role: {"path": str(path), "sha256": bind_host_file(path).digest}
                for role, path in files.items()
            },
        }
        descriptor, name = tempfile.mkstemp(prefix=".receipt-", dir=parent)
        staging = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(json.dumps(record, sort_keys=True).encode("ascii"))
                stream.flush()
                os.fchmod(stream.fileno(), 0o444)
                os.fsync(stream.fileno())
            installation_files(staging)
            # link is an atomic no-replace publication of complete bytes; remove our temporary
            # second link before authenticating the receipt's required single-link identity.
            os.link(staging, receipt, follow_symlinks=False)
        finally:
            staging.unlink()
        os.fsync(directory)
        return installation_files(receipt)


def validate_generation_catalog(
    directory: Path,
    contents: dict[str, bytes],
    executables: set[str],
) -> None:
    """Validate bounded immutable file inputs before protected staging.

    Args:
        directory (Path): Intended private installation root.
        contents (dict[str, bytes]): Literal relative file paths and immutable reviewed bytes.
        executables (set[str]): Subset of files requiring executable publication.

    Raises:
        ValueError: Paths, bytes, executable selection or file/directory bounds are invalid.
    """
    if (
        not isinstance(contents, dict)
        or not 0 < len(contents) <= MAX_NATIVE_INSTALLATION_ENTRIES
        or not isinstance(executables, set)
        or not executables <= contents.keys()
        or any(
            not isinstance(name, str)
            or not 0 < len(name) <= 1024
            or re.fullmatch(r"[A-Za-z0-9_./-]+", name) is None
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or Path(name).as_posix() != name
            or name == "."
            or not isinstance(data, bytes)
            for name, data in contents.items()
        )
        or sum(len(data) for data in contents.values()) > MAX_RETAINED_BYTES
    ):
        raise ValueError("Invalid immutable generation file catalog.")
    expected = {directory / name for name in contents}
    parents = {directory} | {
        parent
        for path in expected
        for parent in path.parents
        if parent != directory and parent.is_relative_to(directory)
    }
    if expected & parents or len(expected | parents) > MAX_NATIVE_INSTALLATION_ENTRIES:
        raise ValueError("Conflicting or oversized generation file inventory.")


def stage_generation_files(
    directory: Path,
    contents: dict[str, bytes],
    executables: set[str],
    lock: HostFile,
) -> dict[str, HostFile]:
    """Publish or exactly reuse protected immutable runtime files.

    Args:
        directory (Path): Canonical protected generation directory.
        contents (dict[str, bytes]): Complete reviewed relative path and byte catalog.
        executables (set[str]): Catalog members to publish with executable mode.
        lock (HostFile): Protected installation management lock.

    Returns:
        dict[str, HostFile]: Published file bindings by relative name.

    Raises:
        PermissionError: Initial-host administrator authority is absent.
        ValueError: Input, existing files or ancestry differs from reviewed authority.
        OSError: Contention or publication fails.
    """
    validate_generation_catalog(directory, contents, executables)
    expected = {directory / name for name in contents}
    parents = {directory} | {
        parent
        for path in expected
        for parent in path.parents
        if parent != directory and parent.is_relative_to(directory)
    }
    with _administration(lock), ExitStack() as pins:
        if protected_path(directory, directory=True) != directory:
            raise ValueError("Generation staging directory must use canonical protected spelling.")
        root = directory.lstat()
        rows = read_mountinfo()
        if not rows or any(len(row) < 5 or Path(row[4]).is_relative_to(directory) for row in rows):
            raise ValueError("Mounted or ambiguous generation staging inventory.")
        allowed = expected | parents | {lock.path}
        for count, path in enumerate(directory.rglob("*"), 1):
            if count > len(allowed) or path not in allowed:
                raise ValueError("Unknown generation staging inventory retained.")

        def bind(name):
            """Require exact canonical same-device bytes and the prescribed immutable mode."""
            path = directory / name
            bound = bind_host_file(path)
            if (
                bound.path != path
                or bound.identity[0] != root.st_dev
                or stat.S_IMODE(bound.identity[2]) != (0o555 if name in executables else 0o444)
                or bound.digest != hashlib.sha256(contents[name]).hexdigest()
            ):
                raise ValueError("Generation file differs from fixed immutable authority.")
            return bound

        existing = {}
        for name in contents:
            path = directory / name
            if path.exists() or path.is_symlink():
                existing[name] = bind(name)
                pins.enter_context(existing[name].open_verified())
        for parent in parents:
            if (parent.exists() or parent.is_symlink()) and (
                protected_path(parent, directory=True) != parent
                or parent.lstat().st_dev != root.st_dev
            ):
                raise ValueError("Unsafe generation staging ancestry.")
        descriptors = {}
        identities = {}
        for parent in sorted(parents, key=lambda path: len(path.parts)):
            if not parent.exists():
                parent.mkdir(mode=0o755)
            if (
                protected_path(parent, directory=True) != parent
                or parent.lstat().st_dev != root.st_dev
            ):
                raise ValueError("Unsafe generation staging ancestry.")
            descriptor = os.open(
                parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            )
            pins.callback(os.close, descriptor)
            descriptors[parent] = descriptor
            details = os.fstat(descriptor)
            identities[parent] = details.st_dev, details.st_ino
        result = {}
        for name, data in contents.items():
            path = directory / name
            if name not in existing:
                descriptor, temporary = tempfile.mkstemp(prefix=".loop-file-", dir=path.parent)
                staging = Path(temporary)
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fchmod(stream.fileno(), 0o555 if name in executables else 0o444)
                        os.fsync(stream.fileno())
                    os.link(staging, path, follow_symlinks=False)
                finally:
                    staging.unlink()
            result[name] = bind(name)
        for parent, descriptor in descriptors.items():
            details = parent.lstat()
            if (details.st_dev, details.st_ino) != identities[parent]:
                raise ValueError("Generation staging directory identity changed.")
            os.fsync(descriptor)
        for bound in (lock, *result.values()):
            pins.enter_context(bound.open_verified())
        return result


def generation_limits(
    reservation: HostFile,
    generation: str,
    *,
    deadline: float | None = None,
) -> tuple[int, int]:
    """Authenticate a fixed generation budget, including in a mapped setup child.

    Bind this administrator-owned record in the initial host view before UID mapping. The
    whole installation budget is reserved per generation: 32 profiles and sixteen MiB. Two
    profile slots and two MiB are reserved for the attachment and outer bootstrap; the immediate
    namespace gets 30 profiles including its bootstrap and fourteen MiB of compiled exports.
    Only one retained generation fits; refresh must first
    prove emptiness and explicitly reclaim it. No payload/root API can silently free capacity.

    Args:
        reservation (HostFile): Host-bound immutable generation reservation.
        generation (str): Exact trusted immediate AppArmor namespace identity.
        deadline (float | None): Optional absolute monotonic bound for capacity authentication.

    Returns:
        tuple[int, int]: Payload profile count excluding bootstrap and aggregate byte ceiling.

    Raises:
        ValueError: Record identity, permissions, size or exact bytes are invalid.
        OSError: Bound reservation authority cannot be read or has changed.
        TimeoutError: The supplied capacity authentication budget expires.
    """
    expected = f"{generation}\n{_GENERATION_PROFILES}\n{_GENERATION_BYTES}\n".encode("ascii")
    if (
        _NAME.fullmatch(generation) is None
        or reservation.identity[2] & 0o222
        or reservation.identity[4] != len(expected)
    ):
        raise ValueError("Invalid generation capacity authority.")
    with reservation.open_verified(deadline=deadline) as descriptor:
        if os.pread(descriptor, _MAX_RESERVATION_RECORD_BYTES, 0) != expected:
            raise ValueError("Generation capacity authority differs from bound input.")
    return _GENERATION_PROFILES - _GENERATION_BOOTSTRAP_PROFILE_COUNT, _GENERATION_BYTES


def reserve_generation(directory: Path, lock: HostFile, generation: str) -> HostFile:
    """Reserve installation capacity before generation creation, without parsing requests.

    The fixed root setup supplies only its reviewed namespace identity. One root-protected
    immutable record reserves the full finite generation ceiling, including bootstrap, before
    any profile is loaded. Ordinary preparation then uses generation_limits without elevation.
    Every retained record counts: exact reuse consumes nothing, another generation refuses with
    ENOSPC, and unknown/changed records refuse. Reservations survive setup failure. There is no
    implicit release API; checked reclamation requires authenticated recursive emptiness and
    verified policy/artifact/tracking removal before releasing capacity.
    This conservative reservation does not measure actual retained kernel allocation. Supporting
    simultaneous smaller generations requires separately qualified fractional-budget enforcement.

    Args:
        directory (Path): Existing canonical protected installation-wide reservation directory.
        lock (HostFile): Persistent read-only management lock shared across all generations.
        generation (str): Fixed trusted setup namespace identity, at most 128 ASCII characters.

    Returns:
        HostFile: Durable host-authenticated reservation for mapped bootstrap enforcement.

    Raises:
        PermissionError: Caller lacks initial-host administrator authority.
        ValueError: Identity, directory, inventory or retained capacity authority is invalid.
        OSError: Contention, capacity exhaustion or filesystem failure prevents reservation.
            A partial write remains reserved until checked setup removes it.
    """
    if _NAME.fullmatch(generation) is None:
        raise ValueError("Invalid installation generation identity.")
    with _administration(lock):
        if protected_path(directory, directory=True) != directory:
            raise ValueError("Reservation directory must use its canonical protected spelling.")
        retained = None
        for entry in directory.iterdir():
            if retained is not None or _NAME.fullmatch(entry.name) is None:
                raise ValueError("Invalid retained generation inventory.")
            retained = bind_host_file(entry)
            if retained.path != entry:
                raise ValueError("Retained generation must not alias other authority.")
            generation_limits(retained, entry.name)
        if retained is not None:
            if retained.path.name != generation:
                raise OSError(errno.ENOSPC, "Installation generation capacity exhausted.")
            return retained
        descriptor = os.open(
            directory / generation,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o444,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                f"{generation}\n{_GENERATION_PROFILES}\n{_GENERATION_BYTES}\n".encode("ascii")
            )
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(
            directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        bound = bind_host_file(directory / generation)
        generation_limits(bound, generation)
        return bound


def provision_generation_units(
    config: "GenerationAdmission",
    lock: HostFile,
) -> dict[str, HostFile]:
    """Publish or exactly reuse fixed immutable runtime fragments under the management lock.

    The trusted caller supplies a protected staged slice and installed receiver, after reserving
    this generation's full capacity. Render through the admission owner; client bytes never
    select fragment names or contents. Prevalidate all existing owned fragments before creating
    missing ones. Publication uses atomic no-replace links and syncs the runtime directory.
    A failed publication retains complete fragments and full capacity for explicit exact reuse.
    This operation neither reloads nor activates systemd, loads policy, replaces authority nor
    establishes operational readiness. Refresh must retire the old generation first.

    Args:
        config (GenerationAdmission): Fixed identity, bound immutable staged slice and full
            reservation, with an installed canonical read-only executable receiver.
        lock (HostFile): Persistent installation-wide read-only management lock.

    Returns:
        dict[str, HostFile]: Bound runtime slice_unit, socket_unit, service_unit and receiver.
            The caller uses the returned runtime slice binding for subsequent admission checks.

    Raises:
        PermissionError: Caller lacks initial-host administrator authority.
        ValueError: Inputs, existing fragment bytes, modes, aliases or owned inventory are unsafe.
        OSError: Contention, no-replace publication, authority drift or synchronization fails.
    """
    return _provision_generation_units(config, lock, _administration(lock))


def _provision_generation_units(config: "GenerationAdmission", lock: HostFile, scope):
    """Publish fixed fragments within the caller's continuously held administration scope."""
    sources = config.unit_sources()
    prefix = NATIVE_UNIT_PREFIX + config.generation
    with scope, ExitStack() as pins:
        generation_limits(config.reservation, config.generation)
        receiver = bind_host_file(config.receiver)
        if receiver.path != config.receiver or stat.S_IMODE(receiver.identity[2]) != 0o555:
            raise ValueError("Generation receiver must be canonical and immutable executable.")
        if stat.S_IMODE(config.slice_unit.identity[2]) != 0o444:
            raise ValueError("Staged generation slice must be immutable.")
        staged = pins.enter_context(config.slice_unit.open_verified())
        if os.pread(staged, MAX_NATIVE_UNIT_BYTES + 1, 0) != sources[prefix + ".slice"].encode():
            raise ValueError("Staged generation slice differs from fixed source.")
        for bound in (receiver, config.reservation):
            pins.enter_context(bound.open_verified())
        directory = protected_path(_UNIT_DIRECTORY, directory=True)
        if directory != _UNIT_DIRECTORY:
            raise ValueError("Runtime unit directory must use canonical protected spelling.")
        directory_fd = os.open(
            directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        pins.callback(os.close, directory_fd)
        for path in directory.glob(prefix + "*"):
            if path.name not in sources:
                raise ValueError("Unknown generation runtime fragment retained.")
        existing = {}
        for name, source in sources.items():
            path = directory / name
            if path.exists() or path.is_symlink():
                bound = bind_host_file(path)
                if (
                    bound.path != path
                    or stat.S_IMODE(bound.identity[2]) != 0o444
                    or bound.digest != hashlib.sha256(source.encode()).hexdigest()
                ):
                    raise ValueError("Runtime fragment differs from fixed immutable source.")
                existing[name] = bound
                pins.enter_context(bound.open_verified())
        result = {"receiver": receiver}
        for role, suffix in (
            ("slice_unit", ".slice"),
            ("socket_unit", ".socket"),
            ("service_unit", "@.service"),
        ):
            name = prefix + suffix
            path = directory / name
            if name not in existing:
                descriptor, temporary = tempfile.mkstemp(
                    prefix=".loop-unit-" + config.generation + "-", dir=directory
                )
                staging = Path(temporary)
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(sources[name].encode())
                        stream.flush()
                        os.fchmod(stream.fileno(), 0o444)
                        os.fsync(stream.fileno())
                    os.link(staging, path, follow_symlinks=False)
                finally:
                    staging.unlink()
            os.fsync(directory_fd)
            result[role] = bind_host_file(path)
        for bound in (lock, config.slice_unit, config.reservation, *result.values()):
            with bound.open_verified():
                pass
        return result


def _manager(tool: HostFile, arguments: list[str], deadline: float) -> str:
    """Run fixed management arguments with bounded output and no authentication path."""
    if time.monotonic() >= deadline:
        raise TimeoutError("Generation drain deadline expired.")
    with tool.open_verified():
        process = subprocess.Popen(
            [str(tool.path), "--system", "--no-pager", *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/",
            env={"PATH": SYSTEM_COMMAND_PATH, "LC_ALL": SYSTEM_LOCALE},
            close_fds=True,
            start_new_session=True,
        )
        result = supervise_process(
            process,
            deadline,
            stdout_limit=_MAX_MANAGER_STDOUT_BYTES,
            stderr_limit=_MAX_MANAGER_STDERR_BYTES,
        )
    if (
        result.status is not ProcessCaptureStatus.COMPLETED
        or result.exit_code != 0
        or result.stderr
        or result.stdout_discarded
        or result.stderr_discarded
    ):
        raise OSError("Generation management operation failed; confinement retained.")
    return result.stdout


def _unit_state(
    tool: HostFile,
    unit: HostFile,
    deadline: float,
    *,
    removed: bool = False,
) -> dict[str, str]:
    """Authenticate fixed loaded authority or a known removed slice or socket."""
    wanted = {
        "Id": unit.path.name,
        "LoadState": "loaded",
        "FragmentPath": str(unit.path),
        "DropInPaths": "",
        "NeedDaemonReload": "no",
        "Transient": "no",
    }
    fields = {"ActiveState", "SubState", "ControlGroup"}
    if unit.path.suffix == ".slice":
        wanted["Delegate"] = "no"
    else:
        wanted.update(
            Accept="yes", RemoveOnStop="yes", Listen="/run/" + unit.path.name + " (Stream)"
        )
    if removed:
        parent = protected_path(unit.path.parent, directory=True)
        _removed_artifact(unit, unit.path, parent.lstat().st_dev, 0o444)
        wanted.update(
            FragmentPath="",
            ActiveState="inactive",
            SubState="dead",
            ControlGroup="",
        )
    if removed and unit.path.suffix == ".socket":
        wanted.update(LoadState="not-found", Accept="no", RemoveOnStop="no", Listen="")
    output = _manager(
        tool,
        [
            "show",
            "--all",
            "--property=" + ",".join(sorted(fields | wanted.keys())),
            "--",
            unit.path.name,
        ],
        deadline,
    )
    actual = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in actual:
            raise ValueError("Ambiguous generation unit authority.")
        actual[key] = value
    # systemd omits an empty Listen array even with --all. Permit only that serialization for
    # an explicitly removed socket; not-found authority and all other fields remain mandatory.
    if removed and unit.path.suffix == ".socket" and "Listen" not in actual:
        actual["Listen"] = ""
    if actual.keys() != fields | wanted.keys() or any(
        actual[key] != value for key, value in wanted.items()
    ):
        raise ValueError("Loaded generation authority changed; confinement retained.")
    return actual


def _recursive_idle(group: Path, *, absent: bool = False) -> None:
    """Prove recursive idle membership or explicit absence in the whole-root cgroup2 mount."""
    root = CGROUP_ROOT
    with ExitStack() as pinned:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        pinned.callback(os.close, descriptor)
        ids = read_descriptor_mount_ids(descriptor)
        rows = read_mountinfo()
        visible = [row for row in rows if [row[0]] in ids]
        if (
            len(ids) != 1
            or len(visible) != 1
            or len(visible[0]) < MINIMUM_MOUNTINFO_FIELDS
            or "-" not in visible[0]
            or visible[0][3:5] != ["/", str(root)]
            or visible[0][visible[0].index("-") + 1] != "cgroup2"
            or any(Path(row[4]).is_relative_to(group) for row in rows)
        ):
            raise ValueError("Unverified generation kernel hierarchy; confinement retained.")
        if absent:
            if protected_path(root, directory=True) != root or group.exists() or group.is_symlink():
                raise ValueError("Retired generation tracking remains; capacity retained.")
            return
        if protected_path(group, directory=True) != group:
            raise ValueError("Aliased generation tracking authority.")
        for name in (
            "cgroup.events",
            "cgroup.type",
            "cgroup.procs",
            "cgroup.threads",
            "cgroup.subtree_control",
        ):
            path = group / name
            details = path.lstat()
            if not stat.S_ISREG(details.st_mode) or details.st_uid != 0 or details.st_mode & 0o022:
                raise ValueError("Delegated generation tracking authority.")
        if (group / "cgroup.type").read_text(encoding="ascii") != "domain\n":
            raise ValueError("Generation tracking is not a process domain.")
        fd = os.open(group / "cgroup.events", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        pinned.callback(os.close, fd)
        data = os.read(fd, _MAX_CGROUP_EVENTS_BYTES + 1)
        fields = data.splitlines()
        if (
            len(data) > _MAX_CGROUP_EVENTS_BYTES
            or fields.count(b"populated 0") != 1
            or any(line.startswith(b"populated") and line != b"populated 0" for line in fields)
        ):
            raise OSError(errno.EBUSY, "Generation is live or ambiguous; confinement retained.")


def _removed_artifact(bound: HostFile, path: Path, device: int, mode: int) -> None:
    """Check trusted pre-deletion authority and current absence of one fixed artifact."""
    if (
        bound.path != path
        or len(bound.identity) != 7
        or bound.identity[0] != device
        or not stat.S_ISREG(bound.identity[2])
        or stat.S_IMODE(bound.identity[2]) != mode
        or bound.identity[3] != 1
        or re.fullmatch(r"[0-9a-f]{64}", bound.digest) is None
        or path.exists()
        or path.is_symlink()
    ):
        raise ValueError(f"Invalid known removed {path.name} authority; capacity retained.")


@contextmanager
def drained_generation(
    config: "GenerationAdmission",
    files: dict[str, HostFile],
    lock: HostFile,
    deadline: float,
    *,
    removed_receiver: HostFile | None = None,
    recovery: RetirementState = RetirementState.COMPLETE,
) -> Generator[None, None, None]:
    """Close authenticated admission and hold the installation lock over an idle generation.

    Args:
        config (GenerationAdmission): Fixed trusted generation admission configuration.
        files (dict[str, HostFile]): Bound socket_unit, service_unit and receiver artifacts.
        lock (HostFile): Persistent authenticated installation-wide readonly management lock.
        recovery (RetirementState): Trusted completed retirement prefix; defaults to COMPLETE.
        deadline (float): Absolute monotonic deadline for admission closure and idle proof.
        removed_receiver (HostFile | None): Trusted fixed 0555 receiver binding authenticated before
            known deletion; None (default) requires an intact receiver. Must equal
            files["receiver"].

    Yields:
        None: Exclusive closed-admission idle scope, valid only while the context is held.

    Raises:
        PermissionError: The caller lacks initial-host administrator authority.
        ValueError: Fixed or loaded authority, kernel tracking or native input is unsafe.
        TimeoutError: The drain budget expires before a management operation.
        OSError: Contention, native failure or live/ambiguous membership prevents drain.
    """
    with _drained_generation(
        config, files, lock, deadline, removed_receiver=removed_receiver, recovery=recovery
    ):
        yield


@contextmanager
def _drained_generation(
    config: "GenerationAdmission",
    files: dict[str, HostFile],
    lock: HostFile,
    deadline: float,
    *,
    removed_receiver: HostFile | None = None,
    recovery: RetirementState = RetirementState.COMPLETE,
    scope=None,
) -> Generator[None, None, None]:
    """Close authenticated admission and hold the installation lock over an idle generation.

    Args:
        config (GenerationAdmission): Fixed trusted generation admission configuration.
        files (dict[str, HostFile]): Bound socket_unit, service_unit and receiver artifacts.
        lock (HostFile): Persistent authenticated installation-wide readonly management lock.
        recovery (RetirementState): Trusted completed retirement prefix; defaults to COMPLETE.
        deadline (float): Absolute monotonic deadline for admission closure and idle proof.
        removed_receiver (HostFile | None): Trusted fixed 0555 receiver binding authenticated before
            known deletion; None (default) requires an intact receiver. Must equal
            files["receiver"].

    Yields:
        None: Exclusive closed-admission idle scope, valid only while the context is held.

    Raises:
        PermissionError: The caller lacks initial-host administrator authority.
        ValueError: Fixed or loaded authority, kernel tracking or native input is unsafe.
        TimeoutError: The drain budget expires before a management operation.
        OSError: Contention, native failure or live/ambiguous membership prevents drain.
    """
    if not isinstance(recovery, RetirementState):
        # Keep malformed lifecycle phase inputs under the established ValueError contract.
        raise ValueError("Invalid explicit retirement state.")  # noqa: TRY004
    tracking_removed = recovery >= RetirementState.TRACKING_REMOVED
    slice_removed = recovery >= RetirementState.SLICE_REMOVED
    socket_removed = recovery >= RetirementState.SOCKET_REMOVED
    template_removed = recovery >= RetirementState.TEMPLATE_REMOVED

    deadline = min(deadline, time.monotonic() + NATIVE_MANAGEMENT_TIMEOUT_SECONDS)
    if slice_removed and (removed_receiver is None):
        raise ValueError(
            "Known removed slice requires explicit stopped tracking and receiver binding."
        )
    sources = config.unit_sources()
    if template_removed and (
        files.get("service_unit") is None
        or files["service_unit"].path.name != NATIVE_UNIT_PREFIX + config.generation + "@.service"
        or files["service_unit"].digest
        != hashlib.sha256(sources[files["service_unit"].path.name].encode()).hexdigest()
    ):
        raise ValueError("Known removed template differs from fixed source.")
    if socket_removed and (
        files.get("socket_unit") is None
        or files["socket_unit"].path.name != NATIVE_UNIT_PREFIX + config.generation + ".socket"
        or files["socket_unit"].digest
        != hashlib.sha256(sources[files["socket_unit"].path.name].encode()).hexdigest()
    ):
        raise ValueError("Known removed socket differs from fixed source.")
    if slice_removed and (
        config.slice_unit.digest
        != hashlib.sha256(sources[config.slice_unit.path.name].encode()).hexdigest()
    ):
        raise ValueError("Known removed slice differs from fixed source.")
    tool = bind_host_file(SYSTEMCTL_PATH)
    with _administration(lock) if scope is None else scope, ExitStack() as inputs:
        generation_limits(config.reservation, config.generation)
        if removed_receiver is not None:
            if files.get("receiver") != removed_receiver:
                raise ValueError("Removed receiver differs from fixed drain authority.")
            root = protected_path(config.receiver.parent, directory=True)
            _removed_artifact(removed_receiver, config.receiver, root.lstat().st_dev, 0o555)
        roles = {"slice_unit": ".slice", "socket_unit": ".socket", "service_unit": "@.service"}
        for role, bound in {"slice_unit": config.slice_unit, **files}.items():
            if (
                role == "receiver"
                and removed_receiver is not None
                or role == "slice_unit"
                and slice_removed
                or role == "socket_unit"
                and socket_removed
                or role == "service_unit"
                and template_removed
            ):
                continue
            fd = inputs.enter_context(bound.open_verified())
            if role in roles and (
                bound.path.name != NATIVE_UNIT_PREFIX + config.generation + roles[role]
                or os.pread(fd, MAX_NATIVE_UNIT_BYTES + 1, 0) != sources[bound.path.name].encode()
            ):
                raise ValueError("Generation fragment differs from fixed authority.")
        if (
            set(files) != {"socket_unit", "service_unit", "receiver"}
            or files["receiver"].path != config.receiver
        ):
            raise ValueError("Incomplete generation management authority.")
        socket = files["socket_unit"]
        _unit_state(tool, socket, deadline, removed=socket_removed)
        _unit_state(tool, config.slice_unit, deadline, removed=slice_removed)
        if tracking_removed:
            if removed_receiver is None:
                raise ValueError("Retired tracking requires known removed receiver authority.")
        else:
            _manager(tool, ["stop", "--", socket.path.name], deadline)
        unit_state = _unit_state(tool, socket, deadline, removed=socket_removed)
        if (
            unit_state["ActiveState"] != "inactive"
            or unit_state["SubState"] != "dead"
            or unit_state["ControlGroup"]
        ):
            raise ValueError("Generation listener closure is unverified.")
        listener = Path("/run") / socket.path.with_suffix(".socket").name
        if listener.exists() or listener.is_symlink():
            raise ValueError("Generation listener remains after closure.")
        unit_state = _unit_state(tool, config.slice_unit, deadline, removed=slice_removed)
        group = CGROUP_ROOT / config.slice_unit.path.name
        if tracking_removed:
            if (
                unit_state["ControlGroup"]
                or unit_state["ActiveState"] != "inactive"
                or unit_state["SubState"] != "dead"
            ):
                raise ValueError("Retired generation tracking state is unsafe; capacity retained.")
        elif (
            unit_state["ControlGroup"] != "/" + group.name or unit_state["ActiveState"] != "active"
        ):
            raise ValueError("Generation tracking is missing or inactive; confinement retained.")
        if _manager(
            tool,
            ["list-jobs", "--no-legend", "--", NATIVE_UNIT_PREFIX + config.generation + "*"],
            deadline,
        ).strip():
            raise ValueError("Pending generation activation; confinement retained.")
        _recursive_idle(group, absent=tracking_removed)
        for bound in (lock, config.reservation, config.slice_unit, tool, *files.values()):
            if bound != removed_receiver and not (
                slice_removed
                and bound == config.slice_unit
                or socket_removed
                and bound == socket
                or template_removed
                and bound == files["service_unit"]
            ):
                inputs.enter_context(bound.open_verified())
        if removed_receiver is not None:
            _removed_artifact(removed_receiver, config.receiver, root.lstat().st_dev, 0o555)
        if slice_removed:
            _unit_state(tool, config.slice_unit, deadline, removed=True)
        if socket_removed:
            _unit_state(tool, socket, deadline, removed=True)
        if template_removed:
            _retired_instances(tool, config, files["service_unit"], deadline, removed=True)
        if time.monotonic() >= deadline:
            raise TimeoutError("Generation drain deadline expired.")
        yield


def _policy_directory(path: Path, root: Path, device: int) -> None:
    """Check non-aliased kernel directories below the authenticated policy magic link."""
    relative = path.relative_to(root)
    for depth in range(1, len(relative.parts) + 1):
        directory = root.joinpath(*relative.parts[:depth])
        details = directory.lstat()
        if (
            not stat.S_ISDIR(details.st_mode)
            or details.st_uid != 0
            or details.st_mode & UNSAFE_HOST_PATH_MODE_BITS
            or details.st_dev != device
        ):
            raise ValueError("Unsafe kernel policy directory authority.")


def _policy_root() -> tuple[Path, int]:
    """Authenticate initial-host securityfs and its protected kernel policy jump link."""
    security = Path("/sys/kernel/security")
    rows = read_mountinfo()
    mounts = [row for row in rows if Path(row[4]).is_relative_to(security)]
    if (
        len(mounts) != 1
        or mounts[0][3:5] != ["/", str(security)]
        or "-" not in mounts[0]
        or mounts[0][mounts[0].index("-") + 1] != "securityfs"
        or read_policy_bytes(security / "apparmor/.ns_level", 16).strip() != b"0"
    ):
        raise ValueError("Unverified initial-host policy hierarchy.")
    apparmor = security / "apparmor"
    if protected_path(apparmor, directory=True) != apparmor:
        raise ValueError("Aliased initial-host policy authority.")
    policy = apparmor / "policy"
    link = policy.lstat()
    details = policy.stat()
    # securityfs supplies a kernel jump link, not a pathname for Path.resolve().
    # Its readlink spelling describes the securityfs link inode, not the target inode.
    if (
        not stat.S_ISLNK(link.st_mode)
        or link.st_uid != 0
        or str(policy.readlink()) != f"apparmorfs:[{link.st_ino}]"
        or not stat.S_ISDIR(details.st_mode)
        or details.st_uid != 0
        or details.st_mode & UNSAFE_HOST_PATH_MODE_BITS
    ):
        raise ValueError("Unverified kernel policy magic link.")
    return policy, details.st_dev


def _policy_inventory(policies: dict[str, dict[str, bytes]]) -> None:
    """Validate bounded flat or one-child policy authority before kernel mutation."""
    if (
        not isinstance(policies, dict)
        or "" not in policies
        or len(policies) > _MAX_GENERATION_NAMESPACES
        or any(
            not isinstance(name, str) or (name and _NAME.fullmatch(name) is None)
            for name in policies
        )
        or any(
            not isinstance(profiles, dict)
            or "bootstrap" not in profiles
            or any(
                not isinstance(name, str)
                or _NAME.fullmatch(name) is None
                or not isinstance(data, bytes)
                or not 0 < len(data) <= MAX_COMPILED_BYTES
                for name, data in profiles.items()
            )
            for profiles in policies.values()
        )
        or sum(len(profiles) for profiles in policies.values())
        > MAX_PROFILES - _GENERATION_BOOTSTRAP_PROFILE_COUNT
        or sum(len(data) for profiles in policies.values() for data in profiles.values())
        > MAX_RETAINED_BYTES - _GENERATION_BOOTSTRAP_POLICY_BYTES
    ):
        raise ValueError("Invalid fixed generation policy authority.")


def _policy_tree(
    generation: str,
    identity: tuple[int, int],
    policies: dict[str, dict[str, bytes]],
) -> list[Path]:
    """Authenticate the exact flat or one-child owned AppArmor namespace tree."""
    _policy_inventory(policies)
    policy, device = _policy_root()
    namespace = policy / "namespaces" / generation
    _policy_directory(namespace, policy, device)
    details = namespace.lstat()
    if (details.st_dev, details.st_ino) != identity:
        raise ValueError("Generation namespace identity changed.")
    _policy_directory(namespace / "namespaces", policy, device)
    children = {path.name for path in (namespace / "namespaces").iterdir()}
    if children != set(policies) - {""}:
        raise ValueError("Unexpected generation namespace inventory.")
    directories = []
    for name, expected in policies.items():
        directory = namespace / "namespaces" / name if name else namespace
        _policy_directory(directory / "namespaces", policy, device)
        if name and next((directory / "namespaces").iterdir(), None) is not None:
            raise ValueError("Unexpected nested generation namespace.")
        _policy_directory(directory / "profiles", policy, device)
        loaded = {}
        for entry in (directory / "profiles").iterdir():
            _policy_directory(entry, policy, device)
            if len(loaded) >= len(expected):
                raise ValueError("Unexpected generation policy inventory.")
            profile = (
                read_policy_bytes(entry / "name", MAX_NATIVE_PROFILE_NAME_BYTES)
                .decode("ascii")
                .strip()
            )
            if (
                profile in loaded
                or profile not in expected
                or read_policy_bytes(entry / "mode", MAX_NATIVE_POLICY_MODE_BYTES).strip()
                != b"enforce"
                or read_policy_bytes(entry / "raw_data", MAX_COMPILED_BYTES) != expected[profile]
                or (
                    (entry / "profiles").exists()
                    and next((entry / "profiles").iterdir(), None) is not None
                )
            ):
                raise ValueError("Generation policy differs from fixed authority.")
            loaded[profile] = entry
        if loaded.keys() != expected.keys():
            raise ValueError("Incomplete generation policy inventory.")
        directories.append(directory)
    return sorted(directories, key=lambda path: len(path.parts), reverse=True)


def remove_generation_policy(
    config: "GenerationAdmission",
    files: dict[str, HostFile],
    lock: HostFile,
    namespace_identity: tuple[int, int],
    policies: dict[str, dict[str, bytes]],
    deadline: float,
) -> None:
    """Remove exactly authenticated idle namespaces while retaining generation capacity.

    Close admission and hold its qualified drain scope throughout validation and removal.
    Fixed administrator authority supplies the namespace inode and exact compiled profile bytes,
    never client/model data. Authenticate securityfs, the initial AppArmor namespace and the
    kernel policy jump link without treating its displayed target as a disk pathname. Reject
    aliases, writable ownership or changed devices below that link. Authenticate the complete
    enforcing inventory, including at most one immediate child namespace. Remove child
    before parent, verifying namespace absence after each removal. Retain the full reservation:
    global attachments and other generation artifacts still require checked removal before
    capacity can be reclaimed. A partial failure retains remaining policy and capacity. After a
    child-only removal, trusted management may explicitly supply the exact remaining outer-only
    enforcing inventory with the same bound outer inode. Stale full inventory, changed identity,
    compiled bytes or unexpected children refuse; missing namespace never implies success. This
    primitive does not remove attachments, helpers, units, cache files or shared directories;
    those require subsequent checked transaction integration. It never kills members, restores
    admission or elevates.

    Args:
        config (GenerationAdmission): Fixed trusted admission and capacity authority.
        files (dict[str, HostFile]): Bound socket/service/receiver artifacts for drain.
        lock (HostFile): Persistent installation-wide protected management lock.
        namespace_identity (tuple[int, int]): Administrator-bound namespace device and inode.
        policies (dict[str, dict[str, bytes]]): Exact profile bytes per relative namespace;
            empty string identifies the outer namespace, an optional simple name its child.
            Each namespace includes bootstrap; total namespace profiles/bytes leave the reserved
            global attachment slot and one MiB unused.
        deadline (float): Absolute monotonic deadline, capped to five seconds for the transaction.

    Raises:
        PermissionError: Initial-host administrator authority is absent.
        ValueError: Drain, hierarchy, identity or exact policy/reservation authority is unsafe.
        TimeoutError: The transaction budget expires before a removal step.
        OSError: Live membership, contention, reads, namespace removal or namespace-absence
            verification fails.
    """
    deadline = min(deadline, time.monotonic() + NATIVE_MANAGEMENT_TIMEOUT_SECONDS)
    _remove_generation_policy(
        config,
        files,
        lock,
        namespace_identity,
        policies,
        deadline,
        drained_generation(config, files, lock, deadline),
    )


def generation_policy_cache(
    config: "GenerationAdmission",
    namespace_identity: tuple[int, int],
    policies: dict[str, dict[str, bytes]],
    helper: HostFile,
    compiled: bytes,
) -> dict[str, dict[str, bytes]]:
    """Authenticate installed bootstrap bytes and the bounded add-only payload cache.

    Reserved payload names identify managed cache entries, not new execution authority. Each
    launch still compares its compiled bytes with the kernel entry. Unknown names, nested,
    non-enforcing, empty or oversized exports refuse. Removal callers must already hold closed
    admission and prove recursive drain; independently unconfined profile users are outside
    Loop's lifetime guarantee. Revalidate the complete observed tree and global attachment.

    Args:
        config (GenerationAdmission): Original generation and capacity authority.
        namespace_identity (tuple[int, int]): Original protected namespace device and inode.
        policies (dict[str, dict[str, bytes]]): Immutable installation bootstrap inventory.
        helper (HostFile): Original installed helper.
        compiled (bytes): Original global helper attachment export.

    Returns:
        dict[str, dict[str, bytes]]: Complete authenticated kernel inventory at this check.

    Raises:
        ValueError: Identity, enforcing cache, bootstrap, attachment or budgets differ.
        OSError: Protected native exports or original capacity cannot be inspected.
    """
    _policy_inventory(policies)
    capacity, byte_limit = generation_limits(config.reservation, config.generation)
    policy, device = _policy_root()
    directory = policy / "namespaces" / config.generation
    _policy_directory(directory / "profiles", policy, device)
    observed = {name: dict(profiles) for name, profiles in policies.items()}
    root = {}
    for entry in (directory / "profiles").iterdir():
        _policy_directory(entry, policy, device)
        name = (
            read_policy_bytes(entry / "name", MAX_NATIVE_PROFILE_NAME_BYTES).decode("ascii").strip()
        )
        data = read_policy_bytes(entry / "raw_data", MAX_COMPILED_BYTES)
        if (
            len(root) >= capacity + 1
            or name in root
            or not data
            or (name not in policies[""] and re.fullmatch(r"payload_[a-f0-9]{64}", name) is None)
            or read_policy_bytes(entry / "mode", MAX_NATIVE_POLICY_MODE_BYTES).strip() != b"enforce"
        ):
            raise ValueError("Invalid managed payload cache inventory.")
        root[name] = data
    if any(root.get(name) != data for name, data in policies[""].items()) or (
        sum(map(len, root.values())) > byte_limit
    ):
        raise ValueError("Managed payload cache bootstrap or byte authority differs.")
    observed[""] = root
    _policy_tree(config.generation, namespace_identity, observed)
    _attest_attachment(policy, device, NATIVE_UNIT_PREFIX + config.generation, helper, compiled)
    return observed


def _remove_generation_policy(
    config: "GenerationAdmission",
    files: dict[str, HostFile],
    lock: HostFile,
    namespace_identity: tuple[int, int],
    policies: dict[str, dict[str, bytes]],
    deadline: float,
    scope,
) -> None:
    """Apply checked removal within the caller-owned exclusive drain scope."""
    with scope:
        reservation = config.reservation
        if (
            reservation.path.name != config.generation
            or protected_path(reservation.path.parent, directory=True) != reservation.path.parent
        ):
            raise ValueError("Invalid generation reclamation authority.")
        directories = _policy_tree(config.generation, namespace_identity, policies)
        _recursive_idle(CGROUP_ROOT / config.slice_unit.path.name)
        for directory in directories:
            if time.monotonic() >= deadline:
                raise TimeoutError("Generation removal deadline expired; capacity retained.")
            directory.rmdir()
            if directory.exists() or directory.is_symlink():
                raise ValueError("Generation namespace remains; capacity retained.")
        generation_limits(reservation, config.generation)
        if time.monotonic() >= deadline:
            raise TimeoutError("Generation removal deadline expired; capacity retained.")


def _attachment_entry(policy: Path, device: int, name: str) -> Path | None:
    """Find one exact global profile within a bounded authenticated kernel inventory."""
    profiles = policy / "profiles"
    _policy_directory(profiles, policy, device)
    selected = None
    for index, entry in enumerate(profiles.iterdir()):
        if index >= _MAX_GLOBAL_POLICY_INVENTORY_ENTRIES:
            raise ValueError("Global policy inventory exceeds its bound.")
        _policy_directory(entry, policy, device)
        if (
            read_policy_bytes(entry / "name", MAX_NATIVE_POLICY_NAME_BYTES).decode("utf-8").strip()
            == name
        ):
            if selected is not None:
                raise ValueError("Ambiguous generation attachment inventory.")
            selected = entry
    return selected


def _attachment_authority(config, lock, helper, compiled, source, compiler_lock):
    """Validate the fixed literal attachment and distinct immutable compiler authority."""
    name = NATIVE_UNIT_PREFIX + config.generation
    prefix = f'abi <abi/4.0>,\nprofile {name} "{helper.path}"'.encode()
    headers = (prefix + b" {", prefix + b" flags=(attach_disconnected) {")
    if (
        helper.path != config.receiver.parent / "helper"
        or re.fullmatch(r"/[A-Za-z0-9_./-]+", str(helper.path)) is None
        or not isinstance(source, bytes)
        or not 0 < len(source) <= MAX_SOURCE_BYTES
        or not source.startswith(headers)
        or compiler_lock.identity[2] & 0o222
        or compiler_lock.identity[:2] == lock.identity[:2]
        or not isinstance(compiled, bytes)
        or not 0 < len(compiled) <= MAX_COMPILED_BYTES
    ):
        raise ValueError("Invalid fixed generation attachment authority.")
    return name


def _attest_attachment(policy, device, name, helper, compiled):
    """Require the exact enforcing global profile and literal or compiled-proven attachment."""
    entry = _attachment_entry(policy, device, name)
    if (
        entry is None
        or read_policy_bytes(entry / "mode", MAX_NATIVE_POLICY_MODE_BYTES).strip() != b"enforce"
        or read_policy_bytes(entry / "attach", _MAX_POLICY_ATTACHMENT_BYTES).decode("utf-8").strip()
        not in (str(helper.path), "<unknown>")
        or read_policy_bytes(entry / "raw_data", MAX_COMPILED_BYTES) != compiled
        or (
            (entry / "profiles").exists() and next((entry / "profiles").iterdir(), None) is not None
        )
    ):
        raise ValueError("Generation attachment differs from fixed authority.")


def remove_generation_attachment(
    config: "GenerationAdmission",
    files: dict[str, HostFile],
    lock: HostFile,
    helper: HostFile,
    compiled: bytes,
    source: bytes,
    compiler_lock: HostFile,
    deadline: float,
) -> None:
    """Withdraw helper execution and remove its exact idle global attachment, retaining capacity.

    Require the generation namespace already absent in the authenticated initial-host policy
    hierarchy. Under the qualified drain lock, authenticate the single enforcing global profile
    named loopgen_<generation>, its literal helper attachment and exact compiled bytes. Strictly
    enumerate exactly that profile and recompile its fixed literal source under the distinct
    installation-wide payload compiler lock, requiring byte equality
    with both administrator authority and the loaded export. An opaque kernel attachment display
    is accepted only with this compilation proof; other display drift is refused. The helper
    must be the private single-link protected file named helper beside the trusted receiver.
    Change its mode to 0444 and sync that withdrawal before unloading, denying new ordinary
    execution even through a previously opened descriptor. Recheck recursive emptiness
    and policy authority after this withdrawal. Write only the exact fixed profile name
    to the authenticated kernel removal interface, then prove that profile absent. Retain the
    helper, units, cache and full reservation
    for subsequent checked cleanup; this does not reclaim capacity or complete uninstall.
    Failure after withdrawal leaves the helper non-executable and admission closed. Rebinding
    withdrawn files or retrying partial removal requires trusted management; absence is never
    inferred to mean a prior successful removal.

    Args:
        config (GenerationAdmission): Fixed trusted generation/admission authority.
        files (dict[str, HostFile]): Bound socket/service/receiver artifacts for drain.
        lock (HostFile): Persistent protected installation management lock.
        helper (HostFile): Exact private helper inode/bytes beside the bound receiver.
        compiled (bytes): Fixed administrator-authorized compiled attachment, at most one MiB.
        source (bytes): Fixed administrator source with the exact literal header, at most 64 KiB.
        compiler_lock (HostFile): Distinct installation-wide read-only payload compiler lock.
        deadline (float): Absolute monotonic deadline, capped to five seconds for the transaction.

    Raises:
        PermissionError: Initial-host administrator authority is absent.
        ValueError: Helper, namespace, kernel profile or reservation authority is unsafe or changed.
        TimeoutError: The transaction budget expires before removal or completion.
        OSError: Contention, live membership, kernel access, withdrawal or removal fails.
    """
    deadline = min(deadline, time.monotonic() + NATIVE_MANAGEMENT_TIMEOUT_SECONDS)
    _remove_generation_attachment(
        config,
        files,
        lock,
        helper,
        compiled,
        source,
        compiler_lock,
        deadline,
        drained_generation(config, files, lock, deadline),
    )


def _remove_generation_attachment(
    config: "GenerationAdmission",
    files: dict[str, HostFile],
    lock: HostFile,
    helper: HostFile,
    compiled: bytes,
    source: bytes,
    compiler_lock: HostFile,
    deadline: float,
    scope,
    *,
    tracking_absent: bool = False,
) -> None:
    """Apply checked removal within the caller-owned exclusive drain scope."""
    name = _attachment_authority(config, lock, helper, compiled, source, compiler_lock)
    with scope, helper.open_verified() as descriptor:
        if bind_host_file(helper.path) != helper:
            raise ValueError("Generation helper differs from fixed authority.")
        policy, device = _policy_root()
        _policy_directory(policy / "namespaces", policy, device)
        namespace = policy / "namespaces" / config.generation
        if namespace.exists() or namespace.is_symlink():
            raise ValueError("Generation namespace remains; attachment retained.")
        with ExitStack() as compiler_inputs:
            compiler_descriptor = compiler_inputs.enter_context(compiler_lock.open_verified())
            fcntl.flock(compiler_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (
                compile_policy_source(source, deadline, names=True) != name.encode("ascii") + b"\n"
                or compile_policy_source(source, deadline) != compiled
            ):
                raise ValueError("Generation attachment compilation differs from fixed authority.")
            compiler_inputs.enter_context(compiler_lock.open_verified())
        for withdrawn in (False, True):
            _attest_attachment(policy, device, name, helper, compiled)
            if not withdrawn:

                def withdraw():
                    """Withdraw and synchronize only the originally pinned helper inode."""
                    os.fchmod(descriptor, 0o444)
                    os.fsync(descriptor)

                withdraw()
                rebound = bind_host_file(helper.path)
                if (
                    rebound.identity[:2] != helper.identity[:2]
                    or rebound.digest != helper.digest
                    or stat.S_IMODE(rebound.identity[2]) != 0o444
                ):
                    raise ValueError("Generation helper changed during withdrawal.")
                _recursive_idle(
                    CGROUP_ROOT / config.slice_unit.path.name,
                    absent=tracking_absent,
                )
        if time.monotonic() >= deadline:
            raise TimeoutError("Attachment removal deadline expired; capacity retained.")
        command = name.encode("ascii") + b"\0"
        entry = _attachment_entry(policy, device, name)
        if entry is None:
            raise ValueError("Generation attachment changed before removal; capacity retained.")

        def remove():
            """Write exactly the attested fixed global profile removal command."""
            with (policy.parent / ".remove").open("wb", buffering=0) as output:
                if output.write(command) != len(command):
                    raise OSError("Incomplete generation attachment removal; capacity retained.")

        remove()
        if _attachment_entry(policy, device, name) is not None:
            raise ValueError("Generation attachment remains; capacity retained.")
        generation_limits(config.reservation, config.generation)
        if time.monotonic() >= deadline:
            raise TimeoutError("Attachment removal completion expired; capacity retained.")


def load_generation_policy(
    config: "GenerationAdmission",
    lock: HostFile,
    helper: HostFile,
    sources: dict[str, dict[str, bytes]],
    source: bytes,
    compiler_lock: HostFile,
    deadline: float,
) -> tuple[tuple[int, int], dict[str, dict[str, bytes]], bytes]:
    """Load only a fresh protected generation from bounded trusted single-profile sources.

    Hold installation management and distinct compiler locks throughout compilation, exclusive
    namespace creation, add-only kernel loading and exact enforcing export attestation. Compile
    all sources and reject existing namespace/global attachment authority before loading anything.
    The caller must persist reviewed source and returned compiled authority for later management.
    Failure retains any partial policy and the full reservation; it never opens admission, replaces
    policy or infers reuse. Existing/partial generations safely refuse.

    Args:
        config (GenerationAdmission): Trusted staged receiver/slice and full reserved generation.
        lock (HostFile): Authenticated immutable installation management lock.
        helper (HostFile): Bound private immutable executable helper beside the receiver.
        sources (dict[str, dict[str, bytes]]): Flat or one-child namespace inventory, each with
            bootstrap and trusted sources of at most 64 KiB for each named profile.
        source (bytes): Literal single-profile global helper attachment source, at most 64 KiB.
        compiler_lock (HostFile): Distinct immutable installation-wide compiler lock.
        deadline (float): Absolute monotonic preparation deadline, capped to five seconds.

    Returns:
        tuple[tuple[int, int], dict[str, dict[str, bytes]], bytes]:
            Newly bound namespace device/inode, complete compiled namespace exports and
            compiled global helper attachment.

    Raises:
        PermissionError: Initial-host administrator authority is absent.
        ValueError: Inputs, inventory, kernel endpoint or loaded enforcing authority are unsafe.
        OSError: Contention, compilation, existing namespace or kernel load prevents completion.
        TimeoutError: Bounded preparation expires; partial authority and full capacity remain.
    """
    config.unit_sources()
    _policy_inventory(sources)
    if any(
        len(data) > MAX_SOURCE_BYTES for profiles in sources.values() for data in profiles.values()
    ):
        raise ValueError("Generation policy source exceeds the preparation bound.")
    name = _attachment_authority(
        config, lock, helper, b"pending compilation", source, compiler_lock
    )
    if stat.S_IMODE(helper.identity[2]) != 0o555:
        raise ValueError("Policy loading requires fixed immutable generation authority.")
    deadline = min(deadline, time.monotonic() + NATIVE_MANAGEMENT_TIMEOUT_SECONDS)
    with _administration(lock), ExitStack() as pins:
        generation_limits(config.reservation, config.generation)
        for bound in (helper, config.slice_unit, bind_host_file(config.receiver), compiler_lock):
            pins.enter_context(bound.open_verified())
        compiler = pins.enter_context(compiler_lock.open_verified())
        fcntl.flock(compiler, fcntl.LOCK_EX | fcntl.LOCK_NB)
        policy, device = _policy_root()
        parent = policy / "namespaces"
        _policy_directory(parent, policy, device)
        namespace = parent / config.generation
        if namespace.exists() or namespace.is_symlink() or _attachment_entry(policy, device, name):
            raise ValueError(
                "Generation policy already exists; retained authority requires management."
            )
        compiled = {}
        for relative, profiles in sources.items():
            compiled[relative] = {}
            for profile, data in profiles.items():
                if compile_policy_source(data, deadline, names=True) != profile.encode() + b"\n":
                    raise ValueError("Generation source does not define exactly its fixed profile.")
                compiled[relative][profile] = compile_policy_source(data, deadline)
        if compile_policy_source(source, deadline, names=True) != name.encode() + b"\n":
            raise ValueError("Attachment source does not define exactly its fixed profile.")
        attachment = compile_policy_source(source, deadline)
        _policy_inventory(compiled)
        _attachment_authority(config, lock, helper, attachment, source, compiler_lock)
        if time.monotonic() >= deadline:
            raise TimeoutError("Generation policy loading deadline expired.")
        namespace.mkdir()
        _policy_directory(namespace, policy, device)
        details = namespace.lstat()
        identity = details.st_dev, details.st_ino
        loads = []
        for relative, profiles in compiled.items():
            directory = namespace / "namespaces" / relative if relative else namespace
            if relative:
                directory.mkdir()
            _policy_directory(directory, policy, device)
            loads.extend((directory / ".load", data) for data in profiles.values())
        loads.append((policy / ".load", attachment))
        for path, data in loads:
            _policy_directory(path.parent, policy, device)
            descriptor = os.open(path, os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                info = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != 0
                    or info.st_mode & UNSAFE_HOST_PATH_MODE_BITS
                    or info.st_dev != device
                ):
                    raise ValueError("Unsafe generation kernel load endpoint.")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Generation policy loading deadline expired.")
                if os.write(descriptor, data) != len(data):
                    raise OSError("Incomplete generation kernel load; authority retained.")
            finally:
                os.close(descriptor)
        _policy_tree(config.generation, identity, compiled)
        _attest_attachment(policy, device, name, helper, attachment)
        for bound in (helper, config.slice_unit, config.reservation, compiler_lock):
            pins.enter_context(bound.open_verified())
        if time.monotonic() >= deadline:
            raise TimeoutError("Generation policy loading completion expired.")
        return identity, compiled, attachment


def activate_generation(
    config: "GenerationAdmission",
    lock: HostFile,
    helper: HostFile,
    namespace_identity: tuple[int, int],
    policies: dict[str, dict[str, bytes]],
    compiled: bytes,
    source: bytes,
    compiler_lock: HostFile,
    deadline: float,
) -> tuple["GenerationAdmission", dict[str, HostFile]]:
    """Publish fixed admission and activate only an authenticated idle installed generation.

    The explicitly authorized root installer supplies already loaded, reviewed policy authority.
    Hold management ownership continuously across exact publication, reload, listener closure,
    kernel idle proof and activation. Recompile the literal global attachment under the distinct
    compiler lock and attest all loaded namespace/global bytes before opening admission. Reuse
    preserves capacity; a live generation refuses after closing its listener without signalling
    members. The listener manager cgroup must be the fixed empty socket group, or its absence must
    be proven against the kernel view. No policy creation, old-generation retirement, elevation
    or Linux enablement occurs.
    An activation failure attempts only fixed listener closure with a separate bounded cleanup
    budget; cleanup failure is reported together with the original error, never hidden.

    Args:
        config (GenerationAdmission): Fixed staged slice, installed receiver and full reservation.
        lock (HostFile): Persistent immutable installation management lock.
        helper (HostFile): Bound private executable helper beside the receiver.
        namespace_identity (tuple[int, int]): Trusted loaded namespace device and inode.
        policies (dict[str, dict[str, bytes]]): Complete trusted enforcing namespace tree exports.
        compiled (bytes): Exact reviewed global attachment export, at most one MiB.
        source (bytes): Fixed literal attachment source, at most 64 KiB.
        compiler_lock (HostFile): Distinct immutable installation-wide compiler lock.
        deadline (float): Absolute activation deadline, capped to five seconds; failed activation
            gets at most five additional seconds solely to close its fixed listener.

    Returns:
        tuple[GenerationAdmission, dict[str, HostFile]]: Runtime slice configuration and bound
            socket_unit, service_unit and receiver for ordinary admission. This is not an execution
            grant or full operational/bootstrap readiness result.

    Raises:
        PermissionError: Caller lacks initial-host root authority.
        ValueError: Installed policy, source, units, listener or kernel authority is invalid.
        TimeoutError: Activation budget expires.
        OSError: Contention, live membership, publication or manager operations fail.
        BaseExceptionGroup: Both activation and bounded listener closure fail.
    """
    deadline = min(deadline, time.monotonic() + NATIVE_MANAGEMENT_TIMEOUT_SECONDS)
    name = _attachment_authority(config, lock, helper, compiled, source, compiler_lock)
    with _administration(lock), ExitStack() as pins:
        if stat.S_IMODE(helper.identity[2]) != 0o555:
            raise ValueError("Activation requires an immutable executable helper.")
        pins.enter_context(helper.open_verified())
        compiler = pins.enter_context(compiler_lock.open_verified())
        fcntl.flock(compiler, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (
            compile_policy_source(source, deadline, names=True) != name.encode() + b"\n"
            or compile_policy_source(source, deadline) != compiled
        ):
            raise ValueError("Generation attachment compilation differs from fixed authority.")
        _policy_tree(config.generation, namespace_identity, policies)
        policy, device = _policy_root()
        _attest_attachment(policy, device, name, helper, compiled)
        published = _provision_generation_units(config, lock, nullcontext())
        runtime = replace(config, slice_unit=published.pop("slice_unit"))
        tool = bind_host_file(SYSTEMCTL_PATH)
        pins.enter_context(tool.open_verified())
        _manager(tool, ["daemon-reload"], deadline)
        socket = published["socket_unit"]
        _unit_state(tool, socket, deadline)
        _manager(tool, ["stop", "--", socket.path.name], deadline)
        listener = Path("/run") / socket.path.name
        unit_state = _unit_state(tool, socket, deadline)
        if (
            unit_state["ActiveState"] != "inactive"
            or unit_state["SubState"] != "dead"
            or unit_state["ControlGroup"]
            or listener.exists()
            or listener.is_symlink()
        ):
            raise ValueError("Generation listener closure is unverified.")
        if _manager(tool, ["cat", "--", published["service_unit"].path.name], deadline) != (
            "# "
            + str(published["service_unit"].path)
            + "\n"
            + runtime.unit_sources()[name + "@.service"]
        ):
            raise ValueError("Loaded service template differs from fixed authority.")
        _retired_instances(tool, runtime, published["service_unit"], deadline)
        if _manager(tool, ["list-jobs", "--no-legend", "--", name + "*"], deadline).strip():
            raise ValueError("Pending generation activation; confinement retained.")
        unit_state = _unit_state(tool, runtime.slice_unit, deadline)
        group = CGROUP_ROOT / runtime.slice_unit.path.name
        if unit_state["ActiveState"] == "inactive" and not unit_state["ControlGroup"]:
            _recursive_idle(group, absent=True)
            _manager(tool, ["start", "--", runtime.slice_unit.path.name], deadline)
        unit_state = _unit_state(tool, runtime.slice_unit, deadline)
        if unit_state["ActiveState"] != "active" or unit_state["ControlGroup"] != "/" + group.name:
            raise ValueError("Generation tracking is missing or inactive; confinement retained.")
        _recursive_idle(group)
        _policy_tree(config.generation, namespace_identity, policies)
        _attest_attachment(policy, device, name, helper, compiled)
        for bound in (
            lock,
            compiler_lock,
            helper,
            runtime.reservation,
            runtime.slice_unit,
            tool,
            *published.values(),
        ):
            pins.enter_context(bound.open_verified())
        try:
            _manager(tool, ["start", "--", socket.path.name], deadline)
            unit_state = _unit_state(tool, socket, deadline)
            details = listener.lstat()
            if (
                unit_state["ActiveState"] != "active"
                or unit_state["SubState"] != "listening"
                or not stat.S_ISSOCK(details.st_mode)
                or stat.S_IMODE(details.st_mode) != 0o600
                or (details.st_uid, details.st_gid) != (runtime.uid, runtime.gid)
            ):
                raise ValueError("Activated generation listener differs from fixed authority.")
            socket_group = (CGROUP_ROOT / "system.slice") / socket.path.name
            if unit_state["ControlGroup"] not in ("", "/system.slice/" + socket.path.name):
                raise ValueError(
                    "Activated generation listener tracking differs from fixed authority."
                )
            _recursive_idle(socket_group, absent=not unit_state["ControlGroup"])
            for bound in (
                lock,
                compiler_lock,
                helper,
                runtime.reservation,
                runtime.slice_unit,
                tool,
                *published.values(),
            ):
                pins.enter_context(bound.open_verified())
            if time.monotonic() >= deadline:
                raise TimeoutError("Generation activation deadline expired.")
        except BaseException as error:
            try:
                cleanup_deadline = time.monotonic() + NATIVE_MANAGEMENT_TIMEOUT_SECONDS
                _manager(tool, ["stop", "--", socket.path.name], cleanup_deadline)
                closed = _unit_state(tool, socket, cleanup_deadline)
                if (
                    closed["ActiveState"] != "inactive"
                    or closed["SubState"] != "dead"
                    or closed["ControlGroup"]
                    or listener.exists()
                    or listener.is_symlink()
                ):
                    raise ValueError("Failed activation listener remains; confinement retained.")
            except (OSError, ValueError) as cleanup:
                raise BaseExceptionGroup(
                    "Activation and listener closure failed.", [error, cleanup]
                ) from None
            raise
        return runtime, published


def _retired_properties(
    tool: HostFile,
    name: str,
    fields: set[str],
    deadline: float,
) -> dict[str, str]:
    """Read unambiguous bounded properties for one already selected retired unit."""
    output = _manager(
        tool, ["show", "--property=" + ",".join(sorted(fields)), "--", name], deadline
    )
    actual = {}
    for entry in output.splitlines():
        key, separator, value = entry.partition("=")
        if not separator or key in actual:
            raise ValueError("Ambiguous retired unit authority; capacity retained.")
        actual[key] = value
    return actual


def _retired_instances(
    tool: HostFile,
    config: "GenerationAdmission",
    service: HostFile,
    deadline: float,
    *,
    removed: bool = False,
) -> list[str]:
    """Authenticate bounded inactive instances, or require none after known template deletion."""
    prefix = NATIVE_UNIT_PREFIX + config.generation
    lines = _manager(
        tool,
        ["list-units", "--all", "--plain", "--no-legend", "--", prefix + "@*.service"],
        deadline,
    ).splitlines()
    if removed:
        parent = protected_path(service.path.parent, directory=True)
        _removed_artifact(service, service.path, parent.lstat().st_dev, 0o444)
        if lines:
            raise ValueError(
                "Removed service template still has instance records; capacity retained."
            )
        return []
    names = []
    for line in lines:
        fields = line.split()
        if (
            len(names) >= 32
            or not fields
            or re.fullmatch(re.escape(prefix) + r"@[A-Za-z0-9_.:\\-]{1,128}\.service", fields[0])
            is None
            or fields[0] in names
        ):
            raise ValueError("Ambiguous retired generation instances; capacity retained.")
        name = fields[0]
        wanted = {
            "Id": name,
            "LoadState": "loaded",
            "FragmentPath": str(service.path),
            "DropInPaths": "",
            "NeedDaemonReload": "no",
            "Transient": "no",
            "MainPID": "0",
            "ControlPID": "0",
            "ControlGroup": "",
            "Slice": prefix + ".slice",
            "Delegate": "no",
            "ExitType": "cgroup",
        }
        actual = _retired_properties(tool, name, wanted.keys() | {"ActiveState"}, deadline)
        if (
            actual.keys() != wanted.keys() | {"ActiveState"}
            or any(actual[key] != value for key, value in wanted.items())
            or actual["ActiveState"] not in {"inactive", "failed"}
        ):
            raise ValueError("Retired service authority differs; capacity retained.")
        names.append(name)
    return names


def _retired_records_absent(tool: HostFile, prefix: str, known: set[str], deadline: float) -> None:
    """Allow only known inert not-found records after authenticated fragment removal/reload."""
    lines = _manager(
        tool, ["list-units", "--all", "--plain", "--no-legend", "--", prefix + "*"], deadline
    ).splitlines()
    seen = set()
    for line in lines:
        fields = line.split()
        if not fields or fields[0] not in known or fields[0] in seen:
            raise ValueError("Retired generation authority remains; capacity retained.")
        name = fields[0]
        seen.add(name)
        wanted = {
            "Id": name,
            "LoadState": "not-found",
            "FragmentPath": "",
            "DropInPaths": "",
            "NeedDaemonReload": "no",
            "Transient": "no",
            "ActiveState": "inactive",
            "ControlGroup": "",
        }
        if name.endswith(".service"):
            wanted.update(MainPID="0", ControlPID="0")
        actual = _retired_properties(tool, name, set(wanted), deadline)
        if actual != wanted:
            raise ValueError("Retired generation authority remains; capacity retained.")


def reclaim_generation(
    config: "GenerationAdmission",
    files: dict[str, HostFile],
    lock: HostFile,
    compiler_lock: HostFile,
    artifacts: dict[str, HostFile],
    deadline: float,
    *,
    empty_directories: dict[str, tuple[int, int]] | None = None,
    removed_helper: HostFile | None = None,
    removed_receiver: HostFile | None = None,
    recovery: RetirementState = RetirementState.COMPLETE,
) -> None:
    """Remove exact retired artifacts, then release capacity and the retained record.

    Args:
        config (GenerationAdmission): Fixed trusted admission/generation authority.
        files (dict[str, HostFile]): Bound socket/service/receiver drain artifacts.
        lock (HostFile): Persistent read-only installation management lock.
        compiler_lock (HostFile): Distinct read-only installation-wide payload compiler lock.
        artifacts (dict[str, HostFile]): Complete bound generation artifact inventory by relative
            path.
        recovery (RetirementState): Trusted completed retirement prefix; defaults to COMPLETE.
        deadline (float): Absolute monotonic budget capped to five seconds.
        empty_directories (dict[str, tuple[int, int]] | None): Trusted relative empty cache
            directory
            names bound to device/inode; None (default) allows only file-derived directories.
        removed_helper (HostFile | None): Trusted withdrawn helper binding authenticated before
            known deletion; None (default) requires an intact bound helper in artifacts. Never
            client input or a record reconstructed after loss.
        removed_receiver (HostFile | None): Trusted fixed 0555 receiver binding authenticated before
            known deletion; requires removed_helper and equals files["receiver"]. None (default)
            requires an intact receiver; never inferred from absence or supplied by a client.

    Raises:
        PermissionError: Initial-host administrator authority is absent.
        ValueError: Inventory, policy, helper, tracking, unit or reservation authority is unsafe.
        TimeoutError: The transaction budget expires before final release.
        OSError: Contention, live membership, deletion, manager or release synchronization fails.
    """
    if not isinstance(recovery, RetirementState):
        # Keep malformed lifecycle phase inputs under the established ValueError contract.
        raise ValueError("Invalid explicit retirement state.")  # noqa: TRY004

    deadline = min(deadline, time.monotonic() + NATIVE_MANAGEMENT_TIMEOUT_SECONDS)
    _reclaim_generation(
        config,
        files,
        lock,
        compiler_lock,
        artifacts,
        deadline,
        drained_generation(
            config,
            files,
            lock,
            deadline,
            removed_receiver=removed_receiver,
            recovery=recovery,
        ),
        empty_directories,
        removed_helper,
        removed_receiver,
        recovery,
    )


def _reclaim_generation(
    config: "GenerationAdmission",
    files: dict[str, HostFile],
    lock: HostFile,
    compiler_lock: HostFile,
    artifacts: dict[str, HostFile],
    deadline: float,
    scope,
    empty_directories: dict[str, tuple[int, int]] | None,
    removed_helper: HostFile | None,
    removed_receiver: HostFile | None,
    recovery: RetirementState,
) -> None:
    """Apply checked removal within the caller-owned exclusive drain scope."""
    tracking_removed = recovery >= RetirementState.TRACKING_REMOVED
    slice_removed = recovery >= RetirementState.SLICE_REMOVED
    socket_removed = recovery >= RetirementState.SOCKET_REMOVED
    template_removed = recovery >= RetirementState.TEMPLATE_REMOVED

    root = config.receiver.parent
    retained_record = artifacts.get("management.json")
    if retained_record is not None and retained_record.path != root / "management.json":
        raise ValueError("Invalid retained management record authority.")
    if (
        set(files) != {"socket_unit", "service_unit", "receiver"}
        or len(artifacts) > MAX_NATIVE_INSTALLATION_ENTRIES
        or tracking_removed
        and (removed_receiver is None or set(artifacts) - {"management.json"} or empty_directories)
        or (not artifacts and removed_receiver is None)
        or len(empty_directories or {}) > MAX_NATIVE_INSTALLATION_ENTRIES
        or compiler_lock.identity[2] & 0o222
        or compiler_lock.identity[:2] == lock.identity[:2]
        or config.reservation.path.name != config.generation
    ):
        raise ValueError("Invalid fixed generation cleanup authority.")
    tool = bind_host_file(SYSTEMCTL_PATH)
    units = (config.slice_unit, files["socket_unit"], files["service_unit"])
    preserved = (
        lock,
        compiler_lock,
        config.reservation,
        tool,
        *(
            u
            for u in units
            if not (
                slice_removed
                and u == config.slice_unit
                or socket_removed
                and u == files["socket_unit"]
                or template_removed
                and u == files["service_unit"]
            )
        ),
    )
    with scope, ExitStack() as pinned:
        compiler_fd = pinned.enter_context(compiler_lock.open_verified())
        fcntl.flock(compiler_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        policy, device = _policy_root()
        _policy_directory(policy / "namespaces", policy, device)
        namespace = policy / "namespaces" / config.generation
        prefix = NATIVE_UNIT_PREFIX + config.generation
        if (
            namespace.exists()
            or namespace.is_symlink()
            or _attachment_entry(policy, device, prefix) is not None
        ):
            raise ValueError("Generation policy remains; capacity retained.")
        if protected_path(root, directory=True) != root:
            raise ValueError("Aliased generation artifact directory.")
        root_device = root.lstat().st_dev
        helper = root / "helper"
        if removed_helper is not None:
            _removed_artifact(removed_helper, helper, root_device, 0o444)
            if "helper" in artifacts:
                raise ValueError("Invalid known removed helper authority; capacity retained.")
        if removed_receiver is not None:
            if removed_helper is None or files["receiver"] != removed_receiver:
                raise ValueError(
                    "Removed receiver requires known removed helper and fixed binding."
                )
            _removed_artifact(removed_receiver, config.receiver, root_device, 0o555)
        expected = {bound.path for bound in preserved if bound.path.is_relative_to(root)}
        targets = []
        for name, bound in artifacts.items():
            path = root / name
            if (
                not name
                or Path(name).is_absolute()
                or Path(name).as_posix() != name
                or ".." in Path(name).parts
                or path == root
                or bound.path != path
                or bound.identity[2] & 0o222
                or bound.path in {item.path for item in preserved}
            ):
                raise ValueError("Invalid generation artifact authority.")
            pinned.enter_context(bound.open_verified())
            if bind_host_file(path) != bound:
                raise ValueError("Generation artifact changed; capacity retained.")
            expected.add(path)
            if name != "management.json":
                targets.append(bound)
        required = {helper, config.receiver}
        if removed_helper is not None:
            required.remove(helper)
        if removed_receiver is not None:
            required.remove(config.receiver)
        if required - {bound.path for bound in targets} or (
            removed_helper is None and stat.S_IMODE(helper.lstat().st_mode) != 0o444
        ):
            raise ValueError("Generation helper is not withdrawn; capacity retained.")
        directories = set()
        for path in expected:
            directories.update(
                parent for parent in path.parents if parent != root and parent.is_relative_to(root)
            )
        for name, identity in (empty_directories or {}).items():
            path = root / name
            if (
                not name
                or Path(name).is_absolute()
                or Path(name).as_posix() != name
                or ".." in Path(name).parts
                or path == root
                or protected_path(path, directory=True) != path
            ):
                raise ValueError("Invalid empty generation directory authority.")
            details = path.lstat()
            if (details.st_dev, details.st_ino) != identity or next(
                path.iterdir(), None
            ) is not None:
                raise ValueError("Empty generation directory changed or is not empty.")
            directories.add(path)
            directories.update(
                parent for parent in path.parents if parent != root and parent.is_relative_to(root)
            )
        observed_files = set()
        observed_directories = set()
        directory_identities = {}
        for index, path in enumerate(root.rglob("*")):
            if index >= MAX_NATIVE_INSTALLATION_ENTRIES:
                raise ValueError("Generation artifact inventory exceeds its bound.")
            details = path.lstat()
            if stat.S_ISDIR(details.st_mode):
                if protected_path(path, directory=True) != path or details.st_dev != root_device:
                    raise ValueError("Unsafe generation artifact directory.")
                descriptor = os.open(
                    path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
                )
                pinned.callback(os.close, descriptor)
                pinned_details = os.fstat(descriptor)
                identity = details.st_dev, details.st_ino
                if (pinned_details.st_dev, pinned_details.st_ino) != identity:
                    raise ValueError("Generation artifact directory changed during pinning.")
                directory_identities[path] = identity
                observed_directories.add(path)
            elif stat.S_ISREG(details.st_mode):
                observed_files.add(path)
            else:
                raise ValueError("Unknown generation artifact type; capacity retained.")
        rows = read_mountinfo()
        if (
            observed_files != expected
            or observed_directories != directories
            or any(len(row) < 5 or Path(row[4]).is_relative_to(root) for row in rows)
            or list(config.reservation.path.parent.iterdir()) != [config.reservation.path]
        ):
            raise ValueError("Generation artifact inventory differs; capacity retained.")
        instances = _retired_instances(
            tool, config, files["service_unit"], deadline, removed=template_removed
        )
        for authority in preserved:
            pinned.enter_context(authority.open_verified())
        retained_directories = {
            parent
            for bound in preserved
            for parent in bound.path.parents
            if parent != root and parent.is_relative_to(root)
        }
        ordered = sorted(
            targets,
            key=lambda bound: (
                bound.path in (helper, config.receiver),
                bound.path == config.receiver,
                str(bound.path),
            ),
        )
        for bound in [*ordered, *([None] if removed_receiver is not None else [])]:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Generation artifact removal deadline expired; capacity retained."
                )
            if bound is not None and bind_host_file(bound.path) != bound:
                raise ValueError("Generation artifact changed before removal; capacity retained.")
            if bound is None or bound.path == (
                config.receiver if removed_helper is not None else helper
            ):
                for directory in sorted(
                    directories - retained_directories, key=lambda p: len(p.parts), reverse=True
                ):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Generation directory removal deadline expired; capacity retained."
                        )
                    details = directory.lstat()
                    if (
                        protected_path(directory, directory=True) != directory
                        or (details.st_dev, details.st_ino) != directory_identities[directory]
                    ):
                        raise ValueError("Generation artifact directory changed before removal.")
                    directory.rmdir()
            if bound is not None:
                bound.path.unlink()
        group = CGROUP_ROOT / config.slice_unit.path.name
        if tracking_removed:
            _recursive_idle(group, absent=True)
        else:
            _recursive_idle(group)
            _manager(tool, ["stop", "--", config.slice_unit.path.name], deadline)
        slice_state = _unit_state(tool, config.slice_unit, deadline, removed=slice_removed)
        if (
            slice_state["ActiveState"] != "inactive"
            or slice_state["SubState"] != "dead"
            or slice_state["ControlGroup"]
            or group.exists()
            or group.is_symlink()
        ):
            raise ValueError("Generation tracking remains; capacity retained.")
        if socket_removed:
            _unit_state(tool, files["socket_unit"], deadline, removed=True)
        if template_removed:
            _retired_instances(tool, config, files["service_unit"], deadline, removed=True)
        for bound in units:
            if (
                slice_removed
                and bound == config.slice_unit
                or socket_removed
                and bound == files["socket_unit"]
                or template_removed
                and bound == files["service_unit"]
            ):
                continue
            if bind_host_file(bound.path) != bound:
                raise ValueError("Generation fragment changed before removal; capacity retained.")
            bound.path.unlink()
        _manager(tool, ["daemon-reload"], deadline)
        if instances:
            _manager(tool, ["reset-failed", "--", *instances], deadline)
        _retired_records_absent(
            tool, prefix, {bound.path.name for bound in units} | set(instances), deadline
        )
        if (
            _manager(tool, ["list-jobs", "--no-legend", "--", prefix + "*"], deadline).strip()
            or (Path("/run") / (prefix + ".socket")).exists()
            or any(bound.path.exists() or bound.path.is_symlink() for bound in (*targets, *units))
            or helper.exists()
            or helper.is_symlink()
            or config.receiver.exists()
            or config.receiver.is_symlink()
            or namespace.exists()
            or _attachment_entry(policy, device, prefix) is not None
        ):
            raise ValueError("Retired generation authority remains; capacity retained.")
        if tracking_removed:
            _recursive_idle(group, absent=True)
        generation_limits(config.reservation, config.generation)
        pinned.enter_context(lock.open_verified())
        pinned.enter_context(compiler_lock.open_verified())
        if time.monotonic() >= deadline:
            raise TimeoutError("Generation reclamation deadline expired; capacity retained.")
        reservation_fd = pinned.enter_context(config.reservation.open_verified())
        record = os.pread(reservation_fd, _MAX_RESERVATION_RECORD_BYTES, 0)
        directory_fd = os.open(
            config.reservation.path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        pinned.callback(os.close, directory_fd)

        def release():
            """Release capacity after native reclamation and compensate on sync failure."""
            config.reservation.path.unlink()
            try:
                if config.reservation.path.exists() or config.reservation.path.is_symlink():
                    raise ValueError("Generation reservation remains after release.")
                os.fsync(directory_fd)
            except (OSError, ValueError):
                if (
                    not config.reservation.path.exists()
                    and not config.reservation.path.is_symlink()
                ):
                    fd = os.open(
                        config.reservation.path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                        0o444,
                    )
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(record)
                        stream.flush()
                        os.fchmod(stream.fileno(), 0o444)
                        os.fsync(stream.fileno())
                raise

        release()
        if retained_record is not None:
            # Keep the original manifest until native cleanup and capacity release complete.
            if bind_host_file(retained_record.path) != retained_record:
                raise ValueError("Management record changed before removal.")
            retained_record.path.unlink()


def retire_generation(
    config: "GenerationAdmission",
    files: dict[str, HostFile],
    lock: HostFile,
    namespace_identity: tuple[int, int],
    policies: dict[str, dict[str, bytes]],
    helper: HostFile,
    compiled: bytes,
    source: bytes,
    compiler_lock: HostFile,
    artifacts: dict[str, HostFile],
    deadline: float,
    *,
    recovery: RetirementState = RetirementState.COMPLETE,
    empty_directories: dict[str, tuple[int, int]] | None = None,
    cached_payloads: bool = False,
) -> None:
    """Retire one fixed generation under uninterrupted authenticated management authority.

    Args:
        config (GenerationAdmission): Fixed generation admission and reservation authority.
        files (dict[str, HostFile]): Bound generation socket/service/receiver drain artifacts.
        lock (HostFile): Persistent authenticated read-only installation management lock.
        namespace_identity (tuple[int, int]): Bound initial generation namespace device/inode.
        policies (dict[str, dict[str, bytes]]): Exact compiled enforcing namespace inventory.
        helper (HostFile): Protected private helper authority matching the supplied inventory;
            original executable or bound withdrawn input for explicit continuation; helper_removed
            requires the trusted 0444 binding authenticated before its known deletion.
        compiled (bytes): Exact fixed compiled global attachment policy.
        source (bytes): Exact literal source proving the global attachment binding.
        compiler_lock (HostFile): Distinct read-only installation-wide compiler lock.
        artifacts (dict[str, HostFile]): Complete trusted relative generation artifact inventory.
        deadline (float): Absolute monotonic budget capped to five seconds.
        recovery (RetirementState): Trusted completed removal prefix; defaults to COMPLETE.
        empty_directories (dict[str, tuple[int, int]] | None): Trusted relative empty cache
            directory
            names bound to device/inode; None (default) allows only file-derived directories.
        cached_payloads (bool): Authenticate managed add-only payload entries after closing and
            draining admission; False preserves exact fixed-inventory retirement by default.

    Raises:
        PermissionError: Initial-host administrator authority is absent.
        ValueError: Explicit recovery state, fixed/loaded authority, inventory or withdrawal proof
            is unsafe.
        TimeoutError: The shared transaction deadline expires.
        OSError: Contention, live membership, partial removal or native operations fail.
    """
    deadline = min(deadline, time.monotonic() + NATIVE_MANAGEMENT_TIMEOUT_SECONDS)
    if not isinstance(recovery, RetirementState):
        # Keep malformed lifecycle phase inputs under the established ValueError contract.
        raise ValueError("Invalid explicit retirement state.")  # noqa: TRY004
    if recovery < RetirementState.HELPER_REMOVED and artifacts.get("helper") != helper:
        raise ValueError("Retirement inventory lacks original helper authority.")
    removed_receiver = (
        files.get("receiver") if recovery >= RetirementState.RECEIVER_REMOVED else None
    )
    if recovery >= RetirementState.RECEIVER_REMOVED and removed_receiver is None:
        raise ValueError("Retirement lacks pre-deletion receiver binding.")
    with drained_generation(
        config,
        files,
        lock,
        deadline,
        removed_receiver=removed_receiver,
        recovery=recovery,
    ):
        if recovery is RetirementState.COMPLETE and cached_payloads:
            policies = generation_policy_cache(
                config, namespace_identity, policies, helper, compiled
            )
        if recovery is RetirementState.COMPLETE:
            _remove_generation_policy(
                config, files, lock, namespace_identity, policies, deadline, nullcontext()
            )
        if recovery < RetirementState.ATTACHMENT_REMOVED:
            _remove_generation_attachment(
                config,
                files,
                lock,
                helper,
                compiled,
                source,
                compiler_lock,
                deadline,
                nullcontext(),
            )
        if recovery >= RetirementState.HELPER_REMOVED:
            withdrawn = helper
        else:
            withdrawn = bind_host_file(helper.path)
            if (
                withdrawn.identity[:2] != helper.identity[:2]
                or withdrawn.digest != helper.digest
                or stat.S_IMODE(withdrawn.identity[2]) != 0o444
            ):
                raise ValueError("Retirement helper changed after withdrawal; capacity retained.")
        _reclaim_generation(
            config,
            files,
            lock,
            compiler_lock,
            artifacts
            if recovery >= RetirementState.HELPER_REMOVED
            else {**artifacts, "helper": withdrawn},
            deadline,
            nullcontext(),
            empty_directories,
            withdrawn if recovery >= RetirementState.HELPER_REMOVED else None,
            removed_receiver,
            recovery,
        )
