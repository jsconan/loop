"""Own protected Linux installation records, lifecycle checks and native management."""

import fcntl
import json
import os
import re
import stat
import sys
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path

from ....constants import (
    MAX_NATIVE_INSTALLATION_ENTRIES,
    MAX_NATIVE_POLICY_MODE_BYTES,
    MAX_NATIVE_POLICY_NAME_BYTES,
    MAX_NATIVE_PROFILE_NAME_BYTES,
    MAX_NATIVE_RECEIPT_BYTES,
    MAX_NATIVE_UNIT_BYTES,
    NATIVE_INSTALLATION_DIRECTORY,
    NATIVE_MANAGEMENT_TIMEOUT_SECONDS,
    NATIVE_UNIT_PREFIX,
)
from ....utils.json import load_unique_json
from .admission import GenerationAdmission
from .constants import CGROUP_ROOT, SYSTEMCTL_PATH, UNSAFE_HOST_PATH_MODE_BITS
from .host import (
    MAX_COMPILED_BYTES,
    NATIVE_LOCK_SEEDS,
    HostFile,
    apparmor_library,
    bind_host_file,
    compile_policy_source,
    discover_tool,
    protected_path,
    read_policy_bytes,
)
from .installation import (
    _UNIT_DIRECTORY,
    _administration,
    _attachment_entry,
    _attest_attachment,
    _manager,
    _policy_directory,
    _policy_inventory,
    _policy_root,
    _recursive_idle,
    _remove_generation_attachment,
    _retired_properties,
    _retired_records_absent,
    activate_generation,
    drained_generation,
    generation_limits,
    load_generation_policy,
    provision_generation_units,
    provision_receipt,
    stage_generation_files,
    validate_generation_catalog,
)
from .mountinfo import read_mountinfo
from .records import (
    MANAGEMENT_RECORD_NAME,
    MAX_MANAGEMENT_RECORD_BYTES,
    read_generation,
)

_RECORD = MANAGEMENT_RECORD_NAME
_MAX_RECORD_BYTES = MAX_MANAGEMENT_RECORD_BYTES
_MAX_POLICY_DIRECTORY_ENTRIES = 4096
_MAX_MANAGEMENT_INVENTORY_ENTRIES = 64
_MAX_MANAGEMENT_STATUS_ENTRIES = 512
_MAX_GENERATION_CGROUP_CHILDREN = 32
_MANAGEMENT_STATUS_TIMEOUT_SECONDS = 2.0
_NATIVE_RECORD = NATIVE_INSTALLATION_DIRECTORY / _RECORD


def _staging_temporary(path: Path, limit: int, finals: set[Path]) -> tuple[bytes, int]:
    """Authenticate a bounded installer temporary, including an interrupted hard-link publish."""
    if protected_path(path.parent, directory=True) != path.parent:
        raise ValueError("Partial staging directory aliases other authority.")
    before = path.lstat()
    mode = stat.S_IMODE(before.st_mode)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_mode & UNSAFE_HOST_PATH_MODE_BITS
        or before.st_nlink not in {1, 2}
        or before.st_size > limit
        or "security.capability" in os.listxattr(path)
    ):
        raise ValueError("Unsafe partial installation staging file.")

    def identity(details: os.stat_result) -> tuple[int, ...]:
        """Compare security-relevant inode state without mutable access time."""
        return (
            details.st_dev,
            details.st_ino,
            details.st_mode,
            details.st_nlink,
            details.st_size,
            details.st_mtime_ns,
            details.st_ctime_ns,
        )

    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        if identity(os.fstat(descriptor)) != identity(before):
            raise ValueError("Partial installation staging changed during inspection.")
        data = os.pread(descriptor, limit + 1, 0)
        if identity(os.fstat(descriptor)) != identity(before) or identity(path.lstat()) != identity(
            before
        ):
            raise ValueError("Partial installation staging changed during inspection.")
    finally:
        os.close(descriptor)
    if before.st_nlink == 2 and not any(
        final.exists()
        and final.lstat().st_dev == before.st_dev
        and final.lstat().st_ino == before.st_ino
        for final in finals
    ):
        raise ValueError("Partial installation staging has an unknown hard link.")
    return data, mode


def publish_generation_record(
    config: GenerationAdmission,
    files: dict[str, HostFile],
    lock: HostFile,
    compiler_lock: HostFile,
    namespace: tuple[int, int],
    policies: dict[str, dict[str, bytes]],
    helper: HostFile,
    compiled: bytes,
    source: bytes,
    artifacts: dict[str, HostFile],
) -> HostFile:
    """Persist complete original authority for later status and idle checked uninstall.

    The trusted installer supplies already qualified policy and original artifact bindings.
    Policy bytes must exist in the exact private artifact catalog. Preserve the namespace inode,
    every original file identity and full reservation; do not infer a deletion phase from absence.
    Reuse immutable staging and its management lock for durable publication. Existing records,
    including stale or partial ones, refuse replacement. A failed publication retains authority.
    This record proves file identity only, never operational readiness or backend availability.

    Args:
        config (GenerationAdmission): Fixed installed generation, ordinary identity and reservation.
        files (dict[str, HostFile]): Authenticated installed receiver and socket/service authority.
        lock (HostFile): Persistent immutable installation management lock.
        compiler_lock (HostFile): Distinct persistent immutable policy compiler lock.
        namespace (tuple[int, int]): Original loaded namespace device and inode.
        policies (dict[str, dict[str, bytes]]): Complete original namespace/profile compiled bytes.
        helper (HostFile): Original immutable generation helper before withdrawal.
        compiled (bytes): Exact global helper attachment compiled bytes.
        source (bytes): Exact literal global helper attachment source.
        artifacts (dict[str, HostFile]): Complete private immutable cleanup catalog by relative
            name.

    Returns:
        HostFile: Bound durable management record; all original authority revalidated.

    Raises:
        PermissionError: Initial-host administrator authority is absent.
        ValueError: Authority, schema, policy artifacts, bounds or existing inventory differ.
        OSError: Existing record, contention, publication or synchronization fails.
    """
    directory = config.receiver.parent
    record = directory / _RECORD
    if record.exists() or record.is_symlink():
        raise FileExistsError("Installation management record already exists; authority retained.")
    contents = {}
    for name, bound in artifacts.items():
        with bound.open_verified() as descriptor:
            contents[name] = os.pread(descriptor, bound.identity[4] + 1, 0)

    def artifact(data):
        """Locate exact policy bytes in the caller's original immutable cleanup catalog."""
        names = [name for name, value in contents.items() if value == data]
        if not names:
            raise ValueError("Management policy lacks its original artifact authority.")
        return min(names)

    value = {
        "schema": 1,
        "config": {
            "generation": config.generation,
            "uid": config.uid,
            "gid": config.gid,
            "receiver": files["receiver"].to_record(),
            "slice": config.slice_unit.to_record(),
            "reservation": config.reservation.to_record(),
        },
        "files": {role: bound.to_record() for role, bound in files.items()},
        "lock": lock.to_record(),
        "compiler_lock": compiler_lock.to_record(),
        "artifacts": {name: bound.to_record() for name, bound in artifacts.items()},
        "namespace": list(namespace),
        "policies": {
            relative: {profile: artifact(data) for profile, data in profiles.items()}
            for relative, profiles in policies.items()
        },
        "helper": helper.to_record(),
        "compiled": artifact(compiled),
        "source": artifact(source),
    }
    data = json.dumps(value, sort_keys=True).encode("ascii")
    if len(data) > _MAX_RECORD_BYTES:
        raise ValueError("Installation management record exceeds its bound.")
    # Shared immutable locks/reservations may live inside this generation in disposable setups.
    # Include their exact bytes for whole-inventory staging; they remain outside cleanup authority.
    for bound in (lock, compiler_lock, config.reservation):
        if bound.path.is_relative_to(directory):
            with bound.open_verified() as descriptor:
                contents[str(bound.path.relative_to(directory))] = os.pread(
                    descriptor, bound.identity[4] + 1, 0
                )
    contents[_RECORD] = data
    executable = {name for name, bound in artifacts.items() if bound.identity[2] & 0o111}
    staged = stage_generation_files(directory, contents, executable, lock)
    _, authority = read_generation(record)
    if staged[_RECORD] != authority:
        raise ValueError("Published management record changed.")
    return authority


def generation_status(record: Path) -> str:
    """Report authenticated file authority without claiming live enforcement readiness.

    Args:
        record (Path): Fixed installer-selected protected management record.

    Returns:
        str: Authenticated installed authority status without a runtime readiness claim.

    Raises:
        ValueError: The record is invalid, stale, ambiguous or incomplete.
        OSError: Authority is absent or cannot be authenticated.
    """
    read_generation(record)
    return "Installed authority authenticated; runtime enforcement is checked before each command."


def uninstall_authority(record: Path) -> tuple[HostFile, HostFile]:
    """Bind the complete record and retained exported uninstaller before a human decision.

    Args:
        record (Path): Fixed protected installer-selected management record.

    Returns:
        tuple[HostFile, HostFile]: Original record and present exported uninstaller, or the
            record twice when the uninstaller was removed during checked teardown.

    Raises:
        ValueError: Saved authority changed or the complete exported entry point is absent.
        OSError: Protected installation authority cannot be inspected.
    """
    installed, authority = read_generation(record, partial=True)
    entry = installed.artifacts.get("loop_uninstall.py")
    if entry is None or not entry.identity[2] & 0o111:
        raise ValueError("Installation lacks its original exported uninstaller authority.")
    return authority, entry if entry.path.exists() else authority


def uninstall_generation(record: Path, deadline: float) -> None:
    """Retry exact idle generation removal from the original record and native state.

    Args:
        record (Path): Fixed protected complete installation management record.
        deadline (float): Absolute monotonic budget, capped by retirement to five seconds.

    Raises:
        PermissionError: Initial-host administrator authority is absent.
        ValueError: Saved or live policy, inventory or installed authority differs.
        TimeoutError: The bounded checked retirement transaction expires.
        OSError: Live membership, contention or checked teardown prevents completion.
    """
    installed, authority = read_generation(record, partial=True, deadline=deadline)
    installed.retire(authority, deadline)


def verify_uninstalled(root: Path, deadline: float) -> None:
    """Prove that an absent record has no remaining Loop installation authority.

    Args:
        root (Path): Fixed protected installation directory.
        deadline (float): Absolute monotonic native inspection deadline.

    Raises:
        ValueError: Private files, units, policy, tracking or sockets remain.
        OSError: Protected native state cannot be inspected.
        TimeoutError: The native inspection deadline expires.
    """
    if root.exists() or root.is_symlink():
        if protected_path(root, directory=True) != root:
            raise ValueError("Installation directory aliases other authority.")
        expected = {root / name for name in NATIVE_LOCK_SEEDS} | {root / "reservations"}
        if not set(root.rglob("*")) <= expected:
            raise ValueError("Partial private installation inventory remains.")
        for name, data in NATIVE_LOCK_SEEDS.items():
            if not (root / name).exists() and not (root / name).is_symlink():
                continue
            bound = bind_host_file(root / name)
            with bound.open_verified() as descriptor:
                if (
                    bound.path != root / name
                    or stat.S_IMODE(bound.identity[2]) != 0o444
                    or os.pread(descriptor, len(data) + 1, 0) != data
                ):
                    raise ValueError("Shared installation seed differs from fixed authority.")
        reservations = root / "reservations"
        if (reservations.exists() or reservations.is_symlink()) and (
            protected_path(reservations, directory=True) != reservations
            or list(reservations.iterdir())
        ):
            raise ValueError("Installation capacity remains reserved.")
    if protected_path(_UNIT_DIRECTORY, directory=True) != _UNIT_DIRECTORY:
        raise ValueError("System unit directory aliases other authority.")
    if (
        next(_UNIT_DIRECTORY.glob(NATIVE_UNIT_PREFIX + "*"), None) is not None
        or next(_UNIT_DIRECTORY.glob(".loop-unit-native_*-*"), None) is not None
    ):
        raise ValueError("Loop system unit fragments remain.")
    tool = bind_host_file(SYSTEMCTL_PATH)
    prefix = NATIVE_UNIT_PREFIX + "native_"
    rows = _manager(
        tool, ["list-units", "--all", "--plain", "--no-legend", "--", prefix + "*"], deadline
    ).splitlines()
    names = set()
    for row in rows:
        fields = row.split()
        if not fields or len(names) >= _MAX_MANAGEMENT_INVENTORY_ENTRIES or fields[0] in names:
            raise ValueError("Ambiguous Loop manager inventory remains.")
        name = fields[0]
        if (
            re.fullmatch(
                re.escape(prefix)
                + r"[a-f0-9]{32}(?:\.slice|\.socket|@(?:[A-Za-z0-9_.:\-]{1,128})?\.service)",
                name,
            )
            is None
        ):
            raise ValueError("Unknown Loop manager record remains.")
        names.add(name)
    _retired_records_absent(tool, prefix, names, deadline)
    if _manager(tool, ["list-jobs", "--no-legend", "--", prefix + "*"], deadline).strip():
        raise ValueError("Loop manager jobs remain.")
    if next(CGROUP_ROOT.glob(prefix + "*.slice"), None) is not None:
        raise ValueError("Loop tracking remains.")
    if next(Path("/run").glob(prefix + "*.socket"), None) is not None:
        raise ValueError("Loop listener remains.")
    policy, device = _policy_root()
    namespaces = policy / "namespaces"
    _policy_directory(namespaces, policy, device)
    for count, path in enumerate(namespaces.iterdir(), 1):
        if count > _MAX_POLICY_DIRECTORY_ENTRIES:
            raise ValueError("AppArmor namespace inventory exceeds its bound.")
        if path.name.startswith("native_"):
            raise ValueError("Loop AppArmor namespace remains.")
    profiles = policy / "profiles"
    _policy_directory(profiles, policy, device)
    for count, path in enumerate(profiles.iterdir(), 1):
        if count > _MAX_POLICY_DIRECTORY_ENTRIES:
            raise ValueError("AppArmor profile inventory exceeds its bound.")
        _policy_directory(path, policy, device)
        if (
            read_policy_bytes(path / "name", MAX_NATIVE_POLICY_NAME_BYTES)
            .decode("utf-8")
            .strip()
            .startswith(prefix)
        ):
            raise ValueError("Loop AppArmor attachment remains.")


def clean_unrecorded_installation(
    config: GenerationAdmission,
    lock: HostFile,
    compiler_lock: HostFile,
    contents: dict[str, bytes],
    executables: set[str],
    sources: dict[str, dict[str, bytes]],
    source: bytes,
    deadline: float,
) -> None:
    """Remove an exact pre-record installation from current reviewed package authority.

    Args:
        config (GenerationAdmission): Fixed reserved generation and ordinary identity.
        lock (HostFile): Protected management lock.
        compiler_lock (HostFile): Distinct protected compiler lock.
        contents (dict[str, bytes]): Reviewed runtime catalog.
        executables (set[str]): Executable runtime catalog members.
        sources (dict[str, dict[str, bytes]]): Reviewed namespace policy sources.
        source (bytes): Reviewed global attachment source.
        deadline (float): Absolute monotonic cleanup deadline.

    Raises:
        PermissionError: Initial-host administrator authority is absent.
        ValueError: Files, units, policy, tracking or ownership differ from reviewed input.
        OSError: Live state, contention or native removal prevents cleanup.
        TimeoutError: The cleanup deadline expires.
    """
    root = config.receiver.parent
    if (root / _RECORD).exists() or (root / _RECORD).is_symlink():
        raise ValueError("Recorded installation requires its original lifecycle owner.")
    units = config.unit_sources()
    catalog = {
        **contents,
        config.slice_unit.path.name: units[config.slice_unit.path.name].encode(),
    }
    catalog.update(NATIVE_LOCK_SEEDS)
    cache = _installation_cache(config, contents, executables, sources, source)
    catalog.update({name: value for name, value in cache.items() if value})
    with _administration(lock), ExitStack() as pins:
        pins.enter_context(compiler_lock.open_verified())
        generation_limits(config.reservation, config.generation)
        with compiler_lock.open_verified() as compiler_fd:
            fcntl.flock(compiler_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            compiled = {}
            for relative, profiles in sources.items():
                compiled[relative] = {}
                for profile, data in profiles.items():
                    if (
                        compile_policy_source(data, deadline, names=True)
                        != profile.encode() + b"\n"
                    ):
                        raise ValueError("Reviewed source defines a different AppArmor profile.")
                    value = compile_policy_source(data, deadline)
                    compiled[relative][profile] = value
                    catalog[
                        f"policies/{'child/' + relative if relative else 'root'}/{profile}.bin"
                    ] = value
            _policy_inventory(compiled)
            unit_prefix = NATIVE_UNIT_PREFIX + config.generation
            if compile_policy_source(source, deadline, names=True) != unit_prefix.encode() + b"\n":
                raise ValueError("Reviewed attachment source differs from fixed identity.")
            attachment_bytes = compile_policy_source(source, deadline)
            catalog["attachment.bin"] = attachment_bytes
        validate_generation_catalog(root, catalog, executables)
        expected = {root / name for name in catalog} | {
            root / "receipt.json",
            config.reservation.path,
        }
        expected.update(
            parent
            for path in tuple(expected)
            for parent in path.parents
            if parent.is_relative_to(root)
        )
        if protected_path(root, directory=True) != root:
            raise ValueError("Partial installation directory aliases other authority.")
        receipt = root / "receipt.json"
        observed = set()
        temporary_files = set()
        for count, path in enumerate(root.rglob("*"), 1):
            temporary = (
                path.parent == root
                and path.name.startswith(".receipt-")
                or path.parent in {candidate.parent for candidate in expected}
                and path.name.startswith(".loop-file-")
            )
            if count > MAX_NATIVE_INSTALLATION_ENTRIES or path not in expected and not temporary:
                raise ValueError("Unknown partial installation file inventory.")
            observed.add(path)
            if temporary:
                receipt_stage = path.name.startswith(".receipt-")
                candidates = {
                    root / name: value
                    for name, value in catalog.items()
                    if (root / name).parent == path.parent
                }
                limit = (
                    MAX_NATIVE_RECEIPT_BYTES
                    if receipt_stage
                    else max(map(len, candidates.values()), default=0)
                )
                data, mode = _staging_temporary(
                    path, limit, {receipt} if receipt_stage else set(candidates)
                )
                if (
                    mode not in ({0o600, 0o444} if receipt_stage else {0o600, 0o444, 0o555})
                    or not receipt_stage
                    and not any(value.startswith(data) for value in candidates.values())
                ):
                    raise ValueError(
                        "Partial installation temporary file differs from reviewed staging."
                    )
                temporary_files.add(path)
        for temporary in temporary_files:
            temporary.unlink()
        for name, data in catalog.items():
            path = root / name
            if path not in observed:
                continue
            bound = bind_host_file(path)
            expected_mode = 0o555 if name in executables else 0o444
            if name == "helper" and stat.S_IMODE(bound.identity[2]) == 0o444:
                expected_mode = 0o444
            with bound.open_verified() as descriptor:
                if (
                    stat.S_IMODE(bound.identity[2]) != expected_mode
                    or os.pread(descriptor, len(data) + 1, 0) != data
                ):
                    raise ValueError("Partial installation file differs from reviewed package.")
        if receipt in observed:
            bound_receipt = bind_host_file(receipt)
            if (
                bound_receipt.identity[4] > MAX_NATIVE_RECEIPT_BYTES
                or stat.S_IMODE(bound_receipt.identity[2]) != 0o444
            ):
                raise ValueError("Partial installation receipt differs from fixed file limits.")
        for path in observed:
            if path.is_dir() and protected_path(path, directory=True) != path:
                raise ValueError("Aliased partial installation directory.")
        rows = read_mountinfo()
        if not rows or any(len(row) < 5 or Path(row[4]).is_relative_to(root) for row in rows):
            raise ValueError("Partial installation crosses a mount boundary.")
        if protected_path(_UNIT_DIRECTORY, directory=True) != _UNIT_DIRECTORY:
            raise ValueError("Partial installation unit directory aliases other authority.")
        fragments = tuple(
            _UNIT_DIRECTORY / (unit_prefix + suffix)
            for suffix in (".slice", ".socket", "@.service")
        )
        temporary_units = set(_UNIT_DIRECTORY.glob(".loop-unit-" + config.generation + "-*"))
        if (
            set(_UNIT_DIRECTORY.glob(NATIVE_UNIT_PREFIX + "native_*")) - set(fragments)
            or set(_UNIT_DIRECTORY.glob(".loop-unit-native_*-*")) - temporary_units
        ):
            raise ValueError("Unknown partial installation unit inventory.")
        tool = bind_host_file(SYSTEMCTL_PATH)
        pins.enter_context(tool.open_verified())
        for temporary in temporary_units:
            data, mode = _staging_temporary(
                temporary,
                MAX_NATIVE_UNIT_BYTES,
                {_UNIT_DIRECTORY / unit for unit in units},
            )
            if mode not in {0o600, 0o444} or not any(
                value.encode().startswith(data) for value in units.values()
            ):
                raise ValueError("Partial installation unit staging differs from fixed source.")
            temporary.unlink()
        for fragment in fragments:
            if fragment.exists() or fragment.is_symlink():
                bound = bind_host_file(fragment)
                with bound.open_verified() as descriptor:
                    if (
                        os.pread(descriptor, MAX_NATIVE_UNIT_BYTES + 1, 0)
                        != units[fragment.name].encode()
                    ):
                        raise ValueError("Partial installation unit differs from fixed source.")
        _manager(tool, ["daemon-reload"], deadline)
        native_prefix = NATIVE_UNIT_PREFIX + "native_"
        if _manager(
            tool, ["list-jobs", "--no-legend", "--", native_prefix + "*"], deadline
        ).strip():
            raise ValueError("Partial installation has pending manager jobs.")
        for count, row in enumerate(
            _manager(
                tool,
                ["list-units", "--all", "--plain", "--no-legend", "--", native_prefix + "*"],
                deadline,
            ).splitlines(),
            1,
        ):
            fields = row.split()
            if (
                count > _MAX_MANAGEMENT_INVENTORY_ENTRIES
                or not fields
                or re.fullmatch(
                    re.escape(unit_prefix)
                    + r"(?:\.slice|\.socket|@(?:[A-Za-z0-9_.:\-]{1,128})?\.service)",
                    fields[0],
                )
                is None
            ):
                raise ValueError("Foreign native manager record remains.")
        if _manager(
            tool,
            ["list-units", "--all", "--plain", "--no-legend", "--", unit_prefix + "@*.service"],
            deadline,
        ).strip():
            raise ValueError("Partial installation has receiver instances.")
        _recursive_idle(CGROUP_ROOT / (unit_prefix + ".slice"), absent=True)
        if next(CGROUP_ROOT.glob(native_prefix + "*.slice"), None) is not None:
            raise ValueError("Foreign native tracking remains.")
        listener = Path("/run") / (unit_prefix + ".socket")
        if listener.exists() or listener.is_symlink():
            raise ValueError("Partial installation listener is active.")
        if next(Path("/run").glob(native_prefix + "*.socket"), None) is not None:
            raise ValueError("Foreign native listener remains.")
        for fragment in fragments:
            fields = {
                "LoadState",
                "ActiveState",
                "SubState",
                "ControlGroup",
                "FragmentPath",
                "DropInPaths",
                "NeedDaemonReload",
                "Transient",
            }
            state = _retired_properties(tool, fragment.name, fields, deadline)
            if (
                state.keys() != fields
                or state.get("LoadState") != ("loaded" if fragment.exists() else "not-found")
                or state.get("ActiveState") != "inactive"
                or state.get("SubState") != "dead"
                or state.get("ControlGroup")
                or state.get("FragmentPath") != (str(fragment) if fragment.exists() else "")
                or state.get("DropInPaths")
                or state.get("NeedDaemonReload") != "no"
                or state.get("Transient") != "no"
            ):
                raise ValueError("Partial installation unit remains active or changed.")
        policy, device = _policy_root()
        namespaces = policy / "namespaces"
        _policy_directory(namespaces, policy, device)
        for count, entry in enumerate(namespaces.iterdir(), 1):
            if (
                count > _MAX_POLICY_DIRECTORY_ENTRIES
                or entry.name.startswith("native_")
                and entry.name != config.generation
            ):
                raise ValueError("Foreign native AppArmor namespace remains.")
        profiles = policy / "profiles"
        _policy_directory(profiles, policy, device)
        for count, entry in enumerate(profiles.iterdir(), 1):
            if count > _MAX_POLICY_DIRECTORY_ENTRIES:
                raise ValueError("AppArmor attachment inventory exceeds its bound.")
            _policy_directory(entry, policy, device)
            profile = (
                read_policy_bytes(entry / "name", MAX_NATIVE_POLICY_NAME_BYTES)
                .decode("utf-8")
                .strip()
            )
            if profile.startswith(native_prefix) and profile != unit_prefix:
                raise ValueError("Foreign native AppArmor attachment remains.")
        namespace = policy / "namespaces" / config.generation
        attachment = _attachment_entry(policy, device, unit_prefix)
        if attachment is not None:
            helper_path = root / "helper"
            if helper_path not in observed:
                raise ValueError("Partial attachment lacks its reviewed helper.")
            helper = bind_host_file(helper_path)
            _attest_attachment(policy, device, unit_prefix, helper, attachment_bytes)
        if receipt.exists() or receipt.is_symlink():
            receipt.unlink()
        if namespace.exists() or namespace.is_symlink():
            _policy_directory(namespace, policy, device)
            actual_children = {path.name for path in (namespace / "namespaces").iterdir()}
            if actual_children - (set(compiled) - {""}):
                raise ValueError("Unknown partial namespace child.")
            for relative in ("", *sorted(actual_children)):
                directory = namespace / "namespaces" / relative if relative else namespace
                nested = directory / "namespaces"
                _policy_directory(nested, policy, device)
                if relative and next(nested.iterdir(), None) is not None:
                    raise ValueError("Nested partial AppArmor namespace remains.")
                _policy_directory(directory / "profiles", policy, device)
                seen = set()
                for entry in (directory / "profiles").iterdir():
                    _policy_directory(entry, policy, device)
                    profile = (
                        read_policy_bytes(entry / "name", MAX_NATIVE_PROFILE_NAME_BYTES)
                        .decode("ascii")
                        .strip()
                    )
                    if (
                        profile in seen
                        or profile not in compiled[relative]
                        or read_policy_bytes(entry / "mode", MAX_NATIVE_POLICY_MODE_BYTES).strip()
                        != b"enforce"
                        or read_policy_bytes(entry / "raw_data", MAX_COMPILED_BYTES)
                        != compiled[relative][profile]
                        or (entry / "profiles").exists()
                        and next((entry / "profiles").iterdir(), None) is not None
                    ):
                        raise ValueError("Partial AppArmor profile differs from reviewed source.")
                    seen.add(profile)
            for relative in sorted(actual_children):
                (namespace / "namespaces" / relative).rmdir()
            namespace.rmdir()
        if attachment is not None:
            _remove_generation_attachment(
                config,
                {},
                lock,
                bind_host_file(root / "helper"),
                attachment_bytes,
                source,
                compiler_lock,
                deadline,
                nullcontext(),
                tracking_absent=True,
            )
        for suffix in (".slice", ".socket", "@.service"):
            fragment = _UNIT_DIRECTORY / (unit_prefix + suffix)
            if fragment.exists() or fragment.is_symlink():
                fragment.unlink()
        _manager(tool, ["daemon-reload"], deadline)
        _retired_records_absent(
            tool,
            unit_prefix,
            {unit_prefix + suffix for suffix in (".slice", ".socket", "@.service")},
            deadline,
        )
        for path in sorted(
            (root / name for name in catalog if name not in NATIVE_LOCK_SEEDS), reverse=True
        ):
            if path.exists() or path.is_symlink():
                path.unlink()
        for path in sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_dir() and path != config.reservation.path.parent
            ),
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            path.rmdir()
        config.reservation.path.unlink()
        verify_uninstalled(root, deadline)


def installation_status(*, details: bool = False) -> str:
    """Report whether Linux sandbox setup is installed, optionally with diagnostics.

    Args:
        details (bool): Include the managed file and native-setting inventory when true.

    Returns:
        str: A concise installation summary or the detailed observed inventory.
    """
    if sys.platform != "linux":
        return "Linux installation management is unavailable on this platform."
    if not details:
        return _concise_installation_status()

    # The catalog is absent from the retained isolated uninstaller, which never calls status.
    from .catalog import runtime_member_names

    root = _NATIVE_RECORD.parent
    paths = {
        root,
        _NATIVE_RECORD,
        root / "receipt.json",
        root / "management.lock",
        root / "compiler.lock",
        root / "reservations",
        _UNIT_DIRECTORY,
    }
    paths.update(root / name for name in runtime_member_names())
    generation = None
    paths_safe = True
    inventory_complete = True
    try:
        if protected_path(root, directory=True) != root:
            raise ValueError("Aliased native installation directory.")
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        paths_safe = False
        inventory_complete = False
    else:
        try:
            for count, path in enumerate(root.rglob("*"), 1):
                if count > _MAX_MANAGEMENT_STATUS_ENTRIES:
                    raise ValueError("Installation inventory exceeds inspection bound.")
                paths.add(path)
        except (OSError, ValueError):
            inventory_complete = False
        try:
            record = bind_host_file(_NATIVE_RECORD)
            with record.open_verified() as descriptor:
                saved = load_unique_json(
                    os.pread(descriptor, _MAX_RECORD_BYTES + 1, 0),
                    duplicate_message="Duplicate installation management field.",
                )
            if isinstance(saved, dict) and isinstance(saved.get("config"), dict):
                candidate = saved["config"].get("generation")
                if isinstance(candidate, str) and re.fullmatch(
                    r"[A-Za-z][A-Za-z0-9_]{0,127}", candidate
                ):
                    generation = candidate
                for name in saved.get("artifacts", {}):
                    relative = Path(name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Unsafe management inventory name.")
                    paths.add(root / relative)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError):
            inventory_complete = False
        if generation is None and not (_NATIVE_RECORD.exists() or _NATIVE_RECORD.is_symlink()):
            reservations = root / "reservations"
            try:
                if reservations.exists() or reservations.is_symlink():
                    if protected_path(reservations, directory=True) != reservations:
                        raise ValueError("Aliased native capacity directory.")
                    entries = reservations.iterdir()
                    first = next(entries, None)
                    if next(entries, None) is not None:
                        raise ValueError("Ambiguous native capacity inventory.")
                    if first is not None:
                        if re.fullmatch(r"native_[a-f0-9]{32}", first.name) is None:
                            raise ValueError("Unknown native capacity identity.")
                        generation = first.name
            except (OSError, ValueError):
                inventory_complete = False
    if generation is not None:
        prefix = NATIVE_UNIT_PREFIX + generation
        paths.update(
            _UNIT_DIRECTORY / (prefix + suffix) for suffix in (".slice", ".socket", "@.service")
        )
    unit_safe = True
    try:
        if protected_path(_UNIT_DIRECTORY, directory=True) != _UNIT_DIRECTORY:
            raise ValueError("Aliased system unit directory.")
        for count, path in enumerate(_UNIT_DIRECTORY.glob(NATIVE_UNIT_PREFIX + "*"), 1):
            if count > _MAX_MANAGEMENT_INVENTORY_ENTRIES:
                raise ValueError("System unit inventory exceeds inspection bound.")
            paths.add(path)
    except (OSError, ValueError):
        unit_safe = False
        inventory_complete = False
    try:
        summary = generation_status(_NATIVE_RECORD)
    except (ValueError, OSError):
        summary = "Linux execution unavailable; installation absent, partial or changed."
    files = []
    for path in sorted(paths):
        safe = unit_safe if path.is_relative_to(_UNIT_DIRECTORY) else paths_safe
        files.append(f"  {path}: {_observed_state(path) if safe else 'unknown'}")
    settings = _managed_settings(generation)
    if (
        generation is None
        and summary.startswith("Linux execution unavailable")
        and all(line.split(";", 1)[0].endswith(": absent") for line in settings)
    ):
        summary = "Linux execution unavailable; installation absent."
    if not inventory_complete:
        settings.append("  Inventory inspection: unknown")
    return "\n".join((summary, "Files:", *files, "Settings:", *settings))


def _concise_installation_status() -> str:
    """Return a safe user-facing summary without rendering the diagnostic inventory."""
    try:
        generation_status(_NATIVE_RECORD)
    except FileNotFoundError:
        try:
            verify_uninstalled(
                _NATIVE_RECORD.parent,
                time.monotonic() + _MANAGEMENT_STATUS_TIMEOUT_SECONDS,
            )
        except (OSError, ValueError):
            return (
                "Linux sandbox setup needs attention. Run /sandbox status details=true to inspect."
            )
        return (
            "Not installed. Sandboxed commands are unavailable. Run /sandbox install to set it up."
        )
    except (OSError, ValueError):
        return "Linux sandbox setup needs attention. Run /sandbox status details=true to inspect."
    return "Installed. Runtime enforcement is checked before each command."


def _observed_state(path: Path) -> str:
    """Inspect one path without following its final component."""
    try:
        path.lstat()
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "unknown"
    return "present"


def _managed_settings(generation: str | None) -> list[str]:
    """Inspect the fixed systemd and AppArmor setting kinds without changing them."""
    kinds = (
        "systemd slice",
        "systemd socket",
        "systemd service",
        "AppArmor namespace",
        "AppArmor attachment",
        "AppArmor profiles",
    )
    if generation is None:
        try:
            verify_uninstalled(
                _NATIVE_RECORD.parent,
                time.monotonic() + _MANAGEMENT_STATUS_TIMEOUT_SECONDS,
            )
        except (OSError, ValueError, TimeoutError):
            state = "unknown"
        else:
            state = "absent"
        return [
            *(
                f"  {kind}: {state}; identity=unavailable (no generation); "
                "expected load=not-found active=inactive enablement=absent; "
                f"observed load={'not-found' if state == 'absent' else 'unknown'} "
                f"active={'inactive' if state == 'absent' else 'unknown'} "
                f"enablement={state}"
                for kind in kinds[:3]
            ),
            *(
                f"  {kind}: {state}; identity=unavailable (no generation); "
                f"expected presence=absent; observed presence={state}"
                for kind in kinds[3:]
            ),
        ]
    prefix = NATIVE_UNIT_PREFIX + generation
    settings = []
    try:
        tool = bind_host_file(SYSTEMCTL_PATH)
        with tool.open_verified():
            for kind, suffix in zip(kinds[:3], (".slice", ".socket", "@.service"), strict=True):
                name = prefix + suffix
                inspected = prefix + "@status.service" if suffix == "@.service" else name
                properties = _retired_properties(
                    tool,
                    inspected,
                    {"LoadState", "ActiveState", "UnitFileState"},
                    time.monotonic() + _MANAGEMENT_STATUS_TIMEOUT_SECONDS,
                )
                state = {"loaded": "current", "not-found": "absent"}.get(
                    properties.get("LoadState"), "unknown"
                )
                observed = {
                    field: value if re.fullmatch(r"[a-z-]{1,32}", value) else "unknown"
                    for field, value in properties.items()
                }
                active = "active" if suffix != "@.service" else "inactive (inspection instance)"
                enabled = "static"
                settings.append(
                    f"  {kind}: {state}; {name}; inspected={inspected}; "
                    f"expected load=loaded active={active} "
                    f"enablement={enabled}; observed load={observed.get('LoadState', 'unknown')} "
                    f"active={observed.get('ActiveState', 'unknown')} "
                    f"enablement={observed.get('UnitFileState', 'unknown')}"
                )
    except (OSError, ValueError, TimeoutError):
        settings = [
            f"  {kind}: unknown; {prefix}{suffix}; expected load=loaded "
            f"active={'inactive (inspection instance)' if suffix == '@.service' else 'active'} "
            "enablement=static; "
            "observed load=unknown active=unknown enablement=unknown"
            for kind, suffix in zip(kinds[:3], (".slice", ".socket", "@.service"), strict=True)
        ]
    try:
        policy, device = _policy_root()
        namespace = _observed_state(policy / "namespaces" / generation)
        settings.append(
            f"  AppArmor namespace: {'current' if namespace == 'present' else namespace}; "
            f"{generation}; expected presence=present; observed presence={namespace}"
        )
        attachment = _attachment_entry(policy, device, prefix)
        mode = (
            read_policy_bytes(attachment / "mode", MAX_NATIVE_POLICY_MODE_BYTES)
            .decode("ascii")
            .strip()
            if attachment
            else "absent"
        )
        if mode not in {"enforce", "complain", "kill", "unconfined", "absent"}:
            mode = "unknown"
        settings.append(
            f"  AppArmor attachment: {'current' if attachment else 'absent'}; "
            f"{prefix}; expected mode=enforce; observed mode={mode}"
        )
        try:
            settings.extend(
                _profile_settings(policy, device, generation)
                if namespace == "present"
                else [
                    (
                        f"  AppArmor profiles: {namespace}; {generation}; expected mode=enforce; "
                        f"observed mode={namespace}"
                    )
                ]
            )
        except (OSError, ValueError):
            settings.append(
                f"  AppArmor profiles: unknown; {generation}; "
                "expected mode=enforce; observed mode=unknown"
            )
    except (OSError, ValueError):
        settings.extend(
            (
                (
                    f"  AppArmor namespace: unknown; {generation}; "
                    "expected presence=present; observed presence=unknown"
                ),
                (
                    f"  AppArmor attachment: unknown; {prefix}; "
                    "expected mode=enforce; observed mode=unknown"
                ),
                (
                    f"  AppArmor profiles: unknown; {generation}; "
                    "expected mode=enforce; observed mode=unknown"
                ),
            )
        )
    return settings


def _profile_settings(policy: Path, device: int, generation: str) -> list[str]:
    """List bounded kernel profile identities and observed enforcement modes."""
    namespace = policy / "namespaces" / generation
    _policy_directory(namespace, policy, device)
    children = namespace / "namespaces"
    _policy_directory(children, policy, device)
    directories = [namespace]
    for count, child in enumerate(children.iterdir(), 1):
        if count > _MAX_GENERATION_CGROUP_CHILDREN:
            raise ValueError("Managed AppArmor child inventory exceeds its bound.")
        _policy_directory(child, policy, device)
        directories.append(child)
    names = {}
    for directory in directories:
        profiles = directory / "profiles"
        _policy_directory(profiles, policy, device)
        for count, entry in enumerate(profiles.iterdir(), 1):
            if count > _MAX_MANAGEMENT_INVENTORY_ENTRIES:
                raise ValueError("Managed AppArmor profile inventory exceeds its bound.")
            _policy_directory(entry, policy, device)
            name = (
                read_policy_bytes(entry / "name", MAX_NATIVE_PROFILE_NAME_BYTES)
                .decode("ascii")
                .strip()
            )
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", name):
                raise ValueError("Unknown managed AppArmor profile name.")
            mode = (
                read_policy_bytes(entry / "mode", MAX_NATIVE_POLICY_MODE_BYTES)
                .decode("ascii")
                .strip()
            )
            if mode not in {"enforce", "complain", "kill", "unconfined"}:
                mode = "unknown"
            names[f"{directory.name}/{name}"] = mode
    return [
        f"  AppArmor profiles: {'current' if names else 'absent'}",
        *(
            f"  AppArmor profile {name}: current; expected mode=enforce;"
            f" observed mode={names[name]}"
            for name in sorted(names)
        ),
    ]


def _installation_cache(config, contents, executables, sources, source):
    """Validate required runtime roles and reserve generated policy/receipt catalog names."""
    receiver = config.receiver.name
    bootstrap = "loop/execution/sandbox/linux/bootstrap.py"
    fixed = {receiver, "helper", "loop_uninstall.py", "cache.lock", bootstrap}
    cache = {"attachment.apparmor": source}
    for relative, profiles in sources.items():
        for profile in profiles:
            cache[f"policies/{'child/' + relative if relative else 'root'}/{profile}.bin"] = b""
    cache["attachment.bin"] = b""
    if (
        not fixed <= contents.keys()
        or not {receiver, "helper", "loop_uninstall.py"} <= executables
        or set(contents) & {*cache, "receipt.json", _RECORD}
    ):
        raise ValueError("Incomplete or conflicting installation catalog.")
    config.unit_sources()
    return cache


def install_generation(
    config: GenerationAdmission,
    lock: HostFile,
    compiler_lock: HostFile,
    contents: dict[str, bytes],
    executables: set[str],
    sources: dict[str, dict[str, bytes]],
    source: bytes,
    deadline: float,
) -> HostFile:
    """Compose fresh installation and persist exact management authority before activation.

    The trusted administrator installer supplies a full reserved generation, protected staged
    slice and complete reviewed runtime catalog. Reuse the existing checked owners in order:
    stage immutable runtime, load fresh policy, retain compiled exports, publish fixed units,
    publish the complete installation receipt and management record, then activate admission.
    Each owner serializes and revalidates its original authority; no step adopts partial state.
    Failed post-activation integrity checks close admission through the checked drain owner,
    with a separate five-second closure budget; live descendants remain confined.
    A failure stops immediately and retains the full reservation and any published authority.
    Refresh must retire the old generation first; this operation never replaces an installation.
    The receiver/bootstrap/helper catalog remains trusted installer input, not client or model data.
    Success enables ordinary native admission through the selector; each payload still verifies
    enforcement before source starts. Installation does not certify release qualification.

    Args:
        config (GenerationAdmission): Fixed ordinary identity, staged slice and full reservation.
        lock (HostFile): Persistent immutable installation management lock.
        compiler_lock (HostFile): Distinct immutable installation-wide compiler lock.
        contents (dict[str, bytes]): Complete reviewed private runtime catalog, including original
            receiver, helper, bootstrap module, cache lock and retained uninstall entry.
        executables (set[str]): Exact executable runtime entries; must include helper, receiver
            and loop_uninstall.py. Other entries remain immutable read-only files.
        sources (dict[str, dict[str, bytes]]): Trusted bounded namespace/profile source inventory.
        source (bytes): Fixed literal single-profile global helper attachment source.
        deadline (float): Shared absolute monotonic deadline for loading and activation.

    Returns:
        HostFile: Original immutable complete management record for the activated generation.

    Raises:
        PermissionError: Initial-host administrator authority is absent.
        ValueError: Catalog, original authority, policy, units or activated identity differs.
        OSError: Contention, existing/partial state, publication or native operations fail.
        TimeoutError: The shared installation budget expires; capacity remains retained.
        BaseExceptionGroup: Activation and its bounded listener closure both fail.
    """
    directory = config.receiver.parent
    bootstrap = "loop/execution/sandbox/linux/bootstrap.py"
    cache = _installation_cache(config, contents, executables, sources, source)
    staged = stage_generation_files(directory, dict(contents), set(executables), lock)
    with config.slice_unit.open_verified():
        pass
    namespace, policies, compiled = load_generation_policy(
        config, lock, staged["helper"], sources, source, compiler_lock, deadline
    )
    for relative, profiles in policies.items():
        for profile, data in profiles.items():
            cache[f"policies/{'child/' + relative if relative else 'root'}/{profile}.bin"] = data
    cache["attachment.bin"] = compiled
    artifacts = stage_generation_files(directory, {**contents, **cache}, set(executables), lock)
    artifacts = {
        name: bound
        for name, bound in artifacts.items()
        if bound.path not in {lock.path, compiler_lock.path, config.reservation.path}
    }
    published = provision_generation_units(config, lock)
    runtime = GenerationAdmission(
        config.generation,
        config.uid,
        config.gid,
        config.receiver,
        published["slice_unit"],
        config.reservation,
    )
    files = {role: published[role] for role in ("receiver", "socket_unit", "service_unit")}
    installed = provision_receipt(
        directory / "receipt.json",
        {
            "helper": artifacts["helper"].path,
            "parser": discover_tool("apparmor_parser").path,
            "limiter": discover_tool("prlimit").path,
            "apparmor": apparmor_library().path,
            "bootstrap": artifacts[bootstrap].path,
            "receiver": config.receiver,
            "compiler_lock": compiler_lock.path,
            "cache_lock": artifacts["cache.lock"].path,
            "socket_unit": files["socket_unit"].path,
            "service_unit": files["service_unit"].path,
        },
        lock,
    )
    artifacts["receipt.json"] = installed["receipt"]
    record = publish_generation_record(
        runtime,
        files,
        lock,
        compiler_lock,
        namespace,
        policies,
        artifacts["helper"],
        compiled,
        source,
        artifacts,
    )
    activated, active_files = activate_generation(
        runtime,
        lock,
        artifacts["helper"],
        namespace,
        policies,
        compiled,
        source,
        compiler_lock,
        deadline,
    )
    try:
        if activated != runtime or active_files != files:
            raise ValueError("Activated installation differs from recorded authority.")
        with record.open_verified():
            pass
    except (ValueError, OSError) as error:
        try:
            with drained_generation(
                runtime,
                files,
                lock,
                time.monotonic() + NATIVE_MANAGEMENT_TIMEOUT_SECONDS,
            ):
                pass
        except (ValueError, OSError) as closure:
            raise ExceptionGroup(
                "Installation validation and admission closure failed.", [error, closure]
            ) from error
        raise
    return record
