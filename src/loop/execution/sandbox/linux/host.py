"""Discover and bind administrator-owned Linux preparation inputs in the host view."""

from __future__ import annotations

import ctypes
import hashlib
import io
import os
import re
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from ....constants import MAX_NATIVE_RECEIPT_BYTES, NATIVE_FILE_READ_CHUNK_BYTES
from ....utils.json import load_unique_json
from ....utils.process import ProcessCaptureStatus, supervise_process
from .constants import (
    INITIAL_USER_NAMESPACE_UID_MAP,
    PROC_UID_MAP_PATH,
    SYSTEM_COMMAND_PATH,
    SYSTEM_LOCALE,
    UNSAFE_HOST_PATH_MODE_BITS,
)

_SUDO_FORBIDDEN_MODE_BITS = 0o2022

NATIVE_LOCK_SEEDS = {
    "management.lock": b"Loop native management\n",
    "compiler.lock": b"Loop native compiler\n",
}

_MAX_HOST_FILE_BYTES = 32 * 1024 * 1024
_MAX_SYMLINKS_PER_PREPARATION = 40
_MAX_SYSTEM_SERVICE_UID = 999
MAX_SOURCE_BYTES = 65536
MAX_COMPILED_BYTES = 1048576
MAX_DIAGNOSTIC_BYTES = 65536
COMPILER_ADDRESS_BYTES = 268435456
COMPILER_SECONDS = 5
LIBRARY_DISCOVERY_SECONDS = 5.0
_MAX_LINKER_CACHE_OUTPUT_BYTES = 1_048_576
_MAX_LINKER_CACHE_ERROR_BYTES = 4_096

RUNTIME_READ_FILES = tuple(
    Path(name)
    for name in (
        "/etc/ld.so.cache",
        "/etc/resolv.conf",
        "/etc/hosts",
        "/etc/nsswitch.conf",
        "/etc/gai.conf",
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
        "/etc/ssl/cert.pem",
    )
)


def protected_path(path: Path, *, directory: bool = False, system_elevation: bool = False) -> Path:
    """Resolve a protected host file, accepting protected distribution symlinks.

    Call in the initial host user namespace, before UID mapping. Root administrator and
    dynamic-loader compromise are outside this boundary. Inspect every intermediate symlink
    destination and its ancestry, within a finite symlink ceiling per preparation.

    Args:
        path (Path): Absolute host tool or library spelling independent of workspace PATH.
        directory (bool): Require a directory instead of a single-link regular file.
        system_elevation (bool): Permit only the setuid bit on the resolved fixed /usr/bin/sudo
            executable for explicitly consented system authentication; False by default.

    Returns:
        Path: Canonical file or directory with protected ownership and ancestors.

    Raises:
        ValueError: A spelling, owner, permissions, type, links or capabilities are unsafe,
            or the symlink inspection exceeds its finite bound.
        OSError: Required filesystem metadata cannot be inspected.
    """
    if system_elevation and (path != Path("/usr/bin/sudo") or directory):
        raise ValueError("System elevation requires the fixed sudo executable.")

    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Invalid native host path.")

    resolved = path.resolve(strict=True)
    pending = [path, resolved]
    inspected = set()
    symlinks = 0
    while pending:
        spelling = pending.pop()
        for entry in (*spelling.parents, spelling):
            if entry in inspected:
                continue
            inspected.add(entry)
            details = entry.lstat()
            forbidden = (
                _SUDO_FORBIDDEN_MODE_BITS
                if system_elevation and entry == resolved
                else UNSAFE_HOST_PATH_MODE_BITS
            )
            # ACL effective write permissions are bounded by the group-class mode mask.
            if details.st_uid != 0 or (
                not stat.S_ISLNK(details.st_mode) and details.st_mode & forbidden
            ):
                raise ValueError("Unsafe native host authority.")

            if stat.S_ISLNK(details.st_mode):
                symlinks += 1
                if symlinks > _MAX_SYMLINKS_PER_PREPARATION:
                    raise ValueError("Native host symlink inspection exceeds its bound.")
                target = entry.parent / entry.readlink()
                pending.extend((target, target.resolve(strict=True)))

    details = resolved.stat()
    if directory:
        safe = stat.S_ISDIR(details.st_mode)
    else:
        safe = stat.S_ISREG(details.st_mode) and details.st_nlink == 1

    if system_elevation and not details.st_mode & 0o111:
        raise ValueError("System elevation tool is not executable.")

    if not safe:
        raise ValueError("Unsafe native host file.")

    if "security.capability" in os.listxattr(resolved):
        raise ValueError("Unexpected native host file capabilities.")

    return resolved


def in_initial_host_user_namespace() -> bool:
    """Report whether process IDs retain their initial host namespace meaning.

    Returns:
        bool: Whether the process uses the initial full UID mapping.
    """
    try:
        mapping = tuple(PROC_UID_MAP_PATH.read_text(encoding="ascii").split())
    except OSError:
        return False
    return mapping == INITIAL_USER_NAMESPACE_UID_MAP


def runtime_read_roots() -> tuple[Path, ...]:
    """Discover protected native runtime directory spellings and canonical destinations.

    Cover merged-/usr and separate lib64 layouts without an architecture-specific library
    filename. These roots belong only to the trusted runtime view; approved request/tool roots
    remain separate. No whole-root or home grant is inferred.

    Returns:
        tuple[Path, ...]: Existing protected runtime directories, including distribution aliases.

    Raises:
        ValueError: A runtime directory or encountered alias has unsafe authority.
        OSError: Required runtime metadata or /usr is unavailable.
    """
    roots = []
    for spelling in ("/usr", "/bin", "/sbin", "/lib", "/lib64"):
        path = Path(spelling)
        if spelling == "/usr" or path.exists():
            roots.extend((path, protected_path(path, directory=True)))
    return tuple(dict.fromkeys(roots))


def runtime_read_files(*, deadline: float | None = None) -> tuple[tuple[Path, HostFile], ...]:
    """Bind selected system resolver and CA bundles without exposing host configuration trees.

    Follow administrator-protected distribution symlinks, including the fixed systemd resolver
    files authenticated against the service identity in protected /etc/passwd. Missing optional
    spellings are omitted; dangling or unsafe present inputs refuse preparation. Preserve the
    original spelling as the namespace destination, with immutable canonical file authority.
    User configuration, private keys and whole /etc or /run trees are never inferred.

    Returns:
        tuple[tuple[Path, HostFile], ...]: Fixed destination spellings and authenticated inputs.

    Raises:
        ValueError: A present system input is unsafe, oversized or changes during binding.
        TimeoutError: The optional binding deadline expires.
        OSError: A present input cannot be authenticated, including a dangling symlink.
    """
    return tuple(
        (
            path,
            _bind_file(
                path,
                lambda spelling: _resolver_path(spelling, deadline=deadline),
                deadline=deadline,
            )
            if path == Path("/etc/resolv.conf")
            else bind_host_file(path, deadline=deadline),
        )
        for path in RUNTIME_READ_FILES
        if path.exists() or path.is_symlink()
    )


def _resolver_path(path: Path, *, deadline: float | None = None) -> Path:
    """Authenticate only root configuration or the fixed systemd service-owned resolver data."""
    try:
        return protected_path(path)
    except ValueError:
        pass

    protected_path(path.parent, directory=True)
    link = path.lstat()
    target = Path(os.path.normpath(path.parent / path.readlink()))
    if link.st_uid != 0 or target not in (
        Path("/run/systemd/resolve/stub-resolv.conf"),
        Path("/run/systemd/resolve/resolv.conf"),
    ):
        raise ValueError("Unsafe system resolver alias.")

    protected_path(Path("/run/systemd"), directory=True)
    account = bind_host_file(Path("/etc/passwd"), deadline=deadline)
    with account.open_verified(deadline=deadline) as descriptor:
        lines = os.pread(descriptor, account.identity[4] + 1, 0).splitlines()

    entries = [line.split(b":") for line in lines if line.startswith(b"systemd-resolve:")]
    if (
        len(entries) != 1
        or len(entries[0]) != 7
        or not entries[0][2].isdigit()
        or not 0 < int(entries[0][2]) <= _MAX_SYSTEM_SERVICE_UID
        or int(entries[0][2]) == os.getuid()
    ):
        raise ValueError("Unsafe system resolver service identity.")

    service_uid = int(entries[0][2])
    for entry, kind in ((target.parent, stat.S_ISDIR), (target, stat.S_ISREG)):
        details = entry.lstat()
        if (
            details.st_uid != service_uid
            or not kind(details.st_mode)
            or details.st_mode & UNSAFE_HOST_PATH_MODE_BITS
            or (entry == target and (details.st_nlink != 1 or details.st_mode & 0o111))
            or "security.capability" in os.listxattr(entry)
        ):
            raise ValueError("Unsafe system resolver service data.")

    return target


def library_candidates(stem: str, *, deadline: float | None = None) -> tuple[Path, ...]:
    """Discover protected installed library candidates through bounded host linker tooling.

    Args:
        stem (str): Simple library stem such as libseccomp or libapparmor, without a SONAME.
        deadline (float | None): Optional absolute monotonic request deadline; discovery also
            retains its five-second ceiling.

    Returns:
        tuple[Path, ...]: Protected canonical candidates, including possible foreign multilibs.

    Raises:
        ValueError: The stem, discovered path or host authority is unsafe.
        OSError: Discovery is missing, unsuccessful, truncated or has diagnostics.
        TimeoutError: The request budget expired before discovery.
    """
    if re.fullmatch(r"lib[a-z][a-z0-9_]*", stem) is None:
        raise ValueError("Invalid native library stem.")
    linker = next(
        (path for path in (Path("/sbin/ldconfig"), Path("/usr/sbin/ldconfig")) if path.exists()),
        None,
    )
    if linker is None:
        raise OSError("Host linker-cache discovery is unavailable.")

    discovery_deadline = time.monotonic() + LIBRARY_DISCOVERY_SECONDS
    deadline = min(deadline, discovery_deadline) if deadline is not None else discovery_deadline
    if time.monotonic() >= deadline:
        raise TimeoutError("Native library discovery deadline expired.")
    linker = protected_path(linker)
    process = subprocess.Popen(
        [str(linker), "-p"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="ascii",
        errors="replace",
        env={"PATH": SYSTEM_COMMAND_PATH, "LC_ALL": SYSTEM_LOCALE},
        cwd="/",
        close_fds=True,
        start_new_session=True,
    )
    result = supervise_process(
        process,
        deadline,
        stdout_limit=_MAX_LINKER_CACHE_OUTPUT_BYTES,
        stderr_limit=_MAX_LINKER_CACHE_ERROR_BYTES,
    )

    if (
        result.status is not ProcessCaptureStatus.COMPLETED
        or result.exit_code != 0
        or result.stderr
        or result.stdout_discarded != 0
        or result.stderr_discarded != 0
    ):
        raise OSError("Host linker-cache discovery failed.")

    candidates = []
    for line in result.stdout.splitlines():
        name, separator, location = line.strip().partition(" => ")
        if separator and name.split()[0].startswith(stem + ".so"):
            path = Path(location)
            if not path.is_absolute():
                raise ValueError("Invalid native library location.")
            candidates.append(protected_path(path))

    if not candidates:
        raise OSError(f"Installed {stem} is unavailable.")

    return tuple(candidates)


def _identity(details: os.stat_result) -> tuple[int, ...]:
    """Retain inode and byte-change identity that survives an ordinary UID mapping."""
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_nlink,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _digest(descriptor: int, deadline: float | None = None) -> str:
    """Hash bounded pinned bytes in chunks, stopping when the caller's deadline expires."""
    digest = hashlib.sha256()
    offset = 0
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Native host authority deadline expired.")
        size = min(NATIVE_FILE_READ_CHUNK_BYTES, _MAX_HOST_FILE_BYTES + 1 - offset)
        content = os.pread(descriptor, size, offset)
        if not content:
            break
        if len(content) > size:
            raise ValueError("Native host file exceeds the preparation bound.")
        offset += len(content)
        if offset > _MAX_HOST_FILE_BYTES:
            raise ValueError("Native host file exceeds the preparation bound.")
        digest.update(content)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Native host authority deadline expired.")
    return digest.hexdigest()


@dataclass(frozen=True)
class HostFile:
    """Carry host-authenticated file authority into a trusted mapped bootstrap.

    Construct current authority through host binding functions in the initial host view. Decoded
    saved records carry historical fields only until independently authenticated and revalidated.
    This record is trusted setup input, never model/request input. UID/GID are authenticated there;
    mapped revalidation
    compares inode, bytes and permissions without mistaking overflow UID for host root.

    Args:
        path (Path): Canonical protected host spelling.
        identity (tuple[int, ...]): Device, inode, mode, links, size and change timestamps.
        digest (str): SHA256 of the bound file bytes.
    """

    path: Path
    identity: tuple[int, ...]
    digest: str

    @classmethod
    def from_record(cls, value: dict) -> HostFile:
        """Decode recorded original fields without authenticating current filesystem state.

        Decode only previously authenticated administrator metadata. This does not adopt a
        replacement inode, prove current presence or authorize execution/removal. Consumers must
        separately authenticate current files or retain these fields solely as historical evidence.

        Args:
            value (dict): Saved path, seven exact nonnegative inode/change integers and SHA256.

        Returns:
            HostFile: Original recorded fields, requiring independent trust and revalidation.

        Raises:
            ValueError: Saved fields, exact integer types or digest shape are invalid.
        """
        if (
            not isinstance(value, dict)
            or set(value) != {"path", "identity", "sha256"}
            or not isinstance(value["path"], str)
            or not isinstance(value["identity"], list)
            or len(value["identity"]) != 7
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in value["identity"]
            )
            or not isinstance(value["sha256"], str)
            or re.fullmatch("[a-f0-9]{64}", value["sha256"]) is None
        ):
            raise ValueError("Invalid saved native file authority.")

        return cls(Path(value["path"]), tuple(value["identity"]), value["sha256"])

    def to_record(self, *, deadline: float | None = None) -> dict[str, object]:
        """Encode original file authority after revalidating its inode and bytes.

        Returns:
            dict[str, object]: Original canonical path, seven inode/change fields and SHA256;
                serialization does not rebind a replaced file or grant execution authority.

        Args:
            deadline (float | None): Optional absolute monotonic bound for revalidation.

        Raises:
            ValueError: Prepared file identity or bytes have changed.
            TimeoutError: The optional verification deadline expires.
            OSError: The original file cannot be pinned or inspected.
        """
        with self.open_verified(deadline=deadline):
            return {"path": str(self.path), "identity": list(self.identity), "sha256": self.digest}

    @contextmanager
    def open_verified(self, *, deadline: float | None = None) -> Generator[int, None, None]:
        """Pin and revalidate the exact prepared inode, including in a mapped child.

        Args:
            deadline (float | None): Optional absolute monotonic bound for verification.

        Yields:
            int: Owned close-on-exec read descriptor valid only within this context.

        Raises:
            ValueError: The file identity or bytes differ from authenticated preparation.
            TimeoutError: The optional verification deadline expires.
            OSError: The no-follow read descriptor or metadata is unavailable.
        """
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Native host authority deadline expired.")
        descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            if (
                _identity(os.fstat(descriptor)) != self.identity
                or _digest(descriptor, deadline) != self.digest
                or _identity(os.fstat(descriptor)) != self.identity
                or _identity(self.path.stat()) != self.identity
            ):
                raise ValueError("Prepared native host authority changed.")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Native host authority deadline expired.")
            yield descriptor
        finally:
            os.close(descriptor)


def bind_host_file(
    path: Path,
    *,
    system_elevation: bool = False,
    deadline: float | None = None,
) -> HostFile:
    """Authenticate a host file and bind its inode and bytes before UID mapping.

    Args:
        path (Path): Absolute protected tool/library candidate in the initial host view.
        system_elevation (bool): Permit the fixed sudo executable’s expected setuid bit only;
            False retains the ordinary no-privilege host-file contract.
        deadline (float | None): Optional absolute monotonic bound for identity and digest work.

    Returns:
        HostFile: Trusted authority for later inode/byte revalidation, not execution consent.

    Raises:
        ValueError: Host authority is unsafe, oversized or changes during binding.
        TimeoutError: The optional binding deadline expires.
        OSError: Required filesystem or descriptor operations fail.
    """
    return _bind_file(
        path,
        lambda spelling: protected_path(spelling, system_elevation=system_elevation),
        deadline=deadline,
    )


def _bind_file(
    spelling: Path,
    authenticate: Callable[[Path], Path],
    *,
    deadline: float | None = None,
) -> HostFile:
    """Pin bounded bytes while rechecking the specific host-input authority contract."""
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Native host authority deadline expired.")
    path = authenticate(spelling)
    before = _identity(path.stat())
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        digest = _digest(descriptor, deadline)
        if (
            _identity(os.fstat(descriptor)) != before
            or _identity(authenticate(spelling).stat()) != before
        ):
            raise ValueError("Native host authority changed during preparation.")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Native host authority deadline expired.")
        return HostFile(path, before, digest)
    finally:
        os.close(descriptor)


def discover_tool(name: str, *, deadline: float | None = None) -> HostFile:
    """Bind a required native management tool independently of workspace PATH.

    Args:
        name (str): bwrap, apparmor_parser or prlimit; no caller-selected search path.
        deadline (float | None): Optional absolute monotonic bound for binding the tool.

    Returns:
        HostFile: Protected installed tool identity; no version or enforcement claim.

    Raises:
        ValueError: The name or installed authority is unsafe or changed.
        TimeoutError: The optional binding deadline expires.
        OSError: The tool is absent or cannot be inspected.
    """
    locations = {
        "bwrap": ("/usr/bin/bwrap", "/bin/bwrap"),
        "apparmor_parser": ("/usr/sbin/apparmor_parser", "/sbin/apparmor_parser"),
        "prlimit": ("/usr/bin/prlimit", "/bin/prlimit"),
    }
    if name not in locations:
        raise ValueError("Invalid native management tool.")
    for spelling in locations[name]:
        path = Path(spelling)
        if path.exists():
            bound = bind_host_file(path, deadline=deadline)
            if not bound.identity[2] & 0o111:
                raise ValueError("Native management tool is not executable.")
            return bound
    raise OSError(f"Installed {name} is unavailable.")


def apparmor_library() -> HostFile:
    """Bind a loadable protected AppArmor library with the required transition API.

    Returns:
        HostFile: Host-authenticated native library to pass to the trusted bootstrap.

    Raises:
        ValueError: A discovered library or its bound identity is unsafe.
        OSError: No loadable candidate or required AppArmor API is available.
    """
    for path in library_candidates("libapparmor"):
        bound = bind_host_file(path)
        with bound.open_verified():
            try:
                library = ctypes.CDLL(str(bound.path), use_errno=True)
            except OSError:
                continue
            try:
                transition = library.aa_change_profile
                transition.argtypes = [ctypes.c_char_p]
                transition.restype = ctypes.c_int
            except AttributeError as error:
                raise OSError("Required libapparmor API is unavailable.") from error
            return bound
    raise OSError("Installed libapparmor is unavailable.")


def installation_files(receipt: Path, *, deadline: float | None = None) -> dict[str, HostFile]:
    """Authenticate installed artifacts against one protected, bounded receipt.

    Call in the initial host view, using the management owner's fixed receipt location.
    A receipt is administrator authority, never model/request data. Its exact schema is
    {"schema": 1, "files": {role: {"path": absolute_spelling, "sha256": digest}}}.
    Every required role must be present, with no extras, aliases to another role or missing
    bytes. Bind the receipt itself too, so later setup can revalidate it after decisions.
    Successful authentication proves installed file identity only; loaded policy, systemd
    admission and an operational transition still need their own checks before readiness.

    Args:
        receipt (Path): Fixed administrator-owned installation receipt, at most 64 KiB.
        deadline (float | None): Optional absolute monotonic bound for artifact authentication.

    Returns:
        dict[str, HostFile]: Exact receipt and bound helper, parser, limiter, library, bootstrap,
            receiver, locks and socket/service unit identities.

    Raises:
        ValueError: Receipt schema, trust, size, artifact identity or bytes are invalid.
        TimeoutError: The optional authentication deadline expires.
        OSError: Required installed files or native metadata cannot be inspected.
    """
    authority = bind_host_file(receipt, deadline=deadline)
    if authority.identity[4] > MAX_NATIVE_RECEIPT_BYTES:
        raise ValueError("Installation receipt exceeds its bound.")
    with authority.open_verified(deadline=deadline) as descriptor:
        contents = os.pread(descriptor, MAX_NATIVE_RECEIPT_BYTES + 1, 0)
    if len(contents) > MAX_NATIVE_RECEIPT_BYTES:
        raise ValueError("Installation receipt exceeds its bound.")

    record = load_unique_json(contents, duplicate_message="Duplicate installation receipt field.")
    roles = {
        "helper",
        "parser",
        "limiter",
        "apparmor",
        "bootstrap",
        "receiver",
        "compiler_lock",
        "cache_lock",
        "socket_unit",
        "service_unit",
    }
    if (
        not isinstance(record, dict)
        or set(record) != {"schema", "files"}
        or not isinstance(record["schema"], int)
        or isinstance(record["schema"], bool)
        or record["schema"] != 1
        or not isinstance(record["files"], dict)
        or set(record["files"]) != roles
    ):
        raise ValueError("Invalid installation receipt schema.")

    files = {"receipt": authority}
    identities = {authority.identity[:2]}
    for role, entry in record["files"].items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or not isinstance(entry["path"], str)
            or not isinstance(entry["sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]) is None
        ):
            raise ValueError("Invalid installation artifact record.")

        bound = bind_host_file(Path(entry["path"]), deadline=deadline)
        if bound.digest != entry["sha256"] or bound.identity[:2] in identities:
            raise ValueError("Installed artifact differs from receipt authority.")

        if role in {"helper", "parser", "limiter", "receiver"} and not bound.identity[2] & 0o111:
            raise ValueError("Installed entry point is not executable.")

        if role in {"compiler_lock", "cache_lock"} and bound.identity[2] & 0o222:
            raise ValueError("Installed lock must be read-only.")

        files[role] = bound
        identities.add(bound.identity[:2])
    with authority.open_verified(deadline=deadline):
        for bound in files.values():
            with bound.open_verified(deadline=deadline):
                pass
    return files


def read_policy_bytes(path: Path, limit: int) -> bytes:
    """Bound a kernel policy export independently of its reported filesystem size.

    The caller must authenticate the securityfs namespace before reading. Follow protected
    kernel-generated export links, including raw_data; this helper grants no policy authority.

    Args:
        path (Path): Export within the caller-authenticated kernel policy hierarchy.
        limit (int): Maximum permitted export bytes selected by the native owner, from zero
            through one MiB.

    Returns:
        bytes: Complete export contents within the selected bound.

    Raises:
        ValueError: The bound is invalid or export contents exceed it.
        OSError: The authenticated export cannot be read.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or not 0 <= limit <= 1_048_576:
        raise ValueError("Invalid kernel policy export bound.")

    with path.open("rb") as stream:
        data = stream.read(limit + 1)

    if len(data) > limit:
        raise ValueError("Oversized kernel policy export.")

    return data


def compile_policy_source(source: bytes, deadline: float, *, names: bool = False) -> bytes:
    """Compile bounded trusted source or enumerate its profiles without loading policy.

    The caller must serialize preparation under its authenticated installation or compiler lock.
    Pin and reverify protected parser, limiter and ABI inputs. Use a private anonymous input,
    empty parser configuration, one worker, no cache, no kernel loading and strict diagnostics.
    Bound address space to 256 MiB, CPU to five seconds, output to one MiB and diagnostics to
    64 KiB. Both enumeration and compilation consume the caller's existing deadline.

    Args:
        source (bytes): Nonempty administrator-generated policy source, at most 64 KiB.
        deadline (float): Absolute monotonic budget, capped to five seconds.
        names (bool): Enumerate profile names instead of compiling bytes; defaults to False.

    Returns:
        bytes: Nonempty exact parser output with byte-preserving transport.

    Raises:
        ValueError: Source or protected host input authority is invalid or changes.
        TimeoutError: The compilation budget expires.
        OSError: Native discovery, diagnostics, cancellation, overflow or compilation fails.
    """
    if not isinstance(source, bytes) or not 0 < len(source) <= MAX_SOURCE_BYTES:
        raise ValueError("Policy source exceeds the preparation bound.")

    deadline = min(deadline, time.monotonic() + COMPILER_SECONDS)
    if time.monotonic() >= deadline:
        raise TimeoutError("Policy compilation deadline expired.")

    parser = discover_tool("apparmor_parser", deadline=deadline)
    limiter = discover_tool("prlimit", deadline=deadline)
    abi = bind_host_file(Path("/etc/apparmor.d/abi/4.0"), deadline=deadline)
    with ExitStack() as inputs:
        for authority in (parser, limiter, abi):
            inputs.enter_context(authority.open_verified(deadline=deadline))
        stream = inputs.enter_context(tempfile.TemporaryFile())
        stream.write(source)
        stream.seek(0)

        if time.monotonic() >= deadline:
            raise TimeoutError("Policy compilation deadline expired.")

        process = subprocess.Popen(
            [
                str(limiter.path),
                f"--as={COMPILER_ADDRESS_BYTES}:{COMPILER_ADDRESS_BYTES}",
                f"--cpu={COMPILER_SECONDS}:{COMPILER_SECONDS}",
                "--core=0:0",
                "--",
                str(parser.path),
                "--config-file=/dev/null",
                "--base=/etc/apparmor.d",
                "--jobs=0",
                "--names" if names else "--stdout",
                "--skip-kernel-load",
                "--skip-cache",
                "--warn=all",
                "--Werror",
            ],
            stdin=stream,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env={"PATH": SYSTEM_COMMAND_PATH, "LC_ALL": SYSTEM_LOCALE},
            close_fds=True,
            start_new_session=True,
        )
        # Latin-1 is a byte-preserving transport for the shared text supervisor. Disabling
        # newline conversion is essential: compiled bytes can contain CR, LF and CRLF.
        process.stdout = io.TextIOWrapper(process.stdout, encoding="latin1", newline="")
        process.stderr = io.TextIOWrapper(process.stderr, encoding="latin1", newline="")
        result = supervise_process(
            process,
            deadline,
            stdout_limit=MAX_COMPILED_BYTES,
            stderr_limit=MAX_DIAGNOSTIC_BYTES,
        )
        if result.status is ProcessCaptureStatus.TIMED_OUT:
            raise TimeoutError("Policy compilation deadline expired.")

        if (
            result.status is not ProcessCaptureStatus.COMPLETED
            or result.exit_code != 0
            or not result.stdout
            or result.stderr
            or result.stdout_discarded != 0
            or result.stderr_discarded != 0
        ):
            raise OSError("Bounded policy compilation failed.")

        for authority in (parser, limiter, abi):
            inputs.enter_context(authority.open_verified(deadline=deadline))

        return result.stdout.encode("latin1")
