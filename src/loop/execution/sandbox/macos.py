"""Launch approved shell commands under a native macOS Seatbelt profile."""

from __future__ import annotations

import json
import os
import platform
import pwd
import re
import secrets
import select
import stat
import subprocess
import tempfile
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import BinaryIO, cast

from ... import constants
from ...permissions.protection import (
    PRIVATE_READ_COMPONENTS,
    ProtectedWorkspacePaths,
    is_protected_external_read,
)
from ...telemetry import telemetry_audit
from ...utils import (
    ProcessCapture,
    ProcessCaptureStatus,
    kill_process_group,
    read_bounded_stream,
    supervise_process,
)
from .contracts import CommandProcessResult, SandboxOutcome, SandboxRequest

_LAUNCHER = "/usr/bin/sandbox-exec"
_SHELL = "/bin/sh"
_SHELL_STARTED = "LOOP_SHELL_STARTED\n"
_SHELL_WRAPPER = "printf 'LOOP_SHELL_STARTED\\n'; exec /bin/sh -c \"$1\""
_POLICY_VERSION = "macos-seatbelt-v1"
_CLANG_OBJECT = re.compile(r"[^/]*-[0-9A-Fa-f]{6}\.o")
_XCRUN_CACHE = re.compile(r"xcrun_db-[A-Za-z0-9]*")
_SYSTEM_READ_ROOTS = tuple(
    Path(path)
    for path in (
        "/System/Library",
        "/usr/bin",
        "/usr/lib",
        "/usr/libexec",
        "/usr/sbin",
        "/usr/share",
        "/bin",
        "/sbin",
        "/Library/Developer/CommandLineTools",
        "/dev",
        "/etc",
        "/var/select",
        "/private/var/select",
    )
)
_MANAGED_TOOLCHAIN_PREFIXES = (Path("/opt/homebrew/Cellar"), Path("/usr/local/Cellar"))
_MAX_RECEIPT_BYTES = 128 * 1024
_LOG_LIMIT = 64 * 1024
_LOG_TAG = re.compile(r"\n(LOOP_SBX_[0-9a-f]{32})$")
_LOG_READ_CHUNK_BYTES = 4096
_MAX_DENIAL_RECORDS = 32
_DENIAL_STREAM_STARTUP_SECONDS = 2.0
_DENIAL_RECORD_WAIT_SECONDS = 0.15
_DENIAL_STREAM_SHUTDOWN_SECONDS = 1.0
_DENIAL_LOG_WAIT_SECONDS = 2.0
_DENIAL_LOG_ATTEMPTS = 3
_DENIAL_LOG_RETRY_SECONDS = 0.05
_MAX_DIAGNOSTIC_CHARS = 500
_MAX_INSTALLED_TOOL_DEPENDENCIES = 32
_HELPER_COPY_CHUNK_BYTES = 1024 * 1024
_MIN_PROBE_TIMEOUT_SECONDS = 0.001


class DenialMonitor:
    """Keep one bounded live kernel-denial stream for concurrent command attempts."""

    _process: subprocess.Popen[bytes]
    _reader: threading.Thread
    _ready: threading.Event
    _lock: threading.Lock
    _pending: dict[str, tuple[threading.Event, list[tuple[int, str]]]]

    def __init__(self, deadline: float) -> None:
        self._lock = threading.Lock()
        self._pending = {}
        self._ready = threading.Event()
        self._process = subprocess.Popen(
            [
                "/usr/bin/log",
                "stream",
                "--style",
                "ndjson",
                "--predicate",
                'eventMessage CONTAINS "LOOP_SBX_"',
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        self._reader = threading.Thread(target=self._consume, daemon=True)
        self._reader.start()
        if not self._ready.wait(
            max(0, min(_DENIAL_STREAM_STARTUP_SECONDS, deadline - time.monotonic()))
        ):
            self.close()
            raise OSError("Seatbelt denial stream did not become ready.")
        if self._process.poll() is not None:
            self.close()
            raise OSError("Seatbelt denial stream exited before command launch.")

    def healthy(self) -> bool:
        """Report whether the log process and its reader are still active."""
        return self._process.poll() is None and self._reader.is_alive()

    def register(self, tag: str) -> None:
        """Track an exact attempt tag before its sandboxed child starts."""
        if not self.healthy():
            raise OSError("Seatbelt denial stream stopped before command launch.")
        with self._lock:
            self._pending[tag] = (threading.Event(), [])

    def denial(self, tag: str, deadline: float) -> str | None:
        """Wait briefly for a live, trusted denial from this exact attempt."""
        with self._lock:
            pending = self._pending.get(tag)
        if pending is None:
            return None
        pending[0].wait(max(0, min(_DENIAL_RECORD_WAIT_SECONDS, deadline - time.monotonic())))
        with self._lock:
            current = self._pending.get(tag)
        return (
            max(current[1], key=lambda item: item[0])[1]
            if current is not None and current[1]
            else None
        )

    def unregister(self, tag: str) -> None:
        """Discard one attempt's diagnostic state after its result is created."""
        with self._lock:
            self._pending.pop(tag, None)

    def close(self) -> None:
        """Stop the managed log process and bound reader cleanup."""
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait(timeout=_DENIAL_STREAM_SHUTDOWN_SECONDS)
        cast(BinaryIO, self._process.stdout).close()
        self._reader.join(timeout=_DENIAL_STREAM_SHUTDOWN_SECONDS)

    def _consume(self) -> None:
        """Validate only tagged kernel records while discarding all other log content."""
        try:
            stream = cast(BinaryIO, self._process.stdout)
            oversized = False
            while raw := stream.readline(_LOG_LIMIT + 1):
                self._ready.set()
                if oversized:
                    oversized = not raw.endswith(b"\n")
                    continue
                if len(raw) > _LOG_LIMIT or not raw.endswith(b"\n"):
                    oversized = not raw.endswith(b"\n")
                    continue
                try:
                    event = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(event, dict):
                    continue
                message = event.get("eventMessage")
                if not isinstance(message, str) or (match := _LOG_TAG.search(message)) is None:
                    continue
                tag = match.group(1)
                denial = _denial_record(event, tag)
                if denial is None:
                    continue
                with self._lock:
                    pending = self._pending.get(tag)
                    if pending is not None:
                        pending[1].append(denial)
                        del pending[1][:-_MAX_DENIAL_RECORDS]
                        pending[0].set()
        except (OSError, ValueError):
            pass
        finally:
            self._ready.set()


class SeatbeltDiagnosticService:
    """Own one live denial stream and process-scoped readiness cache.

    Instances are shared by backends in one application runtime and closed with that runtime.
    """

    _monitor: DenialMonitor | None
    _pid: int | None
    _verified_pid: int | None
    _darwin_temp_path: Path | None
    _lock: threading.Lock

    def __init__(self) -> None:
        self._monitor = None
        self._pid = None
        self._verified_pid = None
        self._darwin_temp_path = None
        self._lock = threading.Lock()

    def monitor(self, deadline: float) -> DenialMonitor:
        """Return a healthy shared stream for this process, starting it when needed.

        Args:
            deadline (float): Monotonic startup deadline.

        Returns:
            DenialMonitor: Live monitor used for tagged denial attribution.
        """
        with self._lock:
            pid = os.getpid()
            if self._pid != pid:
                self._monitor = None
                self._verified_pid = None
                self._darwin_temp_path = None
                self._pid = pid
            if self._monitor is not None and self._monitor.healthy():
                return self._monitor
            if self._monitor is not None:
                self._monitor.close()
            self._monitor = DenialMonitor(deadline)
            return self._monitor

    def parent_verified(self) -> bool:
        """Return whether this process passed the Seatbelt launch probe.

        Returns:
            bool: Whether the current process owns a verified probe result.
        """
        return self._verified_pid == os.getpid()

    def mark_parent_verified(self) -> None:
        """Record a successful Seatbelt launch probe for this process."""
        self._verified_pid = os.getpid()

    def darwin_temp(self, deadline: float) -> Path:
        """Return the validated compiler cache root, rechecking entries on each request.

        Args:
            deadline (float): Absolute monotonic command deadline.

        Returns:
            Path: Canonical private Darwin user temporary directory.

        Raises:
            TimeoutError: If validation exceeds the command deadline.
        """
        if self._pid != os.getpid():
            self._darwin_temp_path = None
            self._verified_pid = None
            self._pid = os.getpid()
        root = _darwin_temp(self._darwin_temp_path, deadline)
        self._darwin_temp_path = root
        return root

    def close(self) -> None:
        """Stop the current process's stream and clear cached readiness."""
        with self._lock:
            if self._pid == os.getpid() and self._monitor is not None:
                self._monitor.close()
            self._monitor = None
            self._verified_pid = None
            self._darwin_temp_path = None
            self._pid = None


def _literal(path: Path | str) -> str:
    """Quote an absolute pathname as a Seatbelt literal."""
    return json.dumps(str(path))


def _check_deadline(deadline: float) -> float:
    """Return remaining preparation time or stop before the command can launch."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Sandbox preparation exceeded the command deadline.")
    return remaining


def _request_failure(request: SandboxRequest) -> CommandProcessResult | None:
    """Distinguish changed authorization and expired budgets from native unavailability."""
    if not request.paths_are_current():
        return CommandProcessResult(SandboxOutcome.STALE, detail="Approved path identity changed.")
    if time.monotonic() >= request.deadline:
        return CommandProcessResult(SandboxOutcome.TIMED_OUT, detail="Command deadline expired.")
    return None


def _walk_granted_root(
    root: Path,
    deadline: float,
    *,
    rename_anchors: set[Path] | None = None,
    protection: ProtectedWorkspacePaths | None = None,
) -> None:
    """Reject unsafe aliases and bind ancestors of protected entries against relocation."""
    original = root.lstat()
    if not stat.S_ISDIR(original.st_mode) or root.resolve(strict=True) != root:
        raise ValueError(f"Granted root is not a real directory: {root}")
    protected_directories = (
        tuple(Path(name).parts for name in protection.directories) if protection is not None else ()
    )
    protected_files = (
        {*protection.instruction_names, *protection.files} if protection is not None else set()
    )
    protected_directory_names = {parts[-1] for parts in protected_directories}
    private_names = {name.casefold() for name in PRIVATE_READ_COMPONENTS}
    pending = [root]
    while pending:
        _check_deadline(deadline)
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                _check_deadline(deadline)
                details = entry.stat(follow_symlinks=False)
                if stat.S_ISREG(details.st_mode) and details.st_nlink > 1:
                    raise ValueError(f"Granted root contains a hardlinked file: {entry.path}")
                if rename_anchors is not None:
                    name = entry.name
                    protected = name.casefold() in private_names or name in protected_files
                    if not protected and name in protected_directory_names:
                        relative = Path(entry.path).relative_to(root)
                        protected = any(
                            relative.parts[-len(parts) :] == parts
                            for parts in protected_directories
                        )
                    if protected:
                        anchor = Path(entry.path)
                        if not stat.S_ISDIR(details.st_mode):
                            anchor = anchor.parent
                        while anchor != root:
                            rename_anchors.add(anchor)
                            anchor = anchor.parent
                if stat.S_ISDIR(details.st_mode):
                    pending.append(Path(entry.path))
    latest = root.lstat()
    if (original.st_dev, original.st_ino) != (latest.st_dev, latest.st_ino):
        raise ValueError(f"Granted root changed during preflight: {root}")


def _darwin_temp(cached: Path | None, deadline: float) -> Path:
    """Validate the narrow compiler cache area from trusted host state."""
    root = cached
    if root is None:
        remaining = _check_deadline(deadline)
        result = subprocess.run(
            ["/usr/bin/getconf", "DARWIN_USER_TEMP_DIR"],
            capture_output=True,
            text=True,
            check=True,
            close_fds=True,
            timeout=remaining,
        )
        root = Path(result.stdout.strip()).resolve(strict=True)
    before = root.lstat()
    if (
        not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o077
    ):
        raise ValueError(f"Unsafe Darwin user temporary directory: {root}")
    with os.scandir(root) as entries:
        for entry in entries:
            _check_deadline(deadline)
            name = entry.name
            if (
                name != "xcrun_db"
                and not _XCRUN_CACHE.fullmatch(name)
                and not _CLANG_OBJECT.fullmatch(name)
            ):
                continue
            details = entry.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
                or details.st_uid != os.getuid()
                or _CLANG_OBJECT.fullmatch(name)
            ):
                raise ValueError(f"Unsafe Darwin compiler cache entry: {entry.path}")
    after = root.lstat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise ValueError(f"Darwin user temporary directory changed: {root}")
    return root


def _trusted_home() -> Path:
    """Resolve the current account's real home without trusting ambient HOME."""
    try:
        home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
        details = home.stat()
    except (KeyError, OSError, RuntimeError) as exc:
        raise ValueError("Current account home directory is unavailable.") from exc
    if not home.is_dir() or details.st_uid != os.getuid():
        raise ValueError("Current account home directory is unsafe.")
    return home


def _search_roots(request: SandboxRequest) -> tuple[tuple[Path, Path], ...]:
    """Select safe lookup-only PATH roots without widening read authority."""
    roots = []
    home = _trusted_home()
    for path, real in request.path_roots:
        if (
            home.is_relative_to(path)
            or path in {Path("/private"), Path("/private/tmp"), Path("/tmp"), Path("/var")}
            or (
                is_protected_external_read(path)
                and not any(path.is_relative_to(system) for system in _SYSTEM_READ_ROOTS)
            )
        ) or (
            home.is_relative_to(real)
            or real in {Path("/private"), Path("/private/tmp"), Path("/tmp"), Path("/var")}
            or (
                is_protected_external_read(real)
                and not any(real.is_relative_to(system) for system in _SYSTEM_READ_ROOTS)
            )
        ):
            continue
        roots.append((path, real))
    return tuple(dict.fromkeys(roots))


def _python_environment_configs(request: SandboxRequest) -> tuple[Path, ...]:
    """Find the measured virtual-environment metadata needed by PATH interpreters."""
    configs = []
    search_roots = _search_roots(request)
    for _, root in search_roots:
        if root.name != "bin" or not any(
            root.is_relative_to(grant)
            for grant in (*_SYSTEM_READ_ROOTS, request.workspace, *request.read_roots)
        ):
            continue
        config = root.parent / "pyvenv.cfg"
        try:
            details = config.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise ValueError(f"Unsafe Python environment configuration: {config}")
        configs.append(config)
    return tuple(configs)


def _profile(
    request: SandboxRequest,
    scratch: Path,
    darwin_temp: Path,
    tag: str,
    compatibility_files: tuple[Path, ...] = (),
    rename_anchors: tuple[Path, ...] = (),
) -> str:
    """Build the reviewed Seatbelt profile from bound roots only."""
    for root in (request.workspace, *request.read_roots, *request.read_aliases):
        if is_protected_external_read(root):
            raise ValueError(f"Protected directory cannot be a read root: {root}")
    search_roots = _search_roots(request)
    reads = (
        *_SYSTEM_READ_ROOTS,
        request.workspace,
        scratch,
        *(alias.parent for _, alias, _ in request.aliases),
        *(target for _, _, target in request.aliases),
        *request.read_roots,
        *request.read_aliases,
    )
    search_paths = tuple(dict.fromkeys(path for pair in search_roots for path in pair))
    limited_search = tuple(
        path for path in search_paths if not any(path.is_relative_to(root) for root in reads)
    )
    ancestors = {parent for root in (*reads, *search_paths) for parent in root.parents}
    read_rules = (
        " ".join(
            f"(literal {_literal(path)})"
            for path in sorted(ancestors - {Path("/private/tmp")}, key=str)
        )
        + " "
        + " ".join(f"(subpath {_literal(path)})" for path in reads)
        + " "
        + " ".join(f"(literal {_literal(path)})" for path in compatibility_files)
        + " "
        + " ".join(f"(literal {_literal(alias)})" for _, alias, _ in request.aliases)
    )
    search_metadata = (
        " ".join(
            f"(literal {_literal(path)})"
            for path in sorted(ancestors & {Path("/private/tmp")}, key=str)
        )
        + " "
        + " ".join(f"(literal {_literal(path)})" for path in limited_search)
        + " "
        + " ".join(f'(regex #"^{re.escape(str(path))}/[^/]+$")' for path in limited_search)
    )
    search_metadata_rule = (
        f"(allow file-read-metadata {search_metadata})" if search_metadata.strip() else ""
    )
    write_rules = " ".join(
        f"(subpath {_literal(path)})" for path in (*request.write_roots, scratch)
    )
    git_granted = (
        request.git_create or request.workspace / constants.GIT_DIRECTORY in request.write_roots
    )
    workspace_pattern = re.escape(str(request.workspace))
    protection = request.protected_paths
    protected_patterns = [
        (
            f"^{workspace_pattern}/(.*/)?"
            f"({'|'.join(re.escape(name) for name in protection.instruction_names)})$"
        )
    ]
    for directory in protection.directories:
        prefix = "(.*/)?"
        if git_granted and directory == constants.GIT_DIRECTORY.as_posix():
            prefix = ".+/"
        protected_patterns.append(f"^{workspace_pattern}/{prefix}{re.escape(directory)}(/.*)?$")
    protected_patterns.extend(
        f"^{workspace_pattern}/(.*/)?{re.escape(file)}$" for file in protection.files
    )
    protected = " ".join(f'(regex #"{pattern}")' for pattern in protected_patterns)
    private_names = "(" + "|".join(re.escape(name) for name in PRIVATE_READ_COMPONENTS) + ")"
    private_roots = dict.fromkeys(
        (
            *_SYSTEM_READ_ROOTS,
            request.workspace,
            *request.read_roots,
            *request.read_aliases,
            *(path for path, _ in search_roots),
            *(real for _, real in search_roots),
            _trusted_home(),
        )
    )
    private_reads = " ".join(
        f'(regex #"^{re.escape(str(root).rstrip("/"))}/(.*/)?{private_names}(/.*)?$")'
        for root in private_roots
    )
    temp_prefix = re.escape(str(darwin_temp))
    hex_suffix = "[0-9A-Fa-f]" * 6
    temp_rules = " ".join(
        f'(regex #"^{pattern}$")'
        for pattern in (
            temp_prefix + r"/xcrun_db-[A-Za-z0-9]*",
            temp_prefix + r"/[^/]*-" + hex_suffix + r"\.o",
        )
    )
    temp_cache = f"(literal {_literal(darwin_temp / 'xcrun_db')}) {temp_rules}"
    network = "(allow network*)" if request.network else ""
    rename_rule = (
        "(deny file-write-unlink "
        + " ".join(f"(literal {_literal(path)})" for path in rename_anchors)
        + f' (with message "{tag}"))'
        if rename_anchors
        else ""
    )
    return f"""(version 1)
(deny default (with message "{tag}"))
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))
(allow process-info* (target same-sandbox))
(allow sysctl-read (sysctl-name-prefix "hw.") (sysctl-name-prefix "kern.proc.")
  (sysctl-name-prefix "kern.os") (sysctl-name "kern.argmax")
  (sysctl-name "kern.hostname") (sysctl-name "kern.maxfilesperproc")
  (sysctl-name "kern.maxproc") (sysctl-name "kern.version")
  (sysctl-name "vm.loadavg"))
(allow mach-lookup (global-name "com.apple.system.opendirectoryd.libinfo")
  (global-name "com.apple.bsd.dirhelper"))
(allow file-read* {read_rules})
{search_metadata_rule}
(allow file-read* (literal {_literal(darwin_temp)}) {temp_cache})
(deny file-read* {private_reads} (with message "{tag}"))
(allow file-write* {write_rules})
(allow file-write* {temp_cache})
(allow file-write-data (require-all (path "/dev/null") (vnode-type CHARACTER-DEVICE)))
(deny file-write* {protected} (with message "{tag}"))
(deny file-write-unlink {private_reads} (with message "{tag}"))
{rename_rule}
(deny file-write-unlink (require-all (literal {_literal(request.workspace)})
  (vnode-type DIRECTORY)) (with message "{tag}"))
(deny file-write-unlink (require-all (literal {_literal(scratch)})
  (vnode-type DIRECTORY)) (with message "{tag}"))
(deny system-fcntl (fcntl-command 80 110) (with message "{tag}"))
{network}
"""


def _denial_record(event: dict, tag: str) -> tuple[int, str] | None:
    """Accept only one exact tagged, relevant kernel Seatbelt denial."""
    sender = event.get("senderImagePath")
    if (
        event.get("processID") != 0
        or event.get("processImagePath") != "/kernel"
        or not isinstance(sender, str)
        or "/Sandbox.kext/" not in sender
    ):
        return None
    message = event.get("eventMessage", "")
    if not isinstance(message, str) or not message.endswith("\n" + tag):
        return None
    denial = message.split("\n", 1)[0]
    if (
        " deny(" not in denial
        or " /dev/" in denial
        or " file-read-data /Library/Preferences/Logging/" in denial
        or denial.endswith(
            (" network-outbound /private/var/run/syslog", " network-outbound /var/run/syslog")
        )
    ):
        return None
    if " network-" in denial:
        priority = 3
    elif " file-write" in denial:
        priority = 2
    elif " file-read-data " in denial:
        priority = 1
    else:
        return None
    return priority, denial[:_MAX_DIAGNOSTIC_CHARS]


def _denial_from_log(tag: str, started: str, deadline: float) -> str | None:
    """Retry briefly for a tagged denial while the kernel log catches up."""
    limit = min(deadline, time.monotonic() + _DENIAL_LOG_WAIT_SECONDS)
    for attempt in range(_DENIAL_LOG_ATTEMPTS):
        denial = _read_denial_from_log(tag, started, limit)
        if denial is not None:
            return denial
        if time.monotonic() + _DENIAL_LOG_RETRY_SECONDS >= limit:
            break
        if attempt < _DENIAL_LOG_ATTEMPTS - 1:
            time.sleep(_DENIAL_LOG_RETRY_SECONDS)
    return None


def _read_denial_from_log(tag: str, started: str, deadline: float) -> str | None:
    """Read one bounded kernel-log snapshot for this exact Seatbelt attempt."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    predicate = (
        f'processID == 0 AND eventMessage CONTAINS "{tag}" AND eventMessage CONTAINS "Sandbox:"'
    )
    try:
        process = subprocess.Popen(
            [
                "/usr/bin/log",
                "show",
                "--start",
                started,
                "--style",
                "ndjson",
                "--predicate",
                predicate,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except OSError:
        return None
    try:
        if process.stdout is None:
            return None
        limit = time.monotonic() + remaining
        buffered = b""
        seen = 0
        best = None
        while time.monotonic() < limit and seen < _LOG_LIMIT:
            readable, _, _ = select.select(
                [process.stdout], [], [], max(0, limit - time.monotonic())
            )
            if not readable:
                break
            data = os.read(process.stdout.fileno(), min(_LOG_READ_CHUNK_BYTES, _LOG_LIMIT - seen))
            if not data:
                break
            seen += len(data)
            buffered += data
            while b"\n" in buffered:
                line, buffered = buffered.split(b"\n", 1)
                try:
                    event = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(event, dict):
                    continue
                denial = _denial_record(event, tag)
                if denial is None:
                    continue
                if best is None or denial[0] >= best[0]:
                    best = denial
        return best[1] if best is not None else None
    except (OSError, ValueError):
        return None
    finally:
        try:
            if process.poll() is None:
                process.kill()
        except OSError:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=_DENIAL_STREAM_SHUTDOWN_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except OSError:
                    pass


class MacOSSeatbeltBackend:
    """Run approved requests through Apple's fixed Seatbelt launcher.

    Args:
        diagnostics (SeatbeltDiagnosticService | None): Application-owned diagnostic service,
            or None to create one owned by this backend.
    """

    _diagnostics: SeatbeltDiagnosticService
    _owns_diagnostics: bool

    def __init__(self, diagnostics: SeatbeltDiagnosticService | None = None) -> None:
        self._diagnostics = diagnostics or SeatbeltDiagnosticService()
        self._owns_diagnostics = diagnostics is None

    def close(self) -> None:
        """Close diagnostics when this backend owns their lifecycle."""
        if self._owns_diagnostics:
            self._diagnostics.close()

    @property
    def system_read_roots(self) -> tuple[Path, ...]:
        """Return the reviewed system roots readable without an extra tool grant.

        Returns:
            tuple[Path, ...]: Static macOS baseline paths used by the native profile.
        """
        return _SYSTEM_READ_ROOTS

    @property
    def policy_version(self) -> str:
        """Return the reviewed Seatbelt policy revision.

        Returns:
            str: Stable identity of this macOS policy compiler.
        """
        return _POLICY_VERSION

    @property
    def scratch_prefix(self) -> str:
        """Return the private temporary-directory prefix validated at launch.

        Returns:
            str: Prefix for bound Seatbelt scratch directories.
        """
        return "loop-seatbelt-"

    @staticmethod
    def managed_tool_root(root: Path) -> bool:
        """Verify a versioned Homebrew package root with an installation receipt.

        Args:
            root (Path): Canonical candidate package directory.

        Returns:
            bool: Whether the root qualifies for automatic read authority.
        """
        return (
            any(root.is_relative_to(prefix) for prefix in _MANAGED_TOOLCHAIN_PREFIXES)
            and (root / "INSTALL_RECEIPT.json").is_file()
        )

    @staticmethod
    def installed_tool_roots(
        resolved: Path,
    ) -> tuple[tuple[Path, ...], tuple[tuple[Path, int, int, Path], ...]]:
        """Select a narrow versioned package and bind its declared support aliases.

        Args:
            resolved (Path): Canonical executable selected from the host PATH.

        Returns:
            tuple[tuple[Path, ...], tuple[tuple[Path, int, int, Path], ...]]: Read roots and
                alias device/inode bindings for this installed executable.

        Raises:
            ValueError: If installed package metadata is unsafe or unavailable.
        """
        parts = resolved.parts
        roots = [resolved.parent]
        opt_aliases = []
        if "Cellar" in parts:
            index = parts.index("Cellar")
            if len(parts) > index + 2:
                root = Path(*parts[: index + 3])
                roots = [root]
                cellar = Path(*parts[: index + 1])
                prefix = cellar.parent
                receipt = root / "INSTALL_RECEIPT.json"
                if receipt.is_file():
                    try:
                        with receipt.open("rb") as stream:
                            content = stream.read(_MAX_RECEIPT_BYTES + 1)
                        if len(content) > _MAX_RECEIPT_BYTES:
                            raise ValueError("Installed-tool metadata is too large.")
                        dependencies = json.loads(content).get("runtime_dependencies", [])
                        if (
                            not isinstance(dependencies, list)
                            or len(dependencies) > _MAX_INSTALLED_TOOL_DEPENDENCIES
                        ):
                            raise ValueError("Installed-tool dependency metadata is invalid.")
                        for dependency in dependencies:
                            name_value = dependency.get("full_name")
                            if (
                                not isinstance(name_value, str)
                                or len(name_value) > constants.MAX_COMMAND_PATH_LENGTH
                                or re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9@+._-]+$", name_value)
                                is None
                            ):
                                raise ValueError("Installed-tool dependency name is invalid.")
                            alias = prefix / "opt" / name_value
                            target = alias.resolve(strict=True)
                            if not alias.is_symlink() or target.parent != cellar / name_value:
                                raise ValueError("Installed-tool dependency alias is invalid.")
                            roots.append(target)
                    except (OSError, TypeError, AttributeError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            "Installed-tool dependency metadata is unavailable."
                        ) from exc
                for package_root in roots:
                    alias = prefix / "opt" / package_root.parent.name
                    if alias.is_symlink() and alias.resolve(strict=True) == package_root:
                        alias_details = alias.lstat()
                        opt_aliases.append(
                            (alias, alias_details.st_dev, alias_details.st_ino, package_root)
                        )
        return tuple(dict.fromkeys(roots)), tuple(opt_aliases)

    def run_read_only_argv(
        self,
        executable: Path,
        arguments: tuple[str, ...],
        descriptors: tuple[int, ...],
        deadline: float,
    ) -> ProcessCapture | None:
        """Run fixed argv under Seatbelt with only preopened content descriptors.

        Args:
            executable (Path): Canonical installed helper to copy into private state.
            arguments (tuple[str, ...]): Fixed-argv helper arguments with selected fd paths.
            descriptors (tuple[int, ...]): Already validated selected file descriptors.
            deadline (float): Absolute monotonic completion deadline.

        Returns:
            ProcessCapture | None: Bounded native output, including a typed timeout, or None
                if native execution is unavailable or preparation fails closed.
        """
        if platform.system() != "Darwin":
            return None
        if time.monotonic() >= deadline:
            return ProcessCapture(ProcessCaptureStatus.TIMED_OUT)
        try:
            with (
                tempfile.TemporaryDirectory(prefix=self.scratch_prefix) as directory,
                tempfile.TemporaryDirectory(prefix=self.scratch_prefix) as scratch_name,
            ):
                workspace = Path(directory).resolve(strict=True)
                scratch = Path(scratch_name).resolve(strict=True)
                helper = workspace / "helper"
                _check_deadline(deadline)
                descriptor = os.open(executable, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb") as source, helper.open("wb") as target:
                    executable_details = os.fstat(source.fileno())
                    if (
                        not stat.S_ISREG(executable_details.st_mode)
                        or executable_details.st_nlink != 1
                    ):
                        raise ValueError("Unsafe selected search helper.")
                    while chunk := source.read(_HELPER_COPY_CHUNK_BYTES):
                        _check_deadline(deadline)
                        target.write(chunk)
                helper.chmod(executable_details.st_mode & 0o777)
                request = SandboxRequest.create(
                    source="read-only argv",
                    cwd=workspace,
                    workspace=workspace,
                    read_roots=(),
                    write_roots=(),
                    network=False,
                    environment={"PATH": "/usr/bin:/bin", "LANG": "C", "TMPDIR": str(scratch)},
                    policy_version=self.policy_version,
                    deadline=deadline,
                    workspace_id="read-only-argv",
                )
                _walk_granted_root(workspace, deadline)
                if not request.paths_are_current():
                    return None
                _check_deadline(deadline)
                profile = _profile(
                    request,
                    scratch,
                    self._diagnostics.darwin_temp(deadline),
                    "LOOP_SBX_" + secrets.token_hex(16),
                )
                _check_deadline(deadline)
                process = subprocess.Popen(
                    [_LAUNCHER, "-p", profile, str(helper), *arguments],
                    cwd=workspace,
                    env=request.shell_environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    pass_fds=descriptors,
                    start_new_session=True,
                )
                return supervise_process(
                    process,
                    deadline,
                    stdout_limit=constants.MAX_OUTPUT_CHARS,
                    stderr_limit=constants.MAX_OUTPUT_CHARS,
                )
        except (TimeoutError, subprocess.TimeoutExpired):
            return ProcessCapture(ProcessCaptureStatus.TIMED_OUT)
        except (OSError, ValueError):
            return None

    def run(self, request: SandboxRequest) -> CommandProcessResult:
        """Run one bound shell request, failing closed on preparation or launch errors.

        Args:
            request (SandboxRequest): Approved shell source and operating-system authority.

        Returns:
            CommandProcessResult: Bounded output and exit status or a typed boundary failure.
        """
        if platform.system() != "Darwin" or request.policy_version != _POLICY_VERSION:
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail="Unsupported macOS policy.",
                failure_context="No compatible native sandbox policy is available.",
            )
        if failure := _request_failure(request):
            return failure
        try:
            bound_scratch = request.shell_environment().get("TMPDIR")
            scratch_context = (
                nullcontext(bound_scratch)
                if bound_scratch is not None
                else tempfile.TemporaryDirectory(prefix=self.scratch_prefix)
            )
            with scratch_context as directory:
                scratch = Path(directory).resolve(strict=True)
                details = scratch.lstat()
                if (
                    not stat.S_ISDIR(details.st_mode)
                    or details.st_uid != os.getuid()
                    or stat.S_IMODE(details.st_mode) != 0o700
                    or (bound_scratch is not None and str(scratch) != directory)
                    or not scratch.name.startswith(self.scratch_prefix)
                ):
                    raise ValueError("Unsafe command scratch directory.")
                _search_roots(request)
                rename_anchors = set()
                for root in dict.fromkeys(
                    (
                        request.workspace,
                        *request.read_roots,
                        *request.write_roots,
                        scratch,
                    )
                ):
                    _walk_granted_root(
                        root,
                        request.deadline,
                        rename_anchors=rename_anchors
                        if root == request.workspace or root in request.write_roots
                        else None,
                        protection=request.protected_paths if root == request.workspace else None,
                    )
                guarded_anchors = tuple(sorted(rename_anchors))
                darwin_temp = self._diagnostics.darwin_temp(request.deadline)
                environment = request.shell_environment()
                environment["TMPDIR"] = str(scratch)
                if failure := _request_failure(request):
                    return failure
                if not self._diagnostics.parent_verified():
                    probe_tag = "LOOP_SBX_" + secrets.token_hex(16)
                    compatibility_files = _python_environment_configs(request)
                    profile = _profile(
                        request,
                        scratch,
                        darwin_temp,
                        probe_tag,
                        compatibility_files,
                        guarded_anchors,
                    )
                    probe = subprocess.run(
                        [_LAUNCHER, "-p", profile, "/usr/bin/true"],
                        cwd=request.cwd,
                        env=environment,
                        capture_output=True,
                        text=True,
                        close_fds=True,
                        timeout=max(
                            _MIN_PROBE_TIMEOUT_SECONDS, request.deadline - time.monotonic()
                        ),
                        check=False,
                    )
                    if probe.returncode:
                        return CommandProcessResult(
                            SandboxOutcome.UNAVAILABLE,
                            detail=f"Seatbelt launch check failed (exit {probe.returncode}): "
                            f"{probe.stderr.strip()[:_MAX_DIAGNOSTIC_CHARS]}",
                            failure_context="Native sandbox launch check failed.",
                        )
                    self._diagnostics.mark_parent_verified()
                if failure := _request_failure(request):
                    return failure
                tag = "LOOP_SBX_" + secrets.token_hex(16)
                compatibility_files = _python_environment_configs(request)
                profile = _profile(
                    request,
                    scratch,
                    darwin_temp,
                    tag,
                    compatibility_files,
                    guarded_anchors,
                )
                monitor = self._diagnostics.monitor(request.deadline)
                monitor.register(tag)
                try:
                    if failure := _request_failure(request):
                        return failure
                    return self._execute(request, profile, environment, tag, monitor)
                finally:
                    monitor.unregister(tag)
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            return CommandProcessResult(SandboxOutcome.TIMED_OUT, detail=str(exc))
        except ValueError as exc:
            return CommandProcessResult(SandboxOutcome.INVALID, detail=str(exc))
        except OSError as exc:
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail=str(exc),
                failure_context="Native sandbox preparation or process launch failed.",
            )
        except KeyboardInterrupt:
            return CommandProcessResult(SandboxOutcome.CANCELLED, detail="Command cancelled.")

    @staticmethod
    def _execute(
        request: SandboxRequest,
        profile: str,
        environment: dict[str, str],
        tag: str,
        monitor: DenialMonitor,
    ) -> CommandProcessResult:
        """Capture both child streams with bounded storage and a single deadline."""
        started = time.strftime("%Y-%m-%d %H:%M:%S%z", time.localtime())
        telemetry_audit(
            "sandbox.launching",
            workspace_id=request.workspace_id,
            attempt_id=request.attempt_id,
        )
        process = subprocess.Popen(
            [
                _LAUNCHER,
                "-p",
                profile,
                _SHELL,
                "-c",
                _SHELL_WRAPPER,
                "loop-shell",
                request.execution_source,
            ],
            cwd=request.cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            close_fds=True,
            start_new_session=True,
        )
        telemetry_audit(
            "sandbox.launched",
            workspace_id=request.workspace_id,
            attempt_id=request.attempt_id,
        )
        capture = supervise_process(
            process,
            request.deadline,
            stdout_limit=constants.MAX_OUTPUT_CHARS + len(_SHELL_STARTED),
            terminate=kill_process_group,
            reader=read_bounded_stream,
            thread_factory=threading.Thread,
        )
        if capture.status is ProcessCaptureStatus.UNAVAILABLE:
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail="Child pipes unavailable.",
                failure_context="Native command output pipes could not be initialized.",
                possible_effects=True,
            )
        if capture.status is not ProcessCaptureStatus.COMPLETED:
            return CommandProcessResult(
                SandboxOutcome(capture.status.value),
                stdout=capture.stdout.removeprefix(_SHELL_STARTED),
                stderr=capture.stderr,
                stdout_discarded=capture.stdout_discarded,
                stderr_discarded=capture.stderr_discarded,
                possible_effects=True,
            )
        started_shell = capture.stdout.startswith(_SHELL_STARTED)
        if not started_shell:
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                stderr=capture.stderr,
                detail=f"Seatbelt failed before command launch (exit {capture.exit_code}).",
                failure_context="Native sandbox failed before the command could start.",
            )
        observed_denial = (
            monitor.denial(tag, request.deadline)
            or _denial_from_log(tag, started, request.deadline)
            if capture.exit_code
            else None
        )
        return CommandProcessResult(
            SandboxOutcome.COMPLETED,
            exit_code=capture.exit_code,
            stdout=capture.stdout[len(_SHELL_STARTED) :],
            stderr=capture.stderr,
            stdout_discarded=capture.stdout_discarded,
            stderr_discarded=capture.stderr_discarded,
            observed_denial=observed_denial or "",
        )
