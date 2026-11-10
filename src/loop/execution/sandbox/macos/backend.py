"""Launch approved shell commands under a native macOS Seatbelt profile."""

from __future__ import annotations

import errno
import fcntl
import io
import json
import os
import platform
import plistlib
import pwd
import re
import secrets
import select
import shlex
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, NamedTuple, cast

from .... import constants
from ....permissions.protection import (
    PRIVATE_READ_COMPONENTS,
    ProtectedWorkspacePaths,
    is_protected_external_read,
)
from ....telemetry import telemetry_audit
from ....utils import (
    ProcessCapture,
    ProcessCaptureStatus,
    kill_process_group,
    open_selected_file,
    read_bounded_stream,
    supervise_process,
)
from ....utils.path import has_extended_access_acl
from ....utils.process import is_finite_deadline
from ..constants import (
    LANG_ENVIRONMENT_VARIABLE,
    MINIMAL_SYSTEM_PATH,
    PATH_ENVIRONMENT_VARIABLE,
    PYTHON_BIN_DIRECTORY_NAME,
    PYTHON_ENVIRONMENT_CONFIG_NAME,
    SYSTEM_LOCALE,
    TMPDIR_ENVIRONMENT_VARIABLE,
)
from ..contracts import (
    CommandProcessResult,
    DirectoryIdentity,
    QualificationProbeLayout,
    SandboxCapability,
    SandboxOutcome,
    SandboxRequest,
)
from .egress import MacOSEgressBroker

if TYPE_CHECKING:
    from ....commands import CommandRegistration
    from ....permissions.manager import PermissionManager

_LAUNCHER = "/usr/bin/sandbox-exec"
_SHELL_STARTED_TAG = "LOOP_SHELL_STARTED"
_SHELL_STARTED = f"{_SHELL_STARTED_TAG}\n"
_SHELL_WRAPPER = (
    f"{constants.SHELL_PRINT_BUILTIN} '{_SHELL_STARTED_TAG}\\n'; exec "
    f"{shlex.quote(constants.SHELL_COMMAND_PATH)} {shlex.quote(constants.SHELL_COMMAND_OPTION)} "
    '"$1"'
)
_POLICY_VERSION = "macos-seatbelt-v2"
_PRIVATE_TEMP_ROOT = Path("/private/tmp")
_LOCAL_SOCKET_DIRECTORY_ENVIRONMENT_VARIABLE = "LOOP_LOCAL_SOCKET_DIR"
_LOCAL_SOCKET_VOLUME_SECTORS = 16384
_LOCAL_SOCKET_VOLUME_SETUP_TIMEOUT_SECONDS = 30
_LOCAL_SOCKET_REGISTRY_CHARS = 256 * 1024
_LOCAL_SOCKET_REGISTRY_NODES = 128
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
        "/private/etc/ssl/cert.pem",
        "/private/etc/ssl/openssl.cnf",
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
_MAX_DIAGNOSTIC_RESULT_CHARS = 2000
_MAX_DIAGNOSTIC_CANDIDATES = 4
_DENIAL_RECORDS_PREFIX = "\x1eLOOP_DENIALS:"
_MAX_DIRECTORY_PROBES = 3
_MAX_INSTALLED_TOOL_DEPENDENCIES = 32
_HELPER_COPY_CHUNK_BYTES = 1024 * 1024
_MIN_PROBE_TIMEOUT_SECONDS = 0.001
_TERMINAL_ENVIRONMENT_LIMIT = 1024 * 1024
_TERMINAL_INPUT_WAIT_SECONDS = 1.0
_TERMINAL_BOOTSTRAP = """import fcntl, json, os, sys, termios
environment_fd, ready_fd = int(sys.argv[1]), int(sys.argv[2])
payload = bytearray()
while chunk := os.read(environment_fd, 65536):
    payload.extend(chunk)
    if len(payload) > 1048576:
        raise ValueError('Terminal environment exceeds its bound.')
os.close(environment_fd)
environment = json.loads(payload)
if not isinstance(environment, dict) or any(
    not isinstance(key, str) or not isinstance(value, str)
    for key, value in environment.items()
):
    raise ValueError('Invalid terminal environment.')
os.setsid()
slave = os.open(os.ttyname(0), os.O_RDWR)
try:
    fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
finally:
    os.close(slave)
os.write(ready_fd, b'R')
os.close(ready_fd)
os.execve('/usr/bin/sandbox-exec', ['/usr/bin/sandbox-exec', '-p', *sys.argv[3:]], environment)
"""


class LocalSocketVolume(NamedTuple):
    """Identify one owned, separate filesystem for attempt-local Unix sockets.

    Args:
        device (str): BSD device node returned by the owned RAM attachment.
        mount (Path): Verified mount path for this filesystem.
        device_id (int): Filesystem device number used for isolation checks.
        name (str): Unique mounted volume name.
        volume_uuid (str): Stable filesystem UUID checked before eject.
    """

    device: str
    mount: Path
    device_id: int
    name: str
    volume_uuid: str


class LocalSocketAttachmentError(OSError):
    """Retain an unverified RAM attachment for checked recovery.

    Args:
        device (str): Device name returned by the original attachment.
        marker (str): Unique registry marker supplied when the image was attached.
    """

    device: str
    marker: str

    def __init__(self, device: str, marker: str) -> None:
        super().__init__("The macOS local socket attachment needs checked cleanup.")
        self.device = device
        self.marker = marker


class OwnedTerminal(NamedTuple):
    """Hold one trusted PTY master and its verified slave until a job ends."""

    master: int
    slave: BinaryIO
    path: Path


@contextmanager
def _owned_terminal() -> Generator[OwnedTerminal, None, None]:
    """Open a fresh PTY without exposing its master to the sandboxed command."""
    master, slave_fd = os.openpty()
    try:
        with os.fdopen(slave_fd, "r+b", buffering=0) as slave:
            path = Path(os.ttyname(slave_fd))
            details = path.stat()
            bound = os.fstat(slave_fd)
            if (
                re.fullmatch(r"/dev/ttys[0-9]+", str(path)) is None
                or not stat.S_ISCHR(details.st_mode)
                or (details.st_dev, details.st_ino, details.st_rdev)
                != (bound.st_dev, bound.st_ino, bound.st_rdev)
                or details.st_uid != os.getuid()
            ):
                raise OSError("The owned macOS terminal slave could not be verified.")
            yield OwnedTerminal(master, slave, path)
    finally:
        os.close(master)


def _spawn_terminal(
    request: SandboxRequest,
    profile: str,
    environment: dict[str, str],
    terminal: OwnedTerminal,
) -> subprocess.Popen[str]:
    """Start the trusted controlling-TTY bootstrap before applying Seatbelt."""
    payload = json.dumps(environment, ensure_ascii=True, separators=(",", ":")).encode()
    if len(payload) > _TERMINAL_ENVIRONMENT_LIMIT:
        raise ValueError("Terminal command environment exceeds its bound.")
    environment_read, environment_write = os.pipe()
    try:
        ready_read, ready_write = os.pipe()
    except BaseException:
        os.close(environment_read)
        os.close(environment_write)
        raise
    process = None
    ready = False
    deadline = (
        request.deadline - request.runtime_timeout
        if request.runtime_timeout is not None
        else request.deadline
    )
    try:
        with (
            os.fdopen(environment_read, "rb", buffering=0) as child_environment,
            os.fdopen(environment_write, "wb", buffering=0) as parent_environment,
            os.fdopen(ready_read, "rb", buffering=0) as parent_ready,
            os.fdopen(ready_write, "wb", buffering=0) as child_ready,
        ):
            process = subprocess.Popen(
                [
                    "/usr/bin/python3",
                    "-I",
                    "-S",
                    "-c",
                    _TERMINAL_BOOTSTRAP,
                    str(child_environment.fileno()),
                    str(child_ready.fileno()),
                    profile,
                    constants.SHELL_COMMAND_PATH,
                    constants.SHELL_COMMAND_OPTION,
                    "exec 2>&1; " + _SHELL_WRAPPER,
                    "loop-shell",
                    request.execution_source,
                ],
                cwd=request.cwd,
                env={
                    PATH_ENVIRONMENT_VARIABLE: MINIMAL_SYSTEM_PATH,
                    LANG_ENVIRONMENT_VARIABLE: "C",
                },
                stdin=terminal.slave,
                stdout=terminal.slave,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                close_fds=True,
                pass_fds=(child_environment.fileno(), child_ready.fileno()),
            )
            child_environment.close()
            child_ready.close()
            terminal.slave.close()
            offset = 0
            while offset < len(payload):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [parent_environment], [], remaining)[1]:
                    raise TimeoutError("Terminal launch preparation exceeded its deadline.")
                written = os.write(parent_environment.fileno(), payload[offset:])
                if written <= 0:
                    raise OSError("Terminal environment write did not advance.")
                offset += written
            parent_environment.close()
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([parent_ready], [], [], remaining)[0]:
                raise TimeoutError("Terminal launch preparation exceeded its deadline.")
            ready = os.read(parent_ready.fileno(), 1) == b"R"
            if not ready:
                raise OSError("The trusted terminal bootstrap did not acquire its PTY.")
        process.stdout = io.TextIOWrapper(
            io.BufferedReader(os.fdopen(os.dup(terminal.master), "rb", buffering=0)),
            encoding="utf-8",
            errors="replace",
        )
        return process
    except BaseException:
        if process is not None:
            if ready:
                kill_process_group(process)
            else:
                process.kill()
            process.wait(timeout=constants.COMMAND_CLEANUP_WAIT_SECONDS)
            if process.stderr is not None:
                process.stderr.close()
        raise


def _write_terminal_control(master: int, frame: bytes) -> None:
    """Apply one validated bounded control frame to the owned PTY master."""
    marker = frame[:1]
    if marker == b"I" and len(frame) >= 3:
        size = int.from_bytes(frame[1:3], "big")
        if not 0 < size <= 4096 or len(frame) != size + 3:
            raise ValueError("Invalid terminal input frame.")
        payload = frame[3:]
    elif marker == b"Z" and len(frame) == 5:
        rows, columns = struct.unpack("!HH", frame[1:])
        if not 1 <= rows <= 500 or not 1 <= columns <= 500:
            raise ValueError("Invalid terminal size.")
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
        return
    elif frame == b"F":
        payload = b"\x04"
    elif frame == b"X":
        payload = b"\x03"
    else:
        raise ValueError("Invalid terminal control frame.")
    deadline = time.monotonic() + _TERMINAL_INPUT_WAIT_SECONDS
    offset = 0
    while offset < len(payload):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([], [master], [], remaining)[1]:
            raise TimeoutError("Terminal input remained blocked.")
        written = os.write(master, payload[offset : offset + 1])
        if written != 1:
            raise OSError("Terminal input write did not advance.")
        offset += written


def _local_socket_setup_timeout(deadline: float) -> float:
    """Return the remaining bounded time for native socket-volume preparation."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Local socket filesystem preparation exceeded its deadline.")
    return min(remaining, _LOCAL_SOCKET_VOLUME_SETUP_TIMEOUT_SECONDS)


def _local_socket_disk_info(device: str, deadline: float) -> dict[str, object] | None:
    """Read the current disk identity without treating missing media as an owned disk."""
    inspected = subprocess.run(
        ["/usr/sbin/diskutil", "info", "-plist", device],
        capture_output=True,
        timeout=_local_socket_setup_timeout(deadline),
        check=False,
    )
    if inspected.returncode:
        return None
    try:
        identity = plistlib.loads(inspected.stdout)
    except (plistlib.InvalidFileException, ValueError) as exc:
        raise OSError("Invalid macOS local socket filesystem identity.") from exc
    if not isinstance(identity, dict):
        raise OSError("Invalid macOS local socket filesystem identity.")
    return identity


def _local_socket_registry(marker: str, deadline: float) -> list[object] | None:
    """Read the bounded registry entries for one attachment marker."""
    remaining = _local_socket_setup_timeout(deadline)
    process = subprocess.Popen(
        ["/usr/sbin/ioreg", "-a", "-l", "-w0", "-r", "-k", marker],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        start_new_session=True,
    )
    inspected = supervise_process(
        process,
        min(deadline, time.monotonic() + remaining),
        stdout_limit=_LOCAL_SOCKET_REGISTRY_CHARS,
        stderr_limit=_MAX_DIAGNOSTIC_CHARS,
    )
    if (
        inspected.status is not ProcessCaptureStatus.COMPLETED
        or inspected.exit_code != 0
        or inspected.stdout_discarded != 0
        or inspected.stderr_discarded != 0
    ):
        return None
    try:
        entries = plistlib.loads(inspected.stdout.encode("utf-8"))
    except (plistlib.InvalidFileException, ValueError):
        return None
    return entries if isinstance(entries, list) else None


def _local_socket_attachment_matches(device: str, marker: str, deadline: float) -> bool:
    """Identify the marked RAM image and its exact whole-disk descendant in the registry."""
    entries = _local_socket_registry(marker, deadline)
    if entries is None or len(entries) != 1 or not isinstance(entries[0], dict):
        return False
    root = entries[0]
    if (
        root.get(marker) != 1
        or root.get("owner-uid") != os.getuid()
        or root.get("Physical Interconnect Location") != "RAM"
    ):
        return False
    pending = [root]
    matches = 0
    visited = 0
    while pending:
        node = pending.pop()
        visited += 1
        if visited > _LOCAL_SOCKET_REGISTRY_NODES or not isinstance(node, dict):
            return False
        if node.get("BSD Name") == Path(device).name:
            if (
                node.get("IOObjectClass") != "IOMedia"
                or node.get("Whole") is not True
                or node.get("Writable") is not True
                or node.get("Size") != _LOCAL_SOCKET_VOLUME_SECTORS * 512
            ):
                return False
            matches += 1
        children = node.get("IORegistryEntryChildren", [])
        if not isinstance(children, list):
            return False
        pending.extend(children)
    return matches == 1


def _pin_local_socket_attachment(device: str, marker: str, deadline: float) -> int:
    """Pin the exact marked RAM device before formatting or checked cleanup."""
    raw_device = device.replace("/dev/disk", "/dev/rdisk", 1)
    descriptor = os.open(raw_device, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        pinned = os.fstat(descriptor)
        current = os.stat(raw_device, follow_symlinks=False)
        if (
            not stat.S_ISCHR(pinned.st_mode)
            or (pinned.st_dev, pinned.st_ino, pinned.st_rdev)
            != (current.st_dev, current.st_ino, current.st_rdev)
            or not _local_socket_attachment_matches(device, marker, deadline)
        ):
            raise OSError("Unverified macOS local socket attachment.")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _remove_local_socket_attachment(device: str, marker: str, deadline: float) -> None:
    """Remove a partial RAM attachment only after revalidating its original identity."""
    descriptor = _pin_local_socket_attachment(device, marker, deadline)
    os.close(descriptor)
    if not _local_socket_attachment_matches(device, marker, deadline):
        raise OSError("The macOS local socket attachment changed before eject.")
    identity = _local_socket_disk_info(device, deadline)
    if (
        identity is None
        or identity.get("DeviceNode") != device
        or identity.get("BusProtocol") != "Disk Image"
        or identity.get("WholeDisk") is not True
    ):
        raise OSError("The macOS local socket attachment identity is uncertain.")
    result = subprocess.run(
        ["/usr/sbin/diskutil", "eject", device],
        capture_output=True,
        timeout=_local_socket_setup_timeout(deadline),
        check=False,
    )
    if result.returncode:
        raise OSError("Unable to eject the macOS local socket attachment.")


def _eject_local_socket_volume(volume: LocalSocketVolume) -> None:
    """Eject a RAM volume only while its original filesystem identity is current."""
    identity = _local_socket_disk_info(
        volume.device, time.monotonic() + _LOCAL_SOCKET_VOLUME_SETUP_TIMEOUT_SECONDS
    )
    if identity is None and not volume.mount.exists():
        return
    if (
        identity is None
        or identity.get("DeviceNode") != volume.device
        or identity.get("MountPoint") != str(volume.mount)
        or identity.get("VolumeName") != volume.name
        or identity.get("VolumeUUID") != volume.volume_uuid
        or identity.get("FilesystemType") != "hfs"
        or not volume.mount.is_mount()
        or volume.mount.stat().st_dev != volume.device_id
    ):
        raise OSError("The macOS local socket filesystem identity changed before eject.")
    result = subprocess.run(
        ["/usr/sbin/diskutil", "eject", volume.device],
        capture_output=True,
        timeout=_LOCAL_SOCKET_VOLUME_SETUP_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode:
        raise OSError("Unable to eject the owned macOS local socket filesystem.")


def _mount_local_socket_volume(deadline: float) -> LocalSocketVolume:
    """Mount an owned RAM filesystem so host socket nodes cannot be relocated into it."""
    name = f"{constants.APP_DISPLAY_NAME}IPC-{os.getpid()}-{secrets.token_hex(4)}"
    marker = f"LoopOwned{os.getpid():x}{secrets.token_hex(16)}"
    expected_mount = Path("/Volumes") / name
    if expected_mount.exists() or expected_mount.is_symlink():
        raise OSError("Local socket filesystem name is already in use.")
    try:
        attached = subprocess.run(
            [
                "/usr/bin/hdiutil",
                "attach",
                "-nomount",
                "-drivekey",
                f"{marker}=1",
                f"ram://{_LOCAL_SOCKET_VOLUME_SECTORS}",
            ],
            capture_output=True,
            text=True,
            timeout=_local_socket_setup_timeout(deadline),
            check=False,
        )
    except BaseException:
        telemetry_audit("sandbox.local_socket_volume.cleanup_pending", attachment=marker)
        raise
    if (
        attached.returncode
        or (match := re.fullmatch(r"\s*(/dev/disk[0-9]+)\s*", attached.stdout)) is None
    ):
        telemetry_audit("sandbox.local_socket_volume.cleanup_pending", attachment=marker)
        raise OSError("Unable to attach an owned macOS local socket filesystem.")
    device = match.group(1)
    descriptor = None
    detached = False
    try:
        descriptor = _pin_local_socket_attachment(device, marker, deadline)
        pin_test = subprocess.run(
            ["/usr/bin/hdiutil", "detach", "-force", device],
            capture_output=True,
            timeout=_local_socket_setup_timeout(deadline),
            check=False,
        )
        if pin_test.returncode == 0:
            detached = _local_socket_registry(marker, deadline) == []
        if pin_test.returncode != errno.EBUSY or not _local_socket_attachment_matches(
            device, marker, deadline
        ):
            raise OSError("The macOS local socket attachment cannot be safely pinned.")
        formatted = subprocess.run(
            [
                "/sbin/newfs_hfs",
                "-U",
                str(os.getuid()),
                "-G",
                str(os.getgid()),
                "-M",
                "0700",
                "-v",
                name,
                device,
            ],
            capture_output=True,
            timeout=_local_socket_setup_timeout(deadline),
            check=False,
        )
        if formatted.returncode:
            raise OSError("Unable to format the owned macOS local socket filesystem.")
        mounted = subprocess.run(
            ["/usr/sbin/diskutil", "mount", device],
            capture_output=True,
            timeout=_local_socket_setup_timeout(deadline),
            check=False,
        )
        if mounted.returncode:
            raise OSError("Unable to mount the owned macOS local socket filesystem.")
        identity = _local_socket_disk_info(device, deadline)
        if identity is None:
            raise OSError("Unable to inspect the owned macOS local socket filesystem.")
        if (
            identity.get("DeviceNode") != device
            or identity.get("MountPoint") != str(expected_mount)
            or identity.get("VolumeName") != name
            or not isinstance(identity.get("VolumeUUID"), str)
            or identity.get("FilesystemType") != "hfs"
            or not expected_mount.is_mount()
            or expected_mount.is_symlink()
            or expected_mount.resolve(strict=True) != expected_mount
        ):
            raise OSError("Unverified macOS local socket filesystem identity.")
        os.chmod(expected_mount, constants.PRIVATE_DIRECTORY_MODE)
        details = expected_mount.stat()
        if (
            details.st_uid != os.getuid()
            or stat.S_IMODE(details.st_mode) != constants.PRIVATE_DIRECTORY_MODE
            or details.st_dev == _PRIVATE_TEMP_ROOT.stat().st_dev
        ):
            raise OSError("Unsafe macOS local socket filesystem ownership or device.")
        return LocalSocketVolume(
            device, expected_mount, details.st_dev, name, identity["VolumeUUID"]
        )
    except BaseException as error:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if detached:
            raise
        try:
            _remove_local_socket_attachment(
                device, marker, time.monotonic() + _LOCAL_SOCKET_VOLUME_SETUP_TIMEOUT_SECONDS
            )
        except (OSError, TimeoutError, subprocess.TimeoutExpired):
            telemetry_audit(
                "sandbox.local_socket_volume.cleanup_pending", device=device, attachment=marker
            )
            if isinstance(error, Exception):
                raise LocalSocketAttachmentError(device, marker) from error
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


class DenialMonitor:
    """Keep one bounded live kernel-denial stream for concurrent command attempts."""

    _process: subprocess.Popen[bytes]
    _reader: threading.Thread
    _ready: threading.Event
    _available: bool
    _lock: threading.Lock
    _pending: dict[str, tuple[threading.Event, list[tuple[int, str]]]]

    def __init__(self, deadline: float) -> None:
        self._lock = threading.Lock()
        self._pending = {}
        self._ready = threading.Event()
        self._available = False
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
            return
        if self._process.poll() is not None:
            self.close()
            return
        self._available = True

    @property
    def available(self) -> bool:
        """Report whether monitor startup completed without an operational failure.

        Returns:
            bool: True when the live stream reached readiness and remained running.
        """
        return self._available

    def healthy(self) -> bool:
        """Report whether the log process and its reader are still active."""
        return self._process.poll() is None and self._reader.is_alive()

    def register(self, tag: str) -> bool:
        """Track an exact attempt tag, or report that the stream has stopped."""
        if not self.healthy():
            return False
        with self._lock:
            self._pending[tag] = (threading.Event(), [])
        return True

    def denial(
        self,
        tag: str,
        deadline: float,
        *,
        all_records: bool = False,
        startup_denials: frozenset[tuple[str, str]] = frozenset(),
    ) -> str | None:
        """Wait briefly for bounded, trusted denials from this exact attempt.

        Args:
            tag (str): Unique tag bound to the sandbox attempt.
            deadline (float): Monotonic deadline for diagnostic collection.
            all_records (bool): Return a bounded set instead of the highest-ranked record.
            startup_denials (frozenset[tuple[str, str]]): Successful-startup operation and
                resource pairs used only to order bounded results.

        Returns:
            str | None: One record or an encoded bounded collection, or None.
        """
        with self._lock:
            pending = self._pending.get(tag)
        if pending is None:
            return None
        pending[0].wait(max(0, min(_DENIAL_RECORD_WAIT_SECONDS, deadline - time.monotonic())))
        with self._lock:
            current = self._pending.get(tag)
            records = tuple(current[1]) if current is not None else ()
        if not records:
            return None
        if all_records:
            return _join_denials(records, startup_denials)
        return max(records, key=lambda item: item[0])[1]

    def unregister(self, tag: str) -> None:
        """Discard one attempt's diagnostic state after its result is created."""
        with self._lock:
            self._pending.pop(tag, None)

    def fingerprints(self, tag: str) -> frozenset[tuple[str, str]]:
        """Return complete operation/resource pairs retained for one registered attempt.

        Args:
            tag (str): Unique tag bound to the sandbox attempt.

        Returns:
            frozenset[tuple[str, str]]: Complete bounded operation and resource pairs.
        """
        with self._lock:
            current = self._pending.get(tag)
            records = tuple(current[1]) if current is not None else ()
        return frozenset(
            fingerprint
            for _priority, record in records
            if (fingerprint := _denial_fingerprint(record)) is not None
        )

    def close(self) -> None:
        """Stop the managed log process and bound reader cleanup."""
        self._available = False
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
    _system_directory: DirectoryIdentity | None
    _directory_probe_attempts: int
    _startup_denials: set[tuple[str, str]]
    _lock: threading.Lock

    def __init__(self) -> None:
        self._monitor = None
        self._pid = None
        self._verified_pid = None
        self._system_directory = None
        self._directory_probe_attempts = 0
        self._startup_denials = set()
        self._lock = threading.Lock()

    def monitor(self, deadline: float) -> DenialMonitor | None:
        """Return a healthy shared stream for this process, starting it when needed.

        Args:
            deadline (float): Monotonic startup deadline.

        Returns:
            DenialMonitor | None: Live monitor, or None when the stream failed to become ready.
        """
        with self._lock:
            pid = os.getpid()
            if self._pid != pid:
                self._monitor = None
                self._verified_pid = None
                self._system_directory = None
                self._directory_probe_attempts = 0
                self._startup_denials = set()
                self._pid = pid
            if self._monitor is not None and self._monitor.healthy():
                return self._monitor
            if self._monitor is not None:
                self._monitor.close()
            self._monitor = DenialMonitor(deadline)
            if self._monitor.available is False:
                self._monitor = None
                return None
            return self._monitor

    def parent_verified(self) -> bool:
        """Return whether this process passed the Seatbelt launch probe.

        Returns:
            bool: Whether the current process owns a verified probe result.
        """
        with self._lock:
            pid = os.getpid()
            if self._pid != pid:
                self._monitor = None
                self._verified_pid = None
                self._system_directory = None
                self._directory_probe_attempts = 0
                self._startup_denials = set()
                self._pid = pid
            return self._verified_pid == pid

    def mark_parent_verified(self, *, directory_probe_attempted: bool = True) -> None:
        """Record a successful Seatbelt launch probe for this process.

        Args:
            directory_probe_attempted (bool): Whether a diagnostic stream observed the probe.
        """
        with self._lock:
            self._verified_pid = os.getpid()
            if directory_probe_attempted:
                self._directory_probe_attempts += 1

    def directory_probe_pending(self) -> bool:
        """Return whether this process still needs a monitored startup probe.

        Returns:
            bool: Whether another bounded monitored startup probe can be attempted.
        """
        with self._lock:
            return (
                self._pid == os.getpid() and self._directory_probe_attempts < _MAX_DIRECTORY_PROBES
            )

    def system_directory(self, deadline: float) -> Path | None:
        """Revalidate an observed system directory before using its narrow read rule.

        Args:
            deadline (float): Monotonic deadline for filesystem validation.

        Returns:
            Path | None: Current independently verified directory, or None when unavailable.
        """
        with self._lock:
            candidate = self._system_directory if self._pid == os.getpid() else None
        if candidate is None:
            return None
        current = _optional_trusted_probe_directory(candidate.path, deadline)
        if current == candidate:
            return candidate.path
        with self._lock:
            if self._system_directory == candidate:
                self._system_directory = None
                self._verified_pid = None
                self._directory_probe_attempts = 0
        return None

    def set_system_directory(self, identity: DirectoryIdentity | None) -> None:
        """Store a directory identity independently validated from a probe candidate.

        Args:
            identity (DirectoryIdentity | None): Validated identity, or None when unavailable.
        """
        with self._lock:
            if identity is not None and self._system_directory is None:
                self._system_directory = identity
                self._directory_probe_attempts = _MAX_DIRECTORY_PROBES

    def observe_startup_denial(
        self,
        denial: str | None,
        fingerprints: frozenset[tuple[str, str]] = frozenset(),
    ) -> None:
        """Remember complete observations from a successful fixed startup probe.

        Args:
            denial (str | None): Bounded kernel records from the startup probe.
            fingerprints (frozenset[tuple[str, str]]): Complete records retained before the
                display limit was applied.
        """
        records = _denial_records(denial) if isinstance(denial, str) else ()
        observed = {
            fingerprint
            for record in records
            if (fingerprint := _denial_fingerprint(record)) is not None
        }
        observed.update(fingerprints)
        with self._lock:
            remaining = max(0, _MAX_DENIAL_RECORDS - len(self._startup_denials))
            self._startup_denials.update(sorted(observed)[:remaining])

    def startup_denials(self) -> frozenset[tuple[str, str]]:
        """Return this process's observed successful-startup fingerprints.

        Returns:
            frozenset[tuple[str, str]]: Exact operation and resource pairs, if observed.
        """
        with self._lock:
            return frozenset(self._startup_denials) if self._pid == os.getpid() else frozenset()

    def close(self) -> None:
        """Stop the current process's stream and clear cached readiness."""
        with self._lock:
            if self._pid == os.getpid() and self._monitor is not None:
                self._monitor.close()
            self._monitor = None
            self._verified_pid = None
            self._system_directory = None
            self._directory_probe_attempts = 0
            self._startup_denials = set()
            self._pid = None


def _literal(path: Path | str) -> str:
    """Quote an absolute pathname as a Seatbelt literal."""
    return json.dumps(str(path), ensure_ascii=False)


def _regex(pattern: str) -> str:
    """Encode regex syntax as an ordinary Seatbelt string, not a raw regex literal."""
    return f"(regex {_literal(pattern)})"


def _check_deadline(deadline: float) -> float:
    """Return remaining preparation time or stop before the command can launch."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Sandbox preparation exceeded the command deadline.")
    return remaining


def _trusted_probe_directory(path: Path, deadline: float) -> DirectoryIdentity | None:
    """Authenticate a probe-suggested system directory without trusting its pathname alone."""
    if (
        not path.is_absolute()
        or path == Path("/")
        or ".." in path.parts
        or len(str(path)) > constants.MAX_PATH_LENGTH
    ):
        return None
    private = {name.casefold() for name in PRIVATE_READ_COMPONENTS}
    for index, part in enumerate(path.parts):
        if part.casefold() in private and (
            part.casefold() != "library"
            or index == len(path.parts) - 1
            or path.parts[index - 1].casefold() != "system"
        ):
            return None
    home = _trusted_home()
    if path.is_relative_to(home) or home.is_relative_to(path):
        return None
    if not path.is_dir():
        return None
    immutable = getattr(stat, "SF_IMMUTABLE", 0)
    no_unlink = getattr(stat, "SF_NOUNLINK", 0)
    restricted = getattr(stat, "SF_RESTRICTED", 0)
    if not (immutable or restricted):
        return None
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    with ExitStack() as stack:
        descriptor = os.open("/", flags)
        stack.callback(os.close, descriptor)
        previous = os.fstat(descriptor)
        if (
            previous.st_uid != 0
            or not stat.S_ISDIR(previous.st_mode)
            or has_extended_access_acl(descriptor)
        ):
            return None
        for part in path.parts[1:]:
            _check_deadline(deadline)
            descriptor = os.open(part, flags, dir_fd=descriptor)
            stack.callback(os.close, descriptor)
            current = os.fstat(descriptor)
            if (
                current.st_uid != 0
                or not stat.S_ISDIR(current.st_mode)
                or has_extended_access_acl(descriptor)
                or (
                    previous.st_mode & constants.GROUP_AND_OTHER_WRITE_MODE_MASK
                    and not getattr(current, "st_flags", 0) & (immutable | no_unlink)
                )
            ):
                return None
            previous = current
        if previous.st_mode & constants.GROUP_AND_OTHER_WRITE_MODE_MASK or not getattr(
            previous, "st_flags", 0
        ) & (immutable | restricted):
            return None
        identity = DirectoryIdentity(path, previous.st_dev, previous.st_ino)
        return identity if identity.is_current() else None


def _optional_trusted_probe_directory(path: Path, deadline: float) -> DirectoryIdentity | None:
    """Ignore a candidate that disappeared or was replaced while being inspected."""
    try:
        return _trusted_probe_directory(path, deadline)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ESTALE):
            return None
        raise


def _probe_system_directory(denial: object, deadline: float) -> DirectoryIdentity | None:
    """Treat one trusted startup denial only as a candidate for independent validation."""
    if not isinstance(denial, str):
        return None
    if len(denial) >= _MAX_DIAGNOSTIC_CHARS or denial.endswith("…"):
        return None
    _, separator, path = denial.partition(" file-read-data ")
    if not separator or not path.startswith("/"):
        return None
    return _optional_trusted_probe_directory(Path(path), deadline)


def _request_failure(request: SandboxRequest) -> CommandProcessResult | None:
    """Distinguish changed authorization and expired budgets from native unavailability."""
    if not is_finite_deadline(request.deadline):
        return CommandProcessResult(SandboxOutcome.INVALID, detail="Command deadline is invalid.")
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
    ordinary_names: bool = False,
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
    private_names = (
        set() if ordinary_names else {name.casefold() for name in PRIVATE_READ_COMPONENTS}
    )
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
    """Find additional virtual-environment metadata outside existing read roots."""
    configs = []
    search_roots = _search_roots(request)
    read_roots = {*_SYSTEM_READ_ROOTS, request.workspace, *request.read_roots}
    for _, root in search_roots:
        if (
            root.name != PYTHON_BIN_DIRECTORY_NAME
            or root not in read_roots
            or not read_roots.isdisjoint(root.parents)
        ):
            continue
        config = root.parent / PYTHON_ENVIRONMENT_CONFIG_NAME
        try:
            details = config.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise ValueError(f"Unsafe Python environment configuration: {config}")
        configs.append(config)
    return tuple(configs)


def _exact_executable_reads(request: SandboxRequest) -> tuple[Path, ...]:
    """Validate exact external executable authority without exposing its directory."""
    reads = (*_SYSTEM_READ_ROOTS, request.workspace, *request.read_roots)
    paths = set()
    for _, spelling, resolved, device, inode in request.executable_identities:
        if any(resolved.is_relative_to(root) for root in reads):
            continue
        root = next(
            (
                canonical
                for _, canonical in _search_roots(request)
                if resolved.is_relative_to(canonical)
            ),
            None,
        )
        if (
            root is None
            or is_protected_external_read(spelling)
            or is_protected_external_read(resolved)
        ):
            raise ValueError("Unsafe exact executable path.")
        directory = root.stat()
        details = resolved.lstat()
        if (
            spelling.resolve(strict=True) != resolved
            or resolved.resolve(strict=True) != resolved
            or not stat.S_ISREG(details.st_mode)
            or (details.st_dev, details.st_ino) != (device, inode)
            or directory.st_uid not in (0, os.getuid())
            or directory.st_mode & constants.OTHER_WRITE_MODE_BIT
            or directory.st_mode & constants.GROUP_WRITE_MODE_BIT
            and directory.st_gid != os.getgid()
            or details.st_uid not in (0, os.getuid())
            or details.st_mode & constants.GROUP_AND_OTHER_WRITE_MODE_MASK
            or details.st_nlink != 1
        ):
            raise ValueError("Unsafe exact executable authority.")
        paths.update((spelling, resolved))
    return tuple(sorted(paths))


def _profile(
    request: SandboxRequest,
    scratch: Path,
    tag: str,
    compatibility_files: tuple[Path, ...] = (),
    rename_anchors: tuple[Path, ...] = (),
    *,
    owned_scratch: bool = False,
    system_directory: Path | None = None,
    local_socket_directory: Path | None = None,
    egress_socket: Path | None = None,
    terminal_slave: Path | None = None,
) -> str:
    """Build the reviewed Seatbelt profile from bound roots only."""
    confined_scratch = request.session_scratch or owned_scratch
    for root in (
        request.workspace,
        *request.read_roots,
        *request.read_aliases,
        *request.read_alias_targets,
        *request.write_roots,
        scratch,
    ):
        if is_protected_external_read(root) and not (confined_scratch and root == scratch):
            raise ValueError(f"Protected directory cannot be a read root: {root}")
    search_roots = _search_roots(request)
    local_roots = (local_socket_directory,) if local_socket_directory is not None else ()
    reads = (
        *_SYSTEM_READ_ROOTS,
        request.workspace,
        scratch,
        *request.read_roots,
        *request.read_aliases,
        *local_roots,
    )
    search_paths = tuple(dict.fromkeys(path for pair in search_roots for path in pair))
    limited_search = tuple(
        path for path in search_paths if not any(path.is_relative_to(root) for root in reads)
    )
    ancestors = {parent for root in (*reads, *search_paths) for parent in root.parents}
    read_rules = (
        " ".join(
            f"(require-all (subpath {_literal(path)}) (require-not {_regex(r'^/dev/ttys[0-9]+$')}))"
            if path == Path("/dev")
            else f"(subpath {_literal(path)})"
            for path in reads
        )
        + " "
        + " ".join(f"(literal {_literal(path)})" for path in compatibility_files)
        + " "
        + " ".join(f"(literal {_literal(path)})" for path in _exact_executable_reads(request))
    )
    system_metadata = (
        " ".join(
            f"(literal {_literal(path)})" for path in (system_directory, *system_directory.parents)
        )
        if system_directory is not None
        else ""
    )
    search_metadata = (
        " ".join(f"(literal {_literal(path)})" for path in sorted(ancestors, key=str))
        + " "
        + " ".join(f"(literal {_literal(path)})" for path in limited_search)
        + " "
        + " ".join(_regex(f"^{re.escape(str(path))}/[^/]+$") for path in limited_search)
        + " "
        + system_metadata
        + (
            " "
            + " ".join(
                f"(literal {_literal(path)})" for path in (egress_socket, *egress_socket.parents)
            )
            if egress_socket is not None
            else ""
        )
    )
    search_metadata_rule = (
        f"(allow file-read-metadata {search_metadata})" if search_metadata.strip() else ""
    )
    write_rules = " ".join(
        f"(subpath {_literal(path)})" for path in (*request.write_roots, scratch, *local_roots)
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
    protected = " ".join(_regex(pattern) for pattern in protected_patterns)
    private_names = (
        "("
        + "|".join(
            "".join(
                f"[{character.lower()}{character.upper()}]"
                if character.isalpha()
                else re.escape(character)
                for character in name
            )
            for name in PRIVATE_READ_COMPONENTS
        )
        + ")"
    )
    # Session storage is re-used, so enforce private-name guards within it as well.
    # Ungranted home paths are denied by default; direct scratch under home is writable.
    unbound_home_scratch = not confined_scratch and scratch.is_relative_to(_trusted_home())
    private_roots = dict.fromkeys(
        root
        for root in (
            *reads,
            *request.write_roots,
            *search_paths,
        )
        if root != scratch or unbound_home_scratch
    )

    def private_rules(*, system_reads: bool) -> str:
        """Protect private names while retaining independently guarded system roots."""
        discovered = (system_directory,) if system_directory is not None else ()
        anchors = (*_SYSTEM_READ_ROOTS, *discovered) if system_reads else ()
        roots = (*private_roots, *discovered) if system_reads else private_roots
        rules = []
        for root in roots:
            pattern = _regex(f"^{re.escape(str(root).rstrip('/'))}/(.*/)?{private_names}(/.*)?$")
            nested = tuple(path for path in anchors if path != root and path.is_relative_to(root))
            if nested:
                exclusions = " ".join(f"(subpath {_literal(path)})" for path in nested)
                pattern = f"(require-all {pattern} (require-not (require-any {exclusions})))"
            rules.append(pattern)
        result = " ".join(rules)
        if confined_scratch:
            result = (
                f"(require-all (require-not (subpath {_literal(scratch)})) (require-any {result})) "
            )
            if not owned_scratch and not request.scratch_data:
                result += _regex(f"^{re.escape(str(scratch))}/(.*/)?{private_names}(/.*)?$")
        return result

    private_reads = private_rules(system_reads=False)
    private_data_reads = private_rules(system_reads=True)
    scratch_ancestors = " ".join(f"(literal {_literal(path)})" for path in scratch.parents)
    private_metadata = (
        f"(require-all (require-any {private_data_reads}) "
        f"(require-not (require-any {scratch_ancestors} {system_metadata})))"
    )
    system_directory_rule = (
        f"(allow file-read-data (require-all (literal {_literal(system_directory)}) "
        "(vnode-type DIRECTORY)))"
        if system_directory is not None
        else ""
    )
    if request.network:
        network = "(allow network*)"
    else:
        network_rules = []
        if request.local_network and local_socket_directory is not None:
            socket_path = f"(subpath {_literal(local_socket_directory)})"
            network_rules.extend(
                (
                    f"(allow network-bind (local unix-socket {socket_path}))",
                    f"(allow network-outbound (remote unix-socket {socket_path}))",
                )
            )
        if egress_socket is not None:
            network_rules.append(
                f"(allow network-outbound (remote unix-socket (literal {_literal(egress_socket)})))"
            )
        network = (
            "(allow system-socket (socket-domain AF_UNIX))\n" + "\n".join(network_rules)
            if network_rules
            else ""
        )
    resolver_lookup = (
        '(allow mach-lookup (global-name "com.apple.system.opendirectoryd.libinfo"))'
        if request.network
        else ""
    )
    rename_rule = (
        "(deny file-write-unlink "
        + " ".join(f"(literal {_literal(path)})" for path in rename_anchors)
        + f' (with message "{tag}"))'
        if rename_anchors
        else ""
    )
    terminal_rules = (
        "(allow pseudo-tty)\n"
        '(allow file-read* file-write* file-ioctl (literal "/dev/ptmx"))\n'
        "(allow file-read* file-write* file-ioctl (require-all "
        '(regex #"^/dev/ttys[0-9]+$") (extension "com.apple.sandbox.pty")))\n'
        f"(allow file-ioctl (literal {_literal(terminal_slave)}))\n"
        "(deny file-read* file-write* (require-all "
        '(regex #"^/dev/ttys[0-9]+$") '
        '(require-not (extension "com.apple.sandbox.pty"))) '
        f'(with message "{tag}"))\n'
        "(deny file-ioctl (require-all "
        '(regex #"^/dev/ttys[0-9]+$") '
        '(require-not (extension "com.apple.sandbox.pty")) '
        f"(require-not (literal {_literal(terminal_slave)}))) "
        f'(with message "{tag}"))\n'
        f'(deny file-ioctl (ioctl-command {termios.TIOCSTI}) (with message "{tag}"))'
        if terminal_slave is not None
        else ""
    )
    # The directory helper returns shared host temporary paths; denying it lets compatible
    # native tools fall back to the bound TMPDIR without granting shared-directory access.
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
{resolver_lookup}
(allow file-read* {read_rules})
(allow file-read-data (literal "/"))
{system_directory_rule}
{search_metadata_rule}
(deny file-read-data {private_data_reads} (with message "{tag}"))
(deny file-read-metadata {private_metadata} (with message "{tag}"))
(allow file-write* {write_rules})
(allow file-write-data (require-all (path "/dev/null") (vnode-type CHARACTER-DEVICE)))
(deny file-write* {protected} (with message "{tag}"))
(deny file-write* {private_reads} (with message "{tag}"))
{rename_rule}
{terminal_rules}
(deny file-write-unlink (require-all (literal {_literal(request.workspace)})
  (vnode-type DIRECTORY)) (with message "{tag}"))
(deny file-write-unlink (require-all (literal {_literal(scratch)})
  (vnode-type DIRECTORY)) (with message "{tag}"))
(deny system-fcntl (fcntl-command 80 110) (with message "{tag}"))
{network}
"""


def _denial_fingerprint(record: str) -> tuple[str, str] | None:
    """Identify an exact, complete operation and resource without trusting clipped text."""
    if len(record) >= _MAX_DIAGNOSTIC_CHARS or record.endswith("…"):
        return None
    if len(tuple(re.finditer(r"\bdeny\(\d+\) \S+", record))) != 1:
        return None
    match = re.search(r"\bdeny\(\d+\) (\S+) (.+)$", record)
    return (match.group(1), match.group(2)) if match else None


def _denial_priority(record: str) -> int | None:
    """Rank a parsed denied operation without treating resource text as an operation."""
    matches = tuple(re.finditer(r"\bdeny\(\d+\) (\S+)", record))
    if not matches:
        return None
    if len(matches) != 1:
        return 1
    operation = matches[0].group(1)
    if operation.startswith("network-"):
        return 3
    if operation.startswith("file-write"):
        return 2
    return 1


def _denial_records(value: str) -> tuple[str, ...]:
    """Decode only the bounded internal collection format; keep single records intact."""
    if not value or len(value) > _MAX_DIAGNOSTIC_RESULT_CHARS:
        return ()
    if not value.startswith(_DENIAL_RECORDS_PREFIX):
        return (value,)
    try:
        records = json.loads(value[len(_DENIAL_RECORDS_PREFIX) :])
    except (ValueError, RecursionError):
        return ()
    if (
        not isinstance(records, list)
        or not 1 <= len(records) <= _MAX_DIAGNOSTIC_CANDIDATES
        or any(
            not isinstance(record, str) or not record or len(record) > _MAX_DIAGNOSTIC_CHARS
            for record in records
        )
    ):
        return ()
    return tuple(records)


def _join_denials(
    records: tuple[tuple[int, str], ...] | list[tuple[int, str]],
    startup_denials: frozenset[tuple[str, str]] = frozenset(),
) -> str | None:
    """Rank and deduplicate recent observations within the existing result-size bound."""
    selected = []
    seen = set()
    ranked = sorted(
        enumerate(records),
        key=lambda item: (
            _denial_fingerprint(item[1][1]) not in startup_denials,
            item[1][0],
            item[0],
        ),
        reverse=True,
    )
    for _position, (_priority, record) in ranked:
        fingerprint = _denial_fingerprint(record) or record
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        proposed = (*selected, record)
        encoded = (
            record
            if len(proposed) == 1
            else _DENIAL_RECORDS_PREFIX
            + json.dumps(proposed, ensure_ascii=True, separators=(",", ":"))
        )
        if len(encoded) > _MAX_DIAGNOSTIC_RESULT_CHARS:
            continue
        selected.append(record)
        if len(selected) == _MAX_DIAGNOSTIC_CANDIDATES:
            break
    if not selected:
        return None
    if len(selected) == 1:
        return selected[0]
    return _DENIAL_RECORDS_PREFIX + json.dumps(selected, ensure_ascii=True, separators=(",", ":"))


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
    priority = _denial_priority(denial)
    if priority is None:
        return None
    incomplete = len(denial) >= _MAX_DIAGNOSTIC_CHARS or message.count("\n") != 1
    return priority, denial[: _MAX_DIAGNOSTIC_CHARS - 1] + "…" if incomplete else denial


def _denial_from_log(
    tag: str,
    started: str,
    deadline: float,
    startup_denials: frozenset[tuple[str, str]] = frozenset(),
) -> str | None:
    """Retry briefly for a tagged denial while the kernel log catches up."""
    limit = min(deadline, time.monotonic() + _DENIAL_LOG_WAIT_SECONDS)
    for attempt in range(_DENIAL_LOG_ATTEMPTS):
        denial = _read_denial_from_log(tag, started, limit, startup_denials)
        if denial is not None:
            return denial
        if time.monotonic() + _DENIAL_LOG_RETRY_SECONDS >= limit:
            break
        if attempt < _DENIAL_LOG_ATTEMPTS - 1:
            time.sleep(_DENIAL_LOG_RETRY_SECONDS)
    return None


def _read_denial_from_log(
    tag: str,
    started: str,
    deadline: float,
    startup_denials: frozenset[tuple[str, str]] = frozenset(),
) -> str | None:
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
        records = []
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
                records.append(denial)
                del records[:-_MAX_DENIAL_RECORDS]
        return _join_denials(records, startup_denials)
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


def _installed_package_layout(
    resolved: Path, spelling: Path | None
) -> tuple[Path, Path, Path] | None:
    """Find a selected package, its store and prefix from a bound alias or Cellar path."""
    if spelling is not None and spelling.is_absolute():
        for alias in spelling.parents:
            if alias.parent.name != "opt" or not alias.is_symlink():
                continue
            target = alias.resolve(strict=True)
            if resolved.is_relative_to(target) and len(target.parts) > 2:
                return target, target.parent.parent, alias.parent.parent
    parts = resolved.parts
    for index in range(len(parts) - 3, -1, -1):
        if parts[index] == "Cellar":
            cellar = Path(*parts[: index + 1])
            return Path(*parts[: index + 3]), cellar, cellar.parent
    return None


class MacOSSeatbeltBackend:
    """Run approved requests through Apple's fixed Seatbelt launcher.

    Prepare an optional separate socket filesystem after core Seatbelt qualification.

    Args:
        diagnostics (SeatbeltDiagnosticService | None): Application-owned diagnostic service,
            or None to create one owned by this backend.
    """

    _diagnostics: SeatbeltDiagnosticService
    _owns_diagnostics: bool
    _enforcement_verified_pid: int | None
    _local_volume_lock: threading.Lock
    _local_volume: LocalSocketVolume | None
    _pending_local_attachment: LocalSocketAttachmentError | None
    _local_volume_users: int
    _local_volume_closed: bool
    capabilities: frozenset[SandboxCapability] = frozenset({SandboxCapability.LOCAL_NETWORK})
    qualification_capabilities: frozenset[SandboxCapability] = frozenset()

    def __init__(self, diagnostics: SeatbeltDiagnosticService | None = None) -> None:
        self._diagnostics = diagnostics or SeatbeltDiagnosticService()
        self._owns_diagnostics = diagnostics is None
        self._enforcement_verified_pid = None
        self._local_volume_lock = threading.Lock()
        self._local_volume = None
        self._pending_local_attachment = None
        self._local_volume_users = 0
        self._local_volume_closed = False

    def qualification_layout(self) -> QualificationProbeLayout:
        """Keep the probe workspace outside protected private temporary ancestry.

        Returns:
            QualificationProbeLayout: Safe workspace parent and adjacent outside canary.
        """
        parent = (
            _PRIVATE_TEMP_ROOT
            if is_protected_external_read(Path(tempfile.gettempdir()).resolve(strict=True))
            else None
        )
        return QualificationProbeLayout(parent, None)

    def qualification_failure(
        self,
        result: CommandProcessResult | None,
        reason: str,
    ) -> str | CommandProcessResult:
        """Preserve invalid native policy failures as typed selection results.

        Args:
            result (CommandProcessResult | None): Native probe result, if execution began.
            reason (str): Sanitized probe failure explanation.

        Returns:
            str | CommandProcessResult: Typed invalid result or ordinary unavailability reason.
        """
        if result is not None and result.outcome in {
            SandboxOutcome.INVALID,
            SandboxOutcome.STALE,
            SandboxOutcome.UNSUPPORTED,
        }:
            return CommandProcessResult(result.outcome, detail=reason)
        return reason

    def capability_failure(self) -> None:
        """Report no preflight failure before native macOS enforcement is probed."""

    def close(self) -> None:
        """Close owned diagnostics and eject the socket volume after its active uses settle.

        Raises:
            OSError: The owned volume cannot be safely verified or ejected.
            subprocess.TimeoutExpired: Native volume inspection or eject exceeds its bound.
        """
        self._enforcement_verified_pid = None
        if self._owns_diagnostics:
            self._diagnostics.close()
        with self._local_volume_lock:
            self._local_volume_closed = True
            if self._local_volume is not None and self._local_volume_users == 0:
                _eject_local_socket_volume(self._local_volume)
                self._local_volume = None
            if self._pending_local_attachment is not None:
                pending = self._pending_local_attachment
                try:
                    _remove_local_socket_attachment(
                        pending.device,
                        pending.marker,
                        time.monotonic() + _LOCAL_SOCKET_VOLUME_SETUP_TIMEOUT_SECONDS,
                    )
                except (OSError, TimeoutError, subprocess.TimeoutExpired) as error:
                    raise pending from error
                self._pending_local_attachment = None

    def _qualify_local_network(self, request: SandboxRequest) -> None:
        """Advertise local IPC only after its volume and native socket exchange work."""
        try:
            with self._local_volume_lock:
                if self._pending_local_attachment is not None:
                    raise self._pending_local_attachment
                try:
                    self._local_volume = _mount_local_socket_volume(
                        time.monotonic() + _LOCAL_SOCKET_VOLUME_SETUP_TIMEOUT_SECONDS
                    )
                except LocalSocketAttachmentError as error:
                    self._pending_local_attachment = error
                    raise
            source = "/usr/bin/perl -MSocket -e " + shlex.quote(
                f'my $path = "$ENV{{{_LOCAL_SOCKET_DIRECTORY_ENVIRONMENT_VARIABLE}}}/probe.sock"; '
                "socket(my $server, PF_UNIX, SOCK_STREAM, 0) or die $!; "
                "bind($server, pack_sockaddr_un($path)) or die $!; "
                "listen($server, 1) or die $!; "
                "socket(my $client, PF_UNIX, SOCK_STREAM, 0) or die $!; "
                "connect($client, pack_sockaddr_un($path)) or die $!; "
                "defined accept(my $peer, $server) or die $!;"
            )
            probe = replace(
                request,
                source=source,
                execution_source=source,
                local_network=True,
                deadline=time.monotonic() + _LOCAL_SOCKET_VOLUME_SETUP_TIMEOUT_SECONDS,
            )
            result = self.qualification_probe(probe)
            if result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0:
                return
        except (OSError, TimeoutError, subprocess.TimeoutExpired):
            pass
        self.capabilities = frozenset()
        with self._local_volume_lock:
            if self._local_volume is not None:
                try:
                    _eject_local_socket_volume(self._local_volume)
                except (OSError, TimeoutError, subprocess.TimeoutExpired):
                    telemetry_audit(
                        "sandbox.local_socket_volume.cleanup_pending",
                        workspace_id=request.workspace_id,
                        attempt_id=request.attempt_id,
                    )
                else:
                    self._local_volume = None

    def _qualify_mediated_egress(self, request: SandboxRequest) -> None:
        """Advertise mediation only after exact-path native socket enforcement works."""
        if SandboxCapability.LOCAL_NETWORK not in self.capabilities:
            return
        scratch_value = request.shell_environment().get(TMPDIR_ENVIRONMENT_VARIABLE)
        if scratch_value is None:
            return
        scratch = Path(scratch_value)
        try:
            with (
                self._local_socket_directory(
                    request, scratch, prefix=f"{constants.APP_NAME}-egress-probe-"
                ) as directory,
                self._local_socket_directory(
                    request, scratch, prefix=f"{constants.APP_NAME}-egress-other-"
                ) as other_directory,
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener,
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as other,
                socket.socket(socket.AF_INET, socket.SOCK_STREAM) as host,
                tempfile.TemporaryDirectory(prefix="le-a-", dir=directory.parent) as alias_root,
            ):
                allowed = directory / "broker.sock"
                forbidden = other_directory / "broker.sock"
                alias_directory = Path(alias_root)
                alias = alias_directory / "a.sock"
                listener.bind(str(allowed))
                listener.listen(1)
                other.bind(str(forbidden))
                other.listen(1)
                host.bind(("127.0.0.1", 0))
                host.listen(1)
                alias.symlink_to(allowed)
                profile = _profile(
                    replace(
                        request,
                        local_network=False,
                        write_roots=(*request.write_roots, alias_directory),
                    ),
                    scratch,
                    "LOOP_SBX_" + secrets.token_hex(16),
                    owned_scratch=False,
                    egress_socket=allowed,
                )
                source = (
                    "use Socket; socket(my $client, PF_UNIX, SOCK_STREAM, 0) or die $!; "
                    f"connect($client, pack_sockaddr_un({_literal(allowed)})) or die $!; "
                    "socket(my $other, PF_UNIX, SOCK_STREAM, 0) or die $!; "
                    f"connect($other, pack_sockaddr_un({_literal(forbidden)})) "
                    "and die 'cross-use'; "
                    "($!{EPERM} || $!{EACCES}) or die 'cross-use uncertain'; "
                    "socket(my $alias, PF_UNIX, SOCK_STREAM, 0) or die $!; "
                    f"connect($alias, pack_sockaddr_un({_literal(alias)})) and die 'alias'; "
                    "($!{EPERM} || $!{EACCES}) or die 'alias uncertain'; "
                    f"unlink({_literal(allowed)}) and die 'unlink'; "
                    "($!{EPERM} || $!{EACCES}) or die 'unlink uncertain'; "
                    "if (socket(my $ip, PF_INET, SOCK_STREAM, 0)) { "
                    f"connect($ip, pack_sockaddr_in({host.getsockname()[1]}, "
                    "inet_aton('127.0.0.1'))) and die 'host-loopback'; "
                    "($!{EPERM} || $!{EACCES}) or die 'host-loopback uncertain'; "
                    "} elsif (!$!{EPERM} && !$!{EACCES}) { die $!; } "
                    "if (socket(my $direct, PF_INET, SOCK_DGRAM, 0)) { "
                    "connect($direct, pack_sockaddr_in(53, inet_aton('1.1.1.1'))) "
                    "and die 'direct-dns'; "
                    "($!{EPERM} || $!{EACCES}) or die 'direct-dns uncertain'; "
                    "} elsif (!$!{EPERM} && !$!{EACCES}) { die $!; }"
                )
                result = subprocess.run(
                    [_LAUNCHER, "-p", profile, "/usr/bin/perl", "-MSocket", "-e", source],
                    cwd=request.cwd,
                    env=request.shell_environment(),
                    capture_output=True,
                    timeout=_LOCAL_SOCKET_VOLUME_SETUP_TIMEOUT_SECONDS,
                    check=False,
                )
                if result.returncode == 0:
                    self.capabilities = self.capabilities | {SandboxCapability.MEDIATED_EGRESS}
        except (OSError, ValueError, TimeoutError, subprocess.TimeoutExpired):
            return

    def _qualify_terminal(self, request: SandboxRequest) -> None:
        """Advertise PTYs only after native ownership and inner-allocation controls pass."""
        try:
            with _owned_terminal() as foreign:
                source = (
                    "set -e; test -t 0; test -t 1; : </dev/tty; "
                    "/usr/bin/script -q /dev/null /bin/sh -c "
                    + shlex.quote("test -t 0 && test -t 1 && printf INNER_PTY_READY")
                    + "; "
                    f"if ( : < {shlex.quote(str(foreign.path))} ) 2>/dev/null; then exit 9; fi; "
                    f"if ( : > {shlex.quote(str(foreign.path))} ) 2>/dev/null; then exit 10; fi"
                )
                probe = replace(
                    request,
                    source=source,
                    execution_source=source,
                    terminal=True,
                    deadline=time.monotonic() + _LOCAL_SOCKET_VOLUME_SETUP_TIMEOUT_SECONDS,
                )
                result = self._run(
                    probe,
                    cancellation_event=None,
                    qualification=True,
                    terminal_input=lambda: None,
                )
                if (
                    result.outcome is SandboxOutcome.COMPLETED
                    and result.exit_code == 0
                    and "INNER_PTY_READY" in result.stdout
                ):
                    self.capabilities = self.capabilities | {SandboxCapability.TERMINAL}
        except (OSError, ValueError, TimeoutError, subprocess.TimeoutExpired):
            return

    @contextmanager
    def _local_socket_directory(
        self,
        request: SandboxRequest,
        scratch: Path,
        *,
        prefix: str = f"{constants.APP_NAME}-local-",
    ) -> Generator[Path, None, None]:
        """Lease one isolated filesystem directory for a sandbox attempt."""
        with self._local_volume_lock:
            if self._local_volume_closed:
                raise OSError("The macOS local socket filesystem is closed.")
            if self._pending_local_attachment is not None:
                raise self._pending_local_attachment
            if self._local_volume is None:
                try:
                    self._local_volume = _mount_local_socket_volume(request.deadline)
                except LocalSocketAttachmentError as error:
                    self._pending_local_attachment = error
                    raise
            volume = self._local_volume
            if volume.mount.stat().st_dev != volume.device_id or not volume.mount.is_dir():
                raise OSError("The macOS local socket filesystem changed.")
            if any(
                volume.mount.is_relative_to(root) or root.stat().st_dev == volume.device_id
                for root in (*request.write_roots, scratch)
            ):
                raise OSError("A writable command root shares the local socket filesystem.")
            self._local_volume_users += 1
        try:
            with tempfile.TemporaryDirectory(prefix=prefix, dir=volume.mount) as directory:
                yield Path(directory).resolve(strict=True)
        finally:
            with self._local_volume_lock:
                self._local_volume_users -= 1
                if (
                    self._local_volume is not None
                    and self._local_volume_closed
                    and not self._local_volume_users
                ):
                    _eject_local_socket_volume(self._local_volume)
                    self._local_volume = None

    def classify_denial_candidates(self, observed: str) -> tuple[tuple[tuple[str, bool], ...], str]:
        """Identify observations also seen in a successful fixed startup probe.

        Args:
            observed (str): Bounded denial records from one command attempt.

        Returns:
            tuple[tuple[tuple[str, bool], ...], str]: Records and startup-match flags,
                followed by observed or not-observed startup status.
        """
        baseline = self._diagnostics.startup_denials()
        records = tuple(
            (
                record,
                (fingerprint := _denial_fingerprint(record)) is not None
                and fingerprint in baseline,
            )
            for record in _denial_records(observed)
        )
        return tuple(sorted(records, key=lambda item: item[1])), (
            "observed" if baseline else "not_observed"
        )

    @property
    def trusted_file_owner_ids(self) -> tuple[int, ...]:
        """Return file owners trusted for automatic executable binding.

        Returns:
            tuple[int, ...]: Current user and any verified host administrator identity.
        """
        return tuple(dict.fromkeys((0, os.getuid())))

    def trusted_system_path(self, path: Path, *, directory: bool = False) -> bool:
        """Authenticate a root-owned system path on a read-only filesystem.

        Args:
            path (Path): Absolute system path to validate without following aliases.
            directory (bool): Require a directory instead of a regular executable file.

        Returns:
            bool: Whether the path and its ancestry are root-owned and read-only.

        Raises:
            OSError: System path or filesystem metadata cannot be inspected.
        """
        del self
        if not path.is_absolute() or path.resolve(strict=True) != path:
            return False
        for entry in (path, *path.parents):
            details = entry.lstat()
            if (
                stat.S_ISLNK(details.st_mode)
                or details.st_uid != 0
                or details.st_mode & constants.GROUP_AND_OTHER_WRITE_MODE_MASK
                or not os.statvfs(entry).f_flag & os.ST_RDONLY
                or entry != path
                and not stat.S_ISDIR(details.st_mode)
            ):
                return False
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            if stat.S_ISDIR(details.st_mode):
                flags |= os.O_DIRECTORY
            descriptor = os.open(entry, flags)
            try:
                bound = os.fstat(descriptor)
                latest = entry.lstat()
                if (
                    (bound.st_dev, bound.st_ino, bound.st_mode, bound.st_uid, bound.st_gid)
                    != (
                        details.st_dev,
                        details.st_ino,
                        details.st_mode,
                        details.st_uid,
                        details.st_gid,
                    )
                    or has_extended_access_acl(descriptor)
                    or (latest.st_dev, latest.st_ino, latest.st_mode, latest.st_uid, latest.st_gid)
                    != (
                        details.st_dev,
                        details.st_ino,
                        details.st_mode,
                        details.st_uid,
                        details.st_gid,
                    )
                ):
                    return False
            finally:
                os.close(descriptor)
        selected = path.lstat()
        return stat.S_ISDIR(selected.st_mode) if directory else stat.S_ISREG(selected.st_mode)

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
        return f"{constants.APP_NAME}-seatbelt-"

    def automatic_tool_read_roots(
        self,
        workspace: Path,
        path_roots: tuple[tuple[Path, Path], ...],
        executable_identities: tuple[tuple[str, Path, Path, int, int], ...],
    ) -> tuple[Path, ...]:
        """Return no extra runtime roots beyond macOS's existing profile defaults.

        Args:
            workspace (Path): Canonical selected workspace.
            path_roots (tuple[tuple[Path, Path], ...]): Bound PATH spellings and directories.
            executable_identities (tuple[tuple[str, Path, Path, int, int], ...]): Bound named
                executable files and their device/inode identities.

        Returns:
            tuple[Path, ...]: Empty tuple; Seatbelt compatibility is expressed in its profile.
        """
        del self, workspace, path_roots, executable_identities
        return ()

    @staticmethod
    def installed_package_root(root: Path) -> bool:
        """Check whether a candidate package root has an ordinary installation receipt.

        Args:
            root (Path): Canonical candidate package directory.

        Returns:
            bool: Whether the root has a single-link regular receipt outside private paths.

        Raises:
            OSError: If the package receipt cannot be inspected.
        """
        if is_protected_external_read(root):
            return False
        try:
            receipt = (root / "INSTALL_RECEIPT.json").lstat()
        except FileNotFoundError:
            return False
        return stat.S_ISREG(receipt.st_mode) and receipt.st_nlink == 1

    @staticmethod
    def managed_tool_root(root: Path) -> bool:
        """Verify a versioned Homebrew package root with an installation receipt.

        Args:
            root (Path): Canonical candidate package directory.

        Returns:
            bool: Whether the root qualifies for automatic read authority.

        Raises:
            OSError: If the package receipt cannot be inspected.
        """
        return any(
            root.is_relative_to(prefix) for prefix in _MANAGED_TOOLCHAIN_PREFIXES
        ) and MacOSSeatbeltBackend.installed_package_root(root)

    @staticmethod
    def installed_tool_roots(
        resolved: Path,
    ) -> tuple[tuple[Path, ...], tuple[tuple[Path, int, int, Path], ...]]:
        """Select the installed package roots required by a host executable.

        Args:
            resolved (Path): Canonical executable selected from the host PATH.

        Returns:
            tuple[tuple[Path, ...], tuple[tuple[Path, int, int, Path], ...]]: Read roots and
                alias identity bindings for the executable's installation.

        Raises:
            ValueError: If installed package metadata is invalid.
            OSError: If the installation receipt cannot be read.
        """
        installation = MacOSSeatbeltBackend.installed_runtime_roots(resolved)
        if installation is None:
            raise ValueError("Installed-tool dependency metadata is invalid or unavailable.")
        return installation

    @staticmethod
    def installed_runtime_roots(
        resolved: Path,
        *,
        spelling: Path | None = None,
        workspace: Path | None = None,
    ) -> tuple[tuple[Path, ...], tuple[tuple[Path, int, int, Path], ...]] | None:
        """Select a narrow versioned package and bind its declared support aliases.

        Args:
            resolved (Path): Canonical executable selected from the host PATH.
            spelling (Path | None): Selected executable spelling used to identify its prefix,
                or None to inspect the canonical package layout alone.
            workspace (Path | None): Workspace that inferred package roots must not contain,
                or None for an explicit installed-tool lookup.

        Returns:
            tuple[tuple[Path, ...], tuple[tuple[Path, int, int, Path], ...]] | None: Read roots
                and alias identity bindings, or None when package metadata is invalid.

        Raises:
            OSError: If the installation receipt cannot be read.
        """
        if is_protected_external_read(resolved):
            return None
        roots = [resolved.parent]
        opt_aliases = []
        layout = _installed_package_layout(resolved, spelling)
        if layout is not None:
            root, cellar, prefix = layout
            home = _trusted_home() if workspace is not None else None
            if workspace is not None and (
                workspace.is_relative_to(root) or home.is_relative_to(root)
            ):
                return None
            roots = [root]
            receipt = root / "INSTALL_RECEIPT.json"
            try:
                receipt_details = receipt.lstat()
            except FileNotFoundError:
                receipt_details = None
            if receipt_details is not None:
                if (
                    not stat.S_ISREG(receipt_details.st_mode)
                    or receipt_details.st_nlink != 1
                    or is_protected_external_read(receipt)
                ):
                    return None
                # Descriptor-relative opens otherwise hide the original path from audit hooks.
                sys.audit("open", str(receipt), "r", os.O_RDONLY | os.O_CLOEXEC)
                descriptor = open_selected_file(
                    Path("/"), receipt, (receipt_details.st_dev, receipt_details.st_ino)
                )
                try:
                    before = os.fstat(descriptor)
                    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                        return None
                    content = os.read(descriptor, _MAX_RECEIPT_BYTES + 1)
                    after = os.fstat(descriptor)
                finally:
                    os.close(descriptor)
                if len(content) > _MAX_RECEIPT_BYTES:
                    return None
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    return None
                try:
                    dependencies = json.loads(content).get("runtime_dependencies", [])
                except (TypeError, AttributeError, json.JSONDecodeError, UnicodeDecodeError):
                    return None
                if (
                    not isinstance(dependencies, list)
                    or len(dependencies) > _MAX_INSTALLED_TOOL_DEPENDENCIES
                ):
                    return None
                for dependency in dependencies:
                    if not isinstance(dependency, dict):
                        return None
                    name_value = dependency.get("full_name")
                    if (
                        not isinstance(name_value, str)
                        or len(name_value) > constants.MAX_COMMAND_PATH_LENGTH
                        or re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9@+._-]+$", name_value) is None
                    ):
                        return None
                    alias = prefix / "opt" / name_value
                    if not alias.is_symlink() or not alias.exists():
                        return None
                    target = alias.resolve(strict=True)
                    if target.parent != cellar / name_value or (
                        workspace is not None
                        and (workspace.is_relative_to(target) or home.is_relative_to(target))
                    ):
                        return None
                    roots.append(target)
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
        """Run fixed argv with selected content and verified installed-tool runtime reads.

        Args:
            executable (Path): Canonical installed helper to copy into private state.
            arguments (tuple[str, ...]): Fixed-argv helper arguments with selected fd paths.
            descriptors (tuple[int, ...]): Already validated selected file descriptors.
            deadline (float): Absolute monotonic completion deadline.

        Returns:
            ProcessCapture | None: Bounded native output, including a typed timeout, or None
                if native execution is unavailable or preparation fails closed.

        Raises:
            OSError: If installed-tool dependency metadata cannot be read.
        """
        if platform.system() != "Darwin" or not is_finite_deadline(deadline):
            return None
        if time.monotonic() >= deadline:
            return ProcessCapture(ProcessCaptureStatus.TIMED_OUT)
        system_directory = self._diagnostics.system_directory(deadline)
        installation = self.installed_runtime_roots(executable)
        if installation is None:
            return None
        tool_roots, tool_aliases = installation
        if not all(self.managed_tool_root(root) for root in tool_roots):
            tool_roots, tool_aliases = (), ()
        try:
            workspace_parent = (
                _PRIVATE_TEMP_ROOT
                if is_protected_external_read(Path(tempfile.gettempdir()).resolve(strict=True))
                else None
            )
            with (
                tempfile.TemporaryDirectory(
                    prefix=self.scratch_prefix, dir=workspace_parent
                ) as directory,
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
                helper.chmod(executable_details.st_mode & constants.FILE_MODE_PERMISSION_MASK)
                request = SandboxRequest.create(
                    source="read-only argv",
                    cwd=workspace,
                    workspace=workspace,
                    read_roots=tool_roots,
                    read_aliases=tuple(alias for alias, _, _, _ in tool_aliases),
                    write_roots=(),
                    network=False,
                    environment={
                        PATH_ENVIRONMENT_VARIABLE: MINIMAL_SYSTEM_PATH,
                        LANG_ENVIRONMENT_VARIABLE: SYSTEM_LOCALE,
                        TMPDIR_ENVIRONMENT_VARIABLE: str(scratch),
                    },
                    policy_version=self.policy_version,
                    deadline=deadline,
                    workspace_id="read-only-argv",
                    scratch_root=scratch,
                    scratch_data=True,
                )
                for root in (workspace, *tool_roots):
                    _walk_granted_root(root, deadline)
                if not request.paths_are_current():
                    return None
                _check_deadline(deadline)
                profile = _profile(
                    request,
                    scratch,
                    "LOOP_SBX_" + secrets.token_hex(16),
                    system_directory=system_directory,
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

    def run(
        self,
        request: SandboxRequest,
        *,
        cancellation_event: threading.Event | None = None,
        output_observer: Callable[[str, str], None] | None = None,
        terminal_input: Callable[[], bytes | None] | None = None,
        policy_authority: PermissionManager | None = None,
        approved_revision: int | None = None,
    ) -> CommandProcessResult:
        """Run one bound shell request, failing closed on preparation or launch errors.

        Args:
            request (SandboxRequest): Approved shell source and operating-system authority.
            cancellation_event (threading.Event | None): Optional signal preventing launch when
                observed before spawn, or requesting cleanup after launch.
            output_observer (Callable[[str, str], None] | None): Optional observer for bounded
                output read from the verified command process.
            terminal_input (Callable[[], bytes | None] | None): Next bounded transient control
                frame for an owned interactive terminal.
            policy_authority (PermissionManager | None): Active mediator authority, when requested.
            approved_revision (int | None): Exact revision captured with mediated approval.

        Returns:
            CommandProcessResult: Bounded output and exit status or a typed boundary failure.
        """
        if request.terminal != (terminal_input is not None):
            return CommandProcessResult(
                SandboxOutcome.INVALID,
                detail="Interactive terminal request and control channel disagree.",
            )
        if request.terminal and SandboxCapability.TERMINAL not in self.capabilities:
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail="Owned interactive terminals are not qualified on this macOS host.",
            )
        if request.terminal and request.egress_origins:
            return CommandProcessResult(
                SandboxOutcome.UNSUPPORTED,
                detail="Interactive terminal and mediated egress cannot share this attempt.",
            )
        if request.egress_origins and (policy_authority is None or approved_revision is None):
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail="Mediated egress requires active policy authority.",
            )
        return self._run(
            request,
            cancellation_event=cancellation_event,
            qualification=False,
            output_observer=output_observer,
            terminal_input=terminal_input,
            policy_authority=policy_authority,
            approved_revision=approved_revision,
        )

    def qualification_probe(self, request: SandboxRequest) -> CommandProcessResult:
        """Run a controlled enforcement probe without requiring diagnostics.

        Args:
            request (SandboxRequest): Owned probe request with observable denied effects.

        Returns:
            CommandProcessResult: Native result for independent enforcement verification.
        """
        return self._run(request, cancellation_event=None, qualification=True)

    def complete_qualification(self, request: SandboxRequest) -> None:
        """Qualify optional native features after independent enforcement verification.

        Args:
            request (SandboxRequest): Owned probe request from the verified native selector.
        """
        self._enforcement_verified_pid = os.getpid()
        self._qualify_local_network(request)
        self._qualify_mediated_egress(request)
        self._qualify_terminal(request)

    def _run(
        self,
        request: SandboxRequest,
        *,
        cancellation_event: threading.Event | None,
        qualification: bool,
        output_observer: Callable[[str, str], None] | None = None,
        terminal_input: Callable[[], bytes | None] | None = None,
        policy_authority: PermissionManager | None = None,
        approved_revision: int | None = None,
    ) -> CommandProcessResult:
        """Launch an approved request or the selector's controlled qualification probe."""
        if platform.system() != "Darwin" or request.policy_version != _POLICY_VERSION:
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail="Unsupported macOS policy.",
                failure_context="No compatible native sandbox policy is available.",
            )
        if (
            request.local_network
            and not request.network
            and SandboxCapability.LOCAL_NETWORK not in self.capabilities
        ):
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail="Attempt-local Unix sockets are not qualified on this macOS host.",
            )
        if request.egress_origins and SandboxCapability.MEDIATED_EGRESS not in self.capabilities:
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail="Attempt-private mediated egress is not qualified on this macOS host.",
            )
        if cancellation_event is not None and cancellation_event.is_set():
            return CommandProcessResult(SandboxOutcome.CANCELLED)
        execution_deadline = request.deadline
        if request.runtime_timeout is not None:
            request = replace(
                request,
                deadline=execution_deadline - request.runtime_timeout,
            )
        if failure := _request_failure(request):
            return failure
        possible_effects = False
        try:
            bound_scratch = request.shell_environment().get(TMPDIR_ENVIRONMENT_VARIABLE)
            scratch_context = (
                nullcontext(bound_scratch)
                if bound_scratch is not None
                else tempfile.TemporaryDirectory(prefix=self.scratch_prefix)
            )
            with scratch_context as directory, ExitStack() as attempt_stack:
                scratch = Path(directory).resolve(strict=True)
                details = scratch.lstat()
                if (
                    not stat.S_ISDIR(details.st_mode)
                    or details.st_uid != os.getuid()
                    or stat.S_IMODE(details.st_mode) != constants.PRIVATE_DIRECTORY_MODE
                    or (bound_scratch is not None and str(scratch) != directory)
                    or (
                        not request.session_scratch
                        and not scratch.name.startswith(self.scratch_prefix)
                    )
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
                        if root in (request.workspace, scratch, *request.write_roots)
                        else None,
                        protection=request.protected_paths if root == request.workspace else None,
                        ordinary_names=request.scratch_data and root == scratch,
                    )
                guarded_anchors = tuple(sorted(rename_anchors))
                environment = request.shell_environment()
                environment[TMPDIR_ENVIRONMENT_VARIABLE] = str(scratch)
                terminal = (
                    attempt_stack.enter_context(_owned_terminal()) if request.terminal else None
                )
                local_socket_directory = None
                if request.local_network and not request.network:
                    local_socket_directory = attempt_stack.enter_context(
                        self._local_socket_directory(request, scratch)
                    )
                    _walk_granted_root(local_socket_directory, request.deadline)
                    environment[_LOCAL_SOCKET_DIRECTORY_ENVIRONMENT_VARIABLE] = str(
                        local_socket_directory
                    )
                egress_socket = None
                broker = None
                if request.egress_origins:
                    if policy_authority is None or approved_revision is None:
                        raise ValueError("Mediated egress authority is unavailable.")
                    directory = attempt_stack.enter_context(
                        self._local_socket_directory(
                            request, scratch, prefix=f"{constants.APP_NAME}-egress-"
                        )
                    )
                    egress_socket = directory / "broker.sock"
                    broker = attempt_stack.enter_context(
                        MacOSEgressBroker(
                            request, egress_socket, policy_authority, approved_revision
                        )
                    )
                    proxy = f"socks5h://localhost{egress_socket}"
                    for key in (
                        "HTTP_PROXY",
                        "HTTPS_PROXY",
                        "ALL_PROXY",
                        "http_proxy",
                        "https_proxy",
                        "all_proxy",
                    ):
                        environment[key] = proxy
                    environment["NO_PROXY"] = ""
                    environment["no_proxy"] = ""
                if failure := _request_failure(request):
                    return failure
                try:
                    monitor = self._diagnostics.monitor(request.deadline)
                except OSError as exc:
                    return CommandProcessResult(
                        SandboxOutcome.INVALID,
                        detail=str(exc),
                        failure_context="Native sandbox diagnostic setup was refused.",
                    )
                system_directory = self._diagnostics.system_directory(request.deadline)
                if not self._diagnostics.parent_verified() or (
                    monitor is not None and self._diagnostics.directory_probe_pending()
                ):
                    probe_tag = "LOOP_SBX_" + secrets.token_hex(16)
                    compatibility_files = _python_environment_configs(request)
                    profile = _profile(
                        request,
                        scratch,
                        probe_tag,
                        compatibility_files,
                        guarded_anchors,
                        owned_scratch=bound_scratch is None,
                        local_socket_directory=local_socket_directory,
                        egress_socket=egress_socket,
                    )
                    try:
                        registered = (
                            monitor is not None and monitor.register(probe_tag) is not False
                        )
                    except OSError as exc:
                        return CommandProcessResult(
                            SandboxOutcome.INVALID,
                            detail=str(exc),
                            failure_context="Native sandbox diagnostic setup was refused.",
                        )
                    try:
                        probe = subprocess.run(
                            [_LAUNCHER, "-p", profile, "/usr/bin/true"],
                            cwd="/",
                            env={
                                PATH_ENVIRONMENT_VARIABLE: MINIMAL_SYSTEM_PATH,
                                LANG_ENVIRONMENT_VARIABLE: SYSTEM_LOCALE,
                                TMPDIR_ENVIRONMENT_VARIABLE: str(scratch),
                            },
                            capture_output=True,
                            text=True,
                            close_fds=True,
                            timeout=max(
                                _MIN_PROBE_TIMEOUT_SECONDS, request.deadline - time.monotonic()
                            ),
                            check=False,
                        )
                        if registered and probe.returncode == 0:
                            probe_denials = monitor.denial(
                                probe_tag, request.deadline, all_records=True
                            )
                            fingerprints = monitor.fingerprints(probe_tag)
                            self._diagnostics.observe_startup_denial(
                                probe_denials,
                                fingerprints
                                if isinstance(fingerprints, frozenset)
                                else frozenset(),
                            )
                            if isinstance(probe_denials, str):
                                for record in _denial_records(probe_denials):
                                    identity = _probe_system_directory(record, request.deadline)
                                    if identity is not None:
                                        self._diagnostics.set_system_directory(identity)
                                        break
                    finally:
                        if registered:
                            monitor.unregister(probe_tag)
                    if probe.returncode:
                        return CommandProcessResult(
                            SandboxOutcome.UNAVAILABLE,
                            detail=f"Seatbelt launch check failed (exit {probe.returncode}): "
                            f"{probe.stderr.strip()[:_MAX_DIAGNOSTIC_CHARS]}",
                            failure_context="Native sandbox launch check failed.",
                        )
                    self._diagnostics.mark_parent_verified(directory_probe_attempted=registered)
                    system_directory = self._diagnostics.system_directory(request.deadline)
                if failure := _request_failure(request):
                    return failure
                tag = "LOOP_SBX_" + secrets.token_hex(16)
                compatibility_files = _python_environment_configs(request)
                profile = _profile(
                    request,
                    scratch,
                    tag,
                    compatibility_files,
                    guarded_anchors,
                    owned_scratch=bound_scratch is None,
                    system_directory=system_directory,
                    local_socket_directory=local_socket_directory,
                    egress_socket=egress_socket,
                    terminal_slave=terminal.path if terminal is not None else None,
                )
                try:
                    if monitor is not None and monitor.register(tag) is False:
                        monitor = None
                except OSError as exc:
                    return CommandProcessResult(
                        SandboxOutcome.INVALID,
                        detail=str(exc),
                        failure_context="Native sandbox diagnostic setup was refused.",
                    )
                if (
                    monitor is None
                    and not qualification
                    and self._enforcement_verified_pid != os.getpid()
                ):
                    return CommandProcessResult(
                        SandboxOutcome.INVALID,
                        detail="Seatbelt denial stream is unavailable.",
                        failure_context="Native sandbox diagnostic setup failed.",
                    )
                try:
                    if failure := _request_failure(request):
                        return failure
                    _exact_executable_reads(request)
                    possible_effects = True
                    result = self._execute(
                        replace(request, deadline=execution_deadline)
                        if request.runtime_timeout is not None
                        else request,
                        profile,
                        environment,
                        tag,
                        monitor,
                        broker=broker,
                        startup_denials=self._diagnostics.startup_denials(),
                        cancellation_event=cancellation_event,
                        terminal=terminal,
                        terminal_input=terminal_input,
                        **({"output_observer": output_observer} if output_observer else {}),
                    )
                    if broker is not None and broker.interrupted:
                        return replace(
                            result,
                            outcome=SandboxOutcome.UNAVAILABLE,
                            exit_code=None,
                            detail="Mediated egress authority ended during this command.",
                            possible_effects=True,
                        )
                    return result
                finally:
                    if monitor is not None:
                        monitor.unregister(tag)
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            return CommandProcessResult(
                SandboxOutcome.TIMED_OUT, detail=str(exc), possible_effects=possible_effects
            )
        except ValueError as exc:
            return CommandProcessResult(
                SandboxOutcome.INVALID, detail=str(exc), possible_effects=possible_effects
            )
        except OSError as exc:
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail=str(exc),
                failure_context="Native sandbox preparation or process launch failed.",
                possible_effects=possible_effects,
            )
        except KeyboardInterrupt:
            return CommandProcessResult(
                SandboxOutcome.CANCELLED,
                detail="Command cancelled.",
                possible_effects=possible_effects,
            )

    @staticmethod
    def _execute(
        request: SandboxRequest,
        profile: str,
        environment: dict[str, str],
        tag: str,
        monitor: DenialMonitor | None,
        *,
        broker: MacOSEgressBroker | None = None,
        startup_denials: frozenset[tuple[str, str]] = frozenset(),
        cancellation_event: threading.Event | None = None,
        output_observer: Callable[[str, str], None] | None = None,
        terminal: OwnedTerminal | None = None,
        terminal_input: Callable[[], bytes | None] | None = None,
    ) -> CommandProcessResult:
        """Capture both child streams with bounded storage and a single deadline."""
        diagnostic_status = "unavailable" if monitor is None else "unknown"
        if cancellation_event is not None and cancellation_event.is_set():
            return CommandProcessResult(
                SandboxOutcome.CANCELLED, diagnostic_status=diagnostic_status
            )
        started = time.strftime("%Y-%m-%d %H:%M:%S%z", time.localtime())
        telemetry_audit(
            "sandbox.launching",
            workspace_id=request.workspace_id,
            attempt_id=request.attempt_id,
        )
        if cancellation_event is not None and cancellation_event.is_set():
            return CommandProcessResult(
                SandboxOutcome.CANCELLED, diagnostic_status=diagnostic_status
            )
        if (
            request.runtime_timeout is not None
            and time.monotonic() >= request.deadline - request.runtime_timeout
        ):
            return CommandProcessResult(
                SandboxOutcome.TIMED_OUT,
                detail="Native command preparation exceeded its deadline.",
                diagnostic_status=diagnostic_status,
            )
        if terminal is not None:
            process = _spawn_terminal(request, profile, environment, terminal)
        else:
            process = subprocess.Popen(
                [
                    _LAUNCHER,
                    "-p",
                    profile,
                    constants.SHELL_COMMAND_PATH,
                    constants.SHELL_COMMAND_OPTION,
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
        if request.runtime_timeout is not None:
            request = replace(
                request,
                deadline=min(request.deadline, time.monotonic() + request.runtime_timeout),
            )
        if broker is not None and not broker.bind_runtime_deadline(request.deadline):
            supervise_process(
                process,
                time.monotonic(),
                terminate=kill_process_group,
                reader=read_bounded_stream,
                thread_factory=threading.Thread,
            )
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail="Mediated egress lease expired during native command launch.",
                possible_effects=True,
            )
        try:
            telemetry_audit(
                "sandbox.launched",
                workspace_id=request.workspace_id,
                attempt_id=request.attempt_id,
            )
        except BaseException:
            supervise_process(
                process,
                time.monotonic(),
                terminate=kill_process_group,
                reader=read_bounded_stream,
                thread_factory=threading.Thread,
            )
            raise
        pending_marker = ""
        pending_stderr: list[str] = []
        pending_stderr_chars = 0
        marker_checked = False
        marker_valid = False
        output_lock = threading.Lock()
        terminal_ready = threading.Event()
        shell_marker = (
            _SHELL_STARTED.replace("\n", "\r\n") if terminal is not None else _SHELL_STARTED
        )

        def observe_output(stream: str, chunk: str) -> None:
            """Publish both streams only after authenticating the shell entry marker."""
            nonlocal pending_marker, pending_stderr_chars, marker_checked, marker_valid
            with output_lock:
                if not marker_checked:
                    if stream == "stderr":
                        retained = chunk[
                            : max(0, constants.MAX_OUTPUT_CHARS - pending_stderr_chars)
                        ]
                        if retained:
                            pending_stderr.append(retained)
                            pending_stderr_chars += len(retained)
                        return
                    pending_marker += chunk
                    if len(pending_marker) < len(shell_marker):
                        return
                    marker_checked = True
                    marker_valid = pending_marker.startswith(shell_marker)
                    if not marker_valid:
                        pending_stderr.clear()
                        return
                    terminal_ready.set()
                    chunk = pending_marker[len(shell_marker) :]
                    pending_marker = ""
                    if output_observer is not None:
                        for pending in pending_stderr:
                            output_observer("stderr", pending)
                    pending_stderr.clear()
                if marker_valid and chunk and output_observer is not None:
                    output_observer(stream, chunk)

        terminal_stop = threading.Event()
        terminal_lock = threading.Lock()
        terminal_failures: list[BaseException] = []

        def drive_terminal() -> None:
            """Drain transient controls without exposing the trusted master descriptor."""
            try:
                while not terminal_stop.is_set():
                    if not terminal_ready.is_set():
                        terminal_stop.wait(0.02)
                        continue
                    frame = terminal_input() if terminal_input is not None else None
                    if frame is None:
                        terminal_stop.wait(0.02)
                    else:
                        with terminal_lock:
                            if terminal_stop.is_set():
                                return
                            _write_terminal_control(terminal.master, frame)
            except BaseException as exc:  # noqa: BLE001 - report lost terminal control.
                terminal_failures.append(exc)
                terminal_stop.set()

        controller = (
            threading.Thread(target=drive_terminal, daemon=True) if terminal is not None else None
        )

        def read_stream(stream, chunks, *, maximum):
            """Treat a closed PTY slave as EOF without hiding pipe failures."""
            return read_bounded_stream(
                stream,
                chunks,
                maximum=maximum,
                eof_errno=errno.EIO if terminal is not None and stream is process.stdout else None,
            )

        try:
            if controller is not None:
                controller.start()
            capture = supervise_process(
                process,
                request.deadline,
                stdout_limit=constants.MAX_OUTPUT_CHARS + len(shell_marker),
                terminate=kill_process_group,
                reader=read_stream if terminal is not None else read_bounded_stream,
                thread_factory=threading.Thread,
                cancel_event=cancellation_event,
                preserve_newlines=True,
                **({"output_observer": observe_output} if output_observer or terminal else {}),
            )
        finally:
            terminal_stop.set()
            if controller is not None:
                controller.join(constants.COMMAND_CLEANUP_WAIT_SECONDS)
                # Synchronize with an in-flight write before its master may be closed.
                with terminal_lock:
                    pass
                if controller.is_alive():
                    terminal_failures.append(
                        TimeoutError("Interactive terminal controller did not stop.")
                    )
        if terminal_failures:
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail="Interactive terminal control failed.",
                possible_effects=True,
                diagnostic_status=diagnostic_status,
            )
        if capture.status is ProcessCaptureStatus.UNAVAILABLE:
            incomplete = capture.stdout_discarded is None or capture.stderr_discarded is None
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                stdout=capture.stdout.removeprefix(shell_marker),
                stderr=capture.stderr,
                stdout_discarded=capture.stdout_discarded,
                stderr_discarded=capture.stderr_discarded,
                detail="Command output capture failed."
                if incomplete
                else "Child pipes unavailable.",
                failure_context="Native command output could not be fully captured."
                if incomplete
                else "Native command output pipes could not be initialized.",
                possible_effects=True,
                diagnostic_status=diagnostic_status,
            )
        if capture.status is not ProcessCaptureStatus.COMPLETED:
            return CommandProcessResult(
                SandboxOutcome(capture.status.value),
                stdout=capture.stdout.removeprefix(shell_marker),
                stderr=capture.stderr,
                stdout_discarded=capture.stdout_discarded,
                stderr_discarded=capture.stderr_discarded,
                possible_effects=True,
                diagnostic_status=diagnostic_status,
            )
        started_shell = capture.stdout.startswith(shell_marker)
        if not started_shell:
            return CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                stdout=capture.stdout,
                stderr=capture.stderr,
                stdout_discarded=capture.stdout_discarded,
                stderr_discarded=capture.stderr_discarded,
                detail=f"Sandboxed command entry could not be verified (exit {capture.exit_code}).",
                failure_context="Native command entry could not be verified.",
                possible_effects=True,
                diagnostic_status=diagnostic_status,
            )
        observed_denial = None
        if monitor is not None:
            live = monitor.denial(
                tag,
                request.deadline if capture.exit_code else time.monotonic(),
                all_records=True,
                startup_denials=startup_denials,
            )
            if not capture.exit_code:
                snapshot = None
                live = live if isinstance(live, str) else ""
            elif isinstance(live, str) and live:
                snapshot = _read_denial_from_log(
                    tag,
                    started,
                    min(request.deadline, time.monotonic() + _DENIAL_RECORD_WAIT_SECONDS),
                    startup_denials,
                )
            else:
                live = ""
                snapshot = _denial_from_log(tag, started, request.deadline, startup_denials)
            records = [
                (priority, record)
                for record in (
                    *(_denial_records(live) if live else ()),
                    *(_denial_records(snapshot) if snapshot else ()),
                )
                if (priority := _denial_priority(record)) is not None
            ]
            observed_denial = _join_denials(records, startup_denials)
        return CommandProcessResult(
            SandboxOutcome.COMPLETED,
            exit_code=capture.exit_code,
            stdout=capture.stdout[len(shell_marker) :],
            stderr=capture.stderr,
            stdout_discarded=capture.stdout_discarded,
            stderr_discarded=capture.stderr_discarded,
            observed_denial=observed_denial or "",
            diagnostic_status=(
                "unavailable" if monitor is None else "available" if observed_denial else "unknown"
            ),
            diagnostic_source="macos-kernel-log" if observed_denial else "unknown",
        )


def create_backend() -> MacOSSeatbeltBackend:
    """Construct the selected macOS native backend.

    Returns:
        MacOSSeatbeltBackend: Backend for the host's Seatbelt confinement.
    """
    return MacOSSeatbeltBackend()


def management_commands(
    main_app: bool,
    refresh_backend: Callable[[], None] | None,
) -> tuple[CommandRegistration, ...]:
    """Return no installation commands for the unprivileged macOS backend.

    Args:
        main_app (bool): Unused direct application status.
        refresh_backend (Callable[[], None] | None): Unused backend refresh callback.

    Returns:
        tuple[CommandRegistration, ...]: Empty command set.
    """
    del main_app, refresh_backend
    return ()
