"""Provide the sole small host process primitive for trusted infrastructure clients."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ... import constants
from ...telemetry import telemetry_audit, telemetry_trace_event
from ..runtime.bootstrap import InstalledExecutable


class ProcessSpawnError(RuntimeError):
    """Report failure to create a classified child process."""


class ProcessTimedOutError(RuntimeError):
    """Report forced termination after a classified process deadline."""


class ProcessCancelledError(RuntimeError):
    """Report forced termination after explicit cancellation."""


@dataclass(frozen=True, slots=True)
class VerifiedExecutable:
    """Bind a host executable path to descriptor-derived content identity.

    Args:
        path (Path): Absolute executable path opened without following symlinks.
        sha256 (str): SHA-256 digest of the opened regular file.
        device (int): Filesystem device identity.
        inode (int): Filesystem inode identity.
        size (int): Executable byte size.
        mtime_ns (int): Executable modification timestamp.
    """

    path: Path
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class VerifiedDirectory:
    """Bind an absolute host directory to an opened filesystem identity.

    Args:
        path (Path): Absolute directory path opened without following symlinks.
        device (int): Filesystem device identity.
        inode (int): Filesystem inode identity.
    """

    path: Path
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class InfrastructureProcessResult:
    """Return bounded infrastructure process output.

    Args:
        exit_code (int): Process exit status.
        stdout (bytes): Bounded standard output.
        stderr (bytes): Bounded standard error.
        stdout_truncated (bool): Whether standard output exceeded its limit.
        stderr_truncated (bool): Whether standard error exceeded its limit.
    """

    exit_code: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool


@dataclass(frozen=True, slots=True)
class InfrastructureProcessCommand:
    """Describe one already-classified fixed process invocation.

    Args:
        argv (tuple[str, ...]): Exact non-shell argv, including executable path.
        environment (Mapping[str, str]): Complete sealed child environment.
        cwd (Path): Private child working directory.
        operation (str): Sanitized domain operation name for audit records.
        executable (InstalledExecutable | VerifiedExecutable | None): Manifest or explicitly
            verified host executable to revalidate. ``None`` is reserved for the allowlisted
            operating-system signature verifier.
        cwd_identity (VerifiedDirectory | None): Optional descriptor-derived cwd authority.
    """

    argv: tuple[str, ...]
    environment: Mapping[str, str]
    cwd: Path
    operation: str
    executable: InstalledExecutable | VerifiedExecutable | None
    cwd_identity: VerifiedDirectory | None = None


class InfrastructureProcessRunner:
    """Run fixed argv with sealed state, bounded pipes, and process-group cleanup.

    Args:
        output_limit (int): Maximum retained bytes per output stream.
    """

    output_limit: int

    def __init__(self, output_limit: int = constants.DEFAULT_EXECUTION_OUTPUT_BYTES) -> None:
        if output_limit <= 0:
            raise ValueError("Infrastructure output limit must be positive.")
        self.output_limit = output_limit

    def run(
        self,
        command: InfrastructureProcessCommand,
        *,
        stdin: bytes | None = None,
        deadline_seconds: float = constants.DEFAULT_COMMAND_TIMEOUT,
        cancellation: Callable[[], bool] = lambda: False,
        started: Callable[[], None] = lambda: None,
        cleanup_descendants: bool = False,
    ) -> InfrastructureProcessResult:
        """Run one classified command without interpreting its arguments.

        Args:
            command (InfrastructureProcessCommand): Exact invocation built by its domain owner.
            stdin (bytes | None): Optional bounded standard input.
            deadline_seconds (float): Positive completion deadline in seconds.
            cancellation (Callable[[], bool]): Predicate that cancels the owned process group.
            started (Callable[[], None]): Notification invoked only after successful creation.
            cleanup_descendants (bool): Whether to kill remaining process-group descendants after
                the leader exits. This is best-effort and is not sandbox containment.

        Returns:
            InfrastructureProcessResult: Bounded output and exit status.

        Raises:
            ValueError: If the command, input, cwd, or deadline is unsafe.
            RuntimeError: If launch, output capture, timeout, or cancellation fails.
        """
        validate_infrastructure_command(command)
        if deadline_seconds <= 0 or stdin is not None and len(stdin) > self.output_limit:
            raise ValueError("Infrastructure process bounds are invalid.")
        telemetry_trace_event(
            "execution.infrastructure.start",
            payload={"operation": command.operation, "argv": command.argv[1:]},
        )
        process: subprocess.Popen[bytes] | None = None
        try:
            cwd_descriptor = (
                open_verified_directory(command.cwd_identity)
                if command.cwd_identity is not None
                else None
            )
            process = subprocess.Popen(
                command.argv,
                cwd=command.cwd,
                env=dict(command.environment),
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                close_fds=True,
                start_new_session=True,
            )
            if cwd_descriptor is not None:
                try:
                    post_launch_descriptor = open_verified_directory(command.cwd_identity)
                except ValueError:
                    _kill_process_group(process)
                    raise
                os.close(post_launch_descriptor)
                os.close(cwd_descriptor)
                cwd_descriptor = None
            try:
                started()
            except Exception as error:
                _kill_process_group(process)
                raise RuntimeError("Classified process start observer failed.") from error
            stdout = BoundedDrain(process.stdout, self.output_limit)
            stderr = BoundedDrain(process.stderr, self.output_limit)
            stdout.start()
            stderr.start()
            if stdin is not None and process.stdin is not None:
                try:
                    process.stdin.write(stdin)
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                _wait_for_process(process, deadline_seconds, cancellation)
            except TimeoutError:
                _kill_process_group(process)
                raise ProcessTimedOutError("Classified process exceeded its deadline.") from None
            except InterruptedError:
                _kill_process_group(process)
                raise ProcessCancelledError("Classified process was cancelled.") from None
            finally:
                if cleanup_descendants and process.poll() is not None:
                    _kill_process_group(process)
                stdout.join()
                stderr.join()
            if stdout.error is not None or stderr.error is not None:
                _kill_process_group(process)
                raise RuntimeError("Infrastructure output reader failed.") from (
                    stdout.error or stderr.error
                )
        except ValueError:
            if process is not None:
                _kill_process_group(process)
            raise
        except OSError as error:
            if process is not None:
                _kill_process_group(process)
            telemetry_audit("execution.infrastructure.spawn_failed", operation=command.operation)
            raise ProcessSpawnError("Classified process could not start.") from error
        finally:
            if "cwd_descriptor" in locals() and cwd_descriptor is not None:
                os.close(cwd_descriptor)
            _finalize_process(process)
        result = InfrastructureProcessResult(
            process.returncode,
            stdout.data,
            stderr.data,
            stdout.truncated,
            stderr.truncated,
        )
        telemetry_audit(
            "execution.infrastructure.finished",
            operation=command.operation,
            exit_code=result.exit_code,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
        )
        return result


def sealed_environment(
    executable: InstalledExecutable,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a complete minimal environment for one manifest executable.

    Args:
        executable (InstalledExecutable): Manifest executable supplying allowed bundle paths.
        extra (Mapping[str, str] | None): Domain-owned fixed additional environment entries.

    Returns:
        dict[str, str]: Complete sealed child environment.

    Raises:
        ValueError: If an environment key or value is invalid.
    """
    directories = (*executable.path_directories, "/usr/bin", "/bin")
    if any(not Path(directory).is_absolute() for directory in directories):
        raise ValueError("Infrastructure PATH contains a nonabsolute directory.")
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": ":".join(dict.fromkeys(directories)),
    }
    if extra is not None:
        environment.update(extra)
    if any(
        not key or not key.replace("_", "A").isalnum() or "\0" in value
        for key, value in environment.items()
    ):
        raise ValueError("Infrastructure environment is invalid.")
    return environment


def validate_installed_executable(executable: InstalledExecutable) -> None:
    """Revalidate a lease-bound executable immediately before process creation.

    Args:
        executable (InstalledExecutable): Executable identity to verify.

    Raises:
        ValueError: If any runtime, lease, path, mode, or content identity changed.
    """
    root = executable.runtime_root
    artifact_root = executable.artifact_root
    lease = executable.lease
    if (
        not root.is_absolute()
        or root.is_symlink()
        or artifact_root.is_symlink()
        or executable.path.is_symlink()
        or executable.artifact_set_digest != lease.artifact_set_digest
        or lease.path.parent != root / constants.RUNTIME_LEASES_DIRECTORY
        or not lease.path.is_file()
        or lease.path.is_symlink()
    ):
        raise ValueError("Infrastructure executable is not bound to an active private runtime.")
    try:
        root_stat = root.stat()
        artifact_relative = artifact_root.relative_to(root)
        executable_relative = executable.path.relative_to(artifact_root)
        lease_payload = json.loads(lease.path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Infrastructure executable provenance is invalid.") from error
    if (
        not artifact_relative.parts
        or artifact_relative.parts[0] != constants.RUNTIME_ARTIFACTS_DIRECTORY
        or not executable_relative.parts
        or lease_payload.get("owner_id") != lease.owner_id
        or lease_payload.get("manifest_digest") != lease.manifest_digest
        or lease_payload.get("artifact_set_digest") != lease.artifact_set_digest
        or not isinstance(lease_payload.get("expiry"), (int, float))
        or lease_payload["expiry"] <= time.time()
    ):
        raise ValueError("Infrastructure executable lease is not active.")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = root_fd
    try:
        for component in (*artifact_relative.parts, *executable_relative.parts[:-1]):
            next_descriptor = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            if descriptor != root_fd:
                os.close(descriptor)
            descriptor = next_descriptor
        executable_fd = os.open(
            executable_relative.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor
        )
    except OSError as error:
        raise ValueError("Infrastructure executable provenance is invalid.") from error
    finally:
        if descriptor != root_fd:
            os.close(descriptor)
        os.close(root_fd)
    try:
        metadata = os.fstat(executable_fd)
        with os.fdopen(os.dup(executable_fd), "rb") as stream:
            identity = hashlib.file_digest(stream, "sha256").hexdigest()
        if (
            not os.path.samestat(root_stat, root.stat())
            or not stat.S_ISREG(metadata.st_mode)
            or identity != executable.expected_identity
            or not metadata.st_mode & stat.S_IXUSR
        ):
            raise ValueError("Infrastructure executable identity changed.")
    finally:
        os.close(executable_fd)


def identify_host_executable(path: Path) -> VerifiedExecutable:
    """Resolve one absolute executable through a no-follow descriptor walk.

    Args:
        path (Path): Absolute host executable path.

    Returns:
        VerifiedExecutable: Descriptor-derived immutable launch identity.

    Raises:
        ValueError: If the path is not an executable regular file or crosses a symlink.
    """
    descriptor = _open_absolute(path, directory=False)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not metadata.st_mode & 0o111:
            raise ValueError("Host executable is not an executable regular file.")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return VerifiedExecutable(
            path=path,
            sha256=digest,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
        )
    finally:
        os.close(descriptor)


def identify_host_directory(path: Path) -> VerifiedDirectory:
    """Resolve one absolute working directory through a no-follow descriptor walk.

    Args:
        path (Path): Absolute host directory path.

    Returns:
        VerifiedDirectory: Descriptor-derived directory identity.

    Raises:
        ValueError: If the path is not a directory or crosses a symlink.
    """
    descriptor = _open_absolute(path, directory=True)
    try:
        metadata = os.fstat(descriptor)
        return VerifiedDirectory(path=path, device=metadata.st_dev, inode=metadata.st_ino)
    finally:
        os.close(descriptor)


def open_verified_directory(directory: VerifiedDirectory) -> int:
    """Open and revalidate a directory identity for descriptor-based child cwd.

    Args:
        directory (VerifiedDirectory): Previously authorized host directory identity.

    Returns:
        int: Owned directory descriptor that the caller must close.

    Raises:
        ValueError: If the directory identity changed.
    """
    descriptor = _open_absolute(directory.path, directory=True)
    metadata = os.fstat(descriptor)
    if (metadata.st_dev, metadata.st_ino) != (directory.device, directory.inode):
        os.close(descriptor)
        raise ValueError("Host working directory identity changed.")
    return descriptor


def validate_verified_executable(executable: VerifiedExecutable) -> None:
    """Revalidate an explicit host executable immediately before launch.

    Args:
        executable (VerifiedExecutable): Previously authorized host executable identity.

    Raises:
        ValueError: If path, metadata, or content identity changed.
    """
    current = identify_host_executable(executable.path)
    if current != executable:
        raise ValueError("Host executable identity changed.")


def _open_absolute(path: Path, *, directory: bool) -> int:
    """Open an absolute path component-by-component without following symlinks."""
    if not path.is_absolute() or not path.parts or path.parts[0] != "/":
        raise ValueError("Host authority paths must be absolute.")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, component in enumerate(path.parts[1:]):
            final = index == len(path.parts[1:]) - 1
            flags = os.O_RDONLY | os.O_NOFOLLOW
            if not final or directory:
                flags |= os.O_DIRECTORY
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as error:
        os.close(descriptor)
        raise ValueError("Host authority path could not be verified.") from error
    return descriptor


def validate_infrastructure_command(command: InfrastructureProcessCommand) -> None:
    """Reject an unsealed or unclassified primitive invocation.

    Args:
        command (InfrastructureProcessCommand): Fixed invocation to validate.

    Raises:
        ValueError: If the invocation lacks a classified executable, sealed working directory,
            operation identity, or safe argument shape.
    """
    if (
        not command.argv
        or not command.operation
        or not command.cwd.is_absolute()
        or command.cwd.is_symlink()
        or not command.cwd.is_dir()
        or any("\0" in value for value in command.argv)
    ):
        raise ValueError("Infrastructure process command is invalid.")
    if command.cwd_identity is not None and command.cwd_identity.path != command.cwd:
        raise ValueError("Infrastructure command cwd identity is inconsistent.")
    if command.executable is None:
        if command.argv[0] != "/usr/bin/codesign":
            raise ValueError("System executable is not classified for infrastructure use.")
    elif isinstance(command.executable, VerifiedExecutable):
        validate_verified_executable(command.executable)
        if command.argv[0] != str(command.executable.path):
            raise ValueError("Host command executable identity is inconsistent.")
    else:
        validate_installed_executable(command.executable)
        if command.argv[0] != str(command.executable.path):
            raise ValueError("Infrastructure command executable identity is inconsistent.")


class BoundedDrain(threading.Thread):
    """Drain one stream while retaining a bounded prefix."""

    _stream: object
    _limit: int
    data: bytes
    truncated: bool
    error: OSError | None

    def __init__(self, stream: object, limit: int) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._limit = limit
        self.data = b""
        self.truncated = False
        self.error = None

    def run(self) -> None:
        """Drain to EOF and discard bytes beyond the limit."""
        retained = bytearray()
        try:
            while chunk := self._stream.read(64 * 1024):
                remaining = self._limit - len(retained)
                if remaining > 0:
                    retained.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.truncated = True
        except OSError as error:
            self.error = error
        self.data = bytes(retained)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    """Kill and reap one runner-owned process group."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _finalize_process(process: subprocess.Popen[bytes] | None) -> None:
    """Reap a created child and close its parent-owned pipes, if creation succeeded."""
    if process is None:
        return
    if process.poll() is None:
        _kill_process_group(process)
    _close_process_streams(process)


def _close_process_streams(process: subprocess.Popen[bytes]) -> None:
    """Close every parent-owned pipe after readers have stopped and the child is reaped."""
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _wait_for_process(
    process: subprocess.Popen[bytes],
    deadline_seconds: float,
    cancellation: Callable[[], bool],
) -> None:
    """Wait for a child while enforcing cancellation and a wall deadline."""
    deadline = time.monotonic() + deadline_seconds
    while process.poll() is None:
        if cancellation():
            raise InterruptedError
        if time.monotonic() >= deadline:
            raise TimeoutError
        time.sleep(0.005)
