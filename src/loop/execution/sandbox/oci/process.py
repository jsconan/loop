"""Own attached OCI management processes, byte streams, and PTY lifecycle."""

from __future__ import annotations

import errno
import fcntl
import os
import pty
import queue
import select
import signal
import struct
import subprocess
import termios
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from ...infrastructure import InfrastructureProcessCommand, validate_infrastructure_command


class OciSessionTimedOut(RuntimeError):
    """Report a session stopped at its authorized wall deadline."""


class OciSessionCancelled(RuntimeError):
    """Report a session stopped by an explicit cancellation request."""


class OciSessionOutputFailure(RuntimeError):
    """Report failed or saturated attached output capture."""


@dataclass(frozen=True, slots=True)
class OciStreamFrame:
    """Carry one ordered opaque byte frame from an OCI process.

    Args:
        sequence (int): Monotonic session-local sequence.
        stream (str): ``stdout``, ``stderr``, or merged ``pty``.
        data (bytes): Unparsed child bytes.
    """

    sequence: int
    stream: str
    data: bytes


class OciProcessSession:
    """Own one attached OCI management process and bounded output queue.

    Args:
        command (InfrastructureProcessCommand): Platform transport command compiled for OCI.
        use_pty (bool): Whether to expose one merged pseudo-terminal stream.
        queue_limit (int): Maximum unread frames before fail-closed cleanup.
        deadline_seconds (float): Positive wall deadline.
        terminal_columns (int | None): Initial PTY width, or ``None`` for the default.
        terminal_rows (int | None): Initial PTY height, or ``None`` for the default.
    """

    _command: InfrastructureProcessCommand
    _deadline: float
    _frames: queue.Queue[OciStreamFrame]
    _closed: threading.Event
    _stdin_closed: threading.Event
    _lock: threading.Lock
    _cleanup_lock: threading.Lock
    _sequence: int
    _readers: list[threading.Thread]
    _master_fd: int | None
    _pty_eof: bytes | None
    _reader_error: OSError | None
    _backpressure_exceeded: bool
    _process: subprocess.Popen[bytes]

    def __init__(
        self,
        command: InfrastructureProcessCommand,
        use_pty: bool,
        queue_limit: int,
        deadline_seconds: float,
        terminal_columns: int | None = None,
        terminal_rows: int | None = None,
    ) -> None:
        if command.executable is None or queue_limit <= 0 or deadline_seconds <= 0:
            raise ValueError("OCI process session bounds are invalid.")
        validate_infrastructure_command(command)
        self._command = command
        self._deadline = time.monotonic() + deadline_seconds
        self._frames: queue.Queue[OciStreamFrame] = queue.Queue(maxsize=queue_limit)
        self._closed = threading.Event()
        self._stdin_closed = threading.Event()
        self._lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._sequence = 0
        self._readers: list[threading.Thread] = []
        self._master_fd: int | None = None
        self._pty_eof: bytes | None = None
        self._reader_error: OSError | None = None
        self._backpressure_exceeded = False
        if use_pty and (terminal_columns is None or terminal_rows is None):
            terminal_columns, terminal_rows = 80, 24
        if not use_pty and (terminal_columns is not None or terminal_rows is not None):
            raise ValueError("Pipe process sessions cannot carry terminal dimensions.")
        try:
            if use_pty:
                master_fd, slave_fd = pty.openpty()
                self._master_fd = master_fd
                try:
                    self._pty_eof = _configure_pty_slave(slave_fd)
                    _set_pty_size(master_fd, terminal_columns, terminal_rows)
                    self._process = _spawn(command, slave_fd, slave_fd, slave_fd)
                finally:
                    os.close(slave_fd)
                self._start_reader("pty", master_fd)
            else:
                self._process = _spawn(command, subprocess.PIPE, subprocess.PIPE, subprocess.PIPE)
                self._start_reader("stdout", self._process.stdout)
                self._start_reader("stderr", self._process.stderr)
        except OSError as error:
            self._closed.set()
            if self._master_fd is not None:
                os.close(self._master_fd)
            raise RuntimeError("OCI process session could not start.") from error

    def write(self, data: bytes) -> None:
        """Write one bounded opaque input frame."""
        if len(data) > 1024 * 1024:
            raise ValueError("OCI process session input exceeds the bounded limit.")
        self._revalidate_live()
        failure: BaseException | None = None
        with self._lock:
            if self._stdin_closed.is_set():
                raise RuntimeError("OCI process session stdin is closed.")
            try:
                if self._master_fd is not None:
                    os.write(self._master_fd, data)
                elif self._process.stdin is not None:
                    self._process.stdin.write(data)
                    self._process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                failure = error
        if failure is not None:
            self.close()
            raise RuntimeError("OCI process session stdin failed.") from failure

    def close_stdin(self) -> None:
        """Close input exactly once without discarding output."""
        with self._lock:
            if self._stdin_closed.is_set():
                return
            self._stdin_closed.set()
            if self._master_fd is not None and self._pty_eof is not None:
                for _ in range(2):
                    try:
                        os.write(self._master_fd, self._pty_eof)
                    except OSError:
                        break
            elif self._process.stdin is not None:
                self._process.stdin.close()

    def resize(self, columns: int, rows: int) -> None:
        """Resize the owned PTY and let the kernel notify its foreground group."""
        if not 0 < columns <= 16384 or not 0 < rows <= 16384:
            raise ValueError("PTY dimensions must be positive bounded values.")
        self._revalidate_live()
        if self._master_fd is None:
            raise RuntimeError("OCI process session does not own a PTY.")
        try:
            _set_pty_size(self._master_fd, columns, rows)
        except (OSError, ProcessLookupError) as error:
            self.close()
            raise RuntimeError("OCI PTY resize failed.") from error

    def read(self, timeout: float | None = None) -> OciStreamFrame | None:
        """Return the next ordered byte frame, or ``None`` on a bounded wait."""
        self._revalidate_live()
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def wait(self, cancellation: Callable[[], bool] = lambda: False) -> int:
        """Wait for terminal completion while enforcing cancellation and cleanup."""
        self._raise_output_failure()
        self._revalidate_live()
        try:
            _wait(self._process, max(0.001, self._deadline - time.monotonic()), cancellation)
        except TimeoutError:
            self.close()
            raise OciSessionTimedOut("OCI process session exceeded its deadline.") from None
        except InterruptedError:
            self.close()
            raise OciSessionCancelled("OCI process session was cancelled.") from None
        self._join_readers()
        self._raise_output_failure()
        self._close_process_streams()
        return self._process.returncode

    def detach(self) -> None:
        """Request the upstream client detach sequence without stopping the container."""
        try:
            self.write(b"\x10\x11")
            if self.wait() != 0:
                raise RuntimeError("OCI process session detach failed.")
        finally:
            self.close()

    def close(self) -> None:
        """Kill, reap, and close this process session exactly once."""
        with self._cleanup_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            _kill(self._process)
            self._join_readers()
            with self._lock:
                master_fd, self._master_fd = self._master_fd, None
            if master_fd is not None:
                try:
                    os.close(master_fd)
                except OSError:
                    pass
            self._close_process_streams()

    def _start_reader(self, stream: str, source: object) -> None:
        """Start one queue-bounded reader."""
        reader = threading.Thread(target=self._read_stream, args=(stream, source), daemon=True)
        self._readers.append(reader)
        reader.start()

    def _read_stream(self, stream: str, source: object) -> None:
        """Copy raw bytes to the queue or close on saturation or reader failure."""
        try:
            descriptor = source if isinstance(source, int) else source.fileno()
            while not self._closed.is_set():
                readable, _, _ = select.select((descriptor,), (), (), 0.05)
                if not readable:
                    continue
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    return
                with self._lock:
                    frame = OciStreamFrame(self._sequence, stream, chunk)
                    self._sequence += 1
                try:
                    self._frames.put(frame, timeout=0.1)
                except queue.Full:
                    self._backpressure_exceeded = True
                    self.close()
                    return
        except OSError as error:
            if (
                not (
                    stream == "pty"
                    and error.errno == errno.EIO
                    and self._process.poll() is not None
                )
                and not self._closed.is_set()
            ):
                self._reader_error = error
                self.close()

    def _join_readers(self) -> None:
        """Join all readers except the current callback thread."""
        current = threading.current_thread()
        for reader in self._readers:
            if reader is not current:
                reader.join(timeout=1)

    def _close_process_streams(self) -> None:
        """Close every parent-owned pipe after process completion and reader shutdown."""
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None:
                stream.close()

    def _revalidate_live(self) -> None:
        """Reject closed, expired, or lease-invalid session access."""
        if self._closed.is_set() or time.monotonic() >= self._deadline:
            self.close()
            raise RuntimeError("OCI process session is no longer active.")
        if self._command.executable is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("OCI process session has no runtime identity.")
        validate_infrastructure_command(self._command)

    def _raise_output_failure(self) -> None:
        """Surface a reader or queue failure before generic closed-session handling."""
        if self._reader_error is not None:
            self.close()
            raise OciSessionOutputFailure(
                "OCI process session output reader failed."
            ) from self._reader_error
        if self._backpressure_exceeded:
            self.close()
            raise OciSessionOutputFailure("OCI process session exceeded its backpressure bound.")


def _spawn(
    command: InfrastructureProcessCommand,
    stdin: int,
    stdout: int,
    stderr: int,
) -> subprocess.Popen[bytes]:
    """Spawn one session command with no inherited process authority."""
    return subprocess.Popen(
        command.argv,
        cwd=command.cwd,
        env=dict(command.environment),
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        shell=False,
        close_fds=True,
        start_new_session=True,
    )


def _configure_pty_slave(descriptor: int) -> bytes:
    """Disable echo and return the configured canonical EOF byte."""
    attributes = termios.tcgetattr(descriptor)
    attributes[3] &= ~termios.ECHO
    termios.tcsetattr(descriptor, termios.TCSANOW, attributes)
    eof = attributes[6][termios.VEOF]
    return bytes((eof if isinstance(eof, int) else ord(eof),))


def _set_pty_size(descriptor: int, columns: int | None, rows: int | None) -> None:
    """Apply one validated PTY geometry without sending a resize signal."""
    if columns is None or rows is None or not 0 < columns <= 16384 or not 0 < rows <= 16384:
        raise ValueError("PTY dimensions must be positive bounded values.")
    fcntl.ioctl(
        descriptor,
        termios.TIOCSWINSZ,
        struct.pack("HHHH", rows, columns, 0, 0),
    )


def _kill(process: subprocess.Popen[bytes]) -> None:
    """Kill and reap one session-owned process group."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _wait(
    process: subprocess.Popen[bytes],
    seconds: float,
    cancellation: Callable[[], bool],
) -> None:
    """Poll one process under cancellation and a wall deadline."""
    deadline = time.monotonic() + seconds
    while process.poll() is None:
        if cancellation():
            raise InterruptedError
        if time.monotonic() >= deadline:
            raise TimeoutError
        time.sleep(0.005)
