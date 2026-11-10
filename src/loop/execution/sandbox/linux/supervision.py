"""Supervise an admitted PID namespace and refuse unverified descendant cleanup.

The trusted receiver must launch authenticated Bubblewrap with a private PID namespace,
--die-with-parent and a new session, before any payload runs. No process-group signal alone
proves cleanup. This owner checks protected kernel cgroup membership after the common
supervisor terminates the owned launcher. It does not launch, authorize or enable Linux.
"""

from __future__ import annotations

import errno
import fcntl
import os
import select
import socket
import struct
import subprocess
import termios
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from ....constants import MAX_OUTPUT_CHARS
from ....utils.process import ProcessCapture, read_bounded_stream, supervise_process
from .admission import GenerationAdmission, admission_group
from .constants import CGROUP_PROCS_FILENAME, NATIVE_CANCEL_FRAME

if TYPE_CHECKING:
    from .egress import EgressBroker

_CLEANUP_SECONDS = 1.0
_CLIENT_DISCONNECT_POLL_MILLISECONDS = 50
_CLIENT_HALF_CLOSE_EVENT = getattr(select, "POLLRDHUP", 0x2000)
_SPAWN_CONTROL_GRACE_SECONDS = 1.0
_CLEANUP_POLL_SECONDS = 0.01
_MAX_CGROUP_MEMBERSHIP_BYTES = 65_536
_MAX_TERMINAL_INPUT_BYTES = 4_096
_TERMINAL_INPUT_SECONDS = 1.0


@contextmanager
def guard_client_disconnect(connection: socket.socket) -> Iterator[threading.Event]:
    """Watch client disconnect and stalled-spawn control without consuming socket data.

    Use only after the receiver's fixed systemd service and cgroup policy have been attested.
    The yielded event must be set immediately after spawn returns. A readable control channel
    while spawn remains pending starts a bounded grace period, after which receiver exit delegates
    process termination to the fixed systemd cgroup policy. An empty half-close is not treated as
    control, and bytes remain available to admission and supervision. Receiver exit does not
    certify that cleanup has completed.

    Args:
        connection (socket.socket): Retained authenticated activated client socket.

    Yields:
        threading.Event: Signal to set when the owned spawn returns.

    Raises:
        OSError: The watcher cannot be started or stopped within the cleanup bound.
        RuntimeError: The watcher thread cannot be started.
    """
    poller = select.poll()
    descriptor = connection.fileno()
    disconnected = select.POLLHUP | select.POLLERR | select.POLLNVAL
    poller.register(descriptor, disconnected | select.POLLIN | _CLIENT_HALF_CLOSE_EVENT)
    stopped = threading.Event()
    spawn_returned = threading.Event()

    def watch() -> None:
        """Exit without interpreting a half-close as a disconnect."""
        status = 1
        exit_receiver = True
        control_deadline = None
        control_observed = False
        try:
            while not stopped.is_set():
                if control_observed and spawn_returned.is_set():
                    control_deadline = None
                if control_deadline is not None and time.monotonic() >= control_deadline:
                    status = 0
                    break
                events = poller.poll(_CLIENT_DISCONNECT_POLL_MILLISECONDS)
                if not events:
                    continue
                if len(events) != 1 or events[0][0] != descriptor:
                    break
                mask = events[0][1]
                if mask & ~(disconnected | select.POLLIN | _CLIENT_HALF_CLOSE_EVENT):
                    break
                if mask & (select.POLLERR | select.POLLNVAL):
                    break
                if mask & select.POLLHUP:
                    status = 0
                    break
                if mask & _CLIENT_HALF_CLOSE_EVENT:
                    pending = connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                    if pending and not spawn_returned.is_set():
                        control_deadline = time.monotonic() + _SPAWN_CONTROL_GRACE_SECONDS
                    poller.modify(descriptor, disconnected)
                    control_observed = True
                elif mask & select.POLLIN:
                    if not spawn_returned.is_set():
                        control_deadline = time.monotonic() + _SPAWN_CONTROL_GRACE_SECONDS
                    poller.modify(descriptor, disconnected)
                    control_observed = True
                else:
                    break
            else:
                exit_receiver = False
        finally:
            if exit_receiver:
                os._exit(status)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        yield spawn_returned
    finally:
        stopped.set()
        watcher.join(_CLEANUP_SECONDS)
        if watcher.is_alive():
            raise OSError("Client disconnect watcher did not stop.")


def supervise_payload(
    process: subprocess.Popen[str],
    config: GenerationAdmission,
    deadline: float,
    *,
    connection: int | None = None,
    mediator: EgressBroker | None = None,
    output_observer: Callable[[str, str], None] | None = None,
    send_lock: threading.Lock | None = None,
    terminal_master: int | None = None,
) -> ProcessCapture:
    """Capture an owned namespace launcher and verify that no admitted descendants remain.

    Call in the authenticated single-request receiver, never in an ordinary client. The
    process must be its directly spawned, new-session Bubblewrap child with a private PID
    namespace and --die-with-parent. Reuse common bounded I/O, timeout, cancellation and
    launcher termination, then require the same immutable service scope to contain only
    this receiver. Detached descendants or ambiguous cleanup raise rather than return a
    successful command result. Existing children remain confined; no foreign PID is killed.

    Args:
        process (subprocess.Popen[str]): Owned authenticated native namespace launcher.
        config (GenerationAdmission): Original installed fixed generation admission.
        deadline (float): Absolute monotonic execution deadline after authorization.
        connection (int | None): Admitted socket descriptor that may carry the fixed
            cancellation byte; a full client disconnect also cancels. ``None`` leaves
            cancellation to the common supervisor. Request half-close is not cancellation.
        mediator (EgressBroker | None): Attempt-owned HTTP broker whose authenticated
            heartbeats and revision acknowledgements share the admitted control socket.
        output_observer (Callable[[str, str], None] | None): Bounded process output receiver.
        send_lock (threading.Lock | None): Shared output/control socket writer lock.
        terminal_master (int | None): Receiver-owned PTY master for bounded interactive controls.

    Returns:
        ProcessCapture: Common typed output/exit result after verified descendant absence.

    Raises:
        ValueError: Protected scope changes, contains subgroups or membership is malformed.
        TimeoutError: Descendants remain after the one-second cleanup budget.
        OSError: Kernel membership or native supervision cannot be inspected.
    """
    try:
        group = admission_group(config)
        original = group.lstat()
    except (OSError, ValueError):
        supervise_process(process, time.monotonic())
        raise
    stopped, cancelled, failed = threading.Event(), threading.Event(), threading.Event()

    def write_terminal(data: bytes) -> None:
        """Write bounded input with a finite per-frame backpressure ceiling."""
        deadline = time.monotonic() + _TERMINAL_INPUT_SECONDS
        offset = 0
        while offset < len(data):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Interactive terminal input remained blocked.")
            writable = select.select([], [terminal_master], [], min(0.05, remaining))[1]
            if writable:
                try:
                    count = os.write(terminal_master, data[offset:])
                except BlockingIOError:
                    continue
                if count <= 0:
                    raise OSError("Interactive terminal input made no progress.")
                offset += count

    def acknowledge(payload: bytes) -> None:
        """Serialize control acknowledgements with optional live output frames."""
        if send_lock is None:
            os.write(connection, payload)
        else:
            with send_lock:
                os.write(connection, payload)

    def watch():
        """Cancel the owned launcher on a bound control byte or full client disconnect."""
        try:
            poller = select.poll()
            disconnected = select.POLLHUP | select.POLLERR | select.POLLNVAL
            poller.register(connection, select.POLLIN | disconnected)
            frame = bytearray()
            while not stopped.is_set():
                if mediator is not None and mediator.lease_expired():
                    if not mediator.close():
                        failed.set()
                    cancelled.set()
                    return
                events = poller.poll(_CLIENT_DISCONNECT_POLL_MILLISECONDS)
                if not events:
                    continue
                mask = events[0][1]
                if mask & select.POLLIN:
                    marker = frame[0] if frame else None
                    target = (
                        9
                        if marker in (ord("H"), ord("R"), ord("P"), ord("U"))
                        else 5
                        if marker == ord("Z")
                        else 3
                        if marker == ord("I") and len(frame) < 3
                        else 3 + int.from_bytes(frame[1:3], "big")
                        if marker == ord("I")
                        else 1
                    )
                    control = os.read(connection, target - len(frame))
                    if control:
                        frame.extend(control)
                        if frame == NATIVE_CANCEL_FRAME:
                            cancelled.set()
                            return
                        if frame[0] in (ord("I"), ord("Z"), ord("F"), ord("X")):
                            if terminal_master is None:
                                failed.set()
                                cancelled.set()
                                return
                            marker = frame[0]
                            if marker == ord("I"):
                                if len(frame) < 3:
                                    continue
                                size = int.from_bytes(frame[1:3], "big")
                                if not 0 < size <= _MAX_TERMINAL_INPUT_BYTES:
                                    failed.set()
                                    cancelled.set()
                                    return
                                if len(frame) < 3 + size:
                                    continue
                                write_terminal(bytes(frame[3:]))
                            elif marker == ord("Z"):
                                if len(frame) < 5:
                                    continue
                                rows, columns = struct.unpack("!HH", frame[1:5])
                                if not 1 <= rows <= 500 or not 1 <= columns <= 500:
                                    failed.set()
                                    cancelled.set()
                                    return
                                fcntl.ioctl(
                                    terminal_master,
                                    termios.TIOCSWINSZ,
                                    struct.pack("HHHH", rows, columns, 0, 0),
                                )
                            else:
                                write_terminal(b"\x04" if marker == ord("F") else b"\x03")
                            frame.clear()
                            continue
                        if mediator is None or frame[0] not in (
                            ord("H"),
                            ord("R"),
                            ord("P"),
                            ord("U"),
                        ):
                            failed.set()
                            cancelled.set()
                            return
                        if len(frame) < 9:
                            continue
                        revision = int.from_bytes(frame[1:], "big")
                        if frame[0] == ord("H"):
                            if not mediator.heartbeat(revision):
                                failed.set()
                                cancelled.set()
                                return
                            acknowledge(b"h" + bytes(frame[1:]))
                        elif frame[0] == ord("P"):
                            if not mediator.pause(revision):
                                failed.set()
                                cancelled.set()
                                return
                            acknowledge(b"p" + bytes(frame[1:]))
                        elif frame[0] == ord("U"):
                            if not mediator.resume(revision):
                                failed.set()
                                cancelled.set()
                                return
                            acknowledge(b"u" + bytes(frame[1:]))
                        else:
                            confirmed = mediator.accept_revision(revision)
                            acknowledge(b"A" + bytes(frame[1:]) + (b"1" if confirmed else b"0"))
                            if not confirmed:
                                failed.set()
                            cancelled.set()
                            return
                        frame.clear()
                        continue
                    poller.modify(connection, disconnected)
                if mask & disconnected:
                    cancelled.set()
                    return
        except (OSError, TimeoutError, ValueError):
            failed.set()
            cancelled.set()

    watcher = threading.Thread(target=watch, daemon=True) if connection is not None else None
    if watcher is not None:
        watcher.start()
    try:

        def read_pty(stream, chunks, *, maximum):
            """Treat EIO as EOF only for the receiver-owned terminal master."""
            return read_bounded_stream(
                stream,
                chunks,
                maximum=maximum,
                eof_errno=errno.EIO if stream is process.stdout else None,
            )

        result = supervise_process(
            process,
            deadline,
            stdout_limit=MAX_OUTPUT_CHARS + 80,
            cancel_event=cancelled,
            preserve_newlines=True,
            **({"output_observer": output_observer} if output_observer else {}),
            **({"reader": read_pty} if terminal_master is not None else {}),
        )
    finally:
        stopped.set()
        if watcher is not None:
            watcher.join(_CLEANUP_SECONDS)
            if watcher.is_alive():
                raise OSError("Client cancellation watcher did not stop.")
    cleanup_deadline = time.monotonic() + _CLEANUP_SECONDS
    while True:
        current = admission_group(config)
        latest = current.lstat()
        if current != group or (latest.st_dev, latest.st_ino) != (
            original.st_dev,
            original.st_ino,
        ):
            raise ValueError("Admitted cleanup scope changed.")
        if any(entry.is_dir() for entry in group.iterdir()):
            raise ValueError("Admitted cleanup scope contains unexpected subgroups.")
        # This is the authoritative kernel domain membership, not a /proc PID inventory.
        with (group / CGROUP_PROCS_FILENAME).open("rb", buffering=0) as stream:
            members = stream.read(_MAX_CGROUP_MEMBERSHIP_BYTES + 1)
        values = members.splitlines()
        if (
            len(members) > _MAX_CGROUP_MEMBERSHIP_BYTES
            or not values
            or any(not value.isdigit() or int(value) <= 0 for value in values)
        ):
            raise ValueError("Admitted cleanup membership is malformed or exceeds its bound.")
        if values == [str(os.getpid()).encode("ascii")]:
            if failed.is_set():
                raise OSError("Client cancellation authority could not be observed.")
            return result
        if time.monotonic() >= cleanup_deadline:
            raise TimeoutError("Admitted descendants remain; confinement and policy retained.")
        time.sleep(min(_CLEANUP_POLL_SECONDS, max(0, cleanup_deadline - time.monotonic())))
