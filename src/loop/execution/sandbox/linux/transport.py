"""Exchange one opaque frame with an authenticated installed ordinary receiver.

Transport supplies no command authorization or execution readiness. The receiver supplies bounded
attempt-bound typed results;
ordinary file preflight does not certify live kernel readiness. Trusted namespace entry verifies
actual enforcing bootstrap/cache exports before source starts.
"""

from __future__ import annotations

import math
import os
import select
import socket
import stat
import struct
import threading
import time
from collections.abc import Callable
from threading import Event
from typing import TYPE_CHECKING

from ....constants import (
    DEFAULT_STREAM_CHUNK_SIZE,
    MAX_NATIVE_RESULT_BYTES,
    MAX_NATIVE_UNIT_BYTES,
    NATIVE_ADMISSION_SOCKET_MODE,
    NATIVE_RESULT_DELIVERY_SECONDS,
    NATIVE_UNIT_PREFIX,
)
from .admission import MAX_FRAME_BYTES, GenerationAdmission
from .constants import (
    NATIVE_CANCEL_FRAME,
    NATIVE_REQUEST_MAGIC,
    PEER_CREDENTIALS_FORMAT,
    PEER_CREDENTIALS_SIZE,
    REQUEST_LENGTH_FORMAT,
    SO_PEERCRED_OPTION_FALLBACK,
    SYSTEM_RUNTIME_DIRECTORY,
    SYSTEMD_SERVICE_TEMPLATE_SUFFIX,
    SYSTEMD_SLICE_UNIT_SUFFIX,
    SYSTEMD_SOCKET_UNIT_SUFFIX,
)
from .host import HostFile, protected_path
from .installation import generation_limits

if TYPE_CHECKING:
    from ....permissions.manager import PermissionManager
    from .diagnostics import LinuxDenialObserver

_SO_PEERCRED_OPTION = getattr(socket, "SO_PEERCRED", SO_PEERCRED_OPTION_FALLBACK)
_RESPONSE_READ_CHUNK_BYTES = 4096
_EGRESS_CONTROL_SECONDS = 2.0
_EGRESS_HEARTBEAT_SECONDS = 0.5
_OUTPUT_FRAME_MAX_BYTES = 4 * DEFAULT_STREAM_CHUNK_SIZE


class EgressLease:
    """Maintain one authenticated heartbeat and bounded revision acknowledgement.

    Args:
        authority (PermissionManager): Manager that approved the exact attempt revision.
        revision (int): Policy revision captured with that approval.
    """

    authority: PermissionManager
    revision: int
    _connection: socket.socket | None
    _lock: threading.Lock
    _stopped: Event
    _acknowledged: Event
    _paused_ack: Event
    _resumed_ack: Event
    _thread: threading.Thread | None
    _last_ack: float
    _last_policy_check: float
    _revoked: bool
    _pending_revision: int | None
    _confirmed: bool
    _timely_confirmed: bool
    _ack_at: float | None
    _bound: bool

    def __init__(self, authority: PermissionManager, revision: int) -> None:
        self.authority = authority
        self.revision = revision
        self._connection = None
        self._lock = threading.Lock()
        self._stopped = Event()
        self._acknowledged = Event()
        self._paused_ack = Event()
        self._resumed_ack = Event()
        self._thread = None
        self._last_ack = 0.0
        self._last_policy_check = 0.0
        self._revoked = False
        self._pending_revision = None
        self._confirmed = False
        self._timely_confirmed = False
        self._ack_at = None
        self._bound = False

    def ready(self, connection: socket.socket) -> None:
        """Bind policy before sending the first heartbeat that opens the mediator route.

        Args:
            connection (socket.socket): Original authenticated receiver socket.

        Raises:
            OSError: The active policy changed before route activation.
        """
        self._connection = connection
        self.authority.bind_egress(self.revision, self)
        self._bound = True
        with self._lock:
            if self._revoked:
                raise OSError("Mediated egress authority was revoked before activation.")
            self._last_ack = time.monotonic()
            self._last_policy_check = self._last_ack
            self._write_frame(b"H" + self.revision.to_bytes(8, "big"))
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def heartbeat_ack(self, revision: int) -> None:
        """Accept only the current attempt revision from the receiver.

        Args:
            revision (int): Exact revision returned by the mediator.

        Raises:
            ValueError: A stale or foreign acknowledgement was received.
        """
        if self._revoked:
            return
        if revision != self.revision:
            raise ValueError("Mediated egress heartbeat acknowledgement differs.")
        self._last_ack = time.monotonic()

    def revision_ack(self, revision: int, confirmed: bool) -> None:
        """Settle only the pending exact revision after receiver closure confirmation.

        Args:
            revision (int): Revision echoed from the receiver.
            confirmed (bool): Whether the receiver confirmed closure.
        """
        with self._lock:
            if (
                self._revoked
                and revision == self._pending_revision
                and not self._acknowledged.is_set()
            ):
                self._confirmed = confirmed
                self._ack_at = time.monotonic()
                self._acknowledged.set()

    def pause_ack(self, revision: int) -> None:
        """Accept an exact pending-admission barrier acknowledgement.

        Args:
            revision (int): Current approving policy revision echoed by the receiver.
        """
        if revision == self.revision:
            self._paused_ack.set()

    def resume_ack(self, revision: int) -> None:
        """Accept an exact unchanged-policy resumption acknowledgement.

        Args:
            revision (int): Current approving policy revision echoed by the receiver.
        """
        if revision == self.revision:
            self._resumed_ack.set()

    def pause(self) -> bool:
        """Block new mediator connections before a policy update mutates active state.

        Returns:
            bool: Whether the independent mediator confirmed its admission barrier.
        """
        if self._revoked:
            return True
        self._paused_ack.clear()
        try:
            self._send(b"P" + self.revision.to_bytes(8, "big"))
            if self._paused_ack.wait(_EGRESS_CONTROL_SECONDS):
                return True
        except (OSError, ValueError):
            pass
        self._fail()
        return False

    def resume(self) -> bool:
        """Resume new connections only after an unchanged policy update settles.

        Returns:
            bool: Whether the independent mediator confirmed unchanged authority.
        """
        if self._revoked:
            return True
        self._resumed_ack.clear()
        try:
            self._send(b"U" + self.revision.to_bytes(8, "big"))
            if self._resumed_ack.wait(_EGRESS_CONTROL_SECONDS):
                return True
        except (OSError, ValueError):
            pass
        self._fail()
        return False

    def revoke_revision(self, revision: int) -> bool:
        """Publish a revision and await closure within its independent two-second deadline.

        Args:
            revision (int): New monotonic active policy revision.

        Returns:
            bool: Whether the exact revision and covered closure were acknowledged in time.
        """
        with self._lock:
            if self._revoked:
                return self._timely_confirmed and revision == self._pending_revision
            if revision <= self.revision:
                return False
            self._revoked = True
            self._pending_revision = revision
        started = time.monotonic()
        deadline = started + _EGRESS_CONTROL_SECONDS
        try:
            self._send(b"R" + revision.to_bytes(8, "big"))
        except (OSError, ValueError):
            self._fail()
            return False
        if time.monotonic() > deadline:
            self._fail()
            return False
        confirmed = (
            self._acknowledged.wait(max(0.0, deadline - time.monotonic()))
            and self._confirmed
            and self._ack_at is not None
            and self._ack_at <= deadline
        )
        if not confirmed:
            self._fail()
        else:
            with self._lock:
                self._timely_confirmed = True
            self._stopped.set()
        return confirmed

    def healthy(self) -> bool:
        """Reject a mediator whose heartbeat acknowledgements missed two seconds.

        Returns:
            bool: Whether this route is live, acknowledged, and not being revoked.
        """
        return time.monotonic() - self._last_ack < _EGRESS_CONTROL_SECONDS

    def close(self) -> None:
        """Stop heartbeats and release only this attempt's manager binding."""
        self._stopped.set()
        if self._bound:
            self.authority.unbind_egress(self)
            self._bound = False
        if self._thread is not None:
            self._thread.join(_EGRESS_CONTROL_SECONDS)

    def _send(self, frame: bytes) -> None:
        """Write one bounded control frame without allowing a stalled peer to block forever."""
        with self._lock:
            self._write_frame(frame)

    def _write_frame(self, frame: bytes) -> None:
        """Write a control frame while holding the framing lock."""
        connection = self._connection
        if connection is None:
            raise OSError("Mediated egress control socket is unavailable.")
        deadline = time.monotonic() + _EGRESS_CONTROL_SECONDS
        remaining = memoryview(frame)
        while remaining:
            allowance = deadline - time.monotonic()
            if allowance <= 0 or not select.select([], [connection], [], allowance)[1]:
                raise TimeoutError("Mediated egress control send timed out.")
            count = connection.send(remaining)
            if count <= 0:
                raise OSError("Mediated egress control socket closed.")
            remaining = remaining[count:]

    def _watch(self) -> None:
        """Check active managed authority and send authenticated half-second heartbeats."""
        while not self._stopped.wait(_EGRESS_HEARTBEAT_SECONDS):
            try:
                if not self._revoked:
                    current = self.authority.try_egress_policy_revision()
                    if current is not None:
                        self._last_policy_check = time.monotonic()
                        if current != self.revision:
                            self._fail()
                            return
                    elif time.monotonic() - self._last_policy_check >= _EGRESS_CONTROL_SECONDS:
                        self._fail()
                        return
                if self._stopped.is_set():
                    return
                self._send(b"H" + self.revision.to_bytes(8, "big"))
            except (OSError, ValueError):
                self._fail()
                return

    def _fail(self) -> None:
        """Drop the original connection so the receiver's independent boundary closes."""
        self._stopped.set()
        if self._connection is not None:
            try:
                self._connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _remaining(deadline: float) -> float:
    """Require remaining time in the supplied absolute phase budget."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Installed request transport deadline expired.")
    return remaining


def _authority(config: GenerationAdmission, files: dict[str, HostFile], deadline: float) -> None:
    """Revalidate original installed receiver, capacity and exact admission unit bytes."""
    _remaining(deadline)
    sources = config.unit_sources()
    generation_limits(config.reservation, config.generation, deadline=deadline)
    if files["receiver"].path != config.receiver:
        raise ValueError("Transport receiver differs from original generation authority.")
    prefix = NATIVE_UNIT_PREFIX + config.generation
    for bound, name in (
        (config.slice_unit, prefix + SYSTEMD_SLICE_UNIT_SUFFIX),
        (files["socket_unit"], prefix + SYSTEMD_SOCKET_UNIT_SUFFIX),
        (files["service_unit"], prefix + SYSTEMD_SERVICE_TEMPLATE_SUFFIX),
    ):
        _remaining(deadline)
        with bound.open_verified(deadline=deadline) as descriptor:
            if os.pread(descriptor, MAX_NATIVE_UNIT_BYTES + 1, 0) != sources[name].encode():
                raise ValueError("Transport unit differs from original generation authority.")
    for bound in files.values():
        _remaining(deadline)
        with bound.open_verified(deadline=deadline):
            pass
    _remaining(deadline)


def _socket_identity(details: os.stat_result) -> tuple[int, ...]:
    """Bind socket authority without access time, which ordinary connections update."""
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_uid,
        details.st_gid,
        details.st_nlink,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def exchange_request(
    config: GenerationAdmission,
    files: dict[str, HostFile],
    payload: bytes,
    deadline: float,
    *,
    cancellation_event: Event | None = None,
    observer: LinuxDenialObserver | None = None,
    egress_lease: EgressLease | None = None,
    output_observer: Callable[[str, str], None] | None = None,
    terminal_input: Callable[[], bytes | None] | None = None,
) -> bytes:
    """Send one bounded opaque request to the original protected generation listener.

    Call from the ordinary client after authorization, using installation authority supplied
    by the management owner. Authenticate exact unit bytes, fixed local socket ownership and
    inode, and the root systemd listener peer before transmitting any content. Send the existing
    version-one admission frame and read a bounded response through EOF. Cancellable clients keep
    the write side open for one supervisor control byte and wait for the receiver's cleanup result.
    Revalidate installed and socket authority before returning. Do not retry: EOF, cancellation,
    a deadline or transport error does not establish whether command effects occurred.

    Responses use the fixed shared result wire ceiling, large enough for the common supervisor's
    two bounded output streams even with worst-case JSON escaping. Request frames retain their
    separate 64 KiB ceiling. This transport neither parses outcomes nor grants execution.

    Args:
        config (GenerationAdmission): Original fixed installed generation and ordinary identity.
        files (dict[str, HostFile]): Original authenticated receipt/admission file bindings.
        payload (bytes): Nonempty already authorized opaque frame, at most 64 KiB.
        deadline (float): Absolute monotonic execution budget for connection and send; response
            delivery
            has a fixed additional five-second cleanup/delivery interval.
        cancellation_event (Event | None): Optional thread-safe run cancellation signal. When set,
            the client requests process cancellation and waits for the receiver's cleanup result.
        observer (LinuxDenialObserver | None): Optional unprivileged denial observation owner.
        egress_lease (EgressLease | None): Exact approval lease for a mediated attempt; absent
            for the original unmodified result wire protocol.
        output_observer (Callable[[str, str], None] | None): Optional receiver output stream
            callback; each frame is bounded and precedes the final typed result.
        terminal_input (Callable[[], bytes | None] | None): Poll one bounded transient terminal
            control frame after the authenticated receiver confirms payload entry.

    Returns:
        bytes: Nonempty response within MAX_NATIVE_RESULT_BYTES, without interpretation or
            disclosure.

    Raises:
        ValueError: Payload, caller, installed/socket/peer authority or response bounds differ.
        TimeoutError: The execution or bounded cleanup/delivery budget expires.
        OSError: Protected authority, socket connection or bounded I/O fails.
    """
    if not math.isfinite(deadline):
        raise ValueError("Installed request transport requires a finite deadline.")
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_FRAME_BYTES:
        raise ValueError("Invalid bounded installed request payload.")
    if (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (
        config.uid,
        config.uid,
        config.gid,
        config.gid,
    ):
        raise ValueError("Transport requires the fixed ordinary generation identity.")
    _authority(config, files, deadline)
    path = SYSTEM_RUNTIME_DIRECTORY / (
        NATIVE_UNIT_PREFIX + config.generation + SYSTEMD_SOCKET_UNIT_SUFFIX
    )
    if protected_path(path.parent, directory=True) != path.parent:
        raise ValueError("Transport socket parent is aliased.")
    original = path.lstat()
    if (
        not stat.S_ISSOCK(original.st_mode)
        or (original.st_uid, original.st_gid) != (config.uid, config.gid)
        or stat.S_IMODE(original.st_mode) != NATIVE_ADMISSION_SOCKET_MODE
    ):
        raise ValueError("Transport socket differs from fixed ordinary listener authority.")
    identity = _socket_identity(original)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(min(_remaining(deadline), threading.TIMEOUT_MAX))
        connection.connect(str(path))
        pid, uid, gid = struct.unpack(
            PEER_CREDENTIALS_FORMAT,
            connection.getsockopt(socket.SOL_SOCKET, _SO_PEERCRED_OPTION, PEER_CREDENTIALS_SIZE),
        )
        if pid <= 0 or (uid, gid) != (0, 0):
            raise ValueError("Transport peer is not the protected system listener.")
        current = path.lstat()
        if _socket_identity(current) != identity:
            raise ValueError("Transport socket changed before transmission.")
        _authority(config, files, deadline)
        if observer is not None:
            observer.start(connection, config.generation)
        connection.settimeout(min(_remaining(deadline), threading.TIMEOUT_MAX))
        connection.sendall(
            NATIVE_REQUEST_MAGIC + struct.pack(REQUEST_LENGTH_FORMAT, len(payload)) + payload
        )
        if cancellation_event is None and egress_lease is None and terminal_input is None:
            connection.shutdown(socket.SHUT_WR)
        delivery_deadline = deadline + NATIVE_RESULT_DELIVERY_SECONDS
        response = bytearray()
        controls = bytearray()
        ready = False
        terminal_ready = False
        cancel_sent = False
        try:
            while True:
                remaining = _remaining(delivery_deadline)
                if terminal_ready and terminal_input is not None and not response:
                    frame = terminal_input()
                    if frame is not None:
                        connection.settimeout(min(remaining, 1.0))
                        connection.sendall(frame)
                connection.settimeout(
                    min(remaining, 0.05)
                    if (cancellation_event is not None and not response and not cancel_sent)
                    or egress_lease is not None
                    or terminal_input is not None
                    else min(remaining, threading.TIMEOUT_MAX)
                )
                try:
                    block = connection.recv(
                        min(_RESPONSE_READ_CHUNK_BYTES, MAX_NATIVE_RESULT_BYTES + 1 - len(response))
                    )
                except TimeoutError:
                    if egress_lease is not None and ready and not egress_lease.healthy():
                        raise OSError("Mediated egress health acknowledgement timed out.") from None
                    if (
                        cancellation_event is None
                        and egress_lease is None
                        and terminal_input is None
                        and remaining <= threading.TIMEOUT_MAX
                    ):
                        raise
                    if (
                        cancellation_event is not None
                        and cancellation_event.is_set()
                        and not response
                        and not cancel_sent
                    ):
                        connection.sendall(NATIVE_CANCEL_FRAME)
                        cancel_sent = True
                    continue
                if not block:
                    break
                if (
                    egress_lease is not None
                    or output_observer is not None
                    or terminal_input is not None
                ) and not response:
                    controls.extend(block)
                    while controls:
                        marker = controls[0]
                        if marker == ord("O") and output_observer is not None:
                            if len(controls) < 6:
                                break
                            size = int.from_bytes(controls[2:6], "big")
                            if controls[1] not in (ord("1"), ord("2")) or not (
                                0 < size <= _OUTPUT_FRAME_MAX_BYTES
                            ):
                                raise ValueError("Invalid native output frame header.")
                            if len(controls) < 6 + size:
                                break
                            try:
                                chunk = controls[6 : 6 + size].decode("utf-8")
                            except UnicodeError as error:
                                raise ValueError("Invalid native output frame text.") from error
                            output_observer(
                                "stdout" if controls[1] == ord("1") else "stderr", chunk
                            )
                            del controls[: 6 + size]
                        elif marker == ord("E") and not ready and egress_lease is not None:
                            del controls[:1]
                            egress_lease.ready(connection)
                            ready = True
                        elif (
                            marker == ord("T") and not terminal_ready and terminal_input is not None
                        ):
                            del controls[:1]
                            terminal_ready = True
                        elif marker == ord("h") and ready and egress_lease is not None:
                            if len(controls) < 9:
                                break
                            egress_lease.heartbeat_ack(int.from_bytes(controls[1:9], "big"))
                            del controls[:9]
                        elif marker in (ord("p"), ord("u")) and ready and egress_lease is not None:
                            if len(controls) < 9:
                                break
                            revision = int.from_bytes(controls[1:9], "big")
                            if marker == ord("p"):
                                egress_lease.pause_ack(revision)
                            else:
                                egress_lease.resume_ack(revision)
                            del controls[:9]
                        elif marker == ord("A") and ready and egress_lease is not None:
                            if len(controls) < 10:
                                break
                            if controls[9] not in (ord("0"), ord("1")):
                                raise ValueError(
                                    "Invalid mediated egress revision acknowledgement."
                                )
                            egress_lease.revision_ack(
                                int.from_bytes(controls[1:9], "big"), controls[9] == ord("1")
                            )
                            del controls[:10]
                        elif marker == ord("{"):
                            response.extend(controls)
                            controls.clear()
                        else:
                            raise ValueError("Invalid native control or output response.")
                else:
                    response.extend(block)
                if len(response) > MAX_NATIVE_RESULT_BYTES:
                    raise ValueError("Installed response exceeds the transport bound.")
        finally:
            if egress_lease is not None:
                egress_lease.close()
        if controls or not response:
            raise ValueError("Installed receiver returned no response.")
        if _socket_identity(path.lstat()) != identity:
            raise ValueError("Transport socket changed during exchange.")
        _authority(config, files, delivery_deadline)
        return bytes(response)
