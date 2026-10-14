"""Exchange one opaque frame with an authenticated installed ordinary receiver.

Transport supplies no command authorization or execution readiness. The receiver supplies bounded
attempt-bound typed results;
ordinary file preflight does not certify live kernel readiness. Trusted namespace entry verifies
actual enforcing bootstrap/cache exports before source starts.
"""

import math
import os
import socket
import stat
import struct
import time

from ....constants import (
    MAX_NATIVE_RESULT_BYTES,
    MAX_NATIVE_UNIT_BYTES,
    NATIVE_ADMISSION_SOCKET_MODE,
    NATIVE_RESULT_DELIVERY_SECONDS,
    NATIVE_UNIT_PREFIX,
)
from .admission import MAX_FRAME_BYTES, GenerationAdmission
from .constants import (
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

_SO_PEERCRED_OPTION = getattr(socket, "SO_PEERCRED", SO_PEERCRED_OPTION_FALLBACK)
_RESPONSE_READ_CHUNK_BYTES = 4096


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
) -> bytes:
    """Send one bounded opaque request to the original protected generation listener.

    Call from the ordinary client after authorization, using installation authority supplied
    by the management owner. Authenticate exact unit bytes, fixed local socket ownership and
    inode, and the root systemd listener peer before transmitting any content. Send the existing
    version-one admission frame and half-close, then read a bounded response through EOF.
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
        connection.settimeout(_remaining(deadline))
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
        connection.settimeout(_remaining(deadline))
        connection.sendall(
            NATIVE_REQUEST_MAGIC + struct.pack(REQUEST_LENGTH_FORMAT, len(payload)) + payload
        )
        connection.shutdown(socket.SHUT_WR)
        delivery_deadline = deadline + NATIVE_RESULT_DELIVERY_SECONDS
        response = bytearray()
        while True:
            connection.settimeout(_remaining(delivery_deadline))
            block = connection.recv(
                min(_RESPONSE_READ_CHUNK_BYTES, MAX_NATIVE_RESULT_BYTES + 1 - len(response))
            )
            if not block:
                break
            response.extend(block)
            if len(response) > MAX_NATIVE_RESULT_BYTES:
                raise ValueError("Installed response exceeds the transport bound.")
        if not response:
            raise ValueError("Installed receiver returned no response.")
        if _socket_identity(path.lstat()) != identity:
            raise ValueError("Transport socket changed during exchange.")
        _authority(config, files, delivery_deadline)
        return bytes(response)
