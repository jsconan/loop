"""Execute one authenticated installed request through the trusted native boundary."""

import os
import socket
import stat
import sys
import time
from pathlib import Path
from typing import Any

_UNSAFE_HOST_PATH_MODE_BITS = 0o6022


def _validate_installed_entry() -> Path:
    """Require the fixed entry and administrator-protected ancestry for an ordinary service."""
    entry = Path(__file__).absolute()
    if entry.name != "loop_receiver.py" or entry.resolve(strict=True) != entry:
        raise ValueError("Receiver requires its fixed canonical installed path.")

    for path in (entry, *entry.parents):
        details = path.lstat()
        if (
            details.st_uid != 0
            or details.st_mode & _UNSAFE_HOST_PATH_MODE_BITS
            or stat.S_ISLNK(details.st_mode)
        ):
            raise PermissionError("Receiver code ancestry is not administrator protected.")

    if os.getuid() == 0:
        raise PermissionError("Receiver requires the fixed ordinary service identity.")
    return entry


def _receive_request(entry: Path) -> bytes:
    """Admit one bounded request after validating retained code and native membership."""
    from loop.execution.sandbox.linux.admission import ADMISSION_SECONDS
    from loop.execution.sandbox.linux.runtime import receive_installed_request

    return receive_installed_request(
        entry.parent / "management.json",
        time.monotonic() + ADMISSION_SECONDS,
    )


def _execute_request(entry: Path, payload: bytes) -> tuple[object, object]:
    """Restore an admitted request and retain its authenticated timeout result."""
    from loop.execution.sandbox.contracts import (
        CommandProcessResult,
        SandboxOutcome,
        SandboxRequest,
        SandboxRequestExpired,
    )
    from loop.execution.sandbox.linux.runtime import execute_installed_request

    try:
        request = SandboxRequest.from_wire(payload)
    except SandboxRequestExpired as error:
        request = error.request
        result = CommandProcessResult(SandboxOutcome.TIMED_OUT)
    else:
        try:
            result = execute_installed_request(entry.parent / "management.json", payload)
        except TimeoutError:
            # Preparation did not produce trustworthy capture or a descendant-cleanup result.
            result = CommandProcessResult(
                SandboxOutcome.TIMED_OUT,
                stdout_discarded=None,
                stderr_discarded=None,
                possible_effects=True,
            )
    return request, result


def _deliver_result(request: Any, result: Any, delivery_seconds: float) -> None:
    """Send one bounded attempt-authenticated result within the fixed delivery allowance."""
    data = result.to_wire(request.attempt_id)
    remaining = request.deadline + delivery_seconds - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Native result delivery deadline expired.")
    # sendall handles short writes on the shared nonblocking socket within a separate fixed limit.
    with socket.fromfd(1, socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(remaining)
        connection.sendall(data)


def main() -> None:
    """Require protected retained code and ordinary admission before one supervised execution.

    Receive under installed peer, receipt, framing and immutable membership checks with a
    separate five-second admission budget. Restore the exact shared snapshot and original
    deadline, then compose strict compilation, trusted namespace entry, payload privilege
    removal and authoritative descendant cleanup. Return only the bounded attempt-bound result.
    Invalid setup never offers a host retry or discloses compiled/control authority.

    Raises:
        PermissionError: The entry is outside protected installed ancestry or runs as root.
        ValueError: The fixed entry, installed record or admission authority differs.
        OSError: Bound state, peer, frame, membership or native reads fail.
        TimeoutError: The separate bounded cleanup/result-delivery deadline expires.
    """
    entry = _validate_installed_entry()
    # Load retained modules only after proving immutable code ancestry.
    sys.path.insert(0, str(entry.parent))
    payload = _receive_request(entry)
    from loop.constants import NATIVE_RESULT_DELIVERY_SECONDS

    request, result = _execute_request(entry, payload)
    _deliver_result(request, result, NATIVE_RESULT_DELIVERY_SECONDS)


if __name__ == "__main__":
    main()
