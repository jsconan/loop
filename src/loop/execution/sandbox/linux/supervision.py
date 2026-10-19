"""Supervise an admitted PID namespace and refuse unverified descendant cleanup.

The trusted receiver must launch authenticated Bubblewrap with a private PID namespace,
--die-with-parent and a new session, before any payload runs. No process-group signal alone
proves cleanup. This owner checks protected kernel cgroup membership after the common
supervisor terminates the owned launcher. It does not launch, authorize or enable Linux.
"""

import os
import select
import subprocess
import threading
import time
from dataclasses import replace

from ....constants import MAX_OUTPUT_CHARS
from ....utils.process import (
    ProcessCapture,
    ProcessCaptureStatus,
    supervise_process,
)
from .admission import GenerationAdmission, admission_group
from .constants import CGROUP_PROCS_FILENAME, NATIVE_CANCEL_FRAME

_CLEANUP_SECONDS = 1.0
_CLIENT_DISCONNECT_POLL_MILLISECONDS = 50
_CLEANUP_POLL_SECONDS = 0.01
_MAX_CGROUP_MEMBERSHIP_BYTES = 65_536


def supervise_payload(
    process: subprocess.Popen[str],
    config: GenerationAdmission,
    deadline: float,
    *,
    connection: int | None = None,
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

    def watch():
        """Cancel the owned launcher on a bound control byte or full client disconnect."""
        try:
            poller = select.poll()
            disconnected = select.POLLHUP | select.POLLERR | select.POLLNVAL
            poller.register(connection, select.POLLIN | disconnected)
            while not stopped.is_set():
                events = poller.poll(_CLIENT_DISCONNECT_POLL_MILLISECONDS)
                if not events:
                    continue
                mask = events[0][1]
                if mask & select.POLLIN:
                    control = os.read(connection, len(NATIVE_CANCEL_FRAME))
                    if control == NATIVE_CANCEL_FRAME:
                        cancelled.set()
                        return
                    if control:
                        failed.set()
                        cancelled.set()
                        return
                    poller.modify(connection, disconnected)
                if mask & disconnected:
                    cancelled.set()
                    return
        except (OSError, ValueError):
            failed.set()
            cancelled.set()

    watcher = threading.Thread(target=watch, daemon=True) if connection is not None else None
    if watcher is not None:
        watcher.start()
    try:
        result = supervise_process(
            process,
            deadline,
            stdout_limit=MAX_OUTPUT_CHARS + 80,
            cancel_event=cancelled,
        )
    finally:
        stopped.set()
        if watcher is not None:
            watcher.join(_CLEANUP_SECONDS)
            if watcher.is_alive():
                raise OSError("Client cancellation watcher did not stop.")
    if cancelled.is_set():
        result = replace(result, status=ProcessCaptureStatus.CANCELLED, exit_code=None)
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
