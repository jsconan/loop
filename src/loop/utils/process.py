"""Provide safe process invocation utilities."""

import ctypes
import errno
import logging
import os
import select
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from .. import constants

_LOGGER = logging.getLogger(__name__)


class TextStream(Protocol):
    """Represent a synchronously readable text stream."""

    def read(self, size: int = -1) -> str:
        """Read at most ``size`` characters from the stream."""

    def close(self) -> None:
        """Release the stream wrapper after its reader exits."""


class ProcessCaptureStatus(StrEnum):
    """Classify a supervised process independently of its exit code."""

    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ProcessCapture:
    """Hold bounded child output and supervision outcome.

    Args:
        status (ProcessCaptureStatus): Lifecycle result.
        exit_code (int | None): Child exit status on completion.
        stdout (str): Captured standard output.
        stderr (str): Captured standard error.
        stdout_discarded (int | None): Characters discarded, or unknown if draining stalled.
        stderr_discarded (int | None): Characters discarded, or unknown if draining stalled.
    """

    status: ProcessCaptureStatus
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_discarded: int | None = 0
    stderr_discarded: int | None = 0


_SHELL_SYNTAX = frozenset("|;&<>`()$")
_PROCESS_WAIT_POLL_SECONDS = 0.01
_MACOS_PROCESS_GROUP_PID_CAPACITY = 65536
_MACOS_PROCESS_GROUP_INSPECTION_PASSES = 8
_MACOS_PROC_PIDTBSDINFO = 3
_MACOS_PROC_BSDINFO_STATUS_OFFSET = 4
_MACOS_ZOMBIE_PROCESS_STATUS = 5  # SZOMB in sys/proc.h.


def parse_command_line(command: str) -> tuple[str, ...]:  # pylint: disable=too-many-branches
    """Parse a restricted command line into an exact argument vector.

    Args:
        command (str): Command text with whitespace separators, quotes, and backslash escapes.

    Returns:
        tuple[str, ...]: Non-empty executable and argument vector.

    Raises:
        ValueError: The command is empty, contains unquoted shell syntax, or has incomplete
            quoting or escaping.
    """
    argv = []
    characters = []
    quote = None
    token_started = False
    index = 0
    while index < len(command):
        character = command[index]
        if quote is not None:
            if character == quote:
                quote = None
            elif character == "\\" and quote == '"':
                index += 1
                if index == len(command):
                    raise ValueError("Command ends with an incomplete escape sequence.")
                characters.append(command[index])
            else:
                characters.append(character)
            index += 1
            continue
        if character.isspace():
            if token_started:
                argv.append("".join(characters))
                characters = []
                token_started = False
            index += 1
            continue
        if character in "'\"":
            quote = character
            token_started = True
        elif character == "\\":
            index += 1
            if index == len(command):
                raise ValueError("Command ends with an incomplete escape sequence.")
            characters.append(command[index])
            token_started = True
        elif character in _SHELL_SYNTAX:
            raise ValueError(
                "Command contains unquoted shell syntax; quote or escape it when it is literal "
                "argument data."
            )
        else:
            characters.append(character)
            token_started = True
        index += 1
    if quote is not None:
        raise ValueError("Command contains an unterminated quoted argument.")
    if token_started:
        argv.append("".join(characters))
    if not argv:
        raise ValueError("Command must include an executable.")
    return tuple(argv)


def read_bounded_stream(
    stream: TextStream,
    chunks: list[str],
    *,
    chunk_size: int = constants.DEFAULT_STREAM_CHUNK_SIZE,
    maximum: int = constants.MAX_OUTPUT_CHARS,
) -> int:
    """Drain and close a text stream while retaining the requested character limit.

    Args:
        stream (TextStream): Stream drained until its ``read`` method returns an empty string.
        chunks (list[str]): Destination receiving retained text chunks.
        chunk_size (int, optional): Number of characters to read per chunk.
            Defaults to ``constants.DEFAULT_STREAM_CHUNK_SIZE``.
        maximum (int, optional): Maximum total characters retained in ``chunks``.
            Defaults to ``constants.MAX_OUTPUT_CHARS``.

    Returns:
        int: Number of characters discarded after the capture limit.
    """
    remaining = maximum
    discarded = 0
    try:
        while True:
            try:
                chunk = stream.read(chunk_size)
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise
                break
            if not chunk:
                break
            discarded += max(0, len(chunk) - remaining)
            if remaining:
                chunks.append(chunk[:remaining])
                remaining -= len(chunk[:remaining])
    finally:
        try:
            stream.close()
        except OSError as exc:
            if exc.errno != errno.EBADF:
                raise
    return discarded


def kill_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate an owned process group before its leader is reaped.

    POSIX callers must create the process with ``start_new_session=True`` so the stored PID is
    also the owned process-group ID. Keep the leader waitable until this signal completes so its
    numeric group ID cannot be reused. Python's portable Windows process API cannot forcibly
    terminate a complete descendant tree, so that platform falls back to the direct process. On
    macOS, a denied group signal is ignored only when bounded process inspection confirms that
    every member is already inactive.

    Args:
        process (subprocess.Popen[str]): Running process to terminate.

    Raises:
        OSError: Group signaling fails while a live member remains or group inspection is
            unavailable.
    """
    if os.name != "posix":
        process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        if sys.platform != "darwin":
            raise
        try:
            if not _macos_process_group_has_live_members(process.pid):
                return
        except OSError as inspection_error:
            raise exc from inspection_error
        raise


def _macos_process_group_has_live_members(process_group_id: int) -> bool:
    """Check a macOS process group for live members after a denied group signal.

    Repeat bounded process-list snapshots until every observed member is inactive and a fresh
    snapshot adds no new members. This avoids treating a descendant that forked during inspection
    as an empty group; inspection errors fail closed.
    """
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    list_group = library.proc_listpgrppids
    list_group.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    list_group.restype = ctypes.c_int
    process_info = library.proc_pidinfo
    process_info.argtypes = (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    )
    process_info.restype = ctypes.c_int

    checked_pids = set()
    for _ in range(_MACOS_PROCESS_GROUP_INSPECTION_PASSES):
        pids = (ctypes.c_int * _MACOS_PROCESS_GROUP_PID_CAPACITY)()
        ctypes.set_errno(0)
        count = list_group(process_group_id, pids, ctypes.sizeof(pids))
        if count < 0:
            error_number = ctypes.get_errno() or errno.EIO
            raise OSError(error_number, os.strerror(error_number))
        if count >= _MACOS_PROCESS_GROUP_PID_CAPACITY:
            raise OSError(errno.EOVERFLOW, "Process group member list exceeded its bound.")

        for process_id in pids[:count]:
            if process_id in checked_pids:
                continue
            info = ctypes.create_string_buffer(256)
            ctypes.set_errno(0)
            info_size = process_info(
                process_id,
                _MACOS_PROC_PIDTBSDINFO,
                0,
                info,
                ctypes.sizeof(info),
            )
            error_number = ctypes.get_errno()
            if info_size <= 0:
                if error_number == errno.ESRCH:
                    checked_pids.add(process_id)
                    continue
                error_number = error_number or errno.EIO
                raise OSError(error_number, os.strerror(error_number))
            if info_size < _MACOS_PROC_BSDINFO_STATUS_OFFSET + ctypes.sizeof(ctypes.c_uint32):
                raise OSError(errno.EIO, "Could not inspect process group member status.")
            status = ctypes.c_uint32.from_buffer(info, _MACOS_PROC_BSDINFO_STATUS_OFFSET).value
            if status != _MACOS_ZOMBIE_PROCESS_STATUS:
                return True
            checked_pids.add(process_id)

        next_pids = (ctypes.c_int * _MACOS_PROCESS_GROUP_PID_CAPACITY)()
        ctypes.set_errno(0)
        next_count = list_group(process_group_id, next_pids, ctypes.sizeof(next_pids))
        if next_count < 0:
            error_number = ctypes.get_errno() or errno.EIO
            raise OSError(error_number, os.strerror(error_number))
        if next_count >= _MACOS_PROCESS_GROUP_PID_CAPACITY:
            raise OSError(errno.EOVERFLOW, "Process group member list exceeded its bound.")
        if set(next_pids[:next_count]).issubset(checked_pids):
            return False

    raise OSError(errno.EAGAIN, "Process group remained active during bounded inspection.")


def _wait_for_process_exit(
    process: subprocess.Popen[str],
    deadline: float,
    cancel_event: threading.Event | None,
) -> ProcessCaptureStatus:
    """Observe process exit without reaping its group leader before owned-group cleanup."""
    if os.name != "posix":
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return ProcessCaptureStatus.CANCELLED
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ProcessCaptureStatus.TIMED_OUT
            try:
                process.wait(timeout=min(remaining, _PROCESS_WAIT_POLL_SECONDS))
                return ProcessCaptureStatus.COMPLETED
            except subprocess.TimeoutExpired:
                continue

    if callable(getattr(os, "waitid", None)) and all(
        hasattr(os, name) for name in ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
    ):
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return ProcessCaptureStatus.CANCELLED
            result = os.waitid(
                os.P_PID,
                process.pid,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
            if result is not None and result.si_pid == process.pid:
                return ProcessCaptureStatus.COMPLETED
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ProcessCaptureStatus.TIMED_OUT
            time.sleep(min(remaining, _PROCESS_WAIT_POLL_SECONDS))

    if hasattr(select, "kqueue"):
        queue = select.kqueue()
        try:
            event = select.kevent(
                process.pid,
                filter=select.KQ_FILTER_PROC,
                flags=select.KQ_EV_ADD | select.KQ_EV_ENABLE,
                fflags=select.KQ_NOTE_EXIT,
            )
            queue.control([event], 0, 0)
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return ProcessCaptureStatus.CANCELLED
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return ProcessCaptureStatus.TIMED_OUT
                if queue.control(
                    None,
                    1,
                    min(remaining, _PROCESS_WAIT_POLL_SECONDS),
                ):
                    return ProcessCaptureStatus.COMPLETED
        except ProcessLookupError:
            # The unpolled Popen child is still waitable, so its PID cannot have been reused.
            return ProcessCaptureStatus.COMPLETED
        finally:
            queue.close()

    raise OSError("POSIX process exit cannot be observed without reaping its leader.")


def supervise_process(
    process: subprocess.Popen[str],
    deadline: float,
    *,
    stdout_limit: int = constants.MAX_OUTPUT_CHARS,
    stderr_limit: int = constants.MAX_OUTPUT_CHARS,
    terminate: Callable[[subprocess.Popen[str]], None] = kill_process_group,
    reader: Callable[..., int] = read_bounded_stream,
    thread_factory: Callable[..., threading.Thread] = threading.Thread,
    cancel_event: threading.Event | None = None,
) -> ProcessCapture:
    """Drain bounded child output and capture the result after owned-group cleanup.

    Observe leader exit without reaping it, then terminate and reap the group before taking the
    output snapshot. This keeps the process-group identity pinned through cleanup. Give readers
    a bounded cleanup interval to consume the resulting pipe EOF; incomplete draining retains
    unknown discard counts. Never extend the execution deadline.

    Args:
        process (subprocess.Popen[str]): Child started in its own process group.
        deadline (float): Absolute monotonic deadline for exit and complete pipe draining.
        stdout_limit (int): Maximum retained standard-output characters.
        stderr_limit (int): Maximum retained standard-error characters.
        terminate (Callable[[subprocess.Popen[str]], None]): Process-group terminator.
        reader (Callable[..., int]): Bounded stream reader.
        thread_factory (Callable[..., threading.Thread]): Reader-thread constructor.
        cancel_event (threading.Event | None): Optional event that requests cancellation before
            the process leader is reaped.

    Returns:
        ProcessCapture: Exit, timeout, cancellation, or missing-pipe result.
    """
    if process.stdout is None or process.stderr is None:
        try:
            terminate(process)
        finally:
            _wait_process(process)
        return ProcessCapture(ProcessCaptureStatus.UNAVAILABLE)

    streams = (process.stdout, process.stderr)
    chunks = ([], [])
    discarded = [0, 0]
    readers = [
        thread_factory(
            target=lambda index=i: discarded.__setitem__(
                index,
                reader(
                    streams[index],
                    chunks[index],
                    maximum=stdout_limit if index == 0 else stderr_limit,
                ),
            ),
            daemon=True,
        )
        for i in range(2)
    ]

    for thread in readers:
        thread.start()

    result = ProcessCapture(ProcessCaptureStatus.UNAVAILABLE)
    try:
        wait_status = _wait_for_process_exit(process, deadline, cancel_event)
        if wait_status is ProcessCaptureStatus.CANCELLED:
            result = ProcessCapture(ProcessCaptureStatus.CANCELLED)
        elif wait_status is ProcessCaptureStatus.TIMED_OUT:
            result = ProcessCapture(
                ProcessCaptureStatus.TIMED_OUT,
                stdout="".join(chunks[0]),
                stderr="".join(chunks[1]),
                stdout_discarded=None,
                stderr_discarded=None,
            )
        else:
            for thread in readers:
                thread.join(max(0, deadline - time.monotonic()))

            if any(thread.is_alive() for thread in readers):
                result = ProcessCapture(
                    ProcessCaptureStatus.TIMED_OUT,
                    stdout="".join(chunks[0]),
                    stderr="".join(chunks[1]),
                    stdout_discarded=discarded[0] if not readers[0].is_alive() else None,
                    stderr_discarded=discarded[1] if not readers[1].is_alive() else None,
                )
            else:
                result = ProcessCapture(
                    ProcessCaptureStatus.COMPLETED,
                    stdout="".join(chunks[0]),
                    stderr="".join(chunks[1]),
                    stdout_discarded=discarded[0],
                    stderr_discarded=discarded[1],
                )
    except KeyboardInterrupt:
        result = ProcessCapture(ProcessCaptureStatus.CANCELLED)
    finally:
        cleanup_error = None
        try:
            if os.name == "posix" or result.status is not ProcessCaptureStatus.COMPLETED:
                terminate(process)
        except (OSError, KeyboardInterrupt) as exc:
            cleanup_error = exc
        try:
            _wait_process(process)
        except (OSError, KeyboardInterrupt) as exc:
            if cleanup_error is None:
                cleanup_error = exc

        for thread in readers:
            thread.join(constants.COMMAND_CLEANUP_WAIT_SECONDS)

        for thread, stream in zip(readers, streams, strict=True):
            if thread.is_alive():
                try:
                    os.close(stream.fileno())
                except (OSError, TypeError, ValueError):
                    _LOGGER.warning("Could not close a process output stream during cleanup.")
            else:
                stream.close()
        if cleanup_error is not None:
            raise cleanup_error

    if result.status is ProcessCaptureStatus.COMPLETED and process.returncode is not None:
        result = replace(result, exit_code=process.returncode)
    return replace(
        result,
        stdout="".join(chunks[0]),
        stderr="".join(chunks[1]),
        stdout_discarded=discarded[0] if not readers[0].is_alive() else None,
        stderr_discarded=discarded[1] if not readers[1].is_alive() else None,
    )


def _wait_process(
    process: subprocess.Popen[str],
    timeout: float = constants.COMMAND_CLEANUP_WAIT_SECONDS,
):
    """Wait for the given process to complete within the cleanup wait time."""
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _LOGGER.warning("Process %s did not exit within the cleanup wait time.", process.pid)
