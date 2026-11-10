"""Provide safe process invocation utilities."""

import codecs
import ctypes
import errno
import io
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
from typing import Protocol, cast

from .. import constants

_LOGGER = logging.getLogger(__name__)


def is_finite_deadline(value: object) -> bool:
    """Check whether a deadline is a finite, representable numeric value.

    Args:
        value (object): Candidate monotonic deadline or duration.

    Returns:
        bool: Whether the value is a non-boolean integer or float representable as a float.
    """
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and -sys.float_info.max <= value <= sys.float_info.max
    )


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


def render_terminal_text(value: str) -> str:
    """Render untrusted terminal controls visibly without executing ANSI or OSC sequences.

    Args:
        value (str): Untrusted text that may contain terminal control characters.

    Returns:
        str: Equal-length text with C0 and DEL shown as control pictures, C1 replaced, and
            line feeds preserved for ordinary reading.
    """
    return "".join(
        "\ufffd"
        if 0x80 <= ord(character) <= 0x9F
        else "\u2421"
        if character == "\x7f"
        else chr(0x2400 + ord(character))
        if ord(character) < 0x20 and character != "\n"
        else character
        for character in value
    )


@dataclass(frozen=True)
class ProcessCapture:
    """Hold bounded child output and supervision outcome.

    Args:
        status (ProcessCaptureStatus): Lifecycle result, or unavailable if capture fails.
        exit_code (int | None): Child exit status on completion.
        stdout (str): Captured standard output.
        stderr (str): Captured standard error.
        stdout_discarded (int | None): Characters discarded, or unknown if draining failed.
        stderr_discarded (int | None): Characters discarded, or unknown if draining failed.
    """

    status: ProcessCaptureStatus
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_discarded: int | None = 0
    stderr_discarded: int | None = 0


class CapturedChunks(list[str]):
    """Retain bounded stream chunks and publish live output to an optional observer.

    Args:
        stream (str): Stream name supplied with each observed chunk.
        observer (Callable[[str, str], None] | None): Recipient of live decoded chunks.
        stop (threading.Event | None): Cooperative reader shutdown signal, when supervised.
        preserve_newlines (bool): Whether to preserve native newline characters.
    """

    _stream: str
    observer: Callable[[str, str], None] | None
    stop: threading.Event | None
    preserve_newlines: bool

    def __init__(
        self,
        stream: str,
        observer: Callable[[str, str], None] | None,
        stop: threading.Event | None = None,
        preserve_newlines: bool = False,
    ) -> None:
        super().__init__()
        self._stream = stream
        self.observer = observer
        self.stop = stop
        self.preserve_newlines = preserve_newlines

    def observe(self, value: str) -> None:
        """Publish a decoded chunk independently of the final capture limit.

        Args:
            value (str): Decoded output to send to the observer.
        """
        if self.observer is not None:
            self.observer(self._stream, value)


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
    eof_errno: int | None = None,
) -> int | None:
    """Drain and close a text stream while retaining the requested character limit.

    Args:
        stream (TextStream): Stream drained until its ``read`` method returns an empty string.
        chunks (list[str]): Destination receiving retained text chunks.
        chunk_size (int, optional): Number of characters to read per chunk.
            Defaults to ``constants.DEFAULT_STREAM_CHUNK_SIZE``.
        maximum (int, optional): Maximum total characters retained in ``chunks``.
            Defaults to ``constants.MAX_OUTPUT_CHARS``.
        eof_errno (int | None): Native PTY read error that means its last slave closed. Ordinary
            pipes leave this unset so an I/O error never becomes a complete capture.

    Returns:
        int | None: Number of characters discarded after the capture limit, or ``None`` when
            a closed descriptor prevents a complete drain.

    Raises:
        OSError: Reading or closing fails for a reason other than a closed descriptor.
    """
    remaining = maximum
    discarded = 0
    incomplete = False
    try:
        buffer = getattr(stream, "buffer", None)
        supervised = (
            isinstance(chunks, CapturedChunks)
            and chunks.stop is not None
            and os.name == "posix"
            and isinstance(stream, io.TextIOWrapper)
            and isinstance(buffer, io.BufferedIOBase)
        )
        descriptor = None
        if supervised:
            try:
                descriptor = stream.fileno()
            except (OSError, ValueError):
                supervised = False
        read_available = (
            getattr(buffer, "read1", None)
            if isinstance(chunks, CapturedChunks)
            and (chunks.observer is not None or supervised)
            and isinstance(buffer, io.BufferedIOBase)
            else None
        )
        decoder = (
            codecs.getincrementaldecoder(stream.encoding or "utf-8")(
                errors=stream.errors or "replace"
            )
            if callable(read_available)
            else None
        )
        if decoder is not None and supervised and chunks.observer is None:
            decoder = io.IncrementalNewlineDecoder(decoder, not chunks.preserve_newlines)
        poller = None
        if supervised:
            poller = select.poll()
            poller.register(descriptor, select.POLLIN | select.POLLHUP | select.POLLERR)
        while True:
            if poller is not None:
                if chunks.stop.is_set():
                    incomplete = True
                    break
                if not poller.poll(50):
                    continue
            try:
                if decoder is None:
                    chunk = stream.read(chunk_size)
                else:
                    data = read_available(chunk_size)
                    chunk = decoder.decode(data, final=not data)
            except OSError as exc:
                if eof_errno is not None and exc.errno == eof_errno:
                    if decoder is not None:
                        tail = decoder.decode(b"", final=True)
                        if tail:
                            discarded += max(0, len(tail) - remaining)
                            if remaining:
                                chunks.append(tail[:remaining])
                            cast(CapturedChunks, chunks).observe(tail)
                    break
                if exc.errno != errno.EBADF:
                    raise
                incomplete = True
                break
            if not chunk:
                if decoder is not None and data:
                    continue
                break
            discarded += max(0, len(chunk) - remaining)
            if remaining:
                chunks.append(chunk[:remaining])
                remaining -= len(chunk[:remaining])
            if isinstance(chunks, CapturedChunks):
                chunks.observe(chunk)
    finally:
        try:
            stream.close()
        except OSError as exc:
            if exc.errno != errno.EBADF:
                raise
            incomplete = True
    return None if incomplete else discarded


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
    reader: Callable[..., int | None] = read_bounded_stream,
    thread_factory: Callable[..., threading.Thread] = threading.Thread,
    cancel_event: threading.Event | None = None,
    preserve_newlines: bool = False,
    output_observer: Callable[[str, str], None] | None = None,
) -> ProcessCapture:
    """Drain bounded child output and attempt owned-group cleanup before returning.

    Observe leader exit without reaping it, then signal the group and wait for its leader before
    taking the output snapshot. The leader remains waitable until signaling is attempted. A
    bounded wait can expire without confirming that the leader was reaped. Give readers
    a bounded cleanup interval to consume the resulting pipe EOF; incomplete draining retains
    unknown discard counts. A reader failure makes an otherwise completed capture unavailable.
    Observe cancellation while draining inherited pipes after leader exit, and never extend the
    execution deadline. Built-in POSIX pipe readers stop cooperatively during cleanup; an injected
    reader or output observer that blocks indefinitely cannot be forcibly stopped by Python.

    Args:
        process (subprocess.Popen[str]): Child started in its own process group.
        deadline (float): Absolute monotonic deadline for exit and complete pipe draining.
        stdout_limit (int): Maximum retained standard-output characters.
        stderr_limit (int): Maximum retained standard-error characters.
        terminate (Callable[[subprocess.Popen[str]], None]): Process-group terminator.
        reader (Callable[..., int | None]): Bounded stream reader; ``None`` means incomplete.
        thread_factory (Callable[..., threading.Thread]): Reader-thread constructor.
        cancel_event (threading.Event | None): Optional event that requests cancellation before
            the process leader is reaped.
        preserve_newlines (bool): Preserve CR and CRLF in the child text pipes. When true,
            both pipes must support ``reconfigure`` and must not have been read yet.
        output_observer (Callable[[str, str], None] | None): Optional callback receiving the
            stream name and each decoded text portion, independently of final capture limits.

    Returns:
        ProcessCapture: Exit, timeout, cancellation, or unavailable capture result.

    Raises:
        BaseException: An operation or cleanup failure prevents a complete capture.
        ValueError: The deadline is invalid; owned child cleanup is attempted before raising.
    """
    if process.stdout is None or process.stderr is None:
        try:
            terminate(process)
        finally:
            _wait_process(process)
        if not is_finite_deadline(deadline):
            raise ValueError("Process deadline must be finite.")
        return ProcessCapture(ProcessCaptureStatus.UNAVAILABLE)

    streams = (process.stdout, process.stderr)
    reader_stop = threading.Event()
    chunks = (
        CapturedChunks("stdout", output_observer, reader_stop, preserve_newlines),
        CapturedChunks("stderr", output_observer, reader_stop, preserve_newlines),
    )
    discarded: list[int | None] = [0, 0]
    readers = []
    started_readers = []
    setup_complete = False
    result = ProcessCapture(ProcessCaptureStatus.UNAVAILABLE)

    def drain(index: int) -> None:
        """Record reader failure without losing already captured output."""
        try:
            discarded[index] = reader(
                streams[index],
                chunks[index],
                maximum=stdout_limit if index == 0 else stderr_limit,
            )
        except BaseException:  # noqa: BLE001 - the supervisor must report every reader failure.
            discarded[index] = None

    try:
        if not is_finite_deadline(deadline):
            raise ValueError("Process deadline must be finite.")
        if preserve_newlines:
            for stream in streams:
                stream.reconfigure(newline="")
        for i in range(2):
            readers.append(
                thread_factory(
                    target=lambda index=i: drain(index),
                    daemon=True,
                )
            )
        for thread in readers:
            try:
                thread.start()
            except BaseException:
                if thread.is_alive():
                    started_readers.append(thread)
                raise
            started_readers.append(thread)
        setup_complete = True
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
                if cancel_event is None:
                    thread.join(max(0, deadline - time.monotonic()))
                else:
                    while thread.is_alive() and not cancel_event.is_set():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        thread.join(min(remaining, _PROCESS_WAIT_POLL_SECONDS))

            if cancel_event is not None and cancel_event.is_set():
                result = ProcessCapture(ProcessCaptureStatus.CANCELLED)
            elif any(thread.is_alive() for thread in readers):
                result = ProcessCapture(
                    ProcessCaptureStatus.TIMED_OUT,
                    stdout="".join(chunks[0]),
                    stderr="".join(chunks[1]),
                    stdout_discarded=discarded[0] if not readers[0].is_alive() else None,
                    stderr_discarded=discarded[1] if not readers[1].is_alive() else None,
                )
            else:
                result = ProcessCapture(
                    ProcessCaptureStatus.UNAVAILABLE
                    if None in discarded
                    else ProcessCaptureStatus.COMPLETED,
                    stdout="".join(chunks[0]),
                    stderr="".join(chunks[1]),
                    stdout_discarded=discarded[0],
                    stderr_discarded=discarded[1],
                )
    except KeyboardInterrupt:
        if not setup_complete:
            raise
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

        reader_stop.set()
        for thread in started_readers:
            thread.join(constants.COMMAND_CLEANUP_WAIT_SECONDS)

        for index, stream in enumerate(streams):
            if (
                index < len(readers)
                and readers[index] in started_readers
                and readers[index].is_alive()
            ):
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
