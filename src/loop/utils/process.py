"""Provide safe process invocation utilities."""

import errno
import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
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
    """Terminate a process group, treating a concurrently exited group as already stopped.

    POSIX callers must create the process with ``start_new_session=True`` so the stored PID is
    also the owned process-group ID. Python's portable Windows process API cannot forcibly
    terminate a complete descendant tree, so that platform falls back to the direct process.

    Args:
        process (subprocess.Popen[str]): Running process to terminate.
    """
    if os.name != "posix":
        process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def supervise_process(
    process: subprocess.Popen[str],
    deadline: float,
    *,
    stdout_limit: int = constants.MAX_OUTPUT_CHARS,
    stderr_limit: int = constants.MAX_OUTPUT_CHARS,
    terminate: Callable[[subprocess.Popen[str]], None] = kill_process_group,
    reader: Callable[..., int] = read_bounded_stream,
    thread_factory: Callable[..., threading.Thread] = threading.Thread,
) -> ProcessCapture:
    """Drain two child pipes within one deadline, then kill and reap the process group.

    Args:
        process (subprocess.Popen[str]): Child started in its own process group.
        deadline (float): Absolute monotonic deadline for exit and complete pipe draining.
        stdout_limit (int): Maximum retained standard-output characters.
        stderr_limit (int): Maximum retained standard-error characters.
        terminate (Callable[[subprocess.Popen[str]], None]): Process-group terminator.
        reader (Callable[..., int]): Bounded stream reader.
        thread_factory (Callable[..., threading.Thread]): Reader-thread constructor.

    Returns:
        ProcessCapture: Exit, timeout, cancellation, or missing-pipe result.
    """
    if process.stdout is None or process.stderr is None:
        terminate(process)
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

    try:
        process.wait(timeout=max(0, deadline - time.monotonic()))
        for thread in readers:
            thread.join(max(0, deadline - time.monotonic()))

        if any(thread.is_alive() for thread in readers):
            return ProcessCapture(
                ProcessCaptureStatus.TIMED_OUT,
                stdout="".join(chunks[0]),
                stderr="".join(chunks[1]),
                stdout_discarded=discarded[0] if not readers[0].is_alive() else None,
                stderr_discarded=discarded[1] if not readers[1].is_alive() else None,
            )

        return ProcessCapture(
            ProcessCaptureStatus.COMPLETED,
            exit_code=process.returncode,
            stdout="".join(chunks[0]),
            stderr="".join(chunks[1]),
            stdout_discarded=discarded[0],
            stderr_discarded=discarded[1],
        )
    except subprocess.TimeoutExpired:
        return ProcessCapture(
            ProcessCaptureStatus.TIMED_OUT,
            stdout="".join(chunks[0]),
            stderr="".join(chunks[1]),
            stdout_discarded=None,
            stderr_discarded=None,
        )
    except KeyboardInterrupt:
        return ProcessCapture(ProcessCaptureStatus.CANCELLED)
    finally:
        terminate(process)
        _wait_process(process)

        for thread, stream in zip(readers, streams, strict=True):
            if thread.is_alive():
                try:
                    os.close(stream.fileno())
                except (OSError, TypeError, ValueError):
                    _LOGGER.warning("Could not close a process output stream during cleanup.")
            else:
                stream.close()

        for thread in readers:
            thread.join(constants.COMMAND_CLEANUP_WAIT_SECONDS)


def _wait_process(
    process: subprocess.Popen[str],
    timeout: float = constants.COMMAND_CLEANUP_WAIT_SECONDS,
):
    """Wait for the given process to complete within the cleanup wait time."""
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _LOGGER.warning("Process %s did not exit within the cleanup wait time.", process.pid)
