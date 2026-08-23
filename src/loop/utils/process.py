"""Provide safe process invocation utilities."""

import os
import signal
import subprocess
from typing import Protocol


class TextStream(Protocol):
    """Represent a synchronously readable text stream."""

    def read(self, size: int = -1) -> str:
        """Read at most ``size`` characters from the stream."""


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


def read_bounded_stream(stream: TextStream, chunks: list[str], maximum: int) -> None:
    """Drain a text stream while retaining no more than the requested character limit.

    Args:
        stream (TextStream): Stream drained until its ``read`` method returns an empty string.
        chunks (list[str]): Destination receiving retained text chunks.
        maximum (int): Maximum total characters retained in ``chunks``.
    """
    remaining = maximum
    while chunk := stream.read(8192):
        if remaining:
            chunks.append(chunk[:remaining])
            remaining -= len(chunk[:remaining])


def kill_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate a process and, on POSIX, its process group.

    Args:
        process (subprocess.Popen[str]): Running process to terminate.
    """
    if os.name != "posix":
        process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
