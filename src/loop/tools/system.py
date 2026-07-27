"""Tools for interacting with the system."""

import os
import signal
import subprocess
import threading
from typing import Annotated

from pydantic import Field

from ..tooling import tool_registry

COMMAND_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 1_000_000


def _read_bounded(stream, chunks: list[str]) -> None:
    """Drain a process stream while retaining only a bounded amount of output."""
    remaining = MAX_OUTPUT_CHARS
    while chunk := stream.read(8192):
        if remaining:
            chunks.append(chunk[:remaining])
            remaining -= len(chunk[:remaining])


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    """Kill the shell and any child processes started by it."""
    if os.name != "posix":
        process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@tool_registry.tool
def run_command(
    self,
    command: Annotated[str, Field(description="The system command to execute.")],
) -> str:
    """Run a system command and return the output."""
    if not self.confirm(f"Agent wants to run command '{command}'. Proceed?"):
        return "Command execution cancelled by user."

    try:
        with subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=os.name == "posix",
        ) as process:
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []
            readers = [
                threading.Thread(
                    target=_read_bounded, args=(process.stdout, stdout_chunks), daemon=True
                ),
                threading.Thread(
                    target=_read_bounded, args=(process.stderr, stderr_chunks), daemon=True
                ),
            ]
            started_readers: list[threading.Thread] = []
            try:
                for reader in readers:
                    reader.start()
                    started_readers.append(reader)

                try:
                    returncode = process.wait(timeout=COMMAND_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    _kill_process_group(process)
                    process.wait()
                    return f"Command timed out after {COMMAND_TIMEOUT_SECONDS} seconds."
            finally:
                if process.poll() is None:
                    _kill_process_group(process)
                    process.wait()
                for reader in started_readers:
                    reader.join()

            output = "".join(stdout_chunks).strip()
            error_msg = "".join(stderr_chunks).strip()
            if returncode != 0:
                return f"Command failed with code {returncode}. Output: {output} Error: {error_msg}"
            return output
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error running command: {exc}"
