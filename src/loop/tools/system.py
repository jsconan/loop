"""Provide tools for interacting with the system."""

import os
import signal
import subprocess
import threading
from typing import Annotated

from pydantic import Field

from .. import constants
from ..context import ToolContext
from ..permissions import Capability, PermissionRequest
from ..tooling import tool_registry


def _read_bounded(stream, chunks: list[str]) -> None:
    """Drain a process stream while retaining only a bounded amount of output."""
    remaining = constants.MAX_OUTPUT_CHARS
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


def _command_permission(arguments: dict[str, object]) -> tuple[PermissionRequest, ...]:
    """Describe exact shell execution authority for one command."""
    return (
        PermissionRequest(
            tool_name="run_command",
            capability=Capability.PROCESS_EXEC,
            resource=str(arguments["command"]),
        ),
    )


@tool_registry.tool(
    capabilities={Capability.PROCESS_EXEC},
    permission_resolver=_command_permission,
)
def run_command(
    context: ToolContext,
    command: Annotated[str, Field(description="The system command to execute.")],
) -> str:
    """Run a system command and return the output."""
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
                    returncode = process.wait(timeout=constants.COMMAND_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    _kill_process_group(process)
                    process.wait()
                    return f"Command timed out after {constants.COMMAND_TIMEOUT_SECONDS} seconds."
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
            # Shell commands may create, remove, or edit instruction files. Their exact effects
            # are intentionally not inferred from arbitrary shell text; a successful command
            # therefore triggers a bounded signature refresh on the next request.
            context.invalidate_instructions()
            return output
    except Exception as exc:  # pylint: disable=broad-except
        return f"Error running command: {exc}"
