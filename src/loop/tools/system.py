"""Provide tools for interacting with the system."""

import logging
import os
import subprocess
import threading
from typing import Annotated

from pydantic import Field

from .. import constants
from ..errors import Problem, log_problem
from ..permissions import Action, Operation, OperationPlan, ProcessBoundary, ProcessTarget
from ..tooling import ToolContext, tool
from ..utils import kill_process_group, parse_command_line, read_bounded_stream

_LOGGER = logging.getLogger(__name__)


def _command_plan(arguments: dict[str, object]) -> OperationPlan:
    """Plan an exact shell-free process invocation."""
    argv = parse_command_line(str(arguments["command"]))
    cwd = os.path.realpath(str(arguments["cwd"]))
    normalized = dict(arguments)
    normalized.update({"cwd": cwd})
    return OperationPlan(
        arguments=normalized,
        operations=(
            Operation(
                tool_id="",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(argv=argv, cwd=cwd, boundary=ProcessBoundary.HOST),
            ),
        ),
    )


@tool(
    actions={Action.PROCESS_EXECUTE},
    operation_planner=_command_plan,
)
def run_command(
    context: ToolContext,
    command: Annotated[
        str,
        Field(
            description="Executable followed by its arguments. "
            "This is a restricted command line, not a shell: quote or escape shell "
            "characters when they are literal argument data.",
            min_length=1,
        ),
    ],
    cwd: Annotated[
        str,
        Field(description="Working directory for the process."),
    ] = ".",
) -> str | Problem:
    """Run a shell-free process and return its output."""
    try:
        operation = context.operations[0] if context.operations else None
        target = operation.target if operation is not None else None
        if not isinstance(target, ProcessTarget):
            raise TypeError("Authorized process target is missing.")
        with subprocess.Popen(
            list(target.argv),
            shell=False,
            cwd=target.cwd,
            env={
                name: value
                for name in ("PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP")
                if (value := os.environ.get(name)) is not None
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=os.name == "posix",
        ) as process:
            stdout_chunks = []
            stderr_chunks = []
            readers = [
                threading.Thread(
                    target=read_bounded_stream,
                    args=(process.stdout, stdout_chunks, constants.MAX_OUTPUT_CHARS),
                    daemon=True,
                ),
                threading.Thread(
                    target=read_bounded_stream,
                    args=(process.stderr, stderr_chunks, constants.MAX_OUTPUT_CHARS),
                    daemon=True,
                ),
            ]
            started_readers = []
            try:
                for reader in readers:
                    reader.start()
                    started_readers.append(reader)

                try:
                    returncode = process.wait(timeout=constants.COMMAND_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    kill_process_group(process)
                    process.wait()
                    return Problem(
                        code="process.timeout",
                        title="Command timed out",
                        detail=(f"Command exceeded {constants.COMMAND_TIMEOUT_SECONDS} seconds."),
                        retryable=True,
                        operation="run_command",
                    )
            finally:
                if process.poll() is None:
                    kill_process_group(process)
                    process.wait()
                for reader in started_readers:
                    reader.join()

            output = "".join(stdout_chunks).strip()
            error_msg = "".join(stderr_chunks).strip()
            if returncode != 0:
                return Problem(
                    code="process.nonzero_exit",
                    title="Command failed",
                    detail=f"Command exited with code {returncode}.",
                    operation="run_command",
                    metadata={"exit_code": returncode, "stdout": output, "stderr": error_msg},
                )
            # Commands may create, remove, or edit instruction files. Their exact effects are
            # intentionally not inferred from arbitrary command text; a successful command
            # therefore triggers a bounded signature refresh on the next request.
            context.invalidate_instructions()
            return output
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
        problem = Problem.from_exception(
            exc,
            code="process.execution_failed",
            title="Could not run command",
            operation="run_command",
        )
        log_problem(_LOGGER, problem, exc)
        return problem
