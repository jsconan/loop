"""Provide tools for interacting with the system."""

import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from pydantic import Field

from .. import constants
from ..errors import Problem, ProblemException, log_problem
from ..permissions import (
    Action,
    Operation,
    OperationPlan,
    OperationPlanningContext,
    ProcessBoundary,
    ProcessTarget,
)
from ..sandbox import SandboxPlan, SandboxUnavailableError, spawn_sandboxed
from ..tooling import ToolContext, ToolPreflightResult, ToolStatus, tool
from ..utils import (
    encode_content_cursor,
    kill_process_group,
    parse_command_line,
    read_bounded_stream,
    store_content,
)

_LOGGER = logging.getLogger(__name__)
_CLEANUP_RESERVE_SECONDS = 0.1
_PROCESS_POLL_SECONDS = 0.01


def _remaining(deadline: float) -> float:
    """Return the non-negative time remaining before a monotonic deadline."""
    return max(0.0, deadline - time.monotonic())


def _read_stream(
    stream,
    chunks: list[str],
    errors: list[Exception | None],
    index: int,
    changed: threading.Event,
    discarded: list[int | None],
) -> None:
    """Capture a bounded stream and retain a reader failure for the calling thread."""
    try:
        discarded[index] = read_bounded_stream(stream, chunks, constants.MAX_OUTPUT_CHARS)
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
        errors[index] = exc
    finally:
        changed.set()


def _join_readers(readers: list[threading.Thread], deadline: float) -> bool:
    """Join readers against one deadline and report whether every reader completed."""
    for reader in readers:
        reader.join(_remaining(deadline))
    return all(not reader.is_alive() for reader in readers)


def _close_process_streams(process: subprocess.Popen[str]) -> None:
    """Close parent-owned process pipe endpoints."""
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _interrupt_process_streams(process: subprocess.Popen[str]) -> None:
    """Close pipe descriptors without waiting for locks held by blocked text readers."""
    for stream in (process.stdout, process.stderr):
        try:
            os.close(stream.fileno())  # type: ignore[union-attr]
        except (OSError, TypeError, ValueError):
            pass


def _wait_for_process(
    process: subprocess.Popen[str],
    changed: threading.Event,
    errors: list[Exception | None],
    deadline: float,
) -> int:
    """Wait for process exit while surfacing reader failures before the deadline."""
    while True:
        if reader_error := next((error for error in errors if error is not None), None):
            raise reader_error
        try:
            return process.wait(timeout=min(_PROCESS_POLL_SECONDS, _remaining(deadline)))
        except subprocess.TimeoutExpired:
            if _remaining(deadline) == 0:
                raise
            changed.clear()
            changed.wait(min(_PROCESS_POLL_SECONDS, _remaining(deadline)))


def _cleanup_process(
    process: subprocess.Popen[str],
    readers: list[threading.Thread],
    deadline: float,
) -> None:
    """Kill owned processes, reap the child, and finish readers within the deadline."""
    kill_process_group(process)
    remaining = _remaining(deadline)
    reap_deadline = time.monotonic() + remaining / 3
    try:
        process.wait(timeout=_remaining(reap_deadline))
    except subprocess.TimeoutExpired:
        pass
    reader_deadline = time.monotonic() + _remaining(deadline) / 2
    if _join_readers(readers, reader_deadline):
        _close_process_streams(process)
        return
    _interrupt_process_streams(process)
    _join_readers(readers, deadline)


def _timeout_error(timeout: float, output: dict) -> Problem:
    """Return a standardized timeout error problem."""
    return Problem(
        code="process.timeout",
        title="Command timed out",
        detail=(f"Command did not complete within {timeout} seconds."),
        retryable=True,
        operation="run_command",
        metadata=output,
    )


def _stream_output(
    chunks: list[str],
    discarded: int | None,
    source: str,
    redactor: Callable[[str], str] | None = None,
) -> dict:
    """Return a recoverable preview and explicit capture-loss status for one stream."""
    content = "".join(chunks)
    if redactor is not None:
        content = redactor(content)
    encoded = content.encode("utf-8")
    preview = encoded[: constants.MAX_TOOL_CONTENT_BYTES // 2].decode("utf-8", errors="ignore")
    included = len(preview.encode("utf-8"))
    result = {
        "content": preview,
        "captured_bytes": len(encoded),
        "included_bytes": included,
        "truncated": included < len(encoded) or discarded != 0,
        "capture_complete": discarded == 0,
        "discarded_characters": discarded,
    }
    if included < len(encoded):
        handle = store_content(encoded, source)
        result.update(
            handle=handle,
            next_cursor=encode_content_cursor(handle, included),
            continuation="Use read_cached_content with this handle and cursor.",
        )
    return result


def _process_output(
    returncode: int | None,
    chunks: list[list[str]],
    discarded: list,
    redactor: Callable[[str], str] | None = None,
) -> dict:
    """Preserve exit status and both streams without hiding incomplete capture."""
    return {
        "exit_code": returncode,
        "stdout": _stream_output(chunks[0], discarded[0], "command stdout", redactor),
        "stderr": _stream_output(chunks[1], discarded[1], "command stderr", redactor),
    }


def _command_plan(
    arguments: dict[str, object],
    planning_context: OperationPlanningContext,
) -> OperationPlan:
    """Plan an exact shell-free process invocation."""
    argv = parse_command_line(str(arguments["command"]))
    cwd = str(arguments["cwd"])
    if planning_context.workspace is None:
        raise ValueError("Sandbox planning requires a configured workspace root.")
    sandbox_plan = SandboxPlan.create(
        argv,
        Path(cwd),
        planning_context.workspace,
        temporary_directory=planning_context.temporary_directory,
        read_only_roots=planning_context.read_only_roots,
        readable_roots=planning_context.readable_roots,
        writable_roots=planning_context.writable_roots,
    )
    normalized = dict(arguments)
    normalized.update({"cwd": str(sandbox_plan.cwd)})
    return OperationPlan(
        arguments=normalized,
        operations=(
            Operation(
                tool_id="",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=sandbox_plan.argv,
                    cwd=str(sandbox_plan.cwd),
                    workspace=str(sandbox_plan.workspace),
                    temporary_directory=(
                        str(sandbox_plan.temporary_directory)
                        if sandbox_plan.temporary_directory is not None
                        else None
                    ),
                    read_only_roots=tuple(str(root) for root in sandbox_plan.read_only_roots),
                    readable_roots=tuple(str(root) for root in sandbox_plan.readable_roots),
                    writable_roots=tuple(str(root) for root in sandbox_plan.writable_roots),
                    boundary=ProcessBoundary.SANDBOXED,
                    sandbox_policy=sandbox_plan.policy_digest,
                ),
            ),
        ),
    )


def _command_preflight() -> ToolPreflightResult:
    """Expose command planning; capability is probed against the authorized plan at execution."""
    return ToolPreflightResult(status=ToolStatus.READY)


@tool(
    actions={Action.PROCESS_EXECUTE},
    operation_planner=_command_plan,
    preflight=_command_preflight,
)
def run_command(
    context: ToolContext,
    command: Annotated[
        str,
        "loop:virtual-command",
        Field(
            description="Executable followed by its arguments. "
            "This is a restricted command line, not a shell: quote or escape shell "
            "characters when they are literal argument data.",
            min_length=1,
        ),
    ],
    cwd: Annotated[
        str,
        "loop:virtual-path",
        "loop:workspace-cwd",
        Field(
            description="Working directory for the process. Use '/workspace' for the workspace "
            "root; 'workspace' is accepted as a workspace-root shorthand."
        ),
    ] = ".",
) -> dict | Problem:
    """Run a shell-free process and return exit status and recoverable stdout/stderr previews."""
    # pylint: disable=too-many-branches,too-many-statements
    del command, cwd  # Execution uses only the canonical, authorized ProcessTarget.
    process = None
    started_readers = []
    timeout = context.settings.command_timeout
    deadline = time.monotonic() + timeout
    cleanup_reserve = min(_CLEANUP_RESERVE_SECONDS, timeout / 2)
    execution_deadline = deadline - cleanup_reserve
    output_redactor = None
    execution_boundary = ProcessBoundary.SANDBOXED.value
    try:
        operation = context.operations[0] if context.operations else None
        target = operation.target if operation is not None else None
        if not isinstance(target, ProcessTarget):
            raise TypeError("Authorized process target is missing.")
        command_cwd = Path(target.cwd)
        if context.instructions_manager is not None:
            output_redactor = context.instructions_manager.virtual_paths.redact
        sandbox_plan = SandboxPlan.create(
            target.argv,
            command_cwd,
            Path(target.workspace),
            temporary_directory=(
                Path(target.temporary_directory) if target.temporary_directory is not None else None
            ),
            read_only_roots=tuple(Path(root) for root in target.read_only_roots),
            readable_roots=tuple(Path(root) for root in target.readable_roots),
            writable_roots=tuple(Path(root) for root in target.writable_roots),
        )
        if target.sandbox_policy != sandbox_plan.policy_digest:
            raise TypeError("Authorized sandbox policy is stale.")
        popen_options = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        try:
            process = spawn_sandboxed(sandbox_plan, popen_options=popen_options)
        except SandboxUnavailableError as exc:
            process = context.request_host_process(sandbox_plan, str(exc), popen_options)
            execution_boundary = ProcessBoundary.HOST.value
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Command process did not expose its output streams.")
        stdout_chunks = []
        stderr_chunks = []
        reader_errors = [None, None]
        reader_changed = threading.Event()
        discarded = [None, None]
        readers = [
            threading.Thread(
                target=_read_stream,
                args=(process.stdout, stdout_chunks, reader_errors, 0, reader_changed, discarded),
                daemon=True,
            ),
            threading.Thread(
                target=_read_stream,
                args=(process.stderr, stderr_chunks, reader_errors, 1, reader_changed, discarded),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()
            started_readers.append(reader)

        try:
            returncode = _wait_for_process(
                process, reader_changed, reader_errors, execution_deadline
            )
        except subprocess.TimeoutExpired:
            _cleanup_process(process, started_readers, deadline)
            timed_out = _process_output(
                None, [stdout_chunks, stderr_chunks], discarded, output_redactor
            )
            timed_out["boundary"] = execution_boundary
            return _timeout_error(
                timeout,
                timed_out,
            )
        if not _join_readers(started_readers, execution_deadline):
            _cleanup_process(process, started_readers, deadline)
            timed_out = _process_output(
                returncode, [stdout_chunks, stderr_chunks], discarded, output_redactor
            )
            timed_out["boundary"] = execution_boundary
            return _timeout_error(
                timeout,
                timed_out,
            )
        if reader_error := next((error for error in reader_errors if error is not None), None):
            raise reader_error

        _close_process_streams(process)
        output = _process_output(
            returncode, [stdout_chunks, stderr_chunks], discarded, output_redactor
        )
        output["boundary"] = execution_boundary
        if returncode != 0:
            return Problem(
                code="process.nonzero_exit",
                title="Command failed",
                detail=f"Command exited with code {returncode}.",
                operation="run_command",
                metadata=output,
            )
        # Commands may create, remove, or edit instruction files. Their exact effects are
        # intentionally not inferred from arbitrary command text; a successful command
        # therefore triggers a bounded signature refresh on the next request.
        context.invalidate_instructions()
        return output
    except ProblemException:
        raise
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
        if process is not None:
            _cleanup_process(process, started_readers, deadline)
        problem = Problem.from_exception(
            exc,
            code="process.execution_failed",
            title="Could not run command",
            operation="run_command",
        )
        log_problem(_LOGGER, problem, exc)
        return problem
