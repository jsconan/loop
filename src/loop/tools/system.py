"""Provide sandboxed command execution tools."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, ValidationError

from ..errors import Problem
from ..execution.contracts import (
    HostExecutionRequest,
    JobHandle,
    JobOperation,
    NetworkConnectionLease,
    NetworkListenerLease,
    SecretExposure,
    TerminalMode,
)
from ..execution.host.models import (
    HostCompleted,
    HostPermissionDenialReason,
    HostPermissionDenied,
)
from ..execution.results import Completed, TimedOut
from ..execution.sandbox.oci.control import OciSignal
from ..permissions import OperationPlan
from ..tooling import ToolContext, tool


def _command_plan(arguments: dict[str, object]) -> OperationPlan:
    """Normalize opaque shell source without claiming host-process authority."""
    return OperationPlan(arguments=dict(arguments))


def _host_command_plan(arguments: dict[str, object]) -> OperationPlan:
    """Keep explicit host authorization inside the dedicated host broker."""
    return OperationPlan(arguments=dict(arguments))


def _job_handle(job_id: str, token: str) -> JobHandle:
    """Validate an opaque durable handle at the public tool boundary."""
    return JobHandle(job_id=job_id, token=token)


def _job_problem(error: Exception, operation: str) -> Problem:
    """Return a sanitized durable lifecycle failure."""
    return Problem(
        code="process.durable_job_failed",
        title="Durable job operation failed",
        detail=f"{operation} could not complete safely: {type(error).__name__}.",
        operation=operation,
    )


def _stream_output(content: bytes, truncated: bool) -> dict[str, object]:
    """Return one bounded UTF-8 stream using the established result shape."""
    text = content.decode("utf-8", errors="replace")
    return {
        "content": text,
        "captured_bytes": len(content),
        "included_bytes": len(content),
        "truncated": truncated,
        "capture_complete": not truncated,
        "discarded_characters": None if truncated else 0,
    }


def _output(result) -> dict[str, object]:
    """Return model-safe bounded output from a closed sandbox result."""
    return {
        "exit_code": getattr(result, "exit_code", None),
        "stdout": _stream_output(result.stdout, result.stdout_truncated),
        "stderr": _stream_output(result.stderr, result.stderr_truncated),
    }


def _host_denial_detail(result: HostPermissionDenied) -> str:
    """Return actionable feedback for one safe explicit-host denial category."""
    if result.reason is HostPermissionDenialReason.HOST_PROCESSES_DISABLED:
        return (
            "This operation requires direct Host execution, but host processes are disabled. "
            "Enable the workspace option with `/permissions limit set workspace host-process "
            "allow`, then retry."
        )
    if result.reason is HostPermissionDenialReason.USER_DENIED:
        return "The explicit Host execution request was not approved."
    if result.reason is HostPermissionDenialReason.APPROVAL_UNAVAILABLE:
        return "Direct Host execution requires approval, but no interactive user is available."
    return "Direct Host execution is not authorized by the current permission policy."


@tool(operation_planner=_command_plan)
def run_command(
    context: ToolContext,
    command: Annotated[
        str,
        Field(
            description=(
                "POSIX shell source executed only inside the managed sandbox. Ordinary developer "
                "tools and language toolchains use the managed sandbox image; never substitute "
                "Host execution for a missing sandbox command."
            ),
            min_length=1,
        ),
    ],
    cwd: Annotated[
        str,
        Field(
            description="Absolute virtual working directory. Use '/workspace' for the root.",
            min_length=1,
        ),
    ] = "/workspace",
    pty: Annotated[
        bool,
        Field(description="Allocate one merged pseudo-terminal for interactive programs."),
    ] = False,
    terminal_columns: Annotated[int, Field(ge=1, le=16384)] = 80,
    terminal_rows: Annotated[int, Field(ge=1, le=16384)] = 24,
    network_connections: Annotated[
        tuple[NetworkConnectionLease, ...],
        Field(
            description=(
                "Explicit pre-resolved outbound destinations. Omit for an offline command."
            )
        ),
    ] = (),
    network_listeners: Annotated[
        tuple[NetworkListenerLease, ...],
        Field(description="Explicit inbound TCP ports to publish through the broker."),
    ] = (),
    secret_exposures: Annotated[
        tuple[SecretExposure, ...],
        Field(
            description=(
                "Explicit audience-bound credential injection or warned raw env/file exposure."
            )
        ),
    ] = (),
    stdin: Annotated[
        str,
        Field(
            description="Bounded UTF-8 input delivered to the command before EOF.", max_length=65536
        ),
    ] = "",
) -> dict | Problem:
    """Run POSIX shell source in the managed sandbox and return bounded output."""
    if context.command_executor is None:
        return Problem(
            code="process.sandbox_unavailable",
            title="Sandbox unavailable",
            detail="Managed command execution is not configured.",
            operation="run_command",
        )
    if not cwd.startswith("/") or "/../" in f"{cwd}/" or "//" in cwd:
        return Problem(
            code="process.invalid_virtual_cwd",
            title="Invalid working directory",
            detail="Command working directory must be a normalized absolute virtual path.",
            operation="run_command",
        )
    try:
        network_connections = tuple(
            value
            if isinstance(value, NetworkConnectionLease)
            else NetworkConnectionLease.model_validate(value)
            for value in network_connections
        )
        network_listeners = tuple(
            value
            if isinstance(value, NetworkListenerLease)
            else NetworkListenerLease.model_validate(value)
            for value in network_listeners
        )
        secret_exposures = tuple(
            value if isinstance(value, SecretExposure) else SecretExposure.model_validate(value)
            for value in secret_exposures
        )
    except (TypeError, ValidationError, ValueError):
        return Problem(
            code="process.invalid_effect_lease",
            title="Invalid effect lease",
            detail="Network and secret effects require exact structured leases.",
            operation="run_command",
        )
    options = {
        "request_id": context.call_id,
        "terminal": TerminalMode.PTY if pty else TerminalMode.PIPE,
        "terminal_columns": terminal_columns if pty else None,
        "terminal_rows": terminal_rows if pty else None,
    }
    if stdin:
        options["stdin"] = stdin.encode()
    if network_connections or network_listeners or secret_exposures:
        options.update(
            network_connections=network_connections,
            network_listeners=network_listeners,
            secret_exposures=secret_exposures,
        )
    result = context.command_executor.execute(
        command,
        cwd,
        context.settings.command_timeout,
        cancellation=context.cancellation,
        **options,
    )
    output = _output(result)
    if isinstance(result, Completed):
        if result.exit_code == 0:
            context.invalidate_instructions()
            return output
        return Problem(
            code="process.nonzero_exit",
            title="Command failed",
            detail=f"Command exited with code {result.exit_code}.",
            operation="run_command",
            metadata=output,
        )
    if isinstance(result, TimedOut):
        return Problem(
            code="process.timeout",
            title="Command timed out",
            detail=(f"Command did not complete within {context.settings.command_timeout} seconds."),
            retryable=True,
            operation="run_command",
            metadata=output,
        )
    return Problem(
        code=f"process.{result.kind}",
        title="Command could not complete",
        detail="Managed sandbox execution stopped safely.",
        operation="run_command",
        metadata={**output, "diagnostic_id": result.diagnostic_id},
    )


@tool(operation_planner=_host_command_plan)
def run_host_command(
    context: ToolContext,
    executable: Annotated[
        str,
        Field(
            description=(
                "Absolute host executable requested through the explicit Host flow, for example "
                "'/usr/bin/git'."
            )
        ),
    ],
    arguments: Annotated[
        tuple[str, ...],
        Field(description="Exact host arguments excluding the executable itself."),
    ] = (),
    cwd: Annotated[
        str,
        Field(
            description=(
                "Absolute authenticated host working directory. Use '/workspace' for the active "
                "project or '/workspace/path' for one of its subdirectories."
            )
        ),
    ] = "/",
    display_cwd: Annotated[
        str,
        Field(description="Sanitized working-directory label shown in the Host warning."),
    ] = "host filesystem",
    reason: Annotated[
        str,
        Field(description="Why the managed sandbox cannot satisfy this operation."),
    ] = "This operation requires an explicitly approved host resource.",
    resource_class: Annotated[
        str,
        Field(description="Bounded host resource category shown during approval."),
    ] = "host process",
) -> dict | Problem:
    """Run an explicit host request after a separate warning, lease, and audit."""
    broker = context.host_execution_broker
    if broker is None:
        return Problem(
            code="host.unavailable",
            title="Host execution unavailable",
            detail="Explicit host execution is not configured.",
            operation="run_host_command",
        )
    try:
        request = HostExecutionRequest(
            request_id=context.call_id or "host-request",
            executable=executable,
            argv=(executable, *arguments),
            cwd=cwd,
            display_cwd=display_cwd,
            resource_class=resource_class,
            reason=reason,
            deadline_seconds=context.settings.command_timeout,
        )
    except ValidationError:
        return Problem(
            code="host.invalid_request",
            title="Invalid host request",
            detail="Explicit host request fields are invalid or ambiguous.",
            operation="run_host_command",
        )
    result = broker.execute(request, context.cancellation)
    output = _output(result)
    if isinstance(result, HostCompleted):
        if result.exit_code == 0:
            context.invalidate_instructions()
            return output
        return Problem(
            code="host.nonzero_exit",
            title="Host command failed",
            detail=f"Host command exited with code {result.exit_code}.",
            operation="run_host_command",
            metadata=output,
        )
    if isinstance(result, HostPermissionDenied):
        return Problem(
            code=result.kind,
            title="Host execution not authorized",
            detail=_host_denial_detail(result),
            operation="run_host_command",
            metadata=output,
        )
    return Problem(
        code=result.kind,
        title="Host command could not complete",
        detail="The separately authorized Host operation stopped safely.",
        operation="run_host_command",
        metadata=output,
    )


@tool(operation_planner=_command_plan)
def start_command_job(
    context: ToolContext,
    command: Annotated[str, Field(min_length=1, description="Opaque POSIX shell source.")],
    cwd: Annotated[
        str, Field(description="Normalized absolute virtual working directory.")
    ] = "/workspace",
    terminal_columns: Annotated[int, Field(ge=1, le=16384)] = 80,
    terminal_rows: Annotated[int, Field(ge=1, le=16384)] = 24,
) -> dict | Problem:
    """Start an explicitly durable managed PTY job."""
    executor = context.command_executor
    if executor is None:
        return Problem(
            code="process.sandbox_unavailable",
            title="Sandbox unavailable",
            detail="Managed durable jobs are not configured.",
            operation="start_command_job",
        )
    try:
        result = executor.start_job(
            command,
            cwd,
            context.settings.command_timeout,
            request_id=context.call_id,
            terminal_columns=terminal_columns,
            terminal_rows=terminal_rows,
        )
    except Exception as error:  # noqa: BLE001 - platform details remain private.
        return _job_problem(error, "start_command_job")
    if isinstance(result, JobHandle):
        return result.model_dump(mode="json")
    return Problem(
        code=f"process.{result.kind}",
        title="Durable job could not start",
        detail="The managed sandbox refused the durable start safely.",
        operation="start_command_job",
    )


def _job_operation(context: ToolContext, job_id: str, token: str, operation: JobOperation):
    """Return a manager, authenticated handle, and fresh operation lease."""
    executor = context.command_executor
    if executor is None:
        raise RuntimeError("Managed durable jobs are unavailable.")
    handle = _job_handle(job_id, token)
    return (
        executor.job_manager(),
        handle,
        executor.job_lease(operation, context.settings.command_timeout),
    )


@tool(operation_planner=_command_plan)
def command_job_status(context: ToolContext, job_id: str, token: str) -> dict | Problem:
    """Return one authenticated durable job status."""
    try:
        manager, handle, lease = _job_operation(context, job_id, token, JobOperation.STATUS)
        return manager.status(handle, lease).model_dump(mode="json")
    except Exception as error:  # noqa: BLE001 - public lifecycle stays sanitized.
        return _job_problem(error, "command_job_status")


@tool(operation_planner=_command_plan)
def attach_command_job(context: ToolContext, job_id: str, token: str) -> dict | Problem:
    """Attach to one authenticated running durable job."""
    try:
        manager, handle, lease = _job_operation(context, job_id, token, JobOperation.ATTACH)
        manager.attach(handle, lease, deadline_seconds=context.settings.command_timeout)
        return {"attached": True}
    except Exception as error:  # noqa: BLE001 - public lifecycle stays sanitized.
        return _job_problem(error, "attach_command_job")


@tool(operation_planner=_command_plan)
def read_command_job(
    context: ToolContext,
    job_id: str,
    token: str,
    timeout_seconds: Annotated[float, Field(ge=0, le=5)] = 0,
) -> dict | Problem:
    """Read one bounded frame from an authenticated durable attachment."""
    try:
        manager, handle, lease = _job_operation(context, job_id, token, JobOperation.ATTACH)
        frame = manager.read(handle, lease, timeout_seconds)
        if frame is None:
            return {"frame": None}
        return {"frame": {"stream": frame.stream, "data": _stream_output(frame.data, False)}}
    except Exception as error:  # noqa: BLE001 - public lifecycle stays sanitized.
        return _job_problem(error, "read_command_job")


@tool(operation_planner=_command_plan)
def write_command_job(
    context: ToolContext,
    job_id: str,
    token: str,
    data: Annotated[str, Field(max_length=65536)] = "",
    eof: bool = False,
) -> dict | Problem:
    """Write bounded UTF-8 input or EOF to an authenticated durable attachment."""
    try:
        manager, handle, lease = _job_operation(context, job_id, token, JobOperation.STDIN)
        if data:
            manager.write(handle, lease, data.encode())
        if eof:
            manager.close_stdin(handle, lease)
        return {"accepted_bytes": len(data.encode()), "eof": eof}
    except Exception as error:  # noqa: BLE001 - public lifecycle stays sanitized.
        return _job_problem(error, "write_command_job")


@tool(operation_planner=_command_plan)
def resize_command_job(
    context: ToolContext,
    job_id: str,
    token: str,
    columns: Annotated[int, Field(ge=1, le=16384)],
    rows: Annotated[int, Field(ge=1, le=16384)],
) -> dict | Problem:
    """Resize an authenticated durable PTY attachment."""
    try:
        manager, handle, lease = _job_operation(context, job_id, token, JobOperation.RESIZE)
        manager.resize(handle, lease, columns, rows)
        return {"columns": columns, "rows": rows}
    except Exception as error:  # noqa: BLE001 - public lifecycle stays sanitized.
        return _job_problem(error, "resize_command_job")


@tool(operation_planner=_command_plan)
def signal_command_job(
    context: ToolContext,
    job_id: str,
    token: str,
    signal: Annotated[str, Field(pattern="^(INT|TERM|HUP)$")],
) -> dict | Problem:
    """Deliver one reviewed signal to an authenticated durable job."""
    signals = {
        "INT": OciSignal.INTERRUPT,
        "TERM": OciSignal.TERMINATE,
        "HUP": OciSignal.HANGUP,
    }
    try:
        manager, handle, lease = _job_operation(context, job_id, token, JobOperation.SIGNAL)
        manager.signal(handle, lease, signals[signal])
        return {"signal": signal}
    except Exception as error:  # noqa: BLE001 - public lifecycle stays sanitized.
        return _job_problem(error, "signal_command_job")


@tool(operation_planner=_command_plan)
def detach_command_job(context: ToolContext, job_id: str, token: str) -> dict | Problem:
    """Detach management I/O without stopping an authenticated durable job."""
    try:
        manager, handle, lease = _job_operation(context, job_id, token, JobOperation.ATTACH)
        manager.detach(handle, lease)
        return {"detached": True}
    except Exception as error:  # noqa: BLE001 - public lifecycle stays sanitized.
        return _job_problem(error, "detach_command_job")


@tool(operation_planner=_command_plan)
def cancel_command_job(context: ToolContext, job_id: str, token: str) -> dict | Problem:
    """Cancel, reap, and finalize an authenticated durable job."""
    try:
        manager, handle, lease = _job_operation(context, job_id, token, JobOperation.CANCEL)
        return manager.cancel(handle, lease).model_dump(mode="json")
    except Exception as error:  # noqa: BLE001 - public lifecycle stays sanitized.
        return _job_problem(error, "cancel_command_job")
