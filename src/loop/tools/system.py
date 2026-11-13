"""Provide tools for interacting with the system."""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, ValidationInfo

from .. import constants
from ..errors import Problem
from ..execution.sandbox.contracts import SandboxCapability, normalize_egress_origin
from ..permissions import Action, Operation, OperationPlan, ProcessBoundary, ProcessTarget
from ..tooling import PlatformArgumentCapability, ToolContext, normalize_optional_null, tool
from ..utils import VirtualPath


def _terminal_content(value: Any, info: ValidationInfo) -> Any:
    """Omit the null spelling only for terminal actions that cannot accept input text."""
    if info.data.get("action") in {"resize", "eof", "interrupt", "close"}:
        return normalize_optional_null(value)
    return value


@tool(requires_execution_service=True)
def resolve_executable(
    context: ToolContext,
    name: Annotated[
        str | list[str],
        Field(
            description=f"One executable name or up to {constants.MAX_COMMAND_PATH_COUNT} "
            "names to check on the host PATH, "
            "without running them. Each tool receives its own lookup result and grant reference."
        ),
    ],
    cwd: Annotated[
        str,
        VirtualPath.METADATA_KEY,
        Field(
            description="Workspace working directory used to resolve relative host PATH entries."
        ),
    ] = ".",
) -> dict | Problem:
    """Check named host executables before requesting sandbox grants."""
    try:
        if context.execution_service is None:
            raise ValueError("Executable lookup service is unavailable.")
        return context.execution_service.resolve_executable(context, name, cwd)
    except ValueError as exc:
        return Problem(
            code="executable.invalid_lookup",
            title="Executable lookup unavailable",
            detail=str(exc),
            operation=context.tool_name,
        )


def _command_plan(arguments: dict[str, object]) -> OperationPlan:
    """Plan an opaque shell source for the native sandbox boundary."""
    source = str(arguments["command"])
    cwd = str(arguments["cwd"])
    normalized = dict(arguments)
    normalized.update({"cwd": cwd})
    origins = arguments.get("egress_origins")
    if origins is not None:
        if not isinstance(origins, list):
            raise ValueError("Mediated egress origins must be a list.")
        normalized["egress_origins"] = list(
            dict.fromkeys(normalize_egress_origin(origin) for origin in origins)
        )
    return OperationPlan(
        arguments=normalized,
        operations=(
            Operation(
                tool_id="",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=(constants.SHELL_COMMAND_PATH, constants.SHELL_COMMAND_OPTION, source),
                    cwd=cwd,
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
    )


def _host_command_plan(arguments: dict[str, object]) -> OperationPlan:
    """Plan an explicit host command through ordinary registry authorization."""
    source = str(arguments["command"])
    cwd = str(Path(str(arguments["cwd"])).resolve(strict=True))
    normalized = dict(arguments)
    normalized["cwd"] = cwd
    return OperationPlan(
        arguments=normalized,
        operations=(
            Operation(
                tool_id="",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=(constants.SHELL_COMMAND_PATH, constants.SHELL_COMMAND_OPTION, source),
                    cwd=cwd,
                    boundary=ProcessBoundary.HOST,
                ),
            ),
        ),
    )


@tool(
    actions={Action.PROCESS_EXECUTE},
    operation_planner=_host_command_plan,
    requires_execution_service=True,
    requires_fresh_host_approval=True,
)
def request_host_command(
    context: ToolContext,
    command: Annotated[
        str,
        Field(description="Exact POSIX shell source to run without the OS sandbox.", min_length=1),
    ],
    reason: Annotated[
        str,
        Field(
            description="Why sandboxed capabilities cannot satisfy this request.",
            min_length=1,
            max_length=2000,
        ),
    ],
    cwd: Annotated[
        str,
        VirtualPath.METADATA_KEY,
        Field(description=f"Working directory inside {VirtualPath.WORKSPACE}/..."),
    ] = ".",
) -> dict | Problem:
    """Request an explicit host command; output may expose physical paths."""
    if context.execution_service is None:
        return Problem(
            code="host_command.unavailable",
            title="Host command unavailable",
            detail="Host command authorization is unavailable.",
            operation=context.tool_name,
        )
    return context.execution_service.request_host_command(context, command, cwd, reason)


@tool(
    actions={Action.PROCESS_EXECUTE},
    operation_planner=_command_plan,
    execution_authorization=True,
    platform_argument_capabilities=(
        PlatformArgumentCapability("interactive", SandboxCapability.TERMINAL),
        PlatformArgumentCapability("egress_origins", SandboxCapability.MEDIATED_EGRESS),
    ),
)
def run_command(
    context: ToolContext,
    command: Annotated[
        str,
        Field(
            description="Opaque POSIX shell source to run under the host OS sandbox. "
            f'Use "{VirtualPath.TEMPORARY}/..." for session temporary files. '
            "Configure each tool's output, cache and environment updates as needed for "
            "read-only workspaces and session scratch. Do not rerun a command that may have "
            "had effects; inspect or read its existing result instead.",
            min_length=1,
        ),
    ],
    cwd: Annotated[
        str,
        VirtualPath.METADATA_KEY,
        Field(
            description=f"Workspace directory using {VirtualPath.WORKSPACE}/...; "
            "defaults to the workspace root. Shell source is passed unchanged."
        ),
    ] = ".",
    read_only: Annotated[
        bool,
        Field(
            description="Disable workspace writes only when this command and its subprocesses "
            "need none. Tests and builds often write coverage, caches, or reports."
        ),
    ] = False,
    read_roots: Annotated[
        list[str] | None,
        Field(
            description="Extra host or virtual directories to read inside the sandbox, subject "
            "to the effective hard or approval-controlled read-root classification. "
            "Installed tools outside reviewed system roots need a grant for their executable "
            "and support files. An identifiable scope is requested in this call when needed; "
            "use resolve_executable and executable_grant when it cannot be identified. "
            "Unused PATH entries grant lookup metadata only; a safe executable named directly "
            "may receive access to that exact file without its neighbors. "
            "When no extra read roots are needed, use JSON null (unquoted)."
        ),
        BeforeValidator(normalize_optional_null),
    ] = None,
    write_roots: Annotated[
        list[str] | None,
        Field(
            description="Extra existing directories to write inside the sandbox. Each root "
            "is bound to its current identity, checked against hard limits and protected "
            "control paths, and shown for scoped approval. "
            "When no extra write roots are needed, use JSON null (unquoted)."
        ),
        BeforeValidator(normalize_optional_null),
    ] = None,
    executable_grant: Annotated[
        str | list[str] | None,
        Field(
            description="Opaque grant_reference values from resolve_executable; each reference "
            "binds installed-tool read access inside the sandbox. Use a reference when a needed "
            "installation cannot be selected automatically. When no reference was returned, "
            "use JSON null (unquoted)."
        ),
        BeforeValidator(normalize_optional_null),
    ] = None,
    network: Annotated[
        bool,
        Field(
            description="Request general network access only when the effective private-network "
            "limit is disabled and network origins include '*'. The native general grant is not "
            "destination-scoped and can reach private host services."
        ),
    ] = False,
    egress_origins: Annotated[
        list[str] | None,
        Field(
            description="Exact HTTP(S) origins requested through a qualified native mediator. "
            "Each new connection is checked against hard limits and the approved scope; "
            "unsupported clients fail closed without a host retry. "
            "The native mediator configures a proxy in the command environment; use a client that "
            "supports its transport. "
            "Pass a list of origins when needed; otherwise use JSON null (unquoted)."
        ),
        BeforeValidator(normalize_optional_null),
    ] = None,
    git_write: Annotated[
        bool,
        Field(
            description="Request a separate sandboxed Git write grant for opaque Git invocations. "
            "Recognized Git state-changing commands request approval automatically."
        ),
    ] = False,
    reason: Annotated[
        str | None,
        Field(
            description="Why requested additional sandbox capabilities are needed. "
            "When no additional rationale is needed, use JSON null (unquoted), never the string "
            '"None".',
            max_length=2000,
        ),
    ] = None,
    background: Annotated[
        bool,
        Field(
            description="Return a session-owned job handle after approval while the same command "
            "continues. Set false for the ordinary one-call wait and final result."
        ),
    ] = False,
    interactive: Annotated[
        bool,
        Field(
            description="Open one owned sandboxed pseudo-terminal and return its handle after "
            "approval. Use read_command_job and control_command_terminal for the same process. "
            "Cannot combine with egress_origins."
        ),
    ] = False,
) -> dict | Problem:
    """Run sandboxed shell source with approved host recovery; output may expose physical paths."""
    if context.execution_service is None:
        return Problem(
            code="sandbox.unavailable",
            title="Sandbox unavailable",
            detail="Native command execution service is unavailable.",
            operation=context.tool_name,
        )
    reason_argument = {"reason": reason} if reason is not None else {}
    return context.execution_service.run_command(
        context,
        command,
        cwd,
        read_only,
        read_roots,
        write_roots,
        executable_grant,
        network,
        git_write,
        egress_origins=egress_origins,
        background=background,
        interactive=interactive,
        **reason_argument,
    )


@tool(requires_execution_service=True)
def inspect_command_jobs(context: ToolContext) -> dict | Problem:
    """List bounded command continuations owned by this session."""
    if context.execution_service is None:
        return Problem(
            code="command_job.unavailable",
            title="Command jobs unavailable",
            detail="Native command execution service is unavailable.",
            operation=context.tool_name,
        )
    return context.execution_service.inspect_command_jobs(context)


@tool(requires_execution_service=True)
def read_command_job(
    context: ToolContext,
    handle: Annotated[str, Field(description="Opaque handle from run_command.", min_length=32)],
    cursor: Annotated[
        int,
        Field(
            description="Start at zero; for each subsequent read pass the previous response's "
            "next_cursor. An earlier cursor replays output and returns immediately.",
            ge=0,
        ),
    ] = 0,
    wait_seconds: Annotated[
        float | None,
        BeforeValidator(normalize_optional_null),
        Field(
            description="Wait for output or completion, or use JSON null (unquoted) for the "
            "configured default. Available output returns immediately. Capped below "
            "tools.command_timeout; "
            "use 0 to poll now.",
            ge=0,
            allow_inf_nan=False,
        ),
    ] = None,
    include_output: Annotated[
        bool,
        Field(description="Include the final captured stream previews when needed."),
    ] = False,
) -> dict | Problem:
    """Read until result is non-null, then stop; pass each response's next_cursor to continue."""
    if context.execution_service is None:
        return Problem(
            code="command_job.unavailable",
            title="Command job unavailable",
            detail="Native command execution service is unavailable.",
            operation=context.tool_name,
        )
    return context.execution_service.read_command_job(
        context,
        handle,
        cursor,
        wait_seconds,
        include_output,
    )


@tool(requires_execution_service=True)
def cancel_command_job(
    context: ToolContext,
    handle: Annotated[str, Field(description="Opaque handle from run_command.", min_length=32)],
) -> dict | Problem:
    """Request cancellation of one owned command and inspect its current state."""
    if context.execution_service is None:
        return Problem(
            code="command_job.unavailable",
            title="Command job unavailable",
            detail="Native command execution service is unavailable.",
            operation=context.tool_name,
        )
    return context.execution_service.cancel_command_job(context, handle)


@tool(
    requires_execution_service=True,
    private_arguments=True,
    platform_capability=SandboxCapability.TERMINAL,
)
def control_command_terminal(
    context: ToolContext,
    handle: Annotated[str, Field(description="Owned interactive handle.", min_length=32)],
    action: Annotated[
        Literal["write", "resize", "eof", "interrupt", "close"],
        Field(
            description="Terminal action for the same owned process. "
            "Write needs content; resize needs rows and columns."
        ),
    ],
    content: Annotated[
        str | None,
        Field(
            description="Exact text for write. Use JSON null (unquoted) for other actions.",
            max_length=4096,
        ),
        BeforeValidator(_terminal_content),
    ] = None,
    rows: Annotated[
        int | None,
        Field(
            description="Height for resize, between 1 and 500. "
            "Use JSON null (unquoted) for other actions.",
            ge=1,
            le=500,
        ),
        BeforeValidator(normalize_optional_null),
    ] = None,
    columns: Annotated[
        int | None,
        Field(
            description="Width for resize, between 1 and 500. "
            "Use JSON null (unquoted) for other actions.",
            ge=1,
            le=500,
        ),
        BeforeValidator(normalize_optional_null),
    ] = None,
) -> dict | Problem:
    """Send bounded input or terminal controls to one owned sandboxed job."""
    if context.execution_service is None:
        return Problem(
            code="command_terminal.unavailable",
            title="Interactive terminal unavailable",
            detail="Native command execution service is unavailable.",
            operation=context.tool_name,
        )
    return context.execution_service.control_command_terminal(
        context, handle, action, content, rows, columns
    )
