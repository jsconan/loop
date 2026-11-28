"""Provide tools for interacting with the system."""

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, ValidationInfo
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import PydanticCustomError

from .. import constants
from ..errors import Problem
from ..execution.sandbox.constants import OUTPUT_DIRECTORY_ENVIRONMENT_VARIABLE
from ..execution.sandbox.contracts import SandboxCapability, normalize_egress_origin
from ..models import ToolDefinition
from ..permissions import Action, Operation, OperationPlan, ProcessBoundary, ProcessTarget
from ..sandbox_catalog import SANDBOX_OPTIONS
from ..tooling import (
    PlatformArgumentCapability,
    ToolContext,
    ToolRuntimeSettings,
    normalize_optional_null,
    tool,
)
from ..tooling.context import CommandEgressAuthority
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
            "names to check on the host PATH without running them. A default_read result "
            "is ready to use. An external installation may return a grant_reference."
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
    """Check whether named executables are installed and readable by a sandbox command."""
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
        if origins and arguments.get("network"):
            raise ValueError(
                "For exact HTTP(S) destinations, keep egress_origins and set network=false. "
                "network=true requests general access."
            )
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


def _wait_value(value: Any) -> Any:
    """Accept optional numeric duration spellings while refusing boolean scheduling values."""
    if value is True or value is False:
        raise PydanticCustomError("wait_duration", "The command wait must be a nonnegative number.")
    return normalize_optional_null(value)


def _bounded_wait(wait_seconds: float | None, settings: ToolRuntimeSettings) -> float | None:
    """Honor an earlier return and bound longer requests to the configured per-call wait."""
    return None if wait_seconds is None else min(wait_seconds, settings.command_max_wait_seconds)


def _configure_wait_definition(
    definition: ToolDefinition,
    settings: ToolRuntimeSettings,
    *,
    default_wait_seconds: float,
    default_setting: str,
) -> ToolDefinition:
    """Advertise the active wait cap and defaults for a nullable duration."""
    wait_schema = definition.parameters["properties"]["wait_seconds"]
    maximum = settings.command_max_wait_seconds
    wait_schema["anyOf"][0]["maximum"] = maximum
    wait_schema["description"] = wait_schema["description"].format(
        maximum_seconds=maximum,
        maximum_setting="tools.command_max_wait_seconds",
        default_seconds=default_wait_seconds,
        default_setting=default_setting,
        runtime_limit_seconds=settings.command_job_runtime_limit_seconds,
        runtime_limit_setting="tools.command_job_runtime_limit_seconds",
    )
    return definition


def _run_command_definition(
    definition: ToolDefinition,
    settings: ToolRuntimeSettings,
    services: tuple[str, ...],
) -> ToolDefinition:
    """Include command usage, active wait durations, and available runtime services."""
    definition.instructions = (
        "Run the requested command with its normal options and configuration. "
        'Use `"write_roots": null` for ordinary commands so Loop can apply the workspace policy '
        "and include any eligible write scope in the command review. "
        'Use `"write_roots": []` when the checkout must stay read-only. '
        "Use the program's supported output settings to put "
        f'generated files under "${OUTPUT_DIRECTORY_ENVIRONMENT_VARIABLE}". '
        f"Set wait_seconds to {settings.command_max_wait_seconds:g} when you need the completed "
        "result, or to 0 when you need a job handle immediately. "
        "Run the command directly when you only need to limit displayed output. "
        "Loop returns a bounded preview and retains output for paging. "
        "When the operation needs a pipeline, verify the relevant stage from its own result. "
        "The shell's exit code can reflect only the final stage. "
        "Label any reported pipeline exit code as the shell's exit code."
    )
    properties = definition.parameters["properties"]
    if services:
        field = properties["runtime_services"]
        field["anyOf"][0]["items"]["enum"] = list(services)
        field["description"] += " " + " ".join(
            f"{name}: {SANDBOX_OPTIONS[name]['label']}." for name in services
        )
    else:
        properties.pop("runtime_services")
        definition.parameters["required"].remove("runtime_services")
    return _configure_wait_definition(
        definition,
        settings,
        default_wait_seconds=settings.command_default_initial_wait_seconds,
        default_setting="tools.command_default_initial_wait_seconds",
    )


def _poll_command_job_definition(
    definition: ToolDefinition, settings: ToolRuntimeSettings, _services: tuple[str, ...]
) -> ToolDefinition:
    """Include active poll-wait and whole-command durations in the schema."""
    return _configure_wait_definition(
        definition,
        settings,
        default_wait_seconds=settings.command_default_poll_wait_seconds,
        default_setting="tools.command_default_poll_wait_seconds",
    )


def _command_worker_context(context: ToolContext, worker_context: ToolContext) -> ToolContext:
    """Add the command tool's narrowly scoped authorities to its worker context."""
    permissions = context.permission_manager
    instructions = context.instructions_manager
    return replace(
        worker_context,
        standalone_execution=(
            permissions.standalone_execution if permissions is not None else None
        ),
        invalidate_detached=(
            partial(instructions.invalidate, None) if instructions is not None else None
        ),
        egress_authority=(
            CommandEgressAuthority(
                permissions.bind_egress,
                permissions.unbind_egress,
                permissions.try_egress_policy_revision,
            )
            if permissions is not None
            else None
        ),
        audit_host_detached=(permissions.record_host_boundary if permissions is not None else None),
        host_recovery_detached=(
            permissions.host_recovery_revision if permissions is not None else None
        ),
        command_admission=context.command_admission,
        command_authorities=(permissions, instructions),
        command_paths=context.command_paths,
        admit_skills_detached=(
            instructions.admit_command_skills if instructions is not None else None
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
    definition_transform=_run_command_definition,
    worker_context_transform=_command_worker_context,
    platform_argument_capabilities=(
        PlatformArgumentCapability("interactive", SandboxCapability.TERMINAL),
        PlatformArgumentCapability("egress_origins", SandboxCapability.MEDIATED_EGRESS),
        PlatformArgumentCapability("local_ipc", SandboxCapability.LOCAL_IPC),
        PlatformArgumentCapability("ingress_port", SandboxCapability.INGRESS),
        PlatformArgumentCapability("write_files", SandboxCapability.WRITE_FILES),
    ),
)
def run_command(
    context: ToolContext,
    command: Annotated[
        str,
        Field(
            description="Opaque POSIX shell source to run under the host OS sandbox. "
            "Use '.' to source a POSIX script. Select an explicit interpreter for other syntax. "
            "Preserve the exit status when running checks. Use output pagination for long output. "
            "Appending echo or piping a check into a display command can hide its failing exit "
            "status. The result includes the shell's exit code. "
            f'Use "{VirtualPath.TEMPORARY}/..." for session temporary files. '
            f'"${OUTPUT_DIRECTORY_ENVIRONMENT_VARIABLE}" is the private output directory created '
            "for this launch. Use the session temporary directory for files shared between "
            "commands. Preserve the requested wrapper and checks. Use the tool's output settings "
            "in the first command to put generated files and caches there when using private "
            "scratch. Review possible effects and `write_scope.retry` before any authorized retry.",
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
        list[str] | None | SkipJsonSchema[Literal["scratch"]],
        Field(
            description="Write scope for the command. JSON null (unquoted) applies the default "
            "workspace policy when no exact files are requested. Loop can include eligible "
            "checkout writes in the command review. "
            "Use [] for private scratch and a read-only workspace. "
            f'Configure generated outputs under "${OUTPUT_DIRECTORY_ENVIRONMENT_VARIABLE}". '
            "A nonempty list names existing directories subject to filesystem policy.",
        ),
        BeforeValidator(normalize_optional_null),
    ] = None,
    write_files: Annotated[
        Annotated[list[str], Field(json_schema_extra={"minItems": 1})] | None,
        Field(
            description="Request write access to exact workspace file paths. Give a "
            "$WORKSPACE_ROOT path or a workspace-relative path. A file can be absent if its "
            "parent directory already exists. Each path covers one known fixed filename. "
            "Use private scratch or a dedicated directory in write_roots for outputs with "
            "generated names, siblings, temporary files or caches. Use JSON null (unquoted) "
            "for no exact file writes."
        ),
        BeforeValidator(normalize_optional_null),
    ] = None,
    executable_grant: Annotated[
        str | list[str] | None,
        Field(
            description="For an already readable tool, use JSON null (unquoted). An external "
            "tool lookup can return a grant_reference. Copy its exact value here only when "
            "the requested installation needs it."
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
    runtime_services: Annotated[
        list[str] | None,
        Field(
            description="Request additional runtime services only when the command or its "
            "dependencies need them. Use JSON null (unquoted) for ordinary execution."
        ),
        BeforeValidator(normalize_optional_null),
    ] = None,
    ingress_port: Annotated[
        int | None,
        Field(
            description="Publish this attempt on exactly 127.0.0.1:<port> after approval. "
            "Publication ends with this command or policy revision. "
            "Cannot combine with general networking, egress_origins or interactive mode. "
            "When no publication is needed, use JSON null (unquoted).",
            strict=True,
            ge=1,
            le=65535,
        ),
        BeforeValidator(normalize_optional_null),
    ] = None,
    local_ipc: Annotated[
        bool,
        Field(
            description="Request private local communication for this attempt. Defaults off. "
            "This does not publish a host port or grant outbound IP access."
        ),
    ] = False,
    egress_origins: Annotated[
        list[str] | None,
        Field(
            description="Exact HTTP(S) origins requested through a qualified native mediator. "
            "Keep network=false for this scope. "
            "Each new connection is checked against hard limits and the approved scope; "
            "unsupported clients fail closed without a host retry. "
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
    wait_seconds: Annotated[
        float | None,
        Field(
            description=(
                "Wait for up to {maximum_seconds:g} seconds ({maximum_setting}) during the "
                "initial call. The default is {default_seconds:g} seconds ({default_setting}); "
                "use JSON null (unquoted) for the default, including for long commands, or 0 "
                "for an immediate response after approval. The whole command has a runtime "
                "limit of {runtime_limit_seconds:g} seconds ({runtime_limit_setting}). "
                "If the command remains active, poll the returned job handle with "
                "poll_command_job to continue waiting for the same process. Use "
                "read_command_output only when more output pages are needed."
            ),
            ge=0,
            allow_inf_nan=False,
        ),
        BeforeValidator(_wait_value),
    ] = None,
    interactive: Annotated[
        bool,
        Field(
            description="Open one owned sandboxed pseudo-terminal and return its handle after "
            "approval. Use poll_command_job and its output cursors for the same process; use "
            "read_command_output for more output pages and control_command_terminal for input. "
            "Cannot combine with egress_origins."
        ),
    ] = False,
) -> dict | Problem:
    """Run sandboxed shell source. Use the default wait for long jobs and poll the returned handle
    with poll_command_job until completion. Output may expose physical paths."""
    if context.execution_service is None:
        return Problem(
            code="sandbox.unavailable",
            title="Sandbox unavailable",
            detail="Native command execution service is unavailable.",
            operation=context.tool_name,
        )
    if runtime_services:
        if context.sandbox_output_only:
            return Problem(
                code="sandbox.unsupported",
                title="Saved command scope is fixed",
                detail="A saved command can use only the runtime services in its saved scope.",
                operation=context.tool_name,
            )
        if any(
            name not in context.execution_service.backend_options
            or name not in SANDBOX_OPTIONS
            or SANDBOX_OPTIONS[name].get("automatic")
            for name in runtime_services
        ):
            return Problem(
                code="sandbox.unsupported",
                title="Requested sandbox access unavailable",
                detail="The active sandbox does not offer the requested runtime service.",
                operation=context.tool_name,
            )
    reason_argument = {"reason": reason} if reason is not None else {}
    return context.execution_service.run_command(
        context,
        command,
        cwd,
        read_roots,
        [] if write_roots == "scratch" else write_roots,
        executable_grant,
        network,
        git_write,
        egress_origins=egress_origins,
        local_ipc=local_ipc,
        backend_options=tuple(runtime_services or context.sandbox_backend_options),
        ingress_port=ingress_port,
        interactive=interactive,
        wait_seconds=_bounded_wait(wait_seconds, context.settings),
        write_files=write_files,
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


@tool(
    requires_execution_service=True,
    definition_transform=_poll_command_job_definition,
)
def poll_command_job(
    context: ToolContext,
    handle: Annotated[str, Field(description="Opaque handle from run_command.", min_length=32)],
    wait_seconds: Annotated[
        float | None,
        Field(
            description=(
                "Wait for command completion or this wait's timeout, up to {maximum_seconds:g} "
                "seconds ({maximum_setting}). The default is {default_seconds:g} seconds "
                "({default_setting}); use JSON null (unquoted) for the default or 0 for an "
                "immediate status check. The whole command has a runtime limit of "
                "{runtime_limit_seconds:g} seconds ({runtime_limit_setting}). Output arriving "
                "during this wait does not return early. The result includes one available "
                "page from each output stream."
            ),
            ge=0,
            allow_inf_nan=False,
        ),
        BeforeValidator(_wait_value),
    ] = None,
    stdout_cursor: Annotated[
        str | None,
        BeforeValidator(normalize_optional_null),
        Field(
            description="Cursor from stdout.next_cursor in a prior run or poll result; "
            "JSON null (unquoted) or an empty string starts at the beginning."
        ),
    ] = None,
    stderr_cursor: Annotated[
        str | None,
        BeforeValidator(normalize_optional_null),
        Field(
            description="Cursor from stderr.next_cursor in a prior run or poll result; "
            "JSON null (unquoted) or an empty string starts at the beginning."
        ),
    ] = None,
) -> dict | Problem:
    """Wait for command completion or timeout and return one page from each output stream."""
    if context.execution_service is None:
        return Problem(
            code="command_job.unavailable",
            title="Command job unavailable",
            detail="Native command execution service is unavailable.",
            operation=context.tool_name,
        )
    return context.execution_service.poll_command_job(
        context,
        handle,
        _bounded_wait(wait_seconds, context.settings),
        stdout_cursor or None,
        stderr_cursor or None,
    )


@tool(requires_execution_service=True)
def read_command_output(
    context: ToolContext,
    handle: Annotated[str, Field(description="Opaque handle from run_command.", min_length=32)],
    stream: Annotated[Literal["stdout", "stderr"], Field(description="Process output stream.")],
    cursor: Annotated[
        str | None,
        BeforeValidator(normalize_optional_null),
        Field(
            description="Opaque continuation cursor. Use JSON null (unquoted) or an empty string "
            "for the start of the stream."
        ),
    ] = None,
    max_bytes: Annotated[
        int,
        Field(
            description="Maximum UTF-8 bytes to return; a character may adjust the boundary.",
            ge=1,
            le=constants.MAX_TOOL_CONTENT_BYTES,
        ),
    ] = constants.MAX_TOOL_CONTENT_BYTES,
) -> dict | Problem:
    """Immediately read more from one command stream when poll pages are insufficient."""
    if context.execution_service is None:
        return Problem(
            code="command_job.unavailable",
            title="Command output unavailable",
            detail="Native command execution service is unavailable.",
            operation=context.tool_name,
        )
    return context.execution_service.read_command_output(
        context,
        handle,
        stream,
        cursor or None,
        max_bytes,
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
