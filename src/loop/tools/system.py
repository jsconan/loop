"""Provide tools for interacting with the system."""

from typing import Annotated

from pydantic import Field

from ..errors import Problem
from ..permissions import Action, Operation, OperationPlan, ProcessBoundary, ProcessTarget
from ..tooling import ToolContext, tool
from ..utils import VirtualPath


@tool(requires_execution_service=True)
def resolve_executable(
    context: ToolContext,
    name: Annotated[
        str,
        Field(description="One executable name to check on the host PATH, without running it."),
    ],
    cwd: Annotated[
        str,
        "loop:virtual-path",
        "loop:workspace-cwd",
        Field(
            description="Workspace working directory used to resolve relative host PATH entries."
        ),
    ] = ".",
) -> dict | Problem:
    """Check whether a named host executable exists before requesting a sandbox grant."""
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
    return OperationPlan(
        arguments=normalized,
        operations=(
            Operation(
                tool_id="",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", source), cwd=cwd, boundary=ProcessBoundary.SANDBOXED
                ),
            ),
        ),
    )


@tool(
    actions={Action.PROCESS_EXECUTE},
    operation_planner=_command_plan,
    execution_authorization=True,
)
def run_command(
    context: ToolContext,
    command: Annotated[
        str,
        Field(
            description="Opaque POSIX shell source to run under the host OS sandbox.",
            min_length=1,
        ),
    ],
    cwd: Annotated[
        str,
        "loop:virtual-path",
        "loop:workspace-cwd",
        Field(
            description="Virtual working directory. 'workspace' selects "
            f"{VirtualPath.WORKSPACE}; relative paths use that root. "
            "Literal virtual paths in shell source are mapped "
            "to approved local paths before execution."
        ),
    ] = ".",
    read_only: Annotated[
        bool,
        Field(description="Disable workspace writes while retaining sandboxed shell access."),
    ] = False,
    read_roots: Annotated[
        list[str] | None,
        Field(
            description="Extra host or virtual directories to read inside the sandbox. "
            "Installed tools outside reviewed system roots need a grant for their executable "
            "and support files; use resolve_executable and executable_grant when host paths "
            "are private. PATH alone grants lookup metadata."
        ),
    ] = None,
    executable_grant: Annotated[
        str | list[str] | None,
        Field(
            description="Optional opaque grant_reference values from resolve_executable "
            "for installations outside the automatically verified managed toolchains. "
            "Each reference binds installed-tool read access inside the sandbox."
        ),
    ] = None,
    network: Annotated[
        bool,
        Field(description="Request a separately approved general network grant."),
    ] = False,
    git_write: Annotated[
        bool,
        Field(
            description="Request a separate sandboxed Git write grant for opaque Git invocations. "
            "Recognized Git state-changing commands request approval automatically."
        ),
    ] = False,
) -> dict | Problem:
    """Run POSIX shell source in the native sandbox, with separately approved host recovery."""
    if context.execution_service is None:
        return Problem(
            code="sandbox.unavailable",
            title="Sandbox unavailable",
            detail="Native command execution service is unavailable.",
            operation=context.tool_name,
        )
    return context.execution_service.run_command(
        context,
        command,
        cwd,
        read_only,
        read_roots,
        executable_grant,
        network,
        git_write,
    )
