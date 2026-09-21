"""Compose the built-in tool catalog without sharing runtime registry state."""

from collections.abc import Callable

from ..execution.command import SandboxCommandExecutor
from ..execution.host import HostExecutionBroker
from ..interaction import Interaction
from ..permissions import PermissionManager
from ..tooling import ToolRegistry, ToolRuntimeSettings
from .dates import get_current_datetime
from .files import (
    delete_path,
    edit_text_file,
    list_folder,
    read_text_file,
    search_text,
    write_text_file,
)
from .skills import activate_skill, manage_skills
from .system import (
    attach_command_job,
    cancel_command_job,
    command_job_status,
    detach_command_job,
    read_command_job,
    resize_command_job,
    run_command,
    run_host_command,
    signal_command_job,
    start_command_job,
    write_command_job,
)
from .web import fetch_content, read_cached_content

BUILTIN_TOOLS = (
    get_current_datetime,
    list_folder,
    read_text_file,
    search_text,
    write_text_file,
    edit_text_file,
    delete_path,
    activate_skill,
    manage_skills,
    run_command,
    start_command_job,
    command_job_status,
    attach_command_job,
    read_command_job,
    write_command_job,
    resize_command_job,
    signal_command_job,
    detach_command_job,
    cancel_command_job,
    run_host_command,
    fetch_content,
    read_cached_content,
)
"""Built-in tool declarations available for explicit registry composition."""


def create_default_tool_registry(
    *,
    interaction: Interaction | None = None,
    permission_manager: PermissionManager | None = None,
    settings: ToolRuntimeSettings | None = None,
    command_executor: SandboxCommandExecutor | None = None,
    host_execution_broker: HostExecutionBroker | None = None,
    cancellation: Callable[[], bool] = lambda: False,
) -> ToolRegistry:
    """Create an isolated registry containing every built-in tool.

    Args:
        interaction (Interaction | None): Default interaction for context-aware dispatch, or
            ``None`` to require an invocation-specific interaction.
        permission_manager (PermissionManager | None): Policy manager guarding calls, or ``None``
            to create the registry's default manager.
        settings (ToolRuntimeSettings | None): Scoped settings supplied to context-aware built-in
            tools, or ``None`` to use built-in tool defaults.
        command_executor (SandboxCommandExecutor | None): Sandbox-only ordinary command service.
        host_execution_broker (HostExecutionBroker | None): Separately authorized host-only
            execution service.
        cancellation (Callable[[], bool]): Predicate requesting cancellation of blocking tools.

    Returns:
        ToolRegistry: A new registry containing the complete built-in tool manifest.
    """
    return ToolRegistry(
        BUILTIN_TOOLS,
        interaction=interaction,
        permission_manager=permission_manager,
        settings=settings or ToolRuntimeSettings(),
        command_executor=command_executor,
        host_execution_broker=host_execution_broker,
        cancellation=cancellation,
    )
