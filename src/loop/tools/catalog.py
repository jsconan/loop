"""Compose the built-in tool catalog without sharing runtime registry state."""

from ..interaction import Interaction
from ..permissions import PermissionManager
from ..tooling import ToolRegistry
from .dates import get_current_datetime
from .files import delete_path, edit_text_file, list_folder, read_text_file, write_text_file
from .skills import manage_skills
from .system import run_command
from .web import fetch_content, read_cached_content

BUILTIN_TOOLS = (
    get_current_datetime,
    list_folder,
    read_text_file,
    write_text_file,
    edit_text_file,
    delete_path,
    manage_skills,
    run_command,
    fetch_content,
    read_cached_content,
)
"""Built-in tool declarations available for explicit registry composition."""


def create_default_tool_registry(
    *,
    interaction: Interaction | None = None,
    permission_manager: PermissionManager | None = None,
) -> ToolRegistry:
    """Create an isolated registry containing every built-in tool.

    Args:
        interaction (Interaction | None): Default interaction for context-aware dispatch, or
            ``None`` to require an invocation-specific interaction.
        permission_manager (PermissionManager | None): Policy manager guarding calls, or ``None``
            to create the registry's default manager.

    Returns:
        ToolRegistry: A new registry containing the complete built-in tool manifest.
    """
    return ToolRegistry(
        BUILTIN_TOOLS,
        interaction=interaction,
        permission_manager=permission_manager,
    )
