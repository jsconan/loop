"""Define command invocation context."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..commands.command_manager import CommandManager
    from ..interaction import Interaction
    from ..permissions import PermissionManager
    from ..skills import SkillManager
    from ..tooling import ToolRegistry


@dataclass(frozen=True)
class CommandContext:
    """Provide runtime metadata and interaction services to a command.

    Args:
        name (str): Public name of the command being invoked.
        interaction (Interaction): Service used to communicate with the user.
        manager (CommandManager | None): Manager dispatching the command, or ``None`` when the
            command is invoked independently.
        permission_manager (PermissionManager | None): Tool policy manager controlled by the
            command, or ``None`` when permission management is unavailable.
        skill_manager (SkillManager | None): Manager exposing the skill catalog, or ``None``
            when skill listing is unavailable.
        tool_registry (ToolRegistry | None): Registry exposing the tool catalog, or ``None``
            when tool listing is unavailable.
    """

    name: str
    interaction: Interaction
    manager: CommandManager | None = None
    permission_manager: PermissionManager | None = None
    skill_manager: SkillManager | None = None
    tool_registry: ToolRegistry | None = None
