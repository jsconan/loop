"""Define command invocation context."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..commands.command_manager import CommandManager
    from ..interaction import Interaction
    from ..permissions import PermissionManager


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
    """

    name: str
    interaction: Interaction
    manager: CommandManager | None = None
    permission_manager: PermissionManager | None = None
