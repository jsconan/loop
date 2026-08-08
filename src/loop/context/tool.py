"""Define tool invocation context."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..interaction import Interaction

if TYPE_CHECKING:
    from ..skills import InstructionsManager


@dataclass(frozen=True)
class ToolContext:
    """Provide runtime services and metadata to a context-aware tool.

    Args:
        interaction (Interaction): Service used to communicate with the user.
        tool_name (str): Public name of the tool being invoked.
        instructions_manager (InstructionsManager | None): Manager for instructions active in the
            current conversation, or ``None`` when instruction management is unavailable.
    """

    interaction: Interaction
    tool_name: str
    instructions_manager: InstructionsManager | None = None

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask the user to confirm an action through the interaction service.

        Args:
            message (str): Confirmation question to display.
            default (bool): Answer to use when the user enters no response.

        Returns:
            bool: Whether the user approved the action.
        """
        return self.interaction.confirm(message, default=default)
