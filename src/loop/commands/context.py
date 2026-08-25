"""Define command invocation context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..interaction import Interaction


@dataclass(frozen=True)
class CommandContext:
    """Provide runtime metadata and interaction services to a command.

    Args:
        name (str): Public name of the command being invoked.
        interaction (Interaction): Service used to communicate with the user.
    """

    name: str
    interaction: Interaction
