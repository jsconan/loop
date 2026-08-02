"""Define user commands handled locally by the conversation loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..interaction import Interaction
    from .command_manager import CommandManager


@dataclass(frozen=True)
class Command:
    """Describe a locally handled user command.

    Args:
        name (str): Command name including its leading slash.
        description (str): Short description shown by command discovery.
        handler (Callable[[CommandManager, Interaction, str], None]): Function receiving the
            active manager, interaction, and stripped argument text.
    """

    name: str
    description: str
    handler: Callable[[CommandManager, Interaction, str], None]
