"""Define command invocation context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..models import CommandCapability

if TYPE_CHECKING:
    from ..interaction import Interaction
    from .command import Command


@dataclass(frozen=True)
class CommandContext:
    """Provide runtime metadata and interaction services to a command.

    Args:
        name (str): Public name of the command being invoked.
        interaction (Interaction): Service used to communicate with the user.
        available_commands (tuple[Command, ...] | None): Context-filtered catalog available to the
            command, used by help. Defaults to ``None`` for the full registry.
        capabilities (frozenset[CommandCapability] | None): Capabilities of the current runtime
            context. ``None`` means that command availability is unrestricted.
    """

    name: str
    interaction: Interaction
    available_commands: tuple[Command, ...] | None = None
    capabilities: frozenset[CommandCapability] | None = None
