"""Define commands available in every conversation loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..interaction import Interaction
    from .command_manager import CommandManager


def help_command(
    manager: CommandManager,
    interaction: Interaction,
    arguments: str,
) -> None:
    """Display the available commands.

    Args:
        manager (CommandManager): Manager providing the authoritative command catalog.
        interaction (Interaction): Active interaction used to display help or argument errors.
        arguments (str): Stripped argument text, which must be empty.
    """
    if arguments:
        interaction.warning("/help does not accept arguments.")
        return
    command_width = max(len(command.name) for command in manager.commands) + 1
    lines = ["Available commands:", ""]
    lines.extend(
        f"  {command.name:<{command_width}} {command.description}" for command in manager.commands
    )
    interaction.info("\n".join(lines))


def exit_command(
    manager: CommandManager,
    interaction: Interaction,
    arguments: str,
) -> None:
    """Request conversation termination.

    Args:
        manager (CommandManager): Manager whose termination state should be updated.
        interaction (Interaction): Active interaction used to display argument errors.
        arguments (str): Stripped argument text, which must be empty.
    """
    if arguments:
        interaction.warning("Exit commands do not accept arguments.")
        return
    manager.request_exit()
