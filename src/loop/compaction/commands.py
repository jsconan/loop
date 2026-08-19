"""Expose active context compaction through user commands."""

from ..commands import CommandContext, CommandRegistration
from .compaction import ContextCompaction


class CompactionCommands:
    """Expose context compaction through an interactive command.

    Args:
        compaction (ContextCompaction): Active conversation compaction feature.
    """

    _compaction: ContextCompaction

    def __init__(self, compaction: ContextCompaction) -> None:
        self._compaction = compaction

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return the compaction command registration.

        Returns:
            tuple[CommandRegistration, ...]: Context compaction command.
        """
        return (CommandRegistration(self.compact, name="compact"),)

    def compact(self, context: CommandContext) -> None:
        """Compact the active session context."""
        del context
        self._compaction.compact()
