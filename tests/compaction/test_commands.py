"""Tests for context compaction commands."""

from unittest.mock import Mock

from loop import CommandManager, ContextCompaction, Interaction
from loop.compaction.commands import CompactionCommands


def test_compaction_command_forces_active_context_compaction():
    """The compact command delegates to the active compaction feature."""
    interaction = Mock(spec=Interaction)
    compaction = Mock(spec=ContextCompaction)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(CompactionCommands(compaction))

    manager.call("compact")

    compaction.compact.assert_called_once_with()
