"""Tests for tool invocation context."""

from unittest.mock import Mock

from loop.context import ToolContext
from loop.interaction import Interaction


def test_confirm_delegates_to_the_interaction():
    """Confirmation forwards the question and default through the interaction boundary."""
    interaction = Mock(spec=Interaction)
    interaction.confirm.return_value = True
    context = ToolContext(interaction=interaction, tool_name="guarded")

    assert context.confirm("Continue?", default=True) is True
    interaction.confirm.assert_called_once_with("Continue?", default=True)
