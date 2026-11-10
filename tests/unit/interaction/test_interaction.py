"""Verify optional interaction scopes for existing synchronous adapters."""

from unittest.mock import Mock

import pytest

from loop import Interaction


def test_unspecified_interaction_does_not_claim_fresh_host_approval():
    """Existing input adapters fail closed until they opt into fresh host decisions."""
    assert Interaction.supports_fresh_host_approval.fget(Mock()) is False


def test_default_terminal_context_preserves_synchronous_execution_and_errors():
    """Legacy adapters require no terminal handoff and do not suppress operation failures."""
    operation = Mock()
    with Interaction.terminal_context(Mock()):
        operation()
    operation.assert_called_once_with()
    with pytest.raises(ValueError, match="operation failed"), Interaction.terminal_context(Mock()):
        raise ValueError("operation failed")
