"""Tests for context compaction commands."""

import asyncio
from unittest.mock import Mock

import pytest

from loop import CommandManager, ContextCompaction, Interaction
from loop.backend import BackendNotFoundError
from loop.compaction import CompactionCommands
from loop.instructions import InstructionBudgetExceededError
from loop.model_selection import ModelSelectionError


def test_compaction_command_forces_active_context_compaction():
    """The compact command delegates to the active compaction feature."""
    interaction = Mock(spec=Interaction)
    compaction = Mock(spec=ContextCompaction)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(CompactionCommands(compaction))

    manager.call("compact")

    compaction.compact.assert_called_once_with()
    compaction.compact_async.assert_not_awaited()


def test_async_compaction_command_uses_the_async_owner():
    """Async dispatch runs compaction on its cancellable provider path."""
    compaction = Mock(spec=ContextCompaction)
    manager = CommandManager(interaction=Mock(spec=Interaction))
    manager.register_provider(CompactionCommands(compaction))

    asyncio.run(manager.call_async("compact"))

    compaction.compact_async.assert_awaited_once_with()
    compaction.compact.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "error, code",
    [
        (
            BackendNotFoundError(
                "model unavailable",
                provider="test",
                operation="compact",
                status_code=404,
                code="model_not_found",
                request_id="req_fixture",
            ),
            "backend.request_failed",
        ),
        (
            InstructionBudgetExceededError("Current instructions are too large."),
            "compaction.failed",
        ),
        (ModelSelectionError("No model was selected."), "compaction.failed"),
    ],
    ids=["provider", "instructions", "model"],
)
def test_compaction_command_reports_expected_failures(asynchronous, error, code):
    """Expected compaction failures return through command diagnostics."""
    interaction = Mock(spec=Interaction)
    compaction = Mock(spec=ContextCompaction)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(CompactionCommands(compaction))
    compaction.compact.side_effect = error
    compaction.compact_async.side_effect = error

    if asynchronous:
        asyncio.run(manager.call_async("compact"))
    else:
        manager.call("compact")

    problem = interaction.report.call_args.args[0]
    assert problem.code == code
    if isinstance(error, BackendNotFoundError):
        assert problem.metadata["provider_code"] == "model_not_found"
        assert problem.metadata["request_id"] == "req_fixture"
    else:
        assert problem.detail == str(error)
