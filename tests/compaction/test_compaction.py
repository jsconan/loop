"""Tests for active conversation context compaction."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop import (
    CompactionContextItem,
    CompactionResult,
    ContextCompaction,
    InstructionsManager,
    Interaction,
    Message,
    ModelSelection,
    Session,
    SessionManager,
    Usage,
)
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry


def compaction_feature(
    *,
    session: Session | None = None,
    compact: Mock | None = None,
    threshold: float = 0.8,
):
    """Build context compaction with isolated collaborators."""
    manager = SessionManager(session=session)
    backend = SimpleNamespace(
        compact=compact or Mock(),
        default_model="model",
        get_context_window=Mock(return_value=100),
    )
    selection = ModelSelection(backend, manager)
    interaction = Mock(spec=Interaction)
    instructions = InstructionsManager(project_instructions="instructions")
    feature = ContextCompaction(
        backend,
        manager,
        selection,
        instructions,
        interaction,
        lambda: "/project",
        threshold=threshold,
    )
    return feature, manager, backend, interaction


@pytest.mark.parametrize("threshold", [0, 1, -0.1, 1.1])
def test_compaction_rejects_invalid_thresholds(threshold):
    """Automatic compaction utilization must remain strictly between zero and one."""
    with pytest.raises(ValueError, match="between zero and one"):
        compaction_feature(threshold=threshold)


def test_compaction_detects_known_capacity_and_new_history():
    """Automatic policy requires known capacity, sufficient usage, and uncompacted history."""
    session = Session(messages=[Message(role="user", content="hello")], tokens=79)
    feature, manager, backend, _ = compaction_feature(session=session)

    assert feature.threshold == 0.8
    assert feature.can_compact() is True
    assert feature.needed() is False
    session.tokens = 80
    assert feature.needed() is True
    manager.session.context_window = None
    backend.get_context_window.return_value = None
    assert feature.needed() is False


def test_compaction_declines_absent_or_unneeded_context():
    """Explicit and automatic requests avoid backend work when their policy is not satisfied."""
    feature, _, backend, interaction = compaction_feature()

    assert feature.compact() is False
    interaction.warning.assert_called_once_with("There is no new session context to compact.")
    assert feature.compact(force=False) is False
    backend.compact.assert_not_called()

    session = Session(messages=[Message(role="user", content="hello")], tokens=79)
    feature, _, backend, interaction = compaction_feature(session=session)
    assert feature.compact(force=False) is False
    assert feature.compact_if_needed() is False
    interaction.warning.assert_not_called()
    backend.compact.assert_not_called()


@pytest.mark.parametrize("unsupported", ["error", "empty", "no_items"])
def test_compaction_reports_backend_that_cannot_produce_context(unsupported):
    """Unsupported and unusable backend results leave active context unchanged."""
    result = CompactionResult(items=()) if unsupported == "no_items" else None
    compact = Mock(
        side_effect=NotImplementedError if unsupported == "error" else None,
        return_value=result,
    )
    session = Session(messages=[Message(role="user", content="hello")], tokens=80)
    feature, _, _, interaction = compaction_feature(session=session, compact=compact)
    telemetry = Telemetry(MemoryTelemetryAdapter(), flush_seconds=0.01)
    if unsupported != "no_items":
        set_telemetry(telemetry)

    try:
        assert feature.compact_if_needed() is False
    finally:
        telemetry.close(1)
        set_telemetry(None)
    assert session.compactions == []
    assert interaction.warning.call_args.args[0].startswith("The selected backend")
    if unsupported == "error":
        unobserved, _, _, _ = compaction_feature(
            session=Session(messages=[Message(role="user", content="hello")], tokens=80),
            compact=Mock(side_effect=NotImplementedError),
        )
        assert unobserved.compact_if_needed() is False


def test_compaction_persists_replacement_context_and_reports_usage():
    """Successful compaction snapshots state and replaces model context without losing history."""
    message = Message(role="user", content="hello")
    item = CompactionContextItem(provider="openai", data={"type": "compaction"})
    compact = Mock(
        return_value=CompactionResult(
            items=(item,),
            usage=Usage(total_tokens=100),
            context_tokens=20,
        )
    )
    session = Session(messages=[message], tokens=80)
    feature, manager, backend, interaction = compaction_feature(
        session=session,
        compact=compact,
    )
    telemetry = Telemetry(MemoryTelemetryAdapter(), flush_seconds=0.01)
    set_telemetry(telemetry)

    try:
        assert feature.compact_if_needed() is True
    finally:
        telemetry.close(1)
        set_telemetry(None)

    assert session.messages == [message]
    assert manager.model_context == [item]
    assert session.tokens == 20
    assert session.compactions[0].instructions.working_directory == "/project"
    assert session.compactions[0].instructions.content == "instructions"
    backend.compact.assert_called_once_with(
        [message],
        instructions="instructions",
        model="model",
    )
    interaction.info.assert_any_call("Compacted session context from 80 to 20 tokens.")


def test_compaction_reports_success_without_remeasured_usage():
    """Compaction confirms a checkpoint when the backend cannot report replacement usage."""
    item = CompactionContextItem(provider="openai", data={"type": "compaction"})
    session = Session(messages=[Message(role="user", content="hello")], tokens=20)
    feature, _, _, interaction = compaction_feature(
        session=session,
        compact=Mock(return_value=CompactionResult(items=(item,))),
    )

    assert feature.compact() is True
    interaction.info.assert_any_call("Compacted session context.")


def test_compaction_requires_an_effective_model():
    """Compaction rejects configured sessions that cannot resolve an effective model."""
    manager = SessionManager(session=Session(messages=[Message(role="user", content="hello")]))
    backend = SimpleNamespace(
        compact=Mock(),
        default_model=None,
        get_context_window=Mock(return_value=None),
    )
    feature = ContextCompaction(
        backend,
        manager,
        ModelSelection(backend, manager),
        InstructionsManager(),
        Mock(spec=Interaction),
        lambda: "/project",
    )

    with pytest.raises(ValueError, match="No model was selected"):
        feature.compact()
