"""Tests for active conversation context compaction."""

import asyncio
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from loop import (
    Agent,
    Backend,
    CompactionContextItem,
    CompactionResult,
    InstructionsManager,
    Interaction,
    Message,
    ModelSelection,
    Session,
    SessionManager,
    Usage,
)
from loop.compaction import ContextCompaction
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry
from loop.utils import PathHolder


def prepared_instructions():
    """Build a provider of instructions for one reusable test agent definition."""
    instructions = InstructionsManager()
    agent = Agent("Assistant")
    return lambda: instructions.prepare(agent)


def compaction_feature(
    *,
    session: Session | None = None,
    compact: Mock | None = None,
    threshold: float = 0.8,
    instruction_provider=None,
    instructions_async=None,
):
    """Build context compaction with isolated collaborators."""
    manager = SessionManager(session=session)
    backend = Mock(
        spec=Backend,
        compact=compact or Mock(),
        default_model="model",
        get_context_window=Mock(return_value=100),
    )
    selection = ModelSelection(backend, manager)
    interaction = Mock(spec=Interaction)
    instructions = InstructionsManager(project_instructions="instructions")
    agent = Agent("Assistant")
    feature = ContextCompaction(
        backend,
        manager,
        selection,
        instruction_provider or (lambda: instructions.prepare(agent)),
        interaction,
        PathHolder("/project"),
        threshold=threshold,
        instructions_async=instructions_async,
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


def test_compaction_reconfigures_backend_and_threshold():
    """Future compaction reads an owner-updated threshold and replacement backend."""
    feature, _, backend, _ = compaction_feature(threshold=0.8)
    replacement = Mock()

    feature.backend = replacement
    feature.threshold = 0.7

    assert feature.backend is replacement
    assert feature.threshold == 0.7
    assert backend is not replacement
    with pytest.raises(ValueError, match="between zero and one"):
        feature.threshold = 1


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
    assert session.compactions[0].instructions.content.endswith("instructions")
    assert "<agent_instructions" in session.compactions[0].instructions.content
    backend.compact.assert_called_once_with(
        [message],
        instructions=session.compactions[0].instructions.content,
        model="model",
    )
    interaction.info.assert_any_call("Compacted session context from 80 to 20 tokens.")


def test_async_compaction_persists_replacement_context_after_backend_completion():
    """Async compaction commits the owner state only after a complete backend result."""
    result = CompactionResult(
        items=(
            CompactionContextItem(
                provider="loop",
                data={"role": "user", "content": "checkpoint"},
            ),
        ),
        context_tokens=20,
    )
    session = Session(messages=[Message(role="user", content="hello")], tokens=80)
    feature, manager, backend, interaction = compaction_feature(session=session)
    previous_context = manager.model_context
    backend.compact_async = AsyncMock(return_value=result)

    applied = asyncio.run(feature.compact_if_needed_async())

    assert applied is True
    backend.compact_async.assert_awaited_once()
    assert backend.compact_async.call_args.args[0] == previous_context
    assert backend.compact_async.call_args.kwargs["instructions"]
    assert backend.compact_async.call_args.kwargs["model"] == "model"
    assert len(session.compactions) == 1
    interaction.info.assert_any_call("Compacting session context...")


def test_async_compaction_runs_legacy_instruction_provider_on_a_worker():
    """The compatibility provider is isolated from the event-loop thread."""
    session = Session(messages=[Message(role="user", content="hello")], tokens=80)
    worker_threads = []
    snapshot = prepared_instructions()()

    def provide_instructions():
        """Return the captured instruction snapshot and record provider ownership."""
        worker_threads.append(get_ident())
        return snapshot

    feature, _, backend, _ = compaction_feature(
        session=session,
        instruction_provider=provide_instructions,
    )
    backend.compact_async = AsyncMock(return_value=CompactionResult(items=()))
    owner_thread = get_ident()

    asyncio.run(feature.compact_async())

    assert worker_threads and worker_threads[0] != owner_thread


def test_async_compaction_awaits_a_native_instruction_provider():
    """A configured async instruction provider is awaited directly by compaction."""
    session = Session(messages=[Message(role="user", content="hello")], tokens=80)
    snapshot = prepared_instructions()()
    instructions_async = AsyncMock(return_value=snapshot)
    feature, _, backend, _ = compaction_feature(
        session=session,
        instructions_async=instructions_async,
    )
    backend.compact_async = AsyncMock(return_value=CompactionResult(items=()))

    asyncio.run(feature.compact_async())

    instructions_async.assert_awaited_once_with()


def test_async_compaction_uses_the_explicit_prepared_snapshot():
    """An explicit instruction snapshot bypasses both provider paths."""
    session = Session(messages=[Message(role="user", content="hello")], tokens=80)
    feature, _, backend, _ = compaction_feature(session=session)
    backend.compact_async = AsyncMock(return_value=CompactionResult(items=()))
    snapshot = prepared_instructions()()

    asyncio.run(feature.compact_async(instructions=snapshot))

    backend.compact_async.assert_awaited_once()
    assert backend.compact_async.call_args.kwargs["instructions"] == snapshot.content


def test_async_compaction_skips_absent_or_unneeded_context():
    """Async compaction honors empty-history and utilization boundaries before backend work."""
    feature, _, backend, interaction = compaction_feature()

    assert asyncio.run(feature.compact_async()) is False
    assert asyncio.run(feature.compact_async(force=False)) is False
    assert asyncio.run(feature.compact_if_needed_async()) is False
    interaction.warning.assert_called_once_with("There is no new session context to compact.")
    backend.compact_async.assert_not_awaited()

    session = Session(messages=[Message(role="user", content="hello")], tokens=79)
    feature, _, backend, interaction = compaction_feature(session=session)
    assert asyncio.run(feature.compact_async(force=False)) is False
    assert asyncio.run(feature.compact_if_needed_async()) is False
    interaction.warning.assert_not_called()
    backend.compact_async.assert_not_awaited()


def test_async_compaction_cancellation_does_not_commit_partial_context():
    """Cancellation during model compaction leaves the active session history unchanged."""
    session = Session(messages=[Message(role="user", content="hello")], tokens=80)
    feature, _, backend, _ = compaction_feature(session=session)
    started = asyncio.Event()

    async def compact(*args, **kwargs):
        """Pause after compaction starts so the test can cancel the operation."""
        started.set()
        await asyncio.Event().wait()

    backend.compact_async = AsyncMock(side_effect=compact)

    async def cancel_compaction():
        """Cancel the backend request at a deterministic suspension point."""
        task = asyncio.create_task(feature.compact_if_needed_async())
        try:
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(cancel_compaction())

    assert session.compactions == []
    assert session.messages == [Message(role="user", content="hello")]


@pytest.mark.parametrize("unsupported", ["error", "empty", "no_items"])
def test_async_compaction_reports_unavailable_backend_results(unsupported):
    """Async compaction leaves history intact when the backend cannot produce a checkpoint."""
    result = CompactionResult(items=()) if unsupported == "no_items" else None
    compact = AsyncMock(
        side_effect=NotImplementedError if unsupported == "error" else None,
        return_value=result,
    )
    session = Session(messages=[Message(role="user", content="hello")], tokens=80)
    feature, _, backend, interaction = compaction_feature(session=session)
    backend.compact_async = compact

    assert asyncio.run(feature.compact_if_needed_async()) is False

    assert session.compactions == []
    assert interaction.warning.call_args.args[0].startswith("The selected backend")


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
        prepared_instructions(),
        Mock(spec=Interaction),
        PathHolder("/project"),
    )

    with pytest.raises(ValueError, match="No model was selected"):
        feature.compact()
