"""Exercise captured compaction ranges, fallback, and atomic context publication."""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from loop import Agent, Message, ModelSelection, Session, SessionManager
from loop.agent import AgentContext
from loop.backend import estimate_context
from loop.compaction import ContextCompaction
from loop.configuration import ApplicationSettings, ContextSettings
from loop.instructions import InstructionsManager
from loop.interaction import Interaction
from loop.models import CompactionContextItem, CompactionResult, DeveloperInput, Usage
from loop.session import ModelWindow, WindowSettings
from loop.utils import PathHolder
from tests.backend_doubles import model_backend


def _history(turns=7):
    """Build unmistakable complete turns with an unfinished current request."""
    return [
        Message(role="user", content="opening"),
        Message(role="user", content="opening addition"),
        Message(role="assistant", content="first answer"),
        *(
            item
            for turn in range(1, turns)
            for item in (
                Message(role="user", content=f"request {turn}"),
                Message(role="assistant", content=f"answer {turn}"),
            )
        ),
        Message(role="user", content="active request"),
    ]


def _result(*, native=False):
    """Produce a replacement block with billing distinct from generation context."""
    item = (
        CompactionContextItem(data={"type": "compaction", "encrypted_content": "opaque"})
        if native
        else Message(role="user", content="Middle summary")
    )
    return CompactionResult(
        items=(item,), usage=Usage(input_tokens=100, output_tokens=10, total_tokens=110)
    )


def _feature(tmp_path, *, session=None, threshold=0.8):
    """Compose a compactor around an isolated session and backend transport."""
    owner = SessionManager(session=session)
    backend = model_backend(
        get_context_window=Mock(return_value=32768),
        get_context_window_async=AsyncMock(return_value=32768),
        instruction_layout=lambda model: "inline",
    )
    selection = ModelSelection(backend, owner)
    interaction = Mock(spec=Interaction)
    instructions = InstructionsManager(project_instructions="project guidance")
    agent = Agent("Assistant")
    context = AgentContext(
        agent,
        backend,
        instructions,
        owner,
        selection,
        PathHolder(tmp_path),
        settings=ContextSettings(),
    )
    feature = ContextCompaction(owner, context, interaction, tmp_path, threshold=threshold)
    return feature, owner, backend, interaction, instructions, agent, context


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1])
def test_rejects_invalid_thresholds(tmp_path, threshold):
    """Automatic utilization must be in the same range as configuration."""
    with pytest.raises(ValueError, match="greater than zero and at most one"):
        _feature(tmp_path, threshold=threshold)


def test_threshold_accepts_full_capacity(tmp_path):
    """A threshold of one compacts only when context reaches the reported limit."""
    session = Session(messages=[Message(role="user", content="request")])
    feature, manager, *_ = _feature(tmp_path, session=session, threshold=1)
    manager.context_window = 100

    assert not feature.needed(99)
    assert feature.needed(100)


def test_threshold_counts_projected_input_and_requires_known_capacity(tmp_path):
    """A projected request can trigger policy before the last observed usage crosses it."""
    session = Session(messages=[Message(role="user", content="hello")], tokens=79)
    feature, manager, *_ = _feature(tmp_path, session=session)
    manager.context_window = 100
    assert feature.threshold == 0.8
    assert feature.can_compact() is True
    assert feature.needed() is False
    assert feature.needed(80) is True
    session.tokens = 80
    assert feature.needed() is True
    manager.context_window = None
    assert feature.needed(100) is False
    feature.threshold = 0.7
    assert feature.threshold == 0.7
    feature.threshold = 1
    assert feature.threshold == 1


def test_skips_no_context_and_empty_middle_needs_no_summary(tmp_path):
    """Explicit rebuild can install instructions for an entirely retained window."""
    feature, owner, backend, interaction, *_ = _feature(tmp_path)
    assert feature.compact() is False
    interaction.warning.assert_called_once_with("There is no new session context to compact.")
    owner.add_user_message("Only the active request")
    assert feature.compact_if_needed() is False
    assert feature.compact() is True
    backend.compact.assert_not_called()
    checkpoint = owner.session.compactions[-1]
    assert checkpoint.context == ()
    assert checkpoint.coverage.tail == (0,)
    assert feature.compact() is False
    assert feature.can_compact() is False
    candidate = owner.session.window.model_copy(update={"checkpoint_id": None})
    assert feature.needed(32768, window=candidate)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("outcome", ["unsupported", "empty"])
def test_empty_or_unsupported_middle_result_keeps_previous_window(tmp_path, asynchronous, outcome):
    """An unusable nonempty summary attempt leaves the source history and window eligible."""
    feature, owner, backend, interaction, *_ = _feature(
        tmp_path, session=Session(messages=_history(), tokens=29000)
    )
    method = backend.compact_async if asynchronous else backend.compact
    if outcome == "unsupported":
        method.side_effect = NotImplementedError
    else:
        method.return_value = None
    if asynchronous:
        completed = asyncio.run(feature.compact_if_needed_async())
    else:
        completed = feature.compact_if_needed()
    assert completed is False
    assert owner.session.window is None
    assert not owner.session.compactions
    assert owner.tokens == 29000
    problem = interaction.warning.call_args.args[0]
    assert (
        "does not support context compaction"
        if outcome == "unsupported"
        else "did not produce compacted context"
    ) in problem


@pytest.mark.parametrize("asynchronous", [False, True])
def test_retains_opening_and_recent_turns_around_middle(tmp_path, asynchronous):
    """The checkpoint replaces only the captured middle and estimates the whole generation."""
    messages = _history()
    feature, owner, backend, interaction, *_, context = _feature(
        tmp_path, session=Session(messages=messages, tokens=29000)
    )
    method = backend.compact_async if asynchronous else backend.compact
    method.return_value = _result()
    if asynchronous:
        assert asyncio.run(feature.compact_if_needed_async())
    else:
        assert feature.compact_if_needed()
    submitted = method.call_args.args[0]
    assert [item.content for item in submitted] == [
        "first answer",
        "request 1",
        "answer 1",
        "request 2",
        "answer 2",
    ]
    assert method.call_args.kwargs == {
        "model": "default-model",
        "prefix": True,
    }
    checkpoint = owner.session.compactions[-1]
    assert checkpoint.boundary == len(messages)
    assert checkpoint.coverage.opening == (0, 1)
    assert checkpoint.coverage.tail == tuple(range(7, len(messages)))
    assert checkpoint.input_tokens_before == 29000
    assert checkpoint.input_tokens_after == owner.tokens > 10
    reconstructed = owner.session.model_context()
    assert isinstance(reconstructed[0], DeveloperInput)
    assert "project guidance" in reconstructed[0].content
    assert reconstructed[1:3] == messages[:2]
    assert reconstructed[3] == _result().items[0]
    assert reconstructed[4:] == messages[7:]
    request = context.prepare(None)
    assert checkpoint.input_tokens_after == estimate_context(
        request.input, instructions=request.instructions, tools=request.window.tools
    )
    assert owner.messages == messages
    assert "estimated" in interaction.info.call_args.args[0]


def test_native_headroom_failure_uses_only_canonical_middle_for_portable_fallback(tmp_path):
    """An unmeasurable opaque block never becomes durable continuation data."""
    feature, owner, backend, *_, context = _feature(
        tmp_path, session=Session(messages=_history(), tokens=30000)
    )
    context.configure(ApplicationSettings(context=ContextSettings(mode="off")))
    backend.compact.side_effect = [_result(native=True), _result()]
    backend.estimate_generation = Mock(side_effect=[None, 1200])
    assert feature.compact()
    assert backend.compact.call_count == 2
    assert backend.compact.call_args.kwargs["native"] is False
    assert all(isinstance(item, Message) for item in backend.compact.call_args.args[0])
    assert owner.session.compactions[-1].context == _result().items
    assert owner.tokens == 1200


def test_unfit_portable_candidate_keeps_original_context(tmp_path):
    """An oversized rebuilt request does not replace a usable prior prefix."""
    feature, owner, backend, interaction, *_ = _feature(
        tmp_path, session=Session(messages=_history(), tokens=29000)
    )
    backend.compact.return_value = _result()
    backend.estimate_generation = Mock(return_value=40000)
    assert feature.compact() is False
    assert not owner.session.compactions
    assert "exceed the available context limit" in interaction.warning.call_args.args[0]


@pytest.mark.parametrize("layout", ["leading", "inline", "suffix"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_rebuild_reserves_disclosure_headroom_without_sending_facts(tmp_path, layout, asynchronous):
    """Budget default disclosure while keeping the backend estimator's inputs stable."""
    feature, owner, backend, interaction, *_, context = _feature(
        tmp_path, session=Session(messages=_history())
    )
    backend.instruction_layout = lambda model: layout
    backend.compact.return_value = backend.compact_async.return_value = _result()
    available = owner.context_window - backend.generation_reserve(owner.context_window)
    estimate = AsyncMock(return_value=available) if asynchronous else Mock(return_value=available)
    if asynchronous:
        backend.estimate_generation_async = estimate
    else:
        backend.estimate_generation = estimate

    outcome = asyncio.run(feature.compact_async()) if asynchronous else feature.compact()

    assert outcome is False
    assert not owner.session.compactions
    assert "exceed the available context limit" in interaction.warning.call_args.args[0]
    assert "<environment_context>" not in (estimate.call_args.kwargs["instructions"] or "")
    assert "<environment_context>" not in repr(estimate.call_args.args[0])

    context.configure(ApplicationSettings(context=ContextSettings(mode="off")))
    outcome = asyncio.run(feature.compact_async()) if asynchronous else feature.compact()

    assert outcome is True
    assert owner.tokens == available


@pytest.mark.parametrize("native", [False, True])
def test_async_compaction_rechecks_headroom_after_disclosure_changes(tmp_path, native):
    """An update during async instruction validation affects the fit of the candidate."""
    feature, owner, backend, interaction, *_, context = _feature(
        tmp_path, session=Session(messages=_history())
    )
    context.configure(ApplicationSettings(context=ContextSettings(mode="off")))
    backend.compact.return_value = _result()
    assert feature.compact()
    previous = owner.session.checkpoint()
    owner.add_message(Message(role="assistant", content="Finished the previous request"))
    owner.add_user_message("Continue from the completed work")
    available = owner.context_window - backend.generation_reserve(owner.context_window)
    backend.compact_async.side_effect = [_result(native=native), _result()]
    backend.estimate_generation_async = AsyncMock(side_effect=[available, 1200])

    async def compact_with_live_update():
        """Enable disclosure while the compactor is waiting for revalidation."""
        validating = asyncio.Event()
        updated = asyncio.Event()
        validate = context.check_rebuild_async

        async def wait_for_update(window):
            """Suspend instruction validation at its public async boundary."""
            validating.set()
            await updated.wait()
            return await validate(window)

        context.check_rebuild_async = wait_for_update
        async with asyncio.TaskGroup() as group:
            pending = group.create_task(feature.compact_async())
            await asyncio.wait_for(validating.wait(), timeout=5)
            context.configure(ApplicationSettings(context=ContextSettings(mode="basic")))
            updated.set()
        return pending.result()

    assert asyncio.run(compact_with_live_update()) is native
    if native:
        assert backend.compact_async.await_count == 2
        assert backend.compact_async.call_args.kwargs["native"] is False
        assert owner.session.checkpoint().context == _result().items
        assert 1200 < owner.tokens <= available
    else:
        backend.compact_async.assert_awaited_once()
        assert owner.session.compactions == [previous]
        assert "exceed the available context limit" in interaction.warning.call_args.args[0]


def test_appended_input_replays_once_after_captured_frontier(tmp_path):
    """A turn admitted while the provider summarizes is outside the checkpoint coverage."""
    messages = _history()
    feature, owner, backend, *_ = _feature(
        tmp_path, session=Session(messages=messages, tokens=29000)
    )
    captured_end = len(messages)

    async def complete(*args, **kwargs):
        """Admit a new user item before the provider returns its summary."""
        await owner.add_user_message_async("while compacting")
        return _result()

    backend.compact_async.side_effect = complete
    assert asyncio.run(feature.compact_async())
    assert owner.session.compactions[-1].boundary == captured_end
    texts = [item.content for item in owner.model_context if isinstance(item, Message)]
    assert texts.count("while compacting") == 1
    assert texts[-2:] == ["active request", "while compacting"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_compaction_discards_a_result_when_the_active_session_was_replaced(tmp_path, asynchronous):
    """A stale rebuild leaves the prior checkpoint and the replacement session intact."""
    feature, owner, backend, interaction, *_ = _feature(
        tmp_path, session=Session(messages=_history())
    )
    backend.compact.return_value = _result()
    assert feature.compact()
    original = owner.session
    checkpoint = original.checkpoint()
    owner.add_message(Message(role="assistant", content="finish earlier request"))
    owner.add_user_message("another request")
    replacement = Session(messages=[Message(role="user", content="Different conversation")])

    def provider_result(*args, **kwargs):
        """Replace the active snapshot before publishing the provider's result."""
        owner.load_session(replacement)
        return _result()

    if asynchronous:
        backend.compact_async = AsyncMock(side_effect=provider_result)
        assert not asyncio.run(feature.compact_async())
    else:
        backend.compact.side_effect = provider_result
        assert not feature.compact()

    assert original.compactions == [checkpoint]
    assert owner.session is replacement and not replacement.compactions
    assert "changed during compaction" in interaction.warning.call_args.args[0]


def test_async_cancellation_keeps_checkpoint_and_effective_window(tmp_path):
    """Cancelling before provider completion leaves the committed prefix in place."""
    feature, owner, backend, *_ = _feature(
        tmp_path, session=Session(messages=_history(), tokens=29000)
    )
    backend.compact.return_value = _result()
    assert feature.compact()
    window = owner.session.window
    checkpoint = owner.session.compactions[-1]
    owner.add_message(Message(role="assistant", content="The active answer"))
    owner.add_user_message("a new active request")

    async def cancel_attempt():
        """Cancel the task after the provider has received the captured plan."""
        started = asyncio.Event()

        async def wait_for_cancellation(*args, **kwargs):
            """Leave provider completion pending until cancelled."""
            started.set()
            await asyncio.Event().wait()

        backend.compact_async.side_effect = wait_for_cancellation
        attempt = asyncio.create_task(feature.compact_async())
        await started.wait()
        attempt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await attempt

    asyncio.run(cancel_attempt())
    assert owner.session.window == window
    assert owner.session.compactions == [checkpoint]


def test_bound_candidate_changes_are_atomic_with_validation(tmp_path):
    """A stale instruction candidate never adopts its pending settings."""
    feature, owner, backend, interaction, instructions, agent, context = _feature(tmp_path)
    owner.add_user_message("opening")
    assert feature.compact()
    original = owner.session.window
    candidate = ModelWindow(
        id="candidate",
        route=backend.route,
        model="default-model",
        layout="inline",
        settings=WindowSettings(agent_name="Updated"),
        baseline=instructions.prepare(replace(agent, identity="Updated")),
        tools=(),
    )
    context.capture = Mock(return_value=candidate)
    context.capture_async = AsyncMock(return_value=candidate)
    context.check_rebuild = Mock(return_value=False)
    context.apply_window = Mock()
    assert not feature.compact()
    assert owner.session.window == original
    context.apply_window.assert_not_called()
    assert "changed during compaction" in interaction.warning.call_args.args[0]
    context.check_rebuild_async = AsyncMock(return_value=True)
    assert asyncio.run(feature.compact_async())
    assert owner.session.window.settings.agent_name == "Updated"
    assert owner.session.compactions[0].boundary == owner.session.compactions[1].boundary
    context.apply_window.assert_called_once_with(owner.session.window)


def test_async_portable_rebuild_resumes_from_canonical_middle_when_native_count_is_unavailable(
    tmp_path,
):
    """Async fallback estimates the portable rebuild before publishing the same capture."""
    feature, owner, backend, *_ = _feature(
        tmp_path, session=Session(messages=_history(), tokens=30000)
    )
    backend.compact_async.side_effect = [_result(native=True), _result()]
    backend.estimate_generation_async = AsyncMock(side_effect=[None, 1200])
    assert asyncio.run(feature.compact_async())
    assert backend.compact_async.await_count == 2
    assert backend.compact_async.call_args.kwargs["native"] is False
    assert owner.session.compactions[-1].context == _result().items


@pytest.mark.parametrize("asynchronous", [False, True])
def test_lower_capacity_abandons_unfit_pending_prefix(tmp_path, asynchronous):
    """An oversized instruction and tool prefix fails before any inference."""
    feature, owner, backend, interaction, *_ = _feature(
        tmp_path, session=Session(messages=_history(), tokens=30000)
    )
    owner.context_window = 16
    if asynchronous:
        assert not asyncio.run(feature.compact_async())
    else:
        assert not feature.compact()
    interaction.warning.assert_called_once_with(
        "Instructions and tools exceed the available context limit."
    )
    assert not owner.session.compactions
    backend.compact.assert_not_called()
    backend.compact_async.assert_not_called()


def test_repeated_compaction_reuses_eligible_coverage_once(tmp_path):
    """A later middle includes the previous checkpoint and each newly uncovered item once."""
    feature, owner, backend, *_ = _feature(
        tmp_path, session=Session(messages=_history(), tokens=30000)
    )
    backend.compact.return_value = _result()
    assert feature.compact()
    old = owner.session.compactions[-1]
    owner.add_message(Message(role="assistant", content="finish active"))
    owner.add_user_message("new active")
    assert feature.compact()
    input_items = backend.compact.call_args.args[0]
    assert input_items[0] == old.context[0]
    assert [item.content for item in input_items[1:]] == ["request 3", "answer 3"]
    assert owner.session.compactions[-1].boundary > old.boundary


def test_async_compaction_declines_to_pin_empty_fresh_history(tmp_path):
    """Manual compaction before any input leaves the agent window unpinned."""
    feature, owner, backend, *_ = _feature(tmp_path)
    assert not asyncio.run(feature.compact_async())
    assert owner.session.window is None
    backend.compact_async.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_portable_summary_without_reported_capacity(tmp_path, asynchronous, empty):
    """Without a reported capacity, compaction uses a bounded, nonempty portable summary."""
    feature, owner, backend, *_ = _feature(tmp_path, session=Session(messages=_history()))
    owner.context_window = None
    method = backend.compact_async if asynchronous else backend.compact
    method.return_value = CompactionResult(items=()) if empty else _result()
    if asynchronous:
        outcome = asyncio.run(feature.compact_async())
    else:
        outcome = feature.compact()
    assert outcome is not empty
    assert method.call_args.kwargs["native"] is False
    assert bool(owner.session.compactions) is not empty


@pytest.mark.parametrize("asynchronous", [False, True])
def test_portable_batches_stop_before_reusing_an_oversized_summary(tmp_path, asynchronous):
    """A provider summary must fit alongside the next batch before Loop sends it."""
    history = [
        Message(role="user", content="first request"),
        Message(role="assistant", content="x" * 33000),
        Message(role="user", content="another historical request"),
        Message(role="assistant", content="y" * 33000),
        *_history(),
    ]
    feature, owner, backend, interaction, *_ = _feature(tmp_path, session=Session(messages=history))
    owner.context_window = None
    method = backend.compact_async if asynchronous else backend.compact
    method.return_value = CompactionResult(items=(Message(role="user", content="z" * 90000),))

    outcome = asyncio.run(feature.compact_async()) if asynchronous else feature.compact()

    assert outcome is False
    assert method.call_count == 1
    interaction.warning.assert_called_once_with(
        "The portable checkpoint exceeds the summary limit."
    )
    assert not owner.session.compactions


@pytest.mark.parametrize("asynchronous", [False, True])
def test_ineligible_native_output_is_replaced_once_with_portable_context(tmp_path, asynchronous):
    """Replay compatibility is checked before estimating or writing a native candidate."""
    feature, owner, backend, *_ = _feature(tmp_path, session=Session(messages=_history()))
    backend.can_replay_checkpoint = Mock(side_effect=[False, True])
    method = backend.compact_async if asynchronous else backend.compact
    method.side_effect = [_result(native=True), _result()]
    if asynchronous:
        assert asyncio.run(feature.compact_async())
    else:
        assert feature.compact()
    assert owner.session.compactions[-1].context == _result().items


@pytest.mark.parametrize("asynchronous", [False, True])
def test_repeated_native_compaction_uses_measured_cost_instead_of_wire_size(tmp_path, asynchronous):
    """Opaque bytes do not disqualify a measured checkpoint from a repeated rebuild."""
    feature, owner, backend, *_ = _feature(tmp_path, session=Session(messages=_history()))
    native = CompactionContextItem(
        data={"type": "compaction", "encrypted_content": "opaque" * 32000}
    )
    backend.compact.return_value = CompactionResult(items=(native,))
    backend.estimate_generation = Mock(return_value=1000)
    assert feature.compact()
    owner.add_message(Message(role="assistant", content="finish earlier request"))
    owner.add_user_message("another request")
    method = backend.compact_async if asynchronous else backend.compact
    method.return_value = _result()

    if asynchronous:
        assert asyncio.run(feature.compact_async())
    else:
        assert feature.compact()

    assert method.call_args.args[0][0] == native
    assert method.call_args.kwargs.get("native", True)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_rejected_prior_checkpoint_restarts_summarization_from_canonical_history(
    tmp_path,
    asynchronous,
):
    """An opaque continuation error must not silently lose its covered canonical content."""
    feature, owner, backend, *_ = _feature(tmp_path, session=Session(messages=_history()))
    backend.compact.return_value = _result()
    assert feature.compact()
    owner.add_message(Message(role="assistant", content="finish earlier request"))
    owner.add_user_message("another request")
    method = backend.compact_async if asynchronous else backend.compact
    method.side_effect = [None, _result()]
    if asynchronous:
        assert asyncio.run(feature.compact_async())
    else:
        assert feature.compact()
    canonical = method.call_args.args[0]
    assert all(isinstance(item, Message) for item in canonical)
    assert canonical[0].content == "first answer"


def test_async_unfit_or_stale_attempt_never_commits(tmp_path):
    """Headroom and captured instruction checks both precede async persistence."""
    feature, owner, backend, interaction, instructions, agent, context = _feature(
        tmp_path, session=Session(messages=_history())
    )
    backend.compact_async.return_value = _result()
    backend.estimate_generation_async = AsyncMock(side_effect=[99999, 1024])
    assert not asyncio.run(feature.compact_async())
    assert "context limit" in interaction.warning.call_args.args[0]
    candidate = ModelWindow(
        id="candidate",
        route=backend.route,
        model="default-model",
        layout="inline",
        settings=WindowSettings(agent_name="Assistant"),
        baseline=instructions.prepare(agent),
        tools=(),
    )
    context.capture_async = AsyncMock(return_value=candidate)
    context.check_rebuild_async = AsyncMock(return_value=False)
    assert not asyncio.run(feature.compact_async())
    assert "changed during compaction" in interaction.warning.call_args.args[0]
    assert not owner.session.compactions


@pytest.mark.parametrize("eligible_before_planning", [False, True])
def test_ineligible_earlier_coverage_is_not_used_to_compact_again(
    tmp_path, eligible_before_planning
):
    """Previously stored provider blocks are optional when the active route cannot replay them."""
    feature, owner, backend, *_ = _feature(tmp_path, session=Session(messages=_history()))
    backend.compact.return_value = _result()
    assert feature.compact()
    owner.add_message(Message(role="assistant", content="finish earlier request"))
    owner.add_user_message("another request")
    backend.can_replay_checkpoint = Mock(
        side_effect=[True, False, True] if eligible_before_planning else [False, True]
    )
    assert feature.compact()
    assert all(isinstance(item, Message) for item in backend.compact.call_args.args[0])
