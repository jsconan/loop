"""Verify responsive mailbox admission and serialized conversation ownership."""

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from queue import Empty
from threading import Event, Thread, get_ident
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from prompt_toolkit.completion import CompleteEvent, DummyCompleter, WordCompleter
from prompt_toolkit.document import Document

from loop import (
    CommandCapability,
    CommandCompletion,
    CommandContext,
    CommandManager,
    CommandRegistration,
    CompletionManager,
    CompletionValue,
    ConsoleInteraction,
    Interaction,
    MemorySessionStore,
    RunInterrupted,
    RunMetrics,
    SessionController,
    SessionExecutionConflictError,
    SessionManager,
)
from loop.conversation import ConversationCommands, ConversationRecoveryCommands
from loop.conversation import controller as controller_module
from loop.conversation.interaction import MailboxInteraction
from loop.conversation.mailbox import InputMailbox
from loop.conversation.models import (
    EVENT_CONFIRM,
    EVENT_PROMPT,
    EVENT_TERMINAL,
    AdmittedConversationInput,
    InputAdmission,
    InputAdmissionKind,
    InteractionCall,
    PendingInputPriorityOutcome,
    PendingInputRecallOutcome,
    ResponseOutcome,
    SteeringOutcome,
)
from loop.models import SteeringMessage


@pytest.fixture
def ui(monkeypatch):
    """Provide async input and output doubles without global terminal patching."""
    monkeypatch.setattr(controller_module, "patch_stdout", lambda **_kwargs: nullcontext())
    source = MagicMock(spec=ConsoleInteraction)
    source.response_context.side_effect = nullcontext
    source.terminal_context.side_effect = nullcontext
    return source


def read_inputs(ui, inputs):
    """Read normal UI messages from a deterministic async mailbox."""

    async def read(*_args, **_kwargs):
        """Wait for an explicitly supplied user input."""
        return await inputs.get()

    ui.prompt_async.side_effect = read


def test_mailbox_routes_async_input_through_a_bound_sync_owner(ui):
    """Bound synchronous controllers relay async prompts through their UI owner."""
    interaction = MailboxInteraction(ui)
    events = []

    def dispatch(event):
        """Return the deterministic response supplied by the controller owner."""
        events.append(event)
        return "answer"

    assert interaction.worker_calls_routed is False
    interaction.bind(dispatch)

    assert interaction.worker_calls_routed is True
    assert asyncio.run(interaction.prompt_async("Question:")) == "answer"
    assert len(events) == 1
    assert events[0].kind == EVENT_PROMPT
    assert events[0].args == ("Question:",)

    interaction.bind(None)
    assert interaction.worker_calls_routed is False


def test_controller_requires_async_input():
    """Legacy synchronous adapters remain outside the responsive controller."""
    with pytest.raises(TypeError, match="asynchronous input"):
        SessionController(MagicMock(spec=Interaction))


def test_controller_validates_and_exposes_the_active_input_mode(ui):
    """Controller mode values are restricted to the two documented active-run routes."""
    controller = SessionController(ui)

    assert controller.input_mode == "queue"
    controller.input_mode = "steer"
    assert controller.input_mode == "steer"
    with pytest.raises(ValueError, match="queue.*steer"):
        controller.input_mode = "replace"
    with pytest.raises(ValueError, match="queue.*steer"):
        SessionController(ui, input_mode="replace")


def test_controller_pause_parks_the_active_run_without_cancelling_it(ui):
    """A pause request parks the same owner until the user continues it."""
    controller = SessionController(ui)
    assert controller.request_pause() is False
    started = asyncio.Event()
    checkpoint = asyncio.Event()
    resumed = asyncio.Event()

    async def owner():
        """Hold an async run at its next safe checkpoint."""
        with controller.control.execution(async_owner=True):
            started.set()
            await checkpoint.wait()
            await controller.control.checkpoint_async()
            resumed.set()

    async def scenario():
        """Pause the run, verify it remains alive, then continue the same task."""
        task = asyncio.create_task(owner())
        await started.wait()
        assert controller.request_pause() is True
        checkpoint.set()
        await asyncio.sleep(0)
        assert not task.done()
        assert controller.control.cancellation_event.is_set() is False
        assert controller.control.continue_run() is True
        await task

    asyncio.run(scenario())

    assert resumed.is_set()
    assert controller.control.stopped is False


@pytest.mark.parametrize("request_completed", [False, True])
def test_controller_pause_preserves_pending_questions_and_direct_input(ui, request_completed):
    """Pause preserves pending UI work instead of revoking or cancelling it."""
    controller = SessionController(ui)
    request = InteractionCall(ui.prompt, ("Question?",), {})
    replacement = InteractionCall(ui.prompt, ("Replacement?",), {})
    if request_completed:
        request.result.set_result("already answered")
    else:
        request.result.add_done_callback(
            lambda _result: setattr(controller, "_pending", replacement)
        )
    controller._pending = request
    controller._loop = Mock()

    async def scenario():
        """Pause active execution while a separate answer is waiting."""
        answer_input = asyncio.get_running_loop().create_future()
        controller._answer_input = answer_input
        with controller.control.execution():
            assert controller.request_pause() is True
        return answer_input

    answer_input = asyncio.run(scenario())

    if request_completed:
        assert controller._pending is request
        assert request.result.result() == "already answered"
    else:
        assert controller._pending is request
        assert request.result.done() is False
    assert not answer_input.cancelled()


def test_cold_stop_rejects_when_the_run_is_idle(ui):
    """A cold stop is rejected when there is no active run to settle."""
    controller = SessionController(ui)

    assert controller.request_stop_run() is False


def test_cold_stop_preserves_the_owner_and_durable_pending_inputs(ui):
    """A cold stop leaves the controller alive and the durable mailbox intact."""
    manager = SessionManager(session_store=MemorySessionStore())
    manager.add_user_message("active request")
    manager.begin_run()
    manager.queue_input("waiting", priority=1)
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    with controller.control.execution():
        assert controller.request_stop_run() is True

    assert controller.control.stopped is False
    assert controller.control.cancellation_event.is_set() is False
    assert manager.continuable is True
    assert [item.content for item in manager.session.pending_inputs] == ["waiting"]


def test_sync_admission_wakes_an_async_recall_waiter(ui, monkeypatch):
    """A synchronous mailbox writer notifies an asynchronous mutation waiting for its lock."""
    admitted, active = asyncio.Event(), asyncio.Event()
    release = Event()
    finish = asyncio.Event()
    owner_loop = None

    class HeldMailbox(InputMailbox):
        """Hold admission while the controller owns its mailbox mutation lock."""

        def try_put(self, admission):
            """Expose the owned write before allowing the admission to finish."""
            owner_loop.call_soon_threadsafe(admitted.set)
            release.wait()
            return super().try_put(admission)

    monkeypatch.setattr(controller_module, "InputMailbox", HeldMailbox)
    controller = SessionController(ui)

    async def conversation(receive):
        """Keep execution active while concurrent public queue operations settle."""
        with controller.control.execution(async_owner=True):
            active.set()
            await finish.wait()

    async def scenario():
        """Recall an input while a synchronous writer still owns the mailbox lock."""
        nonlocal owner_loop
        owner_loop = asyncio.get_running_loop()
        owner = asyncio.create_task(controller.run_async(conversation))
        await active.wait()
        writer = asyncio.create_task(asyncio.to_thread(controller.queue, "recall me", ui))
        recall = None
        try:
            await admitted.wait()
            recall = asyncio.create_task(controller.recall_latest_async())
            await asyncio.sleep(0)
            assert not recall.done()
            release.set()
            assert await writer
            assert await asyncio.wait_for(recall, timeout=1) is PendingInputRecallOutcome.RECALLED
        finally:
            release.set()
            finish.set()
            await asyncio.gather(
                owner, writer, *([recall] if recall else []), return_exceptions=True
            )

    asyncio.run(scenario())


def test_async_mailbox_lock_waits_without_blocking_the_event_loop(ui):
    """Async mailbox access waits for a synchronous owner and resumes when it releases."""
    controller = SessionController(ui)
    controller._mailbox_lock.acquire()

    async def scenario():
        """Hold an async acquisition until the existing lock owner releases it."""
        waiting = asyncio.create_task(controller._acquire_mailbox_lock_async())
        await asyncio.sleep(0)
        assert waiting.done() is False
        controller._release_mailbox_lock()
        await waiting
        controller._release_mailbox_lock()

    asyncio.run(scenario())


def test_async_mailbox_lock_retries_after_a_release_race(ui):
    """A lock released just before the wait is registered is acquired on the retry."""
    controller = SessionController(ui)
    lock = Mock()
    lock.acquire.side_effect = [False, True]
    controller._mailbox_lock = lock

    asyncio.run(controller._acquire_mailbox_lock_async())
    controller._release_mailbox_lock()

    lock.release.assert_called_once_with()


def test_mailbox_lock_release_is_safe_before_async_waiters_exist(ui):
    """Synchronous mailbox use does not require an async availability event."""
    controller = SessionController(ui)
    controller._mailbox_lock.acquire()

    controller._release_mailbox_lock()

    assert controller._mailbox_available is None


def test_controller_exposes_setup_interaction_and_a_single_execution_lifecycle(ui):
    """Setup uses the original UI and a settled controller cannot be started twice."""
    controller = SessionController(ui)
    assert controller.running is False
    assert controller.control.stopped is False
    controller.interaction.info("setup")
    ui.info.assert_called_once_with("setup")

    controller.run(lambda _receive: None)

    assert controller.running is False
    controller.interaction.info("settled")
    with pytest.raises(RuntimeError, match="Bind the command manager before running"):
        controller.bind_command_manager(CommandManager())
    with pytest.raises(RuntimeError, match="only run once"):
        controller.run(lambda _receive: None)


def test_sync_controller_run_routes_callback_prompts_through_the_async_ui(ui):
    """A synchronous caller keeps prompt ownership on the controller's async UI loop."""
    controller = SessionController(ui)
    ui.prompt_async = AsyncMock(return_value="answer")
    answers = []

    controller.run(lambda _receive: answers.append(controller.interaction.prompt("Question?")))

    assert answers == ["answer"]
    ui.prompt_async.assert_awaited_once_with(
        "Question?",
        completer=None,
        choices=None,
        index=None,
        default=None,
        secret=False,
    )


def test_sync_controller_run_rejects_an_active_event_loop_without_claiming_controller(ui):
    """The synchronous entry point does not nest a loop or consume the lifecycle on failure."""
    controller = SessionController(ui)
    assert not inspect.iscoroutinefunction(controller.run)
    assert inspect.iscoroutinefunction(controller.run_async)

    async def invoke():
        """Attempt synchronous execution from an async owner."""
        with pytest.raises(RuntimeError, match="await SessionController.run_async"):
            controller.run(lambda _receive: None)

    asyncio.run(invoke())

    assert controller.running is False
    assert controller._used is False


def test_controller_restores_and_records_durable_input_delivery(ui):
    """A restored mailbox item reaches its owner with its stable session identity."""
    manager = SessionManager(session_store=MemorySessionStore())
    accepted = manager.queue_input("restored message", priority=1, bypass_commands=True)
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    received = []

    def conversation(receive) -> None:
        """Read one previously accepted durable input."""
        received.append(receive(DummyCompleter()))

    async def run_controller() -> None:
        """Run the controller until the restored owner operation settles."""
        read_inputs(ui, asyncio.Queue())
        await controller._run_sync_callback_async(conversation)

    asyncio.run(run_controller())

    assert received == [
        AdmittedConversationInput(
            input_id=accepted.input_id,
            session_id=manager.session.id,
            message="restored message",
            bypass_commands=True,
        )
    ]
    assert manager.session.pending_inputs[0].input_id == accepted.input_id
    assert manager.session.events[-1].type == "input_delivered"

    manager.add_user_message(received[0].message, input_id=received[0].input_id)
    assert manager.session.pending_inputs == ()


def test_controller_holds_queued_input_until_interrupted_run_is_recovered(ui):
    """Pending follow-ups remain queued until the existing recovery gate is cleared."""
    manager = SessionManager(session_store=MemorySessionStore())
    manager.add_user_message("interrupted request")
    manager.begin_run()
    accepted = manager.queue_input("follow-up", priority=1)
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    received = []

    def conversation(receive) -> None:
        """Wait for input while recovery is still required."""
        received.append(receive(DummyCompleter()))

    async def run_controller() -> None:
        """Stop without allowing an unrecovered message to reach the owner."""
        inputs = asyncio.Queue()
        inputs.put_nowait("new message")
        inputs.put_nowait(False)
        read_inputs(ui, inputs)
        await controller._run_sync_callback_async(conversation)

    asyncio.run(run_controller())

    assert received == [False]
    assert [event.input_id for event in manager.session.pending_inputs] == [accepted.input_id]
    assert not any(event.type == "input_delivered" for event in manager.session.events)
    ui.warning.assert_called_once_with(
        "Recover the interrupted run or start a new session before sending a message."
    )


def test_controller_persists_interaction_lifecycle_without_answer_values(ui):
    """The owner receives a prompt answer only after its safe lifecycle record settles."""
    store = MemorySessionStore()
    manager = SessionManager(session_store=store)
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    ui.confirm_async = AsyncMock(return_value=True)
    answers = []

    def conversation(_receive) -> None:
        """Ask the UI one confirmation through the owner mailbox."""
        answers.append(controller.interaction.confirm("Approve this request?", default=False))

    async def run_controller() -> None:
        """Settle the correlated prompt through the controller UI loop."""
        read_inputs(ui, asyncio.Queue())
        await controller._run_sync_callback_async(conversation)

    asyncio.run(run_controller())

    assert answers == [True]
    assert manager.session.pending_interactions == ()
    requested, settled = manager.session.events
    assert requested.type == "interaction_requested"
    assert settled.type == "interaction_settled"
    assert settled.outcome == "answered"
    assert "Approve this request?" not in manager.session.serialize()
    assert "answer" not in settled.model_dump(mode="json")


@pytest.mark.parametrize("answer", [False, True])
def test_idle_confirmation_keeps_session_ownership_until_settlement(ui, answer):
    """Live idle confirmations exclude recovery and session switching until their answer settles."""
    store = MemorySessionStore()
    manager = SessionManager(session_store=store)
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    answers = []

    async def confirm(*_args, **_kwargs):
        """Challenge recovery while the idle command is waiting for its actual answer."""
        snapshot = store.load(manager.session.id).serialize()
        with pytest.raises(SessionExecutionConflictError):
            SessionManager(session=manager.session.id, session_store=store)
        with pytest.raises(SessionExecutionConflictError):
            manager.new_session()
        assert store.load(manager.session.id).serialize() == snapshot
        return answer

    async def conversation(_receive):
        """Ask a blocking question from session-selection admission, as idle commands do."""
        with manager.execution(allow_session_selection=True):
            answers.append(await controller.interaction.confirm_async("Replace export?"))
            manager.new_session()

    ui.confirm_async.side_effect = confirm
    previous = manager.session
    read_inputs(ui, asyncio.Queue())
    asyncio.run(controller.run_async(conversation))

    assert answers == [answer]
    persisted = store.load(previous.id)
    assert persisted.pending_interactions == ()
    assert persisted.events[-1].outcome == "answered"
    assert manager.session is not previous


def test_controller_does_not_present_interaction_when_request_commit_fails(ui):
    """A failed lifecycle write blocks the prompt and returns the storage failure to its owner."""
    store = MemorySessionStore()
    manager = SessionManager(session_store=store)
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    ui.confirm_async = AsyncMock(return_value=True)

    def conversation(_receive) -> None:
        """Request a confirmation that must first be durably registered."""
        controller.interaction.confirm("Approve this request?", default=False)

    async def run_controller() -> None:
        """Attempt the interaction while the store rejects its lifecycle record."""
        read_inputs(ui, asyncio.Queue())
        with (
            patch.object(store, "save", side_effect=OSError("store unavailable")),
            pytest.raises(OSError, match="store unavailable"),
        ):
            await controller._run_sync_callback_async(conversation)

    asyncio.run(run_controller())

    ui.confirm_async.assert_not_awaited()
    assert manager.session.pending_interactions == ()


def test_controller_does_not_acknowledge_input_when_admission_commit_fails(ui):
    """The UI reports no queue admission when durable input persistence fails."""
    store = MemorySessionStore()
    manager = SessionManager(session_store=store)
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    received = []

    def conversation(receive) -> None:
        """Wait for the input whose admission cannot be persisted."""
        received.append(receive(DummyCompleter()))

    async def run_controller() -> None:
        """Fail the durable queue write and settle the owner."""
        inputs = asyncio.Queue()
        inputs.put_nowait("unpersisted input")
        read_inputs(ui, inputs)
        with (
            patch.object(store, "save", side_effect=OSError("store unavailable")),
            pytest.raises(OSError, match="store unavailable"),
        ):
            await controller._run_sync_callback_async(conversation)

    asyncio.run(run_controller())

    assert received == [False]
    assert manager.session.pending_inputs == ()
    assert manager.messages == []
    assert not any("Queued input" in str(call.args) for call in ui.info.call_args_list)


def test_durable_sync_queue_and_steer_persist_before_admission(ui):
    """Synchronous control APIs queue durable text while execution is active."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)

    with controller.control.execution():
        assert controller.queue("follow up", ui) is True
        assert controller.steer("correction", ui) is SteeringOutcome.INTERRUPTED

    assert {event.content for event in manager.session.pending_inputs} == {
        "follow up",
        "correction",
    }
    assert {item.input_id for item in controller._inbox.snapshot()} == {
        event.input_id for event in manager.session.pending_inputs
    }
    assert all(event.bypass_commands for event in manager.session.pending_inputs)


def test_sync_steer_interrupt_rejection_preserves_the_durable_input(ui):
    """An admitted correction remains durable if the active run declines interruption."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    controller.request_interrupt = Mock(return_value=False)

    with controller.control.execution():
        assert controller.steer("correction", ui) is SteeringOutcome.INTERRUPT_REJECTED

    assert [event.content for event in manager.session.pending_inputs] == ["correction"]
    assert controller._inbox.size() == 1


def test_sync_steer_without_durable_storage_preserves_legacy_outcomes(ui):
    """Standalone steering retains its bounded queue and interruption outcomes."""
    controller = SessionController(ui)
    controller.request_interrupt = Mock(return_value=True)

    with controller.control.execution():
        assert controller.steer("correction", ui) is SteeringOutcome.INTERRUPTED

    controller = SessionController(ui)
    for sequence in range(controller._inbox.capacity):
        controller._inbox.try_put(InputAdmission(1, sequence, InputAdmissionKind.MESSAGE, "queued"))
    with controller.control.execution():
        assert controller.steer("overflow", ui) is SteeringOutcome.QUEUE_FULL


def test_async_steer_and_queue_return_inactive_without_admitting_input(ui):
    """Async control APIs preserve their inactive outcomes without creating durable records."""
    controller = SessionController(ui)

    async def scenario():
        """Request both controls before an execution scope has started."""
        assert await controller.steer_async("correction", ui) is SteeringOutcome.RUN_INACTIVE
        assert await controller.queue_async("follow up", ui) is False

    asyncio.run(scenario())
    assert controller.status_snapshot().queued_inputs == 0


def test_async_steer_reports_interrupt_rejection_after_admission(ui):
    """An accepted async correction remains queued when interruption is declined."""
    controller = SessionController(ui)
    controller.request_interrupt = Mock(return_value=False)

    async def scenario():
        """Admit the correction while execution is active but decline interruption."""
        with controller.control.execution():
            return await controller.steer_async("correction", ui)

    assert asyncio.run(scenario()) is SteeringOutcome.INTERRUPT_REJECTED
    assert controller._inbox.size() == 1


def test_controller_accepts_durable_input_when_mailbox_has_capacity(ui):
    """Ordinary UI text enters durable pending state before its mailbox acknowledgment."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    controller.bind_command_manager(CommandManager())

    async def scenario():
        """Submit ordinary text through the controller admission path."""
        submitted = asyncio.get_running_loop().create_future()
        submitted.set_result("accepted")
        await controller._accept_async(submitted, route_idle_commands=False)

    asyncio.run(scenario())

    assert [event.content for event in manager.session.pending_inputs] == ["accepted"]
    assert controller._inbox.size() == 1


def test_controller_accepts_durable_input_during_its_live_run(ui):
    """A controller-owned durable run does not activate the abandoned-run recovery gate."""
    manager = SessionManager(session_store=MemorySessionStore())
    manager.add_user_message("active request")
    manager.begin_run()
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    controller.bind_command_manager(CommandManager())

    async def scenario():
        """Submit ordinary text while the controller owns an active run."""
        submitted = asyncio.get_running_loop().create_future()
        submitted.set_result("follow-up")
        with controller.control.execution():
            await controller._accept_async(submitted, route_idle_commands=False)

    asyncio.run(scenario())

    assert manager.recovery_state is not None
    assert manager.recovery_pending is True
    assert manager.continuable is True
    assert [event.content for event in manager.session.pending_inputs] == ["follow-up"]
    assert controller._inbox.size() == 1
    ui.warning.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "expected_priority", "steering_pending", "interrupted"),
    [
        ("queue", 1, False, False),
        ("steer", 0, True, False),
    ],
)
def test_controller_routes_active_plain_text_through_the_configured_mode(
    ui, mode, expected_priority, steering_pending, interrupted
):
    """Active plain text follows configured priority and interruption semantics."""
    manager = SessionManager(session_store=MemorySessionStore())
    manager.add_user_message("active request")
    manager.begin_run()
    controller = SessionController(ui, input_mode=mode)
    controller.bind_session_manager(manager)
    controller.bind_command_manager(CommandManager())

    async def scenario():
        """Submit ordinary text inside the active run's admission scope."""
        submitted = asyncio.get_running_loop().create_future()
        submitted.set_result("follow-up")
        with controller.control.execution():
            await controller._accept_async(submitted, route_idle_commands=False)
            assert controller.control.steering_pending is steering_pending
            assert controller.control.interrupted is interrupted

    asyncio.run(scenario())

    assert manager.session.pending_inputs[0].priority == expected_priority
    assert controller._inbox.size() == 1
    assert ui.warning.call_count == 0


@pytest.mark.parametrize("admitted", [False, True])
def test_controller_marks_in_memory_plain_text_steering_only_after_admission(ui, admitted):
    """In-memory steer mode marks steering only after its message enters the mailbox."""
    controller = SessionController(ui, input_mode="steer")
    controller.bind_command_manager(CommandManager())
    if not admitted:
        for sequence in range(controller._inbox.capacity):
            controller._inbox.try_put(
                InputAdmission(1, sequence, InputAdmissionKind.MESSAGE, "queued")
            )

    async def scenario():
        """Submit plain text to an active non-durable controller."""
        submitted = asyncio.get_running_loop().create_future()
        submitted.set_result("follow-up")
        with controller.control.execution():
            await controller._accept_async(submitted, route_idle_commands=False)
            assert controller.control.steering_pending is admitted
            assert controller.control.interrupted is False

    asyncio.run(scenario())
    assert controller._inbox.size() == (1 if admitted else controller._inbox.capacity)
    assert ui.warning.call_count == (0 if admitted else 1)


def test_async_steer_input_waiting_between_runs_is_queued_for_the_next_run(ui):
    """An explicit correction remains accepted when its controller has no active run."""
    controller = SessionController(ui)

    async def scenario():
        """Keep the controller lifecycle open without activating a model run."""
        controller._loop = asyncio.get_running_loop()
        controller._ui_thread = get_ident()
        result = await controller.steer_async("correction", ui)
        fallback = await controller._enqueue_steering_async("ordinary fallback")
        controller._loop = None
        controller._ui_thread = None
        return result, fallback

    result, fallback = asyncio.run(scenario())

    assert result is SteeringOutcome.QUEUED
    assert fallback is True
    assert [
        (item.priority, str(item.message)) for item in sorted(controller._inbox.snapshot())
    ] == [
        (0, "correction"),
        (1, "ordinary fallback"),
    ]


def test_async_steer_input_between_runs_reports_a_full_mailbox(ui):
    """An explicit correction between runs is rejected visibly when capacity is exhausted."""
    controller = SessionController(ui)
    for sequence in range(controller._inbox.capacity):
        controller._inbox.try_put(
            InputAdmission(1, sequence, InputAdmissionKind.MESSAGE, "waiting")
        )

    async def scenario():
        """Keep the controller alive between runs while the mailbox remains full."""
        controller._loop = asyncio.get_running_loop()
        controller._ui_thread = get_ident()
        result = await controller.steer_async("correction", ui)
        controller._loop = None
        controller._ui_thread = None
        return result

    assert asyncio.run(scenario()) is SteeringOutcome.QUEUE_FULL
    ui.warning.assert_called_once_with(
        "Input queue is full; this input was not accepted. Try again later."
    )


def test_enqueue_methods_reject_new_input_after_shutdown_starts(ui):
    """Shutdown closes both synchronous and async mailbox admission without a false warning."""
    controller = SessionController(ui)
    controller._stopping = True

    async def scenario():
        """Try each asynchronous admission path after its shutdown guard is set."""
        assert await controller._enqueue_async("ordinary") is False
        with controller.control.execution():
            assert await controller._enqueue_steering_async("steering") is False

    assert controller._enqueue("ordinary") is False
    with controller.control.execution():
        assert controller._enqueue_steering("steering") is False
    asyncio.run(scenario())
    ui.warning.assert_not_called()


def test_steering_admission_reports_full_capacity_and_wakes_the_ui_owner(ui):
    """Full steering queues report rejection while accepted steering wakes the UI loop."""
    controller = SessionController(ui)
    for sequence in range(controller._inbox.capacity):
        controller._inbox.try_put(
            InputAdmission(1, sequence, InputAdmissionKind.MESSAGE, "waiting")
        )
    with controller.control.execution():
        assert controller._enqueue_steering("overflow") is False
    ui.warning.assert_called_once_with(
        "Input queue is full; this input was not accepted. Try again later."
    )

    controller = SessionController(ui)

    async def admit_from_ui_thread():
        """Use the synchronous library path while the current event loop owns the UI."""
        controller._loop = asyncio.get_running_loop()
        controller._ui_thread = get_ident()
        with controller.control.execution():
            assert controller._enqueue_steering("accepted") is True
        assert controller._input_signal.is_set()
        controller._loop = None
        controller._ui_thread = None

    asyncio.run(admit_from_ui_thread())


def test_async_admission_reports_full_capacity_when_requested(ui):
    """Async ordinary and steering admission report their own bounded-queue rejection."""
    controller = SessionController(ui)
    for sequence in range(controller._inbox.capacity):
        controller._inbox.try_put(
            InputAdmission(1, sequence, InputAdmissionKind.MESSAGE, "waiting")
        )

    async def scenario():
        """Exercise the reporting async paths against a full mailbox."""
        assert await controller._enqueue_async("overflow") is False
        with controller.control.execution():
            assert await controller._enqueue_steering_async("overflow") is False

    asyncio.run(scenario())
    assert ui.warning.call_count == 2
    assert all(
        item.args[0] == "Input queue is full; this input was not accepted. Try again later."
        for item in ui.warning.call_args_list
    )


def test_controller_priority_commands_update_the_latest_waiting_message(ui):
    """Explicit priority commands durably reorder only messages still waiting in the mailbox."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    received = []

    async def scenario():
        """Run commands while a durable conversation input owner is active."""
        inputs = asyncio.Queue()
        release = asyncio.Event()
        working = asyncio.Event()
        queued_twice = asyncio.Event()
        steered = asyncio.Event()
        read_inputs(ui, inputs)
        queued_count = 0

        async def conversation(receive):
            """Hold a run while queuing, steering, then receive the priority order."""
            await receive(DummyCompleter())
            with controller.control.execution():
                controller.interaction.info("working")
                await release.wait()
            received.append((await receive(DummyCompleter())).message)
            received.append((await receive(DummyCompleter())).message)

        def report(message=""):
            """Expose command feedback as deterministic synchronization points."""
            nonlocal queued_count
            if message == "working":
                working.set()
            elif message.startswith("Queued input"):
                queued_count += 1
                if queued_count == 2:
                    queued_twice.set()
            elif message.startswith("Interrupt requested"):
                steered.set()

        ui.info.side_effect = report
        task = asyncio.create_task(controller.run_async(conversation))
        try:
            inputs.put_nowait("start")
            await working.wait()
            inputs.put_nowait("/queue first")
            inputs.put_nowait("/queue second")
            await queued_twice.wait()
            inputs.put_nowait("/steer")
            await steered.wait()
            release.set()
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert received == ["second", "first"]
    assert [
        (item.content, item.priority, item.sequence) for item in manager.session.pending_inputs
    ] == [
        ("second", 0, 2),
        ("start", 1, 0),
        ("first", 1, 1),
    ]
    assert [
        event.type for event in manager.session.events if event.type == "input_priority_changed"
    ] == ["input_priority_changed"]


def test_controller_priority_write_failure_keeps_the_mailbox_item_unchanged(ui):
    """A failed durable priority transition leaves its queued input at its previous priority."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    manager.change_input_priority_async = AsyncMock(side_effect=OSError("disk full"))

    async def scenario():
        """Queue input and fail its durable priority update before the mailbox mutation."""
        with controller.control.execution():
            assert await controller.queue_async("waiting", ui) is True
            with pytest.raises(OSError, match="disk full"):
                await controller.reprioritize_latest_async(steer=True)
            admission = await controller._get_input_async()
        return admission

    admission = asyncio.run(scenario())

    assert admission.priority == 1
    assert manager.session.pending_inputs[0].priority == 1
    assert [event.type for event in manager.session.events] == ["input_queued"]


def test_controller_reprioritization_skips_inputs_already_delivered(ui):
    """The physical queue cannot reprioritize an input after durable delivery begins."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)

    async def scenario():
        """Mark the accepted message delivered while leaving its stale envelope in the queue."""
        admission = await controller._durable_admission_async(
            "delivered",
            priority=1,
            bypass_commands=True,
        )
        assert controller._enqueue(admission.message, admission=admission) is True
        await manager.mark_input_delivered_async(admission.input_id)
        result = await controller.reprioritize_latest_async(steer=True)
        return admission, result

    admission, result = asyncio.run(scenario())

    assert result is PendingInputPriorityOutcome.NO_PENDING_INPUT
    assert controller._inbox.get_nowait() == admission
    assert manager.session.pending_inputs[0].priority == 1


def test_controller_reprioritization_reports_when_target_was_removed_during_write(ui):
    """A shutdown during durable priority persistence leaves the queue safely stopped."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)

    async def remove_during_write(input_id, priority):
        """Stop the mailbox, then finish the durable priority transition."""
        controller._stop()
        manager.change_input_priority(input_id, priority)

    manager.change_input_priority_async = remove_during_write

    async def scenario():
        """Select one queued message and remove it while the persistence await is pending."""
        admission = await controller._durable_admission_async(
            "waiting",
            priority=1,
            bypass_commands=True,
        )
        assert controller._enqueue(admission.message, admission=admission) is True
        return await controller.reprioritize_latest_async(steer=True)

    assert asyncio.run(scenario()) is PendingInputPriorityOutcome.NO_PENDING_INPUT
    assert manager.session.pending_inputs[0].priority == 0
    assert controller._inbox.get_nowait().kind is InputAdmissionKind.STOP


@pytest.mark.parametrize(
    ("steer", "initial_priority", "expected_priority"),
    [(False, 0, 1), (True, 1, 0)],
)
def test_controller_reprioritizes_memory_only_input_when_storage_is_unbound(
    ui, steer, initial_priority, expected_priority
):
    """Memory-only messages can move either way without a durable identity."""
    controller = SessionController(ui)
    controller._inbox.try_put(
        InputAdmission(initial_priority, 0, InputAdmissionKind.MESSAGE, "waiting")
    )
    controller.request_interrupt = Mock(return_value=False)

    async def scenario():
        """Change a memory-only priority and decline any optional interruption."""
        if steer:
            with controller.control.execution():
                return await controller.reprioritize_latest_async(steer=steer)
        return await controller.reprioritize_latest_async(steer=False)

    result = asyncio.run(scenario())

    assert result is PendingInputPriorityOutcome.UPDATED
    assert controller._inbox.get_nowait().priority == expected_priority


def test_controller_refuses_to_reprioritize_durable_identity_without_its_manager(ui):
    """A durable identity without its owning session cannot be priority updated."""
    controller = SessionController(ui)
    controller._inbox.try_put(
        InputAdmission(
            0,
            0,
            InputAdmissionKind.MESSAGE,
            "waiting",
            input_id="input-1",
        )
    )

    assert asyncio.run(controller.reprioritize_latest_async(steer=False)) is (
        PendingInputPriorityOutcome.NO_PENDING_INPUT
    )
    assert controller._inbox.get_nowait().priority == 0


@pytest.mark.parametrize("committed", [False, True])
def test_controller_priority_cancellation_keeps_durable_and_mailbox_state_aligned(ui, committed):
    """Cancellation leaves the mailbox unchanged or synchronizes a completed durable write."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)

    async def scenario():
        """Cancel reprioritization before or just after its durable transition commits."""
        with controller.control.execution():
            assert await controller.queue_async("waiting", ui) is True

            async def cancel_priority(input_id, priority):
                """Optionally commit the update before propagating task cancellation."""
                if committed:
                    manager.change_input_priority(input_id, priority)
                raise asyncio.CancelledError

            manager.change_input_priority_async = cancel_priority
            with pytest.raises(asyncio.CancelledError):
                await controller.reprioritize_latest_async(steer=True)
            return await controller._get_input_async()

    admission = asyncio.run(scenario())
    expected_priority = 0 if committed else 1

    assert admission.priority == expected_priority
    assert manager.session.pending_inputs[0].priority == expected_priority


def test_controller_recall_does_not_remove_input_when_durable_write_fails(ui):
    """A failed recall write leaves the accepted message in the mailbox and session."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    manager.recall_input_async = AsyncMock(side_effect=OSError("disk full"))

    async def scenario():
        """Queue one durable input and attempt to recall it during active ownership."""
        with controller.control.execution():
            assert await controller.queue_async("waiting", ui) is True
            with pytest.raises(OSError, match="disk full"):
                await controller.recall_latest_async()

    asyncio.run(scenario())

    assert controller.status_snapshot().queued_inputs == 1
    assert [item.content for item in manager.session.pending_inputs] == ["waiting"]


def test_controller_recall_reports_empty_and_supports_memory_only_inputs(ui):
    """Recall reports an empty mailbox and removes in-memory messages without storage."""
    controller = SessionController(ui)

    async def scenario():
        """Exercise the public recall boundary with no input and then one queued input."""
        assert await controller.recall_latest_async() is PendingInputRecallOutcome.NO_PENDING_INPUT
        with controller.control.execution():
            assert await controller.queue_async("waiting", ui) is True
            result = await controller.recall_latest_async()
        return result

    assert asyncio.run(scenario()) is PendingInputRecallOutcome.RECALLED
    assert controller.status_snapshot().queued_inputs == 0


@pytest.mark.parametrize("settlement", ["delivered", "consumed"])
def test_controller_recall_skips_delivered_or_consumed_inputs(ui, settlement):
    """Mailbox text already delivered or recorded in history cannot be recalled."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)

    async def scenario():
        """Close an accepted input at the requested durable boundary before recall."""
        with controller.control.execution():
            assert await controller.queue_async("waiting", ui) is True
            accepted = manager.session.pending_inputs[0]
            if settlement == "delivered":
                await manager.mark_input_delivered_async(accepted.input_id)
            else:
                await manager.add_user_message_async("waiting", input_id=accepted.input_id)
            return await controller.recall_latest_async()

    assert asyncio.run(scenario()) is PendingInputRecallOutcome.NO_PENDING_INPUT
    assert controller.status_snapshot().queued_inputs == 1
    assert not any(event.type == "input_recalled" for event in manager.session.events)


@pytest.mark.parametrize("committed", [False, True])
def test_controller_recall_cancellation_keeps_mailbox_aligned_with_durable_state(ui, committed):
    """Cancellation preserves an uncommitted input or removes a durably recalled one."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)

    async def scenario():
        """Cancel before or after the session has persisted the recalled outcome."""
        with controller.control.execution():
            assert await controller.queue_async("waiting", ui) is True
            accepted = manager.session.pending_inputs[0]

            async def cancel_recall(input_id):
                """Optionally commit the terminal event before propagating cancellation."""
                if committed:
                    manager.recall_input(input_id)
                raise asyncio.CancelledError

            manager.recall_input_async = cancel_recall
            with pytest.raises(asyncio.CancelledError):
                await controller.recall_latest_async()
            return accepted.input_id

    asyncio.run(scenario())

    if committed:
        assert controller.status_snapshot().queued_inputs == 0
        assert manager.session.pending_inputs == ()
    else:
        assert controller.status_snapshot().queued_inputs == 1
        assert [item.content for item in manager.session.pending_inputs] == ["waiting"]


@pytest.mark.parametrize(
    ("edit_command", "cancel_edit"),
    [("/edit", False), ("/edit", True), ("/recall", False), ("/recall", True)],
)
def test_edit_recalls_latest_message_and_submits_it_as_a_new_input(ui, edit_command, cancel_edit):
    """Edit restores the newest queued text, while cancellation keeps it recalled."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    commands = CommandManager(providers=(ConversationCommands(controller),), interaction=ui)
    controller.bind_command_manager(commands)
    received = []

    async def scenario():
        """Run the editor through queue admission, recall, and optional resubmission."""
        inputs = asyncio.Queue()
        working = asyncio.Event()
        queued_twice = asyncio.Event()
        draft_ready = asyncio.Event()
        release = asyncio.Event()
        queued_count = 0

        async def prompt(*_args, **kwargs):
            """Return submitted commands and expose recalled text as an editor default."""
            if "initial" in kwargs:
                assert kwargs["initial"] == "/exit"
                draft_ready.set()
                return False if cancel_edit else "/exit edited"
            return await inputs.get()

        def report(message=""):
            """Signal after both follow-ups are durably admitted."""
            nonlocal queued_count
            if message.startswith("Queued input"):
                queued_count += 1
                if queued_count == 2:
                    queued_twice.set()

        ui.prompt_async.side_effect = prompt
        ui.info.side_effect = report

        async def conversation(receive):
            """Receive the initial prompt and both remaining mailbox messages."""
            received.append(await receive(DummyCompleter()))
            with controller.control.execution():
                working.set()
                await release.wait()
            received.append(await receive(DummyCompleter()))
            if not cancel_edit:
                received.append(await receive(DummyCompleter()))

        task = asyncio.create_task(controller.run_async(conversation))
        try:
            inputs.put_nowait("start")
            await working.wait()
            inputs.put_nowait("/steer first")
            inputs.put_nowait("/queue /exit")
            await queued_twice.wait()
            inputs.put_nowait(edit_command)
            await draft_ready.wait()
            release.set()
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    queued = [event for event in manager.session.events if event.type == "input_queued"]
    recalled = [event for event in manager.session.events if event.type == "input_recalled"]
    assert [event.content for event in queued[:3]] == ["start", "first", "/exit"]
    assert (queued[1].priority, queued[2].priority) == (0, 1)
    assert len(recalled) == 1
    assert recalled[0].input_id == queued[2].input_id
    assert queued[2].input_id not in {event.input_id for event in manager.session.pending_inputs}
    if cancel_edit:
        assert len(queued) == 3
        assert received[1] is False
    else:
        assert [event.content for event in queued] == ["start", "first", "/exit", "/exit edited"]
        assert queued[-1].input_id != recalled[0].input_id
        assert queued[-1].sequence > queued[-2].sequence
        assert received[1].message == "first"
        assert received[2].message == "/exit edited"


@pytest.mark.parametrize("method", ["queue", "steer"])
def test_durable_sync_control_rejects_mailbox_overflow(ui, method):
    """Durable synchronous controls close accepted records when the mailbox is full."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    controller.bind_command_manager(CommandManager())
    for sequence in range(controller._inbox.capacity):
        controller._inbox.try_put(InputAdmission(1, sequence, InputAdmissionKind.MESSAGE, "queued"))

    with controller.control.execution():
        if method == "queue":
            result = controller.queue("overflow", ui)
            assert result is False
        else:
            result = controller.steer("overflow", ui)
            assert result is SteeringOutcome.QUEUE_FULL

    assert manager.session.pending_inputs == ()
    assert manager.session.events[-1].type == "input_rejected"
    ui.warning.assert_called_once_with(
        "Input queue is full; this input was not accepted. Try again later."
    )


def test_async_durable_control_admission_and_overflow(ui):
    """Async controls persist accepted input and reject mailbox overflow safely."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)

    async def scenario():
        """Exercise async queue and steer success plus both overflow paths."""
        with controller.control.execution():
            assert await controller.queue_async("follow up", ui) is True
            assert await controller.steer_async("correction", ui) is SteeringOutcome.INTERRUPTED
            for sequence in range(controller._inbox.capacity - controller._inbox.size()):
                controller._inbox.try_put(
                    InputAdmission(1, sequence, InputAdmissionKind.MESSAGE, "queued")
                )
            assert await controller.queue_async("queue overflow", ui) is False
            assert (
                await controller.steer_async("steer overflow", ui)
            ) is SteeringOutcome.QUEUE_FULL

    asyncio.run(scenario())

    assert {event.content for event in manager.session.pending_inputs} == {
        "follow up",
        "correction",
    }
    assert [event.type for event in manager.session.events[-4:]] == [
        "input_queued",
        "input_rejected",
        "input_queued",
        "input_rejected",
    ]
    assert ui.warning.call_count == 2


def test_controller_bind_session_manager_after_start_is_rejected(ui):
    """Session persistence must be attached before the controller lifecycle starts."""
    controller = SessionController(ui)
    asyncio.run(controller._run_sync_callback_async(lambda _receive: None))

    with pytest.raises(RuntimeError, match="Bind session persistence before running"):
        controller.bind_session_manager(SessionManager())


def test_controller_receives_durable_inputs_as_async_envelopes(ui):
    """Async owners receive delivery-confirmed identities for durable mailbox items."""
    manager = SessionManager(session_store=MemorySessionStore())
    event = manager.queue_input("async message", priority=1, bypass_commands=True)
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    controller._inbox.try_put(controller._admission_from_event(event))

    async def send(event):
        """Acknowledge the direct unit-test owner boundary."""
        event.result.set_result(None)

    controller._send_async = send

    result = asyncio.run(controller._receive_async(DummyCompleter()))

    assert result == AdmittedConversationInput(
        input_id=event.input_id,
        session_id=manager.session.id,
        message="async message",
        bypass_commands=True,
    )
    assert manager.session.events[-1].type == "input_delivered"


def test_sync_receive_rechecks_the_queue_after_an_empty_read_race(ui):
    """A message inserted between an empty read and condition wait is not lost."""
    controller = SessionController(ui)
    admission = InputAdmission(1, 0, InputAdmissionKind.MESSAGE, "arrived")
    original_get = controller._inbox.get_nowait
    inserted = False

    def get_with_racing_insert(**kwargs):
        """Insert input after the first empty read, before the receiver waits."""
        nonlocal inserted
        if not inserted:
            inserted = True
            controller._inbox.try_put(admission)
            raise Empty
        return original_get(**kwargs)

    controller._inbox.get_nowait = get_with_racing_insert
    controller._send = lambda event: event.result.set_result(None)

    with controller._gate:
        result = controller._receive(DummyCompleter())

    assert result == "arrived"


def test_sync_receive_returns_when_a_closed_mailbox_has_no_items(ui):
    """Finish input reception if shutdown was already consumed by another owner."""
    controller = SessionController(ui)
    stop = InputAdmission(1, 0, InputAdmissionKind.STOP)
    controller._inbox.stop_and_replace(stop)
    controller._inbox.get_nowait()
    controller._send = lambda event: event.result.set_result(None)

    with controller._gate:
        result = controller._receive(DummyCompleter())

    assert result is False


@pytest.mark.parametrize("asynchronous", [False, True])
def test_controller_skips_mailbox_items_from_another_session(ui, asynchronous):
    """A stale durable envelope is dropped before it reaches the conversation owner."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    controller._inbox.try_put(
        InputAdmission(
            0,
            0,
            InputAdmissionKind.MESSAGE,
            "stale",
            input_id="stale-input",
            session_id="different-session",
        )
    )
    controller._inbox.try_put(InputAdmission(1, 1, InputAdmissionKind.MESSAGE, "current"))

    def send(event):
        """Acknowledge the direct unit-test owner boundary."""
        event.result.set_result(None)

    async def send_async(event):
        """Acknowledge the direct async unit-test owner boundary."""
        event.result.set_result(None)

    controller._send = send
    controller._send_async = send_async

    if asynchronous:
        result = asyncio.run(controller._receive_async(DummyCompleter()))
    else:
        with controller._gate:
            result = controller._receive(DummyCompleter())

    assert result == "current"
    ui.warning.assert_called_once_with(
        "Queued input belongs to a different session and was skipped."
    )


def test_controller_durable_admission_reject_helpers_preserve_memory_only_inputs(ui):
    """In-memory admission records need no durable rejection writes."""
    controller = SessionController(ui)
    admission = InputAdmission(1, 0, InputAdmissionKind.MESSAGE, "ordinary")

    controller._reject_admission(admission)
    asyncio.run(controller._reject_admission_async(admission))

    assert controller._queued_input_ids == set()


def test_controller_restores_pending_inputs_up_to_mailbox_capacity(ui):
    """Recovery restores pending accepted inputs in order and respects bounded capacity."""
    manager = SessionManager(session_store=MemorySessionStore())
    accepted = [manager.queue_input(str(index), priority=1) for index in range(17)]
    controller = SessionController(ui)
    controller.bind_session_manager(manager)

    asyncio.run(controller._restore_pending_inputs_async())

    restored = [controller._inbox.get_nowait() for _ in range(controller._inbox.capacity)]
    assert [admission.input_id for admission in restored] == [
        event.input_id for event in accepted[: controller._inbox.capacity]
    ]
    assert len(controller._queued_input_ids) == controller._inbox.capacity


def test_controller_pending_input_restore_handles_queue_race_and_skip_states(ui, monkeypatch):
    """Restore tolerates a full mailbox race and skips stopped or absent persistence."""
    manager = SessionManager(session_store=MemorySessionStore())
    manager.queue_input("pending", priority=1)
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    for sequence in range(controller._inbox.capacity):
        controller._inbox.try_put(InputAdmission(1, sequence, InputAdmissionKind.MESSAGE, "full"))
    monkeypatch.setattr(controller._inbox, "size", lambda: 0)
    asyncio.run(controller._restore_pending_inputs_async())

    controller._session_manager = None
    asyncio.run(controller._restore_pending_inputs_async())
    controller._session_manager = manager
    controller._stopping = True
    asyncio.run(controller._restore_pending_inputs_async())


def test_controller_stops_restoring_when_shutdown_races_with_mailbox_recovery(ui, monkeypatch):
    """A shutdown during pending-input restoration leaves only the stop marker in memory."""
    manager = SessionManager(session_store=MemorySessionStore())
    manager.queue_input("pending", priority=1)
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    put = controller._put_mailbox_item_locked

    def stop_before_restore(admission):
        """Start shutdown just before the restore attempts to add its record."""
        controller._stop()
        return put(admission)

    monkeypatch.setattr(controller, "_put_mailbox_item_locked", stop_before_restore)

    asyncio.run(controller._restore_pending_inputs_async())

    assert controller._inbox.size() == 1
    assert controller._inbox.get_nowait().kind is InputAdmissionKind.STOP
    assert controller._queued_input_ids == set()


def test_controller_async_durable_rejection_discards_queued_identity(ui):
    """A rejected durable admission is removed from the controller's deduplication set."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    event = manager.queue_input("rejected", priority=1)
    admission = controller._admission_from_event(event)

    asyncio.run(controller._reject_admission_async(admission))

    assert event.input_id not in controller._queued_input_ids
    assert manager.session.pending_inputs == ()


def test_controller_durable_admission_without_session_manager_remains_in_memory(ui):
    """Standalone controllers preserve their legacy in-memory admission behavior."""
    controller = SessionController(ui)

    admission = asyncio.run(
        controller._durable_admission_async("correction", priority=0, bypass_commands=True)
    )

    assert isinstance(admission.message, SteeringMessage)
    assert admission.message == SteeringMessage("correction")
    assert admission.input_id is None


@pytest.mark.parametrize("active", [False, True])
def test_controller_accept_rejects_durable_input_when_mailbox_is_full(ui, active):
    """New durable input is closed when either ordinary or steering admission is full."""
    manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    controller.bind_command_manager(CommandManager())
    for sequence in range(controller._inbox.capacity):
        controller._inbox.try_put(InputAdmission(1, sequence, InputAdmissionKind.MESSAGE, "queued"))

    async def scenario():
        """Submit an ordinary message under the requested controller activity state."""
        submitted = asyncio.get_running_loop().create_future()
        submitted.set_result("overflow")
        if active:
            with controller.control.execution():
                await controller._accept_async(submitted, route_idle_commands=False)
        else:
            await controller._accept_async(submitted, route_idle_commands=False)

    asyncio.run(scenario())

    assert manager.session.pending_inputs == ()
    assert manager.session.events[-1].type == "input_rejected"
    ui.warning.assert_called_once_with(
        "Input queue is full; this input was not accepted. Try again later."
    )


def test_controller_reports_generic_mailbox_overflow(ui):
    """Direct bounded admission reports overflow only when requested by its caller."""
    controller = SessionController(ui)
    for sequence in range(controller._inbox.capacity):
        controller._inbox.try_put(InputAdmission(1, sequence, InputAdmissionKind.MESSAGE, "queued"))

    assert controller._enqueue("overflow", report_full=False) is False
    ui.warning.assert_not_called()
    assert controller._enqueue("overflow") is False
    ui.warning.assert_called_once_with(
        "Input queue is full; this input was not accepted. Try again later."
    )


def test_durable_interaction_request_cancellation_is_propagated(ui):
    """Cancellation during durable interaction registration reaches the async owner."""
    manager = SessionManager(session_store=MemorySessionStore())
    manager.record_interaction_requested_async = AsyncMock(side_effect=asyncio.CancelledError())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    request = InteractionCall(ui.prompt, ("Question?",), {}, kind=EVENT_PROMPT)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(controller._deliver_async(request))

    assert isinstance(request.result.exception(), asyncio.CancelledError)
    ui.prompt_async.assert_not_awaited()


def test_revoked_request_commit_failure_releases_interaction_ownership(ui):
    """A revoked registration failure preserves cancellation and releases its session lease."""
    manager = SessionManager()
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    owners, cancelled = [], []

    async def fail_registration(*_args, **_kwargs):
        """Cancel the owner request while its durable registration is failing."""
        owners[0].cancel()
        await asyncio.sleep(0)
        raise OSError("registration unavailable")

    async def conversation(_receive):
        """Handle cancellation of the waiting owner before registration completes."""
        owners.append(asyncio.current_task())
        try:
            await controller.interaction.prompt_async("Question?")
        except asyncio.CancelledError:
            cancelled.append(True)

    manager.record_interaction_requested_async = fail_registration
    read_inputs(ui, asyncio.Queue())
    asyncio.run(controller.run_async(conversation))

    assert cancelled == [True]
    ui.prompt_async.assert_not_awaited()
    manager.new_session()


@pytest.mark.parametrize("revocation", ["stop", "interrupt", "cancel"])
def test_successful_registration_rechecks_revocation_before_presenting_input(ui, revocation):
    """A committed request revoked during registration never opens or grants its prompt."""
    store = MemorySessionStore()
    manager = SessionManager(interaction=ui, session_store=store)
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    register = manager.record_interaction_requested_async
    owners, answers, revoked = [], [], []

    async def register_then_revoke(*args, **kwargs):
        """Commit the request before the controller observes cancellation at the await boundary."""
        await register(*args, **kwargs)
        if revocation == "stop":
            controller.stop()
        elif revocation == "interrupt":
            assert controller.request_interrupt()
        else:
            owners[0].cancel()
        await asyncio.sleep(0)

    async def conversation(_receive):
        """Catch revocation of an idle-command question without accepting an answer."""
        owners.append(asyncio.current_task())
        with (
            manager.execution(allow_session_selection=True),
            controller.control.execution(async_owner=True),
        ):
            try:
                answers.append(await controller.interaction.confirm_async("Overwrite?"))
            except (KeyboardInterrupt, RunInterrupted, asyncio.CancelledError):
                revoked.append(True)

    manager.record_interaction_requested_async = register_then_revoke
    ui.confirm_async = AsyncMock(return_value=True)
    read_inputs(ui, asyncio.Queue())
    asyncio.run(controller.run_async(conversation))

    assert revoked == [True] and answers == []
    ui.confirm_async.assert_not_awaited()
    assert manager.session.pending_interactions == ()
    assert manager.session.events[-1].outcome == "revoked"
    assert sum(event.type == "interaction_settled" for event in manager.session.events) == 1
    manager.new_session()


@pytest.mark.parametrize("late_value", [False, True])
def test_requester_cancellation_settles_its_open_input_without_an_answer(ui, late_value):
    """Cancelling an owner cancels its matching input and ignores any stale value it returns."""
    manager = SessionManager(interaction=ui, session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    ready, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    owners, answers = [], []

    async def confirm(*_args, **_kwargs):
        """Keep input open until cancellation, optionally returning a stale affirmative value."""
        ready.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            if not late_value:
                raise
        return True

    async def conversation(_receive):
        """End the waiting request without abandoning controller settlement."""
        owners.append(asyncio.current_task())
        with manager.execution(allow_session_selection=True):
            try:
                answers.append(await controller.interaction.confirm_async("Overwrite?"))
            except asyncio.CancelledError:
                pass

    async def scenario():
        """Cancel the owner and observe input cancellation before releasing the fallback gate."""
        task = asyncio.create_task(controller.run_async(conversation))
        try:
            await ready.wait()
            owners[0].cancel()
            await owners[0]
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert cancelled.is_set()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        await task

    ui.confirm_async.side_effect = confirm
    read_inputs(ui, asyncio.Queue())
    asyncio.run(scenario())

    assert answers == []
    assert manager.session.pending_interactions == ()
    assert manager.session.events[-1].outcome == "revoked"
    manager.new_session()


def test_requester_cancellation_ignores_an_input_value_already_returned(ui):
    """A completed input task cannot grant an answer to an owner that has just cancelled."""
    manager = SessionManager(interaction=ui)
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    owners, answers = [], []

    async def confirm(*_args, **_kwargs):
        """Cancel the owner immediately before the input coroutine returns its stale value."""
        owners[0].cancel()
        return True

    async def conversation(_receive):
        """Observe cancellation rather than consuming a returned input value."""
        owners.append(asyncio.current_task())
        try:
            answers.append(await controller.interaction.confirm_async("Overwrite?"))
        except asyncio.CancelledError:
            pass

    ui.confirm_async.side_effect = confirm
    read_inputs(ui, asyncio.Queue())
    asyncio.run(controller.run_async(conversation))

    assert answers == []
    assert manager.session.pending_interactions == ()
    assert manager.session.events[-1].outcome == "revoked"


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (KeyboardInterrupt(), "revoked"),
        (RunInterrupted(), "revoked"),
        (ValueError("stale response"), "failed"),
    ],
)
def test_controller_interaction_settlement_tracks_revocation_and_failure(ui, error, outcome):
    """Durable settlement reflects revoked owner requests and failed UI answers."""
    manager = SessionManager(session_store=MemorySessionStore())
    request_id = "settlement-request"
    asyncio.run(
        manager.record_interaction_requested_async(
            request_id,
            kind=EVENT_PROMPT,
            session_id=manager.session.id,
            run_id="run-id",
            operation_id="operation-id",
        )
    )
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    request = InteractionCall(
        ui.prompt,
        ("Question?",),
        {},
        kind=EVENT_PROMPT,
        request_id=request_id,
    )
    request.result.set_exception(error)
    controller._pending = request
    ui.prompt_async = AsyncMock(return_value="answer")

    asyncio.run(controller._answer_async())

    assert manager.session.pending_interactions == ()
    assert manager.session.events[-1].outcome == outcome


@pytest.mark.parametrize("prompt_fails", [False, True])
def test_controller_surfaces_interaction_settlement_storage_failure(ui, prompt_fails):
    """Settlement write failures propagate or annotate the original prompt error."""
    store = MemorySessionStore()
    manager = SessionManager(session_store=store)
    request_id = "settlement-failure"
    asyncio.run(
        manager.record_interaction_requested_async(
            request_id,
            kind=EVENT_PROMPT,
            session_id=manager.session.id,
            run_id=None,
            operation_id=None,
        )
    )
    controller = SessionController(ui)
    controller.bind_session_manager(manager)
    request = InteractionCall(
        ui.prompt,
        ("Question?",),
        {},
        kind=EVENT_PROMPT,
        request_id=request_id,
    )
    controller._pending = request
    if prompt_fails:
        ui.prompt_async = AsyncMock(side_effect=ValueError("prompt failed"))
        expected_error = "prompt failed"
    else:
        ui.prompt_async = AsyncMock(return_value="answer")
        expected_error = "store unavailable"

    async def scenario():
        """Fail the settlement snapshot after the request has already been recorded."""
        with (
            patch.object(store, "save", side_effect=OSError("store unavailable")),
            pytest.raises((ValueError, OSError), match=expected_error) as captured,
        ):
            await controller._answer_async()
        return captured.value

    failure = asyncio.run(scenario())

    if prompt_fails:
        assert any("settlement also failed with OSError" in note for note in failure.__notes__)
        assert isinstance(request.result.exception(), ValueError)
    else:
        assert isinstance(failure, OSError)
        assert request.result.done()
        assert request.result.exception() is failure
    assert manager.session.pending_interactions


def test_controller_commands_appear_in_help_and_complete_values_while_busy(ui):
    """Registered responsive controls and later argument values remain discoverable."""
    controller = SessionController(ui)
    manager = CommandManager(providers=(ConversationCommands(controller),), interaction=ui)
    controller.bind_command_manager(manager)

    def choose(context: CommandContext, color: str) -> None:
        """Choose a configured color."""

    def queued() -> None:
        """Run after the active operation settles."""

    def unavailable() -> None:
        """Remain unavailable while the active operation is running."""

    manager.register(
        CommandRegistration(
            choose,
            capabilities=frozenset({CommandCapability.RUN_ACTIVE}),
            completion=CommandCompletion(values=(CompletionValue("blue"),)),
        )
    )
    manager.register(queued)
    manager.register(unavailable)
    manager.call("help")
    setup_names = {item.name for item in ui.table.call_args.args[0]}
    assert {
        "interrupt",
        "pause",
        "stop",
        "respond",
        "status",
        "steer",
        "queue",
        "choose",
    }.isdisjoint(setup_names)
    assert "queued" in setup_names

    async def scenario():
        """Read a completion request while the conversation owns active work."""
        inputs = asyncio.Queue()
        working, captured = asyncio.Event(), asyncio.Event()
        release = Event()
        busy_completions = []
        busy_command_completions = []
        busy_help_names = []

        async def read(*_args, **kwargs):
            """Capture command completion before waiting for the next mailbox input."""
            completer = kwargs["completer"]
            completions = list(completer.get_completions(Document("/choose b"), CompleteEvent()))
            if working.is_set():
                busy_completions[:] = completions
                busy_command_completions[:] = list(
                    completer.get_completions(Document("/"), CompleteEvent())
                )
                captured.set()
            return await inputs.get()

        ui.prompt_async.side_effect = read

        def conversation(receive):
            """Hold execution ownership after accepting the initial message."""
            receive(DummyCompleter())
            with controller.control.execution():
                controller.interaction.info("working")
                release.wait()
            receive(DummyCompleter())

        ui.info.side_effect = lambda message="": working.set() if message == "working" else None
        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("start")
            await working.wait()
            await manager.handle_user_command_async("/help", interaction=ui)
            busy_help_names[:] = [item.name for item in ui.table.call_args.args[0]]
            inputs.put_nowait("/status")
            await captured.wait()
            assert [item.text for item in busy_completions] == ["blue"]
            commands = {item.text: item for item in busy_command_completions}
            assert "/choose" in commands
            assert "/respond" not in commands
            assert "/queued" not in commands
            assert "/unavailable" not in commands
            assert "/edit" in commands
            assert "/recall" in commands
            assert "/pause" in commands
            assert "/stop" in commands
            assert set(busy_help_names) == {
                "help",
                "exit",
                "quit",
                "interrupt",
                "pause",
                "stop",
                "status",
                "steer",
                "queue",
                "edit",
                "recall",
                "choose",
            }
            release.set()
            inputs.put_nowait(False)
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert "status" not in {item.name for item in manager.active_commands}


def test_controller_rejects_inactive_and_unknown_slash_input_without_queueing(ui):
    """Inactive commands and unknown slash input are explained instead of queued as messages."""
    controller = SessionController(ui)
    manager = CommandManager(providers=(ConversationCommands(controller),), interaction=ui)

    def serialized() -> None:
        """Represent a command owned by the conversation loop."""

    manager.register(serialized)
    controller.bind_command_manager(manager)
    received = []

    async def scenario():
        """Reject slash input during execution, then admit an ordinary follow-up."""
        inputs = asyncio.Queue()
        working = asyncio.Event()
        inactive_rejected = asyncio.Event()
        unknown_rejected = asyncio.Event()
        release = Event()
        read_inputs(ui, inputs)
        ui.info.side_effect = lambda message="": working.set() if message == "working" else None
        ui.warning.side_effect = lambda _message: inactive_rejected.set()
        ui.report.side_effect = lambda _problem: unknown_rejected.set()

        def conversation(receive):
            """Consume a normal input before and after an active run scope."""
            received.append(receive(DummyCompleter()))
            with controller.control.execution():
                controller.interaction.info("working")
                release.wait()
            received.append(receive(DummyCompleter()))

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("start")
            await working.wait()
            inputs.put_nowait("/serialized")
            await inactive_rejected.wait()
            inputs.put_nowait("/missing value")
            await unknown_rejected.wait()
            assert received == ["start"]
            release.set()
            inputs.put_nowait("follow-up")
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert received == ["start", "follow-up"]


def test_controller_routes_idle_commands_through_the_conversation_owner(ui):
    """Idle commands can prompt through the async UI without sync prompt-toolkit calls."""
    controller = SessionController(ui)
    selected = []

    def choose_scope(context: CommandContext) -> None:
        """Choose a scope through the command interaction."""
        selected.append(
            context.interaction.prompt(
                "Choose configuration scope:",
                choices={"user": "Save for current user", "session": "This session only"},
                index={"user": "u", "session": "s"},
            )
        )

    manager = CommandManager(
        commands=(choose_scope,),
        interaction=controller.interaction,
    )
    controller.bind_command_manager(manager)
    prompts = []
    received = []
    inputs = asyncio.Queue()

    async def prompt(*args, **kwargs):
        """Return deterministic editor input and a selected native prompt response."""
        if args == ("Choose configuration scope:",):
            prompts.append((args, kwargs))
            return "session"
        return await inputs.get()

    async def scenario():
        """Dispatch an idle command and open its choice prompt without another Enter."""
        ui.prompt_async.side_effect = prompt
        ui.prompt.side_effect = AssertionError("Synchronous UI input used during async runtime.")
        inputs.put_nowait("/choose_scope")
        inputs.put_nowait(False)

        def conversation(receive):
            """Dispatch commands on the conversation owner between idle reads."""
            while (value := receive(DummyCompleter())) is not False:
                received.append(value)
                manager.handle_user_command(value)

        task = asyncio.create_task(
            controller._run_sync_callback_async(conversation, route_idle_commands=True)
        )
        try:
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert received == ["/choose_scope"]
    assert selected == ["session"]
    assert len(prompts) == 1
    assert prompts[0][0] == ("Choose configuration scope:",)
    assert prompts[0][1]["choices"] == {
        "user": "Save for current user",
        "session": "This session only",
    }
    assert prompts[0][1]["index"] == {"user": "u", "session": "s"}
    ui.prompt.assert_not_called()


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("prompt", "session"), ("confirm", True)],
)
def test_active_commands_use_native_async_blocking_interactions(ui, kind, expected):
    """Run-active commands resolve prompts and confirmations through the native async UI."""
    controller = SessionController(ui)
    release = Event()
    active, question_open, completed = (asyncio.Event() for _ in range(3))
    answers = []

    def choose(context: CommandContext) -> None:
        """Request the configured blocking interaction from an eligible command."""
        if kind == "prompt":
            answer = context.interaction.prompt(
                "Choose a scope:", choices={"session": "This session only"}
            )
        else:
            answer = context.interaction.confirm("Continue?", default=False)
        answers.append(answer)
        context.interaction.info("command completed")

    manager = CommandManager(
        commands=(
            CommandRegistration(choose, capabilities=frozenset({CommandCapability.RUN_ACTIVE})),
        ),
        interaction=controller.interaction,
    )
    controller.bind_command_manager(manager)
    inputs = asyncio.Queue()

    async def prompt(*args, **_kwargs):
        """Answer the command question and wait for ordinary editor input otherwise."""
        if kind == "prompt" and args == ("Choose a scope:",):
            question_open.set()
            return expected
        return await inputs.get()

    async def confirm(*_args, **_kwargs):
        """Answer the command confirmation through the async confirmation API."""
        question_open.set()
        return expected

    def info(message=""):
        """Expose worker activity and command completion to the test event loop."""
        if message == "working":
            active.set()
        elif message == "command completed":
            completed.set()

    ui.prompt_async.side_effect = prompt
    ui.confirm_async.side_effect = confirm
    ui.info.side_effect = info

    def conversation(receive):
        """Hold an active execution while the command asks its blocking question."""
        receive(DummyCompleter())
        with controller.control.execution():
            controller.interaction.info("working")
            release.wait()
        receive(DummyCompleter())

    async def scenario():
        """Submit an active command and wait for its UI question to resolve."""
        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("start")
            await active.wait()
            inputs.put_nowait("/choose")
            await question_open.wait()
            await completed.wait()
            release.set()
            controller.stop()
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert answers == [expected]
    if kind == "prompt":
        ui.prompt_async.assert_any_await(
            "Choose a scope:",
            completer=None,
            choices={"session": "This session only"},
            index=None,
            default=None,
            secret=False,
        )
        ui.confirm_async.assert_not_awaited()
    else:
        ui.confirm_async.assert_awaited_once_with("Continue?", default=False)


def test_active_command_finishes_after_conversation_worker_and_keeps_prompt_open(ui):
    """Finish an active command's prompt after its conversation worker returns."""
    controller = SessionController(ui)
    command_started, prompt_returned, command_release = (Event() for _ in range(3))
    prompt_open = asyncio.Event()
    inputs = asyncio.Queue()
    answers = []

    def choose(context: CommandContext) -> None:
        """Wait for one choice and hold command completion until the test releases it."""
        command_started.set()
        answers.append(
            context.interaction.prompt("Choose a scope:", choices={"session": "Session"})
        )
        prompt_returned.set()
        command_release.wait()

    manager = CommandManager(
        commands=(
            CommandRegistration(choose, capabilities=frozenset({CommandCapability.RUN_ACTIVE})),
        ),
        interaction=controller.interaction,
    )
    controller.bind_command_manager(manager)

    async def prompt(*args, **_kwargs):
        """Return ordinary input or wait for the blocking scope response."""
        if args == ("Choose a scope:",):
            prompt_open.set()
        return await inputs.get()

    ui.prompt_async.side_effect = prompt

    def conversation(receive):
        """Return when active command dispatch proves its prompt has started."""
        receive(DummyCompleter())
        command_started.wait()

    async def scenario():
        """Resolve the prompt after conversation execution ends, then finish the command."""
        with controller.control.execution():
            task = asyncio.create_task(controller._run_sync_callback_async(conversation))
            try:
                inputs.put_nowait("start")
                inputs.put_nowait("/choose")
                await prompt_open.wait()
                inputs.put_nowait("session")
                await asyncio.to_thread(prompt_returned.wait)
                command_release.set()
                await task
            finally:
                command_release.set()
                controller.stop()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert answers == ["session"]


def test_stopping_releases_an_active_command_confirmation(ui):
    """Shutdown cancels an active command confirmation without granting approval."""
    session_manager = SessionManager(session_store=MemorySessionStore())
    controller = SessionController(ui)
    controller.bind_session_manager(session_manager)
    release = Event()
    active, question_open = asyncio.Event(), asyncio.Event()
    answers = []

    def approve(context: CommandContext) -> None:
        """Request approval through the controller mailbox."""
        answers.append(context.interaction.confirm("Continue?", default=False))

    manager = CommandManager(
        commands=(
            CommandRegistration(approve, capabilities=frozenset({CommandCapability.RUN_ACTIVE})),
        ),
        interaction=controller.interaction,
    )
    controller.bind_command_manager(manager)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)

    async def confirm(*_args, **_kwargs):
        """Keep confirmation open until controller shutdown cancels it."""
        question_open.set()
        await asyncio.Future()

    def info(message=""):
        """Signal when conversation execution has become active."""
        if message == "working":
            active.set()

    ui.confirm_async.side_effect = confirm
    ui.info.side_effect = info

    def conversation(receive):
        """Hold active execution while the confirmation command is entered."""
        receive(DummyCompleter())
        with controller.control.execution():
            controller.interaction.info("working")
            release.wait()

    async def scenario():
        """Stop after the command confirmation opens and wait for clean settlement."""
        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("start")
            await active.wait()
            inputs.put_nowait("/approve")
            await question_open.wait()
            release.set()
            controller.stop()
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert answers == []
    ui.confirm_async.assert_awaited_once_with("Continue?", default=False)
    assert session_manager.session.pending_interactions == ()
    assert session_manager.session.events[-1].type == "interaction_settled"
    assert session_manager.session.events[-1].outcome == "revoked"


def test_command_dispatch_preserves_unexpected_keyboard_interrupt(ui):
    """Propagate an unexpected interruption instead of treating it as shutdown control flow."""
    controller = SessionController(ui)

    def interrupt() -> None:
        """Raise the keyboard interruption being handled by command dispatch."""
        raise KeyboardInterrupt

    manager = CommandManager(
        commands=(interrupt,),
        interaction=ui,
    )
    with pytest.raises(KeyboardInterrupt):
        controller._run_command(manager, "/interrupt")


def test_sync_command_dispatch_suppresses_interrupt_after_shutdown(ui):
    """A synchronous command interruption is normal once controller shutdown is requested."""
    controller = SessionController(ui)
    controller.control.stop()
    manager = Mock()
    manager.handle_user_command.side_effect = KeyboardInterrupt

    assert controller._run_command(manager, "/interrupt") is True


def test_async_command_dispatch_normalizes_only_expected_interruptions(ui):
    """Async command dispatch preserves unexpected interrupts and consumes stopped work."""
    controller = SessionController(ui)
    interrupted = Mock()
    interrupted.handle_user_command_async = AsyncMock(side_effect=RunInterrupted())
    assert asyncio.run(controller._run_command_async(interrupted, "/interrupt")) is True

    unexpected = Mock()
    unexpected.handle_user_command_async = AsyncMock(side_effect=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(controller._run_command_async(unexpected, "/interrupt"))

    controller.control.stop()
    stopped = Mock()
    stopped.handle_user_command_async = AsyncMock(side_effect=KeyboardInterrupt())
    assert asyncio.run(controller._run_command_async(stopped, "/interrupt")) is True


@pytest.mark.parametrize("async_owner", [False, True])
def test_controller_dispatches_registered_async_commands_on_the_async_path(ui, async_owner):
    """Registered asynchronous commands run through async dispatch in either controller mode."""
    controller = SessionController(ui)
    called = asyncio.Event()

    async def asynchronous_command():
        """Record execution on the event loop that accepted the command."""
        called.set()

    commands = CommandManager(commands=(asynchronous_command,))
    controller.bind_command_manager(commands)
    controller._async_owner = async_owner

    async def scenario():
        """Submit one registered asynchronous callback through controller input handling."""
        submitted = asyncio.get_running_loop().create_future()
        submitted.set_result("/asynchronous_command")
        await controller._accept_async(submitted, route_idle_commands=False)
        await controller._command_task

    asyncio.run(scenario())
    assert called.is_set()


def test_interrupted_active_command_does_not_stop_the_controller(ui):
    """A run interruption releases a blocking command and leaves shutdown unset."""
    controller = SessionController(ui)
    release = Event()
    active, question_open = asyncio.Event(), asyncio.Event()

    def approve(context: CommandContext) -> None:
        """Request approval through the controller mailbox."""
        context.interaction.confirm("Continue?", default=False)

    manager = CommandManager(
        commands=(
            CommandRegistration(approve, capabilities=frozenset({CommandCapability.RUN_ACTIVE})),
        ),
        interaction=controller.interaction,
    )
    controller.bind_command_manager(manager)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)

    async def confirm(*_args, **_kwargs):
        """Wait until an active run interruption cancels this confirmation."""
        question_open.set()
        await asyncio.Future()

    ui.confirm_async.side_effect = confirm
    ui.info.side_effect = lambda message="": active.set() if message == "working" else None

    def conversation(receive):
        """Hold active execution until the confirmation request is interrupted."""
        receive(DummyCompleter())
        with controller.control.execution():
            controller.interaction.info("working")
            release.wait()

    async def scenario():
        """Interrupt the pending approval and allow both owners to settle normally."""
        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("start")
            await active.wait()
            inputs.put_nowait("/approve")
            await question_open.wait()
            assert controller.request_interrupt() is True
            release.set()
            await task
            assert controller.control.stopped is False
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert controller.running is False
    ui.confirm_async.assert_awaited_once_with("Continue?", default=False)


def test_failed_active_command_is_reported_during_controller_settlement(ui):
    """Preserve a UI failure and note a distinct active-command failure during settlement."""
    controller = SessionController(ui)
    release = Event()
    active, question_open = asyncio.Event(), asyncio.Event()

    def fail(context: CommandContext) -> None:
        """Replace an interaction failure with a command failure for cleanup reporting."""
        try:
            context.interaction.confirm("Continue?", default=False)
        except RuntimeError as error:
            raise ValueError("command failed") from error

    manager = CommandManager(
        commands=(
            CommandRegistration(fail, capabilities=frozenset({CommandCapability.RUN_ACTIVE})),
        ),
        interaction=controller.interaction,
    )
    controller.bind_command_manager(manager)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)

    async def confirm(*_args, **_kwargs):
        """Release the conversation worker before reporting a simulated UI failure."""
        question_open.set()
        release.set()
        raise RuntimeError("confirmation UI failed")

    ui.confirm_async.side_effect = confirm
    ui.info.side_effect = lambda message="": active.set() if message == "working" else None

    def conversation(receive):
        """Wait for the command interaction while the conversation remains active."""
        receive(DummyCompleter())
        with controller.control.execution():
            controller.interaction.info("working")
            release.wait()

    async def scenario():
        """Raise the UI failure after controller cleanup settles both owners."""
        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("start")
            await active.wait()
            inputs.put_nowait("/fail")
            await question_open.wait()
            with pytest.raises(RuntimeError, match="confirmation UI failed") as error:
                await task
            assert any(
                "Command also stopped with ValueError." in note for note in error.value.__notes__
            )
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_response_command_completions_follow_the_pending_request(ui):
    """Selectable response prompts expose their accepted values through the command catalog."""
    controller = SessionController(ui)
    request = InteractionCall(
        function=ui.prompt,
        args=("Choose one:",),
        kwargs={"choices": {"red": "Red", "blue": "Blue"}},
        kind=EVENT_PROMPT,
    )
    controller._pending = request

    assert [item.value for item in controller.response_completions()] == ["Red", "Blue"]
    assert controller.request_response("bLuE") is ResponseOutcome.ANSWERED
    assert request.result.result() == "blue"
    assert controller._pending is None


def test_command_capabilities_track_active_run_pending_response_and_settlement(ui):
    """Conversation controls appear only while the corresponding lifecycle state exists."""
    controller = SessionController(ui)
    manager = CommandManager(providers=(ConversationCommands(controller),), interaction=ui)
    controller.bind_command_manager(manager)

    def names():
        return {command.name for command in manager.active_commands}

    assert {
        "interrupt",
        "pause",
        "stop",
        "respond",
        "status",
        "steer",
        "queue",
        "edit",
        "recall",
    }.isdisjoint(names())
    controller._loop = Mock()
    idle_names = names()
    assert {"status", "steer", "queue", "edit", "recall"} <= idle_names
    assert {"interrupt", "pause", "stop"}.isdisjoint(idle_names)
    with controller.control.execution():
        active_names = names()
        assert {
            "interrupt",
            "pause",
            "stop",
            "status",
            "steer",
            "queue",
            "edit",
            "recall",
        } <= active_names
        assert "respond" not in active_names
        controller._pending = InteractionCall(
            function=ui.prompt,
            args=("Choose:",),
            kwargs={"choices": {"yes": "Yes", "no": "No"}},
            kind=EVENT_PROMPT,
        )
        assert "respond" in names()
        completer = CompletionManager((manager.completion_adapter,))
        completions = list(completer.get_completions(Document("/respond y"), CompleteEvent()))
        assert [item.text for item in completions] == ["Yes"]
    controller._pending = None
    controller._loop = None
    assert {
        "interrupt",
        "pause",
        "stop",
        "respond",
        "status",
        "steer",
        "queue",
        "edit",
        "recall",
    }.isdisjoint(names())


def test_recovery_command_visibility_follows_run_events_and_live_pause(ui):
    """Continue and retry visibility follows durable run events and cooperative pause state."""
    session_manager = SessionManager(session_store=MemorySessionStore())
    session_manager.add_user_message("unfinished")
    run = session_manager.begin_run()
    controller = SessionController(ui)
    controller.bind_session_manager(session_manager)

    class RecoveryControl:
        """Expose durable and live recovery state to the command registry."""

        @property
        def continuable(self):
            """Return whether a durable boundary or live pause can continue."""
            return controller.control.paused or session_manager.continuable

        @property
        def retry_pending(self):
            """Return whether a durable failure event authorizes retry."""
            return session_manager.retry_pending

        async def recover_pending_async(self, *, retry=False):
            """Resume a live pause or persist retry authorization."""
            if retry:
                await session_manager.begin_run_retry_async()
            if controller.control.continue_run():
                return False
            return False

    commands = CommandManager(
        providers=(
            ConversationCommands(controller),
            ConversationRecoveryCommands(RecoveryControl()),
        ),
        interaction=ui,
    )
    controller.bind_command_manager(commands)
    controller._loop = Mock()

    active_names = {command.name for command in commands.active_commands}
    assert "continue" in active_names
    assert "retry" not in active_names

    session_manager.record_run_failure()
    retry_names = {command.name for command in commands.active_commands}
    assert "retry" in retry_names
    assert "continue" not in retry_names
    completion = CompletionManager((commands.completion_adapter,))
    suggestions = {item.text for item in completion.get_completions(Document("/"), CompleteEvent())}
    assert "/retry" in suggestions
    assert "/continue" not in suggestions

    session_manager.begin_run_retry()
    with controller.control.execution():
        controller.control.pause()
        active_names = {command.name for command in commands.active_commands}
        assert {"continue", "pause", "stop"} <= active_names
        assert controller.status_snapshot().state == "Paused"

    assert "retry" not in {command.name for command in commands.active_commands}
    session_manager.record_run(
        "completed",
        run.started_at,
        RunMetrics(
            active_duration_seconds=0,
            model_duration_seconds=0,
            tool_duration_seconds=0,
            message_count=1,
            item_count=1,
        ),
    )
    assert {"continue", "retry"}.isdisjoint(command.name for command in commands.active_commands)
    assert controller.status_snapshot().state == "Idle"


def test_pause_and_continue_restore_the_same_conversation_input_owner(ui):
    """A paused run continues in place and restores the same input prompt afterward."""
    controller = SessionController(ui)
    session_manager = SessionManager(session_store=MemorySessionStore())
    controller.bind_session_manager(session_manager)
    run_started, pause_reported, resumed, run_completed, followup_received = (
        asyncio.Event() for _ in range(5)
    )

    class RecoveryControl:
        """Resume the controller's parked run through its shared state control."""

        @property
        def continuable(self):
            """Return whether the run is parked or has a durable recovery boundary."""
            return controller.control.paused or session_manager.continuable

        @property
        def retry_pending(self):
            """Return whether a failed event can be retried."""
            return session_manager.retry_pending

        async def recover_pending_async(self, *, retry=False):
            """Release the cooperative checkpoint after the continue command arrives."""
            return not controller.control.continue_run()

    manager = CommandManager(
        providers=(
            ConversationCommands(controller),
            ConversationRecoveryCommands(RecoveryControl()),
        ),
        interaction=ui,
    )
    controller.bind_command_manager(manager)
    inputs = asyncio.Queue()
    prompt_started = [asyncio.Event() for _ in range(4)]
    prompt_count = 0
    messages = []

    async def prompt(*_args, **_kwargs):
        """Read scripted input and expose each restored editor to the test."""
        nonlocal prompt_count
        prompt_count += 1
        if prompt_count <= len(prompt_started):
            prompt_started[prompt_count - 1].set()
        return await inputs.get()

    def info(message=""):
        """Track pause feedback and follow-up admission."""
        if message == "Run paused at its next safe checkpoint.":
            pause_reported.set()
        elif message == "Run continued.":
            resumed.set()

    ui.prompt_async.side_effect = prompt
    ui.info.side_effect = info

    async def conversation(receive):
        """Run one durable turn, pause it, continue it, then receive another input."""
        value = await receive(DummyCompleter())
        await session_manager.add_user_message_async(value.message, input_id=value.input_id)
        messages.append(value.message)
        run = await session_manager.begin_run_async()
        with controller.control.execution(async_owner=True):
            run_started.set()
            await controller.control.wait_async(3600)
        await session_manager.record_run_async(
            "completed",
            run.started_at,
            RunMetrics(
                active_duration_seconds=0,
                model_duration_seconds=0,
                tool_duration_seconds=0,
                message_count=1,
                item_count=1,
            ),
        )
        run_completed.set()
        value = await receive(DummyCompleter())
        if isinstance(value, AdmittedConversationInput):
            messages.append(value.message)
        followup_received.set()

    async def scenario():
        """Pause and continue a run, confirm prompt restoration, then exit the controller."""
        task = asyncio.create_task(controller.run_async(conversation, route_idle_commands=True))
        try:
            await prompt_started[0].wait()
            await inputs.put("start")
            await run_started.wait()
            await prompt_started[1].wait()
            await inputs.put("/pause")
            await pause_reported.wait()
            await prompt_started[2].wait()
            assert controller.status_snapshot().state == "Paused"
            assert "continue" in {command.name for command in manager.active_commands}
            await inputs.put("/continue")
            await resumed.wait()
            await run_completed.wait()
            await prompt_started[3].wait()
            await inputs.put("after continue")
            await followup_received.wait()
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert messages == ["start", "after continue"]
    assert prompt_count >= 4
    assert controller.control.cancelled is False


def test_idle_command_activity_waiter_is_cancelled_when_the_owner_finishes(ui):
    """Release the recovery activity waiter when an idle command ends its owner."""
    controller = SessionController(ui)
    waiter_started, waiter_cancelled = asyncio.Event(), asyncio.Event()

    async def wait_until_active():
        """Hold the activity waiter until controller shutdown settles it."""
        waiter_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            waiter_cancelled.set()
            raise

    controller.control.wait_until_active_async = wait_until_active
    ui.prompt_async.return_value = "/finish"

    def finish():
        """Provide a registered command that the owner can consume and then finish."""

    manager = CommandManager(commands=(finish,), interaction=ui)
    controller.bind_command_manager(manager)

    async def conversation(receive):
        """Return after receiving the idle command and observing the activity waiter."""
        assert await receive(DummyCompleter()) == "/finish"
        await waiter_started.wait()

    asyncio.run(controller.run_async(conversation, route_idle_commands=True))

    assert waiter_cancelled.is_set()


def test_idle_command_activity_waiter_completion_releases_input_routing(ui):
    """An active owner releases the gate for queued idle-command dispatch."""
    controller = SessionController(ui)
    controller._events = asyncio.Queue()
    controller._idle_command_pending = True
    controller.control.wait_until_active_async = AsyncMock(return_value=True)

    async def scenario():
        """Complete the activity waiter and worker in the same event-loop turn."""
        worker = asyncio.create_task(asyncio.sleep(0))
        await controller._drive_async(worker, route_idle_commands=True)

    asyncio.run(scenario())

    controller.control.wait_until_active_async.assert_awaited_once_with()
    assert controller._idle_command_pending is False


def test_command_capabilities_during_shutdown_allow_only_inspection(ui):
    """Shutdown keeps inspection available while refusing conversation controls and edits."""
    controller = SessionController(ui)
    controller._loop = Mock()
    controller._stopping = True

    assert controller.command_capabilities() == frozenset({CommandCapability.CONTEXT_READ})


def test_response_completion_caches_one_shot_prompt_choices(ui):
    """Completion does not consume one-shot choices before direct response validation."""
    controller = SessionController(ui)
    request = InteractionCall(
        ui.prompt,
        ("Choose one:",),
        {"choices": (value for value in ("red", "blue"))},
        kind=EVENT_PROMPT,
    )
    controller._pending = request

    assert [item.value for item in controller.response_completions()] == ["red", "blue"]
    assert controller.request_response("blue") is ResponseOutcome.ANSWERED
    assert request.result.result() == "blue"


@pytest.mark.parametrize(
    ("kind", "kwargs", "answer", "outcome"),
    [
        (EVENT_PROMPT, {}, "A free form answer", ResponseOutcome.ANSWERED),
        (EVENT_PROMPT, {"secret": True}, "secret", ResponseOutcome.MASKED_INPUT_REQUIRED),
        (EVENT_TERMINAL, {}, "answer", ResponseOutcome.PROMPT_REQUIRED),
        (EVENT_PROMPT, {"choices": ("first", "second")}, "3", ResponseOutcome.INVALID),
        (EVENT_PROMPT, {"choices": ("first", "second")}, "2", ResponseOutcome.ANSWERED),
        (
            EVENT_PROMPT,
            {"choices": ("first", "second"), "index": ("same", "same")},
            "first",
            ResponseOutcome.INVALID,
        ),
        (EVENT_CONFIRM, {}, "y", ResponseOutcome.ANSWERED),
        (EVENT_CONFIRM, {}, "maybe", ResponseOutcome.INVALID),
        ("unknown", {}, "answer", ResponseOutcome.PROMPT_REQUIRED),
    ],
)
def test_direct_response_resolves_supported_prompt_kinds(ui, kind, kwargs, answer, outcome):
    """Direct responses preserve free text, typed choices, and confirmation values."""
    controller = SessionController(ui)
    request = InteractionCall(function=ui.prompt, args=("Question",), kwargs=kwargs, kind=kind)
    controller._pending = request

    assert controller.request_response(answer) is outcome
    if outcome is ResponseOutcome.ANSWERED:
        assert request.result.done()
    else:
        assert not request.result.done()


@pytest.mark.parametrize(
    ("kind", "kwargs", "expected"),
    [
        (EVENT_CONFIRM, {}, False),
        (EVENT_CONFIRM, {"default": True}, True),
        (EVENT_PROMPT, {"default": "fallback"}, "fallback"),
    ],
)
def test_empty_direct_response_uses_the_prompt_default(ui, kind, kwargs, expected):
    """An empty direct answer applies a declared confirmation or prompt default."""
    controller = SessionController(ui)
    request = InteractionCall(function=ui.prompt, args=("Question",), kwargs=kwargs, kind=kind)
    controller._pending = request

    assert controller.request_response("") is ResponseOutcome.ANSWERED
    assert request.result.result() == expected


def test_empty_direct_response_without_a_default_is_rejected(ui):
    """An empty free-form answer stays pending when the prompt has no default."""
    controller = SessionController(ui)
    request = InteractionCall(function=ui.prompt, args=("Question",), kwargs={}, kind=EVENT_PROMPT)
    controller._pending = request

    assert controller.request_response("") is ResponseOutcome.INVALID
    assert controller._pending is request


@pytest.mark.parametrize(
    ("kind", "kwargs"),
    [(EVENT_PROMPT, {}), (EVENT_CONFIRM, {})],
)
def test_direct_response_does_not_bypass_an_active_native_prompt(ui, kind, kwargs):
    """Direct responses cannot resolve a request while its native UI owns the input."""
    controller = SessionController(ui)
    request = InteractionCall(function=ui.prompt, args=("Question",), kwargs=kwargs, kind=kind)
    controller._pending = request

    async def scenario():
        """Keep a live async input task while rejecting a competing direct answer."""
        answer_input = asyncio.get_running_loop().create_future()
        controller._answer_input = answer_input
        try:
            assert controller.request_response("answer") is ResponseOutcome.PROMPT_REQUIRED
            assert controller._pending is request
            assert not request.result.done()
        finally:
            answer_input.cancel()
            controller._answer_input = None

    asyncio.run(scenario())


def test_request_response_without_value_keeps_the_native_prompt_active(ui):
    """A bare response request leaves an unanswered question active for its native prompt."""
    controller = SessionController(ui)
    assert controller.request_response() is ResponseOutcome.NO_REQUEST
    request = InteractionCall(function=ui.prompt, args=("Question",), kwargs={}, kind=EVENT_PROMPT)
    controller._pending = request

    assert controller.request_response() is ResponseOutcome.PROMPTED
    assert controller._pending is request


def test_request_response_rejects_a_revoked_request(ui):
    """A completed stale request cannot receive another direct answer."""
    controller = SessionController(ui)
    request = InteractionCall(ui.prompt, ("Question",), {}, kind=EVENT_PROMPT)
    request.result.set_exception(KeyboardInterrupt())
    controller._pending = request

    assert controller.request_response("late answer") is ResponseOutcome.NO_REQUEST
    assert controller._pending is None


def test_request_response_does_not_clear_a_replacement_request(ui):
    """A concurrent replacement stays pending when an older request is already revoked."""
    controller = SessionController(ui)
    request = InteractionCall(ui.prompt, ("Old question",), {}, kind=EVENT_PROMPT)
    replacement = InteractionCall(ui.prompt, ("New question",), {}, kind=EVENT_PROMPT)
    controller._pending = request

    def done():
        """Simulate a new request becoming current before stale-state cleanup."""
        controller._pending = replacement
        return True

    request.result.done = done

    assert controller.request_response("late answer") is ResponseOutcome.NO_REQUEST
    assert controller._pending is replacement


def test_interrupt_preserves_a_response_that_won_the_race(ui):
    """An interrupt cannot overwrite a response future completed just before revocation."""
    controller = SessionController(ui)
    request = InteractionCall(ui.prompt, ("Question",), {}, kind=EVENT_PROMPT)
    request.result.set_result("answered")
    controller._pending = request

    with controller.control.execution():
        assert controller.request_interrupt() is True

    assert request.result.result() == "answered"
    assert controller._pending is None


def test_interrupt_does_not_clear_a_replacement_request(ui):
    """An interrupt leaves a newer request untouched when revocation races with delivery."""
    controller = SessionController(ui)
    request = InteractionCall(ui.prompt, ("Old question",), {}, kind=EVENT_PROMPT)
    replacement = InteractionCall(ui.prompt, ("New question",), {}, kind=EVENT_PROMPT)
    controller._pending = request

    def done():
        """Simulate delivery replacing the old request during revocation."""
        controller._pending = replacement
        return True

    request.result.done = done

    with controller.control.execution():
        assert controller.request_interrupt() is True

    assert controller._pending is replacement


def test_shutdown_preserves_a_response_that_won_the_race(ui):
    """Shutdown leaves an already completed request intact while clearing pending state."""
    controller = SessionController(ui)
    request = InteractionCall(ui.prompt, ("Question",), {}, kind=EVENT_PROMPT)
    request.result.set_result("answered")
    controller._pending = request

    controller._stop()

    assert request.result.result() == "answered"
    assert controller._pending is None


def test_shutdown_does_not_clear_a_replacement_request(ui):
    """Shutdown cleanup cannot erase a request installed after its stale snapshot."""
    controller = SessionController(ui)
    request = InteractionCall(ui.prompt, ("Old question",), {}, kind=EVENT_PROMPT)
    replacement = InteractionCall(ui.prompt, ("New question",), {}, kind=EVENT_PROMPT)
    controller._pending = request

    def done():
        """Simulate a replacement appearing while shutdown revokes the old future."""
        controller._pending = replacement
        return True

    request.result.done = done
    controller._stop()

    assert controller._pending is replacement


def test_deliver_ignores_an_interaction_revoked_before_ui_delivery(ui):
    """A revoked request waiting in the event queue is not presented to the user."""
    controller = SessionController(ui)
    request = InteractionCall(ui.prompt, ("Question",), {}, kind=EVENT_PROMPT)
    request.result.set_exception(KeyboardInterrupt())

    asyncio.run(controller._deliver_async(request))

    assert controller._pending is None
    ui.prompt_async.assert_not_awaited()


def test_late_prompt_value_cannot_complete_a_revoked_interaction(ui):
    """Shutdown ignores a UI adapter that returns a value after prompt cancellation."""
    controller = SessionController(ui)
    started = asyncio.Event()

    async def late_prompt(*_args, **_kwargs):
        """Return a stale value after cancellation to exercise the revoked request race."""
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            return "stale approval"

    ui.prompt_async.side_effect = late_prompt
    request = InteractionCall(ui.prompt, ("Approve?",), {}, kind=EVENT_PROMPT)
    controller._pending = request

    async def scenario():
        """Cancel the native prompt and let its adapter return a late value."""
        answering = asyncio.create_task(controller._answer_async())
        await started.wait()
        controller._stop()
        await answering

    asyncio.run(scenario())

    assert isinstance(request.result.exception(), KeyboardInterrupt)
    assert controller._pending is None


@pytest.mark.parametrize("late_error", [False, True])
def test_late_prompt_completion_preserves_a_replacement_answer_task(ui, late_error):
    """A late prompt value or error cannot clear a newer answer task."""
    controller = SessionController(ui)
    started = asyncio.Event()
    release = asyncio.Event()

    async def late_prompt(*_args, **_kwargs):
        """Wait through cancellation until the test releases a stale completion."""
        started.set()
        await release.wait()
        if late_error:
            raise ValueError("stale prompt failure")
        return "stale answer"

    ui.prompt_async.side_effect = late_prompt
    request = InteractionCall(ui.prompt, ("Question?",), {}, kind=EVENT_PROMPT)
    controller._pending = request

    async def scenario():
        """Revoke one request while another input task occupies the controller slot."""
        answering = asyncio.create_task(controller._answer_async())
        await started.wait()
        replacement_input = asyncio.get_running_loop().create_future()
        controller._answer_input = replacement_input
        controller._stop()
        release.set()
        await answering
        assert controller._answer_input is replacement_input

    asyncio.run(scenario())

    assert isinstance(request.result.exception(), KeyboardInterrupt)
    assert controller._pending is None


def test_response_completions_handle_confirm_secret_free_text_and_invalid_choices(ui):
    """Only valid, non-secret choices are completed for pending responses."""
    controller = SessionController(ui)
    assert controller.response_completions() == ()

    controller._pending = InteractionCall(ui.prompt, ("Confirm?",), {}, kind=EVENT_CONFIRM)
    assert [item.value for item in controller.response_completions()] == ["Yes", "No"]

    controller._pending = InteractionCall(
        ui.prompt, ("Secret?",), {"secret": True, "choices": ("a", "b")}, kind=EVENT_PROMPT
    )
    assert controller.response_completions() == ()

    controller._pending = InteractionCall(ui.prompt, ("Exclusive?",), {}, kind=EVENT_TERMINAL)
    assert controller.response_completions() == ()

    controller._pending = InteractionCall(ui.prompt, ("Text?",), {}, kind=EVENT_PROMPT)
    assert controller.response_completions() == ()

    controller._pending = InteractionCall(
        ui.prompt,
        ("Invalid choices?",),
        {"choices": ("a", "b"), "index": ("same", "same")},
        kind=EVENT_PROMPT,
    )
    assert controller.response_completions() == ()

    controller._pending = InteractionCall(ui.prompt, ("Unknown?",), {}, kind="unknown")
    assert controller.response_completions() == ()


def test_steering_reports_when_active_control_already_rejects_interruption(ui):
    """A stopped control refuses interruption even while its execution scope is unwinding."""
    controller = SessionController(ui)

    with controller.control.execution():
        controller.control.stop()
        outcome = controller.steer("correction", ui)

    assert outcome is SteeringOutcome.INTERRUPT_REJECTED


def test_steering_rejects_when_no_run_is_active(ui):
    """Steering cannot admit corrections while the run control is idle."""
    controller = SessionController(ui)

    assert controller.steer("correction", ui) is SteeringOutcome.RUN_INACTIVE
    assert controller.status_snapshot().queued_inputs == 0


def test_status_reports_cancellation_in_progress_and_after_settlement(ui):
    """Status distinguishes active cancellation cleanup from a settled cancelled run."""
    controller = SessionController(ui)

    with controller.control.execution():
        assert controller.control.interrupt() is True
        assert controller.status_snapshot().state == "Interrupting"

    assert controller.status_snapshot().state == "Cancelled"

    with controller.control.execution():
        controller.control.stop()
        assert controller.status_snapshot().state == "Cancelling"

    assert controller.status_snapshot().state == "Cancelled"


def test_queue_requires_a_run_and_preserves_command_like_text(ui):
    """Queued messages preserve text that resembles commands."""
    controller = SessionController(ui)

    assert controller.queue("/exit", ui) is False
    with controller.control.execution():
        assert controller.queue("/exit", ui) is True

    admission = controller._inbox.get_nowait()
    assert admission.priority == 1
    assert admission.kind is InputAdmissionKind.MESSAGE
    assert admission.message == SteeringMessage("/exit")
    assert isinstance(admission.message, SteeringMessage)


def test_default_steering_admission_falls_back_to_ordinary_input_while_idle(ui):
    """A steering admission racing with an idle boundary remains ordinary input."""
    controller = SessionController(ui)

    assert controller._enqueue_steering("message") is True

    admission = controller._inbox.get_nowait()
    assert admission.kind is InputAdmissionKind.MESSAGE
    assert admission.message == "message"
    assert not isinstance(admission.message, SteeringMessage)


def test_steering_admission_without_a_controller_loop_still_queues(ui):
    """An active execution can admit steering before controller startup."""
    controller = SessionController(ui)

    with controller.control.execution():
        assert controller._enqueue_steering("message") is True

    admission = controller._inbox.get_nowait()
    assert admission.kind is InputAdmissionKind.MESSAGE
    assert admission.message == SteeringMessage("message")


def test_steering_admission_from_a_non_ui_thread_wakes_the_controller_loop(ui):
    """Cross-thread steering admission schedules its mailbox wake on the owning loop."""
    controller = SessionController(ui)

    async def scenario():
        """Use a live loop with a distinct UI-thread marker to exercise thread-safe wakeup."""
        controller._loop = asyncio.get_running_loop()
        controller._ui_thread = -1
        with controller.control.execution():
            assert controller._enqueue_steering("message") is True
        await asyncio.sleep(0)
        return controller._input_signal.is_set()

    assert asyncio.run(scenario()) is True


def test_controller_accepts_messages_and_status_during_serialized_work(ui):
    """UI input remains live while execution retains one owner and processes queued inputs FIFO."""
    controller = SessionController(ui)
    release = Event()
    received, execution_threads, output_threads = [], [], []

    async def scenario():
        """Queue messages behind a controlled execution boundary."""
        inputs = asyncio.Queue()
        started, queued, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        read_inputs(ui, inputs)

        def output(message=""):
            """Observe presentation on the UI thread and signal execution boundaries."""
            output_threads.append(get_ident())
            if message == "working":
                started.set()
            elif message.startswith("Queued input"):
                queued.set()
            elif message == "processed":
                finished.set()

        ui.info.side_effect = output

        def conversation(receive):
            """Own mutable conversation state while blocking work is in progress."""
            while (message := receive(DummyCompleter())) is not False:
                received.append(message)
                execution_threads.append(get_ident())
                if message == "first":
                    controller.interaction.info("working")
                    release.wait()
                else:
                    controller.interaction.info("processed")
            controller.interaction.conversation_ended()

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("first")
            await started.wait()
            inputs.put_nowait("/status")
            inputs.put_nowait("second")
            await queued.wait()
            assert received == ["first"]
            assert controller.running is True
            assert "Working; 0 queued inputs." in [item.args[0] for item in ui.info.call_args_list]
            release.set()
            await finished.wait()
            inputs.put_nowait(False)
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)
        assert set(output_threads) == {get_ident()}

    asyncio.run(scenario())
    assert received == ["first", "second"]
    assert len(set(execution_threads)) == 1
    assert execution_threads[0] != get_ident()
    ui.conversation_ended.assert_called_once_with()


def test_sync_receive_discards_admitted_input_when_stop_arrives_at_boundary(ui):
    """A stop that races input consumption prevents the owner from starting more work."""
    controller = SessionController(ui)
    controller._inbox.try_put(InputAdmission(1, 0, InputAdmissionKind.MESSAGE, "late"))
    events = []

    def send(event):
        """Request stop after publishing the idle boundary but before reading input."""
        events.append(event)
        controller._control.stop()

    controller._send = send
    with controller._gate:
        assert controller._receive(DummyCompleter()) is False

    assert len(events) == 1


def test_async_receive_discards_admitted_input_when_stop_arrives_at_boundary(ui):
    """A stop that races async input consumption prevents the owner from starting more work."""
    controller = SessionController(ui)
    controller._inbox.try_put(InputAdmission(1, 0, InputAdmissionKind.MESSAGE, "late"))
    events = []

    async def send(event):
        """Request stop after publishing the idle boundary but before reading input."""
        events.append(event)
        controller._control.stop()

    controller._send_async = send

    assert asyncio.run(controller._receive_async(DummyCompleter())) is False
    assert len(events) == 1


def test_ordinary_input_during_active_run_is_deferred_steering(ui):
    """Active ordinary input finishes the current operation, then returns as a correction."""
    controller = SessionController(ui, input_mode="steer")
    release = Event()
    received = []

    async def scenario():
        """Submit a correction while an execution scope is active."""
        inputs = asyncio.Queue()
        working, accepted, consumed = (asyncio.Event() for _ in range(3))
        read_inputs(ui, inputs)
        ui.info.side_effect = lambda message="": (
            working.set()
            if message == "working"
            else accepted.set()
            if message.startswith("Steering accepted")
            else None
        )

        def conversation(receive):
            """Complete the current boundary and receive the correction next."""
            received.append(receive(DummyCompleter()))
            with controller.control.execution():
                controller.interaction.info("working")
                release.wait()
                assert controller.control.steering_pending is True
                assert controller.control.interrupted is False
                controller.control.checkpoint()
            received.append(receive(DummyCompleter()))
            consumed.set()
            receive(DummyCompleter())

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("first")
            await working.wait()
            inputs.put_nowait("correction")
            await accepted.wait()
            assert controller.status_snapshot().state == "Steering"
            release.set()
            await consumed.wait()
            inputs.put_nowait(False)
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert received == ["first", SteeringMessage("correction")]
    assert controller.control.steering_pending is False


@pytest.mark.parametrize(
    ("options", "answer", "expected"),
    [
        ({}, "a typed clarification", "a typed clarification"),
        ({"completer": WordCompleter(["custom answer"])}, "custom answer", "custom answer"),
        ({"exit_commands": ("quit",)}, "a typed clarification", "a typed clarification"),
        ({"choices": {"first": "First", "second": "Second"}}, "second", "second"),
    ],
)
def test_controller_uses_the_native_prompt_for_blocking_interactions(ui, options, answer, expected):
    """Blocking questions preserve prompt options for the source's native async UI."""
    controller = SessionController(ui)
    responses = []

    async def scenario():
        """Open a blocking question in the regular editor and submit its response."""
        inputs = asyncio.Queue()
        question_visible, answered = asyncio.Event(), asyncio.Event()

        async def prompt(*args, **kwargs):
            """Answer the blocking request while ordinary input stays in the message editor."""
            if args == ("Choose a target?",):
                question_visible.set()
                return answer
            return await inputs.get()

        ui.prompt_async.side_effect = prompt

        def info(message=""):
            """Observe completion of the blocking request."""
            if message == "answered":
                answered.set()

        ui.info.side_effect = info

        def conversation(receive):
            """Pause the owner on a prompt and then continue its input loop."""
            receive(DummyCompleter())
            responses.append(controller.interaction.prompt("Choose a target?", **options))
            controller.interaction.info("answered")
            receive(DummyCompleter())

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("start")
            await question_visible.wait()
            await answered.wait()
            inputs.put_nowait("/exit")
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert responses == [expected]
    question_calls = [call for call in ui.prompt_async.await_args_list if call.args]
    assert len(question_calls) == 1
    assert question_calls[0].args == ("Choose a target?",)
    assert question_calls[0].kwargs["choices"] == options.get("choices")
    assert question_calls[0].kwargs["completer"] is options.get("completer")
    if options.get("exit_commands"):
        assert question_calls[0].kwargs["exit_commands"] == ("quit",)
    ui.confirm_async.assert_not_awaited()
    ui.prompt.assert_not_called()


def test_controller_answers_blocking_prompt_when_no_editor_task_is_active(ui):
    """A startup question uses the source's native async prompt before chat input starts."""
    controller = SessionController(ui)
    ui.prompt_async.return_value = "answer"
    responses = []

    def conversation(_receive):
        """Request clarification before opening the ordinary message editor."""
        responses.append(controller.interaction.prompt("Clarify this value?"))

    asyncio.run(controller._run_sync_callback_async(conversation))

    assert responses == ["answer"]
    ui.prompt_async.assert_awaited_once()
    assert ui.prompt_async.await_args.args == ("Clarify this value?",)


def test_pending_confirmation_uses_the_native_confirmation_ui(ui):
    """A confirmation delegates unchanged to the source's native confirmation interaction."""
    controller = SessionController(ui)
    ui.confirm_async.return_value = False
    decisions = []

    def conversation(_receive):
        """Ask for approval before opening the ordinary message editor."""
        decisions.append(controller.interaction.confirm("Continue?", default=False))

    asyncio.run(controller._run_sync_callback_async(conversation))

    assert decisions == [False]
    ui.confirm_async.assert_awaited_once_with("Continue?", default=False)
    ui.prompt_async.assert_not_awaited()


def test_pending_question_preserves_choices_for_the_native_prompt(ui):
    """Blocking choice prompts pass their choices to the source UI without rewriting them."""
    controller = SessionController(ui)
    responses = []

    async def scenario():
        """Resolve a selected answer through the source's regular choice prompt."""
        inputs = asyncio.Queue()
        question_open, answered = asyncio.Event(), asyncio.Event()

        async def prompt(*args, **kwargs):
            """Return the selected choice key through the native prompt interface."""
            if args == ("Clarify the target?",):
                question_open.set()
                return "second"
            return await inputs.get()

        ui.prompt_async.side_effect = prompt

        def info(message=""):
            """Observe the completed interaction."""
            if message == "answered":
                answered.set()

        ui.info.side_effect = info

        def conversation(receive):
            """Ask for a selected response and then return to ordinary conversation input."""
            receive(DummyCompleter())
            responses.append(
                controller.interaction.prompt(
                    "Clarify the target?", choices={"first": "First", "second": "Second"}
                )
            )
            controller.interaction.info("answered")
            receive(DummyCompleter())

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("start")
            await question_open.wait()
            await answered.wait()
            inputs.put_nowait("/exit")
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert responses == ["second"]
    question_call = next(call for call in ui.prompt_async.await_args_list if call.args)
    assert question_call.args == ("Clarify the target?",)
    assert question_call.kwargs["choices"] == {"first": "First", "second": "Second"}
    assert question_call.kwargs["completer"] is None


def test_declining_a_pending_question_keeps_the_conversation_available(ui):
    """Declining a native confirmation does not close the regular conversation editor."""
    controller = SessionController(ui)
    answers, messages = [], []

    async def scenario():
        """Decline the active request and then submit another ordinary message."""
        inputs = asyncio.Queue()
        question_open, continued, completed = (asyncio.Event() for _ in range(3))

        async def confirm(*_args, **_kwargs):
            """Return a negative decision through the native confirmation UI."""
            question_open.set()
            return False

        read_inputs(ui, inputs)
        ui.confirm_async.side_effect = confirm
        ui.info.side_effect = lambda message="": (
            continued.set()
            if message == "continued"
            else completed.set()
            if message == "done"
            else None
        )

        def conversation(receive):
            """Continue reading after receiving a negative result from the interaction."""
            receive(DummyCompleter())
            answers.append(controller.interaction.confirm("Approve?"))
            controller.interaction.info("continued")
            messages.append(receive(DummyCompleter()))
            controller.interaction.info("done")
            receive(DummyCompleter())

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("start")
            await question_open.wait()
            await continued.wait()
            assert controller.control.stopped is False
            inputs.put_nowait("later")
            await completed.wait()
            inputs.put_nowait("/exit")
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert answers == [False]
    assert messages == ["later"]


def test_stop_discards_input_completed_at_the_shutdown_boundary(ui):
    """Input that finishes as shutdown begins is not admitted to conversation execution."""
    controller = SessionController(ui)
    messages = []

    async def prompt(*_args, **_kwargs):
        """Request shutdown immediately before returning one late editor value."""
        controller.stop()
        return "late input"

    ui.prompt_async.side_effect = prompt

    def conversation(receive):
        """Wait for input and record the shutdown signal instead of late text."""
        messages.append(receive(DummyCompleter()))

    asyncio.run(controller._run_sync_callback_async(conversation))

    assert messages == [False]


def test_interrupt_cancels_a_modal_prompt_and_restores_the_editor_draft(ui):
    """Interruption closes masked prompts without losing the suspended chat draft."""
    controller = SessionController(ui)
    messages = []

    async def scenario():
        """Interrupt a masked request and continue through the restored main editor."""
        inputs = asyncio.Queue()
        prompt_open, interrupted, processed = (asyncio.Event() for _ in range(3))
        ui.current_input.return_value = "unfinished draft"

        async def prompt(*_args, **kwargs):
            """Block on the masked prompt while normal messages use the input queue."""
            if kwargs.get("secret"):
                prompt_open.set()
                await asyncio.Future()
            return await inputs.get()

        ui.prompt_async.side_effect = prompt
        ui.info.side_effect = lambda message="": (
            interrupted.set()
            if message == "interrupted"
            else processed.set()
            if message == "processed"
            else None
        )

        def conversation(receive):
            """Catch an interrupted secret request and accept a later chat message."""
            receive(DummyCompleter())
            with controller.control.execution(), pytest.raises(RunInterrupted):
                controller.interaction.prompt("Credential:", secret=True, exit_commands=())
            controller.interaction.info("interrupted")
            messages.append(receive(DummyCompleter()))
            controller.interaction.info("processed")
            receive(DummyCompleter())

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("start")
            await prompt_open.wait()
            assert controller.request_interrupt() is True
            await interrupted.wait()
            inputs.put_nowait("later")
            await processed.wait()
            assert ui.set_current_input.call_args.args == ("",)
            assert any(
                call.kwargs.get("initial") == "unfinished draft"
                for call in ui.prompt_async.await_args_list
            )
            inputs.put_nowait("/exit")
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert messages == ["later"]


def test_submitted_chat_remains_queued_when_a_modal_question_arrives(ui):
    """Input completed before a blocking question is delivered stays ordinary conversation text."""
    controller = SessionController(ui)
    request = InteractionCall(
        function=ui.prompt,
        args=("Credential:",),
        kwargs={"secret": True},
        kind=EVENT_PROMPT,
    )
    controller._ready = True
    controller._completer = DummyCompleter()

    async def scenario():
        """Complete normal input and modal interaction together through the controller loop."""
        controller._loop = asyncio.get_running_loop()
        controller._events = asyncio.Queue()
        controller._events.put_nowait(request)
        inputs = []

        async def prompt(*_args, **_kwargs):
            """Return queued chat first and a masked response to the blocking request second."""
            inputs.append("ordinary chat" if not inputs else "private response")
            return inputs[-1]

        ui.prompt_async.side_effect = prompt
        controller.bind_command_manager(
            CommandManager(providers=(ConversationCommands(controller),), interaction=ui)
        )
        worker = asyncio.ensure_future(asyncio.wrap_future(request.result))

        await controller._drive_async(worker, route_idle_commands=False)

    asyncio.run(scenario())
    assert controller._inbox.get_nowait().message == "ordinary chat"
    assert controller._pending is None
    assert request.result.result() == "private response"


def test_cancelled_editor_task_does_not_block_a_modal_question(ui):
    """A cancelled editor task is ignored while its blocking interaction is answered."""
    controller = SessionController(ui)
    request = InteractionCall(
        function=ui.prompt,
        args=("Credential:",),
        kwargs={"secret": True},
        kind=EVENT_PROMPT,
    )
    controller._ready = True
    controller._completer = DummyCompleter()
    prompt_calls = 0

    async def prompt(*_args, **_kwargs):
        """Cancel ordinary editor input, then answer the masked interaction."""
        nonlocal prompt_calls
        prompt_calls += 1
        if prompt_calls == 1:
            raise asyncio.CancelledError
        return "private response"

    async def scenario():
        """Deliver the modal request alongside an editor task that cancels immediately."""
        controller._loop = asyncio.get_running_loop()
        controller._events = asyncio.Queue()
        controller._events.put_nowait(request)
        ui.prompt_async.side_effect = prompt
        controller.bind_command_manager(
            CommandManager(providers=(ConversationCommands(controller),), interaction=ui)
        )
        worker = asyncio.ensure_future(asyncio.wrap_future(request.result))

        await controller._drive_async(worker, route_idle_commands=False)

    asyncio.run(scenario())
    assert request.result.result() == "private response"
    assert prompt_calls == 3
    assert [call.kwargs.get("secret", False) for call in ui.prompt_async.await_args_list] == [
        False,
        True,
        False,
    ]


def test_controller_preserves_startup_prompts_and_masked_input(ui):
    """Prompts before the first ordinary input boundary retain their exact typed results."""
    controller = SessionController(ui)
    ui.prompt_async.return_value = "private-value"
    values = []

    def conversation(_receive):
        """Read a startup secret through the UI without ordinary mailbox input."""
        values.append(controller.interaction.prompt("Credential:", secret=True, exit_commands=()))

    asyncio.run(controller._run_sync_callback_async(conversation))

    assert values == ["private-value"]
    ui.prompt_async.assert_awaited_once_with(
        "Credential:",
        completer=None,
        choices=None,
        index=None,
        default=None,
        secret=True,
        exit_commands=(),
    )
    ui.prompt.assert_not_called()


def test_controller_stops_before_startup_without_opening_input(ui):
    """A pre-start stop request prevents admission and leaves interaction usable afterward."""
    controller = SessionController(ui)
    controller.stop()
    values = []
    asyncio.run(
        controller._run_sync_callback_async(
            lambda receive: values.append(receive(DummyCompleter()))
        )
    )

    assert values == [False]
    assert controller.control.stopped is True
    assert controller.running is False
    ui.prompt_async.assert_not_awaited()
    ui.conversation_ended.assert_called_once_with()


def test_controller_bounds_queued_inputs_and_reports_unexecuted_inputs_on_exit(ui):
    """Full admission explicitly rejects input and shutdown does not execute queued work."""
    controller = SessionController(ui)
    release = Event()
    messages = []

    async def scenario():
        """Fill the bounded queue while execution waits at an owned operation."""
        inputs = asyncio.Queue()
        started, rejected, stopping = asyncio.Event(), asyncio.Event(), asyncio.Event()
        read_inputs(ui, inputs)
        ui.info.side_effect = lambda message="": (
            started.set()
            if message == "working"
            else stopping.set()
            if message.startswith("Stopping")
            else None
        )
        ui.warning.side_effect = lambda message: (
            rejected.set() if "queue is full" in message else None
        )

        def conversation(receive):
            """Keep the execution owner busy until the test releases it."""
            messages.append(receive(DummyCompleter()))
            with controller.control.execution():
                controller.interaction.info("working")
                release.wait()
                messages.append(receive(DummyCompleter()))

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("first")
            await started.wait()
            for index in range(17):
                inputs.put_nowait(f"queued {index}")
            await rejected.wait()
            inputs.put_nowait("/exit")
            await stopping.wait()
            assert not task.done()
            release.set()
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert messages == ["first", False]
    assert any(
        "16 queued inputs were not executed" in item.args[0] for item in ui.warning.call_args_list
    )


def test_controller_revokes_pending_approval_on_shutdown(ui):
    """Shutdown unblocks a waiting execution owner without granting pending approval."""
    controller = SessionController(ui)
    decisions = []

    async def scenario():
        """Stop after the native confirmation prompt opens but before the user answers."""
        inputs = asyncio.Queue()
        prompted = asyncio.Event()
        read_inputs(ui, inputs)

        async def confirm(*_args, **_kwargs):
            """Signal that approval input opened, then wait for its cancellation."""
            prompted.set()
            return await inputs.get()

        ui.confirm_async.side_effect = confirm

        def conversation(receive):
            """Request a decision from the independent approval channel."""
            receive(DummyCompleter())
            decisions.append(controller.interaction.confirm("Approve?", default=True))

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("start")
            await prompted.wait()
            controller.stop()
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert decisions == []
    ui.confirm_async.assert_awaited_once_with("Approve?", default=True)


def test_controller_keeps_live_completion_reads_out_of_active_execution(ui):
    """Live completion works when idle and does not inspect services during active mutation."""
    controller = SessionController(ui)
    release = Event()
    completions = []

    async def scenario():
        """Inspect the supplied completion capability at idle and busy input boundaries."""
        inputs = asyncio.Queue()
        started, busy_read = asyncio.Event(), asyncio.Event()
        ui.info.side_effect = lambda text="": started.set() if text == "working" else None

        async def read(*_args, **kwargs):
            """Use the UI's injected completer without accessing controller state."""
            if started.is_set():
                completions.append(
                    list(kwargs["completer"].get_completions(Document("he"), CompleteEvent()))
                )
                busy_read.set()
            else:
                completions.append(
                    list(kwargs["completer"].get_completions(Document("he"), CompleteEvent()))
                )
            return await inputs.get()

        ui.prompt_async.side_effect = read

        def conversation(receive):
            """Hold the owner gate around a synchronous operation."""
            receive(WordCompleter(["hello"]))
            controller.interaction.info("working")
            release.wait()

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("start")
            await started.wait()
            # An already-open editor can submit status to start a fresh completion request.
            inputs.put_nowait("/status")
            await busy_read.wait()
            release.set()
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert [item.text for item in completions[0]] == ["hello"]
    assert completions[-1] == []


def test_controller_propagates_worker_errors_and_restores_direct_interaction(ui):
    """Execution errors settle their owner before the controller restores setup interaction."""
    controller = SessionController(ui)

    def conversation(_receive):
        """Fail at the execution boundary without leaving background operations."""
        raise ValueError("execution failed")

    with pytest.raises(ValueError, match="execution failed"):
        asyncio.run(controller._run_sync_callback_async(conversation))

    assert controller.running is False
    controller.interaction.info("after failure")
    ui.info.assert_called_with("after failure")


def test_controller_cancellation_waits_for_execution_cleanup(ui):
    """Task cancellation drains UI cleanup and retains ownership until the worker settles."""
    controller = SessionController(ui)
    release = Event()
    cleanup_threads = []

    async def scenario():
        """Cancel during active work and then release its deterministic cleanup boundary."""
        started, cleaned = asyncio.Event(), asyncio.Event()
        ui.info.side_effect = lambda text="": (
            started.set() if text == "working" else cleaned.set() if text == "cleaned" else None
        )

        def conversation(_receive):
            """Finalize presentation after an in-progress synchronous operation settles."""
            try:
                controller.interaction.info("working")
                release.wait()
            finally:
                cleanup_threads.append(get_ident())
                controller.interaction.info("cleaned")

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            await started.wait()
            task.cancel()
            # Scheduling an ordinary UI callback remains possible while shutdown is pending.
            await asyncio.get_running_loop().run_in_executor(None, release.set)
            await cleaned.wait()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert len(cleanup_threads) == 1
    assert controller.running is False


def test_controller_hands_off_exclusive_terminal_without_starting_another_editor(ui):
    """Direct authentication owns stdin until its scope and nested approvals have settled."""
    controller = SessionController(ui)
    release = Event()
    decisions = []

    async def scenario():
        """Explicitly hand off the terminal before simulating direct authentication."""
        inputs = asyncio.Queue()
        held, finished = asyncio.Event(), asyncio.Event()
        read_inputs(ui, inputs)

        def output(message=""):
            """Observe terminal ownership without sharing domain objects."""
            if message == "held":
                held.set()
            elif message == "released":
                finished.set()

        ui.info.side_effect = output
        ui.confirm_async.return_value = False

        def conversation(receive):
            """Reserve stdin around an operation that uses direct authentication."""
            receive(DummyCompleter())
            with controller.interaction.terminal_context():
                decisions.append(controller.interaction.confirm("Install?"))
                controller.interaction.info("held")
                release.wait()
            controller.interaction.info("released")
            receive(DummyCompleter())

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("manage")
            await held.wait()
            count = ui.prompt_async.await_count
            assert count == 2
            ui.confirm_async.assert_awaited_once_with("Install?", default=False)
            release.set()
            await finished.wait()
            inputs.put_nowait(False)
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert decisions == [False]


def test_controller_rejects_blocking_mailbox_calls_from_its_ui_thread(ui):
    """Accidental synchronous UI calls fail explicitly instead of deadlocking the mailbox."""
    controller = SessionController(ui)

    async def scenario():
        """Exercise the public injected interaction while the controller is active."""
        ready = asyncio.Event()

        def show(message=""):
            """Check the UI-thread ownership guard from inside real dispatch."""
            if message == "ready":
                with pytest.raises(RuntimeError, match="outside the UI thread"):
                    controller.interaction.info("invalid nested call")
                ready.set()

        ui.info.side_effect = show
        await controller._run_sync_callback_async(
            lambda _receive: controller.interaction.info("ready")
        )
        assert ready.is_set()

    asyncio.run(scenario())


def test_async_controller_keeps_conversation_on_ui_loop_and_routes_approval(ui):
    """Async work can await a separate approval while the UI driver owns input."""
    controller = SessionController(ui)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)
    ui.confirm_async.return_value = True
    owner_thread = []

    async def conversation(receive):
        """Read admitted input, then request approval through the controller mailbox."""
        owner_thread.append(get_ident())
        assert await receive(DummyCompleter()) == "hello"
        assert await controller.interaction.confirm_async("Approve operation?") is True
        controller.interaction.info("approved")

    ui_thread = get_ident()

    async def scenario():
        """Run the async owner and submit its first ordinary message."""
        task = asyncio.create_task(controller.run_async(conversation))
        await asyncio.sleep(0)
        await inputs.put("hello")
        await task

    asyncio.run(scenario())

    assert owner_thread == [ui_thread]
    ui.confirm_async.assert_awaited_once_with("Approve operation?", default=False)
    ui.info.assert_any_call("approved")
    assert controller.running is False


def test_async_controller_processes_status_while_conversation_is_suspended(ui):
    """A read-only control command remains available during async work."""
    controller = SessionController(ui)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)
    working, status_seen, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    displayed = []

    def record_info(message=""):
        """Capture output and signal only after the status response is displayed."""
        displayed.append(message)
        if message.endswith("queued inputs."):
            status_seen.set()

    ui.info.side_effect = record_info

    async def conversation(receive):
        """Wait at an async operation boundary after consuming its initiating message."""
        assert await receive(DummyCompleter()) == "hello"
        working.set()
        await release.wait()

    async def scenario():
        """Keep the agent suspended while the UI executes its status command."""
        task = asyncio.create_task(controller.run_async(conversation))
        await asyncio.sleep(0)
        await inputs.put("hello")
        await working.wait()
        await inputs.put("/status")
        await status_seen.wait()
        release.set()
        await task

    asyncio.run(scenario())

    assert any(message.endswith("queued inputs.") for message in displayed)


def test_async_controller_rejects_reuse_and_suppresses_requested_keyboard_interrupt(ui):
    """The async controller remains single-use and normalizes stopped input reads."""
    controller = SessionController(ui)
    controller.stop()

    async def conversation(receive):
        """Observe the stop sentinel without running another operation."""
        assert await receive(DummyCompleter()) is False

    async def scenario():
        """Run once after stop, then verify the single-use lifecycle guard."""
        await controller.run_async(conversation)
        with pytest.raises(RuntimeError, match="only run once"):
            await controller.run_async(conversation)

    asyncio.run(scenario())

    ui.conversation_ended.assert_called_once_with()


def test_async_controller_wakes_a_pending_receive_when_stopped(ui):
    """A stop request wakes the async mailbox reader and rejects a later message."""
    controller = SessionController(ui)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)
    input_prompted = asyncio.Event()

    async def prompt(*args, **kwargs):
        """Signal that the owner is waiting for admitted input."""
        input_prompted.set()
        return await inputs.get()

    ui.prompt_async.side_effect = prompt

    async def conversation(receive):
        """Wait for a message that the stop request replaces with its sentinel."""
        assert await receive(DummyCompleter()) is False

    async def scenario():
        """Request shutdown only after the async receive is waiting."""
        task = asyncio.create_task(controller.run_async(conversation))
        await input_prompted.wait()
        controller.stop()
        await task

    asyncio.run(scenario())

    assert controller.control.stopped is True


def test_async_controller_rechecks_queue_after_a_racing_admission(ui):
    """A queued input arriving at the clear-and-wait boundary is not lost."""
    controller = SessionController(ui)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)

    class RacingMailbox(InputMailbox):
        """Force one empty read followed by a visible queued input."""

        _first_read: bool

        def __init__(self):
            """Initialize a pre-admitted message and one synthetic read race."""
            super().__init__(controller._inbox.capacity)
            self._first_read = True
            self.try_put(InputAdmission(1, 0, InputAdmissionKind.MESSAGE, "raced"))

        def get_nowait(self, **kwargs):
            """Publish the raced admission's wakeup after one synthetic empty read."""
            if self._first_read:
                self._first_read = False
                controller._input_signal.set()
                raise Empty
            return super().get_nowait(**kwargs)

    controller._inbox = RacingMailbox()
    received = []

    async def conversation(receive):
        """Read the input admitted just before the mailbox wait."""
        received.append(await receive(DummyCompleter()))

    asyncio.run(controller.run_async(conversation))

    assert received == ["raced"]


def test_async_controller_handles_inline_output_and_rejects_sync_input(ui):
    """Async owners present output inline and cannot issue blocking UI input calls."""
    controller = SessionController(ui)
    ui.info.side_effect = ValueError("display failed")
    terminal_exit = InteractionCall(lambda: None, (), {}, kind="terminal_exit")
    controller._terminal_owned = True

    async def conversation(_receive):
        """Exercise inline output failure, terminal cleanup, and sync-input rejection."""
        with pytest.raises(ValueError, match="display failed"):
            controller._send(InteractionCall(ui.info, ("bad",), {}))
        with pytest.raises(RuntimeError, match="Synchronous mailbox input"):
            controller._send(InteractionCall(ui.confirm, ("Approve?",), {}, kind=EVENT_CONFIRM))
        controller._send(terminal_exit)

    asyncio.run(controller.run_async(conversation))

    assert terminal_exit.result.result() is None
    assert controller._terminal_owned is False


def test_async_controller_settles_failures_and_reports_discarded_input(ui):
    """Execution failures remain primary while queue-reporting errors are attached."""
    controller = SessionController(ui)
    controller._inbox.try_put(InputAdmission(1, 0, InputAdmissionKind.MESSAGE, "queued"))
    ui.warning.side_effect = OSError("warning failed")

    async def conversation(_receive):
        """Fail the async owner after the controller has started it."""
        raise RuntimeError("operation failed")

    with pytest.raises(RuntimeError, match="operation failed") as error:
        asyncio.run(controller.run_async(conversation))

    assert any(
        "Queue cancellation reporting also failed with OSError" in note
        for note in error.value.__notes__
    )
    assert controller.running is False


def test_async_controller_propagates_unsolicited_keyboard_interrupt(ui):
    """Keyboard interrupts remain visible unless the controller requested shutdown."""
    controller = SessionController(ui)

    async def conversation(_receive):
        """Raise an unsolicited interrupt from the conversation owner."""
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(controller.run_async(conversation))


def test_async_controller_waits_for_owned_work_after_repeated_cancellation(ui):
    """A second cancellation cannot abandon async work still owned by the controller."""
    controller = SessionController(ui)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)
    operation_started, cleanup_started, release = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    settle = controller._settle_async

    async def observe_settlement(worker):
        """Expose the cleanup boundary before delegating to the real settlement loop."""
        cleanup_started.set()
        return await settle(worker)

    controller._settle_async = observe_settlement

    async def conversation(receive):
        """Keep owned work active until the test releases it."""
        assert await receive(DummyCompleter()) == "hello"
        operation_started.set()
        await release.wait()

    async def scenario():
        """Cancel twice while cleanup waits, then allow the operation to settle."""
        await inputs.put("hello")
        task = asyncio.create_task(controller.run_async(conversation))
        await operation_started.wait()
        task.cancel()
        await cleanup_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert controller.running is False


def test_async_controller_settles_a_sync_command_after_conversation_completion(ui, monkeypatch):
    """Cancellation awaits a sync command after the conversation owner finishes."""
    controller = SessionController(ui)
    commands = CommandManager()

    def held() -> None:
        """Represent one synchronous extension command held by its executor future."""

    commands.register(
        CommandRegistration(
            held,
            name="held",
            capabilities=frozenset({CommandCapability.CONTEXT_READ}),
        )
    )
    controller.bind_command_manager(commands)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)

    async def scenario():
        """Control owner and command completion independently at their public async boundaries."""
        loop = asyncio.get_running_loop()
        started, command_started, finish_owner, owner_finished, settling = (
            asyncio.Event() for _ in range(5)
        )
        pending = loop.create_future()
        owner = None

        def execute(*args):
            """Represent a still-running command at the native executor boundary."""
            command_started.set()
            return pending

        monkeypatch.setattr(loop, "run_in_executor", execute)
        original_wait = asyncio.wait

        async def observe_wait(futures, **kwargs):
            """Expose command cleanup after the cancelled driver releases its completed owner."""
            if task.cancelling() and controller._command_task in futures:
                assert owner.done()
                settling.set()
            return await original_wait(futures, **kwargs)

        monkeypatch.setattr(controller_module.asyncio, "wait", observe_wait)

        async def conversation(receive):
            """Finish ordinary conversation work while command cleanup remains pending."""
            nonlocal owner
            owner = asyncio.current_task()
            assert await receive(DummyCompleter()) == "hello"
            started.set()
            await finish_owner.wait()
            owner_finished.set()

        task = asyncio.create_task(controller.run_async(conversation))
        try:
            await inputs.put("hello")
            await started.wait()
            await inputs.put("/held")
            await command_started.wait()
            finish_owner.set()
            await owner_finished.wait()
            task.cancel()
            await settling.wait()
            assert not task.done()
            pending.set_result(True)
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            finish_owner.set()
            if not pending.done():
                pending.set_result(True)
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert controller.running is False
    commands.close()


def test_async_controller_preserves_primary_and_secondary_failures(ui):
    """UI, command, and conversation failures remain visible with the primary error first."""
    controller = SessionController(ui)
    command_failure = RuntimeError("command failed")

    async def fail_after_owner(worker, *, route_idle_commands):
        """Wait for the conversation failure, then report a separate UI failure."""
        await asyncio.gather(worker, return_exceptions=True)
        raise ValueError("driver failed")

    async def settle_with_command_failure(_worker):
        """Return a late command failure for the controller to attach."""
        return command_failure

    async def conversation(_receive):
        """Fail the conversation owner independently of the UI driver."""
        raise OSError("conversation failed")

    controller._drive_async = fail_after_owner
    controller._settle_async = settle_with_command_failure

    with pytest.raises(ValueError, match="driver failed") as error:
        asyncio.run(controller.run_async(conversation))

    assert any("Command also stopped with RuntimeError" in note for note in error.value.__notes__)
    assert any("Conversation also stopped with OSError" in note for note in error.value.__notes__)


def test_async_controller_reports_discarded_input_failure_without_primary_error(ui):
    """A failure reporting discarded queued input surfaces after normal settlement."""
    controller = SessionController(ui)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)
    ui.warning.side_effect = OSError("warning failed")
    started, stopping = asyncio.Event(), asyncio.Event()

    def observe_info(message=""):
        """Signal when the UI accepted its terminal-exit response."""
        if message.startswith("Stopping after"):
            stopping.set()

    ui.info.side_effect = observe_info

    async def conversation(receive):
        """Wait for UI exit while a previously queued input is discarded."""
        assert await receive(DummyCompleter()) == "hello"
        controller._inbox.try_put(InputAdmission(1, 0, InputAdmissionKind.MESSAGE, "queued"))
        started.set()
        await stopping.wait()

    async def scenario():
        """Submit a message, then exit while the owner is doing asynchronous work."""
        task = asyncio.create_task(controller.run_async(conversation))
        await inputs.put("hello")
        await started.wait()
        await inputs.put(False)
        with pytest.raises(OSError, match="warning failed"):
            await task

    asyncio.run(scenario())


def test_async_controller_stop_without_start_and_requested_keyboard_interrupt(ui):
    """Stop-before-start wakes no foreign loop and suppresses its owned interrupt."""
    controller = SessionController(ui)
    assert controller._stop() == 0

    async def conversation(_receive):
        """Raise an interrupt after shutdown has already been requested."""
        raise KeyboardInterrupt

    asyncio.run(controller.run_async(conversation))

    ui.conversation_ended.assert_called_once_with()


def test_async_controller_stop_signals_from_a_foreign_thread_identity(ui):
    """The stop signal uses thread-safe scheduling when the caller is not the UI owner."""
    controller = SessionController(ui)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)
    input_prompted = asyncio.Event()

    async def prompt(*args, **kwargs):
        """Signal that the conversation owner is waiting for a message."""
        input_prompted.set()
        return await inputs.get()

    ui.prompt_async.side_effect = prompt

    async def conversation(receive):
        """Wait for the stop sentinel."""
        assert await receive(DummyCompleter()) is False

    async def scenario():
        """Exercise the thread-safe input wakeup while remaining on one event loop."""
        task = asyncio.create_task(controller.run_async(conversation))
        await input_prompted.wait()
        controller._ui_thread = -1
        controller._stop()
        await task

    asyncio.run(scenario())


def test_async_controller_marks_inline_conversation_end_as_delivered(ui):
    """An inline conversation-ended event is not emitted twice after shutdown."""
    controller = SessionController(ui)
    controller.stop()

    async def conversation(_receive):
        """Deliver the terminal event through the async owner's UI thread."""
        controller.interaction.conversation_ended()

    asyncio.run(controller.run_async(conversation))

    ui.conversation_ended.assert_called_once_with()


def test_controller_preserves_output_failures_at_the_execution_boundary(ui):
    """Presentation failures return to their blocked owner and do not leave a running worker."""
    controller = SessionController(ui)
    ui.answer.side_effect = ValueError("render failed")

    with pytest.raises(ValueError, match="render failed"):
        asyncio.run(
            controller._run_sync_callback_async(
                lambda _receive: controller.interaction.answer("answer")
            )
        )

    assert controller.running is False


def test_controller_applies_backpressure_to_concurrent_tool_output(ui):
    """Auxiliary tool threads cannot overflow UI dispatch or render outside its owner."""
    controller = SessionController(ui)
    threads = []
    ui.info.side_effect = lambda _text: threads.append(get_ident())

    def conversation(_receive):
        """Join all tool-owned output producers before releasing conversation ownership."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(controller.interaction.info, (str(index) for index in range(30))))

    asyncio.run(controller._run_sync_callback_async(conversation))

    assert sorted(int(item.args[0]) for item in ui.info.call_args_list) == list(range(30))
    assert set(threads) == {get_ident()}


def test_controller_reports_blank_input_and_missing_response_requests(ui):
    """UI control commands never become model input when their request is absent."""
    controller = SessionController(ui)

    async def scenario():
        """Submit idle UI controls before closing the ordinary input channel."""
        inputs = asyncio.Queue()
        read_inputs(ui, inputs)
        for value in (
            "",
            "/respond",
            "/interrupt",
            "/steer",
            "/status",
            False,
        ):
            inputs.put_nowait(value)
        await controller._run_sync_callback_async(lambda receive: receive(DummyCompleter()))

    asyncio.run(scenario())
    assert [item.args[0] for item in ui.warning.call_args_list] == [
        "Please enter a message!",
        (
            "Command '/respond' is unavailable in the current conversation state. "
            "Type /help to see commands that can run now."
        ),
        (
            "Command '/interrupt' is unavailable in the current conversation state. "
            "Type /help to see commands that can run now."
        ),
        "No pending message to steer",
    ]
    ui.info.assert_any_call("Idle; 0 queued inputs.")


@pytest.mark.parametrize("cancelled", [False, True])
def test_controller_propagates_errors_and_cancellation_from_active_request_input(ui, cancelled):
    """Input failures revoke their own request and leave no blocked execution owner."""
    controller = SessionController(ui)

    async def answer(*_args, **_kwargs):
        """Fail while a startup confirmation owns the input surface."""
        if cancelled:
            raise asyncio.CancelledError
        raise ValueError("input failed")

    ui.confirm_async.side_effect = answer
    expected = asyncio.CancelledError if cancelled else ValueError
    with pytest.raises(expected):
        asyncio.run(
            controller._run_sync_callback_async(
                lambda _receive: controller.interaction.confirm("Approve?")
            )
        )

    assert controller.running is False


def test_controller_external_stop_cancels_an_active_decision_prompt(ui):
    """A stop request revokes an opened approval without waiting for a stale user reply."""
    controller = SessionController(ui)

    async def scenario():
        """Stop while a confirmation is actively waiting for terminal input."""
        opened = asyncio.Event()

        async def answer(*_args, **_kwargs):
            """Keep the approval surface open until the controller cancels it."""
            opened.set()
            await asyncio.Event().wait()

        ui.confirm_async.side_effect = answer
        task = asyncio.create_task(
            controller._run_sync_callback_async(
                lambda _receive: controller.interaction.confirm("Approve?", default=True)
            )
        )
        try:
            await opened.wait()
            controller.stop()
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert controller.running is False
    ui.conversation_ended.assert_called_once_with()


def test_controller_external_stop_cancels_an_active_secret_prompt(ui):
    """Shutdown cancels a masked request before releasing exclusive prompt ownership."""
    controller = SessionController(ui)

    async def scenario():
        """Stop while a masked prompt is actively waiting for terminal input."""
        opened = asyncio.Event()

        async def answer(*_args, **kwargs):
            """Keep the masked request open until the controller cancels its input task."""
            if kwargs.get("secret"):
                opened.set()
                await asyncio.Event().wait()

        ui.prompt_async.side_effect = answer
        task = asyncio.create_task(
            controller._run_sync_callback_async(
                lambda _receive: controller.interaction.prompt("Credential:", secret=True)
            )
        )
        try:
            await opened.wait()
            controller.stop()
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert controller.running is False
    assert any(call.kwargs.get("secret") for call in ui.prompt_async.await_args_list)


@pytest.mark.parametrize("failure", [asyncio.CancelledError, ValueError])
def test_controller_propagates_failures_from_a_masked_prompt(ui, failure):
    """Masked prompt failures are returned to the execution owner without being swallowed."""
    controller = SessionController(ui)
    ui.prompt_async.side_effect = failure

    with pytest.raises(failure):
        asyncio.run(
            controller._run_sync_callback_async(
                lambda _receive: controller.interaction.prompt("Credential:", secret=True)
            )
        )

    assert controller.running is False


def test_controller_rejects_a_new_approval_requested_after_stop(ui):
    """An operation finishing during shutdown cannot open or obtain a new approval."""
    controller = SessionController(ui)
    release = Event()

    async def scenario():
        """Stop while work is active and let it attempt a late input request."""
        started = asyncio.Event()
        ui.info.side_effect = lambda text="": started.set() if text == "working" else None

        def conversation(_receive):
            """Attempt confirmation after the active operation has settled."""
            controller.interaction.info("working")
            release.wait()
            controller.interaction.confirm("Too late", default=True)

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            await started.wait()
            controller.stop()
            release.set()
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    ui.confirm_async.assert_not_awaited()


def test_controller_retains_cleanup_failure_evidence_during_repeated_cancellation(ui):
    """Repeated cancellation cannot detach the owner or hide a cleanup failure."""
    controller = SessionController(ui)
    release = Event()

    async def scenario():
        """Cancel again after the worker has observed shutdown and before releasing it."""
        started, cleaning = asyncio.Event(), asyncio.Event()
        ui.info.side_effect = lambda text="": (
            started.set() if text == "working" else cleaning.set() if text == "cleaning" else None
        )

        def conversation(_receive):
            """Wait for shutdown, report cleanup, then fail after explicit release."""
            controller.interaction.info("working")
            # Waiting uses the public stop signal, with no elapsed-time assertion.
            try:
                controller.control.wait(60)
            except KeyboardInterrupt:
                controller.interaction.info("cleaning")
                release.wait()
            raise ValueError("cleanup failed")

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            await started.wait()
            task.cancel()
            await cleaning.wait()
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError) as raised:
                await task
            assert any("ValueError" in note for note in raised.value.__notes__)
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("returned_input", [False, True])
def test_controller_external_stop_releases_an_idle_reader_without_admitting_late_input(
    ui, returned_input
):
    """Stopping an idle owner cancels its editor and rejects input returned during shutdown."""
    controller = SessionController(ui)
    messages = []

    async def scenario():
        """Stop with either an open editor or a completed input waiting for dispatch."""
        opened = asyncio.Event()

        async def read(*_args, **_kwargs):
            """Return text concurrently with stop or remain suspended until cancelled."""
            opened.set()
            if returned_input:
                controller.stop()
                return "late input"
            await asyncio.Event().wait()

        ui.prompt_async.side_effect = read
        task = asyncio.create_task(
            controller._run_sync_callback_async(
                lambda receive: messages.append(receive(DummyCompleter()))
            )
        )
        try:
            await opened.wait()
            if not returned_input:
                controller.stop()
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert messages == [False]
    assert controller.running is False


def test_controller_preserves_unrequested_worker_interruption(ui):
    """An unexpected worker interrupt is propagated after execution has settled."""
    controller = SessionController(ui)

    def conversation(_receive):
        """Interrupt without receiving a cooperative stop request."""
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(controller._run_sync_callback_async(conversation))
    assert controller.running is False


def test_controller_stop_tolerates_the_event_loop_settling_in_another_thread(ui, monkeypatch):
    """A concurrent stop remains harmless when execution settles before its callback is sent."""
    controller = SessionController(ui)
    release_worker, release_stopper = Event(), Event()
    errors = []
    stopper_id = []

    async def scenario():
        """Hold a public scheduler call across the execution lifecycle's final boundary."""
        started, stopping = asyncio.Event(), asyncio.Event()
        loop = asyncio.get_running_loop()
        schedule = loop.call_soon_threadsafe
        ui.info.side_effect = lambda _text: started.set()

        def call_soon(callback, *args, **kwargs):
            """Model a loop closing after a stopper has read its active reference."""
            if stopper_id and get_ident() == stopper_id[0]:
                schedule(stopping.set)
                release_stopper.wait()
                raise RuntimeError("event loop closed")
            return schedule(callback, *args, **kwargs)

        def stop():
            """Request shutdown from an external caller and retain any unexpected error."""
            stopper_id.append(get_ident())
            try:
                controller.stop()
            except RuntimeError as error:
                errors.append(error)

        def conversation(_receive):
            """Keep execution active until the stop scheduler is paused."""
            controller.interaction.info("working")
            release_worker.wait()

        monkeypatch.setattr(loop, "call_soon_threadsafe", call_soon)
        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        thread = Thread(target=stop)
        try:
            await started.wait()
            thread.start()
            await stopping.wait()
            release_worker.set()
            await task
        finally:
            release_worker.set()
            release_stopper.set()
            monkeypatch.setattr(loop, "call_soon_threadsafe", schedule)
            await loop.run_in_executor(None, thread.join)
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert errors == []
    assert controller.control.stopped is True


def test_controller_stop_preserves_real_scheduler_failures(ui, monkeypatch):
    """An active loop's scheduling failure is not mistaken for an already-settled lifecycle."""
    controller = SessionController(ui)

    async def scenario():
        """Fail a stop submission while the UI still owns the active lifecycle."""
        loop = asyncio.get_running_loop()

        def output(_text):
            """Exercise stop on a blocked owner without affecting later scheduling."""

            def fail(*_args, **_kwargs):
                """Report an actual scheduler failure before execution has settled."""
                raise RuntimeError("scheduler failed")

            with monkeypatch.context() as patch:
                patch.setattr(loop, "call_soon_threadsafe", fail)
                with pytest.raises(RuntimeError, match="scheduler failed"):
                    controller.stop()

        ui.info.side_effect = output
        await controller._run_sync_callback_async(
            lambda _receive: controller.interaction.info("working")
        )

    asyncio.run(scenario())


def test_controller_cancellation_does_not_lose_dequeued_output(ui, monkeypatch):
    """An interaction already removed from the mailbox is acknowledged during cancellation."""
    real_queue = asyncio.Queue
    controller = SessionController(ui)

    async def scenario():
        """Cancel exactly when the public queue receive returns its first interaction."""
        cancelled = False

        class CancellationQueue(real_queue):
            """Expose cancellation at a deterministic mailbox receive boundary."""

            async def get(self):
                """Cancel the owning controller after dequeuing its first item."""
                nonlocal cancelled
                item = await super().get()
                if not cancelled:
                    cancelled = True
                    task.cancel()
                return item

        monkeypatch.setattr(controller_module.asyncio, "Queue", CancellationQueue)
        task = asyncio.create_task(
            controller._run_sync_callback_async(
                lambda _receive: controller.interaction.info("buffered output")
            )
        )
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    ui.info.assert_called_once_with("buffered output")
    assert controller.running is False


@pytest.mark.parametrize("worker_failure", [False, True])
def test_controller_reports_discarded_inputs_without_hiding_primary_failures(ui, worker_failure):
    """Queue cancellation is reported after settlement, with primary failure evidence retained."""
    controller = SessionController(ui)
    release = Event()
    ui.warning.side_effect = ValueError("queue display failed")

    async def scenario():
        """Keep one queued input pending while shutdown or execution failure ends the owner."""
        inputs = asyncio.Queue()
        working, queued, stopping = asyncio.Event(), asyncio.Event(), asyncio.Event()
        read_inputs(ui, inputs)
        ui.info.side_effect = lambda text="": (
            working.set()
            if text == "working"
            else queued.set()
            if text.startswith("Queued")
            else stopping.set()
            if text.startswith("Stopping")
            else None
        )

        def conversation(receive):
            """Fail or settle only after a second input has been admitted to the queue."""
            receive(DummyCompleter())
            controller.interaction.info("working")
            release.wait()
            if worker_failure:
                raise LookupError("worker failed")

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("first")
            await working.wait()
            inputs.put_nowait("unexecuted")
            await queued.wait()
            if not worker_failure:
                inputs.put_nowait(False)
                await stopping.wait()
            release.set()
            expected = LookupError if worker_failure else ValueError
            with pytest.raises(expected) as raised:
                await task
            if worker_failure:
                assert "queue cancellation reporting" in raised.value.__notes__[0].casefold()
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    ui.warning.assert_called_once_with("Stopped; 1 queued inputs were not executed.")


def test_steering_precedes_queued_messages_and_keeps_corrections_in_submission_order(ui):
    """Corrections interrupt the current scope, then retain FIFO ahead of ordinary queued text."""

    controller = SessionController(ui)
    release = Event()
    messages = []

    async def scenario():
        """Keep the execution owner held until both corrections and status are acknowledged."""
        inputs = asyncio.Queue()
        working, interrupted, status, completed = (asyncio.Event() for _ in range(4))
        read_inputs(ui, inputs)
        requests = []

        def info(text=""):
            """Observe UI transitions without inspecting controller state."""
            if text == "working":
                working.set()
            elif text.startswith("Interrupt requested"):
                requests.append(text)
                if len(requests) == 2:
                    interrupted.set()
            elif text == "Interrupting; 4 queued inputs.":
                status.set()
            elif text == "done":
                completed.set()

        ui.info.side_effect = info

        def conversation(receive):
            """Simulate an active tool boundary and then consume four admitted inputs."""
            messages.append(receive(DummyCompleter()))
            with controller.control.execution():
                controller.interaction.info("working")
                release.wait()
                with pytest.raises(RunInterrupted):
                    controller.control.checkpoint()
            for _ in range(4):
                messages.append(receive(DummyCompleter()))
            controller.interaction.info("done")
            receive(DummyCompleter())

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("first")
            await working.wait()
            for value in (
                "/queue queued one",
                "/queue queued two",
                "/steer correction",
                "/steer /new",
            ):
                inputs.put_nowait(value)
            await interrupted.wait()
            inputs.put_nowait("/status")
            await status.wait()
            release.set()
            await completed.wait()
            inputs.put_nowait("/exit")
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert messages == ["first", "correction", "/new", "queued one", "queued two"]
    assert all(isinstance(value, SteeringMessage) for value in messages[1:3])


def test_run_interruption_revokes_pending_approval_without_ending_the_chat(ui):
    """An active approval is revoked by run control and leaves the chat available."""
    controller = SessionController(ui)
    decisions, messages = [], []

    async def scenario():
        """Interrupt the native question, then accept a fresh conversation message."""
        inputs = asyncio.Queue()
        prompted, finished, processed = (asyncio.Event() for _ in range(3))
        read_inputs(ui, inputs)

        async def confirm(*_args, **_kwargs):
            """Signal that approval input opened, then wait until run control cancels it."""
            prompted.set()
            await asyncio.Event().wait()

        ui.confirm_async.side_effect = confirm

        def info(text=""):
            """Observe the execution owner after its approval request is revoked."""
            if text == "interrupted":
                finished.set()
            elif text == "processed":
                processed.set()

        ui.info.side_effect = info

        def conversation(receive):
            """Catch only a run interrupt and leave the reader open for another message."""
            receive(DummyCompleter())
            with controller.control.execution(), pytest.raises(RunInterrupted):
                decisions.append(controller.interaction.confirm("Approve?", default=True))
            controller.interaction.info("interrupted")
            messages.append(receive(DummyCompleter()))
            controller.interaction.info("processed")
            receive(DummyCompleter())

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("first")
            await prompted.wait()
            assert controller.request_interrupt() is True
            await finished.wait()
            assert controller.control.stopped is False
            inputs.put_nowait("later")
            await processed.wait()
            inputs.put_nowait("/exit")
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert decisions == []
    assert messages == ["later"]
    ui.confirm_async.assert_awaited_once_with("Approve?", default=True)


def test_rejected_steering_does_not_interrupt_or_displace_a_full_mailbox(ui):
    """Capacity refusal leaves the active run and all sixteen accepted queued messages intact."""
    controller = SessionController(ui)
    release = Event()
    interrupted = []
    received = []

    async def scenario():
        """Fill the queue behind an active scope, reject steering, then end normally."""
        inputs = asyncio.Queue()
        working, rejected, checked = (asyncio.Event() for _ in range(3))
        read_inputs(ui, inputs)

        def info(text=""):
            """Expose controlled execution boundaries through UI output."""
            if text == "working":
                working.set()
            elif text == "checked":
                checked.set()

        ui.info.side_effect = info
        ui.warning.side_effect = lambda text: rejected.set() if "queue is full" in text else None

        def conversation(receive):
            """Hold active ownership until rejected admission is observable."""
            receive(DummyCompleter())
            with controller.control.execution():
                controller.interaction.info("working")
                release.wait()
                interrupted.append(controller.control.interrupted)
                controller.control.checkpoint()
            for _ in range(16):
                received.append(receive(DummyCompleter()))
            controller.interaction.info("checked")

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("first")
            await working.wait()
            for index in range(16):
                inputs.put_nowait(f"/queue queued {index}")
            inputs.put_nowait("/steer rejected")
            await rejected.wait()
            release.set()
            await checked.wait()
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert interrupted == [False]
    assert received == [f"queued {index}" for index in range(16)]
    assert sum("queue is full" in call.args[0] for call in ui.warning.call_args_list) == 1


@pytest.mark.parametrize("operation", ["prompt", "confirm", "terminal_context"])
def test_interrupted_runs_cannot_open_late_questions_or_terminal_handoffs(ui, operation):
    """A question arriving after the control is accepted is revoked before any UI is opened."""
    controller = SessionController(ui)
    release = Event()

    async def scenario():
        """Accept interruption while the execution owner is held before requesting UI."""
        inputs = asyncio.Queue()
        working, interrupted, completed = (asyncio.Event() for _ in range(3))
        read_inputs(ui, inputs)

        def info(text=""):
            """Expose the request and interrupt ordering through UI acknowledgements."""
            if text == "working":
                working.set()
            elif text.startswith("Interrupt requested"):
                interrupted.set()
            elif text == "completed":
                completed.set()

        ui.info.side_effect = info

        def conversation(receive):
            """Try to open a stale approval or terminal scope after interruption."""
            receive(DummyCompleter())
            with controller.control.execution(), pytest.raises(RunInterrupted):
                controller.interaction.info("working")
                release.wait()
                if operation == "terminal_context":
                    with controller.interaction.terminal_context():
                        pytest.fail("Interrupted work must not obtain terminal ownership")
                else:
                    getattr(controller.interaction, operation)("Approve?")
            controller.interaction.info("completed")
            receive(DummyCompleter())

        task = asyncio.create_task(controller._run_sync_callback_async(conversation))
        try:
            inputs.put_nowait("first")
            await working.wait()
            inputs.put_nowait("/interrupt")
            await interrupted.wait()
            release.set()
            await completed.wait()
            inputs.put_nowait("/exit")
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    ui.confirm_async.assert_not_awaited()
    ui.terminal_context.assert_not_called()
    assert all(not call.args for call in ui.prompt_async.call_args_list)
