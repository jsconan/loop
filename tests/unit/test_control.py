"""Verify cooperative cancellation without real retry delays."""

import asyncio
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop import RunControl, RunInterrupted
from loop import control as control_module
from loop.control import RunStopRequested
from tests.scheduling import deliver_callbacks


def test_control_stays_stopped_after_a_stop_request():
    """A stop signal is sticky and every later checkpoint interrupts execution."""
    control = RunControl()
    assert control.stopped is False
    control.checkpoint()

    control.stop()
    control.stop()

    assert control.stopped is True
    assert control.cancellation_event.is_set()
    with pytest.raises(KeyboardInterrupt):
        control.checkpoint()


@pytest.mark.parametrize("stopped", [True, False])
def test_control_wait_uses_the_stop_event_and_checks_it_after_waiting(monkeypatch, stopped):
    """Retry delays wait on the cancellation signal and cannot continue after a stop."""
    event = Mock()
    monkeypatch.setattr(control_module, "Event", Mock(return_value=event))
    control = RunControl()
    if stopped:
        control.stop()

    if stopped:
        with pytest.raises(KeyboardInterrupt):
            control.wait(2.5)
    else:
        control.wait(2.5)

    event.wait.assert_called_once_with(2.5)


def test_interrupt_is_scoped_and_never_latches_onto_an_idle_or_later_run():
    """Interrupts target the active execution and clear only after its scope settles."""
    control = RunControl()
    assert control.interrupt() is False
    assert control.active is False
    assert control.interrupted is False

    with control.execution():
        assert control.active is True
        assert control.interrupt() is True
        assert control.interrupt() is True
        assert control.interrupted is True
        assert control.cancelling is True
        assert control.cancellation_event.is_set()
        assert control.stopped is False
        with pytest.raises(RunInterrupted):
            control.checkpoint()
        with pytest.raises(RuntimeError, match="already owns"), control.execution():
            pytest.fail("Overlapping executions must be refused")

    assert control.active is False
    assert control.interrupted is False
    assert control.cancelling is False
    assert control.cancelled is True
    assert not control.cancellation_event.is_set()
    with control.execution():
        assert control.cancelled is False
        control.checkpoint()


def test_pause_parks_the_same_async_run_until_continue():
    """A pause keeps its owner alive at a cooperative checkpoint until continued."""
    control = RunControl()
    assert control.pause() is False

    async def scenario():
        """Park the owner and release it through an explicit state transition."""
        started = asyncio.Event()
        checkpoint = asyncio.Event()
        resumed = asyncio.Event()

        async def owner():
            """Wait at a checkpoint while its control state is paused."""
            with control.execution(async_owner=True):
                started.set()
                await checkpoint.wait()
                await control.checkpoint_async()
                resumed.set()

        task = asyncio.create_task(owner())
        await started.wait()
        assert control.pause() is True
        assert control.pause() is True
        checkpoint.set()
        await deliver_callbacks()
        assert task.done() is False
        assert control.paused is True
        assert control.interrupted is False
        assert control.cancellation_event.is_set() is False
        assert control.continue_run() is True
        await task
        assert resumed.is_set()

    asyncio.run(scenario())
    assert control.paused is False
    assert control.cancelled is False


def test_cold_stop_unwinds_at_a_checkpoint_without_cancelling_the_owner():
    """A cold stop returns the run to its caller while preserving the conversation owner."""

    async def scenario():
        """Request a cold stop while the owner waits in a cancellable delay."""
        control = RunControl()
        started = asyncio.Event()

        async def owner():
            """Translate the cooperative stop exception into an observable result."""
            with control.execution(async_owner=True):
                started.set()
                with pytest.raises(RunStopRequested):
                    await control.wait_async(60)

        task = asyncio.create_task(owner())
        await started.wait()
        assert control.request_stop_run() is True
        assert control.request_stop_run() is True
        await task
        return control

    control = asyncio.run(scenario())

    assert control.stopped is False
    assert control.cancelled is False
    assert control.cancellation_event.is_set() is False


def test_pause_cannot_override_sticky_shutdown():
    """A conversation shutdown prevents a later pause from claiming the run."""
    control = RunControl()
    with control.execution():
        control.stop()
        assert control.pause() is False
        assert control.paused is False


def test_cold_stop_supersedes_pause_and_cannot_be_reversed():
    """A cold stop releases a parked run and remains final until the scope settles."""
    control = RunControl()
    with control.execution():
        assert control.pause() is True
        assert control.request_stop_run() is True
        assert control.stop_requested is True
        assert control.continue_run() is False
        assert control.pause() is False
        assert control.interrupt() is False
        with pytest.raises(RunStopRequested):
            control.checkpoint()


@pytest.mark.parametrize(
    ("active", "stopped", "admitted", "expected"),
    [
        (False, False, True, False),
        (True, True, True, False),
        (True, False, False, False),
        (True, False, True, True),
    ],
)
def test_steering_admission_is_scoped_to_an_active_run(active, stopped, admitted, expected):
    """Only admitted steering during a live run publishes the safe-boundary signal."""
    control = RunControl()
    called = []
    scope = control.execution()
    if active:
        scope.__enter__()
    if stopped:
        control.stop()

    result = control.request_steering(lambda: called.append(True) or admitted)

    assert result is expected
    assert bool(called) is (active and not stopped)
    assert control.steering_pending is expected
    assert control.cancellation_event.is_set() is stopped
    if active:
        scope.__exit__(None, None, None)
        assert control.steering_pending is False


def test_steering_requests_are_scoped_and_commit_atomically():
    """Withdrawal affects only its request and cannot undo a committed decision."""
    control = RunControl()
    assert control.commit_steering() is False
    with control.execution():
        assert control.request_steering(lambda: True, key=1)
        assert control.request_steering(lambda: True, key=2)
        control.withdraw_steering(1)
        assert control.steering_pending is True
        assert control.commit_steering() is True
        control.withdraw_steering(2)
        assert control.steering_pending is True
        assert control.commit_steering() is True
    with control.execution():
        assert control.steering_pending is False
        assert control.request_steering(lambda: True, key=2)
        control.withdraw_steering(2)
        assert control.commit_steering() is False


def test_shutdown_supersedes_interruption_and_survives_execution_cleanup():
    """Settling an interrupted scope cannot clear a concurrent sticky shutdown request."""
    control = RunControl()
    with control.execution(), pytest.raises(KeyboardInterrupt) as raised:
        control.interrupt()
        control.stop()
        assert control.cancelling is True
        assert control.interrupt() is False
        control.checkpoint()

    assert type(raised.value) is KeyboardInterrupt
    assert control.active is False
    assert control.interrupted is False
    assert control.cancelling is False
    assert control.cancelled is True
    assert control.stopped is True
    assert control.interrupt() is False


def test_async_execution_cancels_its_owner_and_tracks_settlement():
    """Explicit run interruption cancels its bound task only after cleanup is delivered."""

    async def scenario():
        """Hold an async execution until its control requests cancellation."""
        control = RunControl()
        started = asyncio.Event()

        async def owner():
            """Wait inside the run scope until task cancellation is delivered."""
            with control.execution(async_owner=True):
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(owner())
        await started.wait()
        assert control.cancelling is False
        assert control.interrupt() is True
        assert control.cancelling is True
        with pytest.raises(asyncio.CancelledError):
            await task
        return control

    control = asyncio.run(scenario())

    assert control.active is False
    assert control.cancelling is False
    assert control.cancelled is True


def test_async_owner_scope_rejects_entry_without_an_active_task():
    """Async cancellation ownership requires an asyncio task rather than a loop callback."""

    async def scenario():
        """Enter an async owner scope from a callback running without a Task."""
        control = RunControl()
        finished = asyncio.Event()
        failures = []

        def enter_scope():
            """Capture the missing-task validation result from the loop callback."""
            try:
                with control.execution(async_owner=True):
                    pytest.fail("An event-loop callback must not own an async run.")
            except RuntimeError as error:
                failures.append(str(error))
            finally:
                finished.set()

        asyncio.get_running_loop().call_soon(enter_scope)
        await finished.wait()
        return failures

    assert asyncio.run(scenario()) == ["Async execution ownership requires an asyncio task."]


def test_wait_until_active_async_wakes_when_a_run_starts_or_shutdown_is_requested():
    """Waiters observe either active execution or shutdown without polling."""

    async def wait_for_state(control, started):
        """Register one state waiter before the test changes run control."""
        started.set()
        return await control.wait_until_active_async()

    async def scenario():
        """Exercise activation, shutdown, and immediate lifecycle states."""
        control = RunControl()
        started = asyncio.Event()
        waiter = asyncio.create_task(wait_for_state(control, started))
        await started.wait()
        with control.execution(async_owner=True):
            assert await waiter is True

        stopped = RunControl()
        shutdown_started = asyncio.Event()
        shutdown_waiter = asyncio.create_task(wait_for_state(stopped, shutdown_started))
        await shutdown_started.wait()
        stopped.stop()
        assert await shutdown_waiter is False
        stopped.stop()
        return stopped, control

    stopped, active = asyncio.run(scenario())

    async def check_immediate_states():
        """Read current activity and shutdown without registering a waiter."""
        with active.execution(async_owner=True):
            assert await active.wait_until_active_async() is True
        assert await stopped.wait_until_active_async() is False

    asyncio.run(check_immediate_states())


def test_wait_until_active_async_cleans_up_when_cancelled():
    """Cancelling an activity waiter releases its registration for later runs."""

    async def scenario():
        """Cancel a registered waiter, then verify the next run still wakes observers."""
        control = RunControl()
        registered = asyncio.Event()

        async def wait_for_activity():
            """Signal that waiting begins before awaiting run activity."""
            registered.set()
            return await control.wait_until_active_async()

        waiter = asyncio.create_task(wait_for_activity())
        await registered.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        with control.execution(async_owner=True):
            assert await control.wait_until_active_async() is True

    asyncio.run(scenario())


def test_async_cancel_schedule_failure_keeps_scoped_interruption():
    """A closing loop cannot erase the cooperative signal used by safe-boundary checks."""
    control = RunControl()
    loop = Mock()
    loop.call_soon_threadsafe.side_effect = RuntimeError("closed")

    with control.execution(), pytest.raises(RunInterrupted):
        control._async_loop = loop
        control._async_task = Mock()
        assert control.interrupt() is True
        control.checkpoint()

    loop.call_soon_threadsafe.assert_called_once()
    assert control.cancelled is True


def test_interruption_wakes_retry_waits_and_failure_releases_the_scope(monkeypatch):
    """Retry waits observe run interruption without real delays or poisoning the next run."""
    wake_event, resume_event, cancellation_event = Mock(), Mock(), Mock()
    monkeypatch.setattr(
        control_module,
        "Event",
        Mock(side_effect=[wake_event, resume_event, cancellation_event]),
    )
    control = RunControl()
    with pytest.raises(RunInterrupted), control.execution():
        control.interrupt()
        control.wait(5)

    wake_event.set.assert_called_once_with()
    cancellation_event.set.assert_called_once_with()
    wake_event.wait.assert_called_once_with(5)
    wake_event.clear.assert_called_once_with()
    resume_event.set.assert_called()
    cancellation_event.clear.assert_called_once_with()
    assert control.active is False
    control.checkpoint()


def test_async_wait_keeps_the_event_loop_responsive_until_timeout(monkeypatch):
    """Async retry delays yield control to other tasks while the timer is pending."""

    async def scenario():
        """Let a heartbeat run while a retry timer expires."""
        control = RunControl()
        heartbeat = asyncio.Event()

        async def controlled_timeout(waiter, *, timeout):
            """Expire the retry timer only after the heartbeat proves loop progress."""
            assert timeout == 0.01
            try:
                await heartbeat.wait()
            finally:
                waiter.close()
            raise TimeoutError

        monkeypatch.setattr(
            control_module,
            "asyncio",
            SimpleNamespace(**{**vars(asyncio), "wait_for": controlled_timeout}),
        )

        async def signal():
            """Mark that another task ran during the retry delay."""
            await deliver_callbacks()
            heartbeat.set()

        await asyncio.gather(control.wait_async(0.01), signal())
        return heartbeat.is_set(), control._async_waiters

    heartbeat_ran, waiters = asyncio.run(scenario())

    assert heartbeat_ran
    assert waiters == {}


def test_async_wait_is_woken_by_cross_thread_run_interruption():
    """A UI-thread interruption wakes an async retry wait and preserves scoped semantics."""

    async def scenario():
        """Wait for a cross-thread cancellation signal and capture its exception type."""
        control = RunControl()

        async def wait_for_interruption():
            """Convert the expected interruption into a normal task result."""
            try:
                await control.wait_async(5)
            except RunInterrupted as error:
                return type(error)
            return None

        with control.execution():
            waiting = asyncio.create_task(wait_for_interruption())
            await deliver_callbacks()
            interrupt = Thread(target=control.interrupt)
            interrupt.start()
            result = await waiting
            interrupt.join(timeout=5)
            assert not interrupt.is_alive()
        return result, control._async_waiters, control.active

    result, waiters, active = asyncio.run(scenario())

    assert result is RunInterrupted
    assert waiters == {}
    assert active is False


def test_async_wait_is_woken_by_sticky_shutdown():
    """Shutdown wakes an async waiter and remains a shutdown rather than an interrupt."""

    async def scenario():
        """Wait for shutdown and capture its exception type for the assertion."""
        control = RunControl()

        async def wait_for_shutdown():
            """Convert the expected shutdown exception into a normal task result."""
            try:
                await control.wait_async(5)
            except KeyboardInterrupt as error:
                return type(error)
            return None

        waiting = asyncio.create_task(wait_for_shutdown())
        await deliver_callbacks()
        control.stop()
        result = await waiting
        return result, control._async_waiters, control.stopped

    result, waiters, stopped = asyncio.run(scenario())

    assert result is KeyboardInterrupt
    assert waiters == {}
    assert stopped is True


def test_async_wait_checks_shutdown_set_before_registration():
    """A pre-existing shutdown is observed without waiting for the supplied timeout."""

    async def scenario():
        """Register after shutdown and capture the existing cancellation signal."""
        control = RunControl()
        control.stop()

        async def wait_for_shutdown():
            """Convert the expected shutdown exception into a normal task result."""
            try:
                await control.wait_async(5)
            except KeyboardInterrupt as error:
                return type(error)
            return None

        result = await wait_for_shutdown()
        return result, control._async_waiters

    result, waiters = asyncio.run(scenario())

    assert result is KeyboardInterrupt
    assert waiters == {}


def test_async_wait_checks_scoped_interrupt_set_before_registration():
    """An already-requested run interrupt is observed before starting a delay."""

    async def scenario():
        """Enter an active run, request interruption, and register the async wait."""
        control = RunControl()
        with control.execution():
            assert control.interrupt()
            try:
                await control.wait_async(5)
            except RunInterrupted as error:
                return type(error), control._async_waiters
        return None, control._async_waiters

    result, waiters = asyncio.run(scenario())

    assert result is RunInterrupted
    assert waiters == {}


def test_async_wait_discards_waiter_for_a_closed_event_loop():
    """A loop closing during shutdown cannot leave a stale cancellation registration."""
    control = RunControl()
    wake = asyncio.Event()
    loop = Mock()
    loop.call_soon_threadsafe.side_effect = RuntimeError("closed")
    control._async_waiters[wake] = loop

    control.stop()

    loop.call_soon_threadsafe.assert_called_once_with(wake.set)
    assert control._async_waiters == {}


def test_async_wait_cancellation_unregisters_its_wakeup():
    """Cancelling an async delay removes its event-loop registration promptly."""

    async def scenario():
        """Cancel one pending retry wait and return its registration state."""
        control = RunControl()
        waiting = asyncio.create_task(control.wait_async(5))
        await deliver_callbacks()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        return control._async_waiters

    assert asyncio.run(scenario()) == {}
