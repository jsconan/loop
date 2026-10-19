"""Verify cooperative cancellation without real retry delays."""

import asyncio
from threading import Thread
from unittest.mock import Mock

import pytest

from loop import RunControl, RunInterrupted
from loop import control as control_module


def test_control_stays_stopped_after_a_stop_request():
    """A stop signal is sticky and every later checkpoint interrupts execution."""
    control = RunControl()
    assert control.stopped is False
    control.checkpoint()

    control.stop()
    control.stop()

    assert control.stopped is True
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
        assert control.stopped is False
        with pytest.raises(RunInterrupted):
            control.checkpoint()
        with pytest.raises(RuntimeError, match="already owns"), control.execution():
            pytest.fail("Overlapping executions must be refused")

    assert control.active is False
    assert control.interrupted is False
    with control.execution():
        control.checkpoint()


def test_shutdown_supersedes_interruption_and_survives_execution_cleanup():
    """Settling an interrupted scope cannot clear a concurrent sticky shutdown request."""
    control = RunControl()
    with control.execution(), pytest.raises(KeyboardInterrupt) as raised:
        control.interrupt()
        control.stop()
        assert control.interrupt() is False
        control.checkpoint()

    assert type(raised.value) is KeyboardInterrupt
    assert control.active is False
    assert control.interrupted is False
    assert control.stopped is True
    assert control.interrupt() is False


def test_interruption_wakes_retry_waits_and_failure_releases_the_scope(monkeypatch):
    """Retry waits observe run interruption without real delays or poisoning the next run."""
    event = Mock()
    monkeypatch.setattr(control_module, "Event", Mock(return_value=event))
    control = RunControl()
    with pytest.raises(RunInterrupted), control.execution():
        control.interrupt()
        control.wait(5)

    event.set.assert_called_once_with()
    event.wait.assert_called_once_with(5)
    event.clear.assert_called_once_with()
    assert control.active is False
    control.checkpoint()


def test_async_wait_keeps_the_event_loop_responsive_until_timeout():
    """Async retry delays yield control to other tasks while the timer is pending."""

    async def scenario():
        """Let a heartbeat run while a retry timer expires."""
        control = RunControl()
        heartbeat = asyncio.Event()

        async def signal():
            """Mark that another task ran during the retry delay."""
            await asyncio.sleep(0)
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
            await asyncio.sleep(0)
            interrupt = Thread(target=control.interrupt)
            interrupt.start()
            result = await waiting
            interrupt.join()
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
        await asyncio.sleep(0)
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
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        return control._async_waiters

    assert asyncio.run(scenario()) == {}
