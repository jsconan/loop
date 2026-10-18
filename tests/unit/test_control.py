"""Verify cooperative cancellation without real retry delays."""

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
