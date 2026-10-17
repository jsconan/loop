"""Verify cooperative cancellation without real retry delays."""

from unittest.mock import Mock

import pytest

from loop import RunControl
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
    event.is_set.return_value = stopped
    monkeypatch.setattr(control_module, "Event", Mock(return_value=event))
    control = RunControl()

    if stopped:
        with pytest.raises(KeyboardInterrupt):
            control.wait(2.5)
    else:
        control.wait(2.5)

    event.wait.assert_called_once_with(2.5)
