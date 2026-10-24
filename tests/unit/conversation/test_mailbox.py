"""Verify bounded priority ordering and shutdown for conversation inputs."""

from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full
from threading import Barrier, Condition, Event, Thread

import pytest

from loop.conversation import mailbox as mailbox_module
from loop.conversation.mailbox import InputMailbox
from loop.conversation.models import InputAdmission, InputAdmissionKind


def test_mailbox_orders_inputs_and_enforces_capacity():
    """Return inputs by priority and arrival order within a fixed capacity."""
    mailbox = InputMailbox(2)
    later = InputAdmission(1, 1, InputAdmissionKind.MESSAGE, "later")
    urgent = InputAdmission(0, 2, InputAdmissionKind.MESSAGE, "urgent")

    assert mailbox.capacity == 2
    assert mailbox.try_put(later) is True
    assert mailbox.try_put(urgent) is True
    assert mailbox.snapshot() == (urgent, later)
    assert mailbox.size() == 2
    assert mailbox.is_empty() is False
    with pytest.raises(Full):
        mailbox.try_put(InputAdmission(1, 3, InputAdmissionKind.MESSAGE, "overflow"))
    assert mailbox.get_nowait() == urgent
    assert mailbox.get_nowait() == later
    assert mailbox.is_empty() is True
    with pytest.raises(Empty):
        mailbox.get_nowait()


@pytest.mark.parametrize("commands_only", [False, True])
def test_mailbox_reserves_bounded_command_capacity_and_parks_messages(commands_only):
    """Recovery commands remain deliverable through a full ordinary input mailbox."""
    mailbox = InputMailbox(1)
    message = InputAdmission(0, 0, InputAdmissionKind.MESSAGE, "urgent")
    command = InputAdmission(1, 0, InputAdmissionKind.COMMAND, "/continue")
    mailbox.try_put(message)
    assert mailbox.try_put(command)
    with pytest.raises(Full):
        mailbox.try_put(InputAdmission(1, 1, InputAdmissionKind.COMMAND, "/retry"))
    with pytest.raises(Full):
        mailbox.try_put(InputAdmission(1, 2, InputAdmissionKind.MESSAGE, "overflow"))
    assert mailbox.get_nowait(commands_only=commands_only) is (
        command if commands_only else message
    )
    if commands_only:
        with pytest.raises(Empty):
            mailbox.get_nowait(commands_only=True)
    assert mailbox.get_nowait() is (message if commands_only else command)


def test_mailbox_wait_parks_existing_messages_until_a_command_arrives(monkeypatch):
    """A waiter ignores parked messages and wakes on a newly admitted command."""
    mailbox = InputMailbox(1)
    message = InputAdmission(0, 0, InputAdmissionKind.MESSAGE, "parked")
    command = InputAdmission(1, 1, InputAdmissionKind.COMMAND, "/continue")
    mailbox.try_put(message)
    original_wait = mailbox_module.Condition.wait

    def admit_command(condition, timeout=None):
        """Admit a command at the condition boundary without waiting for elapsed time."""
        mailbox.try_put(command)
        return True

    monkeypatch.setattr(mailbox_module.Condition, "wait", admit_command)
    assert mailbox.wait_until_not_empty(commands_only=True)
    monkeypatch.setattr(mailbox_module.Condition, "wait", original_wait)
    assert mailbox.get_nowait(commands_only=True) is command
    assert mailbox.get_nowait() is message


def test_mailbox_validates_positive_capacity():
    """Reject a mailbox capacity that cannot hold an input."""
    with pytest.raises(ValueError, match="capacity must be positive"):
        InputMailbox(0)


def test_mailbox_reprioritizes_only_the_target_object():
    """Avoid replacing an equal admission that is a different object."""
    mailbox = InputMailbox(2)
    target = InputAdmission(1, 1, InputAdmissionKind.MESSAGE, "target")
    equal_admission = InputAdmission(1, 1, InputAdmissionKind.STOP)
    mailbox.try_put(target)

    assert target == equal_admission
    assert mailbox.reprioritize(equal_admission, 0) is False
    assert mailbox.reprioritize(target, 0) is True
    assert mailbox.get_nowait().priority == 0


def test_mailbox_removes_only_the_target_object():
    """Recall removes the exact selected admission and preserves equal messages."""
    mailbox = InputMailbox(2)
    target = InputAdmission(1, 1, InputAdmissionKind.MESSAGE, "target")
    equal_admission = InputAdmission(1, 1, InputAdmissionKind.STOP)
    mailbox.try_put(target)
    mailbox.try_put(InputAdmission(1, 2, InputAdmissionKind.MESSAGE, "remaining"))

    assert target == equal_admission
    assert mailbox.remove(equal_admission) is False
    assert mailbox.remove(target) is True
    assert mailbox.snapshot() == (InputAdmission(1, 2, InputAdmissionKind.MESSAGE, "remaining"),)


@pytest.mark.parametrize("shutdown", [False, True])
def test_mailbox_wait_wakes_for_admission_or_shutdown(monkeypatch, shutdown):
    """Wake a blocked waiter when input arrives or shutdown replaces pending input."""
    waiting = Event()

    class TrackingCondition(Condition):
        """Signal when the mailbox begins waiting for a notification."""

        def wait(self, timeout=None):
            """Expose the condition wait boundary to coordinate the test."""
            waiting.set()
            return super().wait(timeout)

    monkeypatch.setattr(mailbox_module, "Condition", TrackingCondition)
    mailbox = InputMailbox(1)
    result = []

    def wait_for_input():
        """Collect the public wait result from a worker thread."""
        result.append(mailbox.wait_until_not_empty())

    worker = Thread(target=wait_for_input)
    worker.start()
    assert waiting.wait(timeout=1)
    if shutdown:
        expected = InputAdmission(1, 0, InputAdmissionKind.STOP)
        assert mailbox.stop_and_replace(expected) == 0
    else:
        expected = InputAdmission(1, 0, InputAdmissionKind.MESSAGE, "arrived")
        assert mailbox.try_put(expected) is True
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert result == [True]
    assert mailbox.get_nowait() == expected


def test_mailbox_stop_discards_pending_inputs_and_closes_admission():
    """Replace pending inputs with one stop signal and reject later admissions."""
    mailbox = InputMailbox(2)
    mailbox.try_put(InputAdmission(1, 0, InputAdmissionKind.MESSAGE, "first"))
    mailbox.try_put(InputAdmission(1, 1, InputAdmissionKind.MESSAGE, "second"))
    stop = InputAdmission(1, 2, InputAdmissionKind.STOP)

    assert mailbox.stop_and_replace(stop) == 2
    assert mailbox.snapshot() == (stop,)
    assert mailbox.stop_and_replace(stop) == 0
    assert mailbox.try_put(InputAdmission(1, 3, InputAdmissionKind.MESSAGE, "late")) is False
    assert mailbox.get_nowait() == stop
    assert mailbox.wait_until_not_empty() is False


def test_mailbox_serializes_admission_against_shutdown():
    """Linearize a concurrent admission and shutdown without retaining stale input."""
    mailbox = InputMailbox(1)
    admission = InputAdmission(1, 0, InputAdmissionKind.MESSAGE, "racing")
    stop = InputAdmission(1, 1, InputAdmissionKind.STOP)
    start = Barrier(3)

    def admit():
        """Attempt one admission at the shared synchronization boundary."""
        start.wait(timeout=1)
        return mailbox.try_put(admission)

    def shut_down():
        """Close the mailbox at the shared synchronization boundary."""
        start.wait(timeout=1)
        return mailbox.stop_and_replace(stop)

    with ThreadPoolExecutor(max_workers=2) as workers:
        admission_result = workers.submit(admit)
        shutdown_result = workers.submit(shut_down)
        start.wait(timeout=1)
        accepted = admission_result.result(timeout=1)
        discarded = shutdown_result.result(timeout=1)

    assert mailbox.snapshot() == (stop,)
    assert (accepted, discarded) in {(True, 1), (False, 0)}
    assert mailbox.try_put(InputAdmission(1, 2, InputAdmissionKind.MESSAGE, "late")) is False
