"""Verify admitted namespace supervision and authoritative cleanup refusal."""

import io
import select
import socket
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import pytest

from loop.execution.sandbox.linux import supervision as linux_supervision
from loop.execution.sandbox.linux.constants import NATIVE_CANCEL_FRAME
from loop.utils.process import ProcessCapture, ProcessCaptureStatus


@pytest.fixture
def control_connection():
    """Provide the admitted socket without opening a native endpoint."""
    return Mock(fileno=Mock(return_value=0))


class WatcherExit(BaseException):
    """Record a receiver exit without terminating the unit test process."""


@pytest.mark.parametrize(
    ("events", "status"),
    [
        ([(0, select.POLLHUP)], 0),
        ([(0, select.POLLERR)], 1),
        ([(0, select.POLLNVAL)], 1),
        ([(0, 0)], 1),
        ([(0, select.POLLPRI)], 1),
        ([(0, select.POLLIN | select.POLLPRI)], 1),
        ([(1, select.POLLHUP)], 1),
        ([(0, select.POLLHUP), (0, select.POLLHUP)], 1),
    ],
)
def test_disconnect_guard_exits_receiver_on_hangup_or_ambiguous_event(
    monkeypatch, events, status, control_connection
):
    """A disconnected or ambiguous client ends the systemd-owned receiver."""
    poller = Mock()
    poller.poll.return_value = events
    monkeypatch.setattr(linux_supervision.select, "poll", Mock(return_value=poller))
    thread = Mock()
    thread.is_alive.return_value = False
    factory = Mock(return_value=thread)
    monkeypatch.setattr(linux_supervision.threading, "Thread", factory)
    exit_receiver = Mock(side_effect=WatcherExit())
    monkeypatch.setattr(linux_supervision.os, "_exit", exit_receiver)

    with linux_supervision.guard_client_disconnect(control_connection), pytest.raises(WatcherExit):
        target = factory.call_args.kwargs["target"]
        target()

    poller.register.assert_called_once_with(
        0,
        select.POLLHUP
        | select.POLLERR
        | select.POLLNVAL
        | select.POLLIN
        | getattr(select, "POLLRDHUP", 0x2000),
    )
    exit_receiver.assert_called_once_with(status)
    thread.join.assert_called_once_with(1)


@pytest.mark.parametrize(
    "error", [OSError("poll failed"), ValueError("bad socket"), RuntimeError("watcher failed")]
)
def test_disconnect_guard_exits_on_watcher_failure(monkeypatch, error, control_connection):
    """Losing the disconnect observer cannot leave the receiver running unguarded."""
    poller = Mock()
    poller.poll.side_effect = error
    monkeypatch.setattr(linux_supervision.select, "poll", Mock(return_value=poller))
    thread = Mock()
    thread.is_alive.return_value = False
    factory = Mock(return_value=thread)
    monkeypatch.setattr(linux_supervision.threading, "Thread", factory)
    exit_receiver = Mock(side_effect=WatcherExit())
    monkeypatch.setattr(linux_supervision.os, "_exit", exit_receiver)

    with linux_supervision.guard_client_disconnect(control_connection), pytest.raises(WatcherExit):
        factory.call_args.kwargs["target"]()
    exit_receiver.assert_called_once_with(1)


def test_disconnect_guard_stops_without_reading_control(monkeypatch, control_connection):
    """Closing the guard leaves queued control bytes for the ordinary supervisor."""
    poller = Mock()
    monkeypatch.setattr(linux_supervision.select, "poll", Mock(return_value=poller))
    thread = Mock()
    thread.is_alive.return_value = False
    factory = Mock(return_value=thread)
    monkeypatch.setattr(linux_supervision.threading, "Thread", factory)
    read = Mock()
    monkeypatch.setattr(linux_supervision.os, "read", read)
    exit_receiver = Mock()
    monkeypatch.setattr(linux_supervision.os, "_exit", exit_receiver)

    with linux_supervision.guard_client_disconnect(control_connection):
        target = factory.call_args.kwargs["target"]
    target()

    read.assert_not_called()
    exit_receiver.assert_not_called()
    poller.poll.assert_not_called()


def test_disconnect_guard_continues_after_an_idle_poll(monkeypatch, control_connection):
    """An idle poll does not stop the watcher before a later client disconnect."""
    poller = Mock()
    poller.poll.side_effect = [[], [(0, select.POLLHUP)]]
    monkeypatch.setattr(linux_supervision.select, "poll", Mock(return_value=poller))
    thread = Mock()
    thread.is_alive.return_value = False
    factory = Mock(return_value=thread)
    monkeypatch.setattr(linux_supervision.threading, "Thread", factory)
    exit_receiver = Mock(side_effect=WatcherExit())
    monkeypatch.setattr(linux_supervision.os, "_exit", exit_receiver)

    with linux_supervision.guard_client_disconnect(control_connection), pytest.raises(WatcherExit):
        factory.call_args.kwargs["target"]()

    assert poller.poll.call_count == 2
    exit_receiver.assert_called_once_with(0)


def test_disconnect_guard_bounds_stalled_spawn_with_queued_control(monkeypatch, control_connection):
    """A still-blocked spawn cannot retain a connected client's queued control forever."""
    poller = Mock()
    poller.poll.side_effect = [[(0, select.POLLIN)], []]
    monkeypatch.setattr(linux_supervision.select, "poll", Mock(return_value=poller))
    monkeypatch.setattr(linux_supervision.time, "monotonic", Mock(side_effect=[10.0, 11.0]))
    thread = Mock()
    thread.is_alive.return_value = False
    factory = Mock(return_value=thread)
    monkeypatch.setattr(linux_supervision.threading, "Thread", factory)
    exit_receiver = Mock(side_effect=WatcherExit())
    monkeypatch.setattr(linux_supervision.os, "_exit", exit_receiver)

    with linux_supervision.guard_client_disconnect(control_connection), pytest.raises(WatcherExit):
        factory.call_args.kwargs["target"]()

    poller.modify.assert_called_once_with(0, select.POLLHUP | select.POLLERR | select.POLLNVAL)
    exit_receiver.assert_called_once_with(0)


def test_disconnect_guard_hands_queued_control_to_returned_spawn(monkeypatch, control_connection):
    """A returning spawn keeps its queued control for the ordinary supervisor."""
    poller = Mock()
    monkeypatch.setattr(linux_supervision.select, "poll", Mock(return_value=poller))
    clock = Mock(return_value=10.0)
    monkeypatch.setattr(linux_supervision.time, "monotonic", clock)
    thread = Mock()
    thread.is_alive.return_value = False
    factory = Mock(return_value=thread)
    monkeypatch.setattr(linux_supervision.threading, "Thread", factory)
    exit_receiver = Mock(side_effect=WatcherExit())
    monkeypatch.setattr(linux_supervision.os, "_exit", exit_receiver)

    with linux_supervision.guard_client_disconnect(control_connection) as returned:

        def poll(_timeout):
            """Deliver control, then let spawn return before its grace expires."""
            if poller.poll.call_count == 1:
                return [(0, select.POLLIN)]
            if poller.poll.call_count == 2:
                returned.set()
                clock.return_value = 20.0
                return []
            return [(0, select.POLLHUP)]

        poller.poll.side_effect = poll
        with pytest.raises(WatcherExit):
            factory.call_args.kwargs["target"]()

    assert poller.poll.call_count == 3
    exit_receiver.assert_called_once_with(0)


def test_disconnect_guard_leaves_control_after_spawn_handoff(monkeypatch, control_connection):
    """Input arriving after spawn returns cannot trigger the stalled-spawn abort."""
    poller = Mock()
    poller.poll.side_effect = [[(0, select.POLLIN)], [(0, select.POLLHUP)]]
    monkeypatch.setattr(linux_supervision.select, "poll", Mock(return_value=poller))
    clock = Mock()
    monkeypatch.setattr(linux_supervision.time, "monotonic", clock)
    thread = Mock()
    thread.is_alive.return_value = False
    factory = Mock(return_value=thread)
    monkeypatch.setattr(linux_supervision.threading, "Thread", factory)
    exit_receiver = Mock(side_effect=WatcherExit())
    monkeypatch.setattr(linux_supervision.os, "_exit", exit_receiver)

    with linux_supervision.guard_client_disconnect(control_connection) as returned:
        returned.set()
        with pytest.raises(WatcherExit):
            factory.call_args.kwargs["target"]()

    clock.assert_not_called()
    exit_receiver.assert_called_once_with(0)


@pytest.mark.parametrize("with_data", [False, True])
@pytest.mark.parametrize("spawn_returned", [False, True])
def test_disconnect_guard_distinguishes_half_close_from_queued_control(
    monkeypatch, with_data, spawn_returned, control_connection
):
    """An empty half-close is benign; queued bytes retain the stalled-spawn grace."""
    half_close = getattr(select, "POLLRDHUP", 0x2000)
    poller = Mock()
    poller.poll.side_effect = [
        [(0, half_close | (select.POLLIN if with_data else 0))],
        [(0, select.POLLHUP)],
    ]
    monkeypatch.setattr(linux_supervision.select, "poll", Mock(return_value=poller))
    control_connection.recv.return_value = b"x" if with_data else b""
    clock = Mock(return_value=10)
    monkeypatch.setattr(linux_supervision.time, "monotonic", clock)
    thread = Mock()
    thread.is_alive.return_value = False
    factory = Mock(return_value=thread)
    monkeypatch.setattr(linux_supervision.threading, "Thread", factory)
    exit_receiver = Mock(side_effect=WatcherExit())
    monkeypatch.setattr(linux_supervision.os, "_exit", exit_receiver)

    with linux_supervision.guard_client_disconnect(control_connection) as returned:
        if spawn_returned:
            returned.set()
        with pytest.raises(WatcherExit):
            factory.call_args.kwargs["target"]()

    control_connection.recv.assert_called_once_with(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
    assert clock.called == (with_data and not spawn_returned)
    exit_receiver.assert_called_once_with(0)


def test_disconnect_guard_exits_if_half_close_cannot_be_observed(monkeypatch, control_connection):
    """A failed socket peek cannot leave a stalled spawn without its control observer."""
    poller = Mock()
    poller.poll.return_value = [(0, getattr(select, "POLLRDHUP", 0x2000))]
    monkeypatch.setattr(linux_supervision.select, "poll", Mock(return_value=poller))
    control_connection.recv.side_effect = OSError("peek failed")
    thread = Mock()
    thread.is_alive.return_value = False
    factory = Mock(return_value=thread)
    monkeypatch.setattr(linux_supervision.threading, "Thread", factory)
    exit_receiver = Mock(side_effect=WatcherExit())
    monkeypatch.setattr(linux_supervision.os, "_exit", exit_receiver)

    with linux_supervision.guard_client_disconnect(control_connection), pytest.raises(WatcherExit):
        factory.call_args.kwargs["target"]()
    exit_receiver.assert_called_once_with(1)


def test_disconnect_guard_refuses_unsettled_watcher(monkeypatch, control_connection):
    """An unresponsive watcher is not silently abandoned after the guarded operation."""
    poller = Mock()
    monkeypatch.setattr(linux_supervision.select, "poll", Mock(return_value=poller))
    thread = Mock()
    thread.is_alive.return_value = True
    monkeypatch.setattr(linux_supervision.threading, "Thread", Mock(return_value=thread))
    with (
        pytest.raises(OSError, match="disconnect watcher did not stop"),
        linux_supervision.guard_client_disconnect(control_connection),
    ):
        pass


def test_disconnect_guard_refuses_start_failure(monkeypatch, control_connection):
    """A missing watcher prevents the guarded operation from beginning."""
    poller = Mock()
    monkeypatch.setattr(linux_supervision.select, "poll", Mock(return_value=poller))
    thread = Mock()
    thread.start.side_effect = RuntimeError("thread unavailable")
    monkeypatch.setattr(linux_supervision.threading, "Thread", Mock(return_value=thread))
    with (
        pytest.raises(RuntimeError, match="thread unavailable"),
        linux_supervision.guard_client_disconnect(control_connection),
    ):
        pytest.fail("An unguarded operation started")


@pytest.fixture
def scope(monkeypatch, tmp_path):
    """Provide controlled kernel-domain membership without launching native processes."""
    group = tmp_path / "scope"
    group.mkdir()
    (group / "cgroup.procs").write_bytes(b"42\n")
    state = SimpleNamespace(
        group=group,
        config=object(),
        process=Mock(),
        capture=ProcessCapture(ProcessCaptureStatus.COMPLETED, exit_code=17, stdout="output"),
        clock=Mock(return_value=10),
        sleep=Mock(),
        membership=Mock(return_value=group),
    )
    state.supervise = Mock(side_effect=lambda *args, **kwargs: state.capture)
    monkeypatch.setattr(linux_supervision, "admission_group", state.membership)
    monkeypatch.setattr(linux_supervision, "supervise_process", state.supervise)
    monkeypatch.setattr(linux_supervision.os, "getpid", lambda: 42)
    monkeypatch.setattr(linux_supervision.time, "monotonic", state.clock)
    monkeypatch.setattr(linux_supervision.time, "sleep", state.sleep)
    return state


@pytest.mark.parametrize("status", list(ProcessCaptureStatus))
def test_supervise_payload_preserves_typed_capture_after_kernel_cleanup(scope, status):
    """Every common outcome returns only after the admitted domain contains the receiver alone."""
    scope.capture = ProcessCapture(status, stdout="bounded", stderr_discarded=9)
    assert linux_supervision.supervise_payload(scope.process, scope.config, 50) is scope.capture
    scope.supervise.assert_called_once_with(
        scope.process,
        50,
        stdout_limit=1_000_080,
        cancel_event=ANY,
        preserve_newlines=True,
    )
    assert scope.membership.call_count == 2
    scope.sleep.assert_not_called()


@pytest.mark.parametrize("error", [ValueError("unsafe membership"), OSError("lost kernel view")])
def test_supervise_payload_cleans_owned_launcher_on_initial_refusal(scope, error):
    """Failure before supervision still terminates and drains the owned launcher."""
    scope.membership.side_effect = error
    with pytest.raises(type(error), match=str(error)):
        linux_supervision.supervise_payload(scope.process, scope.config, 50)
    scope.supervise.assert_called_once_with(scope.process, 10)


def test_supervise_payload_propagates_native_supervisor_failure(scope):
    """A native supervision failure cannot return a successful result."""
    scope.supervise.side_effect = OSError("supervision failed")
    with pytest.raises(OSError, match="supervision failed"):
        linux_supervision.supervise_payload(scope.process, scope.config, 50)


@pytest.mark.parametrize("changed", ["path", "inode"])
def test_supervise_payload_refuses_replaced_cleanup_scope(scope, tmp_path, changed):
    """The original protected service domain cannot be replaced by another pathname or inode."""
    if changed == "path":
        other = tmp_path / "other"
        other.mkdir()
        scope.membership.side_effect = [scope.group, other]
    else:

        def replace(*args, **kwargs):
            """Model replacement of a retired domain by a new kernel domain."""
            scope.group.rename(tmp_path / "old")
            scope.group.mkdir()
            return scope.capture

        scope.supervise.side_effect = replace
    with pytest.raises(ValueError, match="scope changed"):
        linux_supervision.supervise_payload(scope.process, scope.config, 50)


def test_supervise_payload_refuses_nested_domains(scope):
    """An unexpected subgroup prevents a partial direct-membership proof."""
    (scope.group / "child").mkdir()
    with pytest.raises(ValueError, match="unexpected subgroups"):
        linux_supervision.supervise_payload(scope.process, scope.config, 50)


@pytest.mark.parametrize("members", [b"", b"zero\n", b"0\n", b"-1\n", b"1\n" * 32769])
def test_supervise_payload_refuses_invalid_or_unbounded_kernel_membership(scope, members):
    """Empty, malformed and oversized domain membership never proves descendant absence."""
    (scope.group / "cgroup.procs").write_bytes(members)
    with pytest.raises(ValueError, match="membership is malformed"):
        linux_supervision.supervise_payload(scope.process, scope.config, 50)


def test_supervise_payload_waits_boundedly_for_namespace_teardown(scope):
    """A detached descendant may disappear during the bounded kernel cleanup interval."""
    (scope.group / "cgroup.procs").write_bytes(b"42\n99\n")
    scope.sleep.side_effect = lambda seconds: (scope.group / "cgroup.procs").write_bytes(b"42\n")
    assert linux_supervision.supervise_payload(scope.process, scope.config, 50) is scope.capture
    scope.sleep.assert_called_once_with(0.01)
    assert scope.membership.call_count == 3


def test_supervise_payload_refuses_live_detached_descendants(scope):
    """Expired cleanup retains confinement and never reports a successful command."""
    (scope.group / "cgroup.procs").write_bytes(b"42\n99\n")
    scope.clock.side_effect = [10, 11]
    with pytest.raises(TimeoutError, match="descendants remain; confinement and policy retained"):
        linux_supervision.supervise_payload(scope.process, scope.config, 50)
    scope.sleep.assert_not_called()


def test_supervise_payload_propagates_membership_read_failure(scope, monkeypatch):
    """Inaccessible kernel membership cannot be replaced by a process-group success claim."""
    monkeypatch.setattr(Path, "open", Mock(side_effect=OSError("kernel unavailable")))
    with pytest.raises(OSError, match="kernel unavailable"):
        linux_supervision.supervise_payload(scope.process, scope.config, 50)


def test_supervise_payload_bounds_even_a_nonconforming_kernel_stream(scope, monkeypatch):
    """The kernel reader requests a finite maximum independently of stream metadata."""
    stream = Mock(wraps=io.BytesIO(b"42\n"))
    stream.__enter__ = Mock(return_value=stream)
    stream.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(Path, "open", Mock(return_value=stream))
    assert linux_supervision.supervise_payload(scope.process, scope.config, 50) is scope.capture
    stream.read.assert_called_once_with(65537)


@pytest.mark.parametrize("event", ["idle", "disconnect", "error", "value", "live", "empty"])
def test_supervise_payload_watches_full_disconnect_and_refuses_observation_loss(
    scope, monkeypatch, event
):
    """A full socket hangup cancels only the owned launcher; watcher failures refuse success."""
    target = []
    poller = Mock()
    poller.poll.return_value = [(0, 16)]
    if event in {"error", "value"}:
        poller.register.side_effect = (
            OSError("poll failed") if event == "error" else ValueError("poll failed")
        )
    monkeypatch.setattr(linux_supervision.select, "poll", lambda: poller)
    thread = Mock()
    thread.is_alive.return_value = event == "live"

    def create(*, target: object, daemon):
        """Retain the synchronous callback without creating a real worker thread."""
        globals_target.append(target)
        return thread

    globals_target = target
    monkeypatch.setattr(linux_supervision.threading, "Thread", create)
    if event == "disconnect":
        scope.supervise.side_effect = lambda *args, **kwargs: ProcessCapture(
            ProcessCaptureStatus.CANCELLED
            if kwargs["cancel_event"].is_set()
            else ProcessCaptureStatus.COMPLETED
        )
    if event != "idle":
        if event == "empty":
            stopped = Mock()
            stopped.is_set.side_effect = [False, True]
            events = [
                stopped,
                Mock(is_set=Mock(return_value=False)),
                Mock(is_set=Mock(return_value=False)),
            ]
            monkeypatch.setattr(linux_supervision.threading, "Event", Mock(side_effect=events))
            poller.poll.return_value = []
        thread.start.side_effect = lambda: target[0]()
    if event in {"error", "value", "live"}:
        with pytest.raises(OSError, match="cancellation"):
            linux_supervision.supervise_payload(scope.process, scope.config, 50, connection=0)
    else:
        result = linux_supervision.supervise_payload(scope.process, scope.config, 50, connection=0)
        assert result.status is (
            ProcessCaptureStatus.CANCELLED
            if event == "disconnect"
            else ProcessCaptureStatus.COMPLETED
        )
    thread.join.assert_called_once_with(1)
    assert scope.supervise.call_args.kwargs["cancel_event"].is_set() is (
        event in {"disconnect", "error", "value", "live"}
    )


def test_supervise_payload_cancels_on_bound_control_and_confirms_cleanup(scope, monkeypatch):
    """A valid control byte kills the owned process and waits for kernel cleanup evidence."""
    poller = Mock()
    poller.poll.return_value = [(0, select.POLLIN)]
    monkeypatch.setattr(linux_supervision.select, "poll", lambda: poller)
    monkeypatch.setattr(linux_supervision.os, "read", lambda *_: NATIVE_CANCEL_FRAME)

    def create(*, target: object, daemon):
        """Run the watcher deterministically in the fixture thread."""
        thread = Mock()
        thread.start.side_effect = target
        thread.is_alive.return_value = False
        return thread

    monkeypatch.setattr(linux_supervision.threading, "Thread", create)
    scope.supervise.side_effect = lambda *args, **kwargs: ProcessCapture(
        ProcessCaptureStatus.CANCELLED
        if kwargs["cancel_event"].is_set()
        else ProcessCaptureStatus.COMPLETED
    )

    result = linux_supervision.supervise_payload(scope.process, scope.config, 50, connection=0)

    assert result.status is ProcessCaptureStatus.CANCELLED
    assert scope.supervise.call_args.kwargs["cancel_event"].is_set()
    assert scope.membership.call_count == 2


@pytest.mark.parametrize("status", [ProcessCaptureStatus.COMPLETED, ProcessCaptureStatus.TIMED_OUT])
def test_supervise_payload_preserves_result_when_cancellation_arrives_after_capture(
    scope, monkeypatch, status
):
    """A control arriving after capture cannot replace the already determined outcome."""
    scope.capture = ProcessCapture(
        status, exit_code=17 if status is ProcessCaptureStatus.COMPLETED else None
    )
    thread = Mock()
    thread.is_alive.return_value = False
    thread.join.side_effect = lambda _: scope.supervise.call_args.kwargs["cancel_event"].set()
    monkeypatch.setattr(linux_supervision.threading, "Thread", Mock(return_value=thread))

    result = linux_supervision.supervise_payload(scope.process, scope.config, 50, connection=0)

    assert result is scope.capture
    assert scope.supervise.call_args.kwargs["cancel_event"].is_set()
    assert scope.membership.call_count == 2


def test_supervise_payload_treats_request_half_close_as_normal(scope, monkeypatch):
    """A request half-close disables control reads without cancelling the child."""
    poller = Mock()
    poller.poll.side_effect = [[(0, select.POLLIN)], []]
    monkeypatch.setattr(linux_supervision.select, "poll", lambda: poller)
    monkeypatch.setattr(linux_supervision.os, "read", lambda *_: b"")
    stopped = Mock()
    stopped.is_set.side_effect = [False, False, True]
    monkeypatch.setattr(
        linux_supervision.threading,
        "Event",
        Mock(
            side_effect=[
                stopped,
                Mock(is_set=Mock(return_value=False)),
                Mock(is_set=Mock(return_value=False)),
            ]
        ),
    )

    def create(*, target: object, daemon):
        """Run the watcher deterministically in the fixture thread."""
        thread = Mock()
        thread.start.side_effect = target
        thread.is_alive.return_value = False
        return thread

    monkeypatch.setattr(linux_supervision.threading, "Thread", create)

    result = linux_supervision.supervise_payload(scope.process, scope.config, 50, connection=0)

    assert result.status is ProcessCaptureStatus.COMPLETED
    assert poller.modify.call_args.args == (0, select.POLLHUP | select.POLLERR | select.POLLNVAL)
    assert scope.supervise.call_args.kwargs["cancel_event"].is_set() is False


def test_supervise_payload_rejects_unrecognized_control_after_cleanup(scope, monkeypatch):
    """Unexpected socket bytes cancel execution and cannot bypass descendant verification."""
    poller = Mock()
    poller.poll.return_value = [(0, select.POLLIN)]
    monkeypatch.setattr(linux_supervision.select, "poll", lambda: poller)
    monkeypatch.setattr(linux_supervision.os, "read", lambda *_: b"Q")

    def create(*, target: object, daemon):
        """Run the watcher deterministically in the fixture thread."""
        thread = Mock()
        thread.start.side_effect = target
        thread.is_alive.return_value = False
        return thread

    monkeypatch.setattr(linux_supervision.threading, "Thread", create)

    with pytest.raises(OSError, match="cancellation authority"):
        linux_supervision.supervise_payload(scope.process, scope.config, 50, connection=0)

    assert scope.supervise.call_args.kwargs["cancel_event"].is_set()
    assert scope.membership.call_count == 2


def test_supervise_payload_applies_bounded_terminal_frames(scope, monkeypatch):
    """Only the owned master receives complete input, resize, EOF, and interrupt controls."""
    parts = [b"I", b"\x00\x02", b"ab", b"Z", b"\x00\x25\x00\x5b", b"F", b"X"]
    poller = Mock()
    stopped = []
    original_event = threading.Event

    def event():
        """Capture the watcher's own stop event for deterministic completion."""
        value = original_event()
        stopped.append(value)
        return value

    def poll(_timeout):
        """Stop only after every complete receiver control has been consumed."""
        if parts:
            return [(0, select.POLLIN)]
        stopped[0].set()
        return []

    poller.poll.side_effect = poll
    monkeypatch.setattr(linux_supervision.threading, "Event", event)
    monkeypatch.setattr(linux_supervision.select, "poll", lambda: poller)
    monkeypatch.setattr(linux_supervision.select, "select", lambda *_: ([], [9], []))
    read = Mock(side_effect=lambda _fd, _size: parts.pop(0))
    write = Mock(side_effect=lambda _fd, data: len(data))
    resize = Mock()
    monkeypatch.setattr(linux_supervision.os, "read", read)
    monkeypatch.setattr(linux_supervision.os, "write", write)
    monkeypatch.setattr(linux_supervision.fcntl, "ioctl", resize)

    def create(*, target, daemon):
        """Run the watcher after installing the controlled poll sequence."""
        worker = Mock()
        worker.start.side_effect = target
        worker.is_alive.return_value = False
        return worker

    monkeypatch.setattr(linux_supervision.threading, "Thread", create)
    assert (
        linux_supervision.supervise_payload(
            scope.process, scope.config, 50, connection=0, terminal_master=9
        )
        is scope.capture
    )
    assert [call.args for call in write.call_args_list] == [
        (9, b"ab"),
        (9, b"\x04"),
        (9, b"\x03"),
    ]
    resize.assert_called_once_with(
        9, linux_supervision.termios.TIOCSWINSZ, b"%\x00[\x00\x00\x00\x00\x00"
    )
    assert callable(scope.supervise.call_args.kwargs["reader"])


@pytest.mark.parametrize(
    "parts",
    [
        [b"I", b"\x00\x00"],
        [b"I", b"\x10\x01"],
        [b"Z", b"\x00\x00\x00\x01"],
        [b"F"],
    ],
)
def test_supervise_payload_refuses_invalid_terminal_control(scope, monkeypatch, parts):
    """Malformed dimensions, input lengths, and unowned controls cancel the launcher."""
    poller = Mock()
    poller.poll.return_value = [(0, select.POLLIN)]
    monkeypatch.setattr(linux_supervision.select, "poll", lambda: poller)
    chunks = list(parts)
    monkeypatch.setattr(linux_supervision.os, "read", Mock(side_effect=lambda *_: chunks.pop(0)))

    def create(*, target, daemon):
        """Run the guarded parser without a native thread."""
        worker = Mock()
        worker.start.side_effect = target
        worker.is_alive.return_value = False
        return worker

    monkeypatch.setattr(linux_supervision.threading, "Thread", create)
    with pytest.raises(OSError, match="cancellation authority"):
        linux_supervision.supervise_payload(
            scope.process,
            scope.config,
            50,
            connection=0,
            **({"terminal_master": 9} if parts[0] != b"F" else {}),
        )


@pytest.mark.parametrize("mode", ["partial", "blocked", "zero", "timeout"])
def test_terminal_input_bounds_progress_and_backpressure(scope, monkeypatch, mode):
    """Partial writes resume; no progress or elapsed input budgets cancel native execution."""
    chunks = [b"I", b"\x00\x02", b"ab"]
    poller = Mock()
    stopped = []
    original_event = threading.Event

    def event():
        """Capture the watcher stop event without starting a native worker."""
        value = original_event()
        stopped.append(value)
        return value

    def poll(_timeout):
        """Present each frame fragment then settle the successful watcher."""
        if chunks:
            return [(0, select.POLLIN)]
        stopped[0].set()
        return []

    poller.poll.side_effect = poll
    monkeypatch.setattr(linux_supervision.threading, "Event", event)
    monkeypatch.setattr(linux_supervision.select, "poll", lambda: poller)
    monkeypatch.setattr(linux_supervision.os, "read", Mock(side_effect=lambda *_: chunks.pop(0)))
    ready = [[], [9]] if mode == "blocked" else [[9]]
    selected = Mock(side_effect=lambda *_: ([], ready.pop(0) if len(ready) > 1 else ready[0], []))
    monkeypatch.setattr(linux_supervision.select, "select", selected)
    write = Mock()
    if mode == "partial":
        write.side_effect = [BlockingIOError(), 1, 1]
    elif mode == "zero":
        write.return_value = 0
    else:
        write.side_effect = lambda _fd, data: len(data)
    monkeypatch.setattr(linux_supervision.os, "write", write)
    if mode == "timeout":
        scope.clock.side_effect = [10, 11.1, 10]

    def create(*, target, daemon):
        """Run the input parser deterministically in the fixture thread."""
        worker = Mock()
        worker.start.side_effect = target
        worker.is_alive.return_value = False
        return worker

    monkeypatch.setattr(linux_supervision.threading, "Thread", create)
    if mode in {"zero", "timeout"}:
        with pytest.raises(OSError, match="cancellation authority"):
            linux_supervision.supervise_payload(
                scope.process, scope.config, 50, connection=0, terminal_master=9
            )
    else:
        assert (
            linux_supervision.supervise_payload(
                scope.process, scope.config, 50, connection=0, terminal_master=9
            )
            is scope.capture
        )
        assert chunks == []
        assert write.call_count == (3 if mode == "partial" else 1)


def test_terminal_reader_applies_pty_eof_only_to_owned_stdout(scope, monkeypatch):
    """The receiver's reader preserves ordinary stderr errors while treating PTY EIO as EOF."""
    stdout, stderr = object(), object()
    scope.process.stdout = stdout
    scope.process.stderr = stderr
    observed = []

    def supervise(*_args, reader, **_kwargs):
        """Exercise both stream identities through the supplied native reader."""
        for stream in (stdout, stderr):
            reader(stream, [], maximum=10)
        return scope.capture

    scope.supervise.side_effect = supervise
    monkeypatch.setattr(
        linux_supervision,
        "read_bounded_stream",
        lambda stream, _chunks, **kwargs: observed.append((stream, kwargs["eof_errno"])),
    )
    assert (
        linux_supervision.supervise_payload(scope.process, scope.config, 50, terminal_master=9)
        is scope.capture
    )
    assert observed == [(stdout, linux_supervision.errno.EIO), (stderr, None)]


@pytest.mark.parametrize("control", [b"H", b"P", b"U", b"R"])
@pytest.mark.parametrize("allow", [True, False])
def test_supervise_payload_mediates_exact_control_frames(scope, monkeypatch, control, allow):
    """The trusted receiver confirms only exact heartbeat, barrier, and revision outcomes."""
    poller = Mock()
    poller.poll.return_value = [(0, select.POLLIN)]
    monkeypatch.setattr(linux_supervision.select, "poll", lambda: poller)
    revision = (8).to_bytes(8, "big")
    frames = [control + revision]
    if allow and control != b"R":
        frames.append(b"R" + (9).to_bytes(8, "big"))
    chunks = [part for frame in frames for part in (frame[:1], frame[1:])]
    monkeypatch.setattr(linux_supervision.os, "read", Mock(side_effect=chunks))
    written = Mock(return_value=10)
    monkeypatch.setattr(linux_supervision.os, "write", written)
    mediator = Mock()
    mediator.lease_expired.return_value = False
    mediator.heartbeat.return_value = allow
    mediator.pause.return_value = allow
    mediator.resume.return_value = allow
    mediator.accept_revision.return_value = allow

    def create(*, target, daemon):
        """Run the receiver control watcher deterministically with synthetic frames."""
        worker = Mock()
        worker.start.side_effect = target
        worker.is_alive.return_value = False
        return worker

    monkeypatch.setattr(linux_supervision.threading, "Thread", create)
    if allow:
        assert (
            linux_supervision.supervise_payload(
                scope.process,
                scope.config,
                50,
                connection=0,
                mediator=mediator,
                send_lock=threading.Lock(),
            )
            is scope.capture
        )
        assert written.call_count == (1 if control == b"R" else 2)
    else:
        with pytest.raises(OSError, match="cancellation authority"):
            linux_supervision.supervise_payload(
                scope.process, scope.config, 50, connection=0, mediator=mediator
            )
        if control == b"R":
            written.assert_called_once_with(0, b"A" + revision + b"0")
        else:
            written.assert_not_called()


@pytest.mark.parametrize("confirmed", [True, False])
def test_supervise_payload_expires_missing_mediator_authority(scope, monkeypatch, confirmed):
    """An independent expired lease closes routes and cancels the owned workload."""
    poller = Mock()
    monkeypatch.setattr(linux_supervision.select, "poll", lambda: poller)
    mediator = Mock()
    mediator.lease_expired.return_value = True
    mediator.close.return_value = confirmed

    def create(*, target, daemon):
        """Drive the expiration before the common supervisor receives a result."""
        worker = Mock()
        worker.start.side_effect = target
        worker.is_alive.return_value = False
        return worker

    monkeypatch.setattr(linux_supervision.threading, "Thread", create)
    if confirmed:
        assert (
            linux_supervision.supervise_payload(
                scope.process, scope.config, 50, connection=0, mediator=mediator
            )
            is scope.capture
        )
    else:
        with pytest.raises(OSError, match="cancellation authority"):
            linux_supervision.supervise_payload(
                scope.process, scope.config, 50, connection=0, mediator=mediator
            )
    mediator.close.assert_called_once()
    poller.poll.assert_not_called()
