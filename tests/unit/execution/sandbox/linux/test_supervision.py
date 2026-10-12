"""Verify admitted namespace supervision and authoritative cleanup refusal."""

import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import pytest

from loop.execution.sandbox.linux import supervision as linux_supervision
from loop.utils.process import ProcessCapture, ProcessCaptureStatus


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
