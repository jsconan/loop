"""Verify serialized request compilation and request revalidation."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from loop.execution.sandbox.linux import compiler as linux_compiler


@pytest.fixture
def compiler(monkeypatch):
    """Provide current request authority and a bounded source compiler collaborator."""

    @contextmanager
    def verified(*, deadline):
        """Model a pinned compiler lock."""
        assert deadline in (state.request.deadline, min(state.request.deadline, 15))
        yield 99

    state = SimpleNamespace(
        lock=SimpleNamespace(identity=(1, 2, 0o100444), open_verified=Mock(side_effect=verified)),
        request=SimpleNamespace(deadline=100, paths_are_current=Mock(return_value=True)),
        source="profile payload {}\n",
        times=iter([10, 10]),
        compile=Mock(return_value=b"compiled"),
        flock=Mock(),
    )
    state.render = Mock(side_effect=lambda *args, **kwargs: state.source)
    monkeypatch.setattr(linux_compiler, "payload_policy", state.render)
    monkeypatch.setattr(linux_compiler, "compile_policy_source", state.compile)
    monkeypatch.setattr(linux_compiler.fcntl, "flock", state.flock)
    monkeypatch.setattr(linux_compiler.time, "monotonic", lambda: next(state.times))
    monkeypatch.setattr(linux_compiler.time, "sleep", Mock())
    return state


def compile_request(state):
    """Prepare a current authorized payload through its public owner."""
    return linux_compiler.compile_payload(
        state.request,
        Path("/scratch"),
        Path("/cache"),
        "payload",
        (Path("/usr"),),
        state.lock,
    )


def test_compile_payload_serializes_and_revalidates(compiler):
    """Shared bounded compilation runs under the installation compiler lock with exact source."""
    assert compile_request(compiler) == b"compiled"
    compiler.compile.assert_called_once_with(compiler.source.encode(), 15)
    assert compiler.render.call_args.kwargs == {"runtime_roots": (Path("/usr"),)}
    compiler.flock.assert_called_once_with(
        99, linux_compiler.fcntl.LOCK_EX | linux_compiler.fcntl.LOCK_NB
    )
    assert compiler.lock.open_verified.call_args_list == [
        call(deadline=compiler.request.deadline),
        call(deadline=min(compiler.request.deadline, 15)),
    ]


def test_compile_payload_waits_for_compiler_lock(compiler):
    """Concurrent compilation joins the lock within the original request budget."""
    compiler.times = iter([10, 10, 10])
    compiler.flock.side_effect = [BlockingIOError("busy"), None]
    assert compile_request(compiler) == b"compiled"
    assert compiler.flock.call_count == 2
    linux_compiler.time.sleep.assert_called_once_with(0.01)


def test_compile_payload_times_out_waiting_for_compiler_lock(compiler):
    """A busy compiler cannot extend the request deadline."""
    compiler.times = iter([10, 100])
    compiler.flock.side_effect = BlockingIOError("busy")
    with pytest.raises(TimeoutError, match="Payload compilation deadline expired"):
        compile_request(compiler)
    compiler.compile.assert_not_called()


@pytest.mark.parametrize("cancelled", [False, True])
def test_compile_payload_observes_cancellation_while_waiting(compiler, cancelled):
    """A cancelled request never compiles while another request owns the lock."""
    compiler.times = iter([10, 10, 10])
    compiler.flock.side_effect = [BlockingIOError("busy"), None]
    check = Mock(return_value=cancelled)
    args = (
        compiler.request,
        Path("/scratch"),
        Path("/cache"),
        "payload",
        (Path("/usr"),),
        compiler.lock,
    )
    if cancelled:
        with pytest.raises(linux_compiler.CompilationCancelled, match="compilation cancelled"):
            linux_compiler.compile_payload(*args, cancelled=check)
        compiler.compile.assert_not_called()
    else:
        assert linux_compiler.compile_payload(*args, cancelled=check) == b"compiled"
    check.assert_called_once_with()


@pytest.mark.parametrize("cause", ["source", "lock", "expired", "drift", "request", "compile"])
def test_compile_payload_refuses_unsafe_preparation(compiler, cause):
    """Unsafe or stale authority and failed preparation never return usable compiled policy."""
    exception, message = ValueError, "bound|read-only|changed"
    if cause == "source":
        compiler.source = "é" * 32769
    elif cause == "lock":
        compiler.lock.identity = (1, 2, 0o100644)
    elif cause == "expired":
        compiler.request.deadline = 10
        exception, message = TimeoutError, "deadline expired"
    elif cause == "drift":
        compiler.lock.open_verified.side_effect = ValueError("authority changed")
    elif cause == "request":
        compiler.request.paths_are_current.return_value = False
    else:
        compiler.compile.side_effect = OSError("compilation failed")
        exception, message = OSError, "compilation failed"
    with pytest.raises(exception, match=message):
        compile_request(compiler)


def test_compile_payload_preserves_shorter_request_budget(compiler):
    """Compilation never expands an existing request deadline."""
    compiler.request.deadline = 12
    compile_request(compiler)
    assert compiler.compile.call_args.args[1] == 12
