"""Test asyncio context inspection utilities."""

import asyncio
import inspect
from contextvars import ContextVar
from threading import BoundedSemaphore, Event, get_ident

import pytest

import loop.utils.asyncio as asyncio_utils
from loop.utils import (
    await_value,
    close_awaitable,
    has_running_event_loop,
    is_async_callable,
    run_in_worker,
)


def test_is_async_callable_recognizes_coroutine_functions_and_objects():
    """Callable inspection recognizes async functions and objects."""

    async def asynchronous() -> None:
        pass

    def synchronous() -> None:
        pass

    class _AsyncCallable:
        async def __call__(self) -> None:
            pass

    assert is_async_callable(asynchronous)
    assert not is_async_callable(synchronous)
    assert is_async_callable(_AsyncCallable())


def test_close_awaitable_closes_coroutines():
    """Rejected coroutine values are closed before they can emit an unawaited warning."""

    async def value() -> None:
        pass

    coroutine = value()
    close_awaitable(coroutine)

    assert inspect.getcoroutinestate(coroutine) is inspect.CORO_CLOSED


def test_close_awaitable_ignores_values_without_a_close_method():
    """Awaitables without a close method remain untouched."""
    runner = asyncio.Runner()
    try:
        future = runner.get_loop().create_future()
        close_awaitable(future)
        assert not future.done()
        future.cancel()
    finally:
        runner.close()


def test_close_awaitable_propagates_close_errors():
    """Cleanup errors from a rejected awaitable remain visible to its caller."""

    class BrokenAwaitable:
        """Expose a close method that fails during cleanup."""

        def __await__(self):
            return iter(())

        def close(self) -> None:
            raise RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed"):
        close_awaitable(BrokenAwaitable())


def test_has_running_event_loop_tracks_the_current_thread_context():
    """The helper distinguishes a sync context from a running async context."""
    assert not has_running_event_loop()

    async def inspect_context():
        return has_running_event_loop()

    assert asyncio.run(inspect_context())


def test_await_value_wraps_futures_for_runner_dispatch():
    """The shared wrapper lets a Runner resolve a Future as a top-level coroutine."""
    runner = asyncio.Runner()
    try:
        future = runner.get_loop().create_future()
        future.set_result("resolved")
        assert runner.run(await_value(future)) == "resolved"
    finally:
        runner.close()


def test_run_in_worker_offloads_and_preserves_context():
    """The bounded fallback runs away from the loop and carries correlation context."""
    correlation = ContextVar("correlation", default="missing")
    correlation.set("turn-42")
    owner = get_ident()

    async def scenario():
        """Return worker identity and its copied correlation value."""
        return await run_in_worker(lambda: (get_ident(), correlation.get()))

    worker, value = asyncio.run(scenario())

    assert worker != owner
    assert value == "turn-42"


def test_run_in_worker_propagates_operation_failures():
    """A worker exception is returned to the async caller unchanged."""

    def fail():
        """Raise the controlled operation failure."""
        raise ValueError("operation failed")

    with pytest.raises(ValueError, match="operation failed"):
        asyncio.run(run_in_worker(fail))


def test_run_in_worker_rejects_full_admission(monkeypatch):
    """A saturated bounded worker rejects work before submitting it."""
    monkeypatch.setattr(asyncio_utils, "_WORKER_SLOTS", BoundedSemaphore(0))

    with pytest.raises(RuntimeError, match="worker queue is full"):
        asyncio.run(run_in_worker(lambda: None))


def test_run_in_worker_releases_admission_when_executor_submission_fails(monkeypatch):
    """Failed executor submission returns its admission slot to the bounded queue."""

    class Slots:
        """Track admission release after the executor rejects a submission."""

        released: int

        def __init__(self):
            """Start with one acquired worker slot."""
            self.released = 0

        def acquire(self, *, blocking):
            """Accept the test operation without blocking."""
            assert blocking is False
            return True

        def release(self):
            """Record that the failed submission returned its slot."""
            self.released += 1

    class Executor:
        """Reject every operation as if the shared executor had closed."""

        def submit(self, function, *args):
            """Raise the executor submission failure."""
            raise RuntimeError("executor closed")

    slots = Slots()
    monkeypatch.setattr(asyncio_utils, "_WORKER_SLOTS", slots)
    monkeypatch.setattr(asyncio_utils, "_WORKER_EXECUTOR", Executor())

    with pytest.raises(RuntimeError, match="executor closed"):
        asyncio.run(run_in_worker(lambda: None))

    assert slots.released == 1


def test_run_in_worker_reports_a_cancelled_executor_future(monkeypatch):
    """An executor-cancelled operation remains a visible cancellation outcome."""

    async def scenario():
        """Return a cancelled executor future through the public worker helper."""
        loop = asyncio.get_running_loop()
        pending = loop.create_future()
        pending.cancel()
        monkeypatch.setattr(loop, "run_in_executor", lambda *args: pending)
        with pytest.raises(asyncio.CancelledError) as captured:
            await run_in_worker(lambda: None)
        return captured.value

    error = asyncio.run(scenario())

    assert any("CancelledError" in note for note in error.__notes__)


@pytest.mark.parametrize("fail", [False, True])
def test_run_in_worker_settles_admitted_work_before_propagating_cancellation(fail):
    """Cancellation waits for an admitted worker and retains any secondary failure."""
    started = Event()
    release = Event()
    completed = Event()

    def work():
        """Wait for cancellation, then complete or fail the owned operation."""
        started.set()
        if not release.wait(timeout=2):
            raise TimeoutError("test worker was not released")
        completed.set()
        if fail:
            raise ValueError("worker failed")
        return "done"

    async def scenario():
        """Cancel twice while the worker remains owned, then release it."""
        task = asyncio.create_task(run_in_worker(work))
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await task
        return captured.value

    error = asyncio.run(scenario())

    assert completed.is_set()
    assert any("ValueError" in note for note in getattr(error, "__notes__", ())) is fail


def test_package_exports_asyncio_helpers():
    """The utility package exposes the asyncio helpers from their topic module."""
    assert await_value is asyncio_utils.await_value
    assert close_awaitable is asyncio_utils.close_awaitable
    assert has_running_event_loop is asyncio_utils.has_running_event_loop
    assert is_async_callable is asyncio_utils.is_async_callable
    assert run_in_worker is asyncio_utils.run_in_worker
