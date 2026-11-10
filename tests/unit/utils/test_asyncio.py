"""Test asyncio context inspection utilities."""

import asyncio
import gc
import inspect
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from threading import Event, Thread
from unittest.mock import Mock

import pytest

import loop.utils.asyncio as asyncio_utils
from loop.utils import (
    await_value_async,
    close_awaitable,
    has_running_event_loop,
    is_async_callable,
    run_in_worker_async,
    settle_future_async,
)
from tests.scheduling import deliver_callbacks


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


def test_settle_future_waits_through_repeated_cancellation():
    """Settlement retains a borrowed future until its owner completes it."""

    async def scenario():
        """Cancel the waiter twice and then complete the original future."""
        future = asyncio.get_running_loop().create_future()
        waiter = asyncio.create_task(settle_future_async(future))
        await deliver_callbacks()
        waiter.cancel()
        await deliver_callbacks()
        waiter.cancel()
        await deliver_callbacks()
        assert not waiter.done()
        assert not future.done()
        future.set_result("done")
        await waiter
        await settle_future_async(future)
        assert future.result() == "done"

    asyncio.run(scenario())


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
        assert runner.run(await_value_async(future)) == "resolved"
    finally:
        runner.close()


def test_run_in_worker_propagates_operation_failures():
    """A worker exception is returned to the async caller unchanged."""

    def fail():
        """Raise the controlled operation failure."""
        raise ValueError("operation failed")

    with pytest.raises(ValueError, match="operation failed"):
        asyncio.run(run_in_worker_async(fail))


def test_run_in_worker_rejects_full_admission(monkeypatch):
    """A saturated bounded worker rejects work before submitting it."""
    monkeypatch.setattr(asyncio_utils, "_WORKER_SLOTS", asyncio_utils.WorkerCapacity(0))

    with pytest.raises(RuntimeError, match="worker queue is full"):
        asyncio.run(run_in_worker_async(lambda: None))


def test_interruption_after_admission_does_not_leak_worker_capacity(monkeypatch):
    """Interrupting immediately after admission leaves the slot available."""
    slots = asyncio_utils.WorkerCapacity(1)
    monkeypatch.setattr(asyncio_utils, "_WORKER_SLOTS", slots)
    source, first_line = inspect.getsourcelines(asyncio_utils.WorkerCapacity.acquire)
    target = first_line + next(
        index for index, line in enumerate(source) if line.strip() == "return True"
    )

    def interrupt(frame, event, argument):
        """Interrupt after the admission token has been inserted."""
        if (
            event == "line"
            and frame.f_code is asyncio_utils.WorkerCapacity.acquire.__code__
            and frame.f_lineno == target
        ):
            sys.settrace(None)
            raise KeyboardInterrupt
        return interrupt

    async def scenario():
        """Interrupt one admission and then successfully use the same capacity."""
        previous = sys.gettrace()
        sys.settrace(interrupt)
        try:
            with pytest.raises(KeyboardInterrupt):
                await run_in_worker_async(lambda: "unexpected")
        finally:
            sys.settrace(previous)
        assert await run_in_worker_async(lambda: "available") == "available"

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["release", "quarantine"])
def test_interrupted_capacity_cleanup_is_retried(operation):
    """One interruption cannot leave a token or ambiguous pool in the wrong state."""
    capacity = asyncio_utils.WorkerCapacity(1)
    token = object()
    assert capacity.acquire(token)
    method = getattr(asyncio_utils.WorkerCapacity, operation)
    source, first_line = inspect.getsourcelines(method)
    marker = (
        "self._entries.discard(token)" if operation == "release" else "self._quarantined = True"
    )
    target = first_line + next(index for index, line in enumerate(source) if marker in line)

    def interrupt(frame, event, argument):
        """Interrupt the first cleanup attempt just before its idempotent mutation."""
        if event == "line" and frame.f_code is method.__code__ and frame.f_lineno == target:
            sys.settrace(None)
            raise KeyboardInterrupt
        return interrupt

    previous = sys.gettrace()
    sys.settrace(interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            capacity.release(token) if operation == "release" else capacity.quarantine()
    finally:
        sys.settrace(previous)
    if operation == "release":
        assert not capacity.busy
    else:
        assert capacity.quarantined
        capacity.release(token)


def test_thread_start_failure_quarantines_pool_without_queue_growth(monkeypatch):
    """A failed worker start refuses later work instead of accumulating queued entries."""
    original_start = Thread.start
    effects = []
    starts = []

    def fail_worker_start(thread):
        """Refuse worker creation at the real executor thread boundary."""
        if thread.name.startswith("loop-test-io"):
            starts.append(thread.name)
            raise RuntimeError("synthetic worker start failure")
        return original_start(thread)

    async def scenario():
        """Fail the first submission and reject subsequent work in the poisoned pool."""
        with monkeypatch.context() as scoped:
            scoped.setattr(Thread, "start", fail_worker_start)
            with pytest.raises(RuntimeError, match="synthetic worker start failure"):
                await run_in_worker_async(lambda: effects.append("entered"), cancel_pending=True)
        for _ in range(5):
            with pytest.raises(RuntimeError, match="restart Loop"):
                await run_in_worker_async(lambda: effects.append("entered"), cancel_pending=True)

    asyncio.run(scenario())
    assert len(starts) == 1
    assert effects == []


def test_interrupted_failed_submission_still_quarantines_pool(monkeypatch):
    """Cleanup interruption cannot make an ambiguously queued worker reusable."""
    slots = asyncio_utils.WorkerCapacity(1)
    monkeypatch.setattr(asyncio_utils, "_WORKER_SLOTS", slots)
    monkeypatch.setattr(asyncio_utils._WORKER_EXECUTOR, "submit", Mock(side_effect=RuntimeError))
    monkeypatch.setattr(
        asyncio_utils.WorkerEntry, "cancel_before_start", Mock(side_effect=KeyboardInterrupt)
    )

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(run_in_worker_async(lambda: None))
    assert slots.quarantined
    assert not slots.acquire(object())


def test_worker_quarantine_preserves_already_admitted_work(monkeypatch):
    """An ambiguous failure cannot cancel a different operation already admitted."""
    original_start = Thread.start
    worker_entered = Event()
    worker_release = Event()
    effects = []

    def refuse_second_worker(thread):
        """Fail a new pool thread after one worker has already entered."""
        if thread.name.startswith("loop-test-io"):
            raise RuntimeError("synthetic worker start failure")
        return original_start(thread)

    def held_work():
        """Hold one admitted worker until the owner loop releases it."""
        worker_entered.set()
        assert worker_release.wait(timeout=5)
        return "settled"

    async def scenario():
        """Refuse new entries while allowing the previously admitted one to finish."""
        first = asyncio.create_task(run_in_worker_async(held_work))
        assert await asyncio.to_thread(worker_entered.wait, 5)
        with monkeypatch.context() as scoped:
            scoped.setattr(Thread, "start", refuse_second_worker)
            with pytest.raises(RuntimeError, match="synthetic worker start failure"):
                await run_in_worker_async(lambda: effects.append("untracked"), cancel_pending=True)
            with pytest.raises(RuntimeError, match="restart Loop"):
                await run_in_worker_async(lambda: None)
        worker_release.set()
        assert await first == "settled"
        with pytest.raises(RuntimeError, match="restart Loop"):
            await run_in_worker_async(lambda: None)

    try:
        asyncio.run(scenario())
    finally:
        worker_release.set()
    assert effects == []


def test_interrupted_thread_start_does_not_replace_ambiguous_pool(monkeypatch):
    """A thread launched before interruption cannot cause automatic pool replacement."""
    original_start = Thread.start
    threads = []
    effects = []

    def start_then_interrupt(thread):
        """Start the real worker but interrupt before its executor records it."""
        if thread.name.startswith("loop-test-io"):
            threads.append(thread)
            original_start(thread)
            raise KeyboardInterrupt
        return original_start(thread)

    async def scenario():
        """Keep the affected pool quarantined after the ambiguous start."""
        with monkeypatch.context() as scoped:
            scoped.setattr(Thread, "start", start_then_interrupt)
            with pytest.raises(KeyboardInterrupt):
                await run_in_worker_async(lambda: effects.append("entered"), cancel_pending=True)
        with pytest.raises(RuntimeError, match="restart Loop"):
            await run_in_worker_async(lambda: effects.append("entered"), cancel_pending=True)
        asyncio_utils._WORKER_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        await asyncio.to_thread(threads[0].join, 5)
        assert not threads[0].is_alive()

    asyncio.run(scenario())
    assert len(threads) == 1
    assert effects == []


def test_run_in_worker_releases_admission_when_executor_submission_fails(monkeypatch):
    """Failed executor submission returns its admission slot to the bounded queue."""

    class Slots:
        """Track admission release after the executor rejects a submission."""

        released: int

        def __init__(self):
            """Start with one acquired worker slot."""
            self.released = 0

        def acquire(self, token):
            """Accept the test operation without blocking."""
            return True

        def release(self, token):
            """Record that the failed submission returned its slot."""
            self.released += 1

        def quarantine(self):
            """Record that the failed executor cannot accept more work."""

    class Executor:
        """Reject every operation as if the shared executor had closed."""

        def submit(self, function, *args):
            """Raise the executor submission failure."""
            raise RuntimeError("executor closed")

    slots = Slots()
    monkeypatch.setattr(asyncio_utils, "_WORKER_SLOTS", slots)
    monkeypatch.setattr(asyncio_utils, "_WORKER_EXECUTOR", Executor())

    with pytest.raises(RuntimeError, match="executor closed"):
        asyncio.run(run_in_worker_async(lambda: None))

    assert slots.released == 1


def test_canceled_queued_worker_keeps_slot_until_its_entry_is_skipped(monkeypatch):
    """Opt-in cancellation prevents entry without accumulating unbounded queue slots."""
    slots = asyncio_utils.WorkerCapacity(1)
    monkeypatch.setattr(asyncio_utils, "_WORKER_SLOTS", slots)
    effects = []
    submissions = []

    class ControlledExecutor:
        """Retain the queued wrapper until the test drains it."""

        def submit(self, function, *args):
            """Return a future for work controlled by the test."""
            pending = Future()
            submissions.append((function, args, pending))
            return pending

    monkeypatch.setattr(asyncio_utils, "_WORKER_EXECUTOR", ControlledExecutor())

    async def scenario():
        """Cancel before the controlled executor runs the submitted wrapper."""
        task = asyncio.create_task(
            run_in_worker_async(lambda: effects.append("entered"), cancel_pending=True)
        )
        await deliver_callbacks()
        assert len(submissions) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        token = object()
        assert not slots.acquire(token)
        function, args, pending = submissions[0]
        pending.set_result(function(*args))
        assert slots.acquire(token)
        slots.release(token)
        await deliver_callbacks()

    asyncio.run(scenario())
    assert effects == []


def test_interrupted_submission_cannot_start_the_queued_operation(monkeypatch):
    """An interruption after submission but before ownership cannot start the work."""
    slots = asyncio_utils.WorkerCapacity(1)
    submissions = []
    effects = []

    class ControlledExecutor:
        """Retain submitted work until its owner has handled the interruption."""

        def submit(self, function, *args):
            """Return a controlled future for the submitted wrapper."""
            pending = Future()
            submissions.append((function, args, pending))
            return pending

    original_submit = asyncio_utils.WorkerEntry.submit

    def interrupted_submit(entry, *args):
        """Interrupt after the worker was queued but before the future is returned."""
        original_submit(entry, *args)
        raise KeyboardInterrupt

    monkeypatch.setattr(asyncio_utils, "_WORKER_SLOTS", slots)
    monkeypatch.setattr(asyncio_utils, "_WORKER_EXECUTOR", ControlledExecutor())
    monkeypatch.setattr(asyncio_utils.WorkerEntry, "submit", interrupted_submit)

    async def scenario():
        """Handle interruption on the owner loop before draining the queued wrapper."""
        with pytest.raises(KeyboardInterrupt):
            await run_in_worker_async(lambda: effects.append("entered"), cancel_pending=True)
        await deliver_callbacks()
        function, args, pending = submissions[0]
        pending.set_result(function(*args))
        await deliver_callbacks()

    asyncio.run(scenario())
    assert effects == []
    token = object()
    assert slots.acquire(token)
    slots.release(token)


def test_interruption_between_submission_and_wait_cannot_start_work(monkeypatch):
    """An interruption at the wait boundary cancels the submitted entry."""
    effects = []
    source, first_line = inspect.getsourcelines(run_in_worker_async)
    await_index = next(i for i, line in enumerate(source) if "return await asyncio.shield" in line)
    wait_line = first_line + await_index

    def interrupt(frame, event, argument):
        """Inject an interruption immediately before the worker wait begins."""
        if (
            event == "line"
            and frame.f_code is run_in_worker_async.__code__
            and frame.f_lineno == wait_line
        ):
            sys.settrace(None)
            raise KeyboardInterrupt
        return interrupt

    async def scenario():
        """Interrupt the submitted operation before the event loop can release its gate."""
        previous = sys.gettrace()
        sys.settrace(interrupt)
        try:
            with pytest.raises(KeyboardInterrupt):
                await run_in_worker_async(lambda: effects.append("entered"), cancel_pending=True)
        finally:
            sys.settrace(previous)
        await deliver_callbacks()

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(asyncio_utils, "_WORKER_EXECUTOR", executor)
        asyncio.run(scenario())
    assert effects == []


def test_cancelled_executor_entry_does_not_leave_an_unretrieved_exception(monkeypatch):
    """Executor shutdown can cancel an already-abandoned queued entry without warnings."""
    submissions = []
    diagnostics = []

    class ControlledExecutor:
        """Expose the executor future while keeping its operation queued."""

        def submit(self, function, *args):
            """Return a future which the executor may cancel before entry."""
            pending = Future()
            submissions.append(pending)
            return pending

    monkeypatch.setattr(asyncio_utils, "_WORKER_EXECUTOR", ControlledExecutor())

    async def scenario():
        """Cancel the dispatcher and then cancel its queued executor future."""
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
        task = asyncio.create_task(run_in_worker_async(lambda: None, cancel_pending=True))
        await deliver_callbacks()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert submissions[0].cancel()
        await deliver_callbacks()
        del task
        gc.collect()
        await deliver_callbacks()

    asyncio.run(scenario())
    assert diagnostics == []


def test_guarded_worker_reports_executor_cancellation(monkeypatch):
    """A canceled executor future releases admission and propagates cancellation."""

    class CancelledExecutor:
        """Return an executor future that was canceled before entry."""

        def submit(self, function, *args):
            """Skip the submitted operation."""
            result = Future()
            result.cancel()
            return result

    monkeypatch.setattr(asyncio_utils, "_WORKER_EXECUTOR", CancelledExecutor())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_in_worker_async(lambda: pytest.fail("entered"), cancel_pending=True))


def test_worker_submission_cancellation_releases_admission(monkeypatch):
    """Cancellation during submission does not strand an acquired worker slot."""
    slots = asyncio_utils.WorkerCapacity(1)
    monkeypatch.setattr(asyncio_utils, "_WORKER_SLOTS", slots)
    monkeypatch.setattr(
        asyncio_utils.WorkerEntry, "submit", Mock(side_effect=asyncio.CancelledError)
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_in_worker_async(lambda: None, cancel_pending=True))
    token = object()
    assert slots.acquire(token)
    slots.release(token)


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
@pytest.mark.parametrize("cancel_pending", [False, True])
def test_interruption_waits_for_a_worker_that_has_entered(monkeypatch, outcome, cancel_pending):
    """An exceptional wait still settles effects already admitted to a worker."""
    release = Event()
    effects = []

    async def scenario():
        """Interrupt after the worker has entered, then release it on the owner loop."""
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def effect():
            """Wait until the owner loop permits the entered work to finish."""
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=5)
            effects.append("settled")
            if outcome == "failure":
                raise ValueError("worker failed")
            if outcome == "cancelled":
                raise asyncio.CancelledError

        async def interrupted_wait(pending):
            """Raise after the callable enters, while it still needs its owner loop."""
            await started.wait()
            loop.call_soon(release.set)
            raise KeyboardInterrupt

        monkeypatch.setattr(asyncio, "shield", interrupted_wait)
        with pytest.raises(KeyboardInterrupt) as captured:
            await run_in_worker_async(effect, cancel_pending=cancel_pending)
        assert effects == ["settled"]
        if outcome == "failure":
            assert captured.value.__notes__ == [
                "The blocking operation also stopped with ValueError."
            ]

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_worker_completion_ignores_an_already_cancelled_destination():
    """Late worker completion cannot replace a destination's cancellation."""
    submitted = []
    capacity = asyncio_utils.WorkerCapacity(1)
    token = object()
    assert capacity.acquire(token)

    class ControlledExecutor:
        """Retain the worker wrapper for explicit execution."""

        def submit(self, function, *args):
            """Return a controllable executor future."""
            result = Future()
            submitted.append((function, args, result))
            return result

    async def scenario():
        """Cancel the owner future before completing the worker."""
        gate = asyncio_utils.WorkerEntry(capacity, token)
        pending = gate.submit(ControlledExecutor(), asyncio.get_running_loop(), lambda: "done")
        pending.cancel()
        await deliver_callbacks()
        function, args, source = submitted[0]
        source.set_result(function(*args))
        await deliver_callbacks()
        assert pending.cancelled()

    asyncio.run(scenario())
    assert not capacity.busy


@pytest.mark.parametrize("loop_closed", [False, True])
def test_worker_completion_handles_closed_or_unavailable_owner_loop(monkeypatch, loop_closed):
    """A closed loop may discard completion; a live loop's scheduling failure is raised."""
    submitted = []
    capacity = asyncio_utils.WorkerCapacity(1)
    token = object()
    assert capacity.acquire(token)

    class ControlledExecutor:
        """Retain the worker wrapper until the owner loop is unavailable."""

        def submit(self, function, *args):
            """Return a controllable executor future."""
            result = Future()
            submitted.append((function, args))
            return result

    loop = asyncio.new_event_loop()
    gate = asyncio_utils.WorkerEntry(capacity, token)
    pending = gate.submit(ControlledExecutor(), loop, lambda: "done")
    loop.run_until_complete(asyncio.sleep(0))
    if loop_closed:
        loop.close()
    else:
        monkeypatch.setattr(
            loop, "call_soon_threadsafe", Mock(side_effect=RuntimeError("unavailable"))
        )
    function, args = submitted[0]
    try:
        if loop_closed:
            assert function(*args) == "done"
        else:
            with pytest.raises(RuntimeError, match="unavailable"):
                function(*args)
    finally:
        if not loop_closed:
            pending.cancel()
            loop.close()
    assert not capacity.busy


def test_run_in_worker_reports_a_cancelled_executor_future(monkeypatch):
    """An executor-cancelled operation remains a visible cancellation outcome."""

    class CancelledExecutor:
        """Cancel an entry before the executor starts it."""

        def submit(self, function, *args):
            """Return a canceled executor future."""
            pending = Future()
            pending.cancel()
            return pending

    monkeypatch.setattr(asyncio_utils, "_WORKER_EXECUTOR", CancelledExecutor())
    with pytest.raises(asyncio.CancelledError) as captured:
        asyncio.run(run_in_worker_async(lambda: None))
    assert any("CancelledError" in note for note in captured.value.__notes__)


def test_package_exports_asyncio_helpers():
    """The utility package exposes the asyncio helpers from their topic module."""
    assert await_value_async is asyncio_utils.await_value_async
    assert close_awaitable is asyncio_utils.close_awaitable
    assert has_running_event_loop is asyncio_utils.has_running_event_loop
    assert is_async_callable is asyncio_utils.is_async_callable
    assert run_in_worker_async is asyncio_utils.run_in_worker_async


def test_run_in_worker_submits_the_captured_context_and_returns_the_result():
    """The worker boundary receives the calling context and returns its completed result."""
    correlation = ContextVar("correlation", default="missing")
    correlation.set("turn-42")
    assert asyncio.run(run_in_worker_async(correlation.get)) == "turn-42"


@pytest.mark.parametrize("fail", [False, True])
def test_run_in_worker_settles_controlled_result_after_cancellation(fail):
    """Cancellation retains ownership until the injected result settles, including failures."""

    async def scenario():
        """Deliver cancellation and completion through explicit callback turns."""
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = Event()

        def operation():
            """Wait for both cancellation requests before settling."""
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=5)
            if fail:
                raise ValueError("worker failed")
            return "done"

        task = asyncio.create_task(run_in_worker_async(operation))
        await started.wait()
        task.cancel()
        await deliver_callbacks()
        task.cancel()
        await deliver_callbacks()
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await task
        return captured.value

    error = asyncio.run(scenario())
    assert any("ValueError" in note for note in getattr(error, "__notes__", ())) is fail
