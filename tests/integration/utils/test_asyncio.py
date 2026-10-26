"""Verify native worker thread ownership, context propagation and cancellation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from threading import BoundedSemaphore, Event, get_ident

import pytest

import loop.utils.asyncio as asyncio_utils
from loop.utils import run_in_worker_async
from tests.scheduling import deliver_callbacks

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def native_executor(monkeypatch):
    """Own and settle the executor used by each native worker scenario."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        monkeypatch.setattr(asyncio_utils, "_WORKER_EXECUTOR", executor)
        monkeypatch.setattr(asyncio_utils, "_WORKER_SLOTS", BoundedSemaphore(4))
        yield


def test_run_in_worker_offloads_and_preserves_context():
    """The bounded fallback runs away from the loop and carries correlation context."""
    correlation = ContextVar("correlation", default="missing")
    correlation.set("turn-42")
    owner = get_ident()

    async def scenario():
        """Return worker identity and its copied correlation value."""
        return await run_in_worker_async(lambda: (get_ident(), correlation.get()))

    worker, value = asyncio.run(scenario())

    assert worker != owner
    assert value == "turn-42"


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
        task = asyncio.create_task(run_in_worker_async(work))
        while not started.is_set():
            await deliver_callbacks()
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

    assert completed.is_set()
    assert any("ValueError" in note for note in getattr(error, "__notes__", ())) is fail
