"""Test asyncio context inspection utilities."""

import asyncio
import inspect

import pytest

import loop.utils.asyncio as asyncio_utils
from loop.utils import await_value, close_awaitable, has_running_event_loop, is_async_callable


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


def test_package_exports_asyncio_helpers():
    """The utility package exposes the asyncio helpers from their topic module."""
    assert await_value is asyncio_utils.await_value
    assert close_awaitable is asyncio_utils.close_awaitable
    assert has_running_event_loop is asyncio_utils.has_running_event_loop
    assert is_async_callable is asyncio_utils.is_async_callable
