"""Provide helpers for asyncio dispatch and coroutine callables."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from threading import BoundedSemaphore
from typing import Any

_WORKER_COUNT = 4
_WORKER_QUEUE_SIZE = 4
_WORKER_EXECUTOR = ThreadPoolExecutor(max_workers=_WORKER_COUNT, thread_name_prefix="loop-io")
_WORKER_SLOTS = BoundedSemaphore(_WORKER_COUNT + _WORKER_QUEUE_SIZE)


async def run_in_worker(function: Callable[..., Any], *args: Any) -> Any:
    """Run synchronous I/O on a shared worker with bounded admission.

    Application services should inject their own worker when available. This process-owned
    fallback keeps standalone async APIs off the event loop without using its default executor.

    Args:
        function (Callable[..., Any]): Synchronous operation to execute.
        args (tuple[Any, ...]): Positional arguments passed to the operation.

    Returns:
        Any: Result returned by the synchronous operation.

    Raises:
        RuntimeError: If the bounded worker queue is full or the executor is unavailable.
        asyncio.CancelledError: After an admitted operation settles when its caller is cancelled.
        Exception: If the synchronous operation raises an exception.
    """
    loop = asyncio.get_running_loop()
    if not _WORKER_SLOTS.acquire(blocking=False):
        raise RuntimeError("The bounded worker queue is full.")

    try:
        pending = loop.run_in_executor(
            _WORKER_EXECUTOR,
            copy_context().run,
            partial(function, *args),
        )
    except BaseException:
        _WORKER_SLOTS.release()
        raise

    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError as cancellation:
        while not pending.done():
            try:
                await asyncio.wait({pending})
            except asyncio.CancelledError:
                continue
        if pending.cancelled():
            failure = asyncio.CancelledError()
        else:
            failure = pending.exception()
        if failure is not None:
            cancellation.add_note(
                f"The blocking operation also stopped with {type(failure).__name__}."
            )
        raise
    finally:
        _WORKER_SLOTS.release()


def is_async_callable(function: Callable[..., Any]) -> bool:
    """Return whether invoking a function or callable object produces a coroutine directly.

    Args:
        function (Callable[..., Any]): Callable whose implementation should be inspected.

    Returns:
        bool: Whether the callable is implemented as a coroutine function.
    """
    return inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(function.__call__)


def close_awaitable(value: Awaitable[Any]) -> None:
    """Close an awaitable when it exposes a callable close method.

    Futures and tasks without ``close()`` remain active; cancellation is left to their owner.

    Args:
        value (Awaitable[Any]): Awaitable to close after synchronous dispatch rejects it.

    Raises:
        Exception: If the awaitable's close method raises an exception.
    """
    close = getattr(value, "close", None)
    if callable(close):
        close()


def has_running_event_loop() -> bool:
    """Return whether the current thread is already running an asyncio event loop.

    Returns:
        bool: Whether synchronous code must avoid taking ownership of another event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


async def await_value[T](value: Awaitable[T]) -> T:
    """Await an awaitable through a coroutine wrapper.

    Args:
        value (Awaitable[T]): Awaitable value to resolve.

    Returns:
        T: Resolved value.
    """
    return await value
