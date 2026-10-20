"""Provide helpers for asyncio dispatch and coroutine callables."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any


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
