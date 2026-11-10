"""Provide helpers for asyncio dispatch and coroutine callables."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from threading import Event, Lock
from typing import Any

_WORKER_COUNT = 4
_WORKER_QUEUE_SIZE = 4
_WORKER_EXECUTOR = ThreadPoolExecutor(max_workers=_WORKER_COUNT, thread_name_prefix="loop-io")


class WorkerCapacity:
    """Bound admitted worker entries and fail closed after ambiguous submission.

    Args:
        limit (int): Maximum number of admitted entries.
    """

    _limit: int
    _entries: set[object]
    _lock: Lock
    _quarantined: bool

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._entries = set()
        self._lock = Lock()
        self._quarantined = False

    def acquire(self, token: object) -> bool:
        """Admit an identity if capacity and executor integrity are available.

        The caller must release its token in a ``finally`` block, even on failure.

        Args:
            token (object): Unique identity of the attempted admission.

        Returns:
            bool: Whether the identity was admitted.
        """
        with self._lock:
            if self._quarantined or len(self._entries) >= self._limit:
                return False
            self._entries.add(token)
            return True

    def release(self, token: object) -> None:
        """Release this identity without affecting any other admission.

        Args:
            token (object): Identity to release, even if it was never admitted.
        """
        try:
            with self._lock:
                self._entries.discard(token)
        except BaseException:
            with self._lock:
                self._entries.discard(token)
            raise

    @property
    def busy(self) -> bool:
        """Return whether work or uncertain executor cleanup remains outstanding.

        Returns:
            bool: Whether a synchronous owner-loop shutdown could block dispatch.
        """
        with self._lock:
            return bool(self._entries) or self._quarantined

    @property
    def quarantined(self) -> bool:
        """Return whether an ambiguous submission made the pool unavailable.

        Returns:
            bool: Whether new work must be refused until the pool is replaced.
        """
        with self._lock:
            return self._quarantined

    def quarantine(self) -> None:
        """Refuse new entries while already-admitted work settles."""
        try:
            with self._lock:
                self._quarantined = True
        except BaseException:
            with self._lock:
                self._quarantined = True
            raise


_WORKER_SLOTS = WorkerCapacity(_WORKER_COUNT + _WORKER_QUEUE_SIZE)


class WorkerEntry:
    """Keep worker capacity until queued work is skipped or running work finishes.

    Args:
        capacity (WorkerCapacity): Admission owner for this entry.
        token (object): Unique identity admitted by the capacity owner.
    """

    _state: Future[None]
    _capacity: WorkerCapacity
    _token: object
    _ready: Event
    _submission_attempted: bool
    _submission_confirmed: bool

    def __init__(self, capacity: WorkerCapacity, token: object) -> None:
        self._state = Future()
        self._capacity = capacity
        self._token = token
        self._ready = Event()
        self._submission_attempted = False
        self._submission_confirmed = False

    @property
    def submission_attempted(self) -> bool:
        """Return whether executor submission may have begun.

        Returns:
            bool: Whether the executor may contain this entry.
        """
        return self._submission_attempted

    @property
    def submission_confirmed(self) -> bool:
        """Return whether the executor accepted this entry completely.

        Returns:
            bool: Whether normal executor ownership of the entry was established.
        """
        return self._submission_confirmed

    def run(self, function: Callable[..., Any], *args: Any) -> Any:
        """Enter work atomically unless cancellation won the admission race.

        Args:
            function (Callable[..., Any]): Callable to run after entry.
            args (tuple[Any, ...]): Positional arguments passed to the callable.

        Returns:
            Any: The callable's result, or ``None`` if entry was cancelled.
        """
        try:
            self._ready.wait()
            if not self._state.set_running_or_notify_cancel():
                return None
            try:
                return function(*args)
            finally:
                self._state.set_result(None)
        finally:
            self.release()

    def cancel_before_start(self) -> bool:
        """Cancel only work that has not won the entry race.

        Returns:
            bool: Whether the callable is guaranteed not to start.
        """
        cancelled = self._state.cancel()
        if cancelled:
            self._ready.set()
        return cancelled

    def submit(
        self,
        executor: Executor,
        loop: asyncio.AbstractEventLoop,
        function: Callable[..., Any],
        *args: Any,
    ) -> asyncio.Future[Any]:
        """Submit guarded work with an owner-loop completion notification.

        Args:
            executor (Executor): Executor that runs the guarded operation.
            loop (asyncio.AbstractEventLoop): Loop receiving the completion result.
            function (Callable[..., Any]): Operation to invoke after worker entry.
            args (tuple[Any, ...]): Positional arguments passed to the operation.

        Returns:
            asyncio.Future[Any]: Future completed on the owner loop.

        Raises:
            BaseException: If the loop or executor cannot accept the submission.
        """
        result = loop.create_future()

        def transfer(value: Any, error: BaseException | None) -> None:
            """Deliver one completed worker outcome on its owner loop."""
            if result.done():
                return
            if isinstance(error, asyncio.CancelledError):
                result.cancel()
            elif error is not None:
                result.set_exception(error)
            else:
                result.set_result(value)

        def notify(value: Any, error: BaseException | None) -> None:
            """Schedule completion if the owner loop is still available."""
            try:
                loop.call_soon_threadsafe(transfer, value, error)
            except RuntimeError:
                if not loop.is_closed():
                    raise

        def invoke() -> Any:
            """Publish the worker outcome even if submission itself was interrupted."""
            try:
                value = self.run(function, *args)
            except BaseException as error:
                notify(None, error)
                raise
            notify(value, None)
            return value

        def release_cancelled(source: Future[Any]) -> None:
            """Release admission if the executor skips the wrapper itself."""
            if source.cancelled() and self.cancel_before_start():
                self.release()
                notify(None, asyncio.CancelledError())

        try:
            self._submission_attempted = True
            submitted = executor.submit(invoke)
            self._submission_confirmed = True
            submitted.add_done_callback(release_cancelled)
            # The owner loop cannot run this callback until the submitter yields,
            # after it has received the completion future and installed cleanup.
            loop.call_soon(self._ready.set)
        except BaseException:
            self.cancel_before_start()
            self._ready.set()
            raise
        return result

    def release(self) -> None:
        """Release this entry's admission without affecting other work."""
        self._capacity.release(self._token)


async def run_in_worker_async(
    function: Callable[..., Any],
    *args: Any,
    cancel_pending: bool = False,
) -> Any:
    """Run synchronous I/O on a shared worker with bounded admission.

    Application services should inject their own worker when available. This process-owned
    fallback keeps standalone async APIs off the event loop without using its default executor.

    Args:
        function (Callable[..., Any]): Synchronous operation to execute.
        args (tuple[Any, ...]): Positional arguments passed to the operation.
        cancel_pending (bool): Prevent entry if cancellation wins before the worker starts;
            defaults to ``False`` so mandatory work still settles after admission.

    Returns:
        Any: Result returned by the synchronous operation.

    Raises:
        RuntimeError: If the bounded worker queue is full or the executor is unavailable.
        asyncio.CancelledError: After an admitted operation settles when its caller is cancelled.
        Exception: If the synchronous operation raises an exception.
    """
    loop = asyncio.get_running_loop()
    token = object()
    gate = None
    pending = None
    try:
        if not _WORKER_SLOTS.acquire(token):
            if _WORKER_SLOTS.quarantined:
                raise RuntimeError(
                    "Worker pool unavailable after submission failure; restart Loop."
                )
            raise RuntimeError("The bounded worker queue is full.")
        gate = WorkerEntry(_WORKER_SLOTS, token)
        context = copy_context()
        operation = partial(context.run, partial(function, *args))
        pending = gate.submit(_WORKER_EXECUTOR, loop, operation)
        return await asyncio.shield(pending)
    except asyncio.CancelledError as cancellation:
        if pending is None:
            raise
        if cancel_pending and gate is not None and gate.cancel_before_start():
            if pending.cancelled():
                gate.release()
            raise
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
    except BaseException as interruption:
        if (
            gate is not None
            and pending is not None
            and (not cancel_pending or not gate.cancel_before_start())
        ):
            if not pending.done():
                await settle_future_async(pending)
            if not pending.cancelled():
                failure = pending.exception()
                if failure is not None and failure is not interruption:
                    interruption.add_note(
                        f"The blocking operation also stopped with {type(failure).__name__}."
                    )
        raise
    finally:
        if gate is None:
            _WORKER_SLOTS.release(token)
        else:
            if pending is None:
                try:
                    gate.cancel_before_start()
                finally:
                    try:
                        if gate.submission_attempted and not gate.submission_confirmed:
                            _WORKER_SLOTS.quarantine()
                    finally:
                        if not gate.submission_confirmed:
                            _WORKER_SLOTS.release(token)
            elif cancel_pending:
                gate.cancel_before_start()


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


async def settle_future_async(future: asyncio.Future[Any]) -> None:
    """Wait for a future to settle despite repeated cancellation of its waiter.

    Args:
        future (asyncio.Future[Any]): Future whose owner must reach a terminal state.
    """
    while not future.done():
        try:
            await asyncio.wait({future})
        except asyncio.CancelledError:
            continue


async def await_value_async[T](value: Awaitable[T]) -> T:
    """Await an awaitable through a coroutine wrapper.

    Args:
        value (Awaitable[T]): Awaitable value to resolve.

    Returns:
        T: Resolved value.
    """
    return await value
