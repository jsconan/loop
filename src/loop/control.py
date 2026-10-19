"""Provide cooperative session shutdown and run interruption."""

import asyncio
from collections.abc import Callable, Generator
from contextlib import contextmanager
from threading import Event, Lock
from typing import Any


class RunInterrupted(KeyboardInterrupt):
    """Return control from the active run without shutting down its conversation."""


class RunControl:
    """Coordinate shutdown and scoped interruption across execution and UI threads.

    Initialize an idle control. Shutdown stays sticky; interruption applies only to the current
    execution scope and is cleared when it settles. Cancellation-aware process supervisors receive
    a thread-safe event; other synchronous tools still finish under their existing deadlines.
    Steering is a separate safe-boundary signal: generation and the current tool finish, while
    later work in the same response is skipped.
    """

    _wake: Event
    _cancellation_event: Event
    _lock: Lock
    _stopped: bool
    _active: bool
    _interrupted: bool
    _steering_pending: bool
    _cancelled: bool
    _async_loop: asyncio.AbstractEventLoop | None
    _async_task: asyncio.Task[Any] | None
    _async_waiters: dict[asyncio.Event, asyncio.AbstractEventLoop]

    def __init__(self) -> None:
        self._wake = Event()
        self._cancellation_event = Event()
        self._lock = Lock()
        self._stopped = False
        self._active = False
        self._interrupted = False
        self._steering_pending = False
        self._cancelled = False
        self._async_loop = None
        self._async_task = None
        self._async_waiters = {}

    @property
    def stopped(self) -> bool:
        """Return whether conversation shutdown was requested.

        Returns:
            bool: Whether the sticky shutdown signal is set.
        """
        with self._lock:
            return self._stopped

    @property
    def active(self) -> bool:
        """Return whether an agent execution scope is active.

        Returns:
            bool: Whether interruption can target the current run.
        """
        with self._lock:
            return self._active

    @property
    def interrupted(self) -> bool:
        """Return whether the active run has an outstanding interrupt request.

        Returns:
            bool: Whether a run interruption awaits settlement.
        """
        with self._lock:
            return self._interrupted

    @property
    def steering_pending(self) -> bool:
        """Return whether accepted steering awaits the next safe execution boundary.

        Returns:
            bool: Whether an accepted steering message should stop further work in this run.
        """
        with self._lock:
            return self._steering_pending

    @property
    def cancelling(self) -> bool:
        """Return whether an active run is waiting for cancellation cleanup.

        Returns:
            bool: Whether shutdown or interruption has been requested for the active run.
        """
        with self._lock:
            return self._active and (self._stopped or self._interrupted)

    @property
    def cancelled(self) -> bool:
        """Return whether the most recent run settled after a cancellation request.

        Returns:
            bool: Whether the most recent execution scope completed cancellation cleanup.
        """
        with self._lock:
            return self._cancelled

    @property
    def cancellation_event(self) -> Event:
        """Return the run-scoped event set by shutdown or interruption requests.

        Returns:
            Event: Thread-safe signal for cancellable blocking operations.
        """
        return self._cancellation_event

    def stop(self) -> None:
        """Request conversation shutdown at the next safe boundary."""
        with self._lock:
            self._stopped = True
            self._cancellation_event.set()
            if self._active:
                self._cancelled = False
            self._wake.set()
            self._wake_async_waiters()
        self._cancel_async_owner()

    def interrupt(self) -> bool:
        """Request interruption of the current run while keeping the conversation open.

        Returns:
            bool: Whether an active run accepted the request. Idle or stopped controls reject it.
        """
        with self._lock:
            if not self._active or self._stopped:
                return False
            self._interrupted = True
            self._cancellation_event.set()
            self._cancelled = False
            self._wake.set()
            self._wake_async_waiters()
        self._cancel_async_owner()
        return True

    def request_steering(self, admit: Callable[[], bool]) -> bool:
        """Atomically admit steering while an active run publishes its safe-boundary signal.

        Args:
            admit (Callable[[], bool]): Nonblocking mailbox admission callback. It must not
                perform UI interaction or wait for another task.

        Returns:
            bool: Whether the active run admitted the message and recorded the signal.
        """
        with self._lock:
            if not self._active or self._stopped or not admit():
                return False
            self._steering_pending = True
            return True

    @contextmanager
    def execution(self, *, async_owner: bool = False) -> Generator[None]:
        """Own one run and clear scoped interruption and steering signals after cleanup.

        Args:
            async_owner (bool): Bind cancellation to the current task when it owns an async run.
                Defaults to ``False`` for synchronous execution.

        Yields:
            None: Control while the run owns its interrupt signal.

        Raises:
            RuntimeError: If another scope is active or async ownership lacks a current task.
        """
        loop = None
        task = None
        if async_owner:
            loop = asyncio.get_running_loop()
            task = asyncio.current_task(loop)
            if task is None:
                raise RuntimeError("Async execution ownership requires an asyncio task.")
        with self._lock:
            if self._active:
                raise RuntimeError("Run control already owns an active execution.")
            self._active = True
            self._cancelled = False
            self._async_loop = loop
            self._async_task = task
        try:
            yield
        finally:
            with self._lock:
                self._cancelled = self._stopped or self._interrupted
                self._active = False
                self._interrupted = False
                self._steering_pending = False
                self._async_loop = None
                self._async_task = None
                if not self._stopped:
                    self._wake.clear()
                    self._cancellation_event.clear()

    def checkpoint(self) -> None:
        """Observe shutdown before a scoped interrupt at a safe execution boundary.

        Raises:
            KeyboardInterrupt: If conversation shutdown was requested.
            RunInterrupted: If the active run was interrupted without shutdown.
        """
        with self._lock:
            if self._stopped:
                raise KeyboardInterrupt
            if self._interrupted:
                raise RunInterrupted

    def wait(self, seconds: float) -> None:
        """Wait for a retry delay or return promptly on either cancellation signal.

        Args:
            seconds (float): Maximum delay in seconds.

        Raises:
            KeyboardInterrupt: If conversation shutdown was requested.
            RunInterrupted: If the active run was interrupted.
        """
        self._wake.wait(seconds)
        self.checkpoint()

    async def wait_async(self, seconds: float) -> None:
        """Wait asynchronously for a retry delay or either cancellation signal.

        Args:
            seconds (float): Maximum delay in seconds.

        Raises:
            KeyboardInterrupt: If conversation shutdown was requested.
            RunInterrupted: If the active run was interrupted.
            asyncio.CancelledError: If the waiting task is cancelled.
        """
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()
        with self._lock:
            self._async_waiters[wake] = loop
            if self._stopped or self._interrupted:
                loop.call_soon(wake.set)
        try:
            try:
                await asyncio.wait_for(wake.wait(), timeout=seconds)
            except TimeoutError:
                pass
            self.checkpoint()
        finally:
            with self._lock:
                self._async_waiters.pop(wake, None)

    def _wake_async_waiters(self) -> None:
        """Schedule every registered async retry waiter on its owning event loop."""
        for wake, loop in tuple(self._async_waiters.items()):
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:
                self._async_waiters.pop(wake, None)

    def _cancel_async_owner(self) -> None:
        """Cancel the bound async run task without abandoning its cleanup path."""
        with self._lock:
            loop, task = self._async_loop, self._async_task
        if loop is not None and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                # A closed loop cannot accept cancellation; the scoped signal remains set so
                # synchronous checkpoints still stop the owner at its next safe boundary.
                pass
