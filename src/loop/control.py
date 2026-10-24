"""Provide cooperative session shutdown and run interruption."""

import asyncio
from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import StrEnum
from threading import Event, Lock
from typing import Any


class RunInterrupted(KeyboardInterrupt):
    """Return control from the active run without shutting down its conversation."""


class RunStopRequested(Exception):
    """Return control from an active run while preserving its durable recovery boundary."""


class RunControlState(StrEnum):
    """Describe the current scoped run state."""

    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    STOP_REQUESTED = "stop_requested"


class RunControl:
    """Coordinate shutdown and scoped interruption across execution and UI threads.

    Initialize an idle control. Shutdown stays sticky; interruption and pause apply only to the
    current execution scope and clear when it settles. A pause preserves the durable run boundary
    for later recovery. Cancellation-aware process supervisors receive a thread-safe event; other
    synchronous tools still finish under their existing deadlines. Steering is a separate
    safe-boundary signal: generation and the current tool finish, while later work in the same
    response is skipped.
    """

    _wake: Event
    _resume: Event
    _cancellation_event: Event
    _lock: Lock
    _stopped: bool
    _state: RunControlState
    _steering_pending: bool
    _cancelled: bool
    _async_loop: asyncio.AbstractEventLoop | None
    _async_task: asyncio.Task[Any] | None
    _async_waiters: dict[asyncio.Event, asyncio.AbstractEventLoop]
    _active_waiters: dict[asyncio.Event, asyncio.AbstractEventLoop]
    _checkpoint_waiters: dict[asyncio.Event, asyncio.AbstractEventLoop]

    def __init__(self) -> None:
        self._wake = Event()
        self._resume = Event()
        self._resume.set()
        self._cancellation_event = Event()
        self._lock = Lock()
        self._stopped = False
        self._state = RunControlState.IDLE
        self._steering_pending = False
        self._cancelled = False
        self._async_loop = None
        self._async_task = None
        self._async_waiters = {}
        self._active_waiters = {}
        self._checkpoint_waiters = {}

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
            bool: Whether the current run has not settled its execution scope.
        """
        with self._lock:
            return self._state != RunControlState.IDLE

    @property
    def interrupted(self) -> bool:
        """Return whether the active run has an outstanding interrupt request.

        Returns:
            bool: Whether a run interruption awaits settlement.
        """
        with self._lock:
            return self._state == RunControlState.INTERRUPTED

    @property
    def paused(self) -> bool:
        """Return whether the active run was asked to preserve a resumable boundary.

        Returns:
            bool: Whether the current interruption represents a pause.
        """
        with self._lock:
            return self._state == RunControlState.PAUSED

    @property
    def stop_requested(self) -> bool:
        """Return whether the active run should settle at its next safe boundary.

        Returns:
            bool: Whether a cold stop request is pending for the current run.
        """
        with self._lock:
            return self._state == RunControlState.STOP_REQUESTED

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
            return self._state != RunControlState.IDLE and (
                self._state == RunControlState.INTERRUPTED or self._stopped
            )

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
            was_active = self._state != RunControlState.IDLE
            self._stopped = True
            self._cancellation_event.set()
            if was_active:
                self._cancelled = False
            self._resume.set()
            self._wake.set()
            self._wake_async_waiters()
            self._wake_active_waiters()
            self._wake_checkpoint_waiters()
        self._cancel_async_owner()

    def interrupt(self) -> bool:
        """Request interruption of the current run while keeping the conversation open.

        Returns:
            bool: Whether an active run accepted the request. Idle or stopped controls reject it.
        """
        with self._lock:
            if self._stopped:
                return False
            if self._state == RunControlState.INTERRUPTED:
                return True
            if self._state not in {RunControlState.ACTIVE, RunControlState.PAUSED}:
                return False
            self._state = RunControlState.INTERRUPTED
            self._cancellation_event.set()
            self._cancelled = False
            self._resume.set()
            self._wake.set()
            self._wake_async_waiters()
            self._wake_checkpoint_waiters()
        self._cancel_async_owner()
        return True

    def pause(self) -> bool:
        """Park the current run at its next cooperative checkpoint.

        Returns:
            bool: Whether an active run accepted the pause request.
        """
        with self._lock:
            if self._state == RunControlState.PAUSED:
                return True
            if self._state != RunControlState.ACTIVE or self._stopped:
                return False
            self._state = RunControlState.PAUSED
            self._resume.clear()
            self._wake.set()
            self._wake_async_waiters()
            self._wake_checkpoint_waiters()
        return True

    def continue_run(self) -> bool:
        """Resume a run parked at a cooperative checkpoint.

        Returns:
            bool: Whether a live paused run was resumed.
        """
        with self._lock:
            if self._state != RunControlState.PAUSED:
                return False
            self._state = RunControlState.ACTIVE
            self._resume.set()
            self._wake.set()
            self._wake_checkpoint_waiters()
        return True

    def request_stop_run(self) -> bool:
        """Settle the active run at its next checkpoint without shutting down its owner.

        Returns:
            bool: Whether a live run accepted the cold-stop request.
        """
        with self._lock:
            if self._state == RunControlState.STOP_REQUESTED:
                return True
            if self._state not in {RunControlState.ACTIVE, RunControlState.PAUSED} or self._stopped:
                return False
            self._state = RunControlState.STOP_REQUESTED
            self._resume.set()
            self._wake.set()
            self._wake_async_waiters()
            self._wake_checkpoint_waiters()
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
            if self._state == RunControlState.IDLE or self._stopped or not admit():
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
            if self._state != RunControlState.IDLE:
                raise RuntimeError("Run control already owns an active execution.")
            self._state = RunControlState.ACTIVE
            self._cancelled = False
            self._resume.set()
            self._async_loop = loop
            self._async_task = task
            self._wake_active_waiters()
        try:
            yield
        finally:
            with self._lock:
                self._cancelled = self._stopped or self._state == RunControlState.INTERRUPTED
                self._state = RunControlState.IDLE
                self._steering_pending = False
                self._async_loop = None
                self._async_task = None
                if not self._stopped:
                    self._wake.clear()
                    self._cancellation_event.clear()

    def checkpoint(self) -> None:
        """Observe shutdown, interruption, pause, and stop at a safe execution boundary.

        Raises:
            KeyboardInterrupt: If conversation shutdown was requested.
            RunInterrupted: If the active run was interrupted without shutdown.
            RunStopRequested: If the run received a cold-stop request.
        """
        while True:
            with self._lock:
                if self._stopped:
                    raise KeyboardInterrupt
                if self._state == RunControlState.INTERRUPTED:
                    raise RunInterrupted
                if self._state == RunControlState.STOP_REQUESTED:
                    raise RunStopRequested
                if self._state != RunControlState.PAUSED:
                    return
            self._resume.wait()

    async def checkpoint_async(self) -> None:
        """Wait asynchronously through a pause and observe other run control states.

        Raises:
            KeyboardInterrupt: If conversation shutdown was requested.
            RunInterrupted: If the active run was interrupted.
            RunStopRequested: If the run received a cold-stop request.
            asyncio.CancelledError: If the waiting task is cancelled.
        """
        while True:
            loop = asyncio.get_running_loop()
            wake = asyncio.Event()
            with self._lock:
                if self._stopped:
                    raise KeyboardInterrupt
                if self._state == RunControlState.INTERRUPTED:
                    raise RunInterrupted
                if self._state == RunControlState.STOP_REQUESTED:
                    raise RunStopRequested
                if self._state != RunControlState.PAUSED:
                    return
                self._checkpoint_waiters[wake] = loop
            try:
                await wake.wait()
            finally:
                with self._lock:
                    self._checkpoint_waiters.pop(wake, None)

    def wait(self, seconds: float) -> None:
        """Wait for a retry delay or return promptly on either cancellation signal.

        Args:
            seconds (float): Maximum delay in seconds.

        Raises:
            KeyboardInterrupt: If conversation shutdown was requested.
            RunInterrupted: If the active run was interrupted.
            RunStopRequested: If the active run received a cold-stop request.
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
            RunStopRequested: If the active run received a cold-stop request.
            asyncio.CancelledError: If the waiting task is cancelled.
        """
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()
        with self._lock:
            self._async_waiters[wake] = loop
            if self._stopped or self._state == RunControlState.INTERRUPTED:
                loop.call_soon(wake.set)
        try:
            try:
                await asyncio.wait_for(wake.wait(), timeout=seconds)
            except TimeoutError:
                pass
            await self.checkpoint_async()
        finally:
            with self._lock:
                self._async_waiters.pop(wake, None)

    async def wait_until_active_async(self) -> bool:
        """Wait until a run starts or shutdown is requested.

        Returns:
            bool: Whether a run became active before shutdown.

        Raises:
            asyncio.CancelledError: If the waiting task is cancelled.
        """
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()
        with self._lock:
            if self._stopped:
                return False
            if self._state != RunControlState.IDLE:
                return True
            self._active_waiters[wake] = loop
        try:
            await wake.wait()
        finally:
            with self._lock:
                self._active_waiters.pop(wake, None)
        with self._lock:
            return not self._stopped

    def _wake_async_waiters(self) -> None:
        """Schedule every registered async retry waiter on its owning event loop."""
        self._wake_waiters(self._async_waiters)

    def _wake_active_waiters(self) -> None:
        """Notify async observers that run activity or shutdown state may have changed."""
        self._wake_waiters(self._active_waiters)

    def _wake_checkpoint_waiters(self) -> None:
        """Notify async checkpoints that the scoped run state may have changed."""
        self._wake_waiters(self._checkpoint_waiters)

    @staticmethod
    def _wake_waiters(waiters: dict[asyncio.Event, asyncio.AbstractEventLoop]) -> None:
        """Schedule registered async waiters and discard registrations for closed loops."""
        for wake, loop in tuple(waiters.items()):
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:
                waiters.pop(wake, None)

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
