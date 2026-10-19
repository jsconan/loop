"""Provide cooperative session shutdown and run interruption."""

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from threading import Event, Lock


class RunInterrupted(KeyboardInterrupt):
    """Return control from the active run without shutting down its conversation."""


class RunControl:
    """Coordinate shutdown and scoped interruption across execution and UI threads.

    Initialize an idle control. Shutdown stays sticky; interruption applies only to the current
    execution scope and is cleared when it settles. Active synchronous tools finish under their
    existing deadlines before a checkpoint observes either signal.
    """

    _wake: Event
    _lock: Lock
    _stopped: bool
    _active: bool
    _interrupted: bool
    _async_waiters: dict[asyncio.Event, asyncio.AbstractEventLoop]

    def __init__(self) -> None:
        self._wake = Event()
        self._lock = Lock()
        self._stopped = False
        self._active = False
        self._interrupted = False
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

    def stop(self) -> None:
        """Request conversation shutdown at the next safe boundary."""
        with self._lock:
            self._stopped = True
            self._wake.set()
            self._wake_async_waiters()

    def interrupt(self) -> bool:
        """Request interruption of the current run while keeping the conversation open.

        Returns:
            bool: Whether an active run accepted the request. Idle or stopped controls reject it.
        """
        with self._lock:
            if not self._active or self._stopped:
                return False
            self._interrupted = True
            self._wake.set()
            self._wake_async_waiters()
            return True

    @contextmanager
    def execution(self) -> Generator[None]:
        """Own one interruptible run and clear its interrupt after cleanup.

        Yields:
            None: Control while the run owns its interrupt signal.

        Raises:
            RuntimeError: If another execution scope is active.
        """
        with self._lock:
            if self._active:
                raise RuntimeError("Run control already owns an active execution.")
            self._active = True
        try:
            yield
        finally:
            with self._lock:
                self._active = False
                self._interrupted = False
                if not self._stopped:
                    self._wake.clear()

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
