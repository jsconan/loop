"""Provide cooperative cancellation for serialized agent execution."""

from threading import Event


class RunControl:
    """Coordinate a stop request across execution and interaction threads.

    Initialize an unset stop signal. Once stopped, this control stays stopped; create a new
    control for a new execution lifecycle. Active synchronous tools finish under their existing
    execution deadlines before the next checkpoint observes cancellation.
    """

    _stopped: Event

    def __init__(self) -> None:
        self._stopped = Event()

    @property
    def stopped(self) -> bool:
        """Return whether execution has been asked to stop.

        Returns:
            bool: Whether a stop was requested.
        """
        return self._stopped.is_set()

    def stop(self) -> None:
        """Request that execution stop at its next safe boundary."""
        self._stopped.set()

    def checkpoint(self) -> None:
        """Stop execution when cancellation has been requested.

        Raises:
            KeyboardInterrupt: If execution has been asked to stop.
        """
        if self._stopped.is_set():
            raise KeyboardInterrupt

    def wait(self, seconds: float) -> None:
        """Wait for a retry delay or stop promptly on cancellation.

        Args:
            seconds (float): Maximum delay in seconds.

        Raises:
            KeyboardInterrupt: If cancellation is requested before the delay ends.
        """
        self._stopped.wait(seconds)
        self.checkpoint()
