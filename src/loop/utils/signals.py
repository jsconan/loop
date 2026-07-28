"""Provide utilities for handling process signals."""

import signal

from ..types import ShutdownRequested


def request_shutdown(_signum: int, _frame: object | None) -> None:
    """Convert a termination signal into a controlled shutdown request."""
    raise ShutdownRequested


def register_shutdown_signals() -> None:
    """Register controlled shutdown handling for supported termination signals."""
    for signal_name in ("SIGTERM", "SIGHUP", "SIGQUIT"):
        if shutdown_signal := getattr(signal, signal_name, None):
            signal.signal(shutdown_signal, request_shutdown)
