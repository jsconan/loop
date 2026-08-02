"""Provide utilities for handling process signals."""

import signal


class ShutdownRequested(Exception):
    """Indicate that the process received a termination signal."""


def register_shutdown_signals() -> None:
    """Register controlled shutdown handling for supported termination signals."""

    def _request_shutdown(_signum: int, _frame: object | None) -> None:

        raise ShutdownRequested()

    for signal_name in ("SIGTERM", "SIGHUP", "SIGQUIT"):
        if shutdown_signal := getattr(signal, signal_name, None):
            signal.signal(shutdown_signal, _request_shutdown)
