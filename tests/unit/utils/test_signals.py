"""Tests for process signal utilities."""

import signal
from unittest.mock import Mock

import pytest

from loop.utils import ShutdownRequested, register_shutdown_signals, signals


def test_termination_signal_requests_shutdown(monkeypatch):
    """A registered termination handler raises the controlled shutdown exception."""
    register = Mock()
    monkeypatch.setattr(signals.signal, "signal", register)

    register_shutdown_signals()

    handler = register.call_args_list[0].args[1]
    with pytest.raises(ShutdownRequested):
        handler(signal.SIGTERM, None)


def test_register_shutdown_signals_registers_supported_termination_signals(monkeypatch):
    """Catchable platform termination signals use the shutdown handler."""
    register = Mock()
    monkeypatch.setattr(signals.signal, "signal", register)

    register_shutdown_signals()

    expected_signals = [
        getattr(signal, name) for name in ("SIGTERM", "SIGHUP", "SIGQUIT") if hasattr(signal, name)
    ]
    assert register.call_count == len(expected_signals)
    handlers = {call.args[1] for call in register.call_args_list}
    assert len(handlers) == 1


def test_register_shutdown_signals_skips_unsupported_signals(monkeypatch):
    """Signals unavailable on the current platform are ignored."""
    register = Mock()
    monkeypatch.delattr(signals.signal, "SIGHUP", raising=False)
    monkeypatch.setattr(signals.signal, "signal", register)

    register_shutdown_signals()

    expected_signals = [
        getattr(signal, name) for name in ("SIGTERM", "SIGQUIT") if hasattr(signal, name)
    ]
    assert [call.args[0] for call in register.call_args_list] == expected_signals
