"""Tests for the command-line entry point."""

from unittest.mock import Mock

import pytest

import main
from loop.types import ShutdownRequested


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, ShutdownRequested])
def test_main_gracefully_handles_shutdown_requests(monkeypatch, capsys, interruption):
    """Interrupts stop the CLI with a friendly message and no exception."""
    loop = Mock()
    loop.run.side_effect = interruption
    register_shutdown_signals = Mock()
    monkeypatch.setattr(main, "StreamingLoop", Mock(return_value=loop))
    monkeypatch.setattr(main, "register_shutdown_signals", register_shutdown_signals)

    main.main()

    assert "Stopping loop. Goodbye!" in capsys.readouterr().out
    register_shutdown_signals.assert_called_once_with()
