"""Tests for the command-line entry point."""

from unittest.mock import Mock, call

import pytest

import main
from loop.types import ShutdownRequested


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt, ShutdownRequested])
def test_main_gracefully_handles_shutdown_requests(monkeypatch, interruption):
    """Interrupts stop the CLI with a friendly message and no exception."""
    interaction = Mock()
    loop = Mock()
    loop.run.side_effect = interruption
    loop_factory = Mock(return_value=loop)
    register_shutdown_signals = Mock()
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "StreamingLoop", loop_factory)
    monkeypatch.setattr(main, "register_shutdown_signals", register_shutdown_signals)

    main.main()

    register_shutdown_signals.assert_called_once_with()
    loop_factory.assert_called_once_with(interaction=interaction)
    assert interaction.info.call_args_list == [
        call("Hello from loop!"),
        call("\nStopping loop. Goodbye!"),
    ]


def test_main_routes_startup_output_through_the_loop_interaction(monkeypatch):
    """A successful run displays startup output through its injected interaction."""
    interaction = Mock()
    loop = Mock()
    loop_factory = Mock(return_value=loop)
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "StreamingLoop", loop_factory)
    monkeypatch.setattr(main, "register_shutdown_signals", Mock())

    main.main()

    loop_factory.assert_called_once_with(interaction=interaction)
    interaction.info.assert_called_once_with("Hello from loop!")
    loop.run.assert_called_once_with()
