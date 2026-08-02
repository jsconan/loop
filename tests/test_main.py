"""Tests for the command-line entry point."""

from unittest.mock import Mock, call

import pytest

import main
from loop import ShutdownRequested


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt, ShutdownRequested])
def test_main_gracefully_handles_shutdown_requests(monkeypatch, interruption):
    """Interrupts stop the CLI with a friendly message and no exception."""
    interaction = Mock()
    loop = Mock()
    loop.run.side_effect = interruption
    loop_factory = Mock(return_value=loop)
    backend = Mock()
    backend_factory = Mock(return_value=backend)
    register_shutdown_signals = Mock()
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "Loop", loop_factory)
    monkeypatch.setattr(main, "OpenAIBackend", backend_factory)
    monkeypatch.setattr(main, "register_shutdown_signals", register_shutdown_signals)
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CONTEXT_WINDOW", raising=False)

    main.main()

    register_shutdown_signals.assert_called_once_with()
    backend_factory.assert_called_once_with(
        base_url="http://localhost:8000/v1",
        default_model="nvidia/Qwen3.6-35B-A3B-NVFP4",
        api_key="local-api-key",
        context_window=None,
    )
    loop_factory.assert_called_once_with(backend, interaction=interaction, stream=True)
    assert interaction.info.call_args_list == [
        call("Hello from loop!"),
        call("\nStopping loop. Goodbye!"),
    ]


def test_main_routes_startup_output_through_the_loop_interaction(monkeypatch):
    """A successful run displays startup output through its injected interaction."""
    interaction = Mock()
    loop = Mock()
    loop_factory = Mock(return_value=loop)
    backend = Mock()
    backend_factory = Mock(return_value=backend)
    monkeypatch.setattr(main, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(main, "Loop", loop_factory)
    monkeypatch.setattr(main, "OpenAIBackend", backend_factory)
    monkeypatch.setattr(main, "register_shutdown_signals", Mock())
    monkeypatch.setenv("BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("DEFAULT_MODEL", "configured-model")
    monkeypatch.setenv("OPENAI_API_KEY", "configured-key")
    monkeypatch.setenv("CONTEXT_WINDOW", "32768")

    main.main()

    backend_factory.assert_called_once_with(
        base_url="https://example.test/v1",
        default_model="configured-model",
        api_key="configured-key",
        context_window=32768,
    )
    loop_factory.assert_called_once_with(backend, interaction=interaction, stream=True)
    interaction.info.assert_called_once_with("Hello from loop!")
    loop.run.assert_called_once_with()
