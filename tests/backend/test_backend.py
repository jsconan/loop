"""Tests for the backend capability contract."""

from loop import Backend, OpenAIBackend


def test_openai_adapter_satisfies_the_complete_backend_contract():
    """The OpenAI adapter exposes every required backend capability."""
    assert isinstance(OpenAIBackend(), Backend)
