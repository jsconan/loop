"""Tests for the completion adapter contract."""

from unittest.mock import Mock

from loop import CompletionAdapter, CompletionMatch


def test_adapter_declarations_default_to_no_activation_or_values():
    """Base adapters expose inert default declarations and require completion behavior."""

    class KeywordAdapter(CompletionAdapter):
        """Provide a declaration-only test adapter."""

        def match(self, document):
            """Remain inactive for every document."""
            return

        def complete(self, match):
            """Return no completion values."""
            return ()

    adapter = KeywordAdapter()

    assert adapter.front_markers == ()
    assert adapter.keywords == ()
    assert adapter.match(Mock()) is None
    assert adapter.complete(CompletionMatch("", "")) == ()
