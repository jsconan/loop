"""Tests for dynamic mention-handler coordination."""

from unittest.mock import Mock

import pytest

from loop import CompletionAdapter, ContextReference, MentionHandler, MentionManager


def handler(marker, candidates, references=()):
    """Build one injectable handler mock."""
    result = Mock(spec=MentionHandler)
    result.marker = marker
    result.candidates.return_value = candidates
    result.resolve.return_value = references
    result.completion_adapter = Mock(spec=CompletionAdapter)
    return result


def test_manager_dispatches_only_exact_mentions_to_injected_handlers():
    """Handlers receive their own exact values and contribute context dynamically."""
    reference = ContextReference(
        kind="file", path="a.py", content="pass", size_bytes=4, included_bytes=4, truncated=False
    )
    files = handler("#", ("a.py",), (reference,))
    people = handler("+", ("sam",))
    manager = MentionManager((files, people))

    assert manager.resolve("Ask +sam about #a.py and #unknown") == (reference,)
    files.resolve.assert_called_once_with(("a.py",))
    people.resolve.assert_called_once_with(("sam",))
    assert manager.completion_adapters == (
        files.completion_adapter,
        people.completion_adapter,
    )


def test_manager_dispatches_each_target_once_in_first_mention_order():
    """The registry guarantees unique values even for handlers that do not deduplicate."""
    files = handler("@", ("my file.py", "other.py"))

    MentionManager((files,)).resolve("@my file.py @other.py @[my file.py]")

    files.resolve.assert_called_once_with(("my file.py", "other.py"))


def test_manager_skips_unmentioned_handlers_and_accepts_an_empty_registry():
    """An injected capability remains idle when its namespace is absent."""
    unused = handler("#", ("a.py",))

    assert not MentionManager((unused,)).resolve("plain text")
    unused.resolve.assert_not_called()
    assert not MentionManager().resolve("#a.py")


def test_manager_rejects_duplicate_handler_markers():
    """Ambiguous injected namespaces fail at registry construction."""
    with pytest.raises(ValueError, match="unique"):
        MentionManager((handler("#", ()), handler("#", ())))
