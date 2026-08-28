"""Tests for project-path completion."""

from unittest.mock import Mock

import pytest
from prompt_toolkit.document import Document

from loop import CompletionManager, ProjectPathCompletionAdapter


def complete(completer: CompletionManager, text: str):
    """Return all completions produced for text with its cursor at the end."""
    return list(completer.get_completions(Document(text), Mock()))


def test_project_path_completion_caches_paths_until_ttl_expires(monkeypatch, tmp_path):
    """Project path completion reuses short-lived snapshots and refreshes after expiry."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    current = [tmp_path]
    now = [10.0]
    monkeypatch.setattr("loop.completion.adapters.project_path.time.monotonic", lambda: now[0])
    completer = CompletionManager((ProjectPathCompletionAdapter("@", lambda: current[0]),))

    assert [item.text for item in complete(completer, "@app")] == ["@src/app.py"]
    (tmp_path / "new.py").write_text("", encoding="utf-8")
    assert complete(completer, "@new") == []
    now[0] += 5.0
    assert [item.text for item in complete(completer, "@new")] == ["@new.py"]
    assert complete(completer, "@ignored") == []
    current[0] = tmp_path / "missing"
    assert complete(completer, "@") == []


def test_project_path_completion_can_disable_cache_and_rejects_negative_ttl(tmp_path):
    """A zero TTL always refreshes paths while negative durations are invalid."""
    adapter = ProjectPathCompletionAdapter("@", tmp_path, cache_ttl=0)
    completer = CompletionManager((adapter,))

    assert complete(completer, "@new") == []
    (tmp_path / "new.py").write_text("", encoding="utf-8")
    assert [item.text for item in complete(completer, "@new")] == ["@new.py"]
    with pytest.raises(ValueError, match="cannot be negative"):
        ProjectPathCompletionAdapter("@", tmp_path, cache_ttl=-1)
