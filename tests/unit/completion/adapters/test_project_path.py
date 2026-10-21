"""Tests for project-path completion."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import get_ident
from unittest.mock import Mock

import pytest
from prompt_toolkit.document import Document

from loop import CompletionManager, ProjectPathCompletionAdapter
from loop.utils import PathHolder


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
    directory = PathHolder(current[0])
    completer = CompletionManager((ProjectPathCompletionAdapter("@", directory),))

    assert [item.text for item in complete(completer, "@app")] == ["@src/app.py"]
    (tmp_path / "new.py").write_text("", encoding="utf-8")
    assert complete(completer, "@new") == []
    now[0] += 5.0
    assert [item.text for item in complete(completer, "@new")] == ["@new.py"]
    assert complete(completer, "@ignored") == []
    directory.set(tmp_path / "missing")
    assert complete(completer, "@") == []


def test_project_path_completion_can_disable_cache_and_rejects_negative_ttl(tmp_path):
    """A zero TTL always refreshes paths while negative durations are invalid."""
    adapter = ProjectPathCompletionAdapter("@", PathHolder(tmp_path), cache_ttl=0)
    completer = CompletionManager((adapter,))

    assert complete(completer, "@new") == []
    (tmp_path / "new.py").write_text("", encoding="utf-8")
    assert [item.text for item in complete(completer, "@new")] == ["@new.py"]
    with pytest.raises(ValueError, match="cannot be negative"):
        ProjectPathCompletionAdapter("@", PathHolder(tmp_path), cache_ttl=-1)


def test_async_project_path_completion_reads_on_a_worker_and_updates_cache_on_owner(
    monkeypatch,
    tmp_path,
):
    """Async path discovery returns detached results and reuses them on the owner."""
    source = tmp_path / "source.py"
    source.write_text("", encoding="utf-8")
    adapter = ProjectPathCompletionAdapter("@", PathHolder(tmp_path))
    workers = []
    root_workers = []
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        """Record where the project root is canonicalized."""
        if path == tmp_path:
            root_workers.append(get_ident())
        return original_resolve(path, *args, **kwargs)

    def discover(root, *, recursive):
        """Record where path enumeration runs and return one visible file."""
        workers.append(get_ident())
        assert root == tmp_path
        assert recursive is True
        return [source]

    monkeypatch.setattr("loop.completion.adapters.project_path.iter_visible_paths", discover)
    monkeypatch.setattr(Path, "resolve", resolve)
    owner = get_ident()

    values = asyncio.run(adapter.values_async())

    assert [(item.value, item.description) for item in values] == [("source.py", "file")]
    assert workers and workers[0] != owner
    assert root_workers and root_workers[0] != owner
    assert asyncio.run(adapter.values_async()) == values
    assert adapter.values() == values
    assert len(workers) == 1


def test_project_path_completion_uses_injected_bounded_ui_worker(tmp_path):
    """Synchronous completion delegates filesystem traversal to its bounded worker."""
    (tmp_path / "source.py").write_text("", encoding="utf-8")
    calls = []

    def run_blocking(function, *args):
        """Record and invoke one completion operation through the injected boundary."""
        calls.append((function, args))
        return function(*args)

    adapter = ProjectPathCompletionAdapter(
        "@",
        PathHolder(tmp_path),
        sync_blocking_runner=run_blocking,
    )

    assert [value.value for value in adapter.values()] == ["source.py"]
    assert len(calls) == 1
    assert calls[0][1] == (tmp_path,)


@pytest.mark.parametrize("switch_on", [1, 2])
def test_async_project_path_completion_discards_results_after_root_change(tmp_path, switch_on):
    """A root switch during async completion cannot publish old-workspace paths."""
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    (old_root / "old.py").write_text("", encoding="utf-8")
    (new_root / "new.py").write_text("", encoding="utf-8")
    directory = PathHolder(old_root)
    calls = 0

    async def run_blocking(function, *args):
        """Run one operation and switch roots before its result returns to the owner."""
        nonlocal calls
        result = function(*args)
        calls += 1
        if calls == switch_on:
            directory.set(new_root)
        return result

    adapter = ProjectPathCompletionAdapter(
        "@",
        directory,
        async_blocking_runner=run_blocking,
    )

    assert asyncio.run(adapter.values_async()) == ()
    assert [value.value for value in asyncio.run(adapter.values_async())] == ["new.py"]


def test_async_project_path_completion_uses_injected_bounded_worker(tmp_path):
    """Async path completion delegates traversal to the configured application worker."""
    (tmp_path / "source.py").write_text("", encoding="utf-8")
    executor = ThreadPoolExecutor(max_workers=1)
    calls = []

    async def run_blocking(function, *args):
        """Submit detached path discovery to the controlled worker pool."""
        calls.append((function, args))
        return await asyncio.get_running_loop().run_in_executor(executor, function, *args)

    adapter = ProjectPathCompletionAdapter(
        "@",
        PathHolder(tmp_path),
        async_blocking_runner=run_blocking,
    )

    try:
        values = asyncio.run(adapter.values_async())
    finally:
        executor.shutdown(wait=True)

    assert [value.value for value in values] == ["source.py"]
    assert len(calls) == 2
    assert calls[0][1] == (tmp_path,)
    assert calls[1][1] == (tmp_path,)
