"""Complete visible project paths after a marker."""

import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ...utils import PathReference, run_in_worker
from ...utils.path import iter_visible_paths
from ..models import CompletionValue
from .marker import MarkerCompletionAdapter


class ProjectPathCompletionAdapter(MarkerCompletionAdapter):
    """Complete current visible project paths after a marker with a short-lived cache.

    Args:
        marker (str): Single symbol that activates path completion.
        working_directory (PathReference): Current project-directory reference.
        cache_ttl (float): Seconds to reuse a discovered path snapshot. Defaults to ``5.0``;
            use ``0`` to disable caching.
        sync_blocking_runner (Callable[..., Any] | None): Bounded worker for synchronous UI
            completion. Defaults to direct traversal for standalone adapters.
        async_blocking_runner (Callable[..., Awaitable[Any]] | None): Bounded worker for async
            filesystem traversal. Defaults to a shared bounded process worker.

    Raises:
        ValueError: If ``cache_ttl`` is negative.
    """

    _working_directory: PathReference
    _cache_ttl: float
    _cached_root: Path | None
    _cached_until: float
    _cached_values: tuple[CompletionValue, ...]
    _sync_blocking_runner: Callable[..., Any] | None
    _async_blocking_runner: Callable[..., Awaitable[Any]] | None

    def __init__(
        self,
        marker: str,
        working_directory: PathReference,
        cache_ttl: float = 5.0,
        *,
        sync_blocking_runner: Callable[..., Any] | None = None,
        async_blocking_runner: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        if cache_ttl < 0:
            raise ValueError("Path completion cache TTL cannot be negative.")
        self._working_directory = working_directory
        self._sync_blocking_runner = sync_blocking_runner
        self._async_blocking_runner = async_blocking_runner
        self._cache_ttl = cache_ttl
        self._cached_root = None
        self._cached_until = 0.0
        self._cached_values = ()
        super().__init__(marker, self.values)

    def values(self) -> tuple[CompletionValue, ...]:
        """Return a current snapshot of visible project paths.

        Returns:
            tuple[CompletionValue, ...]: Cached or newly discovered project-relative paths.
        """
        root = self._working_directory.resolve()
        now = time.monotonic()
        if root == self._cached_root and now < self._cached_until:
            return self._cached_values
        values = (
            self._sync_blocking_runner(self._discover_values, root)
            if self._sync_blocking_runner is not None
            else self._discover_values(root)
        )
        self._cached_root = root
        self._cached_until = time.monotonic() + self._cache_ttl
        self._cached_values = values
        return values

    async def values_async(self) -> tuple[CompletionValue, ...]:
        """Resolve and list visible paths through the bounded worker.

        Returns:
            tuple[CompletionValue, ...]: Cached or newly discovered project-relative paths.

        Raises:
            OSError: The project path listing cannot be read.
            RuntimeError: Bounded worker admission is full or unavailable.
            asyncio.CancelledError: The awaiting task is cancelled.
        """
        root_source = self._working_directory.get()
        root = await (
            self._async_blocking_runner(Path.resolve, root_source)
            if self._async_blocking_runner is not None
            else run_in_worker(Path.resolve, root_source)
        )
        if self._working_directory.get() != root_source:
            return ()
        now = time.monotonic()
        if root == self._cached_root and now < self._cached_until:
            return self._cached_values
        values = await (
            self._async_blocking_runner(self._discover_values, root)
            if self._async_blocking_runner is not None
            else run_in_worker(self._discover_values, root)
        )
        if self._working_directory.get() != root_source:
            return ()
        self._cached_root = root
        self._cached_until = time.monotonic() + self._cache_ttl
        self._cached_values = values
        return values

    @staticmethod
    def _discover_values(root: Path) -> tuple[CompletionValue, ...]:
        """Build a detached snapshot of visible paths under one resolved root."""
        if not root.is_dir():
            return ()
        discovered = []
        for path in iter_visible_paths(root, recursive=True):
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                relative += "/"
            discovered.append(CompletionValue(relative, "directory" if path.is_dir() else "file"))
        return tuple(discovered)
