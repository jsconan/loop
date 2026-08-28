"""Complete visible project paths after a marker."""

import time
from collections.abc import Callable
from pathlib import Path

from ...utils.path import iter_visible_paths
from ..models import CompletionValue
from .marker import MarkerCompletionAdapter


class ProjectPathCompletionAdapter(MarkerCompletionAdapter):
    """Complete current visible project paths after a marker with a short-lived cache.

    Args:
        marker (str): Single symbol that activates path completion.
        working_directory (Path | Callable[[], Path]): Project root or lazy source of its current
            value.
        cache_ttl (float): Seconds to reuse a discovered path snapshot. Defaults to ``5.0``;
            use ``0`` to disable caching.

    Raises:
        ValueError: If ``cache_ttl`` is negative.
    """

    _working_directory: Callable[[], Path]
    _cache_ttl: float
    _cached_root: Path | None
    _cached_until: float
    _cached_values: tuple[CompletionValue, ...]

    def __init__(
        self,
        marker: str,
        working_directory: Path | Callable[[], Path],
        cache_ttl: float = 5.0,
    ) -> None:
        if cache_ttl < 0:
            raise ValueError("Path completion cache TTL cannot be negative.")
        self._working_directory = (
            working_directory if callable(working_directory) else lambda: working_directory
        )
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
        root = self._working_directory().resolve()
        now = time.monotonic()
        if root == self._cached_root and now < self._cached_until:
            return self._cached_values
        if not root.is_dir():
            values = ()
        else:
            discovered = []
            for path in iter_visible_paths(root, recursive=True):
                relative = path.relative_to(root).as_posix()
                if path.is_dir():
                    relative += "/"
                discovered.append(
                    CompletionValue(relative, "directory" if path.is_dir() else "file")
                )
            values = tuple(discovered)
        self._cached_root = root
        self._cached_until = time.monotonic() + self._cache_ttl
        self._cached_values = values
        return values
