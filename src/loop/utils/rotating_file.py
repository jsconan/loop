"""Append owner-private text files with optional size-based rotation."""

from __future__ import annotations

import threading
from pathlib import Path

from filelock import FileLock

from .. import constants


class PrivateRotatingTextFile:
    """Append UTF-8 text to a private file with optional numbered archives.

    Args:
        path (Path | str): Active file path.
        max_bytes (int | None): Maximum active-file size before the next append rotates it.
            ``None`` disables rotation.
        backup_count (int): Number of numbered archives to retain. A value of zero disables
            rotation.

    Raises:
        ValueError: If a size or backup count is negative.
    """

    _path: Path
    _max_bytes: int | None
    _backup_count: int
    _lock: threading.Lock
    _process_lock: FileLock

    def __init__(
        self,
        path: Path | str,
        *,
        max_bytes: int | None = None,
        backup_count: int = 0,
    ) -> None:
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be non-negative or None.")
        if backup_count < 0:
            raise ValueError("backup_count must be non-negative.")
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._lock = threading.Lock()
        self._process_lock = FileLock(f"{self._path}.lock")

    def append(self, text: str) -> None:
        """Append complete text after applying the configured retention policy.

        Args:
            text (str): UTF-8 text to append as one unit.

        Raises:
            OSError: If preparing, rotating, or writing the file fails.
        """
        encoded_size = len(text.encode("utf-8"))
        with self._lock, self._process_lock:
            self._prepare_parent()
            Path(self._process_lock.lock_file).chmod(constants.PRIVATE_FILE_MODE)
            if self._should_rotate(encoded_size):
                self._rotate()
            with self._path.open("a", encoding="utf-8") as output:
                output.write(text)
            self._path.chmod(constants.PRIVATE_FILE_MODE)

    def prepare(self) -> None:
        """Create and secure an empty active file.

        Raises:
            OSError: If preparing the destination fails.
        """
        with self._lock, self._process_lock:
            self._prepare_parent()
            Path(self._process_lock.lock_file).chmod(constants.PRIVATE_FILE_MODE)
            self._path.touch(exist_ok=True)
            self._path.chmod(constants.PRIVATE_FILE_MODE)

    def _prepare_parent(self) -> None:
        """Create and secure the destination directory."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.parent.chmod(constants.PRIVATE_DIRECTORY_MODE)

    def _should_rotate(self, encoded_size: int) -> bool:
        """Return whether the next complete append exceeds the active-file limit."""
        return (
            self._max_bytes is not None
            and self._max_bytes > 0
            and self._backup_count > 0
            and self._path.exists()
            and self._path.stat().st_size + encoded_size > self._max_bytes
        )

    def _rotate(self) -> None:
        """Shift numbered archives and preserve their private file modes."""
        oldest_backup = self._archive_path(self._backup_count)
        oldest_backup.unlink(missing_ok=True)
        for backup_index in range(self._backup_count - 1, 0, -1):
            backup_path = self._archive_path(backup_index)
            if backup_path.exists():
                backup_path.replace(self._archive_path(backup_index + 1))
                self._archive_path(backup_index + 1).chmod(constants.PRIVATE_FILE_MODE)
        self._path.replace(self._archive_path(1))
        self._archive_path(1).chmod(constants.PRIVATE_FILE_MODE)

    def _archive_path(self, index: int) -> Path:
        """Return the path for one numbered archive."""
        return self._path.with_name(f"{self._path.name}.{index}")
