"""Validate the virtual-relative namespace used by the workspace VFS."""

from __future__ import annotations

from pathlib import PurePosixPath


def validate_relative_path(value: str) -> str:
    """Return one normalized relative virtual path.

    Args:
        value (str): Workspace-relative POSIX path, with no host-path representation.

    Returns:
        str: The validated path.

    Raises:
        ValueError: The path is empty, absolute, or escapes the virtual workspace.
    """
    if not value or value.startswith("/") or "\\" in value or "//" in value:
        raise ValueError("Workspace paths must be normalized relative virtual paths.")
    path = PurePosixPath(value)
    if path == PurePosixPath(".") or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Workspace paths must not contain traversal components.")
    return path.as_posix()


def validate_relative_path_or_none(value: str | None) -> str | None:
    """Validate an optional relative virtual path.

    Args:
        value (str | None): Optional workspace-relative POSIX path.

    Returns:
        str | None: The validated path, or ``None``.
    """
    return validate_relative_path(value) if value is not None else None
