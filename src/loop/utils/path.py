"""Provide path discovery helpers."""

from pathlib import Path


def find_project_root(working_directory: Path | str) -> Path | None:
    """Return the closest Git project root containing the working directory.

    Args:
        working_directory: Directory from which to search upward.

    Returns:
        The closest directory containing a ``.git`` marker, or ``None`` when no
        project root is found.
    """
    working_directory = Path(working_directory)
    for directory in (working_directory, *working_directory.parents):
        if (directory / ".git").exists():
            return directory
    return None
