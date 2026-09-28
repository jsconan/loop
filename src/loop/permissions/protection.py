"""Define protected workspace paths shared by permission and native policy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .. import constants


@dataclass(frozen=True)
class ProtectedWorkspacePaths:
    """Bind instruction filenames and control paths for command write protection.

    Args:
        instruction_names (tuple[str, ...]): Protected basenames at every workspace depth.
        directories (tuple[str, ...]): Protected workspace-relative directory paths.
        files (tuple[str, ...]): Protected workspace-relative file paths.
    """

    instruction_names: tuple[str, ...]
    directories: tuple[str, ...]
    files: tuple[str, ...]

    def protects_directory(self, relative: Path) -> bool:
        """Check whether a workspace-relative path lies in a protected directory.

        Args:
            relative (Path): Path relative to the selected workspace.

        Returns:
            bool: Whether a protected directory occurs at an applicable scope.
        """
        parts = relative.parts
        for directory in self.directories:
            protected = Path(directory).parts
            if len(protected) == 1:
                if protected[0] in parts:
                    return True
            elif any(
                parts[index : index + len(protected)] == protected
                for index in range(len(parts) - len(protected) + 1)
            ):
                return True
        return False


def protected_workspace_paths(configured: Iterable[str] = ()) -> ProtectedWorkspacePaths:
    """Build the active protected-path specification from trusted configuration.

    Args:
        configured (Iterable[str]): Active project instruction basenames.

    Returns:
        ProtectedWorkspacePaths: Names and relative paths denied to ordinary commands.

    Raises:
        ValueError: If an instruction name is not a plain filename.
    """
    names = tuple(
        dict.fromkeys(
            (constants.DEFAULT_AGENTS_FILENAME, constants.DEFAULT_SKILL_FILENAME, *configured)
        )
    )
    if any(
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        for name in names
    ):
        raise ValueError("Protected instruction names must be plain filenames.")
    return ProtectedWorkspacePaths(
        instruction_names=names,
        directories=(
            constants.GIT_DIRECTORY.as_posix(),
            constants.APP_DIRECTORY.as_posix(),
            constants.DEFAULT_SKILLS_DIRECTORY.as_posix(),
        ),
        files=(constants.GIT_IGNORE_FILENAME, constants.AGENT_IGNORE_FILENAME),
    )
