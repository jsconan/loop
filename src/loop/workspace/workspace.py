"""Discover workspaces and expose their durable artifact locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import constants
from ..utils import find_project_root


@dataclass(frozen=True, slots=True)
class WorkspaceStorage:
    """Expose every durable artifact owned by one workspace.

    Args:
        root (Path): Directory containing the workspace's application artifacts.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())

    @property
    def configuration(self) -> Path:
        """Return the workspace configuration path.

        Returns:
            Path: Workspace TOML configuration path.
        """
        return self.root / constants.APP_CONFIGURATION_FILENAME

    @property
    def sessions(self) -> Path:
        """Return the workspace session database path.

        Returns:
            Path: Workspace session database path.
        """
        return self.root / constants.SESSION_DATABASE_FILENAME

    @property
    def telemetry(self) -> Path:
        """Return the workspace telemetry database path.

        Returns:
            Path: Workspace telemetry database path.
        """
        return self.root / constants.TELEMETRY_DATABASE_FILENAME

    @property
    def operational_log(self) -> Path:
        """Return the workspace operational log path.

        Returns:
            Path: Workspace operational log path.
        """
        return self.root / constants.OPERATIONAL_LOG_FILENAME

    @property
    def permissions(self) -> Path:
        """Return the workspace permission policy path.

        Returns:
            Path: Workspace permission policy path.
        """
        return self.root / constants.PERMISSIONS_FILENAME

    @property
    def permissions_audit(self) -> Path:
        """Return the workspace permission audit path.

        Returns:
            Path: Workspace permission audit path.
        """
        return self.root / constants.PERMISSIONS_AUDIT_FILENAME


@dataclass(frozen=True, slots=True)
class Workspace:
    """Represent one worktree and its active and durable locations.

    Args:
        root (Path): Canonical workspace or Git worktree root.
        working_directory (Path): Canonical active directory within the workspace.
        storage (WorkspaceStorage): Durable artifact locations owned by the workspace.
    """

    root: Path
    working_directory: Path
    storage: WorkspaceStorage

    def __post_init__(self) -> None:
        root = self.root.resolve()
        working_directory = self.working_directory.resolve()
        if working_directory != root and root not in working_directory.parents:
            raise ValueError("Workspace working directory must be within its root.")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "working_directory", working_directory)

    @classmethod
    def discover(cls, working_directory: Path | str) -> Workspace:
        """Discover the workspace containing an active directory.

        Args:
            working_directory (Path | str): Directory from which the application was started.

        Returns:
            Workspace: Discovered worktree and its local artifact locations.
        """
        directory = Path(working_directory).resolve()
        root = find_project_root(directory) or directory
        return cls(
            root=root,
            working_directory=directory,
            storage=WorkspaceStorage(root / constants.APP_DIRECTORY),
        )
