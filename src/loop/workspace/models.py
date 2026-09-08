"""Represent discovered and initialized workspaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..utils import PathHolder, find_project_root

WorkspaceNameSource = Literal["user", "provider", "remote", "directory", "default"]


class WorkspaceSwitchRequested(Exception):
    """Request a composition-root rebuild for a resolved workspace.

    Args:
        workspace (Workspace): Initialized target workspace.
        on_complete (Callable[[], None] | None): Callback releasing the switch lifecycle token.
    """

    workspace: Workspace
    _on_complete: Callable[[], None] | None

    def __init__(
        self,
        workspace: Workspace,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(f"Switch to workspace {workspace.id}")
        self.workspace = workspace
        self._on_complete = on_complete

    def complete(self) -> None:
        """Release the owned switch lifecycle token exactly once."""
        callback = self._on_complete
        self._on_complete = None
        if callable(callback):
            callback()


@dataclass(slots=True)
class Workspace:
    """Represent one worktree and its durable identity.

    Args:
        root (Path): Canonical workspace or Git worktree root.
        working_directory (Path | PathHolder): Mutable canonical active-directory reference.
        id (str | None): Durable workspace identifier after repository initialization.
        name (str | None): Human-readable workspace name after initialization.
        name_source (WorkspaceNameSource | None): Origin of the initialized name.
        created_at_ns (int | None): Workspace creation time after initialization.
        updated_at_ns (int | None): Last workspace metadata update after initialization.
    """

    root: Path
    working_directory: Path | PathHolder
    id: str | None = None
    name: str | None = None
    name_source: WorkspaceNameSource | None = None
    created_at_ns: int | None = None
    updated_at_ns: int | None = None

    def __post_init__(self) -> None:
        root = self.root.resolve()
        working_directory = PathHolder.from_value(self.working_directory)
        directory = working_directory.resolve()
        if directory != root and root not in directory.parents:
            raise ValueError("Workspace working directory must be within its root.")
        identity = (self.id, self.name, self.name_source, self.created_at_ns, self.updated_at_ns)
        if any(value is not None for value in identity) and any(
            value is None for value in identity
        ):
            raise ValueError("Workspace identity metadata must be complete when initialized.")
        self.root = root
        self.working_directory = working_directory

    @classmethod
    def discover(cls, working_directory: Path | str) -> Workspace:
        """Discover the workspace containing an active directory.

        Args:
            working_directory (Path | str): Directory from which the application was started.

        Returns:
            Workspace: Discovered workspace without durable identity metadata.
        """
        directory = Path(working_directory).resolve()
        return cls(find_project_root(directory) or directory, directory)
