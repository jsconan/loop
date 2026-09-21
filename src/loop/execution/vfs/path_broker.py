"""Authenticate and traverse one workspace root through retained descriptors."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Self

from .models import HostObjectIdentity
from .namespace import validate_relative_path


class WorkspaceRootChangedError(RuntimeError):
    """Report a workspace root or object that changed during a trusted operation."""


class AuthenticatedWorkspaceRoot:
    """Own a verified workspace directory descriptor and its stable identity.

    Args:
        path (Path): User-selected workspace root, which must be a real directory.
    """

    path: Path
    identity: HostObjectIdentity
    _descriptor: int

    def __init__(self, path: Path) -> None:
        """Open and authenticate a workspace root without following a symlink."""
        self.path = path
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        self._descriptor = os.open(path, flags)
        metadata = os.fstat(self._descriptor)
        self.identity = HostObjectIdentity.from_metadata(metadata)
        self.verify_root()

    def close(self) -> None:
        """Close the retained workspace descriptor."""
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def __enter__(self) -> Self:
        """Return this open authenticated workspace root."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the retained workspace descriptor."""
        self.close()

    def verify_root(self) -> None:
        """Fail when the user-visible root name no longer names the retained directory.

        Raises:
            WorkspaceRootChangedError: The root was replaced, removed, or became a symlink.
        """
        try:
            current = os.lstat(self.path)
        except OSError as error:
            raise WorkspaceRootChangedError("Workspace root changed during snapshot.") from error
        if (current.st_dev, current.st_ino) != (
            self.identity.device,
            self.identity.inode,
        ) or not stat.S_ISDIR(current.st_mode):
            raise WorkspaceRootChangedError("Workspace root changed during snapshot.")

    def duplicate_descriptor(self) -> int:
        """Return an independent descriptor for the authenticated root.

        Returns:
            int: A caller-owned directory descriptor.
        """
        self.verify_root()
        return os.dup(self._descriptor)

    def open_directory(self, relative_path: str) -> int:
        """Open a workspace-relative directory without following any component symlink.

        Args:
            relative_path (str): Normalized virtual-relative directory path.

        Returns:
            int: A caller-owned directory descriptor.

        Raises:
            WorkspaceRootChangedError: Traversal encountered a changed or unsafe object.
        """
        validate_relative_path(relative_path)
        descriptor = self.duplicate_descriptor()
        try:
            for component in relative_path.split("/"):
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor
        except OSError as error:
            os.close(descriptor)
            raise WorkspaceRootChangedError("Unsafe workspace directory traversal.") from error
