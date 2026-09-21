"""Discover destination filesystem properties through an authenticated root."""

from __future__ import annotations

import os
import secrets
import sys

from .models import DestinationFilesystemCapabilities
from .path_broker import AuthenticatedWorkspaceRoot


class FilesystemCapabilities:
    """Cache immutable capabilities once for each authenticated root identity."""

    _by_identity: dict[tuple[int, int], DestinationFilesystemCapabilities]

    def __init__(self) -> None:
        """Initialize an empty authenticated-root capability cache."""
        self._by_identity = {}

    def discover(self, root: AuthenticatedWorkspaceRoot) -> DestinationFilesystemCapabilities:
        """Return capabilities measured for one authenticated destination root.

        Args:
            root (AuthenticatedWorkspaceRoot): Retained destination root descriptor.

        Returns:
            DestinationFilesystemCapabilities: Immutable representability facts.
        """
        key = (root.identity.device, root.identity.inode)
        if key not in self._by_identity:
            self._by_identity[key] = _discover(root)
        return self._by_identity[key]


def _discover(root: AuthenticatedWorkspaceRoot) -> DestinationFilesystemCapabilities:
    """Measure name comparison and portable primitive support without following links."""
    descriptor = root.duplicate_descriptor()
    probe = f".loop-vfs-{secrets.token_hex(16)}"
    try:
        os.mkdir(probe, mode=0o700, dir_fd=descriptor)
        probe_descriptor = os.open(
            probe, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
        )
        try:
            os.close(
                os.open(
                    "Case", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=probe_descriptor
                )
            )
            try:
                os.stat("case", dir_fd=probe_descriptor, follow_symlinks=False)
                case_sensitive = False
            except FileNotFoundError:
                case_sensitive = True
            try:
                os.symlink("target", "link", dir_fd=probe_descriptor)
                supports_symlinks = True
            except OSError:
                supports_symlinks = False
            _remove_probe_entries(probe_descriptor)
        finally:
            os.close(probe_descriptor)
        os.rmdir(probe, dir_fd=descriptor)
    except OSError as error:
        raise RuntimeError(
            "Destination filesystem capabilities cannot be safely discovered."
        ) from error
    finally:
        os.close(descriptor)
    return DestinationFilesystemCapabilities(
        case_sensitive=case_sensitive,
        unicode_normalization="nfd" if sys.platform == "darwin" else "none",
        maximum_name_bytes=_pathconf(root, "PC_NAME_MAX"),
        maximum_path_bytes=_pathconf(root, "PC_PATH_MAX"),
        supports_symlinks=supports_symlinks,
        supports_basic_mode=os.name == "posix",
        preserves_ownership=False,
        preserves_acl=False,
        preserves_security_xattrs=False,
    )


def _remove_probe_entries(descriptor: int) -> None:
    """Remove known probe entries while the probe directory remains descriptor-confined."""
    for name in ("Case", "link"):
        try:
            os.unlink(name, dir_fd=descriptor)
        except FileNotFoundError:
            pass


def _pathconf(root: AuthenticatedWorkspaceRoot, name: str) -> int:
    """Read one descriptor-bound filesystem limit without leaking a descriptor."""
    descriptor = root.duplicate_descriptor()
    try:
        return os.fpathconf(descriptor, name)
    finally:
        os.close(descriptor)
