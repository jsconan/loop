"""Collect only inactive, unleased, resolved private runtime content."""

from __future__ import annotations

import json
import os
import shutil
import stat
import time
from pathlib import Path

from ... import constants
from .activation import active_set, rollback_set
from .lease import active_leases


def collect(
    root: Path,
    rollback: str | None = None,
    now: float | None = None,
    reclaim_bytes: int | None = None,
) -> tuple[Path, ...]:
    """Remove inactive unleased artifact digest directories and stale staging trees.

    Args:
        root (Path): Resolved Loop-private runtime root.
        rollback (str | None): Retained known-good artifact-set identity.
        now (float | None): Time used for stale staging checks.
        reclaim_bytes (int | None): Stop after reclaiming at least this many bytes. ``None``
            removes every eligible version.

    Returns:
        tuple[Path, ...]: Removed paths, all formerly below ``root``.

    Raises:
        ActivationError: If the active pointer is malformed.
    """
    root = root.resolve()
    protected_sets = set(active_leases(root))
    active = active_set(root)
    if active is not None:
        protected_sets.add(active)
    retained = rollback_set(root) if rollback is None else rollback
    if retained is not None:
        protected_sets.add(retained)
    protected = _protected_artifact_digests(root, protected_sets)
    if reclaim_bytes is not None and reclaim_bytes < 0:
        raise ValueError("Requested runtime reclamation must not be negative.")
    removed: list[Path] = []
    timestamp = time.time() if now is None else now
    artifacts = root / constants.RUNTIME_ARTIFACTS_DIRECTORY
    if not artifacts.is_dir() or artifacts.is_symlink():
        return ()
    candidates: list[tuple[float, str, str, int]] = []
    artifacts_descriptor = os.open(artifacts, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for artifact_name in sorted(os.listdir(artifacts_descriptor)):
            artifact_stat = os.stat(
                artifact_name, dir_fd=artifacts_descriptor, follow_symlinks=False
            )
            if not stat.S_ISDIR(artifact_stat.st_mode):
                continue
            artifact_descriptor = os.open(
                artifact_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=artifacts_descriptor,
            )
            try:
                versions = []
                for version_name in os.listdir(artifact_descriptor):
                    version_stat = os.stat(
                        version_name, dir_fd=artifact_descriptor, follow_symlinks=False
                    )
                    if stat.S_ISDIR(version_stat.st_mode):
                        versions.append((version_stat.st_mtime, version_name))
                for modified, version_name in versions:
                    if version_name.startswith(constants.RUNTIME_STAGING_PREFIX):
                        if timestamp - modified < constants.RUNTIME_STALE_STAGING_SECONDS:
                            continue
                    elif (
                        len(version_name) != constants.SHA256_HEX_LENGTH
                        or any(character not in "0123456789abcdef" for character in version_name)
                        or version_name in protected
                    ):
                        continue
                    size = 0
                    for _, _, files, directory_descriptor in os.fwalk(
                        version_name, dir_fd=artifact_descriptor, follow_symlinks=False
                    ):
                        size += sum(
                            os.stat(
                                name, dir_fd=directory_descriptor, follow_symlinks=False
                            ).st_size
                            for name in files
                        )
                    candidates.append((modified, artifact_name, version_name, size))
            finally:
                os.close(artifact_descriptor)
        reclaimed = 0
        for _, artifact_name, version_name, size in sorted(candidates):
            if reclaim_bytes is not None and reclaimed >= reclaim_bytes:
                break
            artifact_descriptor = os.open(
                artifact_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=artifacts_descriptor,
            )
            try:
                for _, directories, files, directory_descriptor in os.fwalk(
                    version_name,
                    topdown=False,
                    dir_fd=artifact_descriptor,
                    follow_symlinks=False,
                ):
                    for name in files:
                        os.chmod(
                            name,
                            constants.PRIVATE_FILE_MODE,
                            dir_fd=directory_descriptor,
                            follow_symlinks=False,
                        )
                    for name in directories:
                        os.chmod(
                            name,
                            constants.PRIVATE_DIRECTORY_MODE,
                            dir_fd=directory_descriptor,
                            follow_symlinks=False,
                        )
                os.chmod(version_name, constants.PRIVATE_DIRECTORY_MODE, dir_fd=artifact_descriptor)
                shutil.rmtree(version_name, dir_fd=artifact_descriptor)
            finally:
                os.close(artifact_descriptor)
            reclaimed += size
            removed.append(artifacts / artifact_name / version_name)
    finally:
        os.close(artifacts_descriptor)
    return tuple(removed)


def reset_inactive(root: Path) -> tuple[Path, ...]:
    """Remove inactive Loop-owned runtime cache data without touching active content.

    Args:
        root (Path): Exact private runtime root supplied by application composition.

    Returns:
        tuple[Path, ...]: Removed artifact versions and temporary download files.
    """
    root = root.resolve()
    removed = list(collect(root))
    downloads = root / constants.RUNTIME_DOWNLOADS_DIRECTORY
    if downloads.is_dir() and not downloads.is_symlink():
        descriptor = os.open(downloads, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for name in os.listdir(descriptor):
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if stat.S_ISREG(metadata.st_mode) and name.startswith(
                    constants.RUNTIME_DOWNLOAD_PREFIX
                ):
                    os.unlink(name, dir_fd=descriptor)
                    removed.append(downloads / name)
        finally:
            os.close(descriptor)
    return tuple(removed)


def _protected_artifact_digests(root: Path, artifact_sets: set[str]) -> set[str]:
    """Return content digests referenced by valid active, leased, or rollback sets."""
    protected: set[str] = set()
    for artifact_set in artifact_sets:
        if len(artifact_set) != 64:
            continue
        path = root / constants.RUNTIME_SETS_DIRECTORY / f"{artifact_set}.json"
        try:
            payload = json.loads(path.read_text(encoding="ascii"))
            digests = payload["artifact_digests"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if isinstance(digests, list) and all(
            isinstance(digest, str)
            and len(digest) == constants.SHA256_HEX_LENGTH
            and all(character in "0123456789abcdef" for character in digest)
            for digest in digests
        ):
            protected.update(digests)
    return protected
