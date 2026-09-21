"""Build and atomically publish immutable descriptor-validated workspace snapshots."""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from .manifest import SnapshotManifest, SnapshotManifestEntry
from .models import (
    BaseSnapshotId,
    ContentReference,
    HostObjectIdentity,
    MetadataPreservation,
    ObjectKind,
)
from .path_broker import AuthenticatedWorkspaceRoot, WorkspaceRootChangedError


class WorkspaceBusyError(RuntimeError):
    """Report mutation that prevented a stable immutable workspace snapshot."""


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """Expose a published immutable snapshot only to trusted VFS callers.

    Args:
        manifest (SnapshotManifest): Canonical evidence for the published tree.
        directory (Path): Loop-private immutable snapshot tree, excluding trusted metadata.
    """

    manifest: SnapshotManifest
    directory: Path


class FileCopier:
    """Copy regular files through descriptors with no optimization-dependent correctness path."""

    def copy(self, source_descriptor: int, destination: Path) -> tuple[str, int]:
        """Copy one already-authenticated file and return its SHA-256 evidence.

        Args:
            source_descriptor (int): Open source descriptor positioned arbitrarily.
            destination (Path): New private snapshot file path.

        Returns:
            tuple[str, int]: Hex digest and exact copied byte count.
        """
        digest = hashlib.sha256()
        size = 0
        with destination.open("xb") as output:
            while chunk := os.read(source_descriptor, 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        return digest.hexdigest(), size


class SnapshotBuilder:
    """Create sealed immutable bases from authenticated roots with bounded stability retries.

    Args:
        snapshot_store (Path): Loop-private directory that stores published snapshots.
        retries (int): Number of complete attempts after the first. Defaults to ``2``.
        copier (FileCopier | None): Descriptor-safe file copy implementation.
    """

    snapshot_store: Path
    retries: int
    copier: FileCopier

    def __init__(
        self, snapshot_store: Path, *, retries: int = 2, copier: FileCopier | None = None
    ) -> None:
        if retries < 0:
            raise ValueError("Snapshot retries cannot be negative.")
        self.snapshot_store = snapshot_store
        self.retries = retries
        self.copier = copier or FileCopier()

    def build(self, root: AuthenticatedWorkspaceRoot) -> WorkspaceSnapshot:
        """Build and atomically publish one stable immutable workspace snapshot.

        Args:
            root (AuthenticatedWorkspaceRoot): Retained authenticated source root.

        Returns:
            WorkspaceSnapshot: Published private snapshot and canonical manifest.

        Raises:
            WorkspaceBusyError: The source changes throughout every bounded capture attempt.
        """
        self.snapshot_store.mkdir(mode=0o700, parents=True, exist_ok=True)
        for _ in range(self.retries + 1):
            try:
                return self._build_once(root)
            except WorkspaceRootChangedError:
                continue
        raise WorkspaceBusyError("Workspace remained mutable while creating a snapshot.")

    def discard(self, snapshot: WorkspaceSnapshot) -> None:
        """Reclaim one builder-owned immutable snapshot after its leases end.

        Args:
            snapshot (WorkspaceSnapshot): Published snapshot returned by this builder.

        Raises:
            ValueError: The snapshot is outside or no longer beneath this store.
        """
        published = snapshot.directory.parent
        try:
            store = self.snapshot_store.resolve(strict=True)
            resolved = published.resolve(strict=True)
        except (OSError, ValueError) as error:
            raise ValueError("Snapshot is outside its authenticated store.") from error
        if (
            resolved.parent != store
            or published.is_symlink()
            or snapshot.directory.is_symlink()
            or snapshot.directory.resolve(strict=True) != resolved / "tree"
        ):
            raise ValueError("Snapshot is outside its authenticated store.")
        published = resolved
        self._make_writable(snapshot.directory)
        (published / "manifest.json").chmod(0o600)
        published.chmod(0o700)
        shutil.rmtree(published)
        store_descriptor = os.open(self.snapshot_store, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(store_descriptor)
        finally:
            os.close(store_descriptor)

    def _build_once(self, root: AuthenticatedWorkspaceRoot) -> WorkspaceSnapshot:
        """Capture one complete candidate and discard it if any stability check fails."""
        root.verify_root()
        snapshot_id = BaseSnapshotId(value=f"snapshot-{secrets.token_urlsafe(24)}")
        temporary = self.snapshot_store / f".{snapshot_id.value}.tmp"
        published = self.snapshot_store / snapshot_id.value
        temporary.mkdir(mode=0o700)
        tree = temporary / "tree"
        tree.mkdir(mode=0o700)
        entries: list[SnapshotManifestEntry] = []
        try:
            self._capture_directory(root, root.duplicate_descriptor(), tree, "", entries)
            root.verify_root()
            manifest = SnapshotManifest(
                snapshot_id=snapshot_id, entries=tuple(sorted(entries, key=lambda e: e.path))
            )
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
            with manifest_path.open("rb") as stream:
                os.fsync(stream.fileno())
            manifest_path.chmod(0o400)
            temporary_descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(temporary_descriptor)
            finally:
                os.close(temporary_descriptor)
            os.rename(temporary, published)
            store_descriptor = os.open(self.snapshot_store, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(store_descriptor)
            finally:
                os.close(store_descriptor)
            published.chmod(0o500)
            return WorkspaceSnapshot(manifest=manifest, directory=published / "tree")
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.rmtree(published, ignore_errors=True)
            raise

    def _make_writable(self, directory: Path) -> None:
        """Restore owner write permission solely for trusted recursive reclamation."""
        directory.chmod(stat.S_IMODE(directory.stat(follow_symlinks=False).st_mode) | 0o700)
        for entry in os.scandir(directory):
            if entry.is_symlink():
                continue
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                self._make_writable(path)
            else:
                metadata = entry.stat(follow_symlinks=False)
                path.chmod(stat.S_IMODE(metadata.st_mode) | 0o600, follow_symlinks=False)

    def _capture_directory(
        self,
        root: AuthenticatedWorkspaceRoot,
        descriptor: int,
        destination: Path,
        relative: str,
        entries: list[SnapshotManifestEntry],
    ) -> None:
        """Capture a single directory by no-follow entry descriptors and post-checks."""
        before = os.fstat(descriptor)
        try:
            for name in sorted(os.listdir(descriptor)):
                relative_path = f"{relative}/{name}" if relative else name
                source_metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                destination_path = destination / name
                if stat.S_ISDIR(source_metadata.st_mode):
                    destination_path.mkdir(mode=stat.S_IMODE(source_metadata.st_mode))
                    child = os.open(
                        name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                    )
                    if HostObjectIdentity.from_metadata(
                        os.fstat(child)
                    ) != HostObjectIdentity.from_metadata(source_metadata):
                        os.close(child)
                        raise WorkspaceRootChangedError(
                            "Workspace directory changed during snapshot."
                        )
                    self._capture_directory(root, child, destination_path, relative_path, entries)
                    destination_path.chmod(stat.S_IMODE(source_metadata.st_mode))
                    entries.append(
                        self._entry(relative_path, source_metadata, ObjectKind.DIRECTORY)
                    )
                elif stat.S_ISREG(source_metadata.st_mode):
                    source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
                    try:
                        if HostObjectIdentity.from_metadata(
                            os.fstat(source)
                        ) != HostObjectIdentity.from_metadata(source_metadata):
                            raise WorkspaceRootChangedError(
                                "Workspace file changed during snapshot."
                            )
                        digest, size = self.copier.copy(source, destination_path)
                        destination_path.chmod(stat.S_IMODE(source_metadata.st_mode))
                        final_metadata = os.fstat(source)
                        if (
                            size != source_metadata.st_size
                            or HostObjectIdentity.from_metadata(final_metadata)
                            != HostObjectIdentity.from_metadata(source_metadata)
                            or final_metadata.st_size != source_metadata.st_size
                        ):
                            raise WorkspaceRootChangedError(
                                "Workspace file changed during snapshot."
                            )
                    finally:
                        os.close(source)
                    entries.append(
                        self._entry(relative_path, source_metadata, ObjectKind.FILE, digest=digest)
                    )
                elif stat.S_ISLNK(source_metadata.st_mode):
                    target = os.readlink(name, dir_fd=descriptor)
                    os.symlink(target, destination_path)
                    final_metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if (
                        HostObjectIdentity.from_metadata(final_metadata)
                        != HostObjectIdentity.from_metadata(source_metadata)
                        or final_metadata.st_size != source_metadata.st_size
                    ):
                        raise WorkspaceRootChangedError("Workspace link changed during snapshot.")
                    entries.append(
                        self._entry(
                            relative_path, source_metadata, ObjectKind.SYMLINK, target=target
                        )
                    )
                else:
                    raise WorkspaceRootChangedError(
                        "Workspace contains an unsupported object type."
                    )
            final_directory = os.fstat(descriptor)
            if (
                HostObjectIdentity.from_metadata(final_directory)
                != HostObjectIdentity.from_metadata(before)
                or final_directory.st_size != before.st_size
            ):
                raise WorkspaceRootChangedError("Workspace directory changed during snapshot.")
            destination.chmod(stat.S_IMODE(before.st_mode))
        finally:
            os.close(descriptor)

    @staticmethod
    def _entry(
        path: str,
        metadata: os.stat_result,
        kind: ObjectKind,
        *,
        digest: str | None = None,
        target: str | None = None,
    ) -> SnapshotManifestEntry:
        """Translate trusted status evidence into one canonical manifest entry."""
        mode = stat.S_IMODE(metadata.st_mode)
        content = (
            ContentReference(
                digest=f"sha256:{digest}", size=metadata.st_size, reference=f"content:{digest}"
            )
            if digest is not None
            else None
        )
        return SnapshotManifestEntry(
            path=path,
            object_kind=kind,
            identity=HostObjectIdentity.from_metadata(metadata),
            content=content,
            mode=mode,
            size=metadata.st_size,
            symlink_target=target,
            metadata=MetadataPreservation(basic_mode=mode),
        )
