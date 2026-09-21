"""Define the portable versioned manifest for one immutable base snapshot."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import (
    BaseSnapshotId,
    ContentReference,
    HostObjectIdentity,
    MetadataPreservation,
    ObjectKind,
)
from .namespace import validate_relative_path


class SnapshotManifestEntry(BaseModel):
    """Describe one immutable snapshot object using only virtual-relative data.

    Args:
        path (str): Workspace-relative path.
        object_kind (ObjectKind): Captured object kind.
        identity (HostObjectIdentity): Stable source identity evidence.
        content (ContentReference | None): Content evidence for regular files.
        mode (int): Portable POSIX mode.
        size (int): Captured object size in bytes.
        symlink_target (str | None): Literal target for symbolic links.
        metadata (MetadataPreservation): Captured metadata requirements.
    """

    model_config = ConfigDict(frozen=True)

    path: str
    object_kind: ObjectKind
    identity: HostObjectIdentity
    content: ContentReference | None = None
    mode: int = Field(ge=0, le=0o7777)
    size: int = Field(ge=0)
    symlink_target: str | None = None
    metadata: MetadataPreservation = MetadataPreservation()

    @model_validator(mode="after")
    def validate_entry(self) -> SnapshotManifestEntry:
        """Require a portable representation matching the captured object kind."""
        validate_relative_path(self.path)
        if self.object_kind is ObjectKind.FILE and self.content is None:
            raise ValueError("Regular files require content evidence.")
        if self.object_kind is ObjectKind.SYMLINK and self.symlink_target is None:
            raise ValueError("Symbolic links require their literal target.")
        if self.object_kind is not ObjectKind.SYMLINK and self.symlink_target is not None:
            raise ValueError("Only symbolic links may carry a link target.")
        return self


class SnapshotManifest(BaseModel):
    """Persist a deterministic manifest for one immutable workspace base.

    Args:
        format_version (int): Persisted manifest schema version.
        snapshot_id (BaseSnapshotId): Immutable base snapshot identity.
        entries (tuple[SnapshotManifestEntry, ...]): Strictly path-sorted snapshot entries.
    """

    model_config = ConfigDict(frozen=True)

    format_version: int = Field(default=1, ge=1)
    snapshot_id: BaseSnapshotId
    entries: tuple[SnapshotManifestEntry, ...] = ()

    @model_validator(mode="after")
    def validate_order(self) -> SnapshotManifest:
        """Require deterministic ordering and unique paths."""
        paths = tuple(entry.path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("Snapshot manifest entries must be uniquely path-sorted.")
        return self
