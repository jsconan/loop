"""Define portable, host-path-free records for workspace VFS operations."""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts import DeltaEffect
from .namespace import validate_relative_path


class BaseOpaqueIdentifier(BaseModel):
    """Represent one opaque immutable VFS identifier."""

    model_config = ConfigDict(frozen=True)

    value: str = Field(min_length=1)


class BaseSnapshotId(BaseOpaqueIdentifier):
    """Identify one immutable shared workspace base snapshot."""


class BranchId(BaseOpaqueIdentifier):
    """Identify one opaque private agent branch."""


class GenerationId(BaseOpaqueIdentifier):
    """Identify one monotonically advancing agent generation."""


class LineageEpoch(BaseOpaqueIdentifier):
    """Identify one explicit synchronization epoch in an agent lineage."""


class HostObjectIdentity(BaseModel):
    """Record an object identity captured by the trusted host-side broker.

    Args:
        device (int): Source filesystem device number.
        inode (int): Source filesystem inode number.
        ctime_ns (int): Source status-change time in nanoseconds.
    """

    model_config = ConfigDict(frozen=True)

    device: int = Field(ge=0)
    inode: int = Field(ge=0)
    ctime_ns: int = Field(ge=0)

    @classmethod
    def from_metadata(cls, metadata: os.stat_result) -> Self:
        """Create an identity from trusted filesystem status metadata.

        Args:
            metadata (os.stat_result): No-follow or descriptor status metadata.

        Returns:
            Self: Device, inode, and change-time identity.
        """
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            ctime_ns=metadata.st_ctime_ns,
        )


class AffectedPathIdentity(BaseModel):
    """Record the current published host identity for one affected virtual path.

    Args:
        path (str): Workspace-relative affected path.
        identity (HostObjectIdentity | None): Current identity, or ``None`` when absent.
    """

    model_config = ConfigDict(frozen=True)

    path: str
    identity: HostObjectIdentity | None = None

    @model_validator(mode="after")
    def validate_path(self) -> AffectedPathIdentity:
        """Require one normalized workspace-relative path.

        Returns:
            AffectedPathIdentity: Validated path identity.
        """
        validate_relative_path(self.path)
        return self


class ContentReference(BaseModel):
    """Describe immutable staged content without exposing its storage location.

    Args:
        digest (str): Content digest including its algorithm prefix.
        size (int): Exact byte length of the content.
        reference (str): Opaque trusted content-store reference.
    """

    model_config = ConfigDict(frozen=True)

    digest: str = Field(pattern=r"^[a-z0-9][a-z0-9+._-]*:[0-9a-f]+$")
    size: int = Field(ge=0)
    reference: str = Field(min_length=1)


class ObjectKind(StrEnum):
    """Identify a portable workspace object kind."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"


class MetadataPreservation(BaseModel):
    """Describe metadata that a destination must preserve exactly.

    Args:
        basic_mode (int | None): Portable POSIX mode, when changed or preserved.
        requires_ownership (bool): Whether ownership metadata is required.
        requires_acl (bool): Whether ACL metadata is required.
        requires_security_xattrs (bool): Whether security xattrs are required.
    """

    model_config = ConfigDict(frozen=True)

    basic_mode: int | None = Field(default=None, ge=0, le=0o7777)
    requires_ownership: bool = False
    requires_acl: bool = False
    requires_security_xattrs: bool = False


class CanonicalDeltaEntry(BaseModel):
    """Describe one normalized persistent workspace effect.

    Args:
        effect (DeltaEffect): Persistent effect represented by this entry.
        destination_path (str): Workspace-relative destination path.
        object_kind (ObjectKind | None): Resulting object kind, if one remains.
        source_path (str | None): Workspace-relative source for a rename.
        source_identity (HostObjectIdentity | None): Captured source identity.
        destination_identity (HostObjectIdentity | None): Captured destination identity.
        content (ContentReference | None): Opaque content for a created or replaced file.
        mode (int | None): Resulting portable POSIX mode.
        symlink_target (str | None): Literal target of a resulting symbolic link.
        metadata (MetadataPreservation): Required metadata-preservation policy.
    """

    model_config = ConfigDict(frozen=True)

    effect: DeltaEffect
    destination_path: str
    object_kind: ObjectKind | None = None
    source_path: str | None = None
    source_identity: HostObjectIdentity | None = None
    destination_identity: HostObjectIdentity | None = None
    content: ContentReference | None = None
    mode: int | None = Field(default=None, ge=0, le=0o7777)
    symlink_target: str | None = None
    metadata: MetadataPreservation = MetadataPreservation()

    @model_validator(mode="after")
    def validate_effect_shape(self) -> Self:
        """Require an unambiguous portable effect representation."""
        validate_relative_path(self.destination_path)
        if self.source_path is not None:
            validate_relative_path(self.source_path)
        if self.effect is DeltaEffect.RENAME:
            if self.source_path is None:
                raise ValueError("Rename entries require a source path.")
        elif self.source_path is not None:
            raise ValueError("Only rename entries may include a source path.")
        if self.effect is DeltaEffect.DELETE:
            if any(
                value is not None
                for value in (self.object_kind, self.content, self.mode, self.symlink_target)
            ):
                raise ValueError("Delete entries cannot describe a resulting object.")
        elif self.object_kind is None:
            raise ValueError("Non-delete entries require a resulting object kind.")
        if self.object_kind is ObjectKind.FILE and self.symlink_target is not None:
            raise ValueError("Regular files cannot carry a symlink target.")
        if self.object_kind is ObjectKind.SYMLINK and self.symlink_target is None:
            raise ValueError("Symlink entries require a literal link target.")
        if self.content is not None and self.object_kind is not ObjectKind.FILE:
            raise ValueError("Only regular files may carry content references.")
        return self

    @property
    def ordering_key(self) -> tuple[str, int, int, str, str, str]:
        """Return the stable order used for serialized canonical deltas.

        Returns:
            tuple[str, int, int, str, str, str]: Root, deletion phase, depth, destination,
                source, and effect ordering components. Nested deletions sort deepest first.
        """
        deleting = self.effect is DeltaEffect.DELETE
        depth = self.destination_path.count("/")
        return (
            self.destination_path.partition("/")[0],
            int(deleting),
            -depth if deleting else depth,
            self.destination_path,
            self.source_path or "",
            self.effect.value,
        )


class CanonicalDelta(BaseModel):
    """Carry one versioned, deterministically ordered canonical workspace delta.

    Args:
        format_version (int): Persisted canonical-delta schema version.
        entries (tuple[CanonicalDeltaEntry, ...]): Strictly ordered unique destination effects.
    """

    model_config = ConfigDict(frozen=True)

    format_version: int = Field(default=1, ge=1)
    entries: tuple[CanonicalDeltaEntry, ...] = ()

    @model_validator(mode="after")
    def validate_canonical_order(self) -> Self:
        """Require deterministic ordering and one effect per destination."""
        keys = tuple(entry.ordering_key for entry in self.entries)
        if keys != tuple(sorted(keys)):
            raise ValueError("Canonical delta entries must use deterministic ordering.")
        destinations = tuple(entry.destination_path for entry in self.entries)
        if len(destinations) != len(set(destinations)):
            raise ValueError("Canonical deltas cannot contain duplicate destinations.")
        return self


class DestinationFilesystemCapabilities(BaseModel):
    """Record immutable destination representability capabilities.

    Args:
        format_version (int): Persisted capabilities schema version.
        case_sensitive (bool): Whether destination names are case sensitive.
        unicode_normalization (str): Destination Unicode comparison behavior.
        maximum_name_bytes (int): Maximum encoded single-component length.
        maximum_path_bytes (int): Maximum encoded relative-path length.
        supports_symlinks (bool): Whether symbolic links can be represented.
        supports_basic_mode (bool): Whether portable POSIX modes can be preserved.
        preserves_ownership (bool): Whether requested ownership can be preserved.
        preserves_acl (bool): Whether requested ACLs can be preserved.
        preserves_security_xattrs (bool): Whether requested security xattrs can be preserved.
    """

    model_config = ConfigDict(frozen=True)

    format_version: int = Field(default=1, ge=1)
    case_sensitive: bool
    unicode_normalization: str = Field(pattern=r"^(none|nfc|nfd|unknown)$")
    maximum_name_bytes: int = Field(ge=1)
    maximum_path_bytes: int = Field(ge=1)
    supports_symlinks: bool
    supports_basic_mode: bool
    preserves_ownership: bool
    preserves_acl: bool
    preserves_security_xattrs: bool
