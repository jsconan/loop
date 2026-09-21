"""Normalize trusted platform observations into portable workspace deltas."""

from __future__ import annotations

from enum import StrEnum
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts import DeltaEffect
from .models import (
    CanonicalDelta,
    CanonicalDeltaEntry,
    ContentReference,
    HostObjectIdentity,
    MetadataPreservation,
    ObjectKind,
)
from .namespace import validate_relative_path


class UnrepresentableDeltaError(ValueError):
    """Report an observed effect that cannot be represented losslessly."""


class ObservedObjectKind(StrEnum):
    """Identify object kinds emitted by a trusted platform delta inspector."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    FIFO = "fifo"
    SOCKET = "socket"
    BLOCK_DEVICE = "block_device"
    CHARACTER_DEVICE = "character_device"
    HARDLINK = "hardlink"


class ObservedDeltaEntry(BaseModel):
    """Record one trusted, platform-neutral persistent filesystem observation.

    Args:
        effect (DeltaEffect): Observed persistent effect.
        destination_path (str): Workspace-relative path affected by the effect.
        object_kind (ObservedObjectKind | None): Resulting object kind, if one remains.
        source_path (str | None): Workspace-relative rename source.
        source_identity (HostObjectIdentity | None): Trusted source identity evidence.
        destination_identity (HostObjectIdentity | None): Trusted destination identity evidence.
        content (ContentReference | None): Opaque staged content for a regular file.
        mode (int | None): Resulting portable POSIX mode.
        symlink_target (str | None): Literal target of a symbolic link.
        metadata (MetadataPreservation): Required resulting metadata preservation.
        ownership_changed (bool): Whether ownership changed in the attempt layer.
        acl_changed (bool): Whether ACL metadata changed in the attempt layer.
        security_xattrs_changed (bool): Whether security xattrs changed in the attempt layer.
        hardlink_topology_changed (bool): Whether hardlink topology changed in the attempt layer.
    """

    model_config = ConfigDict(frozen=True)

    effect: DeltaEffect
    destination_path: str
    object_kind: ObservedObjectKind | None = None
    source_path: str | None = None
    source_identity: HostObjectIdentity | None = None
    destination_identity: HostObjectIdentity | None = None
    content: ContentReference | None = None
    mode: int | None = Field(default=None, ge=0, le=0o7777)
    symlink_target: str | None = None
    metadata: MetadataPreservation = MetadataPreservation()
    ownership_changed: bool = False
    acl_changed: bool = False
    security_xattrs_changed: bool = False
    hardlink_topology_changed: bool = False

    @model_validator(mode="after")
    def validate_observation_shape(self) -> ObservedDeltaEntry:
        """Require virtual paths and effect-specific source evidence."""
        validate_relative_path(self.destination_path)
        if self.source_path is not None:
            validate_relative_path(self.source_path)
        if self.effect is DeltaEffect.RENAME:
            if self.source_path is None:
                raise ValueError("Rename observations require a source path.")
        elif self.source_path is not None:
            raise ValueError("Only rename observations may include a source path.")
        if self.effect is DeltaEffect.DELETE and self.object_kind is not None:
            raise ValueError("Delete observations cannot describe a resulting object.")
        if self.effect is not DeltaEffect.DELETE and self.object_kind is None:
            raise ValueError("Non-delete observations require a resulting object kind.")
        return self


class ObservedDelta(BaseModel):
    """Collect one complete trusted inspector feed for a stopped attempt.

    Args:
        entries (tuple[ObservedDeltaEntry, ...]): Unordered observed persistent effects.
        maximum_entries (int): Maximum accepted effects from the trusted inspector.
        maximum_content_bytes (int): Maximum total staged regular-file bytes.
    """

    model_config = ConfigDict(frozen=True)

    entries: tuple[ObservedDeltaEntry, ...] = ()
    maximum_entries: int = Field(default=10_000, ge=0)
    maximum_content_bytes: int = Field(default=1 << 30, ge=0)


def normalize_delta(observed: ObservedDelta) -> CanonicalDelta:
    """Normalize one trusted observed delta into a deterministic portable delta.

    Args:
        observed (ObservedDelta): Trusted, platform-neutral attempt effects.

    Returns:
        CanonicalDelta: Deterministically ordered portable persistent effects.

    Raises:
        UnrepresentableDeltaError: An effect, topology, or configured quota is unsafe to commit.
    """
    if len(observed.entries) > observed.maximum_entries:
        raise UnrepresentableDeltaError("Delta exceeds its configured entry quota.")
    content_bytes = sum(entry.content.size for entry in observed.entries if entry.content)
    if content_bytes > observed.maximum_content_bytes:
        raise UnrepresentableDeltaError("Delta exceeds its configured content quota.")
    _validate_observed_topology(observed.entries)
    entries = tuple(
        sorted(
            (_canonicalize(entry) for entry in observed.entries),
            key=lambda entry: entry.ordering_key,
        )
    )
    return CanonicalDelta(entries=entries)


def _canonicalize(observed: ObservedDeltaEntry) -> CanonicalDeltaEntry:
    """Convert one already-validated portable observation to its canonical record."""
    _reject_unportable_observation(observed)
    return CanonicalDeltaEntry(
        effect=observed.effect,
        destination_path=observed.destination_path,
        object_kind=ObjectKind(observed.object_kind) if observed.object_kind else None,
        source_path=observed.source_path,
        source_identity=observed.source_identity,
        destination_identity=observed.destination_identity,
        content=observed.content,
        mode=observed.mode,
        symlink_target=observed.symlink_target,
        metadata=observed.metadata,
    )


def _reject_unportable_observation(observed: ObservedDeltaEntry) -> None:
    """Reject metadata and node changes that the portable commit format cannot express."""
    if observed.object_kind not in {
        None,
        ObservedObjectKind.FILE,
        ObservedObjectKind.DIRECTORY,
        ObservedObjectKind.SYMLINK,
    }:
        raise UnrepresentableDeltaError("Delta contains an unsupported filesystem node type.")
    if observed.ownership_changed:
        raise UnrepresentableDeltaError("Delta changes ownership metadata.")
    if observed.acl_changed:
        raise UnrepresentableDeltaError("Delta changes ACL metadata.")
    if observed.security_xattrs_changed:
        raise UnrepresentableDeltaError("Delta changes security xattrs.")
    if observed.hardlink_topology_changed:
        raise UnrepresentableDeltaError("Delta changes hardlink topology.")


def _validate_observed_topology(entries: tuple[ObservedDeltaEntry, ...]) -> None:
    """Reject duplicate targets, contradictory trees, and non-committable rename graphs."""
    destinations = [entry.destination_path for entry in entries]
    if len(destinations) != len(set(destinations)):
        raise UnrepresentableDeltaError("Delta contains duplicate destinations.")
    for parent in entries:
        for child in entries:
            if parent is child or not child.destination_path.startswith(
                parent.destination_path + "/"
            ):
                continue
            if parent.object_kind is not ObservedObjectKind.DIRECTORY and not (
                parent.effect is DeltaEffect.DELETE and child.effect is DeltaEffect.DELETE
            ):
                raise UnrepresentableDeltaError(
                    "Delta contains contradictory parent and child effects."
                )
    rename_entries = tuple(entry for entry in entries if entry.effect is DeltaEffect.RENAME)
    rename_sources = tuple(entry.source_path for entry in rename_entries)
    if len(rename_sources) != len(set(rename_sources)):
        raise UnrepresentableDeltaError("Delta contains duplicate rename sources.")
    renames = {cast(str, entry.source_path): entry.destination_path for entry in rename_entries}
    for source in renames:
        visited: set[str] = set()
        current = source
        while current in renames:
            if current in visited:
                raise UnrepresentableDeltaError("Delta contains a rename cycle.")
            visited.add(current)
            current = renames[current]
    rename_sources = set(renames)
    for entry in entries:
        if entry.effect is not DeltaEffect.RENAME and entry.destination_path in rename_sources:
            raise UnrepresentableDeltaError("Delta overwrites a rename source.")
