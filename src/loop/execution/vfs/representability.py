"""Validate canonical workspace deltas against destination filesystem facts."""

from __future__ import annotations

import unicodedata

from .delta import UnrepresentableDeltaError
from .models import (
    CanonicalDelta,
    CanonicalDeltaEntry,
    DestinationFilesystemCapabilities,
    ObjectKind,
)


class FilesystemRepresentabilityValidator:
    """Validate lossless commit representation against one discovered destination filesystem."""

    def validate(
        self,
        delta: CanonicalDelta,
        capabilities: DestinationFilesystemCapabilities,
    ) -> CanonicalDelta:
        """Return a delta only when every effect is exactly representable.

        Args:
            delta (CanonicalDelta): Normalized persistent effects to validate.
            capabilities (DestinationFilesystemCapabilities): Measured destination facts.

        Returns:
            CanonicalDelta: The unchanged validated delta.

        Raises:
            UnrepresentableDeltaError: A path, object, or metadata effect would lose information.
        """
        _validate_paths(delta, capabilities)
        for entry in delta.entries:
            _validate_entry(entry, capabilities)
        return delta


def _validate_paths(delta: CanonicalDelta, capabilities: DestinationFilesystemCapabilities) -> None:
    """Reject path limits and aliases under the destination's comparison behavior."""
    paths = [
        path
        for entry in delta.entries
        for path in (entry.destination_path, entry.source_path)
        if path is not None
    ]
    aliases: dict[str, str] = {}
    for path in paths:
        encoded = path.encode("utf-8")
        if len(encoded) > capabilities.maximum_path_bytes:
            raise UnrepresentableDeltaError("Delta path exceeds the destination path limit.")
        if any(
            len(part.encode("utf-8")) > capabilities.maximum_name_bytes for part in path.split("/")
        ):
            raise UnrepresentableDeltaError("Delta name exceeds the destination name limit.")
        key = _comparison_key(path, capabilities)
        existing = aliases.setdefault(key, path)
        if existing != path:
            raise UnrepresentableDeltaError("Delta contains destination-equivalent path aliases.")


def _comparison_key(path: str, capabilities: DestinationFilesystemCapabilities) -> str:
    """Return one fail-closed destination path comparison key."""
    if capabilities.unicode_normalization == "unknown" and not path.isascii():
        raise UnrepresentableDeltaError("Destination Unicode comparison behavior is unknown.")
    if capabilities.unicode_normalization in {"nfc", "nfd"}:
        path = unicodedata.normalize(capabilities.unicode_normalization.upper(), path)
    return path if capabilities.case_sensitive else path.casefold()


def _validate_entry(
    entry: CanonicalDeltaEntry, capabilities: DestinationFilesystemCapabilities
) -> None:
    """Reject one object or metadata effect that the destination cannot preserve."""
    if entry.object_kind is ObjectKind.SYMLINK and not capabilities.supports_symlinks:
        raise UnrepresentableDeltaError("Destination filesystem does not support symbolic links.")
    if (
        entry.mode is not None or entry.metadata.basic_mode is not None
    ) and not capabilities.supports_basic_mode:
        raise UnrepresentableDeltaError("Destination filesystem cannot preserve basic modes.")
    if entry.metadata.requires_ownership and not capabilities.preserves_ownership:
        raise UnrepresentableDeltaError(
            "Destination filesystem cannot preserve ownership metadata."
        )
    if entry.metadata.requires_acl and not capabilities.preserves_acl:
        raise UnrepresentableDeltaError("Destination filesystem cannot preserve ACL metadata.")
    if entry.metadata.requires_security_xattrs and not capabilities.preserves_security_xattrs:
        raise UnrepresentableDeltaError("Destination filesystem cannot preserve security xattrs.")
