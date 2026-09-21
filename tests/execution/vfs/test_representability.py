"""Test lossless destination-filesystem validation for canonical VFS deltas."""

from __future__ import annotations

import pytest

from loop.execution.contracts import DeltaEffect
from loop.execution.vfs import (
    CanonicalDelta,
    CanonicalDeltaEntry,
    ContentReference,
    DestinationFilesystemCapabilities,
    FilesystemRepresentabilityValidator,
    MetadataPreservation,
    ObjectKind,
    UnrepresentableDeltaError,
)


def _capabilities(**changes: object) -> DestinationFilesystemCapabilities:
    """Return a lossless POSIX-like destination with focused capability overrides."""
    values: dict[str, object] = {
        "case_sensitive": True,
        "unicode_normalization": "none",
        "maximum_name_bytes": 255,
        "maximum_path_bytes": 4096,
        "supports_symlinks": True,
        "supports_basic_mode": True,
        "preserves_ownership": True,
        "preserves_acl": True,
        "preserves_security_xattrs": True,
    }
    values.update(changes)
    return DestinationFilesystemCapabilities(**values)


def _entry(path: str, **changes: object) -> CanonicalDeltaEntry:
    """Return one canonical regular-file creation with optional focused changes."""
    values: dict[str, object] = {
        "effect": DeltaEffect.CREATE,
        "destination_path": path,
        "object_kind": ObjectKind.FILE,
        "content": ContentReference(digest="sha256:ab", size=2, reference="content:one"),
    }
    values.update(changes)
    return CanonicalDeltaEntry(**values)


def test_representability_accepts_lossless_portable_effects():
    """A portable path, mode, link, and required metadata pass matching capabilities."""
    delta = CanonicalDelta(
        entries=(
            _entry("file", mode=0o755, metadata=MetadataPreservation(basic_mode=0o755)),
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="link",
                object_kind=ObjectKind.SYMLINK,
                symlink_target="target",
            ),
        )
    )

    assert FilesystemRepresentabilityValidator().validate(delta, _capabilities()) == delta


@pytest.mark.parametrize(
    ("capabilities", "entry", "message"),
    (
        (
            _capabilities(supports_symlinks=False),
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="link",
                object_kind=ObjectKind.SYMLINK,
                symlink_target="target",
            ),
            "symbolic links",
        ),
        (_capabilities(supports_basic_mode=False), _entry("file", mode=0o755), "basic modes"),
        (
            _capabilities(preserves_ownership=False),
            _entry("file", metadata=MetadataPreservation(requires_ownership=True)),
            "ownership",
        ),
        (
            _capabilities(preserves_acl=False),
            _entry("file", metadata=MetadataPreservation(requires_acl=True)),
            "ACL",
        ),
        (
            _capabilities(preserves_security_xattrs=False),
            _entry("file", metadata=MetadataPreservation(requires_security_xattrs=True)),
            "security xattrs",
        ),
    ),
)
def test_representability_rejects_unsupported_object_and_metadata_capabilities(
    capabilities: DestinationFilesystemCapabilities, entry: CanonicalDeltaEntry, message: str
):
    """Each portable operation is denied if the destination cannot preserve it exactly."""
    with pytest.raises(UnrepresentableDeltaError, match=message):
        FilesystemRepresentabilityValidator().validate(
            CanonicalDelta(entries=(entry,)), capabilities
        )


def test_representability_rejects_case_unicode_and_unknown_normalization_aliases():
    """Destination equivalence rules reject aliases before any approval or staging."""
    validator = FilesystemRepresentabilityValidator()
    with pytest.raises(UnrepresentableDeltaError, match="aliases"):
        validator.validate(
            CanonicalDelta(entries=(_entry("Foo"), _entry("foo"))),
            _capabilities(case_sensitive=False),
        )
    composed = "caf\u00e9"
    decomposed = "cafe\u0301"
    with pytest.raises(UnrepresentableDeltaError, match="aliases"):
        validator.validate(
            CanonicalDelta(
                entries=tuple(
                    sorted(
                        (_entry(composed), _entry(decomposed)), key=lambda entry: entry.ordering_key
                    )
                )
            ),
            _capabilities(unicode_normalization="nfd"),
        )
    with pytest.raises(UnrepresentableDeltaError, match="Unicode comparison"):
        validator.validate(
            CanonicalDelta(entries=(_entry(composed),)),
            _capabilities(unicode_normalization="unknown"),
        )


def test_representability_rejects_destination_path_and_name_overflow():
    """UTF-8 component and full-path ceilings are enforced before a commit can be approved."""
    validator = FilesystemRepresentabilityValidator()
    with pytest.raises(UnrepresentableDeltaError, match="name limit"):
        validator.validate(
            CanonicalDelta(entries=(_entry("\u00e9\u00e9"),)), _capabilities(maximum_name_bytes=3)
        )
    with pytest.raises(UnrepresentableDeltaError, match="path limit"):
        validator.validate(
            CanonicalDelta(entries=(_entry("a/b"),)), _capabilities(maximum_path_bytes=2)
        )
