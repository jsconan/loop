"""Tests for portable, deterministic workspace VFS models."""

import ast
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from loop.execution.contracts import DeltaEffect
from loop.execution.vfs import (
    BaseSnapshotId,
    BranchId,
    CanonicalDelta,
    CanonicalDeltaEntry,
    ContentReference,
    DestinationFilesystemCapabilities,
    GenerationId,
    GenerationView,
    HostObjectIdentity,
    MetadataPreservation,
    ObjectKind,
    SnapshotManifest,
    SnapshotManifestEntry,
    WorkspaceMaterializer,
)
from loop.execution.vfs.namespace import validate_relative_path_or_none


def _content() -> ContentReference:
    """Return one opaque file-content reference."""
    return ContentReference(digest="sha256:ab", size=2, reference="content:one")


def _entry(path: str = "file") -> CanonicalDeltaEntry:
    """Return one canonical regular-file creation."""
    return CanonicalDeltaEntry(
        effect=DeltaEffect.CREATE,
        destination_path=path,
        object_kind=ObjectKind.FILE,
        content=_content(),
        mode=0o644,
    )


def test_canonical_delta_round_trips_only_deterministic_virtual_records():
    """Persisted deltas retain canonical ordering without host locations."""
    delta = CanonicalDelta(entries=(_entry("a"), _entry("z")))

    restored = CanonicalDelta.model_validate_json(delta.model_dump_json())

    assert restored == delta
    assert "workspace" not in delta.model_dump_json()
    with pytest.raises(ValidationError, match="deterministic ordering"):
        CanonicalDelta(entries=(_entry("z"), _entry("a")))
    with pytest.raises(ValidationError, match="duplicate destinations"):
        CanonicalDelta(entries=(_entry("same"), _entry("same")))


@pytest.mark.parametrize("path", ("/workspace/file", "../file", "a//b", "a\\b", "."))
def test_vfs_records_reject_host_or_escaping_path_representations(path: str):
    """VFS records admit only normalized workspace-relative POSIX paths."""
    with pytest.raises(ValidationError, match="relative virtual paths|traversal"):
        _entry(path)


def test_canonical_entry_requires_unambiguous_effect_specific_data():
    """Effect records reject missing rename/link evidence and invalid result shapes."""
    with pytest.raises(ValidationError, match="Rename"):
        CanonicalDeltaEntry(
            effect=DeltaEffect.RENAME,
            destination_path="new",
            object_kind=ObjectKind.FILE,
        )
    with pytest.raises(ValidationError, match="Symlink"):
        CanonicalDeltaEntry(
            effect=DeltaEffect.CREATE,
            destination_path="link",
            object_kind=ObjectKind.SYMLINK,
        )
    with pytest.raises(ValidationError, match="Delete"):
        CanonicalDeltaEntry(
            effect=DeltaEffect.DELETE,
            destination_path="gone",
            object_kind=ObjectKind.FILE,
        )
    with pytest.raises(ValidationError, match="Only rename"):
        CanonicalDeltaEntry(
            effect=DeltaEffect.CREATE,
            destination_path="new",
            source_path="old",
            object_kind=ObjectKind.DIRECTORY,
        )
    with pytest.raises(ValidationError, match="Non-delete"):
        CanonicalDeltaEntry(effect=DeltaEffect.CREATE, destination_path="new")
    with pytest.raises(ValidationError, match="Regular files"):
        CanonicalDeltaEntry(
            effect=DeltaEffect.CREATE,
            destination_path="file",
            object_kind=ObjectKind.FILE,
            symlink_target="target",
        )
    with pytest.raises(ValidationError, match="Only regular files"):
        CanonicalDeltaEntry(
            effect=DeltaEffect.CREATE,
            destination_path="directory",
            object_kind=ObjectKind.DIRECTORY,
            content=_content(),
        )
    assert CanonicalDeltaEntry(effect=DeltaEffect.DELETE, destination_path="gone").effect is (
        DeltaEffect.DELETE
    )
    assert (
        CanonicalDeltaEntry(
            effect=DeltaEffect.RENAME,
            destination_path="new",
            source_path="old",
            object_kind=ObjectKind.FILE,
        ).source_path
        == "old"
    )


def test_manifest_round_trips_ordered_versioned_snapshot_evidence():
    """Snapshot manifests preserve sorted virtual identities and content evidence."""
    identity = HostObjectIdentity(device=1, inode=2, ctime_ns=3)
    manifest = SnapshotManifest(
        snapshot_id=BaseSnapshotId(value="base"),
        entries=(
            SnapshotManifestEntry(
                path="file",
                object_kind=ObjectKind.FILE,
                identity=identity,
                content=_content(),
                mode=0o644,
                size=2,
                metadata=MetadataPreservation(basic_mode=0o644),
            ),
        ),
    )

    assert SnapshotManifest.model_validate_json(manifest.model_dump_json()) == manifest
    with pytest.raises(ValidationError, match="uniquely path-sorted"):
        SnapshotManifest(
            snapshot_id=BaseSnapshotId(value="base"),
            entries=(
                manifest.entries[0].model_copy(update={"path": "z"}),
                manifest.entries[0].model_copy(update={"path": "a"}),
            ),
        )
    with pytest.raises(ValidationError, match="Regular files"):
        SnapshotManifestEntry(
            path="file",
            object_kind=ObjectKind.FILE,
            identity=identity,
            mode=0o644,
            size=0,
        )
    with pytest.raises(ValidationError, match="Symbolic links"):
        SnapshotManifestEntry(
            path="link",
            object_kind=ObjectKind.SYMLINK,
            identity=identity,
            mode=0o777,
            size=0,
        )
    with pytest.raises(ValidationError, match="Only symbolic links"):
        SnapshotManifestEntry(
            path="directory",
            object_kind=ObjectKind.DIRECTORY,
            identity=identity,
            mode=0o755,
            size=0,
            symlink_target="target",
        )


def test_host_object_identity_can_be_created_from_status_metadata(tmp_path: Path):
    """Status metadata maps to the identity fields used for race detection."""
    metadata = os.stat(tmp_path)

    assert HostObjectIdentity.from_metadata(metadata) == HostObjectIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        ctime_ns=metadata.st_ctime_ns,
    )


def test_optional_virtual_path_validation_retains_none_without_host_aliases():
    """Optional path fields distinguish absent values from invalid path spellings."""
    assert validate_relative_path_or_none(None) is None
    assert validate_relative_path_or_none("nested/file") == "nested/file"


def test_capabilities_are_immutable_and_versioned():
    """Destination capabilities retain every representability-relevant fact."""
    capabilities = DestinationFilesystemCapabilities(
        case_sensitive=True,
        unicode_normalization="nfc",
        maximum_name_bytes=255,
        maximum_path_bytes=4096,
        supports_symlinks=True,
        supports_basic_mode=True,
        preserves_ownership=False,
        preserves_acl=False,
        preserves_security_xattrs=False,
    )

    assert (
        DestinationFilesystemCapabilities.model_validate_json(capabilities.model_dump_json())
        == capabilities
    )
    with pytest.raises(ValidationError):
        capabilities.case_sensitive = False


class _DeterministicMaterializer:
    """Provide a test-only implementation of the opaque materializer seam."""

    def create_branch(self, base: BaseSnapshotId) -> BranchId:
        """Return a deterministic branch for a base."""
        return BranchId(value=f"branch:{base.value}")

    def recover_branch(
        self,
        base: BaseSnapshotId,
        branch: BranchId,
        committed_transactions: tuple[tuple[str, CanonicalDelta], ...],
    ) -> None:
        """Accept persisted branch recovery without a runtime resource."""
        del base, branch, committed_transactions

    def generation_view(self, branch: BranchId, generation: GenerationId) -> GenerationView:
        """Return one opaque view without a host path."""
        del branch, generation
        return GenerationView()

    def apply_delta(self, branch: BranchId, transaction_id: str, delta: CanonicalDelta) -> None:
        """Accept one idempotent test transaction."""
        del branch, transaction_id, delta

    def fork_branch(
        self,
        base: BaseSnapshotId,
        committed_transactions: tuple[tuple[str, CanonicalDelta], ...],
    ) -> BranchId:
        """Return a deterministic independent replay branch."""
        return BranchId(value=f"fork:{base.value}:{len(committed_transactions)}")

    def dispose_branch(self, branch: BranchId) -> None:
        """Dispose a test branch without a runtime resource."""
        del branch


def test_workspace_materializer_protocol_exposes_only_opaque_operations():
    """A deterministic test materializer conforms without paths or runtime handles."""
    materializer = _DeterministicMaterializer()

    assert isinstance(materializer, WorkspaceMaterializer)
    assert materializer.create_branch(BaseSnapshotId(value="base")) == BranchId(value="branch:base")
    assert isinstance(
        materializer.generation_view(BranchId(value="branch"), GenerationId(value="1")),
        GenerationView,
    )


def test_vfs_package_has_no_runtime_or_host_process_authority_imports():
    """The portable VFS cannot acquire sandbox, runtime, host, or subprocess authority."""
    root = Path(__file__).parents[3] / "src" / "loop" / "execution" / "vfs"
    forbidden = {
        "subprocess",
        "loop.execution.runtime",
        "loop.execution.sandbox",
        "loop.execution.host",
    }
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not imports & forbidden
