"""Test trusted macOS guest attempt archive inspection."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.execution.contracts import DeltaEffect
from loop.execution.infrastructure import InfrastructureProcessResult
from loop.execution.sandbox.macos import (
    MacosDeltaArchiveInspector,
    MacosWorkspaceCoordinator,
    MacosWorkspaceMaterializer,
)
from loop.execution.vfs import (
    AffectedPathIdentity,
    AgentWorkspaceManager,
    AuthenticatedWorkspaceRoot,
    BaseSnapshotId,
    BranchId,
    CanonicalDelta,
    CanonicalDeltaEntry,
    CommitBroker,
    ContentReference,
    DestinationFilesystemCapabilities,
    GenerationId,
    HostObjectIdentity,
    MetadataPreservation,
    ObjectKind,
    PublicationCoordinator,
    SnapshotManifest,
    SnapshotManifestEntry,
    StagedContentStore,
    TransactionJournal,
    UnrepresentableDeltaError,
)


def _identity(value: int) -> HostObjectIdentity:
    """Return deterministic host identity evidence."""
    return HostObjectIdentity(device=1, inode=value, ctime_ns=value)


def _content(value: bytes, reference: str) -> ContentReference:
    """Return digest evidence for one base-manifest payload."""
    return ContentReference(
        digest=f"sha256:{hashlib.sha256(value).hexdigest()}",
        size=len(value),
        reference=reference,
    )


def _base() -> SnapshotManifest:
    """Return a base with an opaque-directory victim, rename source, and mode target."""
    return SnapshotManifest(
        snapshot_id=BaseSnapshotId(value="base"),
        entries=(
            SnapshotManifestEntry(
                path="directory",
                object_kind=ObjectKind.DIRECTORY,
                identity=_identity(1),
                mode=0o755,
                size=0,
            ),
            SnapshotManifestEntry(
                path="directory/old",
                object_kind=ObjectKind.FILE,
                identity=_identity(2),
                content=_content(b"old", "base:old"),
                mode=0o644,
                size=3,
            ),
            SnapshotManifestEntry(
                path="mode",
                object_kind=ObjectKind.FILE,
                identity=_identity(3),
                content=_content(b"mode", "base:mode"),
                mode=0o644,
                size=4,
            ),
            SnapshotManifestEntry(
                path="source",
                object_kind=ObjectKind.FILE,
                identity=_identity(4),
                content=_content(b"rename", "base:source"),
                mode=0o644,
                size=6,
            ),
        ),
    )


def _add_file(bundle: tarfile.TarFile, name: str, value: bytes, mode: int = 0o644) -> None:
    """Add one regular payload to a test upper-layer archive."""
    member = tarfile.TarInfo(name)
    member.size = len(value)
    member.mode = mode
    bundle.addfile(member, io.BytesIO(value))


def test_archive_inspection_handles_opaque_deletes_metadata_and_rename(tmp_path: Path) -> None:
    """Both overlay metadata forms become complete portable effects with staged content."""
    archive = tmp_path / "upper.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as bundle:
        directory = tarfile.TarInfo("directory")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        directory.pax_headers = {"SCHILY.xattr.user.overlay.opaque": "y"}
        bundle.addfile(directory)
        _add_file(bundle, "directory/new", b"new")
        _add_file(bundle, "mode", b"mode", 0o600)
        whiteout = tarfile.TarInfo("source")
        whiteout.type = tarfile.CHRTYPE
        whiteout.devmajor = 0
        whiteout.devminor = 0
        bundle.addfile(whiteout)
        _add_file(bundle, "renamed", b"rename")

    store = StagedContentStore(tmp_path / "staging")
    delta = MacosDeltaArchiveInspector(store).inspect("attempt", archive, _base())

    assert [(entry.effect, entry.destination_path) for entry in delta.entries] == [
        (DeltaEffect.CREATE, "directory/new"),
        (DeltaEffect.DELETE, "directory/old"),
        (DeltaEffect.METADATA, "mode"),
        (DeltaEffect.RENAME, "renamed"),
    ]
    renamed = delta.entries[-1]
    assert renamed.source_path == "source"
    assert renamed.source_identity == _identity(4)
    created = delta.entries[0]
    assert created.content is not None
    with store.open(created.content) as stream:
        assert stream.read() == b"new"


def test_archive_inspection_handles_native_metacopy_redirect_metadata(tmp_path: Path) -> None:
    """Pinned-guest tar metadata for copy-up, chmod, and rename remains representable."""
    archive = tmp_path / "native-upper.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as bundle:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        root.pax_headers = {
            "SCHILY.xattr.trusted.overlay.uuid": "opaque-kernel-identity",
            "SCHILY.xattr.trusted.overlay.impure": "y",
        }
        bundle.addfile(root)
        _add_file(bundle, "replace", b"replacement")
        renamed = tarfile.TarInfo("renamed")
        renamed.size = 0
        renamed.mode = 0o644
        renamed.pax_headers = {
            "SCHILY.xattr.trusted.overlay.origin": "",
            "SCHILY.xattr.trusted.overlay.metacopy": "",
            "SCHILY.xattr.trusted.overlay.redirect": "source",
        }
        bundle.addfile(renamed, io.BytesIO())
        mode = tarfile.TarInfo("mode")
        mode.size = 0
        mode.mode = 0o600
        mode.pax_headers = {
            "SCHILY.xattr.trusted.overlay.origin": "",
            "SCHILY.xattr.trusted.overlay.metacopy": "",
        }
        bundle.addfile(mode, io.BytesIO())
        for name in ("source", "directory/old"):
            whiteout = tarfile.TarInfo(name)
            whiteout.type = tarfile.CHRTYPE
            whiteout.devmajor = 0
            whiteout.devminor = 0
            bundle.addfile(whiteout)

    delta = MacosDeltaArchiveInspector(StagedContentStore(tmp_path / "store")).inspect(
        "attempt", archive, _base()
    )

    assert [(entry.effect, entry.destination_path) for entry in delta.entries] == [
        (DeltaEffect.DELETE, "directory/old"),
        (DeltaEffect.METADATA, "mode"),
        (DeltaEffect.RENAME, "renamed"),
        (DeltaEffect.CREATE, "replace"),
    ]
    assert delta.entries[2].source_path == "source"


@pytest.mark.parametrize(
    ("member_type", "headers", "message"),
    (
        (tarfile.LNKTYPE, {}, "hardlink"),
        (tarfile.REGTYPE, {"SCHILY.xattr.security.capability": "value"}, "security xattrs"),
        (tarfile.REGTYPE, {"SCHILY.xattr.acl.access": "value"}, "ACL"),
    ),
)
def test_archive_inspection_rejects_unrepresentable_metadata(
    tmp_path: Path,
    member_type: bytes,
    headers: dict[str, str],
    message: str,
) -> None:
    """Hardlinks, ACLs, and security xattrs fail before host publication."""
    archive = tmp_path / f"{message}.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as bundle:
        member = tarfile.TarInfo("entry")
        member.type = member_type
        member.linkname = "source"
        member.pax_headers = headers
        bundle.addfile(member, io.BytesIO())
    inspector = MacosDeltaArchiveInspector(StagedContentStore(tmp_path / f"store-{message}"))
    with pytest.raises(UnrepresentableDeltaError, match=message):
        inspector.inspect("attempt", archive, _base())


def test_archive_inspection_rejects_unsafe_paths_and_bounds(tmp_path: Path) -> None:
    """Traversal, symlinked archives, and content excess fail closed."""
    archive = tmp_path / "upper.tar"
    with tarfile.open(archive, "w") as bundle:
        _add_file(bundle, "../escape", b"x")
    store = StagedContentStore(tmp_path / "staging")
    with pytest.raises(UnrepresentableDeltaError, match="unsafe path"):
        MacosDeltaArchiveInspector(store).inspect("attempt", archive, _base())

    archive.unlink()
    with tarfile.open(archive, "w") as bundle:
        _add_file(bundle, "file", b"too-large")
    with pytest.raises(UnrepresentableDeltaError, match="content quota"):
        MacosDeltaArchiveInspector(store, maximum_content_bytes=1).inspect(
            "attempt", archive, _base()
        )
    alias = tmp_path / "alias.tar"
    alias.symlink_to(archive)
    with pytest.raises(UnrepresentableDeltaError, match="regular file"):
        MacosDeltaArchiveInspector(store).inspect("attempt", alias, _base())


def test_archive_inspection_rejects_malformed_overlay_metadata_and_nodes(tmp_path: Path) -> None:
    """Malformed archives, redirects, metacopy, xattrs, and special nodes fail closed."""
    store = StagedContentStore(tmp_path / "store")
    with pytest.raises(ValueError, match="cannot be negative"):
        MacosDeltaArchiveInspector(store, maximum_entries=-1)
    with pytest.raises(UnrepresentableDeltaError, match="unavailable"):
        MacosDeltaArchiveInspector(store).inspect("attempt", tmp_path / "missing", _base())

    malformed = tmp_path / "malformed.tar"
    malformed.write_bytes(b"not a tar")
    with pytest.raises(UnrepresentableDeltaError, match="malformed"):
        MacosDeltaArchiveInspector(store).inspect("attempt", malformed, _base())

    entries = tmp_path / "entries.tar"
    with tarfile.open(entries, "w") as bundle:
        _add_file(bundle, "one", b"1")
    with pytest.raises(UnrepresentableDeltaError, match="entry quota"):
        MacosDeltaArchiveInspector(store, maximum_entries=0).inspect("attempt", entries, _base())

    cases = (
        (
            "unsafe-redirect",
            {"SCHILY.xattr.trusted.overlay.redirect": "../escape"},
            tarfile.REGTYPE,
            "redirect metadata is unsafe",
        ),
        (
            "missing-redirect",
            {"SCHILY.xattr.trusted.overlay.redirect": "missing"},
            tarfile.REGTYPE,
            "no visible lower object",
        ),
        (
            "conflicting-redirect",
            {
                "SCHILY.xattr.trusted.overlay.redirect": "source",
                "SCHILY.xattr.user.overlay.redirect": "source",
            },
            tarfile.REGTYPE,
            "conflicting spellings",
        ),
        (
            "missing-metacopy",
            {"SCHILY.xattr.trusted.overlay.metacopy": ""},
            tarfile.REGTYPE,
            "no regular lower object",
        ),
        (
            "ordinary-xattr",
            {"SCHILY.xattr.user.comment": "value"},
            tarfile.REGTYPE,
            "unsupported extended attributes",
        ),
        (
            "filesystem-metadata",
            {"SCHILY.fflags": "hidden"},
            tarfile.REGTYPE,
            "unsupported filesystem metadata",
        ),
        ("special-node", {}, tarfile.FIFOTYPE, "unsupported filesystem node"),
    )
    for name, headers, member_type, message in cases:
        archive = tmp_path / f"{name}.tar"
        with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as bundle:
            member = tarfile.TarInfo(name)
            member.type = member_type
            member.size = 0
            member.pax_headers = headers
            bundle.addfile(member, io.BytesIO())
        with pytest.raises(UnrepresentableDeltaError, match=message):
            MacosDeltaArchiveInspector(store).inspect(name, archive, _base())

    no_payload = tmp_path / "no-payload.tar"
    with tarfile.open(no_payload, "w") as bundle:
        _add_file(bundle, "file", b"value")
    patch = pytest.MonkeyPatch()
    patch.setattr(tarfile.TarFile, "extractfile", lambda *args: None)
    try:
        with pytest.raises(UnrepresentableDeltaError, match="no archive payload"):
            MacosDeltaArchiveInspector(store).inspect("no-payload", no_payload, _base())
    finally:
        patch.undo()


def test_archive_inspection_covers_whiteout_spellings_symlinks_and_committed_state(
    tmp_path: Path,
) -> None:
    """Portable state replay and every supported tar object spelling normalize completely."""
    committed = (
        CanonicalDelta(
            entries=(
                CanonicalDeltaEntry(
                    effect=DeltaEffect.DELETE,
                    destination_path="directory/old",
                    destination_identity=_identity(2),
                ),
            )
        ),
        CanonicalDelta(
            entries=(
                CanonicalDeltaEntry(
                    effect=DeltaEffect.RENAME,
                    destination_path="moved",
                    source_path="source",
                    source_identity=_identity(4),
                    object_kind=ObjectKind.FILE,
                    mode=0o644,
                ),
            )
        ),
        CanonicalDelta(
            entries=(
                CanonicalDeltaEntry(
                    effect=DeltaEffect.METADATA,
                    destination_path="mode",
                    destination_identity=_identity(3),
                    object_kind=ObjectKind.FILE,
                    content=_content(b"mode", "base:mode"),
                    mode=0o600,
                    metadata=MetadataPreservation(basic_mode=0o600),
                ),
            )
        ),
        CanonicalDelta(
            entries=(
                CanonicalDeltaEntry(
                    effect=DeltaEffect.CREATE,
                    destination_path="created-directory",
                    object_kind=ObjectKind.DIRECTORY,
                    mode=0o755,
                ),
            )
        ),
        CanonicalDelta(
            entries=(
                CanonicalDeltaEntry(
                    effect=DeltaEffect.CREATE,
                    destination_path="created-file",
                    object_kind=ObjectKind.FILE,
                    content=_content(b"created", "committed:created"),
                    mode=0o644,
                ),
            )
        ),
    )
    archive = tmp_path / "upper.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as bundle:
        opaque = tarfile.TarInfo(".wh..wh..opq")
        opaque.type = tarfile.REGTYPE
        bundle.addfile(opaque, io.BytesIO())
        whiteout = tarfile.TarInfo(".wh.moved")
        whiteout.type = tarfile.REGTYPE
        bundle.addfile(whiteout, io.BytesIO())
        xattr_whiteout = tarfile.TarInfo("mode")
        xattr_whiteout.type = tarfile.REGTYPE
        xattr_whiteout.pax_headers = {"SCHILY.xattr.trusted.overlay.whiteout": ""}
        bundle.addfile(xattr_whiteout, io.BytesIO())
        directory = tarfile.TarInfo("directory")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        bundle.addfile(directory)
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        link.linkname = "target"
        link.pax_headers = {"comment": "portable"}
        bundle.addfile(link)
        _add_file(bundle, "././nested", b"nested")

    delta = MacosDeltaArchiveInspector(StagedContentStore(tmp_path / "store")).inspect(
        "attempt",
        archive,
        _base(),
        committed,
        (AffectedPathIdentity(path="moved", identity=_identity(40)),),
    )

    assert any(entry.destination_path == "link" for entry in delta.entries)
    assert any(entry.destination_path == "moved" for entry in delta.entries)


class _GuestRunner:
    """Record fixed guest and copy operations without starting a VM."""

    def __init__(self) -> None:
        """Initialize deterministic command and archive evidence."""
        self.calls: list[tuple[str, object]] = []
        self.applied: set[str] = set()
        self.payload_members: list[str] = []
        self.branch_recovery_evidence = b"recovered\n"

    def run_guest(self, context: object, argv: tuple[str, ...], **kwargs: object):
        """Return status misses until the corresponding completion command runs."""
        del context
        operation = str(kwargs["operation"])
        self.calls.append((operation, argv))
        if operation == "workspace.branch.recover":
            return InfrastructureProcessResult(0, self.branch_recovery_evidence, b"", False, False)
        if operation == "workspace.transaction.status":
            transaction = argv[-1].removesuffix("/applied")
            return InfrastructureProcessResult(
                0 if transaction in self.applied else 1, b"", b"", False, False
            )
        if operation == "workspace.transaction.complete":
            self.applied.add(argv[-1])
        return InfrastructureProcessResult(0, b"", b"", False, False)

    def copy_to_guest(self, context: object, source: Path, destination: str):
        """Inspect the bounded host payload before reporting a successful copy."""
        del context
        self.calls.append(("copy_to_guest", destination))
        with tarfile.open(source, "r") as bundle:
            self.payload_members.extend(bundle.getnames())
        return InfrastructureProcessResult(0, b"", b"", False, False)

    def copy_from_guest(self, context: object, source: str, destination: Path):
        """Materialize one empty but valid trusted guest archive."""
        del context
        self.calls.append(("copy_from_guest", source))
        with tarfile.open(destination, "w"):
            pass
        return InfrastructureProcessResult(0, b"", b"", False, False)


def _materializer(tmp_path: Path) -> tuple[MacosWorkspaceMaterializer, _GuestRunner]:
    """Build one production materializer over an isolated fake managed guest."""
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True)
    runner = _GuestRunner()
    prepared = SimpleNamespace(
        endpoint=SimpleNamespace(state_path="/home/loop/.local/share/containerd"),
        instance=SimpleNamespace(
            runner=runner,
            context=SimpleNamespace(state_root=state_root),
        ),
    )
    content_store = StagedContentStore(state_root / "content")
    return (
        MacosWorkspaceMaterializer(prepared, content_store, state_root / "archives"),  # type: ignore[arg-type]
        runner,
    )


def test_materializer_owns_branch_attempt_archive_replay_and_cleanup(tmp_path: Path) -> None:
    """Opaque common references compile into fixed guest mounts, replay, export, and teardown."""
    materializer, runner = _materializer(tmp_path)
    branch = materializer.create_branch(BaseSnapshotId(value="base"))
    materializer.recover_branch(BaseSnapshotId(value="base"), branch, ())
    recovery = next(call for call in runner.calls if call[0] == "workspace.branch.recover")
    assert "mountpoint -q" in recovery[1][2]
    initialization = next(call for call in runner.calls if call[0] == "workspace.branch.initialize")
    assert "/etc/subuid" in initialization[1][2]
    assert "/etc/subgid" in initialization[1][2]
    assert 'chmod 755 "$branch/merged"' in initialization[1][2]
    view = materializer.generation_view(branch, GenerationId(value="0"))
    assert view.reference == branch.value
    attempt = materializer.begin_attempt(view, "attempt")
    attempt_create = next(call for call in runner.calls if call[0] == "workspace.attempt.create")
    assert attempt_create[1][-1] == str(256 * 1024 * 1024)
    assert 'chmod 755 "$target/merged"' in attempt_create[1][2]
    assert "/etc/subuid" in attempt_create[1][2]
    assert "/etc/subgid" in attempt_create[1][2]
    assert 'chown "$mapped_uid:$mapped_gid"' in attempt_create[1][2]
    assert 'sudo -n chown "$(id -u):$(id -g)" "$target/layer"' in attempt_create[1][2]
    archive = materializer.archive_attempt(attempt)
    assert archive.is_file()

    content = materializer.content_store.stage("transaction", io.BytesIO(b"new"), maximum_bytes=3)
    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="file",
                object_kind=ObjectKind.FILE,
                content=content,
                mode=0o640,
            ),
            CanonicalDeltaEntry(
                effect=DeltaEffect.METADATA,
                destination_path="mode",
                object_kind=ObjectKind.FILE,
                content=_content(b"base", "content:base"),
                mode=0o600,
            ),
            CanonicalDeltaEntry(
                effect=DeltaEffect.RENAME,
                destination_path="renamed",
                source_path="old",
                object_kind=ObjectKind.FILE,
                mode=0o644,
            ),
        )
    )
    materializer.apply_delta(branch, "transaction", delta)
    calls_after_first_apply = len(runner.calls)
    transaction_operations = [call[0] for call in runner.calls]
    assert transaction_operations[-2:] == [
        "workspace.branch.initialize",
        "workspace.transaction.complete",
    ]
    materializer.apply_delta(branch, "transaction", delta)

    assert runner.payload_members == ["0"]
    assert len(runner.calls) == calls_after_first_apply + 1
    assert runner.calls[-1][0] == "workspace.transaction.status"
    materializer.discard_attempt(attempt)
    assert not archive.exists()
    materializer.dispose_branch(branch)
    assert runner.calls[-1][0] == "workspace.branch.dispose"

    with pytest.raises(ValueError, match="quota"):
        MacosWorkspaceMaterializer(
            materializer.prepared,
            materializer.content_store,
            materializer.archive_directory,
            attempt_write_bytes=0,
        )


def test_materializer_reconstructs_a_branch_lost_with_replaced_guest_state(
    tmp_path: Path,
) -> None:
    """Recovery recreates a missing branch and replays every durable transaction."""
    materializer, runner = _materializer(tmp_path)
    runner.branch_recovery_evidence = b"reconstructed\n"
    branch = BranchId(value="branch-restored")

    materializer.recover_branch(
        BaseSnapshotId(value="base"),
        branch,
        (("committed", CanonicalDelta()),),
    )

    operations = [call[0] for call in runner.calls]
    assert operations[:2] == [
        "workspace.branch.recover",
        "workspace.branch.initialize",
    ]
    assert operations[-1] == "workspace.transaction.complete"


def test_materializer_reconstruction_rejects_ambiguous_evidence_and_cleans_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reconstruction fails closed on ambiguous status and removes partial guest state."""
    materializer, runner = _materializer(tmp_path)
    runner.branch_recovery_evidence = b"ambiguous\n"
    branch = BranchId(value="branch-restored")
    with pytest.raises(RuntimeError, match="invalid evidence"):
        materializer.recover_branch(BaseSnapshotId(value="base"), branch, ())

    original_run = runner.run_guest

    def fail_initialize(context: object, argv: tuple[str, ...], **kwargs: object):
        """Fail initialization after the missing branch overlay was reconstructed."""
        if kwargs["operation"] == "workspace.branch.initialize":
            raise RuntimeError("initialize")
        return original_run(context, argv, **kwargs)

    monkeypatch.setattr(runner, "run_guest", fail_initialize)
    runner.branch_recovery_evidence = b"recovered\n"
    with pytest.raises(RuntimeError, match="initialize"):
        materializer.recover_branch(BaseSnapshotId(value="base"), branch, ())
    assert runner.calls[-1][0] == "workspace.branch.recover"

    runner.branch_recovery_evidence = b"reconstructed\n"
    with pytest.raises(RuntimeError, match="initialize"):
        materializer.recover_branch(BaseSnapshotId(value="base"), branch, ())
    assert runner.calls[-1][0] == "workspace.branch.dispose"


def test_materializer_forks_by_replaying_canonical_transactions(tmp_path: Path) -> None:
    """A child receives an independent branch populated from identified canonical effects."""
    materializer, runner = _materializer(tmp_path)
    delta = CanonicalDelta()
    branch = materializer.fork_branch(
        BaseSnapshotId(value="base"), (("parent-transaction", delta),)
    )

    assert branch.value.startswith("branch-")
    assert any(call[0] == "workspace.branch.create" for call in runner.calls)
    assert any(call[0] == "workspace.transaction.complete" for call in runner.calls)


def test_materializer_rejects_unsafe_identities_and_private_state(tmp_path: Path) -> None:
    """Guest path derivation and opaque identities fail closed before management commands."""
    materializer, _ = _materializer(tmp_path / "valid")
    with pytest.raises(ValueError, match="snapshot identity"):
        materializer.create_branch(BaseSnapshotId(value="../escape"))
    with pytest.raises(ValueError, match="generation identity"):
        materializer.generation_view(
            BranchId(value="branch"),
            GenerationId(value="bad"),
        )

    state_root = tmp_path / "bad" / "state"
    state_root.mkdir(parents=True)
    prepared = SimpleNamespace(
        endpoint=SimpleNamespace(state_path="/foreign/containerd"),
        instance=SimpleNamespace(context=SimpleNamespace(state_root=state_root)),
    )
    with pytest.raises(ValueError, match="private home"):
        MacosWorkspaceMaterializer(  # type: ignore[arg-type]
            prepared,
            StagedContentStore(state_root / "content"),
            state_root / "archives",
        )

    materializer, _ = _materializer(tmp_path / "identities")
    with pytest.raises(ValueError, match="transaction identity"):
        materializer.apply_delta(BranchId(value="branch"), "", CanonicalDelta())

    outside = tmp_path / "outside"
    outside.mkdir()
    prepared = SimpleNamespace(
        endpoint=SimpleNamespace(state_path="/home/loop/.local/share/containerd"),
        instance=SimpleNamespace(
            runner=_GuestRunner(),
            context=SimpleNamespace(state_root=state_root),
        ),
    )
    with pytest.raises(ValueError, match="outside private instance state"):
        MacosWorkspaceMaterializer(  # type: ignore[arg-type]
            prepared,
            StagedContentStore(state_root / "content-outside"),
            outside / "archives",
        )
    real_archive = state_root / "real-archives"
    real_archive.mkdir()
    archive_alias = state_root / "archive-alias"
    archive_alias.symlink_to(real_archive, target_is_directory=True)
    with pytest.raises(ValueError, match="private directory"):
        MacosWorkspaceMaterializer(  # type: ignore[arg-type]
            prepared,
            StagedContentStore(state_root / "content-alias"),
            archive_alias,
        )


def test_materializer_cleans_partial_branches_and_payloads_on_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Initialization, fork, status, transfer, and payload failures leave no host artifacts."""
    materializer, runner = _materializer(tmp_path)
    original_run = runner.run_guest

    def fail_initialize(context: object, argv: tuple[str, ...], **kwargs: object):
        """Fail branch initialization after its overlay was created."""
        if kwargs["operation"] == "workspace.branch.initialize":
            raise RuntimeError("initialize")
        return original_run(context, argv, **kwargs)

    monkeypatch.setattr(runner, "run_guest", fail_initialize)
    with pytest.raises(RuntimeError, match="initialize"):
        materializer.create_branch(BaseSnapshotId(value="base"))
    assert runner.calls[-1][0] == "workspace.branch.dispose"
    monkeypatch.setattr(runner, "run_guest", original_run)

    branch = materializer.create_branch(BaseSnapshotId(value="base"))

    def invalid_status(*_: object, **__: object) -> InfrastructureProcessResult:
        """Return an impossible transaction status."""
        return InfrastructureProcessResult(2, b"", b"", False, False)

    monkeypatch.setattr(runner, "run_guest", invalid_status)
    with pytest.raises(RuntimeError, match="transaction status"):
        materializer.apply_delta(branch, "status", CanonicalDelta())
    monkeypatch.setattr(runner, "run_guest", original_run)

    monkeypatch.setattr(
        runner,
        "copy_to_guest",
        lambda *args: InfrastructureProcessResult(1, b"", b"", False, False),
    )
    with pytest.raises(RuntimeError, match="payload copy"):
        materializer.apply_delta(branch, "copy", CanonicalDelta())
    assert not list(materializer.archive_directory.glob("transaction-*.tar"))

    def fail_open(_: ContentReference):
        """Reject staged content during payload assembly."""
        raise ValueError("content")

    monkeypatch.setattr(materializer.content_store, "open", fail_open)
    content = ContentReference(
        digest="sha256:" + "0" * 64,
        size=1,
        reference="staged:" + "0" * 64 + ":" + "0" * 64,
    )
    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="file",
                object_kind=ObjectKind.FILE,
                content=content,
            ),
        )
    )
    with pytest.raises(ValueError, match="content"):
        materializer.apply_delta(branch, "payload", delta)
    assert not list(materializer.archive_directory.glob("transaction-*.tar"))

    monkeypatch.setattr(
        materializer, "apply_delta", lambda *args: (_ for _ in ()).throw(RuntimeError("fork"))
    )
    with pytest.raises(RuntimeError, match="fork"):
        materializer.fork_branch(BaseSnapshotId(value="base"), (("tx", CanonicalDelta()),))
    assert runner.calls[-1][0] == "workspace.branch.dispose"


def _coordinator(
    tmp_path: Path,
) -> tuple[MacosWorkspaceCoordinator, MacosWorkspaceMaterializer, Path]:
    """Build one real portable publication flow over fake guest management."""
    materializer, _ = _materializer(tmp_path)
    manager = AgentWorkspaceManager(
        materializer,
        lambda context: True,
        tmp_path / "state" / "contexts",
    )
    base = SnapshotManifest(snapshot_id=BaseSnapshotId(value="base"))
    manager.create("workspace", "agent", base.snapshot_id)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = AuthenticatedWorkspaceRoot(workspace)
    journal = TransactionJournal(tmp_path / "state" / "journal")
    publication = PublicationCoordinator(
        manager,
        CommitBroker(root, journal, materializer.content_store),
        journal,
    )
    coordinator = MacosWorkspaceCoordinator(
        manager,
        materializer,
        MacosDeltaArchiveInspector(materializer.content_store),
        publication,
        lambda snapshot_id: (
            base
            if snapshot_id == base.snapshot_id
            else (_ for _ in ()).throw(KeyError(snapshot_id))
        ),
        DestinationFilesystemCapabilities(
            case_sensitive=True,
            unicode_normalization="none",
            maximum_name_bytes=255,
            maximum_path_bytes=4096,
            supports_symlinks=True,
            supports_basic_mode=True,
            preserves_ownership=False,
            preserves_acl=False,
            preserves_security_xattrs=False,
        ),
    )
    return coordinator, materializer, workspace


def test_workspace_coordinator_denies_without_journal_and_publishes_after_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Denied effects vanish, while approved effects publish and advance only after lease release."""
    coordinator, materializer, workspace = _coordinator(tmp_path)

    def observed_archive(attempt: object) -> Path:
        """Return one trusted archive containing a single created workspace file."""
        archive = tmp_path / f"{attempt.layer.attempt_id}.tar"  # type: ignore[attr-defined]
        with tarfile.open(archive, "w") as bundle:
            _add_file(bundle, "created", b"value")
        return archive

    monkeypatch.setattr(
        materializer,
        "archive_attempt",
        lambda layer: observed_archive(SimpleNamespace(layer=layer)),
    )
    with coordinator.lease("workspace", "agent", "denied") as denied:
        coordinator.observe(denied)
    coordinator.deny(denied)
    assert not (workspace / "created").exists()
    assert coordinator.publication.journal.transactions() == ()

    with coordinator.lease("workspace", "agent", "approved") as approved:
        approved_delta = coordinator.observe(approved)
        with pytest.raises(RuntimeError, match="active generation leases"):
            coordinator.publish(approved, "transaction", approved_delta)
    updated = coordinator.publish(approved, "transaction", approved_delta)

    assert (workspace / "created").read_bytes() == b"value"
    assert updated.generation_id == GenerationId(value="1")
    transaction = coordinator.publication.journal.load("transaction")
    assert transaction is not None and transaction.phase.value == "committed"


def test_durable_workspace_lease_releases_resumes_and_discards_exact_generation(
    tmp_path: Path,
) -> None:
    """Durable jobs retain one generation across operations and reacquire it after restart."""
    coordinator, materializer, _ = _coordinator(tmp_path)
    job = coordinator.lease_job("workspace", "agent", "job")
    layer = job.leased.layer
    generation = job.leased.generation_reference
    with pytest.raises(RuntimeError, match="active generation leases"):
        coordinator.manager.require_publishable(job.leased.context)

    job.release()
    job.release()
    coordinator.manager.require_publishable(job.leased.context)
    resumed = coordinator.resume_job("workspace", "agent", generation, layer)
    resumed.discard()
    resumed.discard()
    coordinator.manager.require_publishable(resumed.leased.context)

    with pytest.raises(RuntimeError, match="stale"):
        coordinator.resume_job("workspace", "agent", "generation-stale", layer)
    coordinator.manager.require_publishable(resumed.leased.context)
    assert any(
        call[0] == "workspace.attempt.dispose"
        for call in materializer.prepared.instance.runner.calls
    )


def test_durable_workspace_creation_failure_releases_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed attempt overlay creation cannot strand a logical generation lease."""
    coordinator, materializer, _ = _coordinator(tmp_path)
    context = coordinator.manager.get("workspace", "agent")
    monkeypatch.setattr(
        materializer,
        "begin_attempt",
        lambda view, attempt: (_ for _ in ()).throw(RuntimeError("create")),
    )

    with pytest.raises(RuntimeError, match="create"):
        coordinator.lease_job("workspace", "agent", "job")
    coordinator.manager.require_publishable(context)
