"""Test fail-closed managed runtime bootstrap boundaries."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from loop.execution.runtime.activation import (
    ActivationError,
    activate,
    active_set,
    rollback,
    rollback_set,
)
from loop.execution.runtime.bootstrap import (
    InstalledRuntime,
    RuntimeBootstrapper,
    RuntimeRequirement,
)
from loop.execution.runtime.download import (
    AcquisitionCancelled,
    AcquisitionError,
    ArtifactUnavailable,
    HttpxArtifactTransport,
    IntegrityFailure,
    QuotaExceeded,
    download_artifact,
)
from loop.execution.runtime.gc import collect, reset_inactive
from loop.execution.runtime.install import (
    ArchiveInstaller,
    FileInstaller,
    InstallError,
    UnsafeArchive,
    install_content,
    verify_content,
)
from loop.execution.runtime.lease import active_leases, create_lease
from loop.execution.runtime.manifest import (
    ManifestError,
    RuntimeManifest,
    load_checked_in_candidate,
    load_embedded_release,
)
from loop.execution.runtime.models import (
    AcquisitionKind,
    Artifact,
    ArtifactDependency,
    ArtifactMetadata,
    ArtifactRole,
    InstallLayout,
    ManifestMode,
    OciImageIdentity,
    PlatformSelector,
)


class MemoryTransport:
    """Serve one pinned in-memory response."""

    def __init__(self, body: bytes, url: str = "https://example.test/artifact") -> None:
        self.body = body
        self.url = url
        self.calls = 0

    def stream(self, _url: str) -> Iterator[tuple[str, Iterator[bytes]]]:
        """Yield the configured response."""
        self.calls += 1
        yield self.url, iter((self.body,))


def _artifact(body: bytes, kind: AcquisitionKind = AcquisitionKind.FILE) -> Artifact:
    """Build a single fixture artifact with an exact digest."""
    return Artifact(
        artifact_id="runtime",
        version="1",
        role=ArtifactRole.NERDCTL,
        platform=PlatformSelector(os="linux", architecture="amd64"),
        source="https://example.test/artifact",
        size=len(body),
        digest=f"sha256:{hashlib.sha256(body).hexdigest()}",
        acquisition=kind,
        media_type="application/octet-stream",
        layout=InstallLayout(
            files=("bin/tool",),
            executables=("bin/tool",),
            identities={
                "bin/tool": hashlib.sha256(
                    body if kind is AcquisitionKind.FILE else b"x"
                ).hexdigest()
            },
        ),
        capabilities=("core",),
        sbom="https://example.test/sbom",
        notices="https://example.test/notices",
    )


def _manifest(body: bytes) -> RuntimeManifest:
    """Build one fixture-only test manifest."""
    return RuntimeManifest(
        schema_version=1, mode=ManifestMode.FIXTURE, artifacts=(_artifact(body),)
    )


def test_bootstrap_downloads_once_then_reuses_verified_content(tmp_path: Path):
    """Warm bootstrap reuses immutable content without another download."""
    transport = MemoryTransport(b"tool")
    bootstrapper = RuntimeBootstrapper(
        _manifest(b"tool"), tmp_path, transport, {AcquisitionKind.FILE: FileInstaller()}, 1024
    )
    requirement = RuntimeRequirement(
        PlatformSelector(os="linux", architecture="amd64"), frozenset({"core"})
    )
    first = bootstrapper.ensure(requirement)
    second = bootstrapper.ensure(requirement)
    assert transport.calls == 1
    assert first.artifact_set_digest == second.artifact_set_digest
    assert first.artifacts["runtime"].joinpath("bin/tool").read_bytes() == b"tool"


def test_bootstrap_rejects_precreated_incomplete_content(tmp_path: Path):
    """A pre-created digest directory is never trusted without layout verification."""
    artifact = _artifact(b"tool")
    incomplete = tmp_path / "artifacts" / artifact.artifact_id / artifact.digest[7:]
    incomplete.mkdir(parents=True)
    transport = MemoryTransport(b"tool")
    bootstrapper = RuntimeBootstrapper(
        _manifest(b"tool"), tmp_path, transport, {AcquisitionKind.FILE: FileInstaller()}, 1024
    )
    with pytest.raises(InstallError, match="failed verification"):
        bootstrapper.ensure(
            RuntimeRequirement(
                PlatformSelector(os="linux", architecture="amd64"), frozenset({"core"})
            )
        )
    assert transport.calls == 1


def test_download_rejects_unapproved_transport_origin(tmp_path: Path):
    """A test transport cannot bypass manifest final-origin restrictions."""
    transport = MemoryTransport(b"tool", "https://evil.test/artifact")
    bootstrapper = RuntimeBootstrapper(
        _manifest(b"tool"), tmp_path, transport, {AcquisitionKind.FILE: FileInstaller()}, 1024
    )
    with pytest.raises(AcquisitionError, match="unapproved"):
        bootstrapper.ensure(
            RuntimeRequirement(
                PlatformSelector(os="linux", architecture="amd64"), frozenset({"core"})
            )
        )


def test_bootstrap_cancellation_creates_no_content(tmp_path: Path):
    """Cancellation leaves no artifact content available for selection."""
    bootstrapper = RuntimeBootstrapper(
        _manifest(b"tool"),
        tmp_path,
        MemoryTransport(b"tool"),
        {AcquisitionKind.FILE: FileInstaller()},
        1024,
    )
    with pytest.raises(AcquisitionCancelled):
        bootstrapper.ensure(
            RuntimeRequirement(
                PlatformSelector(os="linux", architecture="amd64"), frozenset({"core"})
            ),
            cancellation=lambda: True,
        )
    assert not (tmp_path / "artifacts").exists()


def test_manifest_rejects_unknown_dependency_and_non_release_payload():
    """Manifest validation closes dependency and production trust boundaries."""
    artifact = _artifact(b"x").model_copy(
        update={"dependencies": (ArtifactDependency(artifact_id="missing"),)}
    )
    with pytest.raises(ValueError, match="unknown dependency"):
        RuntimeManifest(schema_version=1, mode=ManifestMode.FIXTURE, artifacts=(artifact,))
    with pytest.raises(ManifestError, match="release"):
        load_embedded_release(_manifest(b"x").model_dump_json().encode())


def test_activation_is_compare_and_swap(tmp_path: Path):
    """Activation rejects stale callers and preserves the active digest."""
    digest = "a" * 64
    assert activate(tmp_path, digest) is None
    assert active_set(tmp_path) == digest
    with pytest.raises(ActivationError, match="changed"):
        activate(tmp_path, "b" * 64, expected=None)


def test_archive_installer_rejects_link_and_extracts_regular_file(tmp_path: Path):
    """Archive installation accepts only declared regular-file content."""
    archive = tmp_path / "artifact.tar"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("bin/tool")
        member.size = 1
        output.addfile(member, io.BytesIO(b"x"))
    destination = tmp_path / "destination"
    destination.mkdir()
    ArchiveInstaller().install(
        _artifact(archive.read_bytes(), AcquisitionKind.ARCHIVE), archive, destination
    )
    assert (destination / "bin/tool").read_bytes() == b"x"
    with tarfile.open(archive, "w") as output:
        output.addfile(tarfile.TarInfo("../../escape"))
    with pytest.raises(InstallError):
        ArchiveInstaller().install(
            _artifact(archive.read_bytes(), AcquisitionKind.ARCHIVE), archive, tmp_path / "bad"
        )


def test_archive_installer_rejects_pax_metadata_and_executable_layout_mismatch(tmp_path: Path):
    """Archives cannot smuggle extended metadata or omit declared executables."""
    archive = tmp_path / "artifact.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as output:
        member = tarfile.TarInfo("bin/tool")
        member.size = 1
        member.pax_headers = {"SCHILY.xattr.user.test": "value"}
        output.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(InstallError, match="special entry"):
        ArchiveInstaller().install(
            _artifact(archive.read_bytes(), AcquisitionKind.ARCHIVE), archive, tmp_path / "bad"
        )
    with pytest.raises(ValueError, match="declared files"):
        InstallLayout(
            files=("bin/tool",),
            executables=("bin/missing",),
            identities={"bin/tool": hashlib.sha256(b"x").hexdigest()},
        )


def test_leases_protect_live_sets_and_discard_expired_records(tmp_path: Path):
    """Only valid nonexpired leases protect an artifact set from collection."""
    artifact_set = "a" * 64
    lease = create_lease(tmp_path, "sha256:manifest", artifact_set, 60)
    assert active_leases(tmp_path) == frozenset({artifact_set})
    with pytest.raises(ValueError, match="future"):
        lease.heartbeat(1.0)
    assert active_leases(tmp_path, now=10**12) == frozenset()
    assert not lease.path.exists()
    with pytest.raises(ValueError, match="positive"):
        create_lease(tmp_path, "manifest", "set", 0)


def test_collection_keeps_content_referenced_by_active_artifact_set(tmp_path: Path):
    """Active set metadata protects its member content rather than the set identifier."""
    transport = MemoryTransport(b"tool")
    bootstrapper = RuntimeBootstrapper(
        _manifest(b"tool"), tmp_path, transport, {AcquisitionKind.FILE: FileInstaller()}, 1024
    )
    installed = bootstrapper.ensure(
        RuntimeRequirement(PlatformSelector(os="linux", architecture="amd64"), frozenset({"core"}))
    )
    assert not collect(tmp_path)
    bootstrapper.activate(installed)
    installed.lease.close()
    assert not collect(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (("files", ("../escape",)), ("executables", ("/absolute",)), ("files", ("x", "x"))),
)
def test_install_layout_rejects_nonportable_paths(field: str, value: tuple[str, ...]):
    """Install layouts never accept an escaping or duplicated path."""
    with pytest.raises(ValueError):
        InstallLayout(
            **{
                field: value,
                **({"files": ("x",)} if field == "executables" else {}),
                "identities": {"x": hashlib.sha256(b"x").hexdigest()},
            }
        )


def test_missing_capability_and_provider_are_explicit(tmp_path: Path):
    """Unavailable capabilities and providers fail without a host fallback."""
    bootstrapper = RuntimeBootstrapper(
        _manifest(b"tool"), tmp_path, MemoryTransport(b"tool"), {}, 1024
    )
    requirement = RuntimeRequirement(
        PlatformSelector(os="macos", architecture="arm64"), frozenset({"core"})
    )
    with pytest.raises(ManifestError, match="platform/capabilities"):
        bootstrapper.ensure(requirement)
    with pytest.raises(Exception, match="No installer"):
        bootstrapper.ensure(
            RuntimeRequirement(
                PlatformSelector(os="linux", architecture="amd64"), frozenset({"core"})
            )
        )


def test_manifest_rejects_duplicate_cycle_sources_and_cross_platform_dependencies():
    """Closed manifests reject ambiguous identity, authority, cycles, and missing selections."""
    artifact = _artifact(b"x")
    with pytest.raises(ValueError, match="duplicate"):
        RuntimeManifest(schema_version=1, mode=ManifestMode.FIXTURE, artifacts=(artifact, artifact))
    cyclic = artifact.model_copy(
        update={"dependencies": (ArtifactDependency(artifact_id="runtime"),)}
    )
    with pytest.raises(ValueError, match="cycle"):
        RuntimeManifest(schema_version=1, mode=ManifestMode.FIXTURE, artifacts=(cyclic,))
    for source in ("http://example.test/a", "https://user@example.test/a"):
        with pytest.raises(ValueError, match="credential-free HTTPS"):
            RuntimeManifest(
                schema_version=1,
                mode=ManifestMode.FIXTURE,
                artifacts=(artifact.model_copy(update={"source": source}),),
            )
    dependency = artifact.model_copy(
        update={
            "artifact_id": "dependency",
            "platform": PlatformSelector(os="macos", architecture="arm64"),
        }
    )
    selected = artifact.model_copy(
        update={"dependencies": (ArtifactDependency(artifact_id="dependency"),)}
    )
    manifest = RuntimeManifest(
        schema_version=1, mode=ManifestMode.FIXTURE, artifacts=(dependency, selected)
    )
    with pytest.raises(ManifestError, match="unavailable"):
        manifest.select(selected.platform, frozenset({"core"}))

    reference_only = artifact.model_copy(update={"layout": None})
    reference_manifest = RuntimeManifest(
        schema_version=1, mode=ManifestMode.FIXTURE, artifacts=(reference_only,)
    )
    with pytest.raises(ManifestError, match="install layout"):
        reference_manifest.select(reference_only.platform, frozenset({"core"}))


def test_manifest_loaders_enforce_distinct_authorities(tmp_path: Path):
    """Developer and production loaders accept only their explicit trust modes."""
    fixture = _manifest(b"x")
    path = tmp_path / "manifest.json"
    path.write_text(fixture.model_dump_json(), encoding="utf-8")
    with pytest.raises(ManifestError, match="candidate"):
        load_checked_in_candidate(path)
    candidate = fixture.model_copy(update={"mode": ManifestMode.CANDIDATE})
    path.write_text(candidate.model_dump_json(), encoding="utf-8")
    assert load_checked_in_candidate(path).mode is ManifestMode.CANDIDATE
    release = fixture.model_copy(update={"mode": ManifestMode.RELEASE})
    assert load_embedded_release(release.model_dump_json().encode()).mode is ManifestMode.RELEASE


def test_fixture_candidate_and_embedded_release_share_one_manifest_schema() -> None:
    """All three authorities validate directly and promotion changes only trust mode."""
    repository = Path(__file__).parents[2]
    fixture = RuntimeManifest.model_validate_json(
        (Path(__file__).parent / "fixtures/runtime-manifest.json").read_bytes()
    )
    candidate = load_checked_in_candidate(
        repository / "scripts/runtime-candidates/macos-arm64-v1.json"
    )
    release = load_embedded_release()

    assert fixture.mode is ManifestMode.FIXTURE
    assert candidate.mode is ManifestMode.CANDIDATE
    assert release.mode is ManifestMode.RELEASE
    assert release.artifacts == candidate.artifacts


def test_download_enforces_empty_multiple_truncated_hash_and_quota(tmp_path: Path):
    """Streaming acquisition rejects every incomplete or over-authority response shape."""
    artifact = _artifact(b"tool")

    class Responses:
        """Yield configured transport responses."""

        def __init__(self, responses):
            self.responses = responses

        def stream(self, _url):
            """Yield configured responses."""
            yield from self.responses

    with pytest.raises(AcquisitionError, match="no response"):
        download_artifact(artifact, tmp_path, Responses(()), 100)
    response = (artifact.source, iter((b"tool",)))
    with pytest.raises(AcquisitionError, match="multiple"):
        download_artifact(artifact, tmp_path, Responses((response, response)), 100)
    with pytest.raises(IntegrityFailure):
        download_artifact(artifact, tmp_path, MemoryTransport(b"to"), 100)
    with pytest.raises(IntegrityFailure):
        download_artifact(
            artifact.model_copy(update={"digest": "0" * 64}),
            tmp_path,
            MemoryTransport(b"tool"),
            100,
        )
    with pytest.raises(QuotaExceeded):
        download_artifact(artifact, tmp_path, MemoryTransport(b"tool"), 1)
    with pytest.raises(AcquisitionError, match="OCI"):
        download_artifact(
            artifact.model_copy(
                update={
                    "size": None,
                    "layout": None,
                    "acquisition": AcquisitionKind.OCI,
                    "digest": "sha256:" + "0" * 64,
                    "source": "repo.test/image@sha256:" + "0" * 64,
                    "oci_identity": OciImageIdentity(
                        index_digest="sha256:" + "0" * 64,
                        manifest_digest="sha256:" + "1" * 64,
                        config_digest="sha256:" + "2" * 64,
                    ),
                }
            ),
            tmp_path,
            MemoryTransport(b""),
            1,
        )


def test_activation_retains_and_swaps_verified_rollback(tmp_path: Path):
    """Activation preserves the prior verified set and rollback swaps without acquisition."""
    first, second = "a" * 64, "b" * 64
    sets = tmp_path / "sets"
    sets.mkdir()
    (sets / f"{first}.json").write_text("{}")
    (sets / f"{second}.json").write_text("{}")
    activate(tmp_path, first)
    activate(tmp_path, second, first)
    assert rollback_set(tmp_path) == first
    assert rollback(tmp_path) == first
    assert active_set(tmp_path) == first
    assert rollback_set(tmp_path) == second
    with pytest.raises(ActivationError, match="invalid"):
        activate(tmp_path, "bad", first)
    (tmp_path / "rollback").write_text("bad")
    with pytest.raises(ActivationError, match="invalid"):
        rollback_set(tmp_path)


def test_bootstrap_status_activation_and_cache_quota_stay_fail_closed(tmp_path: Path):
    """Bootstrap status, activation, rollback, and quota failures stay fail closed."""
    bootstrapper = RuntimeBootstrapper(
        _manifest(b"tool"),
        tmp_path,
        MemoryTransport(b"tool"),
        {AcquisitionKind.FILE: FileInstaller()},
        1024,
    )
    requirement = RuntimeRequirement(
        PlatformSelector(os="linux", architecture="amd64"), frozenset({"core"})
    )
    assert bootstrapper.root == tmp_path
    assert bootstrapper.status()[0]["state"] == "missing"
    installed = bootstrapper.ensure(requirement)
    assert bootstrapper.status()[0]["state"] == "installed"
    bootstrapper.activate(installed)
    assert active_set(tmp_path) is not None
    with pytest.raises(ActivationError, match="rollback"):
        bootstrapper.rollback()
    with pytest.raises(ValueError, match="quotas"):
        RuntimeBootstrapper(_manifest(b"tool"), tmp_path, MemoryTransport(b"tool"), {}, 0)
    limited = RuntimeBootstrapper(
        _manifest(b"tool"),
        tmp_path / "limited",
        MemoryTransport(b"tool"),
        {AcquisitionKind.FILE: FileInstaller()},
        1024,
        cache_limit=1,
    )
    with pytest.raises(QuotaExceeded, match="cache"):
        limited.ensure(requirement)


@pytest.mark.parametrize(
    ("name", "configure", "message"),
    (
        ("too-many", lambda member: None, "file-count"),
        ("privileged", lambda member: setattr(member, "mode", 0o4755), "privileged"),
        ("too-large", lambda member: None, "unpacked-byte"),
        ("absolute", lambda member: setattr(member, "name", "/bin/tool"), "absolute"),
    ),
)
def test_archive_limits_reject_hostile_metadata(tmp_path: Path, name, configure, message):
    """Archive limits reject count, expansion, privilege, and absolute-path attacks."""
    archive = tmp_path / f"{name}.tar"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("bin/tool")
        member.size = 1
        configure(member)
        output.addfile(member, io.BytesIO(b"x"))
    installer = ArchiveInstaller(
        maximum_files=0 if name == "too-many" else 10,
        maximum_bytes=0 if name == "too-large" else 10,
    )
    with pytest.raises((UnsafeArchive, InstallError), match=message):
        installer.install(
            _artifact(archive.read_bytes(), AcquisitionKind.ARCHIVE),
            archive,
            tmp_path / f"out-{name}",
        )


def test_archive_rejects_case_collisions_links_and_broken_inputs(tmp_path: Path):
    """Portable-name collisions, links, invalid tar data, and missing sources fail closed."""
    archive = tmp_path / "hostile.tar"
    with tarfile.open(archive, "w") as output:
        for name in ("Bin/Tool", "bin/tool"):
            member = tarfile.TarInfo(name)
            member.size = 1
            output.addfile(member, io.BytesIO(b"x"))
    artifact = _artifact(archive.read_bytes(), AcquisitionKind.ARCHIVE)
    with pytest.raises(UnsafeArchive, match="duplicate"):
        ArchiveInstaller().install(artifact, archive, tmp_path / "collision")
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("bin/tool")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../escape"
        output.addfile(member)
    with pytest.raises(UnsafeArchive, match="link"):
        ArchiveInstaller().install(artifact, archive, tmp_path / "link")
    archive.write_bytes(b"not a tar")
    with pytest.raises(InstallError, match="extraction"):
        ArchiveInstaller().install(artifact, archive, tmp_path / "invalid")
    with pytest.raises(InstallError, match="requires"):
        ArchiveInstaller().install(artifact, None, tmp_path / "missing")


def test_archive_accepts_only_manifest_declared_contained_symlinks(tmp_path: Path) -> None:
    """A digest-pinned archive may install one exact contained symlink and unlisted files."""
    archive = tmp_path / "linked.tar"
    with tarfile.open(archive, "w") as output:
        root = tarfile.TarInfo("./")
        root.type = tarfile.DIRTYPE
        output.addfile(root)
        member = tarfile.TarInfo("bin/tool")
        member.size = 1
        output.addfile(member, io.BytesIO(b"x"))
        extra = tarfile.TarInfo("share/templates/default.yaml")
        extra.size = 1
        output.addfile(extra, io.BytesIO(b"y"))
        link = tarfile.TarInfo("share/doc/templates")
        link.type = tarfile.SYMTYPE
        link.linkname = "../templates"
        output.addfile(link)
    artifact = _artifact(archive.read_bytes(), AcquisitionKind.ARCHIVE)
    artifact = artifact.model_copy(
        update={
            "layout": artifact.layout.model_copy(
                update={
                    "symlinks": {"share/doc/templates": "../templates"},
                    "allow_unlisted_files": True,
                }
            )
        }
    )
    installed = install_content(artifact, archive, tmp_path / "artifacts", ArchiveInstaller())
    try:
        assert verify_content(installed, artifact)
        assert installed.joinpath("share/doc/templates").readlink() == Path("../templates")
        with pytest.raises(ValueError, match="escapes"):
            artifact.layout.model_copy(
                update={"symlinks": {"link": "../../outside"}}
            ).validate_executables_are_files()
    finally:
        for directory, names, _ in os.walk(installed, topdown=True, followlinks=False):
            Path(directory).chmod(0o700)
            for name in names:
                path = Path(directory, name)
                if not path.is_symlink() and path.is_dir():
                    path.chmod(0o700)


def test_installation_revalidates_content_and_cleans_failed_staging(tmp_path: Path):
    """Atomic installation reuses valid content and removes incomplete staging on failure."""
    artifact = _artifact(b"tool")
    source = tmp_path / "source"
    source.write_bytes(b"tool")
    root = tmp_path / "artifacts"
    installed = install_content(artifact, source, root, FileInstaller())
    assert install_content(artifact, source, root, FileInstaller()) == installed
    assert verify_content(installed, artifact)
    os.chmod(installed / "bin/tool", 0o700)
    assert not verify_content(installed, artifact)
    installed.joinpath("bin/tool").write_bytes(b"changed")
    assert not verify_content(installed, artifact)
    with pytest.raises(InstallError, match="failed verification"):
        install_content(artifact, source, root, FileInstaller())

    class BrokenInstaller:
        """Install unexpected content then fail verification."""

        def install(self, _artifact, _source, destination):
            """Create undeclared content."""
            (destination / "other").write_text("x")
            return destination

    other = artifact.model_copy(update={"artifact_id": "other"})
    with pytest.raises(InstallError, match="declared layout"):
        install_content(other, source, root, BrokenInstaller())
    assert not tuple((root / "other").glob(".staging-*"))
    with pytest.raises(InstallError, match="exactly one"):
        FileInstaller().install(artifact, None, tmp_path / "none")

    class NonExecutableInstaller:
        """Install correct bytes without the required executable mode."""

        def install(self, _artifact, _source, destination):
            target = destination / "bin/tool"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"tool")
            os.chmod(target, 0o600)
            return destination

    nonexecutable = artifact.model_copy(update={"artifact_id": "nonexecutable"})
    with pytest.raises(InstallError, match="declared layout"):
        install_content(nonexecutable, source, root, NonExecutableInstaller())

    outside = tmp_path / "outside-link-target"
    outside.mkdir()

    class SymlinkInstaller:
        """Add an undeclared link beside otherwise valid content."""

        def install(self, _artifact, _source, destination):
            target = destination / "bin/tool"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"tool")
            os.chmod(target, 0o700)
            destination.joinpath("link").symlink_to(outside)
            return destination

    linked = artifact.model_copy(update={"artifact_id": "linked"})
    with pytest.raises(InstallError, match="declared layout"):
        install_content(linked, source, root, SymlinkInstaller())


def test_gc_ignores_links_files_young_staging_and_invalid_set_records(tmp_path: Path):
    """Collection removes only old resolved private trees and never follows hostile entries."""
    artifacts = tmp_path / "artifacts"
    artifact = artifacts / "runtime"
    artifact.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (artifacts / "file").write_text("x")
    (artifacts / "link").symlink_to(outside, target_is_directory=True)
    (artifact / "link").symlink_to(outside, target_is_directory=True)
    (artifact / "file").write_text("x")
    young = artifact / ".staging-young"
    young.mkdir()
    old = artifact / ".staging-old"
    old.mkdir()
    old.touch()
    os.utime(young, (old.stat().st_mtime + 4000, old.stat().st_mtime + 4000))
    digest = "c" * 64
    removable = artifact / digest
    removable.mkdir()
    unowned = artifact / "unowned"
    unowned.mkdir()
    invalid_set = "d" * 64
    (tmp_path / "sets").mkdir()
    (tmp_path / "sets" / f"{invalid_set}.json").write_text("invalid")
    removed = collect(tmp_path, rollback=invalid_set, now=old.stat().st_mtime + 4000)
    assert old in removed and removable in removed
    assert young.exists() and unowned.exists() and outside.exists()
    assert collect(tmp_path, rollback="bad") == ()
    for payload in ('{"artifact_digests":"bad"}', '{"artifact_digests":["bad"]}'):
        (tmp_path / "sets" / f"{'e' * 64}.json").write_text(payload)
        assert collect(tmp_path, rollback="e" * 64) == ()


def test_models_reject_incomplete_identities_and_invalid_acquisition_shapes():
    """Closed artifact models reject missing identities and mismatched OCI/file fields."""
    with pytest.raises(ValueError, match="identity"):
        InstallLayout(files=("x",), identities={})
    artifact = _artifact(b"x")
    with pytest.raises(ValueError, match="File artifacts"):
        Artifact.model_validate({**artifact.model_dump(), "size": None})
    with pytest.raises(ValueError, match="sha256 digest"):
        Artifact.model_validate(
            {
                **artifact.model_dump(),
                "acquisition": "oci",
                "size": None,
                "layout": None,
                "digest": "0" * 64,
                "source": "repo.test/image@sha256:" + "0" * 64,
            }
        )
    with pytest.raises(ValueError, match="metadata names"):
        Artifact(
            **{
                **artifact.model_dump(),
                "metadata": (
                    ArtifactMetadata(name="component.tool", value="1"),
                    ArtifactMetadata(name="component.tool", value="2"),
                ),
            }
        )
    with pytest.raises(ValueError, match="complete OCI identity"):
        Artifact.model_validate(
            {
                **artifact.model_dump(),
                "acquisition": "oci",
                "digest": "sha256:" + "0" * 64,
                "source": "repo.test/image@sha256:" + "0" * 64,
            }
        )


def test_httpx_transport_enforces_redirect_origin_and_maps_network_failure(monkeypatch):
    """Production transport disables ambient proxies and rejects unsafe redirects."""

    class Response:
        """Provide the HTTPX response surface consumed by the transport."""

        def __init__(self, url, history=(), error=None):
            self.url = url
            self.history = history
            self.error = error

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            if self.error is not None:
                raise self.error

        def iter_bytes(self):
            return iter((b"x",))

    class Client:
        """Capture reviewed HTTP client options and return one response."""

        response = Response("https://example.test/final")

        def __init__(self, **options):
            assert options == {"follow_redirects": True, "max_redirects": 3, "trust_env": False}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def stream(self, method, url):
            assert (method, url) == ("GET", "https://example.test/artifact")
            return self.response

    monkeypatch.setattr("loop.execution.runtime.download.httpx.Client", Client)
    transport = HttpxArtifactTransport()
    assert next(transport.stream("https://example.test/artifact"))[0].endswith("/final")
    Client.response = Response(
        "https://example.test/final", history=(Response("http://example.test/redirect"),)
    )
    with pytest.raises(AcquisitionError, match="redirect"):
        next(transport.stream("https://example.test/artifact"))
    Client.response = Response(
        "https://example.test/final",
        error=httpx.ConnectError("offline", request=httpx.Request("GET", "https://example.test")),
    )
    with pytest.raises(ArtifactUnavailable, match="unavailable"):
        next(transport.stream("https://example.test/artifact"))


def test_facade_rejects_foreign_or_unrecorded_activation_and_reports_pointers(tmp_path: Path):
    """Only locally recorded manifest sets can activate, and diagnostics expose stable state."""
    bootstrapper = RuntimeBootstrapper(
        _manifest(b"tool"),
        tmp_path,
        MemoryTransport(b"tool"),
        {AcquisitionKind.FILE: FileInstaller()},
        1024,
    )
    requirement = RuntimeRequirement(
        PlatformSelector(os="linux", architecture="amd64"), frozenset({"core"})
    )
    installed = bootstrapper.ensure(requirement)
    foreign = installed.__class__(
        "sha256:foreign",
        installed.artifact_set_digest,
        {},
        installed.lease,
        installed.declared_artifacts,
        installed.root,
    )
    with pytest.raises(ValueError, match="different manifest"):
        bootstrapper.activate(foreign)
    missing = installed.__class__(
        installed.manifest_digest,
        "f" * 64,
        {},
        installed.lease,
        installed.declared_artifacts,
        installed.root,
    )
    with pytest.raises(ValueError, match="no verified"):
        bootstrapper.activate(missing)
    assert bootstrapper.activate(installed) is None
    assert bootstrapper.activate(installed) == installed.artifact_set_digest
    second = "e" * 64
    (tmp_path / "sets" / f"{second}.json").write_text("{}")
    activate(tmp_path, second, installed.artifact_set_digest)
    rows = bootstrapper.status()
    assert {row["artifact"] for row in rows} >= {"active-set", "rollback-set"}
    assert bootstrapper.rollback() == installed.artifact_set_digest


def test_invalid_active_pointer_and_unavailable_rollback_fail_closed(tmp_path: Path):
    """Malformed activation state and missing retained records never select content."""
    (tmp_path / "active").write_text("bad")
    with pytest.raises(ActivationError, match="invalid"):
        active_set(tmp_path)
    (tmp_path / "active").unlink()
    (tmp_path / "rollback").write_text("a" * 64)
    with pytest.raises(ActivationError, match="No verified"):
        rollback(tmp_path)


def test_invalid_lease_records_are_ignored_and_expired_valid_records_removed(tmp_path: Path):
    """Lease restart recovery ignores corruption and removes only typed expired records."""
    leases = tmp_path / "leases"
    leases.mkdir()
    (leases / "broken.json").write_text("not json")
    (leases / "bad-digest.json").write_text('{"artifact_set_digest":"bad","expiry":9999999999999}')
    (leases / "bad-expiry.json").write_text(
        '{"artifact_set_digest":"' + "a" * 64 + '","expiry":"later"}'
    )
    expired = leases / "expired.json"
    expired.write_text('{"artifact_set_digest":"' + "a" * 64 + '","expiry":1}')
    assert active_leases(tmp_path, now=2) == frozenset()
    assert not expired.exists()


def test_download_cancels_midstream_and_rejects_oversized_chunk(tmp_path: Path):
    """Cancellation and a single oversized chunk remove temporary acquisition state."""
    artifact = _artifact(b"tool")
    with pytest.raises(AcquisitionCancelled):
        download_artifact(artifact, tmp_path, MemoryTransport(b"tool"), 100, lambda: True)
    with pytest.raises(QuotaExceeded, match="declared"):
        download_artifact(artifact, tmp_path, MemoryTransport(b"tools"), 100)
    assert not tuple(tmp_path.glob("artifact-*"))


def test_manifest_accepts_valid_oci_and_orders_dependency_before_root():
    """Immutable OCI references validate and selected dependencies precede their consumer."""
    file_artifact = _artifact(b"x").model_copy(
        update={"artifact_id": "dependency", "capabilities": ()}
    )
    root = _artifact(b"y").model_copy(
        update={"dependencies": (ArtifactDependency(artifact_id="dependency"),)}
    )
    manifest = RuntimeManifest(
        schema_version=1, mode=ManifestMode.FIXTURE, artifacts=(root, file_artifact)
    )
    assert [item.artifact_id for item in manifest.select(root.platform, frozenset({"core"}))] == [
        "dependency",
        "runtime",
    ]
    oci = Artifact(
        artifact_id="image",
        version="1",
        role=ArtifactRole.OCI_PROFILE,
        platform=root.platform,
        source="registry.test/image@sha256:" + "a" * 64,
        digest="sha256:" + "a" * 64,
        acquisition=AcquisitionKind.OCI,
        media_type="application/vnd.oci.image.manifest.v1+json",
        oci_identity=OciImageIdentity(
            index_digest="sha256:" + "a" * 64,
            manifest_digest="sha256:" + "b" * 64,
            config_digest="sha256:" + "c" * 64,
        ),
        capabilities=("image",),
        sbom="https://example.test/sbom",
        notices="https://example.test/notices",
    )
    assert RuntimeManifest(schema_version=1, mode=ManifestMode.FIXTURE, artifacts=(oci,)).select(
        oci.platform, frozenset({"image"})
    ) == (oci,)
    with pytest.raises(ValueError, match="complete OCI identity"):
        RuntimeManifest(
            schema_version=1,
            mode=ManifestMode.FIXTURE,
            artifacts=(oci.model_copy(update={"source": "registry.test/image:latest"}),),
        )


def test_archive_extracts_directories_and_rejects_symlinked_destination(tmp_path: Path):
    """Directory entries work while a replaced extraction parent fails closed."""
    archive = tmp_path / "directory.tar"
    with tarfile.open(archive, "w") as output:
        directory = tarfile.TarInfo("bin")
        directory.type = tarfile.DIRTYPE
        output.addfile(directory)
        member = tarfile.TarInfo("bin/tool")
        member.size = 1
        output.addfile(member, io.BytesIO(b"x"))
    destination = tmp_path / "destination"
    ArchiveInstaller().install(
        _artifact(archive.read_bytes(), AcquisitionKind.ARCHIVE), archive, destination
    )
    assert destination.joinpath("bin/tool").read_bytes() == b"x"
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(InstallError, match="symbolic link"):
        ArchiveInstaller().install(
            _artifact(archive.read_bytes(), AcquisitionKind.ARCHIVE), archive, linked
        )


def test_install_rejects_symlinked_artifact_root_and_preserves_replaced_staging(tmp_path: Path):
    """Install publication cannot follow artifact-root or staging replacement links."""
    artifact = _artifact(b"tool")
    source = tmp_path / "source"
    source.write_bytes(b"tool")
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / artifact.artifact_id).symlink_to(outside, target_is_directory=True)
    with pytest.raises(InstallError, match="private directory"):
        install_content(artifact, source, root, FileInstaller())

    class ReplacingInstaller:
        """Replace staging with a link to simulate a same-user race fault."""

        def install(self, _artifact, _source, destination):
            """Replace and fail after the replacement."""
            destination.rmdir()
            destination.symlink_to(outside, target_is_directory=True)
            raise RuntimeError("fault")

    clean_root = tmp_path / "clean"
    with pytest.raises(RuntimeError, match="fault"):
        install_content(artifact, source, clean_root, ReplacingInstaller())
    assert outside.exists()


@pytest.mark.parametrize("mode", ("missing", "oversized", "truncated"))
def test_archive_stream_faults_fail_closed(monkeypatch, tmp_path: Path, mode: str):
    """Extractor faults cannot publish missing, oversized, or truncated member content."""
    member = tarfile.TarInfo("bin/tool")
    member.size = 1

    class Extracted:
        """Return a fault-shaped member stream."""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            if mode == "oversized":
                mode_value = b"xx"
            else:
                mode_value = b""
            if hasattr(self, "done"):
                return b""
            self.done = True
            return mode_value

    class Archive:
        """Provide one validated member with a controlled extraction result."""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def getmembers(self):
            return [member]

        def extractfile(self, _member):
            return None if mode == "missing" else Extracted()

    monkeypatch.setattr("loop.execution.runtime.install.tarfile.open", lambda *_args: Archive())
    source = tmp_path / "archive"
    source.write_bytes(b"archive")
    with pytest.raises(InstallError):
        ArchiveInstaller().install(
            _artifact(b"archive", AcquisitionKind.ARCHIVE), source, tmp_path / mode
        )


def test_oci_installer_dispatches_without_file_download(tmp_path: Path):
    """OCI acquisition is dispatched publicly without entering the file downloader."""
    oci = Artifact(
        artifact_id="image",
        version="1",
        role=ArtifactRole.OCI_PROFILE,
        platform=PlatformSelector(os="linux", architecture="amd64"),
        source="registry.test/image@sha256:" + "a" * 64,
        digest="sha256:" + "a" * 64,
        acquisition=AcquisitionKind.OCI,
        media_type="application/vnd.oci.image.manifest.v1+json",
        oci_identity=OciImageIdentity(
            index_digest="sha256:" + "a" * 64,
            manifest_digest="sha256:" + "b" * 64,
            config_digest="sha256:" + "c" * 64,
        ),
        capabilities=("image",),
        sbom="https://example.test/sbom",
        notices="https://example.test/notices",
    )

    class OciInstaller:
        """Materialize a fixture OCI content directory."""

        def install(self, _artifact, source, destination):
            assert source is None
            destination.mkdir(exist_ok=True)
            return destination

    manifest = RuntimeManifest(schema_version=1, mode=ManifestMode.FIXTURE, artifacts=(oci,))
    bootstrapper = RuntimeBootstrapper(
        manifest,
        tmp_path,
        MemoryTransport(b"unused"),
        {AcquisitionKind.OCI: OciInstaller()},
        10,
    )
    installed = bootstrapper.ensure(RuntimeRequirement(oci.platform, frozenset({"image"})))
    assert installed.artifacts["image"].is_dir()


def test_cache_collection_retries_capacity_after_evicting_inactive_content(tmp_path: Path):
    """Cache reservation deterministically evicts inactive content before rejecting capacity."""
    stale = tmp_path / "artifacts" / "old" / ("d" * 64)
    stale.mkdir(parents=True)
    stale.joinpath("data").write_bytes(b"old")
    bootstrapper = RuntimeBootstrapper(
        _manifest(b"tool"),
        tmp_path,
        MemoryTransport(b"tool"),
        {AcquisitionKind.FILE: FileInstaller()},
        10,
        cache_limit=5,
    )
    bootstrapper.ensure(
        RuntimeRequirement(PlatformSelector(os="linux", architecture="amd64"), frozenset({"core"}))
    )
    assert not stale.exists()


def test_gc_reclaims_oldest_only_and_reset_removes_owned_downloads(tmp_path: Path):
    """Quota eviction stops at its byte target and reset removes only owned temporary files."""
    artifact = tmp_path / "artifacts" / "runtime"
    first = artifact / ("a" * 64)
    second = artifact / ("b" * 64)
    first.mkdir(parents=True)
    second.mkdir()
    first.joinpath("nested").mkdir()
    first.joinpath("nested/data").write_bytes(b"first")
    second.joinpath("data").write_bytes(b"second")
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))
    removed = collect(tmp_path, reclaim_bytes=1)
    assert removed == (first,)
    assert second.exists()
    with pytest.raises(ValueError, match="must not be negative"):
        collect(tmp_path, reclaim_bytes=-1)
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    temporary = downloads / "artifact-interrupted"
    unrelated = downloads / "unrelated"
    temporary.write_bytes(b"partial")
    unrelated.write_bytes(b"keep")
    removed = reset_inactive(tmp_path)
    assert second in removed and temporary in removed
    assert unrelated.exists()
    assert reset_inactive(tmp_path / "empty") == ()


def test_concurrent_installers_reuse_one_verified_content_directory(tmp_path: Path):
    """Concurrent bootstrap processes converge on one immutable verified installation."""
    transport = MemoryTransport(b"tool")
    bootstrapper = RuntimeBootstrapper(
        _manifest(b"tool"),
        tmp_path,
        transport,
        {AcquisitionKind.FILE: FileInstaller()},
        10,
    )
    requirement = RuntimeRequirement(
        PlatformSelector(os="linux", architecture="amd64"), frozenset({"core"})
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        installed = tuple(executor.map(lambda _: bootstrapper.ensure(requirement), range(2)))
    assert installed[0].artifacts["runtime"] == installed[1].artifacts["runtime"]
    assert verify_content(installed[0].artifacts["runtime"], _artifact(b"tool"))


def test_interrupted_publication_cleans_staging_and_retry_succeeds(monkeypatch, tmp_path: Path):
    """A failed atomic rename leaves no selectable staging tree and a retry succeeds."""
    artifact = _artifact(b"tool")
    source = tmp_path / "source"
    source.write_bytes(b"tool")
    root = tmp_path / "artifacts"
    real_replace = os.replace
    failed = False

    def interrupt(source_path, destination_path):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("interrupted rename")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr("loop.execution.runtime.install.os.replace", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        install_content(artifact, source, root, FileInstaller())
    assert not tuple((root / artifact.artifact_id).glob(".staging-*"))
    assert install_content(artifact, source, root, FileInstaller()).is_dir()


def test_concurrent_activation_has_one_compare_and_swap_winner(tmp_path: Path):
    """Concurrent activation permits one winner and rejects the stale transition."""
    initial, first, second = "a" * 64, "b" * 64, "c" * 64
    activate(tmp_path, initial)

    def transition(target):
        try:
            activate(tmp_path, target, initial)
            return "activated"
        except ActivationError:
            return "stale"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(transition, (first, second)))
    assert sorted(outcomes) == ["activated", "stale"]
    assert active_set(tmp_path) in {first, second}


def test_interrupted_activation_preserves_old_active_set_and_retry_succeeds(
    monkeypatch, tmp_path: Path
):
    """Activation publication failure keeps the prior set active and cleans temporary pointers."""
    initial, replacement = "a" * 64, "b" * 64
    activate(tmp_path, initial)
    real_replace = os.replace
    failed = False

    def interrupt(source, destination):
        nonlocal failed
        if Path(destination).name == "active" and not failed:
            failed = True
            raise OSError("interrupted activation")
        return real_replace(source, destination)

    monkeypatch.setattr("loop.execution.runtime.activation.os.replace", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        activate(tmp_path, replacement, initial)
    assert active_set(tmp_path) == initial
    assert not tuple(tmp_path.glob(".*.new"))
    activate(tmp_path, replacement, initial)
    assert active_set(tmp_path) == replacement


def test_runtime_root_replacement_and_initial_symlink_fail_closed(tmp_path: Path):
    """Runtime operations reject initial and post-construction root identity substitution."""
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        RuntimeBootstrapper(_manifest(b"tool"), linked, MemoryTransport(b"tool"), {}, 10)
    root = tmp_path / "runtime"
    bootstrapper = RuntimeBootstrapper(_manifest(b"tool"), root, MemoryTransport(b"tool"), {}, 10)
    moved = tmp_path / "moved"
    root.rename(moved)
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="identity changed"):
        bootstrapper.status()
    assert not tuple(outside.iterdir())
    missing_root = tmp_path / "missing-runtime"
    missing = RuntimeBootstrapper(
        _manifest(b"tool"), missing_root, MemoryTransport(b"tool"), {}, 10
    )
    missing_root.rename(tmp_path / "missing-runtime-moved")
    with pytest.raises(RuntimeError, match="identity changed"):
        missing.status()


def test_new_manifest_layout_and_executable_edges_retain_full_coverage(tmp_path: Path) -> None:
    """Redirects, symlink collisions/escapes, OCI refs, and bad executables fail closed."""
    with pytest.raises(ValueError, match="symlinks"):
        InstallLayout(files=("link",), identities={"link": "a" * 64}, symlinks={"link": "target"})
    with pytest.raises(ValueError, match="escapes"):
        InstallLayout(
            files=("file",), identities={"file": "a" * 64}, symlinks={"link": "../outside"}
        )
    artifact = _artifact(b"tool")
    with pytest.raises(ValueError, match="redirect origins"):
        artifact.model_copy(
            update={"allowed_redirect_origins": ("http://bad.test",)}
        ).__class__.model_validate(
            artifact.model_copy(
                update={"allowed_redirect_origins": ("http://bad.test",)}
            ).model_dump()
        )

    root = tmp_path / "runtime"
    artifact_root = root / "artifacts" / "runtime" / ("a" * 64)
    artifact_root.mkdir(parents=True)
    target = artifact_root / "bin" / "tool"
    target.mkdir(parents=True)
    lease = create_lease(root, "manifest", "set", 60)
    runtime = InstalledRuntime(
        "manifest", "set", {"runtime": artifact_root}, lease, {"runtime": artifact}, root
    )
    with pytest.raises(ValueError, match="regular file"):
        runtime.executable("runtime", "bin/tool")
    target.rmdir()
    target.write_bytes(b"tool")
    target.chmod(0o600)
    with pytest.raises(ValueError, match="not executable"):
        runtime.executable("runtime", "bin/tool")
    for artifact_id, relative_path in (
        ("missing", "bin/tool"),
        ("runtime", ""),
        ("runtime", "../tool"),
        ("runtime", "/bin/tool"),
        ("runtime", "bin\\tool"),
        ("runtime", "bin/missing"),
    ):
        with pytest.raises(ValueError, match="not part"):
            runtime.executable(artifact_id, relative_path)
