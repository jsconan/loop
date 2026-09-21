"""Test digest-pinned OCI acquisition through the classified runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.execution.infrastructure import InfrastructureProcessResult
from loop.execution.runtime.install import InstallError
from loop.execution.runtime.models import (
    AcquisitionKind,
    Artifact,
    ArtifactRole,
    OciImageIdentity,
    PlatformSelector,
)
from loop.execution.sandbox.oci import OciArtifactInstaller

_DIGEST = "sha256:" + "a" * 64
_MANIFEST_DIGEST = "sha256:" + "b" * 64
_CONFIG_DIGEST = "sha256:" + "c" * 64
_CONTENT_STORE = "sha256:" + "d" * 64


def _inspection(
    *,
    index_digest: str = _DIGEST,
    manifest_digest: str = _MANIFEST_DIGEST,
    config_digest: str = _CONFIG_DIGEST,
    architecture: str = "amd64",
    variant: str | None = None,
) -> str:
    """Return nerdctl 2.2.0 native image-inspection JSON."""
    platform = {"architecture": architecture, "os": "linux"}
    if variant is not None:
        platform["variant"] = variant
    return json.dumps(
        {
            "Image": {
                "Name": f"registry.test/profile@{index_digest}",
                "Labels": {},
                "Target": {
                    "mediaType": "application/vnd.oci.image.index.v1+json",
                    "digest": index_digest,
                    "size": 123,
                },
                "CreatedAt": "2026-09-18T00:00:00Z",
                "UpdatedAt": "2026-09-18T00:00:00Z",
            },
            "IndexDesc": {
                "mediaType": "application/vnd.oci.image.index.v1+json",
                "digest": index_digest,
                "size": 123,
            },
            "Index": {
                "schemaVersion": 2,
                "mediaType": "application/vnd.oci.image.index.v1+json",
                "manifests": [
                    {
                        "mediaType": "application/vnd.oci.image.manifest.v1+json",
                        "digest": manifest_digest,
                        "size": 100,
                        "platform": platform,
                    }
                ],
            },
            "ManifestDesc": {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "digest": manifest_digest,
                "size": 100,
            },
            "Manifest": {
                "schemaVersion": 2,
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "config": {
                    "mediaType": "application/vnd.oci.image.config.v1+json",
                    "digest": config_digest,
                    "size": 80,
                },
                "layers": [],
            },
            "ImageConfigDesc": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": config_digest,
                "size": 80,
            },
            "ImageConfig": {"architecture": architecture, "os": "linux", "config": {}},
            "size": 0,
        }
    )


def _artifact(architecture: str = "amd64") -> Artifact:
    """Build one immutable OCI profile artifact."""
    return Artifact(
        artifact_id="profile",
        version="1",
        role=ArtifactRole.OCI_PROFILE,
        platform=PlatformSelector(os="linux", architecture=architecture),
        source=f"registry.test/profile@{_DIGEST}",
        digest=_DIGEST,
        acquisition=AcquisitionKind.OCI,
        media_type="application/vnd.oci.image.manifest.v1+json",
        oci_identity=OciImageIdentity(
            index_digest=_DIGEST,
            manifest_digest=_MANIFEST_DIGEST,
            config_digest=_CONFIG_DIGEST,
        ),
        sbom="https://example.test/sbom",
        notices="https://example.test/notices",
    )


def _installer(
    tmp_path: Path, inspect_output: str, architecture: str = "amd64"
) -> OciArtifactInstaller:
    """Build an installer backed by an isolated fake OCI transport."""
    del tmp_path
    endpoint = SimpleNamespace(
        namespace="loop-private",
        content_store_identity=_CONTENT_STORE,
        platform=PlatformSelector(os="linux", architecture=architecture),
    )

    class _Transport:
        """Return deterministic pull and inspection results without process creation."""

        def pull_image(self, reference: str) -> InfrastructureProcessResult:
            """Return one successful pull result."""
            del reference
            return InfrastructureProcessResult(0, b"", b"", False, False)

        def inspect_image(self, reference: str) -> InfrastructureProcessResult:
            """Return the configured inspection result."""
            del reference
            return InfrastructureProcessResult(0, inspect_output.encode(), b"", False, False)

    installer = OciArtifactInstaller(_Transport(), endpoint)
    return installer


def test_oci_installer_pulls_and_records_only_an_attested_digest(tmp_path: Path):
    """The management transport must attest the exact manifest digest before publication."""
    installer = _installer(
        tmp_path,
        _inspection(),
    )
    destination = installer.install(_artifact(), None, tmp_path / "content")
    assert destination.joinpath("oci-content.json").read_text(encoding="utf-8") == (
        '{"artifact_id":"profile","config_digest":"'
        + _CONFIG_DIGEST
        + '","content_store_identity":"'
        + _CONTENT_STORE
        + '","index_digest":"'
        + _DIGEST
        + '","manifest_digest":"'
        + _MANIFEST_DIGEST
        + '","namespace":"loop-private","platform":{"architecture":"amd64","os":"linux"}}'
    )


def test_oci_installer_accepts_only_the_pinned_arm64_v8_variant(tmp_path: Path) -> None:
    """Docker arm64 v8 is exact while wrong, extra, and malformed platform data fail closed."""
    installer = _installer(tmp_path, _inspection(architecture="arm64", variant="v8"), "arm64")
    installer.install(_artifact("arm64"), None, tmp_path / "arm64-content")
    for index, platform in enumerate(
        (
            "linux/arm64",
            {"architecture": "amd64", "os": "linux"},
            {"architecture": "arm64", "os": "linux", "variant": "v8", "extra": True},
        )
    ):
        payload = json.loads(_inspection(architecture="arm64", variant="v8"))
        payload["Index"]["manifests"][0]["platform"] = platform
        rejected = _installer(tmp_path, json.dumps(payload), "arm64")
        with pytest.raises(InstallError, match="exactly one"):
            rejected.install(_artifact("arm64"), None, tmp_path / f"rejected-{index}")


def test_oci_installer_rejects_reused_or_linked_staging(tmp_path: Path) -> None:
    """The provider cannot overwrite nonempty or linked artifact staging state."""
    destination = tmp_path / "content"
    destination.mkdir()
    destination.joinpath("foreign").write_text("x", encoding="utf-8")
    with pytest.raises(InstallError, match="empty private"):
        _installer(tmp_path, _inspection()).install(_artifact(), None, destination)
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(InstallError, match="empty private"):
        _installer(tmp_path, _inspection()).install(_artifact(), None, linked)


@pytest.mark.parametrize("output", ("{}", "not-json"))
def test_oci_installer_rejects_missing_or_malformed_inspection_evidence(
    tmp_path: Path, output: str
):
    """Schema drift or a digest mismatch leaves no OCI artifact selected."""
    installer = _installer(tmp_path, output)
    with pytest.raises(InstallError):
        installer.install(_artifact(), None, tmp_path / "content")
    assert not (tmp_path / "content").exists()


def test_oci_installer_rejects_file_and_non_oci_inputs(tmp_path: Path):
    """The OCI provider cannot be repurposed as a generic file installer."""
    installer = _installer(tmp_path, '"' + _DIGEST + '"')
    with pytest.raises(InstallError, match="without a file"):
        installer.install(_artifact(), tmp_path / "source", tmp_path / "content")


def test_oci_installer_wraps_runner_failure_exit_and_identity_mismatch(tmp_path: Path) -> None:
    """Runner faults, CLI rejection, and any native identity conflict fail closed."""
    installer = _installer(tmp_path, "{}")

    class _Runner:
        values: list[InfrastructureProcessResult | Exception]

        def __init__(self, values: list[InfrastructureProcessResult | Exception]) -> None:
            self.values = values

        def _next(self) -> InfrastructureProcessResult:
            """Return or raise the next configured transport outcome."""
            value = self.values.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        def pull_image(self, _reference: str) -> InfrastructureProcessResult:
            """Return the next configured pull outcome."""
            return self._next()

        def inspect_image(self, _reference: str) -> InfrastructureProcessResult:
            """Return the next configured inspection outcome."""
            return self._next()

    installer.transport = _Runner([RuntimeError("cancelled")])  # type: ignore[assignment]
    with pytest.raises(InstallError, match="could not be completed"):
        installer.install(_artifact(), None, tmp_path / "runtime-error")
    failed = InfrastructureProcessResult(1, b"", b"", False, False)
    installer.transport = _Runner([failed, failed])  # type: ignore[assignment]
    with pytest.raises(InstallError, match="rejected"):
        installer.install(_artifact(), None, tmp_path / "exit")
    ok = InfrastructureProcessResult(0, b"", b"", False, False)
    inspected = InfrastructureProcessResult(
        0, _inspection(manifest_digest="sha256:" + "f" * 64).encode(), b"", False, False
    )
    installer.transport = _Runner([inspected])  # type: ignore[assignment]
    with pytest.raises(InstallError, match="did not attest"):
        installer.install(_artifact(), None, tmp_path / "mismatch")

    progress: list[str] = []
    installer.progress = progress.append
    valid = InfrastructureProcessResult(0, _inspection().encode(), b"", False, False)
    missing = InfrastructureProcessResult(0, b"", b"", False, False)
    installer.transport = _Runner([missing, ok, valid])  # type: ignore[assignment]
    installer.install(_artifact(), None, tmp_path / "acquired")
    assert progress == ["Installing sandbox command image…"]


@pytest.mark.parametrize(
    "inspection",
    (
        _inspection(architecture="arm64"),
        _inspection(config_digest="sha256:" + "f" * 64),
        json.dumps({**json.loads(_inspection()), "unknown": True}),
    ),
)
def test_oci_installer_rejects_native_schema_platform_and_digest_drift(
    tmp_path: Path, inspection: str
) -> None:
    """Real native-schema drift and wrong platform or config identity fail closed."""
    installer = _installer(tmp_path, inspection)
    with pytest.raises(InstallError):
        installer.install(_artifact(), None, tmp_path / "content")


def _mutated_inspection(mutate: object) -> str:
    """Return native inspection evidence changed by one adversarial mutation."""
    value = json.loads(_inspection())
    mutate(value)  # type: ignore[operator]
    return json.dumps(value)


def _single_manifest_inspection() -> str:
    """Return native evidence whose target is a single selected manifest."""
    value = json.loads(_inspection())
    value.pop("Index")
    value.pop("IndexDesc")
    value["Image"]["Target"]["digest"] = _MANIFEST_DIGEST
    return json.dumps(value)


def _conflicting_single_manifest_inspection() -> str:
    """Return single-manifest evidence whose target and manifest disagree."""
    value = json.loads(_inspection())
    value.pop("Index")
    value.pop("IndexDesc")
    return json.dumps(value)


@pytest.mark.parametrize(
    "inspection",
    (
        _mutated_inspection(lambda value: value["Index"].update(schemaVersion=1)),
        _mutated_inspection(lambda value: value["Index"].update(manifests=[])),
        _conflicting_single_manifest_inspection(),
        _mutated_inspection(lambda value: value.update(ImageConfig=[])),
        _mutated_inspection(
            lambda value: value["ManifestDesc"].update(digest="sha256:" + "G" * 64)
        ),
    ),
)
def test_oci_installer_rejects_adversarial_native_descriptor_shapes(
    tmp_path: Path, inspection: str
) -> None:
    """Malformed indexes, nested values, and descriptor digests fail closed."""
    installer = _installer(tmp_path, inspection)
    with pytest.raises(InstallError):
        installer.install(_artifact(), None, tmp_path / "content")


def test_oci_installer_accepts_a_digest_pinned_single_manifest(tmp_path: Path) -> None:
    """Native single-manifest evidence is valid when all target identities agree."""
    artifact = _artifact().model_copy(
        update={
            "source": f"registry.test/profile@{_MANIFEST_DIGEST}",
            "digest": _MANIFEST_DIGEST,
            "oci_identity": OciImageIdentity(
                index_digest=_MANIFEST_DIGEST,
                manifest_digest=_MANIFEST_DIGEST,
                config_digest=_CONFIG_DIGEST,
            ),
        }
    )
    destination = _installer(tmp_path, _single_manifest_inspection()).install(
        artifact, None, tmp_path / "content"
    )
    assert destination.is_dir()
