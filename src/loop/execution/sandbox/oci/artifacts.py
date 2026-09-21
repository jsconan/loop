"""Install digest-pinned OCI content through the classified management runner."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ...runtime.install import InstallError
from ...runtime.models import AcquisitionKind, Artifact, OciImageIdentity, PlatformSelector
from .control import OciEndpoint, OciImageTransport


class OciArtifactInstaller:
    """Acquire only manifest-pinned OCI content into Loop's private runtime store.

    Args:
        transport (OciImageTransport): Platform-owned sealed OCI image transport.
        endpoint (OciEndpoint): Attested private containerd endpoint.
        progress (Callable[[str], None]): User-visible image acquisition status callback.
    """

    transport: OciImageTransport
    endpoint: OciEndpoint
    progress: Callable[[str], None]

    def __init__(
        self,
        transport: OciImageTransport,
        endpoint: OciEndpoint,
        *,
        progress: Callable[[str], None] = lambda _: None,
    ) -> None:
        self.transport = transport
        self.endpoint = endpoint
        self.progress = progress

    def install(self, artifact: Artifact, source: Path | None, destination: Path) -> Path:
        """Pull and verify the exact OCI digest without executing image content.

        Args:
            artifact (Artifact): Manifest-selected OCI artifact.
            source (Path | None): Always absent because OCI content is not a downloaded file.
            destination (Path): Private staging directory for an opaque content record.

        Returns:
            Path: Staging directory containing the verified opaque content record.

        Raises:
            InstallError: If the artifact shape, pull, inspection, or resolved digest is unsafe.
        """
        if artifact.acquisition is not AcquisitionKind.OCI or source is not None:
            raise InstallError("OCI installer accepts only manifest OCI artifacts without a file.")
        pull_rejected = False
        try:
            inspected = self.transport.inspect_image(artifact.source)
            if inspected.exit_code != 0 or not inspected.stdout:
                self.progress("Installing sandbox command image…")
                pulled = self.transport.pull_image(artifact.source)
                if pulled.exit_code != 0:
                    pull_rejected = True
                else:
                    inspected = self.transport.inspect_image(artifact.source)
        except (RuntimeError, ValueError) as error:
            raise InstallError("OCI content acquisition could not be completed.") from error
        if pull_rejected or inspected.exit_code != 0:
            raise InstallError("OCI runtime rejected the pinned content reference.")
        evidence = parse_image_evidence(inspected.stdout, self.endpoint.platform)
        identity = artifact.oci_identity
        if identity is None or evidence != identity:
            raise InstallError("OCI inspection did not attest the selected immutable image.")
        destination.mkdir(mode=0o700, parents=True, exist_ok=True)
        if destination.is_symlink() or any(destination.iterdir()):
            raise InstallError("OCI staging destination is not an empty private directory.")
        record = destination / "oci-content.json"
        record.write_text(
            json.dumps(
                {
                    "artifact_id": artifact.artifact_id,
                    "index_digest": identity.index_digest,
                    "manifest_digest": identity.manifest_digest,
                    "config_digest": identity.config_digest,
                    "namespace": self.endpoint.namespace,
                    "content_store_identity": self.endpoint.content_store_identity,
                    "platform": {
                        "os": self.endpoint.platform.os,
                        "architecture": self.endpoint.platform.architecture,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        record.chmod(0o400)
        return destination


def parse_image_evidence(raw: bytes, platform: PlatformSelector) -> OciImageIdentity:
    """Parse exact OCI descriptor identities from nerdctl's native inspection JSON.

    Args:
        raw (bytes): Complete untruncated native image-inspection JSON.
        platform (PlatformSelector): Exact selected image platform.

    Returns:
        OciImageIdentity: Validated index, manifest, and config descriptor identities.

    Raises:
        InstallError: If the schema, platform, or descriptor graph is malformed.
    """
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InstallError("OCI inspection produced an unsupported schema.") from error
    required = {"Image", "ManifestDesc", "Manifest", "ImageConfigDesc", "ImageConfig", "size"}
    optional = {"IndexDesc", "Index"}
    if (
        not isinstance(payload, dict)
        or not required.issubset(payload)
        or set(payload) - required - optional
        or ("IndexDesc" in payload) != ("Index" in payload)
    ):
        raise InstallError("OCI inspection produced an unsupported schema.")
    expected_platform = {"os": platform.os, "architecture": platform.architecture}
    image = _object(payload, "Image")
    target = _object(image, "Target")
    manifest_descriptor = _object(payload, "ManifestDesc")
    manifest = _object(payload, "Manifest")
    config_descriptor = _object(manifest, "config")
    inspected_config_descriptor = _object(payload, "ImageConfigDesc")
    image_config = _object(payload, "ImageConfig")
    index_digest = _digest(target, "digest")
    manifest_digest = _digest(manifest_descriptor, "digest")
    config_digest = _digest(config_descriptor, "digest")
    if (
        manifest.get("schemaVersion") != 2
        or _digest(inspected_config_descriptor, "digest") != config_digest
        or {
            "os": image_config.get("os"),
            "architecture": image_config.get("architecture"),
        }
        != expected_platform
    ):
        raise InstallError("OCI inspection produced an unsupported schema.")
    if "Index" in payload:
        index_descriptor = _object(payload, "IndexDesc")
        index = _object(payload, "Index")
        manifests = index.get("manifests")
        if (
            index.get("schemaVersion") != 2
            or _digest(index_descriptor, "digest") != index_digest
            or not isinstance(manifests, list)
        ):
            raise InstallError("OCI inspection produced an unsupported schema.")
        matches = [
            item
            for item in manifests
            if isinstance(item, dict) and _matches_platform(item.get("platform"), expected_platform)
        ]
        if len(matches) != 1 or _digest(matches[0], "digest") != manifest_digest:
            raise InstallError("OCI inspection did not select exactly one requested platform.")
    elif index_digest != manifest_digest:
        raise InstallError("OCI single-manifest inspection has conflicting target identities.")
    return OciImageIdentity(
        index_digest=index_digest,
        manifest_digest=manifest_digest,
        config_digest=config_digest,
    )


def _matches_platform(value: object, expected: dict[str, object]) -> bool:
    """Match the pinned Docker arm64 descriptor including its required v8 variant."""
    if not isinstance(value, dict) or set(value) - {"os", "architecture", "variant"}:
        return False
    if {"os": value.get("os"), "architecture": value.get("architecture")} != expected:
        return False
    variant = value.get("variant")
    return variant is None or (expected.get("architecture") == "arm64" and variant == "v8")


def _object(value: dict[str, object], key: str) -> dict[str, object]:
    """Return one required JSON object or reject schema drift."""
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise InstallError("OCI inspection produced an unsupported schema.")
    return nested


def _digest(value: dict[str, object], key: str) -> str:
    """Return one exact lowercase SHA-256 descriptor digest."""
    digest = value.get(key)
    if (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest) != 71
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise InstallError("OCI inspection produced an unsupported descriptor digest.")
    return digest
