"""Validate the checked-in macOS runtime manifest without projecting another schema."""

from __future__ import annotations

from pathlib import Path

from ...runtime.manifest import (
    ManifestError,
    RuntimeManifest,
    load_checked_in_candidate,
    load_embedded_release,
)
from ...runtime.models import AcquisitionKind, Artifact, ArtifactRole, PlatformSelector

_MACOS_ARM64 = PlatformSelector(os="macos", architecture="arm64")
_LINUX_ARM64 = PlatformSelector(os="linux", architecture="arm64")
_EXPECTED_ARTIFACTS = {
    "lima": (ArtifactRole.LIMA, _MACOS_ARM64, AcquisitionKind.ARCHIVE),
    "guest-image": (ArtifactRole.GUEST_IMAGE, _LINUX_ARM64, AcquisitionKind.FILE),
    "nerdctl-full": (ArtifactRole.NERDCTL, _LINUX_ARM64, AcquisitionKind.ARCHIVE),
    "envoy": (ArtifactRole.ENVOY, _LINUX_ARM64, AcquisitionKind.OCI),
}
_COMPONENTS = frozenset({"nerdctl", "containerd", "runc", "buildkit"})
_ENTITLEMENT_PREFIX = "codesign-entitlement."


class MacosCandidateError(ValueError):
    """Report an invalid macOS runtime candidate manifest."""


def load_macos_runtime_candidate(path: Path) -> RuntimeManifest:
    """Load and validate one checked-in macOS candidate manifest.

    Args:
        path (Path): Repository-controlled candidate manifest path.

    Returns:
        RuntimeManifest: The authoritative candidate manifest without schema projection.

    Raises:
        MacosCandidateError: If the candidate cannot be read or lacks required identities.
    """
    try:
        manifest = load_checked_in_candidate(path)
        _validate_macos_manifest(manifest)
    except (OSError, ValueError, ManifestError) as error:
        raise MacosCandidateError("macOS runtime candidate manifest is invalid.") from error
    return manifest


def load_macos_runtime_release(payload: bytes | None = None) -> RuntimeManifest:
    """Load and validate the embedded production macOS runtime manifest.

    Args:
        payload (bytes | None): Optional release payload used by isolated verification.

    Returns:
        RuntimeManifest: Validated release manifest with the closed macOS inventory.

    Raises:
        MacosCandidateError: If the release lacks a required identity or artifact.
    """
    try:
        manifest = load_embedded_release(payload)
        _validate_macos_manifest(manifest)
    except (OSError, ValueError, ManifestError) as error:
        raise MacosCandidateError("macOS runtime release manifest is invalid.") from error
    return manifest


def macos_artifact(manifest: RuntimeManifest, artifact_id: str) -> Artifact:
    """Return one required artifact from a validated macOS runtime manifest.

    Args:
        manifest (RuntimeManifest): Validated macOS candidate or release manifest.
        artifact_id (str): Stable required artifact identifier.

    Returns:
        Artifact: The exact immutable artifact record.

    Raises:
        MacosCandidateError: If the artifact is absent.
    """
    try:
        return next(item for item in manifest.artifacts if item.artifact_id == artifact_id)
    except StopIteration as error:
        raise MacosCandidateError("macOS runtime manifest is incomplete.") from error


def macos_component_version(manifest: RuntimeManifest, component: str) -> str:
    """Return one guest runtime component version from manifest metadata.

    Args:
        manifest (RuntimeManifest): Validated macOS candidate or release manifest.
        component (str): Supported guest component identifier.

    Returns:
        str: Exact qualified component version.

    Raises:
        MacosCandidateError: If the component is unsupported or absent.
    """
    if component not in _COMPONENTS:
        raise MacosCandidateError("macOS runtime component is unsupported.")
    return _metadata_value(macos_artifact(manifest, "nerdctl-full"), f"component.{component}")


def macos_entitlement_keys(manifest: RuntimeManifest) -> frozenset[str]:
    """Return the complete expected Lima code-signing entitlement set.

    Args:
        manifest (RuntimeManifest): Validated macOS candidate or release manifest.

    Returns:
        frozenset[str]: Exact expected entitlement names.

    Raises:
        MacosCandidateError: If no valid virtualization entitlement set is declared.
    """
    artifact = macos_artifact(manifest, "lima")
    values = frozenset(
        item.name.removeprefix(_ENTITLEMENT_PREFIX)
        for item in artifact.metadata
        if item.name.startswith(_ENTITLEMENT_PREFIX) and item.value == "true"
    )
    if "com.apple.security.virtualization" not in values or any(
        not value.startswith("com.apple.security.") for value in values
    ):
        raise MacosCandidateError("macOS runtime manifest lacks trusted Lima entitlements.")
    return values


def _metadata_value(artifact: Artifact, name: str) -> str:
    """Return one exact metadata value or reject an incomplete manifest."""
    try:
        return next(item.value for item in artifact.metadata if item.name == name)
    except StopIteration as error:
        raise MacosCandidateError("macOS runtime manifest metadata is incomplete.") from error


def _validate_macos_manifest(manifest: RuntimeManifest) -> None:
    """Require the complete, fixed macOS runtime artifact inventory."""
    if {artifact.artifact_id for artifact in manifest.artifacts} != set(_EXPECTED_ARTIFACTS):
        raise MacosCandidateError("macOS runtime manifest artifact inventory is incomplete.")
    for artifact_id, expected in _EXPECTED_ARTIFACTS.items():
        artifact = macos_artifact(manifest, artifact_id)
        if (artifact.role, artifact.platform, artifact.acquisition) != expected:
            raise MacosCandidateError("macOS runtime artifact identity is invalid.")
    lima = macos_artifact(manifest, "lima")
    if (
        lima.layout is None
        or lima.layout.executables != ("bin/limactl",)
        or "sandbox-management" not in lima.capabilities
    ):
        raise MacosCandidateError("macOS runtime Lima installation is incomplete.")
    envoy = macos_artifact(manifest, "envoy")
    if (
        envoy.oci_identity is None
        or "network-broker" not in envoy.capabilities
        or not envoy.source.startswith(
            f"docker.io/envoyproxy/envoy:distroless-v{envoy.version}@sha256:"
        )
    ):
        raise MacosCandidateError("macOS runtime network broker identity is incomplete.")
    macos_entitlement_keys(manifest)
    for component in _COMPONENTS:
        macos_component_version(manifest, component)
