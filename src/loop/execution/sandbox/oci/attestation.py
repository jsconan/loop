"""Parse and verify the closed OCI inspection evidence schema."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ...contracts import BackendAttestation
from .spec import OciExecutionSpec


class OciAttestationError(ValueError):
    """Report missing, forged, or schema-incompatible OCI evidence."""


class PlatformEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    os: str
    architecture: str


class ImageEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    index_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    platform: PlatformEvidence


class MountEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str
    destination: str
    read_only: bool


class ResourceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    memory_bytes: int
    pids: int
    cpu_quota_us: int
    disk_bytes: int
    temporary_bytes: int
    cache_bytes: int
    persistent_write_bytes: int


class ConfigEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    cwd: str
    environment: tuple[tuple[str, str], ...]
    labels: tuple[tuple[str, str], ...]
    mounts: tuple[MountEvidence, ...]
    capabilities: tuple[str, ...]
    seccomp_mode: str
    seccomp_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    resources: ResourceEvidence
    network_mode: str
    management_sockets: tuple[str, ...]


class InspectionEvidence(BaseModel):
    """Define the single supported normalized nerdctl/containerd evidence schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1]
    container_id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    image: ImageEvidence
    config: ConfigEvidence


def attest_inspection(
    evidence: bytes,
    spec: OciExecutionSpec,
    namespace: str,
    platform_os: str,
    platform_architecture: str,
) -> BackendAttestation:
    """Verify exact normalized inspection evidence before reporting process start.

    Args:
        evidence: UTF-8 JSON emitted by the pinned inspector adapter.
        spec: Previously compiled closed OCI specification.
        namespace: Expected private containerd namespace.
        platform_os: Expected selected operating system.
        platform_architecture: Expected selected architecture.

    Returns:
        BackendAttestation: Sanitized immutable identity evidence.

    Raises:
        OciAttestationError: If evidence is malformed, incomplete, or differs from the spec.
    """
    try:
        payload = json.loads(evidence)
        inspected = InspectionEvidence.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise OciAttestationError("OCI inspection evidence uses an unsupported schema.") from error
    config = inspected.config
    expected_mount = (
        spec.workspace_mount.source_id,
        spec.workspace_mount.destination,
        spec.workspace_mount.read_only,
    )
    observed_mounts = tuple(
        (mount.source_id, mount.destination, mount.read_only) for mount in config.mounts
    )
    if (
        inspected.namespace != namespace
        or (inspected.image.platform.os, inspected.image.platform.architecture)
        != (platform_os, platform_architecture)
        or (
            inspected.image.index_digest,
            inspected.image.manifest_digest,
            inspected.image.config_digest,
        )
        != (spec.image_digest, spec.image_manifest_digest, spec.image_config_digest)
        or config.cwd != spec.cwd
        or config.environment != spec.environment
        or config.labels != spec.labels
        or observed_mounts != (expected_mount,)
        or config.capabilities
        or config.seccomp_mode != "upstream-default"
        or config.resources
        != ResourceEvidence(
            memory_bytes=spec.limits.memory_bytes,
            pids=spec.limits.pids,
            cpu_quota_us=spec.limits.cpu_quota_us,
            disk_bytes=spec.limits.disk_bytes,
            temporary_bytes=spec.limits.temporary_bytes,
            cache_bytes=spec.limits.cache_bytes,
            persistent_write_bytes=spec.limits.persistent_write_bytes,
        )
        or config.network_mode != spec.network_mode
        or config.management_sockets
    ):
        raise OciAttestationError(
            "OCI inspection evidence does not match the authorized specification."
        )
    return BackendAttestation(
        backend_id="oci-nerdctl-v1",
        runtime_digest=spec.image_manifest_digest,
        policy_digest=config.seccomp_digest,
        evidence=(("container_id", inspected.container_id), ("namespace", inspected.namespace)),
    )
