"""Validate independently verifiable runtime SBOM and provenance relationships."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Protocol

from .download import AcquisitionError, ArtifactTransport
from .manifest import ManifestError, RuntimeManifest
from .models import AcquisitionKind, ArtifactEvidence, ArtifactRole, ManifestMode

_EXECUTABLE_ROLES = frozenset({ArtifactRole.LIMA, ArtifactRole.NERDCTL})
_SBOM_TYPES = frozenset({"cyclonedx", "spdx"})
_MEDIA_TYPES = {
    "cyclonedx": "application/vnd.cyclonedx+json",
    "spdx": "application/spdx+json",
    "slsa-provenance": "application/vnd.in-toto+json",
}
_MAXIMUM_EVIDENCE_BYTES = 16 * 1024 * 1024


class EvidenceTransport(Protocol):
    """Resolve immutable evidence through the reviewed artifact transport contract."""

    def stream(self, url: str) -> Iterator[tuple[str, Iterator[bytes]]]:
        """Yield one final HTTPS URL and its bounded evidence chunks."""


def validate_release_provenance(
    manifest: RuntimeManifest,
    transport: ArtifactTransport | EvidenceTransport | None = None,
    maximum_evidence_bytes: int = _MAXIMUM_EVIDENCE_BYTES,
) -> None:
    """Require downloaded, parsed SBOM and provenance evidence for release artifacts.

    Args:
        manifest (RuntimeManifest): Candidate release inventory to validate.
        transport (ArtifactTransport | EvidenceTransport | None): Reviewed evidence-byte
            resolver. It may be omitted only when structural validation already proves the
            release incomplete.
        maximum_evidence_bytes (int): Positive aggregate ceiling for each evidence document.

    Raises:
        ManifestError: If authority, evidence bytes, schema, subject relationship, or coverage
            is missing or invalid.
    """
    if manifest.mode is not ManifestMode.RELEASE:
        raise ManifestError("Provenance promotion accepts only a release manifest.")
    if maximum_evidence_bytes <= 0:
        raise ManifestError("Runtime evidence byte limit must be positive.")
    failures: list[str] = []
    complete: list[tuple[str, str, tuple[ArtifactEvidence, ...]]] = []
    for artifact in manifest.artifacts:
        requires_evidence = (
            artifact.role in _EXECUTABLE_ROLES or artifact.acquisition is AcquisitionKind.OCI
        )
        if not requires_evidence:
            continue
        evidence_types = {entry.evidence_type for entry in artifact.evidence}
        missing = []
        if not evidence_types & _SBOM_TYPES:
            missing.append("CycloneDX/SPDX SBOM")
        if "slsa-provenance" not in evidence_types:
            missing.append("SLSA provenance")
        if missing:
            failures.append(f"'{artifact.artifact_id}': {', '.join(missing)}")
        subject_digest = artifact.digest
        if not subject_digest.startswith("sha256:"):
            subject_digest = f"sha256:{subject_digest}"
        for entry in artifact.evidence:
            if entry.subject_digest != subject_digest:
                failures.append(f"'{artifact.artifact_id}': evidence names another subject")
            if entry.media_type != _MEDIA_TYPES[entry.evidence_type]:
                failures.append(f"'{artifact.artifact_id}': evidence media type is invalid")
        if len({(entry.evidence_type, entry.digest) for entry in artifact.evidence}) != len(
            artifact.evidence
        ):
            failures.append(f"'{artifact.artifact_id}': duplicate evidence records")
        if not missing:
            complete.append((artifact.artifact_id, subject_digest, artifact.evidence))
    if failures:
        raise ManifestError("Runtime provenance validation failed: " + "; ".join(failures) + ".")
    if complete and transport is None:
        raise ManifestError("Runtime provenance validation requires resolved evidence bytes.")
    for artifact_id, subject_digest, evidence in complete:
        for entry in evidence:
            try:
                payload = _download_evidence(entry, transport, maximum_evidence_bytes)
                document = json.loads(payload)
            except (AcquisitionError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ManifestError(
                    f"Runtime provenance validation failed: '{artifact_id}': evidence bytes "
                    "could not be verified."
                ) from error
            if not isinstance(document, dict) or not _names_subject(
                entry.evidence_type, document, subject_digest.removeprefix("sha256:")
            ):
                raise ManifestError(
                    f"Runtime provenance validation failed: '{artifact_id}': parsed "
                    "evidence does not name the pinned subject."
                )


def _download_evidence(
    evidence: ArtifactEvidence,
    transport: ArtifactTransport | EvidenceTransport | None,
    maximum_bytes: int,
) -> bytes:
    """Resolve exactly one digest-verified bounded evidence document."""
    if transport is None:  # pragma: no cover - guarded by the public validator.
        raise AcquisitionError("Runtime evidence transport is unavailable.")
    digest = hashlib.sha256()
    content = bytearray()
    responses = 0
    for final_url, chunks in transport.stream(evidence.source):
        responses += 1
        if responses != 1 or final_url != evidence.source:
            raise AcquisitionError("Runtime evidence source changed during resolution.")
        for chunk in chunks:
            if len(content) + len(chunk) > maximum_bytes:
                raise AcquisitionError("Runtime evidence exceeds its byte limit.")
            digest.update(chunk)
            content.extend(chunk)
    if responses != 1 or f"sha256:{digest.hexdigest()}" != evidence.digest:
        raise AcquisitionError("Runtime evidence SHA-256 identity does not match.")
    return bytes(content)


def _names_subject(evidence_type: str, document: dict[str, object], digest: str) -> bool:
    """Return whether one parsed schema names the exact SHA-256 subject."""
    if evidence_type == "cyclonedx":
        metadata = document.get("metadata")
        component = metadata.get("component") if isinstance(metadata, dict) else None
        hashes = component.get("hashes") if isinstance(component, dict) else None
        return (
            document.get("bomFormat") == "CycloneDX"
            and isinstance(document.get("specVersion"), str)
            and _hash_list_names(hashes, digest)
        )
    if evidence_type == "spdx":
        packages = document.get("packages")
        described = document.get("documentDescribes")
        if not (
            isinstance(document.get("spdxVersion"), str)
            and isinstance(packages, list)
            and isinstance(described, list)
        ):
            return False
        return any(
            isinstance(package, dict)
            and package.get("SPDXID") in described
            and _checksum_list_names(package.get("checksums"), digest)
            for package in packages
        )
    subjects = document.get("subject")
    return (
        document.get("_type") == "https://in-toto.io/Statement/v1"
        and isinstance(document.get("predicateType"), str)
        and "slsa.dev/provenance/" in document["predicateType"]
        and isinstance(subjects, list)
        and any(
            isinstance(subject, dict)
            and isinstance(subject.get("digest"), dict)
            and subject["digest"].get("sha256") == digest
            for subject in subjects
        )
    )


def _hash_list_names(value: object, digest: str) -> bool:
    """Match a CycloneDX SHA-256 component hash."""
    return isinstance(value, list) and any(
        isinstance(item, dict) and item.get("alg") == "SHA-256" and item.get("content") == digest
        for item in value
    )


def _checksum_list_names(value: object, digest: str) -> bool:
    """Match an SPDX SHA-256 package checksum."""
    return isinstance(value, list) and any(
        isinstance(item, dict)
        and item.get("algorithm") == "SHA256"
        and item.get("checksumValue") == digest
        for item in value
    )
