"""Test strict release SBOM and provenance promotion validation."""

import hashlib
import json
from collections.abc import Iterator

import pytest

from loop.execution.runtime.manifest import ManifestError, RuntimeManifest, load_embedded_release
from loop.execution.runtime.provenance import validate_release_provenance


class _EvidenceTransport:
    """Return immutable in-memory evidence through the production stream contract."""

    responses: dict[str, bytes]
    final_url: str | None

    def __init__(self, responses: dict[str, bytes], final_url: str | None = None) -> None:
        self.responses = responses
        self.final_url = final_url

    def stream(self, url: str) -> Iterator[tuple[str, Iterator[bytes]]]:
        """Yield one configured evidence response."""
        yield self.final_url or url, iter((self.responses[url],))


def _release_payload() -> dict[str, object]:
    """Return a mutable minimal release payload with typed evidence."""
    digest = "sha256:" + "a" * 64
    documents = _evidence_documents(digest.removeprefix("sha256:"))
    evidence = [
        {
            "evidence_type": "cyclonedx",
            "media_type": "application/vnd.cyclonedx+json",
            "source": "https://evidence.example/sbom.json",
            "digest": "sha256:" + hashlib.sha256(documents["cyclonedx"]).hexdigest(),
            "subject_digest": digest,
        },
        {
            "evidence_type": "slsa-provenance",
            "media_type": "application/vnd.in-toto+json",
            "source": "https://evidence.example/provenance.json",
            "digest": "sha256:" + hashlib.sha256(documents["slsa-provenance"]).hexdigest(),
            "subject_digest": digest,
        },
    ]
    return {
        "schema_version": 1,
        "mode": "release",
        "artifacts": [
            {
                "artifact_id": "broker",
                "version": "1",
                "role": "envoy",
                "platform": {"os": "linux", "architecture": "arm64"},
                "source": f"registry.example/broker@{digest}",
                "size": None,
                "digest": digest,
                "acquisition": "oci",
                "media_type": "application/vnd.oci.image.manifest.v1+json",
                "layout": None,
                "oci_identity": {
                    "index_digest": digest,
                    "manifest_digest": digest,
                    "config_digest": digest,
                },
                "dependencies": [],
                "capabilities": [],
                "metadata": [],
                "sbom": "https://evidence.example/documentation",
                "evidence": evidence,
                "notices": "https://evidence.example/notices",
                "allowed_redirect_origins": [],
            }
        ],
    }


def _evidence_documents(subject: str = "a" * 64) -> dict[str, bytes]:
    """Return minimal schema-valid evidence naming one exact subject."""
    return {
        "cyclonedx": json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "metadata": {"component": {"hashes": [{"alg": "SHA-256", "content": subject}]}},
            }
        ).encode(),
        "slsa-provenance": json.dumps(
            {
                "_type": "https://in-toto.io/Statement/v1",
                "predicateType": "https://slsa.dev/provenance/v1",
                "subject": [{"name": "artifact", "digest": {"sha256": subject}}],
            }
        ).encode(),
    }


def _transport() -> _EvidenceTransport:
    """Return a resolver for the evidence URLs in the release fixture."""
    documents = _evidence_documents()
    return _EvidenceTransport(
        {
            "https://evidence.example/sbom.json": documents["cyclonedx"],
            "https://evidence.example/provenance.json": documents["slsa-provenance"],
        }
    )


def test_release_provenance_accepts_typed_digest_bound_evidence() -> None:
    """Accept a real SBOM plus SLSA statement bound to the pinned subject digest."""
    manifest = RuntimeManifest.model_validate(_release_payload())

    validate_release_provenance(manifest, _transport())

    payload = _release_payload()
    artifact = payload["artifacts"][0]
    payload["artifacts"].append(
        {
            "artifact_id": "guest",
            "version": "1",
            "role": "guest-image",
            "platform": {"os": "linux", "architecture": "arm64"},
            "source": "https://evidence.example/guest.img",
            "size": 1,
            "digest": "d" * 64,
            "acquisition": "file",
            "media_type": "application/octet-stream",
            "layout": None,
            "dependencies": [],
            "capabilities": [],
            "metadata": [],
            "sbom": "https://evidence.example/guest-documentation",
            "notices": "https://evidence.example/notices",
            "allowed_redirect_origins": [],
        }
    )
    payload["artifacts"].append(
        {
            "artifact_id": "runtime",
            "version": "1",
            "role": "nerdctl",
            "platform": {"os": "linux", "architecture": "arm64"},
            "source": "https://evidence.example/runtime.tar.gz",
            "size": 1,
            "digest": "a" * 64,
            "acquisition": "archive",
            "media_type": "application/gzip",
            "layout": None,
            "dependencies": [],
            "capabilities": [],
            "metadata": [],
            "sbom": "https://evidence.example/runtime-documentation",
            "evidence": artifact["evidence"],
            "notices": "https://evidence.example/notices",
            "allowed_redirect_origins": [],
        }
    )
    validate_release_provenance(RuntimeManifest.model_validate(payload), _transport())


def test_release_provenance_resolves_bounded_digest_and_schema_verified_bytes() -> None:
    """Reject absent resolution, redirects, excess bytes, digests, schemas, and subjects."""
    manifest = RuntimeManifest.model_validate(_release_payload())
    with pytest.raises(ManifestError, match="requires resolved evidence bytes"):
        validate_release_provenance(manifest)
    with pytest.raises(ManifestError, match="evidence bytes could not be verified"):
        validate_release_provenance(manifest, _transport(), maximum_evidence_bytes=1)
    with pytest.raises(ManifestError, match="evidence bytes could not be verified"):
        validate_release_provenance(
            manifest,
            _EvidenceTransport(_transport().responses, "https://other.example/evidence"),
        )
    responses = dict(_transport().responses)
    responses["https://evidence.example/sbom.json"] = b"{}"
    with pytest.raises(ManifestError, match="evidence bytes could not be verified"):
        validate_release_provenance(manifest, _EvidenceTransport(responses))
    payload = _release_payload()
    invalid = json.dumps({"bomFormat": "CycloneDX", "specVersion": "1.6"}).encode()
    payload["artifacts"][0]["evidence"][0]["digest"] = (
        "sha256:" + hashlib.sha256(invalid).hexdigest()
    )
    responses = dict(_transport().responses)
    responses["https://evidence.example/sbom.json"] = invalid
    with pytest.raises(ManifestError, match="does not name the pinned subject"):
        validate_release_provenance(
            RuntimeManifest.model_validate(payload), _EvidenceTransport(responses)
        )
    with pytest.raises(ManifestError, match="byte limit must be positive"):
        validate_release_provenance(manifest, _transport(), maximum_evidence_bytes=0)

    payload = _release_payload()
    spdx = json.dumps(
        {
            "spdxVersion": "SPDX-2.3",
            "documentDescribes": ["SPDXRef-Package"],
            "packages": [
                {
                    "SPDXID": "SPDXRef-Package",
                    "checksums": [{"algorithm": "SHA256", "checksumValue": "a" * 64}],
                }
            ],
        }
    ).encode()
    sbom = payload["artifacts"][0]["evidence"][0]
    sbom.update(
        {
            "evidence_type": "spdx",
            "media_type": "application/spdx+json",
            "source": "https://evidence.example/sbom.spdx.json",
            "digest": "sha256:" + hashlib.sha256(spdx).hexdigest(),
        }
    )
    responses = dict(_transport().responses)
    responses["https://evidence.example/sbom.spdx.json"] = spdx
    validate_release_provenance(
        RuntimeManifest.model_validate(payload), _EvidenceTransport(responses)
    )

    malformed_spdx = json.dumps({"spdxVersion": "SPDX-2.3"}).encode()
    sbom["digest"] = "sha256:" + hashlib.sha256(malformed_spdx).hexdigest()
    responses["https://evidence.example/sbom.spdx.json"] = malformed_spdx
    with pytest.raises(ManifestError, match="does not name the pinned subject"):
        validate_release_provenance(
            RuntimeManifest.model_validate(payload), _EvidenceTransport(responses)
        )


def test_release_provenance_rejects_missing_mismatched_and_nonrelease_evidence() -> None:
    """Reject documentation links, another subject, duplicates, and nonrelease authority."""
    with pytest.raises(ManifestError, match="lima.*CycloneDX/SPDX SBOM.*SLSA provenance"):
        validate_release_provenance(load_embedded_release())

    payload = _release_payload()
    artifact = payload["artifacts"][0]
    artifact["evidence"][0]["subject_digest"] = "sha256:" + "d" * 64
    with pytest.raises(ManifestError, match="another subject"):
        validate_release_provenance(RuntimeManifest.model_validate(payload))

    payload = _release_payload()
    artifact = payload["artifacts"][0]
    artifact["evidence"].append(json.loads(json.dumps(artifact["evidence"][0])))
    with pytest.raises(ManifestError, match="duplicate"):
        validate_release_provenance(RuntimeManifest.model_validate(payload))

    payload = _release_payload()
    payload["mode"] = "candidate"
    with pytest.raises(ManifestError, match="release manifest"):
        validate_release_provenance(RuntimeManifest.model_validate(payload))

    payload = _release_payload()
    payload["artifacts"][0]["evidence"][0]["source"] = "http://unsafe.example/sbom"
    with pytest.raises(ValueError, match="credential-free HTTPS"):
        RuntimeManifest.model_validate(payload)

    payload = _release_payload()
    payload["artifacts"][0]["evidence"][0]["media_type"] = "application/json"
    with pytest.raises(ManifestError, match="media type is invalid"):
        validate_release_provenance(RuntimeManifest.model_validate(payload), _transport())
