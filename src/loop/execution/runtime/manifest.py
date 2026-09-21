"""Validate and select closed runtime manifests before any I/O."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...utils import json_encode, sha256_digest
from .models import AcquisitionKind, Artifact, ManifestMode, PlatformSelector


class ManifestError(ValueError):
    """Report a manifest that cannot safely select runtime artifacts."""


class RuntimeManifest(BaseModel):
    """Describe a versioned closed inventory of runtime artifacts.

    Args:
        schema_version (int): Supported manifest schema version.
        mode (ManifestMode): Trust authority for this inventory.
        artifacts (tuple[Artifact, ...]): Immutable artifact records.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(ge=1, le=1)
    mode: ManifestMode
    artifacts: tuple[Artifact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph(self) -> RuntimeManifest:
        """Validate identifier uniqueness, source authority, and dependency closure.

        Returns:
            RuntimeManifest: This validated manifest.

        Raises:
            ManifestError: If identifiers, sources, or dependency closure are invalid.
        """
        ids = {artifact.artifact_id for artifact in self.artifacts}
        if len(ids) != len(self.artifacts):
            raise ManifestError("Runtime manifest has duplicate artifact identifiers.")
        for artifact in self.artifacts:
            parsed = urlparse(artifact.source)
            if artifact.acquisition is AcquisitionKind.OCI:
                registry = urlparse("//" + artifact.source.partition("@")[0])
                if "@sha256:" not in artifact.source or not registry.hostname:
                    # Artifact validation rejects this before a nested model can reach here.
                    raise ManifestError(  # pragma: no cover
                        "OCI artifacts require an immutable digest reference."
                    )
            elif (
                parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password
            ):
                raise ManifestError("File artifacts require credential-free HTTPS sources.")
            if any(dep.artifact_id not in ids for dep in artifact.dependencies):
                raise ManifestError("Runtime manifest references an unknown dependency.")
        graph = {a.artifact_id: {d.artifact_id for d in a.dependencies} for a in self.artifacts}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(identifier: str) -> None:
            if identifier in visiting:
                raise ManifestError("Runtime manifest dependency graph contains a cycle.")
            if identifier not in visited:
                visiting.add(identifier)
                for dependency in graph[identifier]:
                    visit(dependency)
                visiting.remove(identifier)
                visited.add(identifier)

        for identifier in graph:
            visit(identifier)
        return self

    @property
    def digest(self) -> str:
        """Return the canonical SHA-256 identity of this manifest.

        Returns:
            str: ``sha256:`` prefixed canonical manifest digest.
        """
        payload = json_encode(self.model_dump(mode="json"))
        return f"sha256:{sha256_digest(payload)}"

    def select(
        self,
        platform: PlatformSelector,
        capabilities: frozenset[str],
    ) -> tuple[Artifact, ...]:
        """Return the minimal dependency-closed artifact selection.

        Args:
            platform (PlatformSelector): Exact platform to select.
            capabilities (frozenset[str]): Requested capability triggers.

        Returns:
            tuple[Artifact, ...]: Dependencies-first selected artifacts.

        Raises:
            ManifestError: If no requested capability is available on the platform.
        """
        available = {a.artifact_id: a for a in self.artifacts if a.platform == platform}
        roots = [a for a in available.values() if set(a.capabilities) & capabilities]
        if capabilities and not roots:
            raise ManifestError("No runtime artifact supports the requested platform/capabilities.")
        selected: dict[str, Artifact] = {}

        def add(artifact: Artifact) -> None:
            if artifact.acquisition is not AcquisitionKind.OCI and artifact.layout is None:
                raise ManifestError("Selected file artifact has no install layout.")
            for dependency in artifact.dependencies:
                target = available.get(dependency.artifact_id)
                if target is None:
                    raise ManifestError(
                        "Selected artifact dependency is unavailable on this platform."
                    )
                add(target)
            selected[artifact.artifact_id] = artifact

        for root in roots:
            add(root)
        return tuple(selected.values())


def load_checked_in_candidate(path: Path) -> RuntimeManifest:
    """Load a local candidate manifest for developer or native CI use.

    Args:
        path (Path): Checked-in candidate manifest path.

    Returns:
        RuntimeManifest: Validated candidate manifest.

    Raises:
        ManifestError: If the file is not candidate authority.
    """
    manifest = RuntimeManifest.model_validate_json(path.read_bytes())
    if manifest.mode is not ManifestMode.CANDIDATE:
        raise ManifestError("Developer selection accepts only candidate manifests.")
    return manifest


def load_embedded_release(payload: bytes | None = None) -> RuntimeManifest:
    """Load the distribution-embedded release inventory.

    Args:
        payload (bytes | None): Embedded manifest bytes. Defaults to the packaged release
            inventory.

    Returns:
        RuntimeManifest: Validated release manifest.

    Raises:
        ManifestError: If the payload is not release authority.
    """
    if payload is None:
        payload = resources.files(__package__).joinpath("release-manifest.json").read_bytes()
    manifest = RuntimeManifest.model_validate_json(payload)
    if manifest.mode is not ManifestMode.RELEASE:
        raise ManifestError("Production selection accepts only embedded release manifests.")
    return manifest
