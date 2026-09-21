"""Define closed, immutable runtime bootstrap records."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ... import constants


class ManifestMode(StrEnum):
    """Name the authority allowed to construct a manifest."""

    FIXTURE = "fixture"
    CANDIDATE = "candidate"
    RELEASE = "release"


class AcquisitionKind(StrEnum):
    """Name the reviewed acquisition mechanism for an artifact."""

    FILE = "file"
    ARCHIVE = "archive"
    OCI = "oci"


class ArtifactRole(StrEnum):
    """Name a closed runtime artifact role."""

    LIMA = "lima"
    GUEST_IMAGE = "guest-image"
    NERDCTL = "nerdctl"
    OCI_PROFILE = "oci-profile"
    ENVOY = "envoy"


class PlatformSelector(BaseModel):
    """Select one exact operating-system and architecture pair.

    Args:
        os (str): Canonical operating-system name.
        architecture (str): Canonical CPU architecture name.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    os: str = Field(pattern="^(linux|macos)$")
    architecture: str = Field(pattern="^(amd64|arm64)$")


class InstallLayout(BaseModel):
    """Declare one installed artifact layout without host paths.

    Args:
        files (tuple[str, ...]): Required normalized relative files.
        executables (tuple[str, ...]): Required executable files among ``files``.
        identities (dict[str, str]): SHA-256 identities for every required installed file.
        symlinks (dict[str, str]): Exact relative symlink names and contained targets.
        allow_unlisted_files (bool): Whether a digest-pinned archive may contain additional
            regular files beyond the required identity set.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    files: tuple[str, ...] = Field(min_length=1)
    executables: tuple[str, ...] = ()
    identities: dict[str, str] = Field(default_factory=dict)
    symlinks: dict[str, str] = Field(default_factory=dict)
    allow_unlisted_files: bool = False

    @field_validator("files", "executables")
    @classmethod
    def validate_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject paths which could escape an installation root.

        Args:
            values (tuple[str, ...]): Relative paths to validate.

        Returns:
            tuple[str, ...]: Validated paths.

        Raises:
            ValueError: If a path is absolute, escaping, empty, or duplicated.
        """
        for value in values:
            if not value or value.startswith("/") or "\\" in value or ".." in value.split("/"):
                raise ValueError("Install layout paths must be normalized relative paths.")
        if len(set(values)) != len(values):
            raise ValueError("Install layout paths must be unique.")
        return values

    @model_validator(mode="after")
    def validate_executables_are_files(self) -> InstallLayout:
        """Reject executable declarations absent from the required file layout.

        Returns:
            InstallLayout: This validated layout.

        Raises:
            ValueError: If an executable is not a declared required file.
        """
        if not set(self.executables).issubset(self.files):
            raise ValueError("Install layout executables must be declared files.")
        if set(self.identities) != set(self.files) or any(
            len(value) != constants.SHA256_HEX_LENGTH
            or any(character not in constants.HEX_DIGITS for character in value)
            for value in self.identities.values()
        ):
            raise ValueError("Install layout requires one SHA-256 identity per declared file.")
        file_names = set(self.files)
        for name, target in self.symlinks.items():
            if (
                not name
                or name.startswith("/")
                or "\\" in name
                or ".." in name.split("/")
                or name in file_names
                or not target
                or target.startswith("/")
                or "\\" in target
            ):
                raise ValueError("Install layout symlinks must be unique contained relative paths.")
            resolved = Path(name).parent.joinpath(target)
            depth = 0
            for part in resolved.parts:
                depth += -1 if part == ".." else 0 if part == "." else 1
                if depth < 0:
                    raise ValueError("Install layout symlink target escapes its artifact root.")
        return self


class OciImageIdentity(BaseModel):
    """Declare every immutable OCI descriptor required for one selected platform.

    Args:
        index_digest (str): Requested index or manifest descriptor digest.
        manifest_digest (str): Selected platform manifest descriptor digest.
        config_digest (str): Selected platform configuration descriptor digest.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    index_digest: str = Field(pattern="^sha256:[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern="^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern="^sha256:[0-9a-f]{64}$")


class ArtifactMetadata(BaseModel):
    """Record one validated artifact-specific qualification fact.

    Args:
        name (str): Stable namespaced fact name.
        value (str): Immutable expected value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern="^[a-z][a-z0-9.-]{0,127}$")
    value: str = Field(min_length=1, max_length=512)


class ArtifactDependency(BaseModel):
    """Reference one prerequisite artifact by stable identifier.

    Args:
        artifact_id (str): Stable identifier of the prerequisite artifact.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(pattern="^[a-z][a-z0-9-]{0,63}$")


class ArtifactEvidence(BaseModel):
    """Bind an immutable SBOM or build provenance artifact to one runtime subject.

    Args:
        evidence_type (str): Closed evidence kind: CycloneDX, SPDX, or SLSA provenance.
        media_type (str): Exact machine-readable artifact media type.
        source (str): Immutable credential-free HTTPS evidence URL.
        digest (str): SHA-256 identity of the evidence bytes.
        subject_digest (str): SHA-256 identity of the runtime artifact described by the evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_type: str = Field(pattern="^(cyclonedx|spdx|slsa-provenance)$")
    media_type: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=2048)
    digest: str = Field(pattern="^sha256:[0-9a-f]{64}$")
    subject_digest: str = Field(pattern="^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_source(self) -> ArtifactEvidence:
        """Require an immutable credential-free HTTPS evidence source."""
        parsed = urlparse(self.source)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise ValueError("Runtime evidence requires a credential-free HTTPS source.")
        return self


class Artifact(BaseModel):
    """Describe one immutable runtime artifact.

    Args:
        artifact_id (str): Stable artifact identifier.
        version (str): Immutable upstream version.
        role (ArtifactRole): Closed artifact role.
        platform (PlatformSelector): Exact target platform.
        source (str): HTTPS URL or immutable OCI reference.
        size (int | None): Expected byte size for file material.
        digest (str): SHA-256 or OCI digest.
        acquisition (AcquisitionKind): Reviewed acquisition mechanism.
        media_type (str): Declared content media type.
        layout (InstallLayout | None): Required host-file layout.
        dependencies (tuple[ArtifactDependency, ...]): Prerequisite artifacts.
        capabilities (tuple[str, ...]): Capability triggers.
        metadata (tuple[ArtifactMetadata, ...]): Artifact-specific qualification facts.
        sbom (str): Legacy upstream documentation reference; never accepted as release evidence.
        evidence (tuple[ArtifactEvidence, ...]): Independently digest-verified SBOM and provenance
            artifacts bound to this runtime subject.
        notices (str): Immutable notices reference.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(pattern="^[a-z][a-z0-9-]{0,63}$")
    version: str = Field(min_length=1, max_length=128)
    role: ArtifactRole
    platform: PlatformSelector
    source: str = Field(min_length=1, max_length=2048)
    size: int | None = Field(default=None, ge=0, le=2**40)
    digest: str = Field(pattern="^(sha256:)?[0-9a-f]{64}$")
    acquisition: AcquisitionKind
    media_type: str = Field(min_length=1, max_length=128)
    layout: InstallLayout | None = None
    oci_identity: OciImageIdentity | None = None
    dependencies: tuple[ArtifactDependency, ...] = ()
    capabilities: tuple[str, ...] = ()
    metadata: tuple[ArtifactMetadata, ...] = ()
    sbom: str = Field(min_length=1, max_length=2048)
    evidence: tuple[ArtifactEvidence, ...] = ()
    notices: str = Field(min_length=1, max_length=2048)
    allowed_redirect_origins: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_acquisition_shape(self) -> Artifact:
        """Require the digest and install shape appropriate to acquisition.

        Returns:
            Artifact: This validated artifact.

        Raises:
            ValueError: If the acquisition kind and artifact fields disagree.
        """
        references = (self.sbom, self.notices, *self.allowed_redirect_origins)
        if any(
            urlparse(reference).scheme != "https"
            or not urlparse(reference).netloc
            or urlparse(reference).username
            or urlparse(reference).password
            or urlparse(reference).fragment
            for reference in references
        ):
            raise ValueError("Artifact redirect origins must be HTTPS origins.")
        names = [entry.name for entry in self.metadata]
        if len(names) != len(set(names)):
            raise ValueError("Artifact metadata names must be unique.")
        if self.acquisition is AcquisitionKind.OCI:
            registry = urlparse("//" + self.source.partition("@")[0])
            if not self.digest.startswith("sha256:"):
                raise ValueError("OCI artifacts require a sha256 digest.")
            if (
                self.size is not None
                or self.layout is not None
                or self.oci_identity is None
                or self.oci_identity.index_digest != self.digest
                or not self.source.endswith("@" + self.digest)
                or "://" in self.source
                or not registry.hostname
                or registry.username
                or registry.password
                or registry.query
                or registry.fragment
            ):
                raise ValueError(
                    "OCI artifacts require a digest-pinned source and complete OCI identity."
                )
        elif self.size is None or self.oci_identity is not None:
            raise ValueError("File artifacts require an exact size and no OCI identity.")
        return self
