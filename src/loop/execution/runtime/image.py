"""Validate the single checked-in sandbox image definition and readiness record."""

from __future__ import annotations

import hashlib
import json
import re
from importlib import resources

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...utils import json_encode, sha256_digest
from .models import OciImageIdentity, PlatformSelector

_RUNTIME_PATH = "/tools/bin:/usr/local/bin:/usr/bin:/bin"
_PLATFORMS = frozenset({"linux/amd64", "linux/arm64"})


class RuntimeEnvironment(BaseModel):
    """Declare deterministic process identity and executable lookup.

    Args:
        user (str): Non-root runtime account name.
        uid (int): Numeric runtime user identity.
        gid (int): Numeric runtime group identity.
        path (str): Complete deterministic executable search path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user: str = Field(pattern="^agent$")
    uid: int = Field(ge=1, le=65535)
    gid: int = Field(ge=1, le=65535)
    path: str = Field(pattern="^/tools/bin:/usr/local/bin:/usr/bin:/bin$")


class SandboxToolDefinition(BaseModel):
    """Declare one digest-pinned tool archive installed into the image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(pattern=r"^[0-9]+(?:[.][0-9]+){2}$")
    origin: str = Field(
        pattern=(
            r"^https://releases[.]astral[.]sh/github/uv/releases/download/[0-9]+(?:[.][0-9]+){2}$"
        )
    )
    max_bytes: int = Field(ge=1, le=100 * 1024 * 1024)
    sha256: dict[str, str]

    @model_validator(mode="after")
    def validate_definition(self) -> SandboxToolDefinition:
        """Require one digest for every supported image platform."""
        if not self.origin.endswith(f"/{self.version}"):
            raise ValueError("Sandbox tool origin must match its exact version.")
        if set(self.sha256) != _PLATFORMS or any(
            re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in self.sha256.values()
        ):
            raise ValueError("Sandbox tool archives require exact platform SHA-256 digests.")
        return self


class SandboxImageDefinition(BaseModel):
    """Describe the one locally built sandbox image and locked inventory.

    Args:
        schema_version (int): Supported definition schema version.
        base_image (str): Digest-pinned base image reference.
        debian_snapshot (str): Immutable Debian snapshot timestamp.
        platforms (tuple[str, ...]): Supported OCI platform variants.
        environment (RuntimeEnvironment): Deterministic non-root process environment.
        packages (dict[str, str]): Exact Debian package versions in the image.
        tools (dict[str, SandboxToolDefinition]): Pinned non-Debian tool archives.
        commands (dict[str, str]): Essential commands and their expected image paths.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(ge=1, le=1)
    base_image: str = Field(pattern=r"^docker\.io/library/debian@sha256:[0-9a-f]{64}$")
    debian_snapshot: str = Field(pattern=r"^[0-9]{8}T[0-9]{6}Z$")
    platforms: tuple[str, ...] = Field(min_length=2, max_length=2)
    environment: RuntimeEnvironment
    packages: dict[str, str] = Field(min_length=1)
    tools: dict[str, SandboxToolDefinition] = Field(min_length=1)
    commands: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_definition(self) -> SandboxImageDefinition:
        """Reject mutable or incomplete image definitions.

        Returns:
            SandboxImageDefinition: This validated single-image definition.

        Raises:
            ValueError: If platforms, identities, packages, or commands are not closed.
        """
        if frozenset(self.platforms) != _PLATFORMS or len(set(self.platforms)) != 2:
            raise ValueError("Sandbox image requires exactly linux/arm64 and linux/amd64.")
        if self.environment.path != _RUNTIME_PATH or (
            self.environment.uid,
            self.environment.gid,
        ) != (1000, 1000):
            raise ValueError("Sandbox image requires the reviewed non-root environment.")
        if any(
            not name
            or not version
            or any(character.isspace() for character in name + version)
            or version in {"latest", "stable"}
            for name, version in self.packages.items()
        ):
            raise ValueError("Sandbox packages require exact whitespace-free versions.")
        if set(self.tools) != {"uv"}:
            raise ValueError("Sandbox image requires the reviewed external tool inventory.")
        allowed_directories = tuple(value + "/" for value in _RUNTIME_PATH.split(":"))
        if any(
            not name
            or "/" in name
            or not path.startswith(allowed_directories)
            or path.endswith("/")
            for name, path in self.commands.items()
        ):
            raise ValueError("Sandbox commands require names and deterministic PATH paths.")
        return self

    @property
    def package_arguments(self) -> str:
        """Return stable package-version arguments for the Containerfile.

        Returns:
            str: Space-separated package assignments sorted by package name.
        """
        return " ".join(f"{name}={version}" for name, version in sorted(self.packages.items()))

    @property
    def tool_arguments(self) -> tuple[str, ...]:
        """Return stable external-tool arguments for the Containerfile.

        Returns:
            tuple[str, ...]: Validated build arguments derived from the locked inventory.
        """
        uv = self.tools["uv"]
        return (
            f"UV_VERSION={uv.version}",
            f"UV_ORIGIN={uv.origin}",
            f"UV_MAX_BYTES={uv.max_bytes}",
            f"UV_AARCH64_SHA256={uv.sha256['linux/arm64']}",
            f"UV_X86_64_SHA256={uv.sha256['linux/amd64']}",
        )

    @property
    def digest(self) -> str:
        """Return the canonical inventory digest.

        Returns:
            str: ``sha256:`` prefixed canonical definition digest.
        """
        return f"sha256:{sha256_digest(json_encode(self.model_dump(mode='json')))}"

    @property
    def source_version(self) -> str:
        """Return the version binding the inventory and Containerfile.

        Returns:
            str: ``sha256:`` prefixed checked-in image-source digest.
        """
        digest = hashlib.sha256()
        digest.update(json_encode(self.model_dump(mode="json")).encode())
        digest.update(b"\0")
        digest.update(load_sandbox_containerfile())
        return f"sha256:{digest.hexdigest()}"


class RuntimeImage(BaseModel):
    """Describe the locally built immutable sandbox image.

    Args:
        reference (str): Private local image name pinned to its descriptor digest.
        platform (PlatformSelector): Exact selected OCI platform.
        identity (OciImageIdentity): Exact index, manifest, and config identities.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference: str = Field(pattern=r"^loop\.local/sandbox@sha256:[0-9a-f]{64}$")
    platform: PlatformSelector
    identity: OciImageIdentity


class SandboxImageReadiness(BaseModel):
    """Record only the source version and local image identity needed for reuse.

    Args:
        schema_version (int): Supported readiness schema version.
        source_version (str): Containerfile and inventory source identity.
        image (RuntimeImage): Locally present immutable image identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(ge=1, le=1)
    source_version: str = Field(pattern="^sha256:[0-9a-f]{64}$")
    image: RuntimeImage


def load_sandbox_image_definition(payload: bytes | None = None) -> SandboxImageDefinition:
    """Load the packaged single-image inventory.

    Args:
        payload (bytes | None): Optional isolated-test payload; defaults to packaged inventory.

    Returns:
        SandboxImageDefinition: Validated single-image definition.
    """
    if payload is None:
        payload = resources.files(__package__).joinpath("inventory.json").read_bytes()
    return SandboxImageDefinition.model_validate_json(payload)


def load_sandbox_containerfile() -> bytes:
    """Load the authoritative packaged sandbox Containerfile.

    Returns:
        bytes: Exact checked-in Containerfile bytes used for local construction.
    """
    return resources.files(__package__).joinpath("Containerfile").read_bytes()


def image_has_safe_defaults(raw: bytes, platform: PlatformSelector) -> bool:
    """Return whether native inspection shows the expected unprivileged image defaults.

    Args:
        raw (bytes): Complete native image-inspection JSON.
        platform (PlatformSelector): Expected image platform.

    Returns:
        bool: Whether the image defaults are non-root, deterministic, and non-privileged.
    """
    try:
        payload = json.loads(raw)
        config = payload["ImageConfig"]["config"]
    except (KeyError, TypeError, ValueError):
        return False
    environment = config.get("Env", [])
    expected_path = f"PATH={_RUNTIME_PATH}"
    return bool(
        payload["ImageConfig"].get("os") == platform.os
        and payload["ImageConfig"].get("architecture") == platform.architecture
        and config.get("User") in {"agent:agent", "1000:1000"}
        and config.get("WorkingDir") == "/workspace"
        and expected_path in environment
        and not config.get("Entrypoint")
        and not config.get("ExposedPorts")
        and not config.get("Volumes")
    )
