"""Build, attest, activate, and reuse the single sandbox image inside Lima."""

from __future__ import annotations

import io
import logging
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock

from ....utils import json_encode
from ...runtime.image import (
    RuntimeImage,
    SandboxImageDefinition,
    SandboxImageReadiness,
    image_has_safe_defaults,
    load_sandbox_containerfile,
    load_sandbox_image_definition,
)
from ...runtime.install import InstallError
from ..oci.artifacts import parse_image_evidence
from .control_plane import MacosControlPlane

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InstalledSandboxImage:
    """Expose one ready local sandbox image and its durable readiness record.

    Args:
        image (RuntimeImage): Immutable local image identity.
        readiness (SandboxImageReadiness): Minimal source and image reuse record.
        readiness_path (Path): Private durable readiness-record location.
    """

    image: RuntimeImage
    readiness: SandboxImageReadiness
    readiness_path: Path


class MacosSandboxImageBuilder:
    """Own transient guest construction and atomic single-image activation.

    Args:
        control_plane (MacosControlPlane): Attested guest management boundary.
        state_root (Path): Private host state beneath the Lima invocation root.
        definition (SandboxImageDefinition | None): Checked-in definition or test override.
    """

    _control_plane: MacosControlPlane
    _state_root: Path
    _definition: SandboxImageDefinition

    def __init__(
        self,
        control_plane: MacosControlPlane,
        state_root: Path,
        definition: SandboxImageDefinition | None = None,
    ) -> None:
        if state_root.is_symlink():
            raise ValueError("Sandbox image state must be a private directory.")
        state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        state_root.chmod(0o700)
        if not state_root.is_dir():
            raise ValueError("Sandbox image state must be a private directory.")
        self._control_plane = control_plane
        self._state_root = state_root.resolve()
        self._definition = definition or load_sandbox_image_definition()

    @property
    def definition(self) -> SandboxImageDefinition:
        """Return the validated single-image definition.

        Returns:
            SandboxImageDefinition: Closed checked-in image definition.
        """
        return self._definition

    def prepare(self) -> InstalledSandboxImage:
        """Build or reuse the image after live readiness checks.

        Returns:
            InstalledSandboxImage: Ready local image and minimal durable record.

        Raises:
            RuntimeError: If construction, inspection, inventory, cleanup, or activation fails.
        """
        with FileLock(str(self._state_root / ".build.lock")):
            active = self._load_record("active.json")
            if (
                active is not None
                and active.readiness.source_version == self._definition.source_version
                and self._image_is_ready(active.readiness)
            ):
                return active
            built = self._build()
            try:
                self._activate(built.readiness, active.readiness if active is not None else None)
            except BaseException:
                self._control_plane.remove_image(built.image.reference)
                raise
            return InstalledSandboxImage(
                built.image,
                built.readiness,
                self._state_root / "active.json",
            )

    def _load_record(self, name: str) -> InstalledSandboxImage | None:
        """Load one private readiness record, rejecting malformed durable state."""
        path = self._state_root / name
        if not path.exists():
            return None
        try:
            if path.is_symlink() or path.stat().st_mode & 0o077:
                raise ValueError
            readiness = SandboxImageReadiness.model_validate_json(path.read_bytes())
        except (OSError, ValueError) as error:
            raise RuntimeError("Cached sandbox image readiness is invalid.") from error
        return InstalledSandboxImage(readiness.image, readiness, path)

    def _image_is_ready(self, readiness: SandboxImageReadiness) -> bool:
        """Reinspect one recorded image and verify every local readiness property."""
        inspected = self._control_plane.inspect_image(readiness.image.reference)
        if inspected.exit_code or inspected.stdout_truncated or inspected.stderr_truncated:
            return False
        try:
            identity = parse_image_evidence(inspected.stdout, readiness.image.platform)
        except InstallError:
            # The pinned nerdctl release can report success with empty output when a
            # digest-pinned image is absent. Treat unsupported inspection output as a
            # cache miss; a rebuilt image still receives complete strict attestation.
            return False
        if identity != readiness.image.identity:
            raise RuntimeError("Cached sandbox image identity changed.")
        if not image_has_safe_defaults(inspected.stdout, readiness.image.platform):
            raise RuntimeError("Cached sandbox image defaults are unsafe.")
        return True

    def _build(self) -> InstalledSandboxImage:
        """Run one bounded transient build and validate its complete result."""
        with tempfile.TemporaryDirectory(prefix="sandbox-source-", dir=self._state_root) as temp:
            archive = Path(temp) / "context.tar"
            self._write_context(archive)
            tag = (
                "loop.local/sandbox:" + self._definition.source_version.removeprefix("sha256:")[:24]
            )
            result = self._control_plane.build_sandbox_image(
                archive,
                tag,
                self._definition.base_image,
                self._definition.debian_snapshot,
                self._definition.package_arguments,
                self._definition.tool_arguments,
                tuple(sorted(self._definition.commands)),
            )
        if result.exit_code or result.stdout_truncated or result.stderr_truncated:
            _LOGGER.error(
                "Managed sandbox image build failed: exit=%d stderr=%r",
                result.exit_code,
                result.stderr[-8192:],
            )
            raise RuntimeError("Managed sandbox image build failed.")
        inspected, inventory, cleanup = _parse_build_output(result.stdout)
        if cleanup != "complete":
            raise RuntimeError("Managed sandbox image builder cleanup was incomplete.")
        identity = parse_image_evidence(inspected, result.platform)
        if not image_has_safe_defaults(inspected, result.platform):
            raise RuntimeError("Managed sandbox image defaults are unsafe.")
        _verify_inventory(self._definition, inventory)
        image = RuntimeImage(
            reference=f"loop.local/sandbox@{identity.index_digest}",
            platform=result.platform,
            identity=identity,
        )
        readiness = SandboxImageReadiness(
            schema_version=1,
            source_version=self._definition.source_version,
            image=image,
        )
        return InstalledSandboxImage(image, readiness, self._state_root / "active.json")

    def _activate(
        self,
        readiness: SandboxImageReadiness,
        previous: SandboxImageReadiness | None,
    ) -> None:
        """Atomically activate one image while retaining at most one rollback image."""
        rollback = self._load_record("rollback.json")
        retained_previous = previous is not None and previous.image != readiness.image
        if rollback is not None and (
            previous is None or rollback.image not in {previous.image, readiness.image}
        ):
            removed = self._control_plane.remove_image(rollback.image.reference)
            if removed.exit_code or removed.stdout_truncated or removed.stderr_truncated:
                raise RuntimeError("Obsolete sandbox image could not be reclaimed.")
        if retained_previous:
            _atomic_private_write(
                self._state_root / "rollback.json",
                json_encode(previous.model_dump(mode="json")).encode() + b"\n",
            )
        elif rollback is not None and rollback.image == readiness.image:
            (self._state_root / "rollback.json").unlink()
        _atomic_private_write(
            self._state_root / "active.json",
            json_encode(readiness.model_dump(mode="json")).encode() + b"\n",
        )

    def _write_context(self, destination: Path) -> None:
        """Create the minimal build context exclusively from installed package data."""
        inventory = json_encode(self._definition.model_dump(mode="json")).encode() + b"\n"
        with tarfile.open(destination, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, content in (
                ("Containerfile", load_sandbox_containerfile()),
                ("inventory.json", inventory),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.mode = 0o444
                info.mtime = 0
                archive.addfile(info, fileobj=io.BytesIO(content))
        destination.chmod(0o600)


def _parse_build_output(raw: bytes) -> tuple[bytes, bytes, str]:
    """Split the fixed inspection, inventory, and cleanup output frames."""
    inventory_marker = b"\n--LOOP-INVENTORY--\n"
    cleanup_marker = b"\n--LOOP-CLEANUP--\n"
    try:
        inspected, remainder = raw.split(inventory_marker, 1)
        inventory, cleanup = remainder.split(cleanup_marker, 1)
        cleanup_value = cleanup.decode("ascii").strip()
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("Managed sandbox image build evidence is malformed.") from error
    if not inspected or not inventory:
        raise RuntimeError("Managed sandbox image build evidence is incomplete.")
    return inspected, inventory, cleanup_value


def _verify_inventory(definition: SandboxImageDefinition, inventory: bytes) -> None:
    """Require exact package versions, commands, identity, and executable search path."""
    try:
        lines = inventory.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise RuntimeError("Sandbox image inventory is malformed.") from error
    packages: dict[str, str] = {}
    commands: dict[str, str] = {}
    identity: tuple[str, str, str] | None = None
    for line in lines:
        fields = line.split("\t")
        if len(fields) == 4 and fields[0] == "IDENTITY":
            identity = (fields[1], fields[2], fields[3])
        elif len(fields) == 3 and fields[0] == "PACKAGE":
            name = fields[1].partition(":")[0]
            if name in packages and packages[name] != fields[2]:
                raise RuntimeError("Sandbox image inventory is malformed.")
            packages[name] = fields[2]
        elif len(fields) == 3 and fields[0] == "COMMAND":
            commands[fields[1]] = fields[2]
        else:
            raise RuntimeError("Sandbox image inventory is malformed.")
    expected_identity = (
        str(definition.environment.uid),
        str(definition.environment.gid),
        definition.environment.path,
    )
    if identity != expected_identity:
        raise RuntimeError("Sandbox image process identity differs from its definition.")
    package_mismatches = sorted(
        name for name, version in definition.packages.items() if packages.get(name) != version
    )
    if package_mismatches:
        raise RuntimeError(
            "Sandbox image package inventory differs for: " + ", ".join(package_mismatches)
        )
    command_mismatches = sorted(
        name for name, path in definition.commands.items() if commands.get(name) != path
    )
    if command_mismatches:
        raise RuntimeError(
            "Sandbox image command inventory differs for: " + ", ".join(command_mismatches)
        )


def _atomic_private_write(path: Path, content: bytes) -> None:
    """Replace one private readiness file only after durable complete writing."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
