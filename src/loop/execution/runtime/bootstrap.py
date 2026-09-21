"""Coordinate selection, verified installation, activation, and leasing."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock

from ... import constants
from ...utils import sha256_digest
from .activation import activate, active_set, rollback, rollback_set
from .download import AcquisitionCancelled, ArtifactTransport, QuotaExceeded, download_artifact
from .gc import collect
from .install import ArtifactInstaller, UnsupportedArtifactProvider, install_content, verify_content
from .lease import RuntimeLease, active_leases, create_lease
from .manifest import RuntimeManifest
from .models import AcquisitionKind, Artifact, PlatformSelector


@dataclass(frozen=True, slots=True)
class RuntimeRequirement:
    """Describe exact platform and capabilities needed by one runtime user.

    Args:
        platform (PlatformSelector): Exact target platform.
        capabilities (frozenset[str]): Capabilities required by the user.
    """

    platform: PlatformSelector
    capabilities: frozenset[str]


@dataclass(frozen=True, slots=True)
class InstalledRuntime:
    """Expose immutable installed artifact locations plus their durable lease.

    Args:
        manifest_digest (str): Digest of the selected manifest.
        artifact_set_digest (str): Digest of the selected artifact set.
        artifacts (Mapping[str, Path]): Artifact identifiers mapped to installed locations.
        declared_artifacts (Mapping[str, Artifact]): Manifest records for installed artifacts.
        root (Path): Resolved private runtime root which owns the artifact locations.
        lease (RuntimeLease): Durable lease protecting the artifact set.
    """

    manifest_digest: str
    artifact_set_digest: str
    artifacts: Mapping[str, Path]
    lease: RuntimeLease
    declared_artifacts: Mapping[str, Artifact]
    root: Path

    def executable(self, artifact_id: str, relative_path: str) -> InstalledExecutable:
        """Return one verified executable from this leased runtime.

        Args:
            artifact_id (str): Identifier of the manifest-selected artifact.
            relative_path (str): Normalized path declared executable by that artifact.

        Returns:
            InstalledExecutable: Immutable executable identity for infrastructure use.

        Raises:
            ValueError: If the artifact or executable is not part of this runtime.
        """
        root = self.artifacts.get(artifact_id)
        artifact = self.declared_artifacts.get(artifact_id)
        relative = Path(relative_path)
        if (
            root is None
            or artifact is None
            or artifact.layout is None
            or not relative_path
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in relative_path
        ):
            raise ValueError("Executable is not part of the installed runtime.")
        target = root / relative
        try:
            target.relative_to(root)
            metadata = target.lstat()
        except (FileNotFoundError, ValueError) as error:
            raise ValueError("Executable is not part of the installed runtime.") from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or relative_path not in artifact.layout.executables
        ):
            raise ValueError("Installed executable must be a regular file.")
        if not metadata.st_mode & stat.S_IXUSR:
            raise ValueError("Installed executable is not executable.")
        return InstalledExecutable(
            artifact_id=artifact_id,
            artifact_set_digest=self.artifact_set_digest,
            path=target,
            runtime_root=self.root,
            artifact_root=root,
            lease=self.lease,
            expected_identity=artifact.layout.identities[relative_path],
            path_directories=tuple(
                sorted(
                    {
                        str(root / Path(candidate).parent)
                        for candidate in artifact.layout.executables
                    }
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class InstalledExecutable:
    """Bind a management executable to a verified runtime artifact set.

    Args:
        artifact_id (str): Manifest artifact that supplied the executable.
        artifact_set_digest (str): Leased immutable artifact-set identity.
        path (Path): Verified private absolute executable path.
        runtime_root (Path): Resolved private runtime root for the active lease.
        artifact_root (Path): Content-addressed artifact directory containing ``path``.
        lease (RuntimeLease): Active lease authorizing use of this artifact set.
        expected_identity (str): Manifest-declared SHA-256 identity of ``path``.
        path_directories (tuple[str, ...]): Declared bundle directories permitted in child PATH.
    """

    artifact_id: str
    artifact_set_digest: str
    path: Path
    runtime_root: Path
    artifact_root: Path
    lease: RuntimeLease
    expected_identity: str
    path_directories: tuple[str, ...]


class RuntimeBootstrapper:
    """Ensure the minimal closed manifest selection is installed without process launch.

    Args:
        manifest (RuntimeManifest): Validated runtime manifest to select from.
        root (Path): Private runtime storage root.
        transport (ArtifactTransport): Transport used for file acquisitions.
        installers (Mapping[AcquisitionKind, ArtifactInstaller]): Installers by acquisition kind.
        download_limit (int): Maximum bytes permitted for one download.
    """

    _manifest: RuntimeManifest
    _root: Path
    _root_identity: tuple[int, int]
    _transport: ArtifactTransport
    _installers: dict[AcquisitionKind, ArtifactInstaller]
    _download_limit: int
    _cache_limit: int

    def __init__(
        self,
        manifest: RuntimeManifest,
        root: Path,
        transport: ArtifactTransport,
        installers: Mapping[AcquisitionKind, ArtifactInstaller],
        download_limit: int,
        cache_limit: int = constants.RUNTIME_DEFAULT_CACHE_BYTES,
    ) -> None:
        if root.is_symlink():
            raise ValueError("Runtime root must not be a symbolic link.")
        root.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
        self._manifest = manifest
        self._root = root.resolve()
        root_stat = self._root.stat()
        self._root_identity = (root_stat.st_dev, root_stat.st_ino)
        self._transport = transport
        self._installers = dict(installers)
        self._download_limit = download_limit
        if download_limit <= 0 or cache_limit <= 0:
            raise ValueError("Runtime download and cache quotas must be positive.")
        self._cache_limit = cache_limit

    @property
    def root(self) -> Path:
        """Return the resolved private runtime root.

        Returns:
            Path: Runtime storage root.
        """
        self._validate_root()
        return self._root

    def _validate_root(self) -> None:
        """Reject replacement of the application-private runtime root."""
        try:
            current = self._root.lstat()
        except FileNotFoundError as error:
            raise RuntimeError("Runtime root identity changed.") from error
        if self._root.is_symlink() or (current.st_dev, current.st_ino) != self._root_identity:
            raise RuntimeError("Runtime root identity changed.")

    def status(self) -> tuple[dict[str, str], ...]:
        """Return sanitized stable status rows for manifest artifacts.

        Returns:
            tuple[dict[str, str], ...]: Artifact identity, size, and lifecycle rows.
        """
        self._validate_root()
        active = active_set(self._root)
        retained = rollback_set(self._root)
        leased = active_leases(self._root)
        rows: list[dict[str, str]] = []
        for artifact in self._manifest.artifacts:
            content = (
                self._root
                / constants.RUNTIME_ARTIFACTS_DIRECTORY
                / artifact.artifact_id
                / artifact.digest.removeprefix(constants.SHA256_PREFIX)
            )
            rows.append(
                {
                    "artifact": artifact.artifact_id,
                    "version": artifact.version,
                    "digest": artifact.digest,
                    "size": "oci" if artifact.size is None else str(artifact.size),
                    "state": "installed" if verify_content(content, artifact) else "missing",
                }
            )
        if active is not None:
            rows.append(
                {
                    "artifact": "active-set",
                    "version": "-",
                    "digest": active,
                    "size": "-",
                    "state": "leased" if active in leased else "active",
                }
            )
        if retained is not None:
            rows.append(
                {
                    "artifact": "rollback-set",
                    "version": "-",
                    "digest": retained,
                    "size": "-",
                    "state": "retained",
                }
            )
        return tuple(rows)

    def ensure(
        self,
        requirement: RuntimeRequirement,
        progress: Callable[[str], None] = lambda _: None,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InstalledRuntime:
        """Install and lease the minimal required artifact closure.

        Args:
            requirement (RuntimeRequirement): Platform and capabilities to satisfy.
            progress (Callable[[str], None]): Callback for installation progress messages.
            cancellation (Callable[[], bool]): Predicate checked before and during acquisition.

        Returns:
            InstalledRuntime: Installed artifact locations and their durable lease.

        Raises:
            UnsupportedArtifactProvider: If a required acquisition provider is absent.
            AcquisitionCancelled: If cancellation is requested during bootstrap.
        """
        self._validate_root()
        artifacts = self._manifest.select(requirement.platform, requirement.capabilities)
        locations: dict[str, Path] = {}
        for artifact in artifacts:
            if cancellation():
                raise AcquisitionCancelled("Runtime bootstrap was cancelled.")
            installer = self._installers.get(artifact.acquisition)
            if installer is None:
                raise UnsupportedArtifactProvider(
                    f"No installer for {artifact.acquisition} artifacts."
                )
            destination = self._root / constants.RUNTIME_ARTIFACTS_DIRECTORY
            content = (
                destination
                / artifact.artifact_id
                / artifact.digest.removeprefix(constants.SHA256_PREFIX)
            )
            if verify_content(content, artifact):
                locations[artifact.artifact_id] = content
                continue
            progress(f"Installing {artifact.artifact_id}@{artifact.version}")
            self._reserve_cache(artifact.size or 0)
            source = None
            if artifact.acquisition is not AcquisitionKind.OCI:
                source = download_artifact(
                    artifact,
                    self._root / constants.RUNTIME_DOWNLOADS_DIRECTORY,
                    self._transport,
                    self._download_limit,
                    cancellation,
                )
            try:
                locations[artifact.artifact_id] = install_content(
                    artifact, source, destination, installer
                )
            finally:
                if source is not None:
                    source.unlink(missing_ok=True)
        artifact_set_digest = sha256_digest("\n".join(sorted(a.digest for a in artifacts)))
        self._record_artifact_set(artifact_set_digest, artifacts)
        lease = create_lease(
            self._root,
            self._manifest.digest,
            artifact_set_digest,
            lifetime=constants.RUNTIME_DEFAULT_LEASE_SECONDS,
        )
        return InstalledRuntime(
            self._manifest.digest,
            artifact_set_digest,
            locations,
            lease,
            {artifact.artifact_id: artifact for artifact in artifacts},
            self._root,
        )

    def _reserve_cache(self, incoming: int) -> None:
        """Collect inactive content and fail if the cache quota cannot fit an artifact."""

        def usage() -> int:
            return sum(
                path.stat().st_size
                for path in (self._root / constants.RUNTIME_ARTIFACTS_DIRECTORY).rglob("*")
                if path.is_file() and not path.is_symlink()
            )

        if usage() + incoming <= self._cache_limit:
            return
        collect(self._root, reclaim_bytes=usage() + incoming - self._cache_limit)
        if usage() + incoming > self._cache_limit:
            raise QuotaExceeded("Runtime cache quota is exhausted by protected content.")

    def activate(self, runtime: InstalledRuntime) -> str | None:
        """Activate an installed runtime after its caller has attested it.

        Args:
            runtime (InstalledRuntime): Leased runtime returned by this bootstrapper.

        Returns:
            str | None: Previously active artifact-set digest.

        Raises:
            ValueError: If the runtime was not produced from this manifest or is incomplete.
            ActivationError: If activation loses a concurrent compare-and-swap.
        """
        self._validate_root()
        if runtime.manifest_digest != self._manifest.digest:
            raise ValueError("Installed runtime belongs to a different manifest.")
        record = (
            self._root / constants.RUNTIME_SETS_DIRECTORY / f"{runtime.artifact_set_digest}.json"
        )
        if not record.is_file():
            raise ValueError("Installed runtime has no verified artifact-set record.")
        current = active_set(self._root)
        if current == runtime.artifact_set_digest:
            return current
        return activate(self._root, runtime.artifact_set_digest, current)

    def rollback(self) -> str:
        """Reactivate the retained verified set without acquisition.

        Returns:
            str: Newly active artifact-set digest.

        Raises:
            ActivationError: If no retained verified set is available.
        """
        self._validate_root()
        return rollback(self._root)

    def _record_artifact_set(
        self,
        artifact_set_digest: str,
        artifacts: tuple[Artifact, ...],
    ) -> None:
        """Durably record the verified immutable members of one artifact set."""
        sets = self._root / constants.RUNTIME_SETS_DIRECTORY
        sets.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
        destination = sets / f"{artifact_set_digest}.json"
        payload = json.dumps(
            {
                "manifest_digest": self._manifest.digest,
                "artifact_digests": sorted(
                    artifact.digest.removeprefix(constants.SHA256_PREFIX) for artifact in artifacts
                ),
            },
            sort_keys=True,
        )
        with FileLock(str(sets / ".sets.lock")):
            if destination.exists():
                return
            temporary = destination.with_suffix(f".{os.getpid()}.new")
            with temporary.open("x", encoding="ascii") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            descriptor = os.open(sets, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
