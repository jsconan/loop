"""Compose the managed macOS runtime into one fail-closed substrate boundary."""

from __future__ import annotations

import os
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock

from .... import constants
from ....utils import sha256_digest
from ...infrastructure import InfrastructureProcessRunner
from ...runtime.bootstrap import InstalledRuntime, RuntimeBootstrapper, RuntimeRequirement
from ...runtime.image import RuntimeImage
from ...runtime.manifest import RuntimeManifest
from ...runtime.models import PlatformSelector
from ..oci.control import GuestOciRuntimeEndpoint
from .control_plane import MacosControlPlane, MacosRuntimeNotReadyError
from .images import InstalledSandboxImage, MacosSandboxImageBuilder
from .lima import (
    LimaClient,
    LimaConfigurationError,
    LimaInstanceContext,
    LimaLifecycleState,
    ManagedLimaInstance,
    build_lima_instance_configuration,
)
from .requirements import MacosRequirementEvidence, MacosRequirements

_RUNTIME_READINESS_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class PreparedMacosRuntime:
    """Expose one attested warm runtime and its active immutable lease.

    Args:
        runtime (InstalledRuntime): Lease-bound installed Lima runtime.
        instance (ManagedLimaInstance): Running journal-owned Lima instance.
        control_plane (MacosControlPlane): Attested rootless guest OCI control plane.
        endpoint (GuestOciRuntimeEndpoint): Exact attested guest runtime endpoint.
        requirements (MacosRequirementEvidence): Sanitized supported-host evidence.
        sandbox_image (InstalledSandboxImage): Locally built and readiness-checked command image.
        snapshot_store (Path): Sole read-only host directory shared with the guest.
    """

    runtime: InstalledRuntime
    instance: ManagedLimaInstance
    control_plane: MacosControlPlane
    endpoint: GuestOciRuntimeEndpoint
    requirements: MacosRequirementEvidence
    sandbox_image: InstalledSandboxImage
    snapshot_store: Path

    @property
    def image_reference(self) -> str:
        """Return the digest-pinned local core image reference.

        Returns:
            str: Immutable local core OCI reference.
        """
        return self.sandbox_image.image.reference

    def image_for_digest(self, runtime_digest: str) -> RuntimeImage:
        """Return an already prepared image matching one authorized runtime digest.

        Args:
            runtime_digest (str): Selected platform-manifest digest from the execution lease.

        Returns:
            RuntimeImage: Matching immutable local OCI image.

        Raises:
            ValueError: If no prepared profile has this exact descriptor identity.
        """
        if self.sandbox_image.image.identity.manifest_digest == runtime_digest:
            return self.sandbox_image.image
        raise ValueError("Execution lease does not match the prepared sandbox image.")

    def stop(self) -> None:
        """Stop the owned VM while retaining its private disk and runtime content."""
        self.instance.stop()

    def delete(self) -> None:
        """Delete the owned VM and release the installed-runtime lease."""
        try:
            self.instance.delete()
        finally:
            self.runtime.lease.close()

    def close(self) -> None:
        """Release this caller's runtime lease while leaving the warm VM running."""
        self.runtime.lease.close()

    def renew(self, lifetime_seconds: float) -> None:
        """Extend the runtime lease through one authorized operation deadline.

        Args:
            lifetime_seconds (float): Positive wall-clock lifetime from now.

        Raises:
            ValueError: If the requested lifetime is not positive.
        """
        if lifetime_seconds <= 0:
            raise ValueError("Managed runtime renewal lifetime must be positive.")
        self.runtime.lease.heartbeat(time.time() + lifetime_seconds)


class MacosSandboxBackend:
    """Automatically prepare one workspace-bound Lima and rootless OCI substrate.

    Args:
        candidate (RuntimeManifest): Authoritative checked-in or embedded runtime manifest.
        bootstrapper (RuntimeBootstrapper): Verified private runtime installer.
        process (InfrastructureProcessRunner): Classified fixed-argv host process boundary.
        application_data (Path): Loop-private root for managed VM state.
        workspace_id (str): Authenticated durable workspace identity.
        state_root (Path | None): Short private Lima control root. Defaults to
            ``application_data`` when that path already satisfies platform socket limits.
        minimum_macos_major (int): Minimum supported host macOS major version.
        minimum_free_bytes (int): Required free capacity below private application state.
        progress (Callable[[str], None]): User-visible sandbox preparation status callback.
    """

    _candidate: RuntimeManifest
    _bootstrapper: RuntimeBootstrapper
    _application_data: Path
    _application_identity: tuple[int, int]
    _state_identity: tuple[int, int]
    _workspace_digest: str
    _workspace_key: str
    _instance_name: str
    _state_root: Path
    _snapshot_store: Path
    _snapshot_identity: tuple[int, int]
    _client: LimaClient
    _requirements: MacosRequirements
    _progress: Callable[[str], None]

    def __init__(
        self,
        candidate: RuntimeManifest,
        bootstrapper: RuntimeBootstrapper,
        process: InfrastructureProcessRunner,
        application_data: Path,
        workspace_id: str,
        *,
        state_root: Path | None = None,
        minimum_macos_major: int = 13,
        minimum_free_bytes: int = 4 * 2**30,
        progress: Callable[[str], None] = lambda _: None,
    ) -> None:
        if not workspace_id or "\0" in workspace_id:
            raise ValueError("Managed macOS runtime requires a workspace identity.")
        application_data.mkdir(mode=0o700, parents=True, exist_ok=True)
        resolved_application_data = application_data.resolve()
        working_directory = Path.cwd().resolve()
        home_directory = Path.home().resolve()
        if (
            application_data.is_symlink()
            or not application_data.is_dir()
            or resolved_application_data in {working_directory, home_directory}
            or resolved_application_data in working_directory.parents
            or working_directory in resolved_application_data.parents
        ):
            raise ValueError("Managed macOS runtime requires private application state.")
        resolved_application_data.chmod(0o700)
        application_metadata = resolved_application_data.stat()
        self._candidate = candidate
        self._bootstrapper = bootstrapper
        self._application_data = resolved_application_data
        self._application_identity = (
            application_metadata.st_dev,
            application_metadata.st_ino,
        )
        self._workspace_digest = sha256_digest(workspace_id.encode())
        self._workspace_key = self._workspace_digest[:12]
        selected_state_root = application_data if state_root is None else state_root
        selected_state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._state_root = selected_state_root.resolve()
        state_metadata = selected_state_root.lstat()
        if (
            selected_state_root.is_symlink()
            or not stat.S_ISDIR(state_metadata.st_mode)
            or state_metadata.st_uid != os.getuid()
            or self._state_root in {working_directory, home_directory}
            or self._state_root in working_directory.parents
            or working_directory in self._state_root.parents
        ):
            raise ValueError("Managed macOS runtime requires private short control state.")
        self._state_root.chmod(0o700)
        state_metadata = self._state_root.stat()
        self._state_identity = (state_metadata.st_dev, state_metadata.st_ino)
        instance_key = sha256_digest(f"{self._workspace_digest}\0{self._state_root}")[:12]
        self._instance_name = f"loop-{instance_key}"
        self._snapshot_store = self._application_data / "snapshots" / self._workspace_key
        self._snapshot_store.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._snapshot_store.chmod(0o700)
        snapshot_metadata = self._snapshot_store.stat()
        self._snapshot_identity = (snapshot_metadata.st_dev, snapshot_metadata.st_ino)
        self._bind_workspace()
        self._client = LimaClient(process, self._application_data)
        self._requirements = MacosRequirements.from_candidate(
            self._client,
            candidate,
            minimum_macos_major,
            minimum_free_bytes,
        )
        self._progress = progress

    @property
    def snapshot_store(self) -> Path:
        """Return the workspace-specific immutable-generation store shared read-only.

        Returns:
            Path: Resolved Loop-private snapshot-store root.
        """
        self._validate_private_state()
        return self._snapshot_store

    def prepare(self) -> PreparedMacosRuntime:
        """Install, start, and attest the complete managed macOS substrate.

        Returns:
            PreparedMacosRuntime: Lease-bound running VM, OCI endpoint, and pinned image.

        Raises:
            MacosRequirementError: If the host or installed Lima signature is unsupported.
            LimaConfigurationError: If private state or managed lifecycle evidence is invalid.
            InstallError: If the pinned runtime or core image cannot be verified.
        """

        def report(message: str) -> None:
            """Forward one preparation transition to the product interaction."""
            self._progress(message)

        runtime = self._bootstrapper.ensure(
            RuntimeRequirement(
                platform=PlatformSelector(os="macos", architecture="arm64"),
                capabilities=frozenset({"sandbox-management"}),
            ),
            progress=report,
        )
        runtime.lease.heartbeat(time.time() + constants.RUNTIME_PREPARATION_LEASE_SECONDS)
        created = False
        instance: ManagedLimaInstance | None = None
        try:
            self._validate_private_state()
            executable = runtime.executable("lima", "bin/limactl")
            evidence = self._requirements.verify(executable)
            configuration = build_lima_instance_configuration(
                self._candidate,
                self._instance_name,
                self._snapshot_store,
            )
            context = LimaInstanceContext.create(
                executable,
                self._state_root,
                configuration,
                1,
            )
            instance = ManagedLimaInstance(self._client, context, configuration)
            with FileLock(str(self._state_root / f".{self._instance_name}.prepare.lock")):
                recovered = False
                while True:
                    try:
                        state = instance.state
                    except LimaConfigurationError:
                        report("Repairing isolated command sandbox…")
                        instance.reset_obsolete()
                        recovered = True
                        state = LimaLifecycleState.ABSENT
                    if state not in {
                        LimaLifecycleState.ABSENT,
                        LimaLifecycleState.STOPPED,
                        LimaLifecycleState.RUNNING,
                    }:
                        report("Repairing isolated command sandbox…")
                        instance.reset()
                        state = LimaLifecycleState.ABSENT
                    try:
                        if state is LimaLifecycleState.ABSENT:
                            report("Creating isolated command sandbox…")
                            instance.create()
                            created = True
                            state = LimaLifecycleState.STOPPED
                        if state is LimaLifecycleState.STOPPED:
                            report("Starting isolated command sandbox…")
                            instance.start()
                        else:
                            instance.attest_running()
                        break
                    except LimaConfigurationError:
                        if recovered or instance.state is not LimaLifecycleState.ABSENT:
                            raise
                        report("Repairing isolated command sandbox…")
                        recovered = True
                control_plane = MacosControlPlane(self._client, instance, context)
                endpoint = self._attest_runtime(control_plane)
                # The VM is now complete durable substrate. A later local image-build
                # failure must not discard it and force the next command through another
                # multi-minute create/boot cycle.
                created = False
                image_builder = MacosSandboxImageBuilder(
                    control_plane,
                    self._state_root / "sandbox-image",
                )
                report("Building sandbox command image…")
                sandbox_image = image_builder.prepare()
                self._bootstrapper.activate(runtime)
            self._progress("Isolated command sandbox is ready.")
            return PreparedMacosRuntime(
                runtime,
                instance,
                control_plane,
                endpoint,
                evidence,
                sandbox_image,
                self._snapshot_store,
            )
        except BaseException:
            if created and instance is not None:
                try:
                    instance.delete()
                except (RuntimeError, ValueError):
                    pass
            runtime.lease.close()
            raise

    def manage(self, *, delete: bool) -> bool:
        """Stop or delete an existing journal-owned sandbox without starting it.

        Args:
            delete (bool): Delete the VM and its private disk when ``True``; otherwise stop it.

        Returns:
            bool: Whether owned durable sandbox state existed.

        Raises:
            LimaConfigurationError: If ownership evidence or cleanup fails closed.
        """
        journal = self._state_root / "lima" / f"{self._instance_name}.loop.json"
        if not journal.exists():
            return False
        runtime = self._bootstrapper.ensure(
            RuntimeRequirement(
                platform=PlatformSelector(os="macos", architecture="arm64"),
                capabilities=frozenset({"sandbox-management"}),
            )
        )
        try:
            executable = runtime.executable("lima", "bin/limactl")
            self._requirements.verify(executable)
            configuration = build_lima_instance_configuration(
                self._candidate,
                self._instance_name,
                self._snapshot_store,
            )
            context = LimaInstanceContext.create(
                executable,
                self._state_root,
                configuration,
                1,
            )
            instance = ManagedLimaInstance(self._client, context, configuration)
            with FileLock(str(self._state_root / f".{self._instance_name}.prepare.lock")):
                if delete:
                    instance.reset()
                elif instance.state is LimaLifecycleState.RUNNING:
                    instance.stop(force=True)
                elif instance.state is not LimaLifecycleState.STOPPED:
                    instance.reset()
            return True
        finally:
            runtime.lease.close()

    def status(self) -> tuple[str, LimaLifecycleState] | None:
        """Return the attested state of the current workspace sandbox without starting it.

        Returns:
            tuple[str, LimaLifecycleState] | None: Owned instance name and state, or ``None`` when
            no sandbox ownership journal exists.

        Raises:
            LimaConfigurationError: If ownership evidence or live attestation fails closed.
        """
        journal = self._state_root / "lima" / f"{self._instance_name}.loop.json"
        if not journal.exists():
            return None
        runtime = self._bootstrapper.ensure(
            RuntimeRequirement(
                platform=PlatformSelector(os="macos", architecture="arm64"),
                capabilities=frozenset({"sandbox-management"}),
            )
        )
        try:
            executable = runtime.executable("lima", "bin/limactl")
            self._requirements.verify(executable)
            configuration = build_lima_instance_configuration(
                self._candidate,
                self._instance_name,
                self._snapshot_store,
            )
            context = LimaInstanceContext.create(
                executable,
                self._state_root,
                configuration,
                1,
            )
            instance = ManagedLimaInstance(self._client, context, configuration)
            with FileLock(str(self._state_root / f".{self._instance_name}.prepare.lock")):
                state = instance.state
                if state is LimaLifecycleState.RUNNING:
                    instance.attest_running()
            return self._instance_name, state
        finally:
            runtime.lease.close()

    def list_sandboxes(self) -> tuple[tuple[str, LimaLifecycleState], ...]:
        """List attested sandboxes owned by the current workspace.

        Returns:
            tuple[tuple[str, LimaLifecycleState], ...]: Zero or one owned instance records.

        Raises:
            LimaConfigurationError: If ownership evidence or live attestation fails closed.
        """
        status = self.status()
        return () if status is None else (status,)

    def _attest_runtime(self, control_plane: MacosControlPlane) -> GuestOciRuntimeEndpoint:
        """Wait briefly for rootless startup identity to converge, then fail closed."""
        deadline = time.monotonic() + _RUNTIME_READINESS_SECONDS
        waiting_reported = False
        while True:
            try:
                return control_plane.attest_runtime(self._candidate)
            except MacosRuntimeNotReadyError:
                if not waiting_reported:
                    self._progress("Waiting for isolated command sandbox services…")
                    waiting_reported = True
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    def _bind_workspace(self) -> None:
        """Bind a shortened Lima instance name to the full workspace identity digest."""
        binding = self._application_data / f".{self._instance_name}.workspace"
        lock = FileLock(str(binding.with_suffix(".lock")))
        with lock:
            if binding.exists():
                try:
                    if binding.is_symlink():
                        raise OSError
                    value = binding.read_text(encoding="ascii")
                except OSError as error:
                    raise ValueError("Managed macOS workspace binding is invalid.") from error
                if value != self._workspace_digest:
                    raise ValueError("Managed macOS workspace binding is invalid.")
                return
            try:
                with binding.open("x", encoding="ascii") as stream:
                    stream.write(self._workspace_digest)
                    stream.flush()
                    os.fsync(stream.fileno())
                descriptor = os.open(
                    self._application_data,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                )
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError as error:
                raise ValueError(
                    "Managed macOS workspace binding could not be persisted."
                ) from error

    def _validate_private_state(self) -> None:
        """Reject replacement or permission widening of private VM and snapshot roots."""
        try:
            application = self._application_data.lstat()
            state = self._state_root.lstat()
            snapshot = self._snapshot_store.lstat()
        except OSError as error:
            raise ValueError("Managed macOS private state identity changed.") from error
        if (
            stat.S_ISLNK(application.st_mode)
            or stat.S_ISLNK(state.st_mode)
            or stat.S_ISLNK(snapshot.st_mode)
            or not stat.S_ISDIR(application.st_mode)
            or not stat.S_ISDIR(state.st_mode)
            or not stat.S_ISDIR(snapshot.st_mode)
            or (application.st_dev, application.st_ino) != self._application_identity
            or (state.st_dev, state.st_ino) != self._state_identity
            or state.st_uid != os.getuid()
            or (snapshot.st_dev, snapshot.st_ino) != self._snapshot_identity
            or stat.S_IMODE(application.st_mode) != 0o700
            or stat.S_IMODE(state.st_mode) != 0o700
            or stat.S_IMODE(snapshot.st_mode) != 0o700
        ):
            raise ValueError("Managed macOS private state identity changed.")
