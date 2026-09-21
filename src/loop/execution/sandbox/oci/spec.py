"""Compile authorized sandbox requests into a closed OCI execution specification."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .... import constants
from ...contracts import (
    Capability,
    ExecutionMode,
    SandboxExecutionRequest,
    ShellExecutionRequest,
    TerminalMode,
)
from ...runtime.image import RuntimeImage
from ...runtime.models import Artifact, OciImageIdentity

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_LABEL_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_UNSUPPORTED_CAPABILITIES = frozenset(
    {
        Capability.IPC_CONNECT,
        Capability.CACHE_WRITE,
    }
)


class OciSpecificationError(ValueError):
    """Report a request that cannot become a confined OCI specification."""


@dataclass(frozen=True, slots=True)
class OciResourceLimits:
    """Describe mandatory container resource controls.

    Args:
        memory_bytes (int): Positive memory limit enforced by the runtime.
        pids (int): Positive process-count limit enforced by the runtime.
        cpu_quota_us (int): Positive CPU quota in microseconds for one period.
        open_files (int): Positive per-process open-file descriptor limit.
        disk_bytes (int): Positive private writable ``/var/tmp`` filesystem limit; the image root
            remains read-only.
        temporary_bytes (int): Positive private ``/tmp`` tmpfs limit.
        cache_bytes (int): Positive private ``/cache`` tmpfs limit.
        persistent_write_bytes (int): Positive attempt-overlay write limit.
    """

    memory_bytes: int
    pids: int
    cpu_quota_us: int
    open_files: int = constants.DEFAULT_OCI_OPEN_FILES
    disk_bytes: int = constants.DEFAULT_OCI_DISK_BYTES
    temporary_bytes: int = constants.DEFAULT_OCI_TEMPORARY_BYTES
    cache_bytes: int = constants.DEFAULT_OCI_CACHE_BYTES
    persistent_write_bytes: int = constants.DEFAULT_OCI_PERSISTENT_WRITE_BYTES

    def __post_init__(self) -> None:
        """Reject absent mandatory resource limits."""
        if (
            min(
                self.memory_bytes,
                self.pids,
                self.cpu_quota_us,
                self.open_files,
                self.disk_bytes,
                self.temporary_bytes,
                self.cache_bytes,
                self.persistent_write_bytes,
            )
            <= 0
        ):
            raise OciSpecificationError("OCI resource controls must be positive.")


@dataclass(frozen=True, slots=True)
class OciMount:
    """Name one trusted logical mount without disclosing a host path.

    Args:
        source_id (str): Opaque immutable workspace-generation identity.
        destination (str): Absolute container path.
        read_only (bool): Whether the container may write the mount.
    """

    source_id: str
    destination: str
    read_only: bool


@dataclass(frozen=True, slots=True)
class OciExecutionSpec:
    """Represent one reviewed fixed-shape container invocation.

    This record contains no host executable, socket, or source filesystem path. A
    platform adapter resolves ``workspace_mount.source_id`` only after it has
    verified the leased immutable generation.

    Args:
        request_id (str): Attempt identity.
        container_name (str): Private deterministic container identity.
        image_reference (str): Immutable registry-qualified profile reference.
        image_digest (str): Immutable selected profile digest.
        image_manifest_digest (str): Selected platform manifest digest.
        image_config_digest (str): Selected platform config digest.
        argv (tuple[str, ...]): Exact guest entrypoint and arguments.
        cwd (str): Absolute guest working directory.
        environment (tuple[tuple[str, str], ...]): Sorted explicit guest environment.
        labels (tuple[tuple[str, str], ...]): Sorted ownership labels.
        workspace_mount (OciMount): Trusted logical workspace mount.
        limits (OciResourceLimits): Mandatory container resource controls.
        network_mode (str): Closed network mode for this attempt.
        detached (bool): Whether durable-job ownership is explicit.
        attach_stdin (bool): Whether bounded foreground input requires an attached pipe.
        terminal (TerminalMode): Separated pipes or one merged pseudo-terminal.
        terminal_columns (int | None): Initial PTY width.
        terminal_rows (int | None): Initial PTY height.
    """

    request_id: str
    container_name: str
    image_reference: str
    image_digest: str
    image_manifest_digest: str
    image_config_digest: str
    argv: tuple[str, ...]
    cwd: str
    environment: tuple[tuple[str, str], ...]
    labels: tuple[tuple[str, str], ...]
    workspace_mount: OciMount
    limits: OciResourceLimits
    network_mode: str
    detached: bool
    attach_stdin: bool = False
    terminal: TerminalMode = TerminalMode.PIPE
    terminal_columns: int | None = None
    terminal_rows: int | None = None


class OciSpecCompiler:
    """Compile only leased virtual requests for one immutable OCI profile.

    Args:
        image (Artifact | RuntimeImage): Selected immutable OCI profile identity.
        limits (OciResourceLimits): Mandatory resource controls for every compiled attempt.
    """

    _image: Artifact | RuntimeImage
    _identity: OciImageIdentity
    _limits: OciResourceLimits

    def __init__(self, image: Artifact | RuntimeImage, limits: OciResourceLimits) -> None:
        identity = image.oci_identity if isinstance(image, Artifact) else image.identity
        if identity is None:
            raise OciSpecificationError("OCI execution requires a fully pinned sandbox image.")
        self._image = image
        self._identity = identity
        self._limits = limits

    def compile(
        self, request: SandboxExecutionRequest, workspace_generation: str
    ) -> OciExecutionSpec:
        """Compile one authorized virtual request without host execution.

        Args:
            request: Authorized sandbox request.
            workspace_generation: Opaque immutable workspace-generation identity.

        Returns:
            OciExecutionSpec: Closed container settings for later typed execution.

        Raises:
            OciSpecificationError: If a capability, virtual path, or value is unsafe.
        """
        if not workspace_generation or not _LABEL_VALUE.fullmatch(workspace_generation):
            raise OciSpecificationError("Workspace generation identity is invalid.")
        identity = self._identity
        if request.lease.runtime_digest != identity.manifest_digest:
            raise OciSpecificationError("Execution lease does not match the selected OCI profile.")
        if (
            bool(request.network_connections)
            != (Capability.NETWORK_CONNECT in request.lease.capabilities)
            or bool(request.network_listeners)
            != (Capability.NETWORK_LISTEN in request.lease.capabilities)
            or bool(request.secret_exposures)
            != (Capability.SECRET_USE in request.lease.capabilities)
        ):
            raise OciSpecificationError(
                "OCI capability is unavailable without its exact broker lease."
            )
        unsupported = request.lease.capabilities & _UNSUPPORTED_CAPABILITIES
        if unsupported:
            raise OciSpecificationError(f"OCI capability is unsupported: {min(unsupported).value}.")
        environment = _environment(request.environment)
        writable = Capability.WORKSPACE_WRITE in request.lease.capabilities
        argv = (
            ("/bin/sh", "-c", request.script)
            if isinstance(request, ShellExecutionRequest)
            else request.argv
        )
        if not argv or not all(argument and "\x00" not in argument for argument in argv):
            raise OciSpecificationError("OCI command arguments must be nonempty guest strings.")
        labels = tuple(
            sorted(
                {
                    "io.loop.agent": request.lease.agent_run_id,
                    "io.loop.lease": request.lease.lease_id,
                    "io.loop.request": request.request_id,
                    "io.loop.workspace": request.lease.workspace_id,
                }.items()
            )
        )
        if any(not _LABEL_VALUE.fullmatch(value) for _, value in labels):
            raise OciSpecificationError("OCI ownership identities contain unsafe label values.")
        return OciExecutionSpec(
            request.request_id,
            f"loop-{request.request_id}",
            self._image.source if isinstance(self._image, Artifact) else self._image.reference,
            identity.index_digest,
            identity.manifest_digest,
            identity.config_digest,
            argv,
            request.cwd,
            environment,
            labels,
            OciMount(workspace_generation, "/workspace", not writable),
            self._limits,
            "none",
            request.mode is ExecutionMode.DURABLE_JOB,
            bool(request.stdin),
            request.terminal,
            request.terminal_columns,
            request.terminal_rows,
        )


def _environment(entries: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    """Validate and canonically order the explicit guest environment."""
    values = {
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/agent",
        "PATH": "/tools/bin:/usr/local/bin:/usr/bin:/bin",
        "TMPDIR": "/tmp",
        "XDG_CONFIG_HOME": "/tmp/.config",
    }
    for name, value in entries:
        if not _ENVIRONMENT_NAME.fullmatch(name) or "\x00" in value or name in values:
            raise OciSpecificationError(
                "OCI environment must contain unique safe names and values."
            )
        values[name] = value
    return tuple(sorted(values.items()))
