"""Own a closed, journaled Lima VZ lifecycle below Loop private state."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import stat
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from string import Template
from typing import Any

import yaml

from ....utils import sha256_digest
from ...infrastructure import (
    InfrastructureProcessCommand,
    InfrastructureProcessResult,
    InfrastructureProcessRunner,
    sealed_environment,
)
from ...runtime.bootstrap import InstalledExecutable, RuntimeBootstrapper, RuntimeRequirement
from ...runtime.manifest import RuntimeManifest
from ...runtime.models import PlatformSelector
from ..oci.process import OciProcessSession
from .candidate import macos_artifact

_INSTANCE_NAME = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
_LOGGER = logging.getLogger(__name__)
_LIST_FIELDS = frozenset(
    {
        "additionalDisks",
        "arch",
        "autoStartedIdentifier",
        "config",
        "cpus",
        "dir",
        "disk",
        "driverPID",
        "errors",
        "hostAgentPID",
        "hostname",
        "limaVersion",
        "memory",
        "message",
        "name",
        "network",
        "param",
        "protected",
        "sshAddress",
        "sshConfigFile",
        "sshLocalPort",
        "status",
        "vmType",
        "HostOS",
        "HostArch",
        "LimaHome",
        "IdentityFile",
    }
)
_STATUS_FIELDS = frozenset(
    {
        "running",
        "degraded",
        "exiting",
        "errors",
        "sshLocalPort",
        "cloudInitProgress",
        "portForward",
        "vsock",
    }
)


class LimaConfigurationError(ValueError):
    """Report unsafe managed-Lima state, configuration, or evidence."""


class LimaLifecycleState(StrEnum):
    """Identify every durable managed-instance lifecycle state."""

    ABSENT = "ABSENT"
    CREATING = "CREATING"
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    DELETING = "DELETING"
    FAILED_CLEANUP = "FAILED_CLEANUP"


@dataclass(frozen=True, slots=True)
class LimaInstanceConfiguration:
    """Represent one closed private Lima instance configuration.

    Args:
        instance_name: Validated private Loop instance name.
        document: Canonical UTF-8 Lima YAML document.
        manifest_digest: Authoritative runtime manifest identity.
        lima_version: Exact manifest-pinned Lima version.
    """

    instance_name: str
    document: bytes
    manifest_digest: str
    lima_version: str


@dataclass(frozen=True, slots=True)
class LimaInstanceContext:
    """Bind operations to one private home and active runtime lease.

    Args:
        executable: Lease-bound Lima executable.
        instance_name: Closed Loop instance identity.
        state_root: Resolved Loop-owned lifecycle state root.
        lima_home: Private Lima home below ``state_root``.
        manifest_digest: Manifest identity for this epoch.
        artifact_set_digest: Installed artifact-set identity.
        configuration_digest: Generated configuration identity.
        epoch: Positive lifecycle epoch.
        root_device: Original state-root device identity.
        root_inode: Original state-root inode identity.
        health_nonce: Epoch-specific guest health challenge.
    """

    executable: InstalledExecutable
    instance_name: str
    state_root: Path
    lima_home: Path
    manifest_digest: str
    artifact_set_digest: str
    configuration_digest: str
    epoch: int
    root_device: int
    root_inode: int
    health_nonce: str

    @classmethod
    def create(
        cls,
        executable: InstalledExecutable,
        state_root: Path,
        configuration: LimaInstanceConfiguration,
        epoch: int,
    ) -> LimaInstanceContext:
        """Create a descriptor-identified private context.

        Args:
            executable: Lease-bound Lima executable.
            state_root: Loop-owned absolute lifecycle state root.
            configuration: Generated closed Lima YAML.
            epoch: Positive lifecycle epoch.

        Returns:
            A validated private invocation context.

        Raises:
            LimaConfigurationError: If an identity, lease, or path is invalid.
        """
        if (
            executable.artifact_id != "lima"
            or epoch < 1
            or not state_root.is_absolute()
            or not _INSTANCE_NAME.fullmatch(configuration.instance_name)
            or executable.lease.manifest_digest != configuration.manifest_digest
            or executable.artifact_set_digest != executable.lease.artifact_set_digest
        ):
            raise LimaConfigurationError("Managed Lima context is invalid.")
        try:
            state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            root = state_root.resolve(strict=True)
            metadata = state_root.lstat()
        except OSError as error:
            raise LimaConfigurationError("Managed Lima state root is invalid.") from error
        working_directory = Path.cwd().resolve()
        home_directory = Path.home().resolve()
        runtime_root = executable.runtime_root.resolve()
        artifact_root = executable.artifact_root.resolve()
        forbidden = {working_directory, home_directory, runtime_root, artifact_root}
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or root in forbidden
            or any(root in value.parents for value in forbidden)
            or working_directory in root.parents
            or runtime_root in root.parents
            or artifact_root in root.parents
        ):
            raise LimaConfigurationError("Managed Lima state root is not isolated.")
        os.chmod(root, 0o700)
        home = root / "lima"
        try:
            home.mkdir(mode=0o700, exist_ok=True)
        except OSError as error:
            raise LimaConfigurationError("Managed Lima home is invalid.") from error
        if home.is_symlink() or not home.is_dir():
            raise LimaConfigurationError("Managed Lima home is invalid.")
        os.chmod(home, 0o700)
        private_user_home = root / "home"
        try:
            private_user_home.mkdir(mode=0o700, exist_ok=True)
        except OSError as error:
            raise LimaConfigurationError("Managed Lima private user home is invalid.") from error
        if private_user_home.is_symlink() or not private_user_home.is_dir():
            raise LimaConfigurationError("Managed Lima private user home is invalid.")
        os.chmod(private_user_home, 0o700)
        root_stat = root.stat()
        config_digest = sha256_digest(configuration.document)
        nonce = sha256_digest(
            f"{configuration.manifest_digest}:{executable.artifact_set_digest}:{config_digest}:{epoch}"
        )
        context = cls(
            executable,
            configuration.instance_name,
            root,
            home.resolve(strict=True),
            configuration.manifest_digest,
            executable.artifact_set_digest,
            config_digest,
            epoch,
            root_stat.st_dev,
            root_stat.st_ino,
            nonce,
        )
        context.validate()
        return context

    def validate(self) -> None:
        """Reject replaced private state or an inactive/replaced lease.

        Raises:
            LimaConfigurationError: If state or lease identity is no longer exact.
        """
        try:
            root_stat = self.state_root.lstat()
            home_stat = self.lima_home.lstat()
            user_home_stat = (self.state_root / "home").lstat()
            payload = json.loads(self.executable.lease.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise LimaConfigurationError("Managed Lima context is no longer valid.") from error
        lease = self.executable.lease
        expected_lease = {
            "owner_id": lease.owner_id,
            "manifest_digest": lease.manifest_digest,
            "artifact_set_digest": lease.artifact_set_digest,
            "created": lease.created,
            "expiry": payload.get("expiry"),
        }
        if (
            stat.S_ISLNK(root_stat.st_mode)
            or stat.S_ISLNK(home_stat.st_mode)
            or not stat.S_ISDIR(root_stat.st_mode)
            or not stat.S_ISDIR(home_stat.st_mode)
            or stat.S_ISLNK(user_home_stat.st_mode)
            or not stat.S_ISDIR(user_home_stat.st_mode)
            or (root_stat.st_dev, root_stat.st_ino) != (self.root_device, self.root_inode)
            or self.lima_home.parent != self.state_root
            or stat.S_IMODE(root_stat.st_mode) != 0o700
            or stat.S_IMODE(home_stat.st_mode) != 0o700
            or stat.S_IMODE(user_home_stat.st_mode) != 0o700
            or self.artifact_set_digest != self.executable.artifact_set_digest
            or self.manifest_digest != lease.manifest_digest
            or payload != expected_lease
            or not isinstance(payload.get("expiry"), (int, float))
            or payload["expiry"] <= time.time()
        ):
            raise LimaConfigurationError("Managed Lima context identity is no longer valid.")

    @property
    def instance_directory(self) -> Path:
        """Return the sole expected private instance path."""
        return self.lima_home / self.instance_name


class LimaClient:
    """Own every fixed ``limactl`` and macOS signature invocation.

    Args:
        process (InfrastructureProcessRunner): Small fixed-argv process primitive.
        application_data (Path): Private application-data working directory.

    Raises:
        ValueError: If the working directory cannot be established as private state.
    """

    process: InfrastructureProcessRunner
    application_data: Path

    def __init__(self, process: InfrastructureProcessRunner, application_data: Path) -> None:
        try:
            application_data.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as error:
            raise ValueError("Lima process cwd must be a private directory.") from error
        if application_data.is_symlink() or not application_data.is_dir():
            raise ValueError("Lima process cwd must be a private directory.")
        self.process = process
        self.application_data = application_data.resolve()

    @property
    def output_limit(self) -> int:
        """Return the process boundary's per-stream output limit.

        Returns:
            int: Maximum retained bytes for one infrastructure output stream.
        """
        return self.process.output_limit

    def create(
        self, context: LimaInstanceContext, configuration: bytes
    ) -> InfrastructureProcessResult:
        """Create one named instance from bounded generated configuration bytes."""
        return self._run(
            context,
            ("--tty=false", "create", "--name", context.instance_name, "-"),
            "lima.create",
            stdin=configuration,
            deadline_seconds=1800.0,
        )

    def start(self, context: LimaInstanceContext) -> InfrastructureProcessResult:
        """Start one named private instance."""
        return self._run(
            context,
            ("--tty=false", "start", "--timeout", "10m", context.instance_name),
            "lima.start",
            deadline_seconds=600.0,
        )

    def list(self, context: LimaInstanceContext) -> InfrastructureProcessResult:
        """Inspect every persisted field of one named instance."""
        return self._run(
            context,
            ("--tty=false", "list", "--format", "json", "--all-fields", context.instance_name),
            "lima.list",
        )

    def stop(
        self, context: LimaInstanceContext, *, force: bool = False
    ) -> InfrastructureProcessResult:
        """Stop one named instance, optionally using Lima's force mode."""
        arguments = ("--tty=false", "stop", *(("--force",) if force else ()), context.instance_name)
        return self._run(
            context,
            arguments,
            "lima.force_stop" if force else "lima.stop",
            deadline_seconds=60.0 if force else 120.0,
        )

    def delete(self, context: LimaInstanceContext) -> InfrastructureProcessResult:
        """Delete one named stopped instance."""
        return self._run(
            context,
            ("--tty=false", "delete", "--force", context.instance_name),
            "lima.delete",
            deadline_seconds=300.0,
        )

    def health(self, context: LimaInstanceContext) -> InfrastructureProcessResult:
        """Run the fixed guest health challenge over AF_VSOCK-backed SSH."""
        if not re.fullmatch(r"[0-9a-f]{64}", context.health_nonce):
            raise ValueError("Lima health nonce is invalid.")
        return self.run_guest(
            context,
            (
                "/bin/sh",
                "-c",
                'uname -m; printf "%s\\n" "$1"',
                "loop-health",
                context.health_nonce,
            ),
            operation="lima.health",
            deadline_seconds=60.0,
        )

    def run_guest(
        self,
        context: LimaInstanceContext,
        argv: tuple[str, ...],
        *,
        operation: str,
        deadline_seconds: float = 120.0,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Run exact domain-owned guest argv through fixed host-initiated Lima shell transport."""
        command = self.guest_command(context, argv, operation=operation)
        return self.process.run(
            command, deadline_seconds=deadline_seconds, cancellation=cancellation
        )

    def guest_command(
        self,
        context: LimaInstanceContext,
        argv: tuple[str, ...],
        *,
        operation: str,
        pty: bool = False,
    ) -> InfrastructureProcessCommand:
        """Wrap exact guest argv in a reviewed ``limactl shell`` command shape.

        Args:
            context (LimaInstanceContext): Validated private instance context.
            argv (tuple[str, ...]): Exact trusted guest argument vector.
            operation (str): Sanitized infrastructure operation identity.
            pty (bool): Whether Lima must propagate an attached terminal into the guest.

        Returns:
            InfrastructureProcessCommand: Sealed host-side Lima invocation.

        Raises:
            ValueError: If the guest argument vector is empty or contains a NUL.
        """
        if not argv or any("\0" in value for value in argv):
            raise ValueError("Lima guest command is invalid.")
        return self._command(
            context,
            (f"--tty={'true' if pty else 'false'}", "shell", context.instance_name, "--", *argv),
            operation,
            terminal=pty,
        )

    def copy_to_guest(
        self, context: LimaInstanceContext, source: Path, guest_destination: str
    ) -> InfrastructureProcessResult:
        """Copy one private host file to an absolute guest destination with Lima SCP."""
        context.validate()
        try:
            source = source.resolve(strict=True)
            source.relative_to(context.state_root)
        except (OSError, ValueError) as error:
            raise ValueError("Lima copy source is outside private instance state.") from error
        if not guest_destination.startswith("/") or ".." in guest_destination.split("/"):
            raise ValueError("Lima copy destination is not an absolute guest path.")
        return self._run(
            context,
            (
                "--tty=false",
                "copy",
                "--backend=scp",
                str(source),
                f"{context.instance_name}:{guest_destination}",
            ),
            "lima.copy_to_guest",
        )

    def copy_from_guest(
        self, context: LimaInstanceContext, guest_source: str, destination: Path
    ) -> InfrastructureProcessResult:
        """Copy one trusted guest file into private instance state with Lima SCP.

        Args:
            context (LimaInstanceContext): Validated private instance invocation context.
            guest_source (str): Absolute trusted guest source path.
            destination (Path): New host destination beneath private instance state.

        Returns:
            InfrastructureProcessResult: Bounded copy command result.

        Raises:
            ValueError: A source or destination path crosses its trusted boundary.
        """
        context.validate()
        if not guest_source.startswith("/") or ".." in guest_source.split("/"):
            raise ValueError("Lima copy source is not an absolute guest path.")
        try:
            parent = destination.parent.resolve(strict=True)
            parent.relative_to(context.state_root)
        except (OSError, ValueError) as error:
            raise ValueError("Lima copy destination is outside private instance state.") from error
        if destination.exists() or destination.is_symlink():
            raise ValueError("Lima copy destination must not already exist.")
        return self._run(
            context,
            (
                "--tty=false",
                "copy",
                "--backend=scp",
                f"{context.instance_name}:{guest_source}",
                str(destination),
            ),
            "lima.copy_from_guest",
        )

    def open_guest_session(
        self,
        context: LimaInstanceContext,
        argv: tuple[str, ...],
        *,
        operation: str,
        pty: bool,
        queue_limit: int,
        deadline_seconds: float,
        terminal_columns: int | None = None,
        terminal_rows: int | None = None,
    ) -> OciProcessSession:
        """Open one OCI-owned attached process through fixed Lima shell transport."""
        return OciProcessSession(
            self.guest_command(context, argv, operation=operation, pty=pty),
            pty,
            queue_limit,
            deadline_seconds,
            terminal_columns,
            terminal_rows,
        )

    def write_broker_file(
        self,
        context: LimaInstanceContext,
        guest_path: str,
        content: bytes,
    ) -> InfrastructureProcessResult:
        """Transfer bounded broker data over stdin into one trusted guest-private file.

        Args:
            context (LimaInstanceContext): Validated private instance context.
            guest_path (str): Reviewed lease or CNI configuration path.
            content (bytes): Bounded declarative configuration or secret material.

        Returns:
            InfrastructureProcessResult: Bounded staging result.

        Raises:
            ValueError: If the path or payload crosses the broker staging boundary.
        """
        _validate_broker_path(guest_path)
        if not content or len(content) > 2 * 1024 * 1024:
            raise ValueError("Broker staging content exceeds its bounded shape.")
        return self._run(
            context,
            (
                "--tty=false",
                "shell",
                context.instance_name,
                "--",
                "/bin/sh",
                "-c",
                (
                    'set -eu; umask 077; mkdir -p -- "${1%/*}"; '
                    'chmod 700 -- "${1%/*}"; cat >"$1"; chmod 644 -- "$1"'
                ),
                "loop-broker-stage",
                guest_path,
            ),
            "lima.write_broker_file",
            stdin=content,
        )

    def remove_broker_paths(
        self,
        context: LimaInstanceContext,
        guest_paths: tuple[str, ...],
    ) -> InfrastructureProcessResult:
        """Remove only validated lease-private broker state and CNI configuration.

        Args:
            context (LimaInstanceContext): Validated private instance context.
            guest_paths (tuple[str, ...]): Exact reviewed broker paths to remove.

        Returns:
            InfrastructureProcessResult: Bounded cleanup result.

        Raises:
            ValueError: If a path is broad, duplicated, or outside broker state.
        """
        if not guest_paths or len(set(guest_paths)) != len(guest_paths):
            raise ValueError("Broker cleanup paths are invalid.")
        for value in guest_paths:
            _validate_broker_path(value)
        return self._run(
            context,
            (
                "--tty=false",
                "shell",
                context.instance_name,
                "--",
                "/bin/rm",
                "-rf",
                "--",
                *guest_paths,
            ),
            "lima.remove_broker_paths",
        )

    def verify_codesign(self, executable: InstalledExecutable) -> InfrastructureProcessResult:
        """Verify Lima's seal and return its exact entitlement property list."""
        if executable.artifact_id != "lima":
            raise ValueError("macOS signature verification requires the Lima executable.")
        environment = {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}
        verify = self.process.run(
            InfrastructureProcessCommand(
                ("/usr/bin/codesign", "--verify", "--strict", str(executable.path)),
                environment,
                self.application_data,
                "macos.codesign.verify",
                None,
            )
        )
        if verify.exit_code:
            raise RuntimeError("Installed Lima signature verification failed.")
        result = self.process.run(
            InfrastructureProcessCommand(
                (
                    "/usr/bin/codesign",
                    "-d",
                    "--entitlements",
                    "-",
                    "--xml",
                    str(executable.path),
                ),
                environment,
                self.application_data,
                "macos.codesign.entitlements",
                None,
            )
        )
        if result.exit_code:
            raise RuntimeError("Installed Lima entitlement inspection failed.")
        return result

    def _run(
        self,
        context: LimaInstanceContext,
        arguments: tuple[str, ...],
        operation: str,
        *,
        stdin: bytes | None = None,
        deadline_seconds: float = 30.0,
    ) -> InfrastructureProcessResult:
        """Run one fixed Lima host command through the common process primitive."""
        return self.process.run(
            self._command(context, arguments, operation),
            stdin=stdin,
            deadline_seconds=deadline_seconds,
        )

    def _command(
        self,
        context: LimaInstanceContext,
        arguments: tuple[str, ...],
        operation: str,
        *,
        terminal: bool = False,
    ) -> InfrastructureProcessCommand:
        """Build one sealed host-side Lima command after revalidating private state."""
        context.validate()
        if context.executable.artifact_id != "lima" or not _INSTANCE_NAME.fullmatch(
            context.instance_name
        ):
            raise ValueError("Lima invocation context is invalid.")
        try:
            home = context.lima_home.resolve(strict=True)
            home.relative_to(context.state_root)
        except (OSError, ValueError) as error:
            raise ValueError("Lima home is outside private application state.") from error
        environment = sealed_environment(
            context.executable,
            {
                "HOME": str(context.state_root / "home"),
                "LIMA_HOME": str(home),
                **({"TERM": "xterm-256color"} if terminal else {}),
            },
        )
        return InfrastructureProcessCommand(
            (str(context.executable.path), *arguments),
            environment,
            self.application_data,
            operation,
            context.executable,
        )


@dataclass(frozen=True, slots=True)
class Journal:
    """Represent the durable journal for a managed Lima instance."""

    manifest_digest: str
    artifact_set_digest: str
    configuration_digest: str
    instance_name: str
    epoch: int
    intended_state: str
    last_completed_transition: str
    created_in_epoch: bool


class ManagedLimaInstance:
    """Own a journaled private Lima lifecycle without adopting foreign state.

    Args:
        runner: Sole classified Lima process boundary.
        context: Private lease-bound invocation context.
        configuration: Closed generated Lima configuration.
    """

    runner: LimaClient
    context: LimaInstanceContext
    configuration: LimaInstanceConfiguration
    journal: Path

    def __init__(
        self,
        runner: LimaClient,
        context: LimaInstanceContext,
        configuration: LimaInstanceConfiguration,
    ) -> None:
        if (
            context.configuration_digest != sha256_digest(configuration.document)
            or context.manifest_digest != configuration.manifest_digest
        ):
            raise LimaConfigurationError("Managed Lima configuration identity is invalid.")
        self.runner = runner
        self.context = context
        self.configuration = configuration
        self.journal = context.lima_home / f"{context.instance_name}.loop.json"

    @property
    def state(self) -> LimaLifecycleState:
        """Return durable state, or ``ABSENT`` when no ownership exists."""
        return (
            LimaLifecycleState(self._load().intended_state)
            if self.journal.exists()
            else LimaLifecycleState.ABSENT
        )

    def create(self) -> None:
        """Create, or exactly adopt, one attested stopped instance."""
        if self.journal.exists():
            journal = self._load()
            state = LimaLifecycleState(journal.intended_state)
            if state in {LimaLifecycleState.STOPPED, LimaLifecycleState.RUNNING}:
                self._attest_list(state)
                return
            raise LimaConfigurationError("Managed Lima transition requires recovery.")
        self._record(LimaLifecycleState.CREATING, "absent", True)
        try:
            self._success(self.runner.create(self.context, self.configuration.document), "creation")
            self._attest_list(LimaLifecycleState.STOPPED)
        except (LimaConfigurationError, RuntimeError) as error:
            self._rollback(True)
            raise LimaConfigurationError("Managed Lima creation failed.") from error
        self._record(LimaLifecycleState.STOPPED, "create", True)

    def start(self) -> None:
        """Start and attest VZ, private management, VSOCK, and guest health."""
        journal = self._owned({LimaLifecycleState.STOPPED})
        self._record(
            LimaLifecycleState.STARTING, journal.last_completed_transition, journal.created_in_epoch
        )
        try:
            self._success(self.runner.start(self.context), "start")
            self._attest_running_evidence()
        except (LimaConfigurationError, RuntimeError) as error:
            self._rollback(journal.created_in_epoch)
            raise LimaConfigurationError("Managed Lima start or attestation failed.") from error
        self._record(LimaLifecycleState.RUNNING, "start", journal.created_in_epoch)

    def attest_running(self) -> None:
        """Re-attest a running instance after wake or before reuse."""
        journal = self._owned({LimaLifecycleState.RUNNING})
        try:
            self._attest_running_evidence()
        except (LimaConfigurationError, RuntimeError) as error:
            self._rollback(journal.created_in_epoch)
            raise LimaConfigurationError("Managed Lima running attestation failed.") from error

    def stop(self, *, force: bool = False) -> None:
        """Idempotently stop an owned instance.

        Args:
            force: Use the distinct fixed force-stop cleanup operation.
        """
        journal = self._owned({LimaLifecycleState.RUNNING, LimaLifecycleState.STOPPED})
        if journal.intended_state == LimaLifecycleState.STOPPED:
            self._attest_list(LimaLifecycleState.STOPPED)
            return
        self._record(
            LimaLifecycleState.STOPPING, journal.last_completed_transition, journal.created_in_epoch
        )
        try:
            self._success(self.runner.stop(self.context, force=force), "stop")
            self._attest_list(LimaLifecycleState.STOPPED)
        except (LimaConfigurationError, RuntimeError) as error:
            self._record(LimaLifecycleState.FAILED_CLEANUP, "stop-failed", journal.created_in_epoch)
            raise LimaConfigurationError("Managed Lima stop failed.") from error
        self._record(LimaLifecycleState.STOPPED, "stop", journal.created_in_epoch)

    def delete(self) -> None:
        """Idempotently delete the exact journal-owned instance."""
        if not self.journal.exists():
            return
        journal = self._owned(
            {
                LimaLifecycleState.STOPPED,
                LimaLifecycleState.RUNNING,
                LimaLifecycleState.FAILED_CLEANUP,
            }
        )
        original = LimaLifecycleState(journal.intended_state)
        self._record(
            LimaLifecycleState.DELETING, journal.last_completed_transition, journal.created_in_epoch
        )
        try:
            if original is LimaLifecycleState.RUNNING:
                self._success(
                    self.runner.stop(self.context, force=True),
                    "force stop",
                )
            self._success(self.runner.delete(self.context), "deletion")
        except (LimaConfigurationError, RuntimeError) as error:
            self._record(
                LimaLifecycleState.FAILED_CLEANUP, "delete-failed", journal.created_in_epoch
            )
            raise LimaConfigurationError("Managed Lima deletion failed.") from error
        self._remove_journal()

    def reset(self) -> None:
        """Recover or delete only an exactly journal-owned private instance."""
        if not self.journal.exists():
            return
        journal = self._load()
        self._record(
            LimaLifecycleState.DELETING, journal.last_completed_transition, journal.created_in_epoch
        )
        try:
            force = self.runner.stop(self.context, force=True)
            if force.stdout_truncated or force.stderr_truncated:
                raise LimaConfigurationError("Managed Lima force-stop evidence was truncated.")
            self._success(self.runner.delete(self.context), "reset")
        except (LimaConfigurationError, RuntimeError) as error:
            self._record(
                LimaLifecycleState.FAILED_CLEANUP, "reset-failed", journal.created_in_epoch
            )
            raise LimaConfigurationError("Managed Lima reset failed.") from error
        self._remove_journal()

    def reset_obsolete(self) -> None:
        """Delete a prior-release instance after validating its durable ownership evidence.

        Raises:
            LimaConfigurationError: If the obsolete journal or persisted configuration cannot
                prove ownership, or if the fixed cleanup operations fail.
        """
        self._load_obsolete()
        try:
            force = self.runner.stop(self.context, force=True)
            if force.stdout_truncated or force.stderr_truncated:
                raise LimaConfigurationError("Managed Lima force-stop evidence was truncated.")
            self._success(self.runner.delete(self.context), "obsolete reset")
        except (LimaConfigurationError, RuntimeError) as error:
            raise LimaConfigurationError("Managed obsolete Lima reset failed.") from error
        self._remove_journal()

    def _rollback(self, delete: bool) -> None:
        failed = False
        try:
            stopped = self.runner.stop(self.context, force=True)
            failed = stopped.stdout_truncated or stopped.stderr_truncated
            if delete:
                removed = self.runner.delete(self.context)
                failed = failed or bool(
                    removed.exit_code or removed.stdout_truncated or removed.stderr_truncated
                )
        except RuntimeError:
            failed = True
        if failed:
            self._record(LimaLifecycleState.FAILED_CLEANUP, "rollback-failed", delete)
        elif delete:
            self._remove_journal()
        else:
            self._record(LimaLifecycleState.STOPPED, "rollback", False)

    def _attest_running_evidence(self) -> None:
        record = self._attest_list(LimaLifecycleState.RUNNING)
        self._attest_persisted_configuration()
        self._attest_ssh(record)
        self._attest_vsock(record)
        health = self.runner.health(self.context)
        self._success(health, "guest health")
        if health.stdout != f"aarch64\n{self.context.health_nonce}\n".encode():
            raise LimaConfigurationError("Managed Lima guest health evidence is invalid.")

    def _attest_list(self, state: LimaLifecycleState) -> dict[str, Any]:
        evidence = self.runner.list(self.context)
        self._success(evidence, "list attestation")
        try:
            records = [json.loads(line) for line in evidence.stdout.splitlines() if line.strip()]
            record = records[0]
            required = {"name", "status", "vmType", "arch", "dir", "protected", "limaVersion"}
            valid = (
                len(records) == 1
                and isinstance(record, dict)
                and set(record).issubset(_LIST_FIELDS)
                and required.issubset(record)
                and record["name"] == self.context.instance_name
                and record["status"] == state.value.title()
                and record["vmType"] == "vz"
                and record["arch"] == "aarch64"
                and Path(record["dir"]) == self.context.instance_directory
                and record["protected"] is False
                and record["limaVersion"] == f"v{self.configuration.lima_version}"
                and record.get("HostOS") == "darwin"
                and record.get("HostArch") == "aarch64"
                and record.get("LimaHome") == str(self.context.lima_home)
                and record.get("IdentityFile") == str(self.context.lima_home / "_config" / "user")
                and record.get("errors", []) == []
            )
        except (IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise LimaConfigurationError("Managed Lima list evidence is invalid.") from error
        if not valid:
            raise LimaConfigurationError("Managed Lima list evidence is invalid.")
        return record

    def _attest_persisted_configuration(self) -> None:
        expected = yaml.safe_load(self.configuration.document)
        try:
            actual = yaml.safe_load((self.context.instance_directory / "lima.yaml").read_bytes())
            pairs = (
                (actual["vmType"], expected["vmType"]),
                (actual["arch"], expected["arch"]),
                (actual["images"], expected["images"]),
                (actual["mountType"], "virtiofs"),
                (actual["mounts"], expected["mounts"]),
                (actual["networks"], [{"vzNAT": True}]),
                (actual["portForwards"], expected["portForwards"]),
                (actual["vmOpts"]["vz"]["rosetta"]["enabled"], False),
                (actual["ssh"]["overVsock"], True),
                (actual["ssh"]["localPort"], 0),
                (actual["containerd"]["system"], False),
                (actual["containerd"]["user"], True),
                (actual["containerd"]["archives"], expected["containerd"]["archives"]),
            )
        except (OSError, TypeError, KeyError, yaml.YAMLError) as error:
            raise LimaConfigurationError(
                "Managed Lima persisted configuration is invalid."
            ) from error
        if any(left != right for left, right in pairs):
            raise LimaConfigurationError("Managed Lima persisted configuration changed.")

    def _attest_ssh(self, record: dict[str, Any]) -> None:
        try:
            address = ipaddress.ip_address(record["sshAddress"])
            port = record["sshLocalPort"]
            path = Path(record["sshConfigFile"])
            if (
                not address.is_loopback
                or not isinstance(port, int)
                or isinstance(port, bool)
                or not 0 < port < 65536
                or path != self.context.instance_directory / "ssh.config"
                or path.is_symlink()
            ):
                raise ValueError
            normalized = {
                line.strip().replace("=", " ", 1)
                for line in path.read_text(encoding="utf-8").splitlines()
            }
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise LimaConfigurationError("Managed Lima SSH evidence is invalid.") from error
        if not {f"Hostname {address}", f"Port {port}"}.issubset(normalized):
            raise LimaConfigurationError(
                "Managed Lima SSH configuration conflicts with list evidence."
            )
        for prefix in ("IdentityFile ", "ControlPath "):
            matches = [
                line[len(prefix) :].strip("'\"") for line in normalized if line.startswith(prefix)
            ]
            try:
                if len(matches) != 1 or not Path(matches[0]).is_absolute():
                    raise ValueError
                Path(matches[0]).relative_to(self.context.lima_home)
            except ValueError as error:
                raise LimaConfigurationError(
                    "Managed Lima SSH path escaped private state."
                ) from error

    def _attest_vsock(self, record: dict[str, Any]) -> None:
        path = self.context.instance_directory / "ha.stdout.log"
        try:
            if path.is_symlink() or path.stat().st_size > self.runner.output_limit:
                raise ValueError
            events = [json.loads(line) for line in path.read_bytes().splitlines() if line]
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise LimaConfigurationError("Managed Lima host-agent evidence is invalid.") from error
        vsocks = []
        for event in events:
            if not isinstance(event, dict) or not set(event).issubset({"time", "status"}):
                raise LimaConfigurationError("Managed Lima host-agent schema changed.")
            status_value = event.get("status", {})
            if not isinstance(status_value, dict) or not set(status_value).issubset(_STATUS_FIELDS):
                raise LimaConfigurationError("Managed Lima host-agent status schema changed.")
            if "vsock" in status_value:
                if not isinstance(status_value["vsock"], dict):
                    raise LimaConfigurationError("Managed Lima VSOCK evidence is malformed.")
                vsocks.append(status_value["vsock"])
        expected = {
            "type": "started",
            "hostAddr": f"{record['sshAddress']}:{record['sshLocalPort']}",
            "vsockPort": 22,
        }
        if vsocks != [expected]:
            raise LimaConfigurationError("Managed Lima VSOCK readiness is not exact.")

    def _owned(self, allowed: set[LimaLifecycleState]) -> Journal:
        journal = self._load()
        if LimaLifecycleState(journal.intended_state) not in allowed:
            raise LimaConfigurationError("Managed Lima lifecycle transition is invalid.")
        return journal

    def _load(self) -> Journal:
        journal = self._load_journal()
        expected = (
            self.context.manifest_digest,
            self.context.artifact_set_digest,
            self.context.configuration_digest,
            self.context.instance_name,
            self.context.epoch,
        )
        actual = (
            journal.manifest_digest,
            journal.artifact_set_digest,
            journal.configuration_digest,
            journal.instance_name,
            journal.epoch,
        )
        if actual != expected:
            raise LimaConfigurationError("Managed Lima journal ownership does not match.")
        self.context.validate()
        return journal

    def _load_obsolete(self) -> Journal:
        """Load ownership evidence for a prior release without adopting its state."""
        journal = self._load_journal()
        current = (
            self.context.manifest_digest,
            self.context.artifact_set_digest,
            self.context.configuration_digest,
        )
        recorded = (
            journal.manifest_digest,
            journal.artifact_set_digest,
            journal.configuration_digest,
        )
        configuration = self.context.instance_directory / "lima.yaml"
        try:
            metadata = configuration.lstat()
            persisted_digest = sha256_digest(configuration.read_bytes())
        except OSError as error:
            raise LimaConfigurationError(
                "Managed obsolete Lima configuration is invalid."
            ) from error
        if (
            recorded == current
            or journal.instance_name != self.context.instance_name
            or journal.epoch != self.context.epoch
            or not journal.created_in_epoch
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or persisted_digest != journal.configuration_digest
        ):
            raise LimaConfigurationError("Managed obsolete Lima ownership does not match.")
        self.context.validate()
        return journal

    def _load_journal(self) -> Journal:
        """Parse the exact bounded journal schema without choosing an ownership epoch."""
        fields = {field.name for field in Journal.__dataclass_fields__.values()}
        try:
            value = json.loads(self.journal.read_text(encoding="utf-8"))
            if set(value) != fields:
                raise ValueError
            journal = Journal(**value)
            LimaLifecycleState(journal.intended_state)
        except (OSError, TypeError, ValueError) as error:
            raise LimaConfigurationError("Managed Lima journal is invalid.") from error
        if (
            not isinstance(journal.manifest_digest, str)
            or not journal.manifest_digest.startswith("sha256:")
            or len(journal.manifest_digest) != 71
            or not isinstance(journal.artifact_set_digest, str)
            or len(journal.artifact_set_digest) != 64
            or not isinstance(journal.configuration_digest, str)
            or len(journal.configuration_digest) != 64
            or not isinstance(journal.instance_name, str)
            or not isinstance(journal.epoch, int)
            or isinstance(journal.epoch, bool)
            or journal.epoch < 1
            or not isinstance(journal.created_in_epoch, bool)
            or not isinstance(journal.last_completed_transition, str)
            or len(journal.last_completed_transition) > 64
        ):
            raise LimaConfigurationError("Managed Lima journal is invalid.")
        return journal

    def _record(self, state: LimaLifecycleState, completed: str, created: bool) -> None:
        self.context.validate()
        temporary = self.journal.with_suffix(".new")
        value = Journal(
            self.context.manifest_digest,
            self.context.artifact_set_digest,
            self.context.configuration_digest,
            self.context.instance_name,
            self.context.epoch,
            state.value,
            completed,
            created,
        )
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(asdict(value), stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.journal)
            self._fsync_directory(self.journal.parent)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise LimaConfigurationError("Managed Lima journal could not be persisted.") from error

    def _remove_journal(self) -> None:
        self.journal.unlink()
        self._fsync_directory(self.journal.parent)

    @staticmethod
    def _success(result: InfrastructureProcessResult, operation: str) -> None:
        if result.exit_code or result.stdout_truncated or result.stderr_truncated:
            _LOGGER.error(
                "Managed Lima %s failed: exit=%s stdout=%r stderr=%r truncated=(%s,%s).",
                operation,
                result.exit_code,
                result.stdout,
                result.stderr,
                result.stdout_truncated,
                result.stderr_truncated,
            )
            raise LimaConfigurationError(f"Managed Lima {operation} failed.")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def bootstrap_lima_executable(
    bootstrapper: RuntimeBootstrapper, candidate: RuntimeManifest
) -> InstalledExecutable:
    """Install and lease the manifest-pinned Lima runtime.

    Args:
        bootstrapper (RuntimeBootstrapper): Verified private runtime installer.
        candidate (RuntimeManifest): Authoritative candidate or release manifest.

    Returns:
        InstalledExecutable: Lease-bound verified ``limactl`` executable.

    Raises:
        LimaConfigurationError: If the selected Lima executable layout is invalid.
    """
    artifact = macos_artifact(candidate, "lima")
    runtime = bootstrapper.ensure(
        RuntimeRequirement(
            platform=PlatformSelector(os="macos", architecture="arm64"),
            capabilities=frozenset({"sandbox-management"}),
        )
    )
    if artifact.layout is None or len(artifact.layout.executables) != 1:
        raise LimaConfigurationError("Lima manifest executable layout is invalid.")
    return runtime.executable("lima", artifact.layout.executables[0])


def build_lima_instance_configuration(
    candidate: RuntimeManifest, instance_name: str, snapshot_store: Path
) -> LimaInstanceConfiguration:
    """Build a VZ-only Lima configuration with one read-only snapshot-store share.

    Args:
        candidate (RuntimeManifest): Authoritative candidate or release manifest.
        instance_name (str): Validated private Loop instance name.
        snapshot_store (Path): Workspace-specific Loop-owned immutable snapshot root.

    Returns:
        LimaInstanceConfiguration: Manifest-bound declarative Lima configuration.

    Raises:
        LimaConfigurationError: If the instance name or snapshot store is invalid.
        MacosCandidateError: If a required runtime artifact is absent.
    """
    if not _INSTANCE_NAME.fullmatch(instance_name):
        raise LimaConfigurationError("Managed Lima instance name is invalid.")
    try:
        store = snapshot_store.resolve(strict=True)
        metadata = snapshot_store.lstat()
    except OSError as error:
        raise LimaConfigurationError("Managed Lima snapshot store is invalid.") from error
    if (
        not snapshot_store.is_absolute()
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise LimaConfigurationError("Managed Lima snapshot store is invalid.")
    lima = macos_artifact(candidate, "lima")
    guest_image = macos_artifact(candidate, "guest-image")
    nerdctl = macos_artifact(candidate, "nerdctl-full")
    quote = json.dumps
    template = Template(Path(__file__).with_name("lima-template.yaml").read_text(encoding="utf-8"))
    document = template.substitute(
        guest_image_source=quote(guest_image.source),
        guest_image_digest=quote(_prefixed_digest(guest_image.digest)),
        nerdctl_source=quote(nerdctl.source),
        nerdctl_digest=quote(_prefixed_digest(nerdctl.digest)),
        snapshot_store=quote(str(store)),
    )
    return LimaInstanceConfiguration(
        instance_name,
        document.encode(),
        candidate.digest,
        lima.version,
    )


def _prefixed_digest(value: str) -> str:
    """Return one SHA-256 digest with its algorithm prefix."""
    return value if value.startswith("sha256:") else "sha256:" + value


def _validate_broker_path(value: str) -> None:
    """Reject broker paths outside the two guest-private staging roots."""
    component = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
    lease_root = rf"/run/user/[1-9][0-9]*/loop/brokers/{component}"
    cni_file = rf"/home/{component}/\.config/cni/net\.d/90-loop-{component}\.conflist"
    if (
        "\x00" in value
        or "\n" in value
        or "\r" in value
        or ".." in value.split("/")
        or (
            re.fullmatch(lease_root, value) is None
            and re.fullmatch(lease_root + rf"/{component}", value) is None
            and re.fullmatch(cni_file, value) is None
        )
    ):
        raise ValueError("Broker staging path is invalid.")
