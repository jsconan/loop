"""Define the explicit OCI management surface shared by platform adapters."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ...infrastructure import InfrastructureProcessResult
from ...runtime.models import PlatformSelector
from .process import OciProcessSession, OciStreamFrame
from .spec import OciExecutionSpec


class OciSignal(StrEnum):
    """Identify a closed set of signals accepted by the OCI supervisor."""

    HANGUP = "HUP"
    INTERRUPT = "INT"
    TERMINATE = "TERM"
    WINDOW_CHANGE = "WINCH"
    KILL = "KILL"


class OciEndpoint(Protocol):
    """Expose attested endpoint identity without assuming host-local transport."""

    namespace: str
    content_store_identity: str
    platform: PlatformSelector
    artifact_set_digest: str


@dataclass(frozen=True, slots=True)
class GuestOciRuntimeEndpoint:
    """Bind OCI management to an attested socket inside a managed guest.

    Args:
        socket_path (str): Absolute private socket path inside the guest.
        state_path (str): Absolute rootless containerd state path inside the guest.
        content_path (str): Absolute content-store path inside the guest.
        namespace (str): Closed containerd namespace.
        content_store_identity (str): Manifest-bound SHA-256 identity of the store.
        platform (PlatformSelector): Exact guest OCI platform.
        artifact_set_digest (str): Host substrate artifact-set identity.
        owner_uid (int): Non-root guest owner of the runtime and socket.
        network_namespace_path (str): Attested detached RootlessKit network namespace handle.
        network_namespace_identity (str): Device and inode identity of that namespace.
        nerdctl_version (str): Attested nerdctl version.
        containerd_version (str): Attested containerd version.
        runc_version (str): Attested runc version.
        buildkit_version (str): Attested transient-builder binary version.
    """

    socket_path: str
    state_path: str
    content_path: str
    namespace: str
    content_store_identity: str
    platform: PlatformSelector
    artifact_set_digest: str
    owner_uid: int
    network_namespace_path: str
    network_namespace_identity: str
    nerdctl_version: str
    containerd_version: str
    runc_version: str
    buildkit_version: str

    def __post_init__(self) -> None:
        """Reject non-private, root-owned, relative, or unbound guest evidence."""
        paths = (self.socket_path, self.state_path, self.content_path)
        if (
            self.owner_uid <= 0
            or any(not value.startswith("/") or ".." in value.split("/") for value in paths)
            or re.fullmatch(r"/home/[A-Za-z0-9._-]+/\.local/share/containerd", self.state_path)
            is None
            or self.content_path
            != self.state_path.rstrip("/") + "/io.containerd.content.v1.content"
            or not self.socket_path.startswith("/proc/")
            or not self.socket_path.endswith("/root/run/containerd/containerd.sock")
            or not self.socket_path.removeprefix("/proc/")
            .removesuffix("/root/run/containerd/containerd.sock")
            .isdigit()
            or re.fullmatch(
                r"/proc/[1-9][0-9]*/root/run/user/[1-9][0-9]*/"
                r"containerd-rootless/netns",
                self.network_namespace_path,
            )
            is None
            or self.network_namespace_path.split("/", 3)[2] != self.socket_path.split("/", 3)[2]
            or self.network_namespace_path.split("/", 7)[6] != str(self.owner_uid)
            or re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", self.network_namespace_identity) is None
            or not self.namespace
            or not self.content_store_identity.startswith("sha256:")
            or len(self.content_store_identity) != 71
            or any(
                character not in "0123456789abcdef" for character in self.content_store_identity[7:]
            )
            or len(self.artifact_set_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.artifact_set_digest)
            or not all(
                (
                    self.nerdctl_version,
                    self.containerd_version,
                    self.runc_version,
                    self.buildkit_version,
                )
            )
        ):
            raise ValueError("Guest OCI endpoint evidence is invalid.")


@dataclass(frozen=True, slots=True)
class OciAttemptBindings:
    """Carry platform-resolved broker resources into one OCI attempt.

    Args:
        network_name (str | None): Dedicated CNI network name, absent for raw secrets alone.
        command_address (str | None): Fixed address assigned only to a networked command.
        host_aliases (tuple[tuple[str, str], ...]): Exact hosts entries directed to Envoy.
        dns_servers (tuple[str, ...]): Exact broker gateway used by granted DNS.
        environment_file (str | None): Trusted guest env-file path containing raw secrets.
        secret_mounts (tuple[tuple[str, str], ...]): Trusted guest source and private destination.
        start_gate (str | None): Trusted lease file gating hostname-scoped task exposure.
    """

    network_name: str | None
    command_address: str | None
    host_aliases: tuple[tuple[str, str], ...] = ()
    dns_servers: tuple[str, ...] = ()
    environment_file: str | None = None
    secret_mounts: tuple[tuple[str, str], ...] = ()
    start_gate: str | None = None


class OciControlSession(Protocol):
    """Expose bounded raw streams from one attached OCI management operation."""

    def write(self, data: bytes) -> None:
        """Write one bounded opaque input frame."""

    def close_stdin(self) -> None:
        """Close input exactly once without closing output."""

    def resize(self, columns: int, rows: int) -> None:
        """Resize the attached PTY when the session owns one."""

    def read(self, timeout: float | None = None) -> OciStreamFrame | None:
        """Return the next ordered opaque output frame."""

    def wait(self, cancellation: Callable[[], bool] = lambda: False) -> int:
        """Wait for terminal completion under the session deadline."""

    def detach(self) -> None:
        """Use the runtime client's detach protocol and release the local session."""

    def close(self) -> None:
        """Cancel the session and release its resources."""


class OciImageTransport(Protocol):
    """Expose only the image operations needed by the OCI installer."""

    def pull_image(self, reference: str) -> InfrastructureProcessResult:
        """Pull one immutable image reference.

        Args:
            reference (str): Digest-pinned image reference.

        Returns:
            InfrastructureProcessResult: Bounded raw pull result.
        """

    def inspect_image(self, reference: str) -> InfrastructureProcessResult:
        """Inspect one immutable image reference.

        Args:
            reference (str): Digest-pinned image reference.

        Returns:
            InfrastructureProcessResult: Bounded raw inspection evidence.
        """


class OciCommandTransport(Protocol):
    """Carry exact OCI-owned guest argv across one platform control plane."""

    def run_oci_command(
        self,
        argv: tuple[str, ...],
        operation: str,
        cancellation: Callable[[], bool],
    ) -> InfrastructureProcessResult:
        """Run exact guest argv and return bounded process output.

        Args:
            argv (tuple[str, ...]): OCI-owned guest command.
            operation (str): Sanitized operation identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw process result.
        """

    def open_oci_session(
        self,
        argv: tuple[str, ...],
        operation: str,
        use_pty: bool,
        queue_limit: int,
        deadline_seconds: float,
        terminal_columns: int | None = None,
        terminal_rows: int | None = None,
    ) -> OciProcessSession:
        """Open one attached exact guest argv session.

        Args:
            argv (tuple[str, ...]): OCI-owned guest command.
            operation (str): Sanitized operation identity.
            use_pty (bool): Whether to allocate a pseudo-terminal.
            queue_limit (int): Maximum unread frames before cleanup.
            deadline_seconds (float): Positive wall deadline.
            terminal_columns (int | None): Initial PTY width.
            terminal_rows (int | None): Initial PTY height.

        Returns:
            OciProcessSession: Owned attached transport process.
        """


_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:+=-]{0,511}$")


class OciSupervisor:
    """Own fixed-shape nerdctl operations and attached OCI process handles.

    Args:
        endpoint (GuestOciRuntimeEndpoint): Attested private guest runtime endpoint.
        transport (OciCommandTransport): Platform transport for exact guest argv.
    """

    _endpoint: GuestOciRuntimeEndpoint
    _transport: OciCommandTransport

    def __init__(
        self,
        endpoint: GuestOciRuntimeEndpoint,
        transport: OciCommandTransport,
    ) -> None:
        self._endpoint = endpoint
        self._transport = transport

    def version(
        self, cancellation: Callable[[], bool] = lambda: False
    ) -> InfrastructureProcessResult:
        """Return bounded runtime version evidence.

        Args:
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw version evidence.
        """
        return self._run(("version", "--format", "{{json .}}"), "version", cancellation)

    def pull_image(
        self,
        reference: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Pull one validated immutable image reference.

        Args:
            reference (str): Digest-pinned image reference.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw pull result.

        Raises:
            ValueError: If the reference could be interpreted as an option or malformed value.
        """
        return self._run(
            ("image", "pull", *self._platform_arguments(), self._value(reference)),
            "image_pull",
            cancellation,
        )

    def inspect_image(
        self,
        reference: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Inspect one validated immutable image reference.

        Args:
            reference (str): Digest-pinned image reference.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw inspection evidence.

        Raises:
            ValueError: If the reference could be interpreted as an option or malformed value.
        """
        return self._run(
            (
                "image",
                "inspect",
                "--mode=native",
                "--format",
                "{{json .}}",
                *self._platform_arguments(),
                self._value(reference),
            ),
            "image_inspect",
            cancellation,
        )

    def remove_image(
        self,
        reference: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Remove one validated private image reference.

        Args:
            reference (str): Digest-pinned private image reference.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw removal result.

        Raises:
            ValueError: If the reference could be interpreted as an option or malformed value.
        """
        if re.fullmatch(r"loop\.local/sandbox@sha256:[0-9a-f]{64}", reference) is None:
            raise ValueError("OCI image removal requires a private immutable reference.")
        return self._run(("image", "rm", "--force", reference), "image_remove", cancellation)

    def create_attempt(
        self,
        spec: OciExecutionSpec,
        workspace_source: str,
        *,
        bindings: OciAttemptBindings | None = None,
        queue_limit: int = 128,
        deadline_seconds: float = 30.0,
    ) -> InfrastructureProcessResult:
        """Create an attempt without starting its untrusted process.

        Args:
            spec (OciExecutionSpec): Closed authorized OCI specification.
            workspace_source (str): Platform-resolved immutable runtime mount source.
            bindings (OciAttemptBindings | None): Prepared effect-broker resources.
            queue_limit (int): Positive session bound validated before creation.
            deadline_seconds (float): Positive wall bound validated before creation.

        Returns:
            InfrastructureProcessResult: Bounded container-creation evidence.

        Raises:
            ValueError: If a value or session bound is malformed.
        """
        if queue_limit <= 0 or deadline_seconds <= 0:
            raise ValueError("OCI control session bounds are invalid.")
        return self._run(
            self._attempt_arguments("create", spec, workspace_source, bindings, detached=False),
            "attempt_create",
            lambda: False,
        )

    def run_job(
        self,
        spec: OciExecutionSpec,
        workspace_source: str,
        *,
        queue_limit: int = 128,
        deadline_seconds: float = 30.0,
    ) -> OciControlSession:
        """Create and start one durable job through an attachable interactive session.

        Args:
            spec (OciExecutionSpec): Closed durable OCI specification.
            workspace_source (str): Platform-resolved immutable runtime mount source.
            queue_limit (int): Maximum unread frames before fail-closed cleanup.
            deadline_seconds (float): Positive wall deadline for initial detachment.

        Returns:
            OciControlSession: Initial attached session that must detach gracefully.

        Raises:
            ValueError: If the specification or session bounds are invalid.
        """
        if queue_limit <= 0 or deadline_seconds <= 0:
            raise ValueError("OCI control session bounds are invalid.")
        return self._transport.open_oci_session(
            (
                *self._prefix(),
                *self._attempt_arguments("run", spec, workspace_source, None, detached=True),
            ),
            "job_run",
            spec.terminal.value == "pty",
            queue_limit,
            deadline_seconds,
            spec.terminal_columns if spec.terminal.value == "pty" else None,
            spec.terminal_rows if spec.terminal.value == "pty" else None,
        )

    def start_attempt(
        self,
        spec: OciExecutionSpec,
        *,
        queue_limit: int = 128,
        deadline_seconds: float = 30.0,
    ) -> OciControlSession:
        """Start and attach to one already-created attempt.

        Args:
            spec (OciExecutionSpec): Closed authorized OCI specification.
            queue_limit (int): Maximum unread frames before fail-closed cleanup.
            deadline_seconds (float): Positive wall deadline.

        Returns:
            OciControlSession: Owned attached process handle.

        Raises:
            ValueError: If a session bound is invalid.
        """
        if queue_limit <= 0 or deadline_seconds <= 0:
            raise ValueError("OCI control session bounds are invalid.")
        arguments = (
            *self._prefix(),
            "start",
            "--attach",
            *(("--interactive",) if spec.attach_stdin or spec.terminal.value == "pty" else ()),
            self._value(spec.container_name),
        )
        return self._transport.open_oci_session(
            arguments,
            "attempt_start",
            spec.terminal.value == "pty",
            queue_limit,
            deadline_seconds,
            spec.terminal_columns if spec.terminal.value == "pty" else None,
            spec.terminal_rows if spec.terminal.value == "pty" else None,
        )

    def run_attempt(
        self,
        spec: OciExecutionSpec,
        workspace_source: str,
        *,
        bindings: OciAttemptBindings | None = None,
        queue_limit: int = 128,
        deadline_seconds: float = 30.0,
    ) -> OciControlSession:
        """Run and attach to one closed attempt without a pre-start broker phase."""
        if queue_limit <= 0 or deadline_seconds <= 0:
            raise ValueError("OCI control session bounds are invalid.")
        return self._transport.open_oci_session(
            (
                *self._prefix(),
                *self._attempt_arguments("run", spec, workspace_source, bindings, detached=False),
            ),
            "attempt_run",
            spec.terminal.value == "pty",
            queue_limit,
            deadline_seconds,
            spec.terminal_columns if spec.terminal.value == "pty" else None,
            spec.terminal_rows if spec.terminal.value == "pty" else None,
        )

    def _attempt_arguments(
        self,
        operation: str,
        spec: OciExecutionSpec,
        workspace_source: str,
        bindings: OciAttemptBindings | None,
        *,
        detached: bool,
    ) -> tuple[str, ...]:
        """Compile fixed create/run arguments for one closed attempt."""
        if operation not in {"create", "run"} or spec.detached is not detached:
            raise ValueError("OCI execution lifetime does not match its control operation.")
        mount = (
            f"type=bind,src={self._absolute_path(workspace_source)},"
            f"dst={spec.workspace_mount.destination}"
            + (",readonly" if spec.workspace_mount.read_only else "")
        )
        arguments = [
            operation,
            *(("-i",) if (detached or spec.attach_stdin) and spec.terminal.value != "pty" else ()),
            *(("-i", "-t") if spec.terminal.value == "pty" else ()),
            "--name",
            self._value(spec.container_name),
            "--pull=never",
            "--network",
            self._value(bindings.network_name)
            if bindings is not None and bindings.network_name is not None
            else spec.network_mode,
            "--read-only",
            "--userns",
            "host",
            "--user",
            "1000:1000",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            str(spec.limits.memory_bytes),
            "--pids-limit",
            str(spec.limits.pids),
            "--cpu-quota",
            str(spec.limits.cpu_quota_us),
            "--ulimit",
            f"nofile={spec.limits.open_files}:{spec.limits.open_files}",
            "--workdir",
            spec.cwd,
            "--mount",
            mount,
            "--tmpfs",
            f"/var/tmp:rw,nosuid,nodev,exec,mode=1777,size={spec.limits.disk_bytes}",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,exec,mode=1777,size={spec.limits.temporary_bytes}",
            "--tmpfs",
            f"/cache:rw,nosuid,nodev,noexec,mode=700,size={spec.limits.cache_bytes}",
            "--tmpfs",
            "/home/agent:rw,nosuid,nodev,noexec,mode=700,uid=1000,gid=1000,size=16777216",
        ]
        if bindings is not None:
            if bindings.command_address is not None:
                arguments.extend(("--ip", self._value(bindings.command_address)))
            for hostname, address in bindings.host_aliases:
                arguments.extend(("--add-host", f"{self._value(hostname)}:{self._value(address)}"))
            for address in bindings.dns_servers:
                arguments.extend(("--dns", self._value(address)))
            if bindings.environment_file is not None:
                arguments.extend(("--env-file", self._absolute_path(bindings.environment_file)))
            for source, destination in bindings.secret_mounts:
                arguments.extend(
                    (
                        "--mount",
                        (
                            f"type=bind,src={self._absolute_path(source)},"
                            f"dst={self._absolute_path(destination)},readonly"
                        ),
                    )
                )
            if bindings.start_gate is not None:
                arguments.extend(
                    (
                        "--mount",
                        (
                            f"type=bind,src={self._absolute_path(bindings.start_gate)},"
                            "dst=/run/loop/start-gate,readonly"
                        ),
                    )
                )
        for key, value in spec.environment:
            arguments.extend(("--env", f"{key}={value}"))
        for key, value in spec.labels:
            arguments.extend(("--label", f"{key}={value}"))
        command = spec.argv
        if bindings is not None and bindings.start_gate is not None:
            command = (
                "/bin/sh",
                "-c",
                'until test "$(cat /run/loop/start-gate)" = go; do sleep 0.01; done; exec "$@"',
                "loop-start-gate",
                *spec.argv,
            )
        arguments.extend((self._value(spec.image_reference), *command))
        return tuple(arguments)

    def start_broker(
        self,
        container_id: str,
        image_reference: str,
        configuration_path: str,
    ) -> InfrastructureProcessResult:
        """Start one capability-proportional Envoy sidecar in the trusted runtime network.

        Args:
            container_id (str): Lease-bound broker container identity.
            image_reference (str): Digest-pinned official Envoy image.
            configuration_path (str): Trusted guest bootstrap path.

        Returns:
            InfrastructureProcessResult: Bounded detached-start result.

        Raises:
            ValueError: If any value could alter the fixed command shape.
        """
        mount = (
            f"type=bind,src={self._absolute_path(configuration_path)},"
            "dst=/etc/envoy/envoy.json,readonly"
        )
        return self._run(
            (
                "run",
                "--detach",
                "--name",
                self._value(container_id),
                "--pull=never",
                "--network",
                self._value(f"ns:{self._endpoint.network_namespace_path}"),
                "--read-only",
                # Container UID 0 maps to the unprivileged runtime owner. The bootstrap
                # leaf is readable only through its owner-private guest directory or this mount.
                "--user",
                "0:0",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--memory",
                "134217728",
                "--pids-limit",
                "64",
                "--ulimit",
                "nofile=1024:1024",
                "--mount",
                mount,
                "--entrypoint",
                "/usr/local/bin/envoy",
                self._value(image_reference),
                "-c",
                "/etc/envoy/envoy.json",
                "--concurrency",
                "1",
                "--disable-hot-restart",
            ),
            "broker_start",
            lambda: False,
        )

    def inspect_container_pid(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Inspect the task PID for one authenticated running container.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded decimal task PID evidence.

        Raises:
            ValueError: If the container identity is malformed.
        """
        return self._container_command(
            ("container", "inspect", "--format", "{{.State.Pid}}"),
            "container_pid_inspect",
            container_id,
            cancellation,
        )

    def inspect_container_hosts_path(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Inspect the generated hosts-file path for one stopped container.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded absolute hosts-path evidence.

        Raises:
            ValueError: If the container identity is malformed.
        """
        return self._container_command(
            ("container", "inspect", "--format", "{{.HostsPath}}"),
            "container_hosts_path_inspect",
            container_id,
            cancellation,
        )

    def initialize_network(
        self,
        container_id: str,
        network_name: str,
        command_address: str,
        image_reference: str,
    ) -> InfrastructureProcessResult:
        """Create a CNI bridge using one trusted, immediately removed core container.

        Args:
            container_id (str): Lease-derived initializer container identity.
            network_name (str): Lease-specific CNI network identity.
            command_address (str): Fixed temporary address released before command start.
            image_reference (str): Digest-pinned trusted core image reference.

        Returns:
            InfrastructureProcessResult: Bounded one-shot initialization result.

        Raises:
            ValueError: If any value could alter the fixed command shape.
        """
        return self._run(
            (
                "run",
                "--rm",
                "--name",
                self._value(container_id),
                "--pull=never",
                "--network",
                self._value(network_name),
                "--ip",
                self._value(command_address),
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                self._value(image_reference),
                "/bin/true",
            ),
            "network_initialize",
            lambda: False,
        )

    def inspect_container(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Inspect one authenticated container identity.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw inspection evidence.

        Raises:
            ValueError: If the container identity is malformed.
        """
        return self._container_command(
            ("container", "inspect", "--format", "{{json .}}"),
            "container_inspect",
            container_id,
            cancellation,
        )

    def inspect_container_state(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Inspect only terminal state for one authenticated container identity.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw state evidence.

        Raises:
            ValueError: If the container identity is malformed.
        """
        return self._container_command(
            ("container", "inspect", "--format", "{{json .State}}"),
            "container_state_inspect",
            container_id,
            cancellation,
        )

    def remove_container(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Force-remove one authenticated container identity.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw removal result.

        Raises:
            ValueError: If the container identity is malformed.
        """
        return self._container_command(("rm", "-f"), "container_remove", container_id, cancellation)

    def wait_container(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Wait for one authenticated container identity to terminate.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw wait result.

        Raises:
            ValueError: If the container identity is malformed.
        """
        return self._container_command(
            ("container", "wait"), "container_wait", container_id, cancellation
        )

    def stop_container(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Request bounded graceful stop for one authenticated container identity.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw stop result.

        Raises:
            ValueError: If the container identity is malformed.
        """
        return self._container_command(
            ("container", "stop", "--time", "1"),
            "container_stop",
            container_id,
            cancellation,
        )

    def kill_container(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Kill one authenticated container identity.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw kill result.

        Raises:
            ValueError: If the container identity is malformed.
        """
        return self.signal_container(container_id, OciSignal.KILL, cancellation)

    def signal_container(
        self,
        container_id: str,
        signal: OciSignal,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Send one closed-set signal to an authenticated container task.

        Args:
            container_id (str): Authenticated container identity.
            signal (OciSignal): Reviewed signal delivered by containerd.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw signal result.

        Raises:
            ValueError: If the container identity is malformed.
        """
        return self._container_command(
            ("container", "kill", "--signal", signal.value),
            "container_signal",
            container_id,
            cancellation,
        )

    def pause_container(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Pause one authenticated container identity.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw pause result.

        Raises:
            ValueError: If the container identity is malformed.
        """
        return self._container_command(
            ("container", "pause"), "container_pause", container_id, cancellation
        )

    def unpause_container(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Unpause one authenticated container identity.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw unpause result.

        Raises:
            ValueError: If the container identity is malformed.
        """
        return self._container_command(
            ("container", "unpause"), "container_unpause", container_id, cancellation
        )

    def start_attached(
        self,
        container_id: str,
        *,
        pty: bool = False,
        queue_limit: int = 128,
        deadline_seconds: float = 30.0,
    ) -> OciControlSession:
        """Start and attach to one authenticated container identity.

        Args:
            container_id (str): Authenticated container identity.
            pty (bool): Whether to expose one merged pseudo-terminal stream.
            queue_limit (int): Maximum unread frames before fail-closed cleanup.
            deadline_seconds (float): Positive wall deadline.

        Returns:
            OciControlSession: Owned attached process handle.

        Raises:
            ValueError: If the identity or session bounds are invalid.
        """
        return self._open_session(
            ("start", "--attach"),
            "container_start_attached",
            container_id,
            pty,
            queue_limit,
            deadline_seconds,
            80 if pty else None,
            24 if pty else None,
        )

    def attach(
        self,
        container_id: str,
        *,
        pty: bool = False,
        queue_limit: int = 128,
        deadline_seconds: float = 30.0,
    ) -> OciControlSession:
        """Attach to one authenticated running container identity.

        Args:
            container_id (str): Authenticated container identity.
            pty (bool): Whether to expose one merged pseudo-terminal stream.
            queue_limit (int): Maximum unread frames before fail-closed cleanup.
            deadline_seconds (float): Positive wall deadline.

        Returns:
            OciControlSession: Owned attached process handle.

        Raises:
            ValueError: If the identity or session bounds are invalid.
        """
        return self._open_session(
            ("attach",),
            "container_attach",
            container_id,
            pty,
            queue_limit,
            deadline_seconds,
            80 if pty else None,
            24 if pty else None,
        )

    def _container_command(
        self,
        shape: tuple[str, ...],
        operation: str,
        container_id: str,
        cancellation: Callable[[], bool],
    ) -> InfrastructureProcessResult:
        """Run one fixed container command after validating its sole data slot."""
        return self._run((*shape, self._value(container_id)), operation, cancellation)

    def _run(
        self,
        arguments: tuple[str, ...],
        operation: str,
        cancellation: Callable[[], bool],
    ) -> InfrastructureProcessResult:
        """Run one compiled nerdctl invocation through the platform transport."""
        return self._transport.run_oci_command(
            (*self._prefix(), *arguments), operation, cancellation
        )

    def _open_session(
        self,
        shape: tuple[str, ...],
        operation: str,
        container_id: str,
        pty: bool,
        queue_limit: int,
        deadline_seconds: float,
        terminal_columns: int | None,
        terminal_rows: int | None,
    ) -> OciControlSession:
        """Open one fixed attached operation after validating all bounds."""
        if queue_limit <= 0 or deadline_seconds <= 0:
            raise ValueError("OCI control session bounds are invalid.")
        return self._transport.open_oci_session(
            (*self._prefix(), *shape, self._value(container_id)),
            operation,
            pty,
            queue_limit,
            deadline_seconds,
            terminal_columns,
            terminal_rows,
        )

    def _prefix(self) -> tuple[str, ...]:
        """Return the attested private runtime selector."""
        return (
            "/usr/local/bin/nerdctl",
            "--address",
            self._endpoint.socket_path,
            "--namespace",
            self._endpoint.namespace,
        )

    def _platform_arguments(self) -> tuple[str, ...]:
        """Return the exact selected image platform."""
        return (
            "--platform",
            f"{self._endpoint.platform.os}/{self._endpoint.platform.architecture}",
        )

    @staticmethod
    def _value(value: str) -> str:
        """Return one validated opaque operation value."""
        if not _VALUE.fullmatch(value):
            raise ValueError("OCI control operation has an invalid typed value.")
        return value

    @staticmethod
    def _absolute_path(value: str) -> str:
        """Return one validated platform-resolved absolute runtime path."""
        if (
            not value.startswith("/")
            or ".." in value.split("/")
            or any(character in value for character in ("\x00", ",", "\n", "\r"))
        ):
            raise ValueError("OCI runtime mount source is invalid.")
        return value
