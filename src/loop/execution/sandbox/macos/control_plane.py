"""Route closed OCI control operations through the attested managed Lima guest."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ....utils import sha256_digest
from ...infrastructure import InfrastructureProcessResult
from ...runtime.manifest import RuntimeManifest
from ...runtime.models import PlatformSelector
from ..oci.control import (
    GuestOciRuntimeEndpoint,
    OciAttemptBindings,
    OciControlSession,
    OciSignal,
    OciSupervisor,
)
from ..oci.process import OciProcessSession
from ..oci.spec import OciExecutionSpec
from .candidate import macos_artifact, macos_component_version
from .lima import LimaClient, LimaInstanceContext, ManagedLimaInstance


class MacosRuntimeNotReadyError(ValueError):
    """Report valid startup evidence whose rootless network namespace is not stable yet."""


@dataclass(frozen=True, slots=True)
class SandboxImageBuildResult:
    """Return bounded guest image-build evidence with its attested platform.

    Args:
        process (InfrastructureProcessResult): Complete fixed-shape management result.
        platform (PlatformSelector): Exact native guest platform used for construction.
    """

    process: InfrastructureProcessResult
    platform: PlatformSelector

    @property
    def exit_code(self) -> int:
        """Return the underlying process exit code.

        Returns:
            int: Guest management process exit code.
        """
        return self.process.exit_code

    @property
    def stdout(self) -> bytes:
        """Return complete bounded standard output.

        Returns:
            bytes: Guest build evidence frames.
        """
        return self.process.stdout

    @property
    def stderr(self) -> bytes:
        """Return complete bounded standard error.

        Returns:
            bytes: Guest build diagnostics.
        """
        return self.process.stderr

    @property
    def stdout_truncated(self) -> bool:
        """Return whether standard output exceeded its bound.

        Returns:
            bool: Whether evidence is incomplete.
        """
        return self.process.stdout_truncated

    @property
    def stderr_truncated(self) -> bool:
        """Return whether standard error exceeded its bound.

        Returns:
            bool: Whether diagnostics were incomplete.
        """
        return self.process.stderr_truncated


class MacosControlPlane:
    """Implement common OCI control through one running attested Lima instance.

    Args:
        runner (LimaClient): Sole macOS Lima control client.
        instance (ManagedLimaInstance): Journal-owned instance used for live attestation.
        context (LimaInstanceContext): Lease-bound private Lima invocation context.
    """

    _runner: LimaClient
    _instance: ManagedLimaInstance
    _context: LimaInstanceContext
    _endpoint: GuestOciRuntimeEndpoint | None
    _supervisor: OciSupervisor | None

    def __init__(
        self,
        runner: LimaClient,
        instance: ManagedLimaInstance,
        context: LimaInstanceContext,
    ) -> None:
        self._runner = runner
        self._instance = instance
        self._context = context
        self._endpoint: GuestOciRuntimeEndpoint | None = None
        self._supervisor: OciSupervisor | None = None

    def attest_runtime(self, candidate: RuntimeManifest) -> GuestOciRuntimeEndpoint:
        """Attest the candidate-bound rootless guest runtime and endpoint.

        Args:
            candidate (RuntimeManifest): Closed candidate supplying the runtime digest.

        Returns:
            GuestOciRuntimeEndpoint: Validated guest endpoint and version evidence.

        Raises:
            ValueError: If runtime output is truncated, unsuccessful, malformed, or unsafe.
            LimaConfigurationError: If the managed instance is no longer attested.
        """
        self._instance.attest_running()
        result = self._runner.run_guest(
            self._context,
            ("/bin/sh", "-c", _RUNTIME_ATTESTATION_SCRIPT),
            operation="oci.attest",
            deadline_seconds=60.0,
        )
        if result.exit_code or result.stdout_truncated or result.stderr_truncated:
            raise ValueError("Managed guest OCI runtime attestation failed.")
        try:
            lines = result.stdout.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError("Managed guest OCI runtime evidence is malformed.") from error
        if len(lines) != 16 or any("=" not in line for line in lines):
            raise ValueError("Managed guest OCI runtime evidence is malformed.")
        evidence = dict(line.split("=", 1) for line in lines)
        if set(evidence) != {
            "owner_uid",
            "socket_path",
            "socket_is_socket",
            "socket_uid",
            "state_path",
            "state_exists",
            "content_path",
            "content_exists",
            "rootful_socket_exists",
            "network_namespace_path",
            "network_namespace_identity",
            "entered_network_namespace_identity",
            "nerdctl_version",
            "containerd_version",
            "runc_version",
            "buildkit_version",
        }:
            raise ValueError("Managed guest OCI runtime evidence is malformed.")
        try:
            owner_uid = int(evidence["owner_uid"])
            socket_uid = int(evidence["socket_uid"])
        except ValueError as error:
            raise ValueError("Managed guest OCI runtime evidence is malformed.") from error
        if (
            owner_uid != socket_uid
            or evidence["socket_is_socket"] != "true"
            or evidence["state_exists"] != "true"
            or evidence["rootful_socket_exists"] != "false"
        ):
            raise ValueError("Managed guest runtime ownership evidence is invalid.")
        if evidence["network_namespace_identity"] != evidence["entered_network_namespace_identity"]:
            raise MacosRuntimeNotReadyError(
                "Managed guest rootless network namespace identity is not stable."
            )
        if (
            evidence["nerdctl_version"]
            != f"nerdctl version {macos_component_version(candidate, 'nerdctl')}"
            or re.fullmatch(
                "containerd github.com/containerd/containerd/v2 v"
                + re.escape(macos_component_version(candidate, "containerd"))
                + r" [0-9a-f]{40}",
                evidence["containerd_version"],
            )
            is None
            or evidence["runc_version"]
            != f"runc version {macos_component_version(candidate, 'runc')}"
            or re.fullmatch(
                r"buildkitd github\.com/moby/buildkit v"
                + re.escape(macos_component_version(candidate, "buildkit"))
                + r" [0-9a-f]{7,40}",
                evidence["buildkit_version"],
            )
            is None
        ):
            raise ValueError("Managed guest runtime versions do not match the candidate.")
        identity_payload = "\n".join(
            (
                macos_artifact(candidate, "nerdctl-full").digest.removeprefix("sha256:"),
                *[evidence[key] for key in sorted(evidence)],
                "loop-private",
            )
        )
        endpoint = GuestOciRuntimeEndpoint(
            socket_path=evidence["socket_path"],
            state_path=evidence["state_path"],
            content_path=evidence["content_path"],
            namespace="loop-private",
            content_store_identity="sha256:" + sha256_digest(identity_payload),
            platform=PlatformSelector(os="linux", architecture="arm64"),
            artifact_set_digest=self._context.artifact_set_digest,
            owner_uid=owner_uid,
            network_namespace_path=evidence["network_namespace_path"],
            network_namespace_identity=evidence["network_namespace_identity"],
            nerdctl_version=evidence["nerdctl_version"],
            containerd_version=evidence["containerd_version"],
            runc_version=evidence["runc_version"],
            buildkit_version=evidence["buildkit_version"],
        )
        self._endpoint = endpoint
        self._supervisor = OciSupervisor(endpoint, LimaOciTransport(self._runner, self._context))
        return endpoint

    def version(
        self, cancellation: Callable[[], bool] = lambda: False
    ) -> InfrastructureProcessResult:
        """Return version evidence from the reattested guest runtime.

        Args:
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw version evidence.

        Raises:
            ValueError: If the guest runtime has not been attested.
        """
        return self._live_supervisor().version(cancellation)

    def pull_image(
        self,
        reference: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Pull one immutable image through the reattested guest runtime.

        Args:
            reference (str): Digest-pinned image reference.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw pull result.

        Raises:
            ValueError: If the runtime is unattested or the reference is malformed.
        """
        return self._live_supervisor().pull_image(reference, cancellation)

    def inspect_image(
        self,
        reference: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Inspect one immutable image through the reattested guest runtime.

        Args:
            reference (str): Digest-pinned image reference.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw inspection evidence.

        Raises:
            ValueError: If the runtime is unattested or the reference is malformed.
        """
        return self._live_supervisor().inspect_image(reference, cancellation)

    def remove_image(
        self,
        reference: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Remove one obsolete private sandbox image.

        Args:
            reference (str): Digest-pinned private sandbox image reference.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw removal result.
        """
        return self._live_supervisor().remove_image(reference, cancellation)

    def build_sandbox_image(
        self,
        source_archive: Path,
        image_tag: str,
        base_image: str,
        debian_snapshot: str,
        packages: str,
        tools: tuple[str, ...],
        commands: tuple[str, ...],
    ) -> SandboxImageBuildResult:
        """Build the declared sandbox image with transient bundled BuildKit.

        Args:
            source_archive (Path): Private trusted Containerfile/catalog archive.
            image_tag (str): Private local image tag receiving the build result.
            base_image (str): Exact digest-pinned base image expected by the source.
            debian_snapshot (str): Exact immutable Debian snapshot identifier.
            packages (str): Sorted exact package assignments.
            tools (tuple[str, ...]): Validated external-tool build arguments.
            commands (tuple[str, ...]): Declared executable names to resolve inside the result.

        Returns:
            SandboxImageBuildResult: Framed descriptor, inventory, and cleanup evidence.

        Raises:
            ValueError: If any source, identity, or closed catalog value is malformed.
        """
        supervisor = self._live_supervisor()
        endpoint = self._endpoint
        if endpoint is None:  # pragma: no cover - guarded by _live_supervisor.
            raise ValueError("OCI control requires an attested guest runtime endpoint.")
        try:
            source = source_archive.resolve(strict=True)
            source.relative_to(self._context.state_root)
        except (OSError, ValueError) as error:
            raise ValueError("Sandbox image source is outside private Lima state.") from error
        if source.is_symlink() or not source.is_file():
            raise ValueError("Sandbox image source must be one private regular file.")
        package = re.compile(r"^[A-Za-z0-9][A-Za-z0-9+.-]*=[A-Za-z0-9][A-Za-z0-9.+:~_-]*$")
        if (
            re.fullmatch(r"^loop\.local/sandbox:[0-9a-f]{24}$", image_tag) is None
            or re.fullmatch(r"^docker\.io/library/debian@sha256:[0-9a-f]{64}$", base_image) is None
            or re.fullmatch(r"^[0-9]{8}T[0-9]{6}Z$", debian_snapshot) is None
            or not packages
            or any(package.fullmatch(item) is None for item in packages.split())
            or not _sandbox_tool_arguments_are_valid(tools)
            or not commands
            or any(re.fullmatch(r"^[A-Za-z0-9+_.-]+$", command) is None for command in commands)
        ):
            raise ValueError("Sandbox image build values are invalid.")
        source_digest = sha256_digest(source.read_bytes())
        owner_home = endpoint.state_path.partition("/.local/")[0]
        guest_parent = f"{owner_home}/.loop-build"
        guest_archive = f"{guest_parent}/{source_digest[:24]}-sandbox.tar"
        prepared = self._runner.run_guest(
            self._context,
            ("/bin/mkdir", "-p", guest_parent),
            operation="sandbox_image.prepare_source",
        )
        if prepared.exit_code or prepared.stdout_truncated or prepared.stderr_truncated:
            raise RuntimeError("Sandbox image build source could not be prepared.")
        copied = self._runner.copy_to_guest(self._context, source, guest_archive)
        if copied.exit_code or copied.stdout_truncated or copied.stderr_truncated:
            self._runner.run_guest(
                self._context,
                ("/bin/rm", "-f", "--", guest_archive),
                operation="sandbox_image.cleanup_source",
            )
            raise RuntimeError("Sandbox image build source could not be transferred.")
        inventory_script = _sandbox_image_inventory_script(commands)
        result = self._runner.run_guest(
            self._context,
            (
                "/bin/sh",
                "-c",
                _SANDBOX_IMAGE_BUILD_SCRIPT,
                "loop-sandbox-image-build",
                str(endpoint.owner_uid),
                endpoint.socket_path,
                endpoint.namespace,
                guest_archive,
                source_digest,
                image_tag,
                base_image,
                debian_snapshot,
                packages,
                " ".join(tools),
                inventory_script,
            ),
            operation="sandbox_image.build",
            deadline_seconds=1800.0,
        )
        # Reattest the runtime after the transient builder has exited before accepting evidence.
        version = supervisor.version()
        if version.exit_code or version.stdout_truncated or version.stderr_truncated:
            raise RuntimeError("Managed runtime could not be reattested after image build.")
        return SandboxImageBuildResult(result, endpoint.platform)

    def run_attempt(
        self,
        spec: OciExecutionSpec,
        workspace_source: str,
        *,
        bindings: OciAttemptBindings | None = None,
        queue_limit: int = 128,
        deadline_seconds: float = 30.0,
    ) -> OciControlSession:
        """Start one attached attempt over an immutable shared generation.

        Args:
            spec (OciExecutionSpec): Closed authorized OCI specification.
            workspace_source (str): Trusted absolute guest attempt mount source.
            bindings (OciAttemptBindings | None): Prepared broker resources.
            queue_limit (int): Maximum unread output frames.
            deadline_seconds (float): Positive wall deadline.

        Returns:
            OciControlSession: Attached foreground process handle.

        Raises:
            ValueError: If the runtime is unattested or the source path is malformed.
        """
        if (
            not workspace_source.startswith("/")
            or ".." in workspace_source.split("/")
            or not (
                re.fullmatch(
                    r"/run/loop/snapshots/[A-Za-z0-9][A-Za-z0-9._-]{0,127}/tree",
                    workspace_source,
                )
                or re.fullmatch(
                    r"/home/[A-Za-z0-9._-]+/\.local/share/loop/workspaces/attempts/"
                    r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/merged",
                    workspace_source,
                )
            )
        ):
            raise ValueError("Managed workspace attempt source is invalid.")
        return self._live_supervisor().run_attempt(
            spec,
            workspace_source,
            bindings=bindings,
            queue_limit=queue_limit,
            deadline_seconds=deadline_seconds,
        )

    def create_attempt(
        self,
        spec: OciExecutionSpec,
        workspace_source: str,
        *,
        bindings: OciAttemptBindings | None = None,
        queue_limit: int = 128,
        deadline_seconds: float = 30.0,
    ) -> InfrastructureProcessResult:
        """Create a networked attempt without starting untrusted code."""
        if (
            not workspace_source.startswith("/")
            or ".." in workspace_source.split("/")
            or not (
                re.fullmatch(
                    r"/run/loop/snapshots/[A-Za-z0-9][A-Za-z0-9._-]{0,127}/tree",
                    workspace_source,
                )
                or re.fullmatch(
                    r"/home/[A-Za-z0-9._-]+/\.local/share/loop/workspaces/attempts/"
                    r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/merged",
                    workspace_source,
                )
            )
        ):
            raise ValueError("Managed workspace attempt source is invalid.")
        return self._live_supervisor().create_attempt(
            spec,
            workspace_source,
            bindings=bindings,
            queue_limit=queue_limit,
            deadline_seconds=deadline_seconds,
        )

    def start_attempt(
        self,
        spec: OciExecutionSpec,
        *,
        queue_limit: int = 128,
        deadline_seconds: float = 30.0,
    ) -> OciControlSession:
        """Start and attach to one already-created networked attempt."""
        return self._live_supervisor().start_attempt(
            spec,
            queue_limit=queue_limit,
            deadline_seconds=deadline_seconds,
        )

    def run_job(
        self,
        spec: OciExecutionSpec,
        workspace_source: str,
        *,
        queue_limit: int = 128,
        deadline_seconds: float = 30.0,
    ) -> OciControlSession:
        """Create and start one durable job through its initial attached session.

        Args:
            spec (OciExecutionSpec): Closed durable OCI specification.
            workspace_source (str): Trusted absolute guest attempt mount source.
            queue_limit (int): Maximum unread output frames.
            deadline_seconds (float): Positive bound for initial detachment.

        Returns:
            OciControlSession: Initial attached session that must detach gracefully.

        Raises:
            ValueError: If the runtime is unattested or the source path is malformed.
        """
        if (
            re.fullmatch(
                r"/home/[A-Za-z0-9._-]+/\.local/share/loop/workspaces/attempts/"
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/merged",
                workspace_source,
            )
            is None
        ):
            raise ValueError("Managed durable workspace source is invalid.")
        return self._live_supervisor().run_job(
            spec,
            workspace_source,
            queue_limit=queue_limit,
            deadline_seconds=deadline_seconds,
        )

    def start_broker(
        self,
        container_id: str,
        image_reference: str,
        configuration_path: str,
    ) -> InfrastructureProcessResult:
        """Start one lease-specific Envoy sidecar through the OCI supervisor."""
        return self._live_supervisor().start_broker(
            container_id,
            image_reference,
            configuration_path,
        )

    def container_uses_rootless_network_namespace(self, container_id: str) -> bool:
        """Return whether a running container shares the attested RootlessKit namespace.

        Args:
            container_id (str): Authenticated running container identity.

        Returns:
            bool: Whether the task and detached namespace identities match exactly.

        Raises:
            ValueError: If PID or namespace evidence is unavailable or malformed.
        """
        endpoint = self._live_endpoint()
        pid_result = self._live_supervisor().inspect_container_pid(container_id)
        if (
            pid_result.exit_code
            or pid_result.stdout_truncated
            or pid_result.stderr_truncated
            or not pid_result.stdout.strip().isdigit()
        ):
            raise ValueError("Container network namespace PID evidence is malformed.")
        pid = int(pid_result.stdout.strip())
        if pid <= 0:
            raise ValueError("Container network namespace PID evidence is malformed.")
        result = self._runner.run_guest(
            self._context,
            ("/usr/bin/stat", "-Lc", "%d:%i", f"/proc/{pid}/ns/net"),
            operation="broker.container_network_namespace",
        )
        if result.exit_code or result.stdout_truncated or result.stderr_truncated:
            raise ValueError("Container network namespace evidence is unavailable.")
        try:
            identity = result.stdout.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ValueError("Container network namespace evidence is malformed.") from error
        if re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", identity) is None:
            raise ValueError("Container network namespace evidence is malformed.")
        return identity == endpoint.network_namespace_identity

    def harden_container_hosts(self, container_id: str) -> InfrastructureProcessResult:
        """Make nerdctl's generated hosts mapping immutable before task start.

        Args:
            container_id (str): Authenticated stopped command-container identity.

        Returns:
            InfrastructureProcessResult: Exact resulting file-mode evidence.

        Raises:
            ValueError: If generated hosts-path evidence is unavailable or outside runtime state.
        """
        path = self._container_hosts_path(container_id)
        return self._runner.run_guest(
            self._context,
            (
                "/bin/sh",
                "-c",
                'set -eu; /bin/chmod 0444 -- "$1"; /usr/bin/stat -c %a -- "$1"',
                "loop-harden-hosts",
                path,
            ),
            operation="broker.harden_hosts",
        )

    def release_container_hosts(self, container_id: str) -> InfrastructureProcessResult:
        """Restore trusted write access needed for nerdctl container cleanup.

        Args:
            container_id (str): Authenticated stopped command-container identity.

        Returns:
            InfrastructureProcessResult: Exact resulting file-mode evidence.

        Raises:
            ValueError: If generated hosts-path evidence is unavailable or outside runtime state.
        """
        path = self._container_hosts_path(container_id)
        return self._runner.run_guest(
            self._context,
            (
                "/bin/sh",
                "-c",
                'set -eu; /bin/chmod 0600 -- "$1"; /usr/bin/stat -c %a -- "$1"',
                "loop-release-hosts",
                path,
            ),
            operation="broker.release_hosts",
        )

    def open_container_start_gate(self, guest_path: str) -> InfrastructureProcessResult:
        """Release a prepared command only after hosts hardening passes.

        Args:
            guest_path (str): Lease-private gate source already mounted read-only in the command.

        Returns:
            InfrastructureProcessResult: Bounded trusted staging evidence.
        """
        return self.write_broker_file(guest_path, b"go\n")

    def _container_hosts_path(self, container_id: str) -> str:
        """Return one validated nerdctl-private generated hosts path."""
        endpoint = self._live_endpoint()
        path_result = self._live_supervisor().inspect_container_hosts_path(container_id)
        if path_result.exit_code or path_result.stdout_truncated or path_result.stderr_truncated:
            raise ValueError("Container hosts-path evidence is unavailable.")
        try:
            path = path_result.stdout.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise ValueError("Container hosts-path evidence is malformed.") from error
        owner_home = endpoint.state_path.removesuffix("/.local/share/containerd")
        if (
            not path.startswith(f"{owner_home}/.local/share/nerdctl/")
            or not path.endswith("/hosts")
            or ".." in path.split("/")
        ):
            raise ValueError("Container hosts-path evidence is outside private runtime state.")
        return path

    def initialize_network(
        self,
        container_id: str,
        interface_name: str,
        namespace_path: str,
        configuration_path: str,
    ) -> InfrastructureProcessResult:
        """Create one bridge attachment through the pinned upstream CNI plugin.

        Args:
            container_id (str): Lease-derived initializer identity.
            interface_name (str): Lease-derived initializer interface identity.
            namespace_path (str): Lease-owned persistent target network namespace.
            configuration_path (str): Trusted bridge-plugin configuration path.

        Returns:
            InfrastructureProcessResult: Bounded initialization evidence.
        """
        endpoint = self._live_endpoint()
        return self._runner.run_guest(
            self._context,
            (
                "/bin/sh",
                "-c",
                _CNI_ATTACHMENT_SCRIPT,
                "loop-cni-add",
                str(endpoint.owner_uid),
                "ADD",
                container_id,
                interface_name,
                namespace_path,
                configuration_path,
            ),
            operation="broker.cni_add",
        )

    def remove_network_attachment(
        self,
        container_id: str,
        interface_name: str,
        namespace_path: str,
        configuration_path: str,
    ) -> InfrastructureProcessResult:
        """Remove one exact bridge attachment through the pinned CNI plugin."""
        endpoint = self._live_endpoint()
        return self._runner.run_guest(
            self._context,
            (
                "/bin/sh",
                "-c",
                _CNI_ATTACHMENT_SCRIPT,
                "loop-cni-del",
                str(endpoint.owner_uid),
                "DEL",
                container_id,
                interface_name,
                namespace_path,
                configuration_path,
            ),
            operation="broker.cni_del",
        )

    def harden_network_namespace(self) -> InfrastructureProcessResult:
        """Disable forwarding and permit capability-free transparent listener ports.

        Returns:
            InfrastructureProcessResult: Exact forwarding and privileged-port evidence.
        """
        endpoint = self._live_endpoint()
        return self._runner.run_guest(
            self._context,
            (
                "/bin/sh",
                "-c",
                _HARDEN_NETWORK_NAMESPACE_SCRIPT,
                "loop-harden-network",
                str(endpoint.owner_uid),
            ),
            operation="broker.harden_network",
        )

    def remove_broker_bridge(self, bridge_name: str) -> InfrastructureProcessResult:
        """Remove one lease-derived bridge from the RootlessKit namespace.

        Args:
            bridge_name (str): Validated lease-derived bridge interface.

        Returns:
            InfrastructureProcessResult: Bounded bridge removal evidence.

        Raises:
            ValueError: If the bridge name is outside the broker-owned shape.
        """
        endpoint = self._live_endpoint()
        if re.fullmatch(r"br[0-9a-f]{10}", bridge_name) is None:
            raise ValueError("Broker bridge identity is invalid.")
        return self._runner.run_guest(
            self._context,
            (
                "/bin/sh",
                "-c",
                _REMOVE_BROKER_BRIDGE_SCRIPT,
                "loop-remove-bridge",
                str(endpoint.owner_uid),
                bridge_name,
            ),
            operation="broker.remove_bridge",
        )

    def write_broker_file(
        self,
        guest_path: str,
        content: bytes,
    ) -> InfrastructureProcessResult:
        """Stage one bounded broker file through the fixed Lima stdin boundary.

        Args:
            guest_path (str): Validated lease-private guest destination.
            content (bytes): Bounded declarative or secret bytes.

        Returns:
            InfrastructureProcessResult: Bounded staging evidence.
        """
        self._instance.attest_running()
        return self._runner.write_broker_file(self._context, guest_path, content)

    def remove_broker_paths(
        self,
        guest_paths: tuple[str, ...],
    ) -> InfrastructureProcessResult:
        """Remove exact lease-private broker files through the fixed Lima boundary."""
        self._instance.attest_running()
        return self._runner.remove_broker_paths(self._context, guest_paths)

    def publish_broker_port(
        self,
        address: str,
        port: int,
    ) -> InfrastructureProcessResult:
        """Publish one TCP listener through RootlessKit's builtin port API."""
        endpoint = self._live_endpoint()
        if address not in {"127.0.0.1", "0.0.0.0"} or not 1024 <= port <= 65535:
            raise ValueError("Broker publication endpoint is invalid.")
        return self._runner.run_guest(
            self._context,
            (
                "/usr/local/bin/rootlessctl",
                "--socket",
                f"/run/user/{endpoint.owner_uid}/containerd-rootless/api.sock",
                "add-ports",
                f"{address}:{port}:{port}/tcp4",
            ),
            operation="broker.publish_port",
        )

    def remove_broker_port(self, publication_id: int) -> InfrastructureProcessResult:
        """Remove one exact RootlessKit publication by its returned identifier."""
        endpoint = self._live_endpoint()
        if publication_id <= 0:
            raise ValueError("Broker publication identity is invalid.")
        return self._runner.run_guest(
            self._context,
            (
                "/usr/local/bin/rootlessctl",
                "--socket",
                f"/run/user/{endpoint.owner_uid}/containerd-rootless/api.sock",
                "remove-ports",
                str(publication_id),
            ),
            operation="broker.remove_port",
        )

    def list_broker_ports(self) -> InfrastructureProcessResult:
        """Return structured RootlessKit publication state for cleanup attestation."""
        endpoint = self._live_endpoint()
        return self._runner.run_guest(
            self._context,
            (
                "/usr/local/bin/rootlessctl",
                "--socket",
                f"/run/user/{endpoint.owner_uid}/containerd-rootless/api.sock",
                "list-ports",
                "--json",
            ),
            operation="broker.list_ports",
        )

    def _live_endpoint(self) -> GuestOciRuntimeEndpoint:
        """Return the endpoint only while its managed runtime remains attested."""
        self._live_supervisor()
        return cast(GuestOciRuntimeEndpoint, self._endpoint)

    def inspect_container(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Inspect one authenticated container through the reattested guest runtime.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw inspection evidence.

        Raises:
            ValueError: If the runtime is unattested or the identity is malformed.
        """
        return self._live_supervisor().inspect_container(container_id, cancellation)

    def inspect_container_state(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Inspect terminal state for one authenticated container identity.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw terminal-state evidence.

        Raises:
            ValueError: If the runtime is unattested or the identity is malformed.
        """
        return self._live_supervisor().inspect_container_state(container_id, cancellation)

    def remove_container(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Remove one authenticated container through the reattested guest runtime.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw removal result.

        Raises:
            ValueError: If the runtime is unattested or the identity is malformed.
        """
        return self._live_supervisor().remove_container(container_id, cancellation)

    def wait_container(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Wait for one authenticated container through the reattested guest runtime.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw wait result.

        Raises:
            ValueError: If the runtime is unattested or the identity is malformed.
        """
        return self._live_supervisor().wait_container(container_id, cancellation)

    def stop_container(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Stop one authenticated container through the reattested guest runtime.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw stop result.

        Raises:
            ValueError: If the runtime is unattested or the identity is malformed.
        """
        return self._live_supervisor().stop_container(container_id, cancellation)

    def kill_container(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Kill one authenticated container through the reattested guest runtime.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw kill result.

        Raises:
            ValueError: If the runtime is unattested or the identity is malformed.
        """
        return self._live_supervisor().kill_container(container_id, cancellation)

    def signal_container(
        self,
        container_id: str,
        signal: OciSignal,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Send one reviewed signal through the reattested guest runtime.

        Args:
            container_id (str): Authenticated container identity.
            signal (OciSignal): Reviewed signal delivered by containerd.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw signal result.

        Raises:
            ValueError: If the runtime is unattested or the identity is malformed.
        """
        return self._live_supervisor().signal_container(container_id, signal, cancellation)

    def pause_container(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Pause one authenticated container through the reattested guest runtime.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw pause result.

        Raises:
            ValueError: If the runtime is unattested or the identity is malformed.
        """
        return self._live_supervisor().pause_container(container_id, cancellation)

    def unpause_container(
        self,
        container_id: str,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> InfrastructureProcessResult:
        """Unpause one authenticated container through the reattested guest runtime.

        Args:
            container_id (str): Authenticated container identity.
            cancellation (Callable[[], bool]): Predicate cancelling the control process.

        Returns:
            InfrastructureProcessResult: Bounded raw unpause result.

        Raises:
            ValueError: If the runtime is unattested or the identity is malformed.
        """
        return self._live_supervisor().unpause_container(container_id, cancellation)

    def start_attached(
        self,
        container_id: str,
        *,
        pty: bool = False,
        queue_limit: int = 128,
        deadline_seconds: float = 30.0,
    ) -> OciControlSession:
        """Start and attach to one authenticated container in the reattested runtime.

        Args:
            container_id (str): Authenticated container identity.
            pty (bool): Whether to expose one merged pseudo-terminal stream.
            queue_limit (int): Maximum unread frames before fail-closed cleanup.
            deadline_seconds (float): Positive wall deadline.

        Returns:
            OciControlSession: Owned attached process handle.

        Raises:
            ValueError: If the runtime is unattested or the session arguments are invalid.
        """
        return self._live_supervisor().start_attached(
            container_id,
            pty=pty,
            queue_limit=queue_limit,
            deadline_seconds=deadline_seconds,
        )

    def attach(
        self,
        container_id: str,
        *,
        pty: bool = False,
        queue_limit: int = 128,
        deadline_seconds: float = 30.0,
    ) -> OciControlSession:
        """Attach to one authenticated running container in the reattested runtime.

        Args:
            container_id (str): Authenticated container identity.
            pty (bool): Whether to expose one merged pseudo-terminal stream.
            queue_limit (int): Maximum unread frames before fail-closed cleanup.
            deadline_seconds (float): Positive wall deadline.

        Returns:
            OciControlSession: Owned attached process handle.

        Raises:
            ValueError: If the runtime is unattested or the session arguments are invalid.
        """
        return self._live_supervisor().attach(
            container_id,
            pty=pty,
            queue_limit=queue_limit,
            deadline_seconds=deadline_seconds,
        )

    def _live_supervisor(self) -> OciSupervisor:
        """Return the supervisor only after reattesting its managed instance."""
        if self._supervisor is None:
            raise ValueError("OCI control requires an attested guest runtime endpoint.")
        self._instance.attest_running()
        return self._supervisor


class LimaOciTransport:
    """Carry OCI-owned argv through the macOS Lima client without compiling it."""

    _client: LimaClient
    _context: LimaInstanceContext

    def __init__(self, client: LimaClient, context: LimaInstanceContext) -> None:
        self._client = client
        self._context = context

    def run_oci_command(
        self,
        argv: tuple[str, ...],
        operation: str,
        cancellation: Callable[[], bool],
    ) -> InfrastructureProcessResult:
        """Run one exact OCI command in the managed guest."""
        return self._client.run_guest(
            self._context,
            argv,
            operation=f"oci.{operation}",
            cancellation=cancellation,
        )

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
        """Open one OCI-owned attached process through Lima's fixed shell transport."""
        return self._client.open_guest_session(
            self._context,
            argv,
            operation=f"oci.{operation}",
            pty=use_pty,
            queue_limit=queue_limit,
            deadline_seconds=deadline_seconds,
            terminal_columns=terminal_columns,
            terminal_rows=terminal_rows,
        )


_RUNTIME_ATTESTATION_SCRIPT = (
    "set -u; uid=$(/usr/bin/id -u); "
    'pid=$(/usr/bin/cat "/run/user/$uid/containerd-rootless/child_pid"); '
    "socket=/proc/$pid/root/run/containerd/containerd.sock; "
    "state=$HOME/.local/share/containerd; "
    "content=$state/io.containerd.content.v1.content; "
    'netns="/proc/$pid/root/run/user/$uid/containerd-rootless/netns"; '
    'printf \'owner_uid=%s\\nsocket_path=%s\\nsocket_is_socket=\' "$uid" "$socket"; '
    "test -S \"$socket\" && printf 'true\\n' || printf 'false\\n'; "
    "printf 'socket_uid='; /usr/bin/stat -c %u \"$socket\" 2>/dev/null || printf 'missing\\n'; "
    "printf 'state_path=%s\\nstate_exists=' \"$state\"; "
    "test -d \"$state\" && printf 'true\\n' || printf 'false\\n'; "
    "printf 'content_path=%s\\ncontent_exists=' \"$content\"; "
    "test -d \"$content\" && printf 'true\\n' || printf 'false\\n'; "
    "printf 'rootful_socket_exists='; test -e /run/containerd/containerd.sock "
    "&& printf 'true\\n' || printf 'false\\n'; "
    "printf 'network_namespace_path=%s\\nnetwork_namespace_identity=' \"$netns\"; "
    "/usr/bin/stat -Lc '%d:%i' \"$netns\"; "
    "printf 'entered_network_namespace_identity='; "
    '/usr/bin/nsenter -t "$pid" -m -U --preserve-credentials '
    '/usr/bin/nsenter --net="/run/user/$uid/containerd-rootless/netns" '
    "/usr/bin/stat -Lc '%d:%i' /proc/self/ns/net; "
    "printf 'nerdctl_version='; /usr/local/bin/nerdctl --version; "
    "printf 'containerd_version='; /usr/local/bin/containerd --version; "
    "printf 'runc_version='; /usr/local/bin/runc --version | /usr/bin/head -n 1"
    "; printf 'buildkit_version='; /usr/local/bin/buildkitd --version"
)


def _sandbox_image_inventory_script(commands: tuple[str, ...]) -> str:
    """Return the trusted in-image inventory script for declared command names."""
    command_words = " ".join(commands)
    return (
        'printf \'IDENTITY\\t%s\\t%s\\t%s\\n\' "$(id -u)" "$(id -g)" "$PATH"; '
        "dpkg-query -W -f='PACKAGE\\t${binary:Package}\\t${Version}\\n'; "
        f"for command in {command_words}; do "
        "printf 'checking %s\\n' \"$command\" >&2; "
        'path=$(command -v -- "$command") || exit 1; '
        'printf \'COMMAND\\t%s\\t%s\\n\' "$command" "$path"; done'
    )


def _sandbox_tool_arguments_are_valid(arguments: tuple[str, ...]) -> bool:
    """Return whether external-tool build arguments preserve the reviewed boundary."""
    pattern = re.compile(r"^(UV_(?:VERSION|ORIGIN|MAX_BYTES|AARCH64_SHA256|X86_64_SHA256))=(\S+)$")
    matches = [pattern.fullmatch(argument) for argument in arguments]
    if any(match is None for match in matches):
        return False
    values = {match.group(1): match.group(2) for match in matches if match is not None}
    if len(values) != len(arguments) or set(values) != {
        "UV_VERSION",
        "UV_ORIGIN",
        "UV_MAX_BYTES",
        "UV_AARCH64_SHA256",
        "UV_X86_64_SHA256",
    }:
        return False
    version = values["UV_VERSION"]
    return bool(
        re.fullmatch(r"[0-9]+(?:[.][0-9]+){2}", version)
        and values["UV_ORIGIN"]
        == f"https://releases.astral.sh/github/uv/releases/download/{version}"
        and values["UV_MAX_BYTES"].isdigit()
        and 0 < int(values["UV_MAX_BYTES"]) <= 100 * 1024 * 1024
        and re.fullmatch(r"[0-9a-f]{64}", values["UV_AARCH64_SHA256"])
        and re.fullmatch(r"[0-9a-f]{64}", values["UV_X86_64_SHA256"])
    )


_SANDBOX_IMAGE_BUILD_SCRIPT = r"""
set -eu
uid=$1
runtime_socket=$2
namespace=$3
archive=$4
source_digest=$5
image_tag=$6
base_image=$7
debian_snapshot=$8
packages=$9
tools=${10}
inventory_script=${11}
pid=${runtime_socket#/proc/}
pid=${pid%%/*}
work=${archive%.tar}
builder_socket=$work/run/buildkitd.sock
builder_pid=
resolver_mounted=false
immutable_reference=
success=false
runtime() {
    /usr/bin/nsenter -t "$pid" -m -U --preserve-credentials \
        /usr/bin/nsenter --net="/run/user/$uid/containerd-rootless/netns" "$@"
}
cleanup() {
    status=$?
    if test "$status" -ne 0; then
        for log in "$work/buildkit.log" "$work/build.log" "$work/manage.log" \
                "$work/inventory.log"; do
            if test -f "$log"; then
                /usr/bin/tail -c 32768 "$log" >&2 || true
            fi
        done
    fi
    if test -n "$builder_pid" && kill -0 "$builder_pid" 2>/dev/null; then
        kill "$builder_pid" 2>/dev/null || true
        wait "$builder_pid" 2>/dev/null || true
    fi
    if test "$resolver_mounted" = true; then
        runtime /bin/umount /etc/resolv.conf 2>/dev/null || true
    fi
    if test "$success" != true; then
        if test -n "$immutable_reference"; then
            /usr/local/bin/nerdctl --address "$runtime_socket" \
                --namespace "$namespace" image rm --force "$immutable_reference" \
                >/dev/null 2>&1 || true
        fi
        /usr/local/bin/nerdctl --address "$runtime_socket" \
            --namespace "$namespace" image rm --force "$image_tag" >/dev/null 2>&1 || true
    fi
    rm -rf -- "$work" "$archive"
    exit "$status"
}
trap cleanup EXIT HUP INT TERM
test "$(id -u)" = "$uid"
test "$(/usr/bin/sha256sum "$archive" | /usr/bin/cut -d ' ' -f 1)" = "$source_digest"
mkdir -p -- "$work/context" "$work/root" "$work/run"
/bin/tar -xf "$archive" -C "$work/context"
/bin/grep -Fqx "ARG BASE_IMAGE=$base_image" "$work/context/Containerfile"
/bin/grep -Fqx "ARG DEBIAN_SNAPSHOT=$debian_snapshot" "$work/context/Containerfile"
dns=$(/usr/bin/awk \
    '$1 == "nameserver" && $2 ~ /^[0-9.]+$/ && $2 !~ /^127\./ { print $2; exit }' \
    /run/systemd/resolve/resolv.conf)
test -n "$dns"
printf '[dns]\nnameservers=["%s"]\n' "$dns" >"$work/buildkitd.toml"
printf 'nameserver %s\n' "$dns" >"$work/resolv.conf"
runtime /bin/mount --bind "$work/resolv.conf" /etc/resolv.conf
resolver_mounted=true
runtime /usr/local/bin/buildkitd \
    --config "$work/buildkitd.toml" \
    --addr "unix://$builder_socket" \
    --root "$work/root" \
    --oci-worker=false \
    --containerd-worker=true \
    --containerd-worker-addr=/run/containerd/containerd.sock \
    --containerd-worker-namespace="$namespace" \
    >"$work/buildkit.log" 2>&1 &
builder_pid=$!
count=0
while test ! -S "$builder_socket"; do
    kill -0 "$builder_pid" 2>/dev/null
    count=$((count + 1))
    test "$count" -lt 600
    /bin/sleep 0.1
done
set -- /usr/local/bin/nerdctl --address "$runtime_socket" \
    --namespace "$namespace" build --buildkit-host "unix://$builder_socket" \
    --progress=plain --pull=true \
    --secret id=loop-ca,src=/etc/ssl/certs/ca-certificates.crt \
    --tag "$image_tag" --build-arg "BASE_IMAGE=$base_image" \
    --build-arg "DEBIAN_SNAPSHOT=$debian_snapshot" \
    --build-arg "PACKAGES=$packages"
for tool in $tools; do
    set -- "$@" --build-arg "$tool"
done
"$@" "$work/context" >"$work/build.log" 2>&1
printf 'build-complete\n' >>"$work/manage.log"
/usr/local/bin/nerdctl --address "$runtime_socket" \
    --namespace "$namespace" image inspect --mode=native \
    --format '{{.Image.Target.Digest}}' --platform linux/arm64 "$image_tag" \
    >"$work/descriptor.txt" 2>>"$work/manage.log"
descriptor=$(/bin/cat "$work/descriptor.txt")
printf '%s\n' "$descriptor" | /bin/grep -Eq '^sha256:[0-9a-f]{64}$'
printf 'descriptor-complete\n' >>"$work/manage.log"
immutable_reference="${image_tag%%:*}@$descriptor"
/usr/local/bin/nerdctl --address "$runtime_socket" --namespace "$namespace" \
    tag "$image_tag" "$immutable_reference" >/dev/null 2>>"$work/manage.log"
printf 'alias-complete\n' >>"$work/manage.log"
/usr/local/bin/nerdctl --address "$runtime_socket" --namespace "$namespace" \
    image rm "$image_tag" >/dev/null 2>>"$work/manage.log"
printf 'mutable-name-removed\n' >>"$work/manage.log"
/usr/local/bin/nerdctl --address "$runtime_socket" \
    --namespace "$namespace" image inspect --mode=native --format '{{json .}}' \
    --platform linux/arm64 "$immutable_reference" >"$work/inspect.json" \
    2>>"$work/manage.log"
printf 'inspection-complete\n' >>"$work/manage.log"
/usr/local/bin/nerdctl --address "$runtime_socket" \
    --namespace "$namespace" run --rm --pull=never --network=none --read-only \
    --tmpfs /tmp --tmpfs /workspace "$immutable_reference" \
    /bin/sh -eu -c "$inventory_script" \
    >"$work/inventory.tsv" 2>"$work/inventory.log"
printf 'inventory-complete\n' >>"$work/manage.log"
cat "$work/inspect.json"
printf '\n--LOOP-INVENTORY--\n'
cat "$work/inventory.tsv"
if kill -0 "$builder_pid" 2>/dev/null; then
    kill "$builder_pid"
    wait "$builder_pid" 2>/dev/null || true
fi
builder_pid=
printf 'builder-stop-complete\n' >>"$work/manage.log"
runtime /bin/umount /etc/resolv.conf
resolver_mounted=false
printf 'resolver-cleanup-complete\n' >>"$work/manage.log"
rm -rf -- "$work" "$archive"
test ! -e "$work" && test ! -e "$archive"
success=true
trap - EXIT HUP INT TERM
printf '\n--LOOP-CLEANUP--\ncomplete\n'
"""


_HARDEN_NETWORK_NAMESPACE_SCRIPT = (
    'set -eu; pid=$(/usr/bin/cat "/run/user/$1/containerd-rootless/child_pid"); '
    '/usr/bin/nsenter -t "$pid" -m -U --preserve-credentials '
    '/usr/bin/nsenter --net="/run/user/$1/containerd-rootless/netns" /bin/sh -c '
    "'printf 0 > /proc/sys/net/ipv4/ip_forward; "
    "if test -e /proc/sys/net/ipv6/conf/all/forwarding; then "
    "printf 0 > /proc/sys/net/ipv6/conf/all/forwarding; fi; "
    "printf 0 > /proc/sys/net/ipv4/ip_unprivileged_port_start; "
    'printf "ipv4=%s\\nipv6=%s\\nunprivileged_port_start=%s\\n" '
    '"$(cat /proc/sys/net/ipv4/ip_forward)" '
    '"$(cat /proc/sys/net/ipv6/conf/all/forwarding 2>/dev/null || printf 0)" '
    '"$(cat /proc/sys/net/ipv4/ip_unprivileged_port_start)"\''
)

_REMOVE_BROKER_BRIDGE_SCRIPT = (
    'set -eu; pid=$(/usr/bin/cat "/run/user/$1/containerd-rootless/child_pid"); '
    'exec /usr/bin/nsenter -t "$pid" -m -U --preserve-credentials '
    '/usr/bin/nsenter --net="/run/user/$1/containerd-rootless/netns" '
    '/bin/sh -c \'if /usr/sbin/ip link show "$1" >/dev/null 2>&1; then '
    '/usr/sbin/ip link delete "$1"; fi\' loop-remove-bridge-inner "$2"'
)

_CNI_ATTACHMENT_SCRIPT = (
    'set -eu; pid=$(/usr/bin/cat "/run/user/$1/containerd-rootless/child_pid"); '
    'exec 3<"$6"; exec /usr/bin/nsenter -t "$pid" -m -U --preserve-credentials '
    '/usr/bin/nsenter --net="/run/user/$1/containerd-rootless/netns" '
    '/bin/sh -c \'set -eu; if test "$1" = ADD; then mkdir -p -- "${4%/*}"; '
    'touch -- "$4"; /usr/bin/unshare --net="$4" /bin/true; fi; '
    '/usr/bin/env CNI_COMMAND="$1" CNI_CONTAINERID="$2" CNI_NETNS="$4" '
    'CNI_IFNAME="$3" CNI_PATH=/usr/local/libexec/cni '
    '/usr/local/libexec/cni/bridge <&3; if test "$1" = DEL && test -e "$4"; then '
    '/bin/umount -- "$4"; rm -f -- "$4"; fi\' '
    'loop-cni-inner "$2" "$3" "$4" "$5"'
)
