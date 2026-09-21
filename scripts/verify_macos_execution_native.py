"""Provision and verify the complete managed macOS runtime from clean private state."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import shutil
import socket
import stat
import subprocess
import threading
import time
from pathlib import Path

from sandbox_image_corpus import sandbox_image_corpus

from loop import constants
from loop.execution.command import SandboxCommandExecutor
from loop.execution.contracts import (
    Capability,
    ExecutionLease,
    ExecutionMode,
    NetworkConnectionLease,
    NetworkListenerLease,
    NetworkProtocol,
    SecretExposure,
    SecretMechanism,
    ShellExecutionRequest,
    TerminalMode,
)
from loop.execution.infrastructure import InfrastructureProcessRunner
from loop.execution.results import (
    Cancelled,
    CapabilityDenied,
    CommitConflict,
    Completed,
    TimedOut,
)
from loop.execution.runtime.bootstrap import RuntimeBootstrapper
from loop.execution.runtime.download import HttpxArtifactTransport
from loop.execution.runtime.image import load_sandbox_image_definition
from loop.execution.runtime.install import ArchiveInstaller
from loop.execution.runtime.manifest import RuntimeManifest
from loop.execution.runtime.models import AcquisitionKind
from loop.execution.sandbox.macos import (
    DurableJobState,
    DurableJobStore,
    MacosDeltaArchiveInspector,
    MacosDurableJobManager,
    MacosExecutionAdapter,
    MacosProductAdapter,
    MacosSandboxBackend,
    MacosWorkspaceCoordinator,
    MacosWorkspaceMaterializer,
    PreparedMacosRuntime,
    load_macos_runtime_candidate,
    load_macos_runtime_release,
)
from loop.execution.sandbox.oci.control import OciSignal
from loop.execution.sandbox.oci.spec import OciResourceLimits
from loop.execution.service import AttemptObserver, ExecutionService
from loop.execution.state_machine import AttemptState, AttemptStateMachine
from loop.execution.vfs import (
    AgentWorkspaceManager,
    AuthenticatedWorkspaceRoot,
    CommitBroker,
    FilesystemCapabilities,
    JournalPhase,
    PublicationCoordinator,
    SnapshotBuilder,
    SnapshotManifest,
    StagedContentStore,
    TransactionJournal,
)
from loop.execution.vfs.snapshot import FileCopier
from loop.permissions import (
    ApprovalChoice,
    ExecutionPermissionAdapter,
    PermissionManager,
    SubjectIdentity,
)
from loop.tools import create_default_tool_registry
from loop.utils import sha256_digest


class CountingFileCopier(FileCopier):
    """Count snapshot file copies without changing descriptor-safe behavior."""

    copies: int = 0

    def copy(self, source_descriptor: int, destination: Path) -> tuple[str, int]:
        """Record one snapshot file copy and delegate its exact implementation."""
        self.copies += 1
        return super().copy(source_descriptor, destination)


def _p95(samples: list[float]) -> float:
    """Return the nearest-rank 95th percentile for a nonempty sample."""
    if not samples:
        raise ValueError("Performance samples must not be empty.")
    return sorted(samples)[math.ceil(len(samples) * 0.95) - 1]


def _verify_mounts(backend: MacosSandboxBackend, prepared: PreparedMacosRuntime) -> None:
    """Require exactly one read-only VirtioFS share at the trusted guest path."""
    result = prepared.instance.runner.run_guest(
        prepared.instance.context,
        (
            "/usr/bin/findmnt",
            "--json",
            "--types",
            "virtiofs",
            "--output",
            "TARGET,FSTYPE,OPTIONS",
        ),
        operation="native.mount_attestation",
    )
    if result.exit_code or result.stdout_truncated or result.stderr_truncated:
        raise RuntimeError("Native VirtioFS inspection failed.")
    try:
        filesystems = json.loads(result.stdout)["filesystems"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("Native VirtioFS evidence is malformed.") from error
    if not isinstance(filesystems, list) or len(filesystems) != 1:
        raise RuntimeError("Native runtime exposes an unexpected VirtioFS share set.")
    mount = filesystems[0]
    options = mount.get("options", "").split(",") if isinstance(mount, dict) else []
    if (
        not isinstance(mount, dict)
        or mount.get("target") != "/run/loop/snapshots"
        or mount.get("fstype") != "virtiofs"
        or "ro" not in options
        or "rw" in options
        or not backend.snapshot_store.is_dir()
    ):
        raise RuntimeError("Native snapshot-store share is not read-only and exact.")


def _verify_execution(
    backend: MacosSandboxBackend,
    prepared: PreparedMacosRuntime,
    runtime_manifest,
) -> dict[str, object]:
    """Require the managed runtime and return bounded product-path performance evidence."""
    shell = prepared.instance.runner.run_guest
    context = prepared.instance.context
    result = shell(
        context,
        ("/bin/sh", "-c", "printf loop-native"),
        operation="native.shell",
    )
    if result.exit_code or result.stdout != b"loop-native" or result.stderr:
        raise RuntimeError("Native Lima shell transport is not byte-exact.")
    result = shell(
        context,
        (
            "/usr/local/bin/nerdctl",
            "--namespace",
            prepared.endpoint.namespace,
            "run",
            "--name",
            "native-state-probe",
            "--pull=never",
            "--network=none",
            prepared.image_reference,
            "/bin/sh",
            "-c",
            "printf loop-oci",
        ),
        operation="native.oci",
        deadline_seconds=300.0,
    )
    if result.exit_code or result.stdout != b"loop-oci" or result.stderr:
        raise RuntimeError(
            "Native rootless OCI execution is not byte-exact: "
            f"exit={result.exit_code}, stdout={result.stdout!r}, stderr={result.stderr!r}."
        )
    state = shell(
        context,
        (
            "/usr/local/bin/nerdctl",
            "--namespace",
            prepared.endpoint.namespace,
            "container",
            "inspect",
            "--format",
            "{{json .State}}",
            "native-state-probe",
        ),
        operation="native.oci_state",
    )
    removed = shell(
        context,
        (
            "/usr/local/bin/nerdctl",
            "--namespace",
            prepared.endpoint.namespace,
            "rm",
            "-f",
            "native-state-probe",
        ),
        operation="native.oci_remove",
    )
    try:
        state_payload = json.loads(state.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Native OCI terminal evidence is malformed.") from error
    if (
        state.exit_code
        or not isinstance(state_payload, dict)
        or state_payload.get("Status") != "exited"
        or state_payload.get("Running") is not False
        or state_payload.get("ExitCode") != 0
        or removed.exit_code
    ):
        raise RuntimeError(
            "Native OCI terminal evidence or removal failed: "
            f"state={state.stdout!r}, removal=({removed.exit_code}, "
            f"{removed.stdout!r}, {removed.stderr!r})."
        )

    image = prepared.sandbox_image.image
    private_root = backend.snapshot_store.parents[1]
    workspace_path = private_root / "native-workspace"
    workspace_path.mkdir(mode=0o700)
    (workspace_path / "replace").write_text("old", encoding="utf-8")
    (workspace_path / "delete").write_text("delete", encoding="utf-8")
    (workspace_path / "rename").write_text("rename", encoding="utf-8")
    (workspace_path / "mode").write_text("mode", encoding="utf-8")
    (workspace_path / "tracked.txt").write_text("alpha\n", encoding="utf-8")
    for arguments in (
        ("git", "-C", str(workspace_path), "init", "-q"),
        ("git", "-C", str(workspace_path), "config", "user.name", "Loop"),
        ("git", "-C", str(workspace_path), "config", "user.email", "loop@example.invalid"),
        ("git", "-C", str(workspace_path), "add", "."),
        ("git", "-C", str(workspace_path), "commit", "-qm", "initial"),
    ):
        completed = subprocess.run(arguments, check=False, capture_output=True)
        if completed.returncode:
            raise RuntimeError(f"Native Git fixture setup failed: {completed.stderr!r}.")
    copier = CountingFileCopier()
    snapshot_builder = SnapshotBuilder(backend.snapshot_store, copier=copier)
    root = AuthenticatedWorkspaceRoot(workspace_path)
    snapshot = snapshot_builder.build(root)
    snapshots = [snapshot]
    initial_copy_count = copier.copies
    content_store = StagedContentStore(private_root / "workspace-content")
    materializer = MacosWorkspaceMaterializer(
        prepared,
        content_store,
        private_root / "workspace-archives",
    )
    journal = TransactionJournal(private_root / "workspace-journal")

    def committed(context) -> bool:
        """Return whether every lineage transaction reached its terminal journal phase."""
        return all(
            (record := journal.load(transaction_id)) is not None
            and record.phase is JournalPhase.COMMITTED
            for transaction_id in context.committed_transaction_ids
        )

    manager = AgentWorkspaceManager(
        materializer,
        committed,
        private_root / "workspace-lineages",
    )
    manager.create("native-verifier-workspace", "native-agent", snapshot.manifest.snapshot_id)
    broker = CommitBroker(root, journal, content_store)
    publication = PublicationCoordinator(manager, broker, journal)
    manifests: dict[str, SnapshotManifest] = {
        snapshot.manifest.snapshot_id.value: snapshot.manifest
    }

    def load_manifest(snapshot_id) -> SnapshotManifest:
        """Return only a verifier-created immutable manifest by exact identity."""
        return manifests[snapshot_id.value]

    coordinator = MacosWorkspaceCoordinator(
        manager,
        materializer,
        MacosDeltaArchiveInspector(content_store),
        publication,
        load_manifest,
        FilesystemCapabilities().discover(root),
    )

    def authorize(request, delta) -> bool:
        """Deny the negative case and introduce one deliberate host conflict."""
        del delta
        if request.request_id == "native-write-denied":
            return False
        if request.request_id == "native-write-conflict":
            (workspace_path / "replace").write_text("external", encoding="utf-8")
        return True

    class NativeSecrets:
        """Resolve only verifier-owned secret identities and audiences."""

        def resolve(self, secret_id: str, audience: str) -> bytes:
            """Return one exact native fixture or reject every other binding."""
            values = {
                ("native-env", "native-process"): b"native-environment-secret",
                ("native-file", "native-process"): b"native-file-secret",
                ("native-header", "example.com"): b"Bearer native-broker-secret",
            }
            return values[(secret_id, audience)]

    adapter = MacosExecutionAdapter(
        backend,
        runtime_manifest,
        coordinator,
        OciResourceLimits(256 * 2**20, 64, 100000),
        authorize,
        prepared_runtime=prepared,
        secret_authority=NativeSecrets(),
    )
    service = ExecutionService(adapter)

    def execute(
        request_id: str,
        script: str,
        *,
        write: bool = False,
        agent_run_id: str = "native-agent",
        deadline_seconds: float = 30.0,
        output_limit_bytes: int = 1024 * 1024,
        terminal: TerminalMode = TerminalMode.PIPE,
        terminal_columns: int | None = None,
        terminal_rows: int | None = None,
        network_connections: tuple[NetworkConnectionLease, ...] = (),
        network_listeners: tuple[NetworkListenerLease, ...] = (),
        secret_exposures: tuple[SecretExposure, ...] = (),
        cancellation=lambda: False,
    ):
        """Execute one native shell request through the managed product boundary."""
        capabilities = {
            Capability.WORKSPACE_READ,
            Capability.PROCESS_SPAWN,
            Capability.PROCESS_SIGNAL,
        }
        if write:
            capabilities.add(Capability.WORKSPACE_WRITE)
        if network_connections:
            capabilities.add(Capability.NETWORK_CONNECT)
        if network_listeners:
            capabilities.add(Capability.NETWORK_LISTEN)
        if secret_exposures:
            capabilities.add(Capability.SECRET_USE)
        request = ShellExecutionRequest(
            request_id=request_id,
            lease=ExecutionLease(
                lease_id=f"lease-{request_id}",
                workspace_id="native-verifier-workspace",
                agent_run_id=agent_run_id,
                policy_version="native-verifier",
                runtime_digest=image.identity.manifest_digest,
                expires_at_ns=time.monotonic_ns() + int(deadline_seconds * 2 * 10**9),
                capabilities=frozenset(capabilities),
            ),
            script=script,
            deadline_seconds=deadline_seconds,
            output_limit_bytes=output_limit_bytes,
            terminal=terminal,
            terminal_columns=terminal_columns,
            terminal_rows=terminal_rows,
            network_connections=network_connections,
            network_listeners=network_listeners,
            secret_exposures=secret_exposures,
        )
        return service.execute(request, cancellation)

    def verify_effect_cleanup(request_id: str, *, network: bool) -> None:
        """Require absence of every lease-owned sidecar, path, namespace, and bridge."""
        digest = sha256_digest(f"lease-{request_id}")
        broker = prepared.control_plane.inspect_container_state(f"loop-broker-{digest[:16]}")
        if network and not broker.exit_code:
            raise RuntimeError(f"Native {request_id} Envoy container survived cleanup.")
        owner_home = prepared.endpoint.state_path.removesuffix("/.local/share/containerd")
        root = f"/run/user/{prepared.endpoint.owner_uid}/loop/brokers/{digest[:16]}"
        paths = [root]
        if network:
            paths.extend(
                (
                    f"{owner_home}/.config/cni/net.d/90-loop-loop-{digest[:12]}.conflist",
                    f"/run/user/{prepared.endpoint.owner_uid}/loop/netns/{digest[:16]}",
                )
            )
        path_evidence = shell(
            context,
            (
                "/bin/sh",
                "-c",
                'for path do /usr/bin/test ! -e "$path" || exit 1; done',
                "native.effect_path_cleanup",
                *paths,
            ),
            operation="native.effect_path_cleanup",
        )
        if path_evidence.exit_code:
            raise RuntimeError(f"Native {request_id} lease files survived cleanup.")
        if network:
            bridge = f"br{digest[2:12]}"
            bridge_evidence = shell(
                context,
                (
                    "/bin/sh",
                    "-c",
                    (
                        'pid=$(/usr/bin/cat "/run/user/$1/containerd-rootless/child_pid"); '
                        '/usr/bin/nsenter -t "$pid" -m -U --preserve-credentials '
                        "/usr/bin/nsenter "
                        '--net="/run/user/$1/containerd-rootless/netns" '
                        '/usr/bin/test ! -e "/sys/class/net/$2"'
                    ),
                    "native.bridge_cleanup",
                    str(prepared.endpoint.owner_uid),
                    bridge,
                ),
                operation="native.bridge_cleanup",
            )
            if bridge_evidence.exit_code:
                raise RuntimeError(f"Native {request_id} bridge survived cleanup.")

    outcomes = []
    child_created = False
    durable_managers: list[MacosDurableJobManager] = []
    try:
        corpus = (
            ("native-printf", "printf loop-native", b"loop-native", 0),
            ("native-path", "command -v sh", b"/usr/bin/sh\n", 0),
            ("native-pipeline", "printf abc | wc -c", b"3\n", 0),
            ("native-tmp", "printf private >/tmp/result; cat /tmp/result", b"private", 0),
            ("native-nonzero", "printf failed >&2; exit 7", b"", 7),
            ("native-child", "sh -c 'printf child'", b"child", 0),
            (
                "native-paths",
                'pwd; printf \'%s\n\' "$HOME" "$TMPDIR"',
                b"/workspace\n/home/agent\n/tmp\n",
                0,
            ),
        )
        for request_id, script, expected_stdout, expected_exit in corpus:
            outcome = execute(request_id, script)
            outcomes.append(outcome)
            if (
                not isinstance(outcome, Completed)
                or outcome.exit_code != expected_exit
                or outcome.stdout != expected_stdout
                or (request_id == "native-nonzero" and outcome.stderr != b"failed")
            ):
                raise RuntimeError(f"Native foreground corpus failed at {request_id}: {outcome!r}.")
        approval_prompts: list[str] = []

        class NativeApprovalInteraction:
            """Approve native-verifier prompts while retaining their disclosures."""

            def info(self, message: str) -> None:
                """Retain one permission disclosure for verification."""
                approval_prompts.append(message)

            def prompt(self, *args, **kwargs) -> ApprovalChoice:
                """Select a bounded session approval for the native verifier."""
                del args, kwargs
                return ApprovalChoice.SESSION

        permission_manager = PermissionManager(
            workspace_path,
            configuration_path=private_root / "permissions.yaml",
            workspace_id="native-verifier-workspace",
            interaction=NativeApprovalInteraction(),  # type: ignore[arg-type]
        )
        product_adapter = None
        try:
            execution_permissions = ExecutionPermissionAdapter(
                permission_manager,
                SubjectIdentity(
                    tool_id="run_command",
                    publisher="loop",
                    profile_id="ordinary-shell",
                    profile_version=load_sandbox_image_definition().source_version,
                ),
                "native-verifier-workspace",
                "2",
                "native-agent",
            )
            product_adapter = MacosProductAdapter(
                backend,
                runtime_manifest,
                workspace_path,
                private_root / "product-execution",
                execution_permissions,
                OciResourceLimits(256 * 2**20, 64, 100000),
                secret_authority=NativeSecrets(),
                prepared_runtime=prepared,
            )
            command_executor = SandboxCommandExecutor(
                ExecutionService(product_adapter),
                execution_permissions,
                "native-verifier-workspace",
                "native-agent",
                image.identity.manifest_digest,
                supports_network_effects=True,
                supports_secret_exposures=True,
            )
            registry = create_default_tool_registry(
                permission_manager=permission_manager,
                command_executor=command_executor,
            )

            def run_product(command: str, request_id: str) -> dict[str, object]:
                """Invoke the public command tool through the composed sandbox boundary."""
                return json.loads(
                    registry.call(
                        "run_command",
                        json.dumps({"command": command, "cwd": "/workspace"}),
                        call_id=request_id,
                    )
                )

            corpus = run_product(
                sandbox_image_corpus(),
                "native-product-image-corpus",
            )
            if corpus.get("ok") is not True:
                raise RuntimeError(f"Public sandbox image corpus failed: {corpus!r}.")
            serialized_corpus = json.dumps(corpus)
            if str(workspace_path) in serialized_corpus or str(private_root) in serialized_corpus:
                raise RuntimeError("Public sandbox image corpus leaked a host path.")

            definition = load_sandbox_image_definition()
            installed_image = prepared.sandbox_image
            if (
                installed_image.readiness.source_version != definition.source_version
                or installed_image.readiness.image != installed_image.image
                or installed_image.readiness_path.stat().st_mode & 0o777 != 0o600
            ):
                raise RuntimeError("Native sandbox image readiness record is invalid.")
            builder_cleanup = shell(
                context,
                (
                    "/bin/sh",
                    "-c",
                    (
                        "! pgrep -x buildkitd >/dev/null; "
                        'test ! -d "$HOME/.loop-build" '
                        "|| test -z "
                        '"$(find "$HOME/.loop-build" -mindepth 1 -print -quit)"'
                    ),
                ),
                operation="native.image_builder_cleanup",
            )
            if builder_cleanup.exit_code:
                raise RuntimeError("Native sandbox image builder state survived completion.")

            stdin_result = json.loads(
                registry.call(
                    "run_command",
                    json.dumps(
                        {
                            "command": 'IFS= read -r value; printf \'%s:%s\' "$PWD" "$value"',
                            "cwd": "/workspace",
                            "stdin": "bounded-input\n",
                        }
                    ),
                    call_id="native-product-stdin-eof",
                )
            )
            if (
                stdin_result.get("ok") is not True
                or stdin_result.get("result", {}).get("stdout", {}).get("content")
                != "/workspace:bounded-input"
            ):
                raise RuntimeError(
                    f"Public bounded stdin, EOF, or virtual cwd failed: {stdin_result!r}."
                )

            pty_result = json.loads(
                registry.call(
                    "run_command",
                    json.dumps(
                        {
                            "command": "test -t 0; printf ready; sleep 0.1; stty size",
                            "cwd": "/workspace",
                            "pty": True,
                            "terminal_columns": 101,
                            "terminal_rows": 41,
                        }
                    ),
                    call_id="native-product-pty",
                )
            )
            if pty_result.get("ok") is not True or "41 101" not in pty_result.get("result", {}).get(
                "stdout", {}
            ).get("content", ""):
                raise RuntimeError(
                    f"Public PTY allocation and initial resize failed: {pty_result!r}."
                )

            public_network = json.loads(
                registry.call(
                    "run_command",
                    json.dumps(
                        {
                            "command": "true",
                            "cwd": "/workspace",
                            "network_connections": [
                                {
                                    "hostname": "example.com",
                                    "port": 80,
                                    "protocol": "http",
                                    "addresses": ["93.184.216.34"],
                                    "max_connections": 2,
                                    "max_bytes": 1024,
                                }
                            ],
                        }
                    ),
                    call_id="native-product-network",
                )
            )
            if public_network.get("ok") is not True:
                raise RuntimeError("Public network broker execution failed.")
            public_secret = json.loads(
                registry.call(
                    "run_command",
                    json.dumps(
                        {
                            "command": "true",
                            "cwd": "/workspace",
                            "secret_exposures": [
                                {
                                    "secret_id": "native-env",
                                    "audience": "native-process",
                                    "mechanism": "raw_environment",
                                    "target": "NATIVE_TOKEN",
                                }
                            ],
                        }
                    ),
                    call_id="native-product-secret",
                )
            )
            if public_secret.get("ok") is not True:
                raise RuntimeError("Public secret broker execution failed.")

            public_header = json.loads(
                registry.call(
                    "run_command",
                    json.dumps(
                        {
                            "command": "true",
                            "cwd": "/workspace",
                            "network_connections": [
                                {
                                    "hostname": "example.com",
                                    "port": 80,
                                    "protocol": "http",
                                    "addresses": ["93.184.216.34"],
                                    "max_connections": 2,
                                    "max_bytes": 1024,
                                }
                            ],
                            "secret_exposures": [
                                {
                                    "secret_id": "native-header",
                                    "audience": "example.com",
                                    "mechanism": "request_header",
                                    "target": "Authorization",
                                }
                            ],
                        }
                    ),
                    call_id="native-product-header-secret",
                )
            )
            if public_header.get("ok") is not True:
                raise RuntimeError("Public header-secret broker execution failed.")

            public_listener = json.loads(
                registry.call(
                    "run_command",
                    json.dumps(
                        {
                            "command": "true",
                            "cwd": "/workspace",
                            "network_listeners": [{"port": 18080, "target_port": 18080}],
                        }
                    ),
                    call_id="native-product-listener",
                )
            )
            if public_listener.get("ok") is not True:
                raise RuntimeError("Public listener broker execution failed.")

            started_job = json.loads(
                registry.call(
                    "start_command_job",
                    json.dumps(
                        {
                            "command": (
                                "trap 'exit 0' INT TERM; while :; do "
                                "if IFS= read -r value; then printf 'echo:%s\\n' \"$value\"; "
                                "else sleep 0.05; fi; done"
                            ),
                            "cwd": "/workspace",
                        }
                    ),
                    call_id="native-public-durable",
                )
            )
            if started_job.get("ok") is not True:
                raise RuntimeError("Public durable-job start failed.")
            public_handle = started_job["result"]

            def durable_call(name: str, **arguments) -> dict[str, object]:
                """Invoke one authenticated public durable-job operation."""
                return json.loads(registry.call(name, json.dumps({**public_handle, **arguments})))

            status = durable_call("command_job_status")
            if status.get("result", {}).get("state") != "running":
                raise RuntimeError(
                    f"Public durable job did not remain running after start: {status!r}."
                )
            for operation, arguments in (
                ("attach_command_job", {}),
                ("resize_command_job", {"columns": 102, "rows": 42}),
                ("write_command_job", {"data": "public-input\n"}),
            ):
                if durable_call(operation, **arguments).get("ok") is not True:
                    raise RuntimeError(f"Public durable operation {operation} failed.")
            durable_output = ""
            output_deadline = time.monotonic() + 10
            while time.monotonic() < output_deadline and "echo:public-input" not in durable_output:
                frame_result = durable_call("read_command_job", timeout_seconds=0.2)
                frame = frame_result.get("result", {}).get("frame")
                if frame is not None:
                    durable_output += frame.get("data", {}).get("content", "")
            if "echo:public-input" not in durable_output:
                raise RuntimeError("Public durable input or streaming output failed.")
            for operation in (
                "detach_command_job",
                "attach_command_job",
                "detach_command_job",
            ):
                if durable_call(operation).get("ok") is not True:
                    raise RuntimeError(f"Public durable operation {operation} failed.")
            forged_status = json.loads(
                registry.call(
                    "command_job_status",
                    json.dumps({**public_handle, "token": "0" * 64}),
                )
            )
            if forged_status.get("problem", {}).get("code") != "process.durable_job_failed":
                raise RuntimeError("Public durable tools accepted a forged handle.")
            if durable_call("attach_command_job").get("ok") is not True:
                raise RuntimeError("Public durable reattachment failed.")
            if durable_call("signal_command_job", signal="INT").get("ok") is not True:
                raise RuntimeError("Public durable signal failed.")

            cancelled_job = json.loads(
                registry.call(
                    "start_command_job",
                    json.dumps({"command": "sleep 60", "cwd": "/workspace"}),
                    call_id="native-public-durable-cancel",
                )
            )
            if cancelled_job.get("ok") is not True:
                raise RuntimeError("Public cancellable durable-job start failed.")
            public_handle = cancelled_job["result"]
            if durable_call("cancel_command_job").get("result", {}).get("state") != "cancelled":
                raise RuntimeError("Public durable cancellation failed.")
            if not approval_prompts:
                raise RuntimeError("Public durable lifecycle bypassed permission interaction.")

            for index in range(3):
                warmup = run_product(":", f"native-performance-warmup-{index}")
                if warmup.get("ok") is not True:
                    raise RuntimeError("Native managed performance warm-up failed.")

            managed_durations = []
            for index in range(20):
                started = time.monotonic()
                managed = run_product(":", f"native-performance-managed-{index}")
                managed_durations.append(time.monotonic() - started)
                if managed.get("ok") is not True:
                    raise RuntimeError("Native managed performance sample failed.")
        finally:
            if product_adapter is not None:
                product_adapter.close()
            permission_manager.close()
        if copier.copies != initial_copy_count:
            raise RuntimeError("Warm native commands performed recursive workspace copying.")

        direct_durations = []
        for index in range(20):
            started = time.monotonic()
            direct = shell(
                context,
                (
                    "/usr/local/bin/nerdctl",
                    "--namespace",
                    prepared.endpoint.namespace,
                    "run",
                    "--rm",
                    "--name",
                    f"native-performance-direct-{index}",
                    "--pull=never",
                    "--network=none",
                    prepared.image_reference,
                    "/bin/sh",
                    "-c",
                    ":",
                ),
                operation="native.performance.direct",
                deadline_seconds=30.0,
            )
            direct_durations.append(time.monotonic() - started)
            if direct.exit_code or direct.stdout or direct.stderr:
                raise RuntimeError("Native direct performance baseline failed.")
        direct_p95 = _p95(direct_durations)
        managed_p95 = _p95(managed_durations)
        performance = {
            "direct_nerdctl_p95_seconds": round(direct_p95, 6),
            "managed_run_command_p95_seconds": round(managed_p95, 6),
            "samples": len(managed_durations),
        }
        print(
            json.dumps(
                {"performance": performance},
                sort_keys=True,
            )
        )
        if managed_p95 > 1.0:
            raise RuntimeError(
                "Warm native run_command p95 exceeds 1.0 second "
                f"({managed_p95:.3f}s; direct nerdctl diagnostic {direct_p95:.3f}s)."
            )

        broker_inventory = prepared.control_plane.version()
        if broker_inventory.exit_code:
            raise RuntimeError("Offline native attempt changed OCI control availability.")
        existing_brokers = shell(
            context,
            (
                "/usr/local/bin/nerdctl",
                "--namespace",
                prepared.endpoint.namespace,
                "ps",
                "-a",
                "--filter",
                "name=loop-broker-",
                "--format",
                "{{.Names}}",
            ),
            operation="native.offline_broker_absence",
        )
        if existing_brokers.exit_code or existing_brokers.stdout.strip():
            raise RuntimeError("Offline attempts started or retained an Envoy broker.")

        addresses = tuple(
            sorted(
                {
                    item[4][0]
                    for item in socket.getaddrinfo(
                        "example.com",
                        443,
                        family=socket.AF_INET,
                        type=socket.SOCK_STREAM,
                    )
                }
            )
        )
        connection = NetworkConnectionLease(
            hostname="example.com",
            port=443,
            protocol=NetworkProtocol.HTTPS,
            addresses=addresses,
        )
        network_results = []
        network_runner = threading.Thread(
            target=lambda: network_results.append(
                execute(
                    "native-network",
                    "wget -qO- https://example.com/; status=$?; "
                    "printf '\nnamespace-ready\n'; sleep 15; exit \"$status\"",
                    network_connections=(connection,),
                    deadline_seconds=60.0,
                )
            ),
            daemon=True,
        )
        network_runner.start()
        broker_digest = sha256_digest(b"lease-native-network")[:16]
        broker_container = f"loop-broker-{broker_digest}"
        command_container = "loop-native-network"
        state_deadline = time.monotonic() + 20
        while time.monotonic() < state_deadline:
            broker_state = prepared.control_plane.inspect_container_state(broker_container)
            command_state = prepared.control_plane.inspect_container_state(command_container)
            if not broker_state.exit_code and not command_state.exit_code:
                broker_payload = json.loads(broker_state.stdout)
                command_payload = json.loads(command_state.stdout)
                if (
                    isinstance(broker_payload, dict)
                    and broker_payload.get("Running") is True
                    and isinstance(command_payload, dict)
                    and command_payload.get("Running") is True
                ):
                    break
            time.sleep(0.05)
        else:
            raise RuntimeError("Native broker and command never became concurrently observable.")
        if not prepared.control_plane.container_uses_rootless_network_namespace(broker_container):
            raise RuntimeError("Native Envoy did not share the attested RootlessKit namespace.")
        if prepared.control_plane.container_uses_rootless_network_namespace(command_container):
            raise RuntimeError("Native command escaped into the trusted RootlessKit namespace.")
        network_name = f"loop-{sha256_digest(b'lease-native-network')[:12]}"
        gateway_octet = 16 + int(sha256_digest(b"lease-native-network")[:2], 16) % 224
        gateway = f"10.240.{gateway_octet}.1"
        gateway_evidence = shell(
            context,
            (
                "/bin/sh",
                "-c",
                (
                    'pid=$(/usr/bin/cat "/run/user/$1/containerd-rootless/child_pid"); '
                    '/usr/bin/nsenter -t "$pid" -m -U --preserve-credentials '
                    "/usr/bin/nsenter "
                    '--net="/run/user/$1/containerd-rootless/netns" '
                    '/usr/sbin/ip -o -4 address show dev "$2"'
                ),
                "native.gateway_namespace",
                str(prepared.endpoint.owner_uid),
                f"br{network_name[-10:]}",
            ),
            operation="native.gateway_namespace",
        )
        if gateway_evidence.exit_code or f" {gateway}/24 ".encode() not in gateway_evidence.stdout:
            raise RuntimeError("Native command-bridge gateway is outside the attested namespace.")
        network_runner.join(timeout=65)
        if network_runner.is_alive() or len(network_results) != 1:
            raise RuntimeError("Native leased HTTPS execution did not terminate.")
        networked = network_results[0]
        outcomes.append(networked)
        if not isinstance(networked, Completed) or b"Example Domain" not in networked.stdout:
            raise RuntimeError(f"Native leased HTTPS failed: {networked!r}.")
        verify_effect_cleanup("native-network", network=True)

        denied_egress = execute(
            "native-direct-egress",
            "for address in 1.1.1.1 10.0.0.1 169.254.169.254; do "
            'nc -z -w 1 "$address" 443 && exit 90; done; '
            "nc -z -w 1 127.0.0.1 443 && exit 91; exit 0",
            network_connections=(connection,),
            deadline_seconds=15.0,
        )
        outcomes.append(denied_egress)
        if not isinstance(denied_egress, Completed) or denied_egress.exit_code:
            raise RuntimeError(f"Native direct-egress negative failed: {denied_egress!r}.")
        verify_effect_cleanup("native-direct-egress", network=True)

        wrong_sni = execute(
            "native-wrong-sni",
            "if printf 'GET / HTTP/1.1\r\nHost: example.com\r\nConnection: close\r\n\r\n' "
            "| openssl s_client -quiet -connect example.com:443 "
            "-servername wrong.example 2>/dev/null "
            "| grep -q '^HTTP/'; then exit 98; fi",
            network_connections=(connection,),
            deadline_seconds=20.0,
        )
        outcomes.append(wrong_sni)
        if not isinstance(wrong_sni, Completed) or wrong_sni.exit_code:
            raise RuntimeError(f"Native same-IP/different-SNI negative failed: {wrong_sni!r}.")
        verify_effect_cleanup("native-wrong-sni", network=True)

        rebinding = execute(
            "native-rebinding",
            "if printf '1.1.1.1 example.com\n' >>/etc/hosts 2>/dev/null; then exit 95; fi; "
            "wget -qO- https://example.com/",
            network_connections=(connection,),
            deadline_seconds=60.0,
        )
        outcomes.append(rebinding)
        if not isinstance(rebinding, Completed) or b"Example Domain" not in rebinding.stdout:
            raise RuntimeError(f"Native rebinding negative failed: {rebinding!r}.")
        verify_effect_cleanup("native-rebinding", network=True)

        github_addresses = tuple(
            sorted(
                {
                    item[4][0]
                    for item in socket.getaddrinfo(
                        "github.com",
                        80,
                        family=socket.AF_INET,
                        type=socket.SOCK_STREAM,
                    )
                }
            )
        )
        redirect_connection = NetworkConnectionLease(
            hostname="github.com",
            port=80,
            protocol=NetworkProtocol.HTTP,
            addresses=github_addresses,
        )
        redirect = execute(
            "native-redirect",
            "if wget -S -O /tmp/redirect-body http://github.com/ 2>/tmp/redirect-log; "
            "then exit 96; fi; grep -qi 'Location: https://github.com' /tmp/redirect-log",
            network_connections=(redirect_connection,),
            deadline_seconds=30.0,
        )
        outcomes.append(redirect)
        if not isinstance(redirect, Completed) or redirect.exit_code:
            raise RuntimeError(f"Native redirect negative failed: {redirect!r}.")
        verify_effect_cleanup("native-redirect", network=True)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_probe:
            port_probe.bind(("127.0.0.1", 0))
            ingress_port = port_probe.getsockname()[1]
        ingress_results = []
        ingress_runner = threading.Thread(
            target=lambda: ingress_results.append(
                execute(
                    "native-ingress",
                    f"(sleep 2; printf ingress-ok; sleep 2) | nc -l -p {ingress_port}",
                    network_listeners=(
                        NetworkListenerLease(port=ingress_port, target_port=ingress_port),
                    ),
                    deadline_seconds=30.0,
                )
            ),
            daemon=True,
        )
        ingress_runner.start()
        ingress_deadline = time.monotonic() + 20
        ingress_payload = b""
        while time.monotonic() < ingress_deadline:
            try:
                with socket.create_connection(("127.0.0.1", ingress_port), timeout=0.5) as stream:
                    stream.settimeout(5)
                    ingress_payload = stream.recv(1024)
                    break
            except OSError:
                time.sleep(0.05)
        ingress_runner.join(timeout=30)
        if (
            ingress_payload != b"ingress-ok"
            or ingress_runner.is_alive()
            or len(ingress_results) != 1
            or not isinstance(ingress_results[0], Completed)
        ):
            raise RuntimeError(
                f"Native ingress publication failed: payload={ingress_payload!r}, "
                f"results={ingress_results!r}."
            )
        outcomes.append(ingress_results[0])
        publications = prepared.control_plane.list_broker_ports()
        if publications.exit_code or publications.stdout_truncated or publications.stderr_truncated:
            raise RuntimeError("Native ingress publication cleanup evidence failed.")
        try:
            publication_inventory = (
                json.loads(publications.stdout) if publications.stdout.strip() else []
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "Native ingress publication cleanup evidence is malformed."
            ) from error
        if publication_inventory:
            raise RuntimeError("Native ingress publication survived terminal cleanup.")
        verify_effect_cleanup("native-ingress", network=True)

        raw_secrets = (
            SecretExposure(
                secret_id="native-env",
                audience="native-process",
                mechanism=SecretMechanism.RAW_ENVIRONMENT,
                target="NATIVE_TOKEN",
            ),
            SecretExposure(
                secret_id="native-file",
                audience="native-process",
                mechanism=SecretMechanism.RAW_FILE,
                target="/run/secrets/native-token",
            ),
        )
        secret_result = execute(
            "native-secret-offline",
            'test "$NATIVE_TOKEN" = native-environment-secret; '
            'test "$(cat /run/secrets/native-token)" = native-file-secret; '
            "nc -z -w 1 1.1.1.1 443 && exit 92; printf secret-confined",
            secret_exposures=raw_secrets,
        )
        outcomes.append(secret_result)
        if not isinstance(secret_result, Completed) or secret_result.stdout != b"secret-confined":
            raise RuntimeError(f"Native secret confinement failed: {secret_result!r}.")
        staged_secrets = shell(
            context,
            (
                "/bin/sh",
                "-c",
                (
                    'test ! -e "/run/user/$1/loop/brokers/'
                    '$(printf %s lease-native-secret-offline | sha256sum | cut -c1-16)"'
                ),
                "native.secret_cleanup",
                str(prepared.endpoint.owner_uid),
            ),
            operation="native.secret_cleanup",
        )
        if staged_secrets.exit_code:
            raise RuntimeError("Native secret staging survived terminal cleanup.")
        verify_effect_cleanup("native-secret-offline", network=False)

        header_secret = SecretExposure(
            secret_id="native-header",
            audience="example.com",
            mechanism=SecretMechanism.REQUEST_HEADER,
            target="Authorization",
        )
        http_addresses = tuple(
            sorted(
                {
                    item[4][0]
                    for item in socket.getaddrinfo(
                        "example.com",
                        80,
                        family=socket.AF_INET,
                        type=socket.SOCK_STREAM,
                    )
                }
            )
        )
        header_connection = NetworkConnectionLease(
            hostname="example.com",
            port=80,
            protocol=NetworkProtocol.HTTP,
            addresses=http_addresses,
        )
        secret_exfiltration = execute(
            "native-secret-exfiltration",
            'test -z "${NATIVE_TOKEN+x}"; test ! -e /run/secrets/native-token; '
            "! grep -a -q native-broker-secret /proc/self/environ; "
            "nc -z -w 1 1.1.1.1 80 && exit 99; wget -qO- http://example.com/",
            network_connections=(header_connection,),
            secret_exposures=(header_secret,),
            deadline_seconds=30.0,
        )
        outcomes.append(secret_exfiltration)
        if (
            not isinstance(secret_exfiltration, Completed)
            or b"Example Domain" not in secret_exfiltration.stdout
        ):
            raise RuntimeError(
                f"Native broker-secret exfiltration negative failed: {secret_exfiltration!r}."
            )
        verify_effect_cleanup("native-secret-exfiltration", network=True)

        pty_result = execute(
            "native-pty",
            "test -t 0 || exit 91; printf ready; read value || true; stty size; printf terminal",
            terminal=TerminalMode.PTY,
            terminal_columns=100,
            terminal_rows=40,
        )
        outcomes.append(pty_result)
        if (
            not isinstance(pty_result, Completed)
            or b"40 100" not in pty_result.stdout
            or b"terminal" not in pty_result.stdout
            or pty_result.stderr
        ):
            raise RuntimeError(f"Native PTY resize or merged output failed: {pty_result!r}.")

        eof_result = execute("native-eof", "read value || printf eof")
        outcomes.append(eof_result)
        if not isinstance(eof_result, Completed) or eof_result.stdout != b"eof":
            raise RuntimeError(f"Native stdin EOF failed: {eof_result!r}.")

        descendant_started = time.monotonic()
        descendant = execute("native-descendant", "sleep 30 & printf foreground")
        descendant_elapsed = time.monotonic() - descendant_started
        outcomes.append(descendant)
        if (
            not isinstance(descendant, Completed)
            or descendant.stdout != b"foreground"
            or descendant_elapsed >= 10
        ):
            raise RuntimeError(
                f"Native descendant containment failed after {descendant_elapsed:.3f}s: "
                f"{descendant!r}."
            )

        large = execute(
            "native-large-output",
            "dd if=/dev/zero bs=65536 count=8 2>/dev/null",
            output_limit_bytes=4096,
        )
        outcomes.append(large)
        if (
            not isinstance(large, Completed)
            or len(large.stdout) != 4096
            or not large.stdout_truncated
        ):
            raise RuntimeError("Native foreground output bound is not enforced.")
        timed_out = execute("native-timeout", "sleep 5", deadline_seconds=1.0)
        outcomes.append(timed_out)
        if not isinstance(timed_out, TimedOut):
            raise TypeError(f"Native foreground timeout returned {timed_out!r}.")
        cancel_at = time.monotonic() + 1.0
        cancelled = execute(
            "native-cancel",
            "sleep 5",
            cancellation=lambda: time.monotonic() >= cancel_at,
        )
        outcomes.append(cancelled)
        if not isinstance(cancelled, Cancelled):
            raise TypeError(f"Native foreground cancellation returned {cancelled!r}.")

        def verify_signal(request_id: str, signal: OciSignal, marker: bytes) -> None:
            """Deliver one native container signal and require terminal product cleanup."""
            result = []
            runner = threading.Thread(
                target=lambda: result.append(
                    execute(
                        request_id,
                        f"trap 'printf {marker.decode()}; exit 0' {signal.value}; "
                        "printf ready; while :; do sleep 1; done",
                        deadline_seconds=15.0,
                    )
                ),
                daemon=True,
            )
            runner.start()
            state_deadline = time.monotonic() + 10
            while time.monotonic() < state_deadline:
                state_result = prepared.control_plane.inspect_container_state(f"loop-{request_id}")
                if not state_result.exit_code:
                    state = json.loads(state_result.stdout)
                    if isinstance(state, dict) and state.get("Running") is True:
                        break
                time.sleep(0.05)
            else:
                raise RuntimeError(f"Native {signal.value} target never started.")
            time.sleep(0.1)
            delivered = prepared.control_plane.signal_container(f"loop-{request_id}", signal)
            if delivered.exit_code or delivered.stdout_truncated or delivered.stderr_truncated:
                raise RuntimeError(f"Native {signal.value} delivery failed.")
            runner.join(timeout=10)
            if runner.is_alive() or len(result) != 1:
                raise RuntimeError(f"Native {signal.value} execution did not terminate.")
            outcomes.append(result[0])
            if not isinstance(result[0], Completed) or marker not in result[0].stdout:
                raise RuntimeError(f"Native {signal.value} handling failed: {result[0]!r}.")

        verify_signal("native-interrupt", OciSignal.INTERRUPT, b"interrupted")
        verify_signal("native-hangup", OciSignal.HANGUP, b"hungup")

        changed = execute(
            "native-write-effects",
            "printf created >created; printf replacement >replace; "
            "rm delete; mv rename renamed; chmod 600 mode",
            write=True,
        )
        outcomes.append(changed)
        if not isinstance(changed, Completed):
            raise TypeError(f"Native workspace publication failed: {changed!r}.")
        if (
            (workspace_path / "created").read_text(encoding="utf-8") != "created"
            or (workspace_path / "replace").read_text(encoding="utf-8") != "replacement"
            or (workspace_path / "delete").exists()
            or (workspace_path / "rename").exists()
            or (workspace_path / "renamed").read_text(encoding="utf-8") != "rename"
            or (workspace_path / "mode").stat().st_mode & 0o777 != 0o600
        ):
            raise RuntimeError("Native workspace effects were not published exactly.")

        denied = execute("native-write-denied", "printf denied >denied", write=True)
        outcomes.append(denied)
        if not isinstance(denied, CapabilityDenied) or (workspace_path / "denied").exists():
            raise RuntimeError("Denied native effects changed the host workspace.")
        conflicted = execute("native-write-conflict", "printf conflict >replace", write=True)
        outcomes.append(conflicted)
        if not isinstance(conflicted, CommitConflict):
            raise TypeError(
                f"Concurrent host mutation did not produce a commit conflict: {conflicted!r}."
            )

        manager.fork("native-verifier-workspace", "native-agent", "native-child")
        child_created = True
        fork_read = execute(
            "native-fork-read",
            "cat created renamed",
            agent_run_id="native-child",
        )
        outcomes.append(fork_read)
        if not isinstance(fork_read, Completed) or fork_read.stdout != b"createdrename":
            raise RuntimeError("Native child branch did not inherit committed parent effects.")

        restarted_manager = AgentWorkspaceManager(
            materializer,
            committed,
            private_root / "workspace-lineages",
        )
        restarted_publication = PublicationCoordinator(restarted_manager, broker, journal)
        restarted_publication.recover_incomplete()
        if restarted_manager.get(
            "native-verifier-workspace", "native-agent"
        ).committed_transaction_ids != ("native-write-effects",):
            raise RuntimeError("Native lineage did not survive application restart.")

        refreshed_snapshot = snapshot_builder.build(root)
        snapshots.append(refreshed_snapshot)
        manifests[refreshed_snapshot.manifest.snapshot_id.value] = refreshed_snapshot.manifest
        manager.refresh(
            "native-verifier-workspace",
            "native-agent",
            refreshed_snapshot.manifest.snapshot_id,
        )
        refreshed = execute("native-refresh-read", "cat replace")
        outcomes.append(refreshed)
        if not isinstance(refreshed, Completed) or refreshed.stdout != b"external":
            raise RuntimeError("Explicit native refresh did not replace the agent base.")

        job_store = DurableJobStore(private_root / "jobs")

        def job_request(
            request_id: str,
            script: str,
            *,
            write: bool = False,
            terminal: TerminalMode = TerminalMode.PTY,
        ) -> ShellExecutionRequest:
            """Build one native explicit durable-job request."""
            capabilities = {
                Capability.WORKSPACE_READ,
                Capability.PROCESS_SPAWN,
                Capability.PROCESS_SIGNAL,
            }
            if write:
                capabilities.add(Capability.WORKSPACE_WRITE)
            return ShellExecutionRequest(
                request_id=request_id,
                lease=ExecutionLease(
                    lease_id=f"lease-{request_id}",
                    workspace_id="native-verifier-workspace",
                    agent_run_id="native-agent",
                    policy_version="native-verifier",
                    runtime_digest=image.identity.manifest_digest,
                    expires_at_ns=time.monotonic_ns() + 120 * 10**9,
                    capabilities=frozenset(capabilities),
                ),
                script=script,
                mode=ExecutionMode.DURABLE_JOB,
                deadline_seconds=120,
                terminal=terminal,
                terminal_columns=80 if terminal is TerminalMode.PTY else None,
                terminal_rows=24 if terminal is TerminalMode.PTY else None,
            )

        def job_observer() -> AttemptObserver:
            """Create one correctly authorized lifecycle observer for a durable start."""
            attempt = AttemptStateMachine()
            attempt.transition(AttemptState.AUTHORIZED)
            return AttemptObserver(attempt)

        def operation_lease(request: ShellExecutionRequest, *, expired: bool = False):
            """Issue one native job-operation lease for the exact durable subject."""
            return request.lease.model_copy(
                update={
                    "lease_id": f"operation-{request.request_id}",
                    "expires_at_ns": (0 if expired else time.monotonic_ns() + 120 * 10**9),
                }
            )

        def job_manager(*, warm: PreparedMacosRuntime | None = None):
            """Compose one restartable durable manager over the product boundaries."""
            job_owner = MacosDurableJobManager(
                backend,
                runtime_manifest,
                coordinator,
                OciResourceLimits(256 * 2**20, 64, 100000),
                lambda lease, operation, job_id: True,
                authorize,
                job_store,
                prepared_runtime=warm,
            )
            durable_managers.append(job_owner)
            return job_owner

        interactive = job_request(
            "native-durable-interactive",
            "trap 'exit 0' INT TERM; trap ':' HUP; while :; do "
            "if IFS= read -r value; then "
            'printf "echo:%s\\n" "$value"; else sleep 0.05; fi; done',
            terminal=TerminalMode.PTY,
        )
        durable = job_manager(warm=prepared)
        interactive_handle = durable.start(interactive, job_observer())
        lease = operation_lease(interactive)
        if durable.status(interactive_handle, lease).state is not DurableJobState.RUNNING:
            raise RuntimeError("Native durable job did not remain running after detach.")
        durable.attach(interactive_handle, lease)
        durable.resize(interactive_handle, lease, 100, 40)
        durable.write(interactive_handle, lease, b"native\n")
        output = bytearray()
        output_deadline = time.monotonic() + 10
        while time.monotonic() < output_deadline and b"echo:native" not in output:
            frame = durable.read(interactive_handle, lease, 0.1)
            if frame is not None:
                output.extend(frame.data)
        if b"echo:native" not in output:
            attachment_status = durable.status(interactive_handle, lease)
            raise RuntimeError(
                "Native durable stdin/PTY attachment failed: "
                f"output={bytes(output)!r}, status={attachment_status!r}."
            )
        if durable.suspend(interactive_handle, lease).state is not DurableJobState.PAUSED:
            raise RuntimeError("Native durable suspension failed.")
        if durable.resume(interactive_handle, lease).state is not DurableJobState.RUNNING:
            raise RuntimeError("Native durable resume failed.")
        durable.detach(interactive_handle, lease)
        durable.attach(interactive_handle, lease)
        durable.detach(interactive_handle, lease)
        durable.close()

        durable = job_manager()
        recovered = durable.recover()
        if len(recovered) != 1 or recovered[0].state is not DurableJobState.RUNNING:
            raise RuntimeError("Native durable job did not survive Loop restart.")
        try:
            durable.status(interactive_handle, operation_lease(interactive, expired=True))
        except PermissionError:
            pass
        else:
            raise RuntimeError("Native durable job accepted a stale operation lease.")
        forged = interactive_handle.model_copy(update={"token": "0" * 64})
        try:
            durable.status(forged, lease)
        except PermissionError:
            pass
        else:
            raise RuntimeError("Native durable job accepted a forged handle.")
        durable.signal(interactive_handle, lease, OciSignal.INTERRUPT)
        terminal_deadline = time.monotonic() + 10
        while time.monotonic() < terminal_deadline:
            interactive_status = durable.status(interactive_handle, lease)
            if interactive_status.state is DurableJobState.COMPLETED:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("Native durable signal did not terminate and finalize the job.")

        cancelled_request = job_request("native-durable-cancel", "sleep 60")
        cancelled_handle = durable.start(cancelled_request, job_observer())
        if (
            durable.cancel(cancelled_handle, operation_lease(cancelled_request)).state
            is not DurableJobState.CANCELLED
        ):
            raise RuntimeError("Native durable cancellation failed.")

        write_request = job_request(
            "native-durable-write",
            "printf durable >durable-job; sleep 1",
            write=True,
        )
        write_handle = durable.start(write_request, job_observer())
        write_lease = operation_lease(write_request)
        write_deadline = time.monotonic() + 10
        while time.monotonic() < write_deadline:
            write_status = durable.status(write_handle, write_lease)
            if write_status.state is DurableJobState.COMPLETED:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("Native durable terminal delta did not finalize.")
        if (workspace_path / "durable-job").read_text(encoding="utf-8") != "durable":
            raise RuntimeError("Native durable terminal delta was not published.")

        runtime_request = job_request("native-durable-runtime-restart", "sleep 60")
        runtime_handle = durable.start(runtime_request, job_observer())
        durable.close()
        prepared.stop()
        after_runtime_restart = job_manager()
        restart_states = {status.job_id: status.state for status in after_runtime_restart.recover()}
        if restart_states.get(runtime_handle.job_id) is not DurableJobState.LOST:
            raise RuntimeError("Native runtime restart did not mark its durable job lost.")
        after_runtime_restart.close()
        cleanup_runtime = backend.prepare()
        try:
            for job_id in (
                "native-durable-interactive",
                "native-durable-cancel",
                "native-durable-write",
                "native-durable-runtime-restart",
            ):
                state = cleanup_runtime.control_plane.inspect_container_state(f"loop-{job_id}")
                if not state.exit_code:
                    raise RuntimeError(f"Native durable container {job_id} survived cleanup.")
        finally:
            cleanup_runtime.close()
        if str(private_root) in repr(outcomes):
            raise RuntimeError("Native foreground result leaked a host path.")
    finally:
        for durable_manager in durable_managers:
            durable_manager.close()
        if child_created:
            manager.dispose("native-verifier-workspace", "native-child")
        manager.dispose("native-verifier-workspace", "native-agent")
        for transaction_id in (
            "native-write-effects",
            "native-write-denied",
            "native-write-conflict",
        ):
            content_store.discard(transaction_id)
        for owned_snapshot in snapshots:
            snapshot_builder.discard(owned_snapshot)
        root.close()
    return performance


def _verify_cold_public_execution(
    backend: MacosSandboxBackend,
    runtime_manifest: RuntimeManifest,
    state_root: Path,
    progress: list[str],
) -> None:
    """Require public first-use preparation followed by a preparation-free warm command."""

    class NativeApprovalInteraction:
        """Approve only verifier-owned public command permission prompts."""

        def info(self, message: str) -> None:
            """Accept one verifier-owned disclosure without retaining sensitive detail."""
            del message

        def prompt(self, *args, **kwargs) -> ApprovalChoice:
            """Select one bounded session approval for the native verifier."""
            del args, kwargs
            return ApprovalChoice.SESSION

    private_root = backend.snapshot_store.parents[1]
    if private_root != state_root / "application":
        raise RuntimeError("Cold public verifier resolved unexpected private state.")
    workspace_path = private_root / "cold-public-workspace"
    workspace_path.mkdir(mode=0o700)
    (workspace_path / "tracked.txt").write_text("alpha\n", encoding="utf-8")
    permission_manager = PermissionManager(
        workspace_path,
        configuration_path=private_root / "cold-public-permissions.yaml",
        workspace_id="native-verifier-workspace",
        interaction=NativeApprovalInteraction(),  # type: ignore[arg-type]
    )
    definition = load_sandbox_image_definition()
    execution_permissions = ExecutionPermissionAdapter(
        permission_manager,
        SubjectIdentity(
            tool_id="run_command",
            publisher="loop",
            profile_id="ordinary-shell",
            profile_version=definition.source_version,
        ),
        "native-verifier-workspace",
        "2",
        "native-agent",
    )
    adapter = MacosProductAdapter(
        backend,
        runtime_manifest,
        workspace_path,
        private_root / "cold-public-execution",
        execution_permissions,
        OciResourceLimits(256 * 2**20, 64, 100000),
    )
    try:
        command_executor = SandboxCommandExecutor(
            ExecutionService(adapter),
            execution_permissions,
            "native-verifier-workspace",
            "native-agent",
            definition.source_version,
            runtime_resolver=adapter.ensure_sandbox,
        )
        registry = create_default_tool_registry(
            permission_manager=permission_manager,
            command_executor=command_executor,
        )

        def run(command: str, request_id: str) -> dict[str, object]:
            """Invoke one public command through lazy product composition."""
            return json.loads(
                registry.call(
                    "run_command",
                    json.dumps({"command": command, "cwd": "/workspace"}),
                    call_id=request_id,
                )
            )

        cold = run("printf cold-public", "native-cold-public")
        if (
            cold.get("ok") is not True
            or cold.get("result", {}).get("stdout", {}).get("content") != "cold-public"
        ):
            raise RuntimeError(f"Cold public run_command failed: {cold!r}.")
        after_cold = tuple(progress)
        warm = run("printf warm-public", "native-warm-public")
        if (
            warm.get("ok") is not True
            or warm.get("result", {}).get("stdout", {}).get("content") != "warm-public"
            or tuple(progress) != after_cold
        ):
            raise RuntimeError(f"Warm public run_command repeated preparation: {warm!r}.")
    finally:
        adapter.close()


def main() -> int:
    """Provision, attest, exercise, and remove one clean Apple-Silicon runtime."""
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("scripts/runtime-candidates/macos-arm64-v1.json"),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        help="New file receiving bounded versioned release-gate evidence after success.",
    )
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Native macOS verification requires Apple Silicon.")
    state_root = args.state_root.absolute()
    if state_root.exists() or state_root == Path.home() or state_root == Path.cwd():
        raise RuntimeError("Native verification requires a new isolated state root.")
    candidate = load_macos_runtime_candidate(args.candidate.resolve(strict=True))
    release = load_macos_runtime_release()
    if (
        candidate.schema_version != release.schema_version
        or candidate.artifacts != release.artifacts
    ):
        raise RuntimeError("Embedded macOS release manifest differs from the qualified candidate.")
    bootstrapper = RuntimeBootstrapper(
        release,
        state_root / "runtime",
        HttpxArtifactTransport(),
        {AcquisitionKind.ARCHIVE: ArchiveInstaller()},
        constants.RUNTIME_DEFAULT_DOWNLOAD_LIMIT,
    )
    preparation_progress: list[str] = []
    backend = MacosSandboxBackend(
        release,
        bootstrapper,
        InfrastructureProcessRunner(),
        state_root / "application",
        "native-verifier-workspace",
        progress=preparation_progress.append,
    )
    cleanup_complete = False
    performance: dict[str, object] | None = None
    try:
        _verify_cold_public_execution(
            backend,
            release,
            state_root,
            preparation_progress,
        )
        prepared = backend.prepare()
        try:
            required_progress = {
                "Creating isolated command sandbox…",
                "Starting isolated command sandbox…",
                "Building sandbox command image…",
                "Isolated command sandbox is ready.",
            }
            if not required_progress.issubset(preparation_progress):
                raise RuntimeError("Native sandbox preparation did not publish visible progress.")
            _verify_mounts(backend, prepared)
            performance = _verify_execution(backend, prepared, release)
        finally:
            try:
                prepared.stop()
            finally:
                prepared.delete()
            if prepared.instance.journal.exists() or prepared.runtime.lease.path.exists():
                raise RuntimeError("Native managed-runtime cleanup is incomplete.")
            cleanup_complete = True
    finally:
        if cleanup_complete:
            _remove_state_root(state_root)
    if args.evidence is not None:
        if performance is None:
            raise RuntimeError("Native evidence requires completed product verification.")
        _write_evidence(args.evidence, release, performance)
    return 0


def _write_evidence(path: Path, release, performance: dict[str, object]) -> None:
    """Write one bounded immutable result only after complete cleanup succeeds."""
    path = path.absolute()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "result": "passed",
        "platform": {"os": "macos", "architecture": "arm64", "virtualization": "vz"},
        "runtime_manifest_digest": release.digest,
        "artifacts": {artifact.artifact_id: artifact.digest for artifact in release.artifacts},
        "performance": performance,
        "checks": [
            "public-product-path",
            "native-isolation",
            "failure-cleanup",
            "restart-recovery",
        ],
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(encoded) > 16 * 1024:
        raise RuntimeError("Native evidence exceeded its bounded schema.")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = 0
        while written < len(encoded):
            written += os.write(descriptor, encoded[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_state_root(root: Path) -> None:
    """Remove one verifier-created private root, including immutable artifact modes."""
    if not root.exists():
        return
    metadata = root.lstat()
    if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("Native verification state root changed before cleanup.")

    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        os.chmod(current, current.lstat().st_mode | stat.S_IRWXU, follow_symlinks=False)
        for name in (*names, *files):
            target = current / name
            target_metadata = target.lstat()
            if stat.S_ISLNK(target_metadata.st_mode):
                continue
            owner_mode = stat.S_IRUSR | stat.S_IWUSR
            if stat.S_ISDIR(target_metadata.st_mode):
                owner_mode |= stat.S_IXUSR
            os.chmod(target, target_metadata.st_mode | owner_mode, follow_symlinks=False)
    shutil.rmtree(root)


if __name__ == "__main__":
    raise SystemExit(main())
