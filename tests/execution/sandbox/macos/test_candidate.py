"""Test the closed macOS managed-runtime candidate record."""

from __future__ import annotations

import json
import shutil
import stat
import time
from collections.abc import Callable
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest

from loop.execution.infrastructure import InfrastructureProcessResult
from loop.execution.runtime.bootstrap import InstalledRuntime
from loop.execution.runtime.image import load_sandbox_image_definition
from loop.execution.runtime.lease import create_lease
from loop.execution.runtime.models import (
    AcquisitionKind,
    Artifact,
    ArtifactRole,
    InstallLayout,
    PlatformSelector,
)
from loop.execution.sandbox.macos import (
    LimaConfigurationError,
    LimaInstanceContext,
    LimaLifecycleState,
    MacosCandidateError,
    MacosRequirements,
    ManagedLimaInstance,
    bootstrap_lima_executable,
    build_lima_instance_configuration,
    load_macos_runtime_candidate,
    load_macos_runtime_release,
    macos_artifact,
    macos_component_version,
    macos_entitlement_keys,
)


class InfrastructureOperation(StrEnum):
    """Name fake Lima calls recorded by lifecycle unit tests."""

    LIMA_LIST = "lima.list"
    LIMA_CREATE = "lima.create"
    LIMA_START = "lima.start"
    LIMA_STOP = "lima.stop"
    LIMA_FORCE_STOP = "lima.force_stop"
    LIMA_DELETE = "lima.delete"
    LIMA_HEALTH = "lima.health"


def _candidate_path() -> Path:
    """Return the repository-controlled Apple-Silicon candidate record."""
    return Path(__file__).parents[4] / "scripts/runtime-candidates/macos-arm64-v1.json"


def _snapshot_store(tmp_path: Path) -> Path:
    """Create one private workspace snapshot store for configuration tests."""
    store = (tmp_path / "snapshots").resolve()
    store.mkdir(mode=0o700, parents=True)
    return store


def test_candidate_contains_the_qualified_lima_guest_and_rootless_runtime_inputs():
    """The checked-in record pins each runtime download and Lima signing expectation."""
    candidate = load_macos_runtime_candidate(_candidate_path())
    lima = macos_artifact(candidate, "lima")
    guest = macos_artifact(candidate, "guest-image")
    nerdctl = macos_artifact(candidate, "nerdctl-full")
    assert lima.version == "2.2.0"
    assert (
        lima.digest.removeprefix("sha256:")
        == "bbdef91774885a0d05f7b048c4eb89ae2bcf3a0c252ae7ca7934e63df76d93c3"
    )
    assert lima.layout is not None and lima.layout.executables == ("bin/limactl",)
    assert "com.apple.security.virtualization" in macos_entitlement_keys(candidate)
    assert guest.source.endswith("ubuntu-26.04-server-cloudimg-arm64.img")
    assert nerdctl.source.endswith("nerdctl-full-2.2.0-linux-arm64.tar.gz")
    assert (
        macos_component_version(candidate, "nerdctl"),
        macos_component_version(candidate, "containerd"),
        macos_component_version(candidate, "runc"),
        macos_component_version(candidate, "buildkit"),
    ) == ("2.2.0", "2.2.0", "1.3.3", "0.25.2")
    assert load_sandbox_image_definition().base_image.endswith(
        "@sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251"
    )


def test_release_manifest_uses_the_same_closed_macos_validation() -> None:
    """Production loading accepts only the embedded release authority and inventory."""
    release = load_macos_runtime_release()
    assert release.mode.value == "release"
    with pytest.raises(MacosCandidateError, match="release manifest"):
        load_macos_runtime_release(b"{}")


def test_candidate_accessors_reject_unknown_or_missing_manifest_metadata() -> None:
    """Manifest access fails closed for unknown artifacts, components, and version facts."""
    candidate = load_macos_runtime_candidate(_candidate_path())
    with pytest.raises(MacosCandidateError, match="incomplete"):
        macos_artifact(candidate, "missing")
    with pytest.raises(MacosCandidateError, match="unsupported"):
        macos_component_version(candidate, "unknown")
    nerdctl = macos_artifact(candidate, "nerdctl-full").model_copy(update={"metadata": ()})
    incomplete = candidate.model_copy(
        update={
            "artifacts": tuple(
                nerdctl if item.artifact_id == "nerdctl-full" else item
                for item in candidate.artifacts
            )
        }
    )
    with pytest.raises(MacosCandidateError, match="metadata"):
        macos_component_version(incomplete, "nerdctl")


def test_candidate_rejects_an_unqualified_network_broker(tmp_path: Path) -> None:
    """The macOS candidate accepts only the pinned official distroless Envoy identity."""
    payload = json.loads(_candidate_path().read_text(encoding="utf-8"))
    envoy = next(item for item in payload["artifacts"] if item["artifact_id"] == "envoy")
    envoy["capabilities"] = []
    invalid = tmp_path / "invalid-envoy.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MacosCandidateError, match="candidate manifest"):
        load_macos_runtime_candidate(invalid)


def test_candidate_manifest_directly_selects_its_lima_authority() -> None:
    """The authoritative candidate directly selects the Lima artifact without projection."""
    manifest = load_macos_runtime_candidate(_candidate_path())
    artifact = manifest.select(
        PlatformSelector(os="macos", architecture="arm64"),
        frozenset({"sandbox-management"}),
    )[0]
    assert manifest.mode.value == "candidate"
    assert artifact.artifact_id == "lima"
    assert artifact.version == "2.2.0"
    assert artifact.role.value == "lima"
    assert artifact.platform == PlatformSelector(os="macos", architecture="arm64")
    assert artifact.acquisition.value == "archive"
    assert artifact.media_type == "application/gzip"
    assert artifact.capabilities == ("sandbox-management",)
    assert artifact.layout is not None
    assert artifact.layout.files == ("bin/limactl",)
    assert artifact.layout.executables == ("bin/limactl",)
    assert artifact.layout.allow_unlisted_files


def test_candidate_manifest_does_not_select_a_prepublished_core_image() -> None:
    """The authoritative candidate leaves Loop profiles to managed local construction."""
    manifest = load_macos_runtime_candidate(_candidate_path())
    assert all(artifact.role is not ArtifactRole.OCI_PROFILE for artifact in manifest.artifacts)
    with pytest.raises(ValueError, match="No runtime artifact"):
        manifest.select(
            PlatformSelector(os="linux", architecture="arm64"), frozenset({"command-core"})
        )


def test_requirements_can_only_adopt_the_candidate_entitlement_set(tmp_path: Path) -> None:
    """The macOS probe consumes its entitlement allowlist from checked-in authority."""
    candidate = load_macos_runtime_candidate(_candidate_path())
    requirements = MacosRequirements.from_candidate(
        object(),
        candidate,
        13,
        1,  # type: ignore[arg-type]
    )
    assert requirements.allowed_entitlements == macos_entitlement_keys(candidate)


def test_lima_configuration_is_vz_only_and_contains_no_ambient_shares_or_forwarding(
    tmp_path: Path,
) -> None:
    """Configuration shares only the private snapshot store and blocks ambient forwarding."""
    snapshot_store = _snapshot_store(tmp_path)
    configuration = build_lima_instance_configuration(
        load_macos_runtime_candidate(_candidate_path()), "loop-managed-1", snapshot_store
    )
    document = configuration.document.decode("utf-8")
    assert configuration.instance_name == "loop-managed-1"
    assert 'vmType: "vz"' in document
    assert 'arch: "aarch64"' in document
    assert 'mountType: "virtiofs"' in document
    assert f"- location: {json.dumps(str(snapshot_store))}" in document
    assert 'mountPoint: "/run/loop/snapshots"' in document
    assert "writable: false" in document
    assert document.count("ignore: true") == 4
    assert "overVsock: true" in document
    assert "- vzNAT: true" in document
    assert "localPort: 0" in document
    assert "rosetta:\n      enabled: false" in document
    assert "upgradePackages: false" in document
    assert "current" not in document
    assert "~" not in document
    assert configuration.manifest_digest == load_macos_runtime_candidate(_candidate_path()).digest


@pytest.mark.parametrize("name", ("Loop", "loop_managed", "loop/managed", "x" * 64))
def test_lima_configuration_rejects_nonprivate_instance_names(name: str) -> None:
    """Instance names cannot introduce a path, user state, or CLI option."""
    with pytest.raises(ValueError, match="instance name"):
        build_lima_instance_configuration(
            load_macos_runtime_candidate(_candidate_path()), name, Path.cwd()
        )


@pytest.mark.parametrize("kind", ("missing", "relative", "symlink", "file", "broad"))
def test_lima_configuration_rejects_unsafe_snapshot_stores(tmp_path: Path, kind: str) -> None:
    """Only an existing absolute private non-link directory can cross into VirtioFS."""
    store = tmp_path / "store"
    if kind == "relative":
        store = Path("relative-store")
    elif kind == "symlink":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        store.symlink_to(target)
    elif kind == "file":
        store.write_text("not a directory", encoding="utf-8")
    elif kind == "broad":
        store.mkdir(mode=0o755)
    with pytest.raises(LimaConfigurationError, match="snapshot store"):
        build_lima_instance_configuration(
            load_macos_runtime_candidate(_candidate_path()), "loop-managed-1", store
        )


class _LifecycleRunner:
    """Implement a stateful fake of the closed Lima operation surface."""

    status: str
    failures: set[InfrastructureOperation]
    operations: list[InfrastructureOperation]
    output_limit: int
    mutate_record: Callable[[dict[str, Any]], None] | None
    event_override: object | None
    health_override: bytes | None
    truncate: set[InfrastructureOperation]
    raise_operations: set[InfrastructureOperation]

    def __init__(self, failures: set[InfrastructureOperation] | None = None) -> None:
        self.status = "Absent"
        self.failures = failures or set()
        self.operations = []
        self.output_limit = 1024 * 1024
        self.mutate_record = None
        self.event_override = None
        self.health_override = None
        self.truncate = set()
        self.raise_operations = set()

    def run_lima(
        self,
        context: object,
        operation: InfrastructureOperation,
        configuration: bytes | None = None,
    ) -> InfrastructureProcessResult:
        """Mutate fake state and return exact pinned Lima evidence."""
        typed = context  # Keep the fake signature aligned with the runner protocol.
        self.operations.append(operation)
        if operation in self.raise_operations:
            raise RuntimeError("fake cancellation")
        if operation in self.failures:
            return InfrastructureProcessResult(1, b"", b"", False, False)
        if operation in self.truncate:
            return InfrastructureProcessResult(0, b"{}", b"", True, False)
        if operation is InfrastructureOperation.LIMA_CREATE:
            self.status = "Stopped"
            directory = typed.instance_directory
            directory.mkdir()
            (directory / "lima.yaml").write_bytes(configuration or b"")
        elif operation is InfrastructureOperation.LIMA_START:
            self.status = "Running"
            self._write_running_evidence(typed)
        elif operation in {
            InfrastructureOperation.LIMA_STOP,
            InfrastructureOperation.LIMA_FORCE_STOP,
        }:
            self.status = "Stopped"
        elif operation is InfrastructureOperation.LIMA_DELETE:
            self.status = "Absent"
            shutil.rmtree(typed.instance_directory, ignore_errors=True)
        if operation is InfrastructureOperation.LIMA_LIST:
            record = self._record(typed)
            if self.mutate_record is not None:
                self.mutate_record(record)
            return InfrastructureProcessResult(
                0, json.dumps(record, separators=(",", ":")).encode() + b"\n", b"", False, False
            )
        if operation is InfrastructureOperation.LIMA_HEALTH:
            return InfrastructureProcessResult(
                0,
                self.health_override or f"aarch64\n{typed.health_nonce}\n".encode(),
                b"",
                False,
                False,
            )
        return InfrastructureProcessResult(0, b"", b"", False, False)

    def create(self, context: object, configuration: bytes) -> InfrastructureProcessResult:
        """Record an explicit create call."""
        return self.run_lima(context, InfrastructureOperation.LIMA_CREATE, configuration)

    def start(self, context: object) -> InfrastructureProcessResult:
        """Record an explicit start call."""
        return self.run_lima(context, InfrastructureOperation.LIMA_START)

    def list(self, context: object) -> InfrastructureProcessResult:
        """Record an explicit list call."""
        return self.run_lima(context, InfrastructureOperation.LIMA_LIST)

    def stop(self, context: object, *, force: bool = False) -> InfrastructureProcessResult:
        """Record an explicit normal or forced stop call."""
        operation = (
            InfrastructureOperation.LIMA_FORCE_STOP if force else InfrastructureOperation.LIMA_STOP
        )
        return self.run_lima(context, operation)

    def delete(self, context: object) -> InfrastructureProcessResult:
        """Record an explicit delete call."""
        return self.run_lima(context, InfrastructureOperation.LIMA_DELETE)

    def health(self, context: object) -> InfrastructureProcessResult:
        """Record an explicit health call."""
        return self.run_lima(context, InfrastructureOperation.LIMA_HEALTH)

    def _record(self, context: Any) -> dict[str, Any]:
        """Return exact list JSON for current fake state."""
        value: dict[str, Any] = {
            "name": context.instance_name,
            "status": self.status,
            "vmType": "vz",
            "arch": "aarch64",
            "dir": str(context.instance_directory),
            "protected": False,
            "limaVersion": "v2.2.0",
            "HostOS": "darwin",
            "HostArch": "aarch64",
            "LimaHome": str(context.lima_home),
            "IdentityFile": str(context.lima_home / "_config" / "user"),
        }
        if self.status == "Running":
            value.update(
                sshAddress="127.0.0.1",
                sshLocalPort=60022,
                sshConfigFile=str(context.instance_directory / "ssh.config"),
            )
        return value

    def _write_running_evidence(self, context: Any) -> None:
        """Write the two exact private files consumed by running attestation."""
        (context.instance_directory / "ssh.config").write_text(
            "Host loop-managed-1\n"
            "  Hostname 127.0.0.1\n"
            "  Port 60022\n"
            f'  IdentityFile "{context.lima_home}/_config/user"\n'
            f'  ControlPath "{context.lima_home}/loop-managed-1/ssh.sock"\n',
            encoding="utf-8",
        )
        event = {
            "time": "2026-09-18T00:00:00Z",
            "status": {
                "running": True,
                "vsock": {
                    "type": "started",
                    "hostAddr": "127.0.0.1:60022",
                    "vsockPort": 22,
                },
            },
        }
        events = self.event_override if self.event_override is not None else event
        values = events if isinstance(events, list) else [events]
        (context.instance_directory / "ha.stdout.log").write_text(
            "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
        )


def _lima_executable(tmp_path: Path, manifest_digest: str):
    """Create a real lease-bound fake Lima executable descriptor."""
    root = tmp_path / "runtime"
    artifact_root = root / "artifacts" / "lima" / ("b" * 64)
    artifact_root.mkdir(parents=True)
    executable_path = artifact_root / "bin" / "limactl"
    executable_path.parent.mkdir()
    executable_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable_path.chmod(stat.S_IRUSR | stat.S_IXUSR)
    import hashlib

    identity = hashlib.sha256(executable_path.read_bytes()).hexdigest()
    artifact = Artifact(
        artifact_id="lima",
        version="2.2.0",
        role=ArtifactRole.LIMA,
        platform=PlatformSelector(os="macos", architecture="arm64"),
        source="https://example.test/lima",
        size=executable_path.stat().st_size,
        digest="b" * 64,
        acquisition=AcquisitionKind.FILE,
        media_type="application/octet-stream",
        layout=InstallLayout(
            files=("bin/limactl",),
            executables=("bin/limactl",),
            identities={"bin/limactl": identity},
        ),
        sbom="https://example.test/sbom",
        notices="https://example.test/notices",
    )
    lease = create_lease(root, manifest_digest, "a" * 64, 60)
    runtime = InstalledRuntime(
        manifest_digest, "a" * 64, {"lima": artifact_root}, lease, {"lima": artifact}, root
    )
    return runtime.executable("lima", "bin/limactl")


def _managed_instance(
    tmp_path: Path, failures: set[InfrastructureOperation] | None = None
) -> tuple[ManagedLimaInstance, _LifecycleRunner]:
    """Build a journal-owned lifecycle with isolated sealed test evidence."""
    snapshot_store = _snapshot_store(tmp_path)
    configuration = build_lima_instance_configuration(
        load_macos_runtime_candidate(_candidate_path()), "loop-managed-1", snapshot_store
    )
    context = LimaInstanceContext.create(
        _lima_executable(tmp_path, configuration.manifest_digest),
        (tmp_path / "state").resolve(),
        configuration,
        1,
    )
    runner = _LifecycleRunner(failures)
    return ManagedLimaInstance(runner, context, configuration), runner  # type: ignore[arg-type]


def test_managed_lima_lifecycle_journals_create_start_and_reset(tmp_path: Path) -> None:
    """A successful lifecycle records ownership before actions and removes it at reset."""
    instance, runner = _managed_instance(tmp_path)
    instance.create()
    instance.start()
    assert instance.state is LimaLifecycleState.RUNNING
    instance.stop()
    instance.start()
    instance.stop(force=True)
    instance.reset()
    assert not instance.journal.exists()
    assert InfrastructureOperation.LIMA_HEALTH in runner.operations


def test_managed_lima_lifecycle_rolls_back_a_failed_start(tmp_path: Path) -> None:
    """A failed start with successful cleanup returns to absent and never succeeds."""
    instance, runner = _managed_instance(tmp_path, {InfrastructureOperation.LIMA_START})
    instance.create()
    with pytest.raises(ValueError, match="start or attestation failed"):
        instance.start()
    assert instance.state is LimaLifecycleState.ABSENT
    assert runner.operations[-2:] == [
        InfrastructureOperation.LIMA_FORCE_STOP,
        InfrastructureOperation.LIMA_DELETE,
    ]


def test_managed_lima_adopts_only_an_exact_journal_and_is_idempotent(tmp_path: Path) -> None:
    """Exact owned state is adopted while repeated stop/delete remain safe."""
    instance, runner = _managed_instance(tmp_path)
    instance.create()
    before = list(runner.operations)
    instance.create()
    assert runner.operations == before + [InfrastructureOperation.LIMA_LIST]
    instance.start()
    instance.stop()
    before = list(runner.operations)
    instance.stop()
    assert runner.operations == before + [InfrastructureOperation.LIMA_LIST]
    instance.delete()
    instance.delete()
    assert instance.state is LimaLifecycleState.ABSENT


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "Broken"),
        ("vmType", "qemu"),
        ("arch", "x86_64"),
        ("dir", "/tmp/foreign"),
        ("name", "foreign"),
        ("unknown", True),
    ],
)
def test_managed_lima_rejects_contradictory_or_unknown_list_evidence(
    tmp_path: Path, field: str, value: object
) -> None:
    """Wrong state, identity, driver, architecture, path, and schema fail closed."""
    instance, runner = _managed_instance(tmp_path)
    runner.mutate_record = lambda record: record.__setitem__(field, value)
    with pytest.raises(LimaConfigurationError, match="creation failed"):
        instance.create()
    assert instance.state is LimaLifecycleState.ABSENT
    assert InfrastructureOperation.LIMA_DELETE in runner.operations


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"status": {"vsock": {"type": "failed", "reason": "no guest"}}},
        {"status": {"vsock": {"type": "skipped", "reason": "disabled"}}},
        {"status": {"vsock": {"type": "started", "hostAddr": "127.0.0.1:1", "vsockPort": 22}}},
        {"unknown": True},
        {"status": {"unknown": True}},
        {"status": {"vsock": "malformed"}},
    ],
)
def test_managed_lima_rejects_missing_failed_or_malformed_vsock_events(
    tmp_path: Path, event: object
) -> None:
    """A single exact structured VSOCK-started event is mandatory."""
    instance, runner = _managed_instance(tmp_path)
    instance.create()
    runner.event_override = event
    with pytest.raises(LimaConfigurationError, match="start or attestation failed"):
        instance.start()
    assert instance.state is LimaLifecycleState.ABSENT


def test_managed_lima_rejects_duplicate_vsock_event_bad_health_and_nonloopback(
    tmp_path: Path,
) -> None:
    """Duplicate readiness, stale health, and remote management all fail closed."""
    instance, runner = _managed_instance(tmp_path / "duplicate")
    instance.create()
    started = {
        "status": {
            "vsock": {
                "type": "started",
                "hostAddr": "127.0.0.1:60022",
                "vsockPort": 22,
            }
        }
    }
    runner.event_override = [started, started]
    with pytest.raises(LimaConfigurationError):
        instance.start()

    instance, runner = _managed_instance(tmp_path / "health")
    instance.create()
    runner.health_override = b"aarch64\nstale\n"
    with pytest.raises(LimaConfigurationError):
        instance.start()

    instance, runner = _managed_instance(tmp_path / "remote")
    instance.create()
    runner.mutate_record = lambda record: record.update(sshAddress="192.0.2.1")
    with pytest.raises(LimaConfigurationError):
        instance.start()


def test_managed_lima_rejects_changed_config_missing_agent_and_truncated_evidence(
    tmp_path: Path,
) -> None:
    """Mutable config, lost host agent, and bounded-output overflow cannot attest."""
    instance, runner = _managed_instance(tmp_path / "config")
    instance.create()
    text = (instance.context.instance_directory / "lima.yaml").read_text()
    (instance.context.instance_directory / "lima.yaml").write_text(
        text.replace('vmType: "vz"', 'vmType: "qemu"')
    )
    with pytest.raises(LimaConfigurationError):
        instance.start()

    instance, runner = _managed_instance(tmp_path / "missing")
    instance.create()
    runner.event_override = []
    with pytest.raises(LimaConfigurationError):
        instance.start()

    instance, runner = _managed_instance(tmp_path / "truncated")
    runner.truncate.add(InfrastructureOperation.LIMA_LIST)
    with pytest.raises(LimaConfigurationError):
        instance.create()


def test_managed_lima_cleanup_failure_is_durable_and_foreign_journal_is_untouched(
    tmp_path: Path,
) -> None:
    """Cleanup failure stays recoverable and mismatched ownership launches nothing."""
    instance, runner = _managed_instance(
        tmp_path / "cleanup",
        {InfrastructureOperation.LIMA_START, InfrastructureOperation.LIMA_DELETE},
    )
    instance.create()
    with pytest.raises(LimaConfigurationError):
        instance.start()
    assert (
        json.loads(instance.journal.read_text())["last_completed_transition"] == "rollback-failed"
    )

    instance, runner = _managed_instance(tmp_path / "foreign")
    instance.create()
    value = json.loads(instance.journal.read_text())
    value["manifest_digest"] = "sha256:" + "0" * 64
    instance.journal.write_text(json.dumps(value))
    runner.operations.clear()
    with pytest.raises(LimaConfigurationError, match="ownership"):
        instance.start()
    assert runner.operations == []


def test_managed_lima_resets_only_an_attested_obsolete_release(tmp_path: Path) -> None:
    """Prior-release cleanup requires its exact persisted config and original ownership marker."""
    instance, runner = _managed_instance(tmp_path / "valid")
    instance.create()
    value = json.loads(instance.journal.read_text())
    value["manifest_digest"] = "sha256:" + "0" * 64
    instance.journal.write_text(json.dumps(value))

    instance.reset_obsolete()

    assert not instance.journal.exists()
    assert runner.operations[-2:] == [
        InfrastructureOperation.LIMA_FORCE_STOP,
        InfrastructureOperation.LIMA_DELETE,
    ]

    instance, runner = _managed_instance(tmp_path / "tampered")
    instance.create()
    value = json.loads(instance.journal.read_text())
    value["manifest_digest"] = "sha256:" + "0" * 64
    instance.journal.write_text(json.dumps(value))
    (instance.context.instance_directory / "lima.yaml").write_text("tampered")
    runner.operations.clear()

    with pytest.raises(LimaConfigurationError, match="ownership"):
        instance.reset_obsolete()

    assert runner.operations == []


def test_managed_lima_obsolete_reset_fails_closed_on_missing_or_failed_evidence(
    tmp_path: Path,
) -> None:
    """Missing config, truncated cleanup, and failed deletion retain the obsolete journal."""
    instance, runner = _managed_instance(tmp_path / "missing")
    instance.create()
    value = json.loads(instance.journal.read_text())
    value["manifest_digest"] = "sha256:" + "0" * 64
    instance.journal.write_text(json.dumps(value))
    (instance.context.instance_directory / "lima.yaml").unlink()
    before = list(runner.operations)
    with pytest.raises(LimaConfigurationError, match="configuration is invalid"):
        instance.reset_obsolete()
    assert runner.operations == before

    instance, runner = _managed_instance(tmp_path / "truncated")
    instance.create()
    value = json.loads(instance.journal.read_text())
    value["manifest_digest"] = "sha256:" + "0" * 64
    instance.journal.write_text(json.dumps(value))
    runner.truncate.add(InfrastructureOperation.LIMA_FORCE_STOP)
    with pytest.raises(LimaConfigurationError, match="obsolete Lima reset failed"):
        instance.reset_obsolete()
    assert instance.journal.exists()

    instance, runner = _managed_instance(
        tmp_path / "delete",
        {InfrastructureOperation.LIMA_DELETE},
    )
    instance.create()
    value = json.loads(instance.journal.read_text())
    value["manifest_digest"] = "sha256:" + "0" * 64
    instance.journal.write_text(json.dumps(value))
    with pytest.raises(LimaConfigurationError, match="obsolete Lima reset failed"):
        instance.reset_obsolete()
    assert instance.journal.exists()


def test_managed_lima_rejects_malformed_typed_journal_fields(tmp_path: Path) -> None:
    """Type-confused ownership fields cannot enter current or obsolete lifecycle recovery."""
    instance, runner = _managed_instance(tmp_path)
    instance.create()
    value = json.loads(instance.journal.read_text())
    value["epoch"] = True
    instance.journal.write_text(json.dumps(value))
    runner.operations.clear()

    with pytest.raises(LimaConfigurationError, match="journal is invalid"):
        instance.reset_obsolete()

    assert runner.operations == []


def test_lima_context_rejects_expired_lease_replaced_root_and_forbidden_roots(
    tmp_path: Path,
) -> None:
    """Every operation detects lease expiry and private-root replacement."""
    configuration = build_lima_instance_configuration(
        load_macos_runtime_candidate(_candidate_path()),
        "loop-managed-1",
        _snapshot_store(tmp_path),
    )
    executable = _lima_executable(tmp_path / "expired", configuration.manifest_digest)
    context = LimaInstanceContext.create(
        executable, (tmp_path / "expired-state").resolve(), configuration, 1
    )
    payload = json.loads(executable.lease.path.read_text())
    payload["expiry"] = time.time() - 1
    executable.lease.path.write_text(json.dumps(payload))
    with pytest.raises(LimaConfigurationError, match="no longer valid"):
        context.validate()

    executable = _lima_executable(tmp_path / "replaced", configuration.manifest_digest)
    root = (tmp_path / "replaced-state").resolve()
    context = LimaInstanceContext.create(executable, root, configuration, 1)
    root.rename(tmp_path / "old-state")
    root.mkdir()
    (root / "lima").mkdir()
    with pytest.raises(LimaConfigurationError, match="no longer valid"):
        context.validate()

    executable = _lima_executable(tmp_path / "forbidden", configuration.manifest_digest)
    with pytest.raises(LimaConfigurationError, match="isolated"):
        LimaInstanceContext.create(executable, executable.runtime_root, configuration, 1)


def test_lima_context_and_owner_reject_all_malformed_construction_paths(tmp_path: Path) -> None:
    """Invalid context inputs, linked homes, and mismatched configuration fail before launch."""
    configuration = build_lima_instance_configuration(
        load_macos_runtime_candidate(_candidate_path()),
        "loop-managed-1",
        _snapshot_store(tmp_path),
    )
    executable = _lima_executable(tmp_path / "runtime", configuration.manifest_digest)
    with pytest.raises(LimaConfigurationError, match="context"):
        LimaInstanceContext.create(executable, Path("relative"), configuration, 1)
    file_root = tmp_path / "file-root"
    file_root.write_text("not a directory")
    with pytest.raises(LimaConfigurationError, match="state root"):
        LimaInstanceContext.create(executable, file_root.resolve(), configuration, 1)
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(tmp_path)
    with pytest.raises(LimaConfigurationError, match="isolated"):
        LimaInstanceContext.create(executable, linked_root, configuration, 1)
    home_root = tmp_path / "linked-home"
    home_root.mkdir()
    (home_root / "lima").symlink_to(tmp_path)
    with pytest.raises(LimaConfigurationError, match="home is invalid"):
        LimaInstanceContext.create(executable, home_root, configuration, 1)
    user_root = tmp_path / "linked-user"
    user_root.mkdir()
    (user_root / "home").symlink_to(tmp_path)
    with pytest.raises(LimaConfigurationError, match="user home"):
        LimaInstanceContext.create(executable, user_root, configuration, 1)

    context = LimaInstanceContext.create(executable, tmp_path / "valid", configuration, 1)
    with pytest.raises(LimaConfigurationError, match="configuration identity"):
        ManagedLimaInstance(
            _LifecycleRunner(),
            context,
            replace(configuration, manifest_digest="sha256:" + "0" * 64),
        )


def test_lima_context_accepts_private_application_state_below_the_user_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal platform application data remains eligible beneath the user's home."""
    home = tmp_path / "home-root"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    configuration = build_lima_instance_configuration(
        load_macos_runtime_candidate(_candidate_path()),
        "loop-managed-1",
        _snapshot_store(tmp_path),
    )
    executable = _lima_executable(tmp_path / "runtime-home", configuration.manifest_digest)
    state_root = home / "Library" / "Application Support" / "loop" / "instance"
    context = LimaInstanceContext.create(executable, state_root, configuration, 1)
    assert context.state_root == state_root.resolve()


def test_managed_lima_failure_transitions_and_recovery_are_closed(tmp_path: Path) -> None:
    """Stop, delete, reset, re-attestation, and rollback failures remain journal-owned."""
    instance, runner = _managed_instance(tmp_path / "attest")
    instance.create()
    instance.start()
    runner.health_override = b"bad"
    with pytest.raises(LimaConfigurationError, match="running attestation"):
        instance.attest_running()

    instance, runner = _managed_instance(tmp_path / "stop")
    instance.create()
    instance.start()
    runner.failures.add(InfrastructureOperation.LIMA_STOP)
    with pytest.raises(LimaConfigurationError, match="stop failed"):
        instance.stop()

    instance, runner = _managed_instance(tmp_path / "delete")
    instance.create()
    instance.start()
    runner.failures.add(InfrastructureOperation.LIMA_DELETE)
    with pytest.raises(LimaConfigurationError, match="deletion failed"):
        instance.delete()
    assert InfrastructureOperation.LIMA_FORCE_STOP in runner.operations

    instance, runner = _managed_instance(tmp_path / "reset")
    instance.create()
    runner.truncate.add(InfrastructureOperation.LIMA_FORCE_STOP)
    with pytest.raises(LimaConfigurationError, match="reset failed"):
        instance.reset()

    instance, runner = _managed_instance(tmp_path / "runtime")
    runner.failures.add(InfrastructureOperation.LIMA_CREATE)
    runner.raise_operations.add(InfrastructureOperation.LIMA_FORCE_STOP)
    with pytest.raises(LimaConfigurationError, match="creation failed"):
        instance.create()
    assert instance.state is LimaLifecycleState.FAILED_CLEANUP


def test_managed_lima_rejects_malformed_files_and_transition_journals(tmp_path: Path) -> None:
    """Malformed list, config, SSH, journal, and unfinished state all fail closed."""
    instance, runner = _managed_instance(tmp_path / "list")
    runner.mutate_record = lambda record: record.clear()
    with pytest.raises(LimaConfigurationError):
        instance.create()

    instance, runner = _managed_instance(tmp_path / "config")
    instance.create()
    (instance.context.instance_directory / "lima.yaml").unlink()
    with pytest.raises(LimaConfigurationError):
        instance.start()

    instance, runner = _managed_instance(tmp_path / "ssh")
    instance.create()
    original = runner._write_running_evidence

    def bad_ssh(context: Any) -> None:
        original(context)
        (context.instance_directory / "ssh.config").write_text("Host incomplete\n")

    runner._write_running_evidence = bad_ssh  # type: ignore[method-assign]
    with pytest.raises(LimaConfigurationError):
        instance.start()

    instance, runner = _managed_instance(tmp_path / "journal")
    instance.create()
    instance.journal.write_text("not json")
    with pytest.raises(LimaConfigurationError, match="journal is invalid"):
        instance.start()

    instance, runner = _managed_instance(tmp_path / "transition")
    instance.create()
    value = json.loads(instance.journal.read_text())
    value["intended_state"] = "CREATING"
    instance.journal.write_text(json.dumps(value))
    with pytest.raises(LimaConfigurationError, match="requires recovery"):
        instance.create()


def test_bootstrap_lima_executable_requests_only_the_candidate_capability(tmp_path: Path) -> None:
    """The composition bridge returns only the manifest-declared Lima executable."""
    candidate = load_macos_runtime_candidate(_candidate_path())
    executable = _lima_executable(tmp_path, candidate.digest)
    seen: list[object] = []

    class _Runtime:
        def executable(self, artifact_id: str, relative_path: str):
            assert (artifact_id, relative_path) == ("lima", "bin/limactl")
            return executable

    class _Bootstrapper:
        def ensure(self, requirement: object):
            seen.append(requirement)
            return _Runtime()

    assert bootstrap_lima_executable(_Bootstrapper(), candidate) is executable  # type: ignore[arg-type]
    assert seen[0].capabilities == frozenset({"sandbox-management"})  # type: ignore[attr-defined]
    lima = macos_artifact(candidate, "lima").model_copy(update={"layout": None})
    malformed = candidate.model_copy(
        update={
            "artifacts": tuple(
                lima if item.artifact_id == "lima" else item for item in candidate.artifacts
            )
        }
    )
    with pytest.raises(LimaConfigurationError, match="executable layout"):
        bootstrap_lima_executable(_Bootstrapper(), malformed)  # type: ignore[arg-type]


def test_managed_lima_remaining_fail_closed_file_and_journal_edges(tmp_path: Path) -> None:
    """Malformed bounded files and illegal calls exercise every remaining closed edge."""
    instance, runner = _managed_instance(tmp_path / "edges")
    instance.reset()
    instance.create()
    with pytest.raises(LimaConfigurationError, match="lifecycle transition"):
        instance.start()
        instance.start()
    instance.reset()

    instance, runner = _managed_instance(tmp_path / "json")
    runner.mutate_record = lambda record: record.update(dir=None)
    with pytest.raises(LimaConfigurationError, match="creation failed"):
        instance.create()

    instance, runner = _managed_instance(tmp_path / "direct")
    instance.create()
    record = runner._record(instance.context)
    record.update(
        sshAddress="127.0.0.1",
        sshLocalPort=60022,
        sshConfigFile=str(instance.context.instance_directory / "ssh.config"),
    )
    ssh = instance.context.instance_directory / "ssh.config"
    ssh.write_text("Hostname 127.0.0.1\nPort 60022\nIdentityFile relative\nControlPath relative\n")
    with pytest.raises(LimaConfigurationError, match="escaped"):
        instance._attest_ssh(record)
    event = instance.context.instance_directory / "ha.stdout.log"
    event.write_bytes(b"x" * (runner.output_limit + 1))
    with pytest.raises(LimaConfigurationError, match="host-agent evidence"):
        instance._attest_vsock(record)
    value = json.loads(instance.journal.read_text())
    del value["epoch"]
    instance.journal.write_text(json.dumps(value))
    with pytest.raises(LimaConfigurationError, match="journal is invalid"):
        instance.start()

    instance, runner = _managed_instance(tmp_path / "record")
    instance.journal.with_suffix(".new").write_text("occupied")
    with pytest.raises(LimaConfigurationError, match="could not be persisted"):
        instance.create()

    instance, runner = _managed_instance(tmp_path / "adopted-rollback")
    instance._record(LimaLifecycleState.STARTING, "adopted", False)
    instance._rollback(False)
    assert instance.state is LimaLifecycleState.STOPPED
    assert InfrastructureOperation.LIMA_DELETE not in runner.operations


def test_context_wraps_home_creation_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Filesystem races while creating either private home become typed failures."""
    configuration = build_lima_instance_configuration(
        load_macos_runtime_candidate(_candidate_path()),
        "loop-managed-1",
        _snapshot_store(tmp_path),
    )
    executable = _lima_executable(tmp_path / "runtime", configuration.manifest_digest)
    original = Path.mkdir

    def fail_lima(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == "lima":
            raise OSError("race")
        original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_lima)
    with pytest.raises(LimaConfigurationError, match="home is invalid"):
        LimaInstanceContext.create(executable, tmp_path / "state", configuration, 1)
    monkeypatch.setattr(Path, "mkdir", original)

    def fail_home(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == "home":
            raise OSError("race")
        original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_home)
    with pytest.raises(LimaConfigurationError, match="user home is invalid"):
        LimaInstanceContext.create(executable, tmp_path / "state-2", configuration, 1)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["artifacts"][0].update(source="http://example.test/lima"),
        lambda value: value["artifacts"][0]["layout"]["identities"].update(
            {"bin/limactl": "A" * 64}
        ),
        lambda value: value["artifacts"][0].update(metadata=[]),
        lambda value: value["artifacts"][0].update(capabilities=[]),
        lambda value: value["artifacts"][1].update(role="nerdctl"),
        lambda value: value.update(artifacts=value["artifacts"][:-1]),
        lambda value: value["artifacts"][2].update(metadata=[]),
        lambda value: value["artifacts"][3].update(source="registry.invalid/envoy:latest"),
        lambda value: value["artifacts"][3].update(digest="sha256:" + "f" * 64),
        lambda value: value.update(unexpected=True),
    ],
)
def test_candidate_rejects_untrusted_or_incomplete_records(tmp_path: Path, mutate: object) -> None:
    """Malformed candidate authority fails closed before any runtime action."""
    value = json.loads(_candidate_path().read_text(encoding="utf-8"))
    mutate(value)  # type: ignore[operator]
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(MacosCandidateError, match="candidate"):
        load_macos_runtime_candidate(path)
