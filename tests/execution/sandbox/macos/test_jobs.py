"""Test authenticated durable jobs through isolated macOS and OCI collaborators."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.execution import (
    Capability,
    DeltaEffect,
    ExecutionLease,
    ExecutionMode,
    JobHandle,
    NetworkConnectionLease,
    NetworkProtocol,
    ShellExecutionRequest,
    TerminalMode,
)
from loop.execution.infrastructure import InfrastructureProcessResult
from loop.execution.runtime.image import RuntimeImage
from loop.execution.runtime.models import OciImageIdentity, PlatformSelector
from loop.execution.sandbox.macos import (
    DurableJobState,
    DurableJobStore,
    MacosAttemptLayer,
    MacosDurableJobManager,
    MacosLeasedWorkspace,
    load_macos_runtime_candidate,
)
from loop.execution.sandbox.oci.control import OciSignal
from loop.execution.sandbox.oci.process import OciStreamFrame
from loop.execution.sandbox.oci.spec import OciResourceLimits
from loop.execution.service import AttemptObserver
from loop.execution.state_machine import AttemptStateMachine
from loop.execution.vfs import CanonicalDelta, CanonicalDeltaEntry, ObjectKind
from loop.utils import json_encode, sha256_digest


def _candidate():
    """Load the repository-controlled runtime candidate."""
    return load_macos_runtime_candidate(
        Path(__file__).parents[4] / "scripts/runtime-candidates/macos-arm64-v1.json"
    )


def _request(*, write: bool = False, terminal: TerminalMode = TerminalMode.PTY):
    """Build one explicit durable request with a fresh operation lease."""
    capabilities = {Capability.WORKSPACE_READ, Capability.PROCESS_SPAWN, Capability.PROCESS_SIGNAL}
    if write:
        capabilities.add(Capability.WORKSPACE_WRITE)
    return ShellExecutionRequest(
        request_id="job",
        lease=ExecutionLease(
            lease_id="start-lease",
            workspace_id="workspace",
            agent_run_id="agent",
            policy_version="policy",
            runtime_digest=(
                "sha256:071fa5de01a240dbef5be09d69f8fef2f89d68445d9175393773ee389b6f5935"
            ),
            expires_at_ns=time.monotonic_ns() + 10**12,
            capabilities=frozenset(capabilities),
        ),
        script="sleep 60",
        mode=ExecutionMode.DURABLE_JOB,
        terminal=terminal,
        terminal_columns=80 if terminal is TerminalMode.PTY else None,
        terminal_rows=24 if terminal is TerminalMode.PTY else None,
    )


def _operation_lease(request: ShellExecutionRequest, **changes: object) -> ExecutionLease:
    """Return a fresh operation lease bound to the request identities."""
    values = {
        "lease_id": "operation-lease",
        "expires_at_ns": time.monotonic_ns() + 10**12,
    }
    values.update(changes)
    return request.lease.model_copy(update=values)


class _Session:
    """Record authorized durable attachment operations."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.sizes: list[tuple[int, int]] = []
        self.closed = False
        self.stdin_closed = False

    def write(self, data: bytes) -> None:
        """Record one stdin frame."""
        self.writes.append(data)

    def resize(self, columns: int, rows: int) -> None:
        """Record one PTY size."""
        self.sizes.append((columns, rows))

    def close(self) -> None:
        """Close only the management attachment."""
        self.closed = True

    def detach(self) -> None:
        """Gracefully detach only the management attachment."""
        self.closed = True

    def close_stdin(self) -> None:
        """Provide the complete OCI session interface."""
        self.stdin_closed = True

    def read(self, timeout: float | None = None) -> OciStreamFrame | None:
        """Return no queued output."""
        del timeout
        return None

    def wait(self, cancellation=lambda: False) -> int:
        """Return a successful management status."""
        del cancellation
        return 0


class _Control:
    """Emulate fixed-shape OCI operations without processes or sockets."""

    def __init__(self) -> None:
        self.labels: dict[str, str] = {}
        self.state: dict[str, object] = {"Running": False, "Paused": False, "Status": "created"}
        self.calls: list[str] = []
        self.session = _Session()
        self.fail: str | None = None
        self.inspect_payload: object | None = None
        self.inspect_payloads: list[object] = []

    def _result(self, operation: str) -> InfrastructureProcessResult:
        """Record an operation and return its configured bounded result."""
        self.calls.append(operation)
        return InfrastructureProcessResult(
            1 if self.fail == operation else 0,
            b"",
            b"failed" if self.fail == operation else b"",
            False,
            False,
        )

    def run_job(self, spec, source: str, **kwargs: object):
        """Create an attachable job and return its initial management session."""
        assert source.endswith("/merged") and spec.detached
        assert kwargs["deadline_seconds"] == 30.0
        self.labels = dict(spec.labels)
        self.calls.append(f"run-job:{spec.terminal.value}")
        if self.fail == "run-job":
            raise RuntimeError("run failed")
        if self.fail != "start-state" and self.state["Status"] == "created":
            self.state = {"Running": True, "Paused": False, "Status": "running"}
        return self.session

    def start_attached(self, container: str, **kwargs: object):
        """Start through one attachment that accepts the fixed detach sequence."""
        assert container == "loop-job"
        self.calls.append(f"start-attached:{kwargs['pty']}")
        if self.fail == "start-attached":
            raise RuntimeError("start failed")
        if self.fail != "start-state":
            self.state = {"Running": True, "Paused": False, "Status": "running"}
        return self.session

    def inspect_container(self, container: str):
        """Return exact ownership labels and configured task state."""
        assert container == "loop-job"
        result = self._result("inspect")
        payload = (
            self.inspect_payloads.pop(0)
            if self.inspect_payloads
            else (
                {"Config": {"Labels": self.labels}, "State": self.state}
                if self.inspect_payload is None
                else self.inspect_payload
            )
        )
        stdout = (
            self.inspect_payload
            if isinstance(self.inspect_payload, bytes)
            else json.dumps(payload).encode()
        )
        return result.__class__(
            result.exit_code,
            stdout,
            result.stderr,
            False,
            False,
        )

    def attach(self, container: str, **kwargs: object):
        """Return the sole fake attachment."""
        assert container == "loop-job"
        self.calls.append(f"attach:{kwargs['pty']}")
        return self.session

    def signal_container(self, container: str, signal: OciSignal):
        """Record one reviewed signal."""
        assert container == "loop-job"
        return self._result(f"signal:{signal.value}")

    def pause_container(self, container: str):
        """Pause the complete fake task boundary."""
        assert container == "loop-job"
        result = self._result("pause")
        if not result.exit_code:
            self.state = {"Running": False, "Paused": True, "Status": "paused"}
        return result

    def unpause_container(self, container: str):
        """Resume the complete fake task boundary."""
        assert container == "loop-job"
        result = self._result("unpause")
        if not result.exit_code:
            self.state = {"Running": True, "Paused": False, "Status": "running"}
        return result

    def stop_container(self, container: str):
        """Stop the task with terminal evidence."""
        assert container == "loop-job"
        result = self._result("stop")
        if not result.exit_code:
            self.state = {
                "Running": False,
                "Paused": False,
                "Status": "exited",
                "ExitCode": 143,
            }
        return result

    def kill_container(self, container: str):
        """Force the fake task to terminal state."""
        assert container == "loop-job"
        self.state = {"Running": False, "Paused": False, "Status": "exited", "ExitCode": 137}
        return self._result("kill")

    def wait_container(self, container: str):
        """Record task reaping."""
        assert container == "loop-job"
        return self._result("wait")

    def remove_container(self, container: str):
        """Record container cleanup."""
        assert container == "loop-job"
        return self._result("remove")


class _DurableWorkspace:
    """Hold a fake leased generation until explicit release and discard."""

    def __init__(self) -> None:
        self.leased = MacosLeasedWorkspace(
            context=SimpleNamespace(),  # type: ignore[arg-type]
            layer=MacosAttemptLayer(
                "job",
                "branch",
                "/home/loop/.local/share/loop/workspaces/attempts/job/merged",
                "/home/loop/.local/share/loop/workspaces/attempts/job",
                "/home/loop/.local/share/loop/workspaces/attempts/job/upper.tar",
                Path("/private/job.tar"),
            ),
            generation_reference="generation-" + "a" * 32,
        )
        self.released = False
        self.discarded = False

    def release(self) -> None:
        """Record logical generation release."""
        self.released = True

    def discard(self) -> None:
        """Record attempt overlay disposal."""
        self.released = True
        self.discarded = True


class _Workspace:
    """Record durable generation, observation, and publication operations."""

    def __init__(self) -> None:
        self.active = _DurableWorkspace()
        self.delta = CanonicalDelta()
        self.observations = 0
        self.publications = 0
        self.denials = 0
        self.deny_error = False
        self.resumptions = 0
        self.materializer = SimpleNamespace(discard_attempt=lambda layer: None)

    def lease_job(self, workspace: str, agent: str, attempt: str):
        """Return a fresh durable lease for exact identities."""
        assert (workspace, agent, attempt) == ("workspace", "agent", "job")
        return self.active

    def resume_job(self, workspace: str, agent: str, generation: str, layer: MacosAttemptLayer):
        """Reacquire the persisted generation after manager restart."""
        assert (workspace, agent, generation) == (
            "workspace",
            "agent",
            "generation-" + "a" * 32,
        )
        assert layer.attempt_id == "job"
        self.resumptions += 1
        self.active = _DurableWorkspace()
        return self.active

    def observe(self, leased: MacosLeasedWorkspace) -> CanonicalDelta:
        """Return the configured terminal-only delta."""
        assert leased.layer.attempt_id == "job"
        self.observations += 1
        return self.delta

    def publish(self, leased: MacosLeasedWorkspace, transaction: str, delta: CanonicalDelta):
        """Record one terminal publication."""
        assert leased.layer.attempt_id == transaction == "job" and delta is self.delta
        self.publications += 1
        return object()

    def deny(self, leased: MacosLeasedWorkspace) -> None:
        """Record staged-content denial."""
        assert leased.layer.attempt_id == "job"
        if self.deny_error:
            raise RuntimeError("deny")
        self.denials += 1


class _Prepared:
    """Expose one fake attested runtime and OCI control plane."""

    def __init__(self, control: _Control) -> None:
        self.control_plane = control
        self.endpoint = SimpleNamespace(
            namespace="loop-private",
            platform=SimpleNamespace(os="linux", architecture="arm64"),
            socket_path="/proc/123/root/run/containerd/containerd.sock",
            network_namespace_identity="4:4026533000",
        )
        self.closed = False
        self.renewals: list[float] = []

    def close(self) -> None:
        """Record release of this process's runtime lease."""
        self.closed = True

    def renew(self, lifetime_seconds: float) -> None:
        """Record one operation-bounded runtime lease renewal."""
        self.renewals.append(lifetime_seconds)

    def image_for_digest(self, runtime_digest: str) -> RuntimeImage:
        """Return the fake locally built image only for the authorized digest."""
        assert runtime_digest == _request().lease.runtime_digest
        return RuntimeImage(
            reference="loop.local/sandbox@sha256:" + "a" * 64,
            platform=PlatformSelector(os="linux", architecture="arm64"),
            identity=OciImageIdentity(
                index_digest="sha256:" + "a" * 64,
                manifest_digest=runtime_digest,
                config_digest="sha256:" + "c" * 64,
            ),
        )


class _Backend:
    """Return one configured fake prepared runtime."""

    def __init__(self, prepared: _Prepared) -> None:
        self.prepared = prepared

    def prepare(self) -> _Prepared:
        """Return the attested fake runtime."""
        return self.prepared


def _manager(
    tmp_path: Path,
    request: ShellExecutionRequest | None = None,
    *,
    operation_allowed: bool = True,
    publication_allowed: bool = True,
):
    """Build a durable manager with isolated fake runtime and workspace owners."""
    control = _Control()
    workspace = _Workspace()
    prepared = _Prepared(control)
    manager = MacosDurableJobManager(
        _Backend(prepared),  # type: ignore[arg-type]
        _candidate(),
        workspace,  # type: ignore[arg-type]
        OciResourceLimits(1024, 32, 10000),
        lambda lease, operation, job_id: operation_allowed and job_id == "job",
        lambda execution_request, delta: publication_allowed,
        DurableJobStore(tmp_path / "jobs"),
        prepared_runtime=prepared,  # type: ignore[arg-type]
    )
    return manager, workspace, control, prepared, request or _request()


def _start(manager: MacosDurableJobManager, request: ShellExecutionRequest) -> JobHandle:
    """Start a job through a correctly initialized service observer."""
    attempt = AttemptStateMachine()
    attempt.transition("authorized")
    return manager.start(request, AttemptObserver(attempt))


def test_durable_job_controls_reauthorize_and_cleanup_without_early_delta(tmp_path: Path) -> None:
    """Start, attach, control, cancel, and cleanup never inspect effects while running."""
    request = _request(terminal=TerminalMode.PTY)
    manager, workspace, control, prepared, _ = _manager(tmp_path, request)
    handle = _start(manager, request)
    assert manager.has_live_jobs() is True
    assert prepared.renewals == [90.0]
    assert control.session.writes == [] and control.session.closed
    control.session = _Session()
    lease = _operation_lease(request)

    assert manager.status(handle, lease).state is DurableJobState.RUNNING
    assert workspace.observations == 0
    manager.attach(handle, lease)
    with pytest.raises(RuntimeError, match="active attachment"):
        manager.attach(handle, lease)
    manager.write(handle, lease, b"input")
    manager.close_stdin(handle, lease)
    assert manager.read(handle, lease, 0) is None
    manager.resize(handle, lease, 100, 40)
    manager.signal(handle, lease, OciSignal.INTERRUPT)
    assert manager.suspend(handle, lease).state is DurableJobState.PAUSED
    assert manager.resume(handle, lease).state is DurableJobState.RUNNING
    manager.detach(handle, lease)
    assert control.session.writes == [b"input"] and control.session.stdin_closed
    assert control.session.sizes == [(100, 40)] and control.session.closed
    assert manager.cancel(handle, lease).state is DurableJobState.CANCELLED
    assert manager.has_live_jobs() is False
    assert workspace.active.discarded and workspace.observations == 0
    assert "remove" in control.calls


def test_terminal_job_publishes_only_after_exit_and_survives_manager_restart(
    tmp_path: Path,
) -> None:
    """Restart reattests exact identities before a terminal write is observed and published."""
    request = _request(write=True)
    manager, workspace, control, prepared, _ = _manager(tmp_path, request)
    handle = _start(manager, request)
    manager.attach(handle, _operation_lease(request))
    manager.close()
    assert not prepared.closed
    manager.close()

    workspace.delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="created",
                object_kind=ObjectKind.DIRECTORY,
            ),
        )
    )
    prepared.closed = False
    restarted = MacosDurableJobManager(
        _Backend(prepared),  # type: ignore[arg-type]
        _candidate(),
        workspace,  # type: ignore[arg-type]
        OciResourceLimits(1024, 32, 10000),
        lambda lease, operation, job_id: True,
        lambda execution_request, delta: True,
        DurableJobStore(tmp_path / "jobs"),
        prepared_runtime=prepared,  # type: ignore[arg-type]
    )
    assert restarted.recover()[0].state is DurableJobState.RUNNING
    assert workspace.resumptions == 1 and workspace.observations == 0
    control.state = {"Running": False, "Paused": False, "Status": "exited", "ExitCode": 9}
    status = restarted.status(handle, _operation_lease(request))
    assert (status.state, status.exit_code) == (DurableJobState.COMPLETED, 9)
    assert workspace.observations == workspace.publications == 1
    assert workspace.active.released and workspace.active.discarded


def test_forged_expired_and_ambiguous_jobs_fail_closed_without_publication(tmp_path: Path) -> None:
    """Bad authority and ambiguous recovery cannot expose or commit a durable job delta."""
    manager, workspace, control, prepared, request = _manager(tmp_path)
    handle = _start(manager, request)
    with pytest.raises(PermissionError, match="handle"):
        manager.status(handle.model_copy(update={"token": "0" * 64}), _operation_lease(request))
    with pytest.raises(PermissionError, match="authorized"):
        manager.status(handle, _operation_lease(request, expires_at_ns=0))
    manager.close()

    control.labels["io.loop.epoch"] = "stale"
    restarted = MacosDurableJobManager(
        _Backend(prepared),  # type: ignore[arg-type]
        _candidate(),
        workspace,  # type: ignore[arg-type]
        OciResourceLimits(1024, 32, 10000),
        lambda lease, operation, job_id: True,
        lambda execution_request, delta: True,
        DurableJobStore(tmp_path / "jobs"),
        prepared_runtime=prepared,  # type: ignore[arg-type]
    )
    assert restarted.recover()[0].state is DurableJobState.LOST
    assert workspace.observations == workspace.publications == 0
    assert "kill" in control.calls and "remove" in control.calls


def test_job_store_and_start_failures_reject_corruption_and_cleanup(tmp_path: Path) -> None:
    """Malformed state, invalid requests, and partial starts leave no runnable job authority."""
    store = DurableJobStore(tmp_path / "store")
    with pytest.raises(KeyError):
        store.load("missing")
    with pytest.raises(ValueError, match="identity"):
        store.load("../escape")

    manager, workspace, control, _, request = _manager(tmp_path / "manager")
    with pytest.raises(ValueError, match="explicit durable"):
        _start(manager, request.model_copy(update={"mode": ExecutionMode.FOREGROUND}))
    with pytest.raises(ValueError, match="PTY mode"):
        _start(
            manager,
            request.model_copy(
                update={
                    "terminal": TerminalMode.PIPE,
                    "terminal_columns": None,
                    "terminal_rows": None,
                }
            ),
        )
    control.fail = "run-job"
    with pytest.raises(RuntimeError, match="run failed"):
        _start(manager, request)
    assert workspace.active.discarded and "remove" in control.calls
    assert not tuple((tmp_path / "manager/jobs").glob("*.json"))

    for failure, message in (
        ("run-job", "run failed"),
        ("start-state", "runtime state"),
    ):
        failed, failed_workspace, failed_control, _, failed_request = _manager(tmp_path / failure)
        failed_control.fail = failure
        with pytest.raises(RuntimeError, match=message):
            _start(failed, failed_request)
        assert failed_workspace.active.discarded


def test_store_detects_symlinks_checksums_and_record_identity_changes(tmp_path: Path) -> None:
    """Private state rejects aliased roots and every checksum or identity mutation."""
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="private directory"):
        DurableJobStore(alias)

    manager, _, _, _, request = _manager(tmp_path / "jobs")
    handle = _start(manager, request)
    manager.close()
    store = DurableJobStore(tmp_path / "jobs/jobs")
    path = next(store.directory.glob("*.json"))
    envelope = json.loads(path.read_bytes())
    envelope["checksum"] = "0" * 64
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid"):
        store.load(handle.job_id)

    envelope["payload"]["job_id"] = "other"
    payload = json_encode(envelope["payload"]).encode()
    envelope["checksum"] = sha256_digest(payload)
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity"):
        store.load(handle.job_id)


def test_start_rejects_brokers_duplicates_and_invalid_container_evidence(tmp_path: Path) -> None:
    """Unsupported brokers, duplicate identities, and failed attestation clean up before return."""
    manager, _, _, _, request = _manager(tmp_path / "duplicate")
    _start(manager, request)
    with pytest.raises(ValueError, match="already exists"):
        _start(manager, request)
    manager.close()

    connection = NetworkConnectionLease(
        hostname="example.com",
        port=443,
        protocol=NetworkProtocol.HTTPS,
        addresses=("93.184.216.34",),
    )
    brokered = request.model_copy(
        update={
            "network_connections": (connection,),
            "lease": request.lease.model_copy(
                update={
                    "capabilities": request.lease.capabilities
                    | frozenset({Capability.NETWORK_CONNECT})
                }
            ),
        }
    )
    isolated, _, _, _, _ = _manager(tmp_path / "broker")
    with pytest.raises(ValueError, match="broker leases"):
        _start(isolated, brokered)

    failed, failed_workspace, failed_control, _, failed_request = _manager(tmp_path / "evidence")
    failed_control.fail = "start-state"
    with pytest.raises(RuntimeError, match="runtime state"):
        _start(failed, failed_request)
    assert failed_workspace.active.discarded


@pytest.mark.parametrize(
    "changes",
    (
        {"workspace_id": "other"},
        {"agent_run_id": "other"},
        {"runtime_digest": "sha256:other"},
        {"capabilities": frozenset({Capability.PROCESS_SIGNAL})},
    ),
)
def test_status_reauthorizes_every_bound_lease_identity(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    """Status rejects each independently changed operation-lease binding."""
    manager, _, _, _, request = _manager(tmp_path)
    handle = _start(manager, request)
    with pytest.raises(PermissionError, match="authorized"):
        manager.status(handle, _operation_lease(request, **changes))


def test_operation_denial_missing_attachment_and_terminal_tombstone_are_closed(
    tmp_path: Path,
) -> None:
    """Denied operations and terminal handles cannot create implicit attachment state."""
    manager, workspace, control, prepared, request = _manager(tmp_path)
    handle = _start(manager, request)
    lease = _operation_lease(request)
    with pytest.raises(RuntimeError, match="no active attachment"):
        manager.write(handle, lease, b"data")
    manager.detach(handle, lease)

    denied, _, _, _, _ = _manager(tmp_path, operation_allowed=False)
    with pytest.raises(PermissionError, match="authorized"):
        denied.signal(handle, lease, OciSignal.HANGUP)
    control.fail = "signal:TERM"
    with pytest.raises(RuntimeError, match="signal failed"):
        manager.signal(handle, lease, OciSignal.TERMINATE)
    control.fail = None
    assert manager.cancel(handle, lease).state is DurableJobState.CANCELLED
    restarted = MacosDurableJobManager(
        _Backend(prepared),  # type: ignore[arg-type]
        _candidate(),
        workspace,  # type: ignore[arg-type]
        OciResourceLimits(1024, 32, 10000),
        lambda lease, operation, job_id: True,
        lambda execution_request, delta: True,
        DurableJobStore(tmp_path / "jobs"),
        prepared_runtime=prepared,  # type: ignore[arg-type]
    )
    assert restarted.recover()[0].state is DurableJobState.CANCELLED
    assert restarted.status(handle, lease).state is DurableJobState.CANCELLED
    with pytest.raises(RuntimeError, match="not attachable"):
        restarted.attach(handle, lease)


def test_cancel_fallback_denial_and_terminal_failures_remain_closed(tmp_path: Path) -> None:
    """Forced cancellation, denied deltas, observation errors, and cleanup errors are terminal."""
    request = _request(write=True)
    manager, workspace, control, _, _ = _manager(
        tmp_path / "denied", request, publication_allowed=False
    )
    handle = _start(manager, request)
    workspace.delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="created",
                object_kind=ObjectKind.DIRECTORY,
            ),
        )
    )
    control.fail = "stop"
    assert manager.cancel(handle, _operation_lease(request)).state is DurableJobState.DENIED
    assert workspace.denials == 1 and "kill" in control.calls

    failed, failed_workspace, failed_control, _, failed_request = _manager(
        tmp_path / "failed", request
    )
    failed_handle = _start(failed, failed_request)
    failed_workspace.observe = lambda leased: (_ for _ in ()).throw(RuntimeError("inspect"))
    failed_control.state = {
        "Running": False,
        "Paused": False,
        "Status": "exited",
        "ExitCode": 1,
    }
    assert (
        failed.status(failed_handle, _operation_lease(failed_request)).state
        is DurableJobState.FAILED
    )

    cleanup, _, cleanup_control, _, cleanup_request = _manager(tmp_path / "cleanup")
    cleanup_handle = _start(cleanup, cleanup_request)
    cleanup_control.fail = "remove"
    assert (
        cleanup.cancel(cleanup_handle, _operation_lease(cleanup_request)).state
        is DurableJobState.FAILED
    )


def test_empty_write_terminal_attachment_and_denial_cleanup_are_closed(tmp_path: Path) -> None:
    """Empty writes reclaim staging and terminal attachments close even when denial cleanup fails."""
    request = _request(write=True)
    manager, workspace, control, _, _ = _manager(tmp_path / "empty", request)
    handle = _start(manager, request)
    manager.attach(handle, _operation_lease(request))
    control.state = {
        "Running": False,
        "Paused": False,
        "Status": "exited",
        "ExitCode": 0,
    }
    assert manager.status(handle, _operation_lease(request)).state is DurableJobState.COMPLETED
    assert workspace.denials == 1 and control.session.closed

    failed, failed_workspace, failed_control, _, failed_request = _manager(
        tmp_path / "deny-error", request
    )
    failed_handle = _start(failed, failed_request)
    failed_workspace.observe = lambda leased: (_ for _ in ()).throw(RuntimeError("inspect"))
    failed_workspace.deny_error = True
    failed_control.state = {
        "Running": False,
        "Paused": False,
        "Status": "exited",
        "ExitCode": 1,
    }
    assert (
        failed.status(failed_handle, _operation_lease(failed_request)).state
        is DurableJobState.FAILED
    )


@pytest.mark.parametrize(
    "payload",
    (
        b"not-json",
        [],
        {"Config": {"Labels": {}}, "State": {}},
        {"Config": {"Labels": "labels"}, "State": []},
        {"Config": {"Labels": "labels"}, "State": {"Status": "unknown"}},
    ),
)
def test_malformed_or_ambiguous_inspection_marks_recovery_lost(
    tmp_path: Path, payload: object
) -> None:
    """Malformed, mismatched, or ambiguous runtime evidence never reconnects a job."""
    manager, workspace, control, prepared, request = _manager(tmp_path)
    _start(manager, request)
    manager.close()
    control.inspect_payload = payload
    restarted = MacosDurableJobManager(
        _Backend(prepared),  # type: ignore[arg-type]
        _candidate(),
        workspace,  # type: ignore[arg-type]
        OciResourceLimits(1024, 32, 10000),
        lambda lease, operation, job_id: True,
        lambda execution_request, delta: True,
        DurableJobStore(tmp_path / "jobs"),
        prepared_runtime=prepared,  # type: ignore[arg-type]
    )
    assert restarted.recover()[0].state is DurableJobState.LOST


def test_recovery_without_workspace_lease_discards_private_layer_best_effort(
    tmp_path: Path,
) -> None:
    """A stale generation that cannot be reacquired is lost without inspecting its delta."""
    manager, workspace, control, prepared, request = _manager(tmp_path)
    _start(manager, request)
    manager.close()
    discarded: list[str] = []
    workspace.resume_job = lambda *args: (_ for _ in ()).throw(RuntimeError("stale"))
    workspace.materializer = SimpleNamespace(
        discard_attempt=lambda layer: discarded.append(layer.attempt_id)
    )
    restarted = MacosDurableJobManager(
        _Backend(prepared),  # type: ignore[arg-type]
        _candidate(),
        workspace,  # type: ignore[arg-type]
        OciResourceLimits(1024, 32, 10000),
        lambda lease, operation, job_id: True,
        lambda execution_request, delta: True,
        DurableJobStore(tmp_path / "jobs"),
        prepared_runtime=prepared,  # type: ignore[arg-type]
    )
    assert restarted.recover()[0].state is DurableJobState.LOST
    assert discarded == ["job"] and "kill" in control.calls

    # Cleanup failure cannot turn ambiguous state into publication authority.
    manager, workspace, _, prepared, request = _manager(tmp_path / "cleanup")
    _start(manager, request)
    manager.close()
    workspace.resume_job = lambda *args: (_ for _ in ()).throw(RuntimeError("stale"))
    workspace.materializer = SimpleNamespace(
        discard_attempt=lambda layer: (_ for _ in ()).throw(RuntimeError("cleanup"))
    )
    restarted = MacosDurableJobManager(
        _Backend(prepared),  # type: ignore[arg-type]
        _candidate(),
        workspace,  # type: ignore[arg-type]
        OciResourceLimits(1024, 32, 10000),
        lambda lease, operation, job_id: True,
        lambda execution_request, delta: True,
        DurableJobStore(tmp_path / "cleanup/jobs"),
        prepared_runtime=prepared,  # type: ignore[arg-type]
    )
    assert restarted.recover()[0].state is DurableJobState.LOST


def test_runtime_restart_epoch_marks_job_lost_without_delta_inspection(tmp_path: Path) -> None:
    """A changed rootless runtime process identity makes restart recovery explicitly lost."""
    manager, workspace, _, prepared, request = _manager(tmp_path)
    _start(manager, request)
    manager.close()
    prepared.endpoint.socket_path = "/proc/999/root/run/containerd/containerd.sock"
    restarted = MacosDurableJobManager(
        _Backend(prepared),  # type: ignore[arg-type]
        _candidate(),
        workspace,  # type: ignore[arg-type]
        OciResourceLimits(1024, 32, 10000),
        lambda lease, operation, job_id: True,
        lambda execution_request, delta: True,
        DurableJobStore(tmp_path / "jobs"),
        prepared_runtime=prepared,  # type: ignore[arg-type]
    )
    assert restarted.recover()[0].state is DurableJobState.LOST
    assert workspace.observations == workspace.publications == 0


def test_upstream_labels_are_allowed_but_unexpected_loop_labels_are_rejected(
    tmp_path: Path,
) -> None:
    """Runtime metadata may coexist with the exact closed set of Loop ownership labels."""
    manager, _, control, _, request = _manager(tmp_path / "upstream")
    handle = _start(manager, request)
    control.labels["nerdctl/extra"] = "runtime-owned"
    assert manager.status(handle, _operation_lease(request)).state is DurableJobState.RUNNING

    control.labels["io.loop.unexpected"] = "forged"
    with pytest.raises(RuntimeError, match="ownership"):
        manager.status(handle, _operation_lease(request))


def test_manager_prepares_runtime_lazily_when_no_warm_lease_is_supplied(tmp_path: Path) -> None:
    """A durable start obtains the managed runtime only through its injected backend."""
    control = _Control()
    workspace = _Workspace()
    prepared = _Prepared(control)
    manager = MacosDurableJobManager(
        _Backend(prepared),  # type: ignore[arg-type]
        _candidate(),
        workspace,  # type: ignore[arg-type]
        OciResourceLimits(1024, 32, 10000),
        lambda lease, operation, job_id: True,
        lambda execution_request, delta: True,
        DurableJobStore(tmp_path / "jobs"),
    )
    assert _start(manager, _request()).job_id == "job"
    manager.close()
    assert prepared.closed


@pytest.mark.parametrize(
    ("state", "error"),
    (([], TypeError), ({"Running": False, "Status": "unknown"}, RuntimeError)),
)
def test_runtime_state_race_after_reattest_fails_closed(
    tmp_path: Path, state: object, error: type[Exception]
) -> None:
    """State that becomes malformed or ambiguous after reattestation cannot finalize a job."""
    manager, _, control, _, request = _manager(tmp_path)
    handle = _start(manager, request)
    running = {
        "Config": {"Labels": control.labels},
        "State": {"Running": True, "Paused": False, "Status": "running"},
    }
    control.inspect_payloads = [
        running,
        {"Config": {"Labels": control.labels}, "State": state},
    ]
    with pytest.raises(error, match="state"):
        manager.status(handle, _operation_lease(request))
