"""Test the managed macOS foreground execution adapter."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.execution.contracts import (
    Capability,
    DeltaEffect,
    ExecutionLease,
    ExecutionMode,
    NetworkConnectionLease,
    NetworkProtocol,
    SecretExposure,
    SecretMechanism,
    ShellExecutionRequest,
    TerminalMode,
)
from loop.execution.infrastructure import InfrastructureProcessResult
from loop.execution.results import (
    Cancelled,
    CapabilityDenied,
    CommitConflict,
    Completed,
    InfrastructureFailure,
    TimedOut,
    UnrepresentableDelta,
    UnsupportedCapability,
)
from loop.execution.runtime.image import RuntimeImage
from loop.execution.runtime.models import OciImageIdentity, PlatformSelector
from loop.execution.sandbox.macos import (
    MacosAttemptLayer,
    MacosExecutionAdapter,
    MacosLeasedWorkspace,
    load_macos_runtime_candidate,
)
from loop.execution.sandbox.macos.candidate import macos_artifact
from loop.execution.sandbox.oci.process import (
    OciSessionCancelled,
    OciSessionTimedOut,
    OciStreamFrame,
)
from loop.execution.sandbox.oci.spec import OciResourceLimits
from loop.execution.service import ExecutionService
from loop.execution.vfs import (
    CanonicalDelta,
    CanonicalDeltaEntry,
    CommitConflictError,
    ObjectKind,
    UnrepresentableDeltaError,
)


def _candidate():
    """Load the repository-controlled runtime candidate."""
    return load_macos_runtime_candidate(
        Path(__file__).parents[4] / "scripts/runtime-candidates/macos-arm64-v1.json"
    )


def _request(**changes: object) -> ShellExecutionRequest:
    """Build one read-only foreground shell request."""
    values = {
        "request_id": "request",
        "lease": ExecutionLease(
            lease_id="lease",
            workspace_id="workspace",
            agent_run_id="agent",
            policy_version="policy",
            runtime_digest=(
                "sha256:071fa5de01a240dbef5be09d69f8fef2f89d68445d9175393773ee389b6f5935"
            ),
            expires_at_ns=1,
            capabilities=frozenset({Capability.WORKSPACE_READ, Capability.PROCESS_SPAWN}),
        ),
        "script": "printf safe",
        "output_limit_bytes": 4,
    }
    values.update(changes)
    return ShellExecutionRequest(**values)


class _Workspace:
    """Lease one configured fresh attempt and record observed publication decisions."""

    def __init__(self) -> None:
        self.leases: list[tuple[str, str, str]] = []
        self.delta = CanonicalDelta()
        self.published: list[CanonicalDelta] = []
        self.denied = 0

    @contextmanager
    def lease(self, workspace_id: str, agent_run_id: str, attempt_id: str):
        """Record and yield one platform-private fresh attempt."""
        self.leases.append((workspace_id, agent_run_id, attempt_id))
        yield MacosLeasedWorkspace(
            context=SimpleNamespace(),  # type: ignore[arg-type]
            layer=MacosAttemptLayer(
                attempt_id,
                "branch",
                f"/home/loop/.local/share/loop/workspaces/attempts/{attempt_id}/merged",
                f"/home/loop/.local/share/loop/workspaces/attempts/{attempt_id}",
                f"/home/loop/.local/share/loop/workspaces/attempts/{attempt_id}/upper.tar",
                Path(f"/{attempt_id}.tar"),
            ),
            generation_reference="generation",
        )

    def observe(self, leased: MacosLeasedWorkspace) -> CanonicalDelta:
        """Return the configured complete observed delta."""
        del leased
        return self.delta

    def publish(
        self,
        leased: MacosLeasedWorkspace,
        transaction_id: str,
        delta: CanonicalDelta,
    ) -> object:
        """Record one approved publication."""
        del leased, transaction_id
        self.published.append(delta)
        return object()

    def deny(self, leased: MacosLeasedWorkspace) -> None:
        """Record staged-effect reclamation."""
        del leased
        self.denied += 1


class _Session:
    """Expose deterministic attached frames and terminal behavior."""

    def __init__(self, terminal: str = "completed") -> None:
        self.frames = [
            OciStreamFrame(0, "stdout", b"output"),
            OciStreamFrame(1, "stderr", b"err"),
        ]
        self.terminal = terminal
        self.stdin_closed = False
        self.closed = False
        self.resizes: list[tuple[int, int]] = []
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        """Record bounded input delivered before EOF."""
        self.writes.append(data)

    def close_stdin(self) -> None:
        """Record stdin closure before waiting."""
        if self.terminal == "close_error_late":
            time.sleep(0.05)
            raise RuntimeError("late close failed")
        if self.terminal == "close_error":
            raise RuntimeError("close failed")
        self.stdin_closed = True

    def resize(self, columns: int, rows: int) -> None:
        """Record an initial PTY size."""
        self.resizes.append((columns, rows))

    def read(self, timeout: float | None = None):
        """Return one queued frame without blocking."""
        del timeout
        if self.terminal == "read_error":
            raise OSError("read failed")
        if self.terminal == "slow_reader":
            time.sleep(3)
            return None
        if self.terminal == "late_read_error":
            time.sleep(0.05)
            raise OSError("closed after completion")
        return self.frames.pop(0) if self.frames else None

    def wait(self, cancellation=lambda: False) -> int:
        """Return or raise the configured terminal outcome."""
        del cancellation
        if self.terminal == "timeout":
            raise OciSessionTimedOut("timeout")
        if self.terminal == "cancelled":
            raise OciSessionCancelled("cancelled")
        if self.terminal == "read_error":
            time.sleep(0.01)
        if self.terminal == "close_error":
            time.sleep(0.01)
        return 7

    def close(self) -> None:
        """Record attached transport cleanup."""
        self.closed = True


class _ControlPlane:
    """Record fixed attempt lifecycle calls at the platform boundary."""

    def __init__(self, session: _Session) -> None:
        self.session = session
        self.spec = None
        self.workspace_source = None
        self.state = {"Status": "exited", "Running": False, "Pid": 0, "ExitCode": 7}
        self.remove_exit_code = 0
        self.run_error = False
        self.inspect_result = None
        self.remove_error = False
        self.terminated: list[str] = []
        self.kill_exit_code = 0
        self.wait_exit_code = 0
        self.bindings = None
        self.create_result = InfrastructureProcessResult(0, b"created", b"", False, False)
        self.hosts_result = InfrastructureProcessResult(0, b"444\n", b"", False, False)
        self.hosts_hardened = False
        self.hosts_released = False
        self.release_hosts_result = InfrastructureProcessResult(0, b"600\n", b"", False, False)
        self.release_hosts_error = False
        self.starting = False
        self.gate_opened = False
        self.gate_result = InfrastructureProcessResult(0, b"ok", b"", False, False)

    def run_attempt(self, spec, workspace_source, **bounds):
        """Record the compiled attempt and return its attached session."""
        self.spec = spec
        self.workspace_source = workspace_source
        self.bindings = bounds.pop("bindings", None)
        assert bounds == {"deadline_seconds": 30.0}
        if self.run_error:
            raise RuntimeError("run failed")
        return self.session

    def create_attempt(self, spec, workspace_source, **bounds):
        """Record a pre-start networked attempt."""
        self.spec = spec
        self.workspace_source = workspace_source
        self.bindings = bounds.pop("bindings", None)
        assert bounds == {"deadline_seconds": 30.0}
        return self.create_result

    def start_attempt(self, spec, **bounds):
        """Return the attached session for an already-created attempt."""
        self.spec = spec
        assert bounds == {"deadline_seconds": 30.0}
        self.starting = bool(getattr(self.bindings, "host_aliases", ()))
        return self.session

    def harden_container_hosts(self, container_name: str):
        """Record pre-start immutability enforcement for generated hosts state."""
        assert container_name == "loop-request"
        self.hosts_hardened = True
        return self.hosts_result

    def release_container_hosts(self, container_name: str):
        """Record trusted restoration needed before nerdctl removes the container."""
        assert container_name == "loop-request"
        self.hosts_released = True
        if self.release_hosts_error:
            raise RuntimeError("release failed")
        return self.release_hosts_result

    def open_container_start_gate(self, path: str):
        """Record trusted release of the fixed command wrapper."""
        assert path.endswith("/start-gate")
        self.gate_opened = True
        return self.gate_result

    def inspect_container_state(self, container_name: str):
        """Return structured terminal-state evidence."""
        assert container_name == "loop-request"
        if self.inspect_result is not None:
            return self.inspect_result
        if self.starting:
            self.starting = False
            return InfrastructureProcessResult(
                0,
                b'{"Status":"running","Running":true,"Pid":2,"ExitCode":0}',
                b"",
                False,
                False,
            )
        return InfrastructureProcessResult(0, json.dumps(self.state).encode(), b"", False, False)

    def remove_container(self, container_name: str):
        """Return the configured forced-removal outcome."""
        assert container_name == "loop-request"
        if self.remove_error:
            raise RuntimeError("remove failed")
        return InfrastructureProcessResult(self.remove_exit_code, b"", b"", False, False)

    def kill_container(self, container_name: str):
        """Record one forced task termination."""
        assert container_name == "loop-request"
        self.terminated.append("kill")
        return InfrastructureProcessResult(self.kill_exit_code, b"", b"", False, False)

    def wait_container(self, container_name: str):
        """Record one terminal task reap."""
        assert container_name == "loop-request"
        self.terminated.append("wait")
        return InfrastructureProcessResult(self.wait_exit_code, b"", b"", False, False)


class _Prepared:
    """Expose one fake attested runtime lease."""

    def __init__(self, control_plane: _ControlPlane) -> None:
        self.control_plane = control_plane
        self.endpoint = SimpleNamespace(
            namespace="loop-private",
            platform=SimpleNamespace(os="linux", architecture="arm64"),
            owner_uid=1000,
            state_path="/home/loop/.local/share/containerd",
        )
        self.snapshot_store = Path("/private/tmp/loop-test-snapshots")
        self.closed = False
        self.renewals: list[float] = []

    def close(self) -> None:
        """Record runtime-lease cleanup."""
        self.closed = True

    def renew(self, lifetime_seconds: float) -> None:
        """Record one attempt-bounded runtime lease renewal."""
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
    """Return one configured prepared runtime."""

    def __init__(self, snapshot_store: Path, prepared: _Prepared) -> None:
        self.snapshot_store = snapshot_store
        self.prepared = prepared
        self.calls = 0

    def prepare(self) -> _Prepared:
        """Record runtime preparation."""
        self.calls += 1
        return self.prepared


def _service(
    tmp_path: Path,
    terminal: str = "completed",
    *,
    prepared_runtime: bool = False,
    authorize_publication: bool = False,
    secret_authority=None,
):
    """Build one service with a fresh attempt and fake platform transport."""
    store = tmp_path / "snapshots"
    store.mkdir(parents=True)
    session = _Session(terminal)
    control = _ControlPlane(session)
    prepared = _Prepared(control)
    backend = _Backend(store, prepared)
    workspace = _Workspace()
    adapter = MacosExecutionAdapter(
        backend,  # type: ignore[arg-type]
        _candidate(),
        workspace,
        OciResourceLimits(1024, 32, 10000),
        lambda request, delta: authorize_publication,
        prepared_runtime=prepared if prepared_runtime else None,  # type: ignore[arg-type]
        secret_authority=secret_authority,
    )
    return ExecutionService(adapter), adapter, backend, workspace, control, prepared, session


def test_read_only_foreground_attempt_returns_bounded_streams_and_cleans_up(tmp_path: Path):
    """A completed nonzero shell outcome retains bounded streams and closes every lease."""
    service, adapter, backend, workspace, control, prepared, session = _service(tmp_path)

    result = service.execute(_request(stdin=b"input\n"))

    assert isinstance(result, Completed)
    assert result.exit_code == 7
    assert result.stdout == b"outp"
    assert result.stdout_truncated
    assert result.stderr == b"err"
    assert not result.stderr_truncated
    assert session.stdin_closed and session.closed and not prepared.closed
    assert session.writes == [b"input\n"]
    assert prepared.renewals == [90.0]
    assert workspace.leases == [("workspace", "agent", "request")]
    assert backend.calls == 1
    assert control.workspace_source.endswith("/attempts/request/merged")
    assert control.spec.argv == ("/bin/sh", "-c", "printf safe")
    assert control.spec.workspace_mount.read_only
    adapter.close()
    assert prepared.closed
    adapter.close()

    service, _, _, _, _, _, _ = _service(tmp_path / "short")
    short = service.execute(_request(output_limit_bytes=2))
    assert short.stdout_truncated and short.stderr_truncated


def test_pty_attempt_resizes_and_returns_one_merged_stdout_stream(tmp_path: Path) -> None:
    """PTY mode applies its initial size and maps merged bytes to public stdout."""
    service, _, _, _, _, _, session = _service(tmp_path)
    session.frames = [OciStreamFrame(0, "pty", b"terminal-output")]

    result = service.execute(
        _request(
            terminal=TerminalMode.PTY,
            terminal_columns=100,
            terminal_rows=40,
            output_limit_bytes=32,
        )
    )

    assert isinstance(result, Completed)
    assert result.stdout == b"terminal-output"
    assert result.stderr == b""
    assert session.resizes == [(100, 40), (100, 40)]
    assert session.stdin_closed


@pytest.mark.parametrize(
    ("terminal", "result_type"),
    [("timeout", TimedOut), ("cancelled", Cancelled)],
)
def test_terminal_interruptions_remain_typed_and_cleanup(
    tmp_path: Path, terminal: str, result_type: type
) -> None:
    """Timeout and cancellation stay sandbox outcomes and force container removal."""
    service, adapter, _, workspace, control, prepared, session = _service(
        tmp_path, terminal, authorize_publication=True
    )
    workspace.delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="partial",
                object_kind=ObjectKind.DIRECTORY,
            ),
        )
    )
    request = _request().model_copy(
        update={
            "lease": _request().lease.model_copy(
                update={"capabilities": frozenset({Capability.WORKSPACE_WRITE})}
            )
        }
    )

    result = service.execute(request)

    assert isinstance(result, result_type)
    assert control.terminated == ["kill", "wait"]
    assert session.closed and not prepared.closed
    assert workspace.published == []
    assert workspace.denied == 1
    adapter.close()


@pytest.mark.parametrize("boundary", ["kill_exit_code", "wait_exit_code"])
def test_terminal_interrupt_cleanup_failure_fails_closed(tmp_path: Path, boundary: str) -> None:
    """A timeout is not returned until containerd confirms terminal task cleanup."""
    service, _, _, _, control, _, _ = _service(tmp_path, "timeout")
    setattr(control, boundary, 1)
    control.state = {"Status": "running", "Running": True, "Pid": 2, "ExitCode": 0}
    assert isinstance(service.execute(_request()), InfrastructureFailure)


@pytest.mark.parametrize("boundary", ["kill_exit_code", "wait_exit_code"])
def test_terminal_interrupt_accepts_fresh_already_exited_evidence(
    tmp_path: Path, boundary: str
) -> None:
    """A raced idempotent kill or wait remains typed only after exact terminal attestation."""
    service, _, _, _, control, _, _ = _service(tmp_path, "timeout")
    setattr(control, boundary, 1)

    assert isinstance(service.execute(_request()), TimedOut)


def test_write_requests_publish_only_nonempty_authorized_observed_deltas(tmp_path: Path):
    """Empty and denied effects never publish, while approved complete effects publish once."""
    service, _, _, workspace, _, _, _ = _service(tmp_path / "empty")
    write_request = _request().model_copy(
        update={
            "lease": _request().lease.model_copy(
                update={"capabilities": frozenset({Capability.WORKSPACE_WRITE})}
            )
        }
    )
    assert isinstance(service.execute(write_request), Completed)
    assert workspace.denied == 1 and workspace.published == []

    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="directory",
                object_kind=ObjectKind.DIRECTORY,
            ),
        )
    )
    service, _, _, workspace, _, _, _ = _service(tmp_path / "denied")
    workspace.delta = delta
    assert isinstance(service.execute(write_request), CapabilityDenied)
    assert workspace.denied == 1 and workspace.published == []

    service, _, _, workspace, _, _, _ = _service(tmp_path / "approved", authorize_publication=True)
    workspace.delta = delta
    assert isinstance(service.execute(write_request), Completed)
    assert workspace.published == [delta]


@pytest.mark.parametrize(
    ("boundary", "error", "result_type"),
    (
        ("observe", UnrepresentableDeltaError("unsafe"), UnrepresentableDelta),
        ("observe", CommitConflictError("conflict"), CommitConflict),
        ("publish", CommitConflictError("conflict"), CommitConflict),
        ("publish", RuntimeError("failed"), InfrastructureFailure),
    ),
)
def test_write_boundary_failures_remain_typed_and_never_fall_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    boundary: str,
    error: Exception,
    result_type: type,
) -> None:
    """Observation and publication failures remain closed sandbox results."""
    service, _, _, workspace, _, _, _ = _service(tmp_path, authorize_publication=True)
    workspace.delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="directory",
                object_kind=ObjectKind.DIRECTORY,
            ),
        )
    )

    def fail(*_: object) -> None:
        """Raise the selected boundary failure."""
        raise error

    monkeypatch.setattr(workspace, boundary, fail)
    request = _request().model_copy(
        update={
            "lease": _request().lease.model_copy(
                update={"capabilities": frozenset({Capability.WORKSPACE_WRITE})}
            )
        }
    )

    assert isinstance(service.execute(request), result_type)


def test_durable_jobs_and_workspace_provider_failures_remain_closed(tmp_path: Path):
    """Unsupported lifetime and attempt-creation failure cannot start the managed runtime."""
    service, _, backend, workspace, _, _, _ = _service(tmp_path)
    assert isinstance(
        service.execute(_request(mode=ExecutionMode.DURABLE_JOB)), UnsupportedCapability
    )
    assert backend.calls == 0

    @contextmanager
    def fail_lease(*args: object):
        """Raise before yielding an attempt."""
        del args
        raise RuntimeError("invalid generation")
        yield  # pragma: no cover

    workspace.lease = fail_lease  # type: ignore[method-assign]
    assert isinstance(service.execute(_request()), InfrastructureFailure)
    assert backend.calls == 0


@pytest.mark.parametrize(
    "state",
    [
        {"Status": "running", "Running": True, "ExitCode": 7},
        {"Status": "exited", "Running": True, "ExitCode": 7},
        {"Status": "exited", "Running": False, "ExitCode": 8},
        [],
    ],
)
def test_terminal_attestation_mismatches_fail_closed(tmp_path: Path, state: object):
    """Every nonterminal or mismatched state shape fails closed."""
    service, _, _, _, control, _, _ = _service(tmp_path)
    control.state = state
    assert isinstance(service.execute(_request()), InfrastructureFailure)


def test_terminal_evidence_and_cleanup_errors_fail_closed(tmp_path: Path):
    """Malformed inspection and cleanup failures override a completed program."""
    service, _, _, _, control, _, _ = _service(tmp_path / "inspect-exit")
    control.inspect_result = InfrastructureProcessResult(1, b"", b"", False, False)
    assert isinstance(service.execute(_request()), InfrastructureFailure)

    service, _, _, _, control, _, _ = _service(tmp_path / "inspect-json")
    control.inspect_result = InfrastructureProcessResult(0, b"bad", b"", False, False)
    assert isinstance(service.execute(_request()), InfrastructureFailure)

    service, _, _, _, control, _, _ = _service(tmp_path / "cleanup")
    control.remove_exit_code = 1
    assert isinstance(service.execute(_request()), InfrastructureFailure)

    service, _, _, _, control, _, _ = _service(tmp_path / "cleanup-error")
    control.remove_error = True
    assert isinstance(service.execute(_request()), InfrastructureFailure)


@pytest.mark.parametrize("terminal", ["read_error", "slow_reader", "close_error"])
def test_output_reader_failures_fail_closed(tmp_path: Path, terminal: str):
    """Reader failure or nontermination cannot leak a partial success."""
    service, _, _, _, _, _, _ = _service(tmp_path, terminal)
    assert isinstance(service.execute(_request()), InfrastructureFailure)


def test_reader_error_after_terminal_wait_does_not_replace_completion(tmp_path: Path):
    """A transport-close reader error after terminal state is expected cleanup flow."""
    service, _, _, _, _, _, _ = _service(tmp_path, "late_read_error")
    assert isinstance(service.execute(_request()), Completed)

    service, _, _, _, _, _, _ = _service(tmp_path / "close", "close_error_late")
    assert isinstance(service.execute(_request()), Completed)


def test_runtime_compilation_and_launch_failures_remain_closed(tmp_path: Path):
    """Preparation, lease drift, and launch failures cannot become partial attempts."""
    service, _, backend, _, _, _, _ = _service(tmp_path / "prepare")

    def fail_prepare():
        """Raise one sanitized runtime preparation failure."""
        raise RuntimeError("prepare failed")

    backend.prepare = fail_prepare
    assert isinstance(service.execute(_request()), InfrastructureFailure)

    service, _, _, _, control, _, _ = _service(tmp_path / "compile")
    mismatch = _request().model_copy(
        update={"lease": _request().lease.model_copy(update={"runtime_digest": "sha256:wrong"})}
    )
    assert isinstance(service.execute(mismatch), InfrastructureFailure)
    assert control.spec is None

    service, adapter, backend, _, control, prepared, _ = _service(
        tmp_path / "run", prepared_runtime=True
    )
    control.run_error = True
    assert isinstance(service.execute(_request()), InfrastructureFailure)
    assert backend.calls == 0
    adapter.close()
    assert prepared.closed


def test_network_broker_bindings_wrap_attempt_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An exact network lease is planned before launch and reclaimed after container cleanup."""
    service, adapter, _, _, control, _, _ = _service(tmp_path, prepared_runtime=True)
    bindings = SimpleNamespace(
        network_name="loop-network",
        host_aliases=(("api.example", "10.240.1.1"),),
        start_gate="/run/user/1000/loop/brokers/lease/start-gate",
    )
    lease = SimpleNamespace(bindings=bindings, closed=False)

    def close() -> None:
        """Record broker cleanup after command-container removal."""
        lease.closed = True

    lease.close = close
    broker = SimpleNamespace(prepare=lambda *args: lease, activate=lambda *args: None)
    monkeypatch.setattr(adapter, "_prepare_envoy", lambda *args: None)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.execution.MacosEffectBroker",
        lambda *args: broker,
    )
    connection = NetworkConnectionLease(
        hostname="api.example",
        port=443,
        protocol=NetworkProtocol.HTTPS,
        addresses=("93.184.216.34",),
    )
    request = _request(
        lease=_request().lease.model_copy(
            update={
                "capabilities": frozenset(
                    {
                        Capability.WORKSPACE_READ,
                        Capability.PROCESS_SPAWN,
                        Capability.NETWORK_CONNECT,
                    }
                )
            }
        ),
        network_connections=(connection,),
    )

    assert isinstance(service.execute(request), Completed)
    assert control.bindings is bindings
    assert control.hosts_hardened
    assert control.gate_opened
    assert control.hosts_released
    assert lease.closed


@pytest.mark.parametrize("boundary", ("create", "running", "harden", "gate_missing", "gate"))
def test_network_container_prestart_failures_remain_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, boundary: str
) -> None:
    """Container creation and hosts immutability must pass before untrusted task start."""
    service, adapter, _, _, control, _, session = _service(
        tmp_path,
        prepared_runtime=True,
    )
    bindings = SimpleNamespace(
        network_name="loop-network",
        host_aliases=(("api.example", "10.240.1.1"),),
        start_gate="/run/user/1000/loop/brokers/lease/start-gate",
    )
    lease = SimpleNamespace(bindings=bindings, close=lambda: None)
    broker = SimpleNamespace(prepare=lambda *args: lease, activate=lambda *args: None)
    monkeypatch.setattr(adapter, "_prepare_envoy", lambda *args: None)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.execution.MacosEffectBroker",
        lambda *args: broker,
    )
    if boundary == "create":
        control.create_result = InfrastructureProcessResult(1, b"", b"failed", False, False)
    elif boundary == "running":
        monkeypatch.setattr(adapter, "_await_running", lambda *args: False)
    elif boundary == "harden":
        control.hosts_result = InfrastructureProcessResult(0, b"644\n", b"", False, False)
    elif boundary == "gate_missing":
        bindings.start_gate = None
    else:
        control.gate_result = InfrastructureProcessResult(1, b"", b"bad", False, False)
    connection = NetworkConnectionLease(
        hostname="api.example",
        port=443,
        protocol=NetworkProtocol.HTTPS,
        addresses=("93.184.216.34",),
    )
    request = _request(
        lease=_request().lease.model_copy(
            update={
                "capabilities": frozenset(
                    {
                        Capability.WORKSPACE_READ,
                        Capability.PROCESS_SPAWN,
                        Capability.NETWORK_CONNECT,
                    }
                )
            }
        ),
        network_connections=(connection,),
    )
    assert isinstance(service.execute(request), InfrastructureFailure)
    assert not session.stdin_closed


@pytest.mark.parametrize("failure", ("result", "exception"))
def test_network_hosts_cleanup_failure_overrides_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    """A generated hosts file must be restored before nerdctl cleanup can complete."""
    service, adapter, _, _, control, _, _ = _service(tmp_path, prepared_runtime=True)
    bindings = SimpleNamespace(
        network_name="loop-network",
        host_aliases=(("api.example", "10.240.1.1"),),
        start_gate="/run/user/1000/loop/brokers/lease/start-gate",
    )
    lease = SimpleNamespace(bindings=bindings, close=lambda: None)
    monkeypatch.setattr(adapter, "_prepare_envoy", lambda *args: None)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.execution.MacosEffectBroker",
        lambda *args: SimpleNamespace(
            prepare=lambda *args: lease,
            activate=lambda *args: None,
        ),
    )
    if failure == "result":
        control.release_hosts_result = InfrastructureProcessResult(1, b"", b"bad", False, False)
    else:
        control.release_hosts_error = True
    connection = NetworkConnectionLease(
        hostname="api.example",
        port=443,
        protocol=NetworkProtocol.HTTPS,
        addresses=("93.184.216.34",),
    )
    request = _request(
        lease=_request().lease.model_copy(
            update={
                "capabilities": frozenset(
                    {
                        Capability.WORKSPACE_READ,
                        Capability.PROCESS_SPAWN,
                        Capability.NETWORK_CONNECT,
                    }
                )
            }
        ),
        network_connections=(connection,),
    )
    assert isinstance(service.execute(request), InfrastructureFailure)


def test_missing_secret_authority_and_broker_cleanup_failure_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Secret lookup and broker cleanup errors remain opaque infrastructure failures."""
    service, _, _, _, _, _, _ = _service(tmp_path / "secret", prepared_runtime=True)
    exposure = SecretExposure(
        secret_id="token",
        audience="process",
        mechanism=SecretMechanism.RAW_ENVIRONMENT,
        target="TOKEN",
    )
    request = _request(
        lease=_request().lease.model_copy(
            update={
                "capabilities": frozenset(
                    {
                        Capability.WORKSPACE_READ,
                        Capability.PROCESS_SPAWN,
                        Capability.SECRET_USE,
                    }
                )
            }
        ),
        secret_exposures=(exposure,),
    )
    assert isinstance(service.execute(request), InfrastructureFailure)

    service, adapter, _, _, _, _, _ = _service(tmp_path / "cleanup", prepared_runtime=True)
    connection = NetworkConnectionLease(
        hostname="api.example",
        port=443,
        protocol=NetworkProtocol.HTTPS,
        addresses=("93.184.216.34",),
    )
    network_request = _request(
        lease=_request().lease.model_copy(
            update={
                "capabilities": frozenset(
                    {
                        Capability.WORKSPACE_READ,
                        Capability.PROCESS_SPAWN,
                        Capability.NETWORK_CONNECT,
                    }
                )
            }
        ),
        network_connections=(connection,),
    )
    failed_lease = SimpleNamespace(
        bindings=SimpleNamespace(network_name="loop-network", host_aliases=()),
        close=lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )
    monkeypatch.setattr(adapter, "_prepare_envoy", lambda *args: None)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.execution.MacosEffectBroker",
        lambda *args: SimpleNamespace(prepare=lambda *args: failed_lease),
    )
    assert isinstance(service.execute(network_request), InfrastructureFailure)


def test_raw_secret_only_attempt_uses_bindings_without_starting_envoy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Raw process secrets stage through a broker lease while remaining offline."""
    secrets = SimpleNamespace(resolve=lambda secret_id, audience: b"value")
    service, _, _, _, control, _, _ = _service(
        tmp_path,
        prepared_runtime=True,
        secret_authority=secrets,
    )
    bindings = SimpleNamespace(network_name=None, host_aliases=())
    lease = SimpleNamespace(bindings=bindings, close=lambda: None)
    broker = SimpleNamespace(prepare=lambda *args: lease)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.execution.MacosEffectBroker",
        lambda *args: broker,
    )
    exposure = SecretExposure(
        secret_id="token",
        audience="process",
        mechanism=SecretMechanism.RAW_ENVIRONMENT,
        target="TOKEN",
    )
    request = _request(
        lease=_request().lease.model_copy(
            update={
                "capabilities": frozenset(
                    {
                        Capability.WORKSPACE_READ,
                        Capability.PROCESS_SPAWN,
                        Capability.SECRET_USE,
                    }
                )
            }
        ),
        secret_exposures=(exposure,),
    )
    assert isinstance(service.execute(request), Completed)
    assert control.bindings is bindings


def test_envoy_image_preparation_is_lazy_and_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pinned sidecar image installs exactly once for a prepared runtime."""
    _, adapter, _, _, _, prepared, _ = _service(tmp_path, prepared_runtime=True)
    installs = []

    def install(self, artifact, lockfile, directory) -> None:
        """Record the exact pinned broker artifact and temporary attestation directory."""
        del self, lockfile
        installs.append((artifact, Path(directory)))

    monkeypatch.setattr(
        "loop.execution.sandbox.macos.execution.OciArtifactInstaller.install",
        install,
    )
    envoy = macos_artifact(_candidate(), "envoy")
    adapter._prepare_envoy(prepared, envoy)
    adapter._prepare_envoy(prepared, envoy)
    assert len(installs) == 1
    assert installs[0][0] is envoy


@pytest.mark.parametrize(
    "state_result",
    (
        InfrastructureProcessResult(1, b"", b"bad", False, False),
        InfrastructureProcessResult(0, b"bad-json", b"", False, False),
    ),
)
def test_start_gate_wait_rejects_failed_and_malformed_state(
    monkeypatch: pytest.MonkeyPatch, state_result: InfrastructureProcessResult
) -> None:
    """The fixed wrapper is never released without structured running evidence."""
    moments = iter((0.0, 0.0, 10.0))
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.execution.time.monotonic", lambda: next(moments)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.execution.time.sleep", lambda _: None)
    control = SimpleNamespace(inspect_container_state=lambda _: state_result)
    assert not MacosExecutionAdapter._await_running(control, "loop-request")
