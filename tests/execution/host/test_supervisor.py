"""Test lease validation and closed outcomes in the explicit host supervisor."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from loop.execution.contracts import HostExecutionRequest
from loop.execution.host import supervisor as supervisor_module
from loop.execution.host.models import (
    AuthorizedHostExecution,
    HostAuditEvent,
    HostCancelled,
    HostCompleted,
    HostExecutionLease,
    HostInfrastructureFailure,
    HostIntegrityFailure,
    HostSpawnFailure,
    HostTimedOut,
    host_request_fingerprint,
)
from loop.execution.host.supervisor import HostExecutionSupervisor
from loop.execution.infrastructure import (
    InfrastructureProcessResult,
    ProcessCancelledError,
    ProcessSpawnError,
    ProcessTimedOutError,
    identify_host_directory,
    identify_host_executable,
)


@dataclass
class _Audit:
    """Collect host lifecycle events."""

    events: list[HostAuditEvent] = field(default_factory=list)

    def record(self, event: HostAuditEvent) -> None:
        """Append one event."""
        self.events.append(event)


class _BrokenAudit:
    """Reject every host audit write."""

    def record(self, event: HostAuditEvent) -> None:
        """Raise one deterministic audit failure."""
        del event
        raise RuntimeError("audit unavailable")


class _Runner:
    """Return or raise one configured process primitive outcome."""

    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.calls = []

    def run(self, command, **kwargs):
        """Record the prevalidated command and trigger its start notification."""
        self.calls.append((command, kwargs))
        kwargs["started"]()
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _authorized(tmp_path: Path, **request_changes: object) -> AuthorizedHostExecution:
    """Build one exact live host lease over real descriptor-derived identities."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    executable_path = tmp_path / "tool"
    executable_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable_path.chmod(0o700)
    values = {
        "request_id": "request",
        "executable": str(executable_path),
        "argv": (str(executable_path),),
        "cwd": str(tmp_path),
        "display_cwd": "workspace",
        "resource_class": "native tool",
        "reason": "Requires host integration.",
    }
    values.update(request_changes)
    request = HostExecutionRequest(**values)
    executable = identify_host_executable(executable_path)
    cwd = identify_host_directory(tmp_path)
    lease = HostExecutionLease(
        lease_id="lease",
        request_id=request.request_id,
        request_fingerprint=host_request_fingerprint(request),
        executable_sha256=executable.sha256,
        workspace_id="workspace",
        policy_version="2",
        issued_at_ns=1,
        expires_at_ns=100,
    )
    return AuthorizedHostExecution(request, lease, executable, cwd)


def test_supervisor_uses_scrubbed_environment_and_audits_actual_start(tmp_path: Path) -> None:
    """Only fixed baseline and disclosed safe variables reach a lease-bound host process."""
    authorized = _authorized(tmp_path, environment=(("SDKROOT", "sdk"),))
    runner = _Runner(InfrastructureProcessResult(3, b"out", b"err", False, False))
    audit = _Audit()
    supervisor = HostExecutionSupervisor(
        audit=audit, clock_ns=lambda: 10, runner_factory=lambda _limit: runner
    )

    result = supervisor.execute(authorized)

    assert isinstance(result, HostCompleted)
    assert result.exit_code == 3
    command, options = runner.calls[0]
    assert command.environment == {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": f"{authorized.executable.path.parent}:/usr/bin:/bin",
        "SDKROOT": "sdk",
    }
    assert options["cleanup_descendants"] is True
    assert [event.type.value for event in audit.events] == [
        "execution.host.started",
        "execution.host.terminal",
    ]


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (ProcessSpawnError("spawn"), HostSpawnFailure),
        (ProcessTimedOutError("timeout"), HostTimedOut),
        (ProcessCancelledError("cancel"), HostCancelled),
        (RuntimeError("reader"), HostInfrastructureFailure),
    ],
)
def test_supervisor_maps_process_failures_to_closed_host_results(
    tmp_path: Path, failure: RuntimeError, expected: type
) -> None:
    """Process primitive failures cannot escape or become sandbox outcomes."""
    runner = _Runner(failure)
    result = HostExecutionSupervisor(
        audit=_Audit(), clock_ns=lambda: 10, runner_factory=lambda _limit: runner
    ).execute(_authorized(tmp_path))

    assert isinstance(result, expected)


def test_supervisor_rejects_expired_changed_and_unsafe_environment_authority(
    tmp_path: Path,
) -> None:
    """Expired leases, changed requests, and process-control variables fail before start."""
    runner = _Runner(InfrastructureProcessResult(0, b"", b"", False, False))
    supervisor = HostExecutionSupervisor(
        audit=_Audit(), clock_ns=lambda: 100, runner_factory=lambda _limit: runner
    )
    expired = _authorized(tmp_path)
    assert isinstance(supervisor.execute(expired), HostIntegrityFailure)
    unsafe = _authorized(tmp_path / "unsafe", environment=(("PATH", "/attacker"),))
    unsafe_supervisor = HostExecutionSupervisor(
        audit=_Audit(), clock_ns=lambda: 10, runner_factory=lambda _limit: runner
    )
    assert isinstance(unsafe_supervisor.execute(unsafe), HostIntegrityFailure)
    assert not runner.calls


def test_default_host_audit_emits_only_sanitized_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production audit adapter omits command content, environment, and host paths."""
    calls = []
    monkeypatch.setattr(
        supervisor_module, "telemetry_audit", lambda event, **fields: calls.append((event, fields))
    )

    supervisor_module.TelemetryHostAudit().record(
        HostAuditEvent(
            type="execution.host.started",
            request_id="request",
            lease_id="lease",
            executable_sha256="a" * 64,
        )
    )

    assert calls == [
        (
            "execution.host.started",
            {
                "request_id": "request",
                "lease_id": "lease",
                "executable_sha256": "a" * 64,
                "outcome": None,
            },
        )
    ]


def test_supervisor_returns_closed_failure_when_terminal_audit_fails(tmp_path: Path) -> None:
    """An audit sink failure cannot escape the host result boundary."""
    runner = _Runner(InfrastructureProcessResult(0, b"", b"", False, False))

    result = HostExecutionSupervisor(
        audit=_BrokenAudit(), clock_ns=lambda: 10, runner_factory=lambda _limit: runner
    ).execute(_authorized(tmp_path))

    assert isinstance(result, HostInfrastructureFailure)
