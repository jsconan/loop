"""Test the sole identity, permission, lease, and dispatch path for host execution."""

from dataclasses import dataclass, field
from pathlib import Path

from loop.execution.contracts import HostExecutionRequest
from loop.execution.host.broker import HostExecutionBroker
from loop.execution.host.models import (
    AuthorizedHostExecution,
    HostAuditEvent,
    HostAuthorizationDecision,
    HostCompleted,
    HostInfrastructureFailure,
    HostIntegrityFailure,
    HostPermissionDenialReason,
    HostPermissionDenied,
)


@dataclass
class _Audit:
    """Collect sanitized host events for assertions."""

    events: list[HostAuditEvent] = field(default_factory=list)

    def record(self, event: HostAuditEvent) -> None:
        """Append one immutable event."""
        self.events.append(event)


@dataclass
class _Permissions:
    """Return one configured host authorization decision."""

    allowed: bool
    calls: list[tuple] = field(default_factory=list)

    def authorize(self, request, executable, prompt) -> HostAuthorizationDecision:
        """Record the exact verified request and disclosure."""
        self.calls.append((request, executable, prompt))
        return HostAuthorizationDecision(
            allowed=self.allowed,
            denial_reason=(None if self.allowed else HostPermissionDenialReason.POLICY_DENIED),
        )


class _BrokenAudit:
    """Reject every security audit write."""

    def record(self, event: HostAuditEvent) -> None:
        """Raise one deterministic audit failure."""
        del event
        raise RuntimeError("audit unavailable")


@dataclass
class _FailAfterAudit:
    """Accept a bounded number of events before rejecting audit writes."""

    remaining: int

    def record(self, event: HostAuditEvent) -> None:
        """Accept or reject the event according to the remaining allowance."""
        del event
        if self.remaining == 0:
            raise RuntimeError("audit unavailable")
        self.remaining -= 1


@dataclass
class _Supervisor:
    """Record authorized commands without creating a child."""

    commands: list[AuthorizedHostExecution] = field(default_factory=list)

    def execute(self, authorized, cancellation=lambda: False):
        """Return successful completion for one fully authorized command."""
        del cancellation
        self.commands.append(authorized)
        return HostCompleted(request_id=authorized.request.request_id, exit_code=0)


def _request(executable: Path, cwd: Path) -> HostExecutionRequest:
    """Build one explicit request for a real test executable."""
    return HostExecutionRequest(
        request_id="request",
        executable=str(executable),
        argv=(str(executable), "argument"),
        cwd=str(cwd),
        display_cwd="workspace",
        resource_class="native tool",
        reason="Requires a host-only API.",
    )


def _executable(tmp_path: Path) -> Path:
    """Create one executable regular file with stable content."""
    path = tmp_path / "tool"
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def test_broker_verifies_authorizes_leases_and_dispatches_exact_request(tmp_path: Path) -> None:
    """A host start receives the exact verified request and a bounded matching lease."""
    executable = _executable(tmp_path)
    permissions = _Permissions(True)
    supervisor = _Supervisor()
    audit = _Audit()
    broker = HostExecutionBroker(
        permissions,
        "workspace",
        "2",
        supervisor=supervisor,  # type: ignore[arg-type]
        audit=audit,
        clock_ns=lambda: 10,
        id_factory=lambda: "lease",
    )

    result = broker.execute(_request(executable, tmp_path))

    assert isinstance(result, HostCompleted)
    assert [event.type.value for event in audit.events] == [
        "execution.host.requested",
        "execution.host.authorized",
    ]
    assert supervisor.commands[0].lease.lease_id == "lease"
    assert supervisor.commands[0].lease.executable_sha256 == permissions.calls[0][1].sha256
    assert "sandbox protections do not apply" in permissions.calls[0][2].render()


def test_broker_maps_virtual_workspace_cwd_without_disclosing_host_path(tmp_path: Path) -> None:
    """Explicit host tools accept the public workspace namespace after normal authorization."""
    executable = _executable(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    supervisor = _Supervisor()
    broker = HostExecutionBroker(
        _Permissions(True),
        "workspace",
        "2",
        workspace_root=tmp_path,
        supervisor=supervisor,  # type: ignore[arg-type]
        audit=_Audit(),
        clock_ns=lambda: 10,
    )
    request = _request(executable, tmp_path).model_copy(
        update={"cwd": "/workspace/nested", "display_cwd": "project workspace"}
    )

    assert isinstance(broker.execute(request), HostCompleted)
    assert supervisor.commands[0].cwd.path == nested
    assert supervisor.commands[0].request.cwd == nested.as_posix()

    escaping = request.model_copy(update={"cwd": "/workspace/../outside"})
    assert isinstance(broker.execute(escaping), HostIntegrityFailure)
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    nested.joinpath("escape").symlink_to(outside, target_is_directory=True)
    substituted = request.model_copy(update={"cwd": "/workspace/nested/escape"})
    assert isinstance(broker.execute(substituted), HostIntegrityFailure)


def test_broker_never_starts_without_identity_and_permission(tmp_path: Path) -> None:
    """Invalid identity and permission denial terminate before lease creation or dispatch."""
    supervisor = _Supervisor()
    denied = _Permissions(False)
    broker = HostExecutionBroker(
        denied,
        "workspace",
        "2",
        supervisor=supervisor,  # type: ignore[arg-type]
        audit=_Audit(),
        clock_ns=lambda: 10,
    )
    executable = _executable(tmp_path)

    result = broker.execute(_request(executable, tmp_path))
    assert isinstance(result, HostPermissionDenied)
    assert result.reason is HostPermissionDenialReason.POLICY_DENIED
    assert not supervisor.commands
    executable.unlink()
    assert isinstance(broker.execute(_request(executable, tmp_path)), HostIntegrityFailure)
    assert len(denied.calls) == 1
    assert not supervisor.commands


def test_broker_fails_closed_when_required_audit_cannot_be_recorded(tmp_path: Path) -> None:
    """Audit failure before authorization creates no permission request, lease, or host start."""
    executable = _executable(tmp_path)
    permissions = _Permissions(True)
    supervisor = _Supervisor()
    broker = HostExecutionBroker(
        permissions,
        "workspace",
        "2",
        supervisor=supervisor,  # type: ignore[arg-type]
        audit=_BrokenAudit(),
        clock_ns=lambda: 10,
    )

    assert isinstance(broker.execute(_request(executable, tmp_path)), HostInfrastructureFailure)
    assert not permissions.calls
    assert not supervisor.commands

    broker = HostExecutionBroker(
        permissions,
        "workspace",
        "2",
        supervisor=supervisor,  # type: ignore[arg-type]
        audit=_FailAfterAudit(1),
        clock_ns=lambda: 10,
    )
    assert isinstance(broker.execute(_request(executable, tmp_path)), HostInfrastructureFailure)
    assert len(permissions.calls) == 1
    assert not supervisor.commands
