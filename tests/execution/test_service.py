"""Tests for the ordinary sandbox-only execution service."""

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from loop.execution.contracts import (
    ExecutionLease,
    HostExecutionRequest,
    SandboxExecutionRequest,
    ShellExecutionRequest,
)
from loop.execution.host import HostExecutionBroker
from loop.execution.results import (
    Cancelled,
    CapabilityDenied,
    CommitConflict,
    Completed,
    ExecutableNotFound,
    ExecutionResult,
    InfrastructureFailure,
    ResourceLimitExceeded,
    SpawnFailure,
    TimedOut,
    UnrepresentableDelta,
    UnsupportedCapability,
    WorkspaceBusy,
)
from loop.execution.service import ExecutionService


def _request() -> ShellExecutionRequest:
    """Build one authorized sandbox shell request."""
    return ShellExecutionRequest(
        request_id="request",
        lease=ExecutionLease(
            lease_id="lease",
            workspace_id="workspace",
            agent_run_id="agent",
            policy_version="policy",
            runtime_digest="sha256:runtime",
            expires_at_ns=1,
        ),
        script="false",
    )


@dataclass
class _FakeSandboxAdapter:
    """Return a deterministic sandbox outcome without creating a process."""

    result: ExecutionResult
    requests: list[SandboxExecutionRequest]

    def execute(self, request, observer, cancellation) -> ExecutionResult:
        """Record one sandbox request and return the configured result."""
        del observer, cancellation
        self.requests.append(request)
        return self.result


class _BrokenSandboxAdapter:
    """Raise at the adapter boundary to verify service-level closure."""

    def execute(self, request, observer, cancellation):
        """Raise one unexpected adapter failure."""
        del request, observer, cancellation
        raise RuntimeError("privileged failure")


def test_service_returns_a_sandbox_infrastructure_failure_without_host_dispatch():
    """Every adapter failure remains the terminal sandbox result."""
    failure = InfrastructureFailure(request_id="request", diagnostic_id="opaque")
    adapter = _FakeSandboxAdapter(result=failure, requests=[])

    result = ExecutionService(adapter).execute(_request())

    assert result is failure
    assert adapter.requests == [_request()]


def test_service_rejects_a_host_request_before_it_can_reach_a_sandbox_adapter():
    """The ordinary service cannot convert an explicit host request into sandbox work."""
    adapter = _FakeSandboxAdapter(
        result=InfrastructureFailure(request_id="request"),
        requests=[],
    )
    host_request = HostExecutionRequest(
        request_id="host",
        executable="/host/tool",
        argv=("/host/tool",),
        cwd="/host/workspace",
        display_cwd="workspace",
        resource_class="native-tool",
        reason="required",
    )

    with pytest.raises(ValidationError):
        ExecutionService(adapter).execute(host_request)  # type: ignore[arg-type]

    assert not adapter.requests


def test_service_maps_unexpected_adapter_failures_to_a_closed_result():
    """Unexpected sandbox adapter errors cannot escape or become host work."""
    assert isinstance(
        ExecutionService(_BrokenSandboxAdapter()).execute(_request()), InfrastructureFailure
    )


@pytest.mark.parametrize(
    "result",
    [
        Completed(request_id="request", exit_code=1),
        ExecutableNotFound(request_id="request", executable="missing"),
        CapabilityDenied(request_id="request", capability="network.connect"),
        UnsupportedCapability(request_id="request", capability="pty"),
        WorkspaceBusy(request_id="request"),
        UnrepresentableDelta(request_id="request"),
        CommitConflict(request_id="request"),
        InfrastructureFailure(request_id="request"),
        ResourceLimitExceeded(request_id="request", resource="memory"),
        TimedOut(request_id="request"),
        Cancelled(request_id="request"),
        SpawnFailure(request_id="request"),
    ],
)
def test_every_sandbox_terminal_fault_produces_zero_host_requests_or_starts(
    result: ExecutionResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No closed sandbox outcome can invoke the distinct host broker."""
    host_execute = Mock()
    monkeypatch.setattr(HostExecutionBroker, "execute", host_execute)
    adapter = _FakeSandboxAdapter(result=result, requests=[])

    assert ExecutionService(adapter).execute(_request()) is result
    host_execute.assert_not_called()
