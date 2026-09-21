"""Tests for immutable execution request and result contracts."""

import pytest
from pydantic import TypeAdapter, ValidationError

from loop.execution.contracts import (
    BackendAttestation,
    DeltaEffect,
    DirectExecutionRequest,
    ExecutionLease,
    ExecutionMode,
    HostExecutionRequest,
    ShellExecutionRequest,
    WorkspaceDelta,
    WorkspaceDeltaEntry,
)
from loop.execution.results import TERMINAL_RESULT_KINDS, ExecutionResult


def _lease() -> ExecutionLease:
    """Build one valid sandbox lease."""
    return ExecutionLease(
        lease_id="lease",
        workspace_id="workspace",
        agent_run_id="agent",
        policy_version="policy",
        runtime_digest="sha256:runtime",
        expires_at_ns=1,
    )


def test_sandbox_requests_remain_virtual_and_distinguish_shell_from_direct_execution():
    """Sandbox contracts retain opaque shell syntax and exact direct guest argv."""
    shell = ShellExecutionRequest(request_id="shell", lease=_lease(), script="printf 'x' | cat")
    direct = DirectExecutionRequest(
        request_id="direct",
        lease=_lease(),
        argv=("/tools/bin/example", "argument"),
        mode=ExecutionMode.DURABLE_JOB,
    )

    assert shell.cwd == "/workspace"
    assert shell.boundary == "sandbox"
    assert shell.script == "printf 'x' | cat"
    assert direct.argv == ("/tools/bin/example", "argument")
    assert direct.mode is ExecutionMode.DURABLE_JOB


def test_execution_contracts_reject_relative_cwds_and_empty_host_requests():
    """Contracts reject host-shaped cwd ambiguity and incomplete host authority."""
    try:
        ShellExecutionRequest(request_id="request", lease=_lease(), cwd="relative", script="true")
    except ValidationError as error:
        assert "absolute virtual path" in str(error)
    else:  # pragma: no cover - direct failure reporting for the validation contract.
        raise AssertionError("A relative cwd must be rejected.")

    try:
        HostExecutionRequest(
            request_id="request",
            executable="/host/tool",
            argv=(),
            cwd="/host/workspace",
            display_cwd="workspace",
            resource_class="native-tool",
            reason="needed",
        )
    except ValidationError:
        pass
    else:  # pragma: no cover - direct failure reporting for the validation contract.
        raise AssertionError("A host request must have an exact argv.")


def test_host_and_backend_contracts_are_immutable_and_explicit():
    """Host authorization and backend evidence carry distinct immutable identities."""
    host = HostExecutionRequest(
        request_id="host",
        executable="/host/tool",
        argv=("/host/tool",),
        cwd="/host/workspace",
        display_cwd="workspace",
        resource_class="native-tool",
        reason="macOS-only capability",
    )
    attestation = BackendAttestation(
        backend_id="linux-rootless",
        runtime_digest="sha256:runtime",
        policy_digest="sha256:policy",
    )

    assert host.boundary == "host"
    assert attestation.evidence == ()
    try:
        host.reason = "changed"
    except ValidationError:
        pass
    else:  # pragma: no cover - direct failure reporting for frozen models.
        raise AssertionError("Execution contracts must be immutable.")


def test_workspace_delta_exposes_only_canonical_virtual_effects():
    """Delta contracts retain effect identity without carrying a host workspace path."""
    delta = WorkspaceDelta(
        entries=(
            WorkspaceDeltaEntry(
                effect=DeltaEffect.RENAME,
                path="/workspace/new-name",
                source_path="/workspace/old-name",
            ),
        )
    )

    assert delta.entries[0].effect is DeltaEffect.RENAME
    assert WorkspaceDeltaEntry(effect=DeltaEffect.CREATE, path="/workspace/file").effect is (
        DeltaEffect.CREATE
    )
    with pytest.raises(ValidationError):
        WorkspaceDeltaEntry(effect=DeltaEffect.CREATE, path="relative")
    with pytest.raises(ValidationError):
        WorkspaceDeltaEntry(effect=DeltaEffect.CREATE, path="/workspace/file", source_path="/x")
    with pytest.raises(ValidationError):
        WorkspaceDeltaEntry(effect=DeltaEffect.RENAME, path="/workspace/file")
    with pytest.raises(ValidationError):
        WorkspaceDeltaEntry(
            effect=DeltaEffect.RENAME,
            path="/workspace/file",
            source_path="relative",
        )


def test_result_union_accepts_every_closed_result_kind_and_no_unknown_kind():
    """Result parsing is exhaustive and rejects omitted taxonomy variants."""
    adapter = TypeAdapter(ExecutionResult)
    payloads = {
        "completed": {"exit_code": 1},
        "executable_not_found": {"executable": "missing"},
        "capability_denied": {"capability": "network.connect"},
        "unsupported_capability": {"capability": "pty"},
        "workspace_busy": {},
        "unrepresentable_delta": {},
        "commit_conflict": {},
        "infrastructure_failure": {},
        "resource_limit_exceeded": {"resource": "memory"},
        "timed_out": {},
        "cancelled": {},
        "spawn_failure": {},
    }

    assert set(payloads) == TERMINAL_RESULT_KINDS
    assert {
        adapter.validate_python({"request_id": "request", "kind": kind, **payload}).kind
        for kind, payload in payloads.items()
    } == TERMINAL_RESULT_KINDS
    try:
        adapter.validate_python({"request_id": "request", "kind": "host_fallback"})
    except ValidationError:
        pass
    else:  # pragma: no cover - direct failure reporting for closed result parsing.
        raise AssertionError("Unknown outcomes must not become a fallback result.")
