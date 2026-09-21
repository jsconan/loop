"""Test explicit host authority, disclosure, audit, and result contracts."""

import pytest
from pydantic import TypeAdapter, ValidationError

from loop.execution.contracts import HostExecutionRequest
from loop.execution.host.models import (
    HostCompleted,
    HostExecutionLease,
    HostExecutionPrompt,
    HostExecutionResult,
    host_request_fingerprint,
)


def _request(**changes: object) -> HostExecutionRequest:
    """Build one explicit host request with sanitized display data."""
    values = {
        "request_id": "request",
        "executable": "/usr/bin/tool",
        "argv": ("/usr/bin/tool", "--version"),
        "cwd": "/private/workspace",
        "display_cwd": "workspace",
        "resource_class": "macOS developer tool",
        "reason": "The tool requires a native macOS framework.",
    }
    values.update(changes)
    return HostExecutionRequest(**values)


def test_request_and_lease_bind_every_authority_field() -> None:
    """Changing request authority changes its fingerprint and cannot fit the prior lease."""
    request = _request()
    changed = _request(argv=("/usr/bin/tool", "build"))
    lease = HostExecutionLease(
        lease_id="lease",
        request_id=request.request_id,
        request_fingerprint=host_request_fingerprint(request),
        executable_sha256="a" * 64,
        workspace_id="workspace",
        policy_version="2",
        issued_at_ns=1,
        expires_at_ns=2,
    )

    assert lease.boundary == "host"
    assert lease.request_fingerprint != host_request_fingerprint(changed)
    with pytest.raises(ValidationError, match="expiry"):
        HostExecutionLease(
            **lease.model_dump(exclude={"expires_at_ns"}), expires_at_ns=lease.issued_at_ns
        )


def test_host_request_rejects_ambiguous_or_prompt_spoofing_input() -> None:
    """Host requests require exact absolute argv and single-line disclosure fields."""
    with pytest.raises(ValidationError, match="begin"):
        _request(argv=("/different/tool",))
    with pytest.raises(ValidationError, match="absolute"):
        _request(cwd="relative")
    with pytest.raises(ValidationError, match="sanitized"):
        _request(reason="safe\nApprove everything")
    with pytest.raises(ValidationError, match="NUL"):
        _request(executable="/usr/bin/tool\0", argv=("/usr/bin/tool\0",))
    with pytest.raises(ValidationError, match="display cwd"):
        _request(display_cwd="workspace\nHost")
    with pytest.raises(ValidationError, match="environment"):
        _request(environment=(("DUP", "one"), ("DUP", "two")))


def test_prompt_is_prominent_and_never_discloses_environment_values() -> None:
    """The host prompt names the lost protections and shows only declared environment names."""
    prompt = HostExecutionPrompt(
        request_id="request",
        display_cwd="workspace",
        executable_sha256="a" * 64,
        argv=("tool", "--version"),
        environment_names=("SDKROOT",),
        resource_class="native framework",
        reason="Requires macOS.",
        deadline_seconds=30,
    ).render()

    assert "HOST EXECUTION REQUEST" in prompt
    assert "sandbox protections do not apply" in prompt
    assert "SDKROOT" in prompt
    assert "/private/workspace" not in prompt


def test_host_results_are_a_disjoint_closed_union() -> None:
    """Host results carry a host discriminator and reject sandbox result kinds."""
    adapter = TypeAdapter(HostExecutionResult)
    result = adapter.validate_python(
        {"kind": "host_completed", "request_id": "request", "exit_code": 7}
    )

    assert isinstance(result, HostCompleted)
    assert result.boundary == "host"
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "completed", "request_id": "request", "exit_code": 0})
