"""Test explicit host requests through the retained permission facade."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from loop.execution.contracts import HostExecutionRequest
from loop.execution.host.models import HostExecutionPrompt, HostPermissionDenialReason
from loop.execution.infrastructure import identify_host_executable
from loop.interaction import Interaction
from loop.permissions import (
    ApprovalChoice,
    GrantDecision,
    HostExecutionPermissionAdapter,
    PermissionConfiguration,
    PermissionManager,
    PolicyLimits,
)


def _request(executable: Path, cwd: Path) -> HostExecutionRequest:
    """Build one exact explicit host request."""
    return HostExecutionRequest(
        request_id="request",
        executable=str(executable),
        argv=(str(executable), "--version"),
        cwd=str(cwd),
        display_cwd="workspace",
        resource_class="native tool",
        reason="Requires macOS integration.",
        deadline_seconds=30,
    )


def _prompt(identity: str) -> HostExecutionPrompt:
    """Build one complete host disclosure."""
    return HostExecutionPrompt(
        request_id="request",
        display_cwd="workspace",
        executable_sha256=identity,
        argv=("tool", "--version"),
        environment_names=(),
        resource_class="native tool",
        reason="Requires macOS integration.",
        deadline_seconds=30,
    )


def _adapter(manager: PermissionManager) -> HostExecutionPermissionAdapter:
    """Bind host authorization to one workspace and session."""
    return HostExecutionPermissionAdapter(
        manager, "host_tool", "loop", "workspace", "2", "session", clock_ns=lambda: 10
    )


def test_host_ceiling_denies_before_interaction(tmp_path: Path) -> None:
    """A disabled host ceiling cannot be overridden by a prompt or stored allow."""
    executable_path = tmp_path / "tool"
    executable_path.write_bytes(b"tool")
    executable_path.chmod(0o700)
    executable = identify_host_executable(executable_path)
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(
        tmp_path,
        workspace_id="workspace",
        interaction=interaction,
        configuration=PermissionConfiguration(limits=PolicyLimits(allow_host_processes=False)),
    )

    decision = _adapter(manager).authorize(
        _request(executable_path, tmp_path), executable, _prompt(executable.sha256)
    )
    assert not decision.allowed
    assert decision.denial_reason is HostPermissionDenialReason.HOST_PROCESSES_DISABLED
    interaction.prompt.assert_not_called()


def test_host_prompt_is_prominent_and_scoped_grant_reuses_exact_authority(tmp_path: Path) -> None:
    """Approval uses the host disclosure and reuses only the exact bounded request."""
    executable_path = tmp_path / "tool"
    executable_path.write_bytes(b"tool")
    executable_path.chmod(0o700)
    executable = identify_host_executable(executable_path)
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.SESSION
    manager = PermissionManager(
        tmp_path,
        workspace_id="workspace",
        interaction=interaction,
        configuration=PermissionConfiguration(limits=PolicyLimits(allow_host_processes=True)),
    )
    adapter = _adapter(manager)
    request = _request(executable_path, tmp_path)

    assert adapter.authorize(request, executable, _prompt(executable.sha256)).allowed
    assert "HOST EXECUTION REQUEST" in interaction.info.call_args.args[0]
    assert adapter.authorize(request, executable, _prompt(executable.sha256)).allowed
    assert interaction.prompt.call_count == 1
    changed = request.model_copy(update={"argv": (str(executable_path), "build")})
    interaction.prompt.return_value = ApprovalChoice.DENY
    denied = adapter.authorize(changed, executable, _prompt(executable.sha256))
    assert not denied.allowed
    assert denied.denial_reason is HostPermissionDenialReason.USER_DENIED
    assert interaction.prompt.call_count == 2


def test_host_denial_distinguishes_unavailable_approval_from_other_policy_denial(
    tmp_path: Path,
) -> None:
    """Return safe diagnostics when prompting is impossible or policy rejects authority."""
    executable_path = tmp_path / "tool"
    executable_path.write_bytes(b"tool")
    executable_path.chmod(0o700)
    executable = identify_host_executable(executable_path)
    prompt = _prompt(executable.sha256)
    unavailable = PermissionManager(
        tmp_path,
        workspace_id="workspace",
        configuration=PermissionConfiguration(limits=PolicyLimits(allow_host_processes=True)),
    )

    decision = _adapter(unavailable).authorize(
        _request(executable_path, tmp_path), executable, prompt
    )

    assert not decision.allowed
    assert decision.denial_reason is HostPermissionDenialReason.APPROVAL_UNAVAILABLE

    policy = Mock()
    policy.effective_configuration.limits.allow_host_processes = True
    policy.authorize_execution.return_value = SimpleNamespace(
        decision=GrantDecision.DENY,
        prompted=False,
        reason="A typed deny rejected the request.",
    )

    denied = _adapter(policy).authorize(_request(executable_path, tmp_path), executable, prompt)

    assert not denied.allowed
    assert denied.denial_reason is HostPermissionDenialReason.POLICY_DENIED
