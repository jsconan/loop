"""Verify separate sandbox and exact recorded host retry authorization."""

import json
import os
import sqlite3
import subprocess
import tempfile
import time
from contextlib import closing
from dataclasses import replace
from io import BytesIO, TextIOWrapper
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop import Action, Decision, Interaction, PermissionConfiguration, PermissionManager
from loop.execution.coordinator import (
    CommandExecutionCoordinator,
    HostCommandOffer,
    HostCommandRequest,
    LocalHostCommandExecutor,
)
from loop.execution.sandbox import CommandProcessResult, SandboxOutcome, SandboxRequest
from loop.permissions import (
    ApprovalChoice,
    PermissionRule,
    PolicyLimits,
    PolicyScope,
    ProcessBoundary,
    ProcessTarget,
)
from loop.permissions.managed import ManagedPolicyStore
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry
from loop.workspace import Workspace, WorkspaceRepository


def managed_store(tmp_path, *, rules=(), forbid_host=False):
    """Create an owned administrator policy fixture for coordinator admission."""
    directory = tmp_path / "system" / "loop"
    directory.mkdir(parents=True)
    directory.parent.chmod(0o755)
    directory.chmod(0o755)
    identity = "b" * 32
    (directory.parent / "loop.managed.expect.json").write_text(
        json.dumps({"version": 1, "identity": identity, "minimum_revision": 1})
    )
    (directory / "managed-policy.json").write_text(
        json.dumps(
            {
                "version": 1,
                "identity": identity,
                "revision": 1,
                "rules": list(rules),
                "forbid_host": forbid_host,
            }
        )
    )
    (directory.parent / "loop.managed.expect.json").chmod(0o644)
    (directory / "managed-policy.json").chmod(0o644)
    return ManagedPolicyStore(directory, trust_anchor=tmp_path, owner_uid=os.getuid())


@pytest.mark.parametrize("yolo", [False, True])
def test_managed_host_prohibition_blocks_explicit_request(tmp_path, yolo):
    """Neither YOLO nor a user host grant bypasses managed host prohibition."""
    manager = PermissionManager(
        tmp_path, managed_policy_store=managed_store(tmp_path, forbid_host=True), yolo=yolo
    )
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager.trust_host_command("printf host")
    host = Mock()
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    service = CommandExecutionCoordinator(Mock(), host, manager, interaction)
    assert (
        service.request_host_command(
            trusted_request(tmp_path, manager, "printf host"), "request_host_command", "Needed"
        )
        is None
    )
    host.run_host_command.assert_not_called()
    interaction.prompt.assert_not_called()


def test_managed_human_rule_disables_host_grant_reuse_and_new_grants(tmp_path):
    """An existing host grant cannot satisfy a managed fresh-human requirement."""
    store = managed_store(
        tmp_path,
        rules=(
            {
                "id": "fresh-host",
                "decision": "ask",
                "action": "process.execute",
                "tool": "request_host_command",
            },
        ),
    )
    manager = PermissionManager(tmp_path, managed_policy_store=store)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager.trust_host_command("printf host")
    host = Mock()
    host.run_host_command.return_value = "host-result"
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    service = CommandExecutionCoordinator(Mock(), host, manager, interaction)
    current = trusted_request(tmp_path, manager, "printf host")
    assert service.request_host_command(current, "request_host_command", "Needed") == "host-result"
    assert set(interaction.prompt.call_args.kwargs["choices"]) == {"deny", "approve"}
    interaction.prompt.return_value = "deny"
    assert service.request_host_command(current, "request_host_command", "Needed") is None
    assert host.run_host_command.call_count == 1


def request(tmp_path, **changes):
    """Build a bound sandbox command in the disposable workspace."""
    values = {
        "source": "printf ok",
        "cwd": tmp_path,
        "workspace": tmp_path,
        "read_roots": (tmp_path.parent,),
        "write_roots": (tmp_path,),
        "network": False,
        "environment": {"PATH": "/usr/bin:/bin"},
        "policy_version": "macos-v1",
        "deadline": time.monotonic() + 60,
        "workspace_id": "workspace-1",
    }
    values.update(changes)
    return SandboxRequest.create(**values)


def trusted_request(tmp_path, manager, source, **changes):
    """Build an explicit host request with the current trusted session environment."""
    scratch = manager.temporary_directory
    return request(
        tmp_path,
        source=source,
        scratch_root=scratch,
        environment={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C",
            "WORKSPACE_ROOT": str(tmp_path),
            "TMPDIR": str(scratch),
            "TMP": str(scratch),
            "TEMP": str(scratch),
            "XDG_CACHE_HOME": str(scratch / "cache"),
        },
        **changes,
    )


def coordinator(
    tmp_path, outcome, *, answer="approve", auto_approval=False, defaults=None, headless=False
):
    """Build the application boundary with observable, inert launch collaborators."""
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = answer
    interaction.supports_fresh_host_approval = True
    backend = Mock()
    backend.run.return_value = outcome
    host = Mock()
    host.run_host_command.return_value = "host-result"
    configuration = PermissionConfiguration(limits=PolicyLimits(allow_host_processes=True))
    if defaults:
        configuration.defaults.update(defaults)
    manager = PermissionManager(tmp_path, configuration=configuration)
    return (
        CommandExecutionCoordinator(
            backend, host, manager, None if headless else interaction, auto_approval=auto_approval
        ),
        backend,
        host,
        interaction,
    )


@pytest.fixture
def process_exit_observer(monkeypatch):
    """Report a mocked POSIX child as exited without reaping it."""
    import loop.utils.process as process_utils

    waitid = Mock(side_effect=lambda _idtype, pid, _options: SimpleNamespace(si_pid=pid))
    boundary = SimpleNamespace(
        name="posix",
        P_PID=1,
        WEXITED=1,
        WNOHANG=2,
        WNOWAIT=4,
        close=process_utils.os.close,
        waitid=waitid,
    )
    monkeypatch.setattr(process_utils, "os", boundary)
    return boundary


def test_classified_failure_only_offers_host_retry(tmp_path):
    """A classified failure offers a concise host retry without leaking bound details."""
    service, backend, host, interaction = coordinator(
        tmp_path,
        CommandProcessResult(
            SandboxOutcome.UNAVAILABLE,
            detail=f"nested Seatbelt at {tmp_path}/private",
        ),
    )
    approved = request(tmp_path)

    attempt = service.run_sandboxed(approved, "run_command")

    backend.run.assert_called_once_with(approved)
    host.run_host_command.assert_not_called()
    assert attempt.offer is not None
    assert attempt.offer.warning == (
        'Command: "printf ok"\n'
        "Reason: The sandbox was unavailable. Native sandbox execution could not start. "
        "Running without the OS sandbox can read "
        "private files, change files outside the workspace, and use the network."
    )
    assert str(tmp_path) not in attempt.offer.warning
    assert "PATH=" not in attempt.offer.warning
    assert interaction.prompt.call_count == 1


def test_host_retry_refuses_automatic_interaction_after_sandbox_failure(tmp_path):
    """An automatic answer cannot turn a sandbox failure into a host launch."""
    service, _backend, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    )
    approved = request(tmp_path)
    attempt = service.run_sandboxed(approved, "run_command")
    assert attempt.offer is not None
    interaction.prompt.reset_mock()
    interaction.supports_fresh_host_approval = False

    assert service.retry_host_command(attempt.offer, approved) is None
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


def test_prelaunch_host_offer_displays_executed_source(tmp_path):
    """A host recovery offer identifies a different shell source before any approval."""
    service, backend, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    )
    approved = request(tmp_path, source="printf shown", execution_source="printf executed")

    attempt = service.run_sandboxed(approved, "run_command")

    assert attempt.offer is not None
    assert 'Executed command: "printf executed"' in attempt.offer.warning
    backend.run.assert_called_once_with(approved)
    host.run_host_command.assert_not_called()
    interaction.prompt.assert_called_once()


@pytest.mark.parametrize("outcome", [SandboxOutcome.UNAVAILABLE, SandboxOutcome.DENIED])
def test_possibly_effective_failure_never_offers_host_retry(tmp_path, outcome):
    """A failure that may have changed state cannot issue an automatic host offer."""
    failure = CommandProcessResult(outcome, detail="sandbox stopped", possible_effects=True)
    service, backend, host, interaction = coordinator(tmp_path, failure)
    approved = request(tmp_path)
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)

    try:
        attempt = service.run_sandboxed(approved, "run_command")
        assert telemetry.close(1)
    finally:
        set_telemetry(None)

    assert attempt.result == failure
    assert attempt.offer is None
    backend.run.assert_called_once_with(approved)
    host.run_host_command.assert_not_called()
    assert all(
        call.args[0] != "Run this exact command without the OS sandbox?"
        for call in interaction.prompt.call_args_list
    )
    assert not any(item.event_name == "host_command.offered" for item in adapter.records)


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_deadlines_never_reach_execution(tmp_path, deadline):
    """Directly replaced deadlines and explicit timeout values cannot start either path."""
    failure = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    service, backend, host, _ = coordinator(tmp_path, failure)
    approved = request(tmp_path)
    with pytest.raises(ValueError, match="must be finite"):
        service.run_sandboxed(replace(approved, deadline=deadline), "run_command")
    with pytest.raises(ValueError, match="must be finite"):
        service.run_sandboxed(approved, "run_command", execution_timeout=deadline)
    backend.run.assert_not_called()
    offer = service.run_sandboxed(approved, "run_command").offer
    with pytest.raises(ValueError, match="must be finite"):
        service.retry_host_command(offer, replace(approved, deadline=deadline))
    with pytest.raises(ValueError, match="must be finite"):
        service.retry_host_command(offer, approved, execution_timeout=deadline)
    host.run_host_command.assert_not_called()


def test_overflowed_renewed_deadline_never_reaches_execution(tmp_path, monkeypatch):
    """Finite inputs whose sum overflows are rejected before either execution path."""
    failure = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    service, backend, host, _ = coordinator(tmp_path, failure)
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer
    backend.run.reset_mock()
    monkeypatch.setattr("loop.execution.coordinator.time.monotonic", lambda: 1e308)
    with pytest.raises(ValueError, match="Renewed command deadline"):
        service.run_sandboxed(approved, "run_command", execution_timeout=1e308)
    with pytest.raises(ValueError, match="Renewed command deadline"):
        service.retry_host_command(offer, approved, execution_timeout=1e308)
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()


def test_sandbox_authorization_receives_model_explanation(tmp_path):
    """The coordinator forwards the explanation into the sandbox permission prompt."""
    outcome = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    service, backend, host, interaction = coordinator(tmp_path, outcome)
    approved = request(tmp_path)

    attempt = service.run_sandboxed(approved, "run_command", reason="Read the toolchain")

    assert attempt.result == outcome
    assert 'Agent explanation: "Read the toolchain"' in interaction.info.call_args.args[0]
    backend.run.assert_called_once_with(approved)
    host.run_host_command.assert_not_called()


def test_sandbox_denied_during_approval_never_starts_backend(tmp_path):
    """A denial installed while the sandbox prompt is pending prevents command execution."""
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    backend = Mock()
    host = Mock()
    manager = PermissionManager(tmp_path, interaction=interaction)
    service = CommandExecutionCoordinator(backend, host, manager, interaction)

    def deny_while_prompting(*_args, **_kwargs):
        """Install a sandbox process denial before the prior prompt is answered."""
        manager.add_rule(
            PermissionRule(
                decision=Decision.DENY, tool="run_command", action=Action.PROCESS_EXECUTE
            ),
            scope=PolicyScope.SESSION,
        )
        return "approve"

    interaction.prompt.side_effect = deny_while_prompting
    attempt = service.run_sandboxed(request(tmp_path), "run_command")

    assert attempt.result is None and attempt.offer is None
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()


def test_session_host_approval_is_reused_and_revocable(tmp_path):
    """An exact prelaunch approval is reusable, but cannot retry possible effects."""
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    host_answers = iter(("session", "deny"))

    def answer(prompt, **_kwargs):
        """Approve sandbox prompts and select recorded host decisions."""
        return next(host_answers) if prompt.startswith("Run this exact command") else "approve"

    interaction.prompt.side_effect = answer
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="no profile")
    host = Mock()
    host.run_host_command.return_value = "host-result"
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    first = request(tmp_path)
    offer = service.run_sandboxed(first, "renamed_command").offer
    assert service.retry_host_command(offer, first) == "host-result"
    third = request(tmp_path)
    offer = service.run_sandboxed(third, "renamed_command").offer
    assert service.retry_host_command(offer, third) == "host-result"
    assert (
        sum(
            call.args[0].startswith("Run this exact command")
            for call in interaction.prompt.call_args_list
        )
        == 1
    )
    rules = manager.host_command_rules()
    assert len(rules) == 1
    assert rules[0].source == "printf ok"
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.UNAVAILABLE, possible_effects=True
    )
    blocked = service.run_sandboxed(request(tmp_path), "renamed_command")
    assert blocked.offer is None
    assert host.run_host_command.call_count == 2
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="no profile")
    assert manager.remove_host_command_rule(rules[0].id)
    assert not manager.remove_host_command_rule(rules[0].id)
    fourth = request(tmp_path)
    offer = service.run_sandboxed(fourth, "renamed_command").offer
    assert service.retry_host_command(offer, fourth) is None
    assert host.run_host_command.call_count == 2


def test_session_host_approval_requires_matching_runtime_signature(tmp_path, monkeypatch):
    """A changed runtime authority requires a new host decision."""
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    host_answers = iter(("session", "deny"))
    interaction.prompt.side_effect = lambda prompt, **_kwargs: (
        next(host_answers) if prompt.startswith("Run this exact command") else "approve"
    )
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    host = Mock()
    host.run_host_command.return_value = "host-result"
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    first = request(tmp_path)
    second = request(tmp_path)
    monkeypatch.setattr(
        SandboxRequest,
        "host_command_runtime_signature",
        lambda item: "first" if item.attempt_id == first.attempt_id else "second",
    )
    offer = service.run_sandboxed(first, "run_command").offer
    assert service.retry_host_command(offer, first) == "host-result"

    offer = service.run_sandboxed(second, "run_command").offer

    assert service.retry_host_command(offer, second) is None
    assert host.run_host_command.call_count == 1
    assert (
        sum(
            call.args[0].startswith("Run this exact command")
            for call in interaction.prompt.call_args_list
        )
        == 2
    )


def test_denied_initial_permission_starts_nothing_and_offers_nothing(tmp_path):
    """A user rejection cannot be reclassified as a sandbox inability."""
    service, backend, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.UNAVAILABLE), answer="deny"
    )

    attempt = service.run_sandboxed(
        request(tmp_path),
        "run_command",
        prelaunch_failure=CommandProcessResult(
            SandboxOutcome.UNAVAILABLE, detail="startup probe failed"
        ),
    )

    assert attempt.result is None and attempt.offer is None
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()
    interaction.prompt.assert_called_once()


def test_policy_denial_and_headless_mode_start_nothing(tmp_path):
    """Explicit policy denial and absence of user interaction fail closed."""
    service, backend, host, interaction = coordinator(
        tmp_path,
        CommandProcessResult(SandboxOutcome.UNAVAILABLE),
        defaults={Action.PROCESS_EXECUTE: Decision.DENY},
    )
    assert (
        service.run_sandboxed(
            request(tmp_path),
            "run_command",
            prelaunch_failure=CommandProcessResult(
                SandboxOutcome.UNAVAILABLE, detail="startup probe failed"
            ),
        ).offer
        is None
    )
    interaction.prompt.assert_not_called()
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()


def test_nonzero_exit_timeout_and_cancel_do_not_offer_host(tmp_path):
    """Ordinary command failures are never treated as OS sandbox denial."""
    for result in (
        CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=17),
        CommandProcessResult(SandboxOutcome.TIMED_OUT),
        CommandProcessResult(SandboxOutcome.CANCELLED),
    ):
        service, _, host, _ = coordinator(tmp_path, result)
        attempt = service.run_sandboxed(request(tmp_path), "run_command")
        assert attempt.result == result and attempt.offer is None
        host.run_host_command.assert_not_called()


def test_explicit_host_request_requires_fresh_exact_approval(tmp_path):
    """Explicit host requests show the bound command and require a fresh one-time choice."""
    tmp_path = tmp_path / "workspace é;\nnext line"
    tmp_path.mkdir()
    service, backend, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    )
    approved = request(tmp_path, source="printf host")

    result = service.request_host_command(approved, "run_command", "Need an external service")

    assert result == "host-result"
    backend.run.assert_not_called()
    host.run_host_command.assert_called_once()
    warning = interaction.info.call_args.args[0]
    assert 'Command: "printf host"' in warning
    directory_line = next(
        line for line in warning.splitlines() if line.startswith("Working directory: ")
    )
    assert json.loads(directory_line.removeprefix("Working directory: ")) == str(tmp_path)
    assert 'Reason: "Need an external service"' in warning
    assert "may repeat" in warning
    assert interaction.prompt.call_args.kwargs["choices"] == {
        "deny": "Deny",
        "approve": "Approve this one host run",
    }


@pytest.mark.parametrize("headless", [False, True])
def test_yolo_explicit_host_request_needs_no_gate_or_prompt(tmp_path, headless):
    """An exact host request runs once under YOLO without persisting eligibility or a grant."""
    manager = PermissionManager(tmp_path, yolo=True)
    interaction = None if headless else Mock(spec=Interaction)
    host = Mock()
    host.run_host_command.return_value = "host-result"
    service = CommandExecutionCoordinator(Mock(), host, manager, interaction, auto_approval=True)
    current = request(tmp_path, source="printf host")

    assert (
        service.request_host_command(current, "request_host_command", "explicit host")
        == "host-result"
    )
    host.run_host_command.assert_called_once()
    assert manager.configuration.limits.allow_host_processes is False
    assert manager.host_command_rules() == ()
    if interaction is not None:
        interaction.prompt.assert_not_called()
        assert 'Command: "printf host"' in interaction.warning.call_args.args[0]

    manager.set_yolo(False)
    assert service.request_host_command(current, "request_host_command", "explicit host") is None
    host.run_host_command.assert_called_once()


def test_yolo_rejects_implicit_retry_and_revoked_prelaunch_host_request(tmp_path):
    """A failed sandbox attempt cannot become host execution and revocation stops admission."""
    manager = PermissionManager(tmp_path, yolo=True)
    interaction = Mock(spec=Interaction)
    host = Mock()
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    current = request(tmp_path)
    offer = service.run_sandboxed(current, "run_command").offer
    assert offer is not None
    assert service.retry_host_command(offer, current) is None
    host.run_host_command.assert_not_called()
    interaction.prompt.assert_not_called()

    interaction.warning.side_effect = lambda _message: manager.set_yolo(False)
    assert service.request_host_command(current, "request_host_command", "explicit host") is None
    host.run_host_command.assert_not_called()


def test_yolo_does_not_override_explicit_host_deny(tmp_path):
    """A user hard deny wins before an otherwise eligible YOLO host decision."""
    manager = PermissionManager(tmp_path, yolo=True)
    manager.add_rule(
        PermissionRule(
            decision=Decision.DENY, tool="request_host_command", action=Action.PROCESS_EXECUTE
        ),
        scope=PolicyScope.SESSION,
    )
    host = Mock()
    service = CommandExecutionCoordinator(Mock(), host, manager, None)
    assert (
        service.request_host_command(request(tmp_path), "request_host_command", "explicit host")
        is None
    )
    host.run_host_command.assert_not_called()


def test_explicit_host_request_uses_exact_human_session_grant_headlessly(tmp_path):
    """An exact human grant runs a requested host command without a new interaction."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    grant = manager.trust_host_command("printf host")
    backend = Mock()
    host = Mock()
    host.run_host_command.return_value = "host-result"
    service = CommandExecutionCoordinator(backend, host, manager, None)

    assert (
        service.request_host_command(
            trusted_request(tmp_path, manager, "printf host"), "request_host_command", "Needed"
        )
        == "host-result"
    )
    backend.run.assert_not_called()
    host.run_host_command.assert_called_once()
    assert (
        service.request_host_command(
            trusted_request(tmp_path, manager, "printf other"), "request_host_command", "Needed"
        )
        is None
    )
    assert manager.remove_trusted_host_command_rule(grant.id)
    assert (
        service.request_host_command(
            trusted_request(tmp_path, manager, "printf host"), "request_host_command", "Needed"
        )
        is None
    )
    manager.close()


@pytest.mark.parametrize(
    ("answer", "scope"),
    [
        ("trust_session", ApprovalChoice.SESSION),
        ("trust_workspace", ApprovalChoice.WORKSPACE),
        ("trust_user", ApprovalChoice.USER),
    ],
)
def test_explicit_host_prompt_installs_selected_grant_in_same_decision(tmp_path, answer, scope):
    """One fresh host prompt can run now and remember the exact chosen grant scope."""
    if scope is ApprovalChoice.SESSION:
        manager = PermissionManager(tmp_path)
    else:
        catalog = tmp_path.parent / f"{tmp_path.name}-policy-data" / "workspaces.sqlite"
        workspace = WorkspaceRepository(catalog).initialize(Workspace.discover(tmp_path))
        manager = PermissionManager(
            tmp_path,
            configuration_path=catalog.parent / "workspaces" / workspace.id / "permissions.yaml",
            user_configuration_path=catalog.parent / "user-permissions.yaml",
            workspace_id=workspace.id,
            namespace_catalog_path=catalog,
            user_namespace_identity=str(catalog.parent),
        )
        manager.initialize_policy_namespace("workspace", workspace.id, authorized=True)
        manager.initialize_policy_namespace("user", str(catalog.parent), authorized=True)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = answer
    host = Mock()
    host.run_host_command.return_value = "host-result"
    service = CommandExecutionCoordinator(Mock(), host, manager, interaction)

    assert (
        service.request_host_command(
            trusted_request(tmp_path, manager, "printf host"), "request_host_command", "Needed"
        )
        == "host-result"
    )
    assert answer in interaction.prompt.call_args.kwargs["choices"]
    assert interaction.prompt.call_count == 1
    rules = manager.trusted_host_command_rules()
    assert len(rules) == 1 and rules[0].scope is scope
    assert (
        service.request_host_command(
            trusted_request(tmp_path, manager, "printf host"), "request_host_command", "Needed"
        )
        == "host-result"
    )
    assert interaction.prompt.call_count == 1
    manager.close()


def test_explicit_host_prompt_refuses_stale_grant_installation(tmp_path, monkeypatch):
    """A grant that cannot commit at the reviewed revision never starts a host process."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "trust_session"
    host = Mock()
    service = CommandExecutionCoordinator(Mock(), host, manager, interaction)
    monkeypatch.setattr(
        manager, "remember_trusted_host_command_rule_if_current", lambda *args: None
    )

    assert (
        service.request_host_command(
            trusted_request(tmp_path, manager, "printf host"), "request_host_command", "Needed"
        )
        is None
    )
    host.run_host_command.assert_not_called()
    manager.close()


def test_explicit_host_prompt_rechecks_new_grant_before_launch(tmp_path, monkeypatch):
    """A grant lost after installation cannot authorize the pending host launch."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "trust_session"
    host = Mock()
    service = CommandExecutionCoordinator(Mock(), host, manager, interaction)
    monkeypatch.setattr(manager, "matching_trusted_host_command_rule", lambda *args, **kw: None)

    assert (
        service.request_host_command(
            trusted_request(tmp_path, manager, "printf host"), "request_host_command", "Needed"
        )
        is None
    )
    host.run_host_command.assert_not_called()
    manager.close()


def test_trusted_host_grant_shows_boundary_without_another_prompt(tmp_path):
    """A no-prompt grant still shows unrestricted authority before an interactive launch."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager.trust_host_command("printf host")
    host = Mock()
    host.run_host_command.return_value = "host-result"
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    service = CommandExecutionCoordinator(Mock(), host, manager, interaction)

    assert (
        service.request_host_command(
            trusted_request(tmp_path, manager, "printf host"), "request_host_command", "Needed"
        )
        == "host-result"
    )
    interaction.prompt.assert_not_called()
    assert "host user's authority" in interaction.info.call_args.args[0]
    manager.close()


def test_trusted_host_grant_never_authorizes_a_different_execution_source(tmp_path):
    """A trusted display command cannot launch different shell source headlessly."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager.trust_host_command("printf safe")
    host = Mock()
    service = CommandExecutionCoordinator(Mock(), host, manager, None)
    changed = trusted_request(tmp_path, manager, "printf safe", execution_source="printf different")

    assert service.request_host_command(changed, "request_host_command", "Needed") is None
    host.run_host_command.assert_not_called()
    manager.close()


def test_trusted_host_grant_rejects_changed_execution_environment(tmp_path):
    """A direct request with altered host environment cannot consume a trusted grant."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager.trust_host_command("printf safe")
    host = Mock()
    service = CommandExecutionCoordinator(Mock(), host, manager, None)
    expected = trusted_request(tmp_path, manager, "printf safe")
    environment = expected.shell_environment()
    environment["LANG"] = "C.UTF-8"
    changed = request(
        tmp_path,
        source="printf safe",
        scratch_root=manager.temporary_directory,
        environment=environment,
    )

    assert service.request_host_command(changed, "request_host_command", "Needed") is None
    host.run_host_command.assert_not_called()
    manager.close()


def test_trusted_host_boundary_writes_minimized_durable_audit(tmp_path):
    """No-prompt host approval and execution are durably audited without shell contents."""
    audit_path = tmp_path / "audit.sqlite"
    manager = PermissionManager(tmp_path, audit_path=audit_path, workspace_id="workspace")
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    grant = manager.trust_host_command("printf synthetic-secret")
    host = Mock()
    host.run_host_command.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="synthetic-secret"
    )
    service = CommandExecutionCoordinator(Mock(), host, manager, None)

    result = service.request_host_command(
        trusted_request(tmp_path, manager, "printf synthetic-secret"),
        "request_host_command",
        "Needed",
    )

    assert result is not None and result.outcome is SandboxOutcome.COMPLETED
    manager.remove_trusted_host_command_rule(grant.id)
    with closing(sqlite3.connect(audit_path)) as connection:
        rows = connection.execute(
            "SELECT event_name, payload_json FROM permission_audit_records "
            "WHERE event_name LIKE 'host_command.%' ORDER BY rowid"
        ).fetchall()
    assert [event for event, _ in rows] == [
        "host_command.trusted_rule_added",
        "host_command.offered",
        "host_command.approved",
        "host_command.executed",
        "host_command.trusted_rule_removed",
    ]
    assert all(
        "synthetic-secret" not in payload and str(tmp_path) not in payload for _, payload in rows
    )
    assert json.loads(rows[2][1])["source"] == "trusted"
    assert json.loads(rows[3][1])["outcome"] == "completed"
    manager.close()


def test_final_host_policy_check_rejects_late_deny(tmp_path, monkeypatch):
    """A policy change after human approval and before launch blocks the host shell."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    host = Mock()
    service = CommandExecutionCoordinator(Mock(), host, manager, interaction)
    original = manager.evaluate_with_revision
    evaluations = 0

    def deny_at_final_check(operations):
        """Install a host deny just before the final admission check."""
        nonlocal evaluations
        evaluations += 1
        if evaluations == 3:
            manager.set_limit("allow_host_processes", False, scope=PolicyScope.SESSION)
        return original(operations)

    monkeypatch.setattr(manager, "evaluate_with_revision", deny_at_final_check)
    assert service.request_host_command(request(tmp_path), "request_host_command", "Needed") is None
    assert evaluations == 3
    host.run_host_command.assert_not_called()
    manager.close()


def test_existing_prelaunch_host_rule_reuses_without_new_interaction(tmp_path):
    """A no-effects exact retry keeps its human session approval when input disappears."""
    failed = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    backend = Mock()
    backend.run.return_value = failed
    host = Mock()
    host.run_host_command.return_value = "host-result"
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "session"
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    human = CommandExecutionCoordinator(backend, host, manager, interaction)
    first = request(tmp_path)
    offer = human.run_sandboxed(first, "run_command").offer
    assert offer is not None
    assert human.retry_host_command(offer, first) == "host-result"

    headless = CommandExecutionCoordinator(backend, host, manager, None)
    again = request(tmp_path)
    offer = headless.run_sandboxed(again, "run_command").offer
    assert offer is not None
    assert headless.retry_host_command(offer, again) == "host-result"
    assert host.run_host_command.call_count == 2
    manager.close()


@pytest.mark.parametrize("mode", ["decline", "headless", "automatic", "cancelled", "untrusted"])
def test_explicit_host_request_fails_closed_without_fresh_approval(tmp_path, mode):
    """Decline, missing interaction, automatic approval, and cancellation cannot launch."""
    service, backend, host, interaction = coordinator(
        tmp_path,
        CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0),
        answer="deny" if mode == "decline" else "approve",
        auto_approval=mode == "automatic",
        headless=mode == "headless",
    )
    if mode == "untrusted":
        interaction.supports_fresh_host_approval = False
    cancelled = Event()
    if mode == "cancelled":
        cancelled.set()

    assert (
        service.request_host_command(
            request(tmp_path), "run_command", "Needed", cancellation_event=cancelled
        )
        is None
    )
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()
    if mode != "decline":
        interaction.prompt.assert_not_called()


def test_explicit_host_request_requires_reason(tmp_path):
    """An empty explanation cannot initiate an unrestricted host approval."""
    service, _, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    )
    with pytest.raises(ValueError, match="requires a reason"):
        service.request_host_command(request(tmp_path), "run_command", "  ")
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


def test_explicit_host_request_rechecks_policy_after_fresh_approval(tmp_path):
    """A changed host policy during the explicit prompt prevents the launch."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.supports_fresh_host_approval = True
    host = Mock()
    service = CommandExecutionCoordinator(Mock(), host, manager, interaction)

    def revoke(*_args, **_kwargs):
        """Remove host authority while the user reviews the request."""
        manager.set_limit("allow_host_processes", False, scope=PolicyScope.SESSION)
        return "approve"

    interaction.prompt.side_effect = revoke
    assert service.request_host_command(request(tmp_path), "run_command", "Needed") is None
    host.run_host_command.assert_not_called()


def test_explicit_host_request_cancellation_during_prompt_prevents_launch(tmp_path):
    """Cancellation received while the fresh decision is pending prevents host execution."""
    service, _, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    )
    cancelled = Event()

    def cancel(*_args, **_kwargs):
        """Cancel while the user is choosing."""
        cancelled.set()
        return "approve"

    interaction.prompt.side_effect = cancel
    assert (
        service.request_host_command(
            request(tmp_path), "run_command", "Needed", cancellation_event=cancelled
        )
        is None
    )
    host.run_host_command.assert_not_called()


def test_executable_lookup_exits_do_not_offer_host_for_incidental_denial(tmp_path):
    """Tool lookup failures stay sandboxed despite unrelated trusted OS denials."""
    for exit_code in (72, 126, 127):
        result = CommandProcessResult(
            SandboxOutcome.COMPLETED,
            exit_code=exit_code,
            observed_denial="file-read-data /Library/Preferences/unrelated.plist",
        )
        service, _, host, _ = coordinator(tmp_path, result)
        attempt = service.run_sandboxed(request(tmp_path), "run_command")
        assert attempt.result == result and attempt.offer is None
        host.run_host_command.assert_not_called()


def test_observed_denial_keeps_exit_without_offering_host(tmp_path):
    """A completed shell's observed denial never initiates a host retry offer."""
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "session"
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=7,
        stderr="Operation not permitted: /outside/secret",
        observed_denial="file-read-data /outside/secret",
    )
    host = Mock()
    host.run_host_command.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    approved = request(tmp_path)
    service = CommandExecutionCoordinator(backend, host, manager, interaction)

    attempt = service.run_sandboxed(approved, "run_command")

    assert attempt.result.exit_code == 7
    assert attempt.result.observed_denial == "file-read-data /outside/secret"
    assert attempt.offer is None
    interaction.prompt.assert_called_once()
    assert interaction.prompt.call_args.args[0] == "Allow this command?"
    host.run_host_command.assert_not_called()


def test_observed_denial_cannot_use_automatic_host_approval(tmp_path):
    """Automatic approval cannot launch an uncertain post-launch host retry."""
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=1,
        stderr="connect to 127.0.0.1:443 failed: Operation not permitted",
        observed_denial="network-outbound remote:*:443",
    )
    service, _, host, interaction = coordinator(tmp_path, result, auto_approval=True)
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer

    assert offer is None
    host.run_host_command.assert_not_called()
    interaction.prompt.assert_called_once()
    assert interaction.prompt.call_args.args[0] == "Allow this command?"


@pytest.mark.parametrize(
    ("stderr", "observed_denial"),
    (
        (
            "Permission denied: unrelated operation",
            "network-outbound remote:*:443",
        ),
        (
            "connect failed: Operation not permitted",
            "network-outbound remote:*:443",
        ),
        (
            "connect to 127.0.0.1:444 failed: Operation not permitted",
            "network-outbound remote:*:443",
        ),
        (
            "Operation not permitted: unrelated operation\nconnect to 127.0.0.1:443 failed",
            "network-outbound remote:*:443",
        ),
        (
            "connect to 127.0.0.1:443 failed: Operation not permitted",
            "network-outbound remote:*:*",
        ),
    ),
)
def test_network_denial_requires_correlated_child_failure(tmp_path, stderr, observed_denial):
    """Unrelated or unscoped permission text cannot justify a network host offer."""
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=1,
        stderr=stderr,
        observed_denial=observed_denial,
    )
    service, _, host, interaction = coordinator(tmp_path, result)

    attempt = service.run_sandboxed(request(tmp_path), "run_command")

    assert attempt.result == result
    assert attempt.offer is None
    interaction.prompt.reset_mock()
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


def test_unix_socket_denial_matches_the_reported_connection_path(tmp_path):
    """A Unix socket denial remains diagnostic and cannot create a host offer."""
    target = tmp_path.parent / "service.sock"
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=1,
        stderr=(
            f'  File "<string>", line 1, in <module>\n    s.connect("{target}")\n'
            "PermissionError: [Errno 1] Operation not permitted\n"
        ),
        observed_denial=f"network-outbound {target}",
    )
    service, _, host, _ = coordinator(tmp_path, result)

    attempt = service.run_sandboxed(request(tmp_path), "run_command")

    assert attempt.result == result
    assert attempt.offer is None
    host.run_host_command.assert_not_called()


def test_unrelated_denial_never_offers_host_execution(tmp_path):
    """A tagged read denial cannot turn an unrelated tool error into a host prompt."""
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=2,
        stdout="error: invalid command option",
        observed_denial="Sandbox: uv(100) deny(1) file-read-data /opt/homebrew/etc/openssl.cnf",
    )
    service, _, host, interaction = coordinator(tmp_path, result)

    attempt = service.run_sandboxed(request(tmp_path, read_roots=()), "run_command")

    assert attempt.result == result
    assert attempt.offer is None
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


def test_read_only_workspace_write_denial_stays_in_sandbox(tmp_path):
    """A requested write restriction cannot be escaped through host retry."""
    target = tmp_path / ".coverage"
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=1,
        stderr=f"PermissionError: [Errno 1] Operation not permitted: '{target}'",
        observed_denial=f"Sandbox: Python(100) deny(1) file-write-unlink {target}",
    )
    service, _, host, interaction = coordinator(tmp_path, result)

    attempt = service.run_sandboxed(request(tmp_path, read_roots=(), write_roots=()), "run_command")

    assert attempt.result == result
    assert attempt.offer is None
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


def test_host_retry_decline_and_replay_start_none(tmp_path):
    """A declined offer is consumed and cannot later authorize host execution."""
    service, _, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.DENIED, detail="policy denial")
    )
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer
    interaction.prompt.return_value = "deny"

    assert service.retry_host_command(offer, approved) is None
    interaction.prompt.return_value = "approve"
    assert service.retry_host_command(offer, approved) is None
    host.run_host_command.assert_not_called()


def test_host_command_offer_preserves_default_tool_identity(tmp_path):
    """Directly constructed offers retain the default execution tool identity."""
    offer = HostCommandOffer(
        "token", request(tmp_path), CommandProcessResult(SandboxOutcome.DENIED), ""
    )

    assert offer.tool_id == "run_command"


def test_host_retry_respects_default_host_process_limit(tmp_path):
    """The default host-process boundary denies a retry without prompting for approval."""
    manager = PermissionManager(tmp_path)
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    host = Mock()
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    approved = request(tmp_path)
    attempt = service.run_sandboxed(approved, "renamed_command")
    interaction.prompt.reset_mock()

    assert attempt.offer is not None
    assert service.retry_host_command(attempt.offer, approved) is None
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize("denial", ["limit", "host rule"])
@pytest.mark.parametrize("when", ["before", "during"])
def test_host_retry_rechecks_policy_before_and_after_approval(tmp_path, denial, when):
    """A host boundary or exact deny rule cannot be overridden by a retry approval."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    host = Mock()
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    approved = request(tmp_path)
    attempt = service.run_sandboxed(approved, "renamed_command")
    interaction.prompt.reset_mock()

    def deny_host():
        """Remove host authority without denying the preceding sandbox operation."""
        if denial == "limit":
            manager.set_limit("allow_host_processes", False, scope=PolicyScope.SESSION)
        else:
            manager.add_rule(
                PermissionRule(
                    decision=Decision.DENY,
                    tool="renamed_command",
                    tool_exact=True,
                    action=Action.PROCESS_EXECUTE,
                    target=ProcessTarget(
                        argv=("/bin/sh", "-c", approved.execution_source),
                        cwd=str(approved.cwd),
                        boundary=ProcessBoundary.HOST,
                    ),
                ),
                scope=PolicyScope.SESSION,
            )

    if when == "before":
        deny_host()
    else:

        def answer(*_args, **_kwargs):
            """Change policy while the user reviews the host retry."""
            deny_host()
            return "approve"

        interaction.prompt.side_effect = answer

    assert attempt.offer is not None
    assert service.retry_host_command(attempt.offer, approved) is None
    assert interaction.prompt.call_count == (0 if when == "before" else 1)
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize("deny_logical", [True, False])
def test_host_retry_checks_displayed_and_executed_sources(tmp_path, deny_logical):
    """A host deny for either displayed or executed source prevents launch."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    host = Mock()
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    approved = request(
        tmp_path,
        source="cat $WORKSPACE_ROOT/notes",
        execution_source="printf other-source",
    )
    attempt = service.run_sandboxed(approved, "renamed_command")
    manager.add_rule(
        PermissionRule(
            decision=Decision.DENY,
            tool="renamed_command",
            action=Action.PROCESS_EXECUTE,
            target=ProcessTarget(
                argv=(
                    "/bin/sh",
                    "-c",
                    approved.source if deny_logical else approved.execution_source,
                ),
                cwd=str(approved.cwd),
                boundary=ProcessBoundary.HOST,
            ),
        ),
        scope=PolicyScope.SESSION,
    )
    interaction.prompt.reset_mock()

    assert attempt.offer is not None
    assert service.retry_host_command(attempt.offer, approved) is None
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


def test_explicit_host_ask_requires_a_fresh_decision_despite_session_rule(tmp_path):
    """A new HOST ask rule prevents silent reuse of an existing exact host approval."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    host = Mock()
    host.run_host_command.return_value = "host-result"
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    approved = request(tmp_path)
    first = service.run_sandboxed(approved, "renamed_command")
    interaction.prompt.return_value = "session"
    assert service.retry_host_command(first.offer, approved) == "host-result"
    assert len(manager.host_command_rules()) == 1
    manager.add_rule(
        PermissionRule(
            decision=Decision.ASK,
            tool="renamed_command",
            action=Action.PROCESS_EXECUTE,
            target=ProcessTarget(
                argv=("/bin/sh", "-c", approved.execution_source),
                cwd=str(approved.cwd),
                boundary=ProcessBoundary.HOST,
            ),
        ),
        scope=PolicyScope.SESSION,
    )
    interaction.prompt.return_value = "approve"
    second = service.run_sandboxed(approved, "renamed_command")
    interaction.prompt.reset_mock()
    interaction.prompt.return_value = "deny"

    assert service.retry_host_command(second.offer, approved) is None
    interaction.prompt.assert_called_once()
    host.run_host_command.assert_called_once()

    interaction.prompt.return_value = "approve"
    third = service.run_sandboxed(approved, "renamed_command")
    interaction.prompt.reset_mock()
    assert service.retry_host_command(third.offer, approved) == "host-result"
    interaction.prompt.assert_called_once()
    assert host.run_host_command.call_count == 2


def test_host_ask_added_during_approval_invalidates_decision(tmp_path):
    """A newly added HOST ask rule requires a decision after that rule takes effect."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    host = Mock()
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    approved = request(tmp_path)
    attempt = service.run_sandboxed(approved, "renamed_command")

    def add_ask(*_args, **_kwargs):
        """Install a new exact host ask rule while consent is pending."""
        manager.add_rule(
            PermissionRule(
                decision=Decision.ASK,
                tool="renamed_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", approved.execution_source),
                    cwd=str(approved.cwd),
                    boundary=ProcessBoundary.HOST,
                ),
            ),
            scope=PolicyScope.SESSION,
        )
        return "approve"

    interaction.prompt.side_effect = add_ask

    assert service.retry_host_command(attempt.offer, approved) is None
    host.run_host_command.assert_not_called()


def test_host_retry_rejects_replaced_ask_with_same_rule_id(tmp_path):
    """A replacement host ask cannot reuse consent given for its predecessor."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    host = Mock()
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    approved = request(tmp_path)
    attempt = service.run_sandboxed(approved, "renamed_command")
    ask = PermissionRule(
        decision=Decision.ASK,
        tool="renamed_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", approved.execution_source),
            cwd=str(approved.cwd),
            boundary=ProcessBoundary.HOST,
        ),
    )
    manager.add_rule(ask, scope=PolicyScope.SESSION)

    def replace_ask(*_args, **_kwargs):
        """Replace the reviewed rule without changing its identifier."""
        assert manager.remove_rule(ask.id, scope=PolicyScope.SESSION)
        manager.add_rule(ask.model_copy(update={"tool": "*"}), scope=PolicyScope.SESSION)
        return "approve"

    interaction.prompt.side_effect = replace_ask

    assert attempt.offer is not None
    assert service.retry_host_command(attempt.offer, approved) is None
    host.run_host_command.assert_not_called()
    assert not manager.host_command_rules()


def test_explicit_host_ask_offers_only_one_off_approval(tmp_path):
    """An explicit host ask cannot create an ineffective session approval."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    host = Mock()
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    approved = request(tmp_path)
    attempt = service.run_sandboxed(approved, "renamed_command")
    manager.add_rule(
        PermissionRule(
            decision=Decision.ASK,
            tool="renamed_command",
            action=Action.PROCESS_EXECUTE,
            target=ProcessTarget(
                argv=("/bin/sh", "-c", approved.execution_source),
                cwd=str(approved.cwd),
                boundary=ProcessBoundary.HOST,
            ),
        ),
        scope=PolicyScope.SESSION,
    )
    interaction.prompt.reset_mock()
    interaction.prompt.return_value = "session"

    assert attempt.offer is not None
    assert service.retry_host_command(attempt.offer, approved) is None
    assert set(interaction.prompt.call_args.kwargs["choices"]) == {"approve", "deny"}
    assert not manager.host_command_rules()
    host.run_host_command.assert_not_called()


def test_host_retry_rechecks_policy_after_final_rule_lookup(tmp_path, monkeypatch):
    """A denial published during the last remembered-rule lookup prevents launch."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    host = Mock()
    host.run_host_command.return_value = "host-result"
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    approved = request(tmp_path)
    first = service.run_sandboxed(approved, "renamed_command")
    interaction.prompt.return_value = "session"
    assert service.retry_host_command(first.offer, approved) == "host-result"
    interaction.prompt.return_value = "approve"
    second = service.run_sandboxed(approved, "renamed_command")
    original = manager.matching_host_command_rule
    lookups = 0

    def deny_on_recheck(value):
        """Publish a denial only after the first match has been accepted."""
        nonlocal lookups
        result = original(value)
        lookups += 1
        if lookups == 2:
            manager.add_rule(
                PermissionRule(
                    decision=Decision.DENY,
                    tool="renamed_command",
                    action=Action.PROCESS_EXECUTE,
                    target=ProcessTarget(
                        argv=("/bin/sh", "-c", approved.execution_source),
                        cwd=str(approved.cwd),
                        boundary=ProcessBoundary.HOST,
                    ),
                ),
                scope=PolicyScope.SESSION,
            )
        return result

    monkeypatch.setattr(manager, "matching_host_command_rule", deny_on_recheck)

    assert second.offer is not None
    assert service.retry_host_command(second.offer, approved) is None
    assert lookups == 2
    host.run_host_command.assert_called_once()


def test_host_retry_rejects_policy_change_during_rule_preparation(tmp_path, monkeypatch):
    """Policy changes before a session host rule is stored cannot authorize launch."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    host = Mock()
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    approved = request(tmp_path)
    attempt = service.run_sandboxed(approved, "renamed_command")
    interaction.prompt.return_value = "session"
    original = SandboxRequest.host_command_runtime_signature
    changed = False

    def revoke_during_signature(value):
        """Remove host authority while the remembered binding is computed."""
        nonlocal changed
        signature = original(value)
        if not changed:
            changed = True
            manager.set_limit("allow_host_processes", False, scope=PolicyScope.SESSION)
        return signature

    monkeypatch.setattr(SandboxRequest, "host_command_runtime_signature", revoke_during_signature)

    assert attempt.offer is not None
    assert service.retry_host_command(attempt.offer, approved) is None
    assert changed
    host.run_host_command.assert_not_called()
    assert not manager.host_command_rules()


def test_host_ask_added_during_rule_lookup_prevents_silent_reuse(tmp_path, monkeypatch):
    """A new HOST ask rule invalidates a previously matched session approval."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    host = Mock()
    host.run_host_command.return_value = "host-result"
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "approve"
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    approved = request(tmp_path)
    first = service.run_sandboxed(approved, "renamed_command")
    interaction.prompt.return_value = "session"
    assert service.retry_host_command(first.offer, approved) == "host-result"
    interaction.prompt.return_value = "approve"
    second = service.run_sandboxed(approved, "renamed_command")
    interaction.prompt.reset_mock()
    original = manager.matching_host_command_rule

    def add_ask(value):
        """Add a HOST ask immediately after the old session rule is matched."""
        matched = original(value)
        manager.add_rule(
            PermissionRule(
                decision=Decision.ASK,
                tool="renamed_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", approved.execution_source),
                    cwd=str(approved.cwd),
                    boundary=ProcessBoundary.HOST,
                ),
            ),
            scope=PolicyScope.SESSION,
        )
        return matched

    monkeypatch.setattr(manager, "matching_host_command_rule", add_ask)

    assert service.retry_host_command(second.offer, approved) is None
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_called_once()


def test_host_retry_requires_fresh_one_off_approval_and_runs_once(tmp_path):
    """Only the explicit host prompt launches one audited, exact host request."""
    service, _, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.DENIED, detail="policy denial")
    )
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer

    assert service.retry_host_command(offer, approved) == "host-result"
    assert "Reason: The sandbox denied this command." in interaction.info.call_args.args[0]
    host.run_host_command.assert_called_once_with(HostCommandRequest.from_sandbox(approved))
    assert service.retry_host_command(offer, approved) is None
    host.run_host_command.assert_called_once()


def test_coordinator_passes_run_cancellation_to_both_process_boundaries(tmp_path):
    """Cancellation reaches the sandbox attempt and any separately approved host retry."""
    cancellation_event = Event()
    denied = CommandProcessResult(SandboxOutcome.DENIED, detail="policy denial")
    service, backend, host, _interaction = coordinator(tmp_path, denied)
    approved = request(tmp_path)

    attempt = service.run_sandboxed(
        approved,
        "run_command",
        cancellation_event=cancellation_event,
    )
    assert attempt.offer is not None
    backend.run.assert_called_once_with(approved, cancellation_event=cancellation_event)
    assert (
        service.retry_host_command(
            attempt.offer,
            approved,
            cancellation_event=cancellation_event,
        )
        == "host-result"
    )
    host.run_host_command.assert_called_once_with(
        HostCommandRequest.from_sandbox(approved),
        cancellation_event=cancellation_event,
    )


def test_coordinator_forwards_terminal_input_to_original_backend(tmp_path):
    """One admitted request keeps the transient terminal source on its native run."""
    service, backend, _host, _interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    )
    approved = request(tmp_path)
    terminal_input = Mock(return_value=None)

    attempt = service.run_sandboxed(approved, "run_command", terminal_input=terminal_input)

    assert attempt.result is not None and attempt.result.outcome is SandboxOutcome.COMPLETED
    backend.run.assert_called_once_with(approved, terminal_input=terminal_input)


def test_coordinator_cancellation_before_sandbox_launch_has_no_effects(tmp_path):
    """A cancellation arriving after approval but before launch starts no sandbox process."""
    cancellation_event = Event()
    cancellation_event.set()
    service, backend, _host, _interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    )

    attempt = service.run_sandboxed(
        request(tmp_path),
        "run_command",
        cancellation_event=cancellation_event,
    )

    assert attempt.result is not None
    assert attempt.result.outcome is SandboxOutcome.CANCELLED
    assert attempt.result.possible_effects is False
    assert attempt.offer is None
    backend.run.assert_not_called()


@pytest.mark.parametrize("during_approval", [False, True])
def test_detached_admission_obeys_original_run_stop_before_handoff(
    tmp_path, monkeypatch, during_approval
):
    """A stopped run cannot launch a detached command before job admission completes."""
    origin = Event()
    job = Event()
    service, backend, _host, _interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    )
    admitted = Mock()
    authorization = Mock(return_value=True)
    if during_approval:
        authorization.side_effect = lambda *_args, **_kwargs: origin.set() or True
    else:
        origin.set()
    monkeypatch.setattr(service._permissions, "authorize_sandboxed_command", authorization)

    attempt = service.run_sandboxed(
        request(tmp_path),
        "run_command",
        cancellation_event=job,
        admission_cancellation_event=origin,
        on_admitted=admitted,
    )

    assert attempt.result.outcome is SandboxOutcome.CANCELLED
    assert not attempt.result.possible_effects
    assert authorization.call_count == (1 if during_approval else 0)
    admitted.assert_not_called()
    backend.run.assert_not_called()


def test_cancelled_host_retry_never_opens_the_approval_prompt(tmp_path):
    """A host retry cancelled before input cannot consume or exercise an old offer."""
    service, _backend, host, interaction = coordinator(
        tmp_path,
        CommandProcessResult(SandboxOutcome.DENIED, detail="policy denial"),
    )
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer
    interaction.prompt.reset_mock()
    cancellation_event = Event()
    cancellation_event.set()

    assert (
        service.retry_host_command(
            offer,
            approved,
            cancellation_event=cancellation_event,
        )
        is None
    )
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


def test_host_retry_cancelled_during_approval_never_launches(tmp_path):
    """A reply received after cancellation cannot authorize the separately bound host run."""
    service, _backend, host, interaction = coordinator(
        tmp_path,
        CommandProcessResult(SandboxOutcome.DENIED, detail="policy denial"),
    )
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer
    cancellation_event = Event()

    def cancel_while_prompting(*_args, **_kwargs):
        """Model an interrupt racing with an affirmative approval reply."""
        cancellation_event.set()
        return "approve"

    interaction.prompt.side_effect = cancel_while_prompting

    assert (
        service.retry_host_command(
            offer,
            approved,
            cancellation_event=cancellation_event,
        )
        is None
    )
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"source": "different"},
        {"environment": {"PATH": "/different"}},
        {"workspace_id": "different-workspace"},
        {"read_roots": ()},
        {"write_roots": ()},
        {"network": True},
        {"policy_version": "different-policy"},
        {"file_creation_mask": 0o077},
    ],
)
def test_changed_host_identity_starts_none(tmp_path, change):
    """A change in reviewed command, context, or native authority invalidates the offer."""
    failure = CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="launcher missing")
    service, _, host, _ = coordinator(tmp_path, failure)
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer
    assert service.retry_host_command(offer, request(tmp_path, **change)) is None
    host.run_host_command.assert_not_called()


def test_generated_scratch_classification_cannot_change_a_pending_host_offer(tmp_path):
    """A chosen TMPDIR cannot be reclassified as generated to reuse an older host rule."""
    service, _, host, _ = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="launcher missing")
    )
    with tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as directory:
        scratch = Path(directory).resolve()
        environment = {"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)}
        chosen = request(tmp_path, environment=environment)
        generated = request(tmp_path, environment=environment, generated_environment=("TMPDIR",))
        offer = service.run_sandboxed(chosen, "run_command").offer

        assert offer is not None
        assert service.retry_host_command(offer, generated) is None
        host.run_host_command.assert_not_called()


def test_fresh_scratch_requires_a_new_host_decision(tmp_path):
    """A pending offer cannot authorize a different TMPDIR."""
    service, _, host, interaction = coordinator(
        tmp_path,
        CommandProcessResult(SandboxOutcome.DENIED, detail="policy denial"),
        answer="approve",
    )
    with (
        tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as first_directory,
        tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as second_directory,
    ):
        first = request(tmp_path, environment={"TMPDIR": str(Path(first_directory).resolve())})
        second = request(tmp_path, environment={"TMPDIR": str(Path(second_directory).resolve())})
        offer = service.run_sandboxed(first, "run_command").offer
        assert service.retry_host_command(offer, first) == "host-result"
        offer = service.run_sandboxed(second, "run_command").offer
        assert service.retry_host_command(offer, first) is None
        offer = service.run_sandboxed(second, "run_command").offer
        assert service.retry_host_command(offer, second) == "host-result"
        assert interaction.prompt.call_count == 5
        assert host.run_host_command.call_count == 2


def test_auto_approval_and_changed_root_start_none(tmp_path):
    """Automatic mode and a replaced workspace cannot approve host launch."""
    failure = CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="launcher missing")
    approved = request(tmp_path)
    automatic, automatic_backend, automatic_host, _ = coordinator(
        tmp_path, failure, auto_approval=True
    )
    offer = automatic.run_sandboxed(
        approved,
        "run_command",
        prelaunch_failure=CommandProcessResult(
            SandboxOutcome.UNAVAILABLE, detail="launcher missing"
        ),
    ).offer
    assert automatic.retry_host_command(offer, approved) is None
    automatic_backend.run.assert_not_called()
    automatic_host.run_host_command.assert_not_called()

    service, _, host, interaction = coordinator(tmp_path, failure)
    offer = service.run_sandboxed(
        approved,
        "run_command",
        prelaunch_failure=CommandProcessResult(
            SandboxOutcome.UNAVAILABLE, detail="launcher missing"
        ),
    ).offer
    tmp_path.rename(tmp_path.with_name("old-workspace"))
    tmp_path.mkdir()
    assert service.retry_host_command(offer, approved) is None
    interaction.prompt.assert_called_once()
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize("execution_timeout", [None, 30.0])
def test_changed_root_during_sandbox_approval_produces_no_host_offer(tmp_path, execution_timeout):
    """Renewing an execution budget never accepts a root replaced during authorization."""
    service, backend, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    )
    approved = request(tmp_path)

    def answer(*_args, **_kwargs):
        """Replace the workspace while the user considers the approval."""
        tmp_path.rename(tmp_path.with_name("old-workspace"))
        tmp_path.mkdir()
        return "approve"

    interaction.prompt.side_effect = answer
    attempt = service.run_sandboxed(approved, "run_command", execution_timeout=execution_timeout)

    assert attempt.result.outcome is SandboxOutcome.STALE
    assert attempt.offer is None
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()


def test_sandbox_approval_wait_does_not_consume_execution_budget(tmp_path, monkeypatch):
    """A long interactive wait leaves a full bounded sandbox budget after approval."""
    now = [100.0]
    monkeypatch.setattr("loop.execution.coordinator.time.monotonic", lambda: now[0])
    service, backend, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    )
    approved = request(tmp_path, deadline=130.0)

    def answer(*_args, **_kwargs):
        """Return approval after the original execution deadline would have expired."""
        now[0] = 243.0
        return "approve"

    interaction.prompt.side_effect = answer
    attempt = service.run_sandboxed(approved, "run_command", execution_timeout=30.0)
    launched = backend.run.call_args.args[0]
    assert launched.deadline == 333.0
    assert launched.runtime_timeout == 30.0
    assert launched.source == approved.source
    assert launched.attempt_id == approved.attempt_id
    assert launched.paths_are_current()
    assert attempt.result.outcome is SandboxOutcome.COMPLETED
    assert attempt.offer is None
    host.run_host_command.assert_not_called()


def test_sandbox_admission_precedes_one_observed_native_attempt(tmp_path):
    """A continuation becomes ready once, after authorization and before native execution."""
    result = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    service, backend, _, _ = coordinator(tmp_path, result)
    order = []
    observer = Mock()
    backend.run.side_effect = lambda *_args, **_kwargs: order.append("native") or result

    attempt = service.run_sandboxed(
        request(tmp_path),
        "run_command",
        on_admitted=lambda: order.append("admitted"),
        output_observer=observer,
    )

    assert attempt.result is result
    assert order == ["admitted", "native"]
    assert backend.run.call_args.kwargs["output_observer"] is observer


@pytest.mark.parametrize("network", [False, True])
def test_sandbox_admission_does_not_pause_existing_egress(tmp_path, network):
    """Default and broad-network commands avoid unchanged lease pause replies."""
    result = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    service, backend, _host, _interaction = coordinator(tmp_path, result)
    lease = Mock()
    lease.pause.return_value = False
    manager = service._permissions
    if network:
        manager.set_yolo(True)
        manager.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)
    manager.bind_egress(manager.egress_policy_revision(), lease)
    try:
        attempt = service.run_sandboxed(request(tmp_path, network=network), "run_command")
        assert attempt.result is result
        backend.run.assert_called_once()
        lease.pause.assert_not_called()
        lease.resume.assert_not_called()
        lease.revoke_revision.assert_not_called()
    finally:
        manager.unbind_egress(lease)


@pytest.mark.parametrize("network", [False, True])
def test_policy_change_before_sandbox_admission_prevents_native_launch(
    tmp_path, monkeypatch, network
):
    """A pending sandbox command cannot launch after its approving policy changes."""
    result = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    service, backend, host, _interaction = coordinator(tmp_path, result)
    service._permissions.set_yolo(True)
    if network:
        service._permissions.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)
    current_paths = SandboxRequest.paths_are_current

    def change_policy_after_path_check(current):
        """Revoke the relevant authority after the final path identity check."""
        unchanged = current_paths(current)
        if network:
            service._permissions.set_yolo(False)
        else:
            service._permissions.add_rule(
                PermissionRule(
                    decision=Decision.DENY,
                    action=Action.PROCESS_EXECUTE,
                    tool="run_command",
                )
            )
        return unchanged

    monkeypatch.setattr(SandboxRequest, "paths_are_current", change_policy_after_path_check)
    admitted = Mock()

    attempt = service.run_sandboxed(
        request(tmp_path, network=network), "run_command", on_admitted=admitted
    )

    assert attempt.result.outcome is SandboxOutcome.STALE
    assert attempt.offer is None
    admitted.assert_not_called()
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()


def test_expired_absolute_budget_never_offers_host_execution(tmp_path, monkeypatch):
    """An already expired execution budget remains a timeout rather than an escalation."""
    service, backend, host, _ = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    )
    approved = request(tmp_path, deadline=100.0)
    monkeypatch.setattr("loop.execution.coordinator.time.monotonic", lambda: 101.0)
    attempt = service.run_sandboxed(approved, "run_command")
    assert attempt.result.outcome is SandboxOutcome.TIMED_OUT
    assert attempt.offer is None
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize(
    "outcome", [SandboxOutcome.INVALID, SandboxOutcome.STALE, SandboxOutcome.UNSUPPORTED]
)
def test_unsafe_or_stale_native_preparation_never_offers_host_execution(tmp_path, outcome):
    """Native safety validation failures cannot become unrestricted host authority."""
    service, _, host, _ = coordinator(tmp_path, CommandProcessResult(outcome))
    attempt = service.run_sandboxed(request(tmp_path), "run_command")
    assert attempt.result.outcome is outcome
    assert attempt.offer is None
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize("capture_revision", [True, False])
def test_mediated_attempt_binds_exact_approval_without_host_fallback(
    tmp_path, monkeypatch, capture_revision
):
    """A mediator either receives its approving revision or fails before launch."""
    service, backend, host, _interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    )
    service._permissions.set_yolo(True)
    approved = request(
        tmp_path,
        read_roots=(),
        write_roots=(),
        egress_origins=("http://example.com",),
    )
    if not capture_revision:
        monkeypatch.setattr(
            service._permissions, "authorize_sandboxed_command", Mock(return_value=True)
        )
    admitted = Mock()
    attempt = service.run_sandboxed(approved, "run_command", on_admitted=admitted)
    assert attempt.offer is None
    host.run_host_command.assert_not_called()
    if capture_revision:
        assert attempt.result.outcome is SandboxOutcome.UNAVAILABLE
        assert backend.run.call_args.kwargs["policy_authority"] is service._permissions
        assert backend.run.call_args.kwargs["approved_revision"] == (
            service._permissions.egress_policy_revision()
        )
    else:
        assert attempt.result.outcome is SandboxOutcome.INVALID
        admitted.assert_not_called()
        backend.run.assert_not_called()


@pytest.mark.parametrize("execution_timeout", [None, 30.0])
def test_host_retry_rechecks_root_after_approval(tmp_path, execution_timeout):
    """A root replacement during the one-off prompt invalidates the approved host launch."""
    service, _, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="launcher missing")
    )
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer

    def replace_root(*_args, **_kwargs):
        """Simulate a changed workspace before the user's answer returns."""
        tmp_path.rename(tmp_path.with_name("old-workspace"))
        tmp_path.mkdir()
        return "approve"

    interaction.prompt.side_effect = replace_root
    assert service.retry_host_command(offer, approved, execution_timeout=execution_timeout) is None
    host.run_host_command.assert_not_called()


def test_host_retry_uses_fresh_deadline_with_the_same_command_identity(tmp_path):
    """The host retry can receive a fresh deadline without changing reviewed authority."""
    service, _, host, _ = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="launcher missing")
    )
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer
    renewed = request(tmp_path, deadline=approved.deadline + 60)

    assert service.retry_host_command(offer, renewed) == "host-result"
    host.run_host_command.assert_called_once_with(HostCommandRequest.from_sandbox(renewed))


def test_host_approval_wait_does_not_consume_execution_budget(tmp_path, monkeypatch):
    """The separately approved host attempt starts its budget after the host decision."""
    now = [100.0]
    monkeypatch.setattr("loop.execution.coordinator.time.monotonic", lambda: now[0])
    service, _, host, interaction = coordinator(
        tmp_path,
        CommandProcessResult(
            SandboxOutcome.UNAVAILABLE,
            detail=f"launcher failed at {tmp_path}",
            failure_context="Native sandbox launch check failed.",
        ),
    )
    approved = request(tmp_path, deadline=130.0)
    offer = service.run_sandboxed(approved, "run_command").offer

    def answer(*_args, **_kwargs):
        """Delay the host decision without changing approved authority."""
        now[0] = 300.0
        return "approve"

    interaction.prompt.side_effect = answer
    assert service.retry_host_command(offer, approved, execution_timeout=30.0) == "host-result"
    assert host.run_host_command.call_args.args[0].deadline == 330.0
    assert "Native sandbox launch check failed." in interaction.info.call_args.args[0]
    assert str(tmp_path) not in interaction.info.call_args.args[0]


def test_host_executor_failure_is_audited_and_not_replayed(tmp_path):
    """A failed approved host start consumes the offer before reporting the failure."""
    service, _, host, _ = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.DENIED, detail="policy denial")
    )
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer
    host.run_host_command.side_effect = OSError("spawn failed")

    with pytest.raises(OSError, match="spawn failed"):
        service.retry_host_command(offer, approved)
    assert service.retry_host_command(offer, approved) is None
    host.run_host_command.assert_called_once()


def test_offer_approval_and_host_execution_have_distinct_sanitized_audit_events(tmp_path):
    """Audit records separate the offer, user decision, and host launch without shell text."""
    service, _, _, _ = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.DENIED, detail="policy denial")
    )
    approved = request(tmp_path, source="printf private-command")
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    try:
        offer = service.run_sandboxed(approved, "run_command").offer
        assert service.retry_host_command(offer, approved) == "host-result"
        assert telemetry.close(1)
    finally:
        set_telemetry(None)

    events = [record.event_name for record in adapter.records]
    assert events == [
        "sandbox.permission_decided",
        "sandbox.completed",
        "host_command.offered",
        "host_command.approved",
        "host_command.executed",
    ]
    assert all("private-command" not in str(record.attributes) for record in adapter.records)
    assert all(record.attributes["attempt_id"] == approved.attempt_id for record in adapter.records)


def test_local_host_executor_does_not_launch_after_deadline(tmp_path, monkeypatch):
    """An expired host retry cannot start a process after its bound deadline."""
    launch = Mock()
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", launch)
    approved = HostCommandRequest.from_sandbox(request(tmp_path, deadline=time.monotonic() - 1))

    result = LocalHostCommandExecutor().run_host_command(approved)

    assert result.outcome is SandboxOutcome.TIMED_OUT
    launch.assert_not_called()


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), float("-inf")])
def test_local_host_executor_rejects_nonfinite_deadline_before_spawn(
    tmp_path, monkeypatch, deadline
):
    """A direct host request with a non-finite deadline cannot launch a child."""
    spawn = Mock()
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", spawn)
    approved = HostCommandRequest.from_sandbox(request(tmp_path))
    with pytest.raises(ValueError, match="Command deadline must be finite"):
        LocalHostCommandExecutor().run_host_command(replace(approved, deadline=deadline))
    spawn.assert_not_called()


def test_local_host_executor_allows_completion_near_deadline(
    tmp_path, monkeypatch, process_exit_observer
):
    """A completed child in the final budget interval remains successful without real waits."""
    clock = SimpleNamespace(monotonic=lambda: 100.09)
    monkeypatch.setattr("loop.execution.coordinator.time", clock)
    monkeypatch.setattr("loop.utils.process.time", clock)
    process = Mock(
        stdout=TextIOWrapper(BytesIO(b"finished")), stderr=TextIOWrapper(BytesIO()), returncode=0
    )
    launch = Mock(return_value=process)
    kill = Mock()
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", launch)
    monkeypatch.setattr("loop.execution.coordinator.kill_process_group", kill)

    def reader_thread(*, target, daemon):
        """Drain a fixture stream synchronously without depending on thread scheduling."""
        reader = Mock()
        reader.start.side_effect = target
        reader.is_alive.return_value = False
        return reader

    monkeypatch.setattr("loop.execution.coordinator.threading.Thread", reader_thread)
    approved = HostCommandRequest.from_sandbox(request(tmp_path, deadline=100.1))

    result = LocalHostCommandExecutor().run_host_command(approved)

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.stdout == "finished"
    assert process.wait.called
    assert process_exit_observer.waitid.call_count == 1
    launch.assert_called_once()
    kill.assert_called_once_with(process)


def test_local_host_executor_cancels_and_cleans_up(tmp_path, monkeypatch, process_exit_observer):
    """Interrupting an approved host child returns cancellation after group cleanup."""
    process = Mock(stdout=TextIOWrapper(BytesIO()), stderr=TextIOWrapper(BytesIO()))
    process.wait.return_value = 0
    process_exit_observer.waitid = Mock(side_effect=KeyboardInterrupt)
    kill = Mock()
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", lambda *a, **k: process)
    monkeypatch.setattr("loop.execution.coordinator.kill_process_group", kill)

    result = LocalHostCommandExecutor().run_host_command(
        HostCommandRequest.from_sandbox(request(tmp_path))
    )

    assert result.outcome is SandboxOutcome.CANCELLED
    kill.assert_called_once_with(process)


def test_local_host_executor_passes_cancellation_to_process_supervisor(tmp_path, monkeypatch):
    """Host supervision receives the run signal and reports its cleaned cancellation result."""
    from loop.utils.process import ProcessCapture, ProcessCaptureStatus

    cancellation_event = Event()
    process = Mock(stdout=TextIOWrapper(BytesIO()), stderr=TextIOWrapper(BytesIO()))
    capture = ProcessCapture(ProcessCaptureStatus.CANCELLED, stdout="partial", exit_code=None)
    supervise = Mock(return_value=capture)
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", lambda *a, **k: process)
    monkeypatch.setattr("loop.execution.coordinator.supervise_process", supervise)

    result = LocalHostCommandExecutor().run_host_command(
        HostCommandRequest.from_sandbox(request(tmp_path)),
        cancellation_event=cancellation_event,
    )

    assert result.outcome is SandboxOutcome.CANCELLED
    assert result.possible_effects is True
    assert supervise.call_args.kwargs["cancel_event"] is cancellation_event


def test_local_host_executor_skips_spawn_for_pre_cancelled_request(tmp_path, monkeypatch):
    """A host executor called after cancellation never starts the approved shell."""
    cancellation_event = Event()
    cancellation_event.set()
    launch = Mock()
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", launch)

    result = LocalHostCommandExecutor().run_host_command(
        HostCommandRequest.from_sandbox(request(tmp_path)),
        cancellation_event=cancellation_event,
    )

    assert result.outcome is SandboxOutcome.CANCELLED
    assert result.possible_effects is False
    launch.assert_not_called()


def test_local_host_executor_fails_if_child_pipes_are_missing(tmp_path, monkeypatch):
    """A child with missing pipes is terminated but may already have had effects."""
    process = Mock(stdout=None, stderr=None)
    kill = Mock()
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", lambda *a, **k: process)
    monkeypatch.setattr("loop.execution.coordinator.kill_process_group", kill)

    result = LocalHostCommandExecutor().run_host_command(
        HostCommandRequest.from_sandbox(request(tmp_path))
    )

    assert result.outcome is SandboxOutcome.UNAVAILABLE
    assert result.possible_effects is True
    kill.assert_called_once_with(process)


@pytest.mark.parametrize("failed_stream", ["stdout", "stderr"])
def test_local_host_executor_preserves_partial_output_after_reader_failure(
    tmp_path, monkeypatch, failed_stream
):
    """Incomplete host output remains available without being reported as a successful run."""
    from loop.utils.process import ProcessCapture, ProcessCaptureStatus

    capture = ProcessCapture(
        ProcessCaptureStatus.UNAVAILABLE,
        stdout="partial-out",
        stderr="partial-err",
        stdout_discarded=None if failed_stream == "stdout" else 0,
        stderr_discarded=None if failed_stream == "stderr" else 0,
    )
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", Mock(return_value=Mock()))
    monkeypatch.setattr("loop.execution.coordinator.supervise_process", Mock(return_value=capture))

    result = LocalHostCommandExecutor().run_host_command(
        HostCommandRequest.from_sandbox(request(tmp_path))
    )

    assert result.outcome is SandboxOutcome.UNAVAILABLE
    assert result.possible_effects is True
    assert result.detail == "Host output capture failed."
    assert result.stdout == "partial-out" and result.stderr == "partial-err"
    assert getattr(result, f"{failed_stream}_discarded") is None


def test_host_pipe_failure_does_not_wait_past_deadline(tmp_path, monkeypatch):
    """A broken host child is reported even when post-kill reaping times out."""
    process = Mock(stdout=None, stderr=None)
    process.wait.side_effect = subprocess.TimeoutExpired("sh", 0)
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", lambda *a, **k: process)
    monkeypatch.setattr("loop.execution.coordinator.kill_process_group", Mock())
    result = LocalHostCommandExecutor().run_host_command(
        HostCommandRequest.from_sandbox(request(tmp_path))
    )
    assert result.outcome is SandboxOutcome.UNAVAILABLE


def test_local_host_executor_detects_stuck_reader_and_bounded_cleanup(
    tmp_path, monkeypatch, process_exit_observer
):
    """A reader that outlives the deadline does not hold the host call open."""

    class StuckReader:
        """Simulate an output reader that cannot finish within the deadline."""

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

        def is_alive(self):
            return True

    process = Mock()
    process.stdout = Mock()
    process.stderr = Mock()
    process.wait.side_effect = [0, subprocess.TimeoutExpired("sh", 1)]
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", lambda *a, **k: process)
    monkeypatch.setattr("loop.execution.coordinator.threading.Thread", StuckReader)
    monkeypatch.setattr("loop.execution.coordinator.kill_process_group", Mock())

    result = LocalHostCommandExecutor().run_host_command(
        HostCommandRequest.from_sandbox(request(tmp_path))
    )

    assert result.outcome is SandboxOutcome.TIMED_OUT
    process.stdout.close.assert_not_called()
    process.stderr.close.assert_not_called()
