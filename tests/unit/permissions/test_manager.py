"""Tests for layered operation-policy evaluation, approval, and persistence."""

import asyncio
import errno
import io
import json
import os
import shlex
import shutil
import sqlite3
import stat
import tempfile
import time
from contextlib import ExitStack, closing, contextmanager, nullcontext
from pathlib import Path
from threading import Barrier, Event, Thread, get_ident
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
from rich.console import Console

from loop import (
    Action,
    ApprovalChoice,
    AuthorizationResult,
    Decision,
    FileTarget,
    Interaction,
    NetworkTarget,
    Operation,
    PermissionConfiguration,
    PermissionConfigurationError,
    PermissionLoadPolicy,
    PermissionLoadResult,
    PermissionManager,
    PermissionPreset,
    PermissionPresetError,
    PermissionRule,
    PolicyDecision,
    PolicyLimits,
    PolicyScope,
    ProblemException,
    ProcessBoundary,
    ProcessTarget,
    SessionTarget,
)
from loop.command_authorities import command_authorities
from loop.execution.sandbox import SandboxRequest
from loop.interaction import ConsoleInteraction
from loop.permissions import (
    CommandFinding,
    CommandInspection,
    CommandReviewStatus,
    LimitClassification,
    PermissionLoadFailure,
    PermissionNamespaceError,
    SandboxedCommandRule,
    UserPermissionConfiguration,
)
from loop.permissions import manager as manager_module
from loop.permissions.managed import ManagedPolicyStore
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry
from loop.telemetry.policy import thaw
from loop.utils import PathHolder, ShutdownRequested
from loop.utils.yaml import dump_yaml, load_yaml
from loop.workspace import Workspace, WorkspaceRepository
from tests.scheduling import deliver_callbacks

_HARD_LEGACY_READ_POLICY = "version: 1\nlimits:\n  readable_roots: []\n"


def prompt_scope(message, *, full=False):
    """Read the chosen authority summary from a human approval prompt."""
    prefix = "Full scope: " if full else "Scope: "
    return json.loads(
        next(line.removeprefix(prefix) for line in message.splitlines() if line.startswith(prefix))
    )


def managed_store(
    tmp_path,
    *,
    rules=(),
    require_sandbox=False,
    forbid_host=False,
    deny_local_networks=False,
    forbid_yolo=False,
):
    """Build an owned stand-in for the fixed protected administrator installation."""
    directory = tmp_path / "system" / "loop"
    directory.mkdir(parents=True)
    directory.parent.chmod(0o755)
    directory.chmod(0o755)
    identity = "a" * 32
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
                "require_sandbox": require_sandbox,
                "forbid_host": forbid_host,
                "deny_local_networks": deny_local_networks,
                "forbid_yolo": forbid_yolo,
            }
        )
    )
    (directory.parent / "loop.managed.expect.json").chmod(0o644)
    (directory / "managed-policy.json").chmod(0o644)
    return ManagedPolicyStore(directory, trust_anchor=tmp_path, owner_uid=os.getuid())


def test_uncertain_retry_generation_requires_current_human_decisions(tmp_path):
    """Enabling automatic decisions invalidates a retry even if later disabled."""
    manager = PermissionManager(tmp_path)
    generation = manager.recovery_retry_generation()
    assert generation is not None
    assert manager.recovery_retry_current(generation)
    manager.set_yolo(True)
    assert manager.recovery_retry_generation() is None
    assert not manager.recovery_retry_current(generation)
    manager.set_yolo(False)
    assert not manager.recovery_retry_current(generation)
    assert manager.recovery_retry_current(manager.recovery_retry_generation())
    manager.close()


def test_standalone_execution_retains_scratch_until_all_leases_settle(tmp_path):
    """A live standalone lease prevents scratch cleanup and rebinding, including nested leases."""
    manager = PermissionManager(tmp_path)
    scratch = manager.temporary_directory
    replacement = PathHolder(tmp_path)
    with manager.standalone_execution() as first:
        assert first == scratch and first.is_dir()
        with manager.standalone_execution() as second:
            assert second == first
            with pytest.raises(OSError, match="still owns temporary storage"):
                manager.close()
            with pytest.raises(OSError, match="still owns temporary storage"):
                manager.bind_temporary_directory(replacement)
        with pytest.raises(OSError, match="still owns temporary storage"):
            manager.close()
        assert scratch.is_dir()
    manager.bind_temporary_directory(replacement)
    with pytest.raises(OSError, match="requires its session owner"), manager.standalone_execution():
        pass
    manager.close()
    assert not scratch.exists()


def test_bind_rejects_an_active_shared_command_authority(tmp_path):
    """Session rebinding cannot change a permission manager while a command owns it."""
    manager = PermissionManager(tmp_path)
    original = manager.temporary_directory
    lease = command_authorities.reserve((manager,))
    try:
        with pytest.raises(OSError, match="Command execution still owns temporary storage"):
            manager.bind_temporary_directory(PathHolder(tmp_path))
        assert manager.temporary_directory == original
    finally:
        lease.release()
    manager.bind_temporary_directory(PathHolder(tmp_path))
    manager.close()


def test_standalone_execution_releases_on_exception_and_rejects_closed_manager(tmp_path):
    """Exceptional lease exits release storage, and a closed owner cannot grant new leases."""
    manager = PermissionManager(tmp_path)
    with pytest.raises(ValueError, match="failure"), manager.standalone_execution():
        raise ValueError("failure")
    manager.close()
    manager.close()
    with pytest.raises(OSError, match="manager is closed"), manager.standalone_execution():
        pass
    with pytest.raises(OSError, match="manager is closed"):
        manager.bind_temporary_directory(PathHolder(tmp_path))


def test_failed_scratch_cleanup_blocks_admission_and_can_be_retried(tmp_path, monkeypatch):
    """A failed cleanup remains retryable and never reopens standalone scratch admission."""
    manager = PermissionManager(tmp_path)
    scratch = manager.temporary_directory
    cleanup = manager._temporary_directory.cleanup
    attempts = []

    def clean():
        """Fail the first cleanup before allowing the real cleanup to finish."""
        attempts.append(None)
        if len(attempts) == 1:
            raise OSError("temporary cleanup failed")
        cleanup()

    monkeypatch.setattr(manager._temporary_directory, "cleanup", clean)
    with pytest.raises(OSError, match="temporary cleanup failed"):
        manager.close()
    assert scratch.exists()
    with pytest.raises(OSError, match="closed or closing"), manager.standalone_execution():
        pass
    with pytest.raises(OSError, match="closed or closing"):
        manager.bind_temporary_directory(PathHolder(tmp_path))
    with pytest.raises(OSError, match="policy changed"):
        manager.bind_egress(manager.egress_policy_revision(), Mock())
    manager.close()
    assert not scratch.exists()
    assert len(attempts) == 2


def test_yolo_publication_preserves_only_preexisting_authority(tmp_path):
    """Revoked or nested uncommitted authority stays suspended until publication settles."""
    manager = PermissionManager(tmp_path, yolo=True)
    with manager.suspend_yolo_decisions():
        assert manager.confirmation_decision() is True
        manager.set_yolo(False)
        manager.set_yolo(True)
        assert manager.confirmation_decision() is None
        with manager.suspend_yolo_decisions():
            assert manager.confirmation_decision() is None
        assert manager.confirmation_decision() is None
    assert manager.confirmation_decision() is True
    manager.close()


def test_overlapping_yolo_publications_cannot_inherit_uncommitted_authority(tmp_path):
    """A later publication stays suspended if the earlier publication settles first."""
    manager = PermissionManager(tmp_path)
    with ExitStack() as first:
        first.enter_context(manager.suspend_yolo_decisions())
        manager.set_yolo(True)
        with manager.suspend_yolo_decisions():
            first.close()
            assert manager.confirmation_decision() is None
    assert manager.confirmation_decision() is True
    manager.close()


def test_yolo_suspension_rollback_is_idempotent(tmp_path):
    """Rollback does not change revisions after automatic approval is already off."""
    manager = PermissionManager(tmp_path)
    initial_revision = manager.egress_policy_revision()
    with manager.suspend_yolo_decisions() as rollback:
        rollback()
        assert manager.egress_policy_revision() == initial_revision
        manager.set_yolo(True)
        assert manager.yolo is False
        rollback()
        disabled_revision = manager.egress_policy_revision()
        rollback()
        assert manager.egress_policy_revision() == disabled_revision
    assert manager.yolo is False
    assert manager.confirmation_decision() is None


def test_failed_yolo_rollback_keeps_already_disabled_authority_off(tmp_path):
    """A failed mediator pause cannot enable authority that was already disabled."""
    manager = PermissionManager(tmp_path)
    lease = Mock()
    lease.pause.return_value = False
    lease.revoke_revision.return_value = True
    manager.bind_egress(manager.egress_policy_revision(), lease)
    with (
        manager.suspend_yolo_decisions() as rollback,
        pytest.raises(OSError, match="egress admission pause is uncertain"),
    ):
        rollback()
    assert manager.yolo is False
    assert manager.confirmation_decision() is None
    manager.close()


def test_managed_yolo_prohibition_blocks_startup_and_activation(tmp_path):
    """A protected administrator prohibition overrides both startup and live activation."""
    store = managed_store(tmp_path, forbid_yolo=True)
    startup = PermissionManager(tmp_path, managed_policy_store=store, yolo=True)
    assert startup.yolo is False
    manager = PermissionManager(tmp_path, managed_policy_store=store)
    with pytest.raises(ValueError, match="prohibits YOLO activation"):
        manager.set_yolo(True)
    assert manager.yolo is False
    assert manager.confirmation_decision() is None
    manager.set_yolo(False)


def test_managed_yolo_source_change_disables_mode_until_trusted_reload(tmp_path):
    """A stale source suspends automatic approval and reload applies the new restriction."""
    store = managed_store(tmp_path)
    manager = PermissionManager(tmp_path, managed_policy_store=store, yolo=True)
    policy = store.directory / "managed-policy.json"
    original = json.loads(policy.read_text())
    original.update(revision=2, forbid_yolo=True)
    policy.write_text(json.dumps(original))
    assert manager.yolo is False
    assert manager.confirmation_decision() is None
    with pytest.raises(ValueError, match="recovery required"):
        manager.set_yolo(True)
    manager.reload()
    assert manager.yolo is False
    with pytest.raises(ValueError, match="prohibits YOLO"):
        manager.set_yolo(True)
    original.update(revision=3, forbid_yolo=False)
    policy.write_text(json.dumps(original))
    assert manager.yolo is False
    manager.reload()
    assert manager.yolo is False
    manager.set_yolo(True)
    assert manager.yolo is True


def test_yolo_publication_revalidates_managed_revocation(tmp_path):
    """A changed administrator policy revokes existing authority during publication."""
    store = managed_store(tmp_path)
    manager = PermissionManager(tmp_path, managed_policy_store=store, yolo=True)
    policy = store.directory / "managed-policy.json"
    with manager.suspend_yolo_decisions():
        assert manager.confirmation_decision() is True
        document = json.loads(policy.read_text())
        document.update(revision=2, forbid_yolo=True)
        policy.write_text(json.dumps(document))
        assert manager.confirmation_decision() is None
        manager.reload()
        with pytest.raises(ValueError, match="prohibits YOLO"):
            manager.set_yolo(True)
    assert manager.yolo is False
    manager.close()


def test_managed_reload_revokes_yolo_even_without_intermediate_inspection(tmp_path):
    """A prohibiting policy cannot silently reactivate an old setting after relaxation."""
    store = managed_store(tmp_path)
    manager = PermissionManager(tmp_path, managed_policy_store=store, yolo=True)
    policy = store.directory / "managed-policy.json"
    original = json.loads(policy.read_text())
    original.update(revision=2, forbid_yolo=True)
    policy.write_text(json.dumps(original))
    manager.reload()
    original.update(revision=3, forbid_yolo=False)
    policy.write_text(json.dumps(original))
    manager.reload()
    assert manager.yolo is False
    manager.set_yolo(True)
    assert manager.yolo is True


def test_invalid_managed_source_cannot_enable_yolo(tmp_path):
    """An expected but missing administrator policy fails closed for no-prompt mode."""
    store = managed_store(tmp_path)
    (store.directory / "managed-policy.json").unlink()
    startup = PermissionManager(tmp_path, managed_policy_store=store, yolo=True)
    assert startup.yolo is False
    with pytest.raises(PermissionNamespaceError, match="Administrator policy recovery required"):
        startup.require_policy_ready()
    manager = PermissionManager(tmp_path, managed_policy_store=store)
    with pytest.raises(ValueError, match="Administrator policy recovery required"):
        manager.set_yolo(True)
    assert manager.yolo is False


@pytest.mark.parametrize("managed", [False, True])
def test_local_network_limit_narrows_sandbox_without_denial_prompt(tmp_path, managed):
    """User or managed local deny blocks loopback while preserving ordinary commands."""
    store = managed_store(tmp_path, deny_local_networks=True) if managed else None
    manager = PermissionManager(tmp_path, managed_policy_store=store, yolo=True)
    assert manager.sandbox_local_network_enabled() is not managed
    local = native_request(tmp_path, read_roots=())
    denied = native_request(tmp_path, read_roots=(), local_network=False)
    if not managed:
        manager.set_limit("deny_local_networks", True, scope=PolicyScope.SESSION)
    assert manager.sandbox_local_network_enabled() is False
    assert manager.authorize_sandboxed_command(local, tool_id="run_command") is False
    assert manager.authorize_sandboxed_command(denied, tool_id="run_command") is True
    manager.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)
    assert "local" in manager.sandbox_network_limit("run_command")
    if not managed:
        assert manager.reset_limit("deny_local_networks")
        assert manager.sandbox_local_network_enabled()


@pytest.mark.parametrize("yolo", [False, True])
def test_managed_denial_and_host_prohibition_override_user_grants(tmp_path, yolo):
    """A user allow and YOLO cannot relax a managed deny or host restriction."""
    store = managed_store(
        tmp_path,
        rules=({"id": "deny-session", "decision": "deny", "action": "session.mutate"},),
        require_sandbox=True,
    )
    manager = PermissionManager(tmp_path, managed_policy_store=store, yolo=yolo)
    manager.add_rule(
        PermissionRule(decision=Decision.ALLOW, action=Action.SESSION_MUTATE),
        scope=PolicyScope.SESSION,
    )
    session = (operation(Action.SESSION_MUTATE, target=SessionTarget(identifier="item")),)
    assert manager.evaluate(session).sources == ("managed:deny:deny-session",)
    assert manager.authorize(session).decision is Decision.DENY
    host = (
        operation(
            Action.PROCESS_EXECUTE,
            target=ProcessTarget(
                argv=("/bin/sh", "-c", "printf ok"),
                cwd=str(tmp_path),
                boundary=ProcessBoundary.HOST,
            ),
        ),
    )
    assert manager.evaluate(host).sources == ("managed:sandbox_required",)
    assert manager.host_gate_enable_revision(host) is None
    assert "identity=" in manager.describe("managed")
    protected = (file_operation(Action.FILESYSTEM_READ, store.directory / "managed-policy.json"),)
    assert manager.evaluate(protected).sources == ("boundary:protected_path",)
    expectation = (file_operation(Action.FILESYSTEM_READ, store.expectation_path),)
    assert manager.evaluate(expectation).sources == ("boundary:protected_path",)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("yolo", [False, True])
def test_managed_fresh_human_rule_never_uses_yolo_or_remembered_approval(
    tmp_path, asynchronous, yolo
):
    """A managed human requirement permits only one live approval outside YOLO."""
    store = managed_store(
        tmp_path,
        rules=({"id": "fresh", "decision": "ask", "action": "session.mutate"},),
    )
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt = Mock(return_value=ApprovalChoice.ONCE)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.ONCE)
    manager = PermissionManager(
        tmp_path, managed_policy_store=store, interaction=interaction, yolo=yolo
    )
    manager.add_rule(
        PermissionRule(decision=Decision.ALLOW, action=Action.SESSION_MUTATE),
        scope=PolicyScope.SESSION,
    )
    operations = (operation(Action.SESSION_MUTATE, target=SessionTarget(identifier="item")),)
    result = (
        asyncio.run(manager.authorize_async(operations))
        if asynchronous
        else manager.authorize(operations)
    )
    assert result.decision is (Decision.DENY if yolo else Decision.ALLOW)
    assert result.prompted is not yolo
    assert result.installed_rule_ids == ()
    if yolo:
        interaction.prompt.assert_not_called()
        interaction.prompt_async.assert_not_called()
    else:
        selected = interaction.prompt_async if asynchronous else interaction.prompt
        assert set(selected.call_args.kwargs["choices"]) == {"deny", "once"}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_managed_human_rule_requires_fresh_sandbox_review(tmp_path, asynchronous):
    """Default sandbox access cannot silently satisfy a managed human rule."""
    store = managed_store(
        tmp_path,
        rules=({"id": "fresh-command", "decision": "ask", "action": "process.execute"},),
    )
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt.return_value = "approve"
    interaction.prompt_async = AsyncMock(return_value="approve")
    manager = PermissionManager(tmp_path, managed_policy_store=store, interaction=interaction)
    request = native_request(tmp_path, read_roots=())
    approved = (
        asyncio.run(manager.authorize_sandboxed_command_async(request, tool_id="run_command"))
        if asynchronous
        else manager.authorize_sandboxed_command(request, tool_id="run_command")
    )
    assert approved is True
    selected = interaction.prompt_async if asynchronous else interaction.prompt
    assert set(selected.call_args.kwargs["choices"]) == {"deny", "approve", "details"}
    assert manager.sandboxed_command_rules(ApprovalChoice.SESSION) == ()


def test_async_sandbox_approval_rechecks_managed_replacement(tmp_path):
    """A managed policy replacement during approval revokes a standalone async command."""
    store = managed_store(tmp_path)
    interaction = Mock(spec=ConsoleInteraction)
    policy = store.directory / "managed-policy.json"

    async def replace_policy(*_args, **_kwargs):
        """Replace the protected policy after initial authorization evaluation."""
        payload = json.loads(policy.read_text(encoding="utf-8"))
        payload["revision"] = 2
        payload["rules"] = [{"id": "deny-network", "decision": "deny", "action": "network.request"}]
        policy.write_text(json.dumps(payload), encoding="utf-8")
        return "approve"

    interaction.prompt_async = AsyncMock(side_effect=replace_policy)
    manager = PermissionManager(tmp_path, managed_policy_store=store, interaction=interaction)
    manager.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)

    assert not asyncio.run(
        manager.authorize_sandboxed_command_async(
            native_request(tmp_path, network=True, read_roots=()),
            tool_id="run_command",
        )
    )
    interaction.prompt_async.assert_awaited_once()


def test_managed_replacement_denies_until_trusted_reload(tmp_path):
    """Policy changes, deletion and malformed replacement remain denied until reload."""
    store = managed_store(tmp_path)
    manager = PermissionManager(tmp_path, managed_policy_store=store)
    operations = (
        operation(Action.FILESYSTEM_READ, target=FileTarget(path=str(tmp_path / "file"))),
    )
    assert manager.evaluate(operations).decision is Decision.ALLOW
    policy = store.directory / "managed-policy.json"
    payload = json.loads(policy.read_text())
    payload["revision"] = 2
    payload["rules"] = [{"id": "deny-read", "decision": "deny", "action": "filesystem.read"}]
    policy.write_text(json.dumps(payload))
    assert manager.evaluate(operations).sources == ("managed:recovery_required",)
    assert "recovery required" in manager.describe("managed")
    manager.reload()
    assert manager.evaluate(operations).sources == ("managed:deny:deny-read",)
    payload["revision"] = 1
    policy.write_text(json.dumps(payload))
    manager.reload()
    assert manager.evaluate(operations).sources == ("managed:recovery_required",)
    payload["revision"] = 3
    policy.write_text(json.dumps(payload))
    manager.reload()
    assert manager.evaluate(operations).sources == ("managed:deny:deny-read",)
    policy.unlink()
    assert manager.evaluate(operations).sources == ("managed:recovery_required",)
    store.directory.rmdir()
    restarted = PermissionManager(tmp_path, managed_policy_store=store)
    assert restarted.evaluate(operations).sources == ("managed:recovery_required",)
    (store.directory.parent / "loop.managed.expect.json").unlink()
    manager.reload()
    assert manager.evaluate(operations).sources == ("managed:recovery_required",)
    store.directory.mkdir()
    store.directory.chmod(0o755)
    (store.directory.parent / "loop.managed.expect.json").write_text(
        json.dumps({"version": 1, "identity": "a" * 32, "minimum_revision": 3})
    )
    (store.directory.parent / "loop.managed.expect.json").chmod(0o644)
    policy.write_text("invalid")
    policy.chmod(0o644)
    manager.reload()
    assert manager.evaluate(operations).sources == ("managed:recovery_required",)


def test_unmanaged_platform_and_nonmanaged_views_remain_available(tmp_path):
    """An unsupported platform has no managed source and ordinary views still render."""
    store = Mock(spec=ManagedPolicyStore)
    store.directory = None
    store.expectation_path = None
    store.load.return_value = None
    store.may_be_installed.return_value = False
    manager = PermissionManager(tmp_path, managed_policy_store=store)
    assert "unavailable" in manager.describe("managed")
    assert (
        manager.evaluate(
            (operation(Action.SESSION_MUTATE, target=SessionTarget(identifier="item")),)
        ).decision
        is Decision.ASK
    )
    store.load.assert_called_once()

    active = PermissionManager(tmp_path, managed_policy_store=managed_store(tmp_path))
    assert "Managed restrictions:" not in active.describe("workspace")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("headless", [False, True])
def test_yolo_authorizes_ask_once_without_a_prompt_or_saved_rule(tmp_path, asynchronous, headless):
    """Interactive and headless approval use the same one-request YOLO decision."""
    interaction = None if headless else Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction, yolo=True)
    operations = (operation(Action.SESSION_MUTATE, target=SessionTarget(identifier="item")),)
    result = (
        asyncio.run(manager.authorize_async(operations))
        if asynchronous
        else manager.authorize(operations)
    )
    assert result.decision is Decision.ALLOW
    assert result.source == "yolo"
    assert result.prompted is False
    assert result.approval_choice is ApprovalChoice.ONCE
    assert result.installed_rule_ids == ()
    if interaction is not None:
        interaction.prompt.assert_not_called()

    manager.set_default(Action.SESSION_MUTATE, Decision.DENY)
    denied = (
        asyncio.run(manager.authorize_async(operations))
        if asynchronous
        else manager.authorize(operations)
    )
    assert denied.decision is Decision.DENY
    assert denied.source == "policy"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_yolo_scoped_sandbox_approval_does_not_persist_or_relax_hard_limits(tmp_path, asynchronous):
    """YOLO grants one enforceable sandbox scope but retains hard ceilings and denials."""
    interaction = Mock(spec=ConsoleInteraction)
    manager = PermissionManager(tmp_path, interaction=interaction, yolo=True)
    external = tmp_path.parent / "runtime"
    external.mkdir()
    request = native_request(tmp_path, "printf text > output", read_roots=(external,))
    approved_revision = []

    def authorize():
        """Request the same sandbox scope through the selected dispatch boundary."""
        if asynchronous:
            return asyncio.run(
                manager.authorize_sandboxed_command_async(
                    request, tool_id="run_command", approved_revision=approved_revision
                )
            )
        return manager.authorize_sandboxed_command(
            request, tool_id="run_command", approved_revision=approved_revision
        )

    assert authorize() is True
    assert approved_revision == [manager.egress_policy_revision()]
    assert manager.sandboxed_command_rules(ApprovalChoice.SESSION) == ()
    assert manager.sandboxed_command_rules(ApprovalChoice.WORKSPACE) == ()
    interaction.prompt.assert_not_called()
    interaction.prompt_async.assert_not_called()

    manager.update_limit_values("readable_roots", "workspace", add=False)
    assert authorize() is False
    assert len(approved_revision) == 1
    manager.set_limit("deny_private_networks", True, scope=PolicyScope.SESSION)
    network = native_request(tmp_path, "printf text", read_roots=(), network=True)
    assert (
        asyncio.run(manager.authorize_sandboxed_command_async(network, tool_id="run_command"))
        if asynchronous
        else manager.authorize_sandboxed_command(network, tool_id="run_command")
    ) is False


def test_yolo_host_gate_is_transient_and_explicit_deny_still_wins(tmp_path):
    """Only active YOLO bypasses the user-changeable host gate for an exact host plan."""
    manager = PermissionManager(tmp_path)
    operations = (
        operation(
            Action.PROCESS_EXECUTE,
            target=ProcessTarget(
                argv=("/bin/sh", "-c", "printf ok"),
                cwd=str(tmp_path),
                boundary=ProcessBoundary.HOST,
            ),
        ),
    )
    assert manager.evaluate(operations).decision is Decision.DENY
    assert manager.confirmation_decision() is None
    manager.set_yolo(True)
    manager.set_yolo(True)
    assert manager.evaluate(operations).decision is Decision.ASK
    assert manager.confirmation_decision() is True
    assert manager.confirmation_decision(certain=False) is False
    assert manager.request_permission("Approve?") is ApprovalChoice.ONCE
    assert asyncio.run(manager.request_permission_async("Approve?")) is ApprovalChoice.ONCE
    manager.add_rule(
        PermissionRule(decision=Decision.DENY, tool="demo", action=Action.PROCESS_EXECUTE),
        scope=PolicyScope.SESSION,
    )
    assert manager.evaluate(operations).decision is Decision.DENY
    manager.set_yolo(False)
    assert manager.evaluate(operations).decision is Decision.DENY
    assert manager.confirmation_decision() is None
    assert manager.configuration.limits.allow_host_processes is False


@pytest.mark.parametrize("asynchronous", [False, True])
def test_yolo_rechecks_a_changed_policy_before_authorization(tmp_path, monkeypatch, asynchronous):
    """A deny published between the initial snapshot and final decision prevents admission."""
    manager = PermissionManager(tmp_path, yolo=True)
    operations = (operation(Action.SESSION_MUTATE, target=SessionTarget(identifier="item")),)
    initial = manager.evaluate(operations)
    denied = PolicyDecision(
        decision=Decision.DENY,
        reason="Policy changed while approval was pending.",
        sources=("rule:session:deny",),
    )
    if asynchronous:
        monkeypatch.setattr(manager, "evaluate_async", AsyncMock(side_effect=[initial, denied]))
        result = asyncio.run(manager.authorize_async(operations))
    else:
        monkeypatch.setattr(manager, "evaluate", Mock(side_effect=[initial, denied]))
        result = manager.authorize(operations)
    assert result.decision is Decision.DENY
    assert result.source == "policy"
    assert result.installed_rule_ids == ()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_yolo_never_migrates_a_legacy_hard_read_limit_implicitly(tmp_path, asynchronous):
    """A legacy classification stays hard and no migration prompt appears in YOLO mode."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text(_HARD_LEGACY_READ_POLICY, encoding="utf-8")
    interaction = Mock(spec=ConsoleInteraction)
    manager = PermissionManager(tmp_path, interaction=interaction, yolo=True)
    request = native_request(tmp_path, read_roots=(tmp_path.parent,))
    reviewed = (
        asyncio.run(manager.review_legacy_sandbox_read_limit_async(request, interaction))
        if asynchronous
        else manager.review_legacy_sandbox_read_limit(request, interaction)
    )
    assert reviewed is False
    interaction.prompt.assert_not_called()
    interaction.prompt_async.assert_not_called()


def test_yolo_rejects_invalid_policy_loading_without_recovery_prompt(tmp_path):
    """A malformed policy remains a startup error when no-prompt mode is active."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text("invalid: [", encoding="utf-8")
    interaction = Mock(spec=Interaction)
    with pytest.raises(PermissionConfigurationError):
        PermissionManager(tmp_path, interaction=interaction, yolo=True)
    interaction.prompt.assert_not_called()


@pytest.fixture
def policy_waits(monkeypatch):
    """Expose policy-lock polling as controlled futures rather than elapsed timers."""
    waits = asyncio.Queue()

    async def sleep(seconds):
        """Park one lock poll until the test releases ownership or cancels it."""
        assert seconds == 0.01
        pending = asyncio.get_running_loop().create_future()
        waits.put_nowait(pending)
        await pending

    monkeypatch.setattr(
        manager_module, "asyncio", SimpleNamespace(**{**vars(asyncio), "sleep": sleep})
    )
    return waits


def operation(action: Action, *, tool: str = "demo", target=None, reason=None) -> Operation:
    """Build one representative typed operation."""
    return Operation(tool_id=tool, action=action, target=target, reason=reason)


def file_operation(action: Action, path) -> Operation:
    """Build one filesystem operation for a canonical path."""
    return operation(action, target=FileTarget(path=str(path)))


def path_resolution_threads(monkeypatch):
    """Record the threads that canonicalize paths through pathlib."""
    original_resolve = Path.resolve
    threads = []

    def track_resolve(path, strict=False):
        threads.append(get_ident())
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", track_resolve)
    return threads


def native_request(root: Path, source: str = "git status", **changes) -> SandboxRequest:
    """Bind one native command to disposable policy roots."""
    values = {
        "source": source,
        "cwd": root,
        "workspace": root,
        "read_roots": (root.parent,),
        "write_roots": (root,),
        "network": False,
        "environment": {"PATH": "/usr/bin:/bin", "LANG": "C"},
        "policy_version": "macos-seatbelt-v1",
        "deadline": time.monotonic() + 60,
        "workspace_id": "workspace-1",
    }
    values.update(changes)
    return SandboxRequest.create(**values)


def durable_policy_inputs(root: Path) -> tuple[dict, str, str]:
    """Create an independent workspace registry and return bound policy inputs."""
    catalog = root.parent / f"{root.name}-policy-data" / "workspaces.sqlite"
    workspace = WorkspaceRepository(catalog).initialize(Workspace.discover(root))
    arguments = {
        "workspace_root": root,
        "configuration_path": catalog.parent / "workspaces" / workspace.id / "permissions.yaml",
        "user_configuration_path": root.parent / f"{root.name}-user-permissions.yaml",
        "workspace_id": workspace.id,
        "namespace_catalog_path": catalog,
        "user_namespace_identity": str(catalog.parent),
    }
    return arguments, workspace.id, str(catalog.parent)


def test_policy_namespace_requires_explicit_trusted_first_use(tmp_path):
    """A missing durable namespace denies admission until both scoped identities initialize."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    operation = Operation(
        tool_id="read_text_file",
        action=Action.FILESYSTEM_READ,
        target=FileTarget(path=str(tmp_path / "file")),
    )

    assert manager.evaluate((operation,)).decision is Decision.DENY
    assert "recovery required" in manager.describe()
    with pytest.raises(PermissionNamespaceError, match="Trusted authorization"):
        manager.initialize_policy_namespace("workspace", workspace_id, authorized=False)
    with pytest.raises(PermissionNamespaceError, match="exact namespace identity"):
        manager.initialize_policy_namespace("workspace", "wrong", authorized=True)

    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    assert manager.evaluate((operation,)).decision is Decision.ALLOW
    assert manager.configuration.namespace_generation is not None
    assert (
        manager.configuration.limit_classifications.readable_roots is LimitClassification.APPROVAL
    )
    assert "recovery required" not in manager.describe()

    reloaded = PermissionManager(**arguments)
    assert reloaded.configuration.namespace_generation == manager.configuration.namespace_generation
    assert reloaded.evaluate((operation,)).decision is Decision.ALLOW
    with pytest.raises(PermissionNamespaceError, match="Existing policy"):
        reloaded.initialize_policy_namespace("workspace", workspace_id, authorized=True)


def test_pristine_policy_namespaces_bootstrap_without_user_steps(tmp_path):
    """The application can reserve and publish both built-in policies on first use."""
    arguments, _workspace_id, _ = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments, bootstrap_pristine_policy=True)
    operation = Operation(
        tool_id="list_folder",
        action=Action.FILESYSTEM_LIST,
        target=FileTarget(path=str(tmp_path)),
    )

    assert manager.evaluate((operation,)).decision is Decision.ALLOW
    assert manager.configuration.namespace_generation is not None
    user_path = arguments["user_configuration_path"]
    user_generation = UserPermissionConfiguration.model_validate(
        load_yaml(user_path.read_text(encoding="utf-8"))
    ).namespace_generation
    assert user_generation is not None
    assert arguments["configuration_path"].is_file()
    assert arguments["user_configuration_path"].is_file()
    manager.close()
    reloaded = PermissionManager(**arguments)
    assert reloaded.evaluate((operation,)).decision is Decision.ALLOW
    assert reloaded.configuration.namespace_generation is not None
    reloaded.close()

    second = tmp_path.parent / "second-workspace"
    second.mkdir()
    workspace = WorkspaceRepository(arguments["namespace_catalog_path"]).initialize(
        Workspace.discover(second)
    )
    other = PermissionManager(
        **{
            **arguments,
            "workspace_root": second,
            "workspace_id": workspace.id,
            "configuration_path": arguments["configuration_path"].parent.parent
            / workspace.id
            / "permissions.yaml",
        },
        bootstrap_pristine_policy=True,
    )
    assert other.configuration.namespace_generation is not None
    assert (
        UserPermissionConfiguration.model_validate(
            load_yaml(user_path.read_text(encoding="utf-8"))
        ).namespace_generation
        == user_generation
    )
    other.close()


def test_pristine_bootstrap_keeps_backup_and_initialized_missing_policy_closed(tmp_path):
    """Automatic first use never discards backup evidence or recreates a lost marked policy."""
    arguments, _workspace_id, _ = durable_policy_inputs(tmp_path)
    user_path = arguments["user_configuration_path"]
    backup = user_path.with_name(user_path.name + ".bak")
    backup.write_text(dump_yaml(UserPermissionConfiguration().model_dump()), encoding="utf-8")

    blocked = PermissionManager(**arguments, bootstrap_pristine_policy=True)
    assert blocked.configuration.namespace_generation is not None
    assert "user namespace recovery required" in blocked.describe()
    assert backup.exists() and not user_path.exists()
    blocked.close()

    backup.unlink()
    initialized = PermissionManager(**arguments, bootstrap_pristine_policy=True)
    assert (
        UserPermissionConfiguration.model_validate(
            load_yaml(user_path.read_text(encoding="utf-8"))
        ).namespace_generation
        is not None
    )
    initialized.close()
    user_path.unlink()
    missing = PermissionManager(**arguments, bootstrap_pristine_policy=True)
    assert "Initialized policy is missing" in missing.describe()
    assert not user_path.exists()
    missing.close()


def test_pristine_bootstrap_migrates_existing_user_deny_without_widening(tmp_path):
    """Automatic workspace setup keeps a valid older user denial in force."""
    arguments, _workspace_id, _ = durable_policy_inputs(tmp_path)
    user_path = arguments["user_configuration_path"]
    original = UserPermissionConfiguration(
        rules=[PermissionRule(decision=Decision.DENY, action=Action.FILESYSTEM_READ)]
    )
    user_path.write_text(dump_yaml(original.model_dump(mode="json")), encoding="utf-8")

    manager = PermissionManager(**arguments, bootstrap_pristine_policy=True)
    read = Operation(
        tool_id="read_text_file",
        action=Action.FILESYSTEM_READ,
        target=FileTarget(path=str(tmp_path / "file")),
    )
    assert manager.evaluate((read,)).decision is Decision.DENY
    assert manager.user_rules == tuple(original.rules)
    assert manager.configuration.namespace_generation is not None
    assert next(user_path.parent.glob(f"{user_path.name}.*.legacy.bak")).is_file()
    manager.close()


def test_legacy_namespace_migrates_values_and_rejects_missing_or_restored_marker(tmp_path):
    """A valid markerless policy migrates hard, while a later deletion or backup rolls back."""
    arguments, workspace_id, _ = durable_policy_inputs(tmp_path)
    path = arguments["configuration_path"]
    path.parent.mkdir(parents=True)
    original = PermissionConfiguration(
        limits=PolicyLimits(readable_roots=("workspace",), writable_roots=("workspace",))
    )
    path.write_text(dump_yaml(original.model_dump(mode="json")), encoding="utf-8")

    migrated = PermissionManager(**arguments)

    assert "workspace namespace recovery required" not in migrated.describe(), migrated.describe()
    assert migrated.configuration.limits == original.limits
    assert migrated.configuration.limit_classifications.readable_roots is LimitClassification.HARD
    assert migrated.configuration.limit_classifications.sources["readable_roots"] == "legacy"
    generation = migrated.configuration.namespace_generation
    assert generation is not None
    backup = path.with_name(f"{path.name}.{generation}.legacy.bak")
    assert backup.is_file()
    assert load_yaml(backup.read_text())["limits"] == original.model_dump(mode="json")["limits"]

    saved = path.read_bytes()
    path.unlink()
    missing = PermissionManager(**arguments)
    assert "Initialized policy is missing" in missing.describe()
    with pytest.raises(PermissionNamespaceError, match="Existing policy or backup data"):
        missing.initialize_policy_namespace("workspace", workspace_id, authorized=True)

    path.write_bytes(backup.read_bytes())
    path.chmod(0o600)
    rolled_back = PermissionManager(**arguments)
    assert "mismatched generation" in rolled_back.describe()
    path.write_bytes(saved)
    assert PermissionManager(**arguments).configuration.namespace_generation == generation


def test_markerless_legacy_policy_migrates_missing_local_limit_as_deny(tmp_path):
    """A markerless policy cannot lose a possibly truncated local-network hard deny."""
    arguments, _, _ = durable_policy_inputs(tmp_path)
    audit_path = tmp_path / "audit.db"
    arguments["audit_path"] = audit_path
    path = arguments["configuration_path"]
    path.parent.mkdir(parents=True)
    payload = PermissionConfiguration().model_dump(mode="json")
    payload["version"] = 1
    payload.pop("namespace_generation")
    payload["limits"].pop("deny_local_networks")
    original = dump_yaml(payload).encode("utf-8")
    path.write_bytes(original)
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    try:
        migrated = PermissionManager(**arguments)
        assert telemetry.close(1)
    finally:
        set_telemetry(None)

    assert migrated.configuration.limits.deny_local_networks is True
    assert migrated.configuration.version == 2
    generation = migrated.configuration.namespace_generation
    assert path.with_name(f"{path.name}.{generation}.legacy.bak").read_bytes() == original
    assert any(
        record.event_name == "permission.policy_legacy_migrated"
        and record.attributes["scope"] == "workspace"
        for record in adapter.records
    )
    with closing(sqlite3.connect(audit_path)) as connection:
        row = connection.execute(
            "SELECT payload_json FROM permission_audit_records "
            "WHERE event_name = 'permission.policy_legacy_migrated'"
        ).fetchone()
    assert json.loads(row[0]) == {"scope": "workspace", "version": 2}
    migrated.close()


def test_markerless_legacy_policy_rejects_non_mapping_limits(tmp_path):
    """Legacy schema migration cannot reinterpret a malformed limit collection."""
    arguments, _, _ = durable_policy_inputs(tmp_path)
    path = arguments["configuration_path"]
    path.parent.mkdir(parents=True)
    path.write_text("version: 1\nlimits: []\n", encoding="utf-8")

    refused = PermissionManager(**arguments)
    assert "Policy content is invalid" in refused.describe()
    assert path.read_text(encoding="utf-8") == "version: 1\nlimits: []\n"
    refused.close()


def test_user_namespace_initialization_preserves_plain_legacy_backup(tmp_path):
    """A plain user-policy backup requires restore instead of publishing fresh defaults."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    path = arguments["user_configuration_path"]
    backup = path.with_name(path.name + ".bak")
    backup.write_text(dump_yaml(UserPermissionConfiguration().model_dump()), encoding="utf-8")

    with pytest.raises(PermissionNamespaceError, match="Existing policy or backup data"):
        manager.initialize_policy_namespace("user", user_identity, authorized=True)
    assert not arguments["user_configuration_path"].exists()
    manager.close()


@pytest.mark.parametrize("content", ["", "[]\n", "0\n"])
def test_markerless_policy_rejects_empty_or_nonmapping_document(tmp_path, content):
    """Truncated or malformed legacy YAML never becomes a fresh default policy."""
    arguments, _, _ = durable_policy_inputs(tmp_path)
    path = arguments["configuration_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    manager = PermissionManager(**arguments)

    assert "Policy document must be a mapping" in manager.describe()
    assert path.read_text(encoding="utf-8") == content


def test_legacy_group_writable_policy_migrates_only_beneath_private_ancestor(tmp_path, monkeypatch):
    """A private, ACL-checked ancestor preserves an older file under a permissive umask."""
    arguments, _, _ = durable_policy_inputs(tmp_path)
    path = arguments["configuration_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version: 1\n", encoding="utf-8")
    path.chmod(0o664)

    migrated = PermissionManager(**arguments)

    assert migrated.configuration.namespace_generation is not None
    assert (
        migrated.configuration.limit_classifications.readable_roots is LimitClassification.APPROVAL
    )
    assert path.stat().st_mode & 0o777 == 0o600

    path.unlink()
    path.write_text("version: 1\n", encoding="utf-8")
    path.chmod(0o664)
    with closing(sqlite3.connect(arguments["namespace_catalog_path"])) as connection, connection:
        connection.execute("DELETE FROM policy_namespaces WHERE scope='workspace'")
    monkeypatch.setattr(
        manager_module,
        "ensure_private_directory",
        Mock(side_effect=ValueError("not private")),
    )
    refused = PermissionManager(**arguments)
    assert "Policy file provenance is invalid" in refused.describe()
    assert path.read_text(encoding="utf-8") == "version: 1\n"


def test_marked_group_writable_policy_and_aliased_parent_are_untrusted(tmp_path):
    """A marked policy cannot regain group write or move through a parent symlink."""
    arguments, workspace_id, _ = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    path = arguments["configuration_path"]
    path.chmod(0o664)
    assert "Policy file provenance is invalid" in PermissionManager(**arguments).describe()

    path.chmod(0o600)
    alias = tmp_path / "policy-alias"
    alias.symlink_to(path.parent, target_is_directory=True)
    aliased = {**arguments, "configuration_path": alias / path.name}
    assert "Policy directory has an aliased ancestor" in PermissionManager(**aliased).describe()


def test_policy_namespace_resumes_new_publication_after_interruption(tmp_path, monkeypatch):
    """An initialized reservation publishes defaults after a process dies before the file."""
    arguments, workspace_id, _ = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    original = manager._persist_configuration
    monkeypatch.setattr(
        manager, "_persist_configuration", Mock(side_effect=PermissionError("interrupted"))
    )

    with pytest.raises(PermissionError, match="interrupted"):
        manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)

    monkeypatch.setattr(manager, "_persist_configuration", original)
    recovered = PermissionManager(**arguments)
    assert recovered.configuration.namespace_generation is not None
    assert "workspace namespace recovery required" not in recovered.describe()

    catalog = arguments["namespace_catalog_path"]
    with closing(sqlite3.connect(catalog)) as connection, connection:
        connection.execute(
            "UPDATE policy_namespaces SET status='reserved_new' "
            "WHERE scope='workspace' AND identity=?",
            (workspace_id,),
        )
    resumed = PermissionManager(**arguments)
    assert (
        resumed.configuration.namespace_generation == recovered.configuration.namespace_generation
    )


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_reserved_new_namespace_refuses_later_backup_evidence(tmp_path, monkeypatch, scope):
    """A paused first-use reservation cannot discard backup evidence added before restart."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    if scope == "user":
        manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    monkeypatch.setattr(
        manager, "_persist_configuration", Mock(side_effect=PermissionError("interrupted"))
    )
    with pytest.raises(PermissionError, match="interrupted"):
        manager.initialize_policy_namespace(
            scope, workspace_id if scope == "workspace" else user_identity, authorized=True
        )
    path = arguments["configuration_path" if scope == "workspace" else "user_configuration_path"]
    backup = path.with_name(path.name + ".old.bak")
    backup.write_bytes(b"prior-restrictive-policy")

    blocked = PermissionManager(**arguments, bootstrap_pristine_policy=True)

    assert "Reserved new policy has backup evidence requiring recovery" in blocked.describe()
    assert not path.exists()
    assert backup.read_bytes() == b"prior-restrictive-policy"


def test_pristine_namespace_refuses_unverifiable_backup_listing(tmp_path, monkeypatch):
    """A directory inspection error cannot hide a backup and initialize defaults."""
    arguments, _workspace_id, _user_identity = durable_policy_inputs(tmp_path)
    path = arguments["configuration_path"]
    path.parent.mkdir(parents=True)
    backup = path.with_name(path.name + ".old.bak")
    backup.write_bytes(b"prior-restrictive-policy")
    original = os.scandir

    def refuse_policy_listing(candidate):
        """Refuse only the workspace policy's backup-evidence listing."""
        if not isinstance(candidate, int) and Path(candidate) == path.parent:
            raise PermissionError("policy listing refused")
        return original(candidate)

    monkeypatch.setattr(manager_module.os, "scandir", refuse_policy_listing)
    blocked = PermissionManager(**arguments, bootstrap_pristine_policy=True)

    assert "Policy backup evidence cannot be inspected" in blocked.describe()
    assert not path.exists()
    assert backup.read_bytes() == b"prior-restrictive-policy"


def test_policy_namespace_resumes_legacy_publication_and_checks_backup(tmp_path, monkeypatch):
    """A reserved legacy migration reuses its exact backup and rejects a changed backup."""
    arguments, _workspace_id, _ = durable_policy_inputs(tmp_path)
    path = arguments["configuration_path"]
    path.parent.mkdir(parents=True)
    path.write_text("version: 1\n", encoding="utf-8")
    original = PermissionManager._persist_configuration

    def interrupt(*_args):
        """Leave the legacy reservation and backup before publishing the migrated file."""
        raise PermissionError("interrupted")

    monkeypatch.setattr(PermissionManager, "_persist_configuration", interrupt)
    with pytest.raises(PermissionError, match="interrupted"):
        PermissionManager(**arguments)
    monkeypatch.setattr(PermissionManager, "_persist_configuration", staticmethod(original))

    backup = next(path.parent.glob(f"{path.name}.*.legacy.bak"))
    backup.write_text("different", encoding="utf-8")
    mismatched = PermissionManager(**arguments)
    assert "backup generation does not match" in mismatched.describe()

    backup.unlink()
    innocent = tmp_path / "innocent-backup"
    innocent.write_bytes(path.read_bytes())
    backup.symlink_to(innocent)
    linked = PermissionManager(**arguments)
    assert "backup provenance is invalid" in linked.describe()
    assert innocent.read_bytes() == path.read_bytes()
    backup.unlink()

    backup.write_bytes(path.read_bytes())
    backup.chmod(0o600)
    resumed = PermissionManager(**arguments)
    assert (
        resumed.configuration.limit_classifications.readable_roots is LimitClassification.APPROVAL
    )
    assert "workspace namespace recovery required" not in resumed.describe()


@pytest.mark.parametrize("failure", ["catalog_link", "policy_link", "reset_link"])
def test_policy_namespace_initialization_rejects_untrusted_existing_state(tmp_path, failure):
    """Headless and interactive initialization never replace ambiguous links or registry data."""
    arguments, workspace_id, _ = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    path = arguments["configuration_path"]
    if failure == "catalog_link":
        catalog = arguments["namespace_catalog_path"]
        original = catalog.with_suffix(".original")
        catalog.rename(original)
        catalog.symlink_to(original)
        expected = "registry is untrusted"
    elif failure == "policy_link":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(tmp_path / "absent-policy")
        expected = "Existing policy"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_name(path.name + ".reset-pending").symlink_to(tmp_path / "absent-marker")
        expected = "pending policy reset"

    with pytest.raises(PermissionNamespaceError, match=expected):
        manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)


@pytest.mark.parametrize("failure", ["missing", "malformed", "symlink", "marker", "catalog"])
def test_initialized_namespace_rejects_missing_or_untrusted_state(tmp_path, failure):
    """An initialized policy never loads permissive defaults after storage corruption."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    path = arguments["configuration_path"]
    if failure == "missing":
        path.unlink()
    elif failure == "malformed":
        path.write_text("limits: [invalid\n", encoding="utf-8")
    elif failure == "symlink":
        path.unlink()
        innocent = tmp_path / "innocent"
        innocent.write_text("secret", encoding="utf-8")
        path.symlink_to(innocent)
    elif failure == "marker":
        payload = load_yaml(path.read_text())
        payload["namespace_generation"] = "0" * 32
        path.write_text(dump_yaml(payload), encoding="utf-8")
    else:
        arguments["namespace_catalog_path"].unlink()

    reloaded = PermissionManager(**arguments)
    operation = Operation(
        tool_id="read_text_file",
        action=Action.FILESYSTEM_READ,
        target=FileTarget(path=str(tmp_path / "file")),
    )
    assert reloaded.evaluate((operation,)).decision is Decision.DENY
    assert "workspace namespace recovery required" in reloaded.describe()
    assert asyncio.run(reloaded.evaluate_async((operation,))).decision is Decision.DENY


@pytest.mark.parametrize(
    "missing",
    [
        "workspace-field",
        "limit-field",
        "local-network-field",
        "classification-field",
        "default-action",
        "user-rules",
    ],
)
def test_marked_policy_rejects_incomplete_persisted_schema(tmp_path, missing):
    """A truncated marked policy cannot restore permissive model defaults or drop denials."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    path = (
        arguments["user_configuration_path"]
        if missing == "user-rules"
        else arguments["configuration_path"]
    )
    payload = load_yaml(path.read_text())
    if missing == "workspace-field":
        payload.pop("limit_classifications")
    elif missing == "limit-field":
        payload["limits"].pop("deny_private_networks")
    elif missing == "local-network-field":
        payload["limits"].pop("deny_local_networks")
    elif missing == "classification-field":
        payload["limit_classifications"].pop("sources")
    elif missing == "default-action":
        payload["defaults"].pop(Action.FILESYSTEM_READ.value)
    else:
        payload.pop("rules")
    path.write_text(dump_yaml(payload), encoding="utf-8")

    reloaded = PermissionManager(**arguments)
    assert "Marked policy" in reloaded.describe()
    assert (
        reloaded.evaluate((file_operation(Action.FILESYSTEM_READ, tmp_path),)).decision
        is Decision.DENY
    )


@pytest.mark.parametrize("local_network", [None, False, True])
def test_marked_previous_workspace_schema_migrates_with_exact_backup(tmp_path, local_network):
    """A complete prior schema keeps its decisions and generation across publication."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    manager.close()
    path = arguments["configuration_path"]
    payload = load_yaml(path.read_text(encoding="utf-8"))
    generation = payload["namespace_generation"]
    payload["version"] = 1
    if local_network is None:
        payload["limits"].pop("deny_local_networks")
    else:
        payload["limits"]["deny_local_networks"] = local_network
    payload["defaults"][Action.FILESYSTEM_READ.value] = Decision.DENY.value
    original = dump_yaml(payload).encode("utf-8")
    path.write_bytes(original)
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    try:
        migrated = PermissionManager(**arguments)
        assert telemetry.close(1)
    finally:
        set_telemetry(None)
    assert "recovery required" not in migrated.describe()
    migrated.require_policy_ready()
    assert migrated.configuration.version == 2
    assert migrated.configuration.namespace_generation == generation
    assert migrated.configuration.defaults[Action.FILESYSTEM_READ] is Decision.DENY
    assert migrated.authorize_sandboxed_command(
        native_request(tmp_path, "pwd", read_roots=(), local_network=False), tool_id="run_command"
    )
    saved = load_yaml(path.read_text(encoding="utf-8"))
    assert saved["version"] == 2
    assert saved["limits"]["deny_local_networks"] is (
        True if local_network is None else local_network
    )
    assert saved["namespace_generation"] == generation
    backup = path.with_name(f"{path.name}.{generation}.schema-v1.bak")
    assert backup.read_bytes() == original
    assert any(
        record.event_name == "permission.policy_schema_migrated"
        and record.attributes["scope"] == "workspace"
        for record in adapter.records
    )
    migrated.close()


def test_marked_schema_migration_rejects_conflicting_archive(tmp_path):
    """A conflicting migration backup blocks admission without replacing policy data."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    manager.close()
    path = arguments["configuration_path"]
    payload = load_yaml(path.read_text(encoding="utf-8"))
    generation = payload["namespace_generation"]
    payload["version"] = 1
    payload["limits"].pop("deny_local_networks")
    original = dump_yaml(payload).encode("utf-8")
    path.write_bytes(original)
    backup = path.with_name(f"{path.name}.{generation}.schema-v1.bak")
    backup.write_text("conflict", encoding="utf-8")

    refused = PermissionManager(**arguments)
    assert "Schema policy backup generation does not match" in refused.describe()
    with pytest.raises(PermissionNamespaceError, match="Restore or repair the workspace policy"):
        refused.require_policy_ready()
    assert path.read_bytes() == original
    refused.close()


@pytest.mark.parametrize("observation", [2, 3])
def test_marked_schema_migration_refuses_changed_policy(tmp_path, monkeypatch, observation):
    """A noncooperating writer cannot publish a stale schema migration."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    manager.close()
    path = arguments["configuration_path"]
    payload = load_yaml(path.read_text(encoding="utf-8"))
    payload["version"] = 1
    payload["limits"].pop("deny_local_networks")
    path.write_text(dump_yaml(payload), encoding="utf-8")
    original_read = PermissionManager._read_namespaced_file
    observed = 0

    def changing_read(candidate):
        """Replace only the policy at the selected migration observation."""
        nonlocal observed
        if candidate == path:
            observed += 1
            if observed == observation:
                candidate.write_text("changed", encoding="utf-8")
        return original_read(candidate)

    monkeypatch.setattr(PermissionManager, "_read_namespaced_file", staticmethod(changing_read))
    refused = PermissionManager(**arguments)
    assert "Policy changed during schema migration" in refused.describe()
    assert path.read_text(encoding="utf-8") == "changed"
    refused.close()


def test_managed_policy_recovery_stops_before_model_work(tmp_path):
    """A changed administrator source cannot defer its error to repeated tool calls."""
    store = managed_store(tmp_path)
    manager = PermissionManager(tmp_path, managed_policy_store=store)
    (store.directory / "managed-policy.json").write_text("invalid", encoding="utf-8")

    with pytest.raises(PermissionNamespaceError, match="administrator repair") as raised:
        manager.require_policy_ready()
    assert str(store.expectation_path) in str(raised.value)
    manager.close()


def test_marked_previous_schema_still_rejects_other_missing_limits(tmp_path):
    """Version 1 compatibility cannot fill unrelated truncated policy controls."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    manager.close()
    path = arguments["configuration_path"]
    payload = load_yaml(path.read_text(encoding="utf-8"))
    payload["version"] = 1
    payload["limits"].pop("deny_private_networks")
    path.write_text(dump_yaml(payload), encoding="utf-8")

    refused = PermissionManager(**arguments)
    assert "Marked policy limits or provenance are incomplete" in refused.describe()
    refused.close()


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_invalid_marked_policy_stops_before_model_work_with_repair_path(tmp_path, scope):
    """A corrupt durable policy reports its scope and repair path at startup."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    manager.close()
    path = (
        arguments[f"{scope}_configuration_path"]
        if scope == "user"
        else arguments["configuration_path"]
    )
    payload = load_yaml(path.read_text(encoding="utf-8"))
    payload["version"] = 3
    path.write_text(dump_yaml(payload), encoding="utf-8")

    refused = PermissionManager(**arguments)
    with pytest.raises(PermissionNamespaceError) as raised:
        refused.require_policy_ready()
    assert str(path) in str(raised.value)
    assert f"Restore or repair the {scope} policy and registry" in str(raised.value)
    refused.close()


def test_user_namespace_migrates_legacy_rules_and_disables_missing_user_policy(tmp_path):
    """User-wide remembered grants keep identity and cannot survive an absent policy file."""
    arguments, workspace_id, _ = durable_policy_inputs(tmp_path)
    path = arguments["user_configuration_path"]
    rule = PermissionRule(decision=Decision.DENY, action=Action.FILESYSTEM_READ)
    path.write_text(
        dump_yaml(UserPermissionConfiguration(rules=[rule]).model_dump(mode="json")),
        encoding="utf-8",
    )
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    assert manager.user_rules == (rule,)
    assert load_yaml(path.read_text())["namespace_generation"] is not None
    path.unlink()
    reloaded = PermissionManager(**arguments)
    assert "user namespace recovery required" in reloaded.describe()
    assert reloaded.user_rules == ()
    operation = file_operation(Action.FILESYSTEM_READ, tmp_path / "ordinary")
    assert reloaded.evaluate((operation,)).sources == ("namespace:user:recovery_required",)
    assert asyncio.run(reloaded.evaluate_async((operation,))).sources == (
        "namespace:user:recovery_required",
    )

    safe = tmp_path / "unrelated"
    safe.write_text("version: 1\n", encoding="utf-8")
    path.symlink_to(safe)
    assert "user namespace recovery required" in PermissionManager(**arguments).describe()


def test_policy_namespace_follows_relocation_and_rejects_copied_marker(tmp_path):
    """Registry relocation retains policy while a new copied identity cannot adopt its marker."""
    root = tmp_path / "original"
    root.mkdir()
    arguments, workspace_id, _ = durable_policy_inputs(root)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    source = arguments["configuration_path"]
    saved = source.read_bytes()

    relocated = tmp_path / "relocated"
    root.rename(relocated)
    repository = WorkspaceRepository(arguments["namespace_catalog_path"])
    moved = repository.initialize(Workspace.discover(relocated))
    assert moved.id == workspace_id
    moved_arguments = {**arguments, "workspace_root": relocated}
    assert PermissionManager(**moved_arguments).configuration.namespace_generation == (
        manager.configuration.namespace_generation
    )

    copied = tmp_path / "copied"
    shutil.copytree(relocated, copied)
    copy_identity = repository.initialize(Workspace.discover(copied))
    assert copy_identity.id != workspace_id
    copy_path = source.parent.parent / copy_identity.id / source.name
    copy_path.parent.mkdir()
    copy_path.write_bytes(saved)
    copy_path.chmod(0o600)
    copy_arguments = {
        **arguments,
        "workspace_root": copied,
        "workspace_id": copy_identity.id,
        "configuration_path": copy_path,
    }
    assert (
        "marker has no independent registry anchor"
        in PermissionManager(**copy_arguments).describe()
    )


def test_namespaced_reset_interruption_requires_explicit_restore(tmp_path, monkeypatch):
    """A reset marker and archive keep admission closed until the matching policy is restored."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    path = arguments["configuration_path"]
    generation = manager.configuration.namespace_generation
    original = manager._persist_configuration
    monkeypatch.setattr(
        manager, "_persist_configuration", Mock(side_effect=PermissionError("interrupted"))
    )

    with pytest.raises(PermissionError, match="interrupted"):
        manager.reset_configuration("workspace")

    pending = PermissionManager(**arguments)
    assert "reset is pending" in pending.describe()
    with pytest.raises(PermissionNamespaceError, match="reset is pending"):
        pending.reset_configuration("workspace")
    archive = next(path.parent.glob(f"{path.name}.*.bak"))
    path.write_bytes(archive.read_bytes())
    path.chmod(0o600)
    path.with_name(path.name + ".reset-pending").unlink()
    monkeypatch.setattr(manager, "_persist_configuration", original)
    assert PermissionManager(**arguments).configuration.namespace_generation == generation


def test_user_policy_reset_archives_approvals_and_keeps_namespace_generation(tmp_path):
    """A trusted user reset archives remembered grants and preserves its ledger identity."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    manager.trust_host_command("printf private", ApprovalChoice.USER)
    path = arguments["user_configuration_path"]
    old_bytes = path.read_bytes()
    generation = load_yaml(old_bytes)["namespace_generation"]

    archive = manager.reset_configuration("user")

    assert archive is not None and archive.read_bytes() == old_bytes
    assert manager.trusted_host_command_rules() == ()
    assert not path.with_name(path.name + ".reset-pending").exists()
    reloaded = PermissionManager(**arguments)
    assert load_yaml(path.read_bytes())["namespace_generation"] == generation
    assert reloaded.trusted_host_command_rules() == ()


def test_user_reset_recovers_exact_invalid_policy_but_refuses_a_direct_edit(tmp_path, caplog):
    """A human reset may archive captured invalid bytes without overwriting a later edit."""
    path = tmp_path / "user-policy.yaml"
    path.write_text("version: invalid\n", encoding="utf-8")
    manager = PermissionManager(
        tmp_path, user_configuration_path=path, load_policy=PermissionLoadPolicy.AUTO
    )
    assert "Invalid user-wide approval policy" in caplog.text
    path.write_text("version: another-invalid\n", encoding="utf-8")

    with pytest.raises(OSError, match="changed.*reload"):
        manager.reset_configuration("user")

    assert not path.with_name(path.name + ".reset-pending").exists()
    path.write_text("version: invalid\n", encoding="utf-8")
    archive = manager.reset_configuration("user")
    assert archive is not None and archive.read_text(encoding="utf-8") == "version: invalid\n"
    assert UserPermissionConfiguration.model_validate(load_yaml(path.read_bytes()))


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_reset_never_overwrites_an_existing_archive_name(tmp_path, monkeypatch, scope):
    """A timestamp collision preserves both the active policy and an existing archive."""
    user_path = tmp_path / "user.yaml"
    manager = PermissionManager(tmp_path, user_configuration_path=user_path)
    if scope == "workspace":
        manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
        path = manager.configuration_path
    else:
        user_path.write_text(
            dump_yaml(UserPermissionConfiguration().model_dump(mode="json")),
            encoding="utf-8",
        )
        manager.reload()
        path = user_path
    original = path.read_bytes()
    fixed = manager_module.local_now()
    monkeypatch.setattr(manager_module, "local_now", lambda: fixed)
    archive = path.with_name(f"{path.name}.{fixed.strftime('%Y%m%dT%H%M%S%f%z')}.bak")
    archive.write_bytes(b"existing-archive")

    with pytest.raises(FileExistsError):
        manager.reset_configuration(scope)

    assert archive.read_bytes() == b"existing-archive"
    assert path.read_bytes() == original
    assert path.with_name(path.name + ".reset-pending").exists()


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_reset_preserves_archive_when_rename_reports_failure_after_effect(
    tmp_path, monkeypatch, scope
):
    """An exception after archive publication cannot delete the only prior policy copy."""
    user_path = tmp_path / "user.yaml"
    manager = PermissionManager(tmp_path, user_configuration_path=user_path)
    if scope == "workspace":
        manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
        path = manager.configuration_path
    else:
        user_path.write_text(
            dump_yaml(UserPermissionConfiguration().model_dump(mode="json")),
            encoding="utf-8",
        )
        manager.reload()
        path = user_path
    previous = path.read_bytes()
    original = Path.replace

    def report_after_archive(source, destination):
        """Report a failed rename only after the prior policy reached its archive."""
        result = original(source, destination)
        if source == path:
            raise PermissionError("late archive error")
        return result

    monkeypatch.setattr(Path, "replace", report_after_archive)
    with pytest.raises(PermissionError, match="late archive error"):
        manager.reset_configuration(scope)

    archive = next(path.parent.glob(f"{path.name}.*.bak"))
    assert archive.read_bytes() == previous
    assert not path.exists()
    assert path.with_name(path.name + ".reset-pending").exists()


@pytest.mark.parametrize("inspection_denied", [False, True])
def test_reset_preserves_marker_when_archive_rename_fails_before_effect(
    tmp_path, monkeypatch, inspection_denied
):
    """An early archive failure removes only a verified placeholder and keeps recovery marked."""
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    path = manager.configuration_path
    previous = path.read_bytes()
    original_replace = Path.replace
    original_lstat = Path.lstat

    def refuse_archive(source, destination):
        """Fail before moving the active policy into its reserved archive."""
        if source == path:
            raise PermissionError("archive refused")
        return original_replace(source, destination)

    def inspect(candidate, *args, **kwargs):
        """Optionally refuse inspection of the unused archive placeholder."""
        if (
            inspection_denied
            and candidate.name.startswith(path.name + ".")
            and candidate.name.endswith(".bak")
        ):
            raise PermissionError("archive inspection refused")
        return original_lstat(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "replace", refuse_archive)
    monkeypatch.setattr(Path, "lstat", inspect)
    with pytest.raises(PermissionError, match="archive refused"):
        manager.reset_configuration("workspace")

    archives = tuple(path.parent.glob(f"{path.name}.*.bak"))
    assert len(archives) == int(inspection_denied)
    assert path.read_bytes() == previous
    assert path.with_name(path.name + ".reset-pending").exists()


@pytest.mark.parametrize("yolo", [False, True])
@pytest.mark.parametrize("headless", [False, True])
def test_interrupted_user_reset_blocks_restart_and_yolo(tmp_path, monkeypatch, yolo, headless):
    """A user reset interrupted after archival never loads empty remembered policy."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    manager.trust_host_command("printf private", ApprovalChoice.USER)
    path = arguments["user_configuration_path"]
    original = path.read_bytes()
    monkeypatch.setattr(
        manager, "_persist_configuration", Mock(side_effect=PermissionError("publication refused"))
    )

    with pytest.raises(PermissionError, match="publication refused"):
        manager.reset_configuration("user")

    marker = path.with_name(path.name + ".reset-pending")
    archive = next(path.parent.glob(f"{path.name}.*.bak"))
    assert marker.exists() and archive.read_bytes() == original
    assert not path.exists()
    interaction = None if headless else Mock(spec=Interaction)
    blocked = PermissionManager(**arguments, yolo=yolo, interaction=interaction)
    assert "reset is pending" in blocked.describe()
    if interaction is not None:
        interaction.prompt.assert_not_called()
    with pytest.raises(PermissionNamespaceError, match="reset is pending"):
        blocked.reset_configuration("user")
    archive.replace(path)
    marker.unlink()
    recovered = PermissionManager(**arguments, yolo=yolo)
    assert len(recovered.trusted_host_command_rules()) == 1


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_interrupted_standalone_policy_reset_denies_active_grants_until_reload(
    tmp_path, monkeypatch, scope
):
    """A loaded manager cannot reuse an allow while its own reset marker remains."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.USER
    user_path = tmp_path / "user.yaml"
    manager = PermissionManager(
        tmp_path, user_configuration_path=user_path, interaction=interaction
    )
    request = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.test", origin="https://example.test"),
    )
    if scope == "workspace":
        manager.set_default(Action.NETWORK_REQUEST, Decision.ALLOW)
    else:
        assert manager.authorize((request,)).decision is Decision.ALLOW
    assert manager.evaluate((request,)).decision is Decision.ALLOW
    path = manager.configuration_path if scope == "workspace" else user_path
    monkeypatch.setattr(
        manager, "_persist_configuration", Mock(side_effect=PermissionError("reset stopped"))
    )

    with pytest.raises(PermissionError, match="reset stopped"):
        manager.reset_configuration(scope)

    marker = path.with_name(path.name + ".reset-pending")
    archive = next(path.parent.glob(f"{path.name}.*.bak"))
    assert marker.exists()
    assert manager.evaluate((request,)).sources == (f"namespace:{scope}:recovery_required",)
    archive.replace(path)
    marker.unlink()
    assert manager.evaluate((request,)).decision is Decision.DENY
    manager.reload()
    assert manager.evaluate((request,)).decision is Decision.ALLOW


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_live_manager_denies_a_new_reset_marker_in_either_namespace(tmp_path, scope):
    """A reset started by another writer blocks an already loaded manager's decisions."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    path = arguments["configuration_path" if scope == "workspace" else "user_configuration_path"]
    marker = path.with_name(path.name + ".reset-pending")
    marker.touch()

    decision = manager.evaluate(
        (operation(Action.FILESYSTEM_READ, target=FileTarget(path=str(tmp_path / "ordinary"))),)
    )

    assert decision.decision is Decision.DENY
    assert decision.sources == (f"namespace:{scope}:recovery_required",)
    assert "reset is pending" in decision.reason


def test_policy_namespace_argument_and_protected_storage_guards(tmp_path):
    """Namespace APIs refuse missing identities and a workspace containing app storage."""
    with pytest.raises(ValueError, match="require workspace and user paths and identities"):
        PermissionManager(tmp_path, namespace_catalog_path=tmp_path / "catalog")
    ephemeral = PermissionManager(tmp_path)
    with pytest.raises(ValueError, match="unavailable"):
        ephemeral.policy_namespace_identity("workspace")
    with pytest.raises(ValueError, match="not configured"):
        ephemeral.initialize_policy_namespace("workspace", "id", authorized=True)

    arguments, workspace_id, _ = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    with pytest.raises(ValueError, match="unavailable"):
        manager.policy_namespace_identity("invalid")
    with pytest.raises(ValueError, match="Unknown permission namespace scope"):
        manager.initialize_policy_namespace("invalid", workspace_id, authorized=True)
    missing = PermissionManager(**{**arguments, "namespace_catalog_path": tmp_path / "missing"})
    with pytest.raises(PermissionNamespaceError, match="missing"):
        missing.initialize_policy_namespace("workspace", workspace_id, authorized=True)

    nested = tmp_path / "app-data"
    nested.mkdir()
    assert (
        "workspace intersects protected application storage"
        in PermissionManager(**{**arguments, "protected_host_roots": (nested,)}).describe()
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"workspace_id": None},
        {"user_namespace_identity": None},
        {"workspace_root": None},
        {"user_configuration_path": None},
        {"configuration": PermissionConfiguration()},
    ],
)
def test_durable_policy_namespace_rejects_incomplete_or_injected_inputs(tmp_path, changes):
    """Callers cannot bypass linked policy storage with absent identity or injected defaults."""
    arguments, _, _ = durable_policy_inputs(tmp_path)
    with pytest.raises(ValueError, match="require workspace and user paths and identities"):
        PermissionManager(**{**arguments, **changes})


def test_policy_namespace_rejects_existing_reservation_and_identity_changes(tmp_path):
    """A creator cannot reset a reserved namespace or adopt a changed registry identity."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    path = arguments["configuration_path"]
    path.unlink()
    with pytest.raises(PermissionNamespaceError, match="Existing namespace reservation"):
        manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    with closing(sqlite3.connect(arguments["namespace_catalog_path"])) as connection, connection:
        connection.execute(
            "UPDATE workspace_locations SET canonical_path=? WHERE workspace_id=?",
            (str(tmp_path / "elsewhere"), workspace_id),
        )
    with pytest.raises(PermissionNamespaceError, match="Workspace identity"):
        manager.initialize_policy_namespace("user", user_identity, authorized=True)
    assert "Workspace policy identity does not match" in PermissionManager(**arguments).describe()


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("reserved_new_mismatch", "Reserved new policy generation does not match"),
        ("reserved_legacy_missing", "Reserved legacy policy is missing"),
        ("reserved_legacy_mismatch", "Reserved policy generation"),
        ("invalid_status", "reservation is invalid"),
    ],
)
def test_policy_namespace_rejects_inconsistent_reservations(tmp_path, failure, expected):
    """Only a matching policy generation may complete an interrupted registry transaction."""
    arguments, workspace_id, _ = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    path = arguments["configuration_path"]
    status = "reserved_new" if failure == "reserved_new_mismatch" else "reserved_legacy"
    if failure == "invalid_status":
        status = "unknown"
    with closing(sqlite3.connect(arguments["namespace_catalog_path"])) as connection, connection:
        connection.execute(
            "UPDATE policy_namespaces SET status=? WHERE scope='workspace' AND identity=?",
            (status, workspace_id),
        )
    if failure == "reserved_legacy_missing":
        path.unlink()
    elif failure in {"reserved_new_mismatch", "reserved_legacy_mismatch"}:
        payload = load_yaml(path.read_text())
        payload["namespace_generation"] = "0" * 32
        path.write_text(dump_yaml(payload), encoding="utf-8")
    assert expected in PermissionManager(**arguments).describe()


def test_concurrent_policy_namespace_creators_publish_only_one_generation(tmp_path):
    """Competing trusted creators leave one usable reservation and policy generation."""
    arguments, workspace_id, _ = durable_policy_inputs(tmp_path)
    managers = (PermissionManager(**arguments), PermissionManager(**arguments))
    start = Event()
    outcomes = []

    def create(manager):
        """Attempt one explicit first-use publication after both managers are ready."""
        start.wait()
        try:
            manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
        except (PermissionNamespaceError, BlockingIOError) as error:
            outcomes.append(type(error))
        else:
            outcomes.append(None)

    threads = [Thread(target=create, args=(manager,)) for manager in managers]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert outcomes.count(None) == 1
    assert len(outcomes) == 2
    assert PermissionManager(**arguments).configuration.namespace_generation is not None


def test_legacy_migration_resumes_after_policy_publication(tmp_path):
    """A crash before the ledger commit recognizes the already published matching generation."""
    arguments, workspace_id, _ = durable_policy_inputs(tmp_path)
    path = arguments["configuration_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version: 1\n", encoding="utf-8")
    migrated = PermissionManager(**arguments)
    generation = migrated.configuration.namespace_generation
    with closing(sqlite3.connect(arguments["namespace_catalog_path"])) as connection, connection:
        connection.execute(
            "UPDATE policy_namespaces SET status='reserved_legacy' "
            "WHERE scope='workspace' AND identity=?",
            (workspace_id,),
        )
    assert PermissionManager(**arguments).configuration.namespace_generation == generation


def test_sandbox_read_scope_rejects_protected_application_storage(tmp_path):
    """A broad external read request cannot expose Loop's policy registry or state."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    data = tmp_path / "app-data"
    data.mkdir()
    manager = PermissionManager(workspace, protected_host_roots=(data,))
    request = native_request(workspace, read_roots=(data,))
    assert "intersects protected application policy or state" in manager.sandbox_scope_limit(
        request
    )


def test_namespace_identity_and_recovery_block_policy_writes(tmp_path):
    """Invalid user identity and unresolved namespaces cannot publish policy changes."""
    arguments, _workspace_id, _ = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    with pytest.raises(PermissionNamespaceError, match="missing or ambiguous"):
        manager._persist_checked(arguments["configuration_path"], manager.configuration)
    with pytest.raises(PermissionNamespaceError, match="missing or ambiguous"):
        manager._persist_checked(arguments["user_configuration_path"], manager._user_configuration)
    with pytest.raises(PermissionNamespaceError, match="identity is unavailable"):
        manager._read_namespaced_configuration(arguments["configuration_path"], "workspace", None)

    wrong = PermissionManager(**{**arguments, "user_namespace_identity": "wrong-user"})
    assert "User policy identity does not match" in wrong.describe()
    with pytest.raises(PermissionNamespaceError, match="User identity does not match"):
        wrong.initialize_policy_namespace("user", "wrong-user", authorized=True)


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("initial_stat", "errno 5"),
        ("second_stat", "errno 5"),
        ("changed_inode", "Policy file changed during loading"),
        ("no_nofollow", "No-follow policy loading is unavailable"),
        ("unsafe_mode", "Policy file provenance is invalid"),
        ("hardlink", "Policy file provenance is invalid"),
    ],
)
def test_namespace_policy_reader_rejects_storage_races(tmp_path, monkeypatch, failure, expected):
    """Policy loading refuses changed file identities, unsafe modes and missing OS checks."""
    arguments, workspace_id, _ = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    path = arguments["configuration_path"]
    if failure == "unsafe_mode":
        path.chmod(0o666)
    elif failure == "hardlink":
        os.link(path, tmp_path / "linked-policy")
    elif failure == "no_nofollow":
        monkeypatch.delattr(manager_module.os, "O_NOFOLLOW")
    else:
        original_lstat = Path.lstat
        calls = 0

        def changing_lstat(candidate):
            """Simulate an inaccessible or replaced policy inode at a chosen read point."""
            nonlocal calls
            if candidate == path:
                calls += 1
                if failure == "initial_stat" and calls == 1:
                    raise OSError(5, "stat interrupted")
                if calls == 2:
                    if failure == "second_stat":
                        raise OSError(5, "stat interrupted")
                    if failure == "changed_inode":
                        return SimpleNamespace(st_dev=-1, st_ino=-1)
            return original_lstat(candidate)

        monkeypatch.setattr(Path, "lstat", changing_lstat)
    with pytest.raises(PermissionNamespaceError, match=expected):
        manager._read_namespaced_configuration(path, "workspace", workspace_id)


def test_namespace_rejects_invalid_registry_schema(tmp_path):
    """A registry missing the independent workspace identity table denies loading."""
    arguments, workspace_id, _ = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    with closing(sqlite3.connect(arguments["namespace_catalog_path"])) as connection, connection:
        connection.execute("DROP TABLE workspace_locations")
    with pytest.raises(PermissionNamespaceError, match="registry is invalid"):
        manager._read_namespaced_configuration(
            arguments["configuration_path"], "workspace", workspace_id
        )


@pytest.mark.parametrize("changed_scope", ["workspace", "user"])
def test_live_namespace_rejects_another_managers_policy_change(tmp_path, changed_scope):
    """A loaded manager denies stale policy bytes before the next tool admission."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    first = PermissionManager(**arguments)
    first.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    first.initialize_policy_namespace("user", user_identity, authorized=True)
    second = PermissionManager(**arguments)
    operation = file_operation(Action.FILESYSTEM_READ, tmp_path / "ordinary")
    assert first.evaluate((operation,)).decision is Decision.ALLOW
    if changed_scope == "workspace":
        second.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    else:
        updated = second._user_configuration.model_copy(
            update={
                "rules": [PermissionRule(decision=Decision.DENY, action=Action.FILESYSTEM_READ)]
            },
            deep=True,
        )
        second._replace_user_configuration(updated)

    stale = first.evaluate((operation,))
    assert stale.decision is Decision.DENY
    assert stale.sources == (f"namespace:{changed_scope}:recovery_required",)
    assert asyncio.run(first.evaluate_async((operation,))).decision is Decision.DENY
    first.reload()
    assert first.evaluate((operation,)).decision is Decision.DENY


@pytest.mark.parametrize("confirmed", [True, False])
def test_live_namespace_rechecks_registry_and_final_policy_revision(tmp_path, confirmed):
    """A rekeyed workspace revokes egress and denies a pending sandbox revision."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    operation = file_operation(Action.FILESYSTEM_READ, tmp_path / "ordinary")
    _decision, revision = manager.evaluate_with_revision((operation,))
    lease = Mock()
    lease.revoke_revision.return_value = confirmed
    manager.bind_egress(manager.egress_policy_revision(), lease)
    with closing(sqlite3.connect(arguments["namespace_catalog_path"])) as connection, connection:
        connection.execute(
            "UPDATE workspace_locations SET canonical_path=? WHERE workspace_id=?",
            (str(tmp_path / "rekeyed"), workspace_id),
        )

    if confirmed:
        assert not manager.sandbox_policy_revision_current(revision)
    else:
        with pytest.raises(OSError, match="policy transition is uncertain"):
            manager.sandbox_policy_revision_current(revision)
    lease.revoke_revision.assert_called_once_with(manager.egress_policy_revision())
    manager.unbind_egress(lease)
    assert manager.evaluate((operation,)).sources == ("namespace:workspace:recovery_required",)
    with pytest.raises(PermissionNamespaceError, match="Workspace identity changed"):
        manager.reset_configuration("workspace")


@pytest.mark.parametrize("confirmed", [True, False])
def test_sandbox_revision_validation_exception_closes_live_egress(tmp_path, monkeypatch, confirmed):
    """An interrupted authority inspection closes active egress before failing."""
    store = managed_store(tmp_path)
    manager = PermissionManager(tmp_path, managed_policy_store=store)
    revision = manager.egress_policy_revision()
    lease = Mock()
    lease.revoke_revision.return_value = confirmed
    manager.bind_egress(revision, lease)
    monkeypatch.setattr(store, "load", Mock(side_effect=RuntimeError("inspection interrupted")))

    error = RuntimeError if confirmed else OSError
    message = "inspection interrupted" if confirmed else "revocation closure is uncertain"
    with pytest.raises(error, match=message):
        manager.sandbox_policy_revision_current(revision)
    lease.revoke_revision.assert_called_once_with(revision + 1)
    manager.unbind_egress(lease)
    manager.close()


def test_live_namespace_refuses_catalog_link_swapped_after_initial_validation(
    tmp_path, monkeypatch
):
    """A copied registry reached through a final-component link cannot approve a request."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    catalog = arguments["namespace_catalog_path"]
    copied = catalog.with_suffix(".copy")
    original_validation = manager._validate_namespace_catalog
    swapped = False

    def swap_after_validation(path, candidate):
        """Replace the registry path only after the first provenance result is returned."""
        nonlocal swapped
        identity = original_validation(path, candidate)
        if not swapped:
            copied.write_bytes(catalog.read_bytes())
            catalog.unlink()
            catalog.symlink_to(copied)
            swapped = True
        return identity

    monkeypatch.setattr(manager, "_validate_namespace_catalog", swap_after_validation)
    result = manager.evaluate((file_operation(Action.FILESYSTEM_READ, tmp_path / "ordinary"),))
    assert result.sources == ("namespace:workspace:recovery_required",)
    assert catalog.is_symlink()


@pytest.mark.parametrize("failure", ["generation", "inode", "sqlite", "missing_path"])
def test_live_namespace_refuses_changed_registry_provenance(tmp_path, monkeypatch, failure):
    """A changed registry or inaccessible namespace path denies a loaded policy."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    catalog = arguments["namespace_catalog_path"]
    if failure == "generation":
        with closing(sqlite3.connect(catalog)) as connection, connection:
            connection.execute(
                "UPDATE policy_namespaces SET status='reserved_new' WHERE scope='user'"
            )
    elif failure == "inode":
        original_validation = manager._validate_namespace_catalog
        validations = 0

        def replace_before_final_validation(path, candidate):
            """Replace a valid registry inode after the first read transaction."""
            nonlocal validations
            validations += 1
            if validations == 2:
                replacement = catalog.with_suffix(".replacement")
                replacement.write_bytes(catalog.read_bytes())
                replacement.chmod(catalog.stat().st_mode & 0o777)
                replacement.replace(catalog)
            return original_validation(path, candidate)

        monkeypatch.setattr(manager, "_validate_namespace_catalog", replace_before_final_validation)
    elif failure == "sqlite":
        monkeypatch.setattr(
            manager_module.sqlite3,
            "connect",
            Mock(side_effect=sqlite3.OperationalError("registry unavailable")),
        )
    else:
        manager._user_configuration_path = None

    result = manager.evaluate((file_operation(Action.FILESYSTEM_READ, tmp_path / "ordinary"),))
    assert result.decision is Decision.DENY
    assert result.sources == (
        f"namespace:{'user' if failure == 'generation' else 'workspace'}:recovery_required",
    )


def test_namespaced_async_evaluation_and_final_revision_check(tmp_path):
    """Async admission rechecks a valid namespace and revokes a stale approval revision."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    target = file_operation(Action.FILESYSTEM_READ, tmp_path / "ordinary")
    assert asyncio.run(manager.evaluate_async((target,))).decision is Decision.ALLOW
    _, revision = manager.evaluate_with_revision((target,))
    assert asyncio.run(manager._sandbox_policy_revision_current_async(revision))
    other = PermissionManager(**arguments)
    other.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    assert not asyncio.run(manager._sandbox_policy_revision_current_async(revision))


@pytest.mark.parametrize("closure_confirmed", [False, True])
def test_async_sandbox_admission_revokes_live_egress_on_namespace_change(
    tmp_path, closure_confirmed
):
    """A changed namespace closes existing egress before async admission settles."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    inspection = Mock(spec=CommandInspection)
    manager = PermissionManager(**arguments, command_inspection=inspection)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    other = PermissionManager(**arguments)
    revision = manager.egress_policy_revision()
    lease = Mock()
    lease.revoke_revision.return_value = closure_confirmed
    manager.bind_egress(revision, lease)

    def change_namespace(*_args, **_kwargs):
        """Publish a competing policy update after the first decision snapshot."""
        other.set_default(Action.FILESYSTEM_READ, Decision.DENY)
        return ()

    inspection.inspect.side_effect = change_namespace
    request = native_request(tmp_path, "printf safe", read_roots=())
    try:
        if closure_confirmed:
            assert not asyncio.run(
                manager.authorize_sandboxed_command_async(request, tool_id="run_command")
            )
        else:
            with pytest.raises(OSError, match="policy transition is uncertain"):
                asyncio.run(
                    manager.authorize_sandboxed_command_async(request, tool_id="run_command")
                )
        lease.revoke_revision.assert_called_once_with(revision + 1)
    finally:
        manager.unbind_egress(lease)
        manager.close()
        other.close()


def test_cancelled_async_sandbox_admission_does_not_publish_approval_revision(tmp_path):
    """A cancelled final check cannot leave an approved revision for launch."""
    manager = PermissionManager(tmp_path)
    request = native_request(tmp_path, "printf safe", read_roots=())
    approved_revision = []

    async def cancel_final_check():
        """Cancel admission after the worker completes its final revision check."""
        checked = asyncio.Event()
        release = asyncio.Event()

        async def paused_runner(function, *args):
            """Hold the completed final revision result until the caller is cancelled."""
            result = await asyncio.to_thread(function, *args)
            if function == manager._sandbox_policy_revision_unless_busy:
                checked.set()
                await release.wait()
            return result

        manager.set_async_blocking_runner(paused_runner)
        pending = asyncio.create_task(
            manager.authorize_sandboxed_command_async(
                request, tool_id="run_command", approved_revision=approved_revision
            )
        )
        await checked.wait()
        pending.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending

    try:
        asyncio.run(cancel_final_check())
        assert approved_revision == []
    finally:
        manager.close()


def test_async_sandbox_admission_waits_without_occupying_worker_on_policy_lock(
    tmp_path, policy_waits
):
    """A busy policy lock leaves workers available and admission resumes once it is free."""
    inspection = Mock(spec=CommandInspection)
    manager = PermissionManager(tmp_path, command_inspection=inspection)
    manager.set_async_blocking_runner(lambda function, *args: asyncio.to_thread(function, *args))
    request = native_request(tmp_path, "printf safe", read_roots=())
    entered = Event()
    release = Event()

    def hold_policy_lock():
        """Keep the real policy lock occupied through the first final check."""
        with manager.policy_update():
            entered.set()
            assert release.wait(timeout=3)

    holder = Thread(target=hold_policy_lock)

    def inspect(*_args, **_kwargs):
        """Begin lock contention after the async decision snapshot is complete."""
        holder.start()
        assert entered.wait(timeout=3)
        return ()

    inspection.inspect.side_effect = inspect

    async def authorize_after_lock_release():
        """Advance the retry only after the worker has observed contention."""
        approved_revision = []
        pending = asyncio.create_task(
            manager.authorize_sandboxed_command_async(
                request, tool_id="run_command", approved_revision=approved_revision
            )
        )
        retry = await policy_waits.get()
        release.set()
        holder.join(timeout=3)
        assert not holder.is_alive()
        retry.set_result(None)
        assert await pending
        assert approved_revision == [manager.egress_policy_revision()]

    try:
        asyncio.run(authorize_after_lock_release())
    finally:
        release.set()
        if holder.ident is not None:
            holder.join(timeout=3)
        manager.close()


@pytest.mark.parametrize("boundary", ["evaluation", "final_revision"])
def test_namespaced_async_validation_settles_before_cancellation(tmp_path, boundary):
    """Cancellation waits for the namespace read and cannot release a stale result."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    target = file_operation(Action.FILESYSTEM_READ, tmp_path / "ordinary")
    _, revision = manager.evaluate_with_revision((target,))
    started = asyncio.Event()
    release = asyncio.Event()

    async def paused_worker(function, *args):
        """Keep the validation worker owned until its cancelled caller settles."""
        started.set()
        await release.wait()
        return await asyncio.to_thread(function, *args)

    manager.set_async_blocking_runner(paused_worker)

    async def cancel_check():
        """Cancel the chosen admission boundary while its worker is pending."""
        pending = asyncio.create_task(
            manager.evaluate_async((target,))
            if boundary == "evaluation"
            else manager._sandbox_policy_revision_current_async(revision)
        )
        await started.wait()
        pending.cancel()
        await deliver_callbacks()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(cancel_check())


def test_namespaced_reset_rechecks_policy_before_archiving(tmp_path):
    """A second manager's write prevents reset from replacing its new policy."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    other = PermissionManager(**arguments)
    other.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    with pytest.raises(PermissionNamespaceError, match="recovery required"):
        manager.reset_configuration("workspace")
    assert (
        PermissionManager(**arguments).configuration.defaults[Action.FILESYSTEM_READ]
        is Decision.DENY
    )


def test_namespaced_save_rejects_same_byte_policy_symlink(tmp_path):
    """A final-component link cannot pass a revision check by copying the loaded bytes."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    path = arguments["configuration_path"]
    same_bytes = tmp_path / "policy-copy"
    same_bytes.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(same_bytes)

    with pytest.raises(PermissionNamespaceError, match="symbolic links are not allowed"):
        manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    assert path.is_symlink()
    assert same_bytes.read_bytes() != b""


def test_namespaced_save_rechecks_final_policy_inode_after_live_validation(tmp_path, monkeypatch):
    """A same-byte symlink swapped between live validation and save cannot be published."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    path = arguments["configuration_path"]
    same_bytes = tmp_path / "policy-copy"
    original_bytes = path.read_bytes()
    same_bytes.write_bytes(original_bytes)
    original_check = manager._namespace_recovery_decision

    def swap_after_validation():
        """Replace only the final component after namespace admission succeeded."""
        result = original_check()
        path.unlink()
        path.symlink_to(same_bytes)
        return result

    monkeypatch.setattr(manager, "_namespace_recovery_decision", swap_after_validation)
    with pytest.raises(PermissionNamespaceError, match="symbolic links are not allowed"):
        manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    assert path.is_symlink()
    assert same_bytes.read_bytes() == original_bytes


@pytest.mark.parametrize("change", ["missing", "writable_mode"])
def test_namespaced_save_rechecks_final_policy_provenance_after_live_validation(
    tmp_path, monkeypatch, change
):
    """A removed or group-writable policy cannot pass the final save comparison."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    path = arguments["configuration_path"]
    original_bytes = path.read_bytes()
    original_check = manager._namespace_recovery_decision

    def alter_after_validation():
        """Change file provenance only after the live namespace check passes."""
        result = original_check()
        if change == "missing":
            path.unlink()
        else:
            path.chmod(0o666)
        return result

    monkeypatch.setattr(manager, "_namespace_recovery_decision", alter_after_validation)
    with pytest.raises(PermissionNamespaceError, match="Policy file provenance is invalid"):
        manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    if change == "writable_mode":
        assert path.read_bytes() == original_bytes
    else:
        assert not path.exists()


def test_legacy_namespace_refuses_change_during_migration(tmp_path, monkeypatch):
    """An edited legacy file between reservation and publication stays untouched."""
    arguments, _workspace_id, _ = durable_policy_inputs(tmp_path)
    path = arguments["configuration_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("version: 1\n", encoding="utf-8")
    monkeypatch.setattr(PermissionManager, "_read_storage_revision", Mock(return_value=b"changed"))
    assert "Legacy policy changed during migration" in PermissionManager(**arguments).describe()
    assert path.read_text(encoding="utf-8") == "version: 1\n"


def test_git_sandbox_roots_ignore_commands_without_git_intent(tmp_path):
    """An ordinary command receives no Git metadata authority or creation intent."""
    assert PermissionManager.git_sandbox_roots(tmp_path, read=False, write=False) == (
        (),
        (),
        False,
    )


def test_host_rule_stores_only_virtual_context_and_checks_runtime_binding(tmp_path):
    """A host rule contains no real root and cannot cross to another concrete workspace."""
    other = tmp_path.parent / "other-host-workspace"
    other.mkdir()
    manager = PermissionManager(tmp_path)
    first = native_request(tmp_path, "cat $WORKSPACE_ROOT/file", read_roots=())
    moved = native_request(other, "cat $WORKSPACE_ROOT/file", read_roots=())

    rule = manager.remember_host_command_rule(first)

    assert first.host_command_signature() == moved.host_command_signature()
    assert tmp_path.as_posix() not in rule.model_dump_json()
    assert other.as_posix() not in rule.model_dump_json()
    assert rule.cwd == "$WORKSPACE_ROOT"
    assert rule.source == "<path-bearing-command>"
    assert manager.matching_host_command_rule(first) == rule
    assert manager.matching_host_command_rule(moved) is None
    literal_host = native_request(tmp_path, f"cat {tmp_path / 'private'}", read_roots=())
    literal_rule = manager.remember_host_command_rule(literal_host)
    assert tmp_path.as_posix() not in literal_rule.model_dump_json()
    assert literal_rule.source == "<path-bearing-command>"
    assert manager.matching_host_command_rule(literal_host) == literal_rule
    nested = tmp_path / "nested"
    nested.mkdir()
    nested_rule = manager.remember_host_command_rule(
        native_request(tmp_path, "printf nested", cwd=nested, read_roots=())
    )
    assert nested_rule.cwd == "$WORKSPACE_ROOT/nested"
    assert manager.reset_session()
    assert manager.matching_host_command_rule(first) is None


def test_host_rule_preparation_rejects_session_reset(tmp_path, monkeypatch):
    """A reset during preparation cannot publish an approval from the old session."""
    manager = PermissionManager(tmp_path)
    request = native_request(tmp_path)
    original = SandboxRequest.host_command_runtime_signature

    def reset_during_signature(candidate):
        """Reset the empty session while its approval is being prepared."""
        assert not manager.reset_session()
        return original(candidate)

    monkeypatch.setattr(SandboxRequest, "host_command_runtime_signature", reset_during_signature)
    with pytest.raises(
        ValueError, match="^Session host rules changed while preparing the approval\\.$"
    ):
        manager.remember_host_command_rule(request)
    assert manager.host_command_rules() == ()
    manager.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("change", ["none", "policy", "host"])
def test_revision_bound_host_rule_rejects_changes_during_preparation(
    tmp_path, monkeypatch, asynchronous, change
):
    """A prepared host approval commits only against its original policy and session."""
    manager = PermissionManager(tmp_path)
    request = native_request(tmp_path)
    existing = manager.remember_host_command_rule(request) if change == "host" else None
    _, revision = manager.evaluate_with_revision(())
    original = SandboxRequest.host_command_runtime_signature

    def prepare(candidate):
        """Change state while the filesystem identity is being prepared."""
        if change == "policy":
            manager.set_default(Action.PROCESS_EXECUTE, Decision.DENY, scope=PolicyScope.SESSION)
        elif change == "host":
            assert existing is not None
            assert manager.remove_host_command_rule(existing.id)
        return original(candidate)

    monkeypatch.setattr(SandboxRequest, "host_command_runtime_signature", prepare)
    result = (
        asyncio.run(manager.remember_host_command_rule_if_current_async(request, revision))
        if asynchronous
        else manager.remember_host_command_rule_if_current(request, revision)
    )
    if change == "none":
        assert result is not None
        assert manager.host_command_rules() == (result,)
    else:
        assert result is None
        assert manager.host_command_rules() == ()
    manager.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_revision_bound_host_rule_rejects_peer_policy_change(tmp_path, monkeypatch, asynchronous):
    """A peer policy write before rule publication cannot preserve stale host authority."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    peer = PermissionManager(**arguments)
    request = native_request(tmp_path)
    _, revision = manager.evaluate_with_revision(())
    original = SandboxRequest.host_command_runtime_signature

    def changed(candidate):
        """Change durable policy after the local revision has been captured."""
        peer.set_default(Action.PROCESS_EXECUTE, Decision.DENY)
        return original(candidate)

    monkeypatch.setattr(SandboxRequest, "host_command_runtime_signature", changed)
    result = (
        asyncio.run(manager.remember_host_command_rule_if_current_async(request, revision))
        if asynchronous
        else manager.remember_host_command_rule_if_current(request, revision)
    )
    assert result is None
    assert manager.host_command_rules() == ()
    peer.close()
    manager.close()


def test_trusted_host_grant_binds_exact_root_source_and_registered_tool(tmp_path):
    """A human session grant matches only its exact command and live workspace root."""
    manager = PermissionManager(tmp_path)
    with pytest.raises(ValueError, match="exact host command"):
        manager.trust_host_command(" ")
    with pytest.raises(ValueError, match="exact host command"):
        manager.trust_host_command("printf host", ApprovalChoice.ONCE)
    with pytest.raises(ValueError, match="validated durable user permission policy"):
        manager.trust_host_command("printf host", ApprovalChoice.WORKSPACE)
    rule = manager.trust_host_command(f"cat {tmp_path / 'private'}")
    assert rule.source == "<exact command>"
    assert str(tmp_path) not in repr(rule)
    assert manager.trusted_host_command_rules() == (rule,)
    assert (
        manager.matching_trusted_host_command_rule(
            f"cat {tmp_path / 'private'}", tmp_path, "request_host_command"
        )
        == rule
    )
    assert (
        manager.matching_trusted_host_command_rule("cat other", tmp_path, "request_host_command")
        is None
    )
    assert (
        manager.matching_trusted_host_command_rule(
            f"cat {tmp_path / 'private'}", tmp_path.parent, "request_host_command"
        )
        is None
    )
    assert (
        manager.matching_trusted_host_command_rule(
            f"cat {tmp_path / 'private'}", tmp_path, "run_command"
        )
        is None
    )
    assert (
        manager.matching_trusted_host_command_rule(
            f"cat {tmp_path / 'private'}",
            tmp_path,
            "request_host_command",
            workspace=tmp_path.parent,
        )
        is None
    )
    assert manager.remove_trusted_host_command_rule(rule.id)
    assert not manager.remove_trusted_host_command_rule(rule.id)
    assert manager.trusted_host_command_rules() == ()
    manager.close()


def test_trusted_host_grant_refuses_missing_or_untrusted_workspace(tmp_path):
    """No grant can be installed without a live workspace and valid policy namespace."""
    without_root = PermissionManager()
    with pytest.raises(ValueError, match="current workspace"):
        without_root.trust_host_command("printf host")
    without_root.close()

    root = tmp_path / "workspace"
    root.mkdir()
    manager = PermissionManager(root)
    manager.trust_host_command("printf host")
    root.rename(tmp_path / "moved")
    assert (
        manager.matching_trusted_host_command_rule("printf host", root, "request_host_command")
        is None
    )
    with pytest.raises(ValueError, match="identity is unavailable"):
        manager.trust_host_command("printf host")
    manager.close()

    arguments, _workspace_id, _user_identity = durable_policy_inputs(tmp_path / "moved")
    uninitialized = PermissionManager(**arguments)
    with pytest.raises(ValueError, match="recovery is required"):
        uninitialized.trust_host_command("printf host")
    assert (
        uninitialized.matching_trusted_host_command_rule(
            "printf host", tmp_path / "moved", "request_host_command"
        )
        is None
    )
    uninitialized.close()


def test_host_grants_require_validated_namespace_for_durable_scopes(tmp_path):
    """A standalone user policy cannot create or activate persistent host authority."""
    user_path = tmp_path.parent / f"{tmp_path.name}-user-policy.yaml"
    manager = PermissionManager(tmp_path, user_configuration_path=user_path)
    choices, _ = manager.host_retry_choices(False, trusted=True)
    assert "trust_session" in choices
    assert "trust_workspace" not in choices
    assert "trust_user" not in choices
    with pytest.raises(ValueError, match="validated durable user permission policy"):
        manager.trust_host_command("printf host", ApprovalChoice.USER)
    assert manager.trusted_host_command_rules() == ()
    manager.close()


def test_trusted_host_grant_expires_when_session_scratch_or_path_changes(tmp_path, monkeypatch):
    """A new scratch generation or process PATH invalidates an exact no-prompt grant."""
    manager = PermissionManager(tmp_path)
    rule = manager.trust_host_command("printf host")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    manager.bind_temporary_directory(PathHolder(scratch))
    assert (
        manager.matching_trusted_host_command_rule("printf host", tmp_path, "request_host_command")
        is None
    )
    manager.trust_host_command("printf host")
    monkeypatch.setenv("PATH", "/bin")
    assert (
        manager.matching_trusted_host_command_rule("printf host", tmp_path, "request_host_command")
        is None
    )
    assert rule.id
    manager.close()


def test_trusted_host_grants_persist_by_scope_and_revoke_across_restart(tmp_path):
    """Workspace and user grants survive restart without binding the old session scratch."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    workspace_rule = manager.trust_host_command("printf workspace", ApprovalChoice.WORKSPACE)
    user_rule = manager.trust_host_command("printf user", ApprovalChoice.USER)
    assert {rule.scope for rule in manager.trusted_host_command_rules()} == {
        ApprovalChoice.WORKSPACE,
        ApprovalChoice.USER,
    }
    manager.close()

    unvalidated = PermissionManager(
        tmp_path, user_configuration_path=arguments["user_configuration_path"]
    )
    assert unvalidated.trusted_host_command_rules() == ()
    assert (
        unvalidated.matching_trusted_host_command_rule(
            "printf user", tmp_path, "request_host_command"
        )
        is None
    )
    unvalidated.close()

    reloaded = PermissionManager(**arguments)
    assert (
        reloaded.matching_trusted_host_command_rule(
            "printf workspace", tmp_path, "request_host_command"
        )
        == workspace_rule
    )
    assert (
        reloaded.matching_trusted_host_command_rule("printf user", tmp_path, "request_host_command")
        == user_rule
    )
    assert (
        reloaded.matching_trusted_host_command_rule(
            "printf other", tmp_path, "request_host_command"
        )
        is None
    )
    assert not reloaded.remove_trusted_host_command_rule(user_rule.id, ApprovalChoice.WORKSPACE)
    assert reloaded.remove_trusted_host_command_rule(workspace_rule.id, ApprovalChoice.WORKSPACE)
    assert reloaded.remove_trusted_host_command_rule(user_rule.id, ApprovalChoice.USER)
    reloaded.close()
    final = PermissionManager(**arguments)
    assert final.trusted_host_command_rules() == ()
    final.close()


def test_user_host_grant_matches_another_registered_workspace(tmp_path):
    """A user grant spans qualified workspaces while a workspace grant remains local."""
    first = tmp_path / "first"
    first.mkdir()
    arguments, workspace_id, user_identity = durable_policy_inputs(first)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    manager.trust_host_command("printf workspace", ApprovalChoice.WORKSPACE)
    user_rule = manager.trust_host_command("printf user", ApprovalChoice.USER)
    manager.close()

    second = tmp_path / "second"
    second.mkdir()
    workspace = WorkspaceRepository(arguments["namespace_catalog_path"]).initialize(
        Workspace.discover(second)
    )
    other = PermissionManager(
        **{
            **arguments,
            "workspace_root": second,
            "workspace_id": workspace.id,
            "configuration_path": arguments["configuration_path"].parent.parent
            / workspace.id
            / "permissions.yaml",
        }
    )
    other.initialize_policy_namespace("workspace", workspace.id, authorized=True)
    assert (
        other.matching_trusted_host_command_rule("printf workspace", second, "request_host_command")
        is None
    )
    assert (
        other.matching_trusted_host_command_rule("printf user", second, "request_host_command")
        == user_rule
    )
    other.close()


def test_trusted_host_grant_rejects_stale_installation_and_noncanonical_cwd(tmp_path, monkeypatch):
    """A changed revision or working-directory authority cannot install a host grant."""
    manager = PermissionManager(tmp_path)
    request = native_request(tmp_path, source="printf host")
    _, revision = manager.evaluate_with_revision(())
    manager.set_default(Action.PROCESS_EXECUTE, Decision.DENY, scope=PolicyScope.SESSION)
    assert (
        manager.remember_trusted_host_command_rule_if_current(
            request, ApprovalChoice.SESSION, revision
        )
        is None
    )
    _, revision = manager.evaluate_with_revision(())
    original = Path.is_dir
    monkeypatch.setattr(
        Path, "is_dir", lambda candidate: False if candidate == tmp_path else original(candidate)
    )
    assert (
        manager.remember_trusted_host_command_rule_if_current(
            request, ApprovalChoice.SESSION, revision
        )
        is None
    )
    manager.close()


@pytest.mark.parametrize("missing_host_grants", [False, True])
def test_marked_user_policy_migrates_with_exact_backup(tmp_path, missing_host_grants):
    """Both complete and older marked user policies archive their bytes before upgrade."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    manager.close()
    path = arguments["user_configuration_path"]
    payload = load_yaml(path.read_text(encoding="utf-8"))
    generation = payload["namespace_generation"]
    payload["version"] = 1
    if missing_host_grants:
        payload.pop("trusted_host_command_rules")
    original = dump_yaml(payload).encode("utf-8")
    path.write_bytes(original)
    recovered = PermissionManager(**arguments)
    assert recovered.trusted_host_command_rules() == ()
    assert "recovery required" not in recovered.describe()
    saved = load_yaml(path.read_text(encoding="utf-8"))
    assert saved["version"] == 2
    assert saved["trusted_host_command_rules"] == []
    assert saved["namespace_generation"] == generation
    assert path.with_name(f"{path.name}.{generation}.schema-v1.bak").read_bytes() == original
    recovered.close()


def test_trusted_host_grant_scope_does_not_bypass_workspace_identity(tmp_path):
    """A workspace grant stops matching when the registered workspace inode changes."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    manager.trust_host_command("printf host", ApprovalChoice.WORKSPACE)
    sibling = tmp_path.parent / "replacement"
    tmp_path.rename(sibling)
    tmp_path.mkdir()
    assert (
        manager.matching_trusted_host_command_rule("printf host", tmp_path, "request_host_command")
        is None
    )
    manager.close()


def test_host_gate_offer_requires_only_user_changeable_eligibility(tmp_path):
    """A denied policy or changed gate prevents a stale eligibility grant."""
    manager = PermissionManager(tmp_path)
    host = operation(
        Action.PROCESS_EXECUTE,
        tool="request_host_command",
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "printf host"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.HOST,
        ),
    )
    assert manager.host_gate_enable_revision(()) is None
    assert (
        manager.host_gate_enable_revision((file_operation(Action.FILESYSTEM_READ, tmp_path),))
        is None
    )
    revision = manager.host_gate_enable_revision((host,))
    assert revision is not None
    assert not manager.enable_host_gate_if_current((host,), revision + 1)
    manager.add_rule(
        PermissionRule(
            decision=Decision.DENY,
            tool="request_host_command",
            action=Action.PROCESS_EXECUTE,
        ),
        scope=PolicyScope.SESSION,
    )
    assert manager.host_gate_enable_revision((host,)) is None
    assert not manager.enable_host_gate_if_current((host,), revision)
    manager.reset_session()
    fresh = manager.host_gate_enable_revision((host,))
    assert fresh is not None
    assert manager.enable_host_gate_if_current((host,), fresh)
    assert manager.host_gate_enable_revision((host,)) is None
    manager.close()


def test_revision_bound_host_rule_cancellation_during_policy_wait(tmp_path, policy_waits):
    """Cancelling a contended host approval leaves no remembered authority."""
    manager = PermissionManager(tmp_path)
    request = native_request(tmp_path)
    _, revision = manager.evaluate_with_revision(())
    locked = Event()
    release = Event()

    def hold_policy():
        """Keep the policy transaction owned until the pending operation is cancelled."""
        with manager.policy_update():
            locked.set()
            release.wait(timeout=2)

    async def exercise():
        """Cancel while the owner is cooperatively waiting for the lock."""
        holder = asyncio.create_task(asyncio.to_thread(hold_policy))
        assert await asyncio.to_thread(locked.wait, 2)
        pending = asyncio.create_task(
            manager.remember_host_command_rule_if_current_async(request, revision)
        )
        await policy_waits.get()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()
        await holder

    try:
        asyncio.run(exercise())
        assert manager.host_command_rules() == ()
    finally:
        release.set()
        manager.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_policy_snapshot_revision_identifies_later_change(tmp_path, asynchronous):
    """Policy decisions and their revision describe the same policy snapshot."""
    manager = PermissionManager(tmp_path)
    target = ProcessTarget(
        argv=("/bin/sh", "-c", "true"), cwd=str(tmp_path), boundary=ProcessBoundary.SANDBOXED
    )
    operations = (operation(Action.PROCESS_EXECUTE, target=target),)
    evaluate = lambda: (
        asyncio.run(manager.evaluate_with_revision_async(operations))
        if asynchronous
        else manager.evaluate_with_revision(operations)
    )
    before, revision = evaluate()
    manager.set_default(Action.PROCESS_EXECUTE, Decision.DENY, scope=PolicyScope.SESSION)
    after, changed_revision = evaluate()
    assert before.decision is not Decision.DENY
    assert after.decision is Decision.DENY
    assert revision is not None and changed_revision is not None
    assert changed_revision > revision
    manager.close()


def test_host_policy_owns_input_configuration_and_rule_copies(tmp_path):
    """Mutating supplied models cannot bypass synchronized policy changes."""
    configuration = PermissionConfiguration(limits=PolicyLimits(allow_host_processes=True))
    manager = PermissionManager(tmp_path, configuration=configuration)
    target = ProcessTarget(argv=("/bin/sh", "-c", "true"), cwd=str(tmp_path))
    operations = (operation(Action.PROCESS_EXECUTE, target=target),)
    configuration.limits.allow_host_processes = False
    rule = PermissionRule(decision=Decision.DENY, action=Action.PROCESS_EXECUTE)
    manager.add_rule(rule, scope=PolicyScope.SESSION)
    rule.decision = Decision.ALLOW

    assert manager.evaluate(operations).decision is Decision.DENY
    assert manager.remove_rule(rule.id, scope=PolicyScope.SESSION)
    assert manager.evaluate(operations).decision is Decision.ASK


@pytest.mark.parametrize("write_roots", [(), "workspace"])
def test_native_default_workspace_access_runs_without_prompt(tmp_path, write_roots):
    """Routine workspace commands run in either write mode without a prompt."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    roots = (tmp_path,) if write_roots == "workspace" else ()
    assert manager.authorize_sandboxed_command(
        native_request(tmp_path, "cat file", write_roots=roots, read_roots=()),
        tool_id="run_command",
    )
    interaction.prompt.assert_not_called()


def test_async_sandbox_approval_uses_native_prompt_and_returns_approved_result(tmp_path):
    """Async sandbox authorization awaits approval for a fresh destructive command."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value="approve")
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, "printf text > output", read_roots=())
    approved_revision = []

    assert asyncio.run(
        manager.authorize_sandboxed_command_async(
            request, tool_id="run_command", approved_revision=approved_revision
        )
    )
    assert approved_revision == [manager.egress_policy_revision()]

    interaction.prompt_async.assert_awaited_once()
    assert interaction.prompt_async.await_args.kwargs["choices"] == {
        "deny": "Deny",
        "approve": "Allow once",
        "details": "Show full scope",
    }


def test_async_sandbox_default_workspace_command_runs_without_prompt(tmp_path):
    """Routine workspace commands retain prompt-free sandbox access asynchronously."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock()
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(
        tmp_path,
        "printf safe",
        read_roots=(),
        environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(manager.temporary_directory)},
    )
    approved_revision = []

    assert asyncio.run(
        manager.authorize_sandboxed_command_async(
            request, tool_id="run_command", approved_revision=approved_revision
        )
    )
    assert approved_revision == [manager.egress_policy_revision()]

    interaction.prompt_async.assert_not_awaited()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("decision", [Decision.DENY, Decision.ASK])
def test_default_sandbox_rechecks_policy_after_inspection(tmp_path, asynchronous, decision):
    """A policy change during command inspection cannot authorize the default sandbox."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.info = Mock()
    interaction.prompt = Mock(return_value="deny")
    interaction.prompt_async = AsyncMock(return_value="deny")
    inspection = Mock(spec=CommandInspection)
    manager = PermissionManager(tmp_path, interaction=interaction, command_inspection=inspection)

    def change_policy(*_args, **_kwargs):
        """Change policy while the initial command inspection is pending."""
        manager.add_rule(
            PermissionRule(decision=decision, tool="run_command", action=Action.PROCESS_EXECUTE),
            scope=PolicyScope.SESSION,
        )
        return ()

    inspection.inspect.side_effect = change_policy
    request = native_request(tmp_path, "printf safe", read_roots=())
    if asynchronous:
        approved = asyncio.run(
            manager.authorize_sandboxed_command_async(request, tool_id="run_command")
        )
    else:
        approved = manager.authorize_sandboxed_command(request, tool_id="run_command")

    assert not approved
    interaction.prompt.assert_not_called()
    interaction.prompt_async.assert_not_awaited()


def test_async_sandbox_approval_remembers_session_rule_for_later_requests(tmp_path):
    """Async sandbox approval stores reusable choices and skips matching later prompts."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value="session")
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, "cat file")
    approved_revision = []

    assert asyncio.run(
        manager.authorize_sandboxed_command_async(
            request, tool_id="run_command", approved_revision=approved_revision
        )
    )
    assert approved_revision == [manager.egress_policy_revision()]
    assert manager.sandboxed_command_rules(ApprovalChoice.SESSION)
    interaction.prompt_async.reset_mock()
    approved_revision.clear()
    assert asyncio.run(
        manager.authorize_sandboxed_command_async(
            request, tool_id="run_command", approved_revision=approved_revision
        )
    )
    assert approved_revision == [manager.egress_policy_revision()]
    interaction.prompt_async.assert_not_awaited()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("change", ["revoke", "deny", "ask"])
def test_sandbox_rule_rejects_policy_change_during_lookup(
    tmp_path, monkeypatch, asynchronous, change
):
    """A stale remembered sandbox rule cannot authorize after a concurrent policy change."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.info = Mock()
    interaction.prompt = Mock(return_value="session")
    interaction.prompt_async = AsyncMock(return_value="session")
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, "cat file")
    assert manager.authorize_sandboxed_command(request, tool_id="run_command")
    rule_id = manager.sandboxed_command_rules(ApprovalChoice.SESSION)[0].id
    original = manager.sandboxed_command_rules
    changed = False

    def change_policy(scope):
        """Return the old rule once while publishing a concurrent policy change."""
        nonlocal changed
        previous = original(scope)
        if scope is ApprovalChoice.SESSION and not changed:
            changed = True
            if change == "revoke":
                assert manager.remove_sandboxed_command_rule(ApprovalChoice.SESSION, rule_id)
            else:
                manager.add_rule(
                    PermissionRule(
                        decision=Decision.DENY if change == "deny" else Decision.ASK,
                        tool="run_command",
                        action=Action.PROCESS_EXECUTE,
                    ),
                    scope=PolicyScope.SESSION,
                )
        return previous

    monkeypatch.setattr(manager, "sandboxed_command_rules", change_policy)
    interaction.prompt.reset_mock()
    interaction.prompt_async.reset_mock()
    if asynchronous:
        approved = asyncio.run(
            manager.authorize_sandboxed_command_async(request, tool_id="run_command")
        )
    else:
        approved = manager.authorize_sandboxed_command(request, tool_id="run_command")

    assert changed and not approved
    interaction.prompt.assert_not_called()
    interaction.prompt_async.assert_not_awaited()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sandbox_prompt_does_not_open_after_policy_changes_during_empty_lookup(
    tmp_path, monkeypatch, asynchronous
):
    """An empty remembered-rule lookup cannot open a prompt for stale policy."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.info = Mock()
    interaction.prompt = Mock(return_value="approve")
    interaction.prompt_async = AsyncMock(return_value="approve")
    manager = PermissionManager(tmp_path, interaction=interaction)
    original = manager.sandboxed_command_rules
    changed = False

    def change_policy(scope):
        """Add a denial after an empty remembered-rule snapshot has been read."""
        nonlocal changed
        previous = original(scope)
        if scope is ApprovalChoice.SESSION and not changed:
            changed = True
            manager.add_rule(
                PermissionRule(
                    decision=Decision.DENY, tool="run_command", action=Action.PROCESS_EXECUTE
                ),
                scope=PolicyScope.SESSION,
            )
        return previous

    monkeypatch.setattr(manager, "sandboxed_command_rules", change_policy)
    request = native_request(tmp_path, "cat file")
    if asynchronous:
        approved = asyncio.run(
            manager.authorize_sandboxed_command_async(request, tool_id="run_command")
        )
    else:
        approved = manager.authorize_sandboxed_command(request, tool_id="run_command")

    assert changed and not approved
    interaction.prompt.assert_not_called()
    interaction.prompt_async.assert_not_awaited()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("selected", ["approve", "session", "workspace", "user"])
@pytest.mark.parametrize("decision", [Decision.DENY, Decision.ASK])
def test_sandbox_prompt_rejects_new_policy_before_approval_is_remembered(
    tmp_path, asynchronous, selected, decision
):
    """A new denial or explicit ask cannot be overridden by an earlier prompt."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.info = Mock()
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        user_configuration_path=tmp_path / "user-permissions.yaml",
    )

    def change_policy(*_args, **_kwargs):
        """Change policy while the original sandbox approval is pending."""
        manager.add_rule(
            PermissionRule(decision=decision, tool="run_command", action=Action.PROCESS_EXECUTE),
            scope=PolicyScope.SESSION,
        )
        return selected

    if asynchronous:
        interaction.prompt_async = AsyncMock(side_effect=change_policy)
        approved = asyncio.run(
            manager.authorize_sandboxed_command_async(
                native_request(tmp_path, "cat file"), tool_id="run_command"
            )
        )
    else:
        interaction.prompt = Mock(side_effect=change_policy)
        approved = manager.authorize_sandboxed_command(
            native_request(tmp_path, "cat file"), tool_id="run_command"
        )

    assert not approved
    assert all(
        not manager.sandboxed_command_rules(scope)
        for scope in (ApprovalChoice.SESSION, ApprovalChoice.WORKSPACE, ApprovalChoice.USER)
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("add_ask", [False, True])
def test_sandbox_prompt_checks_reviewed_explicit_ask(tmp_path, asynchronous, add_ask):
    """A reviewed ask remains valid, while a newly added ask needs a fresh decision."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.info = Mock()
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.add_rule(
        PermissionRule(decision=Decision.ASK, tool="run_command", action=Action.PROCESS_EXECUTE),
        scope=PolicyScope.SESSION,
    )

    def answer(*_args, **_kwargs):
        """Optionally add another explicit ask before answering the existing prompt."""
        if add_ask:
            manager.add_rule(
                PermissionRule(
                    decision=Decision.ASK, tool="run_command", action=Action.PROCESS_EXECUTE
                ),
                scope=PolicyScope.SESSION,
            )
        return "approve"

    if asynchronous:
        interaction.prompt_async = AsyncMock(side_effect=answer)
        approved = asyncio.run(
            manager.authorize_sandboxed_command_async(
                native_request(tmp_path, "cat file"), tool_id="run_command"
            )
        )
    else:
        interaction.prompt = Mock(side_effect=answer)
        approved = manager.authorize_sandboxed_command(
            native_request(tmp_path, "cat file"), tool_id="run_command"
        )

    assert approved is not add_ask
    assert not manager.sandboxed_command_rules(ApprovalChoice.SESSION)
    prompt = interaction.prompt_async if asynchronous else interaction.prompt
    assert set(prompt.call_args.kwargs["choices"]) == {"deny", "approve", "details"}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("change", "selected"),
    [("same_ask", "approve"), ("unrelated", "approve"), ("unrelated", "session")],
)
def test_sandbox_prompt_invalidates_any_concurrent_policy_revision(
    tmp_path, asynchronous, selected, change
):
    """A changed policy revision cannot retain an earlier reviewed command decision."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.info = Mock()
    manager = PermissionManager(tmp_path, interaction=interaction)
    rule = PermissionRule(
        id="reviewed-ask", decision=Decision.ASK, tool="run_command", action=Action.PROCESS_EXECUTE
    )
    if change == "same_ask":
        manager.add_rule(rule, scope=PolicyScope.SESSION)

    def answer(*_args, **_kwargs):
        """Reissue the same ask or add an unrelated rule while the prompt is open."""
        if change == "same_ask":
            assert manager.remove_rule(rule.id, scope=PolicyScope.SESSION)
            manager.add_rule(rule, scope=PolicyScope.SESSION)
        else:
            manager.add_rule(
                PermissionRule(decision=Decision.DENY, action=Action.FILESYSTEM_READ),
                scope=PolicyScope.SESSION,
            )
        return selected

    request = native_request(tmp_path, "cat file")
    if asynchronous:
        interaction.prompt_async = AsyncMock(side_effect=answer)
        approved = asyncio.run(
            manager.authorize_sandboxed_command_async(request, tool_id="run_command")
        )
    else:
        interaction.prompt = Mock(side_effect=answer)
        approved = manager.authorize_sandboxed_command(request, tool_id="run_command")

    assert not approved
    assert not manager.sandboxed_command_rules(ApprovalChoice.SESSION)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sandbox_explicit_ask_cannot_be_skipped_by_a_remembered_approval(tmp_path, asynchronous):
    """An explicit ask remains fresh even when an earlier session approval matches."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.info = Mock()
    interaction.prompt = Mock(side_effect=["session", "deny"])
    interaction.prompt_async = AsyncMock(side_effect=["session", "deny"])
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, "cat file")

    if asynchronous:
        first = asyncio.run(
            manager.authorize_sandboxed_command_async(request, tool_id="run_command")
        )
        manager.add_rule(
            PermissionRule(
                decision=Decision.ASK, tool="run_command", action=Action.PROCESS_EXECUTE
            ),
            scope=PolicyScope.SESSION,
        )
        second = asyncio.run(
            manager.authorize_sandboxed_command_async(request, tool_id="run_command")
        )
        assert interaction.prompt_async.await_count == 2
    else:
        first = manager.authorize_sandboxed_command(request, tool_id="run_command")
        manager.add_rule(
            PermissionRule(
                decision=Decision.ASK, tool="run_command", action=Action.PROCESS_EXECUTE
            ),
            scope=PolicyScope.SESSION,
        )
        second = manager.authorize_sandboxed_command(request, tool_id="run_command")
        assert interaction.prompt.call_count == 2

    assert first and not second
    assert manager.sandboxed_command_rules(ApprovalChoice.SESSION)
    prompt = interaction.prompt_async if asynchronous else interaction.prompt
    assert set(prompt.call_args.kwargs["choices"]) == {"deny", "approve", "details"}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sandbox_remembering_rechecks_policy_after_signature_preparation(
    tmp_path, monkeypatch, asynchronous
):
    """A denial during signature preparation prevents publishing a remembered approval."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.info = Mock()
    manager = PermissionManager(tmp_path, interaction=interaction)
    original = SandboxRequest.sandbox_command_signature
    answered = False
    changed = False

    def answer(*_args, **_kwargs):
        """Choose a remembered approval after the original policy review."""
        nonlocal answered
        answered = True
        return "session"

    def change_policy(request, identity):
        """Install a denial after prompt input but before the approval can be stored."""
        nonlocal changed
        signature = original(request, identity)
        if answered and not changed:
            changed = True
            manager.add_rule(
                PermissionRule(
                    decision=Decision.DENY, tool="run_command", action=Action.PROCESS_EXECUTE
                ),
                scope=PolicyScope.SESSION,
            )
        return signature

    monkeypatch.setattr(SandboxRequest, "sandbox_command_signature", change_policy)
    request = native_request(tmp_path, "cat file")
    if asynchronous:
        interaction.prompt_async = AsyncMock(side_effect=answer)
        approved = asyncio.run(
            manager.authorize_sandboxed_command_async(request, tool_id="run_command")
        )
    else:
        interaction.prompt = Mock(side_effect=answer)
        approved = manager.authorize_sandboxed_command(request, tool_id="run_command")

    assert answered and changed and not approved
    assert not manager.sandboxed_command_rules(ApprovalChoice.SESSION)


def test_async_sandbox_approval_obeys_explicit_ask_rules(tmp_path):
    """An explicit command ask rule prompts even inside default workspace authority."""
    console = Console(file=io.StringIO())
    interaction = ConsoleInteraction(console=console, markdown=False)
    interaction.prompt_async = AsyncMock(return_value="deny")
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            rules=[PermissionRule(decision=Decision.ASK, action=Action.PROCESS_EXECUTE)]
        ),
    )
    request = native_request(tmp_path, "printf safe", read_roots=())

    assert not asyncio.run(
        manager.authorize_sandboxed_command_async(request, tool_id="run_command")
    )

    assert "run under your command approval rule" in console.file.getvalue()
    interaction.prompt_async.assert_awaited_once()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sandbox_policy_denies_the_executed_source(tmp_path, asynchronous):
    """A rule denying the actual shell source blocks a different displayed source."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            rules=[
                PermissionRule(
                    decision=Decision.DENY,
                    action=Action.PROCESS_EXECUTE,
                    target=ProcessTarget(
                        argv=("/bin/sh", "-c", "printf blocked"),
                        cwd=str(tmp_path),
                        boundary=ProcessBoundary.SANDBOXED,
                    ),
                )
            ]
        ),
    )
    request = native_request(
        tmp_path, "printf allowed", execution_source="printf blocked", read_roots=()
    )

    if asynchronous:
        approved = asyncio.run(manager.authorize_sandboxed_command_async(request, tool_id="run"))
    else:
        approved = manager.authorize_sandboxed_command(request, tool_id="run")

    assert not approved
    interaction.prompt.assert_not_called()


def test_sandbox_approval_displays_executed_source(tmp_path):
    """Human approval identifies the actual command when it differs from the displayed source."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(
        tmp_path, "printf shown", execution_source="printf executed", read_roots=(tmp_path.parent,)
    )

    assert not manager.authorize_sandboxed_command(request, tool_id="run")
    assert 'Executed command: "printf executed"' in interaction.info.call_args.args[0]


def test_async_sandbox_approval_denies_headless_and_policy_denied_requests(tmp_path):
    """Async sandbox authorization denies absent interactions and explicit deny policy."""
    request = native_request(tmp_path, "cat file")
    headless = PermissionManager(tmp_path)
    assert not asyncio.run(
        headless.authorize_sandboxed_command_async(request, tool_id="run_command")
    )

    manager = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(
            rules=[PermissionRule(decision=Decision.DENY, action=Action.PROCESS_EXECUTE)]
        ),
    )
    assert not asyncio.run(
        manager.authorize_sandboxed_command_async(request, tool_id="run_command")
    )


@pytest.mark.parametrize("selection", [False, "deny"])
def test_async_sandbox_approval_denies_invalid_or_rejected_choices(tmp_path, selection):
    """Async sandbox approval denies cancelled and explicitly rejected selections."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=selection)
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, "printf text > output", read_roots=())

    assert not asyncio.run(
        manager.authorize_sandboxed_command_async(request, tool_id="run_command")
    )


def test_async_sandbox_approval_rejects_sync_only_interaction(tmp_path):
    """Async sandbox authorization requires native asynchronous input for approval."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, "printf text > output", read_roots=())

    with pytest.raises(ProblemException) as captured:
        asyncio.run(manager.authorize_sandboxed_command_async(request, tool_id="run_command"))

    assert captured.value.problem.code == "interaction.async_required"
    interaction.prompt.assert_not_called()


@pytest.mark.parametrize("cancel_stage", ["audit", "remember"])
def test_async_sandbox_approval_cancellation_settles_owned_work(tmp_path, cancel_stage):
    """Cancellation waits for owned audit or approval work and never returns authorization."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value="session")
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, "cat file")

    async def exercise():
        """Cancel authorization while its selected owner operation is settling."""
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def run_blocking(function, *args):
            """Delay the selected operation until cancellation has been requested."""
            nonlocal calls
            calls += 1
            blocked = (cancel_stage == "audit" and calls == 1) or (
                cancel_stage == "remember" and interaction.prompt_async.await_count > 0
            )
            if blocked:
                started.set()
                await release.wait()
            return function(*args)

        manager.set_async_blocking_runner(run_blocking)
        pending = asyncio.create_task(
            manager.authorize_sandboxed_command_async(request, tool_id="run_command")
        )
        await started.wait()
        pending.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(exercise())


def test_native_network_still_requires_permission(tmp_path):
    """General network access stays outside the automatic workspace boundary."""
    manager = PermissionManager(tmp_path)
    assert (
        manager.authorize_sandboxed_command(
            native_request(tmp_path, "cat file", write_roots=(), read_roots=(), network=True),
            tool_id="run_command",
        )
        is False
    )


def test_explicit_process_ask_rule_still_prompts_for_default_sandbox(tmp_path):
    """An explicit user ask rule overrides prompt-free default sandbox access."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            rules=[PermissionRule(decision=Decision.ASK, action=Action.PROCESS_EXECUTE)]
        ),
    )
    request = native_request(tmp_path, "printf safe", read_roots=())
    assert not manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert "run under your command approval rule" in interaction.info.call_args.args[0]
    interaction.prompt.assert_called_once()


@pytest.mark.parametrize(
    "source",
    [
        "rm -rf build",
        "git add file",
        "env -i /bin/rm victim",
        "printf text > victim",
        "sh -c 'git add file'",
        "FOO=1 command mv source existing",
    ],
)
def test_native_destructive_workspace_commands_require_approval(tmp_path, source):
    """Recognizable destructive shell actions ask even within default workspace access."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)

    assert not manager.authorize_sandboxed_command(
        native_request(tmp_path, source, read_roots=()), tool_id="run_command"
    )
    if "git" in source:
        assert "Review a Git state-changing command" in interaction.info.call_args.args[0]
        assert "change repository state" in interaction.info.call_args.args[0]
    elif ">" in source:
        assert "Review an output destination" in interaction.info.call_args.args[0]
    else:
        assert "Review a destructive workspace command" in interaction.info.call_args.args[0]
        assert "delete or overwrite files" in interaction.info.call_args.args[0]
    assert "similar_session" not in interaction.prompt.call_args.kwargs["choices"]
    assert set(interaction.prompt.call_args.kwargs["choices"]) == {"deny", "approve", "details"}


def test_git_write_requires_fresh_approval_after_prior_approval(tmp_path):
    """An earlier one-off Git approval cannot authorize a later Git mutation."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.side_effect = ["approve", "deny"]
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, "git add file", read_roots=())
    assert manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert not manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert interaction.prompt.call_count == 2


def test_combined_command_findings_show_both_reasons_and_allow_once_only(tmp_path):
    """All matching command policies contribute to one fresh approval prompt."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    audit_path = tmp_path / "audit.db"
    manager = PermissionManager(
        tmp_path, interaction=interaction, audit_path=audit_path, workspace_id="workspace-1"
    )

    assert not manager.authorize_sandboxed_command(
        native_request(tmp_path, "git add file && rm file", read_roots=()),
        tool_id="run_command",
    )
    message = interaction.info.call_args.args[0]
    assert "Review a Git state-changing command" in message
    assert "Review a destructive workspace command" in message
    assert "change repository state" in message
    assert "delete or overwrite files in this workspace" in message
    assert set(interaction.prompt.call_args.kwargs["choices"]) == {"deny", "approve", "details"}
    with closing(sqlite3.connect(audit_path)) as connection:
        row = connection.execute(
            "SELECT payload_json FROM permission_audit_records "
            "WHERE event_name = 'sandbox.permission_scope'"
        ).fetchone()
    assert json.loads(row[0])["review_policies"] == ["git_change", "destructive_command"]
    assert json.loads(row[0])["scratch_data"] is False


def test_fresh_command_finding_overrides_matching_remembered_rule(tmp_path):
    """A stored command approval cannot bypass a newly applicable fresh review policy."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "workspace"
    user_path = tmp_path / "user.yaml"
    manager = PermissionManager(
        tmp_path, user_configuration_path=user_path, interaction=interaction
    )
    assert manager.authorize_sandboxed_command(
        native_request(tmp_path, "printf benign"), tool_id="run_command"
    )
    rule = manager.sandboxed_command_rules(ApprovalChoice.WORKSPACE)[0]
    source = "git add file && rm file"
    user_path.write_text(
        dump_yaml(
            {
                "sandboxed_command_rules": [
                    rule.model_copy(update={"source": source}).model_dump(mode="json")
                ]
            }
        ),
        encoding="utf-8",
    )

    interaction.prompt.return_value = "deny"
    reloaded = PermissionManager(
        tmp_path, user_configuration_path=user_path, interaction=interaction
    )
    assert not reloaded.authorize_sandboxed_command(
        native_request(tmp_path, source), tool_id="run_command"
    )
    assert interaction.prompt.call_count == 2


def test_injected_inspection_controls_command_review_and_git_grant_intent(tmp_path):
    """A composed matcher changes this manager's review and Git authority classification."""

    def match_special(command: list[str]) -> CommandFinding | None:
        """Request reusable review and Git authority for a special command."""
        if command != ["special", "command"]:
            return None
        return CommandFinding(
            policy_id="special",
            status=CommandReviewStatus.REVIEW,
            context="Review a special command",
            reason="run a special command",
            requests_git_write=True,
        )

    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    inspection = CommandInspection(()).with_matcher(match_special)
    manager = PermissionManager(tmp_path, interaction=interaction, command_inspection=inspection)
    assert manager.command_needs_git_write("special command")
    assert not manager.command_needs_git_write("git add file")
    assert not manager.authorize_sandboxed_command(
        native_request(tmp_path, "special command", read_roots=()),
        tool_id="run_command",
    )
    assert "Review a special command" in interaction.info.call_args.args[0]
    assert "run a special command" in interaction.info.call_args.args[0]
    assert "session" in interaction.prompt.call_args.kwargs["choices"]


@pytest.mark.parametrize("source", ["printf '%s' 'rm -rf build'", "echo '"])
def test_native_non_destructive_or_invalid_source_does_not_trigger_review(tmp_path, source):
    """Literal output, empty segments, and invalid shell text do not claim destructive intent."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert manager.authorize_sandboxed_command(
        native_request(tmp_path, source, read_roots=()),
        tool_id="run_command",
    )
    interaction.prompt.assert_not_called()


def test_unused_external_path_does_not_expand_read_authority(tmp_path):
    """Search metadata alone does not require permission for routine commands."""
    tools = tmp_path.with_name(tmp_path.name + "-external-tools")
    tools.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    environment = {"PATH": str(tools), "LANG": "C"}
    first = native_request(tmp_path, "printf first", read_roots=(), environment=environment)
    assert manager.authorize_sandboxed_command(first, tool_id="run_command")
    second = native_request(tmp_path, "printf second", read_roots=(), environment=environment)
    assert manager.authorize_sandboxed_command(second, tool_id="run_command")
    interaction.prompt.assert_not_called()
    expanded = native_request(
        tmp_path, "printf first", read_roots=(tools,), environment=environment
    )
    assert not manager.authorize_sandboxed_command(expanded, tool_id="run_command")
    assert interaction.prompt.call_count == 1
    network = native_request(
        tmp_path, "printf second", read_roots=(), environment=environment, network=True
    )
    assert not manager.authorize_sandboxed_command(network, tool_id="run_command")
    assert interaction.prompt.call_count == 1


def test_sandbox_network_requires_enforceable_effective_limits(tmp_path):
    """A broad grant cannot bypass private or origin restrictions, even with approval."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "approve"
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, network=True, read_roots=())

    assert "private-network limit" in manager.sandbox_network_limit("run_command")
    assert not manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert not asyncio.run(
        manager.authorize_sandboxed_command_async(request, tool_id="run_command")
    )
    interaction.prompt.assert_not_called()

    manager.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)
    assert manager.sandbox_network_limit("run_command") is None
    assert manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert interaction.prompt.call_count == 1

    headless = PermissionManager(tmp_path)
    headless.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)
    assert not headless.authorize_sandboxed_command(request, tool_id="run_command")

    manager.update_limit_values("network_origins", "*", add=False, scope=PolicyScope.SESSION)
    assert "network-origin limit" in manager.sandbox_network_limit("run_command")
    assert not manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert interaction.prompt.call_count == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("policy_source", ["workspace", "user", "managed", "default"])
def test_broad_network_grant_respects_network_denial(tmp_path, asynchronous, policy_source):
    """A broad native grant cannot evade network denial through approval or YOLO."""
    rule = PermissionRule(
        id="deny-network",
        decision=Decision.DENY,
        action=Action.NETWORK_REQUEST,
    )
    store = (
        managed_store(tmp_path, rules=(rule.model_dump(mode="json"),))
        if policy_source == "managed"
        else None
    )
    user_path = tmp_path / "user.yaml"
    if policy_source == "user":
        user_path.write_text(
            dump_yaml(UserPermissionConfiguration(rules=[rule]).model_dump(mode="json")),
            encoding="utf-8",
        )
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        managed_policy_store=store,
        user_configuration_path=user_path,
        yolo=True,
    )
    manager.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)
    if policy_source == "workspace":
        manager.add_rule(rule)
    elif policy_source == "default":
        manager.set_default(Action.NETWORK_REQUEST, Decision.DENY)
    request = native_request(tmp_path, network=True, read_roots=())

    if asynchronous:
        authorized = asyncio.run(
            manager.authorize_sandboxed_command_async(request, tool_id="run_command")
        )
    else:
        authorized = manager.authorize_sandboxed_command(request, tool_id="run_command")

    assert not authorized
    interaction.prompt.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("target", ["url", "exact"])
def test_broad_network_grant_rejects_unenforceable_network_rule(tmp_path, asynchronous, target):
    """An exact network rule prevents a native grant to every destination."""
    manager = PermissionManager(tmp_path, yolo=True)
    manager.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)
    matcher = (
        {"resource": "https://example.com/private"}
        if target == "url"
        else {
            "target": NetworkTarget(
                url="https://example.com/private",
                origin="https://example.com",
            )
        }
    )
    manager.add_rule(
        PermissionRule(
            id="deny-specific-url",
            decision=Decision.DENY,
            tool="run_command",
            action=Action.NETWORK_REQUEST,
            **matcher,
        )
    )
    assert "exact-request rule" in manager.sandbox_network_limit("run_command")
    assert manager.sandbox_network_limit("other_tool") is None
    request = native_request(tmp_path, network=True, read_roots=())

    if asynchronous:
        authorized = asyncio.run(
            manager.authorize_sandboxed_command_async(request, tool_id="run_command")
        )
    else:
        authorized = manager.authorize_sandboxed_command(request, tool_id="run_command")

    assert not authorized
    assert manager.authorize_sandboxed_command(request, tool_id="other_tool")


@pytest.mark.parametrize(
    ("resource", "limited"),
    [
        ("/bin/sh -c forbidden*", False),
        ("host*", False),
        ("[p]ython *", False),
        ("[!h]ttps://example.com/private", False),
        ("[]]ttp://example.com/private", False),
        ("[p", False),
        ("h", False),
        ("https://example.com/private", True),
        ("h*private", True),
        ("*private", True),
        ("?ttp://example.com/private", True),
        ("[!p]ttps://example.com/private", True),
        ("[!]]ttps://example.com/private", True),
        ("[h]ttps://example.com/private", True),
    ],
)
def test_generic_resource_rule_only_limits_possible_network_targets(tmp_path, resource, limited):
    """An action-agnostic process glob does not disable unrelated network grants."""
    manager = PermissionManager(tmp_path, yolo=True)
    manager.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)
    manager.add_rule(PermissionRule(decision=Decision.DENY, resource=resource))

    assert (manager.sandbox_network_limit("run_command") is not None) is limited
    assert (
        manager.sandbox_egress_limit("run_command", ("https://example.com",)) is not None
    ) is limited
    assert (
        manager.authorize_sandboxed_command(
            native_request(tmp_path, network=True, read_roots=()),
            tool_id="run_command",
        )
        is not limited
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_mediated_origin_requires_approval_and_respects_origin_limit(tmp_path, asynchronous):
    """An exact origin request keeps the private limit and carries a reviewable scope."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "approve"
    interaction.prompt_async = AsyncMock(return_value="approve")
    interaction.confirm_async = AsyncMock(return_value=True)
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, egress_origins=("https://example.com",), read_roots=())
    approved_revision = []

    def authorize():
        """Exercise the selected public authorization entry point."""
        if asynchronous:
            return asyncio.run(
                manager.authorize_sandboxed_command_async(
                    request, tool_id="run", approved_revision=approved_revision
                )
            )
        return manager.authorize_sandboxed_command(
            request, tool_id="run", approved_revision=approved_revision
        )

    assert authorize()
    assert approved_revision == [manager.egress_policy_revision()]
    selected = interaction.prompt_async if asynchronous else interaction.prompt
    assert selected.call_count == 1
    assert '"egress_origins": ["https://example.com"]' in interaction.info.call_args.args[0]
    manager.update_limit_values("network_origins", "*", add=False, scope=PolicyScope.SESSION)
    assert not authorize()
    assert len(approved_revision) == 1
    assert selected.call_count == 1


def test_mixed_wildcard_origin_limit_allows_mediated_https(tmp_path):
    """A wildcard remains unrestricted when added after a narrower origin."""
    manager = PermissionManager(tmp_path, yolo=True)
    assert manager.update_limit_values(
        "network_origins", "https://specific.example", add=True, scope=PolicyScope.SESSION
    )
    assert manager.update_limit_values("network_origins", "*", add=True, scope=PolicyScope.SESSION)

    assert manager.sandbox_egress_limit("run_command", ("https://example.com",)) is None
    assert manager.authorize_sandboxed_command(
        native_request(tmp_path, egress_origins=("https://example.com",), read_roots=()),
        tool_id="run_command",
    )


def test_mediated_stronger_rule_gate_and_snapshot_share_policy_lock(tmp_path, monkeypatch):
    """A URL-specific rule cannot appear between the mediator ceiling and revision snapshot."""
    manager = PermissionManager(tmp_path, yolo=True)
    manager.add_rule(
        PermissionRule(
            id="deny-specific-url",
            decision=Decision.DENY,
            tool="run",
            action=Action.NETWORK_REQUEST,
            resource="https://example.com/private",
        )
    )
    request = native_request(tmp_path, egress_origins=("https://example.com",))
    gate = manager.sandbox_egress_limit
    snapshot = manager._sandbox_policy_snapshot

    def checked_gate(*args):
        """Verify the policy lock remains held across both admission checks."""
        assert manager._policy_lock._is_owned()
        return gate(*args)

    def checked_snapshot(*args):
        """Verify the revision is captured under that same admission lock."""
        assert manager._policy_lock._is_owned()
        return snapshot(*args)

    monkeypatch.setattr(manager, "sandbox_egress_limit", checked_gate)
    monkeypatch.setattr(manager, "_sandbox_policy_snapshot", checked_snapshot)
    assert not manager.authorize_sandboxed_command(request, tool_id="run")
    assert not asyncio.run(manager.authorize_sandboxed_command_async(request, tool_id="run"))


def test_async_mediated_authorization_refuses_a_policy_change_after_snapshot(tmp_path, monkeypatch):
    """A revision change before the async egress check cannot reuse the old decision."""
    manager = PermissionManager(tmp_path, yolo=True)
    request = native_request(tmp_path, egress_origins=("https://example.com",))
    evaluate = manager.evaluate_with_revision_async

    async def change_after_snapshot(operations):
        """Publish a new policy immediately after the initial decision is captured."""
        decision = await evaluate(operations)
        manager.update_limit_values("network_origins", "*", add=False, scope=PolicyScope.SESSION)
        return decision

    monkeypatch.setattr(manager, "evaluate_with_revision_async", change_after_snapshot)

    assert not asyncio.run(manager.authorize_sandboxed_command_async(request, tool_id="run"))


def test_mediated_policy_revision_check_never_blocks_heartbeat_on_update(tmp_path):
    """A held policy lock yields a pending result to the independent health sender."""
    manager = PermissionManager(tmp_path)
    assert manager.try_egress_policy_revision() == manager.egress_policy_revision()
    original = manager._policy_lock
    occupied = Mock()
    occupied.acquire.return_value = False
    manager._policy_lock = occupied
    try:
        assert manager.try_egress_policy_revision() is None
        occupied.release.assert_not_called()
    finally:
        manager._policy_lock = original


@pytest.mark.parametrize("failure", ["pending", "stale", "managed"])
def test_mediated_lease_binding_requires_active_policy_authority(tmp_path, monkeypatch, failure):
    """A changed, pending, or invalid policy cannot activate a new broker route."""
    manager = PermissionManager(tmp_path)
    lease = Mock()
    if failure == "pending":
        manager._egress_updating = True
    elif failure == "managed":
        monkeypatch.setattr(manager, "_managed_current_error", Mock(return_value="invalid"))
    revision = manager._policy_revision + (1 if failure == "stale" else 0)
    with pytest.raises(OSError, match="policy changed"):
        manager.bind_egress(revision, lease)
    assert lease not in manager._egress_revokers
    manager._egress_updating = False
    manager.close()


def test_mediated_lease_binding_unbinds_and_close_reports_uncertain_cleanup(tmp_path):
    """A manager releases settled leases and reports an unconfirmed active closure."""
    manager = PermissionManager(tmp_path)
    lease = Mock()
    lease.revoke_revision.side_effect = [False, True]
    manager.bind_egress(manager.egress_policy_revision(), lease)
    with pytest.raises(OSError, match="closure could not be confirmed"):
        manager.close()
    with pytest.raises(OSError, match="closed or closing"), manager.standalone_execution():
        pass
    manager.close()
    assert lease.revoke_revision.call_count == 2

    next_manager = PermissionManager(tmp_path)
    next_lease = Mock()
    next_manager.bind_egress(next_manager.egress_policy_revision(), next_lease)
    next_manager.unbind_egress(next_lease)
    next_manager.close()
    next_lease.revoke_revision.assert_not_called()


@pytest.mark.parametrize("confirmed", [True, False])
def test_mediated_managed_policy_change_revokes_before_reporting_error(
    tmp_path, monkeypatch, confirmed
):
    """Managed-source invalidation closes active routes or reports uncertain cleanup."""
    manager = PermissionManager(tmp_path)
    lease = Mock()
    lease.revoke_revision.return_value = confirmed
    manager.bind_egress(manager.egress_policy_revision(), lease)

    def invalidate():
        """Model a newly detected managed revision in the active manager."""
        manager._policy_revision += 1
        return "managed source changed"

    monkeypatch.setattr(manager, "_managed_current_error", invalidate)
    with pytest.raises(OSError, match="unavailable" if confirmed else "uncertain"):
        manager.egress_policy_revision()
    lease.revoke_revision.assert_called_once_with(manager._policy_revision)
    manager.unbind_egress(lease)
    manager.close()


@pytest.mark.parametrize(
    "phase", ["pause", "resume", "revision", "exception", "exception-uncertain"]
)
def test_mediated_policy_transition_requires_confirmed_barrier_and_closure(tmp_path, phase):
    """Policy updates never publish success after a missing admission or closure ack."""
    manager = PermissionManager(tmp_path)
    lease = Mock()
    lease.pause.return_value = phase != "pause"
    lease.resume.return_value = phase != "resume"
    lease.revoke_revision.return_value = phase != "exception-uncertain"
    manager.bind_egress(manager.egress_policy_revision(), lease)
    if phase == "pause":
        with pytest.raises(OSError, match="admission pause"), manager.policy_update():
            pytest.fail("Unconfirmed pause entered the policy mutation")
    elif phase == "resume":
        with pytest.raises(OSError, match="transition is uncertain"), manager.policy_update():
            pass
    elif phase == "revision":
        with manager.policy_update():
            manager._policy_revision += 1
        lease.revoke_revision.assert_called_once_with(manager._policy_revision)
    else:
        expected = OSError if phase == "exception-uncertain" else ValueError
        with pytest.raises(expected), manager.policy_update():
            raise ValueError("failed write")
    assert not manager._egress_updating
    assert manager._policy_update_depth == 0
    manager.unbind_egress(lease)
    manager.close()


def test_mediated_revision_rejects_a_late_positive_ack(tmp_path, monkeypatch):
    """A delayed closure reply cannot turn a spent revision deadline into success."""
    clock = [10.0]
    monkeypatch.setattr(manager_module, "monotonic", lambda: clock[0])
    manager = PermissionManager(tmp_path)
    lease = Mock()

    def late_ack(_revision):
        """Reply positively only after the independent two-second control bound."""
        clock[0] = 12.1
        return True

    lease.revoke_revision.side_effect = late_ack
    manager.bind_egress(manager.egress_policy_revision(), lease)
    with pytest.raises(OSError, match="transition is uncertain"):
        manager.set_yolo(True)
    assert lease in manager._egress_revokers
    manager.unbind_egress(lease)
    manager.close()


def test_mediated_policy_notifies_independent_leases_concurrently(tmp_path):
    """A slow control channel cannot consume another lease's acknowledgement window."""
    manager = PermissionManager(tmp_path)
    first, second = Mock(), Mock()
    for lease in (first, second):
        lease.pause.side_effect = lambda: pause_barrier.wait(timeout=2) >= 0
        lease.resume.side_effect = lambda: resume_barrier.wait(timeout=2) >= 0
        lease.revoke_revision.side_effect = lambda _revision: revoke_barrier.wait(timeout=2) >= 0
        manager.bind_egress(manager.egress_policy_revision(), lease)
    pause_barrier = Barrier(2)
    resume_barrier = Barrier(2)
    revoke_barrier = Barrier(2)
    with manager.policy_update():
        pass
    manager.set_yolo(True)
    assert not manager._egress_revokers
    for lease in (first, second):
        assert lease.pause.call_count == 2
        lease.resume.assert_called_once_with()
        lease.revoke_revision.assert_called_once_with(manager.egress_policy_revision())
        manager.unbind_egress(lease)
    manager.close()


def test_mediated_remembered_approval_captures_its_new_policy_revision(tmp_path):
    """A persisted session grant binds its issued revision before broker activation."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.info = Mock()
    interaction.prompt = Mock(return_value="session")
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, egress_origins=("http://example.com",))
    approved_revision = []
    assert manager.authorize_sandboxed_command(
        request, tool_id="run_command", approved_revision=approved_revision
    )
    assert approved_revision == [manager.egress_policy_revision()]
    assert manager.sandboxed_command_rules(ApprovalChoice.SESSION)
    manager.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_mediated_yolo_does_not_approve_after_a_second_revision_change(
    tmp_path, monkeypatch, asynchronous
):
    """A second policy update during automatic approval denies the stale scope."""
    manager = PermissionManager(tmp_path, yolo=True)
    request = native_request(tmp_path, egress_origins=("http://example.com",))
    current = (
        AsyncMock(side_effect=[True, False]) if asynchronous else Mock(side_effect=[True, False])
    )
    method = (
        "_sandbox_policy_revision_current_async"
        if asynchronous
        else "sandbox_policy_revision_current"
    )
    monkeypatch.setattr(manager, method, current)
    approved_revision = []
    if asynchronous:
        approved = asyncio.run(
            manager.authorize_sandboxed_command_async(
                request, tool_id="run_command", approved_revision=approved_revision
            )
        )
    else:
        approved = manager.authorize_sandboxed_command(
            request, tool_id="run_command", approved_revision=approved_revision
        )
    assert not approved
    assert not approved_revision
    assert current.call_count == 2
    manager.close()


def test_mediated_manager_destructor_does_not_mask_uncertain_closure(tmp_path):
    """Best-effort finalization contains an unconfirmed broker shutdown exception."""
    manager = PermissionManager(tmp_path)
    original = manager.close
    manager.close = Mock(side_effect=OSError("uncertain cleanup"))
    manager.__del__()
    manager.close.assert_called_once()
    manager.close = original
    manager.close()


def test_legacy_roots_remain_hard_until_trusted_reclassification(tmp_path):
    """A valid saved policy keeps root ceilings until the user classifies a scoped grant."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text(_HARD_LEGACY_READ_POLICY, encoding="utf-8")
    external = tmp_path.parent / "external-policy-root"
    external.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "approve"
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, read_roots=(external,))

    assert manager.configuration.limit_classifications.sources["readable_roots"] == "legacy"
    assert "readable_roots limit is hard" in manager.sandbox_scope_limit(request)
    assert not manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert not asyncio.run(
        manager.authorize_sandboxed_command_async(request, tool_id="run_command")
    )
    interaction.prompt.assert_not_called()

    manager.classify_limit("readable_roots", LimitClassification.APPROVAL)
    assert manager.configuration.limit_classifications.sources["readable_roots"] == "user"
    assert manager.sandbox_scope_limit(request) is None
    assert manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert interaction.prompt.call_count == 1

    manager.update_limit_values("readable_roots", "workspace", add=True)
    assert manager.configuration.limit_classifications.readable_roots is LimitClassification.HARD
    assert not manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert not manager.review_legacy_sandbox_read_limit(request, interaction)
    assert interaction.prompt.call_count == 1

    with pytest.raises(ValueError, match="Unknown collection permission limit"):
        manager.classify_limit("deny_private_networks", LimitClassification.APPROVAL)
    manager.classify_limit(
        "readable_roots", LimitClassification.APPROVAL, scope=PolicyScope.SESSION
    )
    assert (
        manager.effective_configuration.limit_classifications.sources["readable_roots"] == "session"
    )


def test_legacy_read_ceiling_preserves_bound_automatic_tool_access(tmp_path):
    """A legacy data-root ceiling does not remove validated default executable access."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text(_HARD_LEGACY_READ_POLICY, encoding="utf-8")
    installation = tmp_path.parent / "managed-tool"
    installation.mkdir()
    executable = installation / "fixture-tool"
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o755)
    details = executable.stat()
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(
        tmp_path,
        source="fixture-tool",
        read_roots=(installation,),
        environment={"PATH": str(installation), "LANG": "C"},
        executable_identities=(
            ("fixture-tool", executable, executable, details.st_dev, details.st_ino),
        ),
        automatic_tool_reads=(installation,),
    )

    assert manager.configuration.limit_classifications.sources["readable_roots"] == "legacy"
    assert manager.sandbox_scope_limit(request) is None
    assert not manager.review_legacy_sandbox_read_limit(request, interaction)
    assert manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert asyncio.run(manager.authorize_sandboxed_command_async(request, tool_id="run_command"))
    interaction.prompt.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    ["version: 1\n", "version: 1\nlimits:\n  readable_roots: [app-temp, workspace]\n"],
)
def test_legacy_default_read_limit_needs_only_scoped_command_approval(tmp_path, payload):
    """An unchanged saved default permits review while a later explicit hard limit persists."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text(payload, encoding="utf-8")
    external = tmp_path.parent / "ordinary-runtime-root"
    external.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "approve"
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, read_roots=(external,))

    assert (
        manager.configuration.limit_classifications.readable_roots is LimitClassification.APPROVAL
    )
    assert manager.configuration.limit_classifications.sources["readable_roots"] == "legacy"
    assert manager.sandbox_scope_limit(request) is None
    assert not manager.review_legacy_sandbox_read_limit(request, interaction)
    assert manager.authorize_sandboxed_command(request, tool_id="run_command")
    interaction.prompt.assert_called_once()
    assert interaction.prompt.call_args.args[0] == "Allow this command?"

    assert manager.update_limit_values("readable_roots", "workspace", add=False)
    assert manager.update_limit_values("readable_roots", "workspace", add=True)
    assert set(manager.configuration.limits.readable_roots) == set(
        PermissionConfiguration().limits.readable_roots
    )
    assert manager.configuration.limit_classifications.readable_roots is LimitClassification.HARD
    assert manager.configuration.limit_classifications.sources["readable_roots"] == "user"
    manager.close()

    reloaded = PermissionManager(tmp_path, interaction=interaction)
    assert "readable_roots limit is hard" in reloaded.sandbox_scope_limit(request)
    reloaded.close()


def test_legacy_read_review_refuses_stale_policy_choice(tmp_path):
    """A policy update during the trusted prompt cannot enable a stale scope review."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text(_HARD_LEGACY_READ_POLICY, encoding="utf-8")
    external = tmp_path.parent / "legacy-tool-root"
    external.mkdir()
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, read_roots=(external,))

    def update_policy(*_args, **_kwargs):
        """Change policy while the user reviews the old classification."""
        manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
        return "review"

    interaction.prompt.side_effect = update_policy
    assert not manager.review_legacy_sandbox_read_limit(request, interaction)
    assert manager.configuration.limit_classifications.sources["readable_roots"] == "legacy"
    assert "readable_roots limit is hard" in manager.sandbox_scope_limit(request)


def test_legacy_read_review_async_prompts_and_persists_only_fresh_trusted_choice(tmp_path):
    """Native async review can migrate a legacy ceiling without approving the command."""
    workspace = tmp_path / "workspace"
    path = workspace / ".loop" / "permissions.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(_HARD_LEGACY_READ_POLICY, encoding="utf-8")
    external = tmp_path / "legacy-tool-root"
    external.mkdir()
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value="review")
    manager = PermissionManager(workspace, interaction=interaction)
    request = native_request(workspace, read_roots=(external,))

    assert asyncio.run(manager.review_legacy_sandbox_read_limit_async(request, interaction))
    interaction.prompt_async.assert_awaited_once()
    assert manager.configuration.limit_classifications.sources["readable_roots"] == "user"
    assert manager.sandbox_scope_limit(request) is None
    assert not asyncio.run(manager.review_legacy_sandbox_read_limit_async(request, interaction))
    interaction.prompt_async.assert_awaited_once()


def test_legacy_read_review_async_preserves_denial_and_stale_choice(tmp_path):
    """Declined or stale async choices cannot change the saved hard ceiling."""
    workspace = tmp_path / "workspace"
    path = workspace / ".loop" / "permissions.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(_HARD_LEGACY_READ_POLICY, encoding="utf-8")
    external = tmp_path / "legacy-tool-root"
    external.mkdir()
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value="deny")
    manager = PermissionManager(workspace, interaction=interaction)
    request = native_request(workspace, read_roots=(external,))

    assert not asyncio.run(manager.review_legacy_sandbox_read_limit_async(request, interaction))

    async def change_policy(*_args, **_kwargs):
        """Invalidate the reviewed revision while input is pending."""
        manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
        return "review"

    interaction.prompt_async.side_effect = change_policy
    assert not asyncio.run(manager.review_legacy_sandbox_read_limit_async(request, interaction))
    assert manager.configuration.limit_classifications.sources["readable_roots"] == "legacy"


@pytest.mark.parametrize("stage", ["preflight", "commit"])
def test_legacy_read_review_async_settles_cancellation(tmp_path, monkeypatch, stage):
    """Cancellation settles owned policy work before propagating to the caller."""
    workspace = tmp_path / "workspace"
    path = workspace / ".loop" / "permissions.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(_HARD_LEGACY_READ_POLICY, encoding="utf-8")
    external = tmp_path / "legacy-tool-root"
    external.mkdir()
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value="review")
    manager = PermissionManager(workspace, interaction=interaction)
    request = native_request(workspace, read_roots=(external,))
    entered = Event()
    release = Event()
    method_name = (
        "_legacy_sandbox_read_review_revision"
        if stage == "preflight"
        else "_complete_legacy_sandbox_read_review"
    )
    original = getattr(manager, method_name)

    def blocked(*args):
        """Hold one policy worker until its owner cancels the review."""
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(manager, method_name, blocked)

    async def cancel_review():
        """Cancel at the selected worker boundary and wait for its settled result."""
        task = asyncio.create_task(
            manager.review_legacy_sandbox_read_limit_async(request, interaction)
        )
        assert await asyncio.to_thread(entered.wait, 5)
        try:
            task.cancel()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_review())
    assert (manager.sandbox_scope_limit(request) is None) is (stage == "commit")


def test_manual_write_roots_cannot_intersect_control_or_private_data(tmp_path):
    """Requested write roots remain disjoint from protected workspace and Loop state."""
    state = tmp_path.parent / "loop-state"
    state.mkdir()
    external = tmp_path.parent / "ordinary-external"
    external.mkdir()
    inner = tmp_path / "ordinary-inner"
    inner.mkdir()
    private = tmp_path.parent / ".ssh"
    private.mkdir()
    manager = PermissionManager(tmp_path, protected_host_roots=(state,))

    assert manager.sandbox_write_root_limit(tmp_path, (external, inner)) is None
    assert "protected workspace" in manager.sandbox_write_root_limit(
        tmp_path, (tmp_path / ".loop",)
    )
    assert "workspace or private" in manager.sandbox_write_root_limit(tmp_path, (tmp_path.parent,))
    assert "workspace or private" in manager.sandbox_write_root_limit(tmp_path, (private,))
    assert "protected application" in manager.sandbox_write_root_limit(tmp_path, (state,))


def test_external_write_roots_under_private_workspace_ancestor_stay_scoped(tmp_path):
    """An already selected private ancestor does not reject an ordinary approved sibling."""
    private_parent = tmp_path / "Library"
    workspace = private_parent / "project"
    external = private_parent / "approved"
    hidden = private_parent / ".ssh"
    for path in (workspace, external, hidden):
        path.mkdir(parents=True, exist_ok=True)
    manager = PermissionManager(workspace)

    assert manager.sandbox_write_root_limit(workspace, (external,)) is None
    assert "private host data" in manager.sandbox_write_root_limit(workspace, (hidden,))


def test_native_git_creation_approval_does_not_reuse_existing_git_grant(tmp_path):
    """A remembered fresh-repository grant cannot silently approve later metadata writes."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "approve"
    manager = PermissionManager(tmp_path, interaction=interaction)
    initial = native_request(tmp_path, source="git init", git_create=True)
    assert manager.authorize_sandboxed_command(initial, tool_id="run_command")
    (tmp_path / ".git").mkdir()
    interaction.prompt.return_value = "deny"
    assert not manager.authorize_sandboxed_command(
        native_request(
            tmp_path,
            source="git init",
            write_roots=(tmp_path, tmp_path / ".git"),
        ),
        tool_id="run_command",
    )
    assert interaction.prompt.call_count == 2


@pytest.mark.parametrize("scope", ["session", "workspace", "user"])
def test_native_scoped_approval_reuses_only_matching_authority(tmp_path, scope):
    """Native grants survive at their chosen scope but cannot approve broader effects."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = scope
    manager = PermissionManager(
        tmp_path,
        user_configuration_path=tmp_path / "user.yaml",
        interaction=interaction,
    )
    manager.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)
    approved = native_request(tmp_path)
    assert manager.authorize_sandboxed_command(approved, tool_id="run_command") is True
    interaction.prompt.reset_mock()
    assert manager.authorize_sandboxed_command(approved, tool_id="run_command") is True
    interaction.prompt.assert_not_called()
    interaction.prompt.return_value = "deny"
    assert (
        manager.authorize_sandboxed_command(
            native_request(tmp_path, network=True), interaction=None, tool_id="run_command"
        )
        is False
    )
    interaction.prompt.assert_called_once()
    assert len(manager.sandboxed_command_rules(ApprovalChoice.WORKSPACE)) == (
        1 if scope == "workspace" else 0
    )
    assert len(manager.user_rules) == 0


def test_native_similar_rule_accepts_simple_prefix_only(tmp_path):
    """A reviewed similar rule never matches a compound shell script."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "similar_session"
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert (
        manager.authorize_sandboxed_command(
            native_request(tmp_path, "git status"), tool_id="run_command"
        )
        is True
    )
    interaction.prompt.reset_mock()
    assert (
        manager.authorize_sandboxed_command(
            native_request(tmp_path, "git status --short"), tool_id="run_command"
        )
        is True
    )
    interaction.prompt.assert_not_called()
    interaction.prompt.return_value = "deny"
    assert (
        manager.authorize_sandboxed_command(
            native_request(tmp_path, "git status; printf unsafe"), tool_id="run_command"
        )
        is False
    )
    assert interaction.prompt.call_count == 1
    assert "similar_session" not in interaction.prompt.call_args.kwargs["choices"]


def test_native_prompt_shows_command_reason_and_scope(tmp_path):
    """Approval text identifies the command, reason, and actual authority."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert (
        manager.authorize_sandboxed_command(
            native_request(tmp_path, "printf change"), tool_id="run_command"
        )
        is False
    )
    message = interaction.info.call_args.args[0]
    assert 'Command: "printf change"' in message
    assert "Reason:" in message
    assert "read files outside this workspace" in message
    assert "PATH:" not in message
    assert "macos-seatbelt" not in message
    assert message.index("Command:") < message.index("Scope:") < message.index("Context:")
    assert prompt_scope(message)["workspace"] == str(tmp_path)


def test_native_prompt_reports_exact_authority_and_escapes_model_reason(tmp_path):
    """Human approval receives the actual scope and an escaped, unaudited explanation."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / 'tool "directory"'
    external.mkdir()
    alias = tmp_path / "tool-alias"
    alias.symlink_to(external, target_is_directory=True)
    executable = external / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o700)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    details = executable.stat()
    interaction = Mock(spec=Interaction)
    interaction.prompt.side_effect = ["details", "deny"]
    audit_path = tmp_path / "audit.db"
    manager = PermissionManager(workspace, interaction=interaction, audit_path=audit_path)
    manager.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)
    request = native_request(
        workspace,
        "tool",
        read_roots=(external,),
        automatic_tool_reads=(external,),
        read_aliases=(alias,),
        write_roots=(workspace, external),
        executable_identities=(("tool", executable, executable, details.st_dev, details.st_ino),),
        environment={"PATH": str(external), "TMPDIR": str(scratch)},
        scratch_root=scratch,
        network=True,
        git_create=True,
    )
    reason = "Install dependencies\n\x1b[2J"

    assert not manager.authorize_sandboxed_command(request, tool_id="run_command", reason=reason)

    message = interaction.info.call_args_list[0].args[0]
    summary = prompt_scope(message)
    assert summary["workspace"] == str(workspace)
    assert summary["workspace_write"] is True
    assert summary["write_roots"] == [str(external)]
    assert summary["network"] is True and summary["git_create"] is True
    assert summary["automatic_tool_read_roots"] == 1
    for hidden in (scratch, alias):
        assert str(hidden) not in message
        assert json.dumps(str(hidden))[1:-1] not in message
    details_message = interaction.info.call_args_list[1].args[0]
    scope = prompt_scope(details_message, full=True)
    assert scope["read_roots"] == [str(external)]
    assert scope["automatic_tool_reads"] == [str(external)]
    assert scope["read_aliases"] == [str(alias)]
    assert scope["write_roots"] == [str(workspace), str(external)]
    assert scope["executables"] == [
        {"name": "tool", "spelling": str(executable), "target": str(executable)}
    ]
    assert scope["network"] is True
    assert scope["git_create"] is True
    assert scope["scratch_root"] == str(scratch)
    assert scope["temporary_directory"] == str(scratch)
    assert message.endswith(f"Agent explanation: {json.dumps(reason)}")
    assert details_message.endswith(f"Agent explanation: {json.dumps(reason)}")
    assert details_message.index("Full scope:") < details_message.index("Context:")
    assert "\x1b" not in message
    with closing(sqlite3.connect(audit_path)) as connection:
        payloads = connection.execute(
            "SELECT payload_json FROM permission_audit_records"
        ).fetchall()
    assert all("Install dependencies" not in payload[0] for payload in payloads)


def test_async_native_prompt_includes_model_reason(tmp_path):
    """Asynchronous approval displays the same escaped model explanation."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.info = Mock()
    interaction.prompt_async = AsyncMock(return_value="deny")
    manager = PermissionManager(tmp_path, interaction=interaction)

    assert not asyncio.run(
        manager.authorize_sandboxed_command_async(
            native_request(tmp_path), tool_id="run_command", reason="Read the toolchain"
        )
    )

    assert 'Agent explanation: "Read the toolchain"' in interaction.info.call_args.args[0]


def test_async_native_prompt_can_show_details_before_approval(tmp_path):
    """Showing the full scope does not itself approve or replace the subsequent decision."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.info = Mock()
    interaction.prompt_async = AsyncMock(side_effect=["details", "approve"])
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, "printf text > output", read_roots=())

    assert asyncio.run(manager.authorize_sandboxed_command_async(request, tool_id="run_command"))

    assert interaction.prompt_async.await_count == 2
    details = interaction.info.call_args_list[1].args[0]
    assert prompt_scope(details, full=True)["workspace"] == str(tmp_path)
    assert details.index("Full scope:") < details.index("Reason:")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("decision", ["approve", "deny"])
def test_sandbox_scope_can_be_toggled_before_deciding(tmp_path, asynchronous, decision):
    """Scope details follow approval choices and toggle between F and R before a decision."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.info = Mock()
    manager = PermissionManager(
        tmp_path, user_configuration_path=tmp_path / "user.yaml", interaction=interaction
    )
    request = native_request(tmp_path, "git status")
    responses = iter(("details", "details", "details", "details", decision))
    prompts = []

    def capture_prompt(*args, **kwargs):
        """Snapshot each choice catalog before the next scope toggle."""
        prompts.append({"choices": dict(kwargs["choices"]), "index": dict(kwargs["index"])})
        return next(responses)

    async def capture_prompt_async(*args, **kwargs):
        """Snapshot each asynchronous choice catalog before the next scope toggle."""
        return capture_prompt(*args, **kwargs)

    if asynchronous:
        interaction.prompt_async = AsyncMock(side_effect=capture_prompt_async)
        approved = asyncio.run(
            manager.authorize_sandboxed_command_async(request, tool_id="run_command")
        )
    else:
        interaction.prompt = Mock(side_effect=capture_prompt)
        approved = manager.authorize_sandboxed_command(request, tool_id="run_command")

    assert approved is (decision == "approve")
    assert [prompt["index"]["details"] for prompt in prompts] == ["F", "R", "F", "R", "F"]
    for prompt in prompts:
        labels = list(prompt["choices"].values())
        assert labels[-1] in {"Show full scope", "Show reduced scope"}
        assert all(label == "Deny" or label.startswith("Allow") for label in labels[:-1])
    assert [prompt["choices"]["details"] for prompt in prompts] == [
        "Show full scope",
        "Show reduced scope",
        "Show full scope",
        "Show reduced scope",
        "Show full scope",
    ]
    messages = [call.args[0] for call in interaction.info.call_args_list]
    assert messages[0] == messages[2]
    assert messages[1].startswith("Command: ")
    assert messages[1].index("Full scope:") < messages[1].index("Context:")
    assert messages[1] == messages[3]
    assert messages[0] == messages[4]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sandbox_details_do_not_bypass_policy_changes(tmp_path, asynchronous):
    """A policy change during detail review invalidates the pending decision."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    manager = PermissionManager(tmp_path, interaction=interaction)

    def show(message):
        """Change policy when the user requests the complete scope."""
        if "\nFull scope: " in message:
            manager.add_rule(
                PermissionRule(
                    decision=Decision.DENY, tool="run_command", action=Action.PROCESS_EXECUTE
                ),
                scope=PolicyScope.SESSION,
            )

    interaction.info = Mock(side_effect=show)
    request = native_request(tmp_path, "printf text > output", read_roots=())
    if asynchronous:
        interaction.prompt_async = AsyncMock(side_effect=["details", "approve"])
        approved = asyncio.run(
            manager.authorize_sandboxed_command_async(request, tool_id="run_command")
        )
    else:
        interaction.prompt = Mock(side_effect=["details", "approve"])
        approved = manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert not approved


def test_native_path_read_authority_is_disclosed_and_bound_to_remembered_rule(tmp_path):
    """A changed external read grant is displayed and requires another approval."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.side_effect = ["session", "deny"]
    manager = PermissionManager(workspace, interaction=interaction)
    initial = native_request(
        workspace, "tool", environment={"PATH": str(first)}, read_roots=(first,)
    )
    assert manager.authorize_sandboxed_command(initial, tool_id="run_command")
    scope = prompt_scope(interaction.info.call_args.args[0])
    assert str(first) in scope["read_roots"]
    changed = native_request(
        workspace, "tool", environment={"PATH": str(second)}, read_roots=(second,)
    )
    assert not manager.authorize_sandboxed_command(changed, tool_id="run_command")
    scope = prompt_scope(interaction.info.call_args.args[0])
    assert str(second) in scope["read_roots"]
    assert interaction.prompt.call_count == 2


def test_native_remembered_read_rule_binds_executable_identity(tmp_path):
    """Approval for one installed binary cannot authorize another in the same read root."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.side_effect = ["session", "deny"]
    manager = PermissionManager(workspace, interaction=interaction)
    for name, expected in (("first", True), ("second", False)):
        executable = installation / name
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
        details = executable.stat()
        bound = native_request(
            workspace,
            "opaque-tool",
            read_roots=(installation,),
            environment={"PATH": str(installation)},
            executable_identities=((name, executable, executable, details.st_dev, details.st_ino),),
        )
        assert manager.authorize_sandboxed_command(bound, tool_id="run_command") is expected
    assert interaction.prompt.call_count == 2


def test_native_prompt_without_path_does_not_claim_search_reads(tmp_path):
    """Approval text omits executable search details when PATH is absent."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, environment={"LANG": "C"})
    assert not manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert "search directories" not in interaction.info.call_args.args[0]


def test_native_persistent_rules_reload_and_can_be_removed(tmp_path):
    """Workspace and user native approvals survive restart and remain revocable."""
    interaction = Mock(spec=Interaction)
    user_path = tmp_path / "user.yaml"
    manager = PermissionManager(
        tmp_path, user_configuration_path=user_path, interaction=interaction
    )
    request = native_request(tmp_path, "printf workspace")
    interaction.prompt.return_value = "workspace"
    assert manager.authorize_sandboxed_command(request, tool_id="run_command")
    interaction.prompt.return_value = "user"
    assert manager.authorize_sandboxed_command(
        native_request(tmp_path, "printf user"), tool_id="run_command"
    )
    loaded = PermissionManager(tmp_path, user_configuration_path=user_path)
    workspace_rules = loaded.sandboxed_command_rules(ApprovalChoice.WORKSPACE)
    user_rules = loaded.sandboxed_command_rules(ApprovalChoice.USER)
    assert len(workspace_rules) == len(user_rules) == 1
    assert loaded.authorize_sandboxed_command(request, tool_id="run_command")
    assert loaded.remove_sandboxed_command_rule(ApprovalChoice.WORKSPACE, workspace_rules[0].id)
    assert len(loaded.sandboxed_command_rules(ApprovalChoice.USER)) == 1
    assert loaded.remove_sandboxed_command_rule(ApprovalChoice.USER, user_rules[0].id)
    assert not loaded.remove_sandboxed_command_rule(ApprovalChoice.USER, user_rules[0].id)
    assert not PermissionManager(
        tmp_path, user_configuration_path=user_path
    ).sandboxed_command_rules(ApprovalChoice.USER)


def test_native_persistent_rules_reject_duplicate_ids_and_unsupported_authority(tmp_path):
    """Malformed persisted sandbox approvals fail visibly before matching or revocation."""
    from loop.permissions import SandboxedCommandRule, UserPermissionConfiguration

    first = SandboxedCommandRule(signature="a", source="one", scope=ApprovalChoice.WORKSPACE)
    second = SandboxedCommandRule(
        id=first.id, signature="b", source="two", scope=ApprovalChoice.USER
    )
    with pytest.raises(ValueError, match="Sandbox command rule identifiers"):
        UserPermissionConfiguration(sandboxed_command_rules=[first, second])
    with pytest.raises(ValueError, match="all_sources"):
        SandboxedCommandRule.model_validate(
            {"signature": "a", "source": "one", "all_sources": True}
        )
    user_path = tmp_path / "user.yaml"
    user_path.write_text(
        dump_yaml(
            {
                "sandboxed_command_rules": [
                    first.model_dump(mode="json"),
                    second.model_dump(mode="json"),
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionConfigurationError, match="identifiers must be unique"):
        PermissionManager(tmp_path, user_configuration_path=user_path)
    user_path.write_text(
        dump_yaml(
            {"sandboxed_command_rules": [{**first.model_dump(mode="json"), "all_sources": True}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionConfigurationError, match="unrecognized field"):
        PermissionManager(tmp_path, user_configuration_path=user_path)


def test_native_workspace_approval_is_not_saved_in_repository_policy(tmp_path):
    """A repository policy cannot become the source of a native allow decision."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "workspace"
    user_path = tmp_path / "user.yaml"
    manager = PermissionManager(
        tmp_path, user_configuration_path=user_path, interaction=interaction
    )
    assert manager.authorize_sandboxed_command(native_request(tmp_path), tool_id="run_command")
    assert manager.sandboxed_command_rules(ApprovalChoice.WORKSPACE)
    workspace_policy = tmp_path / ".loop" / "permissions.yaml"
    assert (
        not workspace_policy.exists()
        or "sandboxed_command_rules" not in workspace_policy.read_text()
    )
    assert "sandboxed_command_rules" in user_path.read_text()


def test_native_session_rule_resets_and_rejects_invalid_scope(tmp_path):
    """Session approval is cleared by reset and cannot use a one-off scope."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "session"
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert manager.authorize_sandboxed_command(native_request(tmp_path), tool_id="run_command")
    assert len(manager.sandboxed_command_rules(ApprovalChoice.SESSION)) == 1
    assert manager.reset_session()
    assert not manager.sandboxed_command_rules(ApprovalChoice.SESSION)
    with pytest.raises(ValueError, match="require session"):
        manager.sandboxed_command_rules(ApprovalChoice.ONCE)


def test_native_similar_rule_does_not_match_changed_environment(tmp_path):
    """A similar source cannot reuse authority after environment changes."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "similar_session"
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert manager.authorize_sandboxed_command(native_request(tmp_path), tool_id="run_command")
    interaction.prompt.return_value = "deny"
    changed = native_request(tmp_path, environment={"PATH": "/bin", "LANG": "C"})
    assert not manager.authorize_sandboxed_command(changed, tool_id="run_command")


def test_native_derived_cache_path_keeps_approval_but_external_cache_does_not(tmp_path):
    """Per-command scratch changes reuse approval while an external cache path cannot."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "session"
    manager = PermissionManager(tmp_path, interaction=interaction)

    def request(scratch: Path, cache: Path, ruff_cache: Path | None = None) -> SandboxRequest:
        """Bind one command with an explicit process cache location."""
        return native_request(
            tmp_path,
            generated_environment=tuple(
                name
                for name, generated in (
                    ("TMPDIR", True),
                    ("XDG_CACHE_HOME", cache == scratch / "cache"),
                    ("RUFF_CACHE_DIR", ruff_cache is None),
                    ("COVERAGE_FILE", True),
                )
                if generated
            ),
            environment={
                "PATH": "/usr/bin:/bin",
                "LANG": "C",
                "TMPDIR": str(scratch),
                "XDG_CACHE_HOME": str(cache),
                "RUFF_CACHE_DIR": str(ruff_cache or scratch / "ruff-cache"),
                "COVERAGE_FILE": str(scratch / ".coverage"),
            },
        )

    with (
        tempfile.TemporaryDirectory(
            prefix="loop-seatbelt-", dir=Path(tempfile.gettempdir()).resolve()
        ) as first_value,
        tempfile.TemporaryDirectory(
            prefix="loop-seatbelt-", dir=Path(tempfile.gettempdir()).resolve()
        ) as second_value,
    ):
        first = Path(first_value)
        second = Path(second_value)
        assert manager.authorize_sandboxed_command(
            request(first, first / "cache"), tool_id="run_command"
        )
        assert manager.authorize_sandboxed_command(
            request(second, second / "cache"), tool_id="run_command"
        )
        interaction.prompt.assert_called_once()
        interaction.prompt.return_value = "deny"
        assert not manager.authorize_sandboxed_command(
            request(second, tmp_path / "external-cache"), tool_id="run_command"
        )
        assert interaction.prompt.call_count == 2
        assert not manager.authorize_sandboxed_command(
            request(second, second / "cache", tmp_path / "external-ruff-cache"),
            tool_id="run_command",
        )
        assert interaction.prompt.call_count == 3


def test_native_exception_grant_prompts_with_exact_path(tmp_path):
    """An extra read root prompts with its exact path."""
    external = tmp_path / "external"
    external.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, "cat extra", read_roots=(external,), write_roots=())
    assert not manager.authorize_sandboxed_command(request, tool_id="run_command")
    assert "read files outside this workspace" in interaction.info.call_args.args[0]
    scope = prompt_scope(interaction.info.call_args.args[0])
    assert str(external) in scope["read_roots"]


def test_native_prompt_shows_workspace_write_when_external_root_contains_it(tmp_path):
    """An approved parent write root also permits writes inside the workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(workspace, interaction=interaction)
    request = native_request(workspace, "true", read_roots=(), write_roots=(tmp_path,))

    assert not manager.authorize_sandboxed_command(request, tool_id="run_command")

    scope = prompt_scope(interaction.info.call_args.args[0])
    assert scope["workspace_write"] is True
    assert scope["write_roots"] == [str(tmp_path)]


def test_native_prompt_discloses_additional_read_alias(tmp_path):
    """An additional read alias and its exact target remain visible."""
    target = tmp_path.parent / "extra-target"
    target.mkdir()
    physical_alias = tmp_path / "extra-alias"
    physical_alias.symlink_to(target, target_is_directory=True)
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(
        tmp_path,
        "true",
        read_roots=(target,),
        read_aliases=(physical_alias,),
    )

    assert not manager.authorize_sandboxed_command(request, tool_id="run_command")

    scope = prompt_scope(interaction.info.call_args.args[0])
    assert scope["read_roots"] == [str(target)]
    assert scope["read_aliases"] == 1


def test_native_read_prompt_distinguishes_tool_folder_data_and_broad_scope(tmp_path):
    """Extra-read requests disclose their practical scope and exact paths."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(workspace, interaction=interaction)
    cases = (
        ((external,), "Use installed python3"),
        ((external,), "Read an additional folder outside the workspace"),
        ((tmp_path,), "Read a broad area outside the workspace"),
    )
    for roots, expected in cases:
        context = expected if expected.startswith("Use installed") else None
        request = native_request(
            workspace, "opaque-command", read_roots=roots, read_context=context
        )
        assert not manager.authorize_sandboxed_command(request, tool_id="run_command")
        message = interaction.info.call_args.args[0]
        assert f"Context: {expected}." in message
        assert 'Command: "opaque-command"' in message
        scope = prompt_scope(message)
        assert scope["read_roots"] == [str(root) for root in roots]


def test_native_automatic_tool_reads_are_not_presented_as_permission_reasons(tmp_path):
    """A destructive review does not claim that already authorized tool reads need approval."""
    tool_root = tmp_path.parent / "installed-tool"
    tool_root.mkdir()
    executable = tool_root / "tool"
    executable.write_text("installed", encoding="utf-8")
    executable.chmod(0o700)
    details = executable.stat()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    bound = native_request(
        tmp_path,
        "rm victim",
        read_roots=(tool_root,),
        automatic_tool_reads=(tool_root,),
        executable_identities=(("tool", executable, executable, details.st_dev, details.st_ino),),
        environment={"PATH": str(tool_root)},
        read_context="Use installed developer tools",
    )
    assert not manager.authorize_sandboxed_command(bound, tool_id="run_command")
    message = interaction.info.call_args.args[0]
    assert "Review a destructive workspace command" in message
    assert "read files outside" not in message
    assert "Use installed developer tools" not in message
    assert str(tool_root) not in message
    assert json.dumps(str(tool_root))[1:-1] not in message
    assert '"automatic_tool_read_roots": 1' in message


@pytest.mark.parametrize("escape", [False, True])
def test_native_redirection_exemption_uses_bound_temporary_root(tmp_path, escape):
    """Only a contained session-scratch destination can bypass output review."""
    scratch = tmp_path.parent / "managed-scratch"
    scratch.mkdir(mode=0o700)
    (scratch / "link").symlink_to(tmp_path, target_is_directory=True)
    suffix = "link/victim" if escape else "report"
    bound = native_request(
        tmp_path,
        f"printf text >{shlex.quote(str(scratch / suffix))}",
        environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)},
        scratch_root=scratch,
        read_roots=(),
        write_roots=(tmp_path, scratch),
    )
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert manager.authorize_sandboxed_command(bound, tool_id="run_command") is not escape
    assert interaction.prompt.call_count == int(escape)


def test_native_literal_command_scratch_redirection_needs_no_review(tmp_path):
    """The bound private command TMPDIR is an output exemption without trusting variables."""
    with tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as directory:
        scratch = Path(directory).resolve()
        interaction = Mock(spec=Interaction)
        interaction.prompt.return_value = "deny"
        manager = PermissionManager(tmp_path, interaction=interaction)
        bound = native_request(
            tmp_path,
            f"printf text >{shlex.quote(str(scratch / 'report'))}",
            read_roots=(),
            environment={"TMPDIR": str(scratch)},
        )
        assert manager.authorize_sandboxed_command(bound, tool_id="run_command")
        interaction.prompt.assert_not_called()


def test_native_read_scope_audit_and_prompt_show_exact_roots(tmp_path):
    """The human prompt and local audit identify the approved roots."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "tool"
    external.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "approve"
    audit_path = tmp_path / "audit.db"
    manager = PermissionManager(
        workspace, interaction=interaction, audit_path=audit_path, workspace_id="workspace"
    )
    approved = native_request(
        workspace, "opaque-tool", read_roots=(external,), read_context="Use installed tool"
    )
    assert manager.authorize_sandboxed_command(approved, tool_id="run_command")
    scope = prompt_scope(interaction.info.call_args.args[0])
    assert str(external) in scope["read_roots"]
    with closing(sqlite3.connect(audit_path)) as connection:
        row = connection.execute(
            "SELECT payload_json FROM permission_audit_records "
            "WHERE event_name = 'sandbox.permission_scope'"
        ).fetchone()
    assert json.loads(row[0])["read_roots"] == [str(external)]


@pytest.mark.parametrize("source", ["python -c print", "git --version"])
def test_native_similar_choice_excludes_script_engines_and_flag_prefixes(tmp_path, source):
    """Broad script engines and option prefixes cannot create similar rules."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert not manager.authorize_sandboxed_command(
        native_request(tmp_path, source), tool_id="run_command"
    )
    assert "similar_session" not in interaction.prompt.call_args.kwargs["choices"]


def test_native_persistence_failure_denies_approval(tmp_path, monkeypatch):
    """A failed remembered-rule write cannot be treated as permission."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "workspace"
    manager = PermissionManager(
        tmp_path, user_configuration_path=tmp_path / "user.yaml", interaction=interaction
    )
    monkeypatch.setattr(
        manager, "_replace_user_configuration", Mock(side_effect=OSError("disk full"))
    )
    assert not manager.authorize_sandboxed_command(native_request(tmp_path), tool_id="run_command")


def test_native_rule_removal_rejects_once_scope_and_removes_session_rule(tmp_path):
    """One-off approvals are never managed as persistent rules."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "session"
    manager = PermissionManager(tmp_path, interaction=interaction)
    assert manager.authorize_sandboxed_command(native_request(tmp_path), tool_id="run_command")
    rule_id = manager.sandboxed_command_rules(ApprovalChoice.SESSION)[0].id
    assert manager.remove_sandboxed_command_rule(ApprovalChoice.SESSION, rule_id)
    with pytest.raises(ValueError, match="require session"):
        manager.remove_sandboxed_command_rule(ApprovalChoice.ONCE, rule_id)


def test_legacy_process_approval_never_approves_native_sandbox_request(tmp_path):
    """Persisted v1 host rules and process defaults cannot skip fresh native approval."""
    configuration = PermissionConfiguration(
        defaults={**PermissionConfiguration().defaults, Action.PROCESS_EXECUTE: Decision.ALLOW},
        limits=PolicyLimits(allow_host_processes=True),
        rules=[
            PermissionRule(
                decision=Decision.ALLOW,
                tool="run_command",
                action=Action.PROCESS_EXECUTE,
                resource="*",
            )
        ],
    )
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "deny"
    policy_path = tmp_path / "permissions.yaml"
    policy_path.write_text(dump_yaml(configuration.model_dump(mode="json")))
    user_rule = PermissionRule(
        decision=Decision.ALLOW,
        tool="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "printf ok"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.HOST,
        ),
    )
    user_path = tmp_path / "user-permissions.yaml"
    user_path.write_text(dump_yaml({"version": 1, "rules": [user_rule.model_dump(mode="json")]}))
    manager = PermissionManager(
        tmp_path, configuration_path=policy_path, user_configuration_path=user_path
    )
    request = SandboxRequest.create(
        source="printf ok",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(tmp_path.parent,),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin"},
        policy_version="macos-v1",
        deadline=time.monotonic() + 60,
        workspace_id="workspace-1",
    )

    assert (
        manager.authorize_sandboxed_command(request, interaction=interaction, tool_id="run_command")
        is False
    )
    interaction.prompt.assert_called_once()
    assert not manager.configuration.rules[0].target
    interaction.prompt.return_value = "approve"
    assert (
        manager.authorize_sandboxed_command(request, interaction=interaction, tool_id="run_command")
        is True
    )
    assert len(manager.configuration.rules) == 1


def test_shutdown_cleanup_releases_live_manager_temporary_directories():
    """Process shutdown cleanup releases temporary directories retained by live managers."""
    manager = PermissionManager()

    manager_module._close_live_managers()  # pylint: disable=protected-access

    assert not manager.temporary_directory.exists()


@pytest.mark.parametrize(
    ("workspace_available", "user_available", "selection", "expected", "workspace_label"),
    [
        (True, True, ApprovalChoice.WORKSPACE, ApprovalChoice.WORKSPACE, True),
        (False, False, ApprovalChoice.SESSION, ApprovalChoice.SESSION, False),
        (True, True, False, ApprovalChoice.DENY, True),
    ],
)
def test_request_permission_offers_valid_scopes_and_fails_closed(
    workspace_available,
    user_available,
    selection,
    expected,
    workspace_label,
    tmp_path,
):
    """Permission prompting owns scoped labels and converts cancellation to denial."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = selection
    recorder = Mock()
    manager = PermissionManager(
        tmp_path if workspace_available else None,
        user_configuration_path=tmp_path / "user.yaml" if user_available else None,
        interaction=interaction,
        recorder=recorder,
    )

    result = manager.request_permission(
        "Approve operations?",
        interaction=interaction,
    )

    assert result is expected
    choices = interaction.prompt.call_args.kwargs["choices"]
    assert (ApprovalChoice.WORKSPACE in choices) is workspace_label
    assert (ApprovalChoice.USER in choices) is user_available


def test_request_permission_forwards_index_map_to_prompt(tmp_path):
    """Permission prompts forward short letter indexes to the generic prompt layer."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(
        tmp_path, user_configuration_path=tmp_path / "user.yaml", interaction=interaction
    )

    manager.request_permission("Approve?")

    index = interaction.prompt.call_args.kwargs["index"]
    assert index is not None
    assert index[ApprovalChoice.DENY] == "N"
    assert index[ApprovalChoice.ONCE] == "Y"
    assert index[ApprovalChoice.SESSION] == "S"
    assert index[ApprovalChoice.USER] == "U"
    interaction.info.assert_called_once_with("Approve?")
    assert interaction.prompt.call_args.args[0] == "Proceed?"


def test_request_permission_includes_workspace_index_when_configured(tmp_path):
    """Workspace scope adds a ``W`` index letter when a configuration path exists."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(tmp_path, interaction=interaction)

    manager.request_permission("Approve?")

    index = interaction.prompt.call_args.kwargs["index"]
    assert index[ApprovalChoice.WORKSPACE] == "W"


def test_request_permission_displays_details_before_choices_and_proceed_prompt(tmp_path):
    """Permission details precede the approval catalog and its final input prompt."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    calls = Mock()
    calls.attach_mock(interaction.info, "info")
    calls.attach_mock(interaction.prompt, "prompt")
    manager = PermissionManager(tmp_path, interaction=interaction)

    manager.request_permission("Approve these operations?")

    assert calls.mock_calls[0] == call.info("Approve these operations?")
    assert calls.mock_calls[1].args == ("Proceed?",)


def test_user_approval_persists_across_workspace_managers(tmp_path):
    """A remembered user approval authorizes its exact operation in another workspace."""
    user_path = tmp_path / "user" / "permissions.yaml"
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.USER
    requested = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com/api", origin="https://example.com"),
    )
    first = PermissionManager(
        first_workspace, user_configuration_path=user_path, interaction=interaction
    )

    approved = first.authorize((requested,))
    second = PermissionManager(second_workspace, user_configuration_path=user_path)

    assert approved.decision is Decision.ALLOW
    assert approved.approval_choice is ApprovalChoice.USER
    assert approved.installed_rule_ids
    assert second.evaluate((requested,)).decision is Decision.ALLOW
    assert len(second.user_rules) == 1


def test_workspace_deny_overrides_a_matching_user_approval(tmp_path):
    """A workspace-local denial remains stronger than a user-wide exact allow rule."""
    user_path = tmp_path / "user" / "permissions.yaml"
    requested = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com/api", origin="https://example.com"),
    )
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.USER
    approving = PermissionManager(
        tmp_path / "first", user_configuration_path=user_path, interaction=interaction
    )
    approving.authorize((requested,))
    denying = PermissionManager(
        tmp_path / "second",
        user_configuration_path=user_path,
        configuration=PermissionConfiguration(
            rules=[
                PermissionRule(
                    decision=Decision.DENY,
                    tool=requested.tool_id,
                    tool_exact=True,
                    action=requested.action,
                    target=requested.target,
                )
            ]
        ),
    )

    result = denying.evaluate((requested,))

    assert result.decision is Decision.DENY
    assert result.sources[0].startswith("rule:workspace:")


@pytest.mark.parametrize("content", ["", "null\n", "false\n", "0\n", "[]\n"])
@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_permission_policy_requires_a_mapping(tmp_path, content, scope):
    """Existing non-mapping policy documents never become default permissions."""
    path = tmp_path / "permissions.yaml"
    path.write_text(content, encoding="utf-8")
    arguments = (
        {"configuration_path": path} if scope == "workspace" else {"user_configuration_path": path}
    )

    with pytest.raises(PermissionConfigurationError, match="Policy document must be a mapping"):
        PermissionManager(tmp_path, **arguments)
    assert path.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_permission_policy_rejects_empty_mapping(tmp_path, scope):
    """A present but empty policy is not interpreted as a default policy."""
    path = tmp_path / "permissions.yaml"
    path.write_text("{}\n", encoding="utf-8")
    arguments = (
        {"configuration_path": path} if scope == "workspace" else {"user_configuration_path": path}
    )
    with pytest.raises(PermissionConfigurationError, match="must not be empty"):
        PermissionManager(tmp_path, **arguments)


@pytest.mark.parametrize("scope", ["workspace", "user"])
@pytest.mark.parametrize(
    "content",
    [
        "version: 1\nversion: 2\n",
        "rules:\n  - decision: deny\n    target: {kind: file, path: /tmp/x, unknown: true}\n",
    ],
)
def test_permission_policy_rejects_ambiguous_or_unknown_fields(tmp_path, scope, content):
    """Persisted rules cannot silently lose unknown fields or duplicate keys."""
    path = tmp_path / "permissions.yaml"
    path.write_text(content, encoding="utf-8")
    arguments = (
        {"configuration_path": path} if scope == "workspace" else {"user_configuration_path": path}
    )
    with pytest.raises(PermissionConfigurationError):
        PermissionManager(tmp_path, **arguments)
    assert path.read_text(encoding="utf-8") == content


def test_invalid_user_policy_can_continue_with_admission_blocked(tmp_path):
    """An explicit recovery choice preserves the invalid file and blocks admission."""
    user_path = tmp_path / "user" / "permissions.yaml"
    user_path.parent.mkdir()
    user_path.write_text("version: invalid\n", encoding="utf-8")
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "continue"

    manager = PermissionManager(
        tmp_path / "workspace", user_configuration_path=user_path, interaction=interaction
    )

    assert not manager.user_rules
    assert user_path.read_text(encoding="utf-8") == "version: invalid\n"
    interaction.report.assert_called_once()
    interaction.warning.assert_called_once_with(
        "Permission operations remain blocked until the policy is repaired."
    )


@pytest.mark.parametrize("failure", ["missing", "invalid"])
def test_user_policy_recovery_retains_restrictions_and_discards_approvals(tmp_path, failure):
    """Unavailable user policy cannot drop restrictions or retain remembered approvals."""
    path = tmp_path / "user.yaml"
    policy = UserPermissionConfiguration(
        rules=[
            PermissionRule(id="deny", decision=Decision.DENY, action=Action.FILESYSTEM_READ),
            PermissionRule(id="ask", decision=Decision.ASK, action=Action.FILESYSTEM_LIST),
            PermissionRule(id="allow", decision=Decision.ALLOW, action=Action.FILESYSTEM_DELETE),
        ],
        sandboxed_command_rules=[
            SandboxedCommandRule(
                id="sandbox", signature="digest", source="true", scope=ApprovalChoice.USER
            ),
            SandboxedCommandRule(
                id="workspace", signature="digest", source="true", scope=ApprovalChoice.WORKSPACE
            ),
        ],
    )
    path.write_text(dump_yaml(policy.model_dump(mode="json")), encoding="utf-8")
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "continue"
    manager = PermissionManager(tmp_path, user_configuration_path=path, interaction=interaction)
    assert manager.explain("demo", Action.FILESYSTEM_DELETE, str(tmp_path / "file")).decision is (
        Decision.ALLOW
    )
    if failure == "missing":
        path.unlink()
    else:
        path.write_text("version: invalid\n", encoding="utf-8")

    manager.reload()

    assert [rule.id for rule in manager.user_rules] == ["deny", "ask"]
    assert not manager.sandboxed_command_rules(ApprovalChoice.USER)
    assert not manager.sandboxed_command_rules(ApprovalChoice.WORKSPACE)
    assert manager.explain("demo", Action.FILESYSTEM_READ, str(tmp_path / "file")).decision is (
        Decision.DENY
    )
    expected = Decision.DENY if failure == "invalid" else Decision.ASK
    assert (
        manager.explain("demo", Action.FILESYSTEM_LIST, str(tmp_path / "file")).decision is expected
    )
    assert (
        manager.explain("demo", Action.FILESYSTEM_DELETE, str(tmp_path / "file")).decision
        is expected
    )
    if failure == "invalid":
        interaction.report.assert_called_once()
        interaction.warning.assert_called_once_with(
            "Permission operations remain blocked until the policy is repaired."
        )
    else:
        interaction.report.assert_not_called()
    path.write_text("version: 1\nrules: []\n", encoding="utf-8")
    manager.reload()
    assert not manager.user_rules


def test_invalid_user_policy_raises_in_strict_mode(tmp_path):
    """Strict loading exposes an invalid user policy instead of applying any approval."""
    user_path = tmp_path / "user" / "permissions.yaml"
    user_path.parent.mkdir()
    user_path.write_text("version: invalid\n", encoding="utf-8")

    with pytest.raises(PermissionConfigurationError, match="permissions.yaml"):
        PermissionManager(tmp_path / "workspace", user_configuration_path=user_path)


def test_invalid_user_policy_reports_without_interaction_in_automatic_mode(tmp_path):
    """Automatic loading logs and ignores invalid user approvals in a headless process."""
    user_path = tmp_path / "user" / "permissions.yaml"
    user_path.parent.mkdir()
    user_path.write_text("version: invalid\n", encoding="utf-8")

    manager = PermissionManager(
        tmp_path / "workspace",
        user_configuration_path=user_path,
        load_policy=PermissionLoadPolicy.AUTO,
    )

    assert not manager.user_rules


def test_unavailable_user_approval_fails_closed(tmp_path, monkeypatch):
    """An unavailable user policy path cannot authorize a spoofed user selection."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    monkeypatch.setattr(manager, "request_permission", Mock(return_value=ApprovalChoice.USER))

    result = manager.authorize((file_operation(Action.FILESYSTEM_CREATE, tmp_path / "new.txt"),))

    assert result.decision is Decision.DENY
    assert result.approval_choice is ApprovalChoice.DENY
    assert manager.user_configuration_path is None


def test_default_policy_allows_scoped_reads_and_fails_closed_for_approval(tmp_path):
    """The supervised default permits workspace inspection and denies headless mutations."""
    manager = PermissionManager(tmp_path)

    read = manager.authorize((file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt"),))
    write = manager.authorize((file_operation(Action.FILESYSTEM_CREATE, tmp_path / "new.txt"),))

    assert read.decision is Decision.ALLOW
    assert read.policy.decision is Decision.ALLOW
    assert write.policy.decision is Decision.ASK
    assert write.decision is Decision.DENY
    assert write.source == "headless"
    assert not (tmp_path / ".loop" / "permissions-audit.jsonl").exists()


def test_permission_audit_uses_central_sqlite_storage(tmp_path):
    """Permission audit storage records workspace-correlated SQLite rows."""
    audit_path = tmp_path / "state" / "audit.db"
    manager = PermissionManager(tmp_path, audit_path=audit_path, workspace_id="workspace")

    for tool_id in ("first", "second", "third", "fourth"):
        manager.authorize((file_operation(Action.FILESYSTEM_READ, tmp_path / f"{tool_id}.txt"),))

    with closing(sqlite3.connect(audit_path)) as connection:
        rows = connection.execute(
            "SELECT workspace_id, event_name FROM permission_audit_records"
        ).fetchall()
    assert rows == [("workspace", "permission.decided")] * 4
    assert audit_path.stat().st_mode & 0o777 == 0o600


def test_permission_audit_rotation_failure_does_not_change_authorization(tmp_path, monkeypatch):
    """A failed audit archive rotation cannot turn an allowed operation into a denial."""
    monkeypatch.setattr("loop.constants.DEFAULT_PERMISSIONS_AUDIT_BYTES", 1)
    manager = PermissionManager(tmp_path)
    operation_set = (file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt"),)
    manager.authorize(operation_set)
    monkeypatch.setattr(Path, "replace", Mock(side_effect=OSError("unavailable")))

    result = manager.authorize(operation_set)

    assert result.decision is Decision.ALLOW


def test_sqlite_permission_audit_failure_does_not_change_authorization(tmp_path, monkeypatch):
    """A failed centralized audit insert cannot change an authorization decision."""
    manager = PermissionManager(
        tmp_path,
        audit_path=tmp_path / "audit.db",
        workspace_id="workspace",
    )
    monkeypatch.setattr(
        "loop.permissions.audit.SQLitePermissionAudit.append",
        Mock(side_effect=sqlite3.OperationalError("busy")),
    )

    result = manager.authorize((file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt"),))

    assert result.decision is Decision.ALLOW


def test_authorization_approves_one_complete_operation_set_and_records_policy(tmp_path):
    """One prompt and recorder event preserve both policy and effective outcomes."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    recorder = Mock()
    manager = PermissionManager(tmp_path, interaction=interaction, recorder=recorder)
    operations = (
        file_operation(Action.FILESYSTEM_CREATE, tmp_path / "a.txt"),
        file_operation(Action.FILESYSTEM_DELETE, tmp_path / "b.txt"),
    )
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)

    try:
        result = manager.authorize(operations)
        assert telemetry.close(1)
    finally:
        set_telemetry(None)

    assert result.policy.decision is Decision.ASK
    assert result.decision is Decision.ALLOW
    assert result.prompted is True
    assert result.prompt is not None
    assert "filesystem.create" in result.prompt
    assert "filesystem.delete" in result.prompt
    interaction.prompt.assert_called_once()
    recorder.record_authorization.assert_called_once_with(result)
    assert adapter.records[0].event_name == "permission.decided"
    assert adapter.records[0].attributes["decision"] == "allow"
    assert adapter.records[1].event_name == "permission.decision"
    assert thaw(adapter.records[1].payload) == result.model_dump(mode="json")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_host_decision_records_omit_command_and_prompt(tmp_path, asynchronous):
    """Routine audit, trace, and session records retain decisions without host secrets."""
    audit_path = tmp_path / "audit.sqlite"
    recorder = Mock()
    manager = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(limits=PolicyLimits(allow_host_processes=True)),
        audit_path=audit_path,
        workspace_id="workspace",
        recorder=recorder,
    )
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW, scope=PolicyScope.SESSION)
    secret = "password=synthetic-secret"
    planned = operation(
        Action.PROCESS_EXECUTE,
        tool="request_host_command",
        target=ProcessTarget(
            argv=("/bin/sh", "-c", f"printf {secret}"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.HOST,
        ),
    )
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    try:
        result = (
            asyncio.run(manager.authorize_async((planned,)))
            if asynchronous
            else manager.authorize((planned,))
        )
        assert telemetry.close(1)
    finally:
        set_telemetry(None)
    assert secret in repr(result)
    recorded = recorder.record_authorization.call_args.args[0]
    assert recorded.decision is Decision.ALLOW
    assert recorded.operations == ()
    assert recorded.prompt is None
    assert secret not in repr(recorded)
    trace = next(record for record in adapter.records if record.event_name == "permission.decision")
    assert secret not in repr(thaw(trace.payload))
    with closing(sqlite3.connect(audit_path)) as connection:
        row = connection.execute(
            "SELECT payload_json FROM permission_audit_records WHERE event_name = ?",
            ("permission.decided",),
        ).fetchone()
    assert row is not None
    payload = json.loads(row[0])
    assert payload["decision"] == "allow"
    assert payload["operation_count"] == 1
    assert secret not in row[0]
    manager.close()


def test_authorization_rejection_and_recorder_override_are_atomic(tmp_path):
    """A rejected batch is recorded once by an invocation-scoped recorder."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.DENY
    configured = Mock()
    override = Mock()
    manager = PermissionManager(tmp_path, interaction=interaction, recorder=configured)

    result = manager.authorize(
        (file_operation(Action.FILESYSTEM_REPLACE, tmp_path / "a.txt"),),
        recorder=override,
    )

    assert result.decision is Decision.DENY
    assert result.source == "user"
    override.record_authorization.assert_called_once_with(result)
    configured.record_authorization.assert_not_called()
    assert manager.recorder is configured


def test_authorize_async_awaits_native_approval_and_records_result(tmp_path):
    """Async authorization awaits native input before recording an approval."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.ONCE)
    recorder = Mock()
    manager = PermissionManager(tmp_path, interaction=interaction)
    operation_set = (
        operation(
            Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com", origin="https://example.com"),
        ),
    )

    result = asyncio.run(manager.authorize_async(operation_set, recorder=recorder))

    assert result.decision is Decision.ALLOW
    assert result.approval_choice is ApprovalChoice.ONCE
    assert result.prompted
    interaction.prompt_async.assert_awaited_once()
    assert interaction.prompt_async.await_args.kwargs["choices"] == {
        ApprovalChoice.DENY: "Deny",
        ApprovalChoice.ONCE: "Allow once",
        ApprovalChoice.SESSION: "Allow for this session",
        ApprovalChoice.WORKSPACE: "Allow in this workspace",
    }
    recorder.record_authorization.assert_called_once_with(result)


def test_async_remembered_approval_writes_off_loop_and_publishes_after_save(tmp_path):
    """Async approval persistence and legacy recording settle outside the event loop."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    policy_path = tmp_path / ".loop" / "permissions.yaml"
    worker_started = Event()
    release_worker = Event()
    blocked_event_loop = Event()
    recorder_threads = []
    worker_calls = []
    manager = PermissionManager(
        tmp_path,
        configuration_path=policy_path,
        interaction=interaction,
    )
    original_persist = manager._persist_configuration

    def slow_persist(path, configuration):
        """Hold policy-file I/O so the test can observe event-loop progress."""
        worker_started.set()
        if not release_worker.wait(timeout=5):
            blocked_event_loop.set()
        original_persist(path, configuration)

    manager._persist_configuration = slow_persist
    recorder = Mock()
    recorder.record_authorization.side_effect = lambda _result: recorder_threads.append(get_ident())

    async def dispatch(function, *args):
        """Run each synchronous authorization effect through a test worker."""
        name = (
            "record_authorization"
            if function is recorder.record_authorization
            else function.__name__
        )
        return await asyncio.to_thread(lambda: (worker_calls.append(name), function(*args))[1])

    manager.set_async_blocking_runner(dispatch)
    operation_set = (
        operation(
            Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com", origin="https://example.com"),
        ),
    )

    async def authorize():
        """Inspect the owner while policy persistence is deliberately stalled."""
        owner_thread = get_ident()
        task = asyncio.create_task(manager.authorize_async(operation_set, recorder=recorder))
        ready = await asyncio.to_thread(worker_started.wait, 2)
        if not ready and task.done():
            result = await task
            pytest.fail(f"Authorization finished before persistence: {result.reason}")
        assert ready
        unpublished_during_write = not manager.configuration.rules
        event_loop_responsive = not blocked_event_loop.is_set()
        release_worker.set()
        result = await task
        return owner_thread, unpublished_during_write, event_loop_responsive, result

    try:
        owner_thread, unpublished, responsive, result = asyncio.run(authorize())
    finally:
        release_worker.set()

    assert result.decision is Decision.ALLOW
    assert unpublished
    assert responsive
    assert not blocked_event_loop.is_set()
    assert [name for name in worker_calls if name != "_namespace_recovery_decision"] == [
        "_persist_configuration_candidate",
        "record_authorization",
    ]
    assert recorder_threads and recorder_threads[0] != owner_thread
    assert manager.configuration.rules
    assert policy_path.exists()


@pytest.mark.parametrize("scope", [ApprovalChoice.WORKSPACE, ApprovalChoice.USER])
def test_async_remembered_approval_publishes_with_durable_write(tmp_path, scope):
    """A policy observer sees the saved approval and revision before its caller resumes."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=scope)
    saved = asyncio.Event()
    release = asyncio.Event()

    async def hold_completed_worker(function, *args):
        """Pause the caller immediately after the policy worker completes its write."""
        result = await asyncio.to_thread(function, *args)
        if function.__name__ == "_persist_configuration_candidate":
            saved.set()
            await release.wait()
        return result

    path = tmp_path / "permissions.yaml"
    manager = PermissionManager(
        tmp_path,
        configuration_path=path if scope is ApprovalChoice.WORKSPACE else None,
        user_configuration_path=path if scope is ApprovalChoice.USER else None,
        interaction=interaction,
        async_blocking_runner=hold_completed_worker,
    )
    target = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com", origin="https://example.com"),
    )
    initial_policy, initial_revision = manager.evaluate_with_revision((target,))

    async def inspect_saved_policy():
        """Inspect the durable and active policy while the worker result remains pending."""
        task = asyncio.create_task(manager.authorize_async((target,)))
        try:
            await asyncio.wait_for(saved.wait(), timeout=3)
            observed, revision = manager.evaluate_with_revision((target,))
            model = (
                PermissionConfiguration
                if scope is ApprovalChoice.WORKSPACE
                else UserPermissionConfiguration
            )
            persisted = model.model_validate(load_yaml(path.read_text()))
            active = manager.configuration if scope is ApprovalChoice.WORKSPACE else None
        finally:
            release.set()
        result = await task
        return observed, revision, persisted, active, result

    observed, revision, persisted, active, result = asyncio.run(inspect_saved_policy())

    assert initial_policy.decision is Decision.ASK
    assert observed.decision is Decision.ALLOW
    assert revision > initial_revision
    assert len(persisted.rules) == 1
    if active is not None:
        assert active == persisted
    assert result.decision is Decision.ALLOW


def test_async_remembered_approval_retries_after_concurrent_policy_update(tmp_path):
    """A concurrent policy change is merged instead of overwritten by a stale approval."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    policy_path = tmp_path / ".loop" / "permissions.yaml"
    worker_started = Event()
    release_worker = Event()
    manager = PermissionManager(
        tmp_path,
        configuration_path=policy_path,
        interaction=interaction,
        async_blocking_runner=lambda function, *args: asyncio.to_thread(function, *args),
    )
    original_candidate = manager._persist_configuration_candidate
    pause_candidate = True
    candidate_writes = []

    def delayed_candidate(path, configuration, revision):
        """Let one sync policy update commit before the async candidate is written."""
        nonlocal pause_candidate
        if pause_candidate:
            pause_candidate = False
            worker_started.set()
            release_worker.wait(timeout=2)
        written = original_candidate(path, configuration, revision)
        candidate_writes.append(written)
        return written

    manager._persist_configuration_candidate = delayed_candidate
    operation_set = (
        operation(
            Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com", origin="https://example.com"),
        ),
    )

    async def authorize():
        """Interleave a serialized sync policy update with remembered approval."""
        task = asyncio.create_task(manager.authorize_async(operation_set))
        assert await asyncio.to_thread(worker_started.wait, 2)
        await asyncio.to_thread(
            manager.set_default,
            Action.FILESYSTEM_READ,
            Decision.DENY,
            scope=PolicyScope.WORKSPACE,
        )
        release_worker.set()
        return await task

    try:
        result = asyncio.run(authorize())
    finally:
        release_worker.set()

    assert result.decision is Decision.ALLOW
    assert candidate_writes == [False, True]
    assert manager.configuration.defaults[Action.FILESYSTEM_READ] is Decision.DENY
    assert manager.configuration.rules
    persisted = PermissionConfiguration.model_validate(
        load_yaml(policy_path.read_text(encoding="utf-8"))
    )
    assert persisted.defaults[Action.FILESYSTEM_READ] is Decision.DENY
    assert persisted.rules


def test_async_remembered_approvals_serialize_write_and_publication(tmp_path, policy_waits):
    """A second approval waits while the first caller settles its completed transaction."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    first_written = asyncio.Event()
    release_first = asyncio.Event()
    writes = []

    async def run_blocking(function, *args):
        """Hold the first completed transaction before its caller receives the result."""
        result = function(*args)
        if function.__name__ == "_persist_configuration_candidate":
            writes.append(result)
            if len(writes) == 1:
                first_written.set()
                await release_first.wait()
        return result

    policy_path = tmp_path / "permissions.yaml"
    manager = PermissionManager(
        tmp_path,
        configuration_path=policy_path,
        interaction=interaction,
        async_blocking_runner=run_blocking,
    )
    first = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://one.example", origin="https://one.example"),
    )
    second = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://two.example", origin="https://two.example"),
    )

    async def authorize():
        """Interleave two independently approved writes after the first transaction settles."""
        first_task = asyncio.create_task(manager.authorize_async((first,)))
        await first_written.wait()
        second_task = asyncio.create_task(manager.authorize_async((second,)))
        waiting = await policy_waits.get()
        assert writes == [True]
        release_first.set()
        first_result = await first_task
        waiting.set_result(None)
        return first_result, await second_task

    first_result, second_result = asyncio.run(authorize())

    assert first_result.decision is second_result.decision is Decision.ALLOW
    assert writes == [True, True]
    persisted = PermissionConfiguration.model_validate(
        load_yaml(policy_path.read_text(encoding="utf-8"))
    )
    assert {rule.target.origin for rule in persisted.rules} == {
        "https://one.example",
        "https://two.example",
    }
    assert persisted.rules == manager.configuration.rules


def test_async_remembered_approval_preserves_policy_changes_after_write(tmp_path):
    """A policy update after publication preserves the already saved approval."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    manager = PermissionManager(
        tmp_path, configuration_path=tmp_path / "permissions.yaml", interaction=interaction
    )
    original = manager._persist_configuration_candidate
    written = []

    def write_then_update(path, configuration, revision):
        """Publish a restrictive update after the first candidate reaches disk."""
        result = original(path, configuration, revision)
        written.append(result)
        if len(written) == 1:
            manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
        return result

    manager._persist_configuration_candidate = write_then_update
    target = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://one.example", origin="https://one.example"),
    )
    result = asyncio.run(manager.authorize_async((target,)))

    assert result.decision is Decision.ALLOW
    assert written == [True]
    persisted = PermissionConfiguration.model_validate(
        load_yaml(manager.configuration_path.read_text(encoding="utf-8"))
    )
    assert persisted.defaults[Action.FILESYSTEM_READ] is Decision.DENY
    assert len(persisted.rules) == 1
    assert persisted.rules[0].target.origin == "https://one.example"
    assert persisted == manager.configuration


def test_async_remembered_approval_respects_deny_after_publication(tmp_path):
    """A deny published after the approval worker finishes still controls the outcome."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    path = tmp_path / "permissions.yaml"
    manager = PermissionManager(tmp_path, configuration_path=path, interaction=interaction)
    original = manager._persist_configuration_candidate
    target = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com", origin="https://example.com"),
    )

    def publish_then_deny(policy_path, configuration, revision):
        """Install an explicit deny after the remembered approval is published."""
        written = original(policy_path, configuration, revision)
        manager.add_rule(
            PermissionRule(
                decision=Decision.DENY,
                tool="demo",
                action=Action.NETWORK_REQUEST,
                target=target.target,
            ),
            scope=PolicyScope.WORKSPACE,
        )
        return written

    manager._persist_configuration_candidate = publish_then_deny
    result = asyncio.run(manager.authorize_async((target,)))

    assert result.decision is Decision.DENY
    assert result.source == "policy"
    persisted = PermissionConfiguration.model_validate(load_yaml(path.read_text()))
    assert {rule.decision for rule in persisted.rules} == {Decision.ALLOW, Decision.DENY}
    assert persisted == manager.configuration


def test_async_remembered_approval_rechecks_a_new_deny_policy(tmp_path):
    """A deny published during durable approval wins over the remembered allow rule."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    worker_started = Event()
    release_worker = Event()
    manager = PermissionManager(
        tmp_path,
        configuration_path=tmp_path / "permissions.yaml",
        interaction=interaction,
        async_blocking_runner=lambda function, *args: asyncio.to_thread(function, *args),
    )
    original_candidate = manager._persist_configuration_candidate
    pause_candidate = True

    def delayed_candidate(path, configuration, revision):
        """Commit an explicit deny while an approval snapshot is being written."""
        nonlocal pause_candidate
        if pause_candidate:
            pause_candidate = False
            worker_started.set()
            release_worker.wait(timeout=2)
        return original_candidate(path, configuration, revision)

    manager._persist_configuration_candidate = delayed_candidate
    operation_set = (
        operation(
            Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com", origin="https://example.com"),
        ),
    )

    async def authorize():
        """Interleave a deny update with the remembered workspace approval."""
        task = asyncio.create_task(manager.authorize_async(operation_set))
        assert await asyncio.to_thread(worker_started.wait, 2)
        await asyncio.to_thread(
            manager.add_rule,
            PermissionRule(
                decision=Decision.DENY,
                tool="demo",
                action=Action.NETWORK_REQUEST,
                target=NetworkTarget(url="https://example.com", origin="https://example.com"),
            ),
            scope=PolicyScope.WORKSPACE,
        )
        release_worker.set()
        return await task

    try:
        result = asyncio.run(authorize())
    finally:
        release_worker.set()

    assert result.decision is Decision.DENY
    assert result.source == "policy"
    assert manager.configuration.rules


@pytest.mark.parametrize("scope", [ApprovalChoice.WORKSPACE, ApprovalChoice.USER])
def test_async_cancelled_stale_approval_does_not_retry_or_record_allow(tmp_path, scope):
    """Cancellation after a stale candidate denies and records without saving a new allow."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=scope)
    worker_started = Event()
    release_worker = Event()
    recorder = Mock()
    path = tmp_path / "permissions.yaml"
    audit_path = tmp_path / "audit.sqlite"
    manager = PermissionManager(
        tmp_path,
        configuration_path=path if scope is ApprovalChoice.WORKSPACE else None,
        user_configuration_path=path if scope is ApprovalChoice.USER else None,
        audit_path=audit_path,
        workspace_id="workspace",
        interaction=interaction,
        async_blocking_runner=lambda function, *args: asyncio.to_thread(function, *args),
    )
    original = manager._persist_configuration_candidate
    writes = []

    def delayed_candidate(policy_path, configuration, revision):
        """Pause the first candidate before its revision is checked."""
        if not writes:
            worker_started.set()
            assert release_worker.wait(timeout=3)
        written = original(policy_path, configuration, revision)
        writes.append(written)
        return written

    manager._persist_configuration_candidate = delayed_candidate
    target = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com", origin="https://example.com"),
    )

    async def cancel_stale_candidate():
        """Make the candidate stale, cancel its owner, and let the owned worker settle."""
        pending = asyncio.create_task(manager.authorize_async((target,), recorder=recorder))
        try:
            assert await asyncio.to_thread(worker_started.wait, 2)
            await asyncio.to_thread(
                manager.set_default,
                Action.FILESYSTEM_READ,
                Decision.DENY,
                scope=PolicyScope.SESSION,
            )
            pending.cancel()
            await deliver_callbacks()
        finally:
            release_worker.set()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(cancel_stale_candidate())

    assert writes == [False]
    assert not path.exists()
    assert manager.evaluate((target,)).decision is Decision.ASK
    recorder.record_authorization.assert_called_once()
    recorded = recorder.record_authorization.call_args.args[0]
    assert recorded.decision is Decision.DENY
    assert recorded.source == "cancelled"
    assert recorded.installed_rule_ids == ()
    with closing(sqlite3.connect(audit_path)) as connection:
        rows = connection.execute(
            "SELECT payload_json FROM permission_audit_records WHERE event_name = ?",
            ("permission.decided",),
        ).fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0][0]) == recorded.model_dump(mode="json")


def test_async_authorization_yields_while_a_policy_update_holds_its_lock(tmp_path, policy_waits):
    """Async policy evaluation waits cooperatively for a concurrent sync update."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.ONCE)
    manager = PermissionManager(tmp_path, interaction=interaction)
    lock_acquired = Event()
    release_lock = Event()
    operation_set = (
        operation(
            Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com", origin="https://example.com"),
        ),
    )

    def hold_policy_lock():
        """Hold the policy lock on another thread until async evaluation yields."""
        with manager.policy_update():
            lock_acquired.set()
            release_lock.wait(timeout=2)

    async def scenario():
        """Check that lock contention suspends authorization instead of blocking this loop."""
        holder = asyncio.create_task(asyncio.to_thread(hold_policy_lock))
        assert await asyncio.to_thread(lock_acquired.wait, 2)
        authorization = asyncio.create_task(manager.authorize_async(operation_set))
        waiting = await policy_waits.get()
        yielded = not authorization.done()
        release_lock.set()
        await holder
        waiting.set_result(None)
        result = await authorization
        return yielded, result

    try:
        yielded, result = asyncio.run(scenario())
    finally:
        release_lock.set()

    assert yielded
    assert result.decision is Decision.ALLOW


@pytest.mark.parametrize("scope", [ApprovalChoice.SESSION, ApprovalChoice.USER])
def test_async_remembered_approval_preserves_each_scope(scope, tmp_path):
    """Async approval writes only the selected user or session rule layer."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=scope)
    user_path = tmp_path / "user-permissions.yaml"
    manager = PermissionManager(
        tmp_path,
        user_configuration_path=user_path if scope is ApprovalChoice.USER else None,
        interaction=interaction,
    )
    operation_set = (
        operation(
            Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com", origin="https://example.com"),
        ),
    )

    result = asyncio.run(manager.authorize_async(operation_set))

    assert result.decision is Decision.ALLOW
    if scope is ApprovalChoice.SESSION:
        assert manager.session_rules
        assert not user_path.exists()
    else:
        assert manager.user_rules
        assert user_path.exists()
        assert UserPermissionConfiguration.model_validate(
            load_yaml(user_path.read_text(encoding="utf-8"))
        ).rules


def test_async_remembered_approval_denies_when_policy_write_fails(tmp_path):
    """A failed async approval write never publishes a remembered allow rule."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    manager = PermissionManager(
        tmp_path,
        configuration_path=tmp_path / "permissions.yaml",
        interaction=interaction,
    )

    def fail_persist(_path, _configuration, _revision):
        """Simulate an atomic policy-file write failure."""
        raise OSError("disk unavailable")

    manager._persist_configuration_candidate = fail_persist
    operation_set = (
        operation(
            Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com", origin="https://example.com"),
        ),
    )

    result = asyncio.run(manager.authorize_async(operation_set))

    assert result.decision is Decision.DENY
    assert result.source == "persistence"
    assert not manager.configuration.rules


def test_async_recorder_cancellation_is_propagated_after_recording_settles(tmp_path):
    """Cancellation from a native async recorder stops authorization after its call settles."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.ONCE)
    recorder = Mock()
    recorder.record_authorization_async = AsyncMock(side_effect=asyncio.CancelledError)
    manager = PermissionManager(tmp_path, interaction=interaction)
    operation_set = (
        operation(
            Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com", origin="https://example.com"),
        ),
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(manager.authorize_async(operation_set, recorder=recorder))

    recorder.record_authorization_async.assert_awaited_once()
    recorder.record_authorization.assert_not_called()


@pytest.mark.parametrize("scope", [ApprovalChoice.WORKSPACE, ApprovalChoice.USER])
@pytest.mark.parametrize("worker_fails", [False, True])
def test_async_remembered_approval_settles_worker_before_cancellation(
    tmp_path, worker_fails, scope
):
    """Remembered policy writes settle after cancellation, including simultaneous write errors."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=scope)
    worker_started = asyncio.Event()
    release_worker = asyncio.Event()

    async def run_blocking_async(function, *args):
        """Keep persistence owned until the test cancels its caller."""
        if getattr(function, "__name__", "") == "_persist_configuration_candidate":
            worker_started.set()
            await release_worker.wait()
            if worker_fails:
                raise OSError("policy storage unavailable")
        return await asyncio.to_thread(function, *args)

    path = tmp_path / "permissions.yaml"
    recorder = Mock()
    manager = PermissionManager(
        tmp_path,
        configuration_path=path if scope is ApprovalChoice.WORKSPACE else None,
        user_configuration_path=path if scope is ApprovalChoice.USER else None,
        interaction=interaction,
        async_blocking_runner=run_blocking_async,
    )
    operation_set = (
        operation(
            Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com", origin="https://example.com"),
        ),
    )

    async def cancel_authorization():
        """Cancel repeatedly while persistence remains admitted and pending."""
        pending = asyncio.create_task(manager.authorize_async(operation_set, recorder=recorder))
        await worker_started.wait()
        pending.cancel()
        await deliver_callbacks()
        pending.cancel()
        release_worker.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await pending
        if worker_fails:
            assert any("OSError" in note for note in captured.value.__notes__)

    asyncio.run(cancel_authorization())
    if worker_fails:
        assert not path.exists()
        recorder.record_authorization.assert_not_called()
    else:
        model = (
            PermissionConfiguration
            if scope is ApprovalChoice.WORKSPACE
            else UserPermissionConfiguration
        )
        persisted = model.model_validate(load_yaml(path.read_text()))
        assert len(persisted.rules) == 1
        assert manager.evaluate(operation_set).decision is Decision.ALLOW
        recorder.record_authorization.assert_called_once()
        recorded = recorder.record_authorization.call_args.args[0]
        assert recorded.decision is Decision.ALLOW
        assert recorded.installed_rule_ids == (persisted.rules[0].id,)


@pytest.mark.parametrize("scope", [ApprovalChoice.WORKSPACE, ApprovalChoice.USER])
def test_async_remembered_approval_cancels_after_publication_while_policy_is_locked(
    tmp_path, policy_waits, scope
):
    """A cancelled final check records denial while preserving the already saved rule."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=scope)
    lock_held = Event()
    release_lock = Event()
    holder_threads = []
    path = tmp_path / "permissions.yaml"
    audit_path = tmp_path / "audit.sqlite"
    recorder = Mock()
    manager = PermissionManager(
        tmp_path,
        configuration_path=path if scope is ApprovalChoice.WORKSPACE else None,
        user_configuration_path=path if scope is ApprovalChoice.USER else None,
        audit_path=audit_path,
        workspace_id="workspace",
        interaction=interaction,
        async_blocking_runner=lambda function, *args: asyncio.to_thread(function, *args),
    )
    original_candidate = manager._persist_configuration_candidate

    def persist_then_hold_lock(path, configuration, revision):
        """Acquire the policy lock in a peer thread after the worker publishes its snapshot."""
        written = original_candidate(path, configuration, revision)

        def hold_lock():
            """Hold the later policy check until the async caller has been cancelled."""
            manager._policy_lock.acquire()
            lock_held.set()
            release_lock.wait(timeout=3)
            manager._policy_lock.release()

        holder = Thread(target=hold_lock)
        holder_threads.append(holder)
        holder.start()
        assert lock_held.wait(timeout=2)
        return written

    manager._persist_configuration_candidate = persist_then_hold_lock
    operation_set = (
        operation(
            Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com", origin="https://example.com"),
        ),
    )

    async def cancel_authorization():
        """Cancel a later policy check while the already published rule remains durable."""
        pending = asyncio.create_task(manager.authorize_async(operation_set, recorder=recorder))
        assert await asyncio.to_thread(lock_held.wait, 2)
        await policy_waits.get()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release_lock.set()
        for holder in holder_threads:
            await asyncio.to_thread(holder.join, 5)
            assert not holder.is_alive()

    try:
        asyncio.run(cancel_authorization())
    finally:
        release_lock.set()
        for holder in holder_threads:
            holder.join(timeout=2)

    assert all(not holder.is_alive() for holder in holder_threads)
    model = (
        PermissionConfiguration
        if scope is ApprovalChoice.WORKSPACE
        else UserPermissionConfiguration
    )
    persisted = model.model_validate(load_yaml(path.read_text(encoding="utf-8")))
    assert len(persisted.rules) == 1
    assert manager.evaluate(operation_set).decision is Decision.ALLOW
    recorder.record_authorization.assert_called_once()
    recorded = recorder.record_authorization.call_args.args[0]
    assert recorded.decision is Decision.DENY
    assert recorded.source == "cancelled"
    assert recorded.installed_rule_ids == (persisted.rules[0].id,)
    with closing(sqlite3.connect(audit_path)) as connection:
        rows = connection.execute(
            "SELECT payload_json FROM permission_audit_records WHERE event_name = ?",
            ("permission.decided",),
        ).fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0][0]) == recorded.model_dump(mode="json")


def test_async_remembered_approval_settles_cancelled_final_file_check(tmp_path):
    """Cancellation settles a final filesystem check before recording the saved approval."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    path = tmp_path / "permissions.yaml"
    audit_path = tmp_path / "audit.sqlite"
    recorder = Mock()
    published, worker_started, release_worker = Event(), Event(), Event()
    manager = PermissionManager(
        tmp_path,
        configuration_path=path,
        audit_path=audit_path,
        workspace_id="workspace",
        interaction=interaction,
        async_blocking_runner=lambda function, *args: asyncio.to_thread(function, *args),
    )
    candidate = manager._persist_configuration_candidate
    boundary = manager._boundary_decision

    def publish_candidate(policy_path, configuration, revision):
        """Mark completion of the real saved and published policy transaction."""
        written = candidate(policy_path, configuration, revision)
        published.set()
        return written

    def pause_file_boundary(value):
        """Hold the post-save filesystem boundary until cancellation settles."""
        if published.is_set() and isinstance(value.target, FileTarget):
            worker_started.set()
            assert release_worker.wait(timeout=3)
        return boundary(value)

    manager._persist_configuration_candidate = publish_candidate
    manager._boundary_decision = pause_file_boundary
    source = tmp_path / "source.txt"
    source.write_text("owned")
    operations = (
        operation(
            Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com", origin="https://example.com"),
        ),
        file_operation(Action.FILESYSTEM_READ, source),
    )

    async def cancel_final_check():
        """Cancel twice while a real file-boundary worker is still owned."""
        pending = asyncio.create_task(manager.authorize_async(operations, recorder=recorder))
        try:
            assert await asyncio.to_thread(worker_started.wait, 2)
            pending.cancel()
            await deliver_callbacks()
            pending.cancel()
            await deliver_callbacks()
            assert not pending.done()
        finally:
            release_worker.set()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(cancel_final_check())
    persisted = PermissionConfiguration.model_validate(load_yaml(path.read_text()))
    recorder.record_authorization.assert_called_once()
    recorded = recorder.record_authorization.call_args.args[0]
    assert recorded.decision is Decision.DENY and recorded.source == "cancelled"
    assert set(recorded.installed_rule_ids) == {rule.id for rule in persisted.rules}
    with closing(sqlite3.connect(audit_path)) as connection:
        rows = connection.execute(
            "SELECT payload_json FROM permission_audit_records WHERE event_name = ?",
            ("permission.decided",),
        ).fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0][0]) == recorded.model_dump(mode="json")


def test_async_cancelled_final_check_settles_audit_after_repeated_cancellation(tmp_path):
    """A saved rule and its cancelled decision remain recorded when audit is interrupted."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    path = tmp_path / "permissions.yaml"
    audit_path = tmp_path / "audit.sqlite"
    published, audit_started, release_audit = Event(), Event(), Event()
    recorder = Mock()
    manager = PermissionManager(
        tmp_path,
        configuration_path=path,
        audit_path=audit_path,
        workspace_id="workspace",
        interaction=interaction,
        async_blocking_runner=lambda function, *args: asyncio.to_thread(function, *args),
    )
    candidate, evaluate, audit = (
        manager._persist_configuration_candidate,
        manager.evaluate_async,
        manager._audit,
    )

    def publish_candidate(policy_path, configuration, revision):
        """Mark the saved approval before its final policy check."""
        written = candidate(policy_path, configuration, revision)
        published.set()
        return written

    async def cancel_final_evaluation(values):
        """Interrupt only the post-publication policy check."""
        if published.is_set():
            raise asyncio.CancelledError()
        return await evaluate(values)

    def hold_audit(result):
        """Delay the real audit write while repeated cancellations arrive."""
        audit_started.set()
        assert release_audit.wait(timeout=3)
        audit(result)

    manager._persist_configuration_candidate = publish_candidate
    manager.evaluate_async = cancel_final_evaluation
    manager._audit = hold_audit
    operations = (
        operation(
            Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com", origin="https://example.com"),
        ),
    )

    async def cancel_audit():
        """Interrupt twice while the first cancelled decision is being audited."""
        pending = asyncio.create_task(manager.authorize_async(operations, recorder=recorder))
        try:
            assert await asyncio.to_thread(audit_started.wait, 2)
            pending.cancel()
            await deliver_callbacks()
            pending.cancel()
            await deliver_callbacks()
            assert not pending.done()
        finally:
            release_audit.set()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(cancel_audit())
    persisted = PermissionConfiguration.model_validate(load_yaml(path.read_text()))
    recorder.record_authorization.assert_called_once()
    recorded = recorder.record_authorization.call_args.args[0]
    assert recorded.decision is Decision.DENY and recorded.source == "cancelled"
    assert recorded.installed_rule_ids == (persisted.rules[0].id,)
    with closing(sqlite3.connect(audit_path)) as connection:
        rows = connection.execute(
            "SELECT payload_json FROM permission_audit_records WHERE event_name = ?",
            ("permission.decided",),
        ).fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0][0]) == recorded.model_dump(mode="json")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
def test_async_recorder_failure_preserves_prior_cancellation(tmp_path, asynchronous, cancelled):
    """A recorder failure preserves a prior cancellation without exposing its message."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    path = tmp_path / "permissions.yaml"
    published = Event()
    manager = PermissionManager(tmp_path, configuration_path=path, interaction=interaction)
    candidate, evaluate = manager._persist_configuration_candidate, manager.evaluate_async
    failure = OSError("private recorder detail")
    recorder = Mock()
    if asynchronous:
        recorder.record_authorization_async = AsyncMock(side_effect=failure)
    else:
        recorder.record_authorization.side_effect = failure

    def publish_candidate(policy_path, configuration, revision):
        """Mark the saved rule before the final policy check."""
        written = candidate(policy_path, configuration, revision)
        published.set()
        return written

    async def evaluate_or_cancel(values):
        """Cancel the final check only for the retained-cancellation control."""
        if cancelled and published.is_set():
            raise asyncio.CancelledError("first")
        return await evaluate(values)

    manager._persist_configuration_candidate = publish_candidate
    manager.evaluate_async = evaluate_or_cancel
    target = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com", origin="https://example.com"),
    )
    expected = asyncio.CancelledError if cancelled else OSError
    with pytest.raises(expected) as captured:
        asyncio.run(manager.authorize_async((target,), recorder=recorder))

    assert len(manager.configuration.rules) == 1
    if cancelled:
        assert captured.value.args == ("first",)
        assert captured.value.__cause__ is failure
        assert captured.value.__notes__ == ["Permission recorder also stopped with OSError."]
    else:
        assert captured.value is failure


def test_async_audit_failure_retains_note_on_first_cancellation(tmp_path):
    """An audit failure during repeated cancellation annotates the original cancellation."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    published, audit_started, release_audit = Event(), Event(), Event()
    manager = PermissionManager(
        tmp_path,
        configuration_path=tmp_path / "permissions.yaml",
        audit_path=tmp_path / "audit.sqlite",
        workspace_id="workspace",
        interaction=interaction,
        async_blocking_runner=lambda function, *args: asyncio.to_thread(function, *args),
    )
    candidate, evaluate = manager._persist_configuration_candidate, manager.evaluate_async
    recorder = Mock()

    def publish_candidate(path, configuration, revision):
        """Mark the completed policy transaction."""
        written = candidate(path, configuration, revision)
        published.set()
        return written

    async def cancel_final_check(values):
        """Generate the first cancellation at the final policy check."""
        if published.is_set():
            raise asyncio.CancelledError("first")
        return await evaluate(values)

    def fail_audit(_result):
        """Fail the owned audit only after another cancellation has arrived."""
        audit_started.set()
        assert release_audit.wait(timeout=3)
        raise OSError("private audit detail")

    manager._persist_configuration_candidate = publish_candidate
    manager.evaluate_async = cancel_final_check
    manager._audit = fail_audit
    target = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com", origin="https://example.com"),
    )

    async def cancel_during_audit():
        """Cancel the still-running audit and retain the original cancellation."""
        pending = asyncio.create_task(manager.authorize_async((target,), recorder=recorder))
        try:
            assert await asyncio.to_thread(audit_started.wait, 2)
            pending.cancel("second")
            await deliver_callbacks()
            assert not pending.done()
        finally:
            release_audit.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await pending
        return captured.value

    cancellation = asyncio.run(cancel_during_audit())
    assert cancellation.args == ("first",)
    assert cancellation.__notes__ == ["Permission audit also stopped with OSError."]
    recorder.record_authorization.assert_called_once()
    assert recorder.record_authorization.call_args.args[0].source == "cancelled"


@pytest.mark.parametrize("recorder_fails", [False, True])
def test_async_recorder_settles_during_second_cancellation(tmp_path, recorder_fails):
    """Owned recorder work settles while the first cancellation remains authoritative."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    published, recorder_started, release_recorder = Event(), Event(), Event()
    manager = PermissionManager(
        tmp_path,
        configuration_path=tmp_path / "permissions.yaml",
        interaction=interaction,
        async_blocking_runner=lambda function, *args: asyncio.to_thread(function, *args),
    )
    candidate, evaluate = manager._persist_configuration_candidate, manager.evaluate_async
    recorder = Mock()
    failure = OSError("private recorder detail")

    def publish_candidate(path, configuration, revision):
        """Mark the completed policy transaction."""
        written = candidate(path, configuration, revision)
        published.set()
        return written

    async def cancel_final_check(values):
        """Generate the first cancellation at the final policy check."""
        if published.is_set():
            raise asyncio.CancelledError("first")
        return await evaluate(values)

    def settle_recorder(_result):
        """Settle the owned recorder after another cancellation has arrived."""
        recorder_started.set()
        assert release_recorder.wait(timeout=3)
        if recorder_fails:
            raise failure

    manager._persist_configuration_candidate = publish_candidate
    manager.evaluate_async = cancel_final_check
    recorder.record_authorization.side_effect = settle_recorder
    target = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com", origin="https://example.com"),
    )

    async def cancel_during_recording():
        """Interrupt the still-running recorder and retain the original cancellation."""
        pending = asyncio.create_task(manager.authorize_async((target,), recorder=recorder))
        try:
            assert await asyncio.to_thread(recorder_started.wait, 2)
            pending.cancel("second")
            await deliver_callbacks()
            assert not pending.done()
        finally:
            release_recorder.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await pending
        return captured.value

    cancellation = asyncio.run(cancel_during_recording())
    assert cancellation.args == ("first",)
    if recorder_fails:
        assert cancellation.__cause__ is failure
        assert cancellation.__notes__ == [
            "The blocking permission operation also stopped with OSError."
        ]
    else:
        assert cancellation.__cause__ is None
        assert cancellation.__notes__ == [
            "An additional cancellation was observed during permission recorder."
        ]
    recorder.record_authorization.assert_called_once()


@pytest.mark.parametrize("same", [False, True])
def test_async_recorder_cancellation_retains_only_sanitized_notes(tmp_path, same):
    """A later native recorder cannot add forged notes or causes to the original signal."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    published = Event()
    manager = PermissionManager(
        tmp_path, configuration_path=tmp_path / "permissions.yaml", interaction=interaction
    )
    candidate, evaluate = manager._persist_configuration_candidate, manager.evaluate_async
    first = asyncio.CancelledError("first")
    later = first if same else asyncio.CancelledError("second")
    later.add_note("The blocking permission operation also stopped with OSError.")
    if not same:
        later.add_note("private recorder detail")
        later.add_note("Permission audit also stopped with SecretToken123.")
        later.__notes__.append(123)
        later.__cause__ = OSError("forged recorder cause")
    recorder = Mock()
    recorder.record_authorization_async = AsyncMock(side_effect=later)

    def publish_candidate(path, configuration, revision):
        """Mark the completed policy transaction."""
        written = candidate(path, configuration, revision)
        published.set()
        return written

    async def cancel_final_check(values):
        """Generate the first cancellation at the final policy check."""
        if published.is_set():
            raise first
        return await evaluate(values)

    manager._persist_configuration_candidate = publish_candidate
    manager.evaluate_async = cancel_final_check
    target = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com", origin="https://example.com"),
    )
    with pytest.raises(asyncio.CancelledError) as captured:
        asyncio.run(manager.authorize_async((target,), recorder=recorder))

    assert captured.value.args == ("first",)
    if same:
        assert captured.value.__notes__ == [
            "The blocking permission operation also stopped with OSError.",
            "An additional cancellation was observed during permission recorder.",
        ]
    else:
        assert captured.value.__notes__ == [
            "An additional cancellation was observed during permission recorder."
        ]
        assert captured.value.__cause__ is None


def test_first_native_recorder_cancellation_remains_unchanged(tmp_path):
    """A recorder cancellation with no earlier cancellation keeps its original details."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.ONCE)
    manager = PermissionManager(tmp_path, interaction=interaction)
    later = asyncio.CancelledError("recorder")
    later.add_note("private recorder detail")
    cause = OSError("recorder cause")
    later.__cause__ = cause
    recorder = Mock()
    recorder.record_authorization_async = AsyncMock(side_effect=later)
    target = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com", origin="https://example.com"),
    )

    with pytest.raises(asyncio.CancelledError) as captured:
        asyncio.run(manager.authorize_async((target,), recorder=recorder))

    assert captured.value.args == ("recorder",)
    assert captured.value.__notes__ == ["private recorder detail"]
    assert captured.value.__cause__ is cause


@pytest.mark.parametrize("concurrent", [False, True])
def test_sync_recorder_cancellation_cannot_forge_owned_diagnostics(tmp_path, concurrent):
    """A sync recorder's own cancellation cannot inject notes or causes into an earlier one."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    published, recorder_started, release_recorder = Event(), Event(), Event()
    manager = PermissionManager(
        tmp_path,
        configuration_path=tmp_path / "permissions.yaml",
        interaction=interaction,
        async_blocking_runner=lambda function, *args: asyncio.to_thread(function, *args),
    )
    candidate, evaluate = manager._persist_configuration_candidate, manager.evaluate_async
    later = asyncio.CancelledError("recorder")
    later.add_note("The blocking permission operation also stopped with SecretToken123.")
    later.__cause__ = OSError("forged recorder cause")
    recorder = Mock()

    def publish_candidate(path, configuration, revision):
        """Mark the saved rule before the final policy check."""
        written = candidate(path, configuration, revision)
        published.set()
        return written

    async def cancel_final_check(values):
        """Generate the first cancellation at the final policy check."""
        if published.is_set():
            raise asyncio.CancelledError("first")
        return await evaluate(values)

    def cancel_recorder(_result):
        """Return a forged recorder cancellation after an optional second stop."""
        recorder_started.set()
        if concurrent:
            assert release_recorder.wait(timeout=3)
        raise later

    manager._persist_configuration_candidate = publish_candidate
    manager.evaluate_async = cancel_final_check
    recorder.record_authorization.side_effect = cancel_recorder
    target = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com", origin="https://example.com"),
    )

    async def exercise():
        """Optionally cancel the owner while the sync recorder remains pending."""
        pending = asyncio.create_task(manager.authorize_async((target,), recorder=recorder))
        try:
            assert await asyncio.to_thread(recorder_started.wait, 2)
            if concurrent:
                pending.cancel("second")
                await deliver_callbacks()
                assert not pending.done()
        finally:
            release_recorder.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await pending
        return captured.value

    cancellation = asyncio.run(exercise())
    assert cancellation.args == ("first",)
    assert cancellation.__notes__ == [
        "An additional cancellation was observed during permission recorder."
    ]
    assert cancellation.__cause__ is None


def test_audit_worker_cancellation_cannot_forge_owned_diagnostics(tmp_path):
    """An audit worker's own cancellation cannot inject notes or causes into an earlier one."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    published = Event()
    manager = PermissionManager(
        tmp_path,
        configuration_path=tmp_path / "permissions.yaml",
        audit_path=tmp_path / "audit.sqlite",
        workspace_id="workspace",
        interaction=interaction,
        async_blocking_runner=lambda function, *args: asyncio.to_thread(function, *args),
    )
    candidate, evaluate = manager._persist_configuration_candidate, manager.evaluate_async
    later = asyncio.CancelledError("audit")
    later.add_note("Permission audit also stopped with SecretToken123.")
    later.__cause__ = OSError("forged audit cause")

    def publish_candidate(path, configuration, revision):
        """Mark the saved rule before the final policy check."""
        written = candidate(path, configuration, revision)
        published.set()
        return written

    async def cancel_final_check(values):
        """Generate the first cancellation at the final policy check."""
        if published.is_set():
            raise asyncio.CancelledError("first")
        return await evaluate(values)

    def cancel_audit(_result):
        """Raise a forged cancellation directly from the audit worker."""
        raise later

    manager._persist_configuration_candidate = publish_candidate
    manager.evaluate_async = cancel_final_check
    manager._audit = cancel_audit
    target = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com", origin="https://example.com"),
    )
    with pytest.raises(asyncio.CancelledError) as captured:
        asyncio.run(manager.authorize_async((target,)))

    assert captured.value.args == ("first",)
    assert captured.value.__notes__ == [
        "An additional cancellation was observed during permission audit."
    ]
    assert captured.value.__cause__ is None


@pytest.mark.parametrize("malformed", [False, True])
def test_owned_cancellation_note_transfer_is_bounded_and_type_checked(tmp_path, malformed):
    """Only a bounded set of well-formed owned notes reaches the earlier cancellation."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.WORKSPACE)
    published = Event()
    manager = PermissionManager(
        tmp_path, configuration_path=tmp_path / "permissions.yaml", interaction=interaction
    )
    candidate, evaluate = manager._persist_configuration_candidate, manager.evaluate_async
    later = asyncio.CancelledError("audit")
    note = "Permission audit also stopped with OSError."
    later.__notes__ = "not-a-list" if malformed else [123, "private", *([note] * 10)]

    async def interrupted_audit(_result, *, _owned_failures):
        """Supply a locally marked audit cancellation with controlled note data."""
        _owned_failures.append((later, OSError("audit failed")))
        raise later

    manager._audit_async = interrupted_audit

    def publish_candidate(path, configuration, revision):
        """Mark the completed policy transaction."""
        written = candidate(path, configuration, revision)
        published.set()
        return written

    async def cancel_final_check(values):
        """Generate the first cancellation at the final policy check."""
        if published.is_set():
            raise asyncio.CancelledError("first")
        return await evaluate(values)

    manager._persist_configuration_candidate = publish_candidate
    manager.evaluate_async = cancel_final_check
    target = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com", origin="https://example.com"),
    )
    with pytest.raises(asyncio.CancelledError) as captured:
        asyncio.run(manager.authorize_async((target,)))

    assert captured.value.args == ("first",)
    if malformed:
        assert not hasattr(captured.value, "__notes__")
    else:
        assert captured.value.__notes__ == [note] * 6


def test_async_remembered_approval_rejects_unavailable_policy_scopes(tmp_path):
    """Only configured workspace or user files can receive async remembered approval."""
    manager = PermissionManager(tmp_path)
    operation_set = (file_operation(Action.FILESYSTEM_READ, tmp_path / "source.txt"),)

    with pytest.raises(ValueError, match="Unsupported remembered approval scope"):
        asyncio.run(manager._remember_approval_async(operation_set, scope=ApprovalChoice.ONCE))
    with pytest.raises(ValueError, match="policy path is unavailable"):
        asyncio.run(manager._remember_approval_async(operation_set, scope=ApprovalChoice.USER))
    with pytest.raises(ValueError, match="user permission policy path is not configured"):
        manager._replace_user_configuration(UserPermissionConfiguration())


def test_async_authorization_uses_bounded_audit_and_native_recorder_paths(tmp_path):
    """Async permission audits and session observations use their owning async boundaries."""
    audit_path = tmp_path / "audit.db"
    worker_calls = []

    async def run_blocking_async(function, *args):
        """Dispatch the audit write through the supplied worker."""
        worker_calls.append(function.__name__)
        return await asyncio.to_thread(function, *args)

    recorder = Mock()
    recorder.record_authorization_async = AsyncMock()
    manager = PermissionManager(
        tmp_path,
        audit_path=audit_path,
        workspace_id="workspace",
        async_blocking_runner=run_blocking_async,
    )
    operation_set = (file_operation(Action.FILESYSTEM_READ, tmp_path / "source.txt"),)

    result = asyncio.run(manager.authorize_async(operation_set, recorder=recorder))

    assert result.decision is Decision.ALLOW
    assert [name for name in worker_calls if name != "_namespace_recovery_decision"] == [
        "_boundary_decision",
        "_audit",
    ]
    recorder.record_authorization_async.assert_awaited_once_with(result)
    recorder.record_authorization.assert_not_called()
    with closing(sqlite3.connect(audit_path)) as connection:
        row = connection.execute(
            "SELECT workspace_id, event_name FROM permission_audit_records"
        ).fetchone()
    assert row == ("workspace", "permission.decided")


def test_async_authorization_keeps_best_effort_audit_failures_out_of_policy_results(tmp_path):
    """A bounded audit admission failure does not change an authorization outcome."""

    async def reject_audit(function, *args):
        """Reject optional audit work after policy evaluation."""
        if function.__name__ == "_audit":
            raise RuntimeError(f"Could not schedule {function.__name__}.")
        return await asyncio.to_thread(function, *args)

    manager = PermissionManager(
        tmp_path,
        audit_path=tmp_path / "audit.db",
        workspace_id="workspace",
        async_blocking_runner=reject_audit,
    )
    operation_set = (file_operation(Action.FILESYSTEM_READ, tmp_path / "source.txt"),)

    result = asyncio.run(manager.authorize_async(operation_set))

    assert result.decision is Decision.ALLOW


def test_async_fresh_file_review_propagates_cancellation_after_path_check(tmp_path):
    """Async authorization settles fresh path review before propagating cancellation."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def run_blocking_async(function, *args):
        """Hold fresh-review path checks until the authorization task is cancelled."""
        if function.__name__ != "_requires_fresh_file_review":
            return await asyncio.to_thread(function, *args)
        started.set()
        await release.wait()
        return function(*args)

    manager = PermissionManager(tmp_path, async_blocking_runner=run_blocking_async)
    operations = (file_operation(Action.FILESYSTEM_CREATE, tmp_path / "AGENTS.md"),)

    async def cancel_authorization():
        """Cancel during fresh-review resolution and wait for its worker to settle."""
        pending = asyncio.create_task(manager.authorize_async(operations))
        await started.wait()
        pending.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(cancel_authorization())


@pytest.mark.parametrize("audit_fails", [False, True])
def test_async_authorization_settles_audit_before_propagating_cancellation(
    tmp_path,
    audit_fails,
):
    """Permission audit work settles before cancellation returns control to the caller."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def run_blocking_async(function, *args):
        """Keep audit work pending until the owner has been cancelled."""
        if function.__name__ != "_audit":
            return await asyncio.to_thread(function, *args)
        started.set()
        await release.wait()
        if audit_fails:
            raise OSError("audit storage unavailable")
        return await asyncio.to_thread(function, *args)

    manager = PermissionManager(
        tmp_path,
        audit_path=tmp_path / "audit.db",
        workspace_id="workspace",
        async_blocking_runner=run_blocking_async,
    )
    operation_set = (file_operation(Action.FILESYSTEM_READ, tmp_path / "source.txt"),)

    async def cancel_authorization():
        """Cancel authorization while its optional audit write remains owned."""
        pending = asyncio.create_task(manager.authorize_async(operation_set))
        await started.wait()
        pending.cancel()
        await deliver_callbacks()
        pending.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await pending
        if audit_fails:
            assert any("OSError" in note for note in captured.value.__notes__)

    asyncio.run(cancel_authorization())


def test_async_authorization_supports_the_standalone_audit_worker(tmp_path):
    """Standalone permission managers still move SQLite audits off the event loop."""
    audit_path = tmp_path / "audit.db"
    manager = PermissionManager(
        tmp_path,
        audit_path=audit_path,
        workspace_id="workspace",
    )
    operation_set = (file_operation(Action.FILESYSTEM_READ, tmp_path / "source.txt"),)

    result = asyncio.run(manager.authorize_async(operation_set))

    assert result.decision is Decision.ALLOW
    with closing(sqlite3.connect(audit_path)) as connection:
        row = connection.execute(
            "SELECT workspace_id, event_name FROM permission_audit_records"
        ).fetchone()
    assert row == ("workspace", "permission.decided")


def test_authorize_async_keeps_fresh_file_changes_one_time_only(tmp_path):
    """A fresh file review cannot install a reusable grant even from a stale response."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.SESSION)
    manager = PermissionManager(tmp_path, interaction=interaction)

    result = asyncio.run(
        manager.authorize_async((file_operation(Action.FILESYSTEM_REPLACE, tmp_path / "a.txt"),))
    )

    assert result.decision is Decision.DENY
    assert result.approval_choice is ApprovalChoice.DENY
    assert result.reason == "Rejected by the user."
    assert interaction.prompt_async.await_args.kwargs["choices"] == {
        ApprovalChoice.DENY: "Deny",
        ApprovalChoice.ONCE: "Allow once",
    }
    assert manager.session_rules == ()


def test_authorize_async_requires_native_async_interaction(tmp_path):
    """Async authorization refuses synchronous approval input on the event loop."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(tmp_path, interaction=interaction)
    operation_set = (
        operation(
            Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com", origin="https://example.com"),
        ),
    )

    with pytest.raises(ProblemException) as captured:
        asyncio.run(manager.authorize_async(operation_set))

    assert captured.value.problem.code == "interaction.async_required"
    interaction.prompt.assert_not_called()


def test_authorize_async_denies_headless_and_propagates_cancellation(tmp_path):
    """Headless approvals deny and cancelled native prompts remain cancellable."""
    operation_set = (
        operation(
            Action.NETWORK_REQUEST,
            target=NetworkTarget(url="https://example.com", origin="https://example.com"),
        ),
    )
    headless = PermissionManager(tmp_path).authorize_async(operation_set)
    result = asyncio.run(headless)
    assert result.decision is Decision.DENY
    assert result.source == "headless"

    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    interaction.prompt_async = AsyncMock(side_effect=asyncio.CancelledError)
    manager = PermissionManager(tmp_path, interaction=interaction)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(manager.authorize_async(operation_set))


def test_session_approval_remembers_exact_batch_targets_and_deduplicates(tmp_path):
    """Session approval installs exact grants once and reuses them without another prompt."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.SESSION
    manager = PermissionManager(tmp_path, interaction=interaction)
    first = file_operation(Action.FILESYSTEM_CREATE, tmp_path / "literal[*].txt")
    second = file_operation(Action.FILESYSTEM_CREATE, tmp_path / "other.txt")

    approved = manager.authorize((first, second))
    repeated = manager.authorize((first, second))
    different = manager.evaluate(
        (file_operation(Action.FILESYSTEM_CREATE, tmp_path / "literal-x.txt"),)
    )

    assert approved.approval_choice is ApprovalChoice.SESSION
    assert len(approved.installed_rule_ids) == 2
    assert repeated.prompted is False
    assert repeated.decision is Decision.ALLOW
    assert different.decision is Decision.ASK
    assert len(manager.session_rules) == 2
    interaction.prompt.assert_called_once()


def test_session_approval_deduplicates_a_rule_added_during_prompt(tmp_path):
    """A user decision does not add a duplicate exact session rule published concurrently."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    target = FileTarget(path=str(tmp_path / "file.txt"))
    requested = operation(Action.FILESYSTEM_CREATE, target=target)

    def approve_after_rule(*_args, **_kwargs):
        """Install the same rule before the in-flight prompt returns."""
        manager.add_rule(
            PermissionRule(
                decision=Decision.ALLOW,
                tool="demo",
                tool_exact=True,
                action=Action.FILESYSTEM_CREATE,
                target=target,
            ),
            scope=PolicyScope.SESSION,
        )
        return ApprovalChoice.SESSION

    interaction.prompt.side_effect = approve_after_rule
    result = manager.authorize((requested,))

    assert result.decision is Decision.ALLOW
    assert result.installed_rule_ids == ()
    assert len(manager.session_rules) == 1


def test_session_approval_matches_tool_identifiers_literally(tmp_path):
    """Generated grants do not interpret glob characters in tool identifiers."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.SESSION
    manager = PermissionManager(tmp_path, interaction=interaction)
    target = FileTarget(path=str(tmp_path / "file.txt"))
    approved = operation(Action.FILESYSTEM_CREATE, tool="writer[*]", target=target)

    manager.authorize((approved,))

    assert manager.evaluate((approved,)).decision is Decision.ALLOW
    assert (
        manager.evaluate(
            (operation(Action.FILESYSTEM_CREATE, tool="writerx", target=target),)
        ).decision
        is Decision.ASK
    )


def test_workspace_approval_persists_exact_rules_and_audit_metadata(tmp_path):
    """Workspace approval atomically persists reusable typed grants and their audit identity."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.WORKSPACE
    manager = PermissionManager(tmp_path, interaction=interaction)
    target = file_operation(Action.FILESYSTEM_CREATE, tmp_path / "module.py")

    approved = manager.authorize((target, target))
    reloaded = PermissionManager(tmp_path)

    assert approved.decision is Decision.ALLOW
    assert approved.approval_choice is ApprovalChoice.WORKSPACE
    assert len(approved.installed_rule_ids) == 1
    assert reloaded.persistent_rules[0].tool_exact is True
    assert "tool_match=exact" in reloaded.describe("workspace")
    assert reloaded.authorize((target,)).decision is Decision.ALLOW


@pytest.mark.parametrize(
    ("action", "relative"),
    [
        (Action.FILESYSTEM_REPLACE, "ordinary.txt"),
        (Action.FILESYSTEM_DELETE, "ordinary.txt"),
        (Action.FILESYSTEM_CREATE, "AGENTS.md"),
        (Action.FILESYSTEM_CREATE, ".agents/skills/demo/SKILL.md"),
        (Action.FILESYSTEM_CREATE, "nested/.agents/skills/demo/asset.txt"),
        (Action.FILESYSTEM_CREATE, "nested/.gitignore"),
        (Action.FILESYSTEM_CREATE, "nested/.agentignore"),
    ],
)
def test_user_data_and_instruction_changes_require_fresh_one_time_approval(
    tmp_path, action, relative
):
    """Broad allow rules and earlier consent never authorize another protected mutation."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.side_effect = [ApprovalChoice.ONCE, ApprovalChoice.SESSION]
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.set_default(action, Decision.ALLOW, scope=PolicyScope.SESSION)
    change = file_operation(action, tmp_path / relative)

    first = manager.authorize((change,))
    second = manager.authorize((change,))

    assert first.decision is Decision.ALLOW
    assert first.approval_choice is ApprovalChoice.ONCE
    assert second.decision is Decision.DENY
    assert not first.installed_rule_ids
    assert not manager.session_rules
    assert interaction.prompt.call_count == 2
    assert all(
        set(call.kwargs["choices"]) == {ApprovalChoice.DENY, ApprovalChoice.ONCE}
        for call in interaction.prompt.call_args_list
    )


def test_new_ordinary_file_keeps_explicit_allow_policy(tmp_path):
    """A normal creation under an allow rule needs no fresh data-loss decision."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.set_default(Action.FILESYSTEM_CREATE, Decision.ALLOW, scope=PolicyScope.SESSION)

    result = manager.authorize((file_operation(Action.FILESYSTEM_CREATE, tmp_path / "new.txt"),))

    assert result.decision is Decision.ALLOW
    interaction.prompt.assert_not_called()


def test_configured_instruction_change_requires_fresh_decision(tmp_path):
    """An active custom instruction name cannot inherit a workspace-wide create grant."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.DENY
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.set_default(Action.FILESYSTEM_CREATE, Decision.ALLOW, scope=PolicyScope.SESSION)

    result = manager.authorize(
        (file_operation(Action.FILESYSTEM_CREATE, tmp_path / "nested" / "POLICY.md"),),
        protected_instruction_names=("POLICY.md",),
    )

    assert result.decision is Decision.DENY
    assert "boundary:fresh_file_review" in result.policy.sources
    assert set(interaction.prompt.call_args.kwargs["choices"]) == {
        ApprovalChoice.DENY,
        ApprovalChoice.ONCE,
    }


def test_manager_rejects_unsafe_instruction_names(tmp_path):
    """Permission authority binds control paths and rejects unsafe instruction names."""
    manager = PermissionManager(tmp_path)

    paths = manager.protected_sandbox_paths(("POLICY.md",))
    assert paths.instruction_names == ("AGENTS.md", "SKILL.md", "POLICY.md")
    assert paths.files == (".gitignore", ".agentignore")

    for invalid in ("../POLICY.md", "a/b.md", "a\\b.md", "", ".", "a\x00b", 7):
        with pytest.raises(ValueError, match="plain filenames"):
            manager.protected_instruction_names((invalid,))


def test_manager_without_workspace_does_not_classify_a_new_file_as_instructions(tmp_path):
    """Only a bound workspace can identify protected instruction creation paths."""
    manager = PermissionManager()

    result = manager.authorize((file_operation(Action.FILESYSTEM_CREATE, tmp_path / "AGENTS.md"),))

    assert "boundary:fresh_file_review" not in result.policy.sources


def test_policy_mutations_write_timestamped_local_and_structured_audit_records(tmp_path):
    """Permission configuration changes remain auditable with and without telemetry storage."""
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    manager = PermissionManager(tmp_path)

    try:
        manager.set_default(
            Action.FILESYSTEM_DELETE,
            Decision.DENY,
            scope=PolicyScope.SESSION,
        )
        assert telemetry.close(1)
    finally:
        set_telemetry(None)

    assert adapter.records[0].event_name == "permission.default_set"
    assert adapter.records[0].attributes["scope"] == "session"


def test_process_grants_distinguish_argument_boundaries_working_directory_and_sandbox(tmp_path):
    """Remembered process approval compares argv, cwd, and execution boundary structurally."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.SESSION
    manager = PermissionManager(tmp_path, interaction=interaction)
    approved = operation(
        Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("tool", "a b"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    ambiguous = operation(
        Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("tool", "a", "b"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )

    manager.authorize((approved,))

    assert manager.evaluate((approved,)).decision is Decision.ALLOW
    assert manager.evaluate((ambiguous,)).decision is Decision.ASK
    assert (
        manager.evaluate(
            (
                approved.model_copy(
                    update={
                        "target": approved.target.model_copy(update={"cwd": str(tmp_path / "sub")})
                    }
                ),
            )
        ).decision
        is Decision.ASK
    )


def test_network_grants_ignore_dns_addresses_but_retain_request_semantics(tmp_path):
    """Network grants survive DNS changes without widening method or body semantics."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.SESSION
    manager = PermissionManager(tmp_path, interaction=interaction)
    get = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(
            url="https://my-host.local/data",
            origin="https://my-host.local",
            addresses=("93.184.216.34",),
        ),
    )

    manager.authorize((get,))

    changed_dns = get.model_copy(
        update={"target": get.target.model_copy(update={"addresses": ("93.184.216.35",)})}
    )
    post = get.model_copy(
        update={"target": get.target.model_copy(update={"method": "POST", "sends_body": True})}
    )
    assert manager.evaluate((changed_dns,)).decision is Decision.ALLOW
    assert manager.evaluate((post,)).decision is Decision.ASK


def test_failed_workspace_persistence_denies_without_activating_grants(tmp_path, monkeypatch):
    """A failed durable write fails closed and leaves the active policy unchanged."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.WORKSPACE
    manager = PermissionManager(tmp_path, interaction=interaction)
    failing_os = SimpleNamespace(
        **{
            **vars(os),
            "O_NOFOLLOW": 1,
            "O_NONBLOCK": 0,
            "open": Mock(side_effect=OSError("disk full")),
        }
    )
    monkeypatch.setattr(manager_module, "os", failing_os)

    result = manager.authorize((file_operation(Action.FILESYSTEM_CREATE, tmp_path / "new.txt"),))

    assert result.decision is Decision.DENY
    assert result.source == "persistence"
    assert not result.installed_rule_ids
    assert not manager.persistent_rules


def test_workspace_choice_is_rejected_when_no_workspace_is_available():
    """A malformed interaction cannot persist workspace trust without a policy path."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.WORKSPACE
    manager = PermissionManager(interaction=interaction)

    result = manager.authorize(
        (operation(Action.SESSION_MUTATE, target=SessionTarget(identifier="setting")),)
    )

    assert result.decision is Decision.DENY
    assert "unavailable" in result.reason
    interaction.prompt.assert_called_once()


def test_no_authority_plan_is_allowed_without_prompting():
    """Pure tool plans require neither policy rules nor interactive approval."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(interaction=interaction)

    result = manager.authorize(())

    assert result.decision is Decision.ALLOW
    assert result.policy.sources == ("no_authority",)
    interaction.confirm.assert_not_called()


def test_interaction_property_and_non_persisted_default_changes():
    """In-memory policy controls can be replaced without requiring a policy path."""
    manager = PermissionManager()
    interaction = Mock(spec=Interaction)

    manager.interaction = interaction
    manager.set_default(Action.FILESYSTEM_READ, Decision.ALLOW, scope=PolicyScope.SESSION)

    assert manager.interaction is interaction
    assert manager.effective_configuration.defaults[Action.FILESYSTEM_READ] is Decision.ALLOW


def test_permission_rules_reject_ambiguous_exact_and_glob_matchers(tmp_path):
    """One rule cannot combine a broad resource glob with an exact typed target."""
    with pytest.raises(ValueError, match="cannot combine"):
        PermissionRule(
            decision=Decision.ALLOW,
            resource="*.txt",
            target=FileTarget(path=str(tmp_path / "file.txt")),
        )


def test_rule_composition_uses_forbid_then_approval_then_permit(tmp_path):
    """Forbid and approval duties monotonically constrain matching permits."""
    rules = [
        PermissionRule(id="permit", decision=Decision.ALLOW, tool="read_*"),
        PermissionRule(
            id="approval",
            decision=Decision.ASK,
            action=Action.FILESYSTEM_READ,
        ),
        PermissionRule(
            id="forbid",
            decision=Decision.DENY,
            resource="*/secret.*",
        ),
    ]
    manager = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(rules=rules),
    )

    secret = manager.evaluate(
        (
            operation(
                Action.FILESYSTEM_READ,
                tool="read_file",
                target=FileTarget(path=str(tmp_path / "secret.env")),
            ),
        )
    )
    ordinary = manager.evaluate(
        (
            operation(
                Action.FILESYSTEM_READ,
                tool="read_file",
                target=FileTarget(path=str(tmp_path / "ordinary.txt")),
            ),
        )
    )

    assert secret.decision is Decision.DENY
    assert secret.sources == ("rule:workspace:forbid",)
    assert ordinary.decision is Decision.ASK
    assert ordinary.sources == ("rule:workspace:approval",)


def test_any_denied_operation_denies_a_batch_without_prompting(tmp_path):
    """A hard denial prevents prompts for other approval-requiring operations."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)

    result = manager.authorize(
        (
            file_operation(Action.FILESYSTEM_CREATE, tmp_path / "new.txt"),
            file_operation(Action.FILESYSTEM_READ, tmp_path.parent / "outside.txt"),
        )
    )

    assert result.policy.decision is Decision.DENY
    assert result.decision is Decision.DENY
    interaction.confirm.assert_not_called()


@pytest.mark.parametrize(
    ("path", "action", "source"),
    [
        ("outside", Action.FILESYSTEM_READ, "limit:workspace:readable_roots"),
        (".loop/policy", Action.FILESYSTEM_READ, "boundary:protected_path"),
        (".git/config", Action.FILESYSTEM_READ, "boundary:protected_path"),
        (".gitignore", Action.FILESYSTEM_REPLACE, "boundary:protected_path"),
        (".agentignore", Action.FILESYSTEM_DELETE, "boundary:protected_path"),
    ],
)
def test_filesystem_boundaries_cannot_be_overridden(tmp_path, path, action, source):
    """Filesystem roots and control paths remain forbidden despite permit rules."""
    target = tmp_path.parent / "outside" if path == "outside" else tmp_path / path
    manager = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(rules=[PermissionRule(decision=Decision.ALLOW)]),
    )

    result = manager.evaluate((file_operation(action, target),))

    assert result.decision is Decision.DENY
    assert result.sources == (source,)


@pytest.mark.parametrize(
    ("action", "target", "limits", "source"),
    [
        (
            Action.FILESYSTEM_READ,
            lambda root: FileTarget(path=str(root.parent / "outside")),
            PolicyLimits(),
            "limit:workspace:readable_roots",
        ),
        (
            Action.NETWORK_REQUEST,
            lambda root: NetworkTarget(
                url="https://outside.test/data", origin="https://outside.test"
            ),
            PolicyLimits(network_origins=("https://allowed.test",)),
            "limit:workspace:network_origins",
        ),
        (
            Action.PROCESS_EXECUTE,
            lambda root: ProcessTarget(
                argv=("git", "status"), cwd=str(root), boundary=ProcessBoundary.HOST
            ),
            PolicyLimits(),
            "limit:workspace:allow_host_processes",
        ),
    ],
)
def test_check_boundaries_returns_first_hard_denial_without_policy_matching(
    tmp_path, action, target, limits, source
):
    """Boundary preflight rejects unsafe effects before rules or defaults are evaluated."""
    manager = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(
            defaults={action: Decision.ALLOW},
            limits=limits,
            rules=[PermissionRule(decision=Decision.ALLOW)],
        ),
    )
    operation_target = target(tmp_path)
    operation_value = operation(action, target=operation_target)

    result = manager.check_boundaries((operation_value,))

    assert result is not None
    assert result.decision is Decision.DENY
    assert result.sources == (source,)


def test_check_boundaries_allows_safe_plans_and_stops_at_first_denial(tmp_path):
    """Boundary preflight ignores policy-only effects and returns the first unsafe effect."""
    manager = PermissionManager(tmp_path)
    safe = file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt")
    unsafe = file_operation(Action.FILESYSTEM_READ, tmp_path.parent / "outside.txt")

    assert manager.check_boundaries((safe,)) is None
    result = manager.check_boundaries((safe, unsafe, safe))

    assert result is not None
    assert result.sources == ("limit:workspace:readable_roots",)


def test_evaluate_async_resolves_file_boundaries_in_a_worker(tmp_path, monkeypatch):
    """Async policy evaluation keeps filesystem canonicalization off its event-loop thread."""
    manager = PermissionManager(tmp_path)
    owner_thread = get_ident()
    resolution_threads = path_resolution_threads(monkeypatch)

    result = asyncio.run(
        manager.evaluate_async((file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt"),))
    )

    assert result.decision is Decision.ALLOW
    assert resolution_threads
    assert owner_thread not in resolution_threads


def test_evaluate_async_accepts_an_empty_operation_set(tmp_path):
    """An empty plan requires no authority and remains an allow decision."""
    result = asyncio.run(PermissionManager(tmp_path).evaluate_async(()))

    assert result.decision is Decision.ALLOW
    assert result.sources == ("no_authority",)


def test_check_boundaries_async_resolves_files_off_loop_and_stops_at_first_denial(
    tmp_path, monkeypatch
):
    """Async boundary preflight keeps canonicalization off-loop and returns the first denial."""
    manager = PermissionManager(tmp_path)
    safe = file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt")
    unsafe = file_operation(Action.FILESYSTEM_READ, tmp_path.parent / "outside.txt")
    owner_thread = get_ident()
    resolution_threads = path_resolution_threads(monkeypatch)

    assert asyncio.run(manager.check_boundaries_async(())) is None
    result = asyncio.run(manager.check_boundaries_async((safe, unsafe, safe)))

    assert result is not None
    assert result.sources == ("limit:workspace:readable_roots",)
    assert resolution_threads
    assert owner_thread not in resolution_threads


def test_evaluate_async_retries_same_owner_policy_mutation(tmp_path):
    """A policy change across a worker await invalidates the first evaluation pass."""
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def run_blocking(function, *args):
        """Pause the first computed boundary before its owner receives the result."""
        result = function(*args)
        if function.__name__ == "_boundary_decision":
            calls.append(function.__name__)
        if calls == ["_boundary_decision"] and function.__name__ == "_boundary_decision":
            entered.set()
            await release.wait()
        return result

    manager = PermissionManager(tmp_path, async_blocking_runner=run_blocking)
    target = file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt")

    async def evaluate():
        """Change the rule while evaluation is suspended on the same owner thread."""
        pending = asyncio.create_task(manager.evaluate_async((target,)))
        await entered.wait()
        manager.set_default(Action.FILESYSTEM_READ, Decision.DENY, scope=PolicyScope.SESSION)
        release.set()
        return await pending

    result = asyncio.run(evaluate())

    assert result.decision is Decision.DENY
    assert result.sources == ("default:session:filesystem.read",)
    assert calls == ["_boundary_decision", "_boundary_decision"]


def test_async_policy_retry_observes_a_rebound_scratch_root(tmp_path):
    """A scratch rebinding invalidates an already-computed file-boundary result."""
    first = tmp_path.parent / "first-scratch"
    second = tmp_path.parent / "second-scratch"
    first.mkdir()
    second.mkdir()
    manager = PermissionManager(tmp_path)
    manager.bind_temporary_directory(PathHolder(first))
    calls = []

    async def rebind_after_first_pass(function, *args):
        """Replace the referenced scratch root after evaluating the first boundary."""
        result = function(*args)
        if function.__name__ == "_boundary_decision":
            calls.append(function.__name__)
            if len(calls) == 1:
                manager.bind_temporary_directory(PathHolder(second))
        return result

    manager.set_async_blocking_runner(rebind_after_first_pass)
    result = asyncio.run(
        manager.evaluate_async((file_operation(Action.FILESYSTEM_READ, first / "f"),))
    )

    assert result.decision is Decision.DENY
    assert result.sources == ("limit:workspace:readable_roots",)
    assert calls == ["_boundary_decision", "_boundary_decision"]


@pytest.mark.parametrize("check_boundaries", [False, True])
def test_async_policy_churn_fails_closed_after_bounded_retries(tmp_path, check_boundaries):
    """Repeated same-owner policy changes cannot cause an unbounded authorization retry."""
    calls = []
    manager = PermissionManager(tmp_path)

    async def mutate_each_pass(function, *args):
        """Publish a change after each real boundary result is computed."""
        result = function(*args)
        if function.__name__ == "_boundary_decision":
            calls.append(function.__name__)
            manager.set_default(Action.FILESYSTEM_READ, Decision.DENY, scope=PolicyScope.SESSION)
        return result

    manager.set_async_blocking_runner(mutate_each_pass)
    target = file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt")
    result = asyncio.run(
        manager.check_boundaries_async((target,))
        if check_boundaries
        else manager.evaluate_async((target,))
    )

    assert result is not None
    assert result.decision is Decision.DENY
    assert result.sources == ("policy:stale",)
    assert calls == ["_boundary_decision"] * 3


def test_async_boundary_rechecks_a_denial_after_same_owner_limit_change(tmp_path):
    """An early boundary denial is retried if its policy changed before it was returned."""
    manager = PermissionManager(tmp_path)
    outside = tmp_path.parent / "shared"
    outside.mkdir()
    calls = []

    async def allow_after_denial(function, *args):
        """Change the root limit after the first denial was computed."""
        result = function(*args)
        calls.append(function.__name__)
        if len(calls) == 1:
            manager.update_limit_values(
                "readable_roots", str(outside), add=True, scope=PolicyScope.SESSION
            )
        return result

    manager.set_async_blocking_runner(allow_after_denial)
    target = file_operation(Action.FILESYSTEM_READ, outside / "file.txt")
    assert asyncio.run(manager.check_boundaries_async((target,))) is None
    assert calls == ["_boundary_decision", "_boundary_decision"]


def test_evaluate_async_settles_file_boundary_before_cancellation(tmp_path):
    """Async policy evaluation settles filesystem boundary work before cancellation returns."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def run_blocking_async(function, *args):
        """Hold filesystem boundary work until the owner has been cancelled."""
        if function.__name__ == "_boundary_decision":
            started.set()
            await release.wait()
        return await asyncio.to_thread(function, *args)

    manager = PermissionManager(tmp_path, async_blocking_runner=run_blocking_async)
    operations = (file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt"),)

    async def cancel_evaluation():
        """Cancel evaluation while its filesystem worker remains owned."""
        pending = asyncio.create_task(manager.evaluate_async(operations))
        await started.wait()
        pending.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(cancel_evaluation())


def test_authorize_async_checks_fresh_file_paths_in_a_worker(tmp_path, monkeypatch):
    """Async fresh-review detection keeps instruction-path resolution off the owner loop."""
    manager = PermissionManager(tmp_path)
    owner_thread = get_ident()
    resolution_threads = path_resolution_threads(monkeypatch)

    result = asyncio.run(
        manager.authorize_async((file_operation(Action.FILESYSTEM_CREATE, tmp_path / "AGENTS.md"),))
    )

    assert result.decision is Decision.DENY
    assert resolution_threads
    assert owner_thread not in resolution_threads


def test_recursive_mutations_cannot_remove_protected_directories(tmp_path):
    """Recursive mutations are denied when they would contain protected control data."""
    target = FileTarget(path=str(tmp_path), recursive=True)
    manager = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(
            rules=[PermissionRule(decision=Decision.ALLOW)],
        ),
    )

    result = manager.evaluate((operation(Action.FILESYSTEM_DELETE, target=target),))

    assert result.decision is Decision.DENY
    assert result.sources == ("boundary:protected_path",)


@pytest.mark.parametrize("workspace_bound", [False, True])
@pytest.mark.parametrize("ancestor", [False, True])
def test_working_root_mutations_cannot_be_overridden(tmp_path, workspace_bound, ancestor):
    """Broad permits cannot delete a working root or an ancestor containing it."""
    manager = PermissionManager(
        tmp_path if workspace_bound else None,
        configuration=PermissionConfiguration(rules=[PermissionRule(decision=Decision.ALLOW)]),
    )
    root = manager.temporary_directory
    target = FileTarget(path=str(root.parent if ancestor else root), recursive=True)

    result = manager.evaluate((operation(Action.FILESYSTEM_DELETE, target=target),))

    assert result.decision is Decision.DENY
    assert result.sources == ("boundary:protected_path",)
    assert (
        manager.evaluate((file_operation(Action.FILESYSTEM_READ, root),)).decision is Decision.ALLOW
    )
    assert (
        manager.evaluate((file_operation(Action.FILESYSTEM_DELETE, root / "child"),)).decision
        is Decision.ALLOW
    )


def test_explicit_filesystem_roots_expand_the_hard_boundary(tmp_path):
    """Configured absolute roots deliberately extend readable and writable scope."""
    outside = tmp_path.parent / "shared"
    configuration = PermissionConfiguration(
        limits=PolicyLimits(
            readable_roots=("workspace", str(outside)),
            writable_roots=(str(outside),),
        )
    )
    manager = PermissionManager(tmp_path, configuration=configuration)

    assert (
        manager.evaluate((file_operation(Action.FILESYSTEM_READ, outside / "a.txt"),)).decision
        is Decision.ALLOW
    )
    assert (
        manager.evaluate((file_operation(Action.FILESYSTEM_CREATE, outside / "a.txt"),)).decision
        is Decision.ASK
    )


def test_loop_temp_is_allowed_by_default_and_system_temp_requires_an_explicit_root(tmp_path):
    """Loop temporary storage is safe by default; the full OS temp directory is opt-in."""

    manager = PermissionManager(tmp_path)
    external_temp = Path(tempfile.gettempdir()) / "outside-loop-temporary-file"

    assert (
        manager.evaluate(
            (file_operation(Action.FILESYSTEM_READ, manager.temporary_directory / "file.txt"),)
        ).decision
        is Decision.ALLOW
    )
    assert manager.evaluate((file_operation(Action.FILESYSTEM_READ, external_temp),)).sources == (
        "limit:workspace:readable_roots",
    )

    manager.update_limit_values("readable_roots", "system-temp", add=True)
    assert (
        manager.evaluate((file_operation(Action.FILESYSTEM_READ, external_temp),)).decision
        is Decision.ALLOW
    )


def test_workspace_root_token_requires_a_configured_workspace(tmp_path):
    """An in-memory manager cannot resolve the special workspace root token."""
    manager = PermissionManager()

    result = manager.evaluate((file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt"),))

    assert result.sources == ("limit:workspace:readable_roots",)


def test_approval_prompt_displays_targets_outside_the_workspace(tmp_path):
    """Approved expanded roots retain their absolute target in the prompt."""
    outside = tmp_path.parent / "shared" / "file.txt"
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(writable_roots=(str(outside.parent),))
        ),
    )

    result = manager.authorize((file_operation(Action.FILESYSTEM_CREATE, outside),))

    assert str(outside) in result.prompt


@pytest.mark.parametrize(
    "url",
    (
        "http://localhost/a",
        "http://127.0.0.1/a",
        "http://169.254.169.254/a",
        "http://[::1]/a",
    ),
)
def test_private_network_targets_are_hard_denied(url):
    """Local and non-global literal network targets cannot be approved."""
    parsed_origin = url.rsplit("/", 1)[0]
    target = NetworkTarget(url=url, origin=parsed_origin)

    result = PermissionManager().evaluate((operation(Action.NETWORK_REQUEST, target=target),))

    assert result.decision is Decision.DENY
    assert result.sources == ("limit:workspace:deny_private_networks",)


def test_network_origin_allowlist_is_a_non_overridable_boundary():
    """Configured origin globs reject otherwise permitted destinations."""
    manager = PermissionManager(
        configuration=PermissionConfiguration(
            limits=PolicyLimits(network_origins=("https://*.my-host.local",))
        )
    )
    allowed = NetworkTarget(url="https://api.my-host.local/a", origin="https://api.my-host.local")
    denied = NetworkTarget(url="https://other.test/a", origin="https://other.test")

    assert (
        manager.evaluate((operation(Action.NETWORK_REQUEST, target=allowed),)).decision
        is Decision.ASK
    )
    assert manager.evaluate((operation(Action.NETWORK_REQUEST, target=denied),)).sources == (
        "limit:workspace:network_origins",
    )


def test_hostname_resolution_fails_closed_for_private_and_unresolved_addresses():
    """Network policy leaves hostname resolution to the pinned request transport."""
    target = NetworkTarget(url="https://service.test/a", origin="https://service.test")
    assert (
        PermissionManager().evaluate((operation(Action.NETWORK_REQUEST, target=target),)).decision
        is Decision.ASK
    )


def test_relative_roots_resolve_against_workspace_and_temp_is_manager_owned(tmp_path):
    """Portable YAML roots and scratch directories are scoped to their manager."""
    first = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(limits=PolicyLimits(readable_roots=("shared",))),
    )
    second = PermissionManager(tmp_path)

    assert (
        first.evaluate((file_operation(Action.FILESYSTEM_READ, tmp_path / "shared/a"),)).decision
        is Decision.ALLOW
    )
    assert first.temporary_directory != second.temporary_directory
    assert first.temporary_directory.is_dir()


def test_host_processes_require_an_explicit_boundary_opt_in(tmp_path):
    """Policy rules cannot authorize host-process execution by default."""
    target = ProcessTarget(argv=("git", "status"), cwd=str(tmp_path), boundary=ProcessBoundary.HOST)
    denied = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(rules=[PermissionRule(decision=Decision.ALLOW)]),
    )
    allowed = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(allow_host_processes=True),
            defaults={Action.PROCESS_EXECUTE: Decision.ALLOW},
        ),
    )

    result = denied.evaluate((operation(Action.PROCESS_EXECUTE, target=target),))
    assert result.sources == ("limit:workspace:allow_host_processes",)
    assert "/permissions limit set session host-process allow" in result.reason
    assert "does not approve" in result.reason
    assert (
        allowed.evaluate((operation(Action.PROCESS_EXECUTE, target=target),)).decision
        is Decision.ALLOW
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_host_limit_in_a_batch_preserves_the_composed_decision(tmp_path, asynchronous):
    """A mixed operation plan retains its batch summary and the determining limit source."""
    manager = PermissionManager(tmp_path)
    target = ProcessTarget(argv=("true",), cwd=str(tmp_path), boundary=ProcessBoundary.HOST)
    operations = (
        operation(Action.PROCESS_EXECUTE, target=target),
        file_operation(Action.FILESYSTEM_READ, tmp_path / "ordinary"),
    )

    result = (
        asyncio.run(manager.evaluate_async(operations))
        if asynchronous
        else manager.evaluate(operations)
    )

    assert result.decision is Decision.DENY
    assert result.reason == "1 of 2 planned operation(s) require deny."
    assert result.sources == ("limit:workspace:allow_host_processes",)


def test_policy_mutations_persist_defaults_and_rule_lifetimes(tmp_path):
    """Policy changes round-trip while session rules remain process-local."""
    manager = PermissionManager(tmp_path)
    persisted = PermissionRule(id="persisted", decision=Decision.ALLOW, tool="read_*")
    transient = PermissionRule(id="transient", decision=Decision.DENY, tool="danger")

    manager.set_default(Action.FILESYSTEM_CREATE, Decision.DENY)
    manager.set_default(Action.NETWORK_REQUEST, Decision.ASK)
    manager.add_rule(persisted)
    manager.add_rule(transient, scope=PolicyScope.SESSION)

    loaded = PermissionManager(tmp_path)
    assert loaded.configuration.defaults[Action.FILESYSTEM_CREATE] is Decision.DENY
    assert loaded.configuration.defaults[Action.NETWORK_REQUEST] is Decision.ASK
    assert loaded.configuration.rules == [persisted]
    assert manager.remove_rule("transient", scope=PolicyScope.SESSION) is True
    assert manager.remove_rule("missing", scope=PolicyScope.SESSION) is False
    assert manager.remove_rule("persisted") is True
    assert manager.remove_rule("missing") is False


def test_preset_replacement_changes_the_selected_defaults_and_rule_layer(tmp_path):
    """A preset replaces selected-scope defaults and rules while preserving limits and overlays."""
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    manager.set_limit("allow_host_processes", True)
    manager.add_rule(PermissionRule(id="old-workspace", decision=Decision.DENY))
    manager.add_rule(
        PermissionRule(
            id="session-guard",
            decision=Decision.DENY,
            action=Action.PROCESS_EXECUTE,
        ),
        scope=PolicyScope.SESSION,
    )

    preview = manager.preview_preset_replacement(
        "workspace",
        scope=PolicyScope.WORKSPACE,
    )
    manager.replace_preset(preview)

    assert [rule.id for rule in preview.removed_rules] == ["old-workspace"]
    assert preview.removed_defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    assert not manager.persistent_rules
    assert manager.configuration.defaults[Action.FILESYSTEM_CREATE] is Decision.ALLOW
    assert manager.configuration.defaults[Action.FILESYSTEM_REPLACE] is Decision.ALLOW
    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.ASK
    assert manager.configuration.limits.allow_host_processes is True
    assert [rule.id for rule in manager.session_rules] == ["session-guard"]
    assert (
        manager.explain("write_text_file", Action.FILESYSTEM_CREATE, str(tmp_path / "new")).decision
        is Decision.ALLOW
    )
    assert not PermissionManager(tmp_path).session_rules


def test_preset_replacement_rejects_a_stale_preview_and_supports_session_scope(tmp_path):
    """Previews cannot overwrite later scoped changes and session profiles never persist."""
    manager = PermissionManager(tmp_path)
    preview = manager.preview_preset_replacement("locked", scope=PolicyScope.SESSION)
    manager.set_default(Action.FILESYSTEM_CREATE, Decision.DENY, scope=PolicyScope.SESSION)

    with pytest.raises(ValueError, match="stale"):
        manager.replace_preset(preview)

    replacement = manager.preview_preset_replacement("locked", scope=PolicyScope.SESSION)
    manager.replace_preset(replacement)

    assert set(manager.session_overrides.defaults.values()) == {Decision.DENY}
    assert manager.configuration.defaults[Action.FILESYSTEM_READ] is Decision.ALLOW
    assert not manager.session_rules
    assert not PermissionManager(tmp_path).persistent_rules


def test_session_preset_rechecks_mutations_after_preview_validation(tmp_path, monkeypatch):
    """A session mutation concurrent with preview validation cannot be overwritten."""
    manager = PermissionManager(tmp_path)
    preview = manager.preview_preset_replacement("locked", scope=PolicyScope.SESSION)

    def race(scope):
        """Return the checked revision while publishing a later user decision."""
        assert scope is PolicyScope.SESSION
        manager.set_default(Action.PROCESS_EXECUTE, Decision.DENY, scope=PolicyScope.SESSION)
        return preview.scope_revision

    monkeypatch.setattr(manager, "_scope_revision", race)
    with pytest.raises(ValueError, match="stale"):
        manager.replace_preset(preview)
    assert manager.session_overrides.defaults == {Action.PROCESS_EXECUTE: Decision.DENY}


def test_workspace_preset_rechecks_policy_after_preview_validation(tmp_path, monkeypatch):
    """A reentrant workspace change invalidates a preset before its persistent replacement."""
    manager = PermissionManager(tmp_path)
    preview = manager.preview_preset_replacement("locked", scope=PolicyScope.WORKSPACE)
    calls = 0

    def race(scope):
        """Return the old revision once, then publish and report a changed policy."""
        nonlocal calls
        assert scope is PolicyScope.WORKSPACE
        calls += 1
        if calls == 1:
            return preview.scope_revision
        manager.set_default(Action.PROCESS_EXECUTE, Decision.DENY)
        return "changed"

    monkeypatch.setattr(manager, "_scope_revision", race)
    with pytest.raises(ValueError, match="stale"):
        manager.replace_preset(preview)
    assert manager.configuration.defaults[Action.PROCESS_EXECUTE] is Decision.DENY


def test_preset_catalog_rejects_duplicate_custom_identifiers(tmp_path):
    """A caller cannot shadow a built-in preset with the same catalog identifier."""
    preset = PermissionPreset.model_validate(
        {
            "metadata": {
                "id": "observe",
                "revision": "custom",
                "title": "Duplicate",
                "description": "Duplicate identifier.",
            },
            "defaults": {action.value: "deny" for action in Action},
            "rules": [],
        }
    )

    with pytest.raises(ValueError, match="identifiers must be unique"):
        PermissionManager(tmp_path, presets=(preset,))


def test_preset_catalog_returns_copies_and_rejects_unknown_ids(tmp_path):
    """Preset lookup exposes isolated artifacts and reports missing catalog entries."""
    manager = PermissionManager(tmp_path)
    presets = manager.presets

    assert [preset.metadata.id for preset in presets] == sorted(
        preset.metadata.id for preset in presets
    )
    assert presets[0] is not manager.presets[0]
    assert manager.preset("workspace") == next(
        preset for preset in presets if preset.metadata.id == "workspace"
    )
    with pytest.raises(ValueError, match="Unknown permission preset 'missing'"):
        manager.preset("missing")


def test_preset_replacement_rejects_rule_id_collisions_with_other_scope(tmp_path):
    """A replacement cannot activate a preset rule already used by the other scope."""
    preset = PermissionPreset.model_validate(
        {
            "metadata": {
                "id": "collision",
                "revision": "1",
                "title": "Collision",
                "description": "Collision test.",
            },
            "defaults": {action.value: "deny" for action in Action},
            "rules": [{"id": "shared", "decision": "allow"}],
        }
    )
    manager = PermissionManager(tmp_path, presets=(preset,))
    manager.add_rule(
        PermissionRule(id="preset:workspace:collision:1:shared", decision=Decision.DENY),
        scope=PolicyScope.SESSION,
    )

    with pytest.raises(ValueError, match="already exists in session"):
        manager.replace_preset(
            manager.preview_preset_replacement("collision", scope=PolicyScope.WORKSPACE)
        )


def test_preset_requires_a_default_for_every_authority_action():
    """A preset cannot leave an action to the policy it is replacing."""
    with pytest.raises(ValueError, match="every known action"):
        PermissionPreset.model_validate(
            {
                "metadata": {
                    "id": "incomplete",
                    "revision": "1",
                    "title": "Incomplete",
                    "description": "Missing fallback decisions.",
                },
                "defaults": {Action.FILESYSTEM_READ.value: Decision.ALLOW.value},
                "rules": [],
            }
        )


def test_preset_rules_retain_the_selected_artifact_provenance(tmp_path):
    """Rules supplied by a complete preset retain its immutable diagnostic identity."""
    preset = PermissionPreset.model_validate(
        {
            "metadata": {
                "id": "custom",
                "revision": "1",
                "title": "Custom",
                "description": "Custom rule-bearing preset.",
            },
            "defaults": {action.value: "deny" for action in Action},
            "rules": [
                {
                    "id": "allow-read",
                    "decision": "allow",
                    "action": "filesystem.read",
                }
            ],
        }
    )
    manager = PermissionManager(tmp_path, presets=(preset,))

    manager.replace_preset(
        manager.preview_preset_replacement("custom", scope=PolicyScope.WORKSPACE)
    )

    installed = manager.persistent_rules[0]
    assert installed.source and installed.source.preset_id == "custom"
    assert "preset=custom@1" in manager.describe("workspace")


def test_rule_identifiers_are_unique_across_workspace_and_session_layers(tmp_path):
    """A rule identity always names exactly one active rule."""
    manager = PermissionManager(tmp_path)
    manager.add_rule(PermissionRule(id="unique", decision=Decision.ALLOW))

    with pytest.raises(ValueError, match="unique"):
        manager.add_rule(
            PermissionRule(id="unique", decision=Decision.DENY),
            scope=PolicyScope.SESSION,
        )

    with pytest.raises(ValueError, match="identifiers must be unique"):
        PermissionConfiguration(
            rules=[
                PermissionRule(id="duplicate", decision=Decision.ALLOW),
                PermissionRule(id="duplicate", decision=Decision.DENY),
            ]
        )


def test_limit_mutations_validate_names_deduplicate_and_support_in_memory_changes(tmp_path):
    """Limit APIs reject unknown fields and report collection changes accurately."""
    manager = PermissionManager(tmp_path)

    with pytest.raises(ValueError, match="Unknown boolean"):
        manager.set_limit("unknown", True)
    with pytest.raises(ValueError, match="Unknown collection"):
        manager.update_limit_values("unknown", "value", add=True)

    assert (
        manager.update_limit_values(
            "readable_roots", "workspace", add=True, scope=PolicyScope.SESSION
        )
        is False
    )
    assert (
        manager.update_limit_values(
            "readable_roots", str(tmp_path / "shared"), add=True, scope=PolicyScope.SESSION
        )
        is True
    )
    assert manager.update_limit_values(
        "readable_roots", str(tmp_path / "other"), add=True, scope=PolicyScope.SESSION
    )
    assert str(tmp_path / "shared") in manager.effective_configuration.limits.readable_roots
    assert str(tmp_path / "other") in manager.effective_configuration.limits.readable_roots
    assert (
        manager.update_limit_values(
            "writable_roots", "relative", add=True, scope=PolicyScope.SESSION
        )
        is True
    )
    assert str(tmp_path / "relative") in manager.effective_configuration.limits.writable_roots
    manager.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)
    assert manager.effective_configuration.limits.deny_private_networks is False


def test_session_overrides_never_leak_into_later_workspace_saves(tmp_path):
    """Persisting workspace changes cannot serialize process-local policy state."""
    manager = PermissionManager(tmp_path)
    session_rule = PermissionRule(id="session-only", decision=Decision.DENY)

    manager.set_default(Action.FILESYSTEM_DELETE, Decision.ALLOW, scope=PolicyScope.SESSION)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager.add_rule(session_rule, scope=PolicyScope.SESSION)
    manager.set_default(Action.NETWORK_REQUEST, Decision.DENY)

    loaded = PermissionManager(tmp_path)
    assert loaded.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.ASK
    assert loaded.configuration.defaults[Action.NETWORK_REQUEST] is Decision.DENY
    assert loaded.configuration.limits.allow_host_processes is False
    assert loaded.configuration.rules == []
    assert manager.effective_configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.ALLOW
    assert manager.effective_configuration.limits.allow_host_processes is True
    assert manager.session_rules == (session_rule,)


def test_session_resets_restore_workspace_inheritance_without_changing_disk(tmp_path):
    """Per-field and whole-session resets reveal the underlying workspace policy."""
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.PROCESS_EXECUTE, Decision.DENY)
    manager.set_limit("allow_host_processes", True)
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW, scope=PolicyScope.SESSION)
    manager.set_limit("allow_host_processes", False, scope=PolicyScope.SESSION)

    assert manager.reset_default(Action.PROCESS_EXECUTE) is True
    assert manager.reset_default(Action.PROCESS_EXECUTE) is False
    assert manager.reset_limit("allow_host_processes") is True
    assert manager.reset_limit("allow_host_processes") is False
    assert manager.effective_configuration.defaults[Action.PROCESS_EXECUTE] is Decision.DENY
    assert manager.effective_configuration.limits.allow_host_processes is True

    manager.add_rule(
        PermissionRule(id="temporary", decision=Decision.ASK),
        scope=PolicyScope.SESSION,
    )
    assert manager.reset_session() is True
    assert manager.reset_session() is False
    assert PermissionManager(tmp_path).configuration == manager.configuration


def test_workspace_resets_restore_application_bootstrap_values(tmp_path):
    """Workspace resets validate names and restore built-in defaults transactionally."""
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    manager.set_limit("deny_private_networks", False)

    assert manager.reset_default(Action.FILESYSTEM_READ, scope=PolicyScope.WORKSPACE) is True
    assert manager.reset_default(Action.FILESYSTEM_READ, scope=PolicyScope.WORKSPACE) is False
    assert manager.reset_limit("deny_private_networks", scope=PolicyScope.WORKSPACE) is True
    assert manager.reset_limit("deny_private_networks", scope=PolicyScope.WORKSPACE) is False
    with pytest.raises(ValueError, match="Unknown permission limit"):
        manager.reset_limit("unknown")


def test_policy_views_validate_names_and_render_sparse_session_overrides(tmp_path):
    """Each policy view is selectable and sparse session sections remain explicit."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)

    assert "Workspace policy:" in manager.describe("workspace")
    assert "Effective policy:" in manager.describe("effective")
    session = manager.describe("session")
    assert "Defaults:\n    none" in session
    assert "deny_private_networks: False" in session
    with pytest.raises(ValueError, match="Unknown permission policy view"):
        manager.describe("unknown")


def test_policy_mutations_are_transactional_when_persistence_fails(tmp_path, monkeypatch):
    """A failed atomic save leaves the active in-memory policy unchanged."""
    manager = PermissionManager(tmp_path)
    failing_os = SimpleNamespace(
        **{
            **vars(os),
            "O_NOFOLLOW": 1,
            "O_NONBLOCK": 0,
            "open": Mock(side_effect=OSError("disk full")),
        }
    )
    monkeypatch.setattr(manager_module, "os", failing_os)

    with pytest.raises(OSError, match="disk full"):
        manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)

    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.ASK


@pytest.mark.parametrize(
    "mode,links,owner",
    [
        (stat.S_IFDIR, 1, os.getuid()),
        (stat.S_IFREG, 2, os.getuid()),
        (stat.S_IFREG, 1, os.getuid() + 1),
        (stat.S_IFREG | 0o660, 1, os.getuid()),
    ],
)
def test_policy_save_refuses_unsafe_staging_inode(tmp_path, monkeypatch, mode, links, owner):
    """An unsafe staging inode is refused before its existing data is truncated."""
    path = tmp_path / "permissions.yaml"
    stage = path.with_suffix(".yaml.tmp")
    stage.write_text("existing", encoding="utf-8")
    manager = PermissionManager(tmp_path, configuration_path=path)
    inspected_os = SimpleNamespace(
        **{
            **vars(os),
            "O_NOFOLLOW": 1,
            "O_NONBLOCK": 0,
            "open": Mock(return_value=123),
            "fstat": Mock(return_value=SimpleNamespace(st_mode=mode, st_nlink=links, st_uid=owner)),
            "close": Mock(),
        }
    )
    monkeypatch.setattr(manager_module, "os", inspected_os)
    monkeypatch.setattr(manager, "_storage_lock", lambda _path: nullcontext())

    with pytest.raises(OSError, match="Unsafe policy staging file"):
        manager.add_rule(PermissionRule(id="deny", decision=Decision.DENY))

    assert stage.read_text(encoding="utf-8") == "existing"
    assert not path.exists()
    assert not manager.persistent_rules
    inspected_os.close.assert_called_once_with(123)


def test_policy_save_preserves_stage_when_fdopen_is_denied(tmp_path, monkeypatch):
    """An audit denial while wrapping the descriptor cannot truncate existing staging data."""
    path = tmp_path / "permissions.yaml"
    stage = path.with_suffix(".yaml.tmp")
    stage.write_text("existing", encoding="utf-8")
    manager = PermissionManager(tmp_path, configuration_path=path)
    blocked_os = SimpleNamespace(
        **{
            **vars(os),
            "O_NOFOLLOW": 1,
            "O_NONBLOCK": 0,
            "open": Mock(return_value=123),
            "fstat": Mock(
                return_value=SimpleNamespace(st_mode=stat.S_IFREG, st_nlink=1, st_uid=os.getuid())
            ),
            "fdopen": Mock(side_effect=PermissionError("descriptor denied")),
            "close": Mock(),
        }
    )
    monkeypatch.setattr(manager_module, "os", blocked_os)
    monkeypatch.setattr(manager, "_storage_lock", lambda _path: nullcontext())

    with pytest.raises(PermissionError, match="descriptor denied"):
        manager.add_rule(PermissionRule(id="deny", decision=Decision.DENY))

    assert stage.read_text(encoding="utf-8") == "existing"
    assert not path.exists()
    blocked_os.close.assert_called_once_with(123)


def test_policy_save_without_nofollow_requires_exclusive_stage(tmp_path, monkeypatch):
    """A platform without no-follow support refuses an existing staging entry."""
    path = tmp_path / "permissions.yaml"
    stage = path.with_suffix(".yaml.tmp")
    manager = PermissionManager(tmp_path, configuration_path=path)
    portable_os = SimpleNamespace(
        **{name: value for name, value in vars(os).items() if name != "O_NOFOLLOW"}
    )
    monkeypatch.setattr(manager_module, "os", portable_os)
    manager.add_rule(PermissionRule(id="first", decision=Decision.DENY))
    stage.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        manager.add_rule(PermissionRule(id="second", decision=Decision.DENY))

    assert stage.read_text(encoding="utf-8") == "existing"
    assert [rule.id for rule in manager.persistent_rules] == ["first"]


def test_namespaced_policy_publication_stays_owner_only_under_permissive_umask(tmp_path):
    """Policy files reload on hosts whose default umask permits group writes."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    previous_umask = os.umask(0o002)
    try:
        manager = PermissionManager(**arguments)
        manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
        manager.initialize_policy_namespace("user", user_identity, authorized=True)
    finally:
        os.umask(previous_umask)

    for path in (arguments["configuration_path"], arguments["user_configuration_path"]):
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
    reloaded = PermissionManager(**arguments)
    assert reloaded.configuration.namespace_generation == manager.configuration.namespace_generation
    assert "recovery required" not in reloaded.describe()


def test_explain_constructs_every_typed_target_without_prompting(tmp_path):
    """Effective-policy explanation handles absolute paths, URLs, processes, and session state."""
    manager = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(limits=PolicyLimits(allow_host_processes=True)),
    )

    assert (
        manager.explain("read", Action.FILESYSTEM_READ, str(tmp_path / "file.txt")).decision
        is Decision.ALLOW
    )
    assert (
        manager.explain("fetch", Action.NETWORK_REQUEST, "https://my-host.local:8443/file").decision
        is Decision.ASK
    )
    assert (
        manager.explain("fetch", Action.NETWORK_REQUEST, "https://my-host.local/file").decision
        is Decision.ASK
    )
    assert manager.explain("run", Action.PROCESS_EXECUTE, "git status").decision is Decision.ASK
    assert (
        manager.explain("skills", Action.SESSION_MUTATE, "activate:review").decision is Decision.ASK
    )
    with pytest.raises(ValueError, match="absolute HTTP"):
        manager.explain("fetch", Action.NETWORK_REQUEST, "relative")
    with pytest.raises(ValueError, match="non-empty command"):
        manager.explain("run", Action.PROCESS_EXECUTE, "")


def test_remove_rule_scans_past_nonmatching_rules():
    """Rule removal finds a requested identity beyond the first list entry."""
    manager = PermissionManager()
    manager.add_rule(PermissionRule(id="first", decision=Decision.ALLOW), scope=PolicyScope.SESSION)
    manager.add_rule(PermissionRule(id="second", decision=Decision.DENY), scope=PolicyScope.SESSION)

    assert manager.remove_rule("second", scope=PolicyScope.SESSION) is True
    result = manager.evaluate(
        (
            operation(
                Action.SESSION_MUTATE,
                target=SessionTarget(identifier="state"),
            ),
        )
    )
    assert result.sources == ("rule:session:first",)


def test_describe_exposes_defaults_boundaries_and_rule_identity(tmp_path):
    """Policy summaries expose every effective policy dimension."""
    manager = PermissionManager(tmp_path)
    manager.add_rule(
        PermissionRule(
            id="docs",
            decision=Decision.ALLOW,
            tool="read_*",
            action=Action.FILESYSTEM_READ,
            resource="*/docs/*",
        ),
        scope=PolicyScope.SESSION,
    )
    description = manager.describe()
    assert "Workspace policy:" in description
    assert "  Rules: none" in description
    assert "Session overrides:" in description

    manager.add_rule(
        PermissionRule(id="persisted", decision=Decision.DENY),
        scope=PolicyScope.SESSION,
    )
    configured = PermissionManager(
        configuration=PermissionConfiguration(
            rules=[PermissionRule(id="docs", decision=Decision.ALLOW)]
        )
    )

    description = configured.describe()
    assert "filesystem.read: allow" in description
    assert "readable_roots: workspace" in description
    assert "docs allow tool=*" in description


def test_in_memory_policy_cannot_be_saved():
    """Persistence without a local configuration path is rejected explicitly."""
    with pytest.raises(ValueError, match="cannot persist"):
        PermissionManager().save()


def test_stale_manager_cannot_overwrite_workspace_policy(tmp_path):
    """Independent managers reject a write based on an outdated workspace policy."""
    first = PermissionManager(tmp_path)
    stale = PermissionManager(tmp_path)
    first.set_default(Action.FILESYSTEM_READ, Decision.DENY)

    with pytest.raises(OSError, match="changed.*reload"):
        stale.set_default(Action.NETWORK_REQUEST, Decision.DENY)

    current = PermissionManager(tmp_path)
    assert current.configuration.defaults[Action.FILESYSTEM_READ] is Decision.DENY
    assert stale.configuration.defaults[Action.NETWORK_REQUEST] is Decision.ASK
    assert stale.reload() is PermissionLoadResult.LOADED
    stale.set_default(Action.NETWORK_REQUEST, Decision.DENY)
    current.reload()
    assert current.configuration.defaults[Action.FILESYSTEM_READ] is Decision.DENY
    assert current.configuration.defaults[Action.NETWORK_REQUEST] is Decision.DENY


def test_stale_manager_cannot_reset_newer_workspace_policy(tmp_path):
    """A reset cannot archive a policy changed by another manager."""
    first = PermissionManager(tmp_path)
    stale = PermissionManager(tmp_path)
    first.set_default(Action.FILESYSTEM_READ, Decision.DENY)

    with pytest.raises(OSError, match="changed.*reload"):
        stale.reset_configuration("workspace")

    assert not tuple(first.configuration_path.parent.glob("*.bak"))
    assert (
        PermissionManager(tmp_path).configuration.defaults[Action.FILESYSTEM_READ] is Decision.DENY
    )


@pytest.mark.parametrize("reset", [False, True])
def test_policy_change_while_acquiring_storage_lock_refuses_stale_candidate(
    tmp_path, monkeypatch, reset
):
    """A reentrant policy change during lock acquisition invalidates the pending update."""
    manager = PermissionManager(tmp_path)
    original = manager._storage_lock

    @contextmanager
    def change_policy(path):
        """Change the active session before returning storage ownership."""
        manager.set_default(Action.FILESYSTEM_READ, Decision.DENY, scope=PolicyScope.SESSION)
        with original(path):
            yield

    monkeypatch.setattr(manager, "_storage_lock", change_policy)
    with pytest.raises(OSError, match="changed.*reload"):
        if reset:
            manager.reset_configuration("workspace")
        else:
            manager.set_default(Action.NETWORK_REQUEST, Decision.DENY)
    assert not manager.configuration_path.exists()


@pytest.mark.parametrize("kind", ["nonregular", "hardlink", "replaced"])
def test_policy_save_refuses_unsafe_lock_descriptor(tmp_path, monkeypatch, kind):
    """A policy lock must be a uniquely linked regular file bound to its name."""
    manager = PermissionManager(tmp_path)
    real_fstat = os.fstat

    def inspect(descriptor):
        """Simulate an unsafe identity for the opened lock."""
        result = real_fstat(descriptor)
        return SimpleNamespace(
            st_mode=stat.S_IFDIR if kind == "nonregular" else result.st_mode,
            st_nlink=2 if kind == "hardlink" else result.st_nlink,
            st_dev=result.st_dev,
            st_ino=result.st_ino + (kind == "replaced"),
        )

    monkeypatch.setattr(manager_module, "os", SimpleNamespace(**{**vars(os), "fstat": inspect}))
    with pytest.raises(OSError, match="Unsafe policy lock"):
        manager.set_default(Action.NETWORK_REQUEST, Decision.DENY)
    assert not manager.configuration_path.exists()


def test_policy_save_fails_closed_without_native_locking(tmp_path, monkeypatch):
    """A host without native advisory locking cannot perform an uncoordinated save."""
    manager = PermissionManager(tmp_path)
    monkeypatch.setattr(manager_module, "import_module", Mock(side_effect=ModuleNotFoundError()))
    with pytest.raises(OSError, match="Native policy locking is unavailable"):
        manager.set_default(Action.NETWORK_REQUEST, Decision.DENY)
    assert not manager.configuration_path.exists()


def test_policy_save_uses_windows_lock_when_posix_locking_is_unavailable(tmp_path, monkeypatch):
    """The native Windows byte-range lock coordinates a policy save without POSIX fcntl."""
    manager = PermissionManager(tmp_path)
    locking = Mock()
    windows = SimpleNamespace(locking=locking, LK_NBLCK=42)

    def platform_lock(name):
        """Expose only the Windows locking interface to the manager."""
        if name == "fcntl":
            raise ModuleNotFoundError(name)
        assert name == "msvcrt"
        return windows

    monkeypatch.setattr(manager_module, "import_module", platform_lock)
    manager.set_default(Action.NETWORK_REQUEST, Decision.DENY)
    assert locking.call_args.args[1:] == (42, 1)
    assert manager.configuration.defaults[Action.NETWORK_REQUEST] is Decision.DENY


def test_policy_save_without_nofollow_refuses_unsafe_existing_lock(tmp_path, monkeypatch):
    """Fallback lock opening rejects an existing nonregular entry before opening it."""
    manager = PermissionManager(tmp_path)
    lock = manager.configuration_path.with_name(manager.configuration_path.name + ".lock")
    lock.parent.mkdir()
    lock.mkdir()
    operating_system = {key: value for key, value in vars(os).items() if key != "O_NOFOLLOW"}
    monkeypatch.setattr(manager_module, "os", SimpleNamespace(**operating_system))

    with pytest.raises(OSError, match="Unsafe policy lock"):
        manager.set_default(Action.NETWORK_REQUEST, Decision.DENY)
    assert not manager.configuration_path.exists()


def test_policy_save_without_nofollow_reuses_safe_lock(tmp_path, monkeypatch):
    """Fallback lock opening permits repeated saves with the same validated regular lock."""
    manager = PermissionManager(tmp_path)
    operating_system = {key: value for key, value in vars(os).items() if key != "O_NOFOLLOW"}
    monkeypatch.setattr(manager_module, "os", SimpleNamespace(**operating_system))

    manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    manager.set_default(Action.NETWORK_REQUEST, Decision.DENY)
    assert manager.configuration.defaults[Action.FILESYSTEM_READ] is Decision.DENY
    assert manager.configuration.defaults[Action.NETWORK_REQUEST] is Decision.DENY


def test_stale_manager_detects_intervening_equal_policy(tmp_path):
    """A new commit has a distinct revision even if its policy values return to their old state."""
    first = PermissionManager(tmp_path)
    first.save()
    stale = PermissionManager(tmp_path)
    first.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    first.set_default(Action.FILESYSTEM_READ, Decision.ALLOW)

    with pytest.raises(OSError, match="changed.*reload"):
        stale.set_default(Action.NETWORK_REQUEST, Decision.DENY)

    assert (
        PermissionManager(tmp_path).configuration.defaults[Action.NETWORK_REQUEST] is Decision.ASK
    )


def test_stale_user_approval_cannot_overwrite_another_manager(tmp_path):
    """A remembered user approval cannot replace another manager's newer approval."""
    user_path = tmp_path / "user" / "permissions.yaml"
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.USER
    first = PermissionManager(
        tmp_path / "first", user_configuration_path=user_path, interaction=interaction
    )
    stale = PermissionManager(
        tmp_path / "second", user_configuration_path=user_path, interaction=interaction
    )
    first_request = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://first.example", origin="https://first.example"),
    )
    second_request = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://second.example", origin="https://second.example"),
    )
    assert first.authorize((first_request,)).decision is Decision.ALLOW

    result = stale.authorize((second_request,))
    assert result.decision is Decision.DENY
    assert "changed; reload" in result.reason

    current = PermissionManager(tmp_path / "third", user_configuration_path=user_path)
    assert current.evaluate((first_request,)).decision is Decision.ALLOW
    assert current.evaluate((second_request,)).decision is Decision.ASK


def test_stale_async_user_approval_cannot_overwrite_another_manager(tmp_path):
    """An asynchronous remembered approval rejects a newer user-wide policy."""
    user_path = tmp_path / "user" / "permissions.yaml"
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.USER
    interaction.prompt_async = AsyncMock(return_value=ApprovalChoice.USER)
    interaction.confirm_async = AsyncMock()
    first = PermissionManager(
        tmp_path / "first", user_configuration_path=user_path, interaction=interaction
    )
    stale = PermissionManager(
        tmp_path / "second", user_configuration_path=user_path, interaction=interaction
    )
    first_request = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://first.example", origin="https://first.example"),
    )
    second_request = operation(
        Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://second.example", origin="https://second.example"),
    )
    assert first.authorize((first_request,)).decision is Decision.ALLOW

    result = asyncio.run(stale.authorize_async((second_request,)))

    assert result.decision is Decision.DENY
    assert "changed; reload" in result.reason
    current = PermissionManager(tmp_path / "third", user_configuration_path=user_path)
    assert current.evaluate((first_request,)).decision is Decision.ALLOW
    assert current.evaluate((second_request,)).decision is Decision.ASK


def test_host_retry_prompt_describes_a_completed_attempt():
    """A completed-attempt warning distinguishes an observed denial from its uncertain cause."""
    warning = PermissionManager.host_retry_prompt("printf ok", "completed", 7, True)
    assert 'Command: "printf ok"' in warning
    assert "exited with code 7" in warning
    assert "denial may be unrelated" in warning
    assert "Retrying may repeat" in warning


def test_empty_permission_mapping_requires_recovery(tmp_path):
    """An empty policy cannot silently become built-in permissions."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text("{}\n", "utf-8")

    with pytest.raises(PermissionConfigurationError, match="must not be empty"):
        PermissionManager(tmp_path)
    assert path.read_text(encoding="utf-8") == "{}\n"


def test_error_policy_raises_configuration_errors_and_preserves_active_policy(tmp_path):
    """Strict startup and reload expose typed errors without replacing valid active policy."""
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    path = tmp_path / ".loop" / "permissions.yaml"
    path.write_text("version: 3\n", "utf-8")

    with pytest.raises(PermissionConfigurationError) as raised:
        manager.reload()

    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    assert raised.value.path == str(path)
    assert raised.value.__cause__ is not None
    with pytest.raises(PermissionConfigurationError):
        PermissionManager(tmp_path)


def test_reload_retains_active_policy_when_configuration_file_disappears(tmp_path):
    """A missing policy on reload cannot discard an active denial."""
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    manager.configuration_path.unlink()

    assert manager.reload() is PermissionLoadResult.RETAINED
    assert manager.configuration.defaults[Action.FILESYSTEM_READ] is Decision.DENY


def test_startup_refuses_reset_started_during_missing_policy_check(tmp_path, monkeypatch):
    """A reset beginning during startup cannot be mistaken for an absent first-use policy."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    marker = path.with_name(path.name + ".reset-pending")
    original = Path.read_bytes

    def start_reset(candidate):
        """Place the reset marker immediately after the first missing-policy observation."""
        if candidate == path and not marker.exists():
            marker.touch()
            raise FileNotFoundError(2, "missing", str(candidate))
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", start_reset)
    with pytest.raises(PermissionConfigurationError, match="reset is pending"):
        PermissionManager(tmp_path)


def test_user_startup_refuses_reset_started_during_missing_policy_check(tmp_path, monkeypatch):
    """A user reset racing a missing-file observation cannot load empty approvals."""
    path = tmp_path / "user.yaml"
    marker = path.with_name(path.name + ".reset-pending")
    original = Path.read_bytes

    def start_reset(candidate):
        """Create reset evidence immediately after the first missing-file observation."""
        if candidate == path and not marker.exists():
            marker.touch()
            raise FileNotFoundError(2, "missing", str(candidate))
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", start_reset)
    with pytest.raises(PermissionConfigurationError, match="reset is pending"):
        PermissionManager(
            tmp_path, user_configuration_path=path, load_policy=PermissionLoadPolicy.AUTO
        )


def test_user_startup_loads_policy_published_during_missing_check(tmp_path, monkeypatch):
    """A user policy published during a missing-file check is loaded on the second read."""
    path = tmp_path / "user.yaml"
    original = Path.read_bytes
    saved = UserPermissionConfiguration(
        rules=[
            PermissionRule(id="deny-network", decision=Decision.DENY, action=Action.NETWORK_REQUEST)
        ]
    )

    def publish(candidate):
        """Publish a restrictive user rule after the first missing-file observation."""
        if candidate == path and not candidate.exists():
            path.write_text(dump_yaml(saved.model_dump(mode="json")), encoding="utf-8")
            raise FileNotFoundError(2, "missing", str(candidate))
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", publish)
    manager = PermissionManager(tmp_path, user_configuration_path=path)
    assert [rule.id for rule in manager.user_rules] == ["deny-network"]


@pytest.mark.parametrize("scope", ["workspace", "user"])
@pytest.mark.parametrize("interactive", [False, True])
def test_policy_load_refuses_file_archived_between_exists_and_read(
    tmp_path, monkeypatch, scope, interactive
):
    """A read racing reset archival fails closed under automatic and interactive recovery."""
    path = tmp_path / (".loop/permissions.yaml" if scope == "workspace" else "user.yaml")
    path.parent.mkdir(exist_ok=True)
    path.write_text("version: 1\n", encoding="utf-8")
    original = Path.read_bytes
    marker = path.with_name(path.name + ".reset-pending")
    archive = path.with_name(path.name + ".raced.bak")

    def archive_before_read(candidate):
        """Archive a policy after its existence was observed but before byte retrieval."""
        if candidate == path:
            marker.touch()
            path.replace(archive)
            raise FileNotFoundError("policy reset raced with read")
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", archive_before_read)
    interaction = Mock(spec=Interaction) if interactive else None
    with pytest.raises(PermissionConfigurationError, match="reset is pending"):
        PermissionManager(
            tmp_path,
            user_configuration_path=path if scope == "user" else None,
            load_policy=(
                PermissionLoadPolicy.INTERACTIVE if interactive else PermissionLoadPolicy.AUTO
            ),
            interaction=interaction,
        )
    assert archive.read_bytes() == b"version: 1\n"
    assert marker.exists()
    if interaction is not None:
        interaction.prompt.assert_not_called()


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_automatic_recovery_never_treats_a_policy_read_veto_as_malformed_data(
    tmp_path, monkeypatch, scope
):
    """A denied policy read remains an error under automatic recovery for either scope."""
    path = tmp_path / (".loop/permissions.yaml" if scope == "workspace" else "user.yaml")
    path.parent.mkdir(exist_ok=True)
    path.write_text("version: 1\n", encoding="utf-8")
    original = Path.read_bytes

    def refuse_read(candidate):
        """Refuse only the selected policy read at the filesystem boundary."""
        if candidate == path:
            raise PermissionError("policy read refused")
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", refuse_read)
    with pytest.raises(PermissionConfigurationError, match="Permission denied") as raised:
        PermissionManager(
            tmp_path,
            user_configuration_path=path if scope == "user" else None,
            load_policy=PermissionLoadPolicy.AUTO,
        )
    assert isinstance(raised.value.__cause__, PermissionError)


def test_startup_reads_policy_published_during_missing_check(tmp_path, monkeypatch):
    """A policy published during startup is loaded instead of replaced by first-use defaults."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    original = Path.read_bytes

    def publish(candidate):
        """Publish a valid policy just after the first missing-policy observation."""
        if candidate == path and not candidate.exists():
            path.write_text("version: 1\ndefaults:\n  filesystem.read: deny\n", "utf-8")
            raise FileNotFoundError(2, "missing", str(candidate))
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", publish)
    assert (
        PermissionManager(tmp_path).configuration.defaults[Action.FILESYSTEM_READ] is Decision.DENY
    )


@pytest.mark.parametrize("load_policy", list(PermissionLoadPolicy))
def test_interrupted_reset_requires_recovery(tmp_path, monkeypatch, load_policy):
    """An interrupted reset cannot silently activate defaults on a subsequent startup."""
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    path = manager.configuration_path
    monkeypatch.setattr(
        manager, "_persist_configuration", Mock(side_effect=PermissionError("reset refused"))
    )

    with pytest.raises(PermissionError, match="reset refused"):
        manager.reset_configuration("workspace")

    archives = tuple(path.parent.glob(f"{path.name}.*.bak"))
    assert len(archives) == 1
    assert not path.exists()
    marker = path.with_name(f"{path.name}.reset-pending")
    assert marker.exists()
    interaction = Mock(spec=Interaction)
    with pytest.raises(PermissionConfigurationError, match="reset is pending"):
        PermissionManager(tmp_path, load_policy=load_policy, interaction=interaction)
    interaction.prompt.assert_not_called()

    archives[0].replace(path)
    marker.unlink()
    recovered = PermissionManager(tmp_path, load_policy=load_policy, interaction=interaction)
    assert recovered.configuration.defaults[Action.FILESYSTEM_READ] is Decision.DENY


def test_async_evaluation_denies_a_new_pending_reset(tmp_path):
    """Async authorization checks a new reset marker before any operation is admitted."""
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.SESSION_MUTATE, Decision.ALLOW)
    operation = Operation(
        tool_id="example", action=Action.SESSION_MUTATE, target=SessionTarget(identifier="test")
    )
    marker = manager.configuration_path.with_name("permissions.yaml.reset-pending")
    marker.touch()

    assert asyncio.run(manager.evaluate_async((operation,))).decision is Decision.DENY
    assert manager.evaluate((operation,)).decision is Decision.DENY
    assert asyncio.run(manager.evaluate_async((operation,))).decision is Decision.DENY


def test_missing_policy_cannot_ignore_reset_marker_inspection_errors(tmp_path, monkeypatch):
    """A refused marker inspection fails closed instead of treating a workspace as new."""
    marker = tmp_path / ".loop" / "permissions.yaml.reset-pending"
    original = Path.lstat

    def deny_stat(path, *args, **kwargs):
        """Deny only inspection of the owned reset marker."""
        if path == marker:
            raise PermissionError("marker inspection refused")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", deny_stat)
    with pytest.raises(PermissionConfigurationError, match="marker inspection refused") as raised:
        PermissionManager(tmp_path, load_policy=PermissionLoadPolicy.AUTO)
    assert isinstance(raised.value.__cause__, PermissionError)


def test_first_use_does_not_require_listing_policy_directory(tmp_path, monkeypatch):
    """A new workspace need not list its policy directory to use defaults."""
    directory = tmp_path / ".loop"
    directory.mkdir()
    monkeypatch.setattr(Path, "iterdir", Mock(side_effect=PermissionError("listing refused")))

    assert PermissionManager(tmp_path).configuration == PermissionConfiguration()


def test_reload_retains_in_memory_policy():
    """Reload cannot replace an in-memory policy with defaults."""
    configuration = PermissionConfiguration()
    configuration.defaults[Action.FILESYSTEM_READ] = Decision.DENY
    manager = PermissionManager(configuration=configuration)

    assert manager.reload() is PermissionLoadResult.RETAINED
    assert manager.configuration.defaults[Action.FILESYSTEM_READ] is Decision.DENY


def test_auto_policy_reports_and_uses_defaults_or_last_known_good(tmp_path, caplog):
    """Automatic recovery reports startup defaults and retains valid policy on reload."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text("version: 3\n", "utf-8")

    manager = PermissionManager(tmp_path, load_policy=PermissionLoadPolicy.AUTO)

    assert manager.configuration == PermissionConfiguration()
    assert "permission operations are blocked" in caplog.text
    with pytest.raises(PermissionNamespaceError):
        manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    path.write_text("version: 1\n", "utf-8")
    manager.reload()
    manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    path.write_text("version: 3\n", "utf-8")
    interaction = Mock(spec=Interaction)
    manager.interaction = interaction

    assert manager.reload() is PermissionLoadResult.BLOCKED
    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    interaction.report.assert_called_once()
    interaction.warning.assert_called_once()


def test_interactive_policy_retries_a_repaired_file(tmp_path):
    """Interactive recovery rereads a file repaired while the recovery prompt is active."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text("version: 3\n", "utf-8")
    interaction = Mock(spec=Interaction)

    def repair(*_args, **_kwargs):
        path.write_text("version: 1\ndefaults:\n  filesystem.delete: deny\n", "utf-8")
        return "retry"

    interaction.prompt.side_effect = repair
    manager = PermissionManager(tmp_path, interaction=interaction)

    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    interaction.report.assert_called_once()
    assert interaction.prompt.call_args.kwargs["index"] == {
        "retry": "R",
        "continue": "C",
        "reset": "T",
        "exit": "Q",
    }


def test_interactive_recovery_does_not_overwrite_a_later_policy_update(tmp_path):
    """A policy changed during recovery is reloaded instead of reset by a stale decision."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    previous = manager.configuration_path.read_bytes()
    manager.configuration_path.write_text("version: 3\n", "utf-8")

    def update_while_prompted(*_args, **_kwargs):
        """Publish a new restrictive policy before the old recovery choice returns."""
        manager.configuration_path.write_bytes(previous)
        manager.reload()
        manager.set_default(Action.PROCESS_EXECUTE, Decision.DENY)
        return "reset"

    interaction.prompt.side_effect = update_while_prompted
    assert manager.reload() is PermissionLoadResult.LOADED
    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    assert manager.configuration.defaults[Action.PROCESS_EXECUTE] is Decision.DENY
    assert not tuple(manager.configuration_path.parent.glob("*.bak"))


def test_interactive_reload_can_retain_the_active_policy(tmp_path):
    """Interactive reload retains last-known-good state when the user continues."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    manager.configuration_path.write_text("version: 3\n", "utf-8")
    interaction.prompt.return_value = "continue"

    assert manager.reload() is PermissionLoadResult.BLOCKED
    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    interaction.warning.assert_called_once_with(
        "Permission operations remain blocked until the policy is repaired."
    )


@pytest.mark.parametrize("choice", [None, False, "unexpected"])
def test_interactive_recovery_requires_an_explicit_choice(tmp_path, choice):
    """An invalid recovery response cannot silently select a default policy."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.configuration_path.parent.mkdir(exist_ok=True)
    manager.configuration_path.write_text("version: invalid\n", encoding="utf-8")
    interaction.prompt.return_value = choice

    with pytest.raises(ShutdownRequested):
        manager.reload()


def test_interactive_loading_requires_an_interaction(tmp_path):
    """Interactive loading cannot silently start or recover without an interaction."""
    with pytest.raises(ValueError, match="requires an Interaction"):
        PermissionManager(load_policy=PermissionLoadPolicy.INTERACTIVE)

    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.interaction = None
    manager.configuration_path.parent.mkdir(exist_ok=True)
    manager.configuration_path.write_text("version: 3\n", "utf-8")
    with pytest.raises(PermissionConfigurationError):
        manager.reload()


def test_preset_failures_raise_strictly_and_report_during_headless_auto_recovery(
    monkeypatch, caplog
):
    """Preset failures follow the same strict or reported recovery policy as configuration."""
    failure = PermissionLoadFailure(
        source="preset", path="presets/broken.yaml", message="invalid schema"
    )
    monkeypatch.setattr(
        PermissionPreset,
        "load_builtin_presets",
        classmethod(lambda cls: ((), (failure,))),
    )

    with pytest.raises(PermissionPresetError) as raised:
        PermissionManager()
    assert raised.value.failures == (failure,)

    manager = PermissionManager(accepted_preset_failures=(failure,))
    assert not manager.presets
    assert "Excluded invalid permission preset" in caplog.text


def test_preset_error_requires_a_failure():
    """The aggregate preset error rejects an empty diagnostic collection."""
    with pytest.raises(ValueError, match="at least one failure"):
        PermissionPresetError(())


def test_reset_configuration_creates_defaults_when_no_policy_file_exists(tmp_path):
    """An explicit reset creates the default policy even before a policy file exists."""
    manager = PermissionManager(tmp_path)

    assert manager.reset_configuration("workspace") is None
    assert (tmp_path / ".loop" / "permissions.yaml").exists()


def test_in_memory_policy_cannot_be_reset():
    """Reset requires a local policy path to preserve its archival contract."""
    with pytest.raises(ValueError, match="cannot reset"):
        PermissionManager().reset_configuration("workspace")
    with pytest.raises(ValueError, match="Unknown permission policy scope"):
        PermissionManager().reset_configuration("unknown")
    with pytest.raises(ValueError, match="user permission policy path"):
        PermissionManager().reset_configuration("user")


def test_diagnostic_audit_failure_does_not_change_authorization(tmp_path, monkeypatch):
    """Unavailable diagnostic JSONL storage cannot turn a permit into a denial."""
    manager = PermissionManager(tmp_path)
    monkeypatch.setattr("pathlib.Path.open", Mock(side_effect=OSError("unavailable")))

    result = manager.authorize((file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt"),))

    assert isinstance(result, AuthorizationResult)
    assert result.decision is Decision.ALLOW


def test_permission_manager_configuration_path_and_recorder(tmp_path):
    """The manager exposes its .loop policy path and a settable recorder sink."""
    recorder = Mock()
    workspace = PermissionManager(tmp_path)
    workspace.recorder = recorder
    assert workspace.recorder is recorder
    assert workspace.configuration_path.parent.parent == tmp_path

    in_memory = PermissionManager()
    assert in_memory.configuration_path is None


def test_approval_prompt_anchors_a_workspace_rooted_target(tmp_path):
    """A prompt for the workspace root renders an explicit workspace anchor."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.set_default(Action.FILESYSTEM_READ, Decision.ASK, scope=PolicyScope.SESSION)

    result = manager.authorize((file_operation(Action.FILESYSTEM_READ, tmp_path),))

    assert result.prompt is not None
    assert "workspace root:" in result.prompt


def test_approval_prompt_renders_session_targets_without_workspace():
    """A manager without a workspace renders session targets without a relative path."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(interaction=interaction)

    result = manager.authorize(
        (operation(Action.SESSION_MUTATE, target=SessionTarget(identifier="config")),)
    )

    assert result.prompt is not None
    assert "config" in result.prompt


def test_process_target_display_resolves_local_paths_to_workspace_virtual_paths(tmp_path):
    """Process target prompts render local workspace paths as VirtualPaths."""
    sub = tmp_path / "src" / "loop"
    sub.mkdir(parents=True)
    readme = tmp_path / "README.md"
    readme.touch()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(allow_host_processes=True),
        ),
    )
    target = ProcessTarget(
        argv=("cat", str(readme)),
        cwd=str(sub),
        boundary=ProcessBoundary.HOST,
    )
    manager.authorize((operation(Action.PROCESS_EXECUTE, target=target),))

    prompt = interaction.info.call_args.args[0]
    assert "cat '$WORKSPACE_ROOT/README.md'" in prompt
    assert "(cwd: $WORKSPACE_ROOT/src/loop)" in prompt
    assert "../" not in prompt


def test_process_target_display_preserves_workspace_relative_argv(tmp_path):
    """Relative argv that stays within workspace passes through cleanly."""
    target = ProcessTarget(
        argv=("cat", "README.md"),
        cwd=str(tmp_path),
        boundary=ProcessBoundary.HOST,
    )
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(allow_host_processes=True),
        ),
    )

    manager.authorize((operation(Action.PROCESS_EXECUTE, target=target),))

    assert "cat README.md (cwd: $WORKSPACE_ROOT)" in interaction.info.call_args.args[0]


def test_process_target_display_preserves_argument_boundaries(tmp_path):
    """Process prompts quote arguments so distinct argv remain visibly distinct."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(allow_host_processes=True),
        ),
    )
    target = ProcessTarget(
        argv=("tool", "a b"),
        cwd=str(tmp_path),
        boundary=ProcessBoundary.HOST,
    )

    manager.authorize((operation(Action.PROCESS_EXECUTE, target=target),))

    assert "tool 'a b' (cwd: $WORKSPACE_ROOT)" in interaction.info.call_args.args[0]


def test_process_target_display_handles_temporary_virtual_paths(tmp_path):
    """Temporary directory paths are rendered below the temporary VirtualPath."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(allow_host_processes=True),
        ),
    )
    scratch_file = manager.temporary_directory / "output.log"
    scratch_file.parent.mkdir(exist_ok=True)
    target = ProcessTarget(
        argv=("cat", str(scratch_file)),
        cwd=str(tmp_path),
        boundary=ProcessBoundary.HOST,
    )

    manager.authorize((operation(Action.PROCESS_EXECUTE, target=target),))

    prompt = interaction.info.call_args.args[0]
    assert "$TMPDIR/output.log" in prompt


def test_process_target_display_preserves_external_temporary_paths(tmp_path):
    """An external temporary path is not redacted in an approval prompt."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(allow_host_processes=True),
        ),
    )
    external_path = Path("/tmp/verify_review.py")
    target = ProcessTarget(
        argv=(".venv/bin/python", str(external_path)),
        cwd=str(tmp_path),
        boundary=ProcessBoundary.HOST,
    )

    manager.authorize((operation(Action.PROCESS_EXECUTE, target=target),))

    prompt = interaction.info.call_args.args[0]
    assert str(external_path) in prompt
    assert "<external>" not in prompt


def test_bound_session_owner_controls_scratch_data_scope(tmp_path):
    """Fresh-data classification follows the trusted owner and resets on restricted binding."""
    manager = PermissionManager(tmp_path)
    reference = PathHolder(manager.temporary_directory)
    state = [True]
    manager.bind_temporary_directory(reference, ordinary_names=lambda: state[0])
    assert manager.temporary_directory_allows_private_names
    state[0] = False
    assert not manager.temporary_directory_allows_private_names
    manager.bind_temporary_directory(reference)
    assert not manager.temporary_directory_allows_private_names


@pytest.mark.parametrize("scope", ["workspace", "user"])
@pytest.mark.parametrize("choice", ["retry", "reset", "exit"])
def test_invalid_policy_recovery_blocks_admission_during_prompt(tmp_path, scope, choice):
    """A concurrent authorization cannot use stale grants while recovery is awaiting input."""
    path = tmp_path / f"{scope}.yaml"
    path.write_text("version: invalid\n", encoding="utf-8")
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    if scope == "workspace":
        manager._configuration_path = path
    else:
        manager._user_configuration_path = path
    operation_set = (file_operation(Action.FILESYSTEM_READ, tmp_path / "file"),)

    def choose(*_args, **_kwargs):
        assert manager.evaluate(operation_set).decision is Decision.DENY
        assert asyncio.run(manager.evaluate_async(operation_set)).decision is Decision.DENY
        if choice == "retry":
            path.write_text("version: 1\n", encoding="utf-8")
        return choice

    interaction.prompt.side_effect = choose
    if choice == "exit":
        with pytest.raises(ShutdownRequested):
            manager.reload()
    else:
        manager.reload()
        assert manager.evaluate(operation_set).decision is not Decision.DENY
    if choice == "reset":
        assert list(tmp_path.glob(f"{scope}.yaml.*.bak"))


@pytest.mark.parametrize("scope,reason", [("invalid", "reason"), ("user", "")])
def test_suspend_policy_requires_a_scope_and_reason(scope, reason):
    """A recovery latch cannot be installed without a meaningful scope and cause."""
    manager = PermissionManager()
    with pytest.raises(ValueError, match="scope and recovery reason"):
        manager.suspend_policy(scope, reason)


def test_namespace_empty_mapping_is_not_a_valid_policy(tmp_path):
    """Even a reserved namespace cannot treat an empty policy as its defaults."""
    arguments, workspace_id, _user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    arguments["configuration_path"].write_text("{}\n", encoding="utf-8")
    manager.reload()
    with pytest.raises(PermissionNamespaceError, match="must not be empty"):
        manager.require_policy_ready()


def test_user_recovery_retries_after_policy_changes_during_prompt(tmp_path):
    """A stale recovery decision is discarded and the changed policy is reread."""
    path = tmp_path / "user.yaml"
    path.write_text("version: invalid\n", encoding="utf-8")
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager._user_configuration_path = path

    def repair(*_args, **_kwargs):
        path.write_text("version: 1\n", encoding="utf-8")
        manager.set_default(Action.FILESYSTEM_READ, Decision.DENY, scope=PolicyScope.SESSION)
        return "continue"

    interaction.prompt.side_effect = repair
    manager.reload()
    assert interaction.prompt.call_count == 1
    assert (
        manager.evaluate((file_operation(Action.FILESYSTEM_READ, tmp_path / "f"),)).decision
        is Decision.DENY
    )


def test_headless_user_recovery_raises_the_failure(tmp_path):
    """A missing interaction cannot turn an invalid user policy into default approvals."""
    path = tmp_path / "user.yaml"
    path.write_text("version: invalid\n", encoding="utf-8")
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager._user_configuration_path = path
    manager._interaction = None
    with pytest.raises(PermissionConfigurationError):
        manager._load_user_configuration()


def test_accepted_preset_exclusion_is_reported_to_interaction(monkeypatch):
    """An explicitly excluded malformed preset remains visible in the current session."""
    failure = PermissionLoadFailure(source="preset", path="broken.yaml", message="invalid")
    monkeypatch.setattr(
        PermissionPreset, "load_builtin_presets", classmethod(lambda cls: ((), (failure,)))
    )
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(interaction=interaction, accepted_preset_failures=(failure,))
    assert manager.presets == ()
    assert "broken.yaml" in interaction.warning.call_args.args[0]


def test_auto_user_policy_failure_reports_blocked_operations(tmp_path):
    """Automatic policy mode reports invalid user rules and still blocks authorization."""
    path = tmp_path / "user.yaml"
    path.write_text("version: invalid\n", encoding="utf-8")
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(
        tmp_path,
        user_configuration_path=path,
        interaction=interaction,
        load_policy=PermissionLoadPolicy.AUTO,
    )
    assert interaction.report.called
    assert manager._namespace_recovery_decision().decision is Decision.DENY


def test_sandbox_authorization_propagates_cancelled_audit(tmp_path):
    """A cancelled sandbox audit never falls through to authorize a command."""
    interaction = ConsoleInteraction(console=Console(file=io.StringIO()), markdown=False)
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = native_request(tmp_path, "printf safe", read_roots=())
    original = manager._run_blocking_settled_async

    async def cancel_audit(function, *args, **kwargs):
        if function == manager._audit_sandbox_scope:
            return None, asyncio.CancelledError()
        return await original(function, *args, **kwargs)

    manager._run_blocking_settled_async = cancel_audit
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(manager.authorize_sandboxed_command_async(request, tool_id="run_command"))


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_boolean_policy_version_requires_explicit_recovery(tmp_path, scope):
    """A boolean schema version is never accepted as policy version one."""
    path = tmp_path / (".loop/permissions.yaml" if scope == "workspace" else "user.yaml")
    path.parent.mkdir(exist_ok=True)
    path.write_text("version: true\n", encoding="utf-8")
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "continue"
    manager = PermissionManager(
        tmp_path,
        user_configuration_path=path if scope == "user" else None,
        interaction=interaction,
    )
    interaction.report.assert_called_once()
    assert "version" in interaction.report.call_args.args[0].detail
    with pytest.raises(PermissionNamespaceError):
        manager.require_policy_ready()


@pytest.mark.parametrize("scope", ["workspace", "user"])
@pytest.mark.parametrize("when", ["before", "after"])
def test_policy_reset_failure_prompts_and_stays_blocked(tmp_path, monkeypatch, scope, when):
    """Failed reset needs a fresh decision and never clears policy restrictions."""
    path = tmp_path / (".loop/permissions.yaml" if scope == "workspace" else "user.yaml")
    path.parent.mkdir(exist_ok=True)
    path.write_text("version: 3\n", encoding="utf-8")
    interaction = Mock(spec=Interaction)
    interaction.prompt.side_effect = ["reset", "continue"]
    if when == "before":
        monkeypatch.setattr(
            PermissionManager,
            "reset_configuration",
            Mock(side_effect=PermissionError("synthetic-secret")),
        )
    else:
        monkeypatch.setattr(
            PermissionManager,
            "_persist_configuration",
            Mock(side_effect=PermissionError("synthetic-secret")),
        )
    manager = PermissionManager(
        tmp_path,
        user_configuration_path=path if scope == "user" else None,
        interaction=interaction,
    )
    assert interaction.prompt.call_count == 2
    problem = interaction.report.call_args_list[-1].args[0]
    assert problem.code == "permission.reset_failed"
    assert "Permission denied while resetting" in problem.detail
    assert "synthetic-secret" not in problem.detail
    assert path.with_name(path.name + ".reset-pending").exists() is (when == "after")
    if when == "after":
        assert next(path.parent.glob(f"{path.name}.*.bak")).read_text() == "version: 3\n"
    with pytest.raises(PermissionNamespaceError):
        manager.require_policy_ready()


@pytest.mark.parametrize("scope", ["workspace", "user"])
@pytest.mark.parametrize("choice", ["retry", "exit"])
def test_policy_reset_failure_requires_fresh_retry_or_halt(tmp_path, monkeypatch, scope, choice):
    """A reset failure cannot silently retry or continue without another decision."""
    from loop.utils import ShutdownRequested

    path = tmp_path / (".loop/permissions.yaml" if scope == "workspace" else "user.yaml")
    path.parent.mkdir(exist_ok=True)
    path.write_text("version: 3\n", encoding="utf-8")
    interaction = Mock(spec=Interaction)
    interaction.prompt.side_effect = ["reset", choice, "exit"]
    monkeypatch.setattr(
        PermissionManager,
        "reset_configuration",
        Mock(side_effect=OSError("Permission policy changed; reload before retrying.")),
    )
    with pytest.raises(ShutdownRequested):
        PermissionManager(
            tmp_path,
            user_configuration_path=path if scope == "user" else None,
            interaction=interaction,
        )
    assert interaction.prompt.call_count == (3 if choice == "retry" else 2)
    reports = [item.args[0] for item in interaction.report.call_args_list]
    assert any(
        "Permission policy changed; reload before retrying." in item.detail for item in reports
    )
    assert path.read_text() == "version: 3\n"


def test_policy_reset_does_not_convert_egress_transition_failure_into_recovery(
    tmp_path, monkeypatch
):
    """An uncertain egress transition remains a hard failure during policy recovery."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    path = manager.configuration_path
    path.parent.mkdir(exist_ok=True)
    path.write_text("version: 3\n", encoding="utf-8")
    lease = Mock()
    lease.pause.return_value = True
    lease.resume.return_value = True
    lease.revoke_revision.return_value = True

    def select_reset(*_args, **_kwargs):
        """Fail the policy lock transition after the initial recovery prompt."""
        manager.bind_egress(manager.egress_policy_revision(), lease)
        lease.pause.return_value = False
        return "reset"

    interaction.prompt.side_effect = select_reset
    with pytest.raises(OSError, match="egress admission pause is uncertain"):
        manager.reload(interaction=interaction)
    assert interaction.prompt.call_count == 1
    assert all(
        item.args[0].code != "permission.reset_failed" for item in interaction.report.call_args_list
    )
    lease.pause.return_value = True
    manager.unbind_egress(lease)
    manager.close()


@pytest.mark.parametrize(
    "code,expected",
    [
        (errno.EACCES, "Permission denied"),
        (errno.EMFILE, f"errno {errno.EMFILE}"),
        (errno.ENOENT, "changed during loading"),
        (errno.ELOOP, "symbolic links"),
        (None, "errno unavailable"),
    ],
)
def test_namespaced_policy_open_failure_is_reported_and_blocked(
    tmp_path, monkeypatch, code, expected
):
    """Namespaced policy open errors identify their safe cause and deny admission."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    path = arguments["configuration_path"]
    original = os.open

    def refuse(candidate, flags, *args, **kwargs):
        """Deny only the selected policy read after its provenance check."""
        if candidate == path and flags & os.O_NOFOLLOW:
            raise OSError(code, "synthetic-secret")
        return original(candidate, flags, *args, **kwargs)

    monkeypatch.setattr(manager_module.os, "open", refuse)
    manager.reload()
    with pytest.raises(PermissionNamespaceError, match=expected) as raised:
        manager.require_policy_ready()
    assert "synthetic-secret" not in str(raised.value)


@pytest.mark.parametrize("operation", ["fstat", "read"])
def test_namespaced_policy_read_failure_requires_recovery(tmp_path, monkeypatch, operation):
    """Descriptor inspection and content I/O failures preserve the fail-closed namespace."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    path = arguments["configuration_path"]
    original_open, original_fstat, original_fdopen = os.open, os.fstat, os.fdopen
    selected = set()

    def observe_open(candidate, flags, *args, **kwargs):
        """Record the actual descriptor for the selected policy."""
        descriptor = original_open(candidate, flags, *args, **kwargs)
        if candidate == path and flags & os.O_NOFOLLOW:
            selected.add(descriptor)
        return descriptor

    def inspect(descriptor):
        """Fail inspection only for the selected policy descriptor."""
        if descriptor in selected:
            raise OSError(5, "synthetic-secret")
        return original_fstat(descriptor)

    def open_stream(descriptor, *args, **kwargs):
        """Return a failing reader while preserving descriptor cleanup."""
        stream = original_fdopen(descriptor, *args, **kwargs)
        if descriptor not in selected:
            return stream

        class FailedReader:
            """Close the owned stream and simulate a content read error."""

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                stream.close()

            def read(self):
                """Refuse the selected policy read."""
                raise OSError(5, "synthetic-secret")

        return FailedReader()

    monkeypatch.setattr(manager_module.os, "open", observe_open)
    if operation == "fstat":
        monkeypatch.setattr(manager_module.os, "fstat", inspect)
    else:
        monkeypatch.setattr(manager_module.os, "fdopen", open_stream)
    manager.reload()
    with pytest.raises(PermissionNamespaceError, match="errno 5") as raised:
        manager.require_policy_ready()
    assert "synthetic-secret" not in str(raised.value)


def test_user_policy_recovery_preserves_newer_revision(tmp_path, monkeypatch):
    """A newer successful reload during a failed-reset prompt cannot be overwritten."""
    path = tmp_path / "user.yaml"
    path.write_text("version: 2\n", encoding="utf-8")
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, user_configuration_path=path, interaction=interaction)
    path.write_text("version: 3\n", encoding="utf-8")
    monkeypatch.setattr(manager, "reset_configuration", Mock(side_effect=OSError(5, "refused")))
    selections = iter(["reset", "continue"])

    def prompt(*_args, **_kwargs):
        """Publish and load a repaired policy before the blocked continuation is chosen."""
        choice = next(selections)
        if choice == "continue":
            path.write_text("version: 2\n", encoding="utf-8")
            manager.reload()
        return choice

    interaction.prompt.side_effect = prompt
    manager.reload(interaction=interaction)
    manager.require_policy_ready()
    assert path.read_text() == "version: 2\n"


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_boolean_version_does_not_trigger_policy_migration(tmp_path, scope):
    """A boolean version cannot be treated as version one or cause policy writes."""
    arguments, workspace_id, user_identity = durable_policy_inputs(tmp_path)
    manager = PermissionManager(**arguments)
    manager.initialize_policy_namespace("workspace", workspace_id, authorized=True)
    manager.initialize_policy_namespace("user", user_identity, authorized=True)
    path = arguments["configuration_path" if scope == "workspace" else "user_configuration_path"]
    payload = load_yaml(path.read_bytes())
    payload["version"] = True
    original = dump_yaml(payload).encode()
    path.write_bytes(original)
    manager.reload()
    with pytest.raises(PermissionNamespaceError, match="version"):
        manager.require_policy_ready()
    assert path.read_bytes() == original
    assert not tuple(path.parent.glob(f"{path.name}.*.schema-v1.bak"))


@pytest.mark.parametrize(
    "error, expected",
    [
        (OSError("Unsafe policy lock file."), "Unsafe policy lock file."),
        (OSError("Unsafe policy staging file."), "Unsafe policy staging file."),
        (OSError(5, "synthetic-secret"), "errno 5"),
        (OSError("synthetic-secret"), "errno unavailable"),
    ],
)
def test_policy_reset_failure_reports_safe_specific_cause(tmp_path, monkeypatch, error, expected):
    """Reset failures explain trusted or operating-system causes without exception contents."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text("version: 3\n", encoding="utf-8")
    interaction = Mock(spec=Interaction)
    interaction.prompt.side_effect = ["reset", "continue"]
    monkeypatch.setattr(PermissionManager, "reset_configuration", Mock(side_effect=error))
    manager = PermissionManager(tmp_path, interaction=interaction)
    problem = interaction.report.call_args.args[0]
    assert expected in problem.detail
    assert "synthetic-secret" not in problem.detail
    with pytest.raises(PermissionNamespaceError):
        manager.require_policy_ready()


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_policy_stat_denial_never_discards_existing_denials(tmp_path, monkeypatch, scope):
    """A metadata lookup failure cannot make an existing policy disappear."""
    path = tmp_path / (".loop/permissions.yaml" if scope == "workspace" else "user.yaml")
    path.parent.mkdir(exist_ok=True)
    if scope == "workspace":
        path.write_text("version: 2\ndefaults:\n  filesystem.read: deny\n", encoding="utf-8")
    else:
        path.write_text(
            "version: 2\nrules:\n  - id: deny-read\n    decision: deny\n    action: filesystem.read\n",
            encoding="utf-8",
        )
    interaction = Mock(spec=Interaction)
    original = os.stat

    def refuse_stat(candidate, *args, **kwargs):
        """Deny metadata access only for the selected policy."""
        if os.fspath(candidate) == str(path):
            raise PermissionError(13, "denied", str(path))
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(manager_module.os, "stat", refuse_stat)
    manager = PermissionManager(
        tmp_path, user_configuration_path=path if scope == "user" else None, interaction=interaction
    )
    if scope == "workspace":
        assert manager.configuration.defaults[Action.FILESYSTEM_READ] is Decision.DENY
    else:
        assert [rule.id for rule in manager.user_rules] == ["deny-read"]
    interaction.prompt.assert_not_called()


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_unreadable_policy_prompts_without_permitting_reset(tmp_path, monkeypatch, scope):
    """An unreadable policy requires explicit recovery and retains fail-closed admission."""
    path = tmp_path / (".loop/permissions.yaml" if scope == "workspace" else "user.yaml")
    path.parent.mkdir(exist_ok=True)
    path.write_text("version: 2\n", encoding="utf-8")
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "continue"
    original = Path.read_bytes

    def refuse_read(candidate):
        """Deny the selected policy read without exposing exception text."""
        if candidate == path:
            raise PermissionError(13, "synthetic-secret", str(path))
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", refuse_read)
    manager = PermissionManager(
        tmp_path, user_configuration_path=path if scope == "user" else None, interaction=interaction
    )
    assert "reset" not in interaction.prompt.call_args.kwargs["choices"]
    detail = interaction.report.call_args.args[0].detail
    assert "Permission denied" in detail
    assert "synthetic-secret" not in detail
    with pytest.raises(PermissionNamespaceError):
        manager.require_policy_ready()


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_dangling_policy_symlink_cannot_load_default_permissions(tmp_path, scope):
    """A dangling policy link is not mistaken for a first-use missing file."""
    path = tmp_path / (".loop/permissions.yaml" if scope == "workspace" else "user.yaml")
    path.parent.mkdir(exist_ok=True)
    path.symlink_to(tmp_path / "missing-policy.yaml")
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "continue"
    manager = PermissionManager(
        tmp_path, user_configuration_path=path if scope == "user" else None, interaction=interaction
    )
    assert "reset" not in interaction.prompt.call_args.kwargs["choices"]
    with pytest.raises(PermissionNamespaceError):
        manager.require_policy_ready()
    assert path.is_symlink()


@pytest.mark.parametrize("scope", ["workspace", "user"])
@pytest.mark.parametrize("choice", ["retry", "continue"])
def test_unreadable_policy_reload_preserves_denials(tmp_path, monkeypatch, scope, choice):
    """A read failure requires a decision and cannot discard an already loaded denial."""
    path = tmp_path / (".loop/permissions.yaml" if scope == "workspace" else "user.yaml")
    path.parent.mkdir(exist_ok=True)
    if scope == "workspace":
        path.write_text("version: 2\ndefaults:\n  filesystem.read: deny\n", encoding="utf-8")
    else:
        path.write_text(
            "version: 2\nrules:\n  - id: deny-read\n    decision: deny\n    action: filesystem.read\n",
            encoding="utf-8",
        )
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(
        tmp_path, user_configuration_path=path if scope == "user" else None, interaction=interaction
    )
    original = Path.read_bytes
    denied = True

    def refuse_read(candidate):
        """Refuse reads only until the user explicitly selects a retry."""
        if candidate == path and denied:
            raise PermissionError(13, "synthetic-secret", str(path))
        return original(candidate)

    def decide(*_args, **_kwargs):
        """Repair the read boundary only when retrying."""
        nonlocal denied
        if choice == "retry":
            denied = False
        return choice

    monkeypatch.setattr(Path, "read_bytes", refuse_read)
    interaction.prompt.side_effect = decide
    result = manager.reload()
    assert "reset" not in interaction.prompt.call_args.kwargs["choices"]
    if scope == "workspace":
        assert manager.configuration.defaults[Action.FILESYSTEM_READ] is Decision.DENY
    else:
        assert [rule.id for rule in manager.user_rules] == ["deny-read"]
    if choice == "continue":
        assert result is PermissionLoadResult.BLOCKED
        with pytest.raises(PermissionNamespaceError):
            manager.require_policy_ready()
    else:
        manager.require_policy_ready()


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_broken_policy_directory_symlink_requires_recovery(tmp_path, scope):
    """A broken policy ancestor cannot be treated as an absent first-use policy."""
    directory = tmp_path / "broken"
    directory.symlink_to(tmp_path / "missing-directory", target_is_directory=True)
    path = directory / "permissions.yaml"
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "continue"
    arguments = (
        {"configuration_path": path} if scope == "workspace" else {"user_configuration_path": path}
    )
    manager = PermissionManager(tmp_path, interaction=interaction, **arguments)
    assert "reset" not in interaction.prompt.call_args.kwargs["choices"]
    with pytest.raises(PermissionNamespaceError):
        manager.require_policy_ready()
    assert directory.is_symlink()


@pytest.mark.parametrize("quote", ["", '"'])
def test_policy_accepts_natural_boolean_strings(tmp_path, quote):
    """Natural boolean strings resolve to their intended permission limits."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text(
        f"version: 2\nlimits:\n  deny_private_networks: {quote}yes{quote}\n"
        f"  allow_host_processes: {quote}no{quote}\n",
        encoding="utf-8",
    )
    manager = PermissionManager(tmp_path)
    assert manager.configuration.limits.deny_private_networks is True
    assert manager.configuration.limits.allow_host_processes is False


def test_policy_accepts_explicit_yaml_11(tmp_path):
    """A policy declaring YAML 1.1 keeps its declared scalar interpretation."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text(
        "%YAML 1.1\n---\nversion: 2\nlimits:\n  deny_private_networks: yes\n",
        encoding="utf-8",
    )
    assert PermissionManager(tmp_path).configuration.limits.deny_private_networks is True
