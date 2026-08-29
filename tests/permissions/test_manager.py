"""Tests for layered operation-policy evaluation, approval, and persistence."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

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
    PolicyLimits,
    PolicyScope,
    ProcessBoundary,
    ProcessTarget,
    SessionTarget,
)
from loop.permissions import PermissionLoadFailure
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry
from loop.telemetry.policy import thaw


def operation(action: Action, *, tool: str = "demo", target=None, reason=None) -> Operation:
    """Build one representative typed operation."""
    return Operation(tool_id=tool, action=action, target=target, reason=reason)


def file_operation(action: Action, path) -> Operation:
    """Build one filesystem operation for a canonical path."""
    return operation(action, target=FileTarget(path=str(path)))


@pytest.mark.parametrize(
    ("workspace_available", "selection", "expected", "workspace_label"),
    [
        (True, ApprovalChoice.WORKSPACE, ApprovalChoice.WORKSPACE, True),
        (False, ApprovalChoice.SESSION, ApprovalChoice.SESSION, False),
        (True, False, ApprovalChoice.DENY, True),
    ],
)
def test_request_permission_offers_valid_scopes_and_fails_closed(
    workspace_available,
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
        tmp_path if workspace_available else None, interaction=interaction, recorder=recorder
    )

    result = manager.request_permission(
        "Approve operations?",
        interaction=interaction,
    )

    assert result is expected
    choices = interaction.prompt.call_args.kwargs["choices"]
    assert (ApprovalChoice.WORKSPACE in choices) is workspace_label


def test_request_permission_forwards_index_map_to_prompt(tmp_path):
    """Permission prompts forward short letter indexes to the generic prompt layer."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(tmp_path, interaction=interaction)

    manager.request_permission("Approve?")

    index = interaction.prompt.call_args.kwargs["index"]
    assert index is not None
    assert index[ApprovalChoice.DENY] == "N"
    assert index[ApprovalChoice.ONCE] == "Y"
    assert index[ApprovalChoice.SESSION] == "S"


def test_request_permission_includes_workspace_index_when_configured(tmp_path):
    """Workspace scope adds a ``W`` index letter when a configuration path exists."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.ONCE
    manager = PermissionManager(tmp_path, interaction=interaction)

    manager.request_permission("Approve?")

    index = interaction.prompt.call_args.kwargs["index"]
    assert index[ApprovalChoice.WORKSPACE] == "W"


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
    audit = tmp_path / ".loop" / "permissions-audit.jsonl"
    assert json.loads(audit.read_text("utf-8").splitlines()[-1])["source"] == "headless"


def test_permission_audit_rotates_valid_private_jsonl_archives(tmp_path, monkeypatch):
    """Permission audit storage retains bounded, complete, private JSONL archives."""
    monkeypatch.setattr("loop.constants.DEFAULT_PERMISSIONS_AUDIT_BYTES", 1)
    monkeypatch.setattr("loop.constants.DEFAULT_PERMISSIONS_AUDIT_BACKUPS", 2)
    manager = PermissionManager(tmp_path)

    for tool_id in ("first", "second", "third", "fourth"):
        manager.authorize((file_operation(Action.FILESYSTEM_READ, tmp_path / f"{tool_id}.txt"),))

    audit_path = tmp_path / ".loop" / "permissions-audit.jsonl"
    audit_paths = (
        audit_path,
        *(audit_path.with_name(f"{audit_path.name}.{index}") for index in (1, 2)),
    )
    assert all(path.exists() for path in audit_paths)
    assert not audit_path.with_name(f"{audit_path.name}.3").exists()
    assert all(json.loads(path.read_text("utf-8")) for path in audit_paths)
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in audit_paths)
    assert (audit_path.parent.stat().st_mode & 0o777) == 0o700


def test_permission_audit_rotation_failure_does_not_change_authorization(tmp_path, monkeypatch):
    """A failed audit archive rotation cannot turn an allowed operation into a denial."""
    monkeypatch.setattr("loop.constants.DEFAULT_PERMISSIONS_AUDIT_BYTES", 1)
    manager = PermissionManager(tmp_path)
    operation_set = (file_operation(Action.FILESYSTEM_READ, tmp_path / "file.txt"),)
    manager.authorize(operation_set)
    monkeypatch.setattr(Path, "replace", Mock(side_effect=OSError("unavailable")))

    result = manager.authorize(operation_set)

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


def test_session_approval_remembers_exact_batch_targets_and_deduplicates(tmp_path):
    """Session approval installs exact grants once and reuses them without another prompt."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = ApprovalChoice.SESSION
    manager = PermissionManager(tmp_path, interaction=interaction)
    first = file_operation(Action.FILESYSTEM_CREATE, tmp_path / "literal[*].txt")
    second = file_operation(Action.FILESYSTEM_DELETE, tmp_path / "old.txt")

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
    target = file_operation(Action.FILESYSTEM_REPLACE, tmp_path / "module.py")

    approved = manager.authorize((target, target))
    reloaded = PermissionManager(tmp_path)

    assert approved.decision is Decision.ALLOW
    assert approved.approval_choice is ApprovalChoice.WORKSPACE
    assert len(approved.installed_rule_ids) == 1
    assert reloaded.persistent_rules[0].tool_exact is True
    assert "tool_match=exact" in reloaded.describe("workspace")
    assert reloaded.authorize((target,)).decision is Decision.ALLOW
    stored = json.loads(
        (tmp_path / ".loop" / "permissions-audit.jsonl").read_text("utf-8").splitlines()[0]
    )
    assert datetime.fromisoformat(stored["timestamp"]).tzinfo is UTC
    assert stored["audit_schema_version"] == 1
    assert stored["event_name"] == "permission.decided"
    assert stored["approval_choice"] == "workspace"
    assert stored["installed_rule_ids"] == list(approved.installed_rule_ids)


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

    record = json.loads(
        (tmp_path / ".loop" / "permissions-audit.jsonl").read_text("utf-8").splitlines()[0]
    )
    assert record["event_name"] == "permission.default_set"
    assert datetime.fromisoformat(record["timestamp"]).tzinfo is UTC
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
    monkeypatch.setattr("pathlib.Path.write_text", Mock(side_effect=OSError("disk full")))

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

    assert denied.evaluate((operation(Action.PROCESS_EXECUTE, target=target),)).sources == (
        "limit:workspace:allow_host_processes",
    )
    assert (
        allowed.evaluate((operation(Action.PROCESS_EXECUTE, target=target),)).decision
        is Decision.ALLOW
    )


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
    monkeypatch.setattr("pathlib.Path.write_text", Mock(side_effect=OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)

    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.ASK


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


def test_empty_yaml_loads_as_default_policy(tmp_path):
    """An empty policy file is accepted as the supervised default."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text("", "utf-8")

    assert PermissionManager(tmp_path).configuration == PermissionConfiguration()


def test_error_policy_raises_configuration_errors_and_preserves_active_policy(tmp_path):
    """Strict startup and reload expose typed errors without replacing valid active policy."""
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    path = tmp_path / ".loop" / "permissions.yaml"
    path.write_text("version: 2\n", "utf-8")

    with pytest.raises(PermissionConfigurationError) as raised:
        manager.reload()

    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    assert raised.value.path == str(path)
    assert raised.value.__cause__ is not None
    with pytest.raises(PermissionConfigurationError):
        PermissionManager(tmp_path)


def test_auto_policy_reports_and_uses_defaults_or_last_known_good(tmp_path, caplog):
    """Automatic recovery reports startup defaults and retains valid policy on reload."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text("version: 2\n", "utf-8")

    manager = PermissionManager(tmp_path, load_policy=PermissionLoadPolicy.AUTO)

    assert manager.configuration == PermissionConfiguration()
    assert "automatic permission policy" in caplog.text
    manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    path.write_text("version: 2\n", "utf-8")
    interaction = Mock(spec=Interaction)
    manager.interaction = interaction

    assert manager.reload() is PermissionLoadResult.RETAINED
    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    interaction.report.assert_called_once()
    interaction.warning.assert_called_once()


def test_interactive_policy_retries_a_repaired_file(tmp_path):
    """Interactive recovery rereads a file repaired while the recovery prompt is active."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text("version: 2\n", "utf-8")
    interaction = Mock(spec=Interaction)

    def repair(*_args, **_kwargs):
        path.write_text("version: 1\ndefaults:\n  filesystem.delete: deny\n", "utf-8")
        return "retry"

    interaction.prompt.side_effect = repair
    manager = PermissionManager(tmp_path, interaction=interaction)

    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    interaction.report.assert_called_once()


@pytest.mark.parametrize(
    ("choice", "expected"),
    [("continue", PermissionLoadResult.RETAINED), (None, PermissionLoadResult.RETAINED)],
)
def test_interactive_reload_can_retain_the_active_policy(tmp_path, choice, expected):
    """Interactive reload retains last-known-good state when the user continues."""
    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    manager.configuration_path.write_text("version: 2\n", "utf-8")
    interaction.prompt.return_value = choice

    assert manager.reload() is expected
    assert manager.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    interaction.warning.assert_called_once_with("Keeping the current permission policy.")


def test_interactive_loading_requires_an_interaction(tmp_path):
    """Interactive loading cannot silently start or recover without an interaction."""
    with pytest.raises(ValueError, match="requires an Interaction"):
        PermissionManager(load_policy=PermissionLoadPolicy.INTERACTIVE)

    interaction = Mock(spec=Interaction)
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.interaction = None
    manager.configuration_path.parent.mkdir(exist_ok=True)
    manager.configuration_path.write_text("version: 2\n", "utf-8")
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

    manager = PermissionManager(load_policy=PermissionLoadPolicy.AUTO)
    assert not manager.presets
    assert "Excluded invalid permission preset" in caplog.text


def test_preset_error_requires_a_failure():
    """The aggregate preset error rejects an empty diagnostic collection."""
    with pytest.raises(ValueError, match="at least one failure"):
        PermissionPresetError(())


def test_reset_configuration_creates_defaults_when_no_policy_file_exists(tmp_path):
    """An explicit reset creates the default policy even before a policy file exists."""
    manager = PermissionManager(tmp_path)

    assert manager.reset_configuration() is None
    assert (tmp_path / ".loop" / "permissions.yaml").exists()


def test_in_memory_policy_cannot_be_reset():
    """Reset requires a local policy path to preserve its archival contract."""
    with pytest.raises(ValueError, match="cannot reset"):
        PermissionManager().reset_configuration()


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
