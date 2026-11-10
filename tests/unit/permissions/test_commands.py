"""Tests for complete operation-policy user commands."""

import shlex
import time
from unittest.mock import Mock

import pytest
from prompt_toolkit.document import Document

from loop import (
    Action,
    ApprovalChoice,
    CommandCapability,
    CommandCompletionAdapter,
    CommandManager,
    CompletionManager,
    CompletionProviderRegistration,
    CompletionValue,
    Decision,
    FileTarget,
    Interaction,
    Operation,
    PermissionConfiguration,
    PermissionManager,
    PermissionPreset,
    PermissionRule,
    PolicyScope,
)
from loop.execution.sandbox import SandboxRequest
from loop.interaction import ConsoleInteraction
from loop.permissions import PermissionCommands
from loop.workspace import Workspace, WorkspaceRepository


def test_permissions_sandbox_rules_can_be_listed_and_removed(tmp_path):
    """The permission command exposes a revocation path for native approvals."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt.return_value = "session"
    permissions = PermissionManager(tmp_path, interaction=interaction)
    command = command_manager(permissions, interaction)
    approved = SandboxRequest.create(
        source="git status",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(tmp_path.parent,),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin"},
        policy_version="macos-seatbelt-v1",
        deadline=time.monotonic() + 60,
        workspace_id="test",
    )
    assert permissions.authorize_sandboxed_command(approved, tool_id="run_command")
    rule = permissions.sandboxed_command_rules(ApprovalChoice.SESSION)[0]
    command.call("permissions", "sandbox")
    assert rule.id in interaction.info.call_args.args[0]
    command.call("permissions", f"sandbox remove session {rule.id}")
    assert not permissions.sandboxed_command_rules(ApprovalChoice.SESSION)
    command.call("permissions", f"sandbox remove session {rule.id}")
    assert "No matching" in interaction.info.call_args.args[0]
    command.call("permissions", "sandbox unknown")
    interaction.report.assert_called()


def test_permissions_host_rules_can_be_listed_and_revoked(tmp_path):
    """A recorded host choice has a user-visible session revocation command."""
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path, interaction=interaction)
    command = command_manager(permissions, interaction)
    command.call("permissions", "host list")
    assert "none" in interaction.info.call_args.args[0]
    approved = SandboxRequest.create(
        source="printf host",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin"},
        policy_version="macos-seatbelt-v1",
        deadline=time.monotonic() + 60,
        workspace_id="test",
    )
    rule = permissions.remember_host_command_rule(approved)
    command.call("permissions", "host list")
    assert rule.id in interaction.info.call_args.args[0]
    command.call("permissions", f"host remove {rule.id}")
    assert not permissions.host_command_rules()
    command.call("permissions", f"host remove {rule.id}")
    assert "No matching" in interaction.info.call_args.args[0]
    command.call("permissions", "host unknown")
    interaction.report.assert_called()


def test_permissions_trusted_host_grant_is_confirmed_listed_and_revoked(tmp_path):
    """Only the trusted human command installs an exact session no-prompt host grant."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.supports_fresh_host_approval = True
    interaction.prompt.return_value = "deny"
    permissions = PermissionManager(tmp_path)
    command = command_manager(permissions, interaction)

    command.call("permissions", "host trust 'printf host'")
    assert not permissions.trusted_host_command_rules()
    interaction.prompt.return_value = "trust"
    command.call("permissions", "host trust 'printf host'")
    rule = permissions.trusted_host_command_rules()[0]
    assert "host user's authority" in interaction.info.call_args_list[-2].args[0]
    command.call("permissions", "host trusted")
    assert rule.id in interaction.info.call_args.args[0]
    command.call("permissions", f"host untrust {rule.id}")
    assert not permissions.trusted_host_command_rules()
    command.call("permissions", f"host untrust {rule.id}")
    assert "No matching" in interaction.info.call_args.args[0]
    permissions.close()


def test_yolo_human_host_trust_command_needs_no_second_confirmation(tmp_path):
    """An explicit trusted human grant command remains usable without a second Loop prompt."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.supports_fresh_host_approval = True
    permissions = PermissionManager(tmp_path, yolo=True)
    command = command_manager(permissions, interaction)

    command.call("permissions", "host trust 'printf host'")

    assert len(permissions.trusted_host_command_rules()) == 1
    interaction.prompt.assert_not_called()
    permissions.close()


@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_explicit_policy_reset_command_uses_no_confirmation(tmp_path, scope):
    """A human policy reset archives the selected scope without a second prompt."""
    interaction = Mock(spec=ConsoleInteraction)
    user_path = tmp_path / "user-policy.yaml"
    permissions = PermissionManager(tmp_path, user_configuration_path=user_path, yolo=True)
    command = command_manager(permissions, interaction)
    if scope == "workspace":
        permissions.set_default(Action.FILESYSTEM_READ, Decision.DENY)

    command.call("permissions", f"reset {scope}")

    assert "prior policy archive:" in interaction.info.call_args.args[0]
    assert (permissions.configuration_path if scope == "workspace" else user_path).exists()
    interaction.prompt.assert_not_called()
    permissions.close()


def test_user_reset_command_recovers_invalid_policy_with_an_archive(tmp_path):
    """The trusted command can recover captured invalid user policy without a prompt."""
    interaction = Mock(spec=ConsoleInteraction)
    user_path = tmp_path / "user-policy.yaml"
    user_path.write_text("version: invalid\n", encoding="utf-8")
    permissions = PermissionManager(
        tmp_path, user_configuration_path=user_path, interaction=interaction
    )
    command = command_manager(permissions, interaction)

    command.call("permissions", "reset user")

    assert "prior policy archive:" in interaction.info.call_args.args[0]
    assert next(tmp_path.glob("user-policy.yaml.*.bak")).read_text(encoding="utf-8") == (
        "version: invalid\n"
    )
    interaction.prompt.assert_not_called()
    permissions.close()


def test_bare_reset_clears_only_session_policy(tmp_path):
    """Bare reset clears session overrides without changing the persisted workspace policy."""
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path)
    command = command_manager(permissions, interaction)
    permissions.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    permissions.set_default(Action.FILESYSTEM_READ, Decision.DENY, scope=PolicyScope.SESSION)
    workspace_bytes = permissions.configuration_path.read_bytes()

    command.call("permissions", "reset")

    assert not permissions.session_overrides.defaults
    assert permissions.configuration_path.read_bytes() == workspace_bytes
    interaction.info.assert_called_with("Cleared every session permission override.")
    permissions.close()


def test_permissions_trusted_host_grant_refuses_automatic_input(tmp_path):
    """Automatic command input cannot create future no-prompt host authority."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.supports_fresh_host_approval = False
    permissions = PermissionManager(tmp_path)
    command = command_manager(permissions, interaction)

    command.call("permissions", "host trust 'printf host'")

    assert not permissions.trusted_host_command_rules()
    interaction.prompt.assert_not_called()
    interaction.report.assert_called_once()
    assert "fresh human host input" in interaction.report.call_args.args[0].detail
    permissions.close()


@pytest.mark.parametrize("scope", (ApprovalChoice.WORKSPACE, ApprovalChoice.USER))
def test_permissions_host_trust_command_manages_durable_scopes(tmp_path, scope):
    """The trusted command exposes and revokes each durable host grant scope."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.supports_fresh_host_approval = True
    interaction.prompt.return_value = "trust"
    catalog = tmp_path.parent / f"{tmp_path.name}-policy-data" / "workspaces.sqlite"
    workspace = WorkspaceRepository(catalog).initialize(Workspace.discover(tmp_path))
    permissions = PermissionManager(
        tmp_path,
        configuration_path=catalog.parent / "workspaces" / workspace.id / "permissions.yaml",
        user_configuration_path=catalog.parent / "user-permissions.yaml",
        workspace_id=workspace.id,
        namespace_catalog_path=catalog,
        user_namespace_identity=str(catalog.parent),
    )
    permissions.initialize_policy_namespace("workspace", workspace.id, authorized=True)
    permissions.initialize_policy_namespace("user", str(catalog.parent), authorized=True)
    command = command_manager(permissions, interaction)

    command.call("permissions", f"host trust {scope.value} 'printf host'")
    rule = permissions.trusted_host_command_rules()[0]
    assert rule.scope is scope
    command.call("permissions", "host trusted")
    assert f"{scope.value} {rule.id}" in interaction.info.call_args.args[0]
    command.call("permissions", f"host untrust {scope.value} {rule.id}")
    assert permissions.trusted_host_command_rules() == ()
    permissions.close()


def test_permissions_sandbox_cli_preserves_other_persisted_scope(tmp_path):
    """CLI revocation of a workspace approval leaves a user approval visible."""
    interaction = Mock(spec=ConsoleInteraction)
    user_path = tmp_path / "user.yaml"
    permissions = PermissionManager(
        tmp_path, user_configuration_path=user_path, interaction=interaction
    )
    command = command_manager(permissions, interaction)
    request = SandboxRequest.create(
        source="printf workspace",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(tmp_path.parent,),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin"},
        policy_version="macos-seatbelt-v1",
        deadline=time.monotonic() + 60,
        workspace_id="test",
    )
    interaction.prompt.return_value = "workspace"
    assert permissions.authorize_sandboxed_command(request, tool_id="run_command")
    interaction.prompt.return_value = "user"
    assert permissions.authorize_sandboxed_command(
        SandboxRequest.create(
            source="printf user",
            cwd=tmp_path,
            workspace=tmp_path,
            read_roots=(tmp_path.parent,),
            write_roots=(tmp_path,),
            network=False,
            environment={"PATH": "/usr/bin:/bin"},
            policy_version="macos-seatbelt-v1",
            deadline=time.monotonic() + 60,
            workspace_id="test",
        ),
        tool_id="run_command",
    )
    workspace_rule = permissions.sandboxed_command_rules(ApprovalChoice.WORKSPACE)[0]
    user_rule = permissions.sandboxed_command_rules(ApprovalChoice.USER)[0]
    command.call("permissions", f"sandbox remove workspace {workspace_rule.id}")
    command.call("permissions", "sandbox list")
    assert user_rule.id in interaction.info.call_args.args[0]
    assert workspace_rule.id not in interaction.info.call_args.args[0]


def command_manager(permissions, interaction):
    """Return a command manager exposing the supplied permission policy."""
    manager = CommandManager(interaction=interaction)
    manager.register_provider(PermissionCommands(permissions))
    return manager


def test_permission_policy_changes_apply_immediately_during_runs(tmp_path):
    """Safe policy edits serialize with checks and affect the next authorization decision."""
    defaults = PermissionConfiguration().defaults
    policy = PermissionConfiguration(defaults={**defaults, Action.FILESYSTEM_READ: Decision.ALLOW})
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path, configuration=policy)
    provider = PermissionCommands(permissions)
    manager = command_manager(permissions, interaction)
    manager.set_capability_provider(
        lambda: frozenset(
            {
                CommandCapability.CONTEXT_READ,
                CommandCapability.CONTEXT_UPDATE,
                CommandCapability.CONVERSATION_ACTIVE,
                CommandCapability.RUN_ACTIVE,
            }
        )
    )
    requested = Operation(
        tool_id="read_text_file",
        action=Action.FILESYSTEM_READ,
        target=FileTarget(path=str(tmp_path / "doc.txt")),
    )
    assert permissions.evaluate((requested,)).decision is Decision.ALLOW

    assert manager.handle_user_command("/permissions default set session filesystem.read deny")
    assert permissions.evaluate((requested,)).decision is Decision.DENY
    interaction.prompt.assert_not_called()

    tool_completions = Mock()
    tool_completions.get_completion_providers.return_value = (
        CompletionProviderRegistration(
            "tools", lambda: (CompletionValue("read_text_file", "tool"),)
        ),
    )
    completer = CompletionManager(
        (
            CommandCompletionAdapter(
                lambda: manager.active_commands,
                providers=(provider, tool_completions),
                capabilities=lambda: manager.capabilities,
            ),
        )
    )
    actions = {item.text for item in complete(completer, "/permissions ")}
    assert "reload" not in actions
    assert "reset" not in actions
    assert {"default", "limit", "rule", "show"} <= actions
    preset_actions = {item.text for item in complete(completer, "/permissions preset ")}
    assert preset_actions == {"list", "show", "diff"}
    rule_actions = {
        item.text for item in complete(completer, "/permissions rule add session allow ")
    }
    assert rule_actions == {"read_text_file"}


def test_permission_commands_reject_prompting_changes_during_runs(tmp_path):
    """Broad rules and policy reload or replacement cannot enter synchronous prompts mid-run."""
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path, interaction=interaction)
    manager = command_manager(permissions, interaction)
    manager.set_capability_provider(
        lambda: frozenset(
            {
                CommandCapability.CONTEXT_READ,
                CommandCapability.CONTEXT_UPDATE,
                CommandCapability.RUN_ACTIVE,
            }
        )
    )

    for command in (
        "/permissions rule add session allow",
        "/permissions rule add",
        "/permissions reload",
        "/permissions reset",
        "/permissions reset workspace",
        "/permissions preset replace session basic",
    ):
        assert manager.handle_user_command(command)
        assert "cannot" in interaction.report.call_args.args[0].detail
    interaction.confirm.assert_not_called()

    assert manager.handle_user_command(
        "/permissions rule add session allow read_text_file filesystem.read /docs"
    )
    assert permissions.session_rules[-1].resource == "/docs"

    for command in (
        "/permissions rule list",
        "/permissions sandbox list",
        "/permissions host list",
        "/permissions preset list",
    ):
        assert manager.handle_user_command(command)
    interaction.prompt.assert_not_called()


@pytest.mark.parametrize(
    ("command", "heading"),
    [
        ("/permissions sandbox", "Sandboxed command approvals:"),
        ("/permissions host", "Session host command approvals:"),
    ],
)
def test_bare_approval_listings_are_available_during_runs(tmp_path, command, heading):
    """Bare sandbox and host commands list approvals without prompting during a run."""
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path, interaction=interaction)
    manager = command_manager(permissions, interaction)
    manager.set_capability_provider(
        lambda: frozenset(
            {
                CommandCapability.CONTEXT_READ,
                CommandCapability.CONTEXT_UPDATE,
                CommandCapability.RUN_ACTIVE,
            }
        )
    )

    assert manager.handle_user_command(command)

    interaction.info.assert_called_once()
    assert interaction.info.call_args.args[0].startswith(heading)
    interaction.report.assert_not_called()
    interaction.prompt.assert_not_called()


def test_permission_change_during_approval_prompt_overrides_stale_approval(tmp_path):
    """A policy tightened while approval waits is rechecked before the operation is authorized."""
    defaults = PermissionConfiguration().defaults
    policy = PermissionConfiguration(defaults={**defaults, Action.FILESYSTEM_READ: Decision.ASK})
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path, interaction=interaction, configuration=policy)
    manager = command_manager(permissions, interaction)
    manager.set_capability_provider(
        lambda: frozenset(
            {
                CommandCapability.CONTEXT_READ,
                CommandCapability.CONTEXT_UPDATE,
                CommandCapability.RUN_ACTIVE,
            }
        )
    )
    interaction.prompt.side_effect = lambda *_args, **_kwargs: (
        manager.handle_user_command("/permissions default set session filesystem.read deny")
        and ApprovalChoice.SESSION
    )
    requested = Operation(
        tool_id="read_text_file",
        action=Action.FILESYSTEM_READ,
        target=FileTarget(path=str(tmp_path / "doc.txt")),
    )

    result = permissions.authorize((requested,))

    assert result.decision is Decision.DENY
    assert result.policy.decision is Decision.DENY
    assert result.source == "policy"
    assert "changed to deny" in result.reason
    assert not permissions.session_rules


def test_permissions_command_manages_defaults_limits_and_complete_rule_lifetimes(tmp_path):
    """Commands persist every policy dimension and manage process-local rules separately."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm.return_value = True
    permissions = PermissionManager(tmp_path)
    manager = command_manager(permissions, interaction)

    manager.call("permissions", "default set workspace network.request deny")
    manager.call("permissions", "limit set workspace host-process allow")
    manager.call("permissions", "limit set workspace private-network allow")
    manager.call("permissions", "limit add workspace write-root shared")
    manager.call("permissions", "limit add workspace network-origin https://my-host.local")
    manager.call(
        "permissions",
        "rule add workspace allow read_text_file filesystem.read '/project/*' 'Read docs'",
    )
    persistent_id = permissions.persistent_rules[0].id
    manager.call("permissions", "rule add session deny run_command process.execute")
    session_id = permissions.session_rules[0].id

    loaded = PermissionManager(tmp_path)
    assert loaded.configuration.defaults[Action.NETWORK_REQUEST] is Decision.DENY
    assert loaded.configuration.limits.allow_host_processes is True
    assert loaded.configuration.limits.deny_private_networks is False
    assert loaded.configuration.limits.network_origins == ("https://my-host.local",)
    assert loaded.configuration.rules[0].description == "Read docs"
    assert permissions.session_rules[0].resource is None

    manager.call("permissions", f"rule remove workspace {persistent_id}")
    manager.call("permissions", f"rule remove session {session_id}")
    assert not permissions.persistent_rules
    assert not permissions.session_rules


def test_permissions_rule_add_defaults_omitted_matchers_to_wildcards(tmp_path):
    """Omitted trailing tool, action, and resource matchers each mean wildcard."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm.return_value = True
    permissions = PermissionManager(tmp_path)
    manager = command_manager(permissions, interaction)

    variants = (
        ("rule add session allow", "*", None, None),
        ("rule add session allow run_command", "run_command", None, None),
        (
            "rule add session allow run_command process.execute",
            "run_command",
            Action.PROCESS_EXECUTE,
            None,
        ),
        (
            "rule add session allow run_command process.execute 'git *'",
            "run_command",
            Action.PROCESS_EXECUTE,
            "git *",
        ),
    )
    for command, tool, action, resource in variants:
        manager.call("permissions", command)
        rule = permissions.session_rules[-1]
        assert (rule.tool, rule.action, rule.resource) == (tool, action, resource)

    interaction.report.assert_not_called()


@pytest.mark.parametrize("declined", [False, object()])
def test_permissions_rule_add_confirms_broad_matchers_before_mutation(tmp_path, declined):
    """Broad rules require explicit approval and declined creation leaves policy unchanged."""
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path)
    manager = command_manager(permissions, interaction)

    interaction.confirm.return_value = declined
    manager.call("permissions", "rule add session allow run_command process.execute")

    assert not permissions.session_rules
    prompt = interaction.confirm.call_args.args[0]
    assert "session allow rule" in prompt
    assert "every resource" in prompt
    assert interaction.confirm.call_args.kwargs == {"default": False}
    interaction.warning.assert_called_once_with("Permission rule creation was not approved.")

    interaction.confirm.return_value = True
    manager.call("permissions", "rule add session allow run_command process.execute")

    assert len(permissions.session_rules) == 1
    assert permissions.session_rules[0].resource is None

    interaction.confirm.return_value = False
    manager.call("permissions", "rule add session deny * process.execute git")
    prompt = interaction.confirm.call_args.args[0]
    assert "every tool" in prompt
    assert "resource" not in prompt
    assert len(permissions.session_rules) == 1


def test_permissions_rule_without_mutation_lists_scoped_rules(tmp_path):
    """Bare and explicitly scoped rule inspection list rules without changing policy."""
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path)
    permissions.add_rule(PermissionRule(id="workspace-rule", decision=Decision.ASK))
    permissions.add_rule(
        PermissionRule(id="session-rule", decision=Decision.ALLOW, tool="run_command"),
        scope=PolicyScope.SESSION,
    )
    manager = command_manager(permissions, interaction)

    manager.call("permissions", "rule")
    listed = interaction.info.call_args.args[0]
    assert "Workspace permission rules:\n  workspace-rule" in listed
    assert "Session permission rules:\n  session-rule" in listed

    manager.call("permissions", "rule list session")
    scoped = interaction.info.call_args.args[0]
    assert "Session permission rules:\n  session-rule" in scoped
    assert "Workspace permission rules:" not in scoped
    interaction.report.assert_not_called()


def test_permissions_show_and_explain_report_the_complete_effective_policy(tmp_path):
    """Policy output distinguishes defaults, limits, persistent rules, and session rules."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm.return_value = True
    permissions = PermissionManager(tmp_path)
    manager = command_manager(permissions, interaction)
    manager.call("permissions", "rule add workspace allow read_text_file filesystem.read '*'")
    manager.call("permissions", "rule add session deny read_text_file filesystem.read '*'")

    manager.call("permissions", "show")
    shown = interaction.info.call_args.args[0]
    assert "deny_private_networks: True" in shown
    assert "Workspace policy:" in shown
    assert "Session overrides:" in shown
    assert "[workspace]" in shown
    assert "[session]" in shown
    assert "Precedence:" in shown

    manager.call("permissions", "help")
    assert "limit" in interaction.info.call_args.args[0]
    assert "  /permissions reset\n" in interaction.info.call_args.args[0]
    assert "  /permissions reset user\n" in interaction.info.call_args.args[0]

    manager.call("permissions", "explain read_text_file filesystem.read file.txt")
    explained = interaction.info.call_args.args[0]
    assert "Effective decision: deny" in explained
    assert "rule:" in explained


def test_permissions_reload_and_limit_removal_report_outcomes(tmp_path):
    """Reload validates disk policy and unchanged collection operations produce warnings."""
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path)
    manager = command_manager(permissions, interaction)

    manager.call("permissions", "default set workspace filesystem.delete deny")
    permissions.set_default(Action.FILESYSTEM_DELETE, Decision.ALLOW, scope=PolicyScope.SESSION)
    manager.call("permissions", "reload")
    assert permissions.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY

    manager.call("permissions", "limit remove workspace read-root workspace")
    manager.call("permissions", "limit remove workspace read-root workspace")
    assert "already unchanged" in interaction.warning.call_args.args[0]
    manager.call("permissions", "rule remove workspace missing")
    assert "was not found" in interaction.warning.call_args.args[0]


def test_permissions_reload_reports_invalid_policy_without_replacing_active_policy(tmp_path):
    """Reload shows recovery diagnostics while preserving the last valid workspace policy."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt.return_value = "continue"
    permissions = PermissionManager(tmp_path, interaction=interaction)
    permissions.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    (tmp_path / ".loop" / "permissions.yaml").write_text("version: 3\n", "utf-8")
    manager = command_manager(permissions, interaction)

    manager.call("permissions", "reload")

    assert permissions.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    interaction.report.assert_called_once()
    assert "Keeping" in interaction.warning.call_args.args[0]
    assert "Retained" in interaction.info.call_args.args[0]

    interaction.prompt.return_value = "reset"
    manager.call("permissions", "reload")
    assert "supervised defaults" in interaction.info.call_args.args[0]


def test_sync_permissions_command_bridges_recovery_to_legacy_input(tmp_path):
    """A synchronous command host keeps permission recovery on its caller interaction."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "continue"
    permissions = PermissionManager(tmp_path, interaction=interaction)
    permissions.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    permissions.configuration_path.write_text("version: 3\n", "utf-8")
    manager = command_manager(permissions, interaction)

    manager.call("permissions", "reload")

    assert permissions.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    interaction.prompt.assert_called_once()
    assert "Keeping" in interaction.warning.call_args.args[0]
    manager.close()


def test_permissions_commands_manage_and_display_session_boundaries(tmp_path):
    """Session boundaries affect effective output without changing workspace YAML."""
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path)
    manager = command_manager(permissions, interaction)

    manager.call("permissions", "limit set session host-process allow")
    manager.call("permissions", "limit add session network-origin https://my-host.local")
    manager.call("permissions", "default set session process.execute allow")
    manager.call("permissions", "show session")
    shown = interaction.info.call_args.args[0]
    assert "allow_host_processes: True" in shown
    assert "network_origins: https://my-host.local" in shown
    assert PermissionManager(tmp_path).configuration.limits == permissions.configuration.limits
    manager.call("permissions", "show managed")
    assert "Managed enforcement: unavailable" in interaction.info.call_args.args[0]

    manager.call("permissions", "limit reset session host-process")
    manager.call("permissions", "default reset session process.execute")
    manager.call("permissions", "default reset session process.execute")
    assert "already inherited" in interaction.warning.call_args.args[0]
    manager.call("permissions", "reset")
    manager.call("permissions", "reset session")
    assert "already empty" in interaction.warning.call_args.args[0]
    assert not permissions.session_overrides.defaults
    assert not permissions.session_rules
    assert permissions.effective_configuration == permissions.configuration


def test_permissions_limit_without_mutation_lists_scoped_values(tmp_path):
    """Bare and explicitly scoped limit inspection distinguish values from inheritance."""
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager = command_manager(permissions, interaction)

    manager.call("permissions", "limit")
    listed = interaction.info.call_args.args[0]
    assert "Workspace permission limits:" in listed
    assert "Session permission limits:" in listed
    assert "  host-process: true" in listed
    assert "  private-network: inherited" in listed

    manager.call("permissions", "limit list workspace")
    scoped = interaction.info.call_args.args[0]
    assert "Workspace permission limits:" in scoped
    assert "Session permission limits:" not in scoped
    interaction.report.assert_not_called()


def test_permissions_local_network_limit_is_a_trusted_boolean_command(tmp_path):
    """A trusted session command can revoke and restore attempt-local networking."""
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path)
    manager = command_manager(permissions, interaction)
    manager.call("permissions", "limit set session local-network deny")
    assert permissions.effective_configuration.limits.deny_local_networks is True
    manager.call("permissions", "limit reset session local-network")
    assert permissions.effective_configuration.limits.deny_local_networks is False


def test_permissions_classifies_collection_limits_through_trusted_command(tmp_path):
    """A human command changes classification and shows the value, scope, and source."""
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path)
    manager = command_manager(permissions, interaction)

    manager.call("permissions", "limit classify workspace read-root hard")
    manager.call("permissions", "limit classify session write-root approval")
    manager.call("permissions", "limit list")

    output = interaction.info.call_args.args[0]
    assert "read-root: workspace, app-temp (classification=hard, source=user)" in output
    assert "write-root: inherited (classification=approval, source=session)" in output
    assert permissions.configuration.limit_classifications.readable_roots.value == "hard"
    assert (
        permissions.effective_configuration.limit_classifications.writable_roots.value == "approval"
    )

    manager.call("permissions", "limit classify workspace private-network approval")
    interaction.report.assert_called()


def test_permissions_initialize_binds_exact_workspace_and_user_identity(tmp_path):
    """The human policy command provisions only the explicit empty namespace identity."""
    catalog = tmp_path.parent / f"{tmp_path.name}-policy-data" / "workspaces.sqlite"
    workspace = WorkspaceRepository(catalog).initialize(Workspace.discover(tmp_path))
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(
        tmp_path,
        configuration_path=catalog.parent / "workspaces" / workspace.id / "permissions.yaml",
        user_configuration_path=tmp_path.parent / f"{tmp_path.name}-user-permissions.yaml",
        workspace_id=workspace.id,
        namespace_catalog_path=catalog,
        user_namespace_identity=str(catalog.parent),
    )
    manager = command_manager(permissions, interaction)

    manager.call("permissions", "initialize workspace wrong")
    interaction.report.assert_called()
    assert "workspace namespace recovery required" in permissions.describe()

    manager.call("permissions", f"initialize workspace {workspace.id}")
    manager.call("permissions", f"initialize user {shlex.quote(str(catalog.parent))}")

    assert permissions.configuration.namespace_generation is not None
    assert "recovery required" not in permissions.describe()
    assert "Initialized the user permission policy." in interaction.info.call_args.args[0]


@pytest.mark.parametrize("declined", [False, object()])
def test_permissions_preset_commands_preview_and_confirm_scoped_policy_replacement(
    tmp_path, declined
):
    """Preset commands replace confirmed selected-scope defaults and rules only."""
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path)
    permissions.add_rule(PermissionRule(id="workspace-rule", decision=Decision.DENY))
    permissions.add_rule(
        PermissionRule(id="session-rule", decision=Decision.DENY),
        scope=PolicyScope.SESSION,
    )
    manager = command_manager(permissions, interaction)

    manager.call("permissions", "preset list")
    assert "workspace@1" in interaction.info.call_args.args[0]
    manager.call("permissions", "preset show workspace")
    assert "Permission preset: workspace@1" in interaction.info.call_args.args[0]
    assert "filesystem.create: allow" in interaction.info.call_args.args[0]
    manager.call("permissions", "preset diff workspace workspace")
    assert "Unchanged: enforcement limits" in interaction.info.call_args.args[0]

    interaction.confirm.return_value = declined
    manager.call("permissions", "preset replace workspace workspace")
    assert [rule.id for rule in permissions.persistent_rules] == ["workspace-rule"]
    assert "not approved" in interaction.warning.call_args.args[0]

    interaction.confirm.return_value = True
    manager.call("permissions", "preset replace workspace workspace")
    assert permissions.persistent_rules == ()
    assert permissions.configuration.defaults[Action.FILESYSTEM_CREATE] is Decision.ALLOW
    assert [rule.id for rule in permissions.session_rules] == ["session-rule"]
    prompt = interaction.confirm.call_args.args[0]
    assert "defaults and rules" in prompt
    assert "Non-overridable boundaries" in prompt


def test_permissions_preset_list_reports_an_empty_catalog(monkeypatch, tmp_path):
    """The preset list command explains when no selectable artifacts are available."""
    monkeypatch.setattr(
        PermissionPreset,
        "load_builtin_presets",
        classmethod(lambda cls: ((), ())),
    )
    interaction = Mock(spec=ConsoleInteraction)
    manager = command_manager(PermissionManager(tmp_path), interaction)

    manager.call("permissions", "preset")

    interaction.info.assert_called_once_with("Permission presets: none")


def test_permissions_preset_diff_describes_empty_replacements_and_replace_reports_success(tmp_path):
    """Preset diff labels empty layers and confirmed replacement reports completion."""
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path)
    manager = command_manager(permissions, interaction)

    manager.call("permissions", "preset diff session locked")
    diff = interaction.info.call_args.args[0]
    assert "Replaced defaults:" in diff
    assert "Removed rules: none" in diff
    assert "Installed rules: none" in diff

    interaction.confirm.return_value = True
    manager.call("permissions", "preset replace session locked")
    assert "Replaced session defaults and rules" in interaction.info.call_args.args[0]


def test_permissions_unsupervised_preset_warns_and_requires_confirmation(tmp_path):
    """Unsupervised activation warns and a declined confirmation preserves the policy."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm.return_value = False
    permissions = PermissionManager(tmp_path)
    manager = command_manager(permissions, interaction)

    manager.call("permissions", "preset replace workspace unsupervised")

    expected_warning = (
        "Unsupervised mode allows every authority-requiring operation without approval. "
        "Enforcement limits remain active."
    )
    assert interaction.warning.call_args_list[0].args == (expected_warning,)
    interaction.confirm.assert_called_once()
    assert interaction.confirm.call_args.kwargs == {"default": False}
    assert permissions.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.ASK

    interaction.confirm.return_value = True
    manager.call("permissions", "preset replace workspace unsupervised")

    assert all(
        decision is Decision.ALLOW for decision in permissions.configuration.defaults.values()
    )


def test_permissions_preset_diff_renders_installed_rules(tmp_path):
    """Preset diffs include the rendered rule list when an artifact installs rules."""
    preset = PermissionPreset.model_validate(
        {
            "metadata": {
                "id": "rules",
                "revision": "1",
                "title": "Rules",
                "description": "Rule-bearing preset.",
            },
            "defaults": {action.value: "deny" for action in Action},
            "rules": [{"id": "allow-read", "decision": "allow"}],
        }
    )
    interaction = Mock(spec=ConsoleInteraction)
    manager = command_manager(PermissionManager(tmp_path, presets=(preset,)), interaction)

    manager.call("permissions", "preset diff workspace rules")

    assert "preset:workspace:rules:1:allow-read allow" in interaction.info.call_args.args[0]


def test_permissions_command_rejects_every_malformed_branch(tmp_path):
    """Malformed subcommands fail with one consistent usage message."""
    interaction = Mock(spec=ConsoleInteraction)
    manager = command_manager(PermissionManager(tmp_path), interaction)

    invalid = (
        "unknown",
        "show extra",
        "explain tool invalid resource",
        "default set workspace filesystem.read invalid",
        "default invalid workspace filesystem.read",
        "rule add invalid allow tool '*' '*'",
        "rule add workspace invalid tool '*' '*'",
        "rule invalid workspace id",
        "rule remove invalid id",
        "limit add workspace host-process value",
        "limit invalid workspace read-root workspace",
        "limit set workspace private-network invalid",
        "limit set workspace read-root allow",
        "limit reset workspace invalid",
        "limit set invalid host-process allow",
        "preset unknown",
        "preset show missing extra",
        "preset diff invalid workspace",
        "reset invalid",
        "reset session extra",
        "session reset",
        "workspace reset",
        "user reset",
    )
    for arguments in invalid:
        manager.call("permissions", arguments)
        assert "Usage: /permissions" in interaction.report.call_args.args[0].detail


def complete(completer: CompletionManager, text: str):
    """Return all completions produced for text with its cursor at the end."""
    return list(completer.get_completions(Document(text), Mock()))


def test_registered_permissions_grammar_completes_described_policy_domains_and_rules(tmp_path):
    """Permission completion covers commands, limits, selectors, and dynamic rule IDs."""
    permissions = PermissionManager(tmp_path)
    permissions.add_rule(
        PermissionRule(
            id="saved-rule",
            decision=Decision.ALLOW,
            tool="read_text_file",
            action=Action.FILESYSTEM_READ,
            description="Read documentation",
        ),
        scope=PolicyScope.SESSION,
    )
    manager = CommandManager()
    manager.register_provider(PermissionCommands(permissions))
    tool_completions = Mock()
    tool_completions.get_completion_providers.return_value = (
        CompletionProviderRegistration(
            "tools", lambda: (CompletionValue("read_text_file", "tool"),)
        ),
    )
    completer = CompletionManager(
        (
            CommandCompletionAdapter(
                lambda: manager.commands,
                providers=(tool_completions,),
            ),
        )
    )
    assert "managed" in {item.text for item in complete(completer, "/permissions show ")}

    commands = complete(completer, "/permissions ")
    assert {item.text for item in commands} == {
        "show",
        "reload",
        "initialize",
        "explain",
        "default",
        "rule",
        "sandbox",
        "host",
        "limit",
        "preset",
        "reset",
        "help",
    }
    assert all(item.display_meta_text for item in commands)
    assert {item.text for item in complete(completer, "/permissions reset ")} == {
        "workspace",
        "user",
        "session",
    }
    assert complete(completer, "/permissions initialize workspace ") == []
    assert [item.text for item in complete(completer, "/permissions sandbox ")] == [
        "list",
        "remove",
    ]
    assert {item.text for item in complete(completer, "/permissions host ")} == {
        "list",
        "remove",
        "trusted",
        "trust",
        "untrust",
    }
    assert {item.text for item in complete(completer, "/permissions host trust ")} == {
        "'command here'",
        "session",
        "workspace",
        "user",
    }
    assert [item.text for item in complete(completer, "/permissions host trust user ")] == [
        "'command here'"
    ]
    trusted = permissions.trust_host_command("printf completion")
    assert {item.text for item in complete(completer, "/permissions host untrust ")} == {
        trusted.id,
        "session",
        "workspace",
        "user",
    }
    assert [item.text for item in complete(completer, "/permissions host untrust session ")] == [
        trusted.id
    ]
    assert [item.text for item in complete(completer, "/permissions rule add session al")] == [
        "allow"
    ]
    assert [
        item.text for item in complete(completer, "/permissions rule add session allow read")
    ] == ["read_text_file"]
    actions = complete(
        completer,
        "/permissions rule add session allow read_text_file filesystem.rep",
    )
    assert [item.text for item in actions] == ["filesystem.replace"]
    assert actions[0].display_meta_text
    process_actions = complete(
        completer,
        "/permissions rule add session allow read_text_file process.exe",
    )
    assert [item.text for item in process_actions] == ["process.execute"]
    assert process_actions[0].display_meta_text == "Execute a process."
    removable = complete(completer, "/permissions rule remove session saved")
    assert [item.text for item in removable] == ["saved-rule"]
    assert removable[0].display_meta_text == "Read documentation"
    permissions.add_rule(PermissionRule(id="persistent-rule", decision=Decision.ASK))
    persistent = complete(completer, "/permissions rule remove workspace persistent")
    assert [item.text for item in persistent] == ["persistent-rule"]
    limits = complete(completer, "/permissions limit set session host-process ")
    assert [item.text for item in limits] == ["allow", "deny"]
    assert all(item.display_meta_text for item in limits)
    classifications = complete(completer, "/permissions limit classify workspace ")
    assert {item.text for item in classifications} == {
        "read-root",
        "write-root",
        "network-origin",
    }
    assert {
        item.text for item in complete(completer, "/permissions limit classify session read-root ")
    } == {
        "hard",
        "approval",
    }
    presets = complete(completer, "/permissions preset ")
    assert {item.text for item in presets} == {"list", "show", "diff", "replace"}
    presets = complete(completer, "/permissions preset replace session w")
    assert [item.text for item in presets] == ["workspace"]
    roots = complete(completer, "/permissions limit remove workspace read-root ")
    assert [item.text for item in roots] == ["app-temp", "workspace"]
    root_tokens = complete(completer, "/permissions limit add workspace read-root ")
    assert [item.text for item in root_tokens] == [
        "app-temp",
        "system-temp",
        "workspace",
    ]
    assert all(item.display_meta_text for item in root_tokens)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    reset_limits = complete(completer, "/permissions limit reset session host")
    assert [item.text for item in reset_limits] == ["host-process"]


def test_permission_initialize_completion_offers_exact_quoted_identity_only_when_idle(tmp_path):
    """Durable initialization completes its exact identity without exposing an active-run action."""
    catalog = tmp_path.parent / f"{tmp_path.name} policy data" / "workspaces.sqlite"
    workspace = WorkspaceRepository(catalog).initialize(Workspace.discover(tmp_path))
    permissions = PermissionManager(
        tmp_path,
        configuration_path=catalog.parent / "workspaces" / workspace.id / "permissions.yaml",
        user_configuration_path=tmp_path.parent / "user permissions.yaml",
        workspace_id=workspace.id,
        namespace_catalog_path=catalog,
        user_namespace_identity=str(catalog.parent),
    )
    manager = CommandManager()
    manager.register_provider(PermissionCommands(permissions))
    completer = CompletionManager((manager.completion_adapter,))

    assert {item.text for item in complete(completer, "/permissions initialize ")} == {
        "workspace",
        "user",
    }
    workspace_identity = complete(completer, "/permissions initialize workspace ")
    assert [item.text for item in workspace_identity] == [workspace.id]
    user_identity = complete(completer, "/permissions initialize user ")
    assert len(user_identity) == 1
    assert shlex.split(user_identity[0].text) == [str(catalog.parent)]
    assert user_identity[0].display_meta_text

    manager.set_capability_provider(
        lambda: frozenset(
            {
                CommandCapability.CONTEXT_READ,
                CommandCapability.CONTEXT_UPDATE,
                CommandCapability.RUN_ACTIVE,
            }
        )
    )
    assert "initialize" not in {item.text for item in complete(completer, "/permissions ")}
    assert complete(completer, "/permissions initialize user ") == []
