"""Tests for complete operation-policy user commands."""

from unittest.mock import Mock

from prompt_toolkit.document import Document

from loop import (
    Action,
    CommandCompletionAdapter,
    CommandManager,
    CompletionManager,
    CompletionProviderRegistration,
    CompletionValue,
    Decision,
    Interaction,
    PermissionManager,
    PermissionPreset,
    PermissionRule,
    PolicyScope,
)
from loop.permissions import PermissionCommands


def command_manager(permissions, interaction):
    """Return a command manager exposing the supplied permission policy."""
    manager = CommandManager(interaction=interaction)
    manager.register_provider(PermissionCommands(permissions))
    return manager


def test_permissions_command_manages_defaults_limits_and_complete_rule_lifetimes(tmp_path):
    """Commands persist every policy dimension and manage process-local rules separately."""
    interaction = Mock(spec=Interaction)
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
    interaction = Mock(spec=Interaction)
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


def test_permissions_rule_add_confirms_broad_matchers_before_mutation(tmp_path):
    """Broad rules require explicit approval and declined creation leaves policy unchanged."""
    interaction = Mock(spec=Interaction)
    permissions = PermissionManager(tmp_path)
    manager = command_manager(permissions, interaction)

    interaction.confirm.return_value = False
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
    interaction = Mock(spec=Interaction)
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
    interaction = Mock(spec=Interaction)
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

    manager.call("permissions", "explain read_text_file filesystem.read file.txt")
    explained = interaction.info.call_args.args[0]
    assert "Effective decision: deny" in explained
    assert "rule:" in explained


def test_permissions_reload_and_limit_removal_report_outcomes(tmp_path):
    """Reload validates disk policy and unchanged collection operations produce warnings."""
    interaction = Mock(spec=Interaction)
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
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "continue"
    permissions = PermissionManager(tmp_path, interaction=interaction)
    permissions.set_default(Action.FILESYSTEM_DELETE, Decision.DENY)
    (tmp_path / ".loop" / "permissions.yaml").write_text("version: 2\n", "utf-8")
    manager = command_manager(permissions, interaction)

    manager.call("permissions", "reload")

    assert permissions.configuration.defaults[Action.FILESYSTEM_DELETE] is Decision.DENY
    interaction.report.assert_called_once()
    assert "Keeping" in interaction.warning.call_args.args[0]
    assert "Retained" in interaction.info.call_args.args[0]

    interaction.prompt.return_value = "reset"
    manager.call("permissions", "reload")
    assert "supervised defaults" in interaction.info.call_args.args[0]


def test_permissions_commands_manage_and_display_session_boundaries(tmp_path):
    """Session boundaries affect effective output without changing workspace YAML."""
    interaction = Mock(spec=Interaction)
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

    manager.call("permissions", "limit reset session host-process")
    manager.call("permissions", "default reset session process.execute")
    manager.call("permissions", "default reset session process.execute")
    assert "already inherited" in interaction.warning.call_args.args[0]
    manager.call("permissions", "session reset")
    manager.call("permissions", "session reset")
    assert "already empty" in interaction.warning.call_args.args[0]
    assert not permissions.session_overrides.defaults
    assert not permissions.session_rules
    assert permissions.effective_configuration == permissions.configuration


def test_permissions_limit_without_mutation_lists_scoped_values(tmp_path):
    """Bare and explicitly scoped limit inspection distinguish values from inheritance."""
    interaction = Mock(spec=Interaction)
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


def test_permissions_preset_commands_preview_and_confirm_scoped_policy_replacement(tmp_path):
    """Preset commands replace confirmed selected-scope defaults and rules only."""
    interaction = Mock(spec=Interaction)
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

    interaction.confirm.return_value = False
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
    interaction = Mock(spec=Interaction)
    manager = command_manager(PermissionManager(tmp_path), interaction)

    manager.call("permissions", "preset")

    interaction.info.assert_called_once_with("Permission presets: none")


def test_permissions_preset_diff_describes_empty_replacements_and_replace_reports_success(tmp_path):
    """Preset diff labels empty layers and confirmed replacement reports completion."""
    interaction = Mock(spec=Interaction)
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
    interaction = Mock(spec=Interaction)
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
    interaction = Mock(spec=Interaction)
    manager = command_manager(PermissionManager(tmp_path, presets=(preset,)), interaction)

    manager.call("permissions", "preset diff workspace rules")

    assert "preset:workspace:rules:1:allow-read allow" in interaction.info.call_args.args[0]


def test_permissions_command_rejects_every_malformed_branch(tmp_path):
    """Malformed subcommands fail with one consistent usage message."""
    interaction = Mock(spec=Interaction)
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

    commands = complete(completer, "/permissions ")
    assert {item.text for item in commands} == {
        "show",
        "reload",
        "explain",
        "default",
        "rule",
        "limit",
        "preset",
        "session",
        "help",
    }
    assert all(item.display_meta_text for item in commands)
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
    removable = complete(completer, "/permissions rule remove session saved")
    assert [item.text for item in removable] == ["saved-rule"]
    assert removable[0].display_meta_text == "Read documentation"
    permissions.add_rule(PermissionRule(id="persistent-rule", decision=Decision.ASK))
    persistent = complete(completer, "/permissions rule remove workspace persistent")
    assert [item.text for item in persistent] == ["persistent-rule"]
    limits = complete(completer, "/permissions limit set session host-process ")
    assert [item.text for item in limits] == ["allow", "deny"]
    assert all(item.display_meta_text for item in limits)
    presets = complete(completer, "/permissions preset ")
    assert {item.text for item in presets} == {"list", "show", "diff", "replace"}
    presets = complete(completer, "/permissions preset replace session w")
    assert [item.text for item in presets] == ["workspace"]
    roots = complete(completer, "/permissions limit remove workspace read-root ")
    assert [item.text for item in roots] == ["loop-temp", "workspace"]
    root_tokens = complete(completer, "/permissions limit add workspace read-root ")
    assert [item.text for item in root_tokens] == [
        "loop-temp",
        "system-temp",
        "workspace",
    ]
    assert all(item.display_meta_text for item in root_tokens)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    reset_limits = complete(completer, "/permissions limit reset session host")
    assert [item.text for item in reset_limits] == ["host-process"]
