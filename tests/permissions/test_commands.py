"""Tests for complete operation-policy user commands."""

from unittest.mock import Mock

from prompt_toolkit.document import Document

from loop import (
    Action,
    CommandCompletionAdapter,
    CommandManager,
    CompletionManager,
    CompletionValue,
    Decision,
    Interaction,
    PermissionManager,
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
    manager.call("permissions", "limit add workspace network-origin https://example.com")
    manager.call(
        "permissions",
        "rule add workspace allow read_text_file filesystem.read '/project/*' 'Read docs'",
    )
    persistent_id = permissions.persistent_rules[0].id
    manager.call("permissions", "rule add session deny run_command process.execute '*'")
    session_id = permissions.session_rules[0].id

    loaded = PermissionManager(tmp_path)
    assert loaded.configuration.defaults[Action.NETWORK_REQUEST] is Decision.DENY
    assert loaded.configuration.limits.allow_host_processes is True
    assert loaded.configuration.limits.deny_private_networks is False
    assert loaded.configuration.limits.network_origins == ("https://example.com",)
    assert loaded.configuration.rules[0].description == "Read docs"

    manager.call("permissions", f"rule remove workspace {persistent_id}")
    manager.call("permissions", f"rule remove session {session_id}")
    assert not permissions.persistent_rules
    assert not permissions.session_rules


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


def test_permissions_commands_manage_and_display_session_boundaries(tmp_path):
    """Session boundaries affect effective output without changing workspace YAML."""
    interaction = Mock(spec=Interaction)
    permissions = PermissionManager(tmp_path)
    manager = command_manager(permissions, interaction)

    manager.call("permissions", "limit set session host-process allow")
    manager.call("permissions", "limit add session network-origin https://example.com")
    manager.call("permissions", "default set session process.execute allow")
    manager.call("permissions", "show session")
    shown = interaction.info.call_args.args[0]
    assert "allow_host_processes: True" in shown
    assert "network_origins: https://example.com" in shown
    assert PermissionManager(tmp_path).configuration.limits == permissions.configuration.limits

    manager.call("permissions", "limit reset session host-process")
    manager.call("permissions", "default reset session process.execute")
    manager.call("permissions", "default reset session process.execute")
    assert "already inherited" in interaction.warning.call_args.args[0]
    manager.call("permissions", "session reset")
    manager.call("permissions", "session reset")
    assert "already empty" in interaction.warning.call_args.args[0]
    assert permissions.session_overrides.defaults == {}
    assert permissions.session_rules == ()
    assert permissions.effective_configuration == permissions.configuration


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
    )
    for arguments in invalid:
        manager.call("permissions", arguments)
        assert "Usage: /permissions" in interaction.warning.call_args.args[0]


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
    completer = CompletionManager(
        (
            CommandCompletionAdapter(
                lambda: manager.commands,
                providers={"tools": lambda: (CompletionValue("read_text_file", "tool"),)},
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
