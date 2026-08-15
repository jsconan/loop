"""Tests for local permission evaluation, persistence, prompts, and auditing."""

import json
from unittest.mock import Mock

import pytest

from loop import (
    Capability,
    Decision,
    Interaction,
    PermissionConfiguration,
    PermissionManager,
    PermissionMode,
    PermissionRequest,
    PermissionRule,
)


def request(
    capability: Capability = Capability.PURE,
    *,
    tool: str = "demo",
    resource: str | None = None,
    reason: str | None = None,
) -> PermissionRequest:
    """Build a representative permission request."""
    return PermissionRequest(
        tool_name=tool, capability=capability, resource=resource, reason=reason
    )


def manager_for(mode: PermissionMode, tmp_path=None, interaction=None) -> PermissionManager:
    """Build a manager with one explicit mode."""
    return PermissionManager(
        tmp_path,
        interaction=interaction,
        configuration=PermissionConfiguration(mode=mode),
    )


def test_default_configuration_is_local_and_fail_closed_without_a_user(tmp_path):
    """Missing local configuration defaults to confirm-all and headless denial."""
    manager = PermissionManager(tmp_path)

    result = manager.authorize(request())

    assert manager.configuration.mode is PermissionMode.CONFIRM_ALL
    assert manager.configuration_path == tmp_path / ".loop" / "permissions.yaml"
    assert result.decision is Decision.DENY
    audit = tmp_path / ".loop" / "permissions-audit.jsonl"
    assert json.loads(audit.read_text("utf-8"))["result"]["source"] == "headless"


@pytest.mark.parametrize("approved,decision", [(True, Decision.ALLOW), (False, Decision.DENY)])
def test_confirm_all_prompts_with_normalized_context(tmp_path, approved, decision):
    """Interactive confirmation records either a one-call approval or rejection."""
    interaction = Mock(spec=Interaction)
    interaction.confirm.return_value = approved
    manager = manager_for(PermissionMode.CONFIRM_ALL, tmp_path, interaction)
    operation = request(
        Capability.NETWORK_READ,
        resource="https://example.com",
        reason="Needed for documentation.",
    )

    result = manager.authorize(operation)

    assert result.decision is decision
    interaction.confirm.assert_called_once_with(
        "🌐 Agent wants to use 'demo' for network.read on 'https://example.com'. "
        "Needed for documentation. Proceed?",
        default=False,
    )


@pytest.mark.parametrize(
    ("resource", "expected"),
    [
        ("inside.txt", "inside.txt"),
        ("folder/inside.txt", "folder/inside.txt"),
        ("../outside.txt", None),
    ],
)
def test_filesystem_prompt_uses_workspace_relative_paths_unless_resource_escapes(
    tmp_path, resource, expected
):
    """File approval prompts abbreviate in-workspace paths but retain outside absolute paths."""
    interaction = Mock(spec=Interaction)
    interaction.confirm.return_value = True
    manager = manager_for(PermissionMode.CONFIRM_ALL, tmp_path, interaction)
    absolute_resource = (tmp_path / resource).resolve()

    manager.authorize(request(Capability.FILESYSTEM_READ, resource=str(absolute_resource)))

    displayed = expected if expected is not None else str(absolute_resource)
    interaction.confirm.assert_called_once_with(
        f"📖 Agent wants to use 'demo' for filesystem.read on '{displayed}'. Proceed?",
        default=False,
    )


def test_filesystem_prompt_identifies_the_workspace_root_unambiguously(tmp_path):
    """File approval prompts label the workspace root without resembling a child path."""
    interaction = Mock(spec=Interaction)
    interaction.confirm.return_value = True
    manager = manager_for(PermissionMode.CONFIRM_ALL, tmp_path, interaction)

    manager.authorize(request(Capability.FILESYSTEM_READ, resource=str(tmp_path)))

    interaction.confirm.assert_called_once_with(
        f"📖 Agent wants to use 'demo' for filesystem.read on "
        f"'workspace root: {tmp_path}'. Proceed?",
        default=False,
    )


def test_explicit_rules_match_fields_and_deny_precedes_ask_and_allow():
    """Matching deny rules dominate weaker decisions regardless of declaration order."""
    configuration = PermissionConfiguration(
        mode=PermissionMode.UNRESTRICTED,
        rules=[
            PermissionRule(decision=Decision.ALLOW, tool="read_*"),
            PermissionRule(
                decision=Decision.ASK,
                tool="read_file",
                capability=Capability.FILESYSTEM_READ,
            ),
            PermissionRule(
                decision=Decision.DENY,
                tool="read_file",
                resource="*/secret.*",
            ),
        ],
    )
    manager = PermissionManager(configuration=configuration)

    result = manager.evaluate(
        request(
            Capability.FILESYSTEM_READ,
            tool="read_file",
            resource="/project/secret.env",
        )
    )

    assert result.decision is Decision.DENY
    assert manager.evaluate(request(tool="other")).decision is Decision.ALLOW


def test_ignored_files_are_denied_by_default_before_interactive_approval(tmp_path, monkeypatch):
    """Ignore rules create a default filesystem denial that modes cannot prompt around."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("secret.txt\n", "utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("sensitive", "utf-8")
    interaction = Mock(spec=Interaction)
    interaction.confirm.return_value = True
    manager = manager_for(PermissionMode.CONFIRM_ALL, tmp_path, interaction)

    result = manager.authorize(request(Capability.FILESYSTEM_READ, resource=str(secret)))

    assert result.decision is Decision.DENY
    assert result.source == "safety:ignored_path"
    interaction.confirm.assert_not_called()

    monkeypatch.setattr(
        "loop.permissions.manager.is_path_ignored",
        Mock(side_effect=OSError("unreadable ignore policy")),
    )
    assert (
        manager.evaluate(
            request(Capability.FILESYSTEM_READ, resource=str(tmp_path / "other.txt"))
        ).decision
        is Decision.DENY
    )


def test_explicit_allow_rule_cannot_override_an_ignored_resource(tmp_path):
    """Hard ignored-path denial takes precedence over explicit allow rules."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("secret.txt\n", "utf-8")
    secret = tmp_path / "secret.txt"
    configuration = PermissionConfiguration(
        rules=[
            PermissionRule(
                decision=Decision.ALLOW,
                tool="read_text_file",
                capability=Capability.FILESYSTEM_READ,
                resource=str(secret),
            )
        ]
    )
    manager = PermissionManager(tmp_path, configuration=configuration)

    result = manager.evaluate(
        request(
            Capability.FILESYSTEM_READ,
            tool="read_text_file",
            resource=str(secret),
        )
    )

    assert result.decision is Decision.DENY


@pytest.mark.parametrize(
    "mode,capability,expected",
    [
        (PermissionMode.LOCKED_DOWN, Capability.PURE, Decision.DENY),
        (PermissionMode.UNRESTRICTED, Capability.PROCESS_EXEC, Decision.ALLOW),
        (PermissionMode.READ_ONLY, Capability.FILESYSTEM_READ, Decision.ALLOW),
        (PermissionMode.READ_ONLY, Capability.FILESYSTEM_WRITE, Decision.DENY),
        (PermissionMode.READ_ONLY, Capability.FILESYSTEM_DELETE, Decision.DENY),
        (PermissionMode.WORKSPACE_WRITE, Capability.NETWORK_READ, Decision.ALLOW),
        (PermissionMode.WORKSPACE_WRITE, Capability.FILESYSTEM_DELETE, Decision.ASK),
        (PermissionMode.WORKSPACE_WRITE, Capability.PROCESS_EXEC, Decision.ASK),
    ],
)
def test_modes_apply_capability_fallbacks(mode, capability, expected):
    """Each mode applies its documented capability fallback."""
    assert manager_for(mode).evaluate(request(capability)).decision is expected


def test_workspace_write_allows_only_normalized_paths_below_workspace(tmp_path):
    """Workspace write mode does not grant missing or escaping filesystem resources."""
    manager = manager_for(PermissionMode.WORKSPACE_WRITE, tmp_path)

    assert (
        manager.evaluate(
            request(Capability.FILESYSTEM_WRITE, resource=str(tmp_path / "file.txt"))
        ).decision
        is Decision.ALLOW
    )
    assert (
        manager.evaluate(
            request(Capability.FILESYSTEM_WRITE, resource=str(tmp_path.parent / "outside.txt"))
        ).decision
        is Decision.ASK
    )
    assert manager.evaluate(request(Capability.FILESYSTEM_WRITE)).decision is Decision.ASK


def test_ignored_delete_paths_are_denied_before_interactive_approval(tmp_path):
    """Ignore rules prevent deletion even when the user could otherwise approve it."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("secret.txt\n", "utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("sensitive", "utf-8")
    interaction = Mock(spec=Interaction)
    interaction.confirm.return_value = True
    manager = manager_for(PermissionMode.CONFIRM_ALL, tmp_path, interaction)

    result = manager.authorize(request(Capability.FILESYSTEM_DELETE, resource=str(secret)))

    assert result.decision is Decision.DENY
    assert result.source == "safety:ignored_path"
    interaction.confirm.assert_not_called()


def test_persisted_and_session_rules_have_distinct_lifetimes(tmp_path):
    """Persisted rules round-trip through YAML while session rules remain in memory."""
    manager = PermissionManager(tmp_path)
    persisted = PermissionRule(
        decision=Decision.ALLOW, tool="read_*", capability=Capability.FILESYSTEM_READ
    )
    temporary = PermissionRule(decision=Decision.DENY, tool="danger")

    manager.set_mode(PermissionMode.READ_ONLY, persist=False)
    manager.add_rule(persisted)
    manager.add_rule(temporary, persist=False)

    loaded = PermissionManager(tmp_path)
    assert loaded.configuration.mode is PermissionMode.READ_ONLY
    assert loaded.configuration.rules == [persisted]
    assert manager.evaluate(request(tool="danger")).decision is Decision.DENY
    assert loaded.evaluate(request(tool="danger")).decision is Decision.ALLOW


def test_describe_covers_empty_and_populated_in_memory_policies():
    """Policy summaries expose modes, locations, and complete rule selectors."""
    manager = PermissionManager()
    assert manager.describe() == (
        "Permission mode: confirm_all\nConfiguration: in memory\nRules: none"
    )

    rule = PermissionRule(
        decision=Decision.ASK,
        tool="write_*",
        capability=Capability.FILESYSTEM_WRITE,
        resource="/project/*",
    )
    manager = PermissionManager(configuration=PermissionConfiguration(rules=[rule]))
    assert "ask tool=write_* capability=filesystem.write resource=/project/*" in manager.describe()


def test_in_memory_configuration_cannot_be_saved():
    """Persistence without a local configuration path is rejected explicitly."""
    with pytest.raises(ValueError, match="cannot persist"):
        PermissionManager().save()


def test_empty_yaml_loads_as_default_configuration(tmp_path):
    """An empty local YAML file is accepted as the default configuration."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text("", "utf-8")

    assert PermissionManager(tmp_path).configuration == PermissionConfiguration()
