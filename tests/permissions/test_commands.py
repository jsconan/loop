"""Tests for permission-policy user commands."""

from unittest.mock import Mock

from loop import (
    CommandManager,
    Interaction,
    PermissionConfiguration,
    PermissionManager,
    PermissionMode,
)
from loop.permissions import PermissionCommands


def test_permissions_command_shows_and_changes_local_policy(tmp_path):
    """Permission commands display policy and persist modes and rules locally."""
    interaction = Mock(spec=Interaction)
    permissions = PermissionManager(tmp_path)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(PermissionCommands(permissions))

    manager.call("permissions")
    manager.call("permissions", "mode read_only")
    manager.call(
        "permissions",
        "add allow 'read_*' filesystem.read '/project/*'",
    )

    assert interaction.info.call_args_list[0].args[0].startswith("Permission mode: confirm_all")
    loaded = PermissionManager(tmp_path)
    assert loaded.configuration.mode is PermissionMode.READ_ONLY
    assert loaded.configuration.rules[0].tool == "read_*"


def test_permissions_command_adds_session_rules_and_validates_usage():
    """Session rules remain transient and malformed operations are reported."""
    interaction = Mock(spec=Interaction)
    permissions = PermissionManager(
        configuration=PermissionConfiguration(mode=PermissionMode.UNRESTRICTED)
    )
    manager = CommandManager(interaction=interaction)
    manager.register_provider(PermissionCommands(permissions))

    manager.call("permissions", "session deny dangerous '*' '*'")
    assert "Added session deny rule" in interaction.info.call_args.args[0]
    for arguments in (
        "unknown",
        "show extra",
        "mode",
        "mode invalid",
        "mode read_only extra",
        "add invalid tool",
        "add allow tool invalid",
    ):
        manager.call("permissions", arguments)
        assert "Usage: /permissions" in interaction.warning.call_args.args[0]

    assert manager.commands[-1].completion is not None
