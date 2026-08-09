"""Tests for commands available in every conversation."""

from unittest.mock import Mock

import pytest

from loop import (
    CommandContext,
    CommandManager,
    Interaction,
    PermissionConfiguration,
    PermissionManager,
    PermissionMode,
)
from loop.commands import exit as exit_command
from loop.commands import help as help_command
from loop.commands import permissions as permissions_command
from loop.commands import quit as quit_command


def test_help_displays_slash_prefixed_command_catalog():
    """Help adds the presentation-only slash to slash-free command metadata."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)

    @manager.register(name="long-command", description="Run a longer command.")
    def long_command() -> None:
        pass

    help_command(CommandContext("help", interaction, manager))

    help_text = interaction.info.call_args.args[0]
    assert "  /help          Show the available commands." in help_text
    assert "  /long-command  Run a longer command." in help_text


def test_permissions_command_shows_and_changes_local_policy(tmp_path):
    """Permission commands display policy and persist modes and rules locally."""
    interaction = Mock(spec=Interaction)
    permissions = PermissionManager(tmp_path)
    context = CommandContext("permissions", interaction, permission_manager=permissions)

    permissions_command(context)
    permissions_command(context, "mode read_only")
    permissions_command(context, "add allow 'read_*' filesystem.read '/project/*'")

    assert interaction.info.call_args_list[0].args[0].startswith("Permission mode: confirm_all")
    loaded = PermissionManager(tmp_path)
    assert loaded.configuration.mode is PermissionMode.READ_ONLY
    assert loaded.configuration.rules[0].tool == "read_*"


def test_permissions_command_adds_session_rules_and_validates_usage():
    """Session rules remain transient and malformed operations are rejected."""
    interaction = Mock(spec=Interaction)
    permissions = PermissionManager(
        configuration=PermissionConfiguration(mode=PermissionMode.UNRESTRICTED)
    )
    context = CommandContext("permissions", interaction, permission_manager=permissions)

    permissions_command(context, "session deny dangerous * *")
    assert "Added session deny rule" in interaction.info.call_args.args[0]
    with pytest.raises(ValueError, match="Usage"):
        permissions_command(context, "unknown")


def test_permissions_command_requires_a_policy_manager():
    """Independent permission commands reject missing policy state."""
    with pytest.raises(ValueError, match="requires a PermissionManager"):
        permissions_command(CommandContext("permissions", Mock(spec=Interaction)))


@pytest.mark.parametrize("function,name", [(help_command, "help"), (exit_command, "exit")])
def test_builtins_require_manager_context(function, name):
    """Built-ins reject independent invocation because they operate on manager state."""
    context = CommandContext(name, Mock(spec=Interaction))
    with pytest.raises(ValueError, match="requires a CommandManager"):
        function(context)


@pytest.mark.parametrize("function,name", [(exit_command, "exit"), (quit_command, "quit")])
def test_exit_commands_request_termination(function, name):
    """Exit handlers request manager termination through their context."""
    manager = CommandManager()
    function(CommandContext(name, Mock(spec=Interaction), manager))
    assert manager.exit_requested is True
