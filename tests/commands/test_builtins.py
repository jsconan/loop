"""Tests for commands available in every conversation."""

from unittest.mock import Mock

import pytest

from loop import (
    CommandArgumentError,
    CommandContext,
    CommandManager,
    Interaction,
    PermissionConfiguration,
    PermissionManager,
    PermissionMode,
    Skill,
    Tool,
)
from loop.commands import exit as exit_command
from loop.commands import help as help_command
from loop.commands import permissions as permissions_command
from loop.commands import quit as quit_command
from loop.commands import skills as skills_command
from loop.commands import tools as tools_command


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
    permissions_command(context, "mode", "read_only")
    permissions_command(context, "add", "allow", "read_*", "filesystem.read", "/project/*")

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

    permissions_command(context, "session", "deny", "dangerous", "*", "*")
    assert "Added session deny rule" in interaction.info.call_args.args[0]
    for arguments in (
        ("unknown",),
        ("show", "extra"),
        ("mode",),
        ("mode", "invalid"),
        ("mode", "read_only", "extra"),
        ("add", "invalid", "tool"),
        ("add", "allow", "tool", "invalid"),
    ):
        with pytest.raises(CommandArgumentError, match="Usage"):
            permissions_command(context, *arguments)


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


def test_tools_command_displays_registered_tools():
    """Tools command shows each registered tool name and description."""
    interaction = Mock(spec=Interaction)
    registry = Mock()
    registry.tools = [
        Tool("read_file", "Read a file from disk", lambda: None, Mock(), frozenset()),
        Tool("write_file", "Write content to a file", lambda: None, Mock(), frozenset()),
    ]
    context = CommandContext("tools", interaction, CommandManager(tool_registry=registry))

    tools_command(context)

    output = interaction.info.call_args.args[0]
    assert "Registered tools:" in output
    assert "read_file" in output
    assert "Read a file from disk" in output
    assert "write_file" in output
    assert "Write content to a file" in output


def test_tools_command_reports_empty_when_no_tools():
    """Tools command reports no tools when the registry is empty."""
    interaction = Mock(spec=Interaction)
    registry = Mock()
    registry.tools = []
    context = CommandContext("tools", interaction, CommandManager(tool_registry=registry))

    tools_command(context)

    assert "No tools registered." in interaction.info.call_args.args[0]


def test_tools_command_requires_tool_registry():
    """Tools commands reject missing manager and registry dependencies."""
    with pytest.raises(ValueError, match="requires a CommandManager"):
        tools_command(CommandContext("tools", Mock(spec=Interaction)))
    with pytest.raises(ValueError, match="requires a ToolRegistry"):
        tools_command(CommandContext("tools", Mock(spec=Interaction), CommandManager()))


def test_skills_command_discovered_skills():
    """Skills command shows each discovered skill name and description."""
    interaction = Mock(spec=Interaction)
    manager = Mock()
    manager.skills = [
        Skill("coding", "Implement and modify code", Mock()),
        Skill("testing", "Write and run tests", Mock()),
    ]
    context = CommandContext("skills", interaction, CommandManager(skill_manager=manager))

    skills_command(context)

    output = interaction.info.call_args.args[0]
    assert "Discovered skills:" in output
    assert "coding" in output
    assert "Implement and modify code" in output
    assert "testing" in output
    assert "Write and run tests" in output


def test_skills_command_reports_empty_when_no_skills():
    """Skills command reports no skills when the catalog is empty."""
    interaction = Mock(spec=Interaction)
    manager = Mock()
    manager.skills = []
    context = CommandContext("skills", interaction, CommandManager(skill_manager=manager))

    skills_command(context)

    assert "No skills discovered." in interaction.info.call_args.args[0]


def test_skills_command_requires_skill_manager():
    """Skills commands reject missing command and skill manager dependencies."""
    with pytest.raises(ValueError, match="requires a CommandManager"):
        skills_command(CommandContext("skills", Mock(spec=Interaction)))
    with pytest.raises(ValueError, match="requires a SkillManager"):
        skills_command(CommandContext("skills", Mock(spec=Interaction), CommandManager()))
