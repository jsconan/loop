"""Tests for commands available in every conversation."""

from unittest.mock import Mock

import pytest

from loop import (
    CommandArgumentError,
    CommandContext,
    CommandManager,
    InstructionsManager,
    Interaction,
    PermissionConfiguration,
    PermissionManager,
    PermissionMode,
    Skill,
    SkillManager,
    Tool,
)
from loop.commands import call as call_command
from loop.commands import exit as exit_command
from loop.commands import help as help_command
from loop.commands import permissions as permissions_command
from loop.commands import quit as quit_command
from loop.commands import skills as skills_command
from loop.commands import tools as tools_command
from loop.commands import use as use_command


def test_help_displays_slash_prefixed_command_catalog():
    """Help alphabetizes commands and adds the presentation-only slash."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)

    @manager.register(name="zebra", description="Run the last command.")
    def zebra() -> None:
        pass

    @manager.register(name="alpha", description="Run the first command.")
    def alpha() -> None:
        pass

    help_command(CommandContext("help", interaction, manager))

    help_text = interaction.info.call_args.args[0]
    rows = [line.split()[0] for line in help_text.splitlines()[2:]]
    assert rows == sorted(rows, key=str.casefold)
    assert "  /help         Show the available commands." in help_text
    assert "  /zebra        Run the last command." in help_text


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


def test_use_command_loads_skill_instructions_and_reports_repeated_use(tmp_path):
    """Use activates a skill through the instruction lifecycle and is idempotent."""
    location = tmp_path / "SKILL.md"
    location.write_text("---\nname: review\ndescription: Review code.\n---\nCheck carefully.\n")
    instructions = InstructionsManager(
        skill_manager=SkillManager([Skill("review", "Review.", location)])
    )
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction, instructions_manager=instructions)

    manager.call("use", "review")
    manager.call("use", "review")

    assert "Check carefully." in instructions.instructions
    assert interaction.info.call_args_list[0].args[0] == "Loaded skill 'review'."
    assert interaction.info.call_args_list[1].args[0] == "Skill 'review' is already loaded."


def test_use_command_reports_unknown_skills_and_missing_dependencies():
    """Use reports unavailable skills and rejects missing lifecycle dependencies."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction, instructions_manager=InstructionsManager())

    manager.call("use", "missing")

    assert "Skill 'missing' is not available" in interaction.warning.call_args.args[0]
    with pytest.raises(ValueError, match="requires a CommandManager"):
        use_command(CommandContext("use", interaction), "missing")
    with pytest.raises(ValueError, match="requires an InstructionsManager"):
        use_command(CommandContext("use", interaction, CommandManager()), "missing")


def test_use_command_reports_skill_loading_failures():
    """Use converts skill loading failures into command argument errors."""
    interaction = Mock(spec=Interaction)
    instructions = Mock()
    instructions.skill_manager = Mock()
    instructions.activate_skill.side_effect = ValueError("malformed instructions")
    context = CommandContext("use", interaction, CommandManager(instructions_manager=instructions))

    with pytest.raises(CommandArgumentError, match="Could not load skill 'broken'"):
        use_command(context, "broken")


def test_call_command_invokes_tools_with_command_arguments_and_runtime_context():
    """Call forwards command tokens, interaction, instructions, and permissions."""
    interaction = Mock(spec=Interaction)
    registry = Mock()
    registry.command.return_value = "42"
    instructions = InstructionsManager()
    manager = CommandManager(
        interaction=interaction, instructions_manager=instructions, tool_registry=registry
    )

    manager.call("call", 'calculate number=21 label="two words"')

    registry.command.assert_called_once_with(
        "calculate",
        ("number=21", "label=two words"),
        interaction=interaction,
        instructions_manager=instructions,
    )
    interaction.tool_result.assert_called_once_with("calculate", "42")


def test_call_command_defaults_to_empty_json_and_requires_dependencies():
    """Call supports argument-free tools and rejects missing registry dependencies."""
    interaction = Mock(spec=Interaction)
    registry = Mock()
    registry.command.return_value = "done"
    call_command(
        CommandContext("call", interaction, CommandManager(tool_registry=registry)), "ping"
    )
    registry.command.assert_called_once_with(
        "ping", (), interaction=interaction, instructions_manager=None
    )
    with pytest.raises(ValueError, match="requires a CommandManager"):
        call_command(CommandContext("call", interaction), "ping")
    with pytest.raises(ValueError, match="requires a ToolRegistry"):
        call_command(CommandContext("call", interaction, CommandManager()), "ping")
