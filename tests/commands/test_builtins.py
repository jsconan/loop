"""Tests for commands available in every conversation."""

from unittest.mock import Mock

import pytest

from loop import (
    CommandArgumentError,
    CommandContext,
    CommandManager,
    InstructionsManager,
    Interaction,
    MemorySessionStore,
    Message,
    PermissionConfiguration,
    PermissionManager,
    PermissionMode,
    Session,
    SessionManager,
    Skill,
    SkillManager,
    Tool,
)
from loop.commands import call as call_command
from loop.commands import exit as exit_command
from loop.commands import help as help_command
from loop.commands import new as new_command
from loop.commands import permissions as permissions_command
from loop.commands import quit as quit_command
from loop.commands import rename as rename_command
from loop.commands import resume as resume_command
from loop.commands import sessions as sessions_command
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

    displayed = interaction.table.call_args
    rows = [command.name for command in displayed.args[0]]
    assert rows == sorted(rows, key=str.casefold)
    assert displayed.kwargs == {"title": "Available commands:", "prefix": "  /"}


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


def test_session_commands_list_resume_reset_and_rename_sessions():
    """Session commands render metadata and resume directly by persisted ID."""
    interaction = Mock(spec=Interaction)
    store = MemorySessionStore()
    first = Session(name="First topic", name_source="user", model="served-model", tokens=1234)
    first.add_message(Message(role="user", content="Prior question"))
    first_id = store.save(first)
    manager = SessionManager(session_store=store)
    command_manager = CommandManager(
        interaction=interaction,
        session_manager=manager,
    )
    context = CommandContext("sessions", interaction, command_manager)

    sessions_command(context)
    resume_command(context, first_id)
    rename_command(context, "Renamed topic")
    new_command(context)

    assert interaction.table.call_args.kwargs["columns"] == ("name", "updated_at", "message_count")
    interaction.history.assert_called_once_with(first.messages)
    interaction.token_usage.assert_called_once_with("served-model", 1234, None)
    assert [item.args[0] for item in interaction.info.call_args_list[:2]] == [
        "Restoring session history for 'First topic'...",
        "Resumed session 'First topic'.",
    ]
    resume_calls = [
        item for item in interaction.method_calls if item[0] in {"info", "history", "token_usage"}
    ]
    assert [item[0] for item in resume_calls[:4]] == [
        "info",
        "history",
        "token_usage",
        "info",
    ]
    assert store.load(first_id).name == "Renamed topic"
    assert manager.session == Session()


@pytest.mark.parametrize(
    "function,arguments",
    [
        (sessions_command, ()),
        (resume_command, ("session name",)),
        (new_command, ()),
        (rename_command, ("name",)),
    ],
)
def test_session_commands_require_a_session_manager(function, arguments):
    """Session commands reject invocation without session lifecycle state."""
    context = CommandContext("session", Mock(spec=Interaction))

    with pytest.raises(ValueError, match="requires a SessionManager"):
        function(context, *arguments)


def test_resume_reports_ids_missing_from_session_state():
    """Resume reports a selected ID that is absent from persisted session state."""
    interaction = Mock(spec=Interaction)
    manager = SessionManager()
    command_manager = CommandManager(interaction=interaction, session_manager=manager)
    context = CommandContext("resume", interaction, command_manager)

    with pytest.raises(CommandArgumentError, match="Session 'missing-id' was not found"):
        resume_command(context, "missing-id")


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

    interaction.table.assert_called_once_with(registry.tools, title="Registered tools:")


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

    interaction.table.assert_called_once_with(manager.skills, title="Discovered skills:")


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
