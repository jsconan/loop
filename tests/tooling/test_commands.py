"""Tests for tooling-owned user commands."""

from unittest.mock import Mock

from loop import CommandManager, InstructionsManager, Interaction, Tool, ToolExecutionResult
from loop.tooling import ToolCommands


def test_tools_command_displays_registered_tools_and_reports_empty_catalogs():
    """Bare tool discovery renders catalogs and reports empty catalogs."""
    interaction = Mock(spec=Interaction)
    registry = Mock()
    registry.tools = [
        Tool("read_file", "Read a file from disk", lambda: None, Mock(), frozenset()),
    ]
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ToolCommands(registry, InstructionsManager()))

    manager.call("tools")
    interaction.table.assert_called_once_with(registry.tools, title="Registered tools:")

    registry.tools = []
    manager.call("tools")
    assert interaction.info.call_args.args[0] == "No tools registered."
    manager.call("tools", "call")
    assert "requires a tool name" in interaction.report.call_args.args[0].detail


def test_tools_call_and_shortcut_forward_tokens_and_runtime_context():
    """Tool invocation forms forward parsed tokens and active instruction state."""
    interaction = Mock(spec=Interaction)
    registry = Mock()
    arguments_model = Mock()
    tool = Tool(
        lambda: None,
        name="ping",
        description="Ping a service",
        arguments_model=arguments_model,
    )
    registry.tools = [tool]
    registry.command.return_value = ToolExecutionResult("42")
    instructions = InstructionsManager()
    manager = CommandManager(interaction=interaction)
    provider = ToolCommands(registry, instructions)
    manager.register_provider(provider)

    manager.call("tools", 'call calculate number=21 label="two words"')
    manager.call("call", "ping")

    assert registry.command.call_args_list[0].args == (
        "calculate",
        ("number=21", "label=two words"),
    )
    assert registry.command.call_args_list[0].kwargs == {
        "interaction": interaction,
        "instructions_manager": instructions,
    }
    assert registry.command.call_args_list[1].args == ("ping", ())
    interaction.tool_result.assert_called_with("42", ToolExecutionResult("42").presentation)
    tools_command, call_command = manager.commands[-2:]
    assert tools_command.completion.children["call"].next.schema_provider == "tool_arguments"
    assert call_command.completion.next.schema_provider == "tool_arguments"
    values, schemas = provider.get_completion_providers()
    assert [(value.value, value.description) for value in values.provider()] == [
        ("ping", "Ping a service")
    ]
    assert schemas.provider(()) is None
    assert schemas.provider(("ping",)) is arguments_model
    assert schemas.provider(("call", "ping")) is arguments_model


def test_call_command_forwards_quoted_run_command_text_without_tool_specific_rewriting():
    """Quoted command text and named arguments remain generic tool-call tokens."""
    interaction = Mock(spec=Interaction)
    registry = Mock()
    registry.tools = []
    registry.command.return_value = ToolExecutionResult("")
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ToolCommands(registry, InstructionsManager()))

    manager.call("call", 'run_command "git status" cwd="."')

    assert registry.command.call_args.args == (
        "run_command",
        ("git status", "cwd=."),
    )
