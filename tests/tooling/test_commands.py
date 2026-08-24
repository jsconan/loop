"""Tests for tooling-owned user commands."""

from unittest.mock import Mock

from loop import CommandManager, InstructionsManager, Interaction, Tool
from loop.tooling import ToolCommands


def test_tools_command_displays_registered_tools_and_reports_empty_catalogs():
    """Tool discovery renders populated registries and explains empty ones."""
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


def test_call_command_forwards_tokens_and_runtime_context():
    """Tool invocation forwards parsed remainder tokens and active instruction state."""
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
    registry.command.return_value = "42"
    instructions = InstructionsManager()
    manager = CommandManager(interaction=interaction)
    provider = ToolCommands(registry, instructions)
    manager.register_provider(provider)

    manager.call("call", 'calculate number=21 label="two words"')
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
    interaction.tool_result.assert_called_with("ping", "42")
    assert manager.commands[-1].completion.schema_provider is None
    assert manager.commands[-1].completion.next.schema_provider == "tool_arguments"
    values, schemas = provider.get_completion_providers()
    assert [(value.value, value.description) for value in values.provider()] == [
        ("ping", "Ping a service")
    ]
    assert schemas.provider(()) is None
    assert schemas.provider(("ping",)) is arguments_model


def test_call_command_forwards_quoted_run_command_text_without_tool_specific_rewriting():
    """Quoted command text and named arguments remain generic tool-call tokens."""
    interaction = Mock(spec=Interaction)
    registry = Mock()
    registry.tools = []
    registry.command.return_value = ""
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ToolCommands(registry, InstructionsManager()))

    manager.call("call", 'run_command "git status" cwd="."')

    assert registry.command.call_args.args == (
        "run_command",
        ("git status", "cwd=."),
    )
