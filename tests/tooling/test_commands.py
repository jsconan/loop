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
    registry.tools = []
    registry.command.return_value = "42"
    instructions = InstructionsManager()
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ToolCommands(registry, instructions))

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
