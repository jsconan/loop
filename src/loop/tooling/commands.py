"""Expose registered tools through user commands."""

from typing import Annotated

from pydantic import Field

from ..commands import CommandContext, CommandRegistration, CommandRemainder
from ..completion import (
    CommandCompletion,
    CompletionProviderRegistration,
    CompletionValue,
    SchemaCompletionProviderRegistration,
)
from ..skills import InstructionsManager
from .tool_registry import ToolRegistry


class ToolCommands:
    """Expose one tool registry through interactive commands.

    Args:
        tool_registry (ToolRegistry): Tool catalog invoked and displayed by the commands.
        instructions_manager (InstructionsManager): Active instruction lifecycle passed to tools.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        instructions_manager: InstructionsManager,
    ) -> None:
        self._tool_registry = tool_registry
        self._instructions_manager = instructions_manager

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return tooling command registrations.

        Returns:
            tuple[CommandRegistration, ...]: Tool discovery and invocation commands.
        """
        return (
            CommandRegistration(self.tools, name="tools"),
            CommandRegistration(
                self.call,
                name="call",
                completion=CommandCompletion(
                    provider="tools",
                    next=CommandCompletion(schema_provider="tool_arguments"),
                ),
            ),
        )

    def get_completion_providers(
        self,
    ) -> tuple[CompletionProviderRegistration | SchemaCompletionProviderRegistration, ...]:
        """Return dynamic tool value and argument-schema completion sources.

        Returns:
            tuple[CompletionProviderRegistration | SchemaCompletionProviderRegistration, ...]:
                Named tool completion sources.
        """
        return (
            CompletionProviderRegistration("tools", self._tool_values),
            SchemaCompletionProviderRegistration("tool_arguments", self._tool_arguments),
        )

    def _tool_values(self) -> tuple[CompletionValue, ...]:
        """Return currently registered tools."""
        return tuple(
            CompletionValue(tool.name, tool.description) for tool in self._tool_registry.tools
        )

    def _tool_arguments(self, tokens: tuple[str, ...]):
        """Return the argument model for the selected tool, if any."""
        return next(
            (
                tool.arguments_model
                for tool in self._tool_registry.tools
                if tokens and tool.name == tokens[0]
            ),
            None,
        )

    def tools(self, context: CommandContext) -> None:
        """List all registered tools with their descriptions."""
        tools = self._tool_registry.tools
        if not tools:
            context.interaction.info("No tools registered.")
            return
        context.interaction.table(tools, title="Registered tools:")

    def call(
        self,
        context: CommandContext,
        name: Annotated[str, Field(description="Exact registered tool name.")],
        arguments: Annotated[
            tuple[str, ...],
            CommandRemainder(),
            Field(description="Command-like positional and name=value tool arguments."),
        ] = (),
    ) -> None:
        """Call a registered tool with command-like arguments."""
        result = self._tool_registry.command(
            name,
            arguments,
            interaction=context.interaction,
            instructions_manager=self._instructions_manager,
        )
        context.interaction.tool_result(result.output, result.presentation)
