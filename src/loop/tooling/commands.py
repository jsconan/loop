"""Expose registered tools through user commands."""

from typing import Annotated, Literal

from pydantic import Field

from ..commands import CommandArgumentError, CommandContext, CommandRegistration, CommandRemainder
from ..completion import (
    CommandCompletion,
    CompletionProviderRegistration,
    CompletionValue,
    SchemaCompletionProviderRegistration,
)
from ..instructions import InstructionsManager
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
            CommandRegistration(
                self.tools,
                name="tools",
                completion=CommandCompletion(
                    values=(CompletionValue("call", "Call a registered tool."),),
                    children={
                        "call": CommandCompletion(
                            provider="tools",
                            next=CommandCompletion(schema_provider="tool_arguments"),
                        )
                    },
                ),
            ),
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
                if tokens and tool.name == tokens[-1]
            ),
            None,
        )

    def tools(
        self,
        context: CommandContext,
        action: Annotated[
            Literal["call"] | None,
            Field(description="Optional tooling action. Use 'call' to invoke a tool."),
        ] = None,
        name: Annotated[
            str | None,
            Field(description="Exact registered tool name required by the call action."),
        ] = None,
        arguments: Annotated[
            tuple[str, ...],
            CommandRemainder(),
            Field(description="Command-like positional and name=value tool arguments."),
        ] = (),
    ) -> None:
        """List registered tools or call one through the ``call`` action.

        Args:
            context (CommandContext): Interaction services for the command.
            action (Literal["call"] | None): Optional action. Omit it to list tools.
            name (str | None): Registered tool name required by the ``call`` action.
            arguments (tuple[str, ...]): Remaining positional and named tool arguments.

        Raises:
            CommandArgumentError: If the ``call`` action omits its required tool name.
        """
        if action == "call":
            if name is None:
                raise CommandArgumentError("The '/tools call' action requires a tool name.")
            self._call_tool(context, name, arguments)
            return
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
        self._call_tool(context, name, arguments)

    def _call_tool(
        self,
        context: CommandContext,
        name: str,
        arguments: tuple[str, ...],
    ) -> None:
        """Execute one explicitly requested tool and present its result."""
        result = self._tool_registry.command(
            name,
            arguments,
            interaction=context.interaction,
            instructions_manager=self._instructions_manager,
        )
        context.interaction.tool_result(result.output, result.presentation)
