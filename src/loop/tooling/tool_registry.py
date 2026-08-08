"""Register and dispatch typed functions exposed to an LLM."""

from collections.abc import Callable, Iterable
from typing import Any

from ..context import ToolContext
from ..interaction import Interaction
from ..models import ToolDefinition
from ..skills import InstructionsManager
from .tool import Tool
from .utils import (
    ToolRegistrationError,
    get_tool_arguments_model,
    get_tool_description,
    serialize_tool_error,
)


class ToolRegistry:
    """Collect tool declarations and route model calls to their implementations.

    Args:
        tools (Iterable[Callable[..., Any]] | None): Functions to register in iteration order, or
            ``None`` to construct an empty registry.
        interaction (Interaction | None): Default interaction used by context-aware tools when
            dispatch does not provide one, or ``None`` to require an invocation-specific
            interaction.
    """

    _tools: dict[str, Tool]
    _interaction: Interaction | None

    def __init__(
        self,
        tools: Iterable[Callable[..., Any]] | None = None,
        interaction: Interaction | None = None,
    ) -> None:
        self._tools = {}
        self._interaction = interaction
        for function in tools or ():
            self.tool(function)

    @property
    def interaction(self) -> Interaction | None:
        """Return the default interaction used during tool dispatch.

        Returns:
            Interaction | None: The default interaction, or ``None`` when none is configured.
        """
        return self._interaction

    @interaction.setter
    def interaction(self, interaction: Interaction | None) -> None:
        """Set or clear the default interaction used during tool dispatch.

        Args:
            interaction (Interaction | None): Default interaction to use, or ``None`` to clear it.
        """
        self._interaction = interaction

    def tool(
        self,
        function: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[..., Any]:
        """Register a function, usable as ``@tool_registry.tool`` or with options.

        Args:
            function (Callable[..., Any] | None): Function to register when the decorator is used
                without options.
            name (str | None): Public tool name. Defaults to the function name.
            description (str | None): Public description. Defaults to the docstring summary.

        Returns:
            Callable[..., Any]: The registered function, or a decorator when ``function`` is
                omitted.

        Raises:
            ToolRegistrationError: If the name is already registered, the function has no
                description, or its parameters cannot be represented by an arguments model.
        """

        def _register(target: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or target.__name__
            if tool_name in self._tools:
                raise ToolRegistrationError(f"Tool '{tool_name}' is already registered.")
            self._tools[tool_name] = Tool(
                name=tool_name,
                description=description or get_tool_description(target),
                function=target,
                arguments_model=get_tool_arguments_model(target, tool_name),
            )
            return target

        return _register(function) if function is not None else _register

    def definitions(self) -> list[ToolDefinition]:
        """Return definitions for all registered tools.

        Returns:
            list[ToolDefinition]: Function-tool definitions in registration order.
        """
        return [tool.definition() for tool in self._tools.values()]

    def call(
        self,
        name: str,
        arguments: str,
        *,
        interaction: Interaction | None = None,
        instructions_manager: InstructionsManager | None = None,
    ) -> str:
        """Dispatch a synchronous tool call by registered name.

        Args:
            name (str): Registered tool name.
            arguments (str): JSON-encoded arguments supplied by the model.
            interaction (Interaction | None): Interaction for this invocation. Overrides the
                registry default.
            instructions_manager (InstructionsManager | None): Instruction manager active for the
                current conversation.

        Returns:
            str: The serialized tool result or a model-readable error.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        tool = self._tools.get(name)
        if tool is None:
            return serialize_tool_error("unknown_tool", f"Tool '{name}' is not available.")
        return tool.call(
            arguments,
            self._context_for(tool, interaction, instructions_manager),
        )

    async def call_async(
        self,
        name: str,
        arguments: str,
        *,
        interaction: Interaction | None = None,
        instructions_manager: InstructionsManager | None = None,
    ) -> str:
        """Dispatch an asynchronous or synchronous tool call by registered name.

        Args:
            name (str): Registered tool name.
            arguments (str): JSON-encoded arguments supplied by the model.
            interaction (Interaction | None): Interaction for this invocation. Overrides the
                registry default.
            instructions_manager (InstructionsManager | None): Instruction manager active for the
                current conversation.

        Returns:
            str: The serialized tool result or a model-readable error.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        tool = self._tools.get(name)
        if tool is None:
            return serialize_tool_error("unknown_tool", f"Tool '{name}' is not available.")
        return await tool.call_async(
            arguments,
            self._context_for(tool, interaction, instructions_manager),
        )

    def _context_for(
        self,
        tool: Tool,
        interaction: Interaction | None,
        instructions_manager: InstructionsManager | None,
    ) -> ToolContext | None:
        """Build a tool context from the invocation override or registry default."""
        if interaction is None:
            interaction = self._interaction
        if interaction is None:
            return None
        return ToolContext(
            interaction=interaction,
            tool_name=tool.name,
            instructions_manager=instructions_manager,
        )


tool_registry = ToolRegistry()
