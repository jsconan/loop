"""Register and dispatch typed functions exposed to an LLM."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from .interaction import Interaction, ToolContext
from .skills import SkillManager
from .types.tooling import ToolRegistrationError
from .utils.tooling import (
    get_tool_arguments_model,
    get_tool_description,
    get_tool_schema,
    serialize_tool_error,
    serialize_tool_result,
    takes_tool_context,
)


@dataclass(frozen=True)
class Tool:
    """Represent a function with its LLM declaration and argument validator.

    Attributes:
        name: Public name exposed to the model.
        description: Description exposed in the tool declaration.
        function: Python function invoked for the tool.
        arguments_model: Pydantic model used to validate arguments.
    """

    name: str
    description: str
    function: Callable[..., Any]
    arguments_model: type[BaseModel]

    def schema(self) -> dict[str, Any]:
        """Return this tool in the flat Responses API function-tool format.

        Returns:
            The strict function-tool declaration.
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": get_tool_schema(self.arguments_model.model_json_schema()),
            "strict": True,
        }

    def call(self, arguments: str, context: ToolContext | None = None) -> str:
        """Validate JSON arguments, invoke the function, and serialize its result.

        Args:
            arguments: JSON-encoded arguments supplied by the model.
            context: Runtime context supplied to a context-aware tool.

        Returns:
            The serialized function result or a model-readable error.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        validated, validation_error = self._validate_arguments(arguments)
        if validation_error is not None:
            return validation_error

        if inspect.iscoroutinefunction(self.function):
            return serialize_tool_error(
                "async_tool_in_sync_loop",
                f"Tool '{self.name}' must be called through call_async().",
            )

        try:
            result = self._invoke(validated, context)
            return serialize_tool_result(result)
        except Exception as exc:  # pylint: disable=broad-except
            return serialize_tool_error("execution_failed", f"Tool '{self.name}' failed: {exc}")

    async def call_async(self, arguments: str, context: ToolContext | None = None) -> str:
        """Validate and invoke either an asynchronous or synchronous function.

        Args:
            arguments: JSON-encoded arguments supplied by the model.
            context: Runtime context supplied to a context-aware tool.

        Returns:
            The serialized function result or a model-readable error.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        validated, validation_error = self._validate_arguments(arguments)
        if validation_error is not None:
            return validation_error

        try:
            result = self._invoke(validated, context)
            if inspect.isawaitable(result):
                result = await result
            return serialize_tool_result(result)
        except Exception as exc:  # pylint: disable=broad-except
            return serialize_tool_error("execution_failed", f"Tool '{self.name}' failed: {exc}")

    def _invoke(self, arguments: dict[str, Any], context: ToolContext | None) -> Any:
        """Invoke the function with an explicit context when it requests one."""
        if takes_tool_context(self.function):
            if context is None:
                raise ValueError(f"Tool '{self.name}' requires a ToolContext.")
            return self.function(context, **arguments)
        return self.function(**arguments)

    def _validate_arguments(
        self,
        arguments: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Return validated keyword arguments or a model-readable error.

        Args:
            arguments: JSON-encoded arguments supplied by the model.

        Returns:
            A pair containing validated arguments and no error, or no arguments and an error.
        """
        try:
            validated = self.arguments_model.model_validate_json(arguments or "{}")
        except ValidationError as exc:
            return None, serialize_tool_error(
                "invalid_arguments",
                f"Invalid arguments for tool '{self.name}'.",
                details=exc.errors(include_url=False),
            )
        return validated.model_dump(), None


class ToolRegistry:
    """Collect tool declarations and route model calls to their implementations.

    Args:
        interaction: Default interaction used by context-aware tools when dispatch does not
            provide one.
    """

    _tools: dict[str, Tool]
    _interaction: Interaction | None

    def __init__(self, interaction: Interaction | None = None) -> None:
        self._tools = {}
        self._interaction = interaction

    @property
    def interaction(self) -> Interaction | None:
        """Return the default interaction used during tool dispatch."""
        return self._interaction

    @interaction.setter
    def interaction(self, interaction: Interaction | None) -> None:
        """Set or clear the default interaction used during tool dispatch."""
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
            function: Function to register when the decorator is used without options.
            name: Public tool name. Defaults to the function name.
            description: Public description. Defaults to the docstring summary.

        Returns:
            The registered function, or a decorator when ``function`` is omitted.

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

    def schemas(self) -> list[dict[str, Any]]:
        """Return declarations for all registered tools.

        Returns:
            Function-tool declarations in registration order.
        """
        return [tool.schema() for tool in self._tools.values()]

    def call(
        self,
        name: str,
        arguments: str,
        *,
        interaction: Interaction | None = None,
        skill_manager: SkillManager | None = None,
    ) -> str:
        """Dispatch a synchronous tool call by registered name.

        Args:
            name: Registered tool name.
            arguments: JSON-encoded arguments supplied by the model.
            interaction: Interaction for this invocation. Overrides the registry default.
            skill_manager: Skill manager active for the current conversation.

        Returns:
            The serialized tool result or a model-readable error.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        tool = self._tools.get(name)
        if tool is None:
            return serialize_tool_error("unknown_tool", f"Tool '{name}' is not available.")
        return tool.call(
            arguments,
            self._context_for(tool, interaction, skill_manager),
        )

    async def call_async(
        self,
        name: str,
        arguments: str,
        *,
        interaction: Interaction | None = None,
        skill_manager: SkillManager | None = None,
    ) -> str:
        """Dispatch an asynchronous or synchronous tool call by registered name.

        Args:
            name: Registered tool name.
            arguments: JSON-encoded arguments supplied by the model.
            interaction: Interaction for this invocation. Overrides the registry default.
            skill_manager: Skill manager active for the current conversation.

        Returns:
            The serialized tool result or a model-readable error.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        tool = self._tools.get(name)
        if tool is None:
            return serialize_tool_error("unknown_tool", f"Tool '{name}' is not available.")
        return await tool.call_async(
            arguments,
            self._context_for(tool, interaction, skill_manager),
        )

    def _context_for(
        self,
        tool: Tool,
        interaction: Interaction | None,
        skill_manager: SkillManager | None,
    ) -> ToolContext | None:
        """Build a tool context from the invocation override or registry default."""
        if interaction is None:
            interaction = self._interaction
        if interaction is None:
            return None
        return ToolContext(
            interaction=interaction,
            tool_name=tool.name,
            skill_manager=skill_manager,
        )


tool_registry = ToolRegistry()
