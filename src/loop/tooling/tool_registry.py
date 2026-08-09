"""Register and dispatch typed functions exposed to an LLM."""

from collections.abc import Callable, Iterable
from typing import Any

from ..context import ToolContext
from ..interaction import Interaction
from ..models import ToolDefinition
from ..permissions import Capability, Decision, PermissionManager, PermissionRequest
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
        permission_manager (PermissionManager | None): Central policy manager guarding every call.
            Defaults to an in-memory, confirm-all manager.
    """

    _tools: dict[str, Tool]
    _interaction: Interaction | None
    _permission_manager: PermissionManager

    def __init__(
        self,
        tools: Iterable[Callable[..., Any]] | None = None,
        interaction: Interaction | None = None,
        permission_manager: PermissionManager | None = None,
    ) -> None:
        self._tools = {}
        self._interaction = interaction
        self._permission_manager = permission_manager or PermissionManager(interaction=interaction)
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
        self._permission_manager.interaction = interaction

    @property
    def permission_manager(self) -> PermissionManager:
        """Return the permission manager guarding dispatch.

        Returns:
            PermissionManager: Active centralized permission manager.
        """
        return self._permission_manager

    @permission_manager.setter
    def permission_manager(self, manager: PermissionManager) -> None:
        """Replace the permission manager guarding dispatch.

        Args:
            manager (PermissionManager): Manager to use for future calls.
        """
        self._permission_manager = manager

    def tool(
        self,
        function: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        capabilities: Iterable[Capability] | None = None,
        permission_resolver: Callable[[dict[str, Any]], Iterable[PermissionRequest]] | None = None,
    ) -> Callable[..., Any]:
        """Register a function, usable as ``@tool_registry.tool`` or with options.

        Args:
            function (Callable[..., Any] | None): Function to register when the decorator is used
                without options.
            name (str | None): Public tool name. Defaults to the function name.
            description (str | None): Public description. Defaults to the docstring summary.
            capabilities (Iterable[Capability] | None): Static authority required by each call.
                Defaults to the tool's inherited declaration or ``pure``.
            permission_resolver (Callable[[dict[str, Any]], Iterable[PermissionRequest]] | None):
                Optional resolver producing resource-specific requests from validated arguments.

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
            declared_capabilities = frozenset(
                capabilities
                or getattr(target, "__loop_capabilities__", None)
                or {Capability.PURE}
            )
            declared_resolver = (
                permission_resolver
                if permission_resolver is not None
                else getattr(target, "__loop_permission_resolver__", None)
            )
            self._tools[tool_name] = Tool(
                name=tool_name,
                description=description or get_tool_description(target),
                function=target,
                arguments_model=get_tool_arguments_model(target, tool_name),
                capabilities=declared_capabilities,
                permission_resolver=declared_resolver,
            )
            target.__loop_capabilities__ = declared_capabilities
            target.__loop_permission_resolver__ = declared_resolver
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
        permission_manager: PermissionManager | None = None,
    ) -> str:
        """Dispatch a synchronous tool call by registered name.

        Args:
            name (str): Registered tool name.
            arguments (str): JSON-encoded arguments supplied by the model.
            interaction (Interaction | None): Interaction for this invocation. Overrides the
                registry default.
            instructions_manager (InstructionsManager | None): Instruction manager active for the
                current conversation.
            permission_manager (PermissionManager | None): Invocation policy overriding the
                registry default.

        Returns:
            str: The serialized tool result or a model-readable error.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        tool = self._tools.get(name)
        if tool is None:
            return serialize_tool_error("unknown_tool", f"Tool '{name}' is not available.")
        validated, error = tool.validate_arguments(arguments)
        if error is not None:
            return error
        active_permissions = permission_manager or self._permission_manager
        denied, grants = self._authorize(tool, validated, interaction, active_permissions)
        if denied is not None:
            return denied
        context = self._context_for(
            tool, interaction, instructions_manager, active_permissions, grants
        )
        return tool.call(validated, context)

    async def call_async(
        self,
        name: str,
        arguments: str,
        *,
        interaction: Interaction | None = None,
        instructions_manager: InstructionsManager | None = None,
        permission_manager: PermissionManager | None = None,
    ) -> str:
        """Dispatch an asynchronous or synchronous tool call by registered name.

        Args:
            name (str): Registered tool name.
            arguments (str): JSON-encoded arguments supplied by the model.
            interaction (Interaction | None): Interaction for this invocation. Overrides the
                registry default.
            instructions_manager (InstructionsManager | None): Instruction manager active for the
                current conversation.
            permission_manager (PermissionManager | None): Invocation policy overriding the
                registry default.

        Returns:
            str: The serialized tool result or a model-readable error.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        tool = self._tools.get(name)
        if tool is None:
            return serialize_tool_error("unknown_tool", f"Tool '{name}' is not available.")
        validated, error = tool.validate_arguments(arguments)
        if error is not None:
            return error
        active_permissions = permission_manager or self._permission_manager
        denied, grants = self._authorize(tool, validated, interaction, active_permissions)
        if denied is not None:
            return denied
        context = self._context_for(
            tool, interaction, instructions_manager, active_permissions, grants
        )
        return await tool.call_async(validated, context)

    def _authorize(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        interaction: Interaction | None,
        permission_manager: PermissionManager,
    ) -> tuple[str | None, frozenset[PermissionRequest]]:
        """Return a serialized denial or the grants approved for a tool call."""
        active_interaction = interaction if interaction is not None else self._interaction
        previous = permission_manager.interaction
        permission_manager.interaction = active_interaction
        grants = set()
        try:
            for request in tool.permission_requests(arguments):
                result = permission_manager.authorize(request)
                if result.decision is Decision.DENY:
                    return (
                        serialize_tool_error(
                            "tool_call_denied",
                            f"Tool '{tool.name}' was not executed: {result.reason}",
                        ),
                        frozenset(),
                    )
                grants.add(request)
        finally:
            permission_manager.interaction = previous
        return None, frozenset(grants)

    def _context_for(
        self,
        tool: Tool,
        interaction: Interaction | None,
        instructions_manager: InstructionsManager | None,
        permission_manager: PermissionManager,
        grants: frozenset[PermissionRequest] = frozenset(),
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
            permission_manager=permission_manager,
            grants=grants,
        )


tool_registry = ToolRegistry()
