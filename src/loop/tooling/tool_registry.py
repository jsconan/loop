"""Register and dispatch typed functions exposed to an LLM."""

from collections.abc import Callable, Iterable
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from ..commands.models import CommandArgumentError
from ..commands.utils import parse_model_arguments
from ..constants import OMIT, Omit
from ..interaction import Interaction
from ..models import ToolDefinition
from ..permissions import (
    Capability,
    Decision,
    PermissionManager,
    PermissionRecorder,
    PermissionRequest,
)
from ..skills import InstructionsManager
from ..utils import callable_name
from .context import ToolContext
from .models import ToolRegistrationError
from .tool import PermissionResolver, Tool, ToolRegistration
from .utils import serialize_tool_error


class ToolRegistry:
    """Collect tool declarations and route model calls to their implementations.

    Args:
        tools (Iterable[Callable[..., Any] | ToolRegistration] | None): Functions or configured
            registrations to add in iteration order, or ``None`` to construct an empty registry.
            Functions may optionally carry metadata from the standalone ``@tool`` decorator.
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
        tools: Iterable[Callable[..., Any] | ToolRegistration] | None = None,
        interaction: Interaction | None = None,
        permission_manager: PermissionManager | None = None,
    ) -> None:
        self._tools = {}
        self._interaction = interaction
        self._permission_manager = permission_manager or PermissionManager(interaction=interaction)
        for tool in tools or ():
            self.register(tool)

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

    @property
    def tools(self) -> list[Tool]:
        """Return registered tools sorted alphabetically by name.

        Returns:
            list[Tool]: Registered tools.
        """
        return sorted(self._tools.values(), key=lambda tool: tool.name.casefold())

    @property
    def names(self) -> list[str]:
        """Return registered tool names sorted alphabetically.

        Returns:
            list[str]: Registered tool names.
        """
        return sorted(self._tools.keys(), key=str.casefold)

    def register(
        self,
        function: Callable[..., Any] | ToolRegistration,
        *,
        name: str | None = None,
        description: str | None = None,
        capabilities: Iterable[Capability] | None = None,
        permission_resolver: PermissionResolver | None | Omit = OMIT,
    ) -> None:
        """Create and register a tool from a callable or configured registration.

        Args:
            function (Callable[..., Any] | ToolRegistration): Function to expose as a tool, or a
                registration that supplies its local metadata.
            name (str | None): Container-specific public name. Defaults to the declared name or
                function name.
            description (str | None): Container-specific public description. Defaults to the
                declared description or docstring summary.
            capabilities (Iterable[Capability] | None): Container-specific static authority.
                Defaults to declared capabilities or ``pure``.
            permission_resolver (PermissionResolver | None | object): Container-specific resolver
                for resource permission requests. Omit it to inherit the declared resolver; pass
                ``None`` to remove one.

        Raises:
            ToolRegistrationError: If the resolved name is already registered, the function has
                no description, or its parameters cannot be represented by an arguments model.
        """
        if isinstance(function, ToolRegistration):
            registration = function
            function = registration.function
            name = name or registration.name
            description = description or registration.description
            capabilities = capabilities if capabilities is not None else registration.capabilities
            permission_resolver = (
                registration.permission_resolver
                if isinstance(permission_resolver, Omit)
                else permission_resolver
            )
        declared_tool = Tool.get_declaration(function)
        if declared_tool is None:
            declared_tool = Tool(function=function)
        tool_name = name or declared_tool.name or callable_name(function)
        if tool_name in self._tools:
            raise ToolRegistrationError(f"Tool '{tool_name}' is already registered.")
        self._tools[tool_name] = declared_tool.registered(
            name=name,
            description=description,
            capabilities=capabilities,
            permission_resolver=permission_resolver,
        )

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
        permission_recorder: PermissionRecorder | None = None,
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
            permission_recorder (PermissionRecorder | None): Invocation-scoped permission sink.

        Returns:
            str: The serialized tool result or a model-readable error.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        output, _ = self.call_with_timing(
            name,
            arguments,
            interaction=interaction,
            instructions_manager=instructions_manager,
            permission_manager=permission_manager,
            permission_recorder=permission_recorder,
        )
        return output

    def call_with_timing(
        self,
        name: str,
        arguments: str,
        *,
        interaction: Interaction | None = None,
        instructions_manager: InstructionsManager | None = None,
        permission_manager: PermissionManager | None = None,
        permission_recorder: PermissionRecorder | None = None,
    ) -> tuple[str, float]:
        """Dispatch a tool and measure only its function execution.

        Args:
            name (str): Registered tool name.
            arguments (str): JSON-encoded arguments supplied by the model.
            interaction (Interaction | None): Interaction for this invocation.
            instructions_manager (InstructionsManager | None): Active instruction manager.
            permission_manager (PermissionManager | None): Invocation permission policy.
            permission_recorder (PermissionRecorder | None): Invocation-scoped permission sink.

        Returns:
            tuple[str, float]: Serialized result and tool-function duration in seconds. Validation
                and authorization failures have a zero duration because no tool ran.

        Raises:
            ValueError: If the tool requires a context but none is provided.
        """
        tool = self._tools.get(name)
        if tool is None:
            return serialize_tool_error("unknown_tool", f"Tool '{name}' is not available."), 0
        validated, error = tool.validate_arguments(arguments)
        if error is not None:
            return error, 0
        active_permissions = permission_manager or self._permission_manager
        denied, grants = self._authorize(
            tool, validated, interaction, active_permissions, permission_recorder
        )
        if denied is not None:
            return denied, 0
        context = self._context_for(
            tool,
            interaction,
            instructions_manager,
            active_permissions,
            grants,
            permission_recorder,
        )
        started = perf_counter()
        output = tool.call(validated, context)
        return output, perf_counter() - started

    async def call_async(
        self,
        name: str,
        arguments: str,
        *,
        interaction: Interaction | None = None,
        instructions_manager: InstructionsManager | None = None,
        permission_manager: PermissionManager | None = None,
        permission_recorder: PermissionRecorder | None = None,
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
            permission_recorder (PermissionRecorder | None): Invocation-scoped permission sink.

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
        denied, grants = self._authorize(
            tool, validated, interaction, active_permissions, permission_recorder
        )
        if denied is not None:
            return denied
        context = self._context_for(
            tool,
            interaction,
            instructions_manager,
            active_permissions,
            grants,
            permission_recorder,
        )
        return await tool.call_async(validated, context)

    def command(
        self,
        name: str,
        arguments: list[str] | tuple[str, ...],
        *,
        interaction: Interaction | None = None,
        instructions_manager: InstructionsManager | None = None,
    ) -> str:
        """Dispatch a user-command tool call without permission evaluation.

        Args:
            name (str): Registered tool name.
            arguments (list[str] | tuple[str, ...]): Positional and ``name=value`` argument tokens.
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
        try:
            validated = parse_model_arguments(tool.arguments_model, arguments).model_dump()
        except (CommandArgumentError, ValidationError) as exc:
            details = (
                exc.errors(include_url=False)
                if isinstance(exc, ValidationError)
                else [{"type": "argument_binding", "msg": str(exc)}]
            )
            return serialize_tool_error(
                "invalid_arguments",
                f"Invalid arguments for tool '{name}'.",
                details=details,
            )
        context = self._context_for(
            tool,
            interaction,
            instructions_manager,
            permission_manager=None,
        )
        return tool.call(validated, context)

    def _authorize(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        interaction: Interaction | None,
        permission_manager: PermissionManager,
        permission_recorder: PermissionRecorder | None,
    ) -> tuple[str | None, frozenset[PermissionRequest]]:
        """Return a serialized denial or the grants approved for a tool call."""
        active_interaction = interaction if interaction is not None else self._interaction
        grants = set()
        for request in tool.permission_requests(arguments):
            result = permission_manager.authorize(
                request,
                interaction=active_interaction,
                recorder=permission_recorder,
            )
            if result.decision is Decision.DENY:
                return (
                    serialize_tool_error(
                        "tool_call_denied",
                        f"Tool '{tool.name}' was not executed: {result.reason}",
                    ),
                    frozenset(),
                )
            grants.add(request)
        return None, frozenset(grants)

    def _context_for(
        self,
        tool: Tool,
        interaction: Interaction | None,
        instructions_manager: InstructionsManager | None,
        permission_manager: PermissionManager | None,
        grants: frozenset[PermissionRequest] = frozenset(),
        permission_recorder: PermissionRecorder | None = None,
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
            permission_recorder=permission_recorder,
            grants=grants,
        )
