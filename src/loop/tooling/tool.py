"""Declare, validate, and invoke typed functions exposed to an LLM."""

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any, overload

from pydantic import BaseModel, ValidationError

from ..constants import OMIT, Omit
from ..models import ToolDefinition
from ..permissions import Capability, PermissionRequest
from ..utils import callable_name
from .context import ToolContext
from .utils import (
    ToolRegistrationError,
    get_tool_arguments_model,
    get_tool_description,
    get_tool_schema,
    is_async_callable,
    serialize_tool_error,
    serialize_tool_result,
    takes_tool_context,
)

PermissionResolver = Callable[[dict[str, Any]], Iterable[PermissionRequest]]

_TOOL_ATTR = "__loop_tool__"


@dataclass(frozen=True)
class Tool:
    """Describe and, once registered, invoke an LLM-callable function.

    Args:
        function (Callable[..., Any]): Python function invoked for the tool.
        name (str | None): Public name exposed to the model, or ``None`` to use the function name.
        description (str | None): Public description, or ``None`` to use the docstring summary.
        capabilities (frozenset[Capability]): Static authority required by every call.
        permission_resolver (PermissionResolver | None): Optional resolver producing
            resource-specific permission requests from validated arguments.
        arguments_model (type[BaseModel] | None): Pydantic model used to validate arguments after
            registration, or ``None`` for a passive declaration.
    """

    function: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    capabilities: frozenset[Capability] = frozenset({Capability.PURE})
    permission_resolver: PermissionResolver | None = None
    arguments_model: type[BaseModel] | None = None

    def registered(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        capabilities: Iterable[Capability] | None = None,
        permission_resolver: PermissionResolver | None | Omit = OMIT,
    ) -> Tool:
        """Return a registry-ready copy with resolved metadata and argument validation.

        Args:
            name (str | None): Container-specific public name, or ``None`` to inherit or derive it.
            description (str | None): Container-specific description, or ``None`` to inherit or
                derive it.
            capabilities (Iterable[Capability] | None): Container-specific static authority, or
                ``None`` to inherit.
            permission_resolver (PermissionResolver | None | Omit): Container-specific resolver.
                Omit it to inherit; pass ``None`` to remove one.

        Returns:
            Tool: Immutable, fully resolved tool for one registry.
        """
        resolved_name = name or self.name or callable_name(self.function)
        return replace(
            self,
            name=resolved_name,
            description=description or self.description or get_tool_description(self.function),
            capabilities=(
                frozenset(capabilities) if capabilities is not None else self.capabilities
            ),
            permission_resolver=(
                self.permission_resolver
                if isinstance(permission_resolver, Omit)
                else permission_resolver
            ),
            arguments_model=get_tool_arguments_model(self.function, resolved_name),
        )

    def definition(self) -> ToolDefinition:
        """Return the function-tool definition.

        Returns:
            ToolDefinition: The strict function-tool definition.

        Raises:
            ValueError: If the tool has not been registered.
        """
        if self.arguments_model is None or self.name is None or self.description is None:
            raise ValueError("A tool must be registered before it can be defined.")
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=get_tool_schema(self.arguments_model.model_json_schema()),
        )

    def call(self, arguments: dict[str, Any], context: ToolContext | None = None) -> str:
        """Invoke a synchronous registered tool with validated arguments.

        Args:
            arguments (dict[str, Any]): Validated keyword arguments.
            context (ToolContext | None): Runtime tool context.

        Returns:
            str: Serialized tool result or model-readable error.
        """
        if is_async_callable(self.function):
            return serialize_tool_error(
                "async_tool_in_sync_loop",
                f"Tool '{self._name_for_error()}' must be called through call_async().",
            )
        try:
            return serialize_tool_result(self._call_function(arguments, context))
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
            return serialize_tool_error(
                "execution_failed", f"Tool '{self._name_for_error()}' failed: {exc}"
            )

    async def call_async(
        self, arguments: dict[str, Any], context: ToolContext | None = None
    ) -> str:
        """Invoke any registered tool asynchronously with validated arguments.

        Args:
            arguments (dict[str, Any]): Validated keyword arguments.
            context (ToolContext | None): Runtime tool context.

        Returns:
            str: Serialized tool result or model-readable error.
        """
        try:
            result = self._call_function(arguments, context)
            if inspect.isawaitable(result):
                result = await result
            return serialize_tool_result(result)
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
            return serialize_tool_error(
                "execution_failed", f"Tool '{self._name_for_error()}' failed: {exc}"
            )

    def validate_arguments(self, arguments: str) -> tuple[dict[str, Any] | None, str | None]:
        """Return validated keyword arguments or a model-readable error.

        Args:
            arguments (str): JSON-encoded arguments supplied by the model.

        Returns:
            tuple[dict[str, Any] | None, str | None]: Validated arguments and no error, or no
                arguments and an error.

        Raises:
            ValueError: If the tool has not been registered.
        """
        if self.arguments_model is None:
            raise ValueError("A tool must be registered before it can validate arguments.")
        try:
            validated = self.arguments_model.model_validate_json(arguments or "{}")
        except ValidationError as exc:
            return None, serialize_tool_error(
                "invalid_arguments",
                f"Invalid arguments for tool '{self._name_for_error()}'.",
                details=exc.errors(include_url=False),
            )
        return validated.model_dump(), None

    def permission_requests(self, arguments: dict[str, Any]) -> tuple[PermissionRequest, ...]:
        """Return normalized permission requests for validated arguments.

        Args:
            arguments (dict[str, Any]): Validated tool arguments.

        Returns:
            tuple[PermissionRequest, ...]: Static or argument-specific permission requests.
        """
        if self.permission_resolver is not None:
            return tuple(
                request.model_copy(update={"tool_name": self._name_for_error()})
                for request in self.permission_resolver(arguments)
            )
        return tuple(
            PermissionRequest(tool_name=self._name_for_error(), capability=capability)
            for capability in sorted(self.capabilities, key=str)
        )

    def _name_for_error(self) -> str:
        """Return the resolved name or a stable fallback for diagnostics."""
        return self.name or callable_name(self.function)

    def _call_function(self, arguments: dict[str, Any], context: ToolContext | None) -> Any:
        """Call the function with an explicit context when it requests one."""
        if takes_tool_context(self.function):
            if context is None:
                raise ValueError(f"Tool '{self._name_for_error()}' requires a ToolContext.")
            return self.function(context, **arguments)
        return self.function(**arguments)

    @staticmethod
    def get_declaration(function: Callable[..., Any]) -> Tool | None:
        """Return the passive tool declaration attached to a callable."""
        declaration = getattr(function, _TOOL_ATTR, None)
        return declaration if isinstance(declaration, Tool) else None

    @staticmethod
    def set_declaration(function: Callable[..., Any], declaration: Tool) -> None:
        """Attach a passive tool declaration to a callable."""
        setattr(function, _TOOL_ATTR, declaration)


@dataclass(frozen=True)
class ToolRegistration:
    """Configure one callable for registration in a specific container.

    Args:
        function (Callable[..., Any]): Callable to expose as a tool.
        name (str | None): Container-specific name, or ``None`` to inherit or derive it.
        description (str | None): Container-specific description, or ``None`` to inherit or
            derive it.
        capabilities (frozenset[Capability] | None): Container-specific capabilities, or ``None``
            to inherit declared capabilities.
        permission_resolver (PermissionResolver | None | Omit): Container-specific resolver.
            Omit it to inherit a declared resolver; pass ``None`` to remove one.
    """

    function: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    capabilities: frozenset[Capability] | None = None
    permission_resolver: PermissionResolver | None | Omit = OMIT


@overload
def tool[ToolFunction: Callable[..., Any]](function: ToolFunction, /) -> ToolFunction: ...


@overload
def tool[ToolFunction: Callable[..., Any]](
    function: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    capabilities: Iterable[Capability] | None = None,
    permission_resolver: PermissionResolver | None = None,
) -> Callable[[ToolFunction], ToolFunction]: ...


def tool[ToolFunction: Callable[..., Any]](
    function: ToolFunction | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    capabilities: Iterable[Capability] | None = None,
    permission_resolver: PermissionResolver | None = None,
) -> ToolFunction | Callable[[ToolFunction], ToolFunction]:
    """Declare a function as an LLM-callable tool without registering it.

    Args:
        function (ToolFunction | None): Function to declare when used without options.
        name (str | None): Public tool name. Defaults to the function name at registration.
        description (str | None): Public description. Defaults to the docstring summary at
            registration.
        capabilities (Iterable[Capability] | None): Static authority required by every call.
            Defaults to ``pure``.
        permission_resolver (PermissionResolver | None): Optional resolver producing
            resource-specific requests from validated arguments.

    Returns:
        ToolFunction | Callable[[ToolFunction], ToolFunction]: The unchanged declared function,
            or a decorator when ``function`` is omitted.
    """

    def _declare(target: ToolFunction) -> ToolFunction:
        if Tool.get_declaration(target) is not None:
            raise ToolRegistrationError(f"Tool '{callable_name(target)}' is already declared.")
        Tool.set_declaration(
            target,
            Tool(
                function=target,
                name=name,
                description=description,
                capabilities=frozenset({Capability.PURE} if capabilities is None else capabilities),
                permission_resolver=permission_resolver,
            ),
        )
        return target

    return _declare(function) if function is not None else _declare
