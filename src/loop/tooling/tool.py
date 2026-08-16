"""Register and dispatch typed functions exposed to an LLM."""

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from ..context import ToolContext
from ..models import ToolDefinition
from ..permissions import Capability, PermissionRequest
from .utils import (
    get_tool_schema,
    serialize_tool_error,
    serialize_tool_result,
    takes_tool_context,
)


@dataclass(frozen=True)
class Tool:
    """Represent a function with its LLM declaration and argument validator.

    Args:
        name (str): Public name exposed to the model.
        description (str): Description exposed in the tool declaration.
        function (Callable[..., Any]): Python function invoked for the tool.
        arguments_model (type[BaseModel]): Pydantic model used to validate arguments.
        capabilities (frozenset[Capability]): Static authority required by every call.
        permission_resolver (Callable[[dict[str, Any]], Iterable[PermissionRequest]] | None):
            Optional resolver producing argument-specific permission requests.
    """

    name: str
    description: str
    function: Callable[..., Any]
    arguments_model: type[BaseModel]
    capabilities: frozenset[Capability] = frozenset({Capability.PURE})
    permission_resolver: Callable[[dict[str, Any]], Iterable[PermissionRequest]] | None = None

    def definition(self) -> ToolDefinition:
        """Return the function-tool definition.

        Returns:
            ToolDefinition: The strict function-tool definition.
        """
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=get_tool_schema(self.arguments_model.model_json_schema()),
        )

    def call(self, arguments: dict[str, Any], context: ToolContext | None = None) -> str:
        """Invoke a synchronous tool with validated arguments.

        Args:
            arguments (dict[str, Any]): Validated keyword arguments.
            context (ToolContext | None): Runtime tool context.

        Returns:
            str: Serialized tool result or model-readable error.
        """
        if inspect.iscoroutinefunction(self.function):
            return serialize_tool_error(
                "async_tool_in_sync_loop",
                f"Tool '{self.name}' must be called through call_async().",
            )
        try:
            return serialize_tool_result(self._call_function(arguments, context))
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-except
            return serialize_tool_error("execution_failed", f"Tool '{self.name}' failed: {exc}")

    async def call_async(
        self, arguments: dict[str, Any], context: ToolContext | None = None
    ) -> str:
        """Invoke any tool asynchronously with validated arguments.

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
            return serialize_tool_error("execution_failed", f"Tool '{self.name}' failed: {exc}")

    def validate_arguments(self, arguments: str) -> tuple[dict[str, Any] | None, str | None]:
        """Return validated keyword arguments or a model-readable error.

        Args:
            arguments (str): JSON-encoded arguments supplied by the model.

        Returns:
            tuple[dict[str, Any] | None, str | None]: Validated arguments and no error, or no
                arguments and an error.
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

    def permission_requests(self, arguments: dict[str, Any]) -> tuple[PermissionRequest, ...]:
        """Return normalized permission requests for validated arguments.

        Args:
            arguments (dict[str, Any]): Validated tool arguments.

        Returns:
            tuple[PermissionRequest, ...]: Static or argument-specific permission requests.
        """
        if self.permission_resolver is not None:
            return tuple(
                request.model_copy(update={"tool_name": self.name})
                for request in self.permission_resolver(arguments)
            )
        return tuple(
            PermissionRequest(tool_name=self.name, capability=capability)
            for capability in sorted(self.capabilities, key=str)
        )

    def _call_function(self, arguments: dict[str, Any], context: ToolContext | None) -> Any:
        """Call the function with an explicit context when it requests one."""
        if takes_tool_context(self.function):
            if context is None:
                raise ValueError(f"Tool '{self.name}' requires a ToolContext.")
            return self.function(context, **arguments)
        return self.function(**arguments)
