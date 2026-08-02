"""Register and dispatch typed functions exposed to an LLM."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from ..context import ToolContext
from ..models import ToolDefinition
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
    """

    name: str
    description: str
    function: Callable[..., Any]
    arguments_model: type[BaseModel]

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

    def call(self, arguments: str, context: ToolContext | None = None) -> str:
        """Validate JSON arguments, invoke the function, and serialize its result.

        Args:
            arguments (str): JSON-encoded arguments supplied by the model.
            context (ToolContext | None): Runtime context supplied to a context-aware tool.

        Returns:
            str: The serialized function result or a model-readable error.

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
            arguments (str): JSON-encoded arguments supplied by the model.
            context (ToolContext | None): Runtime context supplied to a context-aware tool.

        Returns:
            str: The serialized function result or a model-readable error.

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
            arguments (str): JSON-encoded arguments supplied by the model.

        Returns:
            tuple[dict[str, Any] | None, str | None]: A pair containing validated arguments and no
                error, or no arguments and an error.
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
