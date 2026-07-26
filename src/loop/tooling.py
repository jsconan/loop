"""Typed registration and dispatch for functions exposed to an LLM."""

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, ValidationError, create_model


class ToolRegistrationError(ValueError):
    """Raised when a Python function cannot be registered as a tool."""


def _description(function: Callable[..., Any]) -> str:
    """Return the summary paragraph from a function's docstring.

    Args:
        function: Function whose docstring supplies the description.

    Returns:
        The docstring's first paragraph as a single line.

    Raises:
        ToolRegistrationError: If the function has no docstring.
    """
    docstring = inspect.getdoc(function)
    if not docstring:
        raise ToolRegistrationError(f"Tool '{function.__name__}' must have a docstring.")
    return docstring.split("\n\n", maxsplit=1)[0].replace("\n", " ")


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt Pydantic JSON Schema to OpenAI strict function-tool rules.

    Args:
        schema: Schema to modify recursively in place.

    Returns:
        The adapted schema.
    """
    definitions = schema.get("$defs", {})
    for definition in definitions.values():
        _strict_schema(definition)

    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)

    properties = schema.get("properties")
    if isinstance(properties, dict):
        schema["required"] = list(properties)
        for property_schema in properties.values():
            _strict_schema(property_schema)

    items = schema.get("items")
    if isinstance(items, dict):
        _strict_schema(items)

    for keyword in ("anyOf", "oneOf", "allOf"):
        for alternative in schema.get(keyword, []):
            if isinstance(alternative, dict):
                _strict_schema(alternative)

    if schema.get("default", object()) is None:
        schema.pop("default")
    return schema


def _arguments_model(function: Callable[..., Any], tool_name: str) -> type[BaseModel]:
    """Build a validating Pydantic model from a function signature.

    Args:
        function: Function whose parameters define the model fields.
        tool_name: Public tool name used in the model and error messages.

    Returns:
        A Pydantic model that validates the function's arguments.

    Raises:
        ToolRegistrationError: If a parameter kind is unsupported or lacks a type annotation.
    """
    signature = inspect.signature(function)
    hints = get_type_hints(function, include_extras=True)
    fields: dict[str, tuple[Any, Any]] = {}

    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise ToolRegistrationError(
                f"Tool '{tool_name}' has unsupported parameter '{parameter.name}'."
            )
        if parameter.name not in hints:
            raise ToolRegistrationError(
                f"Tool '{tool_name}' parameter '{parameter.name}' needs a type annotation."
            )

        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (hints[parameter.name], default)

    model_name = "".join(part.title() for part in tool_name.split("_")) + "Arguments"
    return create_model(
        model_name,
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _serialize(result: Any) -> str:
    """Convert a tool result to the string required by ``function_call_output``.

    Args:
        result: Tool result to serialize.

    Returns:
        The original string, serialized Pydantic model, or JSON-encoded value.

    Raises:
        TypeError: If ``result`` contains a value that JSON cannot serialize.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return json.dumps(result)


def _error(kind: str, message: str, **details: Any) -> str:
    """Return a stable, model-readable error result.

    Args:
        kind: Machine-readable error category.
        message: Human-readable error description.
        **details: Additional fields to include in the error object.

    Returns:
        A JSON-encoded error object.

    Raises:
        TypeError: If a detail value cannot be JSON serialized.
    """
    return json.dumps({"error": kind, "message": message, **details})


@dataclass(frozen=True)
class Tool:
    """A function together with its LLM declaration and argument validator.

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
            "parameters": _strict_schema(self.arguments_model.model_json_schema()),
            "strict": True,
        }

    def call(self, arguments: str) -> str:
        """Validate JSON arguments, invoke the function, and serialize its result.

        Args:
            arguments: JSON-encoded arguments supplied by the model.

        Returns:
            The serialized function result or a model-readable error.
        """
        validated, validation_error = self._validate_arguments(arguments)
        if validation_error is not None:
            return validation_error

        if inspect.iscoroutinefunction(self.function):
            return _error(
                "async_tool_in_sync_loop",
                f"Tool '{self.name}' must be called through call_async().",
            )

        try:
            result = self.function(**validated)
            return _serialize(result)
        except Exception as exc:  # pylint: disable=broad-except
            return _error("execution_failed", f"Tool '{self.name}' failed: {exc}")

    async def call_async(self, arguments: str) -> str:
        """Validate and invoke either an asynchronous or synchronous function.

        Args:
            arguments: JSON-encoded arguments supplied by the model.

        Returns:
            The serialized function result or a model-readable error.
        """
        validated, validation_error = self._validate_arguments(arguments)
        if validation_error is not None:
            return validation_error

        try:
            result = self.function(**validated)
            if inspect.isawaitable(result):
                result = await result
            return _serialize(result)
        except Exception as exc:  # pylint: disable=broad-except
            return _error("execution_failed", f"Tool '{self.name}' failed: {exc}")

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
            return None, _error(
                "invalid_arguments",
                f"Invalid arguments for tool '{self.name}'.",
                details=exc.errors(include_url=False),
            )
        return validated.model_dump(), None


class ToolRegistry:
    """Collect tool declarations and route model calls to their implementations."""

    _tools: dict[str, Tool]

    def __init__(self) -> None:
        self._tools = {}

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
                description=description or _description(target),
                function=target,
                arguments_model=_arguments_model(target, tool_name),
            )
            return target

        return _register(function) if function is not None else _register

    def schemas(self) -> list[dict[str, Any]]:
        """Return declarations for all registered tools.

        Returns:
            Function-tool declarations in registration order.
        """
        return [tool.schema() for tool in self._tools.values()]

    def call(self, name: str, arguments: str) -> str:
        """Dispatch a synchronous tool call by registered name.

        Args:
            name: Registered tool name.
            arguments: JSON-encoded arguments supplied by the model.

        Returns:
            The serialized tool result or a model-readable error.
        """
        tool = self._tools.get(name)
        if tool is None:
            return _error("unknown_tool", f"Tool '{name}' is not available.")
        return tool.call(arguments)

    async def call_async(self, name: str, arguments: str) -> str:
        """Dispatch an asynchronous or synchronous tool call by registered name.

        Args:
            name: Registered tool name.
            arguments: JSON-encoded arguments supplied by the model.

        Returns:
            The serialized tool result or a model-readable error.
        """
        tool = self._tools.get(name)
        if tool is None:
            return _error("unknown_tool", f"Tool '{name}' is not available.")
        return await tool.call_async(arguments)
