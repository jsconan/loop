"""Provide utility functions for tool registration and dispatch."""

import inspect
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

from ..errors import Problem
from ..utils import callable_hints, callable_name
from .context import ToolContext
from .models import ToolRegistrationError


def is_async_callable(function: Callable[..., Any]) -> bool:
    """Return whether invoking a function or callable object produces a coroutine directly.

    Args:
        function (Callable[..., Any]): Callable whose implementation should be inspected.

    Returns:
        bool: Whether the callable is implemented as a coroutine function.
    """
    return inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(function.__call__)


def get_tool_description(function: Callable[..., Any]) -> str:
    """Return the summary paragraph from a function's docstring.

    Args:
        function (Callable[..., Any]): Function whose docstring supplies the description.

    Returns:
        str: The docstring's first paragraph as a single line.

    Raises:
        ToolRegistrationError: If the function has no docstring.
    """
    docstring = inspect.getdoc(function)
    if not docstring:
        raise ToolRegistrationError(f"Tool '{callable_name(function)}' must have a docstring.")
    return docstring.split("\n\n", maxsplit=1)[0].replace("\n", " ")


def get_tool_arguments_model(function: Callable[..., Any], tool_name: str) -> type[BaseModel]:
    """Build a validating Pydantic model from a function signature.

    Args:
        function (Callable[..., Any]): Function whose parameters define the model fields.
        tool_name (str): Public tool name used in the model and error messages.

    Returns:
        type[BaseModel]: A Pydantic model that validates the function's arguments.

    Raises:
        ToolRegistrationError: If a parameter kind is unsupported or lacks a type annotation.
    """
    signature = inspect.signature(function)
    hints = callable_hints(function)
    fields = {}

    parameters = list(signature.parameters.values())
    if parameters and hints.get(parameters[0].name) is ToolContext:
        parameters = parameters[1:]

    for parameter in parameters:
        if hints.get(parameter.name) is ToolContext:
            raise ToolRegistrationError(
                f"Tool '{tool_name}' must declare ToolContext as its first parameter."
            )
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


def takes_tool_context(function: Callable[..., Any]) -> bool:
    """Determine whether a function requests an injected tool context.

    Args:
        function (Callable[..., Any]): Function whose first parameter may request a tool context.

    Returns:
        bool: Whether the first parameter is annotated as ``ToolContext``.
    """
    parameters = list(inspect.signature(function).parameters.values())
    if not parameters:
        return False
    hints = callable_hints(function)
    return hints.get(parameters[0].name) is ToolContext


def get_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt Pydantic JSON Schema to OpenAI strict function-tool rules.

    Args:
        schema (dict[str, Any]): Schema to modify recursively in place.

    Returns:
        dict[str, Any]: The adapted schema.
    """
    definitions = schema.get("$defs", {})
    for definition in definitions.values():
        get_tool_schema(definition)

    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)

    properties = schema.get("properties")
    if isinstance(properties, dict):
        schema["required"] = list(properties)
        for property_schema in properties.values():
            get_tool_schema(property_schema)

    items = schema.get("items")
    if isinstance(items, dict):
        get_tool_schema(items)

    for keyword in ("anyOf", "oneOf", "allOf"):
        for alternative in schema.get(keyword, []):
            if isinstance(alternative, dict):
                get_tool_schema(alternative)

    if schema.get("default", object()) is None:
        schema.pop("default")
    return schema


def serialize_tool_result(result: Any) -> str:
    """Wrap a successful tool value in the application result envelope.

    Args:
        result (Any): Tool result to serialize.

    Returns:
        str: JSON-encoded successful result envelope.

    Raises:
        TypeError: If ``result`` contains a value that JSON cannot serialize.
    """
    if isinstance(result, Problem):
        return serialize_tool_problem(result)
    if isinstance(result, BaseModel):
        result = result.model_dump(mode="json")
    return json.dumps({"ok": True, "result": result})


def serialize_tool_problem(problem: Problem) -> str:
    """Wrap a problem in the application result envelope.

    Args:
        problem (Problem): Structured tool failure to serialize.

    Returns:
        str: JSON-encoded failed result envelope.
    """
    return json.dumps({"ok": False, "problem": problem.model_dump(mode="json")})
