"""Provide utility functions for tool registration and dispatch."""

import inspect
import json
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

from ..interaction import ToolContext
from ..types import ToolRegistrationError


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
        raise ToolRegistrationError(f"Tool '{function.__name__}' must have a docstring.")
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
    hints = get_type_hints(function, include_extras=True)
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
    hints = get_type_hints(function, include_extras=True)
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
    """Convert a tool result to the string required by ``function_call_output``.

    Args:
        result (Any): Tool result to serialize.

    Returns:
        str: The original string, serialized Pydantic model, or JSON-encoded value.

    Raises:
        TypeError: If ``result`` contains a value that JSON cannot serialize.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return json.dumps(result)


def serialize_tool_error(kind: str, message: str, **details: Any) -> str:
    """Return a stable, model-readable error result.

    Args:
        kind (str): Machine-readable error category.
        message (str): Human-readable error description.
        **details (Any): Additional fields to include in the error object.

    Returns:
        str: A JSON-encoded error object.

    Raises:
        TypeError: If a detail value cannot be JSON serialized.
    """
    return json.dumps({"error": kind, "message": message, **details})
