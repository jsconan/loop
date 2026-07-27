"""Tests for tooling utility functions."""

import json
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from loop.types.tooling import ToolRegistrationError
from loop.utils.tooling import (
    get_tool_arguments_model,
    get_tool_description,
    get_tool_schema,
    serialize_tool_error,
    serialize_tool_result,
)

# pylint: disable=unused-argument, redefined-outer-name


def test_get_tool_description_returns_flattened_summary():
    """Only the normalized first docstring paragraph becomes the description."""

    def documented() -> None:
        """First line
        continues here.

        Extra details are excluded.
        """

    assert get_tool_description(documented) == "First line continues here."


def test_get_tool_description_rejects_missing_docstring():
    """A callable without documentation cannot provide tool metadata."""

    def undocumented() -> None:
        pass

    with pytest.raises(ToolRegistrationError, match="must have a docstring"):
        get_tool_description(undocumented)


def test_get_tool_arguments_model_builds_strict_fields_and_defaults():
    """Annotations, constraints, defaults, and strict extra handling form the argument model."""

    def choose(
        value: Annotated[int, Field(gt=0)],
        note: str | None = None,
    ) -> None:
        pass

    model = get_tool_arguments_model(choose, "choose_value")

    assert model.__name__ == "ChooseValueArguments"
    assert model.model_validate({"value": 2}).model_dump() == {"value": 2, "note": None}
    with pytest.raises(ValueError):
        model.model_validate({"value": 0})
    with pytest.raises(ValueError):
        model.model_validate({"value": 1, "unexpected": True})


def test_get_tool_arguments_model_omits_method_self_parameter():
    """The bound tool instance is not exposed as a model-supplied argument."""

    def contextual(self, value: int) -> None:
        pass

    model = get_tool_arguments_model(contextual, "contextual")

    assert model.model_json_schema()["properties"] == {
        "value": {"title": "Value", "type": "integer"}
    }
    assert model.model_validate({"value": 2}).model_dump() == {"value": 2}


@pytest.mark.parametrize(
    ("function", "parameter"),
    [
        (lambda value, /: None, "value"),
        (lambda *values: None, "values"),
        (lambda **values: None, "values"),
    ],
)
def test_get_tool_arguments_model_rejects_unsupported_parameter_kinds(function, parameter):
    """Parameters that cannot be supplied as JSON keyword fields are rejected."""
    with pytest.raises(ToolRegistrationError, match=f"unsupported parameter '{parameter}'"):
        get_tool_arguments_model(function, "invalid")


def test_get_tool_arguments_model_rejects_missing_annotation():
    """Every supported parameter requires an explicit type annotation."""

    def untyped(value):
        return value

    with pytest.raises(ToolRegistrationError, match="needs a type annotation"):
        get_tool_arguments_model(untyped, "untyped")


def test_get_tool_schema_recursively_applies_strict_object_rules():
    """Definitions, properties, arrays, and composition alternatives are adapted recursively."""
    schema = {
        "type": "object",
        "$defs": {"Nested": {"type": "object", "properties": {"name": {"type": "string"}}}},
        "properties": {
            "nested": {"$ref": "#/$defs/Nested"},
            "items": {"type": "array", "items": {"type": "object", "properties": {}}},
            "choice": {
                "anyOf": [{"type": "object", "properties": {"a": {"type": "string"}}}, True],
                "oneOf": [{"type": "object", "properties": {"b": {"type": "integer"}}}],
                "allOf": [{"type": "object", "properties": {"c": {"type": "boolean"}}}],
            },
            "optional": {"type": ["string", "null"], "default": None},
            "retained": {"type": "integer", "default": 3},
        },
    }

    result = get_tool_schema(schema)

    assert result is schema
    assert result["additionalProperties"] is False
    assert result["required"] == ["nested", "items", "choice", "optional", "retained"]
    assert result["$defs"]["Nested"]["additionalProperties"] is False
    assert result["properties"]["items"]["items"]["additionalProperties"] is False
    assert result["properties"]["choice"]["anyOf"][0]["additionalProperties"] is False
    assert result["properties"]["choice"]["oneOf"][0]["additionalProperties"] is False
    assert result["properties"]["choice"]["allOf"][0]["additionalProperties"] is False
    assert "default" not in result["properties"]["optional"]
    assert result["properties"]["retained"]["default"] == 3


def test_serialize_tool_result_supports_strings_models_and_json_values():
    """Supported result categories use the wire representation expected by tool outputs."""

    class Result(BaseModel):
        value: int

    assert serialize_tool_result("plain") == "plain"
    assert json.loads(serialize_tool_result(Result(value=4))) == {"value": 4}
    assert json.loads(serialize_tool_result({"ready": True})) == {"ready": True}


def test_serialize_tool_result_rejects_non_json_values():
    """Unsupported result values preserve the JSON encoder's explicit failure."""
    with pytest.raises(TypeError):
        serialize_tool_result(object())


def test_serialize_tool_error_includes_kind_message_and_details():
    """Structured errors contain stable required fields and arbitrary serializable details."""
    assert json.loads(
        serialize_tool_error("invalid_arguments", "Invalid payload.", details=[{"field": "value"}])
    ) == {
        "error": "invalid_arguments",
        "message": "Invalid payload.",
        "details": [{"field": "value"}],
    }


def test_serialize_tool_error_rejects_non_json_details():
    """Unsupported detail values preserve the JSON encoder's explicit failure."""
    with pytest.raises(TypeError):
        serialize_tool_error("failure", "Failed.", detail=object())
