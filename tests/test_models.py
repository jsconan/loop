"""Verify serialization and value semantics for loop models."""

import json

import pytest
from pydantic import BaseModel, Field

from loop import (
    Message,
    Reasoning,
    Response,
    ResponseMetadata,
    StructuredOutputFormat,
    StructuredOutputValidationError,
    ToolCall,
    ToolResult,
    Usage,
)


class Person(BaseModel):
    """Represent a validated person response."""

    name: str
    age: int


class Directory(BaseModel):
    """Represent nested structures used to verify strict schema generation."""

    owner: Person = Field(description="Directory owner.")
    members: list[Person]
    contact: Person | str
    nickname: str | None = None


def test_models_serialize_nested_values_to_python_and_json() -> None:
    """Models recursively expose their content as Python and JSON values."""
    response = Response(
        answer="done",
        reasoning="thought",
        tool_calls=(ToolCall(call_id="call", name="demo", arguments="{}"),),
        items=(Message(role="assistant", content="done"),),
        usage=Usage(total_tokens=12),
        model="served-model",
    )

    expected = {
        "answer": "done",
        "reasoning": "thought",
        "tool_calls": [{"call_id": "call", "name": "demo", "arguments": "{}", "id": None}],
        "items": [{"role": "assistant", "content": "done"}],
        "usage": {"total_tokens": 12},
        "model": "served-model",
    }
    assert response.model_dump(mode="json") == expected
    assert json.loads(response.model_dump_json()) == expected


def test_conversation_items_expose_optional_response_metadata() -> None:
    """Every conversation item can retain response-level usage without changing local inputs."""
    metadata = ResponseMetadata(
        response_id="response_1",
        model="served-model",
        usage=Usage(input_tokens=10, output_tokens=2, total_tokens=12),
    )
    items = (
        Message(role="assistant", content="done", metadata=metadata),
        Reasoning(content="thought", metadata=metadata),
        ToolCall(call_id="call", name="demo", arguments="{}", metadata=metadata),
        ToolResult(call_id="call", output="done", metadata=metadata),
    )

    assert all(item.metadata == metadata for item in items)
    assert Message(role="user", content="hello").model_dump() == {
        "role": "user",
        "content": "hello",
    }
    assert items[0].model_dump()["metadata"] == {
        "response_id": "response_1",
        "model": "served-model",
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    }


def test_structured_output_format_retains_and_uses_a_pydantic_model() -> None:
    """Model-derived formats generate schemas and return validated model instances."""
    output_format = StructuredOutputFormat.from_model(
        Person,
        name="person_record",
        description="A person.",
    )

    assert output_format.name == "person_record"
    assert output_format.description == "A person."
    assert output_format.strict is True
    assert output_format.model is Person
    assert output_format.schema["additionalProperties"] is False
    assert output_format.schema["required"] == ["name", "age"]
    assert output_format.validate('{"name":"Ada","age":36}') == Person(name="Ada", age=36)
    assert StructuredOutputFormat.from_model(Person).name == "Person"
    assert StructuredOutputFormat.from_model(Person, strict=False).schema == (
        Person.model_json_schema()
    )


def test_structured_output_format_normalizes_nested_pydantic_schema() -> None:
    """Strict model schemas normalize definitions, references, arrays, unions, and defaults."""
    schema = StructuredOutputFormat.from_model(Directory).schema

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["owner", "members", "contact", "nickname"]
    assert schema["$defs"]["Person"]["additionalProperties"] is False
    assert schema["properties"]["owner"]["description"] == "Directory owner."
    assert "$ref" not in schema["properties"]["owner"]
    assert schema["properties"]["members"]["items"] == {"$ref": "#/$defs/Person"}
    assert len(schema["properties"]["contact"]["anyOf"]) == 2
    assert "default" not in schema["properties"]["nickname"]


def test_structured_output_format_uses_a_callback_or_returns_decoded_json() -> None:
    """Raw schemas optionally transform decoded JSON through their validator callback."""
    schema = {"type": "array", "items": {"type": "integer"}}
    output_format = StructuredOutputFormat(
        name="numbers",
        schema=schema,
        validator=tuple,
    )

    assert output_format.validate("[1, 2]") == (1, 2)
    assert StructuredOutputFormat(name="numbers", schema=schema).validate("[1, 2]") == [1, 2]


@pytest.mark.parametrize(
    "output_format",
    [
        StructuredOutputFormat.from_model(Person),
        StructuredOutputFormat(
            name="person",
            schema={"type": "object"},
            validator=lambda value: int(value["age"]),
        ),
    ],
)
def test_structured_output_format_wraps_decoding_and_validation_errors(output_format) -> None:
    """Malformed or type-invalid output raises the public structured validation error."""
    text = "not-json" if output_format.model is not None else '{"age":"invalid"}'

    with pytest.raises(StructuredOutputValidationError, match=output_format.name):
        output_format.validate(text)


def test_structured_output_format_rejects_invalid_configuration() -> None:
    """Formats require a name and exactly one optional validation mechanism."""
    with pytest.raises(ValueError, match="name must not be empty"):
        StructuredOutputFormat(name="", schema={})
    with pytest.raises(ValueError, match="both a model and validator"):
        StructuredOutputFormat(
            name="person",
            schema={},
            model=Person,
            validator=int,
        )
