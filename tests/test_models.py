"""Verify serialization and value semantics for loop models."""

import json

from loop import Message, Reasoning, Response, ResponseMetadata, ToolCall, ToolResult, Usage


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
