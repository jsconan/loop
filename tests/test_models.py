"""Verify serialization and value semantics for loop models."""

import json

from loop import Message, Response, ToolCall, Usage


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
