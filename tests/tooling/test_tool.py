"""Tests for individual tool validation and dispatch."""

import asyncio
import json
from unittest.mock import Mock

from pydantic import BaseModel

from loop.context import ToolContext
from loop.tooling import Tool


class Arguments(BaseModel):
    """Validate the integer argument used by tool tests."""

    number: int


def make_tool(function=None) -> Tool:
    """Build a tool with the shared argument model."""
    if function is None:
        function = Mock(return_value=2)
    return Tool("calculate", "Calculate a value.", function, Arguments)


def test_definition_adapts_the_argument_model(monkeypatch):
    """Definition exposes neutral tool metadata and delegates schema adaptation."""
    adapt = Mock(return_value={"adapted": True})
    monkeypatch.setattr("loop.tooling.tool.get_tool_schema", adapt)

    definition = make_tool().definition()

    assert definition.name == "calculate"
    assert definition.description == "Calculate a value."
    assert definition.parameters == {"adapted": True}
    assert definition.strict is True
    adapt.assert_called_once_with(Arguments.model_json_schema())


def test_validate_arguments_prepares_call_arguments():
    """Validation converts serialized model input into typed call arguments."""
    arguments, error = make_tool().validate_arguments('{"number": 3}')

    assert arguments == {"number": 3}
    assert error is None


def test_call_invokes_and_serializes_validated_arguments(monkeypatch):
    """Synchronous dispatch invokes and serializes arguments prepared by validation."""
    function = Mock(return_value=4)
    serialize = Mock(return_value="serialized")
    monkeypatch.setattr("loop.tooling.tool.takes_tool_context", Mock(return_value=False))
    monkeypatch.setattr("loop.tooling.tool.serialize_tool_result", serialize)

    assert make_tool(function).call({"number": 3}) == "serialized"
    function.assert_called_once_with(number=3)
    serialize.assert_called_once_with(4)


def test_call_injects_required_context(monkeypatch):
    """Context-aware dispatch injects context and reports a missing context as an execution error."""
    function = Mock(return_value="done")
    serialize_error = Mock(return_value="error")
    monkeypatch.setattr("loop.tooling.tool.takes_tool_context", Mock(return_value=True))
    monkeypatch.setattr("loop.tooling.tool.serialize_tool_result", lambda result: result)
    monkeypatch.setattr("loop.tooling.tool.serialize_tool_error", serialize_error)
    context = Mock(spec=ToolContext)
    tool = make_tool(function)

    assert tool.call({"number": 3}, context) == "done"
    function.assert_called_once_with(context, number=3)
    assert tool.call({"number": 3}) == "error"
    serialize_error.assert_called_once_with(
        "execution_failed",
        "Tool 'calculate' failed: Tool 'calculate' requires a ToolContext.",
    )


def test_validate_arguments_returns_errors_without_invoking(monkeypatch):
    """Invalid model arguments return structured details without reaching application code."""
    function = Mock()
    serialize_error = Mock(return_value="invalid")
    monkeypatch.setattr("loop.tooling.tool.serialize_tool_error", serialize_error)

    arguments, error = make_tool(function).validate_arguments("not json")

    assert arguments is None
    assert error == "invalid"
    function.assert_not_called()
    assert serialize_error.call_args.args == (
        "invalid_arguments",
        "Invalid arguments for tool 'calculate'.",
    )
    assert serialize_error.call_args.kwargs["details"]


def test_call_rejects_coroutine_functions_in_sync_dispatch(monkeypatch):
    """Synchronous dispatch directs coroutine functions to the asynchronous API."""

    async def calculate(number: int) -> int:
        return number * 2

    serialize_error = Mock(return_value="async error")
    monkeypatch.setattr("loop.tooling.tool.serialize_tool_error", serialize_error)

    assert make_tool(calculate).call({"number": 3}) == "async error"
    serialize_error.assert_called_once_with(
        "async_tool_in_sync_loop",
        "Tool 'calculate' must be called through call_async().",
    )


def test_call_serializes_execution_failures(monkeypatch):
    """Synchronous application failures become model-readable execution errors."""
    function = Mock(side_effect=RuntimeError("boom"))
    serialize_error = Mock(return_value="failed")
    monkeypatch.setattr("loop.tooling.tool.takes_tool_context", Mock(return_value=False))
    monkeypatch.setattr("loop.tooling.tool.serialize_tool_error", serialize_error)

    assert make_tool(function).call({"number": 3}) == "failed"
    serialize_error.assert_called_once_with("execution_failed", "Tool 'calculate' failed: boom")


def test_call_async_supports_sync_and_awaitable_results(monkeypatch):
    """Asynchronous dispatch serializes both immediate and awaitable application results."""

    async def calculate(number: int) -> int:
        return number * 2

    monkeypatch.setattr("loop.tooling.tool.takes_tool_context", Mock(return_value=False))
    monkeypatch.setattr("loop.tooling.tool.serialize_tool_result", lambda result: str(result))

    assert asyncio.run(make_tool(Mock(return_value=4)).call_async({"number": 3})) == "4"
    assert asyncio.run(make_tool(calculate).call_async({"number": 3})) == "6"


def test_call_async_handles_execution_failures(monkeypatch):
    """Asynchronous dispatch returns structured errors for application failures."""
    function = Mock(side_effect=RuntimeError("boom"))
    serialize_error = Mock(side_effect=lambda kind, message, **details: json.dumps({"error": kind}))
    monkeypatch.setattr("loop.tooling.tool.takes_tool_context", Mock(return_value=False))
    monkeypatch.setattr("loop.tooling.tool.serialize_tool_error", serialize_error)
    tool = make_tool(function)

    assert json.loads(asyncio.run(tool.call_async({"number": 3})))["error"] == "execution_failed"
