"""Tests for passive tool declaration, validation, and dispatch."""

import asyncio
import importlib
import json
from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from loop import (
    Action,
    Operation,
    OperationPlan,
    SessionTarget,
    ToolRegistrationError,
    ToolRegistry,
    ToolResultPresentation,
    ToolResultPresentationSpec,
    tool,
)
from loop.tooling import Tool, ToolContext

tool_module = importlib.import_module("loop.tooling.tool")


class Arguments(BaseModel):
    """Validate the integer argument used by tool tests."""

    number: int


def make_tool(function=None) -> Tool:
    """Build a tool with the shared argument model."""
    if function is None:
        function = Mock(return_value=2)
    return Tool(
        function=function,
        name="calculate",
        description="Calculate a value.",
        arguments_model=Arguments,
    )


def test_tool_returns_the_original_function_with_pure_defaults():
    """Bare declaration preserves callability and supplies pure registration metadata."""

    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    declared = tool(calculate)
    registered = ToolRegistry([declared]).tools[0]

    assert declared is calculate
    assert calculate(3) == 3
    assert registered.name == "calculate"
    assert registered.actions == frozenset()


def test_plan_rejects_operations_outside_the_declared_action_bound():
    """A planner cannot silently expand the authority declared by its tool."""

    def planner(arguments):
        return OperationPlan(
            arguments=arguments,
            operations=(
                Operation(
                    tool_id="",
                    action=Action.SESSION_MUTATE,
                    target=SessionTarget(identifier="state"),
                ),
            ),
        )

    planned = Tool(
        function=Mock(),
        name="calculate",
        description="Calculate a value.",
        arguments_model=Arguments,
        operation_planner=planner,
    )

    with pytest.raises(ValueError, match="planned undeclared actions: session.mutate"):
        planned.plan({"number": 3})


def test_tool_options_preserve_an_explicitly_empty_capability_set():
    """Configured declarations retain names, descriptions, and empty capability collections."""

    @tool(name="selected", description="Selected tool.", actions=())
    def calculate(number: int) -> int:
        return number

    registered = ToolRegistry([calculate]).tools[0]

    assert registered.name == "selected"
    assert registered.description == "Selected tool."
    assert registered.actions == frozenset()


def test_tools_are_immutable():
    """Tool declarations cannot be modified after construction."""
    declaration = Tool(function=lambda: None)

    with pytest.raises(FrozenInstanceError):
        declaration.name = "changed"


def test_execution_selects_presentation_from_arguments_and_raw_result():
    """Dynamic presentation selectors receive canonical arguments and unserialized results."""
    selected = ToolResultPresentationSpec(kind=ToolResultPresentation.LIST)
    selector = Mock(return_value=selected)

    @tool(result_presentation=selector)
    def calculate(number: int) -> dict:
        """Calculate a structured value."""
        return {"values": [number]}

    registered = ToolRegistry([calculate]).tools[0]

    execution = registered.execute({"number": 3})

    assert json.loads(execution.output) == {"values": [3]}
    assert execution.presentation is selected
    selector.assert_called_once_with({"number": 3}, {"values": [3]})


@pytest.mark.parametrize("selector", [Mock(side_effect=RuntimeError), Mock(return_value=None)])
def test_execution_falls_back_when_dynamic_presentation_selection_fails(selector):
    """Presentation metadata failures cannot turn successful tool calls into tool errors."""

    @tool(result_presentation=selector)
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number * 2

    execution = ToolRegistry([calculate]).tools[0].execute({"number": 3})

    assert execution.output == "6"
    assert execution.presentation.kind is ToolResultPresentation.RAW


def test_passive_tools_require_registration_for_model_operations():
    """Passive definitions reject definition and argument-validation operations."""
    declaration = Tool(function=lambda: None)

    with pytest.raises(ValueError, match="must be registered"):
        declaration.definition()
    with pytest.raises(ValueError, match="must be registered"):
        declaration.validate_arguments("{}")


def test_tool_rejects_redeclaring_the_same_function():
    """A function has one unambiguous passive declaration."""

    @tool
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    with pytest.raises(ToolRegistrationError, match="already declared"):
        tool(name="other")(calculate)


def test_definition_adapts_the_argument_model(monkeypatch):
    """Definition exposes neutral tool metadata and delegates schema adaptation."""
    adapt = Mock(return_value={"adapted": True})
    monkeypatch.setattr(tool_module, "get_tool_schema", adapt)

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
    monkeypatch.setattr(tool_module, "takes_tool_context", Mock(return_value=False))
    monkeypatch.setattr(tool_module, "serialize_tool_result", serialize)

    assert make_tool(function).call({"number": 3}) == "serialized"
    function.assert_called_once_with(number=3)
    serialize.assert_called_once_with(4)


def test_call_injects_required_context(monkeypatch):
    """Context-aware dispatch injects context and reports a missing context as an execution error."""
    function = Mock(return_value="done")
    serialize_error = Mock(return_value="error")
    monkeypatch.setattr(tool_module, "takes_tool_context", Mock(return_value=True))
    monkeypatch.setattr(tool_module, "serialize_tool_result", lambda result: result)
    monkeypatch.setattr(tool_module, "serialize_tool_error", serialize_error)
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
    monkeypatch.setattr(tool_module, "serialize_tool_error", serialize_error)

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
    monkeypatch.setattr(tool_module, "serialize_tool_error", serialize_error)

    assert make_tool(calculate).call({"number": 3}) == "async error"
    serialize_error.assert_called_once_with(
        "async_tool_in_sync_loop",
        "Tool 'calculate' must be called through call_async().",
    )


def test_call_serializes_execution_failures(monkeypatch):
    """Synchronous application failures become model-readable execution errors."""
    function = Mock(side_effect=RuntimeError("boom"))
    serialize_error = Mock(return_value="failed")
    monkeypatch.setattr(tool_module, "takes_tool_context", Mock(return_value=False))
    monkeypatch.setattr(tool_module, "serialize_tool_error", serialize_error)

    assert make_tool(function).call({"number": 3}) == "failed"
    serialize_error.assert_called_once_with("execution_failed", "Tool 'calculate' failed: boom")


def test_call_async_supports_sync_and_awaitable_results(monkeypatch):
    """Asynchronous dispatch serializes both immediate and awaitable application results."""

    async def calculate(number: int) -> int:
        return number * 2

    monkeypatch.setattr(tool_module, "takes_tool_context", Mock(return_value=False))
    monkeypatch.setattr(tool_module, "serialize_tool_result", lambda result: str(result))

    assert asyncio.run(make_tool(Mock(return_value=4)).call_async({"number": 3})) == "4"
    assert asyncio.run(make_tool(calculate).call_async({"number": 3})) == "6"


def test_call_async_handles_execution_failures(monkeypatch):
    """Asynchronous dispatch returns structured errors for application failures."""
    function = Mock(side_effect=RuntimeError("boom"))
    serialize_error = Mock(side_effect=lambda kind, message, **details: json.dumps({"error": kind}))
    monkeypatch.setattr(tool_module, "takes_tool_context", Mock(return_value=False))
    monkeypatch.setattr(tool_module, "serialize_tool_error", serialize_error)
    tool = make_tool(function)

    assert json.loads(asyncio.run(tool.call_async({"number": 3})))["error"] == "execution_failed"
