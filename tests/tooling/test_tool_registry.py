"""Tests for tool registration and routing."""

import asyncio
import importlib
from unittest.mock import AsyncMock, Mock

import pytest

from loop.context import ToolContext
from loop.interaction import Interaction
from loop.skills import InstructionsManager
from loop.tooling import ToolRegistrationError, ToolRegistry

tool_registry_module = importlib.import_module("loop.tooling.tool_registry")


def register(registry: ToolRegistry):
    """Register and return a simple documented function."""

    @registry.tool
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    return calculate


def test_constructor_registers_tools_in_iteration_order():
    """Construction registers each supplied function through the normal tool path."""

    def first() -> str:
        """Return the first result."""
        return "first"

    def second() -> str:
        """Return the second result."""
        return "second"

    registry = ToolRegistry([first, second])

    assert [definition.name for definition in registry.definitions()] == [
        "first",
        "second",
    ]
    assert registry.call("first", "{}") == "first"


def test_tool_registers_a_function_with_derived_metadata(monkeypatch):
    """The decorator builds and retains a tool while returning the original function."""
    tool = Mock()
    tool_type = Mock(return_value=tool)
    arguments_model = Mock()
    monkeypatch.setattr(tool_registry_module, "Tool", tool_type)
    monkeypatch.setattr(tool_registry_module, "get_tool_description", Mock(return_value="Doc"))
    monkeypatch.setattr(
        tool_registry_module, "get_tool_arguments_model", Mock(return_value=arguments_model)
    )
    registry = ToolRegistry()

    function = register(registry)

    tool_type.assert_called_once_with(
        name="calculate",
        description="Doc",
        function=function,
        arguments_model=arguments_model,
    )


def test_tool_accepts_explicit_metadata_and_rejects_duplicate_names(monkeypatch):
    """Decorator options override metadata and duplicate public names remain invalid."""
    monkeypatch.setattr(tool_registry_module, "Tool", Mock())
    monkeypatch.setattr(tool_registry_module, "get_tool_arguments_model", Mock())
    registry = ToolRegistry()

    @registry.tool(name="selected", description="Explicit")
    def first(number: int) -> int:
        return number

    with pytest.raises(ToolRegistrationError, match="already registered"):
        registry.tool(first, name="selected", description="Explicit")


def test_definitions_delegate_to_registered_tools(monkeypatch):
    """Definition collection preserves registration order and delegates to each tool."""
    first = Mock()
    first.definition.return_value = {"name": "first"}
    second = Mock()
    second.definition.return_value = {"name": "second"}
    tool_type = Mock(side_effect=[first, second])
    monkeypatch.setattr(tool_registry_module, "Tool", tool_type)
    monkeypatch.setattr(tool_registry_module, "get_tool_description", Mock(return_value="Doc"))
    monkeypatch.setattr(tool_registry_module, "get_tool_arguments_model", Mock())
    registry = ToolRegistry()

    register(registry)

    @registry.tool(name="second", description="Second")
    def second() -> None:
        pass

    assert registry.definitions() == [{"name": "first"}, {"name": "second"}]


def test_interaction_property_can_be_replaced_and_cleared():
    """The registry exposes mutable default interaction configuration."""
    first = Mock(spec=Interaction)
    second = Mock(spec=Interaction)
    registry = ToolRegistry(interaction=first)

    assert registry.interaction is first
    registry.interaction = second
    assert registry.interaction is second
    registry.interaction = None
    assert registry.interaction is None


def test_call_reports_unknown_tools(monkeypatch):
    """Synchronous routing serializes an unknown-tool error at the registry boundary."""
    serialize = Mock(return_value="unknown")
    monkeypatch.setattr(tool_registry_module, "serialize_tool_error", serialize)

    assert ToolRegistry().call("missing", "{}") == "unknown"
    serialize.assert_called_once_with("unknown_tool", "Tool 'missing' is not available.")


def test_call_routes_arguments_and_runtime_context(monkeypatch):
    """Synchronous routing forwards arguments and builds context from the runtime interaction."""
    tool = Mock(name="tool", name_attribute="ignored")
    tool.name = "calculate"
    tool.call.return_value = "result"
    monkeypatch.setattr(tool_registry_module, "Tool", Mock(return_value=tool))
    monkeypatch.setattr(tool_registry_module, "get_tool_description", Mock(return_value="Doc"))
    monkeypatch.setattr(tool_registry_module, "get_tool_arguments_model", Mock())
    registry = ToolRegistry(interaction=Mock(spec=Interaction))
    register(registry)
    runtime = Mock(spec=Interaction)
    manager = Mock(spec=InstructionsManager)

    assert (
        registry.call("calculate", "{}", interaction=runtime, instructions_manager=manager)
        == "result"
    )
    context = tool.call.call_args.args[1]
    assert context == ToolContext(runtime, "calculate", manager)


def test_call_uses_default_or_no_context(monkeypatch):
    """Synchronous routing uses the default interaction and omits context when none exists."""
    tool = Mock()
    tool.name = "calculate"
    monkeypatch.setattr(tool_registry_module, "Tool", Mock(return_value=tool))
    monkeypatch.setattr(tool_registry_module, "get_tool_description", Mock(return_value="Doc"))
    monkeypatch.setattr(tool_registry_module, "get_tool_arguments_model", Mock())
    interaction = Mock(spec=Interaction)
    registry = ToolRegistry(interaction=interaction)
    register(registry)

    registry.call("calculate", "{}")
    assert tool.call.call_args.args[1] == ToolContext(interaction, "calculate")
    registry.interaction = None
    registry.call("calculate", "{}")
    assert tool.call.call_args.args[1] is None


def test_call_async_reports_unknown_tools(monkeypatch):
    """Asynchronous routing serializes an unknown-tool error at the registry boundary."""
    monkeypatch.setattr(tool_registry_module, "serialize_tool_error", Mock(return_value="unknown"))

    assert asyncio.run(ToolRegistry().call_async("missing", "{}")) == "unknown"


def test_call_async_routes_arguments_and_context(monkeypatch):
    """Asynchronous routing awaits the registered tool with an invocation context."""
    tool = Mock()
    tool.name = "calculate"
    tool.call_async = AsyncMock(return_value="result")
    monkeypatch.setattr(tool_registry_module, "Tool", Mock(return_value=tool))
    monkeypatch.setattr(tool_registry_module, "get_tool_description", Mock(return_value="Doc"))
    monkeypatch.setattr(tool_registry_module, "get_tool_arguments_model", Mock())
    registry = ToolRegistry()
    register(registry)
    interaction = Mock(spec=Interaction)

    result = asyncio.run(registry.call_async("calculate", "{}", interaction=interaction))

    assert result == "result"
    assert tool.call_async.call_args.args[1] == ToolContext(interaction, "calculate")
