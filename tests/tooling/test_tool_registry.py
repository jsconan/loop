"""Tests for tool registration and routing."""

import asyncio
import importlib
import json
from functools import partial
from unittest.mock import Mock

import pytest

from loop import Capability, PermissionConfiguration, PermissionManager, PermissionMode, ToolContext
from loop.interaction import Interaction
from loop.skills import InstructionsManager
from loop.tooling import ToolRegistration, ToolRegistrationError, ToolRegistry
from loop.tooling import tool as declare_tool

tool_registry_module = importlib.import_module("loop.tooling.tool_registry")


def register(registry: ToolRegistry):
    """Register and return a simple documented function."""

    @declare_tool
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    registry.register(calculate)
    return calculate


def test_constructor_registers_in_order_and_exposes_sorted_snapshots():
    """Construction preserves definitions while public tool snapshots sort and isolate state."""

    @declare_tool
    def zebra() -> str:
        """Return the zebra result."""
        return "zebra"

    @declare_tool
    def alpha() -> str:
        """Return the alpha result."""
        return "alpha"

    permissions = PermissionManager(
        configuration=PermissionConfiguration(mode=PermissionMode.UNRESTRICTED)
    )
    registry = ToolRegistry([zebra, alpha], permission_manager=permissions)
    tools = registry.tools
    names = registry.names

    assert [definition.name for definition in registry.definitions()] == [
        "zebra",
        "alpha",
    ]
    assert [tool.name for tool in tools] == ["alpha", "zebra"]
    assert names == ["alpha", "zebra"]
    tools.clear()
    names.clear()
    assert registry.names == ["alpha", "zebra"]
    assert registry.call("zebra", "{}") == "zebra"


def test_register_resolves_a_tool_from_declared_metadata():
    """Registration resolves an immutable runtime tool from a passive declaration."""
    registry = ToolRegistry()

    function = register(registry)
    registered = registry.tools[0]

    assert registered.function is function
    assert registered.name == "calculate"
    assert registered.description == "Calculate a number."
    assert registered.arguments_model is not None
    assert registered.capabilities == frozenset({Capability.PURE})


def test_register_accepts_explicit_metadata_and_rejects_duplicate_names():
    """Declaration options supply metadata and duplicate public names remain invalid."""
    registry = ToolRegistry()

    @declare_tool(name="selected", description="Explicit")
    def first(number: int) -> int:
        return number

    registry.register(first)
    with pytest.raises(ToolRegistrationError, match="already registered"):
        registry.register(first)


def test_register_accepts_undeclared_functions_with_derived_defaults():
    """Any documented callable can become a pure tool without prior declaration."""

    def ordinary(value: int) -> int:
        """Return an ordinary value."""
        return value

    permissions = PermissionManager(
        configuration=PermissionConfiguration(mode=PermissionMode.UNRESTRICTED)
    )
    registry = ToolRegistry([ordinary], permission_manager=permissions)

    assert registry.names == ["ordinary"]
    assert registry.call("ordinary", '{"value": 3}') == "3"


def test_register_overrides_metadata_for_only_one_container():
    """Registration options customize one registry without changing a reusable declaration."""

    @declare_tool(name="declared", description="Declared", capabilities={Capability.PURE})
    def calculate(number: int) -> int:
        return number

    resolver = Mock(return_value=())
    customized = ToolRegistry()
    customized.register(
        calculate,
        name="customized",
        description="Customized",
        capabilities={Capability.NETWORK_READ},
        permission_resolver=resolver,
    )
    standard = ToolRegistry([calculate])

    custom_tool = customized.tools[0]
    standard_tool = standard.tools[0]
    assert (
        custom_tool.name,
        custom_tool.description,
        custom_tool.capabilities,
        custom_tool.permission_resolver,
    ) == (
        "customized",
        "Customized",
        frozenset({Capability.NETWORK_READ}),
        resolver,
    )
    assert (
        standard_tool.name,
        standard_tool.description,
        standard_tool.capabilities,
        standard_tool.permission_resolver,
    ) == ("declared", "Declared", frozenset({Capability.PURE}), None)


def test_register_can_remove_a_declared_permission_resolver():
    """An explicit null resolver removes inherited resource-specific policy in one registry."""
    resolver = Mock(return_value=())

    @declare_tool(permission_resolver=resolver)
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    inherited = ToolRegistry([calculate])
    overridden = ToolRegistry()
    overridden.register(calculate, permission_resolver=None)

    assert inherited.tools[0].permission_resolver is resolver
    assert overridden.tools[0].permission_resolver is None


def test_register_accepts_container_specific_registration_records():
    """Registration records apply their local metadata through the public register method."""

    def calculate(number: int) -> int:
        return number

    registry = ToolRegistry()
    registry.register(
        ToolRegistration(
            calculate,
            name="local_calculate",
            description="Calculate locally.",
            capabilities=frozenset(),
        )
    )

    registered = registry.tools[0]
    assert registered.name == "local_calculate"
    assert registered.description == "Calculate locally."
    assert registered.capabilities == frozenset()


def test_independent_registries_allow_the_same_name_and_isolate_runtime_state():
    """Separate containers can reuse names while retaining their own interaction and policy."""

    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    first_interaction = Mock(spec=Interaction)
    second_interaction = Mock(spec=Interaction)
    first_permissions = Mock(spec=PermissionManager)
    second_permissions = Mock(spec=PermissionManager)
    first = ToolRegistry(
        [calculate], interaction=first_interaction, permission_manager=first_permissions
    )
    second = ToolRegistry(
        [calculate], interaction=second_interaction, permission_manager=second_permissions
    )

    assert first.names == second.names == ["calculate"]
    assert first.interaction is first_interaction
    assert second.interaction is second_interaction
    assert first.permission_manager is first_permissions
    assert second.permission_manager is second_permissions


def test_registries_compose_different_subsets_of_reusable_functions():
    """Each container selects its own subset from the same available callables."""

    def first() -> str:
        """Return the first result."""
        return "first"

    def second() -> str:
        """Return the second result."""
        return "second"

    assert ToolRegistry([first]).names == ["first"]
    assert ToolRegistry([second]).names == ["second"]


def test_register_accepts_callable_objects():
    """Callable instances derive schemas, descriptions, names, and invocation behavior."""

    class Multiplier:
        """Multiply a number."""

        def __call__(self, number: int) -> int:
            return number * 2

    permissions = PermissionManager(
        configuration=PermissionConfiguration(mode=PermissionMode.UNRESTRICTED)
    )
    registry = ToolRegistry([Multiplier()], permission_manager=permissions)

    assert registry.names == ["Multiplier"]
    assert registry.call("Multiplier", '{"number": 3}') == "6"


def test_register_accepts_partials_with_explicit_local_metadata():
    """Partially bound functions expose only their remaining annotated parameters."""

    def add(first: int, second: int) -> int:
        return first + second

    permissions = PermissionManager(
        configuration=PermissionConfiguration(mode=PermissionMode.UNRESTRICTED)
    )
    registry = ToolRegistry(permission_manager=permissions)
    registry.register(partial(add, 2), name="add_two", description="Add two to a number.")

    assert registry.call("add_two", '{"second": 3}') == "5"


def test_register_dispatches_async_callable_objects_only_through_async_calls():
    """Coroutine-based callable instances are detected before synchronous invocation."""

    class AsyncMultiplier:
        """Multiply a number asynchronously."""

        async def __call__(self, number: int) -> int:
            return number * 2

    permissions = PermissionManager(
        configuration=PermissionConfiguration(mode=PermissionMode.UNRESTRICTED)
    )
    registry = ToolRegistry([AsyncMultiplier()], permission_manager=permissions)

    assert "async_tool_in_sync_loop" in registry.call("AsyncMultiplier", '{"number": 3}')
    assert asyncio.run(registry.call_async("AsyncMultiplier", '{"number": 3}')) == "6"


def test_definitions_preserve_registration_order():
    """Definitions retain the order in which tools join a registry."""
    registry = ToolRegistry()

    register(registry)

    @declare_tool(name="second", description="Second")
    def _second() -> None:
        pass

    registry.register(_second)

    assert [definition.name for definition in registry.definitions()] == ["calculate", "second"]
    assert registry.names == ["calculate", "second"]


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

    replacement = PermissionManager(
        configuration=PermissionConfiguration(mode=PermissionMode.UNRESTRICTED)
    )
    registry.permission_manager = replacement
    assert registry.permission_manager is replacement


def test_call_reports_unknown_tools(monkeypatch):
    """Synchronous routing serializes an unknown-tool error at the registry boundary."""
    serialize = Mock(return_value="unknown")
    monkeypatch.setattr(tool_registry_module, "serialize_tool_error", serialize)

    assert ToolRegistry().call("missing", "{}") == "unknown"
    serialize.assert_called_once_with("unknown_tool", "Tool 'missing' is not available.")


def test_call_routes_arguments_and_runtime_context():
    """Synchronous routing forwards arguments and builds context from the runtime interaction."""
    registry = ToolRegistry(interaction=Mock(spec=Interaction))
    contexts = []

    @declare_tool
    def calculate(context: ToolContext) -> str:
        """Return the current tool name."""
        contexts.append(context)
        return context.tool_name

    registry.register(calculate)
    runtime = Mock(spec=Interaction)
    manager = Mock(spec=InstructionsManager)

    assert (
        registry.call("calculate", "{}", interaction=runtime, instructions_manager=manager)
        == "calculate"
    )
    context = contexts[0]
    assert context.interaction is runtime
    assert context.tool_name == "calculate"
    assert context.instructions_manager is manager


def test_call_uses_default_or_no_context():
    """Synchronous routing uses the default interaction and omits context when none exists."""
    interaction = Mock(spec=Interaction)
    permissions = PermissionManager(
        configuration=PermissionConfiguration(mode=PermissionMode.UNRESTRICTED)
    )
    registry = ToolRegistry(interaction=interaction, permission_manager=permissions)
    seen = []

    @declare_tool
    def calculate(context: ToolContext) -> str:
        """Record the supplied context."""
        seen.append(context)
        return "result"

    registry.register(calculate)

    registry.call("calculate", "{}")
    assert seen[0].interaction is interaction
    registry.interaction = None
    assert "execution_failed" in registry.call("calculate", "{}")
    assert len(seen) == 1


def test_call_command_parses_model_parameters_before_shared_dispatch():
    """Command routing validates parameters while bypassing the permission policy."""
    permissions = Mock(spec=PermissionManager)
    registry = ToolRegistry(interaction=Mock(spec=Interaction), permission_manager=permissions)

    @declare_tool
    def describe(count: int, label: str = "default") -> dict:
        """Describe parsed values."""
        return {"count": count, "label": label}

    @declare_tool
    def has_no_policy(context: ToolContext) -> bool:
        """Report whether command dispatch omitted the permission policy."""
        return context.permission_manager is None

    registry.register(describe)
    registry.register(has_no_policy)

    assert json.loads(registry.command("describe", ("3", "label=two words"))) == {
        "count": 3,
        "label": "two words",
    }
    assert registry.command("has_no_policy", ()) == "true"
    permissions.authorize.assert_not_called()


def test_call_command_reports_unknown_tools_and_invalid_parameters():
    """Command routing serializes lookup, binding, and model validation failures."""
    registry = ToolRegistry()
    register(registry)

    assert "unknown_tool" in registry.command("missing", ())
    assert "invalid_arguments" in registry.command("calculate", ("unknown=1",))
    assert "argument_binding" in registry.command("calculate", ("unknown=1",))
    assert "invalid_arguments" in registry.command("calculate", ("not-an-integer",))


def test_call_async_reports_unknown_tools(monkeypatch):
    """Asynchronous routing serializes an unknown-tool error at the registry boundary."""
    monkeypatch.setattr(tool_registry_module, "serialize_tool_error", Mock(return_value="unknown"))

    assert asyncio.run(ToolRegistry().call_async("missing", "{}")) == "unknown"


def test_call_async_routes_arguments_and_context():
    """Asynchronous routing awaits the registered tool with an invocation context."""
    registry = ToolRegistry()
    contexts = []

    @declare_tool
    async def calculate(context: ToolContext) -> str:
        """Return the supplied context tool name."""
        contexts.append(context)
        return context.tool_name

    registry.register(calculate)
    interaction = Mock(spec=Interaction)

    result = asyncio.run(registry.call_async("calculate", "{}", interaction=interaction))

    assert result == "calculate"
    assert contexts[0].interaction is interaction


def test_call_async_returns_validation_and_permission_denials_before_invocation():
    """Async dispatch stops before invocation for invalid and rejected calls."""
    interaction = Mock(spec=Interaction)
    interaction.confirm.return_value = False
    registry = ToolRegistry(interaction=interaction)
    register(registry)

    invalid = asyncio.run(registry.call_async("calculate", "bad"))
    denied = asyncio.run(registry.call_async("calculate", '{"number": 1}'))

    assert "invalid_arguments" in invalid
    assert "tool_call_denied" in denied
