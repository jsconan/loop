"""Tests for tool registration and routing."""

import asyncio
import importlib
import json
from functools import partial
from unittest.mock import Mock

import pytest

from loop import (
    Action,
    NetworkTarget,
    Operation,
    OperationPlan,
    PermissionManager,
    Problem,
    ProblemException,
    SessionTarget,
    ToolResultPresentation,
    ToolResultPresentationSpec,
)
from loop.interaction import Interaction
from loop.skills import InstructionsManager
from loop.tooling import ToolContext, ToolRegistration, ToolRegistrationError, ToolRegistry
from loop.tooling import tool as declare_tool

tool_registry_module = importlib.import_module("loop.tooling.tool_registry")


def result_value(output: str):
    """Return the successful value from a tool result envelope."""
    payload = json.loads(output)
    assert payload["ok"] is True
    return payload["result"]


def planner_for(action: Action):
    """Return a concrete operation planner for an authority-bearing test tool."""

    def plan(arguments):
        target = (
            NetworkTarget(url="https://example.com", origin="https://example.com")
            if action is Action.NETWORK_REQUEST
            else SessionTarget(identifier="test-state")
        )
        return OperationPlan(
            arguments=arguments,
            operations=(Operation(tool_id="", action=action, target=target),),
        )

    return plan


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

    registry = ToolRegistry([zebra, alpha])
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
    assert result_value(registry.call("zebra", "{}")) == "zebra"


def test_register_resolves_a_tool_from_declared_metadata():
    """Registration resolves an immutable runtime tool from a passive declaration."""
    registry = ToolRegistry()

    function = register(registry)
    registered = registry.tools[0]

    assert registered.function is function
    assert registered.name == "calculate"
    assert registered.description == "Calculate a number."
    assert registered.arguments_model is not None
    assert registered.actions == frozenset()


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

    registry = ToolRegistry([ordinary])

    assert registry.names == ["ordinary"]
    assert result_value(registry.call("ordinary", '{"value": 3}')) == 3


def test_register_overrides_metadata_for_only_one_container():
    """Registration options customize one registry without changing a reusable declaration."""

    @declare_tool(name="declared", description="Declared")
    def calculate(number: int) -> int:
        return number

    planner = Mock(side_effect=lambda arguments: OperationPlan(arguments=arguments))
    customized = ToolRegistry()
    customized.register(
        calculate,
        name="customized",
        description="Customized",
        actions={Action.NETWORK_REQUEST},
        operation_planner=planner,
    )
    standard = ToolRegistry([calculate])

    custom_tool = customized.tools[0]
    standard_tool = standard.tools[0]
    assert (
        custom_tool.name,
        custom_tool.description,
        custom_tool.actions,
        custom_tool.operation_planner,
    ) == (
        "customized",
        "Customized",
        frozenset({Action.NETWORK_REQUEST}),
        planner,
    )
    assert (
        standard_tool.name,
        standard_tool.description,
        standard_tool.actions,
        standard_tool.operation_planner,
    ) == ("declared", "Declared", frozenset(), None)


def test_register_rejects_authority_bearing_tools_without_a_planner():
    """Every authority-bearing registration must identify concrete targets before execution."""

    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    with pytest.raises(ToolRegistrationError, match="without an operation planner"):
        ToolRegistry().register(calculate, actions={Action.SESSION_MUTATE})


def test_register_can_remove_a_declared_operation_planner():
    """An explicit null planner removes inherited operation planning in one registry."""
    planner = Mock(side_effect=lambda arguments: OperationPlan(arguments=arguments))

    @declare_tool(operation_planner=planner)
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    inherited = ToolRegistry([calculate])
    overridden = ToolRegistry()
    overridden.register(calculate, operation_planner=None)

    assert inherited.tools[0].operation_planner is planner
    assert overridden.tools[0].operation_planner is None


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
            actions=frozenset(),
        )
    )

    registered = registry.tools[0]
    assert registered.name == "local_calculate"
    assert registered.description == "Calculate locally."
    assert registered.actions == frozenset()


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

    registry = ToolRegistry([Multiplier()])

    assert registry.names == ["Multiplier"]
    assert result_value(registry.call("Multiplier", '{"number": 3}')) == 6


def test_register_accepts_partials_with_explicit_local_metadata():
    """Partially bound functions expose only their remaining annotated parameters."""

    def add(first: int, second: int) -> int:
        return first + second

    registry = ToolRegistry()
    registry.register(partial(add, 2), name="add_two", description="Add two to a number.")

    assert result_value(registry.call("add_two", '{"second": 3}')) == 5


def test_register_dispatches_async_callable_objects_only_through_async_calls():
    """Coroutine-based callable instances are detected before synchronous invocation."""

    class AsyncMultiplier:
        """Multiply a number asynchronously."""

        async def __call__(self, number: int) -> int:
            return number * 2

    registry = ToolRegistry([AsyncMultiplier()])

    assert "tool.async_required" in registry.call("AsyncMultiplier", '{"number": 3}')
    assert result_value(asyncio.run(registry.call_async("AsyncMultiplier", '{"number": 3}'))) == 6


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

    replacement = PermissionManager()
    registry.permission_manager = replacement
    assert registry.permission_manager is replacement


def test_call_reports_unknown_tools():
    """Synchronous routing serializes an unknown-tool error at the registry boundary."""
    problem = json.loads(ToolRegistry().call("missing", "{}"))["problem"]
    assert problem["code"] == "tool.unknown"
    assert problem["detail"] == "Tool 'missing' is not available."


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
        result_value(
            registry.call("calculate", "{}", interaction=runtime, instructions_manager=manager)
        )
        == "calculate"
    )
    context = contexts[0]
    assert context.interaction is runtime
    assert context.tool_name == "calculate"
    assert context.instructions_manager is manager


def test_call_uses_default_or_no_context():
    """Synchronous routing uses the default interaction and omits context when none exists."""
    interaction = Mock(spec=Interaction)
    registry = ToolRegistry(interaction=interaction)
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


def test_call_with_timing_excludes_permission_confirmation(monkeypatch):
    """Tool timing begins after authorization and uses the policy manager's recorder."""
    interaction = Mock(spec=Interaction)
    interaction.confirm.return_value = True
    recorder = Mock()
    permissions = PermissionManager(interaction=interaction, recorder=recorder)
    registry = ToolRegistry(interaction=interaction, permission_manager=permissions)

    @declare_tool(
        actions={Action.SESSION_MUTATE},
        operation_planner=planner_for(Action.SESSION_MUTATE),
    )
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    registry.register(calculate)
    clock = Mock(side_effect=[10.0, 12.0])
    monkeypatch.setattr(tool_registry_module, "perf_counter", clock)

    output, duration = registry.call_with_timing("calculate", '{"number": 3}')

    assert result_value(output) == 3
    assert duration == 2
    interaction.confirm.assert_called_once()
    recorder.record_authorization.assert_called_once()


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
        """Report successful user-command context construction."""
        return context.tool_name == "has_no_policy"

    registry.register(describe)
    registry.register(has_no_policy)

    assert result_value(registry.command("describe", ("3", "label=two words")).output) == {
        "count": 3,
        "label": "two words",
    }
    assert result_value(registry.command("has_no_policy", ()).output) is True
    permissions.authorize.assert_not_called()


def test_call_command_retains_dynamic_result_presentation():
    """Command dispatch returns the presentation selected for canonical arguments and output."""
    presentation = ToolResultPresentationSpec(kind=ToolResultPresentation.TABLE)

    @declare_tool(result_presentation=lambda arguments, result: presentation)
    def describe(count: int) -> dict:
        """Describe parsed values."""
        return {"count": count}

    execution = ToolRegistry([describe]).command("describe", ("3",))

    assert result_value(execution.output) == {"count": 3}
    assert execution.presentation is presentation


def test_call_command_reports_unknown_tools_and_invalid_parameters():
    """Command routing serializes lookup, binding, validation, and planning failures."""
    registry = ToolRegistry()
    register(registry)

    def invalid_planner(_arguments):
        """Reject planning to verify the user-command error boundary."""
        raise ValueError("invalid operation plan")

    @declare_tool(actions={Action.SESSION_MUTATE}, operation_planner=invalid_planner)
    def invalid_plan() -> None:
        """Expose an intentionally invalid operation planner."""

    registry.register(invalid_plan)

    assert "tool.unknown" in registry.command("missing", ()).output
    assert "tool.invalid_arguments" in registry.command("calculate", ("unknown=1",)).output
    assert "argument_binding" in registry.command("calculate", ("unknown=1",)).output
    assert "tool.invalid_arguments" in registry.command("calculate", ("not-an-integer",)).output
    assert "tool.planning_failed" in registry.command("invalid_plan", ()).output


def test_call_async_reports_unknown_tools():
    """Asynchronous routing serializes an unknown-tool error at the registry boundary."""
    problem = json.loads(asyncio.run(ToolRegistry().call_async("missing", "{}")))["problem"]
    assert problem["code"] == "tool.unknown"


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

    assert result_value(result) == "calculate"
    assert contexts[0].interaction is interaction


def test_call_async_returns_validation_and_permission_denials_before_invocation():
    """Async dispatch stops before invocation for invalid and rejected calls."""
    interaction = Mock(spec=Interaction)
    interaction.confirm.return_value = False
    registry = ToolRegistry(interaction=interaction)

    @declare_tool(
        actions={Action.SESSION_MUTATE},
        operation_planner=planner_for(Action.SESSION_MUTATE),
    )
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    registry.register(calculate)

    invalid = asyncio.run(registry.call_async("calculate", "bad"))
    denied = asyncio.run(registry.call_async("calculate", '{"number": 1}'))

    assert "tool.invalid_arguments" in invalid
    assert "tool.denied" in denied


def test_sync_and_async_dispatch_report_operation_planning_failures():
    """A failed canonical operation plan prevents both dispatch paths from invoking a tool."""
    calls = []

    def calculate() -> str:
        """Return an unreachable result."""
        calls.append(True)
        return "unreachable"

    def fail_plan(_arguments):
        raise ValueError("cannot canonicalize target")

    registry = ToolRegistry()
    registry.register(calculate, operation_planner=fail_plan)

    sync_result, sync_duration = registry.call_with_timing("calculate", "{}")
    async_result, async_duration = asyncio.run(registry.call_with_timing_async("calculate", "{}"))

    assert "tool.planning_failed" in sync_result
    assert "tool.planning_failed" in async_result
    assert sync_duration == async_duration == 0
    assert calls == []


def test_all_dispatch_paths_preserve_structured_operation_planning_problems():
    """Structured planning failures retain their problem contract across every dispatcher."""
    planning_problem = Problem(
        code="test.specific",
        title="Specific planning failure",
        detail="Use different input.",
        operation="calculate",
    )

    def fail_plan(_arguments):
        raise ProblemException(planning_problem)

    @declare_tool(actions={Action.SESSION_MUTATE}, operation_planner=fail_plan)
    def calculate() -> None:
        """Expose a structured planning failure."""

    registry = ToolRegistry([calculate])

    sync = json.loads(registry.call("calculate", "{}"))["problem"]
    asynchronous = json.loads(asyncio.run(registry.call_async("calculate", "{}")))["problem"]
    command = json.loads(registry.command("calculate", ()).output)["problem"]

    assert sync["code"] == asynchronous["code"] == command["code"] == "test.specific"


def test_call_with_timing_async_excludes_permission_confirmation(monkeypatch):
    """Async tool timing spans awaited execution but excludes permission confirmation."""
    interaction = Mock(spec=Interaction)
    interaction.confirm.return_value = True
    recorder = Mock()
    permissions = PermissionManager(interaction=interaction, recorder=recorder)
    registry = ToolRegistry(interaction=interaction, permission_manager=permissions)

    @declare_tool(
        actions={Action.NETWORK_REQUEST},
        operation_planner=planner_for(Action.NETWORK_REQUEST),
    )
    async def calculate(number: int) -> int:
        """Calculate a number asynchronously."""
        return number

    registry.register(calculate)
    clock = Mock(side_effect=[10.0, 12.0])
    monkeypatch.setattr(tool_registry_module, "perf_counter", clock)

    output, duration = asyncio.run(registry.call_with_timing_async("calculate", '{"number": 3}'))

    assert result_value(output) == 3
    assert duration == 2
    interaction.confirm.assert_called_once()
    recorder.record_authorization.assert_called_once()


@pytest.mark.parametrize(
    ("name", "arguments", "error"),
    [
        ("missing", "{}", "tool.unknown"),
        ("calculate", "bad", "tool.invalid_arguments"),
        ("calculate", '{"number": 1}', "tool.denied"),
    ],
)
def test_call_with_timing_async_returns_zero_before_invocation(name, arguments, error, monkeypatch):
    """Async timing remains zero when lookup, validation, or authorization stops dispatch."""
    interaction = Mock(spec=Interaction)
    interaction.confirm.return_value = False
    registry = ToolRegistry(interaction=interaction)

    @declare_tool(
        actions={Action.SESSION_MUTATE},
        operation_planner=planner_for(Action.SESSION_MUTATE),
    )
    def calculate(number: int) -> int:
        """Calculate a number."""
        return number

    registry.register(calculate)
    clock = Mock()
    monkeypatch.setattr(tool_registry_module, "perf_counter", clock)

    output, duration = asyncio.run(registry.call_with_timing_async(name, arguments))

    assert error in output
    assert duration == 0
    clock.assert_not_called()
