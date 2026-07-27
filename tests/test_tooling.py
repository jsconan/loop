"""Tests for typed tool registration, schemas, validation, and dispatch."""

import asyncio
import json

import pytest

from loop.tooling import ToolRegistrationError, ToolRegistry


def test_decorator_registers_and_dispatches_function():
    """A decorated function is exposed by name and available for dispatch."""
    registry = ToolRegistry()

    @registry.tool
    def repeat(
        text: str,
        count: int,
    ) -> list[str]:
        """Repeat text a constrained number of times."""
        return [text] * count

    schema = registry.schemas()[0]

    assert schema["name"] == "repeat"
    assert json.loads(registry.call("repeat", '{"text":"hi","count":2}')) == ["hi", "hi"]


def test_duplicate_names_are_rejected():
    """The registry rejects ambiguous public names."""
    registry = ToolRegistry()

    @registry.tool(name="same")
    def first(value: str) -> str:
        """Return a value."""
        return value

    with pytest.raises(ToolRegistrationError, match="already registered"):

        @registry.tool(name="same")
        def second(value: str) -> str:
            """Return another value."""
            return value


def test_context_aware_function_is_bound_to_its_tool():
    """A function's self is its Tool and is omitted from the public schema."""
    registry = ToolRegistry()

    @registry.tool
    def decorated(self, value: str) -> str:
        """Decorate a value."""
        return f"{self.name}:{value}"

    schema = registry.schemas()[0]

    assert list(schema["parameters"]["properties"]) == ["value"]
    assert registry.call("decorated", '{"value":"done"}') == "decorated:done"


def test_invalid_json_and_arguments_do_not_call_function():
    """Invalid model input fails validation without invoking application code."""
    registry = ToolRegistry()
    calls = []

    @registry.tool
    def positive(number: int) -> int:
        """Return a positive integer."""
        calls.append(number)
        return number

    malformed = json.loads(registry.call("positive", "not json"))
    invalid = json.loads(registry.call("positive", '{"number":"not an integer"}'))
    extra = json.loads(registry.call("positive", '{"number":1,"other":2}'))

    assert malformed["error"] == "invalid_arguments"
    assert invalid["error"] == "invalid_arguments"
    assert extra["error"] == "invalid_arguments"
    assert not calls


def test_unknown_tool_returns_structured_error():
    """Unknown names produce a stable error suitable for returning to the model."""
    result = json.loads(ToolRegistry().call("missing", "{}"))

    assert result == {
        "error": "unknown_tool",
        "message": "Tool 'missing' is not available.",
    }


def test_dispatch_returns_strings_and_catches_execution_failures():
    """Public dispatch returns tool output and catches application exceptions."""
    registry = ToolRegistry()

    @registry.tool
    def text() -> str:
        """Return text."""
        return "plain"

    @registry.tool
    def broken() -> None:
        """Fail deliberately."""
        raise RuntimeError("boom")

    assert registry.call("text", "") == "plain"
    assert json.loads(registry.call("broken", "{}")) == {
        "error": "execution_failed",
        "message": "Tool 'broken' failed: boom",
    }


def test_async_dispatch_supports_sync_and_async_tools():
    """The asynchronous dispatcher awaits coroutine tools."""
    registry = ToolRegistry()

    @registry.tool
    async def double(number: int) -> int:
        """Double an integer."""
        return number * 2

    result = asyncio.run(registry.call_async("double", '{"number":3}'))

    assert result == "6"
    assert json.loads(registry.call("double", '{"number":3}'))["error"] == (
        "async_tool_in_sync_loop"
    )


def test_async_dispatch_validates_unknown_sync_and_failing_tools():
    """Async dispatch handles all public outcomes, including a returned awaitable."""
    registry = ToolRegistry()

    @registry.tool
    def increment(number: int) -> int:
        """Increment a number."""
        return number + 1

    @registry.tool
    def returns_awaitable(number: int):
        """Return an awaitable from a synchronous callable."""

        async def calculate():
            return number * 3

        return calculate()

    @registry.tool
    async def broken() -> None:
        """Fail asynchronously."""
        raise RuntimeError("async boom")

    assert asyncio.run(registry.call_async("increment", '{"number":1}')) == "2"
    assert asyncio.run(registry.call_async("returns_awaitable", '{"number":2}')) == "6"
    assert json.loads(asyncio.run(registry.call_async("broken", "{}")))["error"] == (
        "execution_failed"
    )
    assert json.loads(asyncio.run(registry.call_async("increment", "bad")))["error"] == (
        "invalid_arguments"
    )
    assert json.loads(asyncio.run(registry.call_async("missing", "{}")))["error"] == "unknown_tool"
