"""Tests for individual schema-backed command declarations."""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from loop import Command, CommandContext, Interaction
from loop.commands.utils import get_command_arguments_model


def make_command(function, name: str = "select") -> Command:
    """Build a command schema from a test function."""
    return Command(name, "Select.", function, get_command_arguments_model(function, name))


def test_command_deserializes_json_and_raw_single_values():
    """Commands pass JSON objects and natural raw text to one typed side-effect parameter."""
    selected = []

    def select(count: int) -> None:
        selected.append(count)

    command = make_command(select)
    assert command.call('{"count": 2}') is None
    assert command.call("3") is None
    assert selected == [2, 3]


def test_command_calls_parameterless_function_and_injects_context_when_declared():
    """Empty payloads work and context is injected only as the leading declared parameter."""
    calls = []

    def plain() -> None:
        calls.append("plain")

    def contextual(context: CommandContext) -> None:
        calls.append(context.name)

    interaction = Mock(spec=Interaction)
    assert make_command(plain).call("") is None
    assert make_command(contextual).call("", CommandContext("select", interaction)) is None
    assert calls == ["plain", "select"]
    with pytest.raises(ValueError, match="requires a CommandContext"):
        make_command(contextual).call("")


def test_command_rejects_invalid_multiple_arguments():
    """Multiple typed parameters require a valid JSON object payload."""
    def pair(first: int, second: int) -> None:
        pass

    with pytest.raises(ValidationError):
        make_command(pair).call("invalid")
