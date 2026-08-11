"""Tests for individual schema-backed command declarations."""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from loop import Command, CommandArgumentError, CommandContext, Interaction
from loop.commands.utils import get_command_arguments_model


def make_command(function, name: str = "select") -> Command:
    """Build a command schema from a test function."""
    return Command(name, "Select.", function, get_command_arguments_model(function, name))


def test_command_binds_positional_and_named_arguments_and_decodes_each_value():
    """Commands bind mixed tokens and independently decode values for annotated fields."""
    selected = []

    def select(
        count: int,
        label: str,
        enabled: bool,
        numbers: list[int],
    ) -> None:
        selected.append((count, label, enabled, numbers))

    command = make_command(select)
    assert command.call("3 enabled=true numbers='[1, 2]' label=true") is None
    assert selected == [(3, "true", True, [1, 2])]


def test_command_preserves_raw_strings_and_reports_annotation_errors():
    """Non-JSON text remains a string and invalid decoded structures fail model validation."""

    def select(label: str, numbers: list[int]) -> None:
        pass

    command = make_command(select)
    with pytest.raises(ValidationError):
        command.call("plain null")


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


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ("unknown=1", "Unknown parameter 'unknown'"),
        ("first=1 first=2", "supplied more than once"),
        ("1 2 3", "Too many positional arguments"),
        ('{"first": 1, "second": 2}', "Too many positional arguments"),
        ('"unterminated', "Could not parse arguments"),
    ],
)
def test_command_rejects_invalid_argument_syntax_and_binding(arguments, message):
    """Commands reject unknown, duplicate, excess, and malformed argument input."""

    def pair(first: int, second: int) -> None:
        pass

    with pytest.raises(CommandArgumentError, match=message):
        make_command(pair).call(arguments)
