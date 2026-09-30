"""Tests for command registration and dispatch utilities."""

from typing import Annotated

import pytest

from loop import CommandContext, CommandRegistrationError
from loop.commands.models import CommandRemainder
from loop.commands.utils import get_command_arguments_model, takes_command_context


@pytest.mark.parametrize("kind", ["positional", "args", "kwargs", "untyped", "late_context"])
def test_argument_model_rejects_unsupported_signatures(kind):
    """Command schemas reject parameters that cannot be supplied as typed keyword fields."""

    def positional(value: int, /) -> None:
        pass

    def args(*values: int) -> None:
        pass

    def kwargs(**values: int) -> None:
        pass

    def untyped(value) -> None:
        pass

    def late_context(value: int, context: CommandContext) -> None:
        pass

    with pytest.raises(CommandRegistrationError):
        get_command_arguments_model(locals()[kind], "invalid")


def test_context_detection_requires_a_leading_context_parameter():
    """Context detection accepts only a leading parameter annotated as CommandContext."""

    def plain() -> None:
        pass

    def contextual(context: CommandContext) -> None:
        pass

    assert takes_command_context(plain) is False
    assert takes_command_context(contextual) is True


@pytest.mark.parametrize("kind", ["not_final", "wrong_type", "multiple"])
def test_argument_model_rejects_invalid_remainder_declarations(kind):
    """Remainders must be a single final tuple of strings."""

    def not_final(remainder: Annotated[tuple[str, ...], CommandRemainder()], value: str) -> None:
        pass

    def wrong_type(remainder: Annotated[list[str], CommandRemainder()]) -> None:
        pass

    def multiple(
        first: Annotated[tuple[str, ...], CommandRemainder()],
        second: Annotated[tuple[str, ...], CommandRemainder()],
    ) -> None:
        pass

    with pytest.raises(CommandRegistrationError, match="remainder must be"):
        get_command_arguments_model(locals()[kind], "invalid")
