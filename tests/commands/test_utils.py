"""Tests for command registration and dispatch utilities."""

import pytest

from loop import CommandContext, CommandRegistrationError
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
