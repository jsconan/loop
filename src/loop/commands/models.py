"""Define exceptions and models for slash-command argument handling."""

from dataclasses import dataclass


class CommandArgumentError(ValueError):
    """Indicate invalid slash-command argument syntax or binding."""


class CommandRegistrationError(ValueError):
    """Indicate that a Python function cannot be registered as a command."""


@dataclass(frozen=True)
class CommandRemainder:
    """Mark a final command field that receives every remaining shell token."""
