"""Tests for public tooling exception types."""

from loop import ToolRegistrationError


def test_tool_registration_error_is_a_value_error():
    """Callers may handle registration failures as ordinary value errors."""
    error = ToolRegistrationError("invalid tool")

    assert isinstance(error, ValueError)
    assert str(error) == "invalid tool"
