"""Tests for the built-in date tools."""

import re

from loop import get_current_datetime


def test_current_datetime_has_the_documented_shape():
    """The date tool returns a complete human-readable local timestamp."""
    assert re.fullmatch(
        r"[A-Z][a-z]+, [A-Z][a-z]+ \d{2}, \d{4} - \d{2}:\d{2}:\d{2}",
        get_current_datetime(),
    )
