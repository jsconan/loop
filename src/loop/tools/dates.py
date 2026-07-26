"""Tools for working with dates and times."""

from datetime import datetime

from ..tooling import tool_registry


@tool_registry.tool
def get_current_datetime() -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("%A, %B %d, %Y - %H:%M:%S")
