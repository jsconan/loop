"""Provide tools for working with dates and times."""

from datetime import datetime

from ..permissions import Capability
from ..tooling import tool_registry


@tool_registry.tool(capabilities={Capability.PURE})
def get_current_datetime() -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("%A, %B %d, %Y - %H:%M:%S")
