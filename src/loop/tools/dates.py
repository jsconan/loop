"""Provide tools for working with dates and times."""

from ..tooling import tool
from ..utils import local_now


@tool
def get_current_datetime() -> str:
    """Return the current local date and time."""
    return local_now().strftime("%A, %B %d, %Y - %H:%M:%S")
