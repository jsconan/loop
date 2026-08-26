"""Define tooling-domain models and errors."""

from collections.abc import Callable
from dataclasses import dataclass

from ..models import StrEnum


class ToolStatus(StrEnum):
    """Describe the status of a tool."""

    READY = "ready"
    DEGRADED = "degraded"
    BROKEN = "broken"


@dataclass(frozen=True)
class ToolPreflightResult:
    """Describe whether a tool can be admitted to a registry.

    Args:
        status (ToolStatus): The current status of the tool.
        detail (str | None): Safe actionable reason for unavailability, or ``None`` when
            status is ``READY``.
    """

    status: ToolStatus = ToolStatus.READY
    detail: str | None = None


TOOL_READY = ToolPreflightResult(status=ToolStatus.READY, detail=None)
ToolPreflight = Callable[[], ToolPreflightResult]


class ToolRegistrationError(ValueError):
    """Indicate that a Python function cannot be registered as a tool."""
