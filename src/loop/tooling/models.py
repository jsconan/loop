"""Define tooling-domain models and errors."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from ..constants import DEFAULT_COMMAND_TIMEOUT, DEFAULT_USER_AGENT
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


class ToolExecutionAdmissionRejected(Exception):
    """Indicate that an invocation lost its authority before dispatch."""


class ToolExecutionOutcomeUnknown(RuntimeError):
    """Preserve recovery when interrupted work has no authoritative tool result."""


class ToolExecutionNotStarted(asyncio.CancelledError):
    """Signal cancellation that atomically prevented one tool's entry.

    Args:
        invocation (object | None): Opaque identity of the protected dispatch.
    """

    _invocation: object | None

    def __init__(self, invocation: object | None = None) -> None:
        super().__init__()
        self._invocation = invocation


class ToolExecutionCancelled(asyncio.CancelledError):
    """Carry a confirmed tool result while propagating cancellation to its owner.

    Args:
        output (str): Normalized result of an invocation that settled after cancellation.
        duration (float): Tool execution duration in seconds.
    """

    output: str
    duration: float

    def __init__(self, output: str, duration: float) -> None:
        super().__init__()
        self.output = output
        self.duration = duration


class OwnedToolTaskCancelled(asyncio.CancelledError):
    """Carry an explicitly declared final owned-task outcome through cancellation.

    Args:
        result (object): Settled final result, or ``None`` when the operation failed.
        error (Exception | None): Settled failure, or ``None`` for a successful result.
        invocation (object | None): Opaque identity of the invocation that declared the result.
    """

    result: object
    error: Exception | None
    _invocation: object | None

    def __init__(self, result: object, error: Exception | None, invocation: object | None) -> None:
        super().__init__()
        self.result = result
        self.error = error
        self._invocation = invocation


class ToolFinalInterruption(KeyboardInterrupt):
    """Carry a verified raw final outcome between synchronous tool boundaries.

    Args:
        result (object): Confirmed final value, or ``None`` when the operation failed.
        error (Exception | None): Confirmed final failure, or ``None`` after success.
    """

    result: object
    error: Exception | None

    def __init__(self, result: object, error: Exception | None) -> None:
        super().__init__()
        self.result = result
        self.error = error


class ToolExecutionInterrupted(KeyboardInterrupt):
    """Carry a normalized synchronous result while preserving interruption.

    Args:
        output (str): Serialized, normalized result of the interrupted tool.
        duration (float): Tool execution duration in seconds.
    """

    output: str
    duration: float

    def __init__(self, output: str, duration: float) -> None:
        super().__init__()
        self.output = output
        self.duration = duration


@dataclass
class ToolRuntimeSettings:
    """Provide scoped configuration values to tool implementations.

    Args:
        user_agent (str): HTTP user-agent used by web-content tools.
        command_timeout (float): Finite command runtime budget starting at native launch.
    """

    user_agent: str = DEFAULT_USER_AGENT
    command_timeout: float = DEFAULT_COMMAND_TIMEOUT
