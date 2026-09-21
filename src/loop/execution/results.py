"""Define exhaustive, closed results for sandbox execution."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseExecutionResult(BaseModel):
    """Carry common sanitized execution outcome data and bounded output.

    Args:
        request_id (str): Immutable request identifier.
        diagnostic_id (str | None): Opaque privileged-diagnostic correlation identifier.
        stdout (bytes): Retained standard-output bytes.
        stderr (bytes): Retained standard-error bytes.
        stdout_truncated (bool): Whether standard output exceeded its retained bound.
        stderr_truncated (bool): Whether standard error exceeded its retained bound.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    diagnostic_id: str | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class Completed(BaseExecutionResult):
    """Report a normally completed program, including a nonzero exit status."""

    kind: Literal["completed"] = "completed"
    exit_code: int


class ExecutableNotFound(BaseExecutionResult):
    """Report a missing executable inside the selected sandbox runtime."""

    kind: Literal["executable_not_found"] = "executable_not_found"
    executable: str


class CapabilityDenied(BaseExecutionResult):
    """Report runtime denial of an ungranted capability."""

    kind: Literal["capability_denied"] = "capability_denied"
    capability: str


class UnsupportedCapability(BaseExecutionResult):
    """Report a capability unavailable on the selected sandbox backend."""

    kind: Literal["unsupported_capability"] = "unsupported_capability"
    capability: str


class WorkspaceBusy(BaseExecutionResult):
    """Report a workspace that cannot produce a stable immutable snapshot."""

    kind: Literal["workspace_busy"] = "workspace_busy"


class UnrepresentableDelta(BaseExecutionResult):
    """Report a persistent effect that cannot be committed losslessly."""

    kind: Literal["unrepresentable_delta"] = "unrepresentable_delta"


class CommitConflict(BaseExecutionResult):
    """Report host state changed since the immutable generation was leased."""

    kind: Literal["commit_conflict"] = "commit_conflict"


class InfrastructureFailure(BaseExecutionResult):
    """Report a sanitized backend initialization or integrity failure."""

    kind: Literal["infrastructure_failure"] = "infrastructure_failure"


class ResourceLimitExceeded(BaseExecutionResult):
    """Report runtime enforcement of a named resource limit."""

    kind: Literal["resource_limit_exceeded"] = "resource_limit_exceeded"
    resource: str


class TimedOut(BaseExecutionResult):
    """Report termination after the authorized wall-time deadline."""

    kind: Literal["timed_out"] = "timed_out"


class Cancelled(BaseExecutionResult):
    """Report explicit cancellation after process-tree cleanup."""

    kind: Literal["cancelled"] = "cancelled"


class SpawnFailure(BaseExecutionResult):
    """Report failure to create the sandbox process after attestation."""

    kind: Literal["spawn_failure"] = "spawn_failure"


ExecutionResult = Annotated[
    Completed
    | ExecutableNotFound
    | CapabilityDenied
    | UnsupportedCapability
    | WorkspaceBusy
    | UnrepresentableDelta
    | CommitConflict
    | InfrastructureFailure
    | ResourceLimitExceeded
    | TimedOut
    | Cancelled
    | SpawnFailure,
    Field(discriminator="kind"),
]

TERMINAL_RESULT_KINDS = frozenset(
    {
        "completed",
        "executable_not_found",
        "capability_denied",
        "unsupported_capability",
        "workspace_busy",
        "unrepresentable_delta",
        "commit_conflict",
        "infrastructure_failure",
        "resource_limit_exceeded",
        "timed_out",
        "cancelled",
        "spawn_failure",
    }
)
