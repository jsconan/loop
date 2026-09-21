"""Define authority, disclosure, audit, and result types for explicit host execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...utils import json_encode, sha256_digest
from ..contracts import ExecutionBoundary, HostExecutionRequest
from ..infrastructure import VerifiedDirectory, VerifiedExecutable


def host_request_fingerprint(request: HostExecutionRequest) -> str:
    """Return a stable digest binding every authority-bearing request field.

    Args:
        request (HostExecutionRequest): Explicit unleased host request.

    Returns:
        str: SHA-256 hexadecimal request identity.
    """
    payload = json_encode(request.model_dump(mode="json")).encode()
    return sha256_digest(payload)


class HostExecutionLease(BaseModel):
    """Authorize one exact verified host request for a bounded start window.

    Args:
        lease_id (str): Immutable authorization identity.
        request_id (str): Exact request authorized by the user or a matching grant.
        request_fingerprint (str): Digest of every authority-bearing request field.
        executable_sha256 (str): Verified executable content identity.
        workspace_id (str): Authenticated workspace binding.
        policy_version (str): Permission policy version used for authorization.
        issued_at_ns (int): Trusted issue timestamp.
        expires_at_ns (int): Latest permitted process-start timestamp.
        boundary (Literal[ExecutionBoundary.HOST]): Fixed host boundary discriminator.
    """

    model_config = ConfigDict(frozen=True)

    lease_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    workspace_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    issued_at_ns: int = Field(ge=0)
    expires_at_ns: int = Field(ge=0)
    boundary: Literal[ExecutionBoundary.HOST] = ExecutionBoundary.HOST

    @model_validator(mode="after")
    def validate_lifetime(self) -> HostExecutionLease:
        """Reject a lease whose start window is already empty.

        Returns:
            HostExecutionLease: Validated nonempty host lease.

        Raises:
            ValueError: If expiry does not follow issuance.
        """
        if self.expires_at_ns <= self.issued_at_ns:
            raise ValueError("Host lease expiry must follow issuance.")
        return self


@dataclass(frozen=True, slots=True)
class AuthorizedHostExecution:
    """Carry a request and its revalidated launch authorities to the host supervisor.

    Args:
        request (HostExecutionRequest): Exact explicit host request.
        lease (HostExecutionLease): Authorization bound to the request and executable.
        executable (VerifiedExecutable): Descriptor-derived executable authority.
        cwd (VerifiedDirectory): Descriptor-derived working-directory authority.
    """

    request: HostExecutionRequest
    lease: HostExecutionLease
    executable: VerifiedExecutable
    cwd: VerifiedDirectory


class HostExecutionPrompt(BaseModel):
    """Represent the complete prominent disclosure for one host authorization.

    Args:
        request_id (str): Explicit request identity.
        display_cwd (str): Sanitized user-facing working directory.
        executable_sha256 (str): Verified executable content identity.
        argv (tuple[str, ...]): Exact requested argument vector.
        environment_names (tuple[str, ...]): Explicit environment keys, never values.
        resource_class (str): Requested host resource category.
        reason (str): Why the sandbox cannot satisfy the operation.
        deadline_seconds (float): Best-effort process deadline disclosed to the user.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    display_cwd: str
    executable_sha256: str
    argv: tuple[str, ...]
    environment_names: tuple[str, ...]
    resource_class: str
    reason: str
    deadline_seconds: float = Field(gt=0)

    def render(self) -> str:
        """Render the disclosure consumed by the retained permission interaction.

        Returns:
            str: Prominent host-boundary warning and exact sanitized request details.
        """
        return "\n".join(
            (
                "HOST EXECUTION REQUEST",
                "Warning: sandbox protections do not apply to this operation.",
                f"Reason: {self.reason}",
                f"Working directory: {self.display_cwd}",
                f"Executable identity: sha256:{self.executable_sha256}",
                f"Arguments: {json_encode(self.argv)}",
                f"Environment names: {json_encode(self.environment_names)}",
                f"Host resource class: {self.resource_class}",
                f"Best-effort process deadline: {self.deadline_seconds:g} seconds",
                (
                    "Approval is limited by executable identity, arguments, resource, workspace, "
                    "scope, and lifetime."
                ),
            )
        )


class HostAuditEventType(StrEnum):
    """Identify security-relevant host workflow audit events."""

    REQUESTED = "execution.host.requested"
    AUTHORIZED = "execution.host.authorized"
    DENIED = "execution.host.denied"
    STARTED = "execution.host.started"
    TERMINAL = "execution.host.terminal"


class HostPermissionDenialReason(StrEnum):
    """Identify the safe user-facing reason explicit host authority was unavailable."""

    HOST_PROCESSES_DISABLED = "host_processes_disabled"
    USER_DENIED = "user_denied"
    APPROVAL_UNAVAILABLE = "approval_unavailable"
    POLICY_DENIED = "policy_denied"


@dataclass(frozen=True, slots=True)
class HostAuthorizationDecision:
    """Carry one explicit host authorization decision without policy internals.

    Args:
        allowed (bool): Whether the exact host request may receive a launch lease.
        denial_reason (HostPermissionDenialReason | None): Safe denial category, if denied.
    """

    allowed: bool
    denial_reason: HostPermissionDenialReason | None = None


class HostAuditEvent(BaseModel):
    """Carry one sanitized explicit-host audit record.

    Args:
        type (HostAuditEventType): Security event category.
        request_id (str): Owning explicit host request.
        lease_id (str | None): Host lease identity after authorization.
        executable_sha256 (str | None): Verified executable identity when available.
        outcome (str | None): Sanitized decision or terminal result kind.
    """

    model_config = ConfigDict(frozen=True)

    type: HostAuditEventType
    request_id: str
    lease_id: str | None = None
    executable_sha256: str | None = None
    outcome: str | None = None


class HostAuditSink(Protocol):
    """Persist sanitized host security events without exposing execution content."""

    def record(self, event: HostAuditEvent) -> None:
        """Record one explicit-host audit event.

        Args:
            event (HostAuditEvent): Sanitized immutable event.
        """


class BaseHostExecutionResult(BaseModel):
    """Carry bounded output from the explicitly approved host boundary.

    Args:
        request_id (str): Owning explicit host request identity.
        boundary (Literal[ExecutionBoundary.HOST]): Fixed host boundary discriminator.
        stdout (bytes): Bounded retained standard output.
        stderr (bytes): Bounded retained standard error.
        stdout_truncated (bool): Whether standard output exceeded its bound.
        stderr_truncated (bool): Whether standard error exceeded its bound.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    boundary: Literal[ExecutionBoundary.HOST] = ExecutionBoundary.HOST
    stdout: bytes = b""
    stderr: bytes = b""
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class HostCompleted(BaseHostExecutionResult):
    """Report normal host process completion, including nonzero exit status.

    Args:
        exit_code (int): Native host process exit status.
    """

    kind: Literal["host_completed"] = "host_completed"
    exit_code: int


class HostPermissionDenied(BaseHostExecutionResult):
    """Report failure to obtain explicit host authority.

    Args:
        reason (HostPermissionDenialReason): Safe actionable denial category.
    """

    kind: Literal["host_permission_denied"] = "host_permission_denied"
    reason: HostPermissionDenialReason = HostPermissionDenialReason.POLICY_DENIED


class HostIntegrityFailure(BaseHostExecutionResult):
    """Report changed or invalid host launch authority."""

    kind: Literal["host_integrity_failure"] = "host_integrity_failure"


class HostSpawnFailure(BaseHostExecutionResult):
    """Report failure to create the explicitly approved host process."""

    kind: Literal["host_spawn_failure"] = "host_spawn_failure"


class HostTimedOut(BaseHostExecutionResult):
    """Report best-effort termination after the host deadline."""

    kind: Literal["host_timed_out"] = "host_timed_out"


class HostCancelled(BaseHostExecutionResult):
    """Report best-effort termination after explicit host cancellation."""

    kind: Literal["host_cancelled"] = "host_cancelled"


class HostInfrastructureFailure(BaseHostExecutionResult):
    """Report a sanitized internal host-supervision failure."""

    kind: Literal["host_infrastructure_failure"] = "host_infrastructure_failure"


HostExecutionResult = Annotated[
    HostCompleted
    | HostPermissionDenied
    | HostIntegrityFailure
    | HostSpawnFailure
    | HostTimedOut
    | HostCancelled
    | HostInfrastructureFailure,
    Field(discriminator="kind"),
]
