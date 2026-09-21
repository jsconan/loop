"""Authorize explicit host requests before delegating to the host-only supervisor."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from ..contracts import HostExecutionRequest
from ..infrastructure import (
    VerifiedExecutable,
    identify_host_directory,
    identify_host_executable,
)
from .models import (
    AuthorizedHostExecution,
    HostAuditEvent,
    HostAuditEventType,
    HostAuditSink,
    HostAuthorizationDecision,
    HostExecutionLease,
    HostExecutionPrompt,
    HostExecutionResult,
    HostInfrastructureFailure,
    HostIntegrityFailure,
    HostPermissionDenialReason,
    HostPermissionDenied,
    host_request_fingerprint,
)
from .supervisor import HostExecutionSupervisor, TelemetryHostAudit


class HostPermissionAuthorizer(Protocol):
    """Describe the retained permission facade adapter used by the host broker."""

    def authorize(
        self,
        request: HostExecutionRequest,
        executable: VerifiedExecutable,
        prompt: HostExecutionPrompt,
    ) -> HostAuthorizationDecision:
        """Authorize one exact host request through the application permission facade.

        Args:
            request (HostExecutionRequest): Explicit unleased host request.
            executable (VerifiedExecutable): Broker-verified executable identity.
            prompt (HostExecutionPrompt): Complete prominent disclosure.

        Returns:
            HostAuthorizationDecision: Safe typed authorization outcome.
        """


class HostExecutionBroker:
    """Resolve, authorize, lease, and dispatch the sole explicit host workflow.

    Args:
        permissions (HostPermissionAuthorizer): Adapter to the retained permission facade.
        workspace_id (str): Authenticated workspace identity bound into every lease.
        policy_version (str): Permission policy version bound into every lease.
        workspace_root (Path | None): Authenticated host root represented publicly as
            ``/workspace``. Defaults to no virtual host mapping.
        supervisor (HostExecutionSupervisor | None): Separate host-only process supervisor.
        audit (HostAuditSink | None): Sanitized host security audit destination.
        clock_ns (Callable[[], int]): Trusted timestamp source.
        id_factory (Callable[[], str]): Unique lease identifier source.
    """

    permissions: HostPermissionAuthorizer
    workspace_id: str
    policy_version: str
    workspace_root: Path | None
    supervisor: HostExecutionSupervisor
    audit: HostAuditSink
    _clock_ns: Callable[[], int]
    _id_factory: Callable[[], str]

    def __init__(
        self,
        permissions: HostPermissionAuthorizer,
        workspace_id: str,
        policy_version: str,
        workspace_root: Path | None = None,
        supervisor: HostExecutionSupervisor | None = None,
        audit: HostAuditSink | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        id_factory: Callable[[], str] = lambda: str(uuid4()),
    ) -> None:
        self.permissions = permissions
        self.workspace_id = workspace_id
        self.policy_version = policy_version
        self.workspace_root = workspace_root.resolve() if workspace_root is not None else None
        self.audit = audit if audit is not None else TelemetryHostAudit()
        self.supervisor = (
            supervisor
            if supervisor is not None
            else HostExecutionSupervisor(audit=self.audit, clock_ns=clock_ns)
        )
        self._clock_ns = clock_ns
        self._id_factory = id_factory

    def execute(
        self,
        request: HostExecutionRequest,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> HostExecutionResult:
        """Execute one explicit request only after identity verification and authorization.

        Args:
            request (HostExecutionRequest): New explicit host-boundary request.
            cancellation (Callable[[], bool]): Predicate requesting best-effort cancellation.

        Returns:
            HostExecutionResult: Closed host-only authorization or process outcome.
        """
        if not self._record(
            HostAuditEvent(type=HostAuditEventType.REQUESTED, request_id=request.request_id)
        ):
            return HostInfrastructureFailure(request_id=request.request_id)
        try:
            request_path = Path(request.executable)
            executable = identify_host_executable(request_path)
            host_cwd = self._host_cwd(request.cwd)
            cwd = identify_host_directory(host_cwd)
        except ValueError:
            self._record(
                HostAuditEvent(
                    type=HostAuditEventType.DENIED,
                    request_id=request.request_id,
                    outcome="identity_invalid",
                )
            )
            return HostIntegrityFailure(request_id=request.request_id)
        authorized_request = (
            request
            if host_cwd.as_posix() == request.cwd
            else request.model_copy(update={"cwd": host_cwd.as_posix()})
        )
        prompt = HostExecutionPrompt(
            request_id=request.request_id,
            display_cwd=request.display_cwd,
            executable_sha256=executable.sha256,
            argv=(request_path.name, *request.argv[1:]),
            environment_names=tuple(key for key, _value in request.environment),
            resource_class=request.resource_class,
            reason=request.reason,
            deadline_seconds=request.deadline_seconds,
        )
        authorization = self.permissions.authorize(authorized_request, executable, prompt)
        if not authorization.allowed:
            self._record(
                HostAuditEvent(
                    type=HostAuditEventType.DENIED,
                    request_id=request.request_id,
                    executable_sha256=executable.sha256,
                    outcome="permission_denied",
                )
            )
            return HostPermissionDenied(
                request_id=request.request_id,
                reason=authorization.denial_reason or HostPermissionDenialReason.POLICY_DENIED,
            )
        issued_at = self._clock_ns()
        lease = HostExecutionLease(
            lease_id=self._id_factory(),
            request_id=request.request_id,
            request_fingerprint=host_request_fingerprint(authorized_request),
            executable_sha256=executable.sha256,
            workspace_id=self.workspace_id,
            policy_version=self.policy_version,
            issued_at_ns=issued_at,
            expires_at_ns=issued_at + int(request.deadline_seconds * 1_000_000_000),
        )
        if not self._record(
            HostAuditEvent(
                type=HostAuditEventType.AUTHORIZED,
                request_id=request.request_id,
                lease_id=lease.lease_id,
                executable_sha256=executable.sha256,
            )
        ):
            return HostInfrastructureFailure(request_id=request.request_id)
        return self.supervisor.execute(
            AuthorizedHostExecution(authorized_request, lease, executable, cwd), cancellation
        )

    def _host_cwd(self, cwd: str) -> Path:
        """Resolve the public workspace namespace only within its authenticated host root."""
        virtual = Path(cwd)
        if self.workspace_root is None or not (
            cwd == "/workspace" or cwd.startswith("/workspace/")
        ):
            return virtual
        relative = virtual.relative_to("/workspace")
        if ".." in relative.parts:
            raise ValueError("Virtual host working directory escapes the workspace.")
        resolved = self.workspace_root.joinpath(relative).resolve()
        if resolved != self.workspace_root and self.workspace_root not in resolved.parents:
            raise ValueError("Virtual host working directory escapes the workspace.")
        return resolved

    def _record(self, event: HostAuditEvent) -> bool:
        """Record an audit event without allowing observer failure to start host work."""
        try:
            self.audit.record(event)
        except Exception:  # noqa: BLE001 - security audit failure is a closed host outcome.
            return False
        return True
