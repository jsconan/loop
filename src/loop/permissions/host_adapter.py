"""Route explicit host authority through the retained application permission facade."""

from __future__ import annotations

import time
from collections.abc import Callable

from ..execution.contracts import ExecutionBoundary, HostExecutionRequest
from ..execution.host.models import (
    HostAuthorizationDecision,
    HostExecutionPrompt,
    HostPermissionDenialReason,
)
from ..execution.infrastructure import VerifiedExecutable
from ..utils import json_encode, sha256_digest
from .execution import (
    GrantConstraints,
    GrantDecision,
    GrantEffect,
    OpaqueResource,
    PolicyRequest,
    SubjectIdentity,
)
from .manager import PermissionManager


class HostExecutionPermissionAdapter:
    """Authorize exact host requests through the sole permission manager and store.

    Args:
        manager (PermissionManager): Application-facing permission facade.
        tool_id (str): Authenticated requesting tool identity.
        publisher (str): Authenticated requesting publisher identity.
        workspace_id (str): Durable active workspace identity.
        policy_version (str): Typed policy version for matching and issued grants.
        scope_binding (str | Callable[[], str]): Current process or session identity or resolver.
        clock_ns (Callable[[], int]): Trusted timestamp source.
    """

    manager: PermissionManager
    tool_id: str
    publisher: str
    workspace_id: str
    policy_version: str
    scope_binding: str | Callable[[], str]
    _clock_ns: Callable[[], int]

    def __init__(
        self,
        manager: PermissionManager,
        tool_id: str,
        publisher: str,
        workspace_id: str,
        policy_version: str,
        scope_binding: str | Callable[[], str],
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.manager = manager
        self.tool_id = tool_id
        self.publisher = publisher
        self.workspace_id = workspace_id
        self.policy_version = policy_version
        self.scope_binding = scope_binding
        self._clock_ns = clock_ns

    def authorize(
        self,
        request: HostExecutionRequest,
        executable: VerifiedExecutable,
        prompt: HostExecutionPrompt,
    ) -> HostAuthorizationDecision:
        """Authorize one verified host request with its prominent boundary disclosure.

        Args:
            request (HostExecutionRequest): Exact explicit host request.
            executable (VerifiedExecutable): Descriptor-derived executable identity.
            prompt (HostExecutionPrompt): Complete host-boundary disclosure.

        Returns:
            HostAuthorizationDecision: Safe typed authorization outcome.
        """
        argument_payload = json_encode(
            {
                "argv": request.argv[1:],
                "cwd": request.cwd,
                "environment": request.environment,
                "resource_class": request.resource_class,
            }
        ).encode()
        argument_class = sha256_digest(argument_payload)
        policy_request = PolicyRequest(
            subject=SubjectIdentity(
                tool_id=self.tool_id,
                publisher=self.publisher,
                executable_digest=executable.sha256,
            ),
            boundary=ExecutionBoundary.HOST,
            effect=GrantEffect.HOST_EXECUTE,
            resource=OpaqueResource(name=f"executable:sha256:{executable.sha256}"),
            workspace_id=self.workspace_id,
            policy_version=self.policy_version,
            now_ns=self._clock_ns(),
            constraints=GrantConstraints(
                max_bytes=request.output_limit_bytes,
                max_duration_ns=int(request.deadline_seconds * 1_000_000_000),
                argument_class=argument_class,
            ),
            scope_binding=(
                self.scope_binding() if callable(self.scope_binding) else self.scope_binding
            ),
        )
        host_processes_enabled = self.manager.effective_configuration.limits.allow_host_processes
        result = self.manager.authorize_execution((policy_request,), prompt=prompt.render())
        if result.decision is GrantDecision.ALLOW:
            return HostAuthorizationDecision(allowed=True)
        if not host_processes_enabled:
            reason = HostPermissionDenialReason.HOST_PROCESSES_DISABLED
        elif result.prompted and result.reason == "Rejected by the user.":
            reason = HostPermissionDenialReason.USER_DENIED
        elif "no interactive user" in result.reason:
            reason = HostPermissionDenialReason.APPROVAL_UNAVAILABLE
        else:
            reason = HostPermissionDenialReason.POLICY_DENIED
        return HostAuthorizationDecision(allowed=False, denial_reason=reason)
