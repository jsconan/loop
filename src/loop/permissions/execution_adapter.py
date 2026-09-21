"""Translate sandbox capabilities and observed deltas into canonical permission requests."""

from __future__ import annotations

import shlex
import time
from collections.abc import Callable, Iterable

from ..execution.contracts import (
    Capability,
    DeltaEffect,
    ExecutionBoundary,
    ExecutionLease,
    JobOperation,
    NetworkConnectionLease,
    NetworkListenerLease,
    SandboxExecutionRequest,
    SecretExposure,
)
from ..execution.vfs import CanonicalDelta
from .execution import (
    GrantConstraints,
    GrantDecision,
    GrantEffect,
    NetworkEndpoint,
    NetworkListener,
    OpaqueResource,
    PolicyRequest,
    SecretUse,
    SubjectIdentity,
    VirtualPathTree,
)
from .manager import PermissionManager

_CAPABILITY_EFFECTS = {
    Capability.WORKSPACE_READ: GrantEffect.FS_READ,
    Capability.PROCESS_SPAWN: GrantEffect.PROCESS_SPAWN,
    Capability.PROCESS_SIGNAL: GrantEffect.PROCESS_SIGNAL,
    Capability.NETWORK_CONNECT: GrantEffect.NETWORK_CONNECT,
    Capability.NETWORK_LISTEN: GrantEffect.NETWORK_LISTEN,
    Capability.SECRET_USE: GrantEffect.SECRET_USE,
    Capability.IPC_CONNECT: GrantEffect.IPC_CONNECT,
    Capability.CACHE_WRITE: GrantEffect.CACHE_WRITE,
}
_DELTA_EFFECTS = {
    DeltaEffect.CREATE: GrantEffect.FS_CREATE,
    DeltaEffect.REPLACE: GrantEffect.FS_REPLACE,
    DeltaEffect.DELETE: GrantEffect.FS_DELETE,
    DeltaEffect.RENAME: GrantEffect.FS_RENAME,
    DeltaEffect.METADATA: GrantEffect.FS_METADATA,
}
_EFFECT_DESCRIPTIONS = {
    GrantEffect.FS_CREATE: "Create a file or directory",
    GrantEffect.FS_REPLACE: "Replace an existing file or directory",
    GrantEffect.FS_DELETE: "Delete a file or directory",
    GrantEffect.FS_RENAME: "Rename a file or directory",
    GrantEffect.FS_METADATA: "Change file or directory metadata",
}


class ExecutionPermissionAdapter:
    """Route all sandbox execution authorization through one permission manager.

    Args:
        manager (PermissionManager): Application-facing permission facade and sole policy store.
        subject (SubjectIdentity): Authenticated ordinary-command subject.
        workspace_id (str): Durable identity from the active workspace repository.
        policy_version (str): Typed policy version bound into issued leases and grants.
        scope_binding (str | Callable[[], str]): Current supervised identity or resolver.
        clock_ns (Callable[[], int]): Trusted timestamp source for expiry decisions.
    """

    manager: PermissionManager
    subject: SubjectIdentity
    workspace_id: str
    policy_version: str
    scope_binding: str | Callable[[], str]
    _clock_ns: Callable[[], int]

    def __init__(
        self,
        manager: PermissionManager,
        subject: SubjectIdentity,
        workspace_id: str,
        policy_version: str,
        scope_binding: str | Callable[[], str],
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.manager = manager
        self.subject = subject
        self.workspace_id = workspace_id
        self.policy_version = policy_version
        self.scope_binding = scope_binding
        self._clock_ns = clock_ns

    def authorize_capabilities(
        self,
        capabilities: Iterable[Capability],
        *,
        runtime_digest: str,
        duration_ns: int,
        network_connections: tuple[NetworkConnectionLease, ...] = (),
        network_listeners: tuple[NetworkListenerLease, ...] = (),
        secret_exposures: tuple[SecretExposure, ...] = (),
    ) -> bool:
        """Authorize predicted privileged exposure before a sandbox attempt starts.

        Workspace writes are intentionally omitted: they affect only the private attempt overlay
        until the separately observed and frozen delta is authorized for publication.

        Args:
            capabilities (Iterable[Capability]): Predicted runtime capabilities.
            runtime_digest (str): Attested sandbox image manifest identity.
            duration_ns (int): Maximum attempt lease duration.
            network_connections (tuple[NetworkConnectionLease, ...]): Exact outbound leases.
            network_listeners (tuple[NetworkListenerLease, ...]): Exact inbound leases.
            secret_exposures (tuple[SecretExposure, ...]): Exact secret uses.

        Returns:
            bool: Whether all authority-bearing capabilities are authorized.
        """
        capability_set = frozenset(capabilities)
        if (
            (Capability.NETWORK_CONNECT in capability_set) != bool(network_connections)
            or (Capability.NETWORK_LISTEN in capability_set) != bool(network_listeners)
            or (Capability.SECRET_USE in capability_set) != bool(secret_exposures)
        ):
            return False
        requests = []
        for capability in capability_set:
            if capability is Capability.WORKSPACE_WRITE:
                continue
            if capability in {
                Capability.NETWORK_CONNECT,
                Capability.NETWORK_LISTEN,
                Capability.SECRET_USE,
            }:
                continue
            effect = _CAPABILITY_EFFECTS.get(capability)
            if effect is None:
                return False
            resource = (
                VirtualPathTree(root="/workspace/**", effects=frozenset({effect}))
                if capability is Capability.WORKSPACE_READ
                else OpaqueResource(name="loop-managed-attempt")
            )
            requests.append(
                self._request(
                    effect,
                    resource,
                    constraints=GrantConstraints(
                        max_duration_ns=duration_ns,
                        runtime_digest=runtime_digest,
                    ),
                )
            )
        for lease in network_connections:
            requests.append(
                self._request(
                    GrantEffect.NETWORK_CONNECT,
                    NetworkEndpoint(
                        host=lease.hostname,
                        port=lease.port,
                        protocol=lease.protocol,
                        addresses=lease.addresses,
                    ),
                    constraints=GrantConstraints(
                        max_bytes=lease.max_bytes,
                        max_count=lease.max_connections,
                        max_duration_ns=duration_ns,
                        runtime_digest=runtime_digest,
                    ),
                )
            )
        for lease in network_listeners:
            requests.append(
                self._request(
                    GrantEffect.NETWORK_LISTEN,
                    NetworkListener(
                        port=lease.port,
                        target_port=lease.target_port,
                        externally_visible=lease.externally_visible,
                    ),
                    constraints=GrantConstraints(
                        max_count=lease.max_connections,
                        max_duration_ns=duration_ns,
                        runtime_digest=runtime_digest,
                    ),
                )
            )
        for exposure in secret_exposures:
            requests.append(
                self._request(
                    GrantEffect.SECRET_USE,
                    SecretUse(
                        secret_id=exposure.secret_id,
                        audience=exposure.audience,
                        mechanism=exposure.mechanism.value,
                    ),
                    constraints=GrantConstraints(
                        max_duration_ns=duration_ns,
                        runtime_digest=runtime_digest,
                    ),
                )
            )
        return self.manager.authorize_execution(tuple(requests)).decision is GrantDecision.ALLOW

    def authorize_publication(
        self, request: SandboxExecutionRequest, delta: CanonicalDelta
    ) -> bool:
        """Authorize every exact effect in one frozen representable observed delta.

        Args:
            request (SandboxExecutionRequest): Attempt whose stopped overlay was inspected.
            delta (CanonicalDelta): Complete canonical observed persistent effects.

        Returns:
            bool: Whether the complete delta may be published atomically.
        """
        requests = []
        for entry in delta.entries:
            effect = _DELTA_EFFECTS[entry.effect]
            paths = (entry.destination_path,)
            if entry.source_path is not None:
                paths = (entry.source_path, *paths)
            for path in paths:
                resource = VirtualPathTree(root=f"/workspace/{path}", effects=frozenset({effect}))
                requests.append(
                    self._request(
                        effect,
                        resource,
                        constraints=GrantConstraints(
                            max_bytes=entry.content.size if entry.content is not None else 0,
                            runtime_digest=request.lease.runtime_digest,
                        ),
                    )
                )
        prompt = self._publication_prompt(request, tuple(requests))
        return (
            self.manager.authorize_execution(tuple(requests), prompt=prompt).decision
            is GrantDecision.ALLOW
        )

    @staticmethod
    def _publication_prompt(
        request: SandboxExecutionRequest, requests: tuple[PolicyRequest, ...]
    ) -> str:
        """Explain an observed workspace change in the originating command's context."""
        command = request.script if hasattr(request, "script") else shlex.join(request.argv)
        lines = [
            "Approval required to save command changes",
            f"Command: {command}",
            f"Working directory: {request.cwd}",
            "Purpose: Save the following changes made by this sandboxed command to your workspace:",
        ]
        for policy_request in requests:
            resource = policy_request.resource
            lines.append(f"- {_EFFECT_DESCRIPTIONS[policy_request.effect]}: {resource.root}")
        lines.extend(
            (
                (
                    "Boundary: sandbox (the command has finished; only these observed workspace "
                    "changes will be published)"
                ),
                (
                    "Future coverage: Allow once saves only this result. A remembered approval "
                    "covers only the same change type and path under the same workspace, subject, "
                    "runtime constraints, and sandbox boundary."
                ),
            )
        )
        return "\n".join(lines)

    def authorize_job_operation(
        self,
        lease: ExecutionLease,
        operation: JobOperation,
        job_id: str,
    ) -> bool:
        """Authorize one exact operation on an authenticated durable-job handle.

        Args:
            lease (ExecutionLease): Fresh operation lease bound to the active subject.
            operation (JobOperation): Exact lifecycle action requested by the caller.
            job_id (str): Authenticated durable job identity.

        Returns:
            bool: Whether the retained permission facade authorizes this operation.
        """
        effect = (
            GrantEffect.PROCESS_SPAWN
            if operation is JobOperation.STATUS
            else GrantEffect.PROCESS_SIGNAL
        )
        request = self._request(
            effect,
            OpaqueResource(name=f"durable-job:{job_id}:{operation.value}"),
            constraints=GrantConstraints(
                max_duration_ns=max(0, lease.expires_at_ns - time.monotonic_ns()),
                runtime_digest=lease.runtime_digest,
            ),
        )
        return self.manager.authorize_execution((request,)).decision is GrantDecision.ALLOW

    def _request(
        self,
        effect: GrantEffect,
        resource: VirtualPathTree | NetworkEndpoint | NetworkListener | SecretUse | OpaqueResource,
        *,
        constraints: GrantConstraints,
    ) -> PolicyRequest:
        """Build one request bound to the active authenticated workspace and session."""
        return PolicyRequest(
            subject=self.subject,
            boundary=ExecutionBoundary.SANDBOX,
            effect=effect,
            resource=resource,
            workspace_id=self.workspace_id,
            policy_version=self.policy_version,
            now_ns=self._clock_ns(),
            constraints=constraints,
            scope_binding=(
                self.scope_binding() if callable(self.scope_binding) else self.scope_binding
            ),
        )
