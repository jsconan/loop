"""Build ordinary sandbox requests from the public command interface."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from uuid import uuid4

from .. import constants
from ..permissions.execution_adapter import ExecutionPermissionAdapter
from .contracts import (
    Capability,
    ExecutionLease,
    ExecutionMode,
    JobHandle,
    JobOperation,
    NetworkConnectionLease,
    NetworkListenerLease,
    SecretExposure,
    ShellExecutionRequest,
    TerminalMode,
)
from .results import CapabilityDenied, ExecutionResult, InfrastructureFailure, UnsupportedCapability
from .service import ExecutionService

_ORDINARY_CAPABILITIES = frozenset(
    {
        Capability.WORKSPACE_READ,
        Capability.WORKSPACE_WRITE,
        Capability.PROCESS_SPAWN,
        Capability.PROCESS_SIGNAL,
    }
)
_LOGGER = logging.getLogger(__name__)


class SandboxCommandExecutor:
    """Execute opaque POSIX shell source through one authorized sandbox service.

    Args:
        service (ExecutionService): Sandbox-only execution boundary.
        permissions (ExecutionPermissionAdapter): Canonical permission adapter.
        workspace_id (str): Authenticated active workspace identity.
        agent_run_id (str | Callable[[], str]): Supervised agent execution identity or resolver.
        runtime_digest (str): Sandbox image source identity or already-attested manifest digest.
        runtime_resolver (Callable[[], str] | None): Lazy ``ensure_sandbox`` boundary that returns
            the one attested local image manifest digest.
        policy_version (str): Policy version bound into execution leases.
        output_limit_bytes (int): Per-stream retained output bound.
        supports_network_effects (bool): Whether the composed backend enforces every declared
            network constraint. Defaults to ``False``.
        supports_secret_exposures (bool): Whether the composed backend has an application-owned
            secret authority. Defaults to ``False``.
    """

    service: ExecutionService
    permissions: ExecutionPermissionAdapter
    workspace_id: str
    agent_run_id: str | Callable[[], str]
    runtime_digest: str
    runtime_resolver: Callable[[], str] | None
    runtime_ready: bool
    policy_version: str
    output_limit_bytes: int
    supports_network_effects: bool
    supports_secret_exposures: bool

    def __init__(
        self,
        service: ExecutionService,
        permissions: ExecutionPermissionAdapter,
        workspace_id: str,
        agent_run_id: str | Callable[[], str],
        runtime_digest: str,
        runtime_resolver: Callable[[], str] | None = None,
        policy_version: str = constants.EXECUTION_POLICY_VERSION,
        output_limit_bytes: int = constants.DEFAULT_EXECUTION_OUTPUT_BYTES,
        supports_network_effects: bool = False,
        supports_secret_exposures: bool = False,
    ) -> None:
        self.service = service
        self.permissions = permissions
        self.workspace_id = workspace_id
        self.agent_run_id = agent_run_id
        self.runtime_digest = runtime_digest
        self.policy_version = policy_version
        self.runtime_resolver = runtime_resolver
        self.runtime_ready = runtime_resolver is None
        self.output_limit_bytes = output_limit_bytes
        self.supports_network_effects = supports_network_effects
        self.supports_secret_exposures = supports_secret_exposures

    def execute(
        self,
        script: str,
        cwd: str,
        timeout: float,
        *,
        request_id: str | None = None,
        terminal: TerminalMode = TerminalMode.PIPE,
        terminal_columns: int | None = None,
        terminal_rows: int | None = None,
        network_connections: tuple[NetworkConnectionLease, ...] = (),
        network_listeners: tuple[NetworkListenerLease, ...] = (),
        secret_exposures: tuple[SecretExposure, ...] = (),
        stdin: bytes = b"",
        cancellation: Callable[[], bool] = lambda: False,
    ) -> ExecutionResult:
        """Execute one shell program after authorizing pre-exposure capabilities.

        Args:
            script (str): Opaque POSIX shell source parsed only inside the sandbox.
            cwd (str): Absolute virtual working directory.
            timeout (float): Complete foreground lifecycle deadline in seconds.
            request_id (str | None): Stable caller identifier or a generated opaque ID.
            terminal (TerminalMode): Separated pipes or one merged pseudo-terminal.
            terminal_columns (int | None): Initial PTY width.
            terminal_rows (int | None): Initial PTY height.
            network_connections (tuple[NetworkConnectionLease, ...]): Exact outbound leases.
            network_listeners (tuple[NetworkListenerLease, ...]): Exact inbound leases.
            secret_exposures (tuple[SecretExposure, ...]): Exact audience-bound secret uses.
            stdin (bytes): Bounded input delivered before explicit EOF.
            cancellation (Callable[[], bool]): Predicate requesting cancellation.

        Returns:
            ExecutionResult: Closed sandbox result with bounded output.
        """
        identifier = request_id or str(uuid4())
        try:
            runtime_digest = self._resolve_runtime()
        except Exception as error:  # Preparation failures become closed public results.
            diagnostic_id = f"diag_{uuid4().hex}"
            _LOGGER.error(
                "Managed sandbox preparation failed (%s).",
                diagnostic_id,
                exc_info=error,
            )
            return InfrastructureFailure(
                request_id=identifier,
                diagnostic_id=diagnostic_id,
            )
        if (network_connections or network_listeners) and not self.supports_network_effects:
            return UnsupportedCapability(request_id=identifier, capability="network_byte_limits")
        if secret_exposures and not self.supports_secret_exposures:
            return UnsupportedCapability(request_id=identifier, capability="secret_authority")
        duration_ns = int(timeout * 1_000_000_000)
        capabilities = set(_ORDINARY_CAPABILITIES)
        if network_connections:
            capabilities.add(Capability.NETWORK_CONNECT)
        if network_listeners:
            capabilities.add(Capability.NETWORK_LISTEN)
        if secret_exposures:
            capabilities.add(Capability.SECRET_USE)
        requested_capabilities = frozenset(capabilities)
        if not self.permissions.authorize_capabilities(
            requested_capabilities,
            runtime_digest=runtime_digest,
            duration_ns=duration_ns,
            network_connections=network_connections,
            network_listeners=network_listeners,
            secret_exposures=secret_exposures,
        ):
            return CapabilityDenied(request_id=identifier, capability="command.launch")
        request = ShellExecutionRequest(
            request_id=identifier,
            lease=ExecutionLease(
                lease_id=str(uuid4()),
                workspace_id=self.workspace_id,
                agent_run_id=(
                    self.agent_run_id() if callable(self.agent_run_id) else self.agent_run_id
                ),
                policy_version=self.policy_version,
                runtime_digest=runtime_digest,
                expires_at_ns=time.monotonic_ns() + duration_ns,
                capabilities=requested_capabilities,
            ),
            script=script,
            cwd=cwd,
            deadline_seconds=timeout,
            output_limit_bytes=self.output_limit_bytes,
            terminal=terminal,
            terminal_columns=terminal_columns,
            terminal_rows=terminal_rows,
            network_connections=network_connections,
            network_listeners=network_listeners,
            secret_exposures=secret_exposures,
            stdin=stdin,
        )
        return self.service.execute(request, cancellation)

    def close(self) -> None:
        """Release an owned platform adapter when it exposes deterministic cleanup."""
        close = getattr(self.service.adapter, "close", None)
        if callable(close):
            close()

    def manage_sandbox(self, *, delete: bool) -> bool:
        """Stop or delete platform sandbox state and invalidate cached readiness.

        Args:
            delete (bool): Delete the private VM when ``True``; otherwise stop it.

        Returns:
            bool: Whether owned sandbox state existed.

        Raises:
            NotImplementedError: If the platform adapter does not support management.
            RuntimeError: If platform cleanup fails.
        """
        manage = getattr(self.service.adapter, "manage_sandbox", None)
        if not callable(manage):
            raise NotImplementedError("Sandbox management is unavailable on this platform.")
        managed = manage(delete=delete)
        self.runtime_ready = False
        self.runtime_digest = ""
        return managed

    def sandbox_status(self) -> tuple[str, str] | None:
        """Return the attested active-workspace sandbox state.

        Returns:
            tuple[str, str] | None: Instance name and lifecycle state, or ``None`` when absent.

        Raises:
            NotImplementedError: If the platform adapter does not support status inspection.
            RuntimeError: If platform inspection fails.
        """
        status = getattr(self.service.adapter, "sandbox_status", None)
        if not callable(status):
            raise NotImplementedError("Sandbox status is unavailable on this platform.")
        return status()

    def list_sandboxes(self) -> tuple[tuple[str, str], ...]:
        """List attested sandboxes owned by the active workspace.

        Returns:
            tuple[tuple[str, str], ...]: Owned instance names and lifecycle states.

        Raises:
            NotImplementedError: If the platform adapter does not support sandbox inventory.
            RuntimeError: If platform inspection fails.
        """
        inventory = getattr(self.service.adapter, "list_sandboxes", None)
        if not callable(inventory):
            raise NotImplementedError("Sandbox inventory is unavailable on this platform.")
        return inventory()

    def start_job(
        self,
        script: str,
        cwd: str,
        timeout: float,
        *,
        request_id: str | None = None,
        terminal_columns: int = 80,
        terminal_rows: int = 24,
    ) -> JobHandle | ExecutionResult:
        """Start one explicit PTY durable job through the shared sandbox lifecycle.

        Args:
            script (str): Opaque POSIX shell source.
            cwd (str): Absolute virtual working directory.
            timeout (float): Maximum authorized job lifetime in seconds.
            request_id (str | None): Stable public job identity or generated identifier.
            terminal_columns (int): Initial PTY width.
            terminal_rows (int): Initial PTY height.

        Returns:
            JobHandle | ExecutionResult: Authenticated handle or a closed authorization result.
        """
        identifier = request_id or str(uuid4())
        duration_ns = int(timeout * 1_000_000_000)
        try:
            runtime_digest = self._resolve_runtime()
        except Exception as error:  # Preparation failures become closed public results.
            diagnostic_id = f"diag_{uuid4().hex}"
            _LOGGER.error(
                "Managed sandbox preparation failed (%s).",
                diagnostic_id,
                exc_info=error,
            )
            return InfrastructureFailure(
                request_id=identifier,
                diagnostic_id=diagnostic_id,
            )
        if not self.permissions.authorize_capabilities(
            _ORDINARY_CAPABILITIES,
            runtime_digest=runtime_digest,
            duration_ns=duration_ns,
        ):
            return CapabilityDenied(request_id=identifier, capability="durable_job.start")
        request = ShellExecutionRequest(
            request_id=identifier,
            lease=self._lease(_ORDINARY_CAPABILITIES, duration_ns, runtime_digest),
            script=script,
            cwd=cwd,
            mode=ExecutionMode.DURABLE_JOB,
            deadline_seconds=timeout,
            output_limit_bytes=self.output_limit_bytes,
            terminal=TerminalMode.PTY,
            terminal_columns=terminal_columns,
            terminal_rows=terminal_rows,
        )
        starter = getattr(self.service.adapter, "start_job", None)
        if not callable(starter):
            return UnsupportedCapability(request_id=identifier, capability="durable_job")
        return starter(request)

    def job_lease(self, operation: JobOperation, timeout: float) -> ExecutionLease:
        """Issue a fresh bounded lease reauthorized by the durable lifecycle owner.

        Args:
            operation (JobOperation): Exact requested lifecycle operation.
            timeout (float): Positive operation authorization lifetime in seconds.

        Returns:
            ExecutionLease: Fresh workspace, agent, runtime, and capability-bound lease.
        """
        capabilities = frozenset(
            {
                Capability.PROCESS_SPAWN
                if operation is JobOperation.STATUS
                else Capability.PROCESS_SIGNAL
            }
        )
        return self._lease(
            capabilities,
            int(timeout * 1_000_000_000),
            self._resolve_runtime(),
        )

    def job_manager(self) -> object:
        """Return the composed durable lifecycle owner or reject unsupported composition.

        Returns:
            object: Platform lifecycle facade exposing authenticated durable operations.

        Raises:
            RuntimeError: If the selected platform has no durable lifecycle.
        """
        resolver = getattr(self.service.adapter, "job_manager", None)
        if not callable(resolver):
            raise TypeError("Durable jobs are unavailable.")
        return resolver()

    def _lease(
        self,
        capabilities: frozenset[Capability],
        duration_ns: int,
        runtime_digest: str,
    ) -> ExecutionLease:
        """Build one fresh lease bound to the active logical agent."""
        return ExecutionLease(
            lease_id=str(uuid4()),
            workspace_id=self.workspace_id,
            agent_run_id=(
                self.agent_run_id() if callable(self.agent_run_id) else self.agent_run_id
            ),
            policy_version=self.policy_version,
            runtime_digest=runtime_digest,
            expires_at_ns=time.monotonic_ns() + duration_ns,
            capabilities=capabilities,
        )

    def _resolve_runtime(self) -> str:
        """Return the cached or freshly prepared sandbox image manifest digest."""
        if self.runtime_ready:
            return self.runtime_digest
        if self.runtime_resolver is None:  # pragma: no cover - constructor invariant.
            raise RuntimeError("Sandbox runtime resolver is unavailable.")
        digest = self.runtime_resolver()
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError("Managed sandbox resolver returned an invalid digest.")
        self.runtime_digest = digest
        self.runtime_ready = True
        return digest
