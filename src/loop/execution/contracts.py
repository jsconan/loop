"""Define immutable, boundary-specific execution requests and leases."""

from __future__ import annotations

import re
import secrets
from enum import StrEnum
from ipaddress import ip_address
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .. import constants


class ExecutionBoundary(StrEnum):
    """Identify the independently authorized execution boundary."""

    SANDBOX = "sandbox"
    HOST = "host"


class ExecutionMode(StrEnum):
    """Identify the requested container lifetime."""

    FOREGROUND = "foreground"
    DURABLE_JOB = "durable_job"


class JobOperation(StrEnum):
    """Identify one separately authorized durable-job operation."""

    ATTACH = "attach"
    STDIN = "stdin"
    RESIZE = "resize"
    SIGNAL = "signal"
    STATUS = "status"
    CANCEL = "cancel"
    SUSPEND = "suspend"
    RESUME = "resume"


class JobHandle(BaseModel):
    """Carry an opaque authenticated reference to one durable sandbox job.

    Args:
        job_id (str): Stable public job identity.
        token (str): Unpredictable bearer authenticator retained only by the caller.
    """

    model_config = ConfigDict(frozen=True)

    job_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    token: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)

    @classmethod
    def create(cls, job_id: str) -> JobHandle:
        """Create a handle with a cryptographically unpredictable authenticator.

        Args:
            job_id (str): Stable public job identity.

        Returns:
            JobHandle: Newly authenticated handle.
        """
        return cls(job_id=job_id, token=secrets.token_hex(32))


class TerminalMode(StrEnum):
    """Identify separated-pipe or merged pseudo-terminal foreground I/O."""

    PIPE = "pipe"
    PTY = "pty"


class NetworkProtocol(StrEnum):
    """Identify one broker-enforced network protocol."""

    HTTPS = "https"
    TLS = "tls"
    HTTP = "http"
    TCP = "tcp"
    DNS = "dns"


class SecretMechanism(StrEnum):
    """Identify one supported secret exposure mechanism."""

    REQUEST_HEADER = "request_header"
    RAW_ENVIRONMENT = "raw_environment"
    RAW_FILE = "raw_file"


class NetworkConnectionLease(BaseModel):
    """Pin one authorized outbound destination before sandbox exposure.

    Args:
        hostname (str): Canonical authorized hostname or displayed resolved endpoint.
        port (int): Exact destination and command-facing listener port.
        protocol (NetworkProtocol): Broker protocol and authority enforcement mode.
        addresses (tuple[str, ...]): Pre-resolved public upstream addresses.
        max_connections (int): Maximum accepted downstream connections.
        max_bytes (int): Maximum brokered bytes for the attempt.
    """

    model_config = ConfigDict(frozen=True)

    hostname: str = Field(min_length=1, max_length=253)
    port: int = Field(ge=1, le=65535)
    protocol: NetworkProtocol
    addresses: tuple[str, ...] = Field(min_length=1)
    max_connections: int = Field(
        default=constants.DEFAULT_NETWORK_MAX_CONNECTIONS,
        ge=1,
        le=constants.MAX_NETWORK_MAX_CONNECTIONS,
    )
    max_bytes: int = Field(
        default=constants.DEFAULT_NETWORK_MAX_BYTES,
        ge=1,
        le=constants.MAX_NETWORK_MAX_BYTES,
    )

    @model_validator(mode="after")
    def validate_destination(self) -> NetworkConnectionLease:
        """Reject ambiguous names, duplicate addresses, and non-public upstreams."""
        hostname = self.hostname.rstrip(".").lower()
        labels = hostname.split(".")
        if (
            hostname != self.hostname
            or not hostname
            or any(
                re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) is None
                for label in labels
            )
            or len(set(self.addresses)) != len(self.addresses)
        ):
            raise ValueError("Network destination authority is invalid.")
        parsed = []
        for value in self.addresses:
            try:
                address = ip_address(value)
            except ValueError as error:
                raise ValueError("Network destination address is invalid.") from error
            if not address.is_global:
                raise ValueError("Network destination address is not public.")
            parsed.append(address)
        if self.protocol in {NetworkProtocol.HTTPS, NetworkProtocol.TLS, NetworkProtocol.HTTP}:
            try:
                ip_address(hostname)
            except ValueError:
                pass
            else:
                raise ValueError("Hostname-scoped network leases require a DNS authority.")
        return self


class NetworkListenerLease(BaseModel):
    """Authorize one broker-published inbound TCP listener.

    Args:
        port (int): Parent and guest listener port.
        target_port (int): Fixed command-container destination port.
        externally_visible (bool): Whether publication may bind a wildcard parent address.
        max_connections (int): Maximum concurrent downstream connections.
    """

    model_config = ConfigDict(frozen=True)

    port: int = Field(ge=1024, le=65535)
    target_port: int = Field(ge=1, le=65535)
    externally_visible: bool = False
    max_connections: int = Field(
        default=constants.DEFAULT_NETWORK_MAX_CONNECTIONS,
        ge=1,
        le=constants.MAX_NETWORK_MAX_CONNECTIONS,
    )


class SecretExposure(BaseModel):
    """Bind one secret identifier to an audience and reviewed mechanism.

    Args:
        secret_id (str): Opaque host-secret authority identifier.
        audience (str): Exact authorized remote audience or process identity.
        mechanism (SecretMechanism): Broker or explicit raw exposure shape.
        target (str): Header name, environment name, or absolute private file path.
    """

    model_config = ConfigDict(frozen=True)

    secret_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    audience: str = Field(min_length=1, max_length=253)
    mechanism: SecretMechanism
    target: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_target(self) -> SecretExposure:
        """Constrain raw destinations and operation-level header names."""
        if self.mechanism is SecretMechanism.REQUEST_HEADER:
            if self.target.lower() not in {"authorization", "proxy-authorization"}:
                raise ValueError("Credential header target is unsupported.")
        elif self.mechanism is SecretMechanism.RAW_ENVIRONMENT:
            if not self.target.replace("_", "A").isalnum() or not (
                self.target[0].isalpha() or self.target[0] == "_"
            ):
                raise ValueError("Raw secret environment target is invalid.")
        elif (
            re.fullmatch(
                r"/run/secrets/[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
                r"(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,127})*",
                self.target,
            )
            is None
        ):
            raise ValueError("Raw secret file target is invalid.")
        return self


class Capability(StrEnum):
    """Identify one runtime-enforced effect requested by an execution attempt."""

    WORKSPACE_READ = "workspace.read"
    WORKSPACE_WRITE = "workspace.write"
    PROCESS_SPAWN = "process.spawn"
    PROCESS_SIGNAL = "process.signal"
    NETWORK_CONNECT = "network.connect"
    NETWORK_LISTEN = "network.listen"
    SECRET_USE = "secret.use"
    IPC_CONNECT = "ipc.connect"
    CACHE_WRITE = "cache.write"


class DeltaEffect(StrEnum):
    """Identify one canonical persistent workspace effect."""

    CREATE = "create"
    REPLACE = "replace"
    DELETE = "delete"
    RENAME = "rename"
    METADATA = "metadata"


class WorkspaceDeltaEntry(BaseModel):
    """Describe one inspected persistent effect using only virtual paths.

    Args:
        effect (DeltaEffect): Canonical effect observed in an attempt overlay.
        path (str): Absolute virtual destination path.
        source_path (str | None): Absolute virtual source path for a rename.
        content_digest (str | None): Content identity for created or replaced data.
    """

    model_config = ConfigDict(frozen=True)

    effect: DeltaEffect
    path: str = Field(min_length=1)
    source_path: str | None = None
    content_digest: str | None = None

    @model_validator(mode="after")
    def validate_virtual_paths(self) -> WorkspaceDeltaEntry:
        """Require absolute virtual paths and a source only for renames."""
        if not self.path.startswith("/"):
            raise ValueError("Delta paths must be absolute virtual paths.")
        if self.effect is DeltaEffect.RENAME:
            if self.source_path is None or not self.source_path.startswith("/"):
                raise ValueError("A rename requires an absolute virtual source path.")
        elif self.source_path is not None:
            raise ValueError("Only a rename may have a source path.")
        return self


class WorkspaceDelta(BaseModel):
    """Capture the complete immutable inspected delta of one stopped attempt.

    Args:
        entries (tuple[WorkspaceDeltaEntry, ...]): Canonically ordered virtual effects.
    """

    model_config = ConfigDict(frozen=True)

    entries: tuple[WorkspaceDeltaEntry, ...] = ()


class ExecutionLease(BaseModel):
    """Bind an authorized attempt to identity, policy, and a bounded lifetime.

    Args:
        lease_id (str): Immutable unique authorization identifier.
        workspace_id (str): Authenticated workspace identity, never a host path.
        agent_run_id (str): Owning agent execution identity.
        policy_version (str): Policy version used to issue the authorization.
        runtime_digest (str): Digest of the selected sandbox image manifest.
        expires_at_ns (int): Monotonic-expiry instant represented in nanoseconds.
        capabilities (frozenset[Capability]): Runtime effects authorized for this attempt.
    """

    model_config = ConfigDict(frozen=True)

    lease_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    agent_run_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    runtime_digest: str = Field(min_length=1)
    expires_at_ns: int = Field(ge=0)
    capabilities: frozenset[Capability] = frozenset()


class BaseExecutionRequest(BaseModel):
    """Carry common request fields without permitting a boundary conversion.

    Args:
        request_id (str): Immutable request identifier.
        lease (ExecutionLease): Sandbox authorization bound to this request.
        boundary (Literal[ExecutionBoundary.SANDBOX]): Fixed sandbox boundary discriminator.
        cwd (str): Absolute virtual working directory.
        environment (tuple[tuple[str, str], ...]): Explicit guest environment entries.
        mode (ExecutionMode): Foreground or explicit durable-job lifecycle.
        deadline_seconds (float): Positive wall-clock execution deadline.
        output_limit_bytes (int): Maximum retained bytes for each output stream.
        terminal (TerminalMode): Separated pipes or one merged pseudo-terminal.
        terminal_columns (int | None): Initial PTY width, required only in PTY mode.
        terminal_rows (int | None): Initial PTY height, required only in PTY mode.
        network_connections (tuple[NetworkConnectionLease, ...]): Authorized outbound leases.
        network_listeners (tuple[NetworkListenerLease, ...]): Authorized inbound publications.
        secret_exposures (tuple[SecretExposure, ...]): Authorized brokered or raw secrets.
        stdin (bytes): Bounded input delivered before explicit EOF.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    lease: ExecutionLease
    boundary: Literal[ExecutionBoundary.SANDBOX] = ExecutionBoundary.SANDBOX
    cwd: str = "/workspace"
    environment: tuple[tuple[str, str], ...] = ()
    mode: ExecutionMode = ExecutionMode.FOREGROUND
    deadline_seconds: float = Field(
        default=constants.DEFAULT_COMMAND_TIMEOUT,
        gt=0,
        le=constants.MAX_EXECUTION_TIMEOUT_SECONDS,
    )
    output_limit_bytes: int = Field(
        default=constants.DEFAULT_EXECUTION_OUTPUT_BYTES,
        ge=1,
        le=constants.MAX_EXECUTION_OUTPUT_BYTES,
    )
    terminal: TerminalMode = TerminalMode.PIPE
    terminal_columns: int | None = Field(
        default=None, ge=1, le=constants.MAX_EXECUTION_TERMINAL_DIMENSION
    )
    terminal_rows: int | None = Field(
        default=None, ge=1, le=constants.MAX_EXECUTION_TERMINAL_DIMENSION
    )
    network_connections: tuple[NetworkConnectionLease, ...] = Field(
        default=(), max_length=constants.MAX_EXECUTION_COLLECTION_ITEMS
    )
    network_listeners: tuple[NetworkListenerLease, ...] = Field(
        default=(), max_length=constants.MAX_EXECUTION_COLLECTION_ITEMS
    )
    secret_exposures: tuple[SecretExposure, ...] = Field(
        default=(), max_length=constants.MAX_EXECUTION_COLLECTION_ITEMS
    )
    stdin: bytes = Field(default=b"", max_length=constants.MAX_EXECUTION_STDIN_BYTES)

    @model_validator(mode="after")
    def validate_virtual_cwd(self) -> BaseExecutionRequest:
        """Require a stable virtual working directory."""
        if not self.cwd.startswith("/"):
            raise ValueError("Execution cwd must be an absolute virtual path.")
        dimensions = (self.terminal_columns, self.terminal_rows)
        if self.terminal is TerminalMode.PTY and None in dimensions:
            raise ValueError("PTY execution requires initial terminal dimensions.")
        if self.terminal is TerminalMode.PIPE and any(value is not None for value in dimensions):
            raise ValueError("Pipe execution cannot carry terminal dimensions.")
        if bool(self.network_connections) != (
            Capability.NETWORK_CONNECT in self.lease.capabilities
        ):
            raise ValueError("Outbound network authority and leases must match exactly.")
        if bool(self.network_listeners) != (Capability.NETWORK_LISTEN in self.lease.capabilities):
            raise ValueError("Inbound network authority and leases must match exactly.")
        if bool(self.secret_exposures) != (Capability.SECRET_USE in self.lease.capabilities):
            raise ValueError("Secret authority and exposures must match exactly.")
        audiences = {lease.hostname for lease in self.network_connections}
        for exposure in self.secret_exposures:
            if (
                exposure.mechanism is SecretMechanism.REQUEST_HEADER
                and exposure.audience not in audiences
            ):
                raise ValueError("Credential audience lacks a matching network lease.")
        connection_keys = [
            (lease.hostname, lease.port, lease.protocol) for lease in self.network_connections
        ]
        listener_ports = [lease.port for lease in self.network_listeners]
        secret_targets = [
            (exposure.audience, exposure.mechanism, exposure.target)
            for exposure in self.secret_exposures
        ]
        if (
            len(set(connection_keys)) != len(connection_keys)
            or len(set(listener_ports)) != len(listener_ports)
            or len(set(secret_targets)) != len(secret_targets)
        ):
            raise ValueError("Execution broker leases must be unique.")
        return self


class ShellExecutionRequest(BaseExecutionRequest):
    """Request one opaque POSIX shell script inside the sandbox.

    Args:
        request_id (str): Immutable request identifier.
        lease (ExecutionLease): Sandbox authorization bound to this request.
        cwd (str): Absolute virtual working directory.
        environment (tuple[tuple[str, str], ...]): Explicit virtual environment entries.
        mode (ExecutionMode): Foreground or explicit durable-job lifecycle.
        deadline_seconds (float): Positive wall-clock execution deadline.
        output_limit_bytes (int): Maximum retained bytes for each output stream.
        kind (Literal["shell"]): Request discriminator.
        script (str): Opaque shell source passed only to the sandbox shell.
    """

    kind: Literal["shell"] = "shell"
    script: str = Field(min_length=1)


class DirectExecutionRequest(BaseExecutionRequest):
    """Request direct argv execution inside the sandbox.

    Args:
        request_id (str): Immutable request identifier.
        lease (ExecutionLease): Sandbox authorization bound to this request.
        cwd (str): Absolute virtual working directory.
        environment (tuple[tuple[str, str], ...]): Explicit virtual environment entries.
        mode (ExecutionMode): Foreground or explicit durable-job lifecycle.
        deadline_seconds (float): Positive wall-clock execution deadline.
        output_limit_bytes (int): Maximum retained bytes for each output stream.
        kind (Literal["direct"]): Request discriminator.
        argv (tuple[str, ...]): Exact guest executable and arguments.
    """

    kind: Literal["direct"] = "direct"
    argv: tuple[str, ...] = Field(min_length=1)


SandboxExecutionRequest = Annotated[
    ShellExecutionRequest | DirectExecutionRequest,
    Field(discriminator="kind"),
]


class HostExecutionRequest(BaseModel):
    """Request a new, separately authorized host operation.

    Args:
        request_id (str): Immutable request identifier.
        executable (str): Absolute host executable path for privileged resolution.
        argv (tuple[str, ...]): Exact host executable and arguments.
        cwd (str): Absolute host working directory authenticated by the broker.
        display_cwd (str): Sanitized user-facing working-directory label.
        environment (tuple[tuple[str, str], ...]): Explicit environment additions to disclose.
        resource_class (str): Bounded description of the required host resource.
        reason (str): User-facing reason for crossing the host boundary.
        deadline_seconds (float): Best-effort host process deadline.
        output_limit_bytes (int): Maximum retained bytes for each output stream.
        boundary (Literal[ExecutionBoundary.HOST]): Fixed host boundary discriminator.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    executable: str = Field(min_length=1)
    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str = Field(min_length=1)
    display_cwd: str = Field(min_length=1, max_length=512)
    environment: tuple[tuple[str, str], ...] = Field(default=(), max_length=64)
    resource_class: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1)
    deadline_seconds: float = Field(
        default=constants.DEFAULT_COMMAND_TIMEOUT,
        gt=0,
        le=constants.MAX_EXECUTION_TIMEOUT_SECONDS,
    )
    output_limit_bytes: int = Field(
        default=constants.DEFAULT_EXECUTION_OUTPUT_BYTES,
        ge=1,
        le=constants.MAX_EXECUTION_OUTPUT_BYTES,
    )
    boundary: Literal[ExecutionBoundary.HOST] = ExecutionBoundary.HOST

    @model_validator(mode="after")
    def validate_host_shape(self) -> HostExecutionRequest:
        """Reject ambiguous paths, argv, environment, and display text.

        Returns:
            HostExecutionRequest: Validated explicit request.

        Raises:
            ValueError: If path, argv, environment, or disclosure data is unsafe or ambiguous.
        """
        if not self.executable.startswith("/") or not self.cwd.startswith("/"):
            raise ValueError("Host executable and cwd must be absolute paths.")
        if self.argv[0] != self.executable:
            raise ValueError("Host argv must begin with the requested executable.")
        if any("\0" in value for value in (*self.argv, self.cwd, self.executable)):
            raise ValueError("Host request values cannot contain NUL bytes.")
        keys = [key for key, _value in self.environment]
        if len(set(keys)) != len(keys) or any(
            not key or not key.replace("_", "A").isalnum() or "\0" in value
            for key, value in self.environment
        ):
            raise ValueError("Host request environment is invalid.")
        if any(character in self.display_cwd for character in "\r\n\0"):
            raise ValueError("Host display cwd must be a single sanitized line.")
        if any(
            character in value
            for value in (self.reason, self.resource_class)
            for character in "\r\n\0"
        ):
            raise ValueError("Host disclosure fields must be sanitized single lines.")
        return self


class BackendAttestation(BaseModel):
    """Record the verified identity and confinement evidence of one backend.

    Args:
        backend_id (str): Pinned backend implementation identity.
        runtime_digest (str): Attested OCI runtime digest.
        policy_digest (str): Attested effective upstream policy digest.
        evidence (tuple[tuple[str, str], ...]): Sanitized immutable attestation facts.
    """

    model_config = ConfigDict(frozen=True)

    backend_id: str = Field(min_length=1)
    runtime_digest: str = Field(min_length=1)
    policy_digest: str = Field(min_length=1)
    evidence: tuple[tuple[str, str], ...] = ()
