"""Define typed execution authority owned by the application permission facade."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..execution.contracts import Capability, ExecutionBoundary, NetworkProtocol
from ..utils import json_encode, sha256_digest


class GrantEffect(StrEnum):
    """Identify one runtime capability or observed persistent effect."""

    FS_READ = "fs.read"
    FS_CREATE = "fs.create"
    FS_REPLACE = "fs.replace"
    FS_DELETE = "fs.delete"
    FS_RENAME = "fs.rename"
    FS_METADATA = "fs.metadata"
    FS_EXECUTE = "fs.execute"
    PROCESS_SPAWN = "process.spawn"
    PROCESS_SIGNAL = "process.signal"
    NETWORK_CONNECT = "network.connect"
    NETWORK_LISTEN = "network.listen"
    IPC_CONNECT = "ipc.connect"
    SECRET_USE = "secret.use"
    DEVICE_USE = "device.use"
    CACHE_WRITE = "cache.write"
    SYSTEM_CONFIGURE = "system.configure"
    PACKAGE_INSTALL = "package.install"
    PRIVILEGE_ESCALATE = "privilege.escalate"
    HOST_EXECUTE = "host.execute"

    @property
    def capability(self) -> Capability | None:
        """Return the runtime capability represented by this effect, when any."""
        try:
            return Capability(self.value)
        except ValueError:
            return None


class GrantScope(StrEnum):
    """Identify a typed grant lifetime."""

    ONCE = "once"
    PROCESS = "process"
    SESSION = "session"
    WORKSPACE = "workspace"
    USER_POLICY = "user_policy"


class GrantDecision(StrEnum):
    """Identify an allow, deny, or approval-required typed result."""

    ALLOW = "allow"
    DENY = "deny"
    PROMPT = "prompt"


class GrantAuthority(IntEnum):
    """Order rules from product boundaries through user choices."""

    PRODUCT = 0
    ADMINISTRATOR = 1
    USER_DENY = 2
    USER_ALLOW = 3


class VirtualPathTree(BaseModel):
    """Select an absolute virtual path tree and its permitted effects."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["virtual_path_tree"] = "virtual_path_tree"
    root: str = Field(min_length=1)
    effects: frozenset[GrantEffect] = frozenset()

    @model_validator(mode="after")
    def validate_root(self) -> VirtualPathTree:
        """Require a normalized absolute virtual path tree."""
        base = self.root.removesuffix("/**")
        if not base.startswith("/") or "/../" in f"{base}/" or "//" in base:
            raise ValueError("Virtual path roots must be normalized absolute paths.")
        return self


class NetworkEndpoint(BaseModel):
    """Select one pre-resolved exact outbound network endpoint."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["network_endpoint"] = "network_endpoint"
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    protocol: NetworkProtocol
    addresses: tuple[str, ...] = Field(min_length=1)


class NetworkListener(BaseModel):
    """Select one exact inbound publication and command target."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["network_listener"] = "network_listener"
    port: int = Field(ge=1024, le=65535)
    target_port: int = Field(ge=1, le=65535)
    externally_visible: bool = False


class SecretUse(BaseModel):
    """Select a secret audience and reviewed injection mechanism."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["secret_use"] = "secret_use"
    secret_id: str = Field(min_length=1)
    audience: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)


class OpaqueResource(BaseModel):
    """Select one exact named non-path resource."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["opaque"] = "opaque"
    name: str = Field(min_length=1)


ExecutionResource = VirtualPathTree | NetworkEndpoint | NetworkListener | SecretUse | OpaqueResource


class SubjectIdentity(BaseModel):
    """Bind authority to an authenticated tool and executable or profile identity."""

    model_config = ConfigDict(frozen=True)

    tool_id: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    executable_digest: str | None = None
    profile_id: str | None = None
    profile_version: str | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> SubjectIdentity:
        """Require an executable or complete signed-profile identity."""
        profile_complete = self.profile_id is not None and self.profile_version is not None
        if self.executable_digest is None and not profile_complete:
            raise ValueError("A subject needs an executable digest or a complete profile identity.")
        if (self.profile_id is None) != (self.profile_version is None):
            raise ValueError("Profile identity requires both id and version.")
        return self


class GrantConstraints(BaseModel):
    """Constrain a typed grant beyond its resource selector."""

    model_config = ConfigDict(frozen=True)

    max_bytes: int | None = Field(default=None, ge=0)
    max_duration_ns: int | None = Field(default=None, ge=0)
    max_count: int | None = Field(default=None, ge=1)
    runtime_digest: str | None = None
    argument_class: str | None = None


class PolicyGrant(BaseModel):
    """Represent one versioned and revocable typed execution grant."""

    model_config = ConfigDict(frozen=True)

    grant_id: str = Field(min_length=1)
    subject: SubjectIdentity
    boundary: ExecutionBoundary
    effect: GrantEffect
    resource: ExecutionResource
    decision: GrantDecision
    authority: GrantAuthority
    scope: GrantScope
    workspace_binding: str | None = None
    scope_binding: str | None = None
    constraints: GrantConstraints = GrantConstraints()
    policy_version: str = Field(min_length=1)
    issued_at_ns: int = Field(ge=0)
    expires_at_ns: int | None = Field(default=None, ge=0)
    revoked: bool = False
    origin: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_scope(self) -> PolicyGrant:
        """Require bindings implied by the selected lifetime."""
        if self.scope is GrantScope.WORKSPACE and self.workspace_binding is None:
            raise ValueError("Workspace grants require an authenticated workspace binding.")
        if self.scope in {GrantScope.PROCESS, GrantScope.SESSION} and self.scope_binding is None:
            raise ValueError("Process and session grants require a scope binding.")
        if (
            self.scope is GrantScope.USER_POLICY
            and self.workspace_binding is None
            and isinstance(self.resource, VirtualPathTree)
            and self.resource.root == "/workspace/**"
        ):
            raise ValueError("User policy cannot grant a bare global workspace tree.")
        if self.expires_at_ns is not None and self.expires_at_ns < self.issued_at_ns:
            raise ValueError("Grant expiry cannot predate issuance.")
        return self


class PolicyRequest(BaseModel):
    """Describe one fully typed execution authorization request."""

    model_config = ConfigDict(frozen=True)

    subject: SubjectIdentity
    boundary: ExecutionBoundary
    effect: GrantEffect
    resource: ExecutionResource
    workspace_id: str | None = None
    policy_version: str = Field(min_length=1)
    now_ns: int = Field(ge=0)
    constraints: GrantConstraints = GrantConstraints()
    scope_binding: str | None = None


class PolicyExplanation(BaseModel):
    """Provide a safe explanation for one typed policy result."""

    model_config = ConfigDict(frozen=True)

    decision: GrantDecision
    matched_grant_id: str | None
    reason: str
    boundary: ExecutionBoundary
    resource: ExecutionResource
    reusable: bool


class ExecutionAuthorizationResult(BaseModel):
    """Record an atomic typed authorization decision and any installed grants."""

    model_config = ConfigDict(frozen=True)

    requests: tuple[PolicyRequest, ...]
    explanations: tuple[PolicyExplanation, ...]
    decision: GrantDecision
    prompted: bool = False
    approval_scope: GrantScope | None = None
    installed_grant_ids: tuple[str, ...] = ()
    reason: str


class ProfileSignatureVerifier(Protocol):
    """Verify a detached command-profile signature against trusted publisher keys."""

    def verify(self, publisher: str, payload: bytes, signature: bytes) -> bool:
        """Return whether the exact profile payload has a trusted signature."""


class CommandProfile(BaseModel):
    """Describe signed advisory command classification, never runtime authority."""

    model_config = ConfigDict(frozen=True)

    profile_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    executable_digest: str = Field(min_length=1)
    argument_class: str = Field(min_length=1)
    predicted_effects: frozenset[GrantEffect]
    signature: str = Field(min_length=1)

    def payload(self) -> bytes:
        """Return canonical unsigned bytes for detached signature verification."""
        return json_encode(self.model_dump(exclude={"signature"}, mode="json")).encode()

    def verify(self, verifier: ProfileSignatureVerifier) -> None:
        """Require a trusted detached signature before advisory classification."""
        try:
            signature = bytes.fromhex(self.signature)
        except ValueError as error:
            raise ValueError("Command profile signature is not valid hexadecimal.") from error
        if not verifier.verify(self.publisher, self.payload(), signature):
            raise ValueError("Command profile signature verification failed.")

    def subject(self, tool_id: str) -> SubjectIdentity:
        """Return the exact authenticated subject implied by this profile."""
        return SubjectIdentity(
            tool_id=tool_id,
            publisher=self.publisher,
            executable_digest=self.executable_digest,
            profile_id=self.profile_id,
            profile_version=self.version,
        )

    @property
    def digest(self) -> str:
        """Return a content digest for audit and cache invalidation."""
        return sha256_digest(self.model_dump_json())


def resource_matches(granted: ExecutionResource, requested: ExecutionResource) -> bool:
    """Return whether a request is a strict typed subset of a granted resource."""
    if type(granted) is not type(requested):
        return False
    if isinstance(granted, VirtualPathTree) and isinstance(requested, VirtualPathTree):
        root = granted.root.removesuffix("/**").rstrip("/") or "/"
        path = requested.root.removesuffix("/**")
        return (
            path == root or path.startswith(f"{root}/")
        ) and requested.effects <= granted.effects
    return granted == requested


def evaluate_grant(grants: tuple[PolicyGrant, ...], request: PolicyRequest) -> PolicyExplanation:
    """Evaluate one request using deny dominance and narrowest matching authority."""
    matches = tuple(grant for grant in grants if _grant_matches(grant, request))
    denies = tuple(grant for grant in matches if grant.decision is GrantDecision.DENY)
    allows = tuple(grant for grant in matches if grant.decision is GrantDecision.ALLOW)
    prompts = tuple(grant for grant in matches if grant.decision is GrantDecision.PROMPT)
    if denies:
        winner = min(denies, key=_grant_priority)
        decision = GrantDecision.DENY
        reason = "A matching deny grant takes precedence."
    elif allows:
        winner = min(allows, key=_grant_priority)
        decision = GrantDecision.ALLOW
        reason = "A matching typed grant authorizes this bounded effect."
    else:
        winner = min(prompts, key=_grant_priority, default=None)
        decision = GrantDecision.PROMPT
        reason = "No active typed allow grant covers this request."
    return PolicyExplanation(
        decision=decision,
        matched_grant_id=winner.grant_id if winner is not None else None,
        reason=reason,
        boundary=request.boundary,
        resource=request.resource,
        reusable=(
            decision is GrantDecision.ALLOW
            and winner is not None
            and winner.scope is not GrantScope.ONCE
        ),
    )


def _grant_matches(grant: PolicyGrant, request: PolicyRequest) -> bool:
    """Return whether an active grant applies to one fully typed request."""
    if grant.revoked or grant.policy_version != request.policy_version:
        return False
    if grant.expires_at_ns is not None and request.now_ns >= grant.expires_at_ns:
        return False
    if grant.subject != request.subject or grant.boundary is not request.boundary:
        return False
    if grant.effect is not request.effect or not resource_matches(grant.resource, request.resource):
        return False
    if grant.workspace_binding is not None and grant.workspace_binding != request.workspace_id:
        return False
    if grant.scope_binding is not None and grant.scope_binding != request.scope_binding:
        return False
    for field in ("max_bytes", "max_duration_ns", "max_count"):
        allowed = getattr(grant.constraints, field)
        actual = getattr(request.constraints, field)
        if allowed is not None and (actual is None or actual > allowed):
            return False
    return all(
        getattr(grant.constraints, field) in (None, getattr(request.constraints, field))
        for field in ("runtime_digest", "argument_class")
    )


def _grant_priority(grant: PolicyGrant) -> tuple[int, int, int, str]:
    """Return authority then narrowness ordering for matching grants."""
    resource = grant.resource
    path_size = len(resource.root) if isinstance(resource, VirtualPathTree) else 0
    effect_size = len(resource.effects) if isinstance(resource, VirtualPathTree) else 1
    bound_count = sum(value is not None for value in grant.constraints.model_dump().values())
    return (int(grant.authority), -path_size, effect_size - bound_count, grant.grant_id)
