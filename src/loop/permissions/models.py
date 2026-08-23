"""Define typed operations, policies, authorization decisions, and recording protocols."""

from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class Action(StrEnum):
    """Identify one authority-bearing effect planned by a tool."""

    FILESYSTEM_LIST = "filesystem.list"
    FILESYSTEM_READ = "filesystem.read"
    FILESYSTEM_CREATE = "filesystem.create"
    FILESYSTEM_REPLACE = "filesystem.replace"
    FILESYSTEM_DELETE = "filesystem.delete"
    NETWORK_REQUEST = "network.request"
    PROCESS_EXECUTE = "process.execute"
    SESSION_MUTATE = "session.mutate"

    @property
    def icon(self) -> str:
        """Return the icon used to identify this action in approval prompts.

        Returns:
            str: Icon representing the requested authority.
        """
        return _ACTION_ICONS[self]


_ACTION_ICONS = {
    Action.FILESYSTEM_LIST: "📂",
    Action.FILESYSTEM_READ: "📖",
    Action.FILESYSTEM_CREATE: "📝",
    Action.FILESYSTEM_REPLACE: "✏️",
    Action.FILESYSTEM_DELETE: "🗑️",
    Action.NETWORK_REQUEST: "🌐",
    Action.PROCESS_EXECUTE: "⚙️",
    Action.SESSION_MUTATE: "💾",
}


class Decision(StrEnum):
    """Identify a policy or effective authorization outcome."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PolicyScope(StrEnum):
    """Identify the lifetime of one user-controlled policy change."""

    WORKSPACE = "workspace"
    SESSION = "session"


class ProcessBoundary(StrEnum):
    """Identify the execution boundary supplied by a process executor."""

    HOST = "host"
    SANDBOXED = "sandboxed"


class FileTarget(BaseModel):
    """Identify a canonical filesystem object and its planned state.

    Args:
        kind (Literal["file"]): Target discriminator.
        path (str): Canonical path that the executor must use.
        expected_exists (bool | None): Expected existence for guarded mutations.
        expected_digest (str | None): Expected SHA-256 content digest for replacement.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["file"] = "file"
    path: str
    expected_exists: bool | None = None
    expected_digest: str | None = None


class NetworkTarget(BaseModel):
    """Identify one normalized outbound HTTP request.

    Args:
        kind (Literal["network"]): Target discriminator.
        url (str): Normalized request URL.
        origin (str): Normalized scheme, host, and optional port.
        addresses (tuple[str, ...]): Resolved IP addresses pinned for the request connection.
        method (str): Uppercase HTTP method.
        sends_body (bool): Whether request content leaves the process.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["network"] = "network"
    url: str
    origin: str
    addresses: tuple[str, ...] = ()
    method: str = "GET"
    sends_body: bool = False


class ProcessTarget(BaseModel):
    """Identify an exact process invocation.

    Args:
        kind (Literal["process"]): Target discriminator.
        argv (tuple[str, ...]): Executable and arguments without shell parsing.
        cwd (str): Canonical process working directory.
        boundary (ProcessBoundary): Execution boundary enforced by the process executor.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["process"] = "process"
    argv: tuple[str, ...]
    cwd: str
    boundary: ProcessBoundary = ProcessBoundary.HOST


class SessionTarget(BaseModel):
    """Identify one mutation to in-memory agent state.

    Args:
        kind (Literal["session"]): Target discriminator.
        identifier (str): Stable operation-specific state identifier.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["session"] = "session"
    identifier: str


OperationTarget = Annotated[
    FileTarget | NetworkTarget | ProcessTarget | SessionTarget,
    Field(discriminator="kind"),
]


class Operation(BaseModel):
    """Describe one normalized authority-bearing effect before execution.

    Args:
        tool_id (str): Stable registered tool identity requesting authority.
        action (Action): Semantic effect to authorize.
        target (OperationTarget): Typed target identifying the affected resource.
        reason (str | None): Additional explanation displayed during approval.
    """

    model_config = ConfigDict(frozen=True)

    tool_id: str
    action: Action
    target: OperationTarget
    reason: str | None = None

    @model_validator(mode="after")
    def validate_target_type(self) -> Operation:
        """Require the target type associated with the selected action.

        Returns:
            Operation: This validated operation.

        Raises:
            ValueError: If the action has no target or uses the wrong target kind.
        """
        expected_type = (
            FileTarget
            if self.action.value.startswith("filesystem.")
            else NetworkTarget
            if self.action is Action.NETWORK_REQUEST
            else ProcessTarget
            if self.action is Action.PROCESS_EXECUTE
            else SessionTarget
        )
        if not isinstance(self.target, expected_type):
            raise ValueError(  # noqa: TRY004 - Pydantic wraps ValueError as ValidationError.
                f"Action '{self.action.value}' requires a {expected_type.__name__}."
            )
        return self

    @computed_field
    @property
    def resource(self) -> str:
        """Return a stable rule-matching representation of the target.

        Returns:
            str: Canonical path, URL, command, or session identifier.
        """
        if isinstance(self.target, FileTarget):
            return self.target.path
        if isinstance(self.target, NetworkTarget):
            return self.target.url
        if isinstance(self.target, ProcessTarget):
            return " ".join(self.target.argv)
        if isinstance(self.target, SessionTarget):
            return self.target.identifier
        raise AssertionError("Every operation target has a resource.")  # pragma: no cover


class OperationPlan(BaseModel):
    """Bind normalized call arguments to the complete planned operation set.

    Args:
        arguments (dict[str, object]): Canonical arguments passed to the executor.
        operations (tuple[Operation, ...]): Complete effects requiring authorization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    arguments: dict[str, object]
    operations: tuple[Operation, ...] = ()


OperationPlanner = Callable[[dict[str, Any]], OperationPlan]


class PermissionRule(BaseModel):
    """Match typed operations to one policy effect.

    Args:
        decision (Decision): Decision contributed by a matching rule.
        id (str): Stable identifier used for diagnostics and removal.
        description (str | None): Human-readable policy rationale.
        tool (str): Tool-identity glob, defaulting to every tool.
        action (Action | None): Required action, or every action when omitted.
        resource (str | None): Canonical resource glob, or every target when omitted.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Decision
    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str | None = None
    tool: str = "*"
    action: Action | None = None
    resource: str | None = None


class PolicyLimits(BaseModel):
    """Define user-configurable limits that ordinary policy rules cannot override.

    Args:
        readable_roots (tuple[str, ...]): Allowed filesystem read roots. ``workspace`` and
            ``loop-temp`` are special runtime roots.
        writable_roots (tuple[str, ...]): Allowed mutation roots. ``workspace`` and
            ``loop-temp`` are special runtime roots.
        network_origins (tuple[str, ...]): Allowed origin globs. ``"*"`` permits every origin;
            an empty tuple denies every origin.
        deny_private_networks (bool): Whether literal private and local IP targets are forbidden.
        allow_host_processes (bool): Whether policy may approve a host process.
    """

    model_config = ConfigDict(extra="forbid")

    readable_roots: tuple[str, ...] = ("workspace", "loop-temp")
    writable_roots: tuple[str, ...] = ("workspace", "loop-temp")
    network_origins: tuple[str, ...] = ("*",)
    deny_private_networks: bool = True
    allow_host_processes: bool = False


def _default_decisions() -> dict[Action, Decision]:
    return {
        Action.FILESYSTEM_LIST: Decision.ALLOW,
        Action.FILESYSTEM_READ: Decision.ALLOW,
        Action.FILESYSTEM_CREATE: Decision.ASK,
        Action.FILESYSTEM_REPLACE: Decision.ASK,
        Action.FILESYSTEM_DELETE: Decision.ASK,
        Action.NETWORK_REQUEST: Decision.ASK,
        Action.PROCESS_EXECUTE: Decision.ASK,
        Action.SESSION_MUTATE: Decision.ASK,
    }


class PermissionConfiguration(BaseModel):
    """Represent a complete persisted local operation policy.

    Args:
        version (Literal[1]): Configuration schema version.
        defaults (dict[Action, Decision]): Fallback decision for every known action.
        limits (PolicyLimits): User-configurable ceilings ordinary rules cannot override.
        rules (list[PermissionRule]): Composed explicit policy rules.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    defaults: dict[Action, Decision] = Field(default_factory=_default_decisions)
    limits: PolicyLimits = Field(default_factory=PolicyLimits)
    rules: list[PermissionRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_rule_ids(self) -> PermissionConfiguration:
        """Require every persisted rule to have a distinct identifier.

        Returns:
            PermissionConfiguration: This validated configuration.

        Raises:
            ValueError: If multiple rules use the same identifier.
        """
        identifiers = [rule.id for rule in self.rules]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Permission rule identifiers must be unique.")
        return self


class PolicyLimitOverrides(BaseModel):
    """Override selected workspace limits for the active process.

    Args:
        readable_roots (tuple[str, ...] | None): Replacement readable roots, or inheritance.
        writable_roots (tuple[str, ...] | None): Replacement writable roots, or inheritance.
        network_origins (tuple[str, ...] | None): Replacement network origins, or inheritance.
        deny_private_networks (bool | None): Private-network override, or inheritance.
        allow_host_processes (bool | None): Host-process override, or inheritance.
    """

    model_config = ConfigDict(extra="forbid")

    readable_roots: tuple[str, ...] | None = None
    writable_roots: tuple[str, ...] | None = None
    network_origins: tuple[str, ...] | None = None
    deny_private_networks: bool | None = None
    allow_host_processes: bool | None = None


class SessionPolicyOverrides(BaseModel):
    """Store policy changes that expire with the current process.

    Args:
        defaults (dict[Action, Decision]): Session fallback decisions by action.
        limits (PolicyLimitOverrides): Session replacements for selected workspace limits.
        rules (list[PermissionRule]): Session-only matching rules.
    """

    model_config = ConfigDict(extra="forbid")

    defaults: dict[Action, Decision] = Field(default_factory=dict)
    limits: PolicyLimitOverrides = Field(default_factory=PolicyLimitOverrides)
    rules: list[PermissionRule] = Field(default_factory=list)


class PolicyDecision(BaseModel):
    """Record the policy outcome before any user approval.

    Args:
        decision (Decision): Composed allow, ask, or deny outcome.
        reason (str): Human-readable explanation.
        sources (tuple[str, ...]): Determining boundary, rule, or default identifiers.
    """

    model_config = ConfigDict(frozen=True)

    decision: Decision
    reason: str
    sources: tuple[str, ...]


class AuthorizationResult(BaseModel):
    """Record one atomic authorization of a complete operation set.

    Args:
        operations (tuple[Operation, ...]): Operations evaluated together.
        policy (PolicyDecision): Outcome before interactive approval.
        decision (Decision): Effective allow or deny result.
        prompted (bool): Whether a user approval prompt was displayed.
        prompt (str | None): Exact displayed prompt, when applicable.
        reason (str): Explanation of the effective result.
        source (str): Effective decision source.
    """

    model_config = ConfigDict(frozen=True)

    operations: tuple[Operation, ...]
    policy: PolicyDecision
    decision: Decision
    prompted: bool = False
    prompt: str | None = None
    reason: str
    source: str


class PermissionRecorder(Protocol):
    """Record atomic authorization observations for one invocation scope."""

    def record_authorization(self, result: AuthorizationResult) -> None:
        """Record one complete authorization result.

        Args:
            result (AuthorizationResult): Atomic policy, approval, and effective outcome.
        """
