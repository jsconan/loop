"""Define permission capabilities, requests, rules, and decisions."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Capability(StrEnum):
    """Identify the kind of authority required by a tool operation."""

    PURE = "pure"
    FILESYSTEM_READ = "filesystem.read"
    FILESYSTEM_WRITE = "filesystem.write"
    PROCESS_EXEC = "process.exec"
    NETWORK_READ = "network.read"
    NETWORK_WRITE = "network.write"
    SESSION_WRITE = "session.write"


class Decision(StrEnum):
    """Identify the outcome of a permission evaluation."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PermissionMode(StrEnum):
    """Select the fallback policy applied when no explicit rule matches."""

    CONFIRM_ALL = "confirm_all"
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    LOCKED_DOWN = "locked_down"
    UNRESTRICTED = "unrestricted"


class PermissionRequest(BaseModel):
    """Describe authority requested for one model-originated operation.

    Args:
        tool_name (str): Public tool requesting authority.
        capability (Capability): Kind of operation to authorize.
        resource (str | None): Normalized resource affected by the operation.
        reason (str | None): Additional explanation shown during approval.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str
    capability: Capability
    resource: str | None = None
    reason: str | None = None


class PermissionRule(BaseModel):
    """Match permission requests to a configured decision.

    Args:
        decision (Decision): Decision returned by a matching rule.
        tool (str): Tool-name glob, defaulting to every tool.
        capability (Capability | None): Required capability, or every capability when omitted.
        resource (str | None): Resource glob, or every resource when omitted.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Decision
    tool: str = "*"
    capability: Capability | None = None
    resource: str | None = None


class PermissionConfiguration(BaseModel):
    """Represent persisted local permission configuration.

    Args:
        mode (PermissionMode): Fallback approval mode.
        rules (list[PermissionRule]): Rules evaluated in declared order within each decision tier.
    """

    model_config = ConfigDict(extra="forbid")

    mode: PermissionMode = PermissionMode.CONFIRM_ALL
    rules: list[PermissionRule] = Field(default_factory=list)


class PermissionResult(BaseModel):
    """Record the decision produced for a permission request.

    Args:
        decision (Decision): Effective authorization decision.
        reason (str): Human-readable explanation of the decision.
        source (str): Policy source responsible for the decision.
    """

    model_config = ConfigDict(frozen=True)

    decision: Decision
    reason: str
    source: str
