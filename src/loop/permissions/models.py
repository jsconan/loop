"""Define typed operations, policies, authorization decisions, and recording protocols."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol
from uuid import uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from ..errors import safe_error_detail
from ..sandbox_catalog import SANDBOX_OPTIONS, validate_sandbox_options
from ..utils import VirtualPath, YAMLError, load_yaml, sha256_digest
from .protection import validate_hard_denied_paths

type FileKind = Literal["file", "directory", "symlink"]
type Operations = tuple[Operation, ...]
PERMISSION_POLICY_VERSION = 10


class Action(StrEnum):
    """Identify one authority-bearing effect planned by a tool."""

    FILESYSTEM_LIST = "filesystem.list"
    FILESYSTEM_READ = "filesystem.read"
    FILESYSTEM_EXECUTE = "filesystem.execute"
    FILESYSTEM_CREATE = "filesystem.create"
    FILESYSTEM_REPLACE = "filesystem.replace"
    FILESYSTEM_DELETE = "filesystem.delete"
    NETWORK_REQUEST = "network.request"
    NETWORK_LOCAL = "network.local"
    NETWORK_BROAD = "network.broad"
    NETWORK_INGRESS = "network.ingress"
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
    Action.FILESYSTEM_EXECUTE: "⚙️",
    Action.FILESYSTEM_CREATE: "📝",
    Action.FILESYSTEM_REPLACE: "✏️",
    Action.FILESYSTEM_DELETE: "🗑️",
    Action.NETWORK_REQUEST: "🌐",
    Action.NETWORK_LOCAL: "🔌",
    Action.NETWORK_BROAD: "🌐",
    Action.NETWORK_INGRESS: "🔌",
    Action.PROCESS_EXECUTE: "⚙️",
    Action.SESSION_MUTATE: "💾",
}


class Decision(StrEnum):
    """Identify a policy or effective authorization outcome."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class SandboxPathAccess(StrEnum):
    """Identify a stored sandbox path grant or restriction."""

    READ = "read"
    WRITE = "write"
    WRITE_FILE = "write-file"
    READ_ONLY = "read-only"
    DENY = "deny"

    @classmethod
    def _missing_(cls, value: object) -> SandboxPathAccess | None:
        """Interpret the original persisted read grant without changing its authority."""
        return cls.READ if value == "allow" else None


def _validate_sandbox_paths(value: dict[str, SandboxPathAccess]) -> dict[str, SandboxPathAccess]:
    """Validate exact directory grants and hard path denials for commands."""
    validate_hard_denied_paths(tuple(value))
    if any(
        Path(path).anchor != "/" or (access is not SandboxPathAccess.DENY and path == "/")
        for path, access in value.items()
    ):
        raise ValueError("Sandbox paths require an exact path and a valid access level.")
    return value


type SandboxPaths = Annotated[dict[str, SandboxPathAccess], AfterValidator(_validate_sandbox_paths)]


class CommandReviewStatus(StrEnum):
    """Identify the approval lifetime required by a command inspector."""

    REVIEW = "review"
    FRESH = "fresh"


@dataclass(frozen=True)
class CommandFinding:
    """Describe one applicable command review policy.

    Args:
        policy_id (str): Stable identifier for the matching inspector.
        status (CommandReviewStatus): Required approval behavior.
        context (str): Short human-readable description of the observed command.
        reason (str): Action requiring permission, suitable for a combined prompt.
        requests_git_write (bool): Whether Git metadata write authority should be requested.
    """

    policy_id: str
    status: CommandReviewStatus
    context: str
    reason: str
    requests_git_write: bool = False


class PolicyScope(StrEnum):
    """Identify the lifetime of one user-controlled policy change."""

    WORKSPACE = "workspace"
    SESSION = "session"


class ApprovalChoice(StrEnum):
    """Identify one interactive permission response and its lifetime."""

    DENY = "deny"
    ONCE = "once"
    SESSION = "session"
    WORKSPACE = "workspace"
    USER = "user"


class ProcessBoundary(StrEnum):
    """Identify the execution boundary supplied by a process executor."""

    HOST = "host"
    SANDBOXED = "sandboxed"


class FileManifestEntry(BaseModel):
    """Capture one approved object in a recursive filesystem mutation.

    Args:
        relative_path (str): POSIX path relative to the recursive target, or ``."`` for its root.
        kind (FileKind): Object kind without following symlinks.
        device (int): Device identity where exposed by the platform.
        inode (int): Inode identity where exposed by the platform.
        size (int): Byte size reported by ``lstat``.
        mtime_ns (int): Nanosecond modification time reported by ``lstat``.
        ctime_ns (int | None): Change timestamp, or None for a legacy persisted manifest.
        link_count (int | None): Accounted inode names, or None for a legacy manifest.
        mode (int): Permission and type bits reported by ``lstat``.
        digest (str | None): SHA-256 file-content or symbolic-link-target digest when applicable.
    """

    model_config = ConfigDict(frozen=True)

    relative_path: str
    kind: FileKind
    device: int
    inode: int
    size: int
    mtime_ns: int
    mode: int
    ctime_ns: int | None = None
    link_count: int | None = None
    digest: str | None = None


class FileTarget(BaseModel):
    """Identify a canonical filesystem object and its planned state.

    Args:
        kind (Literal["file"]): Target discriminator.
        path (str): Canonical path that the executor must use.
        expected_exists (bool | None): Expected existence for guarded mutations.
        expected_digest (str | None): Expected SHA-256 content or link-target digest.
        expected_kind (FileKind | None): Expected object kind.
        expected_device (int | None): Expected device identity where exposed by the platform.
        expected_inode (int | None): Expected inode identity where exposed by the platform.
        expected_size (int | None): Expected byte size for a regular file.
        expected_mtime_ns (int | None): Expected nanosecond modification time.
        expected_ctime_ns (int | None): Change timestamp binding a read against inode reuse.
        expected_link_count (int | None): Link count from the authorized alias snapshot.
        expected_mode (int | None): Expected permission and type bits.
        expected_parent_device (int | None): Expected parent-directory device identity.
        expected_parent_inode (int | None): Expected parent-directory inode identity.
        recursive (bool): Whether the target represents a recursive tree mutation.
        manifest (tuple[FileManifestEntry, ...]): Approved recursive descendant states.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["file"] = "file"
    path: str
    expected_exists: bool | None = None
    expected_digest: str | None = None
    expected_kind: FileKind | None = None
    expected_device: int | None = None
    expected_inode: int | None = None
    expected_size: int | None = None
    expected_mtime_ns: int | None = None
    expected_ctime_ns: int | None = None
    expected_link_count: int | None = None
    expected_mode: int | None = None
    expected_parent_device: int | None = None
    expected_parent_inode: int | None = None
    recursive: bool = False
    manifest: tuple[FileManifestEntry, ...] = ()


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


class NetworkAccessTarget(BaseModel):
    """Identify a native network capability separately from an HTTP destination.

    Args:
        kind (Literal["network_access"]): Target discriminator.
        scope (Literal["local", "broad"]): Attempt-local peers or unrestricted networking.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["network_access"] = "network_access"
    scope: Literal["local", "broad"]


class NetworkIngressTarget(BaseModel):
    """Identify one host-loopback publication independently of outbound grants.

    Args:
        kind (Literal["network_ingress"]): Target discriminator.
        port (int): Exact IPv4 loopback TCP port; wildcard interfaces are unavailable.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["network_ingress"] = "network_ingress"
    port: int = Field(strict=True, ge=1, le=65535)


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
    FileTarget
    | NetworkTarget
    | NetworkAccessTarget
    | NetworkIngressTarget
    | ProcessTarget
    | SessionTarget,
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
            else NetworkAccessTarget
            if self.action in {Action.NETWORK_LOCAL, Action.NETWORK_BROAD}
            else NetworkIngressTarget
            if self.action is Action.NETWORK_INGRESS
            else ProcessTarget
            if self.action is Action.PROCESS_EXECUTE
            else SessionTarget
        )
        if not isinstance(self.target, expected_type):
            raise ValueError(  # noqa: TRY004 - Pydantic wraps ValueError as ValidationError.
                f"Action '{self.action.value}' requires a {expected_type.__name__}."
            )
        if isinstance(self.target, NetworkAccessTarget) and (
            self.action.value != "network." + self.target.scope
        ):
            raise ValueError("Network action and capability scope must agree.")
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
        if isinstance(self.target, NetworkIngressTarget):
            return f"tcp://127.0.0.1:{self.target.port}"
        if isinstance(self.target, NetworkAccessTarget):
            return "network:" + self.target.scope
        if isinstance(self.target, ProcessTarget):
            return " ".join(self.target.argv)
        if isinstance(self.target, SessionTarget):
            return self.target.identifier
        raise AssertionError("Every operation target has a resource.")  # pragma: no cover


class OperationPlan(BaseModel):
    """Bind normalized call arguments to the complete planned operation set.

    Args:
        arguments (dict[str, object]): Canonical arguments passed to the executor.
        operations (Operations): Effects requiring authorization at this phase.
        prerequisite_operations (Operations): Earlier inspection effects completed
            before this executable phase was produced.
        boundary_operations (Operations): Future effects whose non-overridable
            boundaries and model-observation preconditions must be checked before
            prerequisite inspection or further planning.
        continuation (Callable[[], OperationPlan] | None): Authorized inspection continuation
            producing the next phase, or ``None`` when this plan is executable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    arguments: dict[str, object]
    operations: Operations = ()
    prerequisite_operations: Operations = ()
    boundary_operations: Operations = ()
    continuation: Callable[[], OperationPlan] | None = Field(default=None, exclude=True)


class PresetSource(BaseModel):
    """Identify the immutable preset snapshot that installed a policy rule.

    Args:
        preset_id (str): Stable identifier of the originating preset.
        revision (str): Preset revision selected by the user.
        content_hash (str): SHA-256 digest of the selected preset content.
        rule_id (str): Identifier of the rule within the originating preset.
    """

    model_config = ConfigDict(frozen=True)

    preset_id: str
    revision: str
    content_hash: str
    rule_id: str


OperationPlanner = Callable[[dict[str, Any]], OperationPlan]


class PermissionRule(BaseModel):
    """Match typed operations to one policy effect.

    Args:
        decision (Decision): Decision contributed by a matching rule.
        id (str): Stable identifier used for diagnostics and removal.
        description (str | None): Human-readable policy rationale.
        tool (str): Tool-identity glob, defaulting to every tool.
        tool_exact (bool): Whether ``tool`` is matched literally instead of as a glob.
        action (Action | None): Required action, or every action when omitted.
        resource (str | None): Canonical resource glob, or every target when omitted.
        target (OperationTarget | None): Exact typed target, or glob matching through ``resource``
            when omitted.
        source (PresetSource | None): Preset snapshot that installed this rule, when any.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Decision
    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str | None = None
    tool: str = "*"
    tool_exact: bool = False
    action: Action | None = None
    resource: str | None = None
    target: OperationTarget | None = None
    source: PresetSource | None = None

    @model_validator(mode="after")
    def validate_matcher(self) -> PermissionRule:
        """Require glob and exact target matchers to be mutually exclusive.

        Returns:
            PermissionRule: This validated permission rule.

        Raises:
            ValueError: If both a resource glob and exact typed target are configured.
        """
        if self.resource is not None and self.target is not None:
            raise PydanticCustomError(
                "loop_rule_matcher", "Permission rules cannot combine resource and target matchers."
            )
        return self


class SandboxedCommandRule(BaseModel):
    """Remember one sandbox-only command approval for a reviewed authority context.

    Args:
        id (str): Stable identifier for managing the approval.
        signature (str): Digest of the sandbox authority and execution context.
        source (str): Exact shell source, restricted leading command phrase, or label for
            a session tool-access rule.
        similar (bool): Whether source may match a longer simple command.
        scope (ApprovalChoice): Lifetime and breadth of this sandbox-only approval.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    signature: str
    source: str
    similar: bool = False
    scope: ApprovalChoice = ApprovalChoice.SESSION


def _validate_sandboxed_rule_ids(rules: list[SandboxedCommandRule]) -> None:
    """Reject repeated identifiers in a persisted sandbox approval collection."""
    if len({rule.id for rule in rules}) != len(rules):
        raise PydanticCustomError(
            "loop_sandbox_rule_ids", "Sandbox command rule identifiers must be unique."
        )


class HostCommandRule(BaseModel):
    """Remember an exact host approval with virtual working-directory context.

    Args:
        id (str): Identifier used to list and revoke the approval.
        authorization_version (int): Approval semantics; version one stays inactive.
        signature (str): Digest of virtual workspace coordinates and safe command shape only.
        cwd (str): Virtual workspace working directory; never a host path.
        source (str): Path-free command label for listing and revocation.
        scope (ApprovalChoice): Lifetime of a trusted grant; retry rules remain session-only.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    authorization_version: int = Field(default=1, ge=1, le=2, strict=True)
    signature: str
    cwd: str
    source: str
    scope: ApprovalChoice = ApprovalChoice.SESSION

    @model_validator(mode="after")
    def validate_virtual_context(self) -> HostCommandRule:
        """Reject real paths and malformed portable rule identities.

        Returns:
            HostCommandRule: Rule containing virtual workspace context only.

        Raises:
            ValueError: If a path or malformed digest is stored in the rule.
        """
        relative = self.cwd.removeprefix(f"{VirtualPath.WORKSPACE}/")
        if (
            (
                self.cwd != VirtualPath.WORKSPACE
                and (
                    not self.cwd.startswith(f"{VirtualPath.WORKSPACE}/")
                    or not relative
                    or any(part in {"", ".", ".."} for part in relative.split("/"))
                )
            )
            or "/" in self.source
            or len(self.signature) != 64
            or any(character not in "0123456789abcdef" for character in self.signature)
        ):
            raise PydanticCustomError(
                "loop_host_rule",
                "Host rule must contain only virtual context and a SHA-256 digest.",
            )
        return self


class PresetMetadata(BaseModel):
    """Describe a versioned, user-selectable permission preset.

    Args:
        id (str): Stable catalog identifier.
        revision (str): Immutable revision selected during installation.
        title (str): Short user-facing name.
        description (str): Explanation of the rule set's intended effect.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    revision: str
    title: str
    description: str


class PermissionLoadFailure(BaseModel):
    """Describe a permission artifact that could not be activated safely.

    Args:
        source (Literal["configuration", "preset"]): Type of artifact that failed to load.
        path (str): Filesystem or package path of the failed artifact.
        message (str): Safe, actionable summary of the loading failure.
    """

    model_config = ConfigDict(frozen=True)

    source: Literal["configuration", "preset"]
    path: str
    message: str


class PermissionLoadPolicy(StrEnum):
    """Control recovery from invalid permission configuration and preset artifacts."""

    INTERACTIVE = "interactive"
    AUTO = "auto"
    ERROR = "error"


class PermissionLoadResult(StrEnum):
    """Describe the configuration selected by a completed load operation."""

    LOADED = "loaded"
    DEFAULTED = "defaulted"
    RETAINED = "retained"
    BLOCKED = "blocked"


class PermissionArtifactError(Exception):
    """Report permission artifacts that could not be activated safely."""


class PermissionConfigurationError(PermissionArtifactError):
    """Report an invalid workspace permission configuration.

    Args:
        path (Path | str): Path of the configuration that could not be loaded.
        message (str): Safe description of the loading failure.
    """

    path: str
    message: str

    def __init__(self, path: Path | str, message: str) -> None:
        self.path = str(path)
        self.message = message
        super().__init__(f"Could not load permission policy at {self.path}: {message}")


class PermissionNamespaceError(PermissionConfigurationError):
    """Report policy provenance or publication state requiring trusted recovery."""


class PermissionPresetError(PermissionArtifactError):
    """Report one or more invalid built-in permission presets.

    Args:
        failures (tuple[PermissionLoadFailure, ...]): Invalid preset diagnostics.
    """

    failures: tuple[PermissionLoadFailure, ...]

    def __init__(self, failures: tuple[PermissionLoadFailure, ...]) -> None:
        if not failures:
            raise ValueError("PermissionPresetError requires at least one failure.")
        self.failures = failures
        super().__init__(
            f"Could not load {len(failures)} built-in permission preset(s): {failures[0].message}"
        )


class PresetRule(BaseModel):
    """Describe one rule before a preset installs it into a policy scope.

    Args:
        id (str): Identifier unique within the containing rule set.
        decision (Decision): Policy effect contributed by this rule.
        description (str | None): Human-readable rationale for the rule.
        tool (str): Tool-identity glob, defaulting to every tool.
        action (Action | None): Required action, or every action when omitted.
        resource (str | None): Canonical resource glob, or every target when omitted.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    decision: Decision
    description: str | None = None
    tool: str = "*"
    action: Action | None = None
    resource: str | None = None


class PermissionPreset(BaseModel):
    """Contain a versioned complete policy-default and rule artifact.

    A preset replaces defaults and rules in one explicit policy scope. It intentionally excludes
    enforcement limits, which remain the user-controlled boundaries for every preset.

    Args:
        version (int): Preset schema version.
        kind (Literal["app.permission-preset"]): Artifact type discriminator.
        metadata (PresetMetadata): Identity and explanation shown before replacement.
        confirmation_warning (str | None): Warning shown before the required confirmation, or
            ``None`` when the standard replacement prompt is sufficient.
        defaults (dict[Action, Decision]): Complete fallback decision for every known action.
        rules (tuple[PresetRule, ...]): Complete rule replacement payload.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1, le=1, strict=True)
    kind: Literal["app.permission-preset"] = "app.permission-preset"
    metadata: PresetMetadata
    confirmation_warning: str | None = None
    defaults: dict[Action, Decision]
    rules: tuple[PresetRule, ...]

    @field_validator("defaults")
    @classmethod
    def add_network_scopes(cls, value: dict[Action, Decision]) -> dict[Action, Decision]:
        """Normalize the complete historical action set without repairing missing restrictions.

        Args:
            value (dict[Action, Decision]): Parsed defaults from the preset artifact.

        Returns:
            dict[Action, Decision]: Complete defaults with separate local and broad decisions.
        """
        if set(value) == set(Action) - {
            Action.NETWORK_LOCAL,
            Action.NETWORK_BROAD,
            Action.NETWORK_INGRESS,
        }:
            return {
                **value,
                Action.NETWORK_LOCAL: Decision.ALLOW,
                Action.NETWORK_BROAD: Decision.ASK,
                Action.NETWORK_INGRESS: Decision.ASK,
            }
        return value

    @model_validator(mode="after")
    def validate_contents(self) -> PermissionPreset:
        """Require a complete default map and unique rule identifiers.

        Returns:
            PermissionPreset: This validated preset.

        Raises:
            ValueError: If defaults omit or add an action, or multiple rules use the same
                identifier.
        """
        if set(self.defaults) != set(Action):
            raise PydanticCustomError(
                "loop_preset_defaults",
                "Permission preset defaults must contain every known action exactly once.",
            )
        identifiers = [rule.id for rule in self.rules]
        if len(identifiers) != len(set(identifiers)):
            raise PydanticCustomError(
                "loop_preset_rule_ids", "Permission preset rule identifiers must be unique."
            )
        return self

    @property
    def content_hash(self) -> str:
        """Return a stable digest for this preset snapshot.

        Returns:
            str: ``sha256:``-prefixed digest of the schema-normalized artifact.
        """
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{sha256_digest(encoded)}"

    @classmethod
    def builtin_presets(cls) -> tuple[PermissionPreset, ...]:
        """Return all validated presets shipped with the application.

        Returns:
            tuple[PermissionPreset, ...]: Catalog entries sorted by stable identifier.

        Raises:
            ValueError: If packaged artifacts duplicate an identifier.
        """
        presets, failures = cls.load_builtin_presets()
        if failures:
            raise ValueError(f"Could not load built-in permission preset: {failures[0].message}")
        return tuple(sorted(presets, key=lambda preset: preset.metadata.id))

    @classmethod
    def load_builtin_presets(
        cls,
    ) -> tuple[tuple[PermissionPreset, ...], tuple[PermissionLoadFailure, ...]]:
        """Load every usable shipped preset and retain diagnostics for unusable artifacts.

        Returns:
            tuple[tuple[PermissionPreset, ...], tuple[PermissionLoadFailure, ...]]: Valid presets
                sorted by stable identifier and failures for artifacts excluded from the catalog.
        """
        presets = []
        failures = []
        directory = files(__package__).joinpath("presets")
        try:
            entries = sorted(
                (entry for entry in directory.iterdir() if entry.name.endswith((".yaml", ".yml"))),
                key=lambda item: item.name,
            )
        except OSError as error:
            return (), (
                PermissionLoadFailure(
                    source="preset",
                    path=str(directory),
                    message=(
                        f"Cannot read preset directory: {error.strerror or type(error).__name__}."
                    ),
                ),
            )
        if not entries:
            failures.append(
                PermissionLoadFailure(
                    source="preset", path=str(directory), message="Preset directory is empty."
                )
            )
        for entry in entries:
            try:
                presets.append(
                    cls.model_validate(load_yaml(entry.read_text("utf-8")), extra="forbid")
                )
            except (OSError, UnicodeError, ValueError, YAMLError) as exc:
                failures.append(
                    PermissionLoadFailure(
                        source="preset",
                        path=str(entry),
                        message=safe_error_detail(
                            exc,
                            cls,
                            trusted_types=(YAMLError,),
                        ),
                    )
                )
        unique_presets = []
        identifiers = set()
        for preset in presets:
            if preset.metadata.id in identifiers:
                failures.append(
                    PermissionLoadFailure(
                        source="preset",
                        path=str(directory),
                        message="Built-in permission preset identifiers must be unique.",
                    )
                )
                continue
            identifiers.add(preset.metadata.id)
            unique_presets.append(preset)
        return (
            tuple(sorted(unique_presets, key=lambda preset: preset.metadata.id)),
            tuple(failures),
        )


class PresetReplacementPreview(BaseModel):
    """Describe one explicit scoped preset replacement before activation.

    Args:
        preset (PermissionPreset): Selected immutable preset snapshot.
        scope (PolicyScope): Defaults and rule layer that would be replaced.
        removed_defaults (dict[Action, Decision]): Existing defaults in the selected layer.
        installed_defaults (dict[Action, Decision]): Complete defaults installed in that layer.
        removed_rules (tuple[PermissionRule, ...]): Existing rules removed from that layer.
        installed_rules (tuple[PermissionRule, ...]): Rules installed into that layer.
        scope_revision (str): Revision required to reject stale replacement previews.
    """

    model_config = ConfigDict(frozen=True)

    preset: PermissionPreset
    scope: PolicyScope
    removed_defaults: dict[Action, Decision]
    installed_defaults: dict[Action, Decision]
    removed_rules: tuple[PermissionRule, ...]
    installed_rules: tuple[PermissionRule, ...]
    scope_revision: str


class PolicyLimits(BaseModel):
    """Define user-configurable limits that ordinary policy rules cannot override.

    Args:
        hard_denied_paths (tuple[str, ...]): Exact absolute file/directory designations that
            ordinary approvals cannot override. Empty by default.
        readable_roots (tuple[str, ...]): Allowed filesystem read roots. ``workspace`` and
            ``app-temp`` are special runtime roots.
        writable_roots (tuple[str, ...]): Allowed mutation roots. ``workspace`` and
            ``app-temp`` are special runtime roots.
        network_origins (tuple[str, ...]): Allowed origin globs. ``"*"`` permits every origin;
            an empty tuple denies every origin.
        deny_private_networks (bool): Whether literal private and local IP targets are forbidden.
        deny_local_networks (bool): Whether sandbox-attempt loopback communication is forbidden.
    """

    model_config = ConfigDict(extra="forbid")

    hard_denied_paths: tuple[str, ...] = ()
    readable_roots: tuple[str, ...] = ("workspace", "app-temp")
    writable_roots: tuple[str, ...] = ("workspace", "app-temp")
    network_origins: tuple[str, ...] = ("*",)
    deny_private_networks: bool = True
    deny_local_networks: bool = False

    _validate_hard_paths = field_validator("hard_denied_paths")(validate_hard_denied_paths)


class LimitClassification(StrEnum):
    """Classify a user-controlled collection limit for scoped sandbox requests."""

    HARD = "hard"
    APPROVAL = "approval"


class PolicyLimitClassifications(BaseModel):
    """Store the provenance-sensitive behavior of collection limits.

    Args:
        readable_roots (LimitClassification): Classification of sandbox read roots.
        writable_roots (LimitClassification): Classification of sandbox write roots.
        network_origins (LimitClassification): Classification of network origins.
        sources (dict[str, str]): Provenance of each persisted classification.
    """

    model_config = ConfigDict(extra="forbid")

    readable_roots: LimitClassification = LimitClassification.APPROVAL
    writable_roots: LimitClassification = LimitClassification.APPROVAL
    network_origins: LimitClassification = LimitClassification.APPROVAL
    sources: dict[str, str] = Field(
        default_factory=lambda: {
            "readable_roots": "builtin",
            "writable_roots": "builtin",
            "network_origins": "builtin",
        }
    )

    @model_validator(mode="after")
    def validate_sources(self) -> PolicyLimitClassifications:
        """Require a complete, known provenance for each classified collection.

        Returns:
            PolicyLimitClassifications: Validated classifications and provenance.

        Raises:
            ValueError: If provenance is missing or unknown.
        """
        if self.sources.keys() != {"readable_roots", "writable_roots", "network_origins"} or any(
            source not in {"builtin", "legacy", "user", "session"}
            for source in self.sources.values()
        ):
            raise PydanticCustomError(
                "loop_limit_provenance", "Permission limit classification provenance is invalid."
            )
        return self


def _default_decisions() -> dict[Action, Decision]:
    return {
        Action.FILESYSTEM_LIST: Decision.ALLOW,
        Action.FILESYSTEM_READ: Decision.ALLOW,
        Action.FILESYSTEM_EXECUTE: Decision.ALLOW,
        Action.FILESYSTEM_CREATE: Decision.ASK,
        Action.FILESYSTEM_REPLACE: Decision.ASK,
        Action.FILESYSTEM_DELETE: Decision.ASK,
        Action.NETWORK_REQUEST: Decision.ASK,
        Action.NETWORK_LOCAL: Decision.ALLOW,
        Action.NETWORK_BROAD: Decision.ASK,
        Action.NETWORK_INGRESS: Decision.ASK,
        Action.PROCESS_EXECUTE: Decision.ASK,
        Action.SESSION_MUTATE: Decision.ASK,
    }


def _review_legacy_network_grants(value: Any) -> Any:
    """Require renewed consent for saved unconstrained egress allows from older schemas."""
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("version", 5), int)
        or isinstance(value.get("version", 5), bool)
        or value.get("version", 5) >= 5
    ):
        return value
    rules = value.get("rules", [])
    if not isinstance(rules, list):
        return value
    return {
        **value,
        "rules": [
            {**rule, "decision": Decision.ASK}
            if isinstance(rule, dict)
            and rule.get("decision") == Decision.ALLOW
            and rule.get("action") == Action.NETWORK_REQUEST
            and rule.get("resource") in (None, "*")
            and rule.get("target") is None
            else rule
            for rule in rules
        ],
    }


class PermissionConfiguration(BaseModel):
    """Represent a complete persisted local operation policy.

    Args:
        version (int): Configuration schema version.
        defaults (dict[Action, Decision]): Fallback decision for every known action.
        limits (PolicyLimits): User-configurable ceilings ordinary rules cannot override.
        limit_classifications (PolicyLimitClassifications): Provenance-sensitive sandbox grant
            behavior, recorded separately from the effective limit values.
        namespace_generation (str | None): Durable policy namespace marker when bound to the
            application registry; markerless valid policies are migrated conservatively.
        rules (list[PermissionRule]): Composed explicit policy rules.
        sandboxed_command_rules (list[SandboxedCommandRule]): Remembered approvals for this
            workspace only.
        sandbox_paths (SandboxPaths): Exact sandbox directory grants and path denials.
        workspace_command_writes (bool): Include the checkout in ordinary command write requests
            unless the invocation supplies a scope.
        backend_options (tuple[str, ...]): Optional backend authorities enabled by the owner.
        command_backend_options (dict[str, tuple[str, ...]]): Exact workspace command signatures
            whose optional authority selection the owner has asked to reuse.
        backend_option_revisions (dict[str, str]): Workspace revocation markers for exact
            command approvals that include a backend option.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = Field(
        default=PERMISSION_POLICY_VERSION, ge=1, le=PERMISSION_POLICY_VERSION, strict=True
    )
    defaults: dict[Action, Decision] = Field(default_factory=_default_decisions)
    limits: PolicyLimits = Field(default_factory=PolicyLimits)
    limit_classifications: PolicyLimitClassifications = Field(
        default_factory=PolicyLimitClassifications
    )
    namespace_generation: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    rules: list[PermissionRule] = Field(default_factory=list)
    sandboxed_command_rules: list[SandboxedCommandRule] = Field(default_factory=list)
    sandbox_paths: SandboxPaths = Field(default_factory=dict)
    workspace_command_writes: bool = Field(default=False, strict=True)
    backend_options: tuple[str, ...] = ()
    command_backend_options: dict[
        Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")],
        Annotated[tuple[str, ...], AfterValidator(validate_sandbox_options)],
    ] = Field(default_factory=dict)
    backend_option_revisions: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]] = Field(
        default_factory=dict
    )

    _validate_options = field_validator("backend_options")(validate_sandbox_options)

    @field_validator("backend_option_revisions")
    @classmethod
    def validate_backend_option_revisions(cls, value: dict[str, str]) -> dict[str, str]:
        """Validate the capability keys in persisted revocation markers.

        Args:
            value (dict[str, str]): Capability keys and their workspace revocation markers.

        Returns:
            dict[str, str]: Validated markers.

        Raises:
            ValueError: A capability key is unknown.
        """
        validate_sandbox_options(tuple(value))
        return value

    @model_validator(mode="after")
    def _initialize_backend_option_revisions(self) -> PermissionConfiguration:
        """Give new durable policies fresh markers and require a complete saved set."""
        if self.namespace_generation is not None:
            if "backend_option_revisions" not in self.model_fields_set:
                self.backend_option_revisions = {name: uuid4().hex for name in SANDBOX_OPTIONS}
            elif self.backend_option_revisions.keys() != SANDBOX_OPTIONS.keys():
                raise ValueError("Workspace backend option revocation markers are incomplete.")
        return self

    @model_validator(mode="before")
    @classmethod
    def preserve_legacy_defaults(cls, value: Any) -> Any:
        """Retain ambiguous saved process asks and add the newly explicit exec capability.

        Args:
            value (Any): Persisted policy mapping or already validated configuration.

        Returns:
            Any: Policy input with conservative legacy defaults.
        """
        value = _review_legacy_network_grants(value)
        if (
            isinstance(value, dict)
            and isinstance(value.get("version", 5), int)
            and not isinstance(value.get("version", 5), bool)
            and value.get("version", 5) < 5
        ):
            value = dict(value)
            defaults = value.get("defaults", _default_decisions())
            if isinstance(defaults, dict):
                defaults = dict(defaults)
                defaults.setdefault(Action.NETWORK_LOCAL, Decision.ALLOW)
                defaults.setdefault(Action.NETWORK_BROAD, Decision.ASK)
                defaults.setdefault(Action.NETWORK_INGRESS, Decision.ASK)
                if defaults.get(Action.NETWORK_REQUEST) == Decision.ALLOW:
                    defaults[Action.NETWORK_REQUEST] = Decision.ASK
                value["defaults"] = defaults
        if (
            isinstance(value, dict)
            and isinstance(value.get("version", 4), int)
            and not isinstance(value.get("version", 4), bool)
            and value.get("version", 4) < 4
        ):
            value = dict(value)
            defaults = value.get("defaults", _default_decisions())
            if isinstance(defaults, dict):
                defaults = dict(defaults)
                defaults.setdefault(Action.FILESYSTEM_EXECUTE, Decision.ALLOW)
                value["defaults"] = defaults
        return value

    @model_validator(mode="after")
    def validate_rule_ids(self) -> PermissionConfiguration:
        """Require every persisted rule to have a distinct identifier.

        Returns:
            PermissionConfiguration: This validated configuration.

        Raises:
            ValueError: If multiple rules share an identifier or a sandbox approval has a
                non-workspace scope.
        """
        identifiers = [rule.id for rule in self.rules]
        if len(identifiers) != len(set(identifiers)):
            raise PydanticCustomError(
                "loop_rule_ids", "Permission rule identifiers must be unique."
            )
        _validate_sandboxed_rule_ids(self.sandboxed_command_rules)
        if any(rule.scope is not ApprovalChoice.WORKSPACE for rule in self.sandboxed_command_rules):
            raise PydanticCustomError(
                "loop_sandbox_rule_scope", "Workspace sandbox approvals require workspace scope."
            )
        return self


class UserPermissionConfiguration(BaseModel):
    """Store remembered exact approvals shared by every workspace.

    User policy contains reviewed approvals and explicitly managed sandbox paths.

    Args:
        version (int): Persisted schema version.
        rules (list[PermissionRule]): Exact user-approved operation rules.
        sandboxed_command_rules (list[SandboxedCommandRule]): User approvals and legacy
            workspace-scoped approvals stored by earlier versions.
        trusted_host_command_rules (list[HostCommandRule]): Human-approved durable host grants.
        namespace_generation (str | None): Durable user-data-scoped policy namespace marker.
        sandbox_paths (SandboxPaths): User-wide native path grants and denials.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = Field(
        default=PERMISSION_POLICY_VERSION, ge=1, le=PERMISSION_POLICY_VERSION, strict=True
    )
    rules: list[PermissionRule] = Field(default_factory=list)
    sandboxed_command_rules: list[SandboxedCommandRule] = Field(default_factory=list)
    trusted_host_command_rules: list[HostCommandRule] = Field(default_factory=list)
    namespace_generation: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    sandbox_paths: SandboxPaths = Field(default_factory=dict)

    _review_network_grants = model_validator(mode="before")(_review_legacy_network_grants)

    @model_validator(mode="after")
    def validate_rule_ids(self) -> UserPermissionConfiguration:
        """Require every persisted rule to have a distinct identifier.

        Returns:
            UserPermissionConfiguration: This validated configuration.

        Raises:
            ValueError: If multiple rules use the same identifier.
        """
        identifiers = [rule.id for rule in self.rules]
        host_identifiers = [rule.id for rule in self.trusted_host_command_rules]
        if len(identifiers) != len(set(identifiers)):
            raise PydanticCustomError(
                "loop_rule_ids", "Permission rule identifiers must be unique."
            )
        _validate_sandboxed_rule_ids(self.sandboxed_command_rules)
        if len(host_identifiers) != len(set(host_identifiers)) or any(
            rule.scope not in {ApprovalChoice.WORKSPACE, ApprovalChoice.USER}
            for rule in self.trusted_host_command_rules
        ):
            raise PydanticCustomError(
                "loop_host_rule_ids", "Durable host grants require distinct IDs and durable scopes."
            )
        return self


class PolicyLimitOverrides(BaseModel):
    """Override selected workspace limits for the active process.

    Args:
        readable_roots (tuple[str, ...] | None): Replacement readable roots, or inheritance.
        writable_roots (tuple[str, ...] | None): Replacement writable roots, or inheritance.
        network_origins (tuple[str, ...] | None): Replacement network origins, or inheritance.
        deny_private_networks (bool | None): Private-network override, or inheritance.
        deny_local_networks (bool | None): Sandbox-local override, or inheritance.
    """

    model_config = ConfigDict(extra="forbid")

    readable_roots: tuple[str, ...] | None = None
    writable_roots: tuple[str, ...] | None = None
    network_origins: tuple[str, ...] | None = None
    deny_private_networks: bool | None = None
    deny_local_networks: bool | None = None


class SessionPolicyOverrides(BaseModel):
    """Store policy changes that expire with the current process.

    Args:
        defaults (dict[Action, Decision]): Session fallback decisions by action.
        limits (PolicyLimitOverrides): Session replacements for selected workspace limits.
        limit_classifications (dict[str, LimitClassification]): Trusted session changes to
            collection classifications; a new session value is hard unless reclassified.
        rules (list[PermissionRule]): Session-only matching rules.
        sandboxed_command_rules (list[SandboxedCommandRule]): Session-only sandbox approvals.
        host_command_rules (list[HostCommandRule]): Session-only exact host retry approvals.
        trusted_host_command_rules (list[HostCommandRule]): Human-installed session host grants.
        sandbox_paths (SandboxPaths): Process-local native path grants and denials.
    """

    model_config = ConfigDict(extra="forbid")

    defaults: dict[Action, Decision] = Field(default_factory=dict)
    limits: PolicyLimitOverrides = Field(default_factory=PolicyLimitOverrides)
    limit_classifications: dict[str, LimitClassification] = Field(default_factory=dict)
    rules: list[PermissionRule] = Field(default_factory=list)
    sandboxed_command_rules: list[SandboxedCommandRule] = Field(default_factory=list)
    host_command_rules: list[HostCommandRule] = Field(default_factory=list)
    trusted_host_command_rules: list[HostCommandRule] = Field(default_factory=list)
    sandbox_paths: SandboxPaths = Field(default_factory=dict)


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
        operations (Operations): Operations evaluated together.
        policy (PolicyDecision): Outcome before interactive approval.
        decision (Decision): Effective allow or deny result.
        prompted (bool): Whether a user approval prompt was displayed.
        prompt (str | None): Exact displayed prompt, when applicable.
        reason (str): Explanation of the effective result.
        source (str): Effective decision source.
        approval_choice (ApprovalChoice | None): Interactive response, when prompted.
        installed_rule_ids (tuple[str, ...]): Remembered rule identifiers installed by approval.
    """

    model_config = ConfigDict(frozen=True)

    operations: Operations
    policy: PolicyDecision
    decision: Decision
    prompted: bool = False
    prompt: str | None = None
    reason: str
    source: str
    approval_choice: ApprovalChoice | None = None
    installed_rule_ids: tuple[str, ...] = ()


class PermissionRecorder(Protocol):
    """Record atomic authorization observations for one invocation scope."""

    def record_authorization(self, result: AuthorizationResult) -> None:
        """Record one complete authorization result.

        Args:
            result (AuthorizationResult): Atomic policy, approval, and effective outcome.
        """
