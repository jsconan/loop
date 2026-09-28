"""Load, evaluate, persist, approve, and record local operation policies."""

# Pylint cannot infer mutable attributes declared by Pydantic models.
# pylint: disable=no-member

from __future__ import annotations

import ipaddress
import json
import logging
import re
import shlex
import sqlite3
import stat
import tempfile
from atexit import register
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from weakref import WeakSet

import yaml

from .. import constants
from ..errors import Problem, log_problem
from ..telemetry import telemetry_audit, telemetry_error, telemetry_trace_event
from ..utils import (
    ShutdownRequested,
    VirtualPath,
    canonical_path,
    local_now,
    sha256_digest,
)
from .audit import SQLitePermissionAudit
from .inspection import CommandAnalysis, CommandInspection
from .models import (
    Action,
    ApprovalChoice,
    AuthorizationResult,
    CommandFinding,
    CommandReviewStatus,
    Decision,
    FileTarget,
    HostCommandRule,
    NetworkTarget,
    Operation,
    Operations,
    OperationTarget,
    PermissionConfiguration,
    PermissionConfigurationError,
    PermissionLoadPolicy,
    PermissionLoadResult,
    PermissionPreset,
    PermissionPresetError,
    PermissionRecorder,
    PermissionRule,
    PolicyDecision,
    PolicyLimitOverrides,
    PolicyLimits,
    PolicyScope,
    PresetReplacementPreview,
    PresetSource,
    ProcessBoundary,
    ProcessTarget,
    SandboxedCommandRule,
    SessionPolicyOverrides,
    SessionTarget,
    UserPermissionConfiguration,
)
from .protection import ProtectedWorkspacePaths, protected_workspace_paths

if TYPE_CHECKING:
    from ..execution.sandbox import SandboxRequest
    from ..interaction import Interaction

_READ_ACTIONS = {Action.FILESYSTEM_LIST, Action.FILESYSTEM_READ}
_WRITE_ACTIONS = {
    Action.FILESYSTEM_CREATE,
    Action.FILESYSTEM_REPLACE,
    Action.FILESYSTEM_DELETE,
}
_LIMIT_NAMES = (
    "readable_roots",
    "writable_roots",
    "network_origins",
    "deny_private_networks",
    "allow_host_processes",
)
_LOGGER = logging.getLogger(__name__)
_SIMPLE_COMMAND = re.compile(r"[A-Za-z0-9_./:@+=,-]+(?: [A-Za-z0-9_./:@+=,-]+)+")
_LIVE_MANAGERS: WeakSet[PermissionManager] = WeakSet()


def _close_live_managers() -> None:
    """Release temporary directories held by managers alive at process shutdown."""
    for manager in _LIVE_MANAGERS:
        manager.close()


register(_close_live_managers)


class PermissionManager:
    """Authorize complete operation plans using limits, policy rules, and approval.

    Args:
        workspace_root (Path | str | None): Workspace used to resolve policy root tokens.
        configuration_path (Path | str | None): YAML policy path. Defaults to
            <workspace_root>/.loop/permissions.yaml when a workspace is supplied.
        user_configuration_path (Path | str | None): User-wide remembered-approval YAML path.
            User scope is unavailable when omitted.
        audit_path (Path | str | None): Central SQLite audit database path.
        workspace_id (str | None): Stable identity attached to centralized audit records.
        interaction (Interaction | None): User interaction used for approval prompts.
        recorder (PermissionRecorder | None): Default sink for authorization observations.
        configuration (PermissionConfiguration | None): Explicit policy instead of the local file.
        presets (Iterable[PermissionPreset] | None): Additional selectable presets alongside
            the built-in catalog. Duplicate identifiers are rejected.
        load_policy (PermissionLoadPolicy | None): Artifact failure behavior. Defaults to
            interactive recovery when an interaction is available and strict errors otherwise.
        command_inspection (CommandInspection | None): Trusted command inspection. Defaults to
            the built-in matchers.

    Raises:
        PermissionConfigurationError: If a policy is invalid in strict mode.
        PermissionPresetError: If built-in presets are invalid in strict mode.
        ShutdownRequested: If the user exits interactive configuration recovery.
        ValueError: If interactive recovery is selected without an interaction or preset
            identifiers are duplicated.
    """

    _workspace_root: Path | None
    _configuration_path: Path | None
    _user_configuration_path: Path | None
    _interaction: Interaction | None
    _recorder: PermissionRecorder | None
    _configuration: PermissionConfiguration
    _user_configuration: UserPermissionConfiguration
    _load_policy: PermissionLoadPolicy
    _session_overrides: SessionPolicyOverrides
    _temporary_directory: tempfile.TemporaryDirectory[str]
    _temporary_path: Path
    _presets: dict[str, PermissionPreset]
    _audit_store: SQLitePermissionAudit | None
    _workspace_id: str | None
    _command_inspection: CommandInspection
    _host_runtime_bindings: dict[str, str]

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        *,
        configuration_path: Path | str | None = None,
        user_configuration_path: Path | str | None = None,
        audit_path: Path | str | None = None,
        workspace_id: str | None = None,
        interaction: Interaction | None = None,
        recorder: PermissionRecorder | None = None,
        configuration: PermissionConfiguration | None = None,
        presets: Iterable[PermissionPreset] | None = None,
        load_policy: PermissionLoadPolicy | None = None,
        command_inspection: CommandInspection | None = None,
    ) -> None:
        self._command_inspection = (
            command_inspection if command_inspection is not None else CommandInspection()
        )
        self._host_runtime_bindings = {}
        self._workspace_root = (
            Path(workspace_root).resolve() if workspace_root is not None else None
        )
        self._temporary_directory = tempfile.TemporaryDirectory(  # pylint: disable=consider-using-with
            prefix=constants.TEMPORARY_DIRECTORY_PREFIX
        )
        self._temporary_path = Path(self._temporary_directory.name).resolve()
        _LIVE_MANAGERS.add(self)
        try:
            self._configuration_path = (
                Path(configuration_path)
                if configuration_path is not None
                else self._workspace_root / constants.APP_DIRECTORY / constants.PERMISSIONS_FILENAME
                if self._workspace_root is not None
                else None
            )
            self._user_configuration_path = (
                Path(user_configuration_path).expanduser().resolve()
                if user_configuration_path is not None
                else None
            )
            self._workspace_id = workspace_id
            self._audit_store = (
                SQLitePermissionAudit(audit_path) if audit_path is not None else None
            )
            self._interaction = interaction
            self._recorder = recorder
            self._load_policy = load_policy or (
                PermissionLoadPolicy.INTERACTIVE
                if interaction is not None
                else PermissionLoadPolicy.ERROR
            )
            if self._load_policy is PermissionLoadPolicy.INTERACTIVE and interaction is None:
                raise ValueError("Interactive permission loading requires an Interaction.")
            self._configuration = configuration or PermissionConfiguration()
            if configuration is None:
                self._load_configuration()
            self._user_configuration = UserPermissionConfiguration()
            self._load_user_configuration()
            self._session_overrides = SessionPolicyOverrides()
            builtin_presets = self._load_presets()
            catalog = (*builtin_presets, *(presets or ()))
            identifiers = [preset.metadata.id for preset in catalog]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("Permission preset identifiers must be unique.")
            self._presets = {preset.metadata.id: preset for preset in catalog}
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release the manager-owned temporary directory."""
        self._temporary_directory.cleanup()

    def __del__(self) -> None:
        self.close()

    @property
    def configuration(self) -> PermissionConfiguration:
        """Return the persisted workspace policy.

        Returns:
            PermissionConfiguration: Workspace defaults, limits, and rules.
        """
        return self._configuration.model_copy(deep=True)

    @property
    def effective_configuration(self) -> PermissionConfiguration:
        """Return the workspace policy with session overrides applied.

        Returns:
            PermissionConfiguration: Complete effective defaults, limits, and active rules.
        """
        defaults = dict(self._configuration.defaults)
        defaults.update(self._session_overrides.defaults)
        limit_updates = {
            name: value
            for name in _LIMIT_NAMES
            if (value := getattr(self._session_overrides.limits, name)) is not None
        }
        limits = self._configuration.limits.model_copy(update=limit_updates)
        return self._configuration.model_copy(
            update={
                "defaults": defaults,
                "limits": limits,
                "rules": [*self._configuration.rules, *self._session_overrides.rules],
            },
            deep=True,
        )

    @property
    def session_overrides(self) -> SessionPolicyOverrides:
        """Return an immutable snapshot of active session policy changes.

        Returns:
            SessionPolicyOverrides: Deep copy of the process-local policy overlay.
        """
        return self._session_overrides.model_copy(deep=True)

    @property
    def configuration_path(self) -> Path | None:
        """Return the local policy path.

        Returns:
            Path | None: YAML path, or None for an in-memory manager.
        """
        return self._configuration_path

    @property
    def user_configuration_path(self) -> Path | None:
        """Return the user-wide remembered-approval policy path.

        Returns:
            Path | None: YAML path, or None when user-wide approvals are unavailable.
        """
        return self._user_configuration_path

    @property
    def temporary_directory(self) -> Path:
        """Return the manager-owned scratch directory allowed by ``loop-temp``.

        Returns:
            Path: Existing private temporary directory removed with this manager.
        """
        return self._temporary_path

    @property
    def interaction(self) -> Interaction | None:
        """Return the interaction used for approvals.

        Returns:
            Interaction | None: Configured interaction, when available.
        """
        return self._interaction

    @interaction.setter
    def interaction(self, interaction: Interaction | None) -> None:
        """Set the interaction used for approvals.

        Args:
            interaction (Interaction | None): New interaction or None for headless use.
        """
        self._interaction = interaction

    @property
    def recorder(self) -> PermissionRecorder | None:
        """Return the authorization observation sink.

        Returns:
            PermissionRecorder | None: Configured recorder, when available.
        """
        return self._recorder

    @recorder.setter
    def recorder(self, recorder: PermissionRecorder | None) -> None:
        """Set the authorization observation sink.

        Args:
            recorder (PermissionRecorder | None): New recorder or None to disable recording.
        """
        self._recorder = recorder

    @property
    def persistent_rules(self) -> tuple[PermissionRule, ...]:
        """Return persisted policy rules in display order.

        Returns:
            tuple[PermissionRule, ...]: Immutable snapshot of persisted rules.
        """
        return tuple(rule.model_copy(deep=True) for rule in self._configuration.rules)

    @property
    def user_rules(self) -> tuple[PermissionRule, ...]:
        """Return user-wide remembered approval rules in display order.

        Returns:
            tuple[PermissionRule, ...]: Immutable snapshot of user-wide rules.
        """
        return tuple(rule.model_copy(deep=True) for rule in self._user_configuration.rules)

    @property
    def session_rules(self) -> tuple[PermissionRule, ...]:
        """Return process-local policy rules in display order.

        Returns:
            tuple[PermissionRule, ...]: Immutable snapshot of session rules.
        """
        return tuple(rule.model_copy(deep=True) for rule in self._session_overrides.rules)

    @property
    def presets(self) -> tuple[PermissionPreset, ...]:
        """Return selectable permission presets ordered by stable identifier.

        Returns:
            tuple[PermissionPreset, ...]: Deep-copied preset catalog entries.
        """
        return tuple(
            self._presets[identifier].model_copy(deep=True) for identifier in sorted(self._presets)
        )

    def authorize(
        self,
        operations: Operations,
        *,
        interaction: Interaction | None = None,
        recorder: PermissionRecorder | None = None,
        protected_instruction_names: tuple[str, ...] = (),
    ) -> AuthorizationResult:
        """Evaluate and approve one complete operation set atomically.

        Args:
            operations (Operations): Complete normalized effects of one tool call.
            interaction (Interaction | None): Invocation interaction overriding the default.
            recorder (PermissionRecorder | None): Invocation recorder overriding the default.
            protected_instruction_names (tuple[str, ...]): Active instruction filenames that
                require a fresh decision when a structured tool changes them.

        Returns:
            AuthorizationResult: Policy, prompt, and effective result for the complete set.
        """
        policy = self.evaluate(operations)
        fresh_file_review = self._requires_fresh_file_review(
            operations, self.protected_instruction_names(protected_instruction_names)
        )
        if fresh_file_review and policy.decision is not Decision.DENY:
            policy = PolicyDecision(
                decision=Decision.ASK,
                reason="A user-managed replacement, deletion, "
                "or instruction change needs fresh approval.",
                sources=("boundary:fresh_file_review",),
            )
        active_interaction = interaction if interaction is not None else self._interaction
        prompt = None
        decision = policy.decision
        reason = policy.reason
        source = "policy"
        approval_choice = None
        installed_rule_ids = ()
        if policy.decision is Decision.ASK:
            if active_interaction is None:
                decision = Decision.DENY
                reason = "Approval is required but no interactive user is available."
                source = "headless"
            else:
                prompt_body = self._prompt(operations)
                prompt = (
                    f"{prompt_body}\nProceed with this one file change?"
                    if fresh_file_review
                    else f"{prompt_body}\nProceed?"
                )
                selected = self.request_permission(
                    prompt_body,
                    interaction=active_interaction,
                    fresh_only=fresh_file_review,
                )
                approval_choice = (
                    selected
                    if isinstance(selected, ApprovalChoice)
                    else ApprovalChoice.ONCE
                    if selected is True
                    else ApprovalChoice.DENY
                )
                if fresh_file_review and approval_choice is not ApprovalChoice.ONCE:
                    approval_choice = ApprovalChoice.DENY
                if approval_choice is ApprovalChoice.WORKSPACE and self._configuration_path is None:
                    approval_choice = ApprovalChoice.DENY
                    decision = Decision.DENY
                    reason = "Workspace approval is unavailable without a workspace policy path."
                if approval_choice is ApprovalChoice.USER and self._user_configuration_path is None:
                    approval_choice = ApprovalChoice.DENY
                    decision = Decision.DENY
                    reason = "User approval is unavailable without a user policy path."
                if approval_choice is not ApprovalChoice.DENY:
                    decision = Decision.ALLOW
                    reason = f"Approved by the user with {approval_choice.value} scope."
                    if approval_choice in {
                        ApprovalChoice.SESSION,
                        ApprovalChoice.WORKSPACE,
                        ApprovalChoice.USER,
                    }:
                        try:
                            installed_rule_ids = self._remember_approval(
                                operations, scope=approval_choice
                            )
                        except OSError as exc:
                            decision = Decision.DENY
                            reason = f"Could not persist the approved permission policy: {exc}"
                            source = "persistence"
                elif reason == policy.reason:
                    decision = Decision.DENY
                    reason = "Rejected by the user."
                if source != "persistence":
                    source = "user"
        result = AuthorizationResult(
            operations=operations,
            policy=policy,
            decision=decision,
            prompted=prompt is not None,
            prompt=prompt,
            reason=reason,
            source=source,
            approval_choice=approval_choice,
            installed_rule_ids=installed_rule_ids,
        )
        telemetry_audit(
            "permission.decided",
            decision=result.decision.value,
            source=result.source,
            prompted=result.prompted,
            approval_scope=(
                result.approval_choice.value if result.approval_choice is not None else None
            ),
            operation_count=len(result.operations),
        )
        telemetry_trace_event(
            "permission.decision",
            payload=result,
        )
        self._audit(result)
        active_recorder = recorder if recorder is not None else self._recorder
        if active_recorder is not None:
            active_recorder.record_authorization(result)
        return result

    def _requires_fresh_file_review(
        self,
        operations: Operations,
        protected_instruction_names: tuple[str, ...],
    ) -> bool:
        """Identify user-owned replacement/deletion and instruction mutations."""
        protection = protected_workspace_paths(protected_instruction_names)
        for operation in operations:
            target = operation.target
            if not isinstance(target, FileTarget) or operation.action not in _WRITE_ACTIONS:
                continue
            if operation.action in {Action.FILESYSTEM_REPLACE, Action.FILESYSTEM_DELETE}:
                return True
            if self._workspace_root is None:
                continue
            path = Path(target.path).resolve(strict=False)
            if not path.is_relative_to(self._workspace_root):
                continue
            relative = path.relative_to(self._workspace_root)
            if relative.name in (
                *protection.instruction_names,
                *protection.files,
            ) or protection.protects_directory(relative):
                return True
        return False

    def inspect_command(self, source: str) -> tuple[CommandFinding, ...]:
        """Collect command findings from this manager's registry.

        Args:
            source (str): Opaque shell source to inspect heuristically.

        Returns:
            tuple[CommandFinding, ...]: Applicable findings; absence does not prove safety.
        """
        return self._command_inspection.inspect(source)

    def analyze_command(self, source: str) -> CommandAnalysis:
        """Collect advisory findings and executable names in one inspection.

        Args:
            source (str): Opaque shell source to inspect heuristically.

        Returns:
            CommandAnalysis: Bounded findings and candidate names; absence proves no safety.
        """
        return self._command_inspection.analyze(source)

    def protected_instruction_names(self, configured: Iterable[str] = ()) -> tuple[str, ...]:
        """Bind configured instruction filenames into native command protection.

        Args:
            configured (Iterable[str]): Active project instruction basenames.

        Returns:
            tuple[str, ...]: Default and active filenames requiring native write denial.

        Raises:
            ValueError: If a configured name is not a plain filename.
        """
        return protected_workspace_paths(configured).instruction_names

    def protected_sandbox_paths(self, configured: Iterable[str] = ()) -> ProtectedWorkspacePaths:
        """Describe workspace paths that ordinary sandbox commands cannot mutate.

        Args:
            configured (Iterable[str]): Active project instruction basenames.

        Returns:
            ProtectedWorkspacePaths: Instruction basenames and protected relative paths.

        Raises:
            ValueError: If a configured instruction name is not a plain filename.
        """
        return protected_workspace_paths(configured)

    @staticmethod
    def git_sandbox_roots(
        workspace: Path,
        *,
        read: bool,
        write: bool,
    ) -> tuple[tuple[Path, ...], tuple[Path, ...], bool]:
        """Bind Git metadata reads, writes, and creation for one workspace.

        Args:
            workspace (Path): Canonical workspace root.
            read (bool): Whether the command inspects Git metadata.
            write (bool): Whether Git metadata mutation was requested.

        Returns:
            tuple[tuple[Path, ...], tuple[Path, ...], bool]: Linked metadata read roots,
                approved metadata write roots, and new repository creation intent.

        Raises:
            ValueError: If a linked worktree pointer or creation target is unsafe.
        """
        if not (read or write):
            return (), (), False
        git_directory = workspace / constants.GIT_DIRECTORY
        if git_directory.is_file():
            details = git_directory.lstat()
            if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
                raise ValueError("Unsafe linked worktree Git pointer.")
            pointer = git_directory.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir: "):
                raise ValueError("Invalid linked worktree Git directory pointer.")
            git_root = (git_directory.parent / pointer[8:]).resolve(strict=True)
            if not git_root.is_dir() or not (git_root / "HEAD").is_file():
                raise ValueError("Invalid linked worktree Git metadata directory.")
            reads = (git_root,)
            common_pointer = git_root / "commondir"
            if common_pointer.exists():
                common_details = common_pointer.lstat()
                if not stat.S_ISREG(common_details.st_mode) or common_details.st_nlink != 1:
                    raise ValueError("Unsafe linked worktree common Git pointer.")
                common_directory = (
                    git_root / common_pointer.read_text(encoding="utf-8").strip()
                ).resolve(strict=True)
                if not common_directory.is_dir() or not (common_directory / "HEAD").is_file():
                    raise ValueError("Invalid linked worktree common Git directory.")
                reads = (*reads, common_directory)
            return reads, reads if write else (), False
        if write:
            if git_directory.is_symlink():
                raise ValueError("Unsafe Git metadata creation target.")
            return (
                (),
                (git_directory,) if git_directory.exists() else (),
                not git_directory.exists(),
            )
        return (), (), False

    @staticmethod
    def describe_sandbox_reads(
        workspace: Path,
        roots: tuple[Path, ...],
        manual_roots: tuple[Path, ...],
        tool_names: tuple[str, ...],
        package_installation: bool,
    ) -> str | None:
        """Describe expanded read authority without exposing host paths in a prompt.

        Args:
            workspace (Path): Selected user workspace root.
            roots (tuple[Path, ...]): All effective extra read roots.
            manual_roots (tuple[Path, ...]): Roots supplied directly by the caller.
            tool_names (tuple[str, ...]): Names of identity-bound installed tools.
            package_installation (bool): Whether all tool roots are a recognized package install.

        Returns:
            str | None: Honest plain-language context for expanded reads, if any.
        """
        if not roots:
            return None
        if any(
            root == Path("/")
            or (workspace.is_relative_to(root) and root != workspace)
            or (root in manual_roots and len(root.parts) <= 3)
            for root in roots
        ):
            return "Read a broad area outside the workspace"
        if tool_names and manual_roots:
            return "Use installed tools and read an additional folder outside the workspace"
        if len(tool_names) > 1:
            return "Use installed developer tools"
        if tool_names:
            return (
                f"Use installed {tool_names[0]}"
                if package_installation
                else f"Read the installed {tool_names[0]} tool folder"
            )
        return "Read an additional folder outside the workspace"

    @staticmethod
    def host_retry_prompt(
        source: str,
        outcome: str,
        exit_code: int | None,
        possible_effects: bool,
    ) -> str:
        """Describe an offered unrestricted host run and possible repeated effects.

        Args:
            source (str): Exact shell source proposed for the host retry.
            outcome (str): Completed, denied, or unavailable sandbox outcome.
            exit_code (int | None): Completed sandbox command's exit code, if any.
            possible_effects (bool): Whether the sandbox attempt may have changed state.

        Returns:
            str: Human warning for the separate host authorization boundary.
        """
        if outcome == "completed":
            reason = (
                "The command exited with code "
                f"{exit_code}, and the OS denied an operation during this attempt. "
                "The denial may be unrelated to the exit."
            )
        else:
            reason = (
                "The sandbox denied this command."
                if outcome == "denied"
                else "The sandbox was unavailable."
            )
        repeat = (
            "\nThe sandboxed attempt may already have had effects. "
            "Retrying may repeat those effects."
            if possible_effects or outcome == "completed"
            else ""
        )
        return (
            f"Command: {json.dumps(source)}\n"
            f"Reason: {reason} Running without the OS sandbox can read private files, "
            "change files outside the workspace, and use the network."
            f"{repeat}"
        )

    @staticmethod
    def host_retry_choices(reusable: bool) -> tuple[dict[str, str], dict[str, str]]:
        """Return permission-owned choices for a fresh or reusable host offer.

        Args:
            reusable (bool): Whether a prelaunch failure permits a stable exact session rule.

        Returns:
            tuple[dict[str, str], dict[str, str]]: Choice labels and keyboard index.
        """
        choices = {"deny": "Deny", "approve": "Approve this one host run"}
        index = {"deny": "N", "approve": "Y"}
        if reusable:
            choices["session"] = "Allow this exact host command for this session"
            index["session"] = "S"
        return choices, index

    def command_needs_git_write(self, source: str) -> bool:
        """Report whether this registry requests Git metadata write authority.

        Args:
            source (str): Opaque shell source to inspect conservatively.

        Returns:
            bool: Whether any inspector requests Git write authority. Unrecognized shell
                indirection remains subject to the OS protection on Git metadata.
        """
        return any(finding.requests_git_write for finding in self.inspect_command(source))

    def authorize_sandboxed_command(
        self,
        request: SandboxRequest,
        *,
        tool_id: str,
        interaction: Interaction | None = None,
    ) -> bool:
        """Authorize a bound sandbox request using sandbox-only scoped approval rules.

        Legacy process allow rules never approve this request. Explicit deny policy still takes
        precedence. Commands within the default workspace sandbox run without a prompt;
        additional authority requires a sandbox-only approval.

        Args:
            request (SandboxRequest): Exact command, authority, and path identities to approve.
            tool_id (str): Registered execution tool identity for policy and audit.
            interaction (Interaction | None): Interactive approval surface.

        Returns:
            bool: Whether policy or the user approved this sandbox attempt.
        """
        policy = self.evaluate((self._sandbox_operation(request, tool_id),))
        if policy.decision is Decision.DENY:
            self._audit_sandbox_denial(request, source="policy")
            return False
        scratch_roots = self._sandbox_scratch_roots(request)
        explicit_ask = policy.decision is Decision.ASK and any(
            source.startswith("rule:") for source in policy.sources
        )
        findings = self.inspect_command(request.source)
        fresh_review = any(finding.status is CommandReviewStatus.FRESH for finding in findings)
        self._audit_sandbox_scope(request, findings)
        if self._has_default_sandbox_access(
            request,
            scratch_roots,
            explicit_ask=explicit_ask,
            findings=findings,
        ):
            self._audit_sandbox_decision(request, decision="allow", source="workspace_sandbox")
            return True
        matched_scope = None if fresh_review else self._matching_sandbox_rule_scope(request)
        if matched_scope is not None:
            self._audit_sandbox_decision(
                request,
                decision="allow",
                source=f"rule:{matched_scope.value}",
            )
            return True
        active_interaction = interaction if interaction is not None else self._interaction
        if active_interaction is None:
            self._audit_sandbox_denial(request, source="headless")
            return False

        approved = self._request_sandbox_approval(
            request,
            interaction=active_interaction,
            scratch_roots=scratch_roots,
            findings=findings,
            explicit_ask=explicit_ask,
            fresh_review=fresh_review,
        )
        self._audit_sandbox_decision(
            request,
            decision="allow" if approved else "deny",
            source="interactive",
        )
        return approved

    @staticmethod
    def _sandbox_operation(request: SandboxRequest, tool_id: str) -> Operation:
        """Build the policy operation representing one sandboxed command."""
        return Operation(
            tool_id=tool_id,
            action=Action.PROCESS_EXECUTE,
            target=ProcessTarget(
                argv=("/bin/sh", "-c", request.source),
                cwd=str(request.cwd),
                boundary=ProcessBoundary.SANDBOXED,
            ),
        )

    @staticmethod
    def _sandbox_scratch_roots(request: SandboxRequest) -> set[Path]:
        """Return host paths backing the request's temporary aliases."""
        return {
            target for virtual, _, target in request.aliases if virtual == VirtualPath.TEMPORARY
        }

    def _audit_sandbox_scope(
        self,
        request: SandboxRequest,
        findings: tuple[CommandFinding, ...],
    ) -> None:
        """Record the exact sandbox authority considered for approval."""
        self._append_audit(
            "sandbox.permission_scope",
            {
                "attempt_id": request.attempt_id,
                "read_roots": [str(root) for root in request.read_roots],
                "automatic_tool_reads": [str(root) for root in request.automatic_tool_reads],
                "read_aliases": [str(alias) for alias in request.read_aliases],
                "executables": [
                    {"name": name, "spelling": str(spelling), "target": str(resolved)}
                    for name, spelling, resolved, _, _ in request.executable_identities
                ],
                "write_roots": [str(root) for root in request.write_roots],
                "policy_version": request.policy_version,
                "review_policies": [finding.policy_id for finding in findings],
            },
        )

    @staticmethod
    def _has_default_sandbox_access(
        request: SandboxRequest,
        scratch_roots: set[Path],
        *,
        explicit_ask: bool,
        findings: tuple[CommandFinding, ...],
    ) -> bool:
        """Report whether a request stays inside the unprompted workspace sandbox."""
        allowed_aliases = all(
            (virtual == VirtualPath.WORKSPACE and target == request.workspace)
            or (virtual == VirtualPath.TEMPORARY and target in scratch_roots)
            for virtual, _, target in request.aliases
        )
        return (
            not explicit_ask
            and not findings
            and set(request.read_roots) <= set(request.automatic_tool_reads)
            and not request.network
            and not request.git_create
            and set(request.write_roots) <= {request.workspace, *scratch_roots}
            and allowed_aliases
        )

    def _matching_sandbox_rule_scope(self, request: SandboxRequest) -> ApprovalChoice | None:
        """Return the first remembered scope that approves this exact request."""
        for scope in (
            ApprovalChoice.SESSION,
            ApprovalChoice.WORKSPACE,
            ApprovalChoice.USER,
        ):
            signature = request.sandbox_command_signature(
                self._sandbox_scope_identity(request, scope)
            )
            if any(
                self._sandbox_matches(rule, request.source, signature)
                for rule in self.sandboxed_command_rules(scope)
            ):
                return scope
        return None

    def _request_sandbox_approval(
        self,
        request: SandboxRequest,
        *,
        interaction: Interaction,
        scratch_roots: set[Path],
        findings: tuple[CommandFinding, ...],
        explicit_ask: bool,
        fresh_review: bool,
    ) -> bool:
        """Prompt for a sandbox command and remember a selected approval when allowed."""
        interaction.info(
            self._sandbox_permission_message(
                request,
                scratch_roots=scratch_roots,
                findings=findings,
                explicit_ask=explicit_ask,
            )
        )
        choices, index, prefix = self._sandbox_approval_choices(request, fresh_review=fresh_review)
        selected = interaction.prompt(
            "Allow this command?", exit_commands=None, choices=choices, index=index
        )
        if selected not in choices or selected == "deny":
            return False
        return selected == "approve" or self._remember_sandbox_approval(
            request,
            selected=selected,
            prefix=prefix,
        )

    def _sandbox_permission_message(
        self,
        request: SandboxRequest,
        *,
        scratch_roots: set[Path],
        findings: tuple[CommandFinding, ...],
        explicit_ask: bool,
    ) -> str:
        """Describe the additional authority requested without exposing local paths."""
        reads_outside_workspace = request.read_roots or any(
            not (
                (virtual == VirtualPath.WORKSPACE and target == request.workspace)
                or (virtual == VirtualPath.TEMPORARY and target in scratch_roots)
            )
            for virtual, _, target in request.aliases
        )
        read_context = self._sandbox_read_context(request, reads_outside_workspace)
        reasons = self._sandbox_permission_reasons(
            request,
            scratch_roots=scratch_roots,
            findings=findings,
            reads_outside_workspace=reads_outside_workspace,
            explicit_ask=explicit_ask,
        )
        contexts = [read_context] if read_context is not None else []
        contexts.extend(finding.context for finding in findings)
        if not contexts:
            contexts.append("Run this command with additional sandbox access")
        return (
            f"Command: {json.dumps(request.source)}\n"
            f"Context: {'; '.join(contexts)}.\n"
            f"Reason: Needs permission to {', '.join(reasons)}."
        )

    @staticmethod
    def _sandbox_read_context(request: SandboxRequest, reads_outside_workspace: bool) -> str | None:
        """Describe extra read authority at a human-safe level."""
        if not reads_outside_workspace:
            return request.read_context
        if request.read_context is not None:
            return request.read_context
        broad_roots = {
            Path("/"),
            Path("/Users"),
            Path("/opt"),
            Path("/usr"),
            Path("/private/tmp"),
            Path("/tmp"),
        }
        if any(
            root in broad_roots
            or (request.workspace.is_relative_to(root) and root != request.workspace)
            for root in request.read_roots
        ):
            return "Read a broad area outside the workspace"
        return "Read an additional folder outside the workspace"

    @staticmethod
    def _sandbox_permission_reasons(
        request: SandboxRequest,
        *,
        scratch_roots: set[Path],
        findings: tuple[CommandFinding, ...],
        reads_outside_workspace: bool,
        explicit_ask: bool,
    ) -> list[str]:
        """List the policy reasons that require an interactive decision."""
        reasons = ["read files outside this workspace"] if reads_outside_workspace else []
        extra_writes = set(request.write_roots) - {request.workspace, *scratch_roots}
        if extra_writes:
            reasons.append(
                "change Git metadata"
                if all(".git" in root.parts for root in extra_writes)
                else "write outside this workspace"
            )
        if request.git_create:
            reasons.append("create Git metadata")
        if request.network:
            reasons.append("use the network")
        reasons.extend(finding.reason for finding in findings)
        if explicit_ask and not reasons:
            reasons.append("run under your command approval rule")
        return reasons

    def _sandbox_approval_choices(
        self,
        request: SandboxRequest,
        *,
        fresh_review: bool,
    ) -> tuple[dict[str, str], dict[str, str], str | None]:
        """Build permitted interactive choices and their optional similar-command prefix."""
        choices = {"deny": "Deny", "approve": "Allow once", "session": "Allow for this session"}
        index = {"deny": "N", "approve": "Y", "session": "S"}
        if fresh_review:
            choices.pop("session")
            index.pop("session")
            return choices, index, None
        if self._user_configuration_path is not None:
            choices["workspace"] = "Allow in this workspace"
            index["workspace"] = "W"
            choices["user"] = "Allow for this user"
            index["user"] = "U"
        prefix = self._similar_prefix(request.source)
        if prefix is not None:
            self._add_similar_sandbox_choices(choices, index, prefix)
        return choices, index, prefix

    @staticmethod
    def _add_similar_sandbox_choices(
        choices: dict[str, str],
        index: dict[str, str],
        prefix: str,
    ) -> None:
        """Add similar-command choices for each available remembered-rule scope."""
        for scope, shortcut in (("session", "A"), ("workspace", "B"), ("user", "C")):
            if scope in choices:
                choices[f"similar_{scope}"] = f"Allow similar '{prefix} …' commands ({scope})"
                index[f"similar_{scope}"] = shortcut

    def _remember_sandbox_approval(
        self,
        request: SandboxRequest,
        *,
        selected: str,
        prefix: str | None,
    ) -> bool:
        """Persist a remembered approval selected through the sandbox command prompt."""
        similar = selected.startswith("similar_")
        scope = ApprovalChoice(selected.removeprefix("similar_"))
        rule = SandboxedCommandRule(
            signature=request.sandbox_command_signature(
                self._sandbox_scope_identity(request, scope)
            ),
            source=prefix if similar else request.source,
            similar=similar,
            scope=scope,
        )
        try:
            if scope is ApprovalChoice.SESSION:
                self._session_overrides.sandboxed_command_rules.append(rule)
            else:
                updated_user = self._user_configuration.model_copy(deep=True)
                updated_user.sandboxed_command_rules.append(rule)
                self._replace_user_configuration(updated_user)
        except OSError:
            return False
        telemetry_audit(
            "sandbox.permission_rule_added",
            workspace_id=request.workspace_id,
            scope=scope.value,
            rule_id=rule.id,
            similar=similar,
        )
        return True

    @staticmethod
    def _audit_sandbox_decision(request: SandboxRequest, *, decision: str, source: str) -> None:
        """Record a completed sandbox authorization decision."""
        telemetry_audit(
            "sandbox.permission_decided",
            workspace_id=request.workspace_id,
            attempt_id=request.attempt_id,
            decision=decision,
            source=source,
        )

    @staticmethod
    def _audit_sandbox_denial(request: SandboxRequest, *, source: str) -> None:
        """Record a sandbox denial before interactive approval is available."""
        telemetry_audit(
            "sandbox.permission_denied",
            workspace_id=request.workspace_id,
            attempt_id=request.attempt_id,
            source=source,
        )

    @staticmethod
    def _sandbox_scope_identity(request: SandboxRequest, scope: ApprovalChoice) -> str:
        """Return the policy identity applicable to one remembered-rule scope."""
        return request.workspace_id if scope is not ApprovalChoice.USER else "user"

    @staticmethod
    def _similar_prefix(source: str) -> str | None:
        """Offer similarity only for simple literal command words."""
        if _SIMPLE_COMMAND.fullmatch(source) is None:
            return None
        words = source.split()
        if words[0] in {
            "python",
            "python3",
            "sh",
            "bash",
            "zsh",
            "node",
            "ruby",
            "perl",
            "env",
        }:
            return None
        if words[1].startswith("-"):
            return None
        return " ".join(words[:2])

    @staticmethod
    def _sandbox_matches(rule: SandboxedCommandRule, source: str, signature: str) -> bool:
        """Match an exact source or reviewed simple prefix at one authority."""
        if rule.signature != signature:
            return False
        if not rule.similar:
            return rule.source == source
        return PermissionManager._similar_prefix(source) == rule.source and (
            source == rule.source or source.startswith(rule.source + " ")
        )

    def sandboxed_command_rules(self, scope: ApprovalChoice) -> tuple[SandboxedCommandRule, ...]:
        """Return sandbox-only approvals stored at one lifetime scope.

        Args:
            scope (ApprovalChoice): Session, workspace, or user policy layer.

        Returns:
            tuple[SandboxedCommandRule, ...]: Copies of remembered sandbox approvals.

        Raises:
            ValueError: If the scope cannot store approvals.
        """
        if scope is ApprovalChoice.SESSION:
            rules = self._session_overrides.sandboxed_command_rules
        elif scope in {ApprovalChoice.WORKSPACE, ApprovalChoice.USER}:
            rules = [
                rule
                for rule in self._user_configuration.sandboxed_command_rules
                if rule.scope is scope
            ]
        else:
            raise ValueError("Native command rules require session, workspace, or user scope.")
        return tuple(rule.model_copy(deep=True) for rule in rules)

    def remove_sandboxed_command_rule(self, scope: ApprovalChoice, rule_id: str) -> bool:
        """Remove one remembered sandbox-only approval by identifier.

        Args:
            scope (ApprovalChoice): Session, workspace, or user policy layer.
            rule_id (str): Identifier of the approval to remove.

        Returns:
            bool: Whether a rule was removed.

        Raises:
            ValueError: If the scope cannot store approvals.
            OSError: If the changed persistent policy cannot be written.
        """
        rules = list(self.sandboxed_command_rules(scope))
        remaining = [rule for rule in rules if rule.id != rule_id]
        if len(remaining) == len(rules):
            return False
        if scope is ApprovalChoice.SESSION:
            self._session_overrides.sandboxed_command_rules = remaining
        else:
            updated_user = self._user_configuration.model_copy(deep=True)
            updated_user.sandboxed_command_rules = [
                rule
                for rule in updated_user.sandboxed_command_rules
                if rule.id != rule_id or rule.scope is not scope
            ]
            self._replace_user_configuration(updated_user)
        telemetry_audit("sandbox.permission_rule_removed", scope=scope.value, rule_id=rule_id)
        return True

    def host_command_rules(self) -> tuple[HostCommandRule, ...]:
        """List revocable exact host approvals for this session.

        Returns:
            tuple[HostCommandRule, ...]: Independent copies of active host rules.
        """
        return tuple(
            rule.model_copy(deep=True) for rule in self._session_overrides.host_command_rules
        )

    def matching_host_command_rule(self, request: SandboxRequest) -> HostCommandRule | None:
        """Find an exact host rule for the current bound command context.

        Args:
            request (SandboxRequest): Freshly validated command request.

        Returns:
            HostCommandRule | None: Matching session approval, if present.
        """
        signature = request.host_command_signature()
        return next(
            (
                rule.model_copy(deep=True)
                for rule in self._session_overrides.host_command_rules
                if rule.signature == signature
                and self._host_runtime_bindings.get(rule.id)
                == request.host_command_runtime_signature()
            ),
            None,
        )

    def remember_host_command_rule(self, request: SandboxRequest) -> HostCommandRule:
        """Record one explicitly approved exact host retry for this session.

        Args:
            request (SandboxRequest): Unchanged context approved at the host boundary.

        Returns:
            HostCommandRule: New revocable host-only decision.
        """
        relative_cwd = request.cwd.relative_to(request.workspace)
        rule = HostCommandRule(
            signature=request.host_command_signature(),
            cwd=(
                VirtualPath.WORKSPACE
                if relative_cwd == Path(".")
                else f"{VirtualPath.WORKSPACE}/{relative_cwd.as_posix()}"
            ),
            source=request.source if "/" not in request.source else "<path-bearing-command>",
        )
        self._session_overrides.host_command_rules.append(rule)
        self._host_runtime_bindings[rule.id] = request.host_command_runtime_signature()
        telemetry_audit(
            "host_command.rule_added", workspace_id=request.workspace_id, rule_id=rule.id
        )
        return rule.model_copy(deep=True)

    def remove_host_command_rule(self, rule_id: str) -> bool:
        """Revoke one session host approval by identifier.

        Args:
            rule_id (str): Identifier of the recorded approval.

        Returns:
            bool: Whether a matching rule was removed.
        """
        before = len(self._session_overrides.host_command_rules)
        self._session_overrides.host_command_rules = [
            rule for rule in self._session_overrides.host_command_rules if rule.id != rule_id
        ]
        removed = len(self._session_overrides.host_command_rules) != before
        if removed:
            self._host_runtime_bindings.pop(rule_id, None)
            telemetry_audit("host_command.rule_removed", rule_id=rule_id)
        return removed

    def request_permission(
        self,
        prompt: str,
        *,
        interaction: Interaction | None = None,
        fresh_only: bool = False,
    ) -> ApprovalChoice:
        """Request one scoped approval through a generic user interaction.

        Args:
            prompt (str): Complete operation description to display.
            interaction (Interaction | None): Invocation interaction overriding the default.
            fresh_only (bool): Offer only a one-time choice for user-managed data changes.

        Returns:
            ApprovalChoice: Selected denial or approval lifetime. Cancellation denies.
        """
        choices = {
            ApprovalChoice.DENY: "Deny",
            ApprovalChoice.ONCE: "Allow once",
            ApprovalChoice.SESSION: "Allow for this session",
        }
        index = {
            ApprovalChoice.DENY: "N",
            ApprovalChoice.ONCE: "Y",
            ApprovalChoice.SESSION: "S",
        }
        if fresh_only:
            choices.pop(ApprovalChoice.SESSION)
            index.pop(ApprovalChoice.SESSION)
        if not fresh_only and self._configuration_path is not None:
            choices[ApprovalChoice.WORKSPACE] = "Allow in this workspace"
            index[ApprovalChoice.WORKSPACE] = "W"
        if not fresh_only and self._user_configuration_path is not None:
            choices[ApprovalChoice.USER] = "Always allow for this user"
            index[ApprovalChoice.USER] = "U"
        active_interaction = interaction if interaction is not None else self._interaction
        active_interaction.info(prompt)
        selected = active_interaction.prompt(
            "Proceed?",
            exit_commands=None,
            choices=choices,
            index=index,
        )
        return selected if isinstance(selected, ApprovalChoice) else ApprovalChoice.DENY

    def _remember_approval(
        self,
        operations: Operations,
        *,
        scope: ApprovalChoice,
    ) -> tuple[str, ...]:
        """Install exact, deduplicated allow rules for one approved operation set."""
        if scope is ApprovalChoice.USER:
            existing = self._user_configuration.rules
        elif scope is ApprovalChoice.WORKSPACE:
            existing = self._configuration.rules
        else:
            existing = self._session_overrides.rules
        additions = []
        for operation in operations:
            target = self._approval_target(operation.target)
            if any(
                rule.decision is Decision.ALLOW
                and rule.tool == operation.tool_id
                and (rule.tool_exact or not any(character in rule.tool for character in "*?["))
                and rule.action is operation.action
                and rule.target == target
                for rule in (*existing, *additions)
            ):
                continue
            additions.append(
                PermissionRule(
                    decision=Decision.ALLOW,
                    tool=operation.tool_id,
                    tool_exact=True,
                    action=operation.action,
                    target=target,
                    description=f"Approved interactively for this {scope.value}.",
                )
            )
        if scope is ApprovalChoice.USER:
            updated = self._user_configuration.model_copy(deep=True)
            updated.rules.extend(additions)
            self._replace_user_configuration(updated)
        elif scope is ApprovalChoice.WORKSPACE:
            updated = self._configuration.model_copy(deep=True)
            updated.rules.extend(additions)
            self._replace_configuration(updated)
        else:
            self._session_overrides.rules.extend(additions)
        return tuple(rule.id for rule in additions)

    @staticmethod
    def _approval_target(target: OperationTarget) -> OperationTarget:
        """Return the stable security-relevant identity of an operation target."""
        if isinstance(target, FileTarget):
            return FileTarget(path=target.path)
        if isinstance(target, NetworkTarget):
            return NetworkTarget(
                url=target.url,
                origin=target.origin,
                method=target.method,
                sends_body=target.sends_body,
            )
        return target.model_copy(deep=True)

    def evaluate(self, operations: Operations) -> PolicyDecision:
        """Evaluate a complete operation set without prompting or recording.

        Args:
            operations (Operations): Complete normalized effects to evaluate.

        Returns:
            PolicyDecision: Composed allow, ask, or deny outcome and determining sources.
        """
        if not operations:
            return PolicyDecision(
                decision=Decision.ALLOW,
                reason="The tool plan requires no authority.",
                sources=("no_authority",),
            )
        decisions = tuple(self._evaluate_operation(operation) for operation in operations)
        for selected in (Decision.DENY, Decision.ASK, Decision.ALLOW):
            determining = tuple(item for item in decisions if item.decision is selected)
            if determining:
                return PolicyDecision(
                    decision=selected,
                    reason=(
                        f"{len(determining)} of {len(operations)} planned operation(s) "
                        f"require {selected.value}."
                    ),
                    sources=tuple(source for item in determining for source in item.sources),
                )
        raise RuntimeError("Every operation must produce a policy decision.")  # pragma: no cover

    def set_default(
        self,
        action: Action,
        decision: Decision,
        *,
        scope: PolicyScope = PolicyScope.WORKSPACE,
    ) -> None:
        """Set the fallback decision for one action.

        Args:
            action (Action): Action whose fallback changes.
            decision (Decision): New fallback outcome.
            scope (PolicyScope): Workspace persistence or process-local lifetime.
        """
        if scope is PolicyScope.WORKSPACE:
            updated = self._configuration.model_copy(deep=True)
            updated.defaults[action] = decision
            self._replace_configuration(updated)
        else:
            self._session_overrides.defaults[action] = decision
        self._audit_policy_change(
            "permission.default_set",
            scope,
            action=action.value,
            decision=decision.value,
        )

    def reset_default(self, action: Action, *, scope: PolicyScope = PolicyScope.SESSION) -> bool:
        """Reset one default to its inherited or application value.

        Args:
            action (Action): Action whose default resets.
            scope (PolicyScope): Workspace bootstrap or session inheritance target.

        Returns:
            bool: Whether the selected layer changed.
        """
        if scope is PolicyScope.SESSION:
            changed = self._session_overrides.defaults.pop(action, None) is not None
            if changed:
                self._audit_policy_change("permission.default_reset", scope, action=action.value)
            return changed
        default = PermissionConfiguration().defaults[action]
        if self._configuration.defaults.get(action, Decision.DENY) is default:
            return False
        updated = self._configuration.model_copy(deep=True)
        updated.defaults[action] = default
        self._replace_configuration(updated)
        self._audit_policy_change("permission.default_reset", scope, action=action.value)
        return True

    def add_rule(
        self,
        rule: PermissionRule,
        *,
        scope: PolicyScope = PolicyScope.WORKSPACE,
    ) -> None:
        """Add one workspace or session-scoped policy rule.

        Args:
            rule (PermissionRule): Rule to add.
            scope (PolicyScope): Workspace persistence or process-local lifetime.

        Raises:
            ValueError: If another active rule already uses the same identifier.
        """
        if any(
            existing.id == rule.id for existing in (*self.persistent_rules, *self.session_rules)
        ):
            raise ValueError(f"Permission rule '{rule.id}' already exists.")
        if scope is PolicyScope.WORKSPACE:
            updated = self._configuration.model_copy(deep=True)
            updated.rules.append(rule)
            self._replace_configuration(updated)
        else:
            self._session_overrides.rules.append(rule)
        self._audit_policy_change(
            "permission.rule_added", scope, rule_id=rule.id, decision=rule.decision.value
        )

    def remove_rule(
        self,
        rule_id: str,
        *,
        scope: PolicyScope = PolicyScope.WORKSPACE,
    ) -> bool:
        """Remove one rule by stable identifier.

        Args:
            rule_id (str): Identifier of the rule to remove.
            scope (PolicyScope): Workspace or session layer to search.

        Returns:
            bool: Whether a rule was removed.
        """
        rules = (
            list(self._configuration.rules)
            if scope is PolicyScope.WORKSPACE
            else self._session_overrides.rules
        )
        for index, rule in enumerate(rules):
            if rule.id == rule_id:
                rules.pop(index)
                if scope is PolicyScope.WORKSPACE:
                    updated = self._configuration.model_copy(update={"rules": rules}, deep=True)
                    self._replace_configuration(updated)
                self._audit_policy_change("permission.rule_removed", scope, rule_id=rule_id)
                return True
        return False

    def preset(self, preset_id: str) -> PermissionPreset:
        """Return one named permission preset artifact.

        Args:
            preset_id (str): Stable preset catalog identifier.

        Returns:
            PermissionPreset: Deep copy of the selected artifact.

        Raises:
            ValueError: If no configured preset uses this identifier.
        """
        try:
            return self._presets[preset_id].model_copy(deep=True)
        except KeyError as exc:
            raise ValueError(f"Unknown permission preset '{preset_id}'.") from exc

    def preview_preset_replacement(
        self,
        preset_id: str,
        *,
        scope: PolicyScope,
    ) -> PresetReplacementPreview:
        """Describe replacement of one scoped defaults-and-rules layer without mutating it.

        Limits and the non-selected policy layer are deliberately absent from this operation. The
        returned preview carries a revision that rejects stale replacements.

        Args:
            preset_id (str): Stable identifier of the selected preset.
            scope (PolicyScope): Workspace persistence or process-local replacement target.

        Returns:
            PresetReplacementPreview: Exact default and rule replacement plus a scope revision.
        """
        preset = self.preset(preset_id)
        return PresetReplacementPreview(
            preset=preset,
            scope=scope,
            removed_defaults=self._defaults_for_scope(scope),
            installed_defaults=dict(preset.defaults),
            removed_rules=self._rules_for_scope(scope),
            installed_rules=self._rules_from_preset(preset, scope),
            scope_revision=self._scope_revision(scope),
        )

    def replace_preset(self, preview: PresetReplacementPreview) -> None:
        """Atomically replace the defaults and rules described by a current preview.

        Args:
            preview (PresetReplacementPreview): Previously reviewed scoped replacement.

        Raises:
            ValueError: If the preview is stale or installs an identifier active in another layer.
        """
        if preview.scope_revision != self._scope_revision(preview.scope):
            raise ValueError("Permission preset preview is stale; generate a new preview.")
        replacement = list(preview.installed_rules)
        other_scope = (
            PolicyScope.SESSION if preview.scope is PolicyScope.WORKSPACE else PolicyScope.WORKSPACE
        )
        other_identifiers = {rule.id for rule in self._rules_for_scope(other_scope)}
        collision = next((rule.id for rule in replacement if rule.id in other_identifiers), None)
        if collision is not None:
            raise ValueError(
                f"Permission rule '{collision}' already exists in {other_scope.value}."
            )
        if preview.scope is PolicyScope.WORKSPACE:
            updated = self._configuration.model_copy(
                update={"defaults": preview.installed_defaults, "rules": replacement}, deep=True
            )
            self._replace_configuration(updated)
        else:
            self._session_overrides.defaults = dict(preview.installed_defaults)
            self._session_overrides.rules = replacement
        self._audit_policy_change(
            "permission.preset_replaced",
            preview.scope,
            preset_id=preview.preset.metadata.id,
            revision=preview.preset.metadata.revision,
        )

    def set_limit(
        self,
        name: str,
        value: bool,
        *,
        scope: PolicyScope = PolicyScope.WORKSPACE,
    ) -> None:
        """Set one boolean enforcement limit.

        Args:
            name (str): ``deny_private_networks`` or ``allow_host_processes``.
            value (bool): New boolean value.
            scope (PolicyScope): Workspace persistence or process-local lifetime.

        Raises:
            ValueError: If ``name`` is not a supported boolean limit.
        """
        if name not in {"deny_private_networks", "allow_host_processes"}:
            raise ValueError(f"Unknown boolean permission limit '{name}'.")
        if scope is PolicyScope.WORKSPACE:
            updated = self._configuration.model_copy(deep=True)
            updated.limits = updated.limits.model_copy(update={name: value})
            self._replace_configuration(updated)
        else:
            self._session_overrides.limits = self._session_overrides.limits.model_copy(
                update={name: value}
            )
        self._audit_policy_change("permission.limit_set", scope, limit=name, enabled=value)

    def update_limit_values(
        self,
        name: str,
        value: str,
        *,
        add: bool,
        scope: PolicyScope = PolicyScope.WORKSPACE,
    ) -> bool:
        """Add or remove one filesystem root or network origin limit.

        Args:
            name (str): ``readable_roots``, ``writable_roots``, or ``network_origins``.
            value (str): Root token/path or origin glob to update.
            add (bool): Add when true; remove when false.
            scope (PolicyScope): Workspace persistence or process-local lifetime.

        Returns:
            bool: Whether the configured collection changed.

        Raises:
            ValueError: If ``name`` is unsupported.
        """
        if name not in {"readable_roots", "writable_roots", "network_origins"}:
            raise ValueError(f"Unknown collection permission limit '{name}'.")
        if name == "network_origins" or value in {"workspace", "loop-temp", "system-temp"}:
            normalized = value
        else:
            path = Path(value)
            if not path.is_absolute() and self._workspace_root is not None:
                path = self._workspace_root / path
            normalized = canonical_path(path)
        source = (
            self._configuration.limits
            if scope is PolicyScope.WORKSPACE
            else self.effective_configuration.limits
        )
        values = list(getattr(source, name))
        if add:
            if normalized in values:
                return False
            if name == "network_origins" and values == ["*"] and normalized != "*":
                values.clear()
            values.append(normalized)
        else:
            if normalized not in values:
                return False
            values.remove(normalized)
        if scope is PolicyScope.WORKSPACE:
            updated = self._configuration.model_copy(deep=True)
            updated.limits = updated.limits.model_copy(update={name: tuple(values)})
            self._replace_configuration(updated)
        else:
            self._session_overrides.limits = self._session_overrides.limits.model_copy(
                update={name: tuple(values)}
            )
        self._audit_policy_change(
            "permission.limit_value_updated",
            scope,
            limit=name,
            operation="add" if add else "remove",
        )
        return True

    def reset_limit(self, name: str, *, scope: PolicyScope = PolicyScope.SESSION) -> bool:
        """Reset one limit to its inherited or application value.

        Args:
            name (str): Field in ``PolicyLimits`` to reset.
            scope (PolicyScope): Workspace bootstrap or session inheritance target.

        Returns:
            bool: Whether the selected layer changed.

        Raises:
            ValueError: If ``name`` is not a policy limit.
        """
        if name not in _LIMIT_NAMES:
            raise ValueError(f"Unknown permission limit '{name}'.")
        if scope is PolicyScope.SESSION:
            if getattr(self._session_overrides.limits, name) is None:
                return False
            self._session_overrides.limits = self._session_overrides.limits.model_copy(
                update={name: None}
            )
            self._audit_policy_change("permission.limit_reset", scope, limit=name)
            return True
        default = getattr(PolicyLimits(), name)
        if getattr(self._configuration.limits, name) == default:
            return False
        updated = self._configuration.model_copy(deep=True)
        updated.limits = updated.limits.model_copy(update={name: default})
        self._replace_configuration(updated)
        self._audit_policy_change("permission.limit_reset", scope, limit=name)
        return True

    def reset_session(self) -> bool:
        """Clear every process-local default, limit, and rule override.

        Returns:
            bool: Whether any session policy state was cleared.
        """
        changed = self._session_overrides != SessionPolicyOverrides()
        self._session_overrides = SessionPolicyOverrides()
        self._host_runtime_bindings.clear()
        if changed:
            self._audit_policy_change("permission.session_reset", PolicyScope.SESSION)
        return changed

    def reload(self) -> PermissionLoadResult:
        """Reload the workspace policy according to the configured failure policy.

        Returns:
            PermissionLoadResult: Whether new policy loaded or the active policy was retained.

        Raises:
            PermissionConfigurationError: If loading fails in strict mode.
            ShutdownRequested: If the user exits interactive recovery.
        """
        result = self._load_configuration(retain_on_failure=True)
        self._load_user_configuration()
        return result

    def reset_configuration(self) -> Path | None:
        """Archive an invalid workspace policy and replace it with supervised defaults.

        Returns:
            Path | None: Backup path for the invalid policy, or None when no local file existed.

        Raises:
            ValueError: If this manager has no configuration path.
            OSError: If the invalid policy cannot be archived or defaults cannot be persisted.
        """
        if self._configuration_path is None:
            raise ValueError("An in-memory PermissionManager cannot reset configuration.")
        backup_path = None
        if self._configuration_path.exists():
            timestamp = local_now().strftime("%Y%m%dT%H%M%S%f%z")
            backup_path = self._configuration_path.with_name(
                f"{self._configuration_path.name}.{timestamp}.bak"
            )
            self._configuration_path.replace(backup_path)
        configuration = PermissionConfiguration()
        self._persist(configuration)
        self._configuration = configuration
        self._audit_policy_change("permission.configuration_reset", PolicyScope.WORKSPACE)
        return backup_path

    def explain(self, tool: str, action: Action, resource: str) -> PolicyDecision:
        """Evaluate one concrete operation without prompting or recording.

        Args:
            tool (str): Registered tool identity to evaluate.
            action (Action): Typed action to evaluate.
            resource (str): Path, URL, command, or session identifier.

        Returns:
            PolicyDecision: Effective policy decision and determining sources.
        """
        if action in _READ_ACTIONS | _WRITE_ACTIONS:
            path = Path(resource)
            if not path.is_absolute() and self._workspace_root is not None:
                path = self._workspace_root / path
            target = FileTarget(path=canonical_path(path))
        elif action is Action.NETWORK_REQUEST:
            parsed = urlsplit(resource)
            if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
                raise ValueError("Network explanation requires an absolute HTTP(S) URL.")
            origin = f"{parsed.scheme}://{parsed.hostname}"
            if parsed.port is not None:
                origin += f":{parsed.port}"
            target = NetworkTarget(url=resource, origin=origin)
        elif action is Action.PROCESS_EXECUTE:
            argv = tuple(shlex.split(resource))
            if not argv:
                raise ValueError("Process explanation requires a non-empty command line.")
            target = ProcessTarget(
                argv=argv,
                cwd=str(self._workspace_root or Path.cwd()),
                boundary=ProcessBoundary.HOST,
            )
        else:
            target = SessionTarget(identifier=resource)
        return self.evaluate((Operation(tool_id=tool, action=action, target=target),))

    def save(self) -> None:
        """Persist the complete policy atomically.

        Raises:
            ValueError: If this manager has no configuration path.
            OSError: If the configuration cannot be written.
        """
        self._persist(self._configuration)

    def _persist(self, configuration: PermissionConfiguration) -> None:
        if self._configuration_path is None:
            raise ValueError("An in-memory PermissionManager cannot persist configuration.")
        self._persist_configuration(self._configuration_path, configuration)

    @staticmethod
    def _persist_configuration(
        path: Path,
        configuration: PermissionConfiguration | UserPermissionConfiguration,
    ) -> None:
        """Atomically persist one validated permission configuration to its owning path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = configuration.model_dump(mode="json")
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(yaml.safe_dump(payload, sort_keys=False), "utf-8")
        temporary_path.replace(path)

    def _replace_configuration(self, configuration: PermissionConfiguration) -> None:
        """Persist and activate one workspace policy transactionally."""
        self._persist(configuration)
        self._configuration = configuration

    def _replace_user_configuration(self, configuration: UserPermissionConfiguration) -> None:
        """Persist and activate one user-wide remembered-approval policy transactionally."""
        configuration = UserPermissionConfiguration.model_validate(configuration.model_dump())
        self._persist_configuration(self._user_configuration_path, configuration)
        self._user_configuration = configuration

    def _rules_for_scope(self, scope: PolicyScope) -> tuple[PermissionRule, ...]:
        """Return deep-copied rules from one exact policy layer."""
        rules = (
            self._configuration.rules
            if scope is PolicyScope.WORKSPACE
            else self._session_overrides.rules
        )
        return tuple(rule.model_copy(deep=True) for rule in rules)

    def _defaults_for_scope(self, scope: PolicyScope) -> dict[Action, Decision]:
        """Return a deep-copied default map from one exact policy layer."""
        defaults = (
            self._configuration.defaults
            if scope is PolicyScope.WORKSPACE
            else self._session_overrides.defaults
        )
        return dict(defaults)

    @staticmethod
    def _rules_from_preset(
        preset: PermissionPreset,
        scope: PolicyScope,
    ) -> tuple[PermissionRule, ...]:
        """Materialize one artifact into collision-resistant, provenance-bearing policy rules."""
        metadata = preset.metadata
        return tuple(
            PermissionRule(
                id=(f"preset:{scope.value}:{metadata.id}:{metadata.revision}:{rule.id}"),
                decision=rule.decision,
                description=rule.description,
                tool=rule.tool,
                action=rule.action,
                resource=rule.resource,
                source=PresetSource(
                    preset_id=metadata.id,
                    revision=metadata.revision,
                    content_hash=preset.content_hash,
                    rule_id=rule.id,
                ),
            )
            for rule in preset.rules
        )

    def _scope_revision(self, scope: PolicyScope) -> str:
        """Return a semantic revision of the exact layer a replacement would overwrite."""
        payload = (
            self._configuration.model_dump(mode="json")
            if scope is PolicyScope.WORKSPACE
            else self._session_overrides.model_dump(mode="json")
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{sha256_digest(encoded)}"

    def describe(self, view: str = "all") -> str:
        """Return a user-facing summary of selected policy layers.

        Args:
            view (str): ``all``, ``workspace``, ``session``, or ``effective``.

        Returns:
            str: Requested defaults, limits, rules, and their policy sources.

        Raises:
            ValueError: If ``view`` is not supported.
        """
        if view not in {"all", "workspace", "session", "effective"}:
            raise ValueError(f"Unknown permission policy view '{view}'.")
        path = str(self._configuration_path) if self._configuration_path else "in memory"
        lines = [f"Permission policy: {path}"]
        if view in {"all", "workspace"}:
            lines.extend(self._describe_configuration("Workspace policy", self._configuration))
        if view in {"all", "session"}:
            lines.extend(self._describe_session_overrides())
        if view in {"all", "effective"}:
            lines.extend(
                self._describe_configuration(
                    "Effective policy", self.effective_configuration, annotate_sources=True
                )
            )
        lines.append(
            "Precedence: protected boundary > deny rule > exact allow rule > ask rule > "
            "allow rule > default"
        )
        return "\n".join(lines)

    def _describe_configuration(
        self,
        heading: str,
        configuration: PermissionConfiguration,
        *,
        annotate_sources: bool = False,
    ) -> list[str]:
        """Return display lines for one complete configuration."""
        lines = [f"{heading}:", "  Defaults:"]
        for action in Action:
            source = (
                (" [session]" if action in self._session_overrides.defaults else " [workspace]")
                if annotate_sources
                else ""
            )
            decision = configuration.defaults.get(action, Decision.DENY)
            lines.append(f"    {action.value}: {decision.value}{source}")
        lines.append("  Limits:")
        for name in _LIMIT_NAMES:
            value = getattr(configuration.limits, name)
            rendered = (", ".join(value) or "none") if isinstance(value, tuple) else str(value)
            source = (
                (
                    " [session]"
                    if getattr(self._session_overrides.limits, name) is not None
                    else " [workspace]"
                )
                if annotate_sources
                else ""
            )
            lines.append(f"    {name}: {rendered}{source}")
        rules = configuration.rules
        lines.append("  Rules:" if rules else "  Rules: none")
        if rules:
            session_ids = {rule.id for rule in self._session_overrides.rules}
            lines.extend(
                f"  {self._describe_rule(rule)}"
                f" [{'session' if rule.id in session_ids else 'workspace'}]"
                if annotate_sources
                else f"  {self._describe_rule(rule)}"
                for rule in rules
            )
        return lines

    def _describe_session_overrides(self) -> list[str]:
        """Return display lines for process-local policy differences."""
        overrides = self._session_overrides
        lines = ["Session overrides:"]
        if (
            not overrides.defaults
            and overrides.limits == PolicyLimitOverrides()
            and not overrides.rules
        ):
            return [*lines, "  none"]
        lines.append("  Defaults:")
        lines.extend(
            f"    {action.value}: {decision.value}"
            for action, decision in sorted(
                overrides.defaults.items(), key=lambda item: item[0].value
            )
        )
        if not overrides.defaults:
            lines.append("    none")
        lines.append("  Limits:")
        limit_count = 0
        for name in _LIMIT_NAMES:
            value = getattr(overrides.limits, name)
            if value is None:
                continue
            rendered = (", ".join(value) or "none") if isinstance(value, tuple) else str(value)
            lines.append(f"    {name}: {rendered}")
            limit_count += 1
        if not limit_count:
            lines.append("    none")
        lines.append("  Rules:" if overrides.rules else "  Rules: none")
        lines.extend(f"  {self._describe_rule(rule)}" for rule in overrides.rules)
        return lines

    @staticmethod
    def _describe_rule(rule: PermissionRule) -> str:
        description = f" — {rule.description}" if rule.description else ""
        source = (
            f" [preset={rule.source.preset_id}@{rule.source.revision} "
            f"hash={rule.source.content_hash}]"
            if rule.source is not None
            else ""
        )
        target = rule.target.model_dump_json() if rule.target is not None else None
        tool_match = " tool_match=exact" if rule.tool_exact else ""
        return (
            f"{rule.id} {rule.decision.value} tool={rule.tool}{tool_match} "
            f"action={rule.action.value if rule.action else '*'} "
            f"resource={target or rule.resource or '*'}{description}{source}"
        )

    def _read_configuration(self) -> PermissionConfiguration:
        """Read and validate the configured policy without changing active state."""
        if self._configuration_path is None or not self._configuration_path.exists():
            return PermissionConfiguration()
        try:
            payload = yaml.safe_load(self._configuration_path.read_text("utf-8"))
            return PermissionConfiguration.model_validate(payload or {})
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            raise PermissionConfigurationError(self._configuration_path, str(exc)) from exc

    def _read_user_configuration(self) -> UserPermissionConfiguration:
        """Read and validate the user-wide approval policy without changing active state."""
        if self._user_configuration_path is None or not self._user_configuration_path.exists():
            return UserPermissionConfiguration()
        try:
            payload = yaml.safe_load(self._user_configuration_path.read_text("utf-8"))
            return UserPermissionConfiguration.model_validate(payload or {})
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            raise PermissionConfigurationError(self._user_configuration_path, str(exc)) from exc

    def _load_user_configuration(self) -> None:
        """Load user-wide approvals and fail closed if their file is invalid."""
        try:
            self._user_configuration = self._read_user_configuration()
        except PermissionConfigurationError as exc:
            if self._load_policy is PermissionLoadPolicy.ERROR:
                raise
            self._user_configuration = UserPermissionConfiguration()
            self._report_user_configuration_error(exc)

    def _load_configuration(
        self,
        *,
        retain_on_failure: bool = False,
    ) -> PermissionLoadResult:
        """Load configuration and completely handle the selected failure policy."""
        while True:
            try:
                configuration = self._read_configuration()
            except PermissionConfigurationError as exc:
                if self._load_policy is PermissionLoadPolicy.ERROR:
                    raise
                if self._load_policy is PermissionLoadPolicy.AUTO:
                    self._report_configuration_error(exc)
                    if retain_on_failure:
                        return PermissionLoadResult.RETAINED
                    self._configuration = PermissionConfiguration()
                    return PermissionLoadResult.DEFAULTED
                if self._interaction is None:
                    raise
                choice = self._recover_configuration_interactively(
                    exc, retain_on_failure=retain_on_failure
                )
                if choice is PermissionLoadResult.LOADED:
                    continue
                return choice
            self._configuration = configuration
            return PermissionLoadResult.LOADED

    def _recover_configuration_interactively(
        self,
        error: PermissionConfigurationError,
        *,
        retain_on_failure: bool,
    ) -> PermissionLoadResult:
        """Report one failure and return the user's selected recovery outcome."""
        problem = Problem.from_exception(
            error,
            code="permission.configuration_invalid",
            title="Invalid permission policy",
            operation="load_permission_policy",
            metadata={"path": error.path},
        )
        log_problem(_LOGGER, problem, error)
        self._interaction.report(problem)
        continue_description = (
            "Keep the current permission policy"
            if retain_on_failure
            else "Use supervised defaults for this session"
        )
        choice = self._interaction.prompt(
            "Permission policy recovery:",
            exit_commands=(),
            choices={
                "retry": "Retry after fixing the permission file",
                "continue": continue_description,
                "reset": "Archive the invalid file and reset to supervised defaults",
                "exit": "Exit application",
            },
            index={"retry": "R", "continue": "S", "reset": "A", "exit": "Q"},
        )
        if choice == "retry":
            return PermissionLoadResult.LOADED
        if choice == "reset":
            backup_path = self.reset_configuration()
            self._interaction.info(
                "Reset permission policy to supervised defaults; "
                f"archived invalid file at {backup_path}."
            )
            return PermissionLoadResult.DEFAULTED
        if choice == "exit" or choice is False:
            raise ShutdownRequested()
        if retain_on_failure:
            self._interaction.warning("Keeping the current permission policy.")
            return PermissionLoadResult.RETAINED
        self._configuration = PermissionConfiguration()
        self._interaction.warning("Using supervised permission defaults for this session.")
        return PermissionLoadResult.DEFAULTED

    def _report_configuration_error(self, error: PermissionConfigurationError) -> None:
        """Report an automatically recovered configuration failure."""
        problem = Problem.from_exception(
            error,
            code="permission.configuration_invalid",
            title="Invalid permission policy",
            operation="load_permission_policy",
            metadata={"path": error.path},
        )
        log_problem(_LOGGER, problem, error)
        if self._interaction is not None:
            self._interaction.report(problem)
            self._interaction.warning("Recovered according to the automatic permission policy.")
        else:
            _LOGGER.warning("Recovered according to the automatic permission policy")

    def _report_user_configuration_error(self, error: PermissionConfigurationError) -> None:
        """Report an invalid user approval policy while retaining fail-closed behavior."""
        problem = Problem.from_exception(
            error,
            code="permission.user_configuration_invalid",
            title="Invalid user permission policy",
            operation="load_user_permission_policy",
            metadata={"path": error.path},
        )
        log_problem(_LOGGER, problem, error)
        if self._interaction is not None:
            self._interaction.report(problem)
            self._interaction.warning("Ignoring user-wide approvals until the policy is fixed.")
        else:
            _LOGGER.warning("Ignoring invalid user-wide approval policy")

    def _load_presets(self) -> tuple[PermissionPreset, ...]:
        """Load presets and completely handle invalid catalog artifacts."""
        presets, failures = PermissionPreset.load_builtin_presets()
        if not failures:
            return presets
        error = PermissionPresetError(failures)
        if self._load_policy is PermissionLoadPolicy.ERROR:
            raise error
        for failure in failures:
            message = f"Excluded invalid permission preset at {failure.path}: {failure.message}"
            if self._interaction is not None:
                self._interaction.warning(message)
            else:
                _LOGGER.warning(
                    "Excluded invalid permission preset",
                    extra={"error.type": "permission.preset_invalid"},
                )
        return presets

    def _evaluate_operation(self, operation: Operation) -> PolicyDecision:
        boundary = self._boundary_decision(operation)
        if boundary is not None:
            return boundary
        matching = [
            (ApprovalChoice.USER.value, rule)
            for rule in self._user_configuration.rules
            if self._matches(rule, operation)
        ]
        matching.extend(
            (ApprovalChoice.WORKSPACE.value, rule)
            for rule in self._configuration.rules
            if self._matches(rule, operation)
        )
        matching.extend(
            (ApprovalChoice.SESSION.value, rule)
            for rule in self._session_overrides.rules
            if self._matches(rule, operation)
        )
        exact_allow = tuple(
            (scope, rule)
            for scope, rule in matching
            if rule.decision is Decision.ALLOW and rule.target is not None
        )
        ordered = (
            (
                Decision.DENY,
                tuple((scope, rule) for scope, rule in matching if rule.decision is Decision.DENY),
            ),
            (Decision.ALLOW, exact_allow),
            (
                Decision.ASK,
                tuple((scope, rule) for scope, rule in matching if rule.decision is Decision.ASK),
            ),
            (
                Decision.ALLOW,
                tuple(
                    (scope, rule)
                    for scope, rule in matching
                    if rule.decision is Decision.ALLOW and rule.target is None
                ),
            ),
        )
        for decision, determining in ordered:
            if determining:
                return PolicyDecision(
                    decision=decision,
                    reason=f"Matched explicit {decision.value} policy rule(s).",
                    sources=tuple(f"rule:{scope}:{rule.id}" for scope, rule in determining),
                )
        session_decision = self._session_overrides.defaults.get(operation.action)
        decision = session_decision or self._configuration.defaults.get(
            operation.action, Decision.DENY
        )
        scope = PolicyScope.SESSION if session_decision is not None else PolicyScope.WORKSPACE
        return PolicyDecision(
            decision=decision,
            reason=f"The {operation.action.value} default is {decision.value}.",
            sources=(f"default:{scope.value}:{operation.action.value}",),
        )

    def _boundary_decision(self, operation: Operation) -> PolicyDecision | None:
        target = operation.target
        limits = self.effective_configuration.limits
        if isinstance(target, FileTarget):
            if self._is_protected_path(
                target.path,
                mutation=operation.action in _WRITE_ACTIONS,
                recursive=target.recursive,
            ):
                return self._boundary_denial("protected_path")
            roots = (
                limits.readable_roots
                if operation.action in _READ_ACTIONS
                else limits.writable_roots
            )
            if not self._in_roots(target.path, roots):
                limit_name = (
                    "readable_roots" if operation.action in _READ_ACTIONS else "writable_roots"
                )
                return self._limit_denial("filesystem_root", limit_name)
        elif isinstance(target, NetworkTarget):
            if not any(fnmatchcase(target.origin, pattern) for pattern in limits.network_origins):
                return self._limit_denial("network_origin", "network_origins")
            if limits.deny_private_networks and self._is_private_network(target):
                return self._limit_denial("private_network", "deny_private_networks")
        elif (
            isinstance(target, ProcessTarget)
            and target.boundary is ProcessBoundary.HOST
            and not limits.allow_host_processes
        ):
            return self._limit_denial("host_process", "allow_host_processes")
        return None

    def _limit_denial(self, name: str, field: str) -> PolicyDecision:
        """Return a denial attributed to the effective limit layer."""
        scope = (
            PolicyScope.SESSION
            if getattr(self._session_overrides.limits, field) is not None
            else PolicyScope.WORKSPACE
        )
        return PolicyDecision(
            decision=Decision.DENY,
            reason=f"The {scope.value} {name} boundary denied this operation.",
            sources=(f"limit:{scope.value}:{field}",),
        )

    @staticmethod
    def _boundary_denial(name: str) -> PolicyDecision:
        return PolicyDecision(
            decision=Decision.DENY,
            reason=f"The non-overridable {name} boundary denied this operation.",
            sources=(f"boundary:{name}",),
        )

    @staticmethod
    def _matches(rule: PermissionRule, operation: Operation) -> bool:
        exact_target = (
            PermissionManager._approval_target(operation.target) == rule.target
            if rule.target is not None
            else True
        )
        return (
            (
                operation.tool_id == rule.tool
                if rule.tool_exact
                else fnmatchcase(operation.tool_id, rule.tool)
            )
            and (rule.action is None or rule.action is operation.action)
            and exact_target
            and (
                rule.resource is None
                or operation.resource is not None
                and fnmatchcase(operation.resource, rule.resource)
            )
        )

    def _in_roots(self, resource: str, configured_roots: tuple[str, ...]) -> bool:
        path = Path(resource).resolve()
        for configured in configured_roots:
            root = self._root_path(configured)
            if root is None:
                continue
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _root_path(self, configured: str) -> Path | None:
        """Resolve one configured filesystem root token or absolute path."""
        if configured == "workspace":
            return self._workspace_root.resolve() if self._workspace_root else None
        if configured == "loop-temp":
            return self._temporary_path
        if configured == "system-temp":
            return Path(tempfile.gettempdir()).resolve()
        path = Path(configured)
        if path.is_absolute():
            return path.resolve()
        return (self._workspace_root / path).resolve() if self._workspace_root is not None else None

    def check_boundaries(self, operations: Operations) -> PolicyDecision | None:
        """Return the first non-overridable boundary denial for planned effects.

        This check performs no policy matching or interactive approval. It lets phased planners
        reject an unsafe eventual effect before prerequisite filesystem inspection.

        Args:
            operations (Operations): Future typed effects to check.

        Returns:
            PolicyDecision | None: First boundary denial, or ``None`` when all effects are inside
                the fixed safety boundaries.
        """
        for operation in operations:
            denial = self._boundary_decision(operation)
            if denial is not None:
                return denial
        return None

    def _is_protected_path(self, resource: str, *, mutation: bool, recursive: bool = False) -> bool:
        if self._workspace_root is None:
            return False
        requested = Path(resource)
        path = requested.parent.resolve() / requested.name if mutation else requested.resolve()
        protected = (
            self._workspace_root / constants.APP_DIRECTORY,
            self._workspace_root / constants.GIT_DIRECTORY,
        )
        if mutation:
            protected += (
                self._workspace_root / constants.GIT_IGNORE_FILENAME,
                self._workspace_root / constants.AGENT_IGNORE_FILENAME,
            )
        for root in protected:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                if not (mutation and recursive):
                    continue
            try:
                root.relative_to(path)
                return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _is_private_network(target: NetworkTarget) -> bool:
        hostname = urlsplit(target.url).hostname
        if hostname is None or hostname.casefold() == "localhost":
            return True
        addresses = target.addresses
        try:
            addresses = addresses or (str(ipaddress.ip_address(hostname)),)
        except ValueError:
            return False
        return not addresses or any(
            not ipaddress.ip_address(address).is_global for address in addresses
        )

    def _prompt(self, operations: Operations) -> str:
        lines = ["Agent requests approval for the following operations:"]
        for operation in operations:
            resource = self._display_resource(operation)
            target = f" on '{resource}'" if resource else ""
            reason = f" — {operation.reason}" if operation.reason else ""
            lines.append(
                f"{operation.action.icon} {operation.tool_id}: "
                f"{operation.action.value}{target}{reason}"
            )
        return "\n".join(lines)

    def _display_resource(self, operation: Operation) -> str | None:
        """Return a workspace-friendly display name for the operation target.

        File paths are shown relative to the workspace root. Process targets include
        their working directory and quote argument boundaries after normalizing
        recognized absolute paths through ``VirtualPath``. Paths outside the
        configured virtual roots remain visible rather than being redacted.

        Args:
            operation (Operation): The operation whose target should be displayed.

        Returns:
            str | None: Cleaned display string, or the raw resource when no
                workspace is configured or the target is not a file or process.
        """
        if operation.resource is None or self._workspace_root is None:
            return operation.resource
        if isinstance(operation.target, FileTarget):
            try:
                relative = Path(operation.target.path).relative_to(self._workspace_root)
                return (
                    f"workspace root: {self._workspace_root}"
                    if relative == Path(".")
                    else str(relative)
                )
            except ValueError:
                return operation.resource
        if isinstance(operation.target, ProcessTarget):
            virtual_paths = VirtualPath(
                workspace=self._workspace_root,
                temporary_directory=self._temporary_path,
            )

            def display_path(path: str) -> str:
                """Render a recognized virtual path or retain an external path verbatim."""
                rendered = virtual_paths.display(path)
                return path if rendered == VirtualPath.EXTERNAL else rendered

            cleaned_argv = tuple(
                display_path(arg) if Path(arg).is_absolute() else arg
                for arg in operation.target.argv
            )
            cwd = display_path(operation.target.cwd)
            return f"{shlex.join(cleaned_argv)} (cwd: {cwd})"
        return operation.resource

    def _audit(self, result: AuthorizationResult) -> None:
        """Append one independently durable, timestamped permission decision."""
        payload = result.model_dump(mode="json")
        self._append_audit("permission.decided", payload)

    def _append_audit(self, event_name: str, payload: dict[str, object]) -> None:
        """Append one bounded audit payload without affecting policy behavior."""
        if self._audit_store is None or self._workspace_id is None:
            return
        try:
            self._audit_store.append(self._workspace_id, event_name, payload)
        except (OSError, sqlite3.Error) as error:
            # The durable session recorder remains authoritative for application behavior;
            # diagnostic JSONL availability must not change an authorization outcome.
            telemetry_error(
                "permission.audit_write_failed",
                error_type="permission.audit_write_failed",
                exception=error,
                component="permission_manager",
            )

    def _audit_policy_change(
        self,
        event_name: str,
        scope: PolicyScope,
        **attributes: object,
    ) -> None:
        """Record one minimized permission-policy mutation."""
        telemetry_audit(event_name, scope=scope.value, **attributes)
        self._append_audit(event_name, {"scope": scope.value, **attributes})
