"""Load, evaluate, persist, approve, and record local operation policies."""

# Pylint cannot infer mutable attributes declared by Pydantic models.
# pylint: disable=no-member

from __future__ import annotations

import ipaddress
import json
import logging
import shlex
import tempfile
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import yaml

from .. import constants
from ..errors import Problem, log_problem
from ..telemetry import telemetry_audit, telemetry_error, telemetry_trace_event
from ..utils import ShutdownRequested, canonical_path, local_now, sha256_digest, utc_now
from .models import (
    Action,
    ApprovalChoice,
    AuthorizationResult,
    Decision,
    FileTarget,
    NetworkTarget,
    Operation,
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
    SessionPolicyOverrides,
    SessionTarget,
)

if TYPE_CHECKING:
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


class PermissionManager:
    """Authorize complete operation plans using limits, policy rules, and approval.

    Args:
        working_directory (Path | str | None): Workspace used to resolve policy root tokens.
        configuration_path (Path | str | None): YAML policy path. Defaults to
            <working_directory>/.loop/permissions.yaml when a workspace is supplied.
        interaction (Interaction | None): User interaction used for approval prompts.
        recorder (PermissionRecorder | None): Default sink for authorization observations.
        configuration (PermissionConfiguration | None): Explicit policy instead of the local file.
        presets (Iterable[PermissionPreset] | None): Additional selectable presets alongside
            the built-in catalog. Duplicate identifiers are rejected.
        load_policy (PermissionLoadPolicy | None): Artifact failure behavior. Defaults to
            interactive recovery when an interaction is available and strict errors otherwise.

    Raises:
        PermissionConfigurationError: If a policy is invalid in strict mode.
        PermissionPresetError: If built-in presets are invalid in strict mode.
        ShutdownRequested: If the user exits interactive configuration recovery.
        ValueError: If interactive recovery is selected without an interaction or preset
            identifiers are duplicated.
    """

    _working_directory: Path | None
    _configuration_path: Path | None
    _interaction: Interaction | None
    _recorder: PermissionRecorder | None
    _configuration: PermissionConfiguration
    _load_policy: PermissionLoadPolicy
    _session_overrides: SessionPolicyOverrides
    _temporary_directory: tempfile.TemporaryDirectory[str]
    _temporary_path: Path
    _presets: dict[str, PermissionPreset]

    def __init__(
        self,
        working_directory: Path | str | None = None,
        *,
        configuration_path: Path | str | None = None,
        interaction: Interaction | None = None,
        recorder: PermissionRecorder | None = None,
        configuration: PermissionConfiguration | None = None,
        presets: Iterable[PermissionPreset] | None = None,
        load_policy: PermissionLoadPolicy | None = None,
    ) -> None:
        self._working_directory = (
            Path(working_directory).resolve() if working_directory is not None else None
        )
        self._temporary_directory = tempfile.TemporaryDirectory(  # pylint: disable=consider-using-with
            prefix=constants.TEMPORARY_DIRECTORY_PREFIX
        )
        self._temporary_path = Path(self._temporary_directory.name).resolve()
        self._configuration_path = (
            Path(configuration_path)
            if configuration_path is not None
            else self._working_directory / constants.APP_DIRECTORY / constants.PERMISSIONS_FILENAME
            if self._working_directory is not None
            else None
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
        self._session_overrides = SessionPolicyOverrides()
        builtin_presets = self._load_presets()
        catalog = (*builtin_presets, *(presets or ()))
        identifiers = [preset.metadata.id for preset in catalog]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Permission preset identifiers must be unique.")
        self._presets = {preset.metadata.id: preset for preset in catalog}

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
        operations: tuple[Operation, ...],
        *,
        interaction: Interaction | None = None,
        recorder: PermissionRecorder | None = None,
    ) -> AuthorizationResult:
        """Evaluate and approve one complete operation set atomically.

        Args:
            operations (tuple[Operation, ...]): Complete normalized effects of one tool call.
            interaction (Interaction | None): Invocation interaction overriding the default.
            recorder (PermissionRecorder | None): Invocation recorder overriding the default.

        Returns:
            AuthorizationResult: Policy, prompt, and effective result for the complete set.
        """
        policy = self.evaluate(operations)
        active_interaction = interaction if interaction is not None else self._interaction
        prompt = None
        prompted = False
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
                prompted = True
                prompt = self._prompt(operations)
                selected = self.request_permission(
                    prompt,
                    interaction=active_interaction,
                )
                approval_choice = (
                    selected
                    if isinstance(selected, ApprovalChoice)
                    else ApprovalChoice.ONCE
                    if selected is True
                    else ApprovalChoice.DENY
                )
                if approval_choice is ApprovalChoice.WORKSPACE and self._configuration_path is None:
                    approval_choice = ApprovalChoice.DENY
                    decision = Decision.DENY
                    reason = "Workspace approval is unavailable without a workspace policy path."
                if approval_choice is not ApprovalChoice.DENY:
                    decision = Decision.ALLOW
                    reason = f"Approved by the user with {approval_choice.value} scope."
                    if approval_choice in {ApprovalChoice.SESSION, ApprovalChoice.WORKSPACE}:
                        scope = PolicyScope(approval_choice.value)
                        try:
                            installed_rule_ids = self._remember_approval(operations, scope=scope)
                        except OSError as exc:
                            decision = Decision.DENY
                            reason = f"Could not persist the approved workspace policy: {exc}"
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
            prompted=prompted,
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

    def request_permission(
        self,
        prompt: str,
        *,
        interaction: Interaction | None = None,
    ) -> ApprovalChoice:
        """Request one scoped approval through a generic user interaction.

        Args:
            prompt (str): Complete operation description to display.
            interaction (Interaction | None): Invocation interaction overriding the default.

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
        if self._configuration_path is not None:
            choices[ApprovalChoice.WORKSPACE] = (
                "Allow in this workspace (save to .loop/permissions.yaml)"
            )
            index[ApprovalChoice.WORKSPACE] = "W"
        active_interaction = interaction if interaction is not None else self._interaction
        selected = active_interaction.prompt(
            prompt,
            exit_commands=None,
            choices=choices,
            index=index,
        )
        return selected if isinstance(selected, ApprovalChoice) else ApprovalChoice.DENY

    def _remember_approval(
        self,
        operations: tuple[Operation, ...],
        *,
        scope: PolicyScope,
    ) -> tuple[str, ...]:
        """Install exact, deduplicated allow rules for one approved operation set."""
        existing = (*self._configuration.rules, *self._session_overrides.rules)
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
        if scope is PolicyScope.WORKSPACE:
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

    def evaluate(self, operations: tuple[Operation, ...]) -> PolicyDecision:
        """Evaluate a complete operation set without prompting or recording.

        Args:
            operations (tuple[Operation, ...]): Complete normalized effects to evaluate.

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
            if not path.is_absolute() and self._working_directory is not None:
                path = self._working_directory / path
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
        return self._load_configuration(retain_on_failure=True)

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
            if not path.is_absolute() and self._working_directory is not None:
                path = self._working_directory / path
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
                cwd=str(self._working_directory or Path.cwd()),
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
        self._configuration_path.parent.mkdir(parents=True, exist_ok=True)
        payload = configuration.model_dump(mode="json")
        temporary_path = self._configuration_path.with_suffix(
            self._configuration_path.suffix + ".tmp"
        )
        temporary_path.write_text(yaml.safe_dump(payload, sort_keys=False), "utf-8")
        temporary_path.replace(self._configuration_path)

    def _replace_configuration(self, configuration: PermissionConfiguration) -> None:
        """Persist and activate one workspace policy transactionally."""
        self._persist(configuration)
        self._configuration = configuration

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
                "exit": "Exit Loop",
            },
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
            (PolicyScope.WORKSPACE, rule)
            for rule in self._configuration.rules
            if self._matches(rule, operation)
        ]
        matching.extend(
            (PolicyScope.SESSION, rule)
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
                    sources=tuple(f"rule:{scope.value}:{rule.id}" for scope, rule in determining),
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
            if self._is_protected_path(target.path, mutation=operation.action in _WRITE_ACTIONS):
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
            return self._working_directory.resolve() if self._working_directory else None
        if configured == "loop-temp":
            return self._temporary_path
        if configured == "system-temp":
            return Path(tempfile.gettempdir()).resolve()
        path = Path(configured)
        if path.is_absolute():
            return path.resolve()
        return (
            (self._working_directory / path).resolve()
            if self._working_directory is not None
            else None
        )

    def _is_protected_path(self, resource: str, *, mutation: bool) -> bool:
        if self._working_directory is None:
            return False
        path = Path(resource)
        protected = (
            self._working_directory / constants.APP_DIRECTORY,
            self._working_directory / constants.GIT_DIRECTORY,
        )
        if mutation:
            protected += (
                self._working_directory / constants.GIT_IGNORE_FILENAME,
                self._working_directory / constants.AGENT_IGNORE_FILENAME,
            )
        for root in protected:
            try:
                path.relative_to(root)
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

    def _prompt(self, operations: tuple[Operation, ...]) -> str:
        lines = ["Agent requests approval for the following operations:"]
        for operation in operations:
            resource = self._display_resource(operation)
            target = f" on '{resource}'" if resource else ""
            reason = f" — {operation.reason}" if operation.reason else ""
            lines.append(
                f"{operation.action.icon} {operation.tool_id}: "
                f"{operation.action.value}{target}{reason}"
            )
        lines.append("Proceed?")
        return "\n".join(lines)

    def _display_resource(self, operation: Operation) -> str | None:
        if (
            operation.resource is None
            or self._working_directory is None
            or not isinstance(operation.target, FileTarget)
        ):
            return operation.resource
        try:
            relative = Path(operation.target.path).relative_to(self._working_directory)
            return (
                f"workspace root: {self._working_directory}"
                if relative == Path(".")
                else str(relative)
            )
        except ValueError:
            return operation.resource

    def _audit(self, result: AuthorizationResult) -> None:
        """Append one independently durable, timestamped permission decision."""
        payload = result.model_dump(mode="json")
        self._append_audit("permission.decided", payload)

    def _append_audit(self, event_name: str, payload: dict[str, object]) -> None:
        """Append one timestamped permission audit payload without affecting policy behavior."""
        if self._configuration_path is None:
            return
        audit_path = self._configuration_path.with_name(constants.PERMISSIONS_AUDIT_FILENAME)
        record = {
            **payload,
            "audit_schema_version": 1,
            "event_name": event_name,
            "timestamp": utc_now().isoformat(),
        }
        try:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as audit:
                audit.write(json.dumps(record, sort_keys=True) + "\n")
            audit_path.chmod(constants.PRIVATE_FILE_MODE)
        except OSError as error:
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
