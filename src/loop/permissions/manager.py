"""Load, evaluate, persist, and audit local tool permissions."""

import json
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .. import constants
from .models import (
    Capability,
    Decision,
    PermissionConfiguration,
    PermissionMode,
    PermissionRequest,
    PermissionResult,
    PermissionRule,
)

if TYPE_CHECKING:
    from ..interaction import Interaction

_READ_CAPABILITIES = {Capability.PURE, Capability.FILESYSTEM_READ, Capability.NETWORK_READ}


class PermissionManager:
    """Authorize tool operations using local rules, modes, and session grants.

    Args:
        working_directory (Path | str | None): Active workspace used for scoped mode decisions.
        configuration_path (Path | str | None): YAML configuration path. Defaults to
            ``<working_directory>/.loop/permissions.yaml`` when a directory is supplied.
        interaction (Interaction | None): User interaction used for approval prompts.
        configuration (PermissionConfiguration | None): Explicit configuration used instead of
            loading the local file.

    Raises:
        OSError: If an existing configuration cannot be read.
        ValueError: If an existing configuration is invalid.
        yaml.YAMLError: If an existing configuration contains invalid YAML.
    """

    _working_directory: Path | None
    _configuration_path: Path | None
    _interaction: Interaction | None
    _configuration: PermissionConfiguration
    _session_rules: list[PermissionRule]

    def __init__(
        self,
        working_directory: Path | str | None = None,
        *,
        configuration_path: Path | str | None = None,
        interaction: Interaction | None = None,
        configuration: PermissionConfiguration | None = None,
    ) -> None:
        self._working_directory = (
            Path(working_directory).resolve() if working_directory is not None else None
        )
        self._configuration_path = (
            Path(configuration_path)
            if configuration_path is not None
            else self._working_directory
            / constants.APP_DIRECTORY
            / constants.PERMISSIONS_FILENAME
            if self._working_directory is not None
            else None
        )
        self._interaction = interaction
        self._configuration = configuration or self._load()
        self._session_rules = []

    @property
    def configuration(self) -> PermissionConfiguration:
        """Return the active configuration.

        Returns:
            PermissionConfiguration: Active mode and persisted rules.
        """
        return self._configuration

    @property
    def configuration_path(self) -> Path | None:
        """Return the local configuration path.

        Returns:
            Path | None: YAML path, or ``None`` for an in-memory manager.
        """
        return self._configuration_path

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
            interaction (Interaction | None): New interaction or ``None`` for headless use.
        """
        self._interaction = interaction

    def authorize(self, request: PermissionRequest) -> PermissionResult:
        """Evaluate and, when required, ask the user about a permission request.

        Args:
            request (PermissionRequest): Normalized operation to authorize.

        Returns:
            PermissionResult: Effective allow or deny result.
        """
        result = self.evaluate(request)
        if result.decision is Decision.ASK:
            if self._interaction is None:
                result = PermissionResult(
                    decision=Decision.DENY,
                    reason="Approval is required but no interactive user is available.",
                    source="headless",
                )
            elif self._interaction.confirm(self._prompt(request), default=False):
                result = PermissionResult(
                    decision=Decision.ALLOW,
                    reason="Approved by the user for this call.",
                    source="user",
                )
            else:
                result = PermissionResult(
                    decision=Decision.DENY,
                    reason="Rejected by the user.",
                    source="user",
                )
        self._audit(request, result)
        return result

    def evaluate(self, request: PermissionRequest) -> PermissionResult:
        """Evaluate a request without prompting or recording it.

        Args:
            request (PermissionRequest): Operation to evaluate.

        Returns:
            PermissionResult: Allow, ask, or deny policy outcome.
        """
        matching = [
            rule
            for rule in (*self._configuration.rules, *self._session_rules)
            if self._matches(rule, request)
        ]
        for decision in (Decision.DENY, Decision.ASK, Decision.ALLOW):
            if any(rule.decision is decision for rule in matching):
                return PermissionResult(
                    decision=decision,
                    reason=f"Matched an explicit {decision.value} rule.",
                    source="rule",
                )
        return self._mode_result(request)

    def set_mode(self, mode: PermissionMode, *, persist: bool = True) -> None:
        """Change the active fallback mode.

        Args:
            mode (PermissionMode): Mode to activate.
            persist (bool): Whether to write the local configuration file.
        """
        self._configuration.mode = mode
        if persist:
            self.save()

    def add_rule(self, rule: PermissionRule, *, persist: bool = True) -> None:
        """Append a permission rule.

        Args:
            rule (PermissionRule): Rule to append.
            persist (bool): Whether to write the local configuration file.
        """
        if persist:
            self._configuration.rules.append(rule)
            self.save()
        else:
            self._session_rules.append(rule)

    def save(self) -> None:
        """Persist the active mode and rules in the local ``.loop`` folder.

        Raises:
            ValueError: If this manager has no configuration path.
            OSError: If the configuration cannot be written.
        """
        if self._configuration_path is None:
            raise ValueError("An in-memory PermissionManager cannot persist configuration.")
        self._configuration_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._configuration.model_dump(mode="json", exclude_defaults=True)
        self._configuration_path.write_text(yaml.safe_dump(payload, sort_keys=False), "utf-8")

    def describe(self) -> str:
        """Return a user-facing summary of the active permission policy.

        Returns:
            str: Mode, configuration path, and configured rules.
        """
        path = str(self._configuration_path) if self._configuration_path else "in memory"
        lines = [f"Permission mode: {self._configuration.mode.value}", f"Configuration: {path}"]
        if not self._configuration.rules:
            lines.append("Rules: none")
        else:
            lines.append("Rules:")
            lines.extend(
                f"  {rule.decision.value} tool={rule.tool} "
                f"capability={rule.capability.value if rule.capability else '*'} "
                f"resource={rule.resource or '*'}"
                for rule in self._configuration.rules
            )
        return "\n".join(lines)

    def _load(self) -> PermissionConfiguration:
        if self._configuration_path is None or not self._configuration_path.exists():
            return PermissionConfiguration()
        payload = yaml.safe_load(self._configuration_path.read_text("utf-8"))
        return PermissionConfiguration.model_validate(payload or {})

    @staticmethod
    def _matches(rule: PermissionRule, request: PermissionRequest) -> bool:
        return (
            fnmatchcase(request.tool_name, rule.tool)
            and (rule.capability is None or rule.capability is request.capability)
            and (
                rule.resource is None
                or request.resource is not None
                and fnmatchcase(request.resource, rule.resource)
            )
        )

    def _mode_result(self, request: PermissionRequest) -> PermissionResult:
        mode = self._configuration.mode
        decision = Decision.ASK
        if mode is PermissionMode.LOCKED_DOWN:
            decision = Decision.DENY
        elif mode is PermissionMode.UNRESTRICTED:
            decision = Decision.ALLOW
        elif mode is PermissionMode.READ_ONLY:
            decision = Decision.ALLOW if request.capability in _READ_CAPABILITIES else Decision.DENY
        elif mode is PermissionMode.WORKSPACE_WRITE:
            if request.capability in _READ_CAPABILITIES:
                decision = Decision.ALLOW
            elif request.capability is Capability.FILESYSTEM_WRITE and self._in_workspace(
                request.resource
            ):
                decision = Decision.ALLOW
        return PermissionResult(
            decision=decision,
            reason=f"Permission mode '{mode.value}' selected {decision.value}.",
            source=f"mode:{mode.value}",
        )

    def _in_workspace(self, resource: str | None) -> bool:
        if resource is None or self._working_directory is None:
            return False
        try:
            Path(resource).resolve().relative_to(self._working_directory)
            return True
        except ValueError:
            return False

    @staticmethod
    def _prompt(request: PermissionRequest) -> str:
        target = f" on '{request.resource}'" if request.resource else ""
        reason = f" {request.reason}" if request.reason else ""
        return (
            f"Agent wants to use '{request.tool_name}' for {request.capability.value}{target}."
            f"{reason} Proceed?"
        )

    def _audit(self, request: PermissionRequest, result: PermissionResult) -> None:
        if self._configuration_path is None:
            return
        audit_path = self._configuration_path.with_name(constants.PERMISSIONS_AUDIT_FILENAME)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "request": request.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
        }
        with audit_path.open("a", encoding="utf-8") as audit:
            audit.write(json.dumps(record, sort_keys=True) + "\n")
