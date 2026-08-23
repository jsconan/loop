"""Expose scoped operation-policy management as a user command."""

from typing import Annotated

from ..commands import CommandArgumentError, CommandContext, CommandRegistration, CommandRemainder
from ..completion import CommandCompletion, CompletionValue
from .manager import PermissionManager
from .models import Action, Decision, PermissionRule, PolicyScope

_ACTION_DESCRIPTIONS = {
    Action.FILESYSTEM_LIST: "List entries under a filesystem path.",
    Action.FILESYSTEM_READ: "Read the contents of a filesystem object.",
    Action.FILESYSTEM_CREATE: "Create a new filesystem object.",
    Action.FILESYSTEM_REPLACE: "Replace an existing filesystem object's contents.",
    Action.FILESYSTEM_DELETE: "Permanently delete a filesystem object.",
    Action.NETWORK_REQUEST: "Send an outbound HTTP request.",
    Action.PROCESS_EXECUTE: "Execute an exact process argument vector.",
    Action.SESSION_MUTATE: "Change process-local agent session state.",
}
_DECISION_DESCRIPTIONS = {
    Decision.ALLOW: "Permit matching operations without prompting.",
    Decision.ASK: "Require approval for every matching operation.",
    Decision.DENY: "Reject matching operations without prompting.",
}
_SCOPE_DESCRIPTIONS = {
    PolicyScope.WORKSPACE: "Persist the change in the workspace policy.",
    PolicyScope.SESSION: "Keep the change only for this process.",
}
_COLLECTION_LIMITS = {
    "read-root": "readable_roots",
    "write-root": "writable_roots",
    "network-origin": "network_origins",
}
_BOOLEAN_LIMITS = {
    "private-network": "deny_private_networks",
    "host-process": "allow_host_processes",
}
_BOOLEAN_LIMIT_POLARITY = {
    "private-network": "deny",
    "host-process": "allow",
}
_LIMIT_FIELDS = {**_COLLECTION_LIMITS, **_BOOLEAN_LIMITS}


def _enum_values(enum_type: type, descriptions: dict) -> tuple[CompletionValue, ...]:
    """Return described completion values for one string-valued enum."""
    return tuple(CompletionValue(item.value, descriptions[item]) for item in enum_type)


class PermissionCommands:
    """Expose one permission manager through an interactive command.

    Args:
        permission_manager (PermissionManager): Policy manager controlled by the command.
    """

    def __init__(self, permission_manager: PermissionManager) -> None:
        self._permission_manager = permission_manager

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return permission command registrations.

        Returns:
            tuple[CommandRegistration, ...]: Permission command registration.
        """
        return (
            CommandRegistration(
                self.permissions,
                name="permissions",
                description="Inspect and manage workspace and session operation policy.",
                completion=self._completion(),
            ),
        )

    def permissions(
        self,
        context: CommandContext,
        arguments: Annotated[tuple[str, ...], CommandRemainder()] = (),
    ) -> None:
        """Show, explain, or change workspace and session operation policy."""
        try:
            self._dispatch(context, arguments)
        except (TypeError, ValueError) as exc:
            raise CommandArgumentError(f"{exc}\n{self._usage()}") from exc

    def _dispatch(self, context: CommandContext, arguments: tuple[str, ...]) -> None:
        manager = self._permission_manager
        if not arguments or arguments == ("show",):
            context.interaction.info(manager.describe())
            return
        if len(arguments) == 2 and arguments[0] == "show":
            context.interaction.info(manager.describe(arguments[1]))
            return
        if arguments == ("reload",):
            manager.reload()
            context.interaction.info(
                "Reloaded the workspace policy; session overrides remain active."
            )
            return
        if arguments == ("help",):
            context.interaction.info(self._help())
            return
        if arguments == ("session", "reset"):
            if manager.reset_session():
                context.interaction.info("Cleared every session permission override.")
            else:
                context.interaction.warning("Session permission overrides were already empty.")
            return
        if len(arguments) == 4 and arguments[0] == "explain":
            _, tool, raw_action, resource = arguments
            decision = manager.explain(tool, Action(raw_action), resource)
            context.interaction.info(
                f"Effective decision: {decision.decision.value}\n"
                f"Reason: {decision.reason}\n"
                f"Determining sources: {', '.join(decision.sources)}"
            )
            return
        if arguments[:1] == ("default",):
            self._change_default(context, arguments[1:])
            return
        if arguments[:1] == ("rule",):
            self._change_rule(context, arguments[1:])
            return
        if arguments[:1] == ("limit",):
            self._change_limit(context, arguments[1:])
            return
        raise ValueError("Invalid permission command arguments.")

    def _change_default(self, context: CommandContext, arguments: tuple[str, ...]) -> None:
        manager = self._permission_manager
        if len(arguments) == 4 and arguments[0] == "set":
            _, raw_scope, raw_action, raw_decision = arguments
            scope = PolicyScope(raw_scope)
            action = Action(raw_action)
            decision = Decision(raw_decision)
            manager.set_default(action, decision, scope=scope)
            context.interaction.info(
                f"{scope.value.title()} default for {action.value} set to {decision.value}."
            )
            return
        if len(arguments) == 3 and arguments[0] == "reset":
            _, raw_scope, raw_action = arguments
            scope = PolicyScope(raw_scope)
            action = Action(raw_action)
            changed = manager.reset_default(action, scope=scope)
            self._report_reset(context, f"{scope.value} default {action.value}", changed)
            return
        raise ValueError("Invalid permission default arguments.")

    def _change_rule(self, context: CommandContext, arguments: tuple[str, ...]) -> None:
        manager = self._permission_manager
        if len(arguments) in {6, 7} and arguments[0] == "add":
            _, raw_scope, raw_decision, tool, raw_action, resource, *description = arguments
            scope = PolicyScope(raw_scope)
            rule = PermissionRule(
                decision=Decision(raw_decision),
                tool=tool,
                action=None if raw_action == "*" else Action(raw_action),
                resource=None if resource == "*" else resource,
                description=description[0] if description else None,
            )
            manager.add_rule(rule, scope=scope)
            context.interaction.info(f"Added {scope.value} {rule.decision.value} rule {rule.id}.")
            return
        if len(arguments) == 3 and arguments[0] == "remove":
            _, raw_scope, rule_id = arguments
            scope = PolicyScope(raw_scope)
            if manager.remove_rule(rule_id, scope=scope):
                context.interaction.info(f"Removed {scope.value} permission rule {rule_id}.")
            else:
                context.interaction.warning(
                    f"{scope.value.title()} permission rule {rule_id} was not found."
                )
            return
        raise ValueError("Invalid permission rule arguments.")

    def _change_limit(self, context: CommandContext, arguments: tuple[str, ...]) -> None:
        manager = self._permission_manager
        if len(arguments) == 4 and arguments[0] in {"add", "remove"}:
            operation, raw_scope, label, value = arguments
            if label not in _COLLECTION_LIMITS:
                raise ValueError("Add and remove require a collection permission limit.")
            scope = PolicyScope(raw_scope)
            changed = manager.update_limit_values(
                _COLLECTION_LIMITS[label], value, add=operation == "add", scope=scope
            )
            message = f"{scope.value.title()} permission limit {label} {value!r}"
            if changed:
                verb = "added" if operation == "add" else "removed"
                context.interaction.info(f"{message} {verb}.")
            else:
                context.interaction.warning(f"{message} was already unchanged.")
            return
        if len(arguments) == 4 and arguments[0] == "set":
            _, raw_scope, label, decision = arguments
            if label not in _BOOLEAN_LIMITS:
                raise ValueError("Set requires a boolean permission limit.")
            if decision not in {"allow", "deny"}:
                raise ValueError("Boolean limit decision must be 'allow' or 'deny'.")
            scope = PolicyScope(raw_scope)
            name = _BOOLEAN_LIMITS[label]
            manager.set_limit(name, decision == _BOOLEAN_LIMIT_POLARITY[label], scope=scope)
            context.interaction.info(
                f"{scope.value.title()} permission limit {label} set to {decision}."
            )
            return
        if len(arguments) == 3 and arguments[0] == "reset":
            _, raw_scope, label = arguments
            if label not in _LIMIT_FIELDS:
                raise ValueError(f"Unknown permission limit '{label}'.")
            scope = PolicyScope(raw_scope)
            changed = manager.reset_limit(_LIMIT_FIELDS[label], scope=scope)
            self._report_reset(context, f"{scope.value} limit {label}", changed)
            return
        raise ValueError("Invalid permission limit arguments.")

    @staticmethod
    def _report_reset(context: CommandContext, label: str, changed: bool) -> None:
        """Report whether one scoped reset changed policy state."""
        if changed:
            context.interaction.info(f"Reset {label}.")
        else:
            context.interaction.warning(f"{label.title()} was already inherited.")

    def _completion(self) -> CommandCompletion:
        decisions = _enum_values(Decision, _DECISION_DESCRIPTIONS)
        actions = _enum_values(Action, _ACTION_DESCRIPTIONS)
        scopes = _enum_values(PolicyScope, _SCOPE_DESCRIPTIONS)
        resource = CommandCompletion(
            values=(CompletionValue("*", "Match every canonical resource."),)
        )
        action = CommandCompletion(
            values=(CompletionValue("*", "Match every action."), *actions), next=resource
        )
        tool = CommandCompletion(
            values=(CompletionValue("*", "Match every registered tool."),),
            provider="tools",
            next=action,
        )
        decision = CommandCompletion(
            values=decisions, children={item.value: tool for item in decisions}
        )
        scoped_decision = CommandCompletion(
            values=scopes,
            children={scope.value: decision for scope in PolicyScope},
        )
        rule = CommandCompletion(
            values=(
                CompletionValue("add", "Create a workspace or session policy rule."),
                CompletionValue("remove", "Remove an existing scoped rule by identifier."),
            ),
            children={
                "add": scoped_decision,
                "remove": CommandCompletion(
                    values=scopes,
                    children={
                        PolicyScope.WORKSPACE.value: CommandCompletion(
                            provider=self._workspace_rule_values
                        ),
                        PolicyScope.SESSION.value: CommandCompletion(
                            provider=self._session_rule_values
                        ),
                    },
                ),
            },
        )
        default = self._default_completion(actions, decisions, scopes)
        limit = self._limit_completion(scopes)
        explain_resources = {
            action.value: CommandCompletion(values=(self._example_resource(action),))
            for action in Action
        }
        explain = CommandCompletion(
            provider="tools",
            next=CommandCompletion(values=actions, children=explain_resources),
        )
        return CommandCompletion(
            values=(
                CompletionValue("show", "Show all policy layers or one selected view."),
                CompletionValue("reload", "Reload only the workspace YAML policy."),
                CompletionValue("session", "Manage the complete process-local policy overlay."),
                CompletionValue("explain", "Explain one concrete effective decision."),
                CompletionValue("default", "Manage scoped fallback decisions by action."),
                CompletionValue("rule", "Manage workspace and session rules."),
                CompletionValue("limit", "Manage scoped roots, network, and process limits."),
                CompletionValue("help", "Explain policy concepts and common commands."),
            ),
            children={
                "show": CommandCompletion(
                    values=tuple(
                        CompletionValue(value, f"Show the {value} policy view.")
                        for value in ("effective", "workspace", "session")
                    )
                ),
                "session": CommandCompletion(
                    values=(CompletionValue("reset", "Clear every session override."),)
                ),
                "explain": explain,
                "default": default,
                "rule": rule,
                "limit": limit,
            },
        )

    @staticmethod
    def _default_completion(
        actions: tuple[CompletionValue, ...],
        decisions: tuple[CompletionValue, ...],
        scopes: tuple[CompletionValue, ...],
    ) -> CommandCompletion:
        """Return the scoped default command grammar."""
        action_decisions = CommandCompletion(
            values=actions,
            children={action.value: CommandCompletion(values=decisions) for action in Action},
        )
        action_reset = CommandCompletion(values=actions)
        return CommandCompletion(
            values=(
                CompletionValue("set", "Set a scoped action fallback."),
                CompletionValue("reset", "Restore a bootstrap or inherited fallback."),
            ),
            children={
                "set": CommandCompletion(
                    values=scopes,
                    children={scope.value: action_decisions for scope in PolicyScope},
                ),
                "reset": CommandCompletion(
                    values=scopes,
                    children={scope.value: action_reset for scope in PolicyScope},
                ),
            },
        )

    def _limit_completion(self, scopes: tuple[CompletionValue, ...]) -> CommandCompletion:
        """Return the scoped limit command grammar."""
        collection_values = (
            CompletionValue("read-root", "Roots from which files may be read."),
            CompletionValue("write-root", "Roots in which files may be changed."),
            CompletionValue("network-origin", "Origins to which requests may be sent."),
        )
        boolean_values = (
            CompletionValue("private-network", "Access to private network targets."),
            CompletionValue("host-process", "Processes run with this application's host access."),
        )

        def scoped_collection(operation: str) -> CommandCompletion:
            children = {
                scope.value: CommandCompletion(
                    values=collection_values,
                    children={
                        label: CommandCompletion(
                            values=(
                                self._root_values()
                                if operation == "add" and label in {"read-root", "write-root"}
                                else (self._limit_example(label),)
                            ),
                            provider=(
                                lambda scope=scope, field=field: self._limit_values(scope, field)
                            )
                            if operation == "remove"
                            else None,
                        )
                        for label, field in _COLLECTION_LIMITS.items()
                    },
                )
                for scope in PolicyScope
            }
            return CommandCompletion(values=scopes, children=children)

        set_decision = CommandCompletion(
            values=(
                CompletionValue("allow", "Open this user policy limit."),
                CompletionValue("deny", "Close this user policy limit."),
            )
        )
        scoped_boolean = CommandCompletion(
            values=scopes,
            children={
                scope.value: CommandCompletion(
                    values=boolean_values,
                    children={label: set_decision for label in _BOOLEAN_LIMITS},
                )
                for scope in PolicyScope
            },
        )
        reset_values = tuple(
            CompletionValue(label, "Reset this limit to its inherited or bootstrap value.")
            for label in _LIMIT_FIELDS
        )
        scoped_reset = CommandCompletion(
            values=scopes,
            children={
                PolicyScope.WORKSPACE.value: CommandCompletion(values=reset_values),
                PolicyScope.SESSION.value: CommandCompletion(
                    provider=self._session_limit_override_values
                ),
            },
        )
        return CommandCompletion(
            values=(
                CompletionValue("set", "Set a scoped boolean limit."),
                CompletionValue("add", "Add a root or origin to a scoped limit."),
                CompletionValue("remove", "Remove a root or origin from a scoped limit."),
                CompletionValue("reset", "Restore a bootstrap or inherited limit."),
            ),
            children={
                "set": scoped_boolean,
                "add": scoped_collection("add"),
                "remove": scoped_collection("remove"),
                "reset": scoped_reset,
            },
        )

    def _workspace_rule_values(self) -> tuple[CompletionValue, ...]:
        return tuple(self._rule_value(rule) for rule in self._permission_manager.persistent_rules)

    def _session_rule_values(self) -> tuple[CompletionValue, ...]:
        return tuple(self._rule_value(rule) for rule in self._permission_manager.session_rules)

    def _limit_values(self, scope: PolicyScope, name: str) -> tuple[CompletionValue, ...]:
        configuration = (
            self._permission_manager.configuration
            if scope is PolicyScope.WORKSPACE
            else self._permission_manager.effective_configuration
        )
        values = getattr(configuration.limits, name)
        return tuple(CompletionValue(value, "Currently effective value.") for value in values)

    def _session_limit_override_values(self) -> tuple[CompletionValue, ...]:
        overrides = self._permission_manager.session_overrides.limits
        return tuple(
            CompletionValue(label, "Reset this session limit to workspace inheritance.")
            for label, field in _LIMIT_FIELDS.items()
            if getattr(overrides, field) is not None
        )

    @staticmethod
    def _limit_example(label: str) -> CompletionValue:
        if label in {"read-root", "write-root"}:
            return CompletionValue("workspace", "Use the workspace root.")
        return CompletionValue("https://example.com", "Example exact HTTP origin.")

    @staticmethod
    def _root_values() -> tuple[CompletionValue, ...]:
        """Return root tokens suitable for adding a filesystem limit."""
        return (
            CompletionValue("workspace", "The active project workspace."),
            CompletionValue("loop-temp", "Loop-owned temporary files; allowed by default."),
            CompletionValue(
                "system-temp",
                "All OS temporary files; "
                "add only when cross-application temporary access is needed.",
            ),
        )

    @staticmethod
    def _example_resource(action: Action) -> CompletionValue:
        if action.value.startswith("filesystem."):
            return CompletionValue("workspace", "Example workspace-relative path.")
        if action is Action.NETWORK_REQUEST:
            return CompletionValue("https://example.com", "Example HTTP request URL.")
        if action is Action.PROCESS_EXECUTE:
            return CompletionValue("git", "Example shell-free process command line.")
        return CompletionValue("state", "Example session-state identifier.")

    @staticmethod
    def _rule_value(rule: PermissionRule) -> CompletionValue:
        action = rule.action.value if rule.action else "*"
        return CompletionValue(
            rule.id,
            rule.description
            or f"{rule.decision.value} {rule.tool} {action} {rule.resource or '*'}",
        )

    @staticmethod
    def _usage() -> str:
        return (
            "Usage: /permissions [show [effective|workspace|session] | "
            "help | reload | session reset | "
            "explain <tool> <action> <resource> | default set <workspace|session> <action> "
            "<decision> | default reset <workspace|session> <action> | rule add "
            "<workspace|session> <decision> <tool> <action|*> <resource|*> [description] | "
            "rule remove <workspace|session> <rule-id> | limit "
            "<set|add|remove|reset> <workspace|session> <name> [value]]"
        )

    @staticmethod
    def _help() -> str:
        """Return a concise guide to policy defaults, rules, and boundaries."""
        return (
            "Use /permissions show to inspect policy and /permissions show effective for its "
            "combined view. Defaults choose allow, ask, or deny by action; rules refine matching "
            "tools, actions, and resources; boundaries restrict reachable roots, origins, and host "
            "processes regardless of rules.\n\n"
            "Examples:\n"
            "  /permissions default set workspace filesystem.delete deny\n"
            "  /permissions rule add session allow read_text_file filesystem.read '*'\n"
            "  /permissions limit add workspace read-root system-temp\n"
            "  /permissions limit set session host-process allow\n"
            "  /permissions default set session process.execute ask\n\n"
            "Host-process permission runs commands with this application's host access; Loop does "
            "not currently provide an OS sandbox executor."
        )
