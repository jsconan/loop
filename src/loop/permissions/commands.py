"""Expose permission policy operations as a user command."""

from typing import Annotated

from pydantic import Field

from ..commands import CommandArgumentError, CommandContext, CommandRegistration
from ..completion import CommandCompletion, CompletionValue
from .manager import PermissionManager
from .models import Capability, Decision, PermissionMode, PermissionRule


def _values(enum_type: type) -> tuple[CompletionValue, ...]:
    """Return completion values for one string-valued enum."""
    return tuple(CompletionValue(member.value) for member in enum_type)


_RESOURCE_COMPLETION = CommandCompletion(values=(CompletionValue("*", "any resource"),))
_CAPABILITY_COMPLETION = CommandCompletion(
    values=(CompletionValue("*", "any capability"), *_values(Capability)),
    next=_RESOURCE_COMPLETION,
)
_TOOL_COMPLETION = CommandCompletion(provider="tools", next=_CAPABILITY_COMPLETION)
_DECISION_COMPLETION = CommandCompletion(
    values=_values(Decision),
    children={decision.value: _TOOL_COMPLETION for decision in Decision},
)
_PERMISSIONS_COMPLETION = CommandCompletion(
    values=tuple(CompletionValue(value) for value in ("show", "mode", "add", "session")),
    children={
        "mode": CommandCompletion(values=_values(PermissionMode)),
        "add": _DECISION_COMPLETION,
        "session": _DECISION_COMPLETION,
    },
)


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
                completion=_PERMISSIONS_COMPLETION,
            ),
        )

    def permissions(
        self,
        context: CommandContext,
        operation: Annotated[
            str,
            Field(description="Permission operation: show, mode, add, or session."),
        ] = "show",
        value: Annotated[
            str | None,
            Field(description="Mode for mode, or decision for add and session."),
        ] = None,
        tool: Annotated[
            str | None,
            Field(description="Tool pattern for add and session."),
        ] = None,
        capability: Annotated[
            str | None,
            Field(description="Optional capability pattern for add and session."),
        ] = None,
        resource: Annotated[
            str | None,
            Field(description="Optional resource pattern for add and session."),
        ] = None,
    ) -> None:
        """Show or change the local tool permission policy."""
        manager = self._permission_manager
        if operation == "show" and all(
            item is None for item in (value, tool, capability, resource)
        ):
            context.interaction.info(manager.describe())
            return
        if operation == "mode" and all(item is None for item in (tool, capability, resource)):
            try:
                mode = PermissionMode(value)
            except TypeError, ValueError:
                pass
            else:
                manager.set_mode(mode)
                context.interaction.info(f"Permission mode set to {value}.")
                return
        if operation in {"add", "session"} and value is not None and tool is not None:
            try:
                rule = PermissionRule(
                    decision=Decision(value),
                    tool=tool,
                    capability=None if capability in {None, "*"} else Capability(capability),
                    resource=None if resource in {None, "*"} else resource,
                )
            except ValueError:
                pass
            else:
                manager.add_rule(rule, persist=operation == "add")
                context.interaction.info(
                    f"Added {operation} {rule.decision.value} rule for '{rule.tool}'."
                )
                return
        raise CommandArgumentError(
            "Usage: /permissions [show | mode <mode> | "
            "add|session <allow|ask|deny> <tool> [capability] [resource]]"
        )
