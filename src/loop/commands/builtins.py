"""Define commands available in every conversation loop."""

from typing import Annotated

from pydantic import Field

from ..completion import COMPLETION_ATTRIBUTE, CommandCompletion, CompletionValue
from ..context import CommandContext
from ..permissions import Capability, Decision, PermissionMode, PermissionRule
from .command import CommandArgumentError


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
PERMISSIONS_COMPLETION = CommandCompletion(
    values=tuple(CompletionValue(value) for value in ("show", "mode", "add", "session")),
    children={
        "mode": CommandCompletion(values=_values(PermissionMode)),
        "add": _DECISION_COMPLETION,
        "session": _DECISION_COMPLETION,
    },
)


def help(context: CommandContext) -> None:  # pylint: disable=redefined-builtin
    """Show the available commands."""
    manager = context.manager
    if manager is None:
        raise ValueError("The help command requires a CommandManager.")
    command_width = max(len(command.name) for command in manager.commands) + 1
    lines = ["Available commands:", ""]
    lines.extend(
        f"  /{command.name:<{command_width}} {command.description}" for command in manager.commands
    )
    context.interaction.info("\n".join(lines))


def permissions(
    context: CommandContext,
    operation: Annotated[
        str,
        Field(description="Permission operation: show, mode, add, or session."),
    ] = "show",
    value: Annotated[
        str | None,
        Field(description="Mode for mode, or decision for add and session."),
    ] = None,
    tool: Annotated[str | None, Field(description="Tool pattern for add and session.")] = None,
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
    manager = context.permission_manager
    if manager is None:
        raise ValueError("The permissions command requires a PermissionManager.")
    if operation == "show" and all(item is None for item in (value, tool, capability, resource)):
        context.interaction.info(manager.describe())
        return
    if operation == "mode" and all(item is None for item in (tool, capability, resource)):
        try:
            mode = PermissionMode(value)
        except (TypeError, ValueError):
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


setattr(permissions, COMPLETION_ATTRIBUTE, PERMISSIONS_COMPLETION)


def exit(context: CommandContext) -> None:  # pylint: disable=redefined-builtin
    """End the conversation."""
    manager = context.manager
    if manager is None:
        raise ValueError("The exit command requires a CommandManager.")
    manager.request_exit()


def quit(context: CommandContext) -> None:  # pylint: disable=redefined-builtin,consider-using-sys-exit
    """End the conversation."""
    exit(context)  # pylint: disable=consider-using-sys-exit
