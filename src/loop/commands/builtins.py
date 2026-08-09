"""Define commands available in every conversation loop."""

import shlex
from typing import Annotated

from pydantic import Field

from ..context import CommandContext
from ..permissions import Capability, Decision, PermissionMode, PermissionRule


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
    action: Annotated[
        str,
        Field(
            description=(
                "Permission operation: show, mode <mode>, add <decision> <tool> "
                "[capability] [resource], or session <decision> <tool> [capability] [resource]."
            )
        ),
    ] = "show",
) -> None:
    """Show or change the local tool permission policy."""
    manager = context.permission_manager
    if manager is None:
        raise ValueError("The permissions command requires a PermissionManager.")
    parts = shlex.split(action)
    operation = parts[0] if parts else "show"
    if operation == "show":
        context.interaction.info(manager.describe())
        return
    if operation == "mode" and len(parts) == 2:
        manager.set_mode(PermissionMode(parts[1]))
        context.interaction.info(f"Permission mode set to {parts[1]}.")
        return
    if operation in {"add", "session"} and 3 <= len(parts) <= 5:
        capability = None if len(parts) < 4 or parts[3] == "*" else Capability(parts[3])
        resource = None if len(parts) < 5 or parts[4] == "*" else parts[4]
        rule = PermissionRule(
            decision=Decision(parts[1]),
            tool=parts[2],
            capability=capability,
            resource=resource,
        )
        manager.add_rule(rule, persist=operation == "add")
        context.interaction.info(
            f"Added {operation} {rule.decision.value} rule for '{rule.tool}'."
        )
        return
    raise ValueError(
        "Usage: /permissions [show | mode <mode> | "
        "add|session <allow|ask|deny> <tool> [capability] [resource]]"
    )


def exit(context: CommandContext) -> None:  # pylint: disable=redefined-builtin
    """End the conversation."""
    manager = context.manager
    if manager is None:
        raise ValueError("The exit command requires a CommandManager.")
    manager.request_exit()


def quit(context: CommandContext) -> None:  # pylint: disable=redefined-builtin,consider-using-sys-exit
    """End the conversation."""
    exit(context)  # pylint: disable=consider-using-sys-exit
