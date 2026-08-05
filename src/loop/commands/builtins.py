"""Define commands available in every conversation loop."""

from ..context import CommandContext


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


def exit(context: CommandContext) -> None:  # pylint: disable=redefined-builtin
    """End the conversation."""
    manager = context.manager
    if manager is None:
        raise ValueError("The exit command requires a CommandManager.")
    manager.request_exit()


def quit(context: CommandContext) -> None:  # pylint: disable=redefined-builtin
    """End the conversation."""
    exit(context)
