"""Define commands available in every conversation loop."""

from typing import Annotated

from pydantic import Field

from ..completion import COMPLETION_ATTRIBUTE, CommandCompletion, CompletionValue
from ..context import CommandContext
from ..permissions import Capability, Decision, PermissionMode, PermissionRule
from .models import CommandArgumentError, CommandRemainder


def _values(enum_type: type) -> tuple[CompletionValue, ...]:
    """Return completion values for one string-valued enum."""
    return tuple(CompletionValue(member.value) for member in enum_type)


_RESOURCE_COMPLETION = CommandCompletion(values=(CompletionValue("*", "any resource"),))
_CAPABILITY_COMPLETION = CommandCompletion(
    values=(CompletionValue("*", "any capability"), *_values(Capability)),
    next=_RESOURCE_COMPLETION,
)
_TOOL_COMPLETION = CommandCompletion(provider="tools", next=_CAPABILITY_COMPLETION)
_CALL_COMPLETION = CommandCompletion(
    provider="tools",
    next=CommandCompletion(schema_provider="tool_arguments"),
)
_USE_COMPLETION = CommandCompletion(provider="skills")
_RESUME_COMPLETION = CommandCompletion(provider="sessions")
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


def help(context: CommandContext) -> None:  # pylint: disable=redefined-builtin
    """Show the available commands."""
    manager = context.manager
    if manager is None:
        raise ValueError("The help command requires a CommandManager.")
    commands = sorted(manager.commands, key=lambda command: command.name.casefold())
    context.interaction.table(commands, title="Available commands:", prefix="  /")


def new(context: CommandContext) -> None:
    """Start a fresh unpersisted session."""
    manager = context.manager
    if manager is None or manager.session_manager is None:
        raise ValueError("The new command requires a SessionManager.")
    session_manager = manager.session_manager
    session_manager.new_session()
    context.interaction.info("Started a new session.")


def rename(
    context: CommandContext,
    name: Annotated[str, Field(description="New human-readable session name.")],
) -> None:
    """Rename the active session."""
    manager = context.manager
    if manager is None or manager.session_manager is None:
        raise ValueError("The rename command requires a SessionManager.")
    session_manager = manager.session_manager
    session_manager.rename_session(name)
    context.interaction.info(f"Renamed session to '{session_manager.session.name}'.")


def resume(
    context: CommandContext,
    session_id: Annotated[str, Field(description="Exact persisted session ID.")],
) -> None:
    """Resume a persisted session."""
    manager = context.manager
    if manager is None or manager.session_manager is None:
        raise ValueError("The resume command requires a SessionManager.")
    session_manager = manager.session_manager
    try:
        session_manager.load_session(session_id)
    except ValueError as error:
        raise CommandArgumentError(str(error)) from error
    context.interaction.info(f"Restoring session history for '{session_manager.session.name}'...")
    context.interaction.history(session_manager.messages)
    context.interaction.token_usage(session_manager.model, session_manager.tokens, None)
    context.interaction.info(f"Resumed session '{session_manager.session.name}'.")


def sessions(context: CommandContext) -> None:
    """List persisted sessions."""
    manager = context.manager
    if manager is None or manager.session_manager is None:
        raise ValueError("The sessions command requires a SessionManager.")
    session_manager = manager.session_manager
    context.interaction.table(
        session_manager.store.list(),
        title="Persisted sessions:",
        columns=("name", "updated_at", "message_count"),
    )


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


def tools(context: CommandContext) -> None:
    """List all registered tools with their descriptions."""
    manager = context.manager
    if manager is None:
        raise ValueError("The tools command requires a CommandManager.")
    tool_registry = manager.tool_registry
    if tool_registry is None:
        raise ValueError("The CommandManager requires a ToolRegistry for the tools command.")
    tool_list = tool_registry.tools
    if not tool_list:
        context.interaction.info("No tools registered.")
        return
    context.interaction.table(tool_list, title="Registered tools:")


def call(
    context: CommandContext,
    name: Annotated[str, Field(description="Exact registered tool name.")],
    arguments: Annotated[
        tuple[str, ...],
        CommandRemainder(),
        Field(description="Command-like positional and name=value tool arguments."),
    ] = (),
) -> None:
    """Call a registered tool with command-like arguments."""
    command_manager = context.manager
    if command_manager is None:
        raise ValueError("The call command requires a CommandManager.")
    tool_registry = command_manager.tool_registry
    if tool_registry is None:
        raise ValueError("The CommandManager requires a ToolRegistry for the call command.")
    result = tool_registry.command(
        name,
        arguments,
        interaction=context.interaction,
        instructions_manager=command_manager.instructions_manager,
    )
    context.interaction.tool_result(name, result)


def skills(context: CommandContext) -> None:
    """List all discovered skills with their descriptions."""
    command_manager = context.manager
    if command_manager is None:
        raise ValueError("The skills command requires a CommandManager.")
    skill_manager = command_manager.skill_manager
    if skill_manager is None:
        raise ValueError("The CommandManager requires a SkillManager for the skills command.")
    skill_list = skill_manager.skills
    if not skill_list:
        context.interaction.info("No skills discovered.")
        return
    context.interaction.table(skill_list, title="Discovered skills:")


def use(
    context: CommandContext,
    name: Annotated[str, Field(description="Exact skill name.")],
) -> None:
    """Load a skill for subsequent model requests."""
    command_manager = context.manager
    if command_manager is None:
        raise ValueError("The use command requires a CommandManager.")
    instructions_manager = command_manager.instructions_manager
    if instructions_manager is None:
        raise ValueError("The CommandManager requires an InstructionsManager for the use command.")
    try:
        result = instructions_manager.activate_skill(name)
    except (OSError, UnicodeError, ValueError) as exc:
        raise CommandArgumentError(f"Could not load skill '{name}': {exc}") from exc
    if "error" in result:
        raise CommandArgumentError(result["message"])
    if result["instructions_updated"]:
        context.interaction.info(f"Loaded skill '{name}'.")
    else:
        context.interaction.info(f"Skill '{name}' is already loaded.")


def exit(context: CommandContext) -> None:  # pylint: disable=redefined-builtin
    """End the conversation."""
    manager = context.manager
    if manager is None:
        raise ValueError("The exit command requires a CommandManager.")
    manager.request_exit()


def quit(context: CommandContext) -> None:  # pylint: disable=redefined-builtin,consider-using-sys-exit
    """End the conversation."""
    exit(context)  # pylint: disable=consider-using-sys-exit


# Register completions for built-in commands
setattr(permissions, COMPLETION_ATTRIBUTE, _PERMISSIONS_COMPLETION)
setattr(use, COMPLETION_ATTRIBUTE, _USE_COMPLETION)
setattr(call, COMPLETION_ATTRIBUTE, _CALL_COMPLETION)
setattr(resume, COMPLETION_ATTRIBUTE, _RESUME_COMPLETION)
