"""Expose local user-command definitions and dispatch."""

__all__ = [
    "BUILTIN_COMMANDS",
    "call",
    "Command",
    "CommandArgumentError",
    "CommandManager",
    "CommandRegistrationError",
    "exit",
    "help",
    "new",
    "permissions",
    "quit",
    "rename",
    "skills",
    "tools",
    "use",
]

from .builtins import (
    call,
    exit,
    help,
    new,
    permissions,
    quit,
    rename,
    skills,
    tools,
    use,
)
from .command import Command
from .command_manager import BUILTIN_COMMANDS, CommandManager
from .models import CommandArgumentError, CommandRegistrationError
