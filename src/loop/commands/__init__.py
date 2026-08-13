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
    "permissions",
    "quit",
    "skills",
    "tools",
    "use",
]

from .builtins import call, exit, help, permissions, quit, skills, tools, use
from .command import Command
from .command_manager import BUILTIN_COMMANDS, CommandManager
from .models import CommandArgumentError, CommandRegistrationError
