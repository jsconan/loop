"""Expose local user-command definitions and dispatch."""

__all__ = [
    "BUILTIN_COMMANDS",
    "Command",
    "CommandArgumentError",
    "CommandManager",
    "CommandRegistrationError",
    "exit",
    "help",
    "permissions",
    "quit",
]

from .builtins import exit, help, permissions, quit
from .command import Command, CommandArgumentError
from .command_manager import BUILTIN_COMMANDS, CommandManager
from .utils import CommandRegistrationError
