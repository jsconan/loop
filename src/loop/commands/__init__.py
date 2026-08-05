"""Expose local user-command definitions and dispatch."""

__all__ = [
    "BUILTIN_COMMANDS",
    "Command",
    "CommandManager",
    "CommandRegistrationError",
    "exit",
    "help",
    "quit",
]

from .builtins import exit, help, quit
from .command import Command
from .command_manager import BUILTIN_COMMANDS, CommandManager
from .utils import CommandRegistrationError
