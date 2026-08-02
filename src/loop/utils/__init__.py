"""Provide utilities."""

__all__ = [
    "find_project_root",
    "format_content_preview",
    "is_path_ignored",
    "iter_visible_paths",
    "register_shutdown_signals",
    "ShutdownRequested",
]


from .path import find_project_root, is_path_ignored, iter_visible_paths
from .signals import ShutdownRequested, register_shutdown_signals
from .text import format_content_preview
