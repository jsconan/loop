"""Expose independently registered interactive completion adapters."""

__all__ = [
    "CommandCompletionAdapter",
    "CompletionAdapter",
    "MarkerCompletionAdapter",
    "ProjectPathCompletionAdapter",
]

from .adapter import CompletionAdapter
from .command import CommandCompletionAdapter
from .marker import MarkerCompletionAdapter
from .project_path import ProjectPathCompletionAdapter
