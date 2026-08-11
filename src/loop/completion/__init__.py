"""Expose declarative completion models, adapters, and aggregation."""

__all__ = [
    "CommandCompletion",
    "CommandCompletionAdapter",
    "COMPLETION_ATTRIBUTE",
    "CompletionAdapter",
    "CompletionManager",
    "CompletionMatch",
    "CompletionProvider",
    "CompletionValue",
    "MarkerCompletionAdapter",
    "ProjectPathCompletionAdapter",
    "SchemaCompletionState",
]

from .adapters import (
    CommandCompletionAdapter,
    CompletionAdapter,
    MarkerCompletionAdapter,
    ProjectPathCompletionAdapter,
)
from .manager import CompletionManager
from .models import (
    COMPLETION_ATTRIBUTE,
    CommandCompletion,
    CompletionMatch,
    CompletionProvider,
    CompletionValue,
    SchemaCompletionState,
)
