"""Expose declarative completion models, adapters, and aggregation."""

__all__ = [
    "CommandCompletion",
    "CommandCompletionAdapter",
    "CompletionAdapter",
    "CompletionManager",
    "CompletionMatch",
    "CompletionProvider",
    "CompletionValue",
    "MarkerCompletionAdapter",
    "ProjectPathCompletionAdapter",
    "SchemaCompletionProvider",
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
    CommandCompletion,
    CompletionMatch,
    CompletionProvider,
    CompletionValue,
    SchemaCompletionProvider,
    SchemaCompletionState,
)
