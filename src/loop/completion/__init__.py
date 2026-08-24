"""Expose declarative completion models, adapters, and aggregation."""

__all__ = [
    "CommandCompletion",
    "CommandCompletionAdapter",
    "CompletionAdapter",
    "CompletionManager",
    "CompletionMatch",
    "CompletionProvider",
    "CompletionProviderRegistration",
    "CompletionProvidersProvider",
    "CompletionValue",
    "MarkerCompletionAdapter",
    "ProjectPathCompletionAdapter",
    "SchemaCompletionProvider",
    "SchemaCompletionProviderRegistration",
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
    CompletionProviderRegistration,
    CompletionProvidersProvider,
    CompletionValue,
    SchemaCompletionProvider,
    SchemaCompletionProviderRegistration,
    SchemaCompletionState,
)
