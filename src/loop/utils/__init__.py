"""Provide utilities."""

__all__ = [
    "bound_tool_result",
    "BoundedTextContent",
    "cached_metadata",
    "cached_path",
    "CachedContentMetadata",
    "find_project_root",
    "format_content_diff",
    "format_content_preview",
    "format_tool_call_arguments",
    "IgnoreRule",
    "IgnoreRules",
    "is_path_ignored",
    "iter_visible_paths",
    "read_bounded_text",
    "register_cached_metadata",
    "register_shutdown_signals",
    "sha256_digest",
    "ShutdownRequested",
    "store_content",
    "store_text_stream",
]


from .content import (
    bound_tool_result,
    cached_metadata,
    cached_path,
    read_bounded_text,
    register_cached_metadata,
    store_content,
    store_text_stream,
)
from .hashing import sha256_digest
from .models import BoundedTextContent, CachedContentMetadata, IgnoreRule, IgnoreRules
from .path import find_project_root, is_path_ignored, iter_visible_paths
from .signals import ShutdownRequested, register_shutdown_signals
from .text import (
    format_content_diff,
    format_content_preview,
    format_tool_call_arguments,
)
