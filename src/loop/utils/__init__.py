"""Provide utilities."""

__all__ = [
    "BoundedTextContent",
    "CachedContentMetadata",
    "ChoiceItem",
    "IgnoreRule",
    "IgnoreRules",
    "ShutdownRequested",
    "TextSearchContext",
    "TextSearchMatch",
    "bound_tool_result",
    "cached_metadata",
    "cached_path",
    "callable_hints",
    "callable_name",
    "canonical_path",
    "choice_items",
    "decode_content_cursor",
    "encode_content_cursor",
    "filter_paths_by_globs",
    "find_project_root",
    "format_content_diff",
    "format_content_preview",
    "format_tool_call_arguments",
    "is_binary_file",
    "is_path_ignored",
    "iter_visible_paths",
    "kill_process_group",
    "parse_command_line",
    "read_bounded_stream",
    "read_bounded_text",
    "register_cached_metadata",
    "register_shutdown_signals",
    "ripgrep_path",
    "search_text_paths",
    "sha256_digest",
    "store_content",
    "store_text_stream",
    "write_text_atomically",
]


from .callables import callable_hints, callable_name
from .content import (
    bound_tool_result,
    cached_metadata,
    cached_path,
    decode_content_cursor,
    encode_content_cursor,
    read_bounded_text,
    register_cached_metadata,
    store_content,
    store_text_stream,
)
from .files import is_binary_file, write_text_atomically
from .hashing import sha256_digest
from .models import (
    BoundedTextContent,
    CachedContentMetadata,
    ChoiceItem,
    IgnoreRule,
    IgnoreRules,
    TextSearchContext,
    TextSearchMatch,
)
from .path import (
    canonical_path,
    filter_paths_by_globs,
    find_project_root,
    is_path_ignored,
    iter_visible_paths,
)
from .process import kill_process_group, parse_command_line, read_bounded_stream
from .search import ripgrep_path, search_text_paths
from .signals import ShutdownRequested, register_shutdown_signals
from .text import (
    choice_items,
    format_content_diff,
    format_content_preview,
    format_tool_call_arguments,
)
