"""Provide utilities."""

__all__ = [
    "BoundedTextContent",
    "CachedContentMetadata",
    "ChoiceItem",
    "IgnoreRule",
    "IgnoreRules",
    "Scalar",
    "ShutdownRequested",
    "TextSearchContext",
    "TextSearchMatch",
    "as_utc",
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
    "local_now",
    "normalized_key",
    "parse_command_line",
    "payload_digest",
    "read_bounded_stream",
    "read_bounded_text",
    "register_cached_metadata",
    "register_shutdown_signals",
    "ripgrep_path",
    "safe_scalar",
    "search_text_paths",
    "sha256_digest",
    "store_content",
    "store_text_stream",
    "utc_now",
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
from .dates import as_utc, local_now, utc_now
from .files import is_binary_file, write_text_atomically
from .hashing import payload_digest, sha256_digest
from .models import (
    BoundedTextContent,
    CachedContentMetadata,
    ChoiceItem,
    IgnoreRule,
    IgnoreRules,
    Scalar,
    TextSearchContext,
    TextSearchMatch,
)
from .normalization import normalized_key, safe_scalar
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
