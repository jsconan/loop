"""Tests for text formatting utilities."""

import pytest

from loop.constants import (
    CONTENT_PREVIEW_MAX_CHARS,
    CONTENT_PREVIEW_MAX_LINES,
)
from loop.utils.text import (
    format_content_diff,
    format_content_preview,
    format_tabular_lines,
    format_tool_call_arguments,
)


def test_format_tool_call_arguments_formats_object_fields_as_parameters():
    """Tool-call displays render short object fields as named parameters."""
    arguments = '{"query":"term","options":{"limit":2,"exact":true},"paths":["one",null]}'

    assert format_tool_call_arguments(arguments) == (
        'query="term", options={"limit":2,"exact":true}, paths=["one",null]'
    )


def test_format_tool_call_arguments_truncates_long_nested_string_values():
    """Tool-call displays retain both ends of long strings at every nesting level."""
    arguments = '{"content":"0123456789abcdefghijklmnop","items":["abcdefghijklmnopqrstuvwxyz"]}'

    assert format_tool_call_arguments(arguments) == (
        'content="0123456789…hijklmnop", items=["abcdefghij…rstuvwxyz"]'
    )


def test_format_tool_call_arguments_accepts_a_custom_value_limit():
    """Tool-call displays apply a caller-provided limit to every string value."""
    arguments = '{"content":"abcdefgh"}'

    assert format_tool_call_arguments(arguments, max_chars=5) == 'content="ab…gh"'


def test_format_tool_call_arguments_rejects_an_insufficient_value_limit():
    """Tool-call displays require room for a prefix, suffix, and ellipsis."""
    with pytest.raises(ValueError, match="at least 3"):
        format_tool_call_arguments('{"content":"abcdefgh"}', max_chars=2)


def test_format_tool_call_arguments_bounds_invalid_json_as_raw_text():
    """Malformed tool-call arguments still receive a bounded terminal display."""
    arguments = "0123456789abcdefghijklmnop"

    assert format_tool_call_arguments(arguments) == "0123456789…hijklmnop"


def test_format_tool_call_arguments_bounds_non_object_json_as_raw_text():
    """Non-object JSON remains a bounded raw display rather than a parameter list."""
    arguments = '"0123456789abcdefghijklmnop"'

    assert format_tool_call_arguments(arguments) == '"012345678…ijklmnop"'


def test_format_content_preview_returns_formatted_lines():
    """Preview includes line numbers and content separated by a pipe."""
    result = format_content_preview("hello\nworld")
    assert "   1 | hello" in result
    assert "   2 | world" in result


def test_format_content_preview_truncates_over_long_content():
    """Content exceeding the line limit shows a truncation notice."""
    long_lines = "\n".join(f"line {i}" for i in range(CONTENT_PREVIEW_MAX_LINES + 10))
    result = format_content_preview(long_lines)
    assert f"({10} more lines omitted)" in result


def test_format_content_preview_preserves_all_lines_when_under_limit():
    """Content within the limit shows all lines without truncation notice."""
    lines = "\n".join(f"line {i}" for i in range(CONTENT_PREVIEW_MAX_LINES))
    result = format_content_preview(lines)
    assert f"({CONTENT_PREVIEW_MAX_LINES} more lines omitted)" not in result
    for i in range(CONTENT_PREVIEW_MAX_LINES):
        assert f"{i + 1:4d} | line {i}" in result


def test_format_content_preview_empty_content():
    """Empty string produces a preview with a single empty line."""
    result = format_content_preview("")
    assert "   1 | " in result


def test_format_content_preview_single_line():
    """A single line is numbered correctly."""
    result = format_content_preview("just one line")
    assert "   1 | just one line" in result


def test_format_content_preview_constants_are_reasonable():
    """Constants have expected magnitudes."""
    assert isinstance(CONTENT_PREVIEW_MAX_LINES, int)
    assert isinstance(CONTENT_PREVIEW_MAX_CHARS, int)
    assert CONTENT_PREVIEW_MAX_LINES > 0
    assert CONTENT_PREVIEW_MAX_CHARS > 0


def test_format_content_preview_max_chars_truncation():
    """Content exceeding max_chars shows a truncation notice."""
    long_content = "x" * (CONTENT_PREVIEW_MAX_CHARS + 100)
    result = format_content_preview(long_content)
    assert f"(truncated, total {len(long_content)} chars)" in result


def test_format_content_preview_max_chars_respects_custom_limit():
    """Custom max_chars is respected when provided."""
    content = "line1\nline2\nline3\nline4\nline5"
    result = format_content_preview(content, max_chars=10)
    assert "   1 | line1" in result
    assert f"(truncated, total {len(content)} chars)" in result


def test_format_content_preview_max_lines_truncation():
    """Content exceeding max_lines shows a truncation notice."""
    long_lines = "\n".join(f"line{i}" for i in range(25))
    result = format_content_preview(long_lines, max_lines=10)
    assert "(15 more lines omitted)" in result


def test_format_content_preview_max_lines_custom_limit():
    """Custom max_lines is respected when provided."""
    content = "\n".join(f"line{i}" for i in range(50))
    result = format_content_preview(content, max_lines=3)
    assert "   1 | line0" in result
    assert "   3 | line2" in result
    assert "(47 more lines omitted)" in result


def test_format_content_diff_shows_complete_unified_hunks():
    """Diff previews summarize and show surrounding unchanged context."""
    result = format_content_diff("one\ntwo\nthree", "one\nchanged\nthree", "notes.txt")
    assert result.startswith("1 addition(s), 1 deletion(s), 1 changed hunk(s)")
    assert "--- a/notes.txt" in result
    assert "+++ b/notes.txt" in result
    assert "-two" in result
    assert "+changed" in result
    assert " one" in result


def test_format_content_diff_omits_whole_hunks_at_the_preview_limit():
    """Diff previews never include a partial hunk when their limit is reached."""
    before = "\n".join(("first", *("same" for _ in range(10)), "last"))
    after = "\n".join(("changed first", *("same" for _ in range(10)), "changed last"))
    result = format_content_diff(before, after, "notes.txt", max_lines=7)
    assert "... (2 changed hunk(s) omitted; preview limit reached)" in result
    assert "\n@@" not in result


def test_format_content_diff_omits_later_hunks_after_a_complete_hunk():
    """Diff previews retain an included hunk when later hunks exceed the limit."""
    before = "\n".join(("first", *("same" for _ in range(10)), "last"))
    after = "\n".join(("changed first", *("same" for _ in range(10)), "changed last"))
    result = format_content_diff(before, after, "notes.txt", max_lines=8)
    assert "... (1 changed hunk(s) omitted; preview limit reached)" in result
    assert result.count("\n@@") == 1


def test_format_content_diff_reports_unchanged_content():
    """Identical before and after content produces a clear no-change summary."""
    assert format_content_diff("same", "same", "notes.txt") == "No content changes."


class _Item:
    """Minimal fixture object with ``name`` and ``description`` attributes."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description


def test_format_tabular_lines_returns_title_and_blank_line_when_title_provided() -> None:
    """When a title is set, the output starts with the title followed by a blank line."""
    result = format_tabular_lines([], title="List")
    assert result == "List\n"


def test_format_tabular_lines_returns_single_line_without_title() -> None:
    """Without a title, the output starts directly with the prefixed row."""
    items = [_Item("a", "desc")]
    result = format_tabular_lines(items)
    lines = result.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("  ")
    assert "a" in lines[0]


def test_format_tabular_lines_left_aligns_columns_to_max_width() -> None:
    """Columns are padded on the right to the longest value in each column."""
    items = [
        _Item("short", "long description value"),
        _Item("a_very_long_name", "short"),
    ]
    result = format_tabular_lines(items)
    lines = result.splitlines()
    assert len(lines) == 2
    prefix = "  "
    sep = "  "
    # width of name column is 16 ("a_very_long_name")
    name_width = 16
    # width of description column is 22 ("long description value")
    desc_width = 22

    for i, line in enumerate(lines):
        rest = line[len(prefix) :]
        name_part = rest[:name_width]
        desc_part = rest[name_width + len(sep) :]
        assert name_part == getattr(items[i], "name").ljust(name_width)
        assert desc_part == getattr(items[i], "description").ljust(desc_width)


def test_format_tabular_lines_respects_custom_prefix() -> None:
    """A custom prefix is prepended to every output line."""
    items = [_Item("item", "desc")]
    result = format_tabular_lines(items, prefix="-> ")
    assert result.startswith("-> ")


def test_format_tabular_lines_uses_custom_columns() -> None:
    """Custom columns are read from the specified attributes."""

    class _WithExtra:
        def __init__(self, name: str, description: str, extra: str) -> None:
            self.name = name
            self.description = description
            self.extra = extra

    items = [_WithExtra("n", "d", "e")]
    result = format_tabular_lines(items, columns=("name", "extra"))
    assert "n" in result
    assert "e" in result
    assert "d" not in result


def test_format_tabular_lines_handles_empty_items_list() -> None:
    """An empty list produces just the title (if present) or an empty string."""
    assert format_tabular_lines([]) == ""
    assert format_tabular_lines([], title="Empty list") == "Empty list\n"


def test_format_tabular_lines_handles_missing_attributes_gracefully() -> None:
    """Items missing an attribute fall back to an empty string."""

    class _Partial:
        name = "ok"

    result = format_tabular_lines([_Partial()])
    # Should not raise; missing "description" renders as empty
    assert "ok" in result


def test_format_tabular_lines_handles_non_string_attributes() -> None:
    """Attribute values that are not strings are converted via str()."""

    class _Mixed:
        def __init__(self, val: object) -> None:
            self.value = val

    items = [_Mixed(42), _Mixed("hello")]
    result = format_tabular_lines(items, columns=("value",))
    assert "42" in result
    assert "hello" in result


def test_format_tabular_lines_handles_longest_column_first() -> None:
    """Column order follows the ``columns`` iterable, not column width."""
    items = [
        _Item("ab", "a very long description"),
        _Item("c", "short"),
    ]
    result = format_tabular_lines(items, columns=("description", "name"))
    lines = result.splitlines()
    assert len(lines) == 2
    prefix = "  "
    sep = "  "
    # First column is description (max width 23), second is name (max width 2)
    desc_width = 23  # "a very long description"
    name_width = 2  # "ab"

    for i, line in enumerate(lines):
        rest = line[len(prefix) :]
        desc_part = rest[:desc_width]
        name_part = rest[desc_width + len(sep) :]
        assert desc_part == getattr(items[i], "description").ljust(desc_width)
        assert name_part == getattr(items[i], "name").ljust(name_width)
