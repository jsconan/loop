"""Tests for text formatting utilities."""

import pytest

from loop.constants import (
    CONTENT_PREVIEW_MAX_CHARS,
    CONTENT_PREVIEW_MAX_LINES,
)
from loop.utils.text import (
    choice_items,
    format_content_diff,
    format_content_preview,
    format_tool_call_arguments,
)


def test_choice_items_normalizes_iterables_and_mappings():
    """Selectable values preserve their order and separate returned keys from displayed labels."""
    assert choice_items(["first", "second"]) == (("first", "first"), ("second", "second"))
    assert choice_items({"first-id": "First", "second-id": "Second"}) == (
        ("first-id", "First"),
        ("second-id", "Second"),
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "choices cannot be empty."),
        ([""], "choice labels cannot be empty."),
        (["same", "SAME"], "choice labels must be unique ignoring case."),
        (["1"], "choice labels cannot conflict with selection numbers."),
    ],
)
def test_choice_items_rejects_ambiguous_labels(values, message):
    """Selectable values reject catalog shapes that cannot be selected unambiguously."""
    with pytest.raises(ValueError, match=message):
        choice_items(values)


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
