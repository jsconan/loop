"""Tests for text formatting utilities."""

from loop.constants import (
    CONTENT_PREVIEW_MAX_CHARS,
    CONTENT_PREVIEW_MAX_LINES,
)
from loop.utils.text import format_content_preview


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
    assert f"(truncated, total {CONTENT_PREVIEW_MAX_CHARS} chars)" in result


def test_format_content_preview_max_chars_respects_custom_limit():
    """Custom max_chars is respected when provided."""
    content = "line1\nline2\nline3\nline4\nline5"
    result = format_content_preview(content, max_chars=10)
    assert "   1 | line1" in result
    assert "(truncated, total 10 chars)" in result


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
