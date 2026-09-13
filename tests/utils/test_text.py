"""Tests for text formatting utilities."""

import pytest

from loop.constants import (
    CONTENT_PREVIEW_MAX_CHARS,
    CONTENT_PREVIEW_MAX_LINES,
)
from loop.utils import ChoiceItem
from loop.utils.text import (
    choice_items,
    format_content_diff,
    format_content_preview,
    format_tool_call_arguments,
    list_terms,
    snippet,
    validate_term,
)


def test_snippet_wraps_text_in_a_default_fenced_code_block():
    """Snippet uses a Markdown code fence and preserves the wrapped text."""
    assert snippet("print('hello')") == "```\nprint('hello')\n```"


def test_snippet_includes_an_optional_language_identifier():
    """Snippet places a provided language identifier after the opening fence."""
    assert snippet("print('hello')", language="python") == "```python\nprint('hello')\n```"


@pytest.mark.parametrize(
    "text",
    [
        "\n\r\ncontent\r\n\n",
        "first\n\nsecond",
        "   ",
    ],
)
def test_snippet_preserves_all_supplied_text_by_default(text):
    """Snippet retains meaningful and boundary whitespace inside its envelope."""
    assert snippet(text) == f"```\n{text}\n```"


def test_snippet_preserves_existing_wrapping_backticks_by_default():
    """Snippet retains an existing Markdown fence as literal referenced content."""
    text = "\n```\ncontent\n```\n"
    assert snippet(text) == f"````\n{text}\n````"


def test_snippet_preserves_meaningful_boundary_backticks():
    """Snippet removes only a complete outer fence, preserving content backticks."""
    assert snippet("`value`") == "```\n`value`\n```"


def test_snippet_preserves_an_existing_fence_language_marker():
    """Snippet retains an existing fence language marker as literal content."""
    text = "```python\nprint('hello')\n```"
    assert snippet(text) == f"````\n{text}\n````"


def test_snippet_replaces_existing_fence_when_preservation_is_disabled():
    """Snippet normalizes an existing fence when preservation is disabled."""
    assert (
        snippet("```python\nprint('hello')\n```", preserve_fence=False)
        == "```\nprint('hello')\n```"
    )


def test_snippet_preserves_code_when_fence_shape_is_not_complete():
    """Snippet keeps content intact when an apparent opening fence has no matching close."""
    text = "```\nv = 1\nprint(v + 1)"
    assert snippet(text) == "````\n```\nv = 1\nprint(v + 1)\n````"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("\r\ncontent\r\n", "content"),
        ("```\r\ncontent\r\n```", "content"),
        ("```\ncontent1\ncontent2\ncontent3\n```", "content1\ncontent2\ncontent3"),
        ("```python\ncontent1\ncontent2\ncontent3\n```", "content1\ncontent2\ncontent3"),
        ("```python\r\ncontent\r\n```", "content"),
        ("```\ncontent\n```   ", "content"),
        ("````\ncontent\n````", "content"),
        ("```\ncontent", "```\ncontent"),
        ("content\n```", "content\n```"),
        ("`content`", "`content`"),
    ],
)
def test_snippet_normalizes_supported_inputs_when_preservation_is_disabled(text, expected):
    """Snippet normalizes wrappers without discarding content from malformed inputs."""
    fence = "````" if "```" in expected else "```"
    assert snippet(text, preserve_fence=False) == f"{fence}\n{expected}\n{fence}"


def test_snippet_expands_the_fence_when_text_contains_it():
    """Snippet expands its delimiter until it cannot prematurely close the block."""
    assert snippet("before ``` after") == "````\nbefore ``` after\n````"


@pytest.mark.parametrize(
    ("terms", "exclusive", "quote", "expected"),
    [
        ([], True, "", ""),
        (["first"], True, "", "first"),
        (["first", "second"], True, "", "first or second"),
        (["first", "second"], False, "", "first and second"),
        (["first", "second", "third"], True, "", "first, second, or third"),
        (["first", "second", "third"], False, "", "first, second, and third"),
        (["first"], True, '"', '"first"'),
        (["first", "second", "third"], True, '"', '"first", "second", or "third"'),
        (["first", "second", "third"], False, "'", "'first', 'second', and 'third'"),
        ([None, "text", "native"], True, "'", "None, 'text', or 'native'"),
    ],
)
def test_list_terms_formats_empty_single_and_multiple_terms(terms, exclusive, quote, expected):
    """List terms use the appropriate conjunction and punctuation for each list size."""
    assert list_terms(terms, exclusive=exclusive, quote=quote) == expected


def test_list_terms_accepts_a_general_iterable():
    """List terms consume iterable inputs without requiring a concrete collection."""
    assert list_terms(term for term in ("first", "second")) == "first or second"


@pytest.mark.parametrize(
    ("value", "values"),
    [
        ("second", ["first", "second"]),
        (None, [None, "text"]),
    ],
)
def test_validate_term_accepts_values_in_the_allowed_list(value, values):
    """Term validation accepts values that are present in the allowed list."""
    validate_term(value, values, "Value must be")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], "Value must be ."),
        (["only"], "Value must be 'only'."),
        (["first", "second"], "Value must be 'first' or 'second'."),
        (["first", "second", "third"], "Value must be 'first', 'second', or 'third'."),
        ([None, "text"], "Value must be None or 'text'."),
    ],
)
def test_validate_term_rejects_values_with_the_allowed_terms(values, expected):
    """Term validation reports the supplied message and every allowed term."""
    with pytest.raises(ValueError, match=f"^{expected}$"):
        validate_term("missing", values, "Value must be")


def test_choice_items_normalizes_iterables_and_mappings():
    """Selectable values preserve their order and generate complete display labels."""
    assert choice_items(["first", "second"]) == (
        ChoiceItem(index="1", value="first", name="first"),
        ChoiceItem(index="2", value="second", name="second"),
    )
    assert choice_items({"first-id": "First", "second-id": "Second"}) == (
        ChoiceItem(index="1", value="first-id", name="First"),
        ChoiceItem(index="2", value="second-id", name="Second"),
    )


def test_choice_items_applies_explicit_indexes_to_choice_models():
    """Explicit indexes replace a model's display selector while retaining its metadata."""
    assert choice_items(
        [ChoiceItem("1", "approve", "Approve", "Proceed")],
        index={"approve": "a"},
    ) == (ChoiceItem("a", "approve", "Approve", "Proceed"),)


@pytest.mark.parametrize(
    "choices",
    [
        [
            ChoiceItem("a", "same", "First", "(a) First"),
            ChoiceItem("b", "same", "Second", "(b) Second"),
        ],
        [ChoiceItem("a", [], "First", "(a) First")],
    ],
)
def test_choice_items_rejects_duplicate_or_unhashable_choice_values(choices):
    """Choice model values must form a unique, hashable selection domain."""
    with pytest.raises(ValueError, match="choice values must be"):
        choice_items(choices)


def test_choice_items_rejects_extra_mapping_indexes():
    """Mapping indexes must not contain entries unrelated to the choice catalog."""
    with pytest.raises(ValueError, match="must map"):
        choice_items(["first"], index={"first": "a", "extra": "e"})


def test_choice_items_rejects_an_index_iterable_with_the_wrong_length():
    """Index iterables must provide precisely one selector for every choice."""
    with pytest.raises(ValueError, match="must map"):
        choice_items(["first", "second"], index=["a"])


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "choices cannot be empty."),
        ([""], "choice names cannot be empty."),
        (["same", "SAME"], "choice names must be unique ignoring case."),
        (["1"], "choice names cannot conflict with selection indexes."),
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


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ("", "+++", "1 addition(s), 0 deletion(s)"),
        ("+++", "", "0 addition(s), 1 deletion(s)"),
    ],
)
def test_format_content_diff_counts_changed_lines_with_header_like_content(before, after, expected):
    """Diff previews count changed lines whose content resembles a diff file header."""
    result = format_content_diff(before, after, "notes.txt")

    assert result.startswith(expected)


def test_format_content_diff_reduces_context_when_the_first_hunk_exceeds_line_limit():
    """Diff previews retain the first complete hunk by reducing unchanged context."""
    result = format_content_diff("one\ntwo\nthree", "one\nchanged\nthree", "notes.txt", max_lines=5)

    assert "@@ -2 +2 @@" in result
    assert "-two" in result
    assert "+changed" in result
    assert "change excerpt" not in result


def test_format_content_diff_omits_later_hunks_after_a_complete_hunk():
    """Diff previews retain an included hunk when later hunks exceed the limit."""
    before = "\n".join(("first", *("same" for _ in range(10)), "last"))
    after = "\n".join(("changed first", *("same" for _ in range(10)), "changed last"))
    result = format_content_diff(before, after, "notes.txt", max_lines=9)
    assert "... (1 changed hunk(s) omitted; preview limit reached)" in result
    assert result.count("\n@@") == 1
    assert len(result.partition("\n")[2].splitlines()) <= 9


def test_format_content_diff_omits_later_reduced_context_hunks_after_the_first():
    """Reduced-context previews still report later hunks that do not fit."""
    before = "\n".join(("first", *("same" for _ in range(10)), "last"))
    after = "\n".join(("changed first", *("same" for _ in range(10)), "changed last"))

    result = format_content_diff(before, after, "notes.txt", max_lines=6)

    assert "@@ -1 +1 @@" in result
    assert "... (1 changed hunk(s) omitted; preview limit reached)" in result


def test_format_content_diff_counts_reduced_context_hunks_consistently():
    """Reduced context reports the hunk count used by its rendered preview."""
    before = "\n".join(("old first", *("same" for _ in range(5)), "old last"))
    after = "\n".join(("new first", *("same" for _ in range(5)), "new last"))

    result = format_content_diff(before, after, "notes.txt", max_lines=6)

    assert result.startswith("2 addition(s), 2 deletion(s), 2 changed hunk(s)")
    assert "... (1 changed hunk(s) omitted; preview limit reached)" in result


def test_format_content_diff_shows_an_excerpt_when_changed_lines_exceed_char_limit():
    """Oversized changed lines retain visible evidence from both versions."""
    result = format_content_diff("a" * 200, "b" * 200, "notes.txt", max_chars=220)
    diff_preview = result.partition("\n")[2]

    assert "change excerpt; not a complete unified diff" in result
    assert "-a" in result
    assert "+b" in result
    assert "chars omitted" in result
    assert len(diff_preview) <= 220


def test_format_content_diff_marks_tiny_changed_line_excerpts():
    """Tight excerpt budgets retain each changed-line marker when text cannot fit."""
    result = format_content_diff("a" * 200, "b" * 200, "notes.txt", max_chars=180)

    assert "-…" in result
    assert "+…" in result


def test_format_content_diff_keeps_short_changed_lines_in_an_excerpt():
    """Line-limited excerpts retain complete short changes when character space permits."""
    before = "\n".join(f"before {index}" for index in range(6))
    after = "\n".join(f"after {index}" for index in range(7))

    result = format_content_diff(before, after, "notes.txt", max_lines=7)

    assert "change excerpt; not a complete unified diff" in result
    assert "-before 0" in result
    assert "+after 0" in result
    assert "chars omitted" not in result


def test_format_content_diff_keeps_the_excerpt_bounded_when_metadata_uses_the_budget():
    """Below the metadata floor the excerpt is trimmed to fit, keeping only the label."""
    result = format_content_diff("a" * 200, "b" * 200, "notes.txt", max_chars=150)
    preview = result.partition("\n")[2]

    assert "change excerpt; not a complete unified diff" in result
    assert "changed line(s)" not in result  # the omission notice cannot fit the tiny budget
    assert len(preview) <= 150


def test_format_content_diff_keeps_label_and_notice_when_the_budget_allows_it():
    """A budget covering the metadata floor keeps the label and notice without content."""
    result = format_content_diff("a" * 200, "b" * 200, "notes.txt", max_chars=160)
    preview = result.partition("\n")[2]

    assert "change excerpt; not a complete unified diff" in result
    assert "changed line(s)" in result
    assert len(preview) <= 160


def test_format_content_diff_excerpt_never_exceeds_the_budget_beyond_the_header_floor():
    """The preview is bounded by max_chars/max_lines unless it is only the file headers."""
    before = "\n".join(f"line-{index}-alpha" for index in range(30))
    after = "\n".join(f"line-{index}-beta" for index in range(30))

    for max_chars in range(50, 130):
        preview = format_content_diff(before, after, "notes.txt", max_chars=max_chars).partition(
            "\n"
        )[2]
        if len(preview.splitlines()) > 2:
            assert len(preview) <= max_chars

    for max_lines in range(1, 8):
        preview = format_content_diff(before, after, "notes.txt", max_lines=max_lines).partition(
            "\n"
        )[2]
        if len(preview.splitlines()) > 2:
            assert len(preview.splitlines()) <= max_lines


def test_format_content_diff_reports_unchanged_content():
    """Identical before and after content produces a clear no-change summary."""
    assert format_content_diff("same", "same", "notes.txt") == "No content changes."
