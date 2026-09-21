"""Tests for bounded in-process text search utilities."""

import pytest

from loop.utils.search import search_text_paths


def test_search_text_paths_handles_empty_literal_case_and_context(tmp_path):
    """Literal search returns deterministic Unicode columns, case modes, and context."""
    first = tmp_path / "first.txt"
    first.write_text("before\n€ match\nafter\nMATCH\n", encoding="utf-8")

    assert search_text_paths([], "match", root=tmp_path) == ([], False)
    matches, truncated = search_text_paths([first], "match", root=tmp_path, context_lines=1)

    assert truncated is False
    assert matches == [
        {
            "path": "first.txt",
            "line": 2,
            "column": 3,
            "text": "€ match",
            "context": [
                {"line": 1, "text": "before"},
                {"line": 3, "text": "after"},
            ],
        },
        {
            "path": "first.txt",
            "line": 4,
            "column": 1,
            "text": "MATCH",
            "context": [{"line": 3, "text": "after"}],
        },
    ]

    sensitive, _ = search_text_paths([first], "MATCH", root=tmp_path)
    assert [match["line"] for match in sensitive] == [4]
    with_two_context, _ = search_text_paths([first], "match", root=tmp_path, context_lines=2)
    assert with_two_context[0]["context"][-1] == {"line": 4, "text": "MATCH"}


def test_search_text_paths_supports_regex_replacement_decoding_and_limits(tmp_path):
    """Regex search replaces invalid bytes and reports result and byte truncation."""
    source = tmp_path / "source.txt"
    source.write_bytes(b"one 123\ninvalid \xff value\ntwo 456\n")

    matches, truncated = search_text_paths(
        [source], r"[0-9]+", root=tmp_path, regex=True, case="sensitive"
    )
    assert truncated is False
    assert [(match["line"], match["column"]) for match in matches] == [(1, 5), (3, 5)]

    limited, truncated = search_text_paths(
        [source], r"[0-9]+", root=tmp_path, regex=True, max_results=1
    )
    assert len(limited) == 1
    assert truncated is True

    limited, truncated = search_text_paths([source], "one", root=tmp_path, max_bytes=1)
    assert limited == []
    assert truncated is True

    limited, truncated = search_text_paths(
        [source], "one", root=tmp_path, context_lines=1, max_bytes=90
    )
    assert len(limited) == 1
    assert truncated is True

    replacement, _ = search_text_paths([source], "�", root=tmp_path)
    assert replacement[0]["line"] == 2


def test_search_text_paths_rejects_invalid_regex_and_propagates_file_errors(tmp_path):
    """Invalid expressions and unreadable selected paths fail without process execution."""
    source = tmp_path / "source.txt"
    source.write_text("text", encoding="utf-8")

    with pytest.raises(ValueError, match="regex parse error"):
        search_text_paths([source], "(", root=tmp_path, regex=True)
    with pytest.raises(OSError):
        search_text_paths([tmp_path / "missing"], "text", root=tmp_path)
