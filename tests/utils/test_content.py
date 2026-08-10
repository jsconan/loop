"""Tests for bounded text ranges and out-of-context content caching."""

import json

import pytest

from loop import constants
from loop.utils.content import (
    bound_tool_result,
    cached_metadata,
    cached_path,
    read_bounded_text,
    register_cached_metadata,
    store_content,
    store_text_stream,
)


def test_content_cache_round_trips_bytes_and_strings():
    """Opaque handles resolve both textual and raw cached content."""
    text_handle = store_content("text", "text source")
    byte_handle = store_content(b"bytes", "byte source")

    text_path, text_source = cached_path(text_handle)
    byte_path, byte_source = cached_path(byte_handle)

    assert (text_path.read_bytes(), text_source) == (b"text", "text source")
    assert (byte_path.read_bytes(), byte_source) == (b"bytes", "byte source")
    assert cached_metadata(text_handle) == {"source": "text source", "reloadable": False}
    assert cached_path("missing") is None
    assert cached_metadata("missing") is None


def test_cached_metadata_rejects_noncanonical_handles_and_invalid_values():
    """Session metadata registration rejects malformed handles and values."""
    handle = store_content("text", "source")

    assert cached_metadata(handle.upper()) is None
    with pytest.raises(ValueError, match="Invalid content handle"):
        register_cached_metadata(handle.upper(), "source", True)
    with pytest.raises(ValueError, match="Invalid cached content metadata"):
        register_cached_metadata(handle, 1, "yes")


def test_content_budget_reserves_serialization_space_with_one_policy_limit():
    """The content ceiling derives from and remains below the complete result ceiling."""
    assert constants.MAX_TOOL_CONTENT_BYTES == constants.MAX_TOOL_RESULT_BYTES * 4 // 5


def test_text_stream_cache_validates_content_and_cleans_up_failures():
    """Streaming cache accepts split UTF-8 and rejects binary, invalid, and excessive input."""
    handle, size = store_text_stream([b"a\xe2", b"\x82\xac"], "stream", 4)

    assert size == 4
    assert cached_path(handle)[0].read_text(encoding="utf-8") == "a€"
    assert cached_metadata(handle) == {"source": "stream", "reloadable": False}
    with pytest.raises(ValueError, match="binary"):
        store_text_stream([b"a\0"], "binary", 4)
    with pytest.raises(UnicodeDecodeError):
        store_text_stream([b"\xff"], "invalid", 4)
    with pytest.raises(ValueError, match="download limit"):
        store_text_stream([b"12345"], "large", 4)


def test_tool_results_remain_unchanged_or_become_bounded_artifacts():
    """Tool-result bounding preserves small values and safely escapes large previews."""
    assert bound_tool_result("small", "tool") == ("small", None)

    bounded, handle = bound_tool_result("\0" * (constants.MAX_TOOL_RESULT_BYTES + 1), "tool")
    payload = json.loads(bounded)

    assert handle == payload["handle"]
    assert len(bounded.encode()) <= constants.MAX_TOOL_RESULT_BYTES
    assert payload["truncated"] is True


@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        ({"start_byte": 1, "start_line": 1}, "either start_byte or start_line"),
        ({"start_byte": -1, "start_line": None}, "start_byte must be non-negative"),
        ({"start_line": 0}, "start_line must be at least 1"),
        ({"max_lines": 0}, "max_lines must be at least 1"),
        ({"max_bytes": 0}, "max_bytes must be between"),
        ({"max_bytes": constants.MAX_TOOL_CONTENT_BYTES + 1}, "max_bytes must be between"),
    ],
)
def test_bounded_text_rejects_invalid_ranges(tmp_path, ranges, message):
    """Every invalid line and byte range fails with a precise validation error."""
    source = tmp_path / "source.txt"
    source.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_bounded_text(source, **ranges)


def test_bounded_text_defaults_when_both_start_positions_are_null(tmp_path):
    """Null start positions fall back to the first logical line."""
    source = tmp_path / "source.txt"
    source.write_text("line", encoding="utf-8")

    result = read_bounded_text(source, start_byte=None, start_line=None)

    assert result["start_line"] == 1
    assert result["content"] == "line"


def test_bounded_text_accepts_equivalent_byte_and_line_origins(tmp_path):
    """Byte zero and line one normalize to legitimate byte-oriented access."""
    source = tmp_path / "source.txt"
    source.write_text("line", encoding="utf-8")

    result = read_bounded_text(source, start_byte=0, start_line=1)

    assert result["content"] == "line"
    assert result["start_byte"] == 0
    assert "start_line" not in result


def test_bounded_text_prefers_line_mode_when_the_shared_origin_has_a_line_limit(tmp_path):
    """An explicit line limit selects line semantics at byte zero and line one."""
    source = tmp_path / "source.txt"
    source.write_text("first\nsecond\n", encoding="utf-8")

    result = read_bounded_text(source, start_byte=0, start_line=1, max_lines=1)

    assert result["content"] == "first\n"
    assert result["start_line"] == 1
    assert result["end_line"] == 1
    assert result["next_start_line"] == 2
    assert result["truncation_reason"] == "lines"


def test_bounded_text_applies_both_ceilings_to_byte_started_reads(tmp_path):
    """Byte-started reads stop at whichever explicit line or byte ceiling is reached first."""
    source = tmp_path / "source.txt"
    source.write_text("skip\nfirst\nsecond\nthird\n", encoding="utf-8")

    line_first = read_bounded_text(
        source, start_byte=5, start_line=None, max_lines=1, max_bytes=100
    )
    byte_first = read_bounded_text(
        source, start_byte=5, start_line=None, max_lines=2, max_bytes=3
    )

    assert line_first["content"] == "first\n"
    assert line_first["truncation_reason"] == "lines"
    assert line_first["next_start_byte"] == 11
    assert "next_start_line" not in line_first
    assert byte_first["content"] == "fir"
    assert byte_first["truncation_reason"] == "bytes"
    assert byte_first["next_start_byte"] == 8


def test_bounded_text_has_no_default_line_limit(tmp_path):
    """Default reads continue beyond 200 lines until EOF or the byte ceiling."""
    source = tmp_path / "source.txt"
    content = "line\n" * 300
    source.write_text(content, encoding="utf-8")

    result = read_bounded_text(source)

    assert result["content"] == content
    assert result["truncated"] is False


def test_bounded_text_seeks_directly_to_a_requested_line(tmp_path):
    """Line-oriented reads begin exactly after the preceding newline."""
    source = tmp_path / "source.txt"
    source.write_text("first\nsecond\n", encoding="utf-8")

    result = read_bounded_text(source, start_line=2)

    assert result["content"] == "second\n"
    assert result["start_byte"] == 6


def test_bounded_text_handles_lines_beyond_eof_and_embedded_binary_data(tmp_path):
    """Line scanning stops at EOF and binary markers are rejected wherever selected."""
    text = tmp_path / "text.txt"
    text.write_text("one\n", encoding="utf-8")
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"x" * 8_192 + b"\0")

    beyond = read_bounded_text(text, start_line=4)

    assert beyond["content"] == ""
    assert beyond["end_line"] == 1
    with pytest.raises(ValueError, match="binary"):
        read_bounded_text(binary, start_byte=8_192, start_line=None)
