"""Tests for bounded text ranges and out-of-context content caching."""

import json
from copy import deepcopy
from hashlib import file_digest
from uuid import uuid4

import pytest

from loop import constants
from loop.utils.content import (
    base64_decode,
    base64_encode,
    bound_tool_result,
    cached_metadata,
    cached_path,
    content_digest,
    content_identity,
    data_url,
    decode_content_cursor,
    encode_content_cursor,
    get_binary,
    read_bounded_text,
    register_cached_metadata,
    store_content,
    store_text_stream,
)


@pytest.mark.parametrize(
    ("content", "expected"),
    [("hello €", "hello €".encode()), (b"\x00\xff", b"\x00\xff")],
)
def test_get_binary_encodes_text_and_preserves_binary_content(content, expected):
    """Binary conversion UTF-8 encodes text and returns bytes unchanged."""
    result = get_binary(content)

    assert result == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("hello €", "aGVsbG8g4oKs"),
        (b"\x00\xff", "AP8="),
    ],
)
def test_base64_encode_encodes_text_and_binary_content(content, expected):
    """Base64 strings encode UTF-8 or raw bytes correctly."""
    assert base64_encode(content) == expected


@pytest.mark.parametrize(
    ("encoded", "expected", "binary"),
    [
        ("aGVsbG8g4oKs", "hello €", False),
        ("AP8=", b"\x00\xff", True),
    ],
)
def test_base64_decode_decodes_text_and_binary_content(encoded, expected, binary):
    """Base64 strings decode back to UTF-8 or raw bytes correctly."""
    assert base64_decode(encoded, binary=binary) == expected


@pytest.mark.parametrize(
    ("content", "media_type", "expected"),
    [
        ("hello €", "text/plain", "data:text/plain;base64,aGVsbG8g4oKs"),
        (b"\x00\xff", "application/octet-stream", "data:application/octet-stream;base64,AP8="),
    ],
)
def test_data_url_encodes_text_and_binary_content(content, media_type, expected):
    """Data URLs retain the declared MIME type and base64-encode UTF-8 or raw bytes."""
    assert data_url(content, media_type) == expected


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


def test_content_identity_separates_random_capabilities_from_stable_digests():
    """Each reference gets an unguessable handle while equal bytes retain one identity."""
    first_handle, first_digest = content_identity("same")
    second_handle, second_digest = content_identity("same")

    assert first_handle != second_handle
    assert first_digest == second_digest == content_digest("same")


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
    assert decode_content_cursor(payload["next_cursor"], handle) == payload["included_bytes"]


@pytest.mark.parametrize(("multiline", "byte_mode"), [(False, False), (True, False), (False, True)])
def test_oversized_file_result_keeps_a_direct_resumable_page_and_cached_snapshot(
    tmp_path, multiline, byte_mode
):
    """Serialization overhead preserves file receipts, byte continuation and the original output."""
    source = tmp_path / "escaped.txt"
    content = (
        '"\n' * (constants.MAX_TOOL_CONTENT_BYTES // 2 + 1)
        if multiline
        else '"' * constants.MAX_TOOL_CONTENT_BYTES
    )
    source.write_text(content, encoding="utf-8")
    page = read_bounded_text(
        source,
        start_byte=1 if byte_mode else None,
        start_line=None if byte_mode else 1,
        preserve_line_boundaries=True,
        include_version=True,
    )
    logical = "$WORKSPACE_ROOT/escaped.txt"
    output = json.dumps(
        {
            "ok": True,
            "result": {"path": logical, **page},
            "read_versions": {logical: page["version"]},
        },
        ensure_ascii=False,
    )
    assert len(output.encode()) > constants.MAX_TOOL_RESULT_BYTES

    bounded, handle = bound_tool_result(output, "synthetic")
    payload = json.loads(bounded)
    result = payload["result"]
    assert handle == payload["handle"]
    assert payload["cached_snapshot_format"] == "serialized_tool_result"
    assert cached_path(handle)[0].read_text(encoding="utf-8") == output
    assert len(bounded.encode()) <= constants.MAX_TOOL_RESULT_BYTES
    assert result["version"] == page["version"] == payload["read_versions"][logical]
    assert result["truncated"] is True
    assert result["truncation_reason"] == "result_limit"
    assert result["end_byte"] == result["next_start_byte"]
    assert result["end_byte"] == page["start_byte"] + len(result["content"].encode())
    assert "next_start_line" not in result
    if multiline:
        assert "next_start_line" in page
        assert result["end_line"] == result["start_line"] + result["content"].count("\n") - int(
            result["content"].endswith("\n")
        )
    remaining = read_bounded_text(source, start_byte=result["next_start_byte"], start_line=None)
    assert result["content"] + remaining["content"] == content[page["start_byte"] :]
    source.write_text("changed", encoding="utf-8")
    assert cached_path(handle)[0].read_text(encoding="utf-8") == output


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("existing_stream_cache", [False, True])
@pytest.mark.parametrize("large_metadata", [False, True])
def test_oversized_command_result_keeps_status_and_resumable_streams(
    failed, existing_stream_cache, large_metadata
):
    """Escaped command streams stay structured and recoverable within the result limit."""
    text = '"' * (constants.MAX_TOOL_CONTENT_BYTES // 2)
    stream = {
        "content": text,
        "captured_bytes": len(text),
        "included_bytes": len(text),
        "truncated": False,
        "capture_complete": True,
        "discarded_characters": 0,
        "decoding_loss": "none",
    }
    if existing_stream_cache:
        cached = text + "tail"
        stream["handle"] = store_content(cached, "synthetic stream")
        stream["next_cursor"] = encode_content_cursor(stream["handle"], len(text))
        stream["continuation"] = "Use read_cached_content with this handle and cursor."
        stream["captured_bytes"] = len(cached)
        stream["truncated"] = True
    body = {
        "boundary": "sandbox",
        "exit_code": 7 if failed else 0,
        "stdout": deepcopy(stream),
        "stderr": deepcopy(stream),
    }
    if large_metadata:
        body["diagnostic"] = "x" * (constants.MAX_TOOL_RESULT_BYTES // 2)
    payload = (
        {"ok": False, "problem": {"code": "process.nonzero_exit", "metadata": body}}
        if failed
        else {"ok": True, "result": body}
    )
    output = json.dumps(payload, ensure_ascii=False)
    assert len(output.encode()) > constants.MAX_TOOL_RESULT_BYTES

    bounded, handle = bound_tool_result(output, "synthetic command")
    delivered = json.loads(bounded)
    actual = delivered["problem"]["metadata"] if failed else delivered["result"]
    assert len(bounded.encode()) <= constants.MAX_TOOL_RESULT_BYTES
    assert cached_path(handle)[0].read_text(encoding="utf-8") == output
    assert actual["boundary"] == body["boundary"] and actual["exit_code"] == body["exit_code"]
    for name in ("stdout", "stderr"):
        item = actual[name]
        assert text.startswith(item["content"])
        assert item["capture_complete"] is True and item["discarded_characters"] == 0
        if len(item["content"]) < len(text):
            assert item["truncated"] is True
            assert (
                decode_content_cursor(item["next_cursor"], item["handle"]) == item["included_bytes"]
            )
            assert cached_path(item["handle"])[0].read_text(encoding="utf-8").startswith(text)
    if large_metadata:
        assert all(len(actual[name]["content"]) < len(text) for name in ("stdout", "stderr"))


@pytest.mark.parametrize(
    "change",
    [
        "bad_status",
        "bad_metadata",
        "bad_boundary",
        "bad_exit_code",
        "missing_stream",
        "bad_stream",
        "unsafe_unicode",
        "bad_captured",
        "bad_included",
        "bad_truncated",
        "bad_complete",
        "bad_discarded",
        "bad_decoding",
        "missing_handle",
        "bad_handle",
        "invalid_handle",
        "unknown_cache",
        "missing_cache_file",
        "wrong_cache_size",
        "wrong_cache_content",
        "wrong_cursor",
        "invalid_cursor",
        "unsafe_metadata_unicode",
        "no_room",
        "no_room_empty_stream",
    ],
)
def test_oversized_unverified_command_results_keep_cache_fallback(change):
    """Malformed stream metadata never becomes an invented command continuation."""
    text = '"' * (constants.MAX_TOOL_CONTENT_BYTES // 2)
    stream = {
        "content": text,
        "captured_bytes": len(text),
        "included_bytes": len(text),
        "truncated": False,
        "capture_complete": True,
        "discarded_characters": 0,
        "decoding_loss": "none",
    }
    body = {
        "boundary": "sandbox",
        "exit_code": 0,
        "stdout": deepcopy(stream),
        "stderr": deepcopy(stream),
    }
    payload = {"ok": True, "result": body}
    if change == "bad_status":
        payload["ok"] = "invalid"
    elif change == "bad_metadata":
        payload = {"ok": False, "problem": {"metadata": "invalid"}, "padding": text * 2}
    elif change == "bad_boundary":
        body["boundary"] = []
    elif change == "bad_exit_code":
        body["exit_code"] = True
    elif change == "missing_stream":
        body.pop("stderr")
        body["padding"] = text
    elif change == "bad_stream":
        body["stdout"] = "invalid"
        body["padding"] = text
    elif change == "unsafe_unicode":
        body["stdout"]["content"] = "\ud800"
        body["padding"] = text
    elif change == "bad_captured":
        body["stdout"]["captured_bytes"] = True
    elif change == "bad_included":
        body["stdout"]["included_bytes"] -= 1
    elif change == "bad_truncated":
        body["stdout"]["truncated"] = None
    elif change == "bad_complete":
        body["stdout"]["capture_complete"] = None
    elif change == "bad_discarded":
        body["stdout"]["discarded_characters"] = -1
    elif change == "bad_decoding":
        body["stdout"]["decoding_loss"] = []
    elif change == "missing_handle":
        body["stdout"]["captured_bytes"] += 1
        body["stdout"]["truncated"] = True
    elif change == "bad_handle":
        body["stdout"]["handle"] = 4
    elif change == "invalid_handle":
        body["stdout"]["handle"] = "invalid"
    elif change in {
        "unknown_cache",
        "missing_cache_file",
        "wrong_cache_size",
        "wrong_cache_content",
        "wrong_cursor",
        "invalid_cursor",
    }:
        handle = (
            uuid4().hex
            if change == "unknown_cache"
            else store_content(
                "x" * len(text) if change == "wrong_cache_content" else text, "fixture"
            )
        )
        body["stdout"]["handle"] = handle
        body["stdout"]["next_cursor"] = encode_content_cursor(
            handle, len(text) - int(change == "wrong_cursor")
        )
        if change == "invalid_cursor":
            body["stdout"]["next_cursor"] = "invalid"
        if change == "missing_cache_file":
            cached_path(handle)[0].unlink()
        if change == "wrong_cache_size":
            body["stdout"]["captured_bytes"] += 1
            body["stdout"]["truncated"] = True
    elif change == "unsafe_metadata_unicode":
        body["padding"] = "\ud800"
    elif change in {"no_room", "no_room_empty_stream"}:
        body["padding"] = "x" * constants.MAX_TOOL_RESULT_BYTES
        if change == "no_room_empty_stream":
            body["stderr"].update(content="", included_bytes=0, captured_bytes=0)

    output = json.dumps(payload)
    assert len(output.encode()) > constants.MAX_TOOL_RESULT_BYTES
    bounded, handle = bound_tool_result(output, "synthetic")
    assert json.loads(bounded)["truncation_reason"] == "tool_result_limit"
    assert cached_path(handle)[0].read_text(encoding="utf-8") == output


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", None),
        ("version", None),
        ("content", ""),
        ("content", ["invalid"]),
        ("content", "\ud800"),
        ("start_byte", True),
        ("start_byte", -1),
        ("end_byte", 1),
        ("size_bytes", 1),
        ("start_line", 0),
        ("start_line", True),
    ],
)
def test_oversized_unverified_file_results_keep_the_existing_cache_fallback(field, value):
    """Malformed file metadata cannot be transformed into a trusted partial read."""
    content = '"' * constants.MAX_TOOL_CONTENT_BYTES
    logical, version = "$WORKSPACE_ROOT/file.txt", "sha256:synthetic"
    page = {
        "path": logical,
        "version": version,
        "content": content,
        "start_byte": 0,
        "end_byte": len(content),
        "size_bytes": len(content),
    }
    page[field] = value
    output = json.dumps(
        {"ok": True, "result": page, "read_versions": {logical: version}, "padding": content}
    )

    bounded, handle = bound_tool_result(output, "synthetic")
    assert json.loads(bounded)["handle"] == handle
    assert "result" not in json.loads(bounded)
    assert cached_path(handle)[0].read_text(encoding="utf-8") == output


@pytest.mark.parametrize(
    "change",
    [
        "invalid_json",
        "duplicate_field",
        "not_mapping",
        "failure",
        "existing_handle",
        "no_result",
        "invalid_receipts",
        "no_receipt",
        "no_room",
        "unsafe_unicode",
    ],
)
def test_oversized_results_without_a_safe_file_page_keep_cache_fallback(change):
    """Only a verified file page with room for content gets a structured continuation."""
    content = '"' * constants.MAX_TOOL_CONTENT_BYTES
    logical, version = "$WORKSPACE_ROOT/file.txt", "sha256:synthetic"
    page = {
        "path": logical,
        "version": version,
        "content": content,
        "start_byte": 0,
        "end_byte": len(content),
        "size_bytes": len(content),
    }
    payload = {"ok": True, "result": deepcopy(page), "read_versions": {logical: version}}
    if change == "failure":
        payload["ok"] = False
    elif change == "existing_handle":
        payload["handle"] = "existing"
    elif change == "no_result":
        payload["result"] = content
    elif change == "invalid_receipts":
        payload["read_versions"] = []
    elif change == "no_receipt":
        payload["read_versions"] = {}
    elif change == "no_room":
        payload["padding"] = "x" * constants.MAX_TOOL_RESULT_BYTES
    elif change == "unsafe_unicode":
        payload["metadata"] = "\ud800"
    if change == "invalid_json":
        output = "{" + content * 2
    elif change == "duplicate_field":
        output = '{"ok":true,"ok":true,"result":' + json.dumps(page) + "}"
    elif change == "not_mapping":
        output = json.dumps([content])
    else:
        output = json.dumps(payload)

    bounded, handle = bound_tool_result(output, "synthetic")
    assert json.loads(bounded)["handle"] == handle
    assert "result" not in json.loads(bounded)
    assert cached_path(handle)[0].read_text(encoding="utf-8") == output


def test_content_cursors_are_opaque_handle_bound_offsets():
    """Content cursors round-trip offsets and reject invalid inputs or other handles."""
    handle = store_content("text", "source")
    other_handle = store_content("other", "other source")
    cursor = encode_content_cursor(handle, 3)

    assert decode_content_cursor(cursor, handle) == 3
    with pytest.raises(ValueError, match="non-negative"):
        encode_content_cursor(handle, -1)
    with pytest.raises(ValueError, match="Invalid cached content cursor"):
        decode_content_cursor(cursor, other_handle)
    with pytest.raises(ValueError, match="Invalid cached content cursor"):
        decode_content_cursor("invalid", handle)


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
    byte_first = read_bounded_text(source, start_byte=5, start_line=None, max_lines=2, max_bytes=3)

    assert line_first["content"] == "first\n"
    assert line_first["truncation_reason"] == "lines"
    assert line_first["next_start_byte"] == 11
    assert "next_start_line" not in line_first
    assert byte_first["content"] == "fir"
    assert byte_first["truncation_reason"] == "bytes"
    assert byte_first["next_start_byte"] == 8


def test_bounded_text_trims_utf8_safely_in_byte_resumable_mode(tmp_path):
    """Byte-resumable reads stop before an incomplete UTF-8 character."""
    source = tmp_path / "source.txt"
    source.write_text("a€b", encoding="utf-8")

    first = read_bounded_text(source, max_bytes=3)
    second = read_bounded_text(
        source, start_byte=first["next_start_byte"], start_line=None, max_bytes=4
    )

    assert first["content"] == "a"
    assert first["end_byte"] == 1
    assert second["content"] == "€b"
    with pytest.raises(ValueError, match="not valid UTF-8"):
        read_bounded_text(source, start_byte=2, start_line=None)


@pytest.mark.parametrize("character", ["é", "€", "😀"])
def test_bounded_text_completes_a_character_that_exceeds_the_requested_limit(tmp_path, character):
    """A small byte limit advances by one whole character and reports its adjustment."""
    source = tmp_path / "source.txt"
    source.write_text("a" + character + "z", encoding="utf-8")
    result = read_bounded_text(source, start_byte=1, start_line=None, max_bytes=1)

    assert result["content"] == character
    assert result["requested_max_bytes"] == 1
    assert result["effective_max_bytes"] == len(character.encode("utf-8"))
    assert result["next_start_byte"] == 1 + len(character.encode("utf-8"))
    resumed = read_bounded_text(source, start_byte=result["next_start_byte"], start_line=None)
    assert resumed["content"] == "z"


def test_bounded_text_rejects_incomplete_utf8_at_end_of_file(tmp_path):
    """An incomplete final sequence is invalid text rather than a resumable range."""
    source = tmp_path / "source.txt"
    source.write_bytes(b"prefix\xe2\x82")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        read_bounded_text(source)


@pytest.mark.parametrize("data", [b"\xe2\x82", b"\xe2x"])
def test_bounded_text_rejects_invalid_character_when_expanding_a_page(tmp_path, data):
    """Automatic character completion rejects truncated and malformed sequences."""
    source = tmp_path / "source.txt"
    source.write_bytes(data)
    with pytest.raises(ValueError, match="not valid UTF-8"):
        read_bounded_text(source, max_bytes=1)


def test_bounded_text_has_no_default_line_limit(tmp_path):
    """Default reads continue beyond 200 lines until EOF or the byte ceiling."""
    source = tmp_path / "source.txt"
    content = "line\n" * 300
    source.write_text(content, encoding="utf-8")

    result = read_bounded_text(source)

    assert result["content"] == content
    assert result["truncated"] is False


def test_bounded_text_preserves_line_boundaries_at_the_byte_ceiling(tmp_path):
    """Line reads continue at the next line and automatically expand for the first line."""
    source = tmp_path / "source.txt"
    source.write_text("first\nsecond\n", encoding="utf-8")

    bounded = read_bounded_text(source, max_bytes=8, preserve_line_boundaries=True)
    oversized = read_bounded_text(source, start_line=2, max_bytes=3, preserve_line_boundaries=True)

    assert bounded["content"] == "first\n"
    assert bounded["truncation_reason"] == "bytes"
    assert bounded["next_start_line"] == 2
    assert oversized["content"] == "second\n"
    assert oversized["requested_max_bytes"] == 3
    assert oversized["effective_max_bytes"] == len(b"second\n")
    assert oversized["truncated"] is False


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
    with pytest.raises(ValueError, match="start_byte is beyond the end"):
        read_bounded_text(text, start_byte=text.stat().st_size + 1, start_line=None)

    assert beyond["content"] == ""
    assert beyond["end_line"] == 1
    with pytest.raises(ValueError, match="binary"):
        read_bounded_text(binary, start_byte=8_192, start_line=None)


@pytest.mark.parametrize("include_version", [False, True])
def test_bounded_reads_include_source_versions_only_when_requested(tmp_path, include_version):
    """A partial UTF-8 page includes the complete source identity when requested."""
    source = tmp_path / "source.txt"
    source.write_text("visible\nnot returned\n", encoding="utf-8")
    result = read_bounded_text(source, max_lines=1, include_version=include_version)

    assert result["content"] == "visible\n"
    if include_version:
        assert result["version"] == content_digest("visible\nnot returned\n")
    else:
        assert "version" not in result
    assert result["truncated"] is True


def test_versioned_reads_reject_changes_during_snapshot_capture(tmp_path, monkeypatch):
    """Content cannot be paired with a digest captured after a concurrent save."""
    source = tmp_path / "source.txt"
    source.write_text("before", encoding="utf-8")

    def changed_digest(stream, algorithm):
        """Change the source after reading its bounded text and before hashing it."""
        source.write_text("after", encoding="utf-8")
        return file_digest(stream, algorithm)

    monkeypatch.setattr("loop.utils.content.file_digest", changed_digest)
    with pytest.raises(ValueError, match="File changed while reading; read it again"):
        read_bounded_text(source, include_version=True)
