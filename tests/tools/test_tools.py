"""Tests for the built-in public tools module."""

import re

from loop import get_current_datetime, read_text_file, write_text_file


def test_read_text_file_returns_content_and_reports_empty_or_failed_reads(tmp_path):
    """Reading reports content, empty files, and operating-system failures."""
    populated = tmp_path / "populated.txt"
    populated.write_text("hello", encoding="utf-8")
    empty = tmp_path / "empty.txt"
    empty.touch()

    assert read_text_file.__name__ == "read_text_file"
    assert read_text_file(str(populated)) == "hello"
    assert read_text_file(str(empty)) == f"File '{empty}' is empty."
    assert read_text_file(str(tmp_path / "missing.txt")).startswith("Error reading file:")


def test_write_text_file_requires_confirmation_and_reports_success(tmp_path, monkeypatch):
    """Writing only happens after an affirmative confirmation."""
    target = tmp_path / "written.txt"
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    assert write_text_file(str(target), "blocked") == "Write operation cancelled."
    assert not target.exists()

    monkeypatch.setattr("builtins.input", lambda _prompt: " Y ")
    assert write_text_file(str(target), "saved") == f"Successfully wrote to file '{target}'."
    assert target.read_text(encoding="utf-8") == "saved"


def test_write_text_file_reports_open_failure(tmp_path, monkeypatch):
    """An invalid destination becomes a readable tool result."""
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    result = write_text_file(str(tmp_path / "missing" / "file.txt"), "content")
    assert result.startswith("Error writing to file:")


def test_current_datetime_has_the_documented_shape():
    """The date tool returns a complete human-readable local timestamp."""
    assert re.fullmatch(
        r"[A-Z][a-z]+, [A-Z][a-z]+ \d{2}, \d{4} - \d{2}:\d{2}:\d{2}",
        get_current_datetime(),
    )
