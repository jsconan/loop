"""Tests for the built-in file access tools."""

import json
from unittest.mock import MagicMock, call

from loop import (
    ConsoleInteraction,
    list_folder,
    read_text_file,
    tool_registry,
)


def write_text_file(path, content):
    """Dispatch the context-aware file-writing tool."""
    return tool_registry.call(
        "write_text_file",
        json.dumps({"path": str(path), "content": content}),
        interaction=ConsoleInteraction(),
    )


def test_list_folder_filters_and_sorts_immediate_entries(tmp_path):
    """Listing filters immediate entries by type and sorts their names."""
    (tmp_path / "zebra.txt").touch()
    (tmp_path / "alpha.txt").touch()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "ignored.txt").touch()

    assert list_folder.__name__ == "list_folder"
    assert list_folder(str(tmp_path)) == [
        {"path": "alpha.txt", "type": "file"},
        {"path": "nested", "type": "folder"},
        {"path": "zebra.txt", "type": "file"},
    ]
    assert list_folder(str(tmp_path), "files") == [
        {"path": "alpha.txt", "type": "file"},
        {"path": "zebra.txt", "type": "file"},
    ]
    assert list_folder(str(tmp_path), "folders") == [
        {"path": "nested", "type": "folder"}
    ]


def test_list_folder_recursively_returns_relative_paths(tmp_path):
    """Recursive listings return selected entries relative to the requested folder."""
    (tmp_path / "root.txt").touch()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "child.txt").touch()
    (tmp_path / "nested" / "deeper").mkdir()

    assert list_folder(str(tmp_path), "files", recursive=True) == [
        {"path": "nested/child.txt", "type": "file"},
        {"path": "root.txt", "type": "file"},
    ]
    assert list_folder(str(tmp_path), "folders", recursive=True) == [
        {"path": "nested", "type": "folder"},
        {"path": "nested/deeper", "type": "folder"},
    ]
    assert list_folder(str(tmp_path), recursive=True) == [
        {"path": "nested", "type": "folder"},
        {"path": "nested/child.txt", "type": "file"},
        {"path": "nested/deeper", "type": "folder"},
        {"path": "root.txt", "type": "file"},
    ]


def test_list_folder_reports_failures(tmp_path):
    """Listing reports an invalid folder as a readable tool result."""
    assert list_folder(str(tmp_path / "missing")).startswith("Error listing folder:")


def test_read_text_file_returns_content_and_reports_empty_binary_or_failed_reads(tmp_path):
    """Reading reports content, empty files, binary files, and failures."""
    populated = tmp_path / "populated.txt"
    populated.write_text("hello", encoding="utf-8")
    empty = tmp_path / "empty.txt"
    empty.touch()
    binary = tmp_path / "binary.dat"
    binary.write_bytes(b"valid UTF-8\0binary payload")

    assert read_text_file.__name__ == "read_text_file"
    assert read_text_file(str(populated)) == "hello"
    assert read_text_file(str(empty)) == f"File '{empty}' is empty."
    assert read_text_file(str(binary)) == (
        f"Error reading file: File '{binary}' appears to be binary."
    )
    assert read_text_file(str(tmp_path / "missing.txt")).startswith("Error reading file:")


def test_write_text_file_requires_confirmation_and_reports_success(tmp_path, monkeypatch):
    """Writing only happens after an affirmative confirmation."""
    target = tmp_path / "written.txt"
    confirm = MagicMock(side_effect=[False, True])
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)

    assert write_text_file(str(target), "blocked") == "Write operation cancelled by user."
    assert not target.exists()

    assert write_text_file(str(target), "saved") == f"Successfully wrote to file '{target}'."
    assert target.read_text(encoding="utf-8") == "saved"
    assert confirm.call_args_list == [
        call(f"Agent wants to write to file '{target}'. Proceed?", default=False),
        call(f"Agent wants to write to file '{target}'. Proceed?", default=False),
    ]


def test_write_text_file_reports_open_failure(tmp_path, monkeypatch):
    """An invalid destination becomes a readable tool result."""
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    result = write_text_file(str(tmp_path / "missing" / "file.txt"), "content")
    assert result.startswith("Error writing to file:")
