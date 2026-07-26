"""Tests for the built-in file access tools."""

from loop import list_files, list_folders, read_text_file, write_text_file


def test_list_files_returns_sorted_file_names_and_reports_failures(tmp_path):
    """Listing returns direct files in name order and reports invalid folders."""
    (tmp_path / "zebra.txt").touch()
    (tmp_path / "alpha.txt").touch()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "ignored.txt").touch()

    assert list_files.__name__ == "list_files"
    assert list_files(str(tmp_path)) == ["alpha.txt", "zebra.txt"]
    assert list_files(str(tmp_path / "missing")).startswith("Error listing files:")


def test_list_folders_returns_sorted_folder_names_and_reports_failures(tmp_path):
    """Listing returns direct folders in name order and reports invalid folders."""
    (tmp_path / "zebra").mkdir()
    (tmp_path / "alpha").mkdir()
    (tmp_path / "file.txt").touch()
    (tmp_path / "alpha" / "ignored").mkdir()

    assert list_folders.__name__ == "list_folders"
    assert list_folders(str(tmp_path)) == ["alpha", "zebra"]
    assert list_folders(str(tmp_path / "missing")).startswith("Error listing folders:")


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
