"""Tests for the built-in file access tools."""

import json
from unittest.mock import MagicMock, call

import pytest

from loop import (
    ConsoleInteraction,
    InstructionsManager,
    ToolContext,
    tool_registry,
)
from loop.tools.files import list_folder as list_folder_tool


@pytest.fixture(autouse=True)
def approve_tool_calls(monkeypatch):
    """Approve central permission prompts unless a case overrides the decision."""
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))


def write_text_file(path, content):
    """Dispatch the context-aware file-writing tool."""
    return tool_registry.call(
        "write_text_file",
        json.dumps({"path": str(path), "content": content}),
        interaction=ConsoleInteraction(),
    )


def read_text_file(path, **ranges):
    """Dispatch the context-aware file-reading tool."""
    return tool_registry.call(
        "read_text_file",
        json.dumps({"path": str(path), **ranges}),
        interaction=ConsoleInteraction(),
    )


def list_folder(path, entry_type="all", recursive=False):
    """Dispatch the context-aware folder-listing tool."""
    result = tool_registry.call(
        "list_folder",
        json.dumps({"path": str(path), "entry_type": entry_type, "recursive": recursive}),
        interaction=ConsoleInteraction(),
    )
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return result


def test_list_folder_filters_and_sorts_immediate_entries(tmp_path):
    """Listing filters immediate entries by type and sorts their names."""
    (tmp_path / "zebra.txt").touch()
    (tmp_path / "alpha.txt").touch()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "ignored.txt").touch()

    assert list_folder(str(tmp_path)) == [
        {"path": "alpha.txt", "type": "file"},
        {"path": "nested", "type": "folder"},
        {"path": "zebra.txt", "type": "file"},
    ]
    assert list_folder(str(tmp_path), "files") == [
        {"path": "alpha.txt", "type": "file"},
        {"path": "zebra.txt", "type": "file"},
    ]
    assert list_folder(str(tmp_path), "folders") == [{"path": "nested", "type": "folder"}]


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


def test_list_folder_respects_git_and_agent_ignore_files(tmp_path):
    """Listings prune ignored paths and give agent rules higher precedence."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "metadata").touch()
    (tmp_path / ".gitignore").write_text(
        "*.log\ngenerated/\n!secret.txt\n",
        encoding="utf-8",
    )
    (tmp_path / ".agentignore").write_text(
        "secret.txt\n!generated/\n",
        encoding="utf-8",
    )
    (tmp_path / "debug.log").touch()
    (tmp_path / "secret.txt").touch()
    (tmp_path / "visible.txt").touch()
    (tmp_path / "generated").mkdir()
    (tmp_path / "generated" / "output.txt").touch()

    assert list_folder(str(tmp_path), "files") == [
        {"path": ".agentignore", "type": "file"},
        {"path": ".gitignore", "type": "file"},
        {"path": "visible.txt", "type": "file"},
    ]
    assert list_folder(str(tmp_path), recursive=True) == [
        {"path": ".agentignore", "type": "file"},
        {"path": ".gitignore", "type": "file"},
        {"path": "generated", "type": "folder"},
        {"path": "generated/output.txt", "type": "file"},
        {"path": "visible.txt", "type": "file"},
    ]


def test_list_folder_rejects_ignored_folder_as_traversal_root(tmp_path):
    """An ignored folder cannot be listed by requesting it directly."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").touch()
    (tmp_path / ".gitignore").write_text("private/\n", encoding="utf-8")
    private = tmp_path / "private"
    private.mkdir()
    (private / "secret.txt").touch()

    git_result = list_folder(str(tmp_path / ".git"))
    private_result = list_folder(str(private))

    assert git_result["error"] == "tool_call_denied"
    assert private_result["error"] == "tool_call_denied"


def test_list_folder_retains_tool_level_ignored_path_protection(tmp_path):
    """Direct tool invocation independently rejects an ignored traversal root."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("private/\n", "utf-8")
    private = tmp_path / "private"
    private.mkdir()
    context = ToolContext(ConsoleInteraction(), "list_folder")

    assert list_folder_tool(context, str(private)) == (
        f"Error listing folder: Path '{private}' is ignored."
    )


def test_list_folder_applies_ancestor_and_nested_ignore_files(tmp_path):
    """Ignore rules follow the project hierarchy and nested rules win."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    (project / ".gitignore").write_text("!keep.tmp\n", encoding="utf-8")
    (project / ".agentignore").write_text("private/\n", encoding="utf-8")
    (project / "drop.tmp").touch()
    (project / "keep.tmp").touch()
    (project / "private").mkdir()
    (project / "private" / "hidden.txt").touch()
    (project / "src").mkdir()
    (project / "src" / ".agentignore").write_text("generated.py\n", encoding="utf-8")
    (project / "src" / "generated.py").touch()
    (project / "src" / "main.py").touch()

    assert list_folder(str(project), "files", recursive=True) == [
        {"path": ".agentignore", "type": "file"},
        {"path": ".gitignore", "type": "file"},
        {"path": "keep.tmp", "type": "file"},
        {"path": "src/.agentignore", "type": "file"},
        {"path": "src/main.py", "type": "file"},
    ]


def test_list_folder_reports_failures(tmp_path):
    """Listing reports an invalid folder as a readable tool result."""
    assert list_folder(str(tmp_path / "missing")).startswith("Error listing folder:")


def test_file_navigation_reports_successful_instruction_context_changes(tmp_path):
    """Registered folder and file tools report only successful navigation to their manager."""
    nested = tmp_path / "nested"
    nested.mkdir()
    target = nested / "file.txt"
    target.write_text("content", encoding="utf-8")
    manager = InstructionsManager.discover(tmp_path)
    interaction = ConsoleInteraction()

    tool_registry.call(
        "list_folder",
        json.dumps({"path": str(nested)}),
        interaction=interaction,
        instructions_manager=manager,
    )
    assert manager.working_directory == nested.resolve()

    written = nested / "written.txt"
    tool_registry.call(
        "write_text_file",
        json.dumps({"path": str(written), "content": "content"}),
        interaction=interaction,
        instructions_manager=manager,
    )
    assert manager.working_directory == nested.resolve()

    tool_registry.call(
        "read_text_file",
        json.dumps({"path": str(target)}),
        interaction=interaction,
        instructions_manager=manager,
    )
    assert manager.working_directory == nested.resolve()

    tool_registry.call(
        "read_text_file",
        json.dumps({"path": str(tmp_path / "missing.txt")}),
        interaction=interaction,
        instructions_manager=manager,
    )
    assert manager.working_directory == nested.resolve()

    tool_registry.call(
        "list_folder",
        json.dumps({"path": str(tmp_path / "missing")}),
        interaction=interaction,
        instructions_manager=manager,
    )
    assert manager.working_directory == nested.resolve()


def test_read_text_file_returns_content_and_reports_empty_binary_or_failed_reads(tmp_path):
    """Reading reports content, empty files, binary files, and failures."""
    populated = tmp_path / "populated.txt"
    populated.write_text("hello", encoding="utf-8")
    empty = tmp_path / "empty.txt"
    empty.touch()
    binary = tmp_path / "binary.dat"
    binary.write_bytes(b"valid UTF-8\0binary payload")

    result = json.loads(read_text_file(str(populated)))
    assert result == {
        "path": str(populated),
        "content": "hello",
        "size_bytes": 5,
        "start_byte": 0,
        "end_byte": 5,
        "included_bytes": 5,
        "truncated": False,
        "start_line": 1,
        "end_line": 1,
    }
    assert read_text_file(str(empty)) == f"File '{empty}' is empty."
    assert read_text_file(str(binary)) == (
        f"Error reading file: File '{binary}' appears to be binary."
    )
    assert read_text_file(str(tmp_path / "missing.txt")).startswith("Error reading file:")


def test_read_text_file_denies_ignored_files_before_confirmation(tmp_path, monkeypatch):
    """Ignored files are denied centrally without offering a confirmation override."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("secret.txt\n", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("sensitive", encoding="utf-8")
    confirm = MagicMock(return_value=True)
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)

    assert '"error": "tool_call_denied"' in read_text_file(secret)
    assert confirm.call_args_list == []


def test_read_text_file_confirms_for_visible_files_by_default(tmp_path, monkeypatch):
    """The confirm-all mode includes ordinary visible files."""
    visible = tmp_path / "visible.txt"
    visible.write_text("hello", encoding="utf-8")
    confirm = MagicMock(return_value=True)
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)

    assert json.loads(read_text_file(visible))["content"] == "hello"
    confirm.assert_called_once()


def test_read_text_file_supports_line_pages_and_byte_continuations(tmp_path):
    """Line pages retain a hard byte limit and expose efficient byte continuation."""
    source = tmp_path / "unicode.txt"
    source.write_text("one\ntwø\nthree\nfour\n", encoding="utf-8")

    first = json.loads(read_text_file(source, max_lines=2, max_bytes=100))
    second = json.loads(
        read_text_file(
            source,
            start_byte=first["next_start_byte"],
            start_line=None,
            max_bytes=100,
        )
    )

    assert first["content"] == "one\ntwø\n"
    assert first["truncation_reason"] == "lines"
    assert first["next_start_line"] == 3
    assert second["content"] == "three\nfour\n"
    assert "start_line" not in second


def test_read_text_file_trims_utf8_safely_at_the_byte_ceiling(tmp_path):
    """A byte ceiling never splits a multibyte UTF-8 character."""
    source = tmp_path / "unicode.txt"
    source.write_text("a€b", encoding="utf-8")

    first = json.loads(read_text_file(source, max_bytes=3))
    second = json.loads(
        read_text_file(
            source,
            start_byte=first["next_start_byte"],
            start_line=None,
            max_bytes=4,
        )
    )

    assert first["content"] == "a"
    assert first["end_byte"] == 1
    assert second["content"] == "€b"


def test_read_text_file_accepts_equivalent_origins_and_rejects_conflicts(tmp_path):
    """Range selection normalizes the origin while rejecting genuine ambiguity."""
    source = tmp_path / "unicode.txt"
    source.write_text("€", encoding="utf-8")

    origin = json.loads(read_text_file(source, start_byte=0))
    conflicting = read_text_file(source, start_byte=1)
    split_character = read_text_file(source, start_byte=1, start_line=None)

    assert origin["content"] == "€"
    assert origin["start_byte"] == 0
    assert "Specify either start_byte or start_line" in conflicting
    assert "not valid UTF-8" in split_character


def test_write_text_file_requires_confirmation_and_reports_success(tmp_path, monkeypatch):
    """Writing only happens after an affirmative confirmation."""
    target = tmp_path / "written.txt"
    confirm = MagicMock(side_effect=[False, True])
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)

    assert '"error": "tool_call_denied"' in write_text_file(str(target), "blocked")
    assert not target.exists()

    assert write_text_file(str(target), "saved") == f"Successfully wrote to file '{target}'."
    assert target.read_text(encoding="utf-8") == "saved"
    assert confirm.call_args_list == [
        call(
            f"Agent wants to use 'write_text_file' for filesystem.write on '{target}'. Proceed?",
            default=False,
        ),
        call(
            f"Agent wants to use 'write_text_file' for filesystem.write on '{target}'. Proceed?",
            default=False,
        ),
    ]


def test_write_text_file_truncation_notice_for_large_content(tmp_path, monkeypatch):
    """Writing content exceeding the character limit appends a truncation notice."""
    target = tmp_path / "big.txt"
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))

    long_content = "x" * 2001
    assert write_text_file(str(target), long_content) == f"Successfully wrote to file '{target}'."
    assert target.read_text(encoding="utf-8") == long_content


def test_write_text_file_reports_open_failure(tmp_path, monkeypatch):
    """An invalid destination becomes a readable tool result."""
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    result = write_text_file(str(tmp_path / "missing" / "file.txt"), "content")
    assert result.startswith("Error writing to file:")
