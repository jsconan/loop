"""Tests for the built-in file access tools."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from loop import (
    BUILTIN_TOOLS,
    Action,
    ConsoleInteraction,
    FileTarget,
    InstructionsManager,
    Operation,
    PermissionManager,
    ToolRegistry,
)
from loop.tooling import ToolContext
from loop.tools import files as files_module
from loop.tools.files import edit_text_file as edit_text_file_tool
from loop.tools.files import list_folder as list_folder_tool
from loop.tools.files import write_text_file as write_text_file_tool

tool_registry = ToolRegistry(BUILTIN_TOOLS)


def dispatched_value(output: str):
    """Return a successful value while preserving failed envelopes for assertions."""
    payload = json.loads(output)
    if not payload["ok"]:
        return output
    result = payload["result"]
    return result if isinstance(result, str) else json.dumps(result)


def problem(output: str):
    """Return the problem from a failed tool result envelope."""
    return json.loads(output)["problem"]


@pytest.fixture(autouse=True)
def approve_tool_calls(monkeypatch, tmp_path):
    """Approve central permission prompts unless a case overrides the decision."""
    global tool_registry  # pylint: disable=global-statement
    monkeypatch.setattr(files_module, "ripgrep_path", MagicMock(return_value="rg"))
    tool_registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(tmp_path),
    )
    monkeypatch.setattr(PermissionManager, "request_permission", MagicMock(return_value=True))


def write_text_file(path, content):
    """Dispatch the context-aware file-writing tool."""
    return dispatched_value(
        tool_registry.call(
            "write_text_file",
            json.dumps({"path": str(path), "content": content}),
            interaction=ConsoleInteraction(),
        )
    )


def edit_text_file(path, old_content, new_content, replace_all=False):
    """Dispatch the context-aware exact text-editing tool."""
    return dispatched_value(
        tool_registry.call(
            "edit_text_file",
            json.dumps(
                {
                    "path": str(path),
                    "old_content": old_content,
                    "new_content": new_content,
                    "replace_all": replace_all,
                }
            ),
            interaction=ConsoleInteraction(),
        )
    )


def delete_path(path):
    """Dispatch the context-aware file-deletion tool."""
    return dispatched_value(
        tool_registry.call(
            "delete_path",
            json.dumps({"path": str(path)}),
            interaction=ConsoleInteraction(),
        )
    )


def read_text_file(path, **ranges):
    """Dispatch the context-aware file-reading tool."""
    return dispatched_value(
        tool_registry.call(
            "read_text_file",
            json.dumps({"path": str(path), **ranges}),
            interaction=ConsoleInteraction(),
        )
    )


def search_text(path, query, **options):
    """Dispatch the context-aware text-searching tool."""
    return dispatched_value(
        tool_registry.call(
            "search_text",
            json.dumps({"path": str(path), "query": query, **options}),
            interaction=ConsoleInteraction(),
        )
    )


def list_folder(path, entry_type="all", recursive=False):
    """Dispatch the context-aware folder-listing tool."""
    result = tool_registry.call(
        "list_folder",
        json.dumps({"path": str(path), "entry_type": entry_type, "recursive": recursive}),
        interaction=ConsoleInteraction(),
    )
    payload = json.loads(result)
    return payload["result"] if payload["ok"] else result


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

    assert problem(git_result)["code"] == "tool.denied"
    assert problem(private_result)["code"] == "filesystem.path_ignored"


def test_list_folder_retains_tool_level_ignored_path_protection(tmp_path):
    """Direct tool invocation independently rejects an ignored traversal root."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("private/\n", "utf-8")
    private = tmp_path / "private"
    private.mkdir()
    context = ToolContext(ConsoleInteraction(), "list_folder")

    result = list_folder_tool(context, str(private))
    assert result.code == "filesystem.path_ignored"
    assert str(private) in result.detail


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
    assert problem(list_folder(str(tmp_path / "missing")))["code"] == "filesystem.list_failed"


def test_file_navigation_reports_successful_instruction_context_changes(tmp_path, monkeypatch):
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

    monkeypatch.setattr(
        "loop.tools.files.search_text_paths",
        MagicMock(
            return_value=(
                [{"path": "file.txt", "line": 1, "column": 1, "text": "content"}],
                False,
            )
        ),
    )
    tool_registry.call(
        "search_text",
        json.dumps({"path": str(nested), "query": "content"}),
        interaction=interaction,
        instructions_manager=manager,
    )
    assert manager.working_directory == nested.resolve()

    tool_registry.call(
        "edit_text_file",
        json.dumps(
            {
                "path": str(target),
                "old_content": "content",
                "new_content": "edited",
                "replace_all": False,
            }
        ),
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
    assert problem(read_text_file(str(binary)))["code"] == "filesystem.binary_file"
    assert (
        problem(read_text_file(str(tmp_path / "missing.txt")))["code"] == "filesystem.read_failed"
    )


def test_read_text_file_allows_explicit_reads_of_ignored_files(tmp_path, monkeypatch):
    """Ignore files limit discovery rather than acting as an authorization boundary."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("secret.txt\n", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("sensitive", encoding="utf-8")
    confirm = MagicMock(return_value=True)
    monkeypatch.setattr(PermissionManager, "request_permission", confirm)

    assert json.loads(read_text_file(secret))["content"] == "sensitive"
    confirm.assert_not_called()


def test_read_text_file_allows_scoped_reads_by_default(tmp_path, monkeypatch):
    """The supervised policy allows reads inside its filesystem boundary."""
    visible = tmp_path / "visible.txt"
    visible.write_text("hello", encoding="utf-8")
    confirm = MagicMock(return_value=True)
    monkeypatch.setattr(PermissionManager, "request_permission", confirm)

    assert json.loads(read_text_file(visible))["content"] == "hello"
    confirm.assert_not_called()


def test_read_text_file_supports_line_pages_and_line_continuations(tmp_path):
    """Line pages retain a hard byte limit and continue with one line coordinate."""
    source = tmp_path / "unicode.txt"
    source.write_text("one\ntwø\nthree\nfour\n", encoding="utf-8")

    first = json.loads(read_text_file(source, max_lines=2, max_bytes=100))
    second = json.loads(read_text_file(source, start_line=first["next_start_line"], max_bytes=100))

    assert first["content"] == "one\ntwø\n"
    assert first["truncation_reason"] == "lines"
    assert first["next_start_line"] == 3
    assert second["content"] == "three\nfour\n"
    assert second["start_line"] == 3


def test_read_text_file_stops_at_lines_before_the_byte_ceiling(tmp_path):
    """A byte ceiling preserves complete lines and exposes a line continuation."""
    source = tmp_path / "unicode.txt"
    source.write_text("a€\nb\n", encoding="utf-8")

    first = json.loads(read_text_file(source, max_bytes=5))
    second = json.loads(read_text_file(source, start_line=first["next_start_line"], max_bytes=5))

    assert first["content"] == "a€\n"
    assert first["end_byte"] == 5
    assert first["truncation_reason"] == "bytes"
    assert first["next_start_line"] == 2
    assert second["content"] == "b\n"


def test_read_text_file_rejects_byte_selection_and_reports_oversized_lines(tmp_path):
    """The public tool is line-only and reports a line that exceeds its byte ceiling."""
    source = tmp_path / "unicode.txt"
    source.write_text("€", encoding="utf-8")

    oversized = json.loads(read_text_file(source, max_bytes=2))
    invalid = json.loads(read_text_file(source, start_byte=0))
    definition = next(
        definition
        for definition in tool_registry.definitions()
        if definition.name == "read_text_file"
    )

    assert oversized["content"] == ""
    assert oversized["truncation_reason"] == "line_too_long"
    assert "next_start_line" not in oversized
    assert invalid["problem"]["code"] == "tool.invalid_arguments"
    assert "start_byte" not in definition.parameters["properties"]


def test_search_text_finds_literal_unicode_with_smart_case_and_context(tmp_path, monkeypatch):
    """Text search returns deterministic lines, columns, Unicode, and neighboring context."""
    source = tmp_path / "source file.txt"
    source.write_text("before\nNeedle € here\nafter\nneedle lower\n", encoding="utf-8")
    matches = [
        {
            "path": "source file.txt",
            "line": 2,
            "column": 1,
            "text": "Needle € here",
            "context": [
                {"line": 1, "text": "before"},
                {"line": 3, "text": "after"},
            ],
        }
    ]
    engine = MagicMock(return_value=(matches, False))
    monkeypatch.setattr("loop.tools.files.search_text_paths", engine)

    result = json.loads(search_text(tmp_path, "Needle", context_lines=1))

    assert result == {"matches": matches, "truncated": False}
    assert engine.call_args.kwargs["case"] == "smart"
    assert engine.call_args.kwargs["context_lines"] == 1


def test_search_text_supports_regex_case_globs_files_and_result_limits(tmp_path, monkeypatch):
    """Regex, explicit case, inclusive globs, file roots, and global limits compose safely."""
    selected = tmp_path / "selected.py"
    selected.write_text("TOKEN 1\ntoken 2\ntoken 3\n", encoding="utf-8")
    (tmp_path / "excluded.txt").write_text("token 4\n", encoding="utf-8")
    engine = MagicMock(
        side_effect=[
            (
                [
                    {"path": "selected.py", "line": 1, "column": 1, "text": "TOKEN 1"},
                    {"path": "selected.py", "line": 2, "column": 1, "text": "token 2"},
                ],
                True,
            ),
            (
                [
                    {"path": "selected.py", "line": 2, "column": 1, "text": "token 2"},
                    {"path": "selected.py", "line": 3, "column": 1, "text": "token 3"},
                ],
                False,
            ),
        ]
    )
    monkeypatch.setattr("loop.tools.files.search_text_paths", engine)

    result = json.loads(
        search_text(
            tmp_path,
            r"token \d",
            regex=True,
            case="insensitive",
            include=["*.py"],
            max_results=2,
        )
    )
    direct = json.loads(search_text(selected, "token", case="sensitive"))

    assert [match["line"] for match in result["matches"]] == [1, 2]
    assert result["truncated"] is True
    assert [match["line"] for match in direct["matches"]] == [2, 3]
    assert list(engine.call_args_list[0].args[0]) == [selected]
    assert engine.call_args_list[0].kwargs["regex"] is True
    assert engine.call_args_list[0].kwargs["case"] == "insensitive"
    assert engine.call_args_list[0].kwargs["max_results"] == 2
    assert engine.call_args_list[1].kwargs["case"] == "sensitive"


def test_search_text_skips_binary_and_ignored_content(tmp_path, monkeypatch):
    """Folder search preserves ignore, binary, and symlink traversal protections."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "binary.dat").write_bytes(b"needle\0binary")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("needle\n", encoding="utf-8")
    (tmp_path / "linked.txt").symlink_to(outside)
    engine = MagicMock(
        return_value=(
            [{"path": "visible.txt", "line": 1, "column": 1, "text": "needle"}],
            False,
        )
    )
    monkeypatch.setattr("loop.tools.files.search_text_paths", engine)

    result = json.loads(search_text(tmp_path, "needle"))
    ignored = search_text(tmp_path / "ignored.txt", "needle")

    assert [match["path"] for match in result["matches"]] == ["visible.txt"]
    assert problem(ignored)["code"] == "filesystem.path_ignored"
    searched = set(engine.call_args.args[0])
    assert tmp_path / "visible.txt" in searched
    assert tmp_path / "ignored.txt" not in searched
    assert tmp_path / "binary.dat" not in searched
    assert tmp_path / "linked.txt" not in searched


def test_search_text_reports_empty_missing_invalid_and_unavailable_searches(tmp_path, monkeypatch):
    """Empty selections and distinct root, regex, and engine failures remain actionable."""
    (tmp_path / "empty.txt").touch()
    engine = MagicMock(return_value=([], False))
    monkeypatch.setattr("loop.tools.files.search_text_paths", engine)
    assert json.loads(search_text(tmp_path, "absent")) == {"matches": [], "truncated": False}
    assert problem(search_text(tmp_path / "missing", "text"))["code"] == (
        "filesystem.path_not_searchable"
    )
    engine.side_effect = RuntimeError("regex parse error")
    assert problem(search_text(tmp_path, "(", regex=True))["code"] == (
        "filesystem.invalid_search_pattern"
    )

    engine.side_effect = FileNotFoundError("rg missing")
    assert problem(search_text(tmp_path, "text"))["code"] == "filesystem.search_unavailable"


def test_write_text_file_requires_confirmation_and_reports_success(tmp_path, monkeypatch):
    """Writing only happens after an affirmative confirmation."""
    target = tmp_path / "written.txt"
    confirm = MagicMock(side_effect=[False, True])
    monkeypatch.setattr(PermissionManager, "request_permission", confirm)

    assert problem(write_text_file(str(target), "blocked"))["code"] == "tool.denied"
    assert not target.exists()

    assert write_text_file(str(target), "saved") == f"Successfully wrote to file '{target}'."
    assert target.read_text(encoding="utf-8") == "saved"
    assert confirm.call_count == 2
    assert all("filesystem.create" in item.args[0] for item in confirm.call_args_list)
    assert "blocked" in confirm.call_args_list[0].args[0]
    assert "saved" in confirm.call_args_list[1].args[0]


def test_write_text_file_truncation_notice_for_large_content(tmp_path, monkeypatch):
    """Writing content exceeding the character limit appends a truncation notice."""
    target = tmp_path / "big.txt"
    monkeypatch.setattr(PermissionManager, "request_permission", MagicMock(return_value=True))

    long_content = "x" * 2001
    assert write_text_file(str(target), long_content) == f"Successfully wrote to file '{target}'."
    assert target.read_text(encoding="utf-8") == long_content


def test_write_text_file_includes_a_diff_in_the_confirmation_prompt(tmp_path, monkeypatch):
    """Replacing UTF-8 text presents the proposed unified diff before approval."""
    target = tmp_path / "written.txt"
    target.write_text("before\nunchanged", encoding="utf-8")
    confirm = MagicMock(return_value=True)
    monkeypatch.setattr(PermissionManager, "request_permission", confirm)

    assert write_text_file(str(target), "after\nunchanged") == (
        f"Successfully wrote to file '{target}'."
    )
    prompt = confirm.call_args.args[0]
    assert "Proposed changes:" in prompt
    assert "--- a/" in prompt
    assert "-before" in prompt
    assert "+after" in prompt
    assert target.read_text(encoding="utf-8") == "after\nunchanged"


def test_write_text_file_falls_back_to_content_preview_for_binary_destination(
    tmp_path, monkeypatch
):
    """Unreadable existing content does not prevent a bounded write preview."""
    target = tmp_path / "written.txt"
    target.write_bytes(b"\xff")
    confirm = MagicMock(return_value=True)
    monkeypatch.setattr(PermissionManager, "request_permission", confirm)

    assert write_text_file(str(target), "replacement") == f"Successfully wrote to file '{target}'."
    prompt = confirm.call_args.args[0]
    assert "Existing content could not be previewed; proposed content:" in prompt
    assert "   1 | replacement" in prompt


def test_write_text_file_reports_open_failure(tmp_path, monkeypatch):
    """An invalid destination becomes a readable tool result."""
    monkeypatch.setattr(PermissionManager, "request_permission", MagicMock(return_value=True))
    result = write_text_file(str(tmp_path / "missing" / "file.txt"), "content")
    assert problem(result)["code"] == "filesystem.write_failed"


def test_write_text_file_removes_its_temporary_file_when_commit_fails(tmp_path, monkeypatch):
    """An atomic replacement failure does not leave staged content in the destination folder."""
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    monkeypatch.setattr(Path, "replace", MagicMock(side_effect=OSError("commit failed")))

    result = write_text_file(target, "content")

    assert problem(result)["detail"] == "commit failed"
    assert set(tmp_path.iterdir()) == {tmp_path / ".loop", target}
    assert target.read_text(encoding="utf-8") == "old"


def test_write_text_file_cancels_when_approved_target_changes(tmp_path, monkeypatch):
    """Replacement consumes the approved digest instead of overwriting changed content."""
    target = tmp_path / "target.txt"
    target.write_text("approved", encoding="utf-8")
    original = Path.read_bytes
    calls = 0

    def changing_read(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.write_text("changed", encoding="utf-8")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", changing_read)

    result = write_text_file(target, "replacement")

    assert problem(result)["detail"].startswith("The target changed after approval")
    assert target.read_text(encoding="utf-8") == "changed"


def test_write_executor_requires_authorized_state_and_existing_replacement(tmp_path, monkeypatch):
    """The executor fails closed without a plan or when a replacement disappears."""
    context = ToolContext(ConsoleInteraction(), "write_text_file")
    target = tmp_path / "target.txt"
    assert write_text_file_tool(context, str(target), "new").detail.startswith(
        "Authorized file-state precondition is missing"
    )

    target.write_text("old", encoding="utf-8")
    monkeypatch.setattr(Path, "is_file", lambda _path: False)
    assert problem(write_text_file(target, "new"))["detail"].startswith(
        "The target changed after approval"
    )


def test_edit_text_file_replaces_inserts_deletes_and_replaces_all(tmp_path):
    """Exact edits support unique replacement, anchored insertion, deletion, and explicit all."""
    target = tmp_path / "target.txt"
    target.write_text("first\nanchor\nremove\nrepeat repeat\n", encoding="utf-8")

    assert "(1 replacement)" in edit_text_file(target, "first", "changed")
    assert "(1 replacement)" in edit_text_file(target, "anchor\n", "inserted\nanchor\n")
    assert "(1 replacement)" in edit_text_file(target, "remove\n", "")
    assert "(2 replacements)" in edit_text_file(target, "repeat", "done", True)

    assert target.read_text(encoding="utf-8") == ("changed\ninserted\nanchor\ndone done\n")


@pytest.mark.parametrize(
    ("old_content", "new_content", "code"),
    [
        ("", "new", "filesystem.empty_match"),
        ("missing", "new", "filesystem.content_not_found"),
        ("same", "same", "filesystem.no_content_change"),
        ("repeat", "new", "filesystem.content_ambiguous"),
    ],
)
def test_edit_text_file_rejects_invalid_or_ambiguous_replacements(
    tmp_path, monkeypatch, old_content, new_content, code
):
    """Invalid exact edits fail with actionable codes before requesting approval."""
    target = tmp_path / "target.txt"
    target.write_text("same repeat repeat", encoding="utf-8")
    confirm = MagicMock(return_value=True)
    monkeypatch.setattr(PermissionManager, "request_permission", confirm)

    result = edit_text_file(target, old_content, new_content)

    assert problem(result)["code"] == code
    confirm.assert_not_called()
    assert target.read_text(encoding="utf-8") == "same repeat repeat"


def test_edit_text_file_rejects_missing_directory_and_non_utf8_targets(tmp_path):
    """Exact editing reports unsupported target and encoding states without mutation."""
    binary = tmp_path / "binary.dat"
    binary.write_bytes(b"\xff")

    assert problem(edit_text_file(tmp_path / "missing", "a", "b"))["code"] == (
        "filesystem.path_not_file"
    )
    assert problem(edit_text_file(tmp_path, "a", "b"))["code"] == ("filesystem.path_not_file")
    assert problem(edit_text_file(binary, "a", "b"))["code"] == "filesystem.binary_file"


def test_edit_text_file_reports_a_planning_read_failure(tmp_path, monkeypatch):
    """An operating-system failure while preparing an edit remains a structured file problem."""
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    monkeypatch.setattr(Path, "read_bytes", MagicMock(side_effect=OSError("read failed")))

    result = edit_text_file(target, "old", "new")

    assert problem(result)["code"] == "filesystem.edit_failed"
    assert problem(result)["detail"] == "read failed"


def test_edit_text_file_requires_approval_and_previews_the_resulting_diff(tmp_path, monkeypatch):
    """Exact editing shows its unified diff and leaves denied content unchanged."""
    target = tmp_path / "target.txt"
    target.write_text("before\nunchanged\n", encoding="utf-8")
    confirm = MagicMock(side_effect=[False, True])
    monkeypatch.setattr(PermissionManager, "request_permission", confirm)

    assert problem(edit_text_file(target, "before", "after"))["code"] == "tool.denied"
    assert target.read_text(encoding="utf-8") == "before\nunchanged\n"
    assert "(1 replacement)" in edit_text_file(target, "before", "after")

    prompt = confirm.call_args.args[0]
    assert "filesystem.replace" in prompt
    assert "Proposed changes:" in prompt
    assert "-before" in prompt
    assert "+after" in prompt


def test_edit_text_file_cancels_when_the_approved_target_changes(tmp_path, monkeypatch):
    """Exact editing consumes the approved digest instead of overwriting later content."""
    target = tmp_path / "target.txt"
    target.write_text("approved", encoding="utf-8")
    original = Path.read_bytes
    calls = 0

    def changing_read(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.write_text("changed", encoding="utf-8")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", changing_read)

    result = edit_text_file(target, "approved", "replacement")

    assert problem(result)["detail"].startswith("The target changed after approval")
    assert target.read_text(encoding="utf-8") == "changed"


def test_edit_text_file_reports_commit_failures_and_cleans_up_staging(tmp_path, monkeypatch):
    """Exact editing preserves original content and reports a failed atomic commit."""
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    monkeypatch.setattr(Path, "replace", MagicMock(side_effect=OSError("commit failed")))

    result = edit_text_file(target, "old", "new")

    assert problem(result)["detail"] == "commit failed"
    assert target.read_text(encoding="utf-8") == "old"
    assert set(tmp_path.iterdir()) == {tmp_path / ".loop", target}


def test_edit_executor_requires_an_authorized_digest(tmp_path):
    """Direct exact-edit execution fails closed without an approved operation state."""
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    context = ToolContext(ConsoleInteraction(), "edit_text_file")

    result = edit_text_file_tool(context, str(target), "old", "new")

    assert result.code == "filesystem.edit_failed"
    assert result.detail.startswith("Authorized file-state precondition is missing")

    incomplete = ToolContext(
        ConsoleInteraction(),
        "edit_text_file",
        operations=(
            Operation(
                tool_id="edit_text_file",
                action=Action.FILESYSTEM_REPLACE,
                target=FileTarget(path=str(target), expected_exists=True),
            ),
        ),
    )
    result = edit_text_file_tool(incomplete, str(target), "old", "new")
    assert result.detail == "Approved edit content is missing."


def test_edit_text_file_schema_exposes_only_the_model_facing_arguments():
    """The exact editor schema documents its compact public contract without internal state."""
    definition = next(
        definition
        for definition in tool_registry.definitions()
        if definition.name == "edit_text_file"
    )

    assert list(definition.parameters["properties"]) == [
        "path",
        "old_content",
        "new_content",
        "replace_all",
    ]


def test_delete_path_requires_confirmation_and_removes_files(tmp_path, monkeypatch):
    """Deleting a file requires approval and permanently removes the selected path."""
    target = tmp_path / "obsolete.txt"
    target.write_text("obsolete", encoding="utf-8")
    confirm = MagicMock(side_effect=[False, True])
    monkeypatch.setattr(PermissionManager, "request_permission", confirm)

    assert problem(delete_path(target))["code"] == "tool.denied"
    assert target.exists()

    assert delete_path(target) == f"Successfully deleted path '{target}'."
    assert not target.exists()
    assert confirm.call_count == 2
    assert all("filesystem.delete" in item.args[0] for item in confirm.call_args_list)
    assert all("Permanently delete this file." in item.args[0] for item in confirm.call_args_list)


def test_delete_path_removes_folder_trees_without_following_symbolic_links(tmp_path, monkeypatch):
    """Folder deletion removes descendants while symbolic-link deletion preserves its target."""
    folder = tmp_path / "obsolete"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    (nested / "child.txt").write_text("obsolete", encoding="utf-8")
    target = tmp_path / "target.txt"
    target.write_text("keep", encoding="utf-8")
    link = tmp_path / "target-link"
    link.symlink_to(target)
    confirm = MagicMock(return_value=True)
    monkeypatch.setattr(PermissionManager, "request_permission", confirm)

    assert delete_path(folder) == f"Successfully deleted path '{folder}'."
    assert not folder.exists()
    assert delete_path(link) == f"Successfully deleted path '{link}'."
    assert not link.exists()
    assert target.read_text(encoding="utf-8") == "keep"
    assert (
        "Permanently delete this folder and all of its contents."
        in (confirm.call_args_list[0].args[0])
    )
    assert (
        "Permanently delete this symbolic link; its target will not be deleted."
        in (confirm.call_args_list[1].args[0])
    )


def test_delete_path_can_delete_ignored_paths_and_rejects_unsupported_targets(
    tmp_path, monkeypatch
):
    """Explicit ignored paths can be approved; missing and special paths are rejected."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("secret.txt\n", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("sensitive", encoding="utf-8")
    confirm = MagicMock(return_value=True)
    monkeypatch.setattr(PermissionManager, "request_permission", confirm)

    assert delete_path(secret) == f"Successfully deleted path '{secret}'."
    assert not secret.exists()
    confirm.assert_called_once()
    assert problem(delete_path(tmp_path / "missing"))["code"] == "filesystem.path_missing"

    fifo = tmp_path / "events"
    os.mkfifo(fifo)
    assert problem(delete_path(fifo))["code"] == "filesystem.unsupported_path"
    assert fifo.exists()


def test_delete_path_reports_removal_failures(tmp_path, monkeypatch):
    """Filesystem removal failures are returned without reporting a successful deletion."""
    target = tmp_path / "protected.txt"
    target.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(PermissionManager, "request_permission", MagicMock(return_value=True))
    monkeypatch.setattr(Path, "unlink", MagicMock(side_effect=OSError("access denied")))

    assert problem(delete_path(target))["detail"] == "access denied"
    assert target.exists()
