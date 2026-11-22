"""Tests for the built-in file access tools."""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from loop import (
    BUILTIN_TOOLS,
    Action,
    ApprovalChoice,
    ConsoleInteraction,
    Decision,
    FileTarget,
    InstructionsManager,
    Operation,
    PermissionConfiguration,
    PermissionManager,
    PermissionRule,
    PolicyLimits,
    PolicyScope,
    ProblemException,
    RuntimeEnvironment,
    SessionManager,
    SQLiteSessionStore,
    ToolRegistry,
    constants,
)
from loop.execution import CommandExecutionService
from loop.models import ToolCall
from loop.tooling import Tool, ToolContext
from loop.tooling.file_observations import FileObservations
from loop.tools import files as files_module
from loop.tools.files import delete_path as delete_path_tool
from loop.tools.files import edit_text_file as edit_text_file_tool
from loop.tools.files import list_folder as list_folder_tool
from loop.tools.files import write_text_file as write_text_file_tool
from loop.utils import (
    PathHolder,
    VirtualPath,
    bound_tool_result,
    cached_metadata,
    content_digest,
    sha256_digest,
)
from loop.utils.process import ProcessCapture, ProcessCaptureStatus

tool_registry: ToolRegistry


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
    context = ToolContext(
        ConsoleInteraction(), "list_folder", permission_manager=PermissionManager(tmp_path)
    )

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


def test_file_discovery_skips_a_protected_ignore_source_and_denied_subtree(tmp_path):
    """Listing and search use only allowed ignore rules and skip withheld directories."""
    ignore = tmp_path / ".gitignore"
    ignore.write_text("visible.txt\n")
    visible = tmp_path / "visible.txt"
    visible.write_text("owned public marker\n")
    withheld = tmp_path / "withheld"
    withheld.mkdir()
    (withheld / "child.txt").write_text("owned public marker\n")
    manager = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(hard_denied_paths=(str(ignore),)),
            rules=[
                PermissionRule(
                    decision=Decision.DENY,
                    action=Action.FILESYSTEM_READ,
                    resource=str(withheld),
                )
            ],
        ),
    )
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=manager)
    try:
        listing = json.loads(
            registry.call("list_folder", json.dumps({"path": str(tmp_path), "recursive": True}))
        )
        assert listing["result"] == [{"path": visible.name, "type": "file"}]
        search = json.loads(
            registry.call("search_text", json.dumps({"path": str(tmp_path), "query": "marker"}))
        )
        assert [entry["path"] for entry in search["result"]["matches"]] == [visible.name]
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("tool", "action", "arguments", "error"),
    [
        ("list_folder", Action.FILESYSTEM_LIST, {}, "filesystem.list_failed"),
        ("search_text", Action.FILESYSTEM_READ, {"query": "marker"}, "filesystem.search_failed"),
    ],
)
def test_file_discovery_rechecks_the_selected_root_after_approval(
    tmp_path, caplog, tool, action, arguments, error
):
    """A policy denial published after planning prevents root inspection."""
    (tmp_path / "visible").write_text("owned marker")
    manager = PermissionManager(tmp_path)
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=manager)

    def withdraw():
        """Publish a denial between authorization and dispatch."""
        manager.add_rule(
            PermissionRule(decision=Decision.DENY, action=action, resource=str(tmp_path)),
            scope=PolicyScope.SESSION,
        )

    try:
        result = registry.call(
            tool,
            json.dumps({"path": str(tmp_path), **arguments}),
            execution_started=withdraw,
        )
        assert problem(result)["code"] == error
        assert "File policy no longer permits" in problem(result)["detail"]
        assert "owned marker" not in result
        assert problem(result)["title"] in caplog.text
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("tool", "action", "arguments", "error"),
    [
        ("list_folder", Action.FILESYSTEM_LIST, {}, "filesystem.list_failed"),
        ("search_text", Action.FILESYSTEM_READ, {"query": "marker"}, "filesystem.search_failed"),
    ],
)
def test_file_discovery_refuses_a_replaced_approved_directory(
    tmp_path, monkeypatch, caplog, tool, action, arguments, error
):
    """One directory approval cannot be consumed against a replacement at the same path."""
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "original").write_text("original marker")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "unreviewed").write_text("replacement marker")
    manager = PermissionManager(tmp_path)
    manager.set_default(action, Decision.ASK, scope=PolicyScope.SESSION)
    approve = MagicMock(return_value=ApprovalChoice.ONCE)
    monkeypatch.setattr(PermissionManager, "request_permission", approve)
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=manager)

    def replace_directory():
        """Switch the directory identity before the approved operation starts."""
        selected.rename(tmp_path / "original")
        replacement.rename(selected)

    try:
        result = registry.call(
            tool,
            json.dumps({"path": str(selected), **arguments}),
            interaction=ConsoleInteraction(),
            execution_started=replace_directory,
        )
        assert problem(result)["code"] == error
        assert "File policy no longer permits" in problem(result)["detail"]
        assert "unreviewed" not in result and "replacement marker" not in result
        assert problem(result)["title"] in caplog.text
        assert approve.call_count == 2
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("tool", "action", "arguments", "error"),
    [
        ("list_folder", Action.FILESYSTEM_LIST, {}, "filesystem.list_failed"),
        ("search_text", Action.FILESYSTEM_READ, {"query": "marker"}, "filesystem.search_failed"),
    ],
)
def test_file_discovery_rechecks_authority_before_returning_results(
    tmp_path, monkeypatch, caplog, tool, action, arguments, error
):
    """Revocation during traversal prevents returning the collected names or matches."""
    (tmp_path / "observed").write_text("owned marker")
    manager = PermissionManager(tmp_path)
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=manager)

    def revoke():
        """Withdraw authority for the selected tree before publishing its result."""
        manager.add_rule(
            PermissionRule(decision=Decision.DENY, action=action, resource=str(tmp_path)),
            scope=PolicyScope.SESSION,
        )

    if tool == "list_folder":
        iterate = files_module.iter_visible_paths

        def list_then_revoke(*args, **kwargs):
            """Let traversal finish before revoking its authority."""
            yield from iterate(*args, **kwargs)
            revoke()

        monkeypatch.setattr(files_module, "iter_visible_paths", list_then_revoke)
    else:
        search = files_module.search_text_paths

        def search_then_revoke(*args, **kwargs):
            """Let search finish before revoking its authority."""
            matches = search(*args, **kwargs)
            revoke()
            return matches

        monkeypatch.setattr(files_module, "search_text_paths", search_then_revoke)

    try:
        result = registry.call(tool, json.dumps({"path": str(tmp_path), **arguments}))
        assert problem(result)["code"] == error
        assert "File policy no longer permits" in problem(result)["detail"]
        assert "observed" not in result and "owned marker" not in result
        assert problem(result)["title"] in caplog.text
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("tool", "is_directory", "arguments"),
    [
        ("list_folder", True, {}),
        ("search_text", True, {"query": "marker"}),
        ("read_text_file", False, {}),
    ],
)
def test_file_tools_keep_default_review_for_bound_aliases(
    tmp_path, monkeypatch, tool, is_directory, arguments
):
    """Ordinary default approval follows the canonical root bound by a benign alias."""
    directory = tmp_path / "directory"
    directory.mkdir()
    file = directory / "visible"
    file.write_text("owned marker")
    target = directory if is_directory else file
    alias = tmp_path / "alternative"
    alias.symlink_to(target, target_is_directory=is_directory)
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.FILESYSTEM_READ, Decision.ASK, scope=PolicyScope.SESSION)
    manager.set_default(Action.FILESYSTEM_LIST, Decision.ASK, scope=PolicyScope.SESSION)
    approve = MagicMock(return_value=ApprovalChoice.ONCE)
    monkeypatch.setattr(PermissionManager, "request_permission", approve)
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=manager)
    try:
        for selected in (target, alias):
            result = json.loads(
                registry.call(
                    tool,
                    json.dumps({"path": str(selected), **arguments}),
                    interaction=ConsoleInteraction(),
                )
            )
            assert result["ok"], result
            if tool == "list_folder":
                assert result["result"] == [{"path": "visible", "type": "file"}]
            elif tool == "search_text":
                assert [item["path"] for item in result["result"]["matches"]] == ["visible"]
            else:
                assert result["result"]["content"] == "owned marker"
        # Literal and aliased targets each complete both authorization phases.
        assert approve.call_count == 4
    finally:
        registry.close()


def test_file_discovery_does_not_transfer_default_review_to_a_changed_alias(tmp_path, caplog):
    """A directory selected through an alias keeps its original bound identity."""
    original = tmp_path / "original"
    unrelated = tmp_path / "unrelated"
    original.mkdir()
    unrelated.mkdir()
    (unrelated / "neighbor").touch()
    alias = tmp_path / "selected"
    alias.symlink_to(original, target_is_directory=True)
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.FILESYSTEM_LIST, Decision.ASK, scope=PolicyScope.SESSION)
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=manager)

    def replace_selection():
        """Redirect the owned selection after both approval phases."""
        alias.unlink()
        alias.symlink_to(unrelated, target_is_directory=True)

    try:
        result = registry.call(
            "list_folder",
            json.dumps({"path": str(alias)}),
            interaction=ConsoleInteraction(),
            execution_started=replace_selection,
        )
        assert problem(result)["code"] == "filesystem.list_failed"
        assert "File policy no longer permits" in problem(result)["detail"]
        assert "neighbor" not in result
        assert "Could not list folder" in caplog.text
    finally:
        registry.close()


@pytest.mark.parametrize("scratch_data", [False, True])
def test_direct_read_obeys_the_bound_scratch_generation_policy(tmp_path, scratch_data):
    """Only the session owner can make private basenames ordinary scratch data."""
    scratch = tmp_path / "private" / "scratch"
    scratch.mkdir(parents=True)
    selected = scratch / ".env"
    selected.write_text("owned scratch data")
    manager = PermissionManager(tmp_path, yolo=True)
    manager.bind_temporary_directory(PathHolder(scratch), ordinary_names=lambda: scratch_data)
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=manager)
    try:
        output = registry.call("read_text_file", json.dumps({"path": str(selected)}))
        result = json.loads(output)
        if scratch_data:
            assert result["result"]["content"] == "owned scratch data"
        else:
            assert result["problem"]["code"] == "tool.denied"
            assert "owned scratch data" not in output
    finally:
        registry.close()


def test_list_folder_reports_failures(tmp_path):
    """Listing reports an invalid folder as a readable tool result."""
    assert problem(list_folder(str(tmp_path / "missing")))["code"] == "filesystem.list_failed"


def test_file_navigation_reports_successful_instruction_context_changes(tmp_path, monkeypatch):
    """File navigation changes instruction scope without changing the runtime path root."""
    nested = tmp_path / "nested"
    nested.mkdir()
    target = nested / "file.txt"
    target.write_text("content", encoding="utf-8")
    manager = InstructionsManager.discover(tmp_path)
    manager.set_runtime_environment(RuntimeEnvironment(tmp_path, tmp_path / "scratch"))
    interaction = ConsoleInteraction()

    tool_registry.call(
        "list_folder",
        json.dumps({"path": "$WORKSPACE_ROOT/nested"}),
        interaction=interaction,
        instructions_manager=manager,
    )
    assert manager.working_directory == nested.resolve()
    assert manager.virtual_paths.resolve(".") == str(tmp_path.resolve())

    tool_registry.call(
        "write_text_file",
        json.dumps({"path": "$WORKSPACE_ROOT/nested/written.txt", "content": "content"}),
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
        "version": content_digest("hello"),
        "content": "hello",
        "size_bytes": 5,
        "start_byte": 0,
        "end_byte": 5,
        "included_bytes": 5,
        "truncated": False,
        "start_line": 1,
        "end_line": 1,
    }
    empty_result = json.loads(read_text_file(str(empty)))
    assert empty_result["content"] == ""
    assert empty_result["version"] == content_digest("")
    assert empty_result["size_bytes"] == 0
    assert problem(read_text_file(str(binary)))["code"] == "filesystem.binary_file"
    assert (
        problem(read_text_file(str(tmp_path / "missing.txt")))["code"] == "filesystem.path_missing"
    )


def test_read_text_file_reports_unexpected_io_failures(tmp_path, monkeypatch, caplog):
    """Unexpected read errors retain the failed-read diagnostic and are logged once."""
    reader = MagicMock(side_effect=OSError("read unavailable"))
    (tmp_path / "source.txt").write_text("content", encoding="utf-8")
    monkeypatch.setattr("loop.tools.files.read_bounded_text_stream", reader)
    result = problem(read_text_file(str(tmp_path / "source.txt")))
    assert result["code"] == "filesystem.read_failed"
    assert "read unavailable" in result["detail"]
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert "Could not read file" in caplog.records[0].message


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


@pytest.mark.parametrize("start_byte", [None, 0])
def test_read_text_file_stops_at_lines_before_the_byte_ceiling(tmp_path, start_byte):
    """A byte ceiling preserves complete lines and exposes a line continuation."""
    source = tmp_path / "unicode.txt"
    source.write_text("a€\nb\n", encoding="utf-8")

    first = json.loads(read_text_file(source, max_bytes=5))
    second = json.loads(
        read_text_file(
            source, start_line=first["next_start_line"], start_byte=start_byte, max_bytes=5
        )
    )

    assert first["content"] == "a€\n"
    assert first["end_byte"] == 5
    assert first["truncation_reason"] == "bytes"
    assert first["next_start_line"] == 2
    assert second["content"] == "b\n"


def test_read_text_file_automatically_pages_oversized_lines(tmp_path):
    """An oversized line returns content immediately and continues by byte offset."""
    source = tmp_path / "unicode.txt"
    content = "€long first line\nnext line\n"
    source.write_text(content, encoding="utf-8")

    first = json.loads(read_text_file(source, max_bytes=2))
    second = json.loads(read_text_file(source, start_line=first["next_start_line"], max_bytes=20))
    definition = next(
        definition
        for definition in tool_registry.definitions()
        if definition.name == "read_text_file"
    )

    assert first["content"] == "€long first line\n"
    assert first["requested_max_bytes"] == 2
    assert first["effective_max_bytes"] == len(first["content"].encode())
    assert first["next_start_line"] == 2
    assert second["content"] == "next line\n"
    assert second["start_byte"] == first["end_byte"]
    assert second["version"] == first["version"]
    assert "start_byte" in definition.parameters["properties"]
    for name in ("start_line", "start_byte", "max_lines"):
        description = definition.parameters["properties"][name]["description"]
        assert "JSON null (unquoted)" in description


@pytest.mark.parametrize("spelling", ["None", "nOnE", "null", "NULL"])
def test_read_text_file_accepts_optional_null_spellings_without_changing_paths(
    tmp_path, monkeypatch, spelling
):
    """String nulls behave like JSON null while a file named None stays addressable."""
    source = tmp_path / "None"
    source.write_text("one\ntwo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    spelled = json.loads(
        read_text_file("None", start_line=spelling, start_byte=4, max_lines=spelling)
    )
    actual = json.loads(read_text_file("None", start_line=None, start_byte=4, max_lines=None))
    assert spelled == actual
    assert spelled["content"] == "two\n"
    assert problem(read_text_file("None", start_line=" None "))["code"] == "tool.invalid_arguments"


@pytest.mark.parametrize("spelling", ["None", "nOnE", "null", "NULL"])
def test_optional_null_spelling_does_not_bypass_file_authorization(tmp_path, monkeypatch, spelling):
    """A normalized range cannot grant a read that the permission manager denies."""
    source = tmp_path / "secret"
    source.write_text("private", encoding="utf-8")
    monkeypatch.undo()
    permissions = PermissionManager(tmp_path)
    permissions.set_default(Action.FILESYSTEM_READ, Decision.DENY, scope=PolicyScope.SESSION)
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=permissions)
    output = registry.call(
        "read_text_file",
        json.dumps({"path": str(source), "start_line": spelling, "max_lines": spelling}),
        interaction=ConsoleInteraction(),
    )
    assert problem(output)["code"] == "tool.denied"
    assert "private" not in output


@pytest.mark.parametrize("spelling", ["None", "nOnE", "null", "NULL"])
def test_search_text_preserves_literal_globs_when_null_spelling_is_supported(
    tmp_path, monkeypatch, spelling
):
    """An omitted filter and literal sentinel globs remain distinct through public dispatch."""
    (tmp_path / "None").write_text("needle", encoding="utf-8")
    (tmp_path / "other.txt").write_text("needle", encoding="utf-8")
    (tmp_path / "null").write_text("needle", encoding="utf-8")
    engine = MagicMock(return_value=([], False))
    monkeypatch.setattr("loop.tools.files.search_text_paths", engine)

    assert json.loads(search_text(tmp_path, "needle", include=spelling))["matches"] == []
    assert {item.name for item in engine.call_args.args[0]} == {"None", "null", "other.txt"}
    assert json.loads(search_text(tmp_path, "needle", include=["None"]))["matches"] == []
    assert [item.name for item in engine.call_args.args[0]] == ["None"]
    assert json.loads(search_text(tmp_path, "needle", include=["null"]))["matches"] == []
    assert [item.name for item in engine.call_args.args[0]] == ["null"]


@pytest.mark.parametrize("character", ["a", "€"])
def test_read_text_file_pages_a_line_above_the_hard_limit(tmp_path, character):
    """A genuinely oversized line remains bounded and resumes without skipping bytes."""
    source = tmp_path / "long-line.txt"
    width = len(character.encode())
    content = character * (constants.MAX_TOOL_CONTENT_BYTES // width) + "€tail\n"
    source.write_text(content, encoding="utf-8")

    first = json.loads(read_text_file(source, max_bytes=2))
    second = json.loads(read_text_file(source, start_byte=first["next_start_byte"], max_bytes=10))
    assert first["content"] + second["content"] == content
    assert first["included_bytes"] == constants.MAX_TOOL_CONTENT_BYTES // width * width
    assert first["requested_max_bytes"] == 2
    assert first["effective_max_bytes"] == constants.MAX_TOOL_CONTENT_BYTES
    assert "next_start_line" not in first
    assert second["start_byte"] == first["end_byte"]
    assert second["version"] == first["version"]


def test_physical_file_read_preserves_virtual_receipt_and_edit_guards(tmp_path):
    """A contained physical path records the same read authority as its logical spelling."""
    source = tmp_path / "observed.txt"
    content = "visible content"
    source.write_text(content, encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(tmp_path, scratch))
    arguments = json.dumps({"path": str(source)})
    output = tool_registry.call(
        "read_text_file",
        arguments,
        interaction=ConsoleInteraction(),
        instructions_manager=instructions,
    )
    result = json.loads(output)
    logical = f"{VirtualPath.WORKSPACE}/{source.name}"
    assert result["ok"] and result["result"]["path"] == logical
    assert result["read_versions"][logical] == result["result"]["version"]
    observations = FileObservations([], instructions.virtual_paths)
    observations.observe_result(
        ToolCall(call_id="read", name="read_text_file", arguments=arguments), output
    )
    operations = (
        Operation(
            tool_id="edit_text_file",
            action=Action.FILESYSTEM_REPLACE,
            target=FileTarget(
                path=str(source), expected_exists=True, expected_digest=sha256_digest(content)
            ),
        ),
    )
    assert observations.check(operations, {"old_content": "visible"}) is None
    assert (
        observations.check(operations, {"old_content": "unseen"}).code
        == "filesystem.unread_content"
    )


def test_byte_pages_preserve_visible_content_guards(tmp_path):
    """Partial byte pages cannot authorize unseen edits or full replacement."""
    source = tmp_path / "long-line.txt"
    content = "a" * constants.MAX_TOOL_CONTENT_BYTES + "€tail\n"
    source.write_text(content, encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(tmp_path, scratch))
    logical = f"{VirtualPath.WORKSPACE}/{source.name}"
    operations = (
        Operation(
            tool_id="edit_text_file",
            action=Action.FILESYSTEM_REPLACE,
            target=FileTarget(
                path=str(source), expected_exists=True, expected_digest=sha256_digest(content)
            ),
        ),
    )
    observations = FileObservations([], instructions.virtual_paths)

    def capture(arguments):
        """Deliver one actual public read and record only its attested visible bytes."""
        request = json.dumps({"path": logical, **arguments})
        output = tool_registry.call(
            "read_text_file",
            request,
            interaction=ConsoleInteraction(),
            instructions_manager=instructions,
        )
        observations.observe_result(
            ToolCall(call_id="read", name="read_text_file", arguments=request), output
        )
        return json.loads(output)["result"]

    first = capture({"max_bytes": 2})
    assert (
        observations.check(operations, {"old_content": "€tail"}).code == "filesystem.unread_content"
    )
    assert (
        observations.check(operations, {"replace_all": True}).code == "filesystem.incomplete_read"
    )
    second = capture({"start_byte": first["next_start_byte"], "max_bytes": 10})
    assert first["content"] + second["content"] == content
    assert observations.check(operations, {"old_content": "€tail"}) is None
    assert observations.check(operations, {"replace_all": True}) is None
    changed = "b" * constants.MAX_TOOL_CONTENT_BYTES + "€tail\n"
    source.write_text(changed, encoding="utf-8")
    changed_page = capture({"start_byte": first["next_start_byte"], "max_bytes": 10})
    assert changed_page["version"] != first["version"]
    changed_operations = (
        Operation(
            tool_id="edit_text_file",
            action=Action.FILESYSTEM_REPLACE,
            target=FileTarget(
                path=str(source), expected_exists=True, expected_digest=sha256_digest(changed)
            ),
        ),
    )
    assert (
        observations.check(changed_operations, {"old_content": "b"}).code
        == "filesystem.unread_content"
    )
    assert (
        observations.check(changed_operations, {"replace_all": True}).code
        == "filesystem.incomplete_read"
    )


def test_result_limited_file_pages_preserve_visible_content_guards(tmp_path):
    """A serialization-limited page preserves receipts without exposing unseen content."""
    source = tmp_path / "escaped.txt"
    content = '"' * constants.MAX_TOOL_CONTENT_BYTES + "unseen"
    source.write_text(content, encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(tmp_path, scratch))
    logical = f"{VirtualPath.WORKSPACE}/{source.name}"
    operations = (
        Operation(
            tool_id="edit_text_file",
            action=Action.FILESYSTEM_REPLACE,
            target=FileTarget(
                path=str(source), expected_exists=True, expected_digest=sha256_digest(content)
            ),
        ),
    )
    observations = FileObservations([], instructions.virtual_paths)

    def capture(arguments):
        """Bound one real tool result and record only its delivered file range."""
        request = json.dumps({"path": logical, **arguments})
        raw = tool_registry.call(
            "read_text_file",
            request,
            interaction=ConsoleInteraction(),
            instructions_manager=instructions,
        )
        bounded, handle = bound_tool_result(raw, "synthetic")
        observations.observe_result(
            ToolCall(call_id="read", name="read_text_file", arguments=request), bounded
        )
        payload = json.loads(bounded)
        return payload["result"], handle or payload.get("handle")

    first, handle = capture({"max_bytes": 2})
    assert handle is not None
    assert first["truncation_reason"] == "result_limit"
    assert first["requested_max_bytes"] == 2
    assert first["effective_max_bytes"] == first["included_bytes"]
    assert (
        observations.check(operations, {"old_content": "unseen"}).code
        == "filesystem.unread_content"
    )
    assert (
        observations.check(operations, {"replace_all": True}).code == "filesystem.incomplete_read"
    )
    second, _ = capture({"start_byte": first["next_start_byte"]})
    assert first["content"] + second["content"] == content
    assert observations.check(operations, {"old_content": "unseen"}) is None
    assert observations.check(operations, {"replace_all": True}) is None


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


def test_public_search_ignores_forged_virtual_environment_ripgrep(tmp_path, monkeypatch):
    """An unverified virtual environment helper cannot forge public search results."""
    source = tmp_path / "source.txt"
    source.write_text("actual needle\n", encoding="utf-8")
    binary = tmp_path / "venv" / "bin" / "rg"
    binary.parent.mkdir(parents=True)
    marker = tmp_path / "forged-helper-ran"
    binary.write_text(f"#!/bin/sh\nprintf forged > {marker}\n", encoding="utf-8")
    binary.chmod(0o755)
    backend = MagicMock()
    backend.system_read_roots = ()
    backend.installed_tool_roots.return_value = ((binary.parent,), ())
    backend.managed_tool_root.return_value = False
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )
    monkeypatch.setattr("loop.execution.facade.ripgrep_path", lambda: str(binary))

    result = json.loads(
        registry.call("search_text", json.dumps({"path": str(source), "query": "needle"}))
    )

    assert result["result"]["matches"][0]["text"] == "actual needle"
    assert not marker.exists()
    backend.run_read_only_argv.assert_not_called()


@pytest.mark.parametrize("provenance", ["missing", "untrusted"])
def test_public_regex_search_uses_python_fallback_without_qualified_ripgrep(
    tmp_path, monkeypatch, provenance
):
    """Unavailable or untrusted ripgrep leaves bounded Python regex semantics in place."""
    source = tmp_path / "source.txt"
    source.write_text("before\n€ needle needle\nafter\nNEEDLE\n", encoding="utf-8")
    executable = tmp_path / "rg"
    executable.write_text("helper", encoding="utf-8")
    backend = MagicMock()
    backend.system_read_roots = ()
    backend.installed_tool_roots.return_value = ((), ())
    if provenance == "missing":
        monkeypatch.setattr(
            "loop.execution.facade.ripgrep_path",
            MagicMock(side_effect=FileNotFoundError("missing")),
        )
    else:
        monkeypatch.setattr("loop.execution.facade.ripgrep_path", lambda: str(executable))
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )

    def search(query, **options):
        """Dispatch one public regex request through the normal registry."""
        return json.loads(
            registry.call(
                "search_text",
                json.dumps({"path": str(source), "query": query, "regex": True, **options}),
            )
        )

    lookbehind = search(r"(?<=€ )needle", context_lines=1)
    assert lookbehind["result"]["matches"][0]["column"] == 3
    assert lookbehind["result"]["matches"][0]["context"] == [
        {"line": 1, "text": "before"},
        {"line": 3, "text": "after"},
    ]
    assert search(r"(needle) \1")["result"]["matches"][0]["column"] == 3
    assert len(search("needle", case="insensitive", max_results=1)["result"]["matches"]) == 1
    assert search("needle", case="insensitive", max_results=1)["result"]["truncated"]
    assert search("(")["problem"]["code"] == "filesystem.invalid_search_pattern"
    backend.run_read_only_argv.assert_not_called()


def test_public_regex_search_uses_qualified_native_runner(tmp_path, monkeypatch):
    """A qualified helper receives regex patterns and reports its own dialect errors."""
    source = tmp_path / "source.txt"
    source.write_text("€ needle\n", encoding="utf-8")
    executable = tmp_path / "rg"
    executable.write_text("helper", encoding="utf-8")
    backend = MagicMock()
    backend.system_read_roots = (tmp_path,)

    def native_search(_executable, arguments, descriptors, _deadline):
        """Return a selected-file event or the native parser's pattern error."""
        if "(?<=€ )needle" in arguments:
            return ProcessCapture(
                ProcessCaptureStatus.COMPLETED,
                exit_code=2,
                stderr="look-around is not supported",
            )
        event = {
            "type": "match",
            "data": {
                "path": {"text": f"/dev/fd/{descriptors[0]:05d}"},
                "lines": {"text": "€ needle\n"},
                "line_number": 1,
                "submatches": [{"start": 4}],
            },
        }
        return ProcessCapture(ProcessCaptureStatus.COMPLETED, exit_code=0, stdout=json.dumps(event))

    backend.run_read_only_argv.side_effect = native_search
    monkeypatch.setattr("loop.execution.facade.ripgrep_path", lambda: str(executable))
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )

    def search(query):
        """Dispatch a regex pattern through the public search tool."""
        return json.loads(
            registry.call(
                "search_text",
                json.dumps({"path": str(source), "query": query, "regex": True}),
            )
        )

    match = search("needle$")
    assert match["result"]["matches"][0]["column"] == 3
    assert "--fixed-strings" not in backend.run_read_only_argv.call_args.args[1]
    assert search(r"(?<=€ )needle")["problem"]["code"] == "filesystem.invalid_search_pattern"
    assert backend.run_read_only_argv.call_count == 2


def test_public_search_reports_native_integrity_failure(tmp_path, monkeypatch):
    """A failed native authority check cannot turn into a successful Python search."""
    source = tmp_path / "source.txt"
    source.write_text("needle\n", encoding="utf-8")
    executable = tmp_path / "rg"
    executable.write_text("helper", encoding="utf-8")
    backend = MagicMock()
    backend.system_read_roots = (tmp_path,)
    backend.run_read_only_argv.side_effect = RuntimeError("Native search authority is invalid.")
    host = MagicMock()
    monkeypatch.setattr("loop.execution.facade.ripgrep_path", lambda: str(executable))
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(backend, host),
    )

    result = json.loads(
        registry.call("search_text", json.dumps({"path": str(source), "query": "needle"}))
    )

    assert result["problem"]["code"] == "filesystem.search_failed"
    assert "result" not in result
    backend.run_read_only_argv.assert_called_once()
    host.run_host_command.assert_not_called()


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


def test_write_text_file_reports_directory_and_special_targets_during_planning(tmp_path):
    """Unsupported existing write targets return structured planning problems."""
    fifo = tmp_path / "events"
    os.mkfifo(fifo)

    assert problem(write_text_file(tmp_path, "content"))["code"] == "filesystem.path_not_file"
    assert problem(write_text_file(fifo, "content"))["code"] == "filesystem.unsupported_path"
    parent_file = tmp_path / "parent-file"
    parent_file.write_text("not a directory", encoding="utf-8")
    assert problem(write_text_file(parent_file / "child", "content"))["code"] == (
        "tool.planning_failed"
    )


def test_write_text_file_removes_its_temporary_file_when_commit_fails(tmp_path, monkeypatch):
    """An atomic replacement failure does not leave staged content in the destination folder."""
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    monkeypatch.setattr(os, "replace", MagicMock(side_effect=OSError("commit failed")))

    result = write_text_file(target, "content")

    assert problem(result)["detail"] == "commit failed"
    assert set(tmp_path.iterdir()) == {target}
    assert target.read_text(encoding="utf-8") == "old"


def test_write_text_file_cancels_when_approved_target_changes(tmp_path, monkeypatch):
    """Replacement consumes the approved digest instead of overwriting changed content."""
    target = tmp_path / "target.txt"
    target.write_text("approved", encoding="utf-8")

    def approve_changed(*_args, **_kwargs):
        """Change the authorized original after its preview has been prepared."""
        target.write_text("changed", encoding="utf-8")
        return True

    monkeypatch.setattr(
        PermissionManager, "request_permission", MagicMock(side_effect=approve_changed)
    )

    result = write_text_file(target, "replacement")

    assert problem(result)["detail"].startswith("The target changed after approval")
    assert target.read_text(encoding="utf-8") == "changed"


def test_write_planning_rejects_kind_and_metadata_changes_during_inspection(tmp_path, monkeypatch):
    """Authorized inspection refuses a changed type or metadata at its descriptor boundary."""
    target = tmp_path / "target.txt"
    target.write_text("approved", encoding="utf-8")
    original = files_module.open_selected_file

    def changed_kind(root, path, identity):
        """Substitute a directory immediately before opening the selected file."""
        target.unlink()
        target.mkdir()
        return original(root, path, identity)

    monkeypatch.setattr(files_module, "open_selected_file", changed_kind)
    assert problem(write_text_file(target, "replacement"))["code"] == "tool.planning_failed"
    assert target.is_dir()
    target.rmdir()
    target.write_text("approved", encoding="utf-8")
    target.chmod(0o644)

    def changed_metadata(root, path, identity):
        """Change the selected inode's mode after capture but before reading."""
        descriptor = original(root, path, identity)
        target.chmod(0o600)
        return descriptor

    monkeypatch.setattr(files_module, "open_selected_file", changed_metadata)
    assert problem(write_text_file(target, "replacement"))["code"] == "tool.planning_failed"
    assert target.read_text(encoding="utf-8") == "approved"


def test_write_preview_rejects_metadata_changed_during_content_read(tmp_path, monkeypatch):
    """A preview cannot publish bytes when the bound source changes during inspection."""
    target = tmp_path / "target"
    target.write_text("approved")
    target.chmod(0o644)
    identity = target.stat().st_ino
    original = os.fdopen

    @contextmanager
    def changed_after_read(descriptor, mode, *args, **kwargs):
        """Change only the owned selected file after its content has been consumed."""
        selected = os.fstat(descriptor).st_ino == identity and mode == "rb"
        with original(descriptor, mode, *args, **kwargs) as stream:
            yield stream
            if selected:
                target.chmod(0o600)

    monkeypatch.setattr(os, "fdopen", changed_after_read)
    result = write_text_file(target, "replacement")
    assert problem(result)["code"] == "tool.planning_failed"
    assert "target changed during planning" in problem(result)["detail"]
    assert target.read_text() == "approved"


def test_write_and_edit_denials_do_not_read_unauthorized_existing_content(tmp_path, monkeypatch):
    """Denied mutation planning never inspects unauthorized file contents or metadata."""
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside secret", encoding="utf-8")
    git = tmp_path / ".git"
    git.mkdir()
    protected = git / "config"
    protected.write_text("protected secret", encoding="utf-8")
    protected_alias = tmp_path / "protected-alias"
    protected_alias.symlink_to(git, target_is_directory=True)
    original = Path.read_bytes
    reads = []
    original_lstat = Path.lstat
    metadata_reads = []

    def tracked_read(path):
        reads.append(path)
        return original(path)

    def tracked_lstat(path):
        metadata_reads.append(path)
        return original_lstat(path)

    monkeypatch.setattr(Path, "read_bytes", tracked_read)
    monkeypatch.setattr(Path, "lstat", tracked_lstat)

    edit_result = edit_text_file(outside, "outside", "changed")
    write_result = write_text_file(protected, "replacement")
    delete_result = delete_path(protected)
    alias_result = edit_text_file(protected_alias / "config", "protected", "changed")

    assert problem(edit_result)["code"] == "tool.denied"
    assert problem(write_result)["code"] == "tool.denied"
    assert problem(delete_result)["code"] == "tool.denied"
    assert problem(alias_result)["code"] == "tool.denied"
    assert reads == []
    assert metadata_reads == []
    assert outside.read_text(encoding="utf-8") == "outside secret"
    assert protected.read_text(encoding="utf-8") == "protected secret"


def test_atomic_replacement_preserves_executable_permission_bits(tmp_path):
    """Editing an executable regular file preserves its existing permission bits."""
    target = tmp_path / "script.sh"
    target.write_text("#!/bin/sh\necho old\n", encoding="utf-8")
    target.chmod(0o755)

    assert "(1 replacement)" in edit_text_file(target, "old", "new")

    assert target.stat().st_mode & 0o777 == 0o755


def test_write_executor_requires_authorized_state_and_existing_replacement(tmp_path, monkeypatch):
    """The executor fails closed without a plan or when a replacement disappears."""
    context = ToolContext(ConsoleInteraction(), "write_text_file")
    target = tmp_path / "target.txt"
    assert write_text_file_tool(context, str(target), "new").detail.startswith(
        "Authorized file-state precondition is missing"
    )

    target.write_text("old", encoding="utf-8")

    def approve_removed(*_args, **_kwargs):
        """Remove the approved target before mutation starts."""
        target.unlink()
        return True

    monkeypatch.setattr(
        PermissionManager, "request_permission", MagicMock(side_effect=approve_removed)
    )
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
    monkeypatch.setattr(
        files_module, "open_selected_file", MagicMock(side_effect=OSError("read failed"))
    )

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

    def approve_changed(*_args, **_kwargs):
        """Change the authorized original after its preview has been prepared."""
        target.write_text("changed", encoding="utf-8")
        return True

    monkeypatch.setattr(
        PermissionManager, "request_permission", MagicMock(side_effect=approve_changed)
    )

    result = edit_text_file(target, "approved", "replacement")

    assert problem(result)["detail"].startswith("The target changed after approval")
    assert target.read_text(encoding="utf-8") == "changed"


def test_edit_text_file_reports_commit_failures_and_cleans_up_staging(tmp_path, monkeypatch):
    """Exact editing preserves original content and reports a failed atomic commit."""
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    monkeypatch.setattr(os, "replace", MagicMock(side_effect=OSError("commit failed")))

    result = edit_text_file(target, "old", "new")

    assert problem(result)["detail"] == "commit failed"
    assert target.read_text(encoding="utf-8") == "old"
    assert set(tmp_path.iterdir()) == {target}


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


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("path", ["$TMPDIR", "$TMPDIR/", "$TMPDIR/."])
def test_delete_path_preserves_the_bound_session_root(tmp_path, saved, path):
    """Root deletion is forbidden while child cleanup and explicit retirement stay usable."""
    store = (
        SQLiteSessionStore(tmp_path / "sessions.db", workspace_id="workspace") if saved else None
    )
    manager = SessionManager(session_store=store)
    root = manager.temporary_directory.get()
    permissions = PermissionManager(tmp_path)
    permissions.bind_temporary_directory(manager.temporary_directory)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, manager.temporary_directory)
    )
    registry = ToolRegistry([delete_path_tool], permission_manager=permissions)
    binding = manager.store.working_binding(manager.session.id)
    child = root / "child"
    child.write_text("obsolete")
    alias = root / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with manager.execution():
        denied = registry.call(
            "delete_path",
            json.dumps({"path": path}),
            interaction=ConsoleInteraction(),
            instructions_manager=instructions,
        )
        assert problem(denied)["code"] == "tool.denied"
        assert root.is_dir() and child.read_text() == "obsolete"
        assert manager.store.working_binding(manager.session.id) == binding
        for target in ("$TMPDIR/child", "$TMPDIR/alias"):
            output = registry.call(
                "delete_path",
                json.dumps({"path": target}),
                interaction=ConsoleInteraction(),
                instructions_manager=instructions,
            )
            assert json.loads(output)["ok"]
        assert root.is_dir() and not child.exists() and not alias.is_symlink()
    with manager.execution() as captured:
        assert captured == root
    manager.discard_working_directory()
    assert manager.temporary_directory.get() != root
    assert not root.exists()


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
    monkeypatch.setattr(os, "unlink", MagicMock(side_effect=OSError("access denied")))

    assert problem(delete_path(target))["detail"] == "access denied"
    assert target.exists()


def test_recursive_delete_rejects_workspace_root_containing_protected_git(tmp_path):
    """Recursive ancestor permission cannot override a protected Git descendant."""
    git_config = tmp_path / ".git" / "config"
    git_config.parent.mkdir()
    git_config.write_text("keep", encoding="utf-8")
    ordinary = tmp_path / "ordinary.txt"
    ordinary.write_text("keep", encoding="utf-8")

    result = delete_path(tmp_path)

    assert problem(result)["code"] == "tool.denied"
    assert git_config.read_text(encoding="utf-8") == "keep"
    assert ordinary.read_text(encoding="utf-8") == "keep"


def test_recursive_delete_rejects_manifest_changes_after_approval(tmp_path):
    """A recursive deletion fails closed when its approved object set changes."""
    target = tmp_path / "tree"
    target.mkdir()
    approved = target / "approved.txt"
    approved.write_text("approved", encoding="utf-8")

    output = tool_registry.call(
        "delete_path",
        json.dumps({"path": str(target)}),
        interaction=ConsoleInteraction(),
        execution_started=lambda: (target / "late.txt").write_text("late", encoding="utf-8"),
    )

    assert problem(output)["code"] == "filesystem.delete_failed"
    assert approved.exists()
    assert (target / "late.txt").exists()


def test_recursive_delete_handles_empty_trees_and_directory_symlinks(tmp_path):
    """Recursive contracts cover empty trees and never traverse directory symlinks."""
    empty = tmp_path / "empty"
    empty.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    kept = outside / "kept.txt"
    kept.write_text("keep", encoding="utf-8")
    tree = tmp_path / "tree-with-link"
    tree.mkdir()
    (tree / "outside-link").symlink_to(outside, target_is_directory=True)

    assert delete_path(empty) == f"Successfully deleted path '{empty}'."
    assert delete_path(tree) == f"Successfully deleted path '{tree}'."
    assert kept.read_text(encoding="utf-8") == "keep"

    special = tmp_path / "special-tree"
    special.mkdir()
    os.mkfifo(special / "events")
    special_result = delete_path(special)
    assert problem(special_result)["code"] == "tool.planning_failed"
    assert special.exists()


def test_recursive_delete_rejects_content_and_kind_changes(tmp_path):
    """Recursive deletion requires an unchanged manifest and selected directory identity."""
    changed = tmp_path / "changed"
    changed.mkdir()
    child = changed / "child.txt"
    child.write_text("approved", encoding="utf-8")
    changed_output = tool_registry.call(
        "delete_path",
        json.dumps({"path": str(changed)}),
        interaction=ConsoleInteraction(),
        execution_started=lambda: child.write_text("changed", encoding="utf-8"),
    )
    assert problem(changed_output)["code"] == "filesystem.delete_failed"
    assert child.read_text(encoding="utf-8") == "changed"

    rebound = tmp_path / "rebound"
    rebound.mkdir()

    def replace_tree_with_file():
        rebound.rmdir()
        rebound.write_text("replacement", encoding="utf-8")

    rebound_output = tool_registry.call(
        "delete_path",
        json.dumps({"path": str(rebound)}),
        interaction=ConsoleInteraction(),
        execution_started=replace_tree_with_file,
    )
    assert problem(rebound_output)["code"] == "filesystem.delete_failed"
    assert rebound.read_text(encoding="utf-8") == "replacement"


@pytest.mark.parametrize("replacement_kind", ["file", "directory", "symlink"])
def test_file_delete_rejects_replacement_after_approval(tmp_path, replacement_kind):
    """Deletion cannot unlink a file path rebound to any unapproved object."""
    target = tmp_path / "target"
    target.write_text("approved", encoding="utf-8")
    link_target = tmp_path / "link-target"
    link_target.write_text("keep", encoding="utf-8")

    def replace_target():
        target.unlink()
        if replacement_kind == "directory":
            target.mkdir()
        elif replacement_kind == "symlink":
            target.symlink_to(link_target)
        else:
            target.write_text("replacement", encoding="utf-8")

    output = tool_registry.call(
        "delete_path",
        json.dumps({"path": str(target)}),
        interaction=ConsoleInteraction(),
        execution_started=replace_target,
    )

    assert problem(output)["code"] == "filesystem.delete_failed"
    assert target.exists() or target.is_symlink()
    if replacement_kind == "symlink":
        assert target.is_symlink()
        assert link_target.read_text(encoding="utf-8") == "keep"


def test_file_delete_rejects_disappearance_and_executor_without_capability(tmp_path):
    """Deletion fails closed when a target disappears or no executable capability is supplied."""
    target = tmp_path / "target"
    target.write_text("approved", encoding="utf-8")
    output = tool_registry.call(
        "delete_path",
        json.dumps({"path": str(target)}),
        interaction=ConsoleInteraction(),
        execution_started=target.unlink,
    )
    assert problem(output)["code"] == "filesystem.delete_failed"

    context = ToolContext(ConsoleInteraction(), "delete_path")
    result = delete_path_tool(context, str(target))
    assert result.code == "filesystem.delete_failed"
    assert result.detail == "Authorized deletion capability is missing."

    unsupported = ToolContext(
        ConsoleInteraction(),
        "delete_path",
        permission_manager=PermissionManager(tmp_path),
        operations=(
            Operation(
                tool_id="delete_path",
                action=Action.FILESYSTEM_DELETE,
                target=FileTarget(
                    path=str(target), expected_exists=True, expected_kind="directory"
                ),
            ),
        ),
    )
    unsupported_result = delete_path_tool(unsupported, str(target))
    assert unsupported_result.detail == (
        "Authorized deletion capability has an unsupported target kind."
    )


def test_file_and_recursive_deletes_reject_rebound_parent_directories(tmp_path):
    """Deletion capabilities bind the canonical parent directory as well as the leaf."""
    parent = tmp_path / "parent"
    parent.mkdir()
    file_target = parent / "target"
    file_target.write_text("approved", encoding="utf-8")
    moved_parent = tmp_path / "moved-parent"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_target = outside / "target"
    outside_target.write_text("outside", encoding="utf-8")

    def rebind_file_parent():
        parent.rename(moved_parent)
        parent.symlink_to(outside, target_is_directory=True)

    file_output = tool_registry.call(
        "delete_path",
        json.dumps({"path": str(file_target)}),
        interaction=ConsoleInteraction(),
        execution_started=rebind_file_parent,
    )
    assert problem(file_output)["code"] == "filesystem.delete_failed"
    assert outside_target.read_text(encoding="utf-8") == "outside"
    assert (moved_parent / "target").read_text(encoding="utf-8") == "approved"

    recursive_parent = tmp_path / "recursive-parent"
    recursive_parent.mkdir()
    tree = recursive_parent / "tree"
    tree.mkdir()
    (tree / "child").write_text("approved", encoding="utf-8")
    moved_recursive_parent = tmp_path / "moved-recursive-parent"
    outside_recursive = tmp_path / "outside-recursive"
    outside_recursive.mkdir()
    outside_tree = outside_recursive / "tree"
    outside_tree.mkdir()
    (outside_tree / "child").write_text("outside", encoding="utf-8")

    def rebind_recursive_parent():
        recursive_parent.rename(moved_recursive_parent)
        recursive_parent.symlink_to(outside_recursive, target_is_directory=True)

    recursive_output = tool_registry.call(
        "delete_path",
        json.dumps({"path": str(tree)}),
        interaction=ConsoleInteraction(),
        execution_started=rebind_recursive_parent,
    )
    assert problem(recursive_output)["code"] == "filesystem.delete_failed"
    assert (outside_tree / "child").read_text(encoding="utf-8") == "outside"
    assert (moved_recursive_parent / "tree" / "child").read_text(encoding="utf-8") == ("approved")


def test_file_delete_rejects_a_missing_parent_after_approval(tmp_path):
    """A missing canonical parent invalidates the executable deletion capability."""
    parent = tmp_path / "parent"
    parent.mkdir()
    target = parent / "target"
    target.write_text("approved", encoding="utf-8")
    moved = tmp_path / "moved"

    output = tool_registry.call(
        "delete_path",
        json.dumps({"path": str(target)}),
        interaction=ConsoleInteraction(),
        execution_started=lambda: parent.rename(moved),
    )

    assert problem(output)["code"] == "filesystem.delete_failed"
    assert (moved / "target").read_text(encoding="utf-8") == "approved"


def test_delete_rejects_targets_rebound_during_authorization_phases(tmp_path, monkeypatch):
    """File execution and directory planning reject targets rebound during authorization."""
    file_target = tmp_path / "file-target"
    file_target.write_text("approved", encoding="utf-8")
    manager = tool_registry.permission_manager
    original_authorize = manager.authorize
    changed_file = False

    def change_file_after_inspection(operations, **kwargs):
        nonlocal changed_file
        result = original_authorize(operations, **kwargs)
        if not changed_file and operations and operations[0].action is Action.FILESYSTEM_DELETE:
            changed_file = True
            file_target.unlink()
            file_target.mkdir()
        return result

    monkeypatch.setattr(manager, "authorize", change_file_after_inspection)
    file_output = delete_path(file_target)
    assert problem(file_output)["code"] == "filesystem.delete_failed"
    assert file_target.is_dir()

    directory = tmp_path / "directory-target"
    directory.mkdir()
    changed_directory = False

    def change_directory_after_list(operations, **kwargs):
        nonlocal changed_directory
        result = original_authorize(operations, **kwargs)
        if not changed_directory and operations and operations[0].action is Action.FILESYSTEM_LIST:
            changed_directory = True
            directory.rmdir()
            directory.write_text("replacement", encoding="utf-8")
        return result

    monkeypatch.setattr(manager, "authorize", change_directory_after_list)
    directory_output = delete_path(directory)
    assert problem(directory_output)["code"] == "tool.planning_failed"
    assert directory.read_text(encoding="utf-8") == "replacement"


def test_file_delete_rejects_kind_change_during_capability_capture(tmp_path, monkeypatch):
    """File deletion fails closed when the target changes during state capture."""
    target = tmp_path / "target"
    target.write_text("approved", encoding="utf-8")
    original_open = files_module.open_selected_file

    def change_before_open(root, path, identity):
        """Replace the selected regular file before its content descriptor opens."""
        if path == target:
            target.unlink()
            target.mkdir()
        return original_open(root, path, identity)

    monkeypatch.setattr(files_module, "open_selected_file", change_before_open)
    output = delete_path(target)
    assert problem(output)["code"] == "tool.planning_failed"
    assert target.is_dir()


def test_recursive_delete_rejects_metadata_drift_between_planning_phases(tmp_path, monkeypatch):
    """A tree change after listing but before digest capture invalidates the plan."""
    directory = tmp_path / "tree"
    directory.mkdir()
    child = directory / "child.txt"
    child.write_text("approved", encoding="utf-8")
    manager = tool_registry.permission_manager
    original_authorize = manager.authorize
    changed = False

    read_authorizations = 0

    def change_after_prerequisite_read(operations, **kwargs):
        nonlocal read_authorizations
        nonlocal changed
        result = original_authorize(operations, **kwargs)
        if operations and operations[0].action is Action.FILESYSTEM_READ:
            read_authorizations += 1
        if not changed and read_authorizations == 2:
            changed = True
            child.write_text("changed", encoding="utf-8")
        return result

    monkeypatch.setattr(manager, "authorize", change_after_prerequisite_read)

    output = delete_path(directory)

    assert problem(output)["code"] == "filesystem.target_changed"
    assert child.read_text(encoding="utf-8") == "changed"


@pytest.mark.parametrize(
    "name", [".env", ".env.local", ".env.production", ".ssh/key", ".config/key"]
)
@pytest.mark.parametrize("yolo", [False, True])
def test_file_tools_enforce_hard_names_before_read_preview_or_mutation(tmp_path, name, yolo):
    """Hard data never reaches explicit reads, mutation previews, listing or search results."""
    source = tmp_path / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("hard-canary", encoding="utf-8")
    permissions = PermissionManager(tmp_path, yolo=yolo)
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=permissions)
    for tool_name, extra in (
        ("read_text_file", {}),
        ("write_text_file", {"content": "changed"}),
        ("edit_text_file", {"old_content": "hard", "new_content": "changed"}),
        ("delete_path", {}),
        ("search_text", {"query": "hard-canary"}),
    ):
        payload = json.loads(registry.call(tool_name, json.dumps({"path": str(source), **extra})))
        assert payload["ok"] is False
        assert payload["problem"]["code"] == "tool.denied"
        assert "hard-canary" not in json.dumps(payload)
    listing = json.loads(
        registry.call("list_folder", json.dumps({"path": str(tmp_path), "recursive": True}))
    )
    assert listing["ok"] is True
    assert name not in {entry["path"] for entry in listing["result"]}
    search = json.loads(
        registry.call("search_text", json.dumps({"path": str(tmp_path), "query": "hard-canary"}))
    )
    assert search["result"]["matches"] == []
    assert source.read_text(encoding="utf-8") == "hard-canary"


@pytest.mark.parametrize(
    "name", [".env.example", ".env.production.example", ".env.sample", ".env.template", "cache"]
)
def test_public_templates_and_undesignated_ignored_files_stay_readable(tmp_path, name):
    """Ignore discovery does not declassify or hard-deny explicit template/cache reads."""
    source = tmp_path / name
    source.write_text("public content", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(name + "\n", encoding="utf-8")
    assert json.loads(read_text_file(source))["content"] == "public content"


@pytest.mark.parametrize("tool_name", ["write_text_file", "edit_text_file", "delete_path"])
def test_pending_mutation_obeys_trusted_policy_reload(tmp_path, tool_name):
    """A trusted hard designation after approval prevents the pending mutation."""
    source = tmp_path / ".env.example"
    source.write_text("original", encoding="utf-8")
    policy = tmp_path / ".loop/permissions.yaml"
    policy.parent.mkdir()
    policy.write_text(PermissionConfiguration().model_dump_json(), encoding="utf-8")
    manager = PermissionManager(tmp_path, configuration_path=policy, yolo=True)
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=manager)

    def designate():
        """Reload the new exact hard designation after ordinary authorization."""
        configuration = PermissionConfiguration(
            limits=PolicyLimits(hard_denied_paths=(str(source),))
        )
        policy.write_text(configuration.model_dump_json(), encoding="utf-8")
        manager.reload()

    arguments = {"path": str(source)}
    if tool_name == "write_text_file":
        arguments["content"] = "changed"
    elif tool_name == "edit_text_file":
        arguments.update(old_content="original", new_content="changed")
    try:
        result = registry.call(tool_name, json.dumps(arguments), execution_started=designate)
        assert problem(result)["code"] == "tool.denied"
        assert source.read_text(encoding="utf-8") == "original"
    finally:
        registry.close()


@pytest.mark.parametrize("tool_name", ["write_text_file", "edit_text_file", "delete_path"])
def test_mutation_capability_requires_current_permission_authority(tmp_path, tool_name, caplog):
    """A capability without a current policy owner cannot open or mutate a file."""
    source = tmp_path / "source"
    source.write_text("original", encoding="utf-8")
    context = ToolContext(
        ConsoleInteraction(),
        tool_name,
        operations=(
            Operation(
                tool_id=tool_name,
                action=Action.FILESYSTEM_DELETE
                if tool_name == "delete_path"
                else Action.FILESYSTEM_REPLACE,
                target=FileTarget(
                    path=str(source),
                    expected_exists=True,
                    expected_kind="file",
                    expected_digest=sha256_digest(b"original"),
                ),
            ),
        ),
    )
    if tool_name == "write_text_file":
        result = write_text_file_tool(context, str(source), "changed")
    elif tool_name == "edit_text_file":
        result = edit_text_file_tool(context, str(source), "original", "changed")
    else:
        result = delete_path_tool(context, str(source))
    assert result.detail == "File permission authority is unavailable."
    assert result.title in caplog.text
    assert source.read_text(encoding="utf-8") == "original"


def test_trusted_designation_rechecks_cached_file_results_and_declassification(tmp_path):
    """A retained artifact keeps physical source authority across trusted policy reloads."""
    source = tmp_path / ".env.example"
    source.write_text('"' * constants.MAX_TOOL_CONTENT_BYTES, encoding="utf-8")
    policy = tmp_path / ".loop/permissions.yaml"
    policy.parent.mkdir()
    policy.write_text(PermissionConfiguration().model_dump_json(), encoding="utf-8")
    manager = PermissionManager(tmp_path, configuration_path=policy)
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=manager)
    original = json.loads(registry.call("read_text_file", json.dumps({"path": str(source)})))
    handle = original["handle"]
    assert str(source) in cached_metadata(handle)["file_sources"]
    assert "file_sources" not in original
    denied = PermissionConfiguration(limits=PolicyLimits(hard_denied_paths=(str(source),)))
    policy.write_text(denied.model_dump_json(), encoding="utf-8")
    manager.reload()
    for tool_name, args in (
        ("read_text_file", {"path": str(source)}),
        ("read_cached_content", {"handle": handle}),
    ):
        result = json.loads(registry.call(tool_name, json.dumps(args)))
        assert result["problem"]["code"] == "tool.denied"
    policy.write_text(PermissionConfiguration().model_dump_json(), encoding="utf-8")
    manager.reload()
    resumed = json.loads(
        registry.call("read_cached_content", json.dumps({"handle": handle, "max_bytes": 100}))
    )
    assert resumed["ok"] is True
    assert resumed["result"]["content"]
    (tmp_path / ".env.local").write_text("still denied", encoding="utf-8")
    assert (
        json.loads(
            registry.call("read_text_file", json.dumps({"path": str(tmp_path / ".env.local")}))
        )["problem"]["code"]
        == "tool.denied"
    )


@pytest.mark.parametrize("alias_kind", ["outside", "protected"])
def test_direct_tools_refuse_incomplete_or_protected_hardlinks(tmp_path, alias_kind):
    """Direct tools refuse incomplete alias sets and every known alias of hard data."""
    original = tmp_path.parent / "outside" if alias_kind == "outside" else tmp_path / ".env"
    original.write_text("alias-canary", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.hardlink_to(original)
    for invoke in (
        lambda: read_text_file(alias),
        lambda: write_text_file(alias, "changed"),
        lambda: delete_path(alias),
    ):
        assert problem(invoke())["code"] == "tool.denied"
    assert original.read_text(encoding="utf-8") == "alias-canary"


def test_direct_tools_allow_complete_benign_aliases_without_mutating_other_names(tmp_path):
    """Reads and atomic replacement/deletion accept verified links and preserve other names."""
    source = tmp_path / "source"
    source.write_text("ordinary")
    alias = tmp_path / "alias"
    alias.hardlink_to(source)
    assert json.loads(read_text_file(alias))["content"] == "ordinary"
    assert "Successfully edited" in edit_text_file(alias, "ordinary", "edited")
    assert source.read_text() == "ordinary"
    alias.unlink()
    alias.hardlink_to(source)
    assert "Successfully wrote" in write_text_file(alias, "replacement")
    assert source.read_text() == "ordinary"
    alias.unlink()
    alias.hardlink_to(source)
    assert "Successfully deleted" in delete_path(alias)
    assert source.read_text() == "ordinary"


def test_recursive_delete_accounts_for_its_own_benign_alias_removals(tmp_path):
    """Deleting one approved alias does not invalidate the next name of that same inode."""
    directory = tmp_path / "tree"
    directory.mkdir()
    first = directory / "first"
    first.write_text("ordinary")
    (directory / "second").hardlink_to(first)
    assert "Successfully deleted" in delete_path(directory)
    assert not directory.exists()


def test_file_functions_fail_closed_without_boundary_owner(tmp_path, caplog):
    """Calling a file function without dispatch cannot obtain raw filesystem authority."""
    source = tmp_path / "source"
    source.write_text("public", encoding="utf-8")
    context = ToolContext(None, "read_text_file")
    result = files_module.read_text_file(context, str(source))
    assert result.code == "filesystem.read_failed"
    assert "permission authority is unavailable" in result.detail
    result = files_module.list_folder(context, str(tmp_path))
    assert result.code == "filesystem.list_failed"
    assert "permission authority is unavailable" in result.detail
    assert "Could not read file" in caplog.text
    assert "Could not list folder" in caplog.text


def test_read_rechecks_policy_after_consuming_descriptor(tmp_path, monkeypatch, caplog):
    """A hard designation added during reading suppresses the result and its receipt."""
    source = tmp_path / "public"
    source.write_text("canary", encoding="utf-8")
    secret = tmp_path / ".env"
    secret.write_text("secret", encoding="utf-8")
    original = files_module.read_bounded_text_stream

    def rebind_after_read(stream, **kwargs):
        """Rebind the original source before the final hard-boundary check."""
        result = original(stream, **kwargs)
        source.unlink()
        source.symlink_to(secret)
        return result

    monkeypatch.setattr(files_module, "read_bounded_text_stream", rebind_after_read)
    output = read_text_file(source)
    assert problem(output)["code"] == "filesystem.read_failed"
    assert "File policy changed while reading" in problem(output)["detail"]
    assert "canary" not in output
    assert "Could not read file" in caplog.text


def test_write_planner_rejects_kind_change_before_content_capture(tmp_path, monkeypatch):
    """An admitted inspection cannot treat a changed directory as a regular-file source."""
    source = tmp_path / "file"
    source.write_text("content", encoding="utf-8")
    plan = Tool.get_declaration(write_text_file_tool).plan({"path": str(source), "content": "new"})
    original = Path.lstat
    changed = False

    def replace_after_kind(path, *args, **kwargs):
        """Replace the object after the initial kind inspection returns its metadata."""
        nonlocal changed
        status = original(path, *args, **kwargs)
        if path == source and not changed:
            changed = True
            source.unlink()
            source.mkdir()
        return status

    monkeypatch.setattr(Path, "lstat", replace_after_kind)
    with pytest.raises(ValueError, match="not a regular file"):
        plan.continuation()


def test_file_delete_planner_rechecks_kind_at_the_capture_phase(tmp_path):
    """The final file-only deletion phase rejects a directory substituted after inspection."""
    source = tmp_path / "file"
    source.write_text("content", encoding="utf-8")
    plan = Tool.get_declaration(delete_path_tool).plan({"path": str(source)}).continuation()
    source.unlink()
    source.mkdir()
    with pytest.raises(ProblemException) as error:
        plan.continuation()
    assert error.value.problem.code == "filesystem.target_changed"


def test_mutation_preview_rejects_a_new_hardlink_before_open(tmp_path, monkeypatch):
    """A new alias between metadata capture and opening cannot enter a mutation preview."""
    source = tmp_path / "file"
    source.write_text("content", encoding="utf-8")
    original = files_module.open_selected_file

    def link_before_open(root, path, identity):
        """Introduce another name only at the selected descriptor boundary."""
        (tmp_path / "alias").hardlink_to(source)
        return original(root, path, identity)

    monkeypatch.setattr(files_module, "open_selected_file", link_before_open)
    assert problem(write_text_file(source, "new"))["code"] == "tool.planning_failed"
    assert source.read_text(encoding="utf-8") == "content"


@pytest.mark.parametrize("change", ["replace", "link"])
def test_delete_checks_the_final_descriptor_relative_entry(tmp_path, monkeypatch, caplog, change):
    """Deletion rechecks identity and link count at its final parent-relative operation."""
    source = tmp_path / "file"
    source.write_text("content", encoding="utf-8")
    replacement = tmp_path / "replacement"
    replacement.write_text("replacement", encoding="utf-8")
    original = os.stat

    def change_before_final_stat(path, *args, **kwargs):
        """Change the leaf just before its final descriptor-relative metadata inspection."""
        if path == source.name and kwargs.get("dir_fd") is not None:
            if change == "replace":
                replacement.replace(source)
            else:
                (tmp_path / "alias").hardlink_to(source)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", change_before_final_stat)
    assert problem(delete_path(source))["code"] == "filesystem.delete_failed"
    assert source.exists()
    assert "Could not delete path" in caplog.text


def test_search_binds_a_symbolic_selection_to_its_authorized_identity(tmp_path, caplog):
    """Resolving a search pathname does not discard the authorized source identity."""
    original = tmp_path / "original"
    original.write_text("ordinary", encoding="utf-8")
    replacement = tmp_path / "replacement"
    replacement.write_text("replacement-canary", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(original)

    def swap():
        """Retarget the source alias only after the search operation is authorized."""
        alias.unlink()
        alias.symlink_to(replacement)

    output = tool_registry.call(
        "search_text",
        json.dumps({"path": str(alias), "query": "replacement-canary"}),
        execution_started=swap,
    )
    assert json.loads(output)["ok"] is False
    assert "replacement-canary" not in output
    assert "Could not search text" in caplog.text


def test_cached_search_retains_its_original_symbolic_selection(tmp_path):
    """Cached provenance includes the authorized spelling as well as resolved file sources."""
    source = tmp_path / "source"
    source.write_text(("needle " + '"' * 300 + "\n") * 100, encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    result = json.loads(
        tool_registry.call("search_text", json.dumps({"path": str(alias), "query": "needle"}))
    )
    metadata = cached_metadata(result["handle"])
    assert str(alias) in metadata["file_sources"]
    assert str(source) in metadata["file_sources"]
    assert "file_sources" not in result
