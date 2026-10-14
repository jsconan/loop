"""Verify public text search semantics and filesystem authority."""

import json
import os
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.macos import MacOSSeatbeltBackend

_RIPGREP_CANDIDATES = tuple(
    Path(root) / "rg" for root in os.environ.get("PATH", "").split(os.pathsep)
)

pytestmark = [pytest.mark.integration, pytest.mark.e2e, pytest.mark.macos]


@pytest.mark.parametrize("query", ["needle", "needle$", r"(?<=€ )needle"])
def test_public_search_preserves_unicode_and_python_regex(public_command, native_workspace, query):
    """Public search composes permissions, context and Unicode columns without a host helper."""
    (native_workspace / "search.txt").write_text("before\n€ needle\nafter\n", encoding="utf-8")
    result = public_command.call(
        name="search_text",
        path="/workspace/search.txt",
        query=query,
        regex=query != "needle",
        context_lines=1,
    )
    assert result["ok"] is True
    assert result["result"]["matches"] == [
        {
            "path": "search.txt",
            "line": 2,
            "column": 3,
            "text": "€ needle",
            "context": [{"line": 1, "text": "before"}, {"line": 3, "text": "after"}],
        }
    ]
    public_command.host.run_host_command.assert_not_called()


def test_hostile_path_helper_and_outside_search_are_rejected(
    public_command, native_workspace, monkeypatch
):
    """Keep public search from running arbitrary PATH helpers or outside aliases."""
    hostile = native_workspace.parent / "hostile-bin"
    hostile.mkdir()
    marker = native_workspace.parent / "hostile-ran"
    helper = hostile / "rg"
    helper.write_text(f"#!/bin/sh\nprintf ran > {shlex.quote(str(marker))}\n")
    helper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{hostile}:/usr/bin:/bin")
    result = public_command.call(name="search_text", path="/workspace/ordinary", query="ordinary")
    assert result["ok"] is True
    assert not marker.exists()
    outside = public_command.call(
        name="search_text", path="/workspace/external-link", query="fake-outside"
    )
    assert outside["ok"] is False
    assert "fake-outside-canary" not in json.dumps(outside)
    public_command.host.run_host_command.assert_not_called()


def test_public_search_binary_bounds_and_backreferences(public_command, native_workspace):
    """Preserve Python backreferences, skip binary files and bound search results."""
    (native_workspace / "search-first.txt").write_text("x" * 2000 + "\nneedle needle\n")
    (native_workspace / "search-second.txt").write_text("needle\n")
    (native_workspace / "search-binary.bin").write_bytes(b"\0needle\n")
    backreference = public_command.call(
        name="search_text", path="/workspace/search-first.txt", query=r"(needle) \1", regex=True
    )
    assert backreference["result"]["matches"][0]["column"] == 1
    bounded = public_command.call(
        name="search_text", path="/workspace", query="needle", max_results=1
    )
    assert len(bounded["result"]["matches"]) == 1
    assert bounded["result"]["truncated"] is True
    binary = public_command.call(
        name="search_text", path="/workspace/search-binary.bin", query="needle"
    )
    assert binary["result"]["matches"] == []


@pytest.fixture
def native_search_helper(public_command, monkeypatch):
    """Select a genuinely qualified installed helper and forbid Python matching fallback."""
    selected = None
    for candidate in _RIPGREP_CANDIDATES:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve(strict=True)
        roots, _ = MacOSSeatbeltBackend.installed_tool_roots(resolved)
        if resolved.stat().st_nlink == 1 and (
            any(resolved.is_relative_to(root) for root in public_command.backend.system_read_roots)
            or all(MacOSSeatbeltBackend.managed_tool_root(root) for root in roots)
        ):
            selected = candidate
            break
    if selected is None:
        pytest.skip("requires a qualified installed ripgrep on the qualification PATH")
    monkeypatch.setenv("PATH", str(selected.parent) + ":/usr/bin:/bin")
    captures = []
    original = MacOSSeatbeltBackend.run_read_only_argv

    def run(backend, executable, arguments, descriptors, deadline):
        """Observe actual native captures without replacing OS execution or enforcement."""
        capture = original(backend, executable, arguments, descriptors, deadline)
        assert capture is not None, "qualified native helper did not launch"
        captures.append(capture)
        return capture

    def no_fallback(*args, **kwargs):
        """Reject a switch from the selected native helper into Python pattern matching."""
        pytest.fail("qualified public native search used Python fallback")

    monkeypatch.setattr(MacOSSeatbeltBackend, "run_read_only_argv", run)
    monkeypatch.setattr("loop.utils.search.re.compile", no_fallback)
    public_command.helper = selected.resolve(strict=True)
    return captures


def test_public_native_search_preserves_dialect_unicode_and_bounds(
    public_command, native_workspace, native_search_helper, caplog
):
    """Real public ripgrep preserves its regex dialect and bounded Unicode results."""
    selected = native_workspace / "selected"
    selected.write_text("before\n€ needle needle\nafter\n")
    for query, regex in (("needle", False), ("needle$", True)):
        value = public_command.call(
            name="search_text",
            path="/workspace/selected",
            query=query,
            regex=regex,
            context_lines=1,
        )
        assert value["ok"], value
        match = value["result"]["matches"][0]
        assert match["line"] == 2 and match["column"] == (3 if not regex else 10)
        assert len(match["context"]) == 2
    for query in (r"(?<=€ )needle", r"(needle) \1"):
        value = public_command.call(
            name="search_text", path="/workspace/selected", query=query, regex=True
        )
        assert not value["ok"] and "regex parse error" in json.dumps(value)
    assert sum("Could not search text" in record.message for record in caplog.records) == 2
    selected.write_text("needle\n" * 20000)
    bounded = public_command.call(
        name="search_text", path="/workspace/selected", query="needle", max_results=1
    )
    assert bounded["ok"] and len(bounded["result"]["matches"]) == 1
    assert bounded["result"]["truncated"]
    assert len(native_search_helper) == 5
    assert [capture.exit_code for capture in native_search_helper[:4]] == [0, 0, 2, 2]
    public_command.host.run_host_command.assert_not_called()


def test_public_native_search_keeps_selected_files_isolated(
    public_command, native_workspace, native_search_helper
):
    """Concurrent native searches disclose only selected inputs and deny outside aliases."""
    first = native_workspace / "first"
    second = native_workspace / "second"
    first.write_text("first needle\n")
    second.write_text("second needle\n")
    (native_workspace / "unselected").write_text("unselected-private-canary")
    with ThreadPoolExecutor(max_workers=2) as workers:
        values = list(
            workers.map(
                lambda name: public_command.call(
                    name="search_text", path="/workspace/" + name, query="needle"
                ),
                ("first", "second"),
            )
        )
    assert all(value["ok"] for value in values), values
    assert [value["result"]["matches"][0]["text"] for value in values] == [
        "first needle",
        "second needle",
    ]
    assert "unselected-private-canary" not in json.dumps(values)
    descriptor = os.open(first, os.O_RDONLY)
    try:
        capture = public_command.backend.run_read_only_argv(
            public_command.helper,
            (
                "--no-config",
                "--json",
                "--regexp",
                "unselected-private-canary",
                "--",
                str(native_workspace / "unselected"),
                f"/dev/fd/{descriptor:05d}",
            ),
            (descriptor,),
            time.monotonic() + 5,
        )
    finally:
        os.close(descriptor)
    assert capture.exit_code == 2
    assert "unselected-private-canary" not in capture.stdout
    before = len(native_search_helper)
    refusal = Mock()
    refusal.prompt.return_value = "deny"
    for path in ("/workspace/.git/config", "/workspace/external-link"):
        value = public_command.call(
            name="search_text", path=path, query="canary", chosen_interaction=refusal
        )
        assert not value["ok"], (path, value)
        assert "fake-private-canary" not in json.dumps(value)
        assert "fake-outside-canary" not in json.dumps(value)
    assert len(native_search_helper) == before
    public_command.host.run_host_command.assert_not_called()


def test_native_selected_helper_bounds_a_running_child(
    public_command, native_workspace, native_search_helper
):
    """A qualified helper and its system preprocessor stop within the original deadline."""
    selected = native_workspace / "selected"
    selected.write_text("selected input\n")
    descriptor = os.open(selected, os.O_RDONLY)
    started = time.monotonic()
    try:
        capture = public_command.backend.run_read_only_argv(
            public_command.helper,
            (
                "--no-config",
                "--json",
                "--pre=/usr/bin/yes",
                "--regexp",
                "no-match",
                "--",
                f"/dev/fd/{descriptor:05d}",
            ),
            (descriptor,),
            started + 0.5,
        )
    finally:
        os.close(descriptor)
    assert capture.status.value == "timed_out", capture
    assert time.monotonic() - started < 3
    assert len(capture.stdout) <= 65536
    public_command.host.run_host_command.assert_not_called()
