"""Verify public text search semantics and filesystem authority."""

import json
import shlex

import pytest

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
    """Public search never executes an arbitrary PATH helper or follows an unauthorized outside alias."""
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
    """Public search preserves Python backreferences, ignores binary files and bounds total results."""
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
