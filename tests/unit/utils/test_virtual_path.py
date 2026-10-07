"""Verify model path translation, shell aliases and disclosure boundaries."""

import shlex
from pathlib import Path

import pytest

from loop.utils.virtual_path import VirtualPath


def test_virtual_paths_preserve_identity_without_disclosing_local_roots(tmp_path):
    """Virtual paths resolve typed roots while keeping host roots out of model-facing metadata."""
    scratch = tmp_path / "temporary"
    scratch.mkdir()
    paths = VirtualPath(tmp_path, scratch)
    assert paths.resolve("/workspace/a b/é.txt") == str(tmp_path / "a b/é.txt")
    assert paths.resolve("notes.txt") == str(tmp_path / "notes.txt")
    assert paths.display(str(tmp_path)) == "/workspace"
    assert paths.display(str(tmp_path / "notes.txt")) == "/workspace/notes.txt"
    assert paths.display(str(scratch / "notes.txt")) == "/tmp/notes.txt"
    assert paths.display("notes.txt") == "notes.txt"
    assert paths.display(str(tmp_path.parent / "external")) == "<external>"
    assert paths.resolve_command(
        "git -C /workspace --root=/workspace/config.toml /tmp/output.txt"
    ) == shlex.join(
        (
            "git",
            "-C",
            str(tmp_path),
            f"--root={tmp_path}/config.toml",
            f"{scratch}/output.txt",
        )
    )
    link = tmp_path / "link"
    link.symlink_to(tmp_path.parent, target_is_directory=True)
    assert paths.resolve("/workspace/link") == str(link)
    assert paths.metadata(
        {"rows": [{"path": str(scratch / "notes.txt")}], "content": str(tmp_path)},
        (("rows", "*", "path"),),
    ) == {"rows": [{"path": "/tmp/notes.txt"}], "content": str(tmp_path)}
    assert paths.redact(f"Failed at {tmp_path}/notes.txt") == "Failed at /workspace/notes.txt"
    for value in (
        "/workspace/../escape",
        "/workspace//outside",
        "/skills/missing/file",
        "../escape",
    ):
        with pytest.raises(ValueError):
            paths.resolve(value)

    assert paths.metadata({"other": "value"}, (("missing",),)) == {"other": "value"}
    assert paths.metadata({"items": "not-a-list"}, (("items", "*", "path"),)) == {
        "items": "not-a-list"
    }
    assert paths.metadata({"path": 42}, (("path",),)) == {"path": 42}


def test_virtual_paths_reconstruct_after_workspace_relocation(tmp_path):
    """A stable virtual path addresses the equivalent file after a local root is relocated."""
    original = VirtualPath(tmp_path / "old")
    moved = VirtualPath(tmp_path / "new")
    virtual = original.display(str(tmp_path / "old" / "notes.txt"))
    assert moved.resolve(virtual) == str(tmp_path / "new" / "notes.txt")


def test_virtual_paths_redact_canonical_symlink_targets(tmp_path):
    """Command output through a symlinked workspace cannot expose its canonical host root."""
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    paths = VirtualPath(linked)

    assert paths.redact(f"{actual}/file {linked}/file") == "/workspace/file /workspace/file"


def test_virtual_shell_roots_translate_in_one_pass_without_reparsing(tmp_path):
    """Literal roots translate across quoting while path-like nonmatches stay intact."""
    paths = VirtualPath(
        tmp_path / "workspace", tmp_path / "temporary", {"guide": tmp_path / "skill"}
    )
    aliases = {
        "/workspace": Path("/private/tmp/loop-vpath-safe/r0"),
        "/tmp": Path("/private/tmp/loop-vpath-safe/r1"),
        "/skills/guide": Path("/private/tmp/loop-vpath-safe/r2"),
    }
    assert [root for root, _ in paths.command_roots()] == ["/skills/guide", "/workspace", "/tmp"]
    source = (
        "cat /workspace/a '/workspace/b' \"/tmp/c\" --file=/skills/guide/d; "
        "sh -c 'cat /workspace/e'; echo /workspace-old /foo/workspace "
        "https://example.test/workspace"
    )
    translated = paths.translate_shell_source(source, aliases)
    assert translated == (
        "cat /private/tmp/loop-vpath-safe/r0/a '/private/tmp/loop-vpath-safe/r0/b' "
        '"/private/tmp/loop-vpath-safe/r1/c" --file=/private/tmp/loop-vpath-safe/r2/d; '
        "sh -c 'cat /private/tmp/loop-vpath-safe/r0/e'; echo /workspace-old "
        "/foo/workspace https://example.test/workspace"
    )
    assert paths.translate_shell_source(source, {}) == source
    assert paths.translate_shell_source("ls /workspace", aliases) == (
        "ls /private/tmp/loop-vpath-safe/r0/"
    )
