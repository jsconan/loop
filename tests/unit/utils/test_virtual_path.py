"""Verify model path translation, shell aliases and disclosure boundaries."""

import json
import re
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
    assert paths.display(str(scratch / "notes.txt")) == "$TMPDIR/notes.txt"
    assert paths.display("notes.txt") == "notes.txt"
    assert paths.display(str(tmp_path.parent / "external")) == "<external>"
    assert paths.resolve_command(
        "git -C /workspace --root=/workspace/config.toml '$TMPDIR/output.txt'"
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
    ) == {"rows": [{"path": "$TMPDIR/notes.txt"}], "content": str(tmp_path)}
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
        "$TMPDIR": Path("/private/tmp/loop-vpath-safe/r1"),
        "/skills/guide": Path("/private/tmp/loop-vpath-safe/r2"),
    }
    assert [root for root, _ in paths.command_roots()] == ["/skills/guide", "/workspace", "$TMPDIR"]
    source = (
        "cat /workspace/a '/workspace/b' \"$TMPDIR/c\" --file=/skills/guide/d; "
        "sh -c 'cat /workspace/e'; echo /workspace-old /foo/workspace "
        "https://example.test/workspace"
    )
    translated = paths.translate_shell_source(source, aliases)
    assert translated == (
        "cat /private/tmp/loop-vpath-safe/r0/a '/private/tmp/loop-vpath-safe/r0/b' "
        '"$TMPDIR/c" --file=/private/tmp/loop-vpath-safe/r2/d; '
        "sh -c 'cat /private/tmp/loop-vpath-safe/r0/e'; echo /workspace-old "
        "/foo/workspace https://example.test/workspace"
    )
    assert paths.translate_shell_source(source, {}) == source
    assert paths.translate_shell_source("ls /workspace", aliases) == (
        "ls /private/tmp/loop-vpath-safe/r0/"
    )


@pytest.mark.parametrize(
    "value",
    [
        "$TMPDIR-other/file",
        "$TMPDIRsuffix",
        "${TMPDIR}/file",
        "$HOME/file",
        "$TMPDIR/../escape",
        "$TMPDIR//file",
        "/tmp/file",
    ],
)
def test_session_token_requires_an_exact_contained_prefix(tmp_path, value):
    """Only the literal session token and its contained slash suffix select working storage."""
    paths = VirtualPath(tmp_path / "work", tmp_path / "session")
    with pytest.raises(ValueError):
        paths.resolve(value)
    assert paths.resolve("$TMPDIR") == str(tmp_path / "session")
    assert paths.resolve("$TMPDIR/space é;$/file") == str(tmp_path / "session/space é;$/file")


def test_session_source_and_operating_system_tmp_remain_opaque():
    """Session environment expansion and embedded program text are never substituted."""
    source = '''python -c 'print("/tmp/data", "$TMPDIR/data")'; cat "$TMPDIR/report"'''
    assert VirtualPath.translate_shell_source(source, {"$TMPDIR": Path("/alias")}) == source


@pytest.mark.parametrize("home", ["/Users/synthetic-account", "/home/synthetic-account"])
def test_native_diagnostic_variants_hide_session_and_private_ancestors(home, monkeypatch):
    """Raw, quoted, JSON, shell, canonical and abbreviated diagnostics use logical roots."""
    # These are diagnostic spellings, not host accounts or automounted home directories.
    monkeypatch.setattr(Path, "resolve", lambda path, **kwargs: path)
    root = Path(home) / "Library/Application Support/synthetic é;$/files"
    paths = VirtualPath(Path(home) / "work", root)
    variants = [
        str(root),
        shlex.quote(str(root)),
        json.dumps(str(root), ensure_ascii=True)[1:-1],
        json.dumps(str(root), ensure_ascii=False)[1:-1],
        str(root).replace("/", "\\/"),
        re.sub(r"([^\w/.-])", r"\\\1", str(root)),
        "~/" + "/".join(root.parts[3:]),
    ]
    for variant in variants:
        assert paths.redact(variant) == "$TMPDIR"
    for ancestor in (Path(home), root.parent, Path(home) / "Library"):
        assert paths.redact(str(ancestor)) == "<external>"
    assert paths.redact("~/Library/Application Support") == "<external>"
