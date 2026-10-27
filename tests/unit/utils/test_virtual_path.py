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
    assert paths.resolve("$WORKSPACE_ROOT/a b/é.txt") == str(tmp_path / "a b/é.txt")
    assert paths.resolve("notes.txt") == str(tmp_path / "notes.txt")
    assert paths.display(str(tmp_path)) == "$WORKSPACE_ROOT"
    assert paths.display(str(tmp_path / "notes.txt")) == "$WORKSPACE_ROOT/notes.txt"
    assert paths.display(str(scratch / "notes.txt")) == "$TMPDIR/notes.txt"
    assert paths.display("notes.txt") == "notes.txt"
    assert paths.display(str(tmp_path.parent / "external")) == "<external>"
    assert paths.resolve_command(
        "git -C '$WORKSPACE_ROOT' --root='$WORKSPACE_ROOT/config.toml' '$TMPDIR/output.txt'"
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
    assert paths.resolve("$WORKSPACE_ROOT/link") == str(link)
    assert paths.metadata(
        {"rows": [{"path": str(scratch / "notes.txt")}], "content": str(tmp_path)},
        (("rows", "*", "path"),),
    ) == {"rows": [{"path": "$TMPDIR/notes.txt"}], "content": str(tmp_path)}
    assert paths.redact(f"Failed at {tmp_path}/notes.txt") == "Failed at $WORKSPACE_ROOT/notes.txt"
    for value in (
        "$WORKSPACE_ROOT/../escape",
        "$WORKSPACE_ROOT//outside",
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

    assert (
        paths.redact(f"{actual}/file {linked}/file") == "$WORKSPACE_ROOT/file $WORKSPACE_ROOT/file"
    )


def test_virtual_shell_roots_translate_in_one_pass_without_reparsing():
    """Internal literal aliases translate without replacing environment-variable prefixes."""
    aliases = {
        "/bound-work": Path("/private/tmp/loop-vpath-safe/r0"),
        VirtualPath.WORKSPACE: Path("/ignored-workspace"),
        VirtualPath.TEMPORARY: Path("/ignored-scratch"),
    }
    source = (
        "cat /bound-work/a '/bound-work/b'; "
        'cat "$WORKSPACE_ROOT/c" "$TMPDIR/d"; '
        "echo /bound-work-old /foo/bound-work https://example.test/bound-work"
    )
    assert VirtualPath.translate_shell_source(source, aliases) == (
        "cat /private/tmp/loop-vpath-safe/r0/a '/private/tmp/loop-vpath-safe/r0/b'; "
        'cat "$WORKSPACE_ROOT/c" "$TMPDIR/d"; '
        "echo /bound-work-old /foo/bound-work https://example.test/bound-work"
    )
    assert VirtualPath.translate_shell_source(source, {}) == source
    assert VirtualPath.translate_shell_source("ls /bound-work", aliases) == (
        "ls /private/tmp/loop-vpath-safe/r0/"
    )
    paths = VirtualPath(Path("/work"), Path("/scratch"), {"guide": Path("/skill")})
    assert [root for root, _ in paths.command_roots()] == [
        VirtualPath.WORKSPACE,
        "/skills/guide",
        VirtualPath.TEMPORARY,
    ]


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


def test_canonical_prefixes_round_trip_and_reject_physical_paths(tmp_path):
    """One portable spelling round-trips without accepting host or obsolete root aliases."""
    workspace = tmp_path / "work space"
    temporary = tmp_path / "session"
    skill = tmp_path / "skill"
    paths = VirtualPath(workspace, temporary, {"guide": skill})
    for root, physical in ((VirtualPath.WORKSPACE, workspace), (VirtualPath.TEMPORARY, temporary)):
        value = root + "/space é;$/file"
        assert paths.resolve(value) == str(physical / "space é;$/file")
        assert paths.display(paths.resolve(value)) == value
        assert paths.display(paths.resolve(paths.display(paths.resolve(value)))) == value
        with pytest.raises(ValueError, match="escapes"):
            paths.resolve(root + "/../outside")
    assert paths.display(str(skill / "resource")) == "/skills/guide/resource"
    text = f"{workspace}/file {temporary}/file {skill}/resource {tmp_path}/outside"
    assert paths.redact(text) == (
        "$WORKSPACE_ROOT/file $TMPDIR/file /skills/guide/resource <external>/outside"
    )
    assert paths.redact(shlex.quote(str(workspace))) == VirtualPath.WORKSPACE
    assert paths.redact("unchanged", roots=()) == "unchanged"
    for value in ("/workspace/file", str(workspace / "file"), str(temporary / "file")):
        with pytest.raises(ValueError, match="configured"):
            paths.resolve(value)
