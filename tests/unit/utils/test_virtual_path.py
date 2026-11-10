"""Verify model path translation, shell aliases and disclosure boundaries."""

import json
import re
import shlex
from pathlib import Path

import pytest

from loop.utils.virtual_path import VirtualPath


def test_virtual_paths_preserve_identity_without_disclosing_local_roots(tmp_path):
    """Virtual paths resolve typed roots while keeping host roots out of model-facing metadata."""
    assert VirtualPath.METADATA_KEY == "app:virtual-path"
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


def test_diagnostic_redaction_does_not_mislabel_sibling_path_prefixes(tmp_path):
    """A path sharing only a root's textual prefix is not identified as that root."""
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "session"
    paths = VirtualPath(workspace, scratch)

    assert paths.redact(f"{workspace}/file") == "$WORKSPACE_ROOT/file"
    assert paths.redact(f"{workspace}-other/file") == "<external>-other/file"
    assert paths.redact(f"{scratch}.old/file") == "<external>.old/file"
    assert paths.redact(f"https://example.test{workspace}/file") == (
        "https://example.test<external>/file"
    )


def test_command_roots_are_ordered_by_longest_virtual_prefix():
    """Configured command roots retain their longest-prefix ordering."""
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


def test_canonical_prefixes_round_trip_and_accept_bound_physical_paths(tmp_path):
    """Portable and physical paths address the same bound roots, but outsiders remain invalid."""
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
    for physical in (workspace, temporary, skill):
        assert paths.resolve(str(physical / "file")) == str(physical / "file")
    for value in ("/workspace/file", str(tmp_path / "outside"), str(workspace) + "-other/file"):
        with pytest.raises(ValueError, match="configured"):
            paths.resolve(value)


def test_physical_root_aliases_remain_contained_and_preserve_symlink_leaves(tmp_path):
    """Canonical and lexical root spellings never accept parent traversal or sibling prefixes."""
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (actual / "link").symlink_to(outside, target_is_directory=True)
    paths = VirtualPath(alias)

    assert paths.resolve(str(alias / "link" / "file")) == str(alias / "link" / "file")
    assert paths.resolve(str(actual / "link" / "file")) == str(alias / "link" / "file")
    with pytest.raises(ValueError, match="escapes"):
        paths.resolve(f"{actual}/../outside/file")
    with pytest.raises(ValueError, match="configured"):
        paths.resolve(f"{actual}-other/file")


@pytest.mark.parametrize("failure", [OSError, RuntimeError])
def test_physical_root_resolution_failure_does_not_authorize_an_unknown_path(
    tmp_path, monkeypatch, failure
):
    """An unavailable canonical alias never expands physical path authority."""
    root = tmp_path / "workspace"
    paths = VirtualPath(root)

    def unavailable(*_args, **_kwargs):
        """Simulate a missing or circular canonical root."""
        raise failure()

    monkeypatch.setattr(Path, "resolve", unavailable)
    assert paths.resolve(str(root / "file")) == str(root / "file")
    with pytest.raises(ValueError, match="configured"):
        paths.resolve(str(tmp_path / "outside"))
