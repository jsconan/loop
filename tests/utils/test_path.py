"""Tests for path discovery helpers."""

import shlex

import pytest

from loop.utils import (
    VirtualPath,
    canonical_path,
    filter_paths_by_globs,
    find_project_root,
    is_path_ignored,
    iter_visible_paths,
)


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


def test_canonical_path_handles_existing_and_missing_targets(tmp_path):
    """Canonical paths resolve existing targets and missing target parents."""
    existing = tmp_path / "existing.txt"
    existing.write_text("content", "utf-8")

    assert canonical_path(existing) == str(existing)
    assert canonical_path(tmp_path / "missing.txt") == str(tmp_path / "missing.txt")

    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    assert canonical_path(link) == str(target)


def test_filter_paths_by_globs_selects_git_style_relative_patterns(tmp_path):
    """Inclusive path filtering supports recursive Git-style globs and an unfiltered mode."""
    nested = tmp_path / "tests" / "unit.py"
    nested.parent.mkdir()
    nested.touch()
    root_file = tmp_path / "main.py"
    root_file.touch()
    text = tmp_path / "notes.txt"
    text.touch()
    paths = [nested, root_file, text]

    assert list(filter_paths_by_globs(paths, tmp_path, ["*.py"])) == [nested, root_file]
    assert list(filter_paths_by_globs(paths, tmp_path, ["tests/**"])) == [nested]
    assert list(filter_paths_by_globs(paths, tmp_path, None)) == paths


def test_find_project_root_returns_none_outside_a_git_project(tmp_path):
    """A path without a Git marker in its ancestry has no project root."""
    working_directory = tmp_path / "src"
    working_directory.mkdir()

    assert find_project_root(working_directory) is None


def test_find_project_root_returns_the_closest_git_ancestor(tmp_path):
    """The closest Git marker defines the project when repositories are nested."""
    nested_project = tmp_path / "outer" / "nested"
    working_directory = nested_project / "src"
    working_directory.mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    (nested_project / ".git").mkdir()

    assert find_project_root(working_directory) == nested_project


def test_find_project_root_accepts_a_git_file_marker(tmp_path):
    """Git worktree marker files identify project roots as well as directories."""
    working_directory = tmp_path / "src"
    working_directory.mkdir()
    (tmp_path / ".git").write_text("gitdir: elsewhere", encoding="utf-8")

    assert find_project_root(working_directory) == tmp_path


def test_find_project_root_accepts_a_string_path(tmp_path):
    """String working directories are accepted during project discovery."""
    working_directory = tmp_path / "src"
    working_directory.mkdir()
    (tmp_path / ".git").mkdir()

    assert find_project_root(str(working_directory)) == tmp_path


def test_is_path_ignored_checks_hierarchical_rules_and_agent_precedence(tmp_path):
    """On-demand checks share traversal hierarchy and source precedence."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("*.tmp\nsecret.txt\n", encoding="utf-8")
    (tmp_path / ".agentignore").write_text("!secret.txt\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / ".agentignore").write_text("private.txt\n", encoding="utf-8")
    temporary = nested / "cache.tmp"
    secret = nested / "secret.txt"
    private = nested / "private.txt"
    visible = nested / "visible.txt"
    for path in (temporary, secret, private, visible):
        path.touch()

    assert is_path_ignored(temporary)
    assert not is_path_ignored(secret)
    assert is_path_ignored(private)
    assert not is_path_ignored(visible)


def test_is_path_ignored_does_not_load_rules_below_an_ignored_parent(tmp_path):
    """Files cannot be re-included by ignore files inside a pruned directory."""
    (tmp_path / ".agentignore").write_text("private/\n", encoding="utf-8")
    private = tmp_path / "private"
    private.mkdir()
    (private / ".agentignore").write_text("!secret.txt\n", encoding="utf-8")
    secret = private / "secret.txt"
    secret.touch()

    assert is_path_ignored(secret, root=tmp_path)


def test_is_path_ignored_always_excludes_the_application_directory(tmp_path):
    """The application directory is ignored by name, unlike ordinary directories."""
    app_directory = tmp_path / ".loop"
    app_directory.mkdir()
    ordinary = tmp_path / "data"
    ordinary.mkdir()

    assert is_path_ignored(app_directory)
    assert not is_path_ignored(ordinary)


def test_is_path_ignored_falls_back_to_path_parent_and_rejects_outside_root(tmp_path):
    """Ignore checks work outside repositories and enforce an explicit root boundary."""
    ignored = tmp_path / "ignored.txt"
    ignored.touch()
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    assert is_path_ignored(ignored)

    with pytest.raises(ValueError):
        is_path_ignored(tmp_path.parent / "outside.txt", root=tmp_path)


def test_iter_visible_paths_prunes_rules_recurses_and_does_not_follow_symlinks(tmp_path):
    """Traversal yields visible entries, prunes ignored folders, and lists symlink folders only."""
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    visible = tmp_path / "visible"
    visible.mkdir()
    (visible / "child.txt").touch()
    ignored = tmp_path / "ignored"
    ignored.mkdir()
    (ignored / "hidden.txt").touch()
    link = tmp_path / "linked"
    link.symlink_to(visible, target_is_directory=True)

    assert set(iter_visible_paths(tmp_path)) == {tmp_path / ".gitignore", visible, link}
    paths = set(iter_visible_paths(tmp_path, recursive=True))
    assert visible in paths and visible / "child.txt" in paths and link in paths
    assert ignored not in paths and ignored / "hidden.txt" not in paths
    assert link / "child.txt" not in paths


def test_iter_visible_paths_loads_rules_between_project_root_and_folder(tmp_path):
    """Traversal includes ignore files inherited by a nested starting folder."""
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / ".gitignore").write_text("hidden.txt\n", encoding="utf-8")
    hidden = nested / "hidden.txt"
    visible = nested / "visible.txt"
    hidden.touch()
    visible.touch()

    assert set(iter_visible_paths(nested)) == {nested / ".gitignore", visible}
