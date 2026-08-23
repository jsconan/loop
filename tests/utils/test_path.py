"""Tests for path discovery helpers."""

from loop.utils import canonical_path, find_project_root, is_path_ignored


def test_canonical_path_handles_existing_and_missing_targets(tmp_path):
    """Canonical paths resolve existing targets and missing target parents."""
    existing = tmp_path / "existing.txt"
    existing.write_text("content", "utf-8")

    assert canonical_path(existing) == str(existing)
    assert canonical_path(tmp_path / "missing.txt") == str(tmp_path / "missing.txt")


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
