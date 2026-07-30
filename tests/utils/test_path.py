"""Tests for path discovery helpers."""

from loop.utils.path import find_project_root


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
