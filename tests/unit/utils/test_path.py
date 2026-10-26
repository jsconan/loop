"""Tests for private directory preparation and path discovery helpers."""

import errno
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.utils import (
    canonical_path,
    ensure_private_directory,
    filter_paths_by_globs,
    find_project_root,
    is_path_ignored,
    iter_visible_paths,
)
from loop.utils import path as path_module
from loop.utils.path import private_directory_lock


def test_ensure_private_directory_creates_and_reopens_private_storage(tmp_path):
    """Private directory preparation creates missing parents and preserves existing files."""
    directory = tmp_path / "parent" / "private"
    ensure_private_directory(directory)
    (directory / "report").write_text("draft")

    ensure_private_directory(directory)

    assert (directory / "report").read_text() == "draft"
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700


@pytest.mark.parametrize("search_only", [True, False])
def test_ensure_private_directory_uses_available_directory_search_access(
    monkeypatch, tmp_path, search_only
):
    """Select search-only or fallback flags without requiring either native open capability."""
    directory = tmp_path / "parent" / "private"
    directory.mkdir(mode=0o700, parents=True)
    details = directory.lstat()
    search_flag = 1 << 40
    if search_only:
        monkeypatch.setattr(os, "O_SEARCH", search_flag, raising=False)
    else:
        monkeypatch.delattr(os, "O_SEARCH", raising=False)
    opened = []
    closed = Mock()
    monkeypatch.setattr(os, "close", closed)
    monkeypatch.setattr(os, "fstat", Mock(return_value=details))
    monkeypatch.setattr(path_module, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(os, "getxattr", Mock(return_value=b""), raising=False)

    def open_directory(path, flags, *args, **kwargs):
        """Model descriptor acquisition independently of the host OS and sandbox."""
        access = search_flag if search_only else os.O_RDONLY
        assert flags == access | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor = 100 + len(opened)
        opened.append((path, descriptor))
        return descriptor

    monkeypatch.setattr(os, "open", open_directory)

    ensure_private_directory(directory)
    ensure_private_directory(directory, create=False)

    assert directory.stat().st_mode & 0o777 == 0o700
    assert opened[0][0] == directory.anchor
    assert directory.name in [path for path, _ in opened]
    assert [call.args[0] for call in closed.call_args_list] == [
        descriptor for _, descriptor in opened
    ]


@pytest.mark.parametrize("component", ["target", "ancestor", "relative"])
def test_ensure_private_directory_rejects_aliased_roots_before_creating_files(tmp_path, component):
    """Symbolic directory substitutions and relative roots never grant working storage."""
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    root = tmp_path / "scratch"
    if component == "target":
        root.symlink_to(outside, target_is_directory=True)
    elif component == "ancestor":
        root.symlink_to(outside, target_is_directory=True)
        root = root / "nested"
    else:
        root = Path("relative")

    with pytest.raises(ValueError, match="private real directory"):
        ensure_private_directory(root)
    assert not list(outside.iterdir())


def test_ensure_private_directory_rejects_existing_nonprivate_modes(tmp_path):
    """Existing shared directories are refused rather than silently changing their modes."""
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    directory.chmod(0o755)
    with pytest.raises(ValueError, match="private real directory"):
        ensure_private_directory(directory)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o755


@pytest.mark.parametrize("change", ["owner", "type", "alias"])
def test_ensure_private_directory_rejects_changed_directory_authority(
    monkeypatch, tmp_path, change
):
    """A directory whose owner, type, or spelling changes during preparation is refused."""
    root = tmp_path / "scratch"
    root.mkdir(mode=0o700)
    original_stat = Path.lstat
    original_resolve = Path.resolve
    resolutions = 0

    def resolve(path, *args, **kwargs):
        """Replace the root spelling only after its initial validation."""
        nonlocal resolutions
        if path == root:
            resolutions += 1
            if change == "alias" and resolutions > 1:
                return tmp_path
        return original_resolve(path, *args, **kwargs)

    def lstat(path, *args, **kwargs):
        """Present changed ownership or file type at the inspection boundary."""
        details = original_stat(path, *args, **kwargs)
        if path == root:
            return SimpleNamespace(
                st_uid=os.getuid() + 1 if change == "owner" else os.getuid(),
                st_mode=stat.S_IFREG | 0o700 if change == "type" else details.st_mode,
            )
        return details

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="private real directory"):
        ensure_private_directory(root)


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


def test_private_directory_creation_never_follows_a_concurrently_substituted_parent(
    monkeypatch, tmp_path
):
    """A symlink inserted after canonical validation cannot redirect directory creation."""
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o700)
    original = tmp_path / "original"
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    target = parent / "created"
    original_open = os.open

    def open_directory(path, flags, *args, **kwargs):
        """Substitute the parent immediately before its descriptor is opened."""
        if path == parent.name:
            parent.rename(original)
            parent.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_directory)
    with pytest.raises(OSError):
        ensure_private_directory(target)
    assert not (outside / "created").exists()
    assert not (original / "created").exists()


def test_private_directory_creation_tolerates_another_creator(monkeypatch, tmp_path):
    """Another caller creating the same missing directory does not cause spurious failure."""
    path = tmp_path / "private"
    original_mkdir = os.mkdir

    def concurrent_mkdir(name, *args, **kwargs):
        """Create the selected directory before reporting the other creator's race."""
        original_mkdir(name, *args, **kwargs)
        if name == path.name:
            raise FileExistsError("another caller created the directory")

    monkeypatch.setattr(os, "mkdir", concurrent_mkdir)
    ensure_private_directory(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o700


def test_private_directory_validation_rejects_a_replaced_inode(monkeypatch, tmp_path):
    """A real directory replacing the opened inode is not accepted as the original target."""
    path = tmp_path / "private"
    path.mkdir(mode=0o700)
    original_lstat = Path.lstat
    replacement = tmp_path / "replacement"

    def lstat(current, *args, **kwargs):
        """Replace the selected directory during the final pathname identity check."""
        if current == path:
            path.rename(replacement)
            path.mkdir(mode=0o700)
        return original_lstat(current, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="private real directory"):
        ensure_private_directory(path)


def test_private_directory_reopen_never_creates_missing_components(tmp_path):
    """Reopening missing storage fails without creating a replacement directory."""
    missing = tmp_path / "missing" / "files"
    with pytest.raises(FileNotFoundError):
        ensure_private_directory(missing, create=False)
    assert not missing.parent.exists()


@pytest.mark.parametrize(
    ("acl", "entry", "error", "expected"),
    [
        (None, 0, 2, None),
        (None, 0, 13, OSError),
        (123, -1, 22, None),
        (123, -1, 1, OSError),
        (123, 0, 0, ValueError),
    ],
)
def test_private_directory_checks_darwin_access_lists(
    monkeypatch, tmp_path, acl, entry, error, expected
):
    """Darwin extended ACLs are refused and inspection failures remain explicit."""

    native = SimpleNamespace(
        acl_get_fd_np=Mock(return_value=acl),
        acl_get_entry=Mock(return_value=entry),
        acl_free=Mock(),
    )
    monkeypatch.setattr(path_module, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(path_module.ctypes, "CDLL", Mock(return_value=native))
    monkeypatch.setattr(path_module.ctypes, "get_errno", Mock(return_value=error))
    directory = tmp_path / "private"
    if expected:
        with pytest.raises(expected):
            ensure_private_directory(directory)
    else:
        ensure_private_directory(directory)
    if acl:
        native.acl_free.assert_called_once_with(acl)


@pytest.mark.parametrize("value", [b"extended-acl", b"", "missing", "unsupported", "denied"])
def test_private_directory_checks_linux_access_lists(monkeypatch, tmp_path, value):
    """Linux ACL absence is accepted while extended grants and unexpected errors fail closed."""

    errors = {"missing": errno.ENODATA, "unsupported": errno.ENOTSUP, "denied": errno.EACCES}
    reader = Mock()
    if isinstance(value, str):
        reader.side_effect = OSError(errors[value], "synthetic ACL diagnostic")
    else:
        reader.return_value = value
    monkeypatch.setattr(path_module, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(path_module.os, "getxattr", reader, raising=False)
    directory = tmp_path / "private"
    if value == b"extended-acl":
        with pytest.raises(ValueError, match="private real directory"):
            ensure_private_directory(directory)
    elif value == "denied":
        with pytest.raises(OSError, match="synthetic ACL diagnostic"):
            ensure_private_directory(directory)
    else:
        ensure_private_directory(directory)


@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "/absolute"])
def test_private_directory_lock_rejects_unchecked_names(tmp_path, name):
    """A lock name cannot supply path traversal or select its parent directory."""

    with pytest.raises(ValueError, match="lock name"), private_directory_lock(tmp_path, name):
        pytest.fail("invalid lock admitted")


@pytest.mark.parametrize("kind", ["shared", "hardlink", "directory"])
def test_private_directory_lock_refuses_unsafe_existing_entries(tmp_path, kind):
    """Existing lock entries are validated rather than repaired or followed."""

    lock = tmp_path / "owner"
    if kind == "directory":
        lock.mkdir()
    else:
        lock.write_text("preserve")
        lock.chmod(0o644 if kind == "shared" else 0o600)
        if kind == "hardlink":
            (tmp_path / "alias").hardlink_to(lock)
    with pytest.raises((OSError, ValueError)), private_directory_lock(tmp_path, "owner"):
        pytest.fail("unsafe lock admitted")


def test_private_directory_anchors_creation_and_validation_at_approved_descriptor(tmp_path):
    """Approved descriptors support private descendants without reopening every ancestor."""
    root = tmp_path / "private"
    ensure_private_directory(root)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        ensure_private_directory(root / "generation/files", anchor=(root, descriptor))
        ensure_private_directory(root, create=False, anchor=(root, descriptor))
        assert (root / "generation/files").is_dir()
        with pytest.raises(ValueError, match="authority changed"):
            ensure_private_directory(tmp_path / "outside", anchor=(root, descriptor))
        moved = tmp_path / "moved"
        root.rename(moved)
        root.mkdir(mode=0o700)
        with pytest.raises(ValueError, match="authority changed"):
            ensure_private_directory(root / "files", anchor=(root, descriptor))
        assert not (root / "files").exists()
        assert not (moved / "files").exists()
    finally:
        os.close(descriptor)


def test_native_ownership_unavailability_does_not_break_portable_imports(monkeypatch, tmp_path):
    """Platforms without advisory locking fail closed when execution requests ownership."""
    monkeypatch.setattr(
        path_module, "import_module", Mock(side_effect=ModuleNotFoundError("fcntl"))
    )
    with (
        pytest.raises(OSError, match="requires macOS or Linux") as failure,
        private_directory_lock(tmp_path, "owner"),
    ):
        pytest.fail("ownership admitted without native locks")
    assert failure.value.errno == errno.ENOTSUP
    assert not (tmp_path / "owner").exists()
