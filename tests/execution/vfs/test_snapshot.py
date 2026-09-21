"""Test authenticated workspace roots, immutable snapshots, and filesystem capabilities."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from loop.execution.vfs import (
    AuthenticatedWorkspaceRoot,
    FilesystemCapabilities,
    SnapshotBuilder,
    WorkspaceBusyError,
)
from loop.execution.vfs.snapshot import FileCopier


def test_snapshot_captures_real_files_and_literal_symlinks_without_following_them(tmp_path: Path):
    """A native descriptor walk copies files and preserves, rather than follows, symlinks."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nested").mkdir()
    (workspace / "nested" / "file.txt").write_text("content", encoding="utf-8")
    os.symlink("/outside/never-read", workspace / "outside")

    builder = SnapshotBuilder(tmp_path / "snapshots")
    with AuthenticatedWorkspaceRoot(workspace) as root:
        snapshot = builder.build(root)

    assert (
        snapshot.directory.joinpath("nested", "file.txt").read_text(encoding="utf-8") == "content"
    )
    assert os.readlink(snapshot.directory / "outside") == "/outside/never-read"
    assert [entry.path for entry in snapshot.manifest.entries] == [
        "nested",
        "nested/file.txt",
        "outside",
    ]
    assert snapshot.manifest.entries[1].content is not None
    assert snapshot.manifest.snapshot_id.value.startswith("snapshot-")
    assert snapshot.directory.name == "tree"
    assert not (snapshot.directory.parent / "manifest.json").samefile(snapshot.directory)
    assert snapshot.directory.parent.stat().st_mode & 0o777 == 0o500
    assert snapshot.directory.stat().st_mode & 0o777 == workspace.stat().st_mode & 0o777
    assert (snapshot.directory / "nested" / "file.txt").stat().st_mode & 0o777 == (
        workspace / "nested" / "file.txt"
    ).stat().st_mode & 0o777
    assert not (snapshot.directory.parent / "manifest.json").stat().st_mode & 0o222
    builder.discard(snapshot)
    assert not snapshot.directory.parent.exists()


def test_authenticated_root_rejects_symlinks_and_root_replacement(tmp_path: Path):
    """Root authentication refuses aliases and detects replacement before descriptor use."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "alias"
    os.symlink(workspace, alias)
    with pytest.raises(OSError):
        AuthenticatedWorkspaceRoot(alias)

    with AuthenticatedWorkspaceRoot(workspace) as root:
        replacement = tmp_path / "replacement"
        replacement.mkdir()
        workspace.rename(tmp_path / "old")
        replacement.rename(workspace)
        with pytest.raises(RuntimeError, match="root changed"):
            root.duplicate_descriptor()


def test_authenticated_root_confines_directory_traversal_and_closes_idempotently(tmp_path: Path):
    """Descriptor traversal opens real descendants and rejects a symlink component."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nested").mkdir()
    os.symlink(tmp_path, workspace / "escape")

    root = AuthenticatedWorkspaceRoot(workspace)
    descriptor = root.open_directory("nested")
    os.close(descriptor)
    with pytest.raises(RuntimeError, match="Unsafe workspace directory traversal"):
        root.open_directory("escape")
    root.close()
    root.close()


def test_authenticated_root_fails_closed_when_its_name_is_removed(tmp_path: Path):
    """A retained descriptor cannot authorize a root path that disappeared from its parent."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with AuthenticatedWorkspaceRoot(workspace) as root:
        workspace.rmdir()
        with pytest.raises(RuntimeError, match="root changed"):
            root.verify_root()


def test_snapshot_builder_rejects_invalid_retry_counts(tmp_path: Path):
    """A bounded retry policy cannot be configured with a negative attempt count."""
    with pytest.raises(ValueError, match="cannot be negative"):
        SnapshotBuilder(tmp_path / "snapshots", retries=-1)


def test_snapshot_discard_rejects_missing_and_foreign_publications(tmp_path: Path) -> None:
    """Reclamation cannot follow a missing or foreign snapshot identity."""
    builder = SnapshotBuilder(tmp_path / "snapshots")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with AuthenticatedWorkspaceRoot(workspace) as root:
        snapshot = builder.build(root)
    foreign = snapshot.__class__(snapshot.manifest, tmp_path / "foreign" / "tree")
    with pytest.raises(ValueError, match="outside"):
        builder.discard(foreign)
    misplaced = snapshot.__class__(
        snapshot.manifest,
        snapshot.directory.parent / "manifest.json",
    )
    with pytest.raises(ValueError, match="outside"):
        builder.discard(misplaced)
    builder.discard(snapshot)


class _MutatingCopier(FileCopier):
    """Change a source after copying it to exercise the post-copy stability gate."""

    source_path: Path

    def __init__(self, source_path: Path) -> None:
        """Remember the test source that is deliberately changed after copying."""
        self.source_path = source_path

    def copy(self, source_descriptor: int, destination: Path) -> tuple[str, int]:
        """Copy then mutate the source descriptor's contents before its post-check."""
        result = super().copy(source_descriptor, destination)
        self.source_path.write_text("changed", encoding="utf-8")
        return result


def test_snapshot_retries_and_fails_closed_when_a_file_changes_during_every_capture(tmp_path: Path):
    """Concurrent native source mutation never publishes a mixed snapshot."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file").write_text("value", encoding="utf-8")

    with (
        AuthenticatedWorkspaceRoot(workspace) as root,
        pytest.raises(WorkspaceBusyError, match="remained mutable"),
    ):
        SnapshotBuilder(
            tmp_path / "snapshots", retries=1, copier=_MutatingCopier(workspace / "file")
        ).build(root)

    assert list((tmp_path / "snapshots").iterdir()) == []


def test_snapshot_rejects_unsupported_native_nodes_without_publication(tmp_path: Path):
    """A FIFO cannot be silently represented as a portable workspace object."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    os.mkfifo(workspace / "pipe")

    with AuthenticatedWorkspaceRoot(workspace) as root, pytest.raises(WorkspaceBusyError):
        SnapshotBuilder(tmp_path / "snapshots", retries=0).build(root)


@pytest.mark.parametrize("kind", ("directory", "file", "symlink"))
def test_snapshot_rejects_native_entry_replacement_races(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
):
    """An entry replaced between inspection and use is never included in a snapshot."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "entry"
    if kind == "directory":
        target.mkdir()
    elif kind == "file":
        target.write_text("old", encoding="utf-8")
    else:
        os.symlink("old", target)
    original_open = os.open
    original_readlink = os.readlink
    changed = False

    def race_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        """Replace the inspected file or directory immediately before descriptor opening."""
        nonlocal changed
        if path == "entry" and kind != "symlink" and not changed:
            changed = True
            if kind == "directory":
                target.rmdir()
                target.mkdir()
            else:
                target.write_text("new", encoding="utf-8")
        return original_open(path, flags, *args, **kwargs)

    def race_readlink(path: object, *args: object, **kwargs: object) -> str:
        """Replace the inspected link after its literal target is read."""
        nonlocal changed
        value = original_readlink(path, *args, **kwargs)
        if path == "entry" and not changed:
            changed = True
            target.unlink()
            os.symlink("new", target)
        return value

    monkeypatch.setattr(os, "open", race_open)
    if kind == "symlink":
        monkeypatch.setattr(os, "readlink", race_readlink)
    with AuthenticatedWorkspaceRoot(workspace) as root, pytest.raises(WorkspaceBusyError):
        SnapshotBuilder(tmp_path / "snapshots", retries=0).build(root)


class _ParentMutatingCopier(FileCopier):
    """Add a sibling after a nested file copy to exercise the directory post-check."""

    parent: Path

    def __init__(self, parent: Path) -> None:
        """Remember the directory to mutate after the nested capture starts."""
        self.parent = parent

    def copy(self, source_descriptor: int, destination: Path) -> tuple[str, int]:
        """Copy the file then add a new parent entry before parent verification."""
        result = super().copy(source_descriptor, destination)
        (self.parent / "late").write_text("late", encoding="utf-8")
        return result


def test_snapshot_rejects_a_parent_directory_that_changes_after_child_capture(tmp_path: Path):
    """A descendant copy cannot hide a concurrent new sibling from the parent post-check."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nested").mkdir()
    (workspace / "nested" / "file").write_text("value", encoding="utf-8")

    with AuthenticatedWorkspaceRoot(workspace) as root, pytest.raises(WorkspaceBusyError):
        SnapshotBuilder(
            tmp_path / "snapshots", retries=0, copier=_ParentMutatingCopier(workspace)
        ).build(root)


def test_filesystem_capabilities_are_natively_discovered_once_per_root_identity(tmp_path: Path):
    """A real probe is cleaned up and repeated discovery returns the cached record."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    capabilities = FilesystemCapabilities()

    with AuthenticatedWorkspaceRoot(workspace) as root:
        first = capabilities.discover(root)
        second = capabilities.discover(root)

    assert first is second
    assert first.maximum_name_bytes >= 1
    assert first.maximum_path_bytes >= first.maximum_name_bytes
    assert first.supports_basic_mode
    assert list(workspace.iterdir()) == []


def test_filesystem_capabilities_fail_closed_when_a_descriptor_probe_cannot_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A failed native capability probe cannot silently claim representability support."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def fail_mkdir(*_: object, **__: object) -> None:
        """Simulate a filesystem that refuses the private probe directory."""
        raise OSError("blocked")

    monkeypatch.setattr(os, "mkdir", fail_mkdir)
    with (
        AuthenticatedWorkspaceRoot(workspace) as root,
        pytest.raises(RuntimeError, match="cannot be safely discovered"),
    ):
        FilesystemCapabilities().discover(root)


def test_filesystem_capabilities_handle_case_and_symlink_probe_outcomes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Capability discovery records both case-sensitive and unavailable-symlink filesystems."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original_stat = os.stat

    def case_sensitive_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        """Make only the alternate-case probe name appear absent."""
        if path == "case":
            raise FileNotFoundError
        return original_stat(path, *args, **kwargs)

    def unavailable_symlink(*_: object, **__: object) -> None:
        """Model a destination filesystem that rejects link creation."""
        raise OSError("unsupported")

    monkeypatch.setattr(os, "stat", case_sensitive_stat)
    monkeypatch.setattr(os, "symlink", unavailable_symlink)
    with AuthenticatedWorkspaceRoot(workspace) as root:
        discovered = FilesystemCapabilities().discover(root)

    assert discovered.case_sensitive
    assert not discovered.supports_symlinks
