"""Verify workspace discovery and durable artifact ownership."""

from pathlib import Path

import pytest

from loop.workspace import Workspace, WorkspaceStorage


def test_discover_uses_the_nearest_git_worktree_root(tmp_path: Path) -> None:
    """Use a Git marker ancestor while preserving the active nested directory."""
    (tmp_path / ".git").mkdir()
    working_directory = tmp_path / "src" / "package"
    working_directory.mkdir(parents=True)

    workspace = Workspace.discover(working_directory)

    assert workspace.root == tmp_path
    assert workspace.working_directory == working_directory
    assert workspace.storage.root == tmp_path / ".loop"


def test_discover_uses_the_active_directory_outside_git(tmp_path: Path) -> None:
    """Treat a non-Git active directory as its own workspace root."""
    workspace = Workspace.discover(tmp_path)

    assert workspace.root == tmp_path
    assert workspace.working_directory == tmp_path


def test_storage_exposes_every_workspace_artifact(tmp_path: Path) -> None:
    """Resolve all durable application artifacts below one storage root."""
    storage = Workspace.discover(tmp_path).storage

    assert storage.configuration == tmp_path / ".loop" / "config.toml"
    assert storage.sessions == tmp_path / ".loop" / "sessions.db"
    assert storage.telemetry == tmp_path / ".loop" / "telemetry.db"
    assert storage.operational_log == tmp_path / ".loop" / "loop.log"
    assert storage.permissions == tmp_path / ".loop" / "permissions.yaml"
    assert storage.permissions_audit == tmp_path / ".loop" / "permissions-audit.jsonl"


def test_workspace_rejects_an_active_directory_outside_its_root(tmp_path: Path) -> None:
    """Prevent a workspace context from combining unrelated filesystem roots."""
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"

    with pytest.raises(ValueError, match="must be within its root"):
        Workspace(root, outside, WorkspaceStorage(root / ".loop"))
