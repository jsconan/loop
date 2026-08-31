"""Verify workspace discovery and durable artifact ownership."""

import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import UUID

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
    assert storage.workspaces == tmp_path / ".loop" / "workspaces.db"
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


def test_workspace_rejects_partial_identity_metadata(tmp_path: Path) -> None:
    """Prevent partially initialized workspace identities from escaping their boundary."""
    with pytest.raises(ValueError, match="identity metadata must be complete"):
        Workspace(
            tmp_path,
            tmp_path,
            WorkspaceStorage(tmp_path / ".loop"),
            id="workspace",
        )


def test_initialize_creates_a_named_workspace_in_separate_storage(tmp_path: Path) -> None:
    """Initialization creates a durable UUID and name outside telemetry storage."""
    workspace = Workspace.discover(tmp_path)

    initialized = workspace.initialize()

    assert UUID(initialized.id).version == 4
    assert initialized.name == tmp_path.name
    assert initialized.name_source == "directory"
    assert initialized.created_at_ns == initialized.updated_at_ns
    assert initialized.initialize() is initialized
    assert initialized.storage.workspaces.is_file()
    assert not initialized.storage.telemetry.exists()
    assert initialized.storage.workspaces.stat().st_mode & 0o777 == 0o600


def test_initialize_preserves_workspace_identity_and_name_after_move(tmp_path: Path) -> None:
    """Moving workspace storage preserves its UUID and original inferred name."""
    original = tmp_path / "original"
    original.mkdir()
    initialized = Workspace.discover(original).initialize()
    moved = tmp_path / "moved"
    original.rename(moved)

    reopened = Workspace.discover(moved).initialize()

    assert reopened.id == initialized.id
    assert reopened.name == "original"
    assert reopened.name_source == "directory"


def test_initialize_names_a_filesystem_root_with_the_default(tmp_path: Path) -> None:
    """A root workspace receives a useful fallback rather than an empty name."""
    workspace = Workspace(Path("/"), Path("/"), WorkspaceStorage(tmp_path / ".loop"))

    initialized = workspace.initialize()

    assert initialized.name == "Untitled workspace"
    assert initialized.name_source == "default"


def test_initialize_rejects_non_positive_busy_timeouts(tmp_path: Path) -> None:
    """Workspace catalog locking must allow forward progress."""
    with pytest.raises(ValueError, match="busy timeout"):
        Workspace.discover(tmp_path).initialize(busy_timeout_ms=0)


def test_initialize_rolls_back_an_invalid_workspace_schema(tmp_path: Path) -> None:
    """Invalid workspace catalog schemas fail without replacing identity."""
    workspace = Workspace.discover(tmp_path)
    workspace.storage.workspaces.parent.mkdir()
    with closing(sqlite3.connect(workspace.storage.workspaces)) as connection:
        connection.execute("CREATE TABLE workspaces (workspace_id TEXT PRIMARY KEY)")

    with pytest.raises(sqlite3.OperationalError, match="no such column: name"):
        workspace.initialize()
