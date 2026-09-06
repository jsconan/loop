"""Verify durable workspace identity repository behavior."""

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import pytest

from loop.workspace import Workspace, WorkspaceRepository


def repository(tmp_path: Path) -> WorkspaceRepository:
    """Return a repository with isolated catalog storage."""
    return WorkspaceRepository(tmp_path / "app" / "workspaces.db")


def test_repository_creates_reopens_and_short_circuits_identity(tmp_path: Path) -> None:
    """Initialization persists one complete UUID-backed identity per canonical location."""
    store = repository(tmp_path)
    workspace = Workspace.discover(tmp_path / "project")

    initialized = store.initialize(workspace)
    reopened = store.initialize(Workspace.discover(tmp_path / "project"))

    assert UUID(initialized.id).version == 4
    assert initialized.name == "project"
    assert initialized.name_source == "directory"
    assert initialized.created_at_ns == initialized.updated_at_ns
    assert reopened.id == initialized.id
    assert store.initialize(initialized) is initialized
    assert not (tmp_path / "project" / ".loop" / "workspace-id").exists()


def test_repository_lists_and_renames_registered_workspaces(tmp_path: Path) -> None:
    """Registry reads and explicit renames remain repository-owned operations."""
    store = repository(tmp_path)
    initialized = store.initialize(Workspace.discover(tmp_path / "project"))

    store.rename(initialized.id, "My project")

    renamed = store.list(initialized.id)
    assert len(renamed) == 1
    assert renamed[0].id == initialized.id
    assert renamed[0].name == "My project"
    assert renamed[0].name_source == "user"
    assert store.list("missing") == ()


def test_repository_names_a_filesystem_root_with_the_default(tmp_path: Path, monkeypatch) -> None:
    """An empty root basename receives a useful fallback name."""
    initialized = repository(tmp_path).initialize(Workspace(Path("/"), Path("/")))

    assert initialized.name == "Untitled workspace"
    assert initialized.name_source == "default"


def test_repository_rejects_non_positive_busy_timeout(tmp_path: Path) -> None:
    """Catalog locking must permit forward progress."""
    with pytest.raises(ValueError, match="busy timeout"):
        repository(tmp_path).initialize(Workspace.discover(tmp_path), busy_timeout_ms=0)


def test_repository_rolls_back_an_invalid_catalog_schema(tmp_path: Path) -> None:
    """Invalid catalog schemas fail without returning partial identity."""
    catalog = tmp_path / "app" / "workspaces.db"
    catalog.parent.mkdir()
    with closing(sqlite3.connect(catalog)) as connection:
        connection.execute("CREATE TABLE workspaces (workspace_id TEXT PRIMARY KEY)")
    store = WorkspaceRepository(catalog)

    with pytest.raises(sqlite3.OperationalError, match="no such column: w.name"):
        store.initialize(Workspace.discover(tmp_path / "project"))


def test_repository_closes_connection_when_schema_setup_fails(tmp_path: Path, monkeypatch) -> None:
    """Catalog setup failure closes its partially initialized connection."""
    connection = Mock()
    connection.execute.side_effect = sqlite3.OperationalError("schema failed")
    catalog = tmp_path / "app" / "workspaces.db"
    catalog.parent.mkdir()
    catalog.touch()
    monkeypatch.setattr("loop.workspace.repository.sqlite3.connect", Mock(return_value=connection))

    with pytest.raises(sqlite3.OperationalError, match="schema failed"):
        WorkspaceRepository(catalog).initialize(Workspace.discover(tmp_path))

    connection.close.assert_called_once_with()


def test_repository_imports_legacy_identity(tmp_path: Path) -> None:
    """A legacy local catalog seeds the centralized workspace identity."""
    project = tmp_path / "project"
    local = project / ".loop"
    local.mkdir(parents=True)
    legacy_store = WorkspaceRepository(local / "workspaces.db")
    legacy = legacy_store.initialize(Workspace.discover(project))
    store = repository(tmp_path)
    imported = store.initialize(Workspace.discover(project))
    assert imported.id == legacy.id


def test_repository_relocates_missing_workspace_and_rejects_live_copy(tmp_path: Path) -> None:
    """A stale registered location can move, but a live duplicate cannot claim its identity."""
    original = tmp_path / "original"
    local = original / ".loop"
    local.mkdir(parents=True)
    legacy_store = WorkspaceRepository(local / "workspaces.db")
    legacy = legacy_store.initialize(Workspace.discover(original))
    copied = tmp_path / "copied"
    shutil.copytree(original, copied)
    store = repository(tmp_path)
    initialized = store.initialize(Workspace.discover(original))
    assert initialized.id == legacy.id

    with pytest.raises(ValueError, match="Copied workspace"):
        store.initialize(Workspace.discover(copied))

    moved_root = tmp_path / "moved"
    original.rename(moved_root)
    moved = store.initialize(Workspace.discover(moved_root))
    assert moved.id == initialized.id
    assert moved.root == moved_root


def test_repository_ignores_malformed_legacy_catalog(tmp_path: Path) -> None:
    """Unreadable legacy identity does not prevent a fresh central identity."""
    project = tmp_path / "project"
    local = project / ".loop"
    local.mkdir(parents=True)
    (local / "workspaces.db").write_text("invalid", encoding="utf-8")

    assert repository(tmp_path).initialize(Workspace.discover(project)).id is not None
