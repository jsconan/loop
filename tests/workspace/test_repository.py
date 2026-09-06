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


def test_repository_uses_unique_locators_but_not_ambiguous_or_unavailable_hints(
    tmp_path: Path, monkeypatch
) -> None:
    """Only one stale locator match may update a canonical location."""
    original = tmp_path / "original"
    original.mkdir()
    store = repository(tmp_path)
    first = store.initialize(Workspace.discover(original))
    moved = tmp_path / "moved"
    original.rename(moved)

    relocated = store.initialize(Workspace.discover(moved))
    assert relocated.id == first.id
    assert relocated.name == "moved"

    unavailable = tmp_path / "unavailable"
    unavailable.mkdir()
    monkeypatch.setattr(WorkspaceRepository, "_locator", staticmethod(lambda _path: (None, None)))
    fresh = store.initialize(Workspace.discover(unavailable))
    assert fresh.id != first.id

    ambiguous = tmp_path / "ambiguous"
    ambiguous.mkdir()
    catalog = tmp_path / "app" / "workspaces.db"
    with closing(sqlite3.connect(catalog)) as connection, connection:
        locator = connection.execute(
            "SELECT locator FROM workspace_locations WHERE workspace_id=?", (first.id,)
        ).fetchone()[0]
        connection.execute(
            "UPDATE workspace_locations SET locator_kind='same',locator=?,status='active'",
            (locator,),
        )
    monkeypatch.setattr(
        WorkspaceRepository, "_locator", staticmethod(lambda _path: ("same", locator))
    )
    distinct = store.initialize(Workspace.discover(ambiguous))
    assert distinct.id not in {first.id, fresh.id}

    live_hint = tmp_path / "live-hint"
    live_hint.mkdir()
    with closing(sqlite3.connect(catalog)) as connection:
        first_locator = connection.execute(
            "SELECT locator_kind,locator FROM workspace_locations WHERE workspace_id=?", (first.id,)
        ).fetchone()
        connection.execute(
            "UPDATE workspace_locations SET locator_kind=NULL,locator=NULL WHERE workspace_id!=?",
            (first.id,),
        )
        connection.commit()
    monkeypatch.setattr(WorkspaceRepository, "_locator", staticmethod(lambda _path: first_locator))
    assert store.initialize(Workspace.discover(live_hint)).id != first.id


def test_repository_attach_forget_rekey_and_conflicts(tmp_path: Path) -> None:
    """Lifecycle operations preserve identities, reject conflicts, and retain old data."""
    store = repository(tmp_path)
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    missing = tmp_path / "missing"
    first_path.mkdir()
    second_path.mkdir()
    first = store.attach(first_path)
    second = store.attach(second_path)

    with pytest.raises(FileNotFoundError):
        store.attach(missing)
    with pytest.raises(ValueError, match="Unknown"):
        store.attach(second_path, "missing")
    with pytest.raises(ValueError, match="another identity"):
        store.attach(first_path, second.id)
    third_path = tmp_path / "third"
    third_path.mkdir()
    with pytest.raises(ValueError, match="active path"):
        store.attach(third_path, first.id)

    assert store.forget(first.id)
    assert not store.forget(first.id)
    attached = store.attach(first_path, first.id)
    assert attached.id == first.id
    rekeyed = store.rekey(first_path)
    assert rekeyed.id != first.id
    assert store.list_by_path(first_path).id == rekeyed.id
    with pytest.raises(FileNotFoundError):
        store.rekey(missing)


def test_repository_refreshes_names_only_from_allowed_sources(tmp_path: Path) -> None:
    """User, trusted remote, directory, and default names follow conservative refresh rules."""
    store = repository(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    workspace = store.initialize(Workspace.discover(project))
    assert not store.refresh_name(workspace.id, "renamed", "directory")
    assert store.refresh_name(workspace.id, "renamed", "directory", relocation_confirmed=True)
    store.rename(workspace.id, "Mine")
    assert not store.refresh_name(workspace.id, "remote", "remote")

    catalog = tmp_path / "app" / "workspaces.db"
    with closing(sqlite3.connect(catalog)) as connection, connection:
        connection.execute(
            "UPDATE workspaces SET name='old',name_source='provider' WHERE workspace_id=?",
            (workspace.id,),
        )
    assert not store.refresh_name(workspace.id, "remote", "remote")
    assert store.refresh_name(workspace.id, "provider", "provider")
    assert not store.refresh_name(workspace.id, "provider", "provider")
    moved = tmp_path / "moved-provider"
    project.rename(moved)
    assert store.initialize(Workspace.discover(moved)).name == "provider"
    with closing(sqlite3.connect(catalog)) as connection, connection:
        connection.execute(
            "UPDATE workspaces SET name='Untitled',name_source='default' WHERE workspace_id=?",
            (workspace.id,),
        )
    assert store.refresh_name(workspace.id, "remote", "remote")
    with pytest.raises(ValueError, match="must not be empty"):
        store.refresh_name(workspace.id, " ", "remote")
    with pytest.raises(ValueError, match="Unknown"):
        store.refresh_name("missing", "name", "remote")


def test_repository_resolves_ids_and_covers_fresh_rekey_and_missing_locator(tmp_path: Path) -> None:
    """Resolver returns registered IDs while fresh rekeys and absent paths remain deterministic."""
    store = repository(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    workspace = store.initialize(Workspace.discover(project))
    assert store.resolve(workspace.id).id == workspace.id

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert store.rekey(fresh).root == fresh
    nonexistent = tmp_path / "nonexistent"
    initialized = store.initialize(Workspace.discover(nonexistent))
    assert initialized.id is not None


def test_repository_rolls_back_failed_rekey_transaction(tmp_path: Path) -> None:
    """A catalog write failure cannot leave a partially committed replacement identity."""
    store = repository(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    workspace = store.initialize(Workspace.discover(project))
    catalog = tmp_path / "app" / "workspaces.db"
    with closing(sqlite3.connect(catalog)) as connection, connection:
        connection.execute(
            "CREATE TRIGGER reject_rekey BEFORE INSERT ON workspaces "
            "BEGIN SELECT RAISE(ABORT, 'blocked'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="blocked"):
        store.rekey(project)
    assert store.list_by_path(project).id == workspace.id


def test_repository_relocates_legacy_identity_when_locators_are_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    """The conservative legacy fallback repairs a missing location without locator support."""
    original = tmp_path / "original"
    local = original / ".loop"
    local.mkdir(parents=True)
    legacy_store = WorkspaceRepository(local / "workspaces.db")
    legacy = legacy_store.initialize(Workspace.discover(original))
    store = repository(tmp_path)
    store.initialize(Workspace.discover(original))
    moved = tmp_path / "moved"
    original.rename(moved)
    monkeypatch.setattr(WorkspaceRepository, "_locator", staticmethod(lambda _path: (None, None)))

    assert store.initialize(Workspace.discover(moved)).id == legacy.id
