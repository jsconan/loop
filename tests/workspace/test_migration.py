"""Verify safe migration of legacy workspace artifacts."""

import sqlite3
from contextlib import closing

import pytest

from loop.workspace import Workspace, WorkspaceMigration


def test_migration_uses_a_consistent_sqlite_snapshot_and_copies_policy(tmp_path) -> None:
    """Session data in WAL mode and permission policy reach central storage intact."""
    project = tmp_path / "project"
    legacy = project / ".loop"
    legacy.mkdir(parents=True)
    source = legacy / "sessions.db"
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE records (value TEXT NOT NULL)")
        connection.execute("INSERT INTO records VALUES ('preserved')")
    (legacy / "permissions.yaml").write_text("version: 2\n", encoding="utf-8")
    workspace = Workspace(project, project, "id", "project", "directory", 1, 1)
    sessions = tmp_path / "central" / "sessions.db"
    permissions = tmp_path / "central" / "permissions.yaml"

    WorkspaceMigration(workspace, sessions, permissions).run()

    with closing(sqlite3.connect(sessions)) as connection:
        assert connection.execute("SELECT value FROM records").fetchone() == ("preserved",)
    assert permissions.read_text(encoding="utf-8") == "version: 2\n"
    permissions.write_text("central\n", encoding="utf-8")
    WorkspaceMigration(workspace, sessions, permissions).run()
    assert permissions.read_text(encoding="utf-8") == "central\n"


def test_migration_requires_initialized_workspace(tmp_path) -> None:
    """Migration cannot target storage without a durable workspace identity."""
    with pytest.raises(ValueError, match="initialized"):
        WorkspaceMigration(Workspace(tmp_path, tmp_path), tmp_path / "s", tmp_path / "p")
