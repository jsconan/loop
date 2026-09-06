"""Verify application-level coordination of legacy artifact imports."""

import json
import sqlite3
from contextlib import closing

import pytest

from loop.application import ApplicationMigration, ApplicationPaths
from loop.workspace import Workspace


def application_paths(tmp_path) -> ApplicationPaths:
    """Return isolated application roots."""
    return ApplicationPaths(tmp_path / "config", tmp_path / "data", tmp_path / "state")


def initialized_workspace(project) -> Workspace:
    """Return an initialized workspace rooted at a test project."""
    return Workspace(project, project, "id", "project", "directory", 1, 1)


def test_migration_coordinates_owned_importers_and_copies_policy(tmp_path) -> None:
    """Every legacy artifact reaches its central owner without deleting its source."""
    project = tmp_path / "project"
    legacy = project / ".loop"
    legacy.mkdir(parents=True)
    sessions_source = legacy / "sessions.db"
    with closing(sqlite3.connect(sessions_source)) as connection, connection:
        connection.execute("CREATE TABLE records (value TEXT NOT NULL)")
        connection.execute("INSERT INTO records VALUES ('preserved')")
    telemetry_source = legacy / "telemetry.db"
    with closing(sqlite3.connect(telemetry_source)) as connection, connection:
        connection.execute(
            "CREATE TABLE telemetry_records(record_id TEXT PRIMARY KEY,timestamp_ns INTEGER NOT NULL,"
            "observed_ns INTEGER NOT NULL,signal TEXT NOT NULL,event_name TEXT NOT NULL,severity TEXT,"
            "workspace_id TEXT,session_id TEXT,message_sequence INTEGER,event_sequence INTEGER NOT NULL,"
            "trace_id TEXT,span_id TEXT,parent_span_id TEXT,attributes TEXT NOT NULL,payload_id INTEGER,"
            "payload_sha256 TEXT,schema_version INTEGER NOT NULL)"
        )
        connection.execute(
            "INSERT INTO telemetry_records VALUES "
            "('record',1,1,'activity','legacy','info',NULL,NULL,NULL,1,NULL,NULL,NULL,'{}',NULL,NULL,1)"
        )
    (legacy / "permissions.yaml").write_text("version: 2\n", encoding="utf-8")
    (legacy / "permissions-audit.jsonl").write_text(
        json.dumps({"record_id": "audit", "event_name": "decision"}) + "\n",
        encoding="utf-8",
    )
    (legacy / "loop.log").write_text('{"event.name":"legacy"}\n', encoding="utf-8")
    workspace = initialized_workspace(project)
    paths = application_paths(tmp_path)
    workspace_paths = paths.for_workspace(workspace.id, workspace.root)

    migration = ApplicationMigration(workspace, paths, workspace_paths)
    migration.run()
    migration.run()

    with closing(sqlite3.connect(workspace_paths.sessions)) as connection:
        assert connection.execute("SELECT value FROM records").fetchone() == ("preserved",)
    with closing(sqlite3.connect(paths.telemetry)) as connection:
        assert connection.execute(
            "SELECT record_id, workspace_id FROM telemetry_records"
        ).fetchall() == [("record", "id")]
    with closing(sqlite3.connect(paths.permissions_audit)) as connection:
        assert connection.execute(
            "SELECT record_id, workspace_id FROM permission_audit_records"
        ).fetchall() == [("audit", "id")]
    assert workspace_paths.permissions.read_text(encoding="utf-8") == "version: 2\n"
    assert paths.operational_log.read_text(encoding="utf-8") == '{"event.name":"legacy"}\n'
    assert all(path.exists() for path in legacy.iterdir())
    assert not (paths.data_root / "application.db").exists()


def test_migration_preserves_existing_central_files(tmp_path) -> None:
    """Opaque files already owned by central storage are never overwritten."""
    project = tmp_path / "project"
    legacy = project / ".loop"
    legacy.mkdir(parents=True)
    (legacy / "permissions.yaml").write_text("legacy\n", encoding="utf-8")
    (legacy / "loop.log").write_text("legacy\n", encoding="utf-8")
    workspace = initialized_workspace(project)
    paths = application_paths(tmp_path)
    workspace_paths = paths.for_workspace(workspace.id, workspace.root)
    workspace_paths.permissions.parent.mkdir(parents=True)
    workspace_paths.permissions.write_text("central\n", encoding="utf-8")
    paths.operational_log.parent.mkdir(parents=True)
    paths.operational_log.write_text("central\n", encoding="utf-8")

    ApplicationMigration(workspace, paths, workspace_paths).run()

    assert workspace_paths.permissions.read_text(encoding="utf-8") == "central\n"
    assert paths.operational_log.read_text(encoding="utf-8") == "central\n"


def test_migration_requires_identity_and_positive_timeout(tmp_path) -> None:
    """Migration rejects incomplete ownership and lock configuration."""
    paths = application_paths(tmp_path)
    workspace = Workspace(tmp_path, tmp_path)
    with pytest.raises(ValueError, match="initialized"):
        ApplicationMigration(workspace, paths, paths.for_workspace("id", tmp_path))
    initialized = initialized_workspace(tmp_path)
    with pytest.raises(ValueError, match="busy timeout"):
        ApplicationMigration(
            initialized,
            paths,
            paths.for_workspace("id", tmp_path),
            busy_timeout_ms=0,
        )
