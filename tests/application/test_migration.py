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
    imported_log = json.loads(paths.operational_log.read_text(encoding="utf-8"))
    assert imported_log["event.name"] == "legacy"
    assert imported_log["migration_id"]
    assert all(path.exists() for path in legacy.iterdir())
    marker = legacy / ".central-storage-v1"
    assert marker.read_bytes() == b""
    assert marker.stat().st_mode & 0o777 == 0o600


def test_migration_preserves_policy_and_appends_to_existing_central_log(tmp_path) -> None:
    """Opaque policies remain untouched while every workspace log is imported once."""
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
    lines = paths.operational_log.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "central"
    assert json.loads(lines[1])["message"] == "legacy"


def test_migration_ignores_obsolete_application_database(tmp_path) -> None:
    """An obsolete migration database cannot prevent source-local migration."""
    project = tmp_path / "project"
    legacy = project / ".loop"
    legacy.mkdir(parents=True)
    (legacy / "permissions.yaml").write_text("legacy\n", encoding="utf-8")
    workspace = initialized_workspace(project)
    paths = application_paths(tmp_path)
    paths.data_root.mkdir(parents=True)
    obsolete = paths.data_root / "application.db"
    with closing(sqlite3.connect(obsolete)) as connection, connection:
        connection.execute("CREATE TABLE migrations(kind TEXT NOT NULL)")

    ApplicationMigration(workspace, paths, paths.for_workspace("id", project)).run()

    assert paths.for_workspace("id", project).permissions.read_text(encoding="utf-8") == "legacy\n"
    assert (legacy / ".central-storage-v1").is_file()


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


def test_migration_marks_only_a_complete_import_and_retries_after_failure(tmp_path) -> None:
    """A failed import remains unmarked and retries after its source is corrected."""
    project = tmp_path / "project"
    legacy = project / ".loop"
    legacy.mkdir(parents=True)
    audit = legacy / "permissions-audit.jsonl"
    audit.write_text("malformed\n", encoding="utf-8")
    workspace = initialized_workspace(project)
    paths = application_paths(tmp_path)
    migration = ApplicationMigration(workspace, paths, paths.for_workspace("id", project))

    with pytest.raises(ValueError, match="Malformed"):
        migration.run()
    marker = legacy / ".central-storage-v1"
    assert not marker.exists()
    audit.write_text('{"record_id":"fixed"}\n', encoding="utf-8")
    migration.run()
    assert marker.is_file()


def test_migration_skips_absent_sources_completed_directories_and_marker_failures(
    tmp_path, monkeypatch
) -> None:
    """Migration needs no state and does not let an unwritable marker block imported data."""
    project = tmp_path / "project"
    legacy = project / ".loop"
    legacy.mkdir(parents=True)
    workspace = initialized_workspace(project)
    paths = application_paths(tmp_path)
    migration = ApplicationMigration(workspace, paths, paths.for_workspace("id", project))

    migration.run()
    assert not (legacy / ".central-storage-v1").exists()
    (legacy / ".central-storage-v1").touch()
    (legacy / "permissions.yaml").write_text("legacy\n", encoding="utf-8")
    migration.run()
    assert not paths.for_workspace("id", project).permissions.exists()

    (legacy / ".central-storage-v1").unlink()
    marker = legacy / ".central-storage-v1"
    chmod = type(marker).chmod

    def reject_marker_chmod(path, mode, *, follow_symlinks=True):
        """Reject only the source-local marker permission update."""
        if path == marker:
            raise PermissionError("read only")
        return chmod(path, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr("loop.application.migration.Path.touch", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("loop.application.migration.Path.chmod", reject_marker_chmod)
    migration.run()
