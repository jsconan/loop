"""Tests for the SQLite telemetry persistence adapter."""

import json
import sqlite3
from contextlib import closing
from dataclasses import replace

import pytest

from loop.telemetry import SQLiteTelemetryAdapter


def test_sqlite_adapter_separates_payloads_and_indexes_metadata(tmp_path, telemetry_record):
    """SQLite batches store indexed metadata separately from complete payload bytes."""
    path = tmp_path / ".loop" / "telemetry.db"
    adapter = SQLiteTelemetryAdapter(path, workspace_id=telemetry_record.workspace_id)
    assert adapter.path == path
    adapter.flush()
    adapter.write_batch(())
    adapter.write_batch((telemetry_record,))
    adapter.write_batch(
        (replace(telemetry_record, record_id="tel_second", event_sequence=4, payload=None),)
    )
    adapter.flush()
    adapter.close()
    adapter.close()

    with closing(sqlite3.connect(path)) as connection:
        metadata = connection.execute(
            "SELECT event_name, payload_sha256, workspace_id FROM telemetry_records"
        ).fetchall()
        payload = connection.execute("SELECT payload FROM telemetry_payloads").fetchone()[0]
        indexes = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    assert metadata == [
        ("test.event", "digest", telemetry_record.workspace_id),
        ("test.event", "digest", telemetry_record.workspace_id),
    ]
    assert json.loads(payload) == {"value": "complete"}
    assert ("ix_telemetry_trace_time",) in indexes
    assert ("ix_telemetry_workspace_time",) in indexes
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600


def test_sqlite_adapter_rejects_non_positive_busy_timeouts(tmp_path):
    """SQLite configuration rejects timeouts that cannot make progress."""
    with pytest.raises(ValueError, match="busy timeout"):
        SQLiteTelemetryAdapter(
            tmp_path / "telemetry.db", workspace_id="workspace", busy_timeout_ms=0
        )


def test_sqlite_adapter_rejects_an_empty_workspace_identifier(tmp_path):
    """Telemetry storage requires an explicit workspace link."""
    with pytest.raises(ValueError, match="Workspace identifier"):
        SQLiteTelemetryAdapter(tmp_path / "telemetry.db", workspace_id="")


def test_sqlite_adapter_rejects_records_owned_by_another_workspace(tmp_path, telemetry_record):
    """Telemetry storage cannot silently accept another workspace's records."""
    adapter = SQLiteTelemetryAdapter(tmp_path / "telemetry.db", workspace_id="another")

    with pytest.raises(ValueError, match="does not match"):
        adapter.write_batch((telemetry_record,))

    adapter.close()


def test_sqlite_adapter_rolls_back_an_invalid_existing_telemetry_schema(tmp_path):
    """Invalid telemetry schemas fail without leaving an open migration transaction."""
    path = tmp_path / "telemetry.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """
            CREATE TABLE telemetry_records (
                record_id TEXT PRIMARY KEY,
                timestamp_ns INTEGER NOT NULL,
                event_name TEXT NOT NULL,
                session_id TEXT,
                event_sequence INTEGER NOT NULL,
                trace_id TEXT,
                span_id TEXT,
                parent_span_id TEXT
            )
            """
        )

    with pytest.raises(sqlite3.OperationalError, match="no such column: attributes"):
        SQLiteTelemetryAdapter(path, workspace_id="workspace")


def test_sqlite_adapter_migrates_existing_records_to_the_registered_workspace(tmp_path):
    """Existing telemetry gains workspace identity and no longer discloses paths."""
    path = tmp_path / "telemetry.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """
            CREATE TABLE telemetry_records (
                record_id TEXT PRIMARY KEY,
                timestamp_ns INTEGER NOT NULL DEFAULT 0,
                observed_ns INTEGER NOT NULL DEFAULT 0,
                signal TEXT NOT NULL DEFAULT 'trace',
                event_name TEXT NOT NULL DEFAULT 'legacy',
                severity TEXT,
                session_id TEXT,
                message_sequence INTEGER,
                event_sequence INTEGER NOT NULL DEFAULT 0,
                trace_id TEXT,
                span_id TEXT,
                parent_span_id TEXT,
                attributes TEXT NOT NULL,
                payload_id INTEGER,
                payload_sha256 TEXT,
                schema_version INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        connection.executemany(
            "INSERT INTO telemetry_records(record_id, attributes) VALUES (?, ?)",
            (
                ("valid", '{"workspace.root":"/secret/workspace","model":"test"}'),
                ("ordinary", '{"model":"test"}'),
                ("malformed", "not-json"),
            ),
        )
        connection.commit()

    workspace_id = "workspace"
    adapter = SQLiteTelemetryAdapter(path, workspace_id=workspace_id)
    adapter.close()

    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute(
            "SELECT record_id, workspace_id, attributes FROM telemetry_records ORDER BY record_id"
        ).fetchall()
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert rows == [
        ("malformed", workspace_id, "not-json"),
        ("ordinary", workspace_id, '{"model":"test"}'),
        ("valid", workspace_id, '{"model":"test"}'),
    ]
    assert schema_version == 1


def test_sqlite_adapter_does_not_repeat_completed_record_migrations(tmp_path):
    """A recorded schema version prevents record scans and rewrites on later startups."""
    path = tmp_path / "telemetry.db"
    adapter = SQLiteTelemetryAdapter(path, workspace_id="workspace")
    adapter.close()
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """
            INSERT INTO telemetry_records(
                record_id, timestamp_ns, observed_ns, signal, event_name, event_sequence,
                workspace_id, attributes, schema_version
            ) VALUES ('later', 0, 0, 'trace', 'test', 0, NULL, ?, 2)
            """,
            ('{"model":"test"}',),
        )
        connection.commit()

    reopened = SQLiteTelemetryAdapter(path, workspace_id="workspace")
    reopened.close()

    with closing(sqlite3.connect(path)) as connection:
        row = connection.execute(
            "SELECT workspace_id, attributes FROM telemetry_records WHERE record_id = 'later'"
        ).fetchone()
    assert row == (None, '{"model":"test"}')


def test_sqlite_adapter_rejects_a_newer_database_schema(tmp_path):
    """Opening storage from a newer Loop version fails instead of risking corruption."""
    path = tmp_path / "telemetry.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA user_version=2")

    with pytest.raises(RuntimeError, match="newer than supported"):
        SQLiteTelemetryAdapter(path, workspace_id="workspace")
