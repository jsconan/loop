"""Tests for the SQLite telemetry persistence adapter."""

import json
import sqlite3
from contextlib import closing
from dataclasses import replace

from loop.telemetry import SQLiteTelemetryAdapter


def test_sqlite_adapter_separates_payloads_and_indexes_metadata(tmp_path, telemetry_record):
    """SQLite batches store indexed metadata separately from complete payload bytes."""
    path = tmp_path / ".loop" / "telemetry.db"
    adapter = SQLiteTelemetryAdapter(path)
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
            "SELECT event_name, payload_sha256 FROM telemetry_records"
        ).fetchall()
        payload = connection.execute("SELECT payload FROM telemetry_payloads").fetchone()[0]
        indexes = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    assert metadata == [("test.event", "digest"), ("test.event", "digest")]
    assert json.loads(payload) == {"value": "complete"}
    assert ("ix_telemetry_trace_time",) in indexes
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600
