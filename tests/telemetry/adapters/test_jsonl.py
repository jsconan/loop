"""Tests for the JSON Lines telemetry persistence adapter."""

import json

from loop.telemetry import JSONLTelemetryAdapter


def test_jsonl_adapter_writes_complete_canonical_records(tmp_path, telemetry_record):
    """JSONL persistence creates its parent lazily and emits complete objects."""
    path = tmp_path / "nested" / "telemetry.jsonl"
    adapter = JSONLTelemetryAdapter(path)
    adapter.write_batch(())
    adapter.write_batch((telemetry_record,))
    adapter.flush()
    adapter.close()

    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["payload"] == {"value": "complete"}
    assert value["trace_id"] == "a" * 32
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600
