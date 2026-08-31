"""Tests for the JSON Lines telemetry persistence adapter."""

import json
from dataclasses import replace

import pytest

from loop.telemetry import JSONLTelemetryAdapter


def test_jsonl_adapter_writes_complete_canonical_records(tmp_path, telemetry_record):
    """JSONL persistence creates its parent lazily and emits complete objects."""
    path = tmp_path / "nested" / "telemetry.jsonl"
    adapter = JSONLTelemetryAdapter(path, workspace_id=telemetry_record.workspace_id)
    adapter.write_batch(())
    adapter.write_batch((telemetry_record,))
    adapter.flush()
    adapter.close()

    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["payload"] == {"value": "complete"}
    assert value["workspace_id"] == telemetry_record.workspace_id
    assert value["trace_id"] == "a" * 32
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600


def test_jsonl_adapter_rejects_an_empty_workspace_identifier(tmp_path):
    """JSONL persistence requires a non-empty configured workspace identifier."""
    with pytest.raises(ValueError, match="must not be empty"):
        JSONLTelemetryAdapter(tmp_path / "telemetry.jsonl", workspace_id="")


def test_jsonl_adapter_rejects_records_from_another_workspace(tmp_path, telemetry_record):
    """JSONL persistence rejects records that belong to another workspace."""
    adapter = JSONLTelemetryAdapter(tmp_path / "telemetry.jsonl", workspace_id="workspace")

    with pytest.raises(ValueError, match="does not match storage workspace"):
        adapter.write_batch((replace(telemetry_record, workspace_id="another"),))
