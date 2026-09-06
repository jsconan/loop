"""Verify centralized SQLite permission audit persistence and export."""

import json
import sqlite3

import pytest

import loop.permissions.audit as audit_module
from loop.permissions import SQLitePermissionAudit


def test_audit_appends_private_records_and_exports_filtered_jsonl(tmp_path):
    """Central audit records retain workspace identity and export a stable filtered schema."""
    audit = SQLitePermissionAudit(tmp_path / "state" / "audit.db")
    audit.append("first", "permission.decided", {"decision": "allow"})
    audit.append("second", "permission.decided", {"decision": "deny"})

    destination = audit.export_jsonl(tmp_path / "exports" / "audit.jsonl", workspace_id="first")
    record = json.loads(destination.read_text(encoding="utf-8"))

    assert record["workspace_id"] == "first"
    assert record["payload"] == {"decision": "allow"}
    assert audit.path.stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(audit.path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_audit_validates_inputs_and_refuses_export_overwrite(tmp_path):
    """Invalid audit configuration and destructive exports fail explicitly."""
    with pytest.raises(ValueError, match="busy timeout"):
        SQLitePermissionAudit(tmp_path / "audit.db", busy_timeout_ms=0)
    audit = SQLitePermissionAudit(tmp_path / "audit.db")
    with pytest.raises(ValueError, match="non-empty"):
        audit.append("", "event", {})
    with pytest.raises(ValueError, match="workspace"):
        audit.import_legacy_jsonl(tmp_path / "missing.jsonl", workspace_id="")
    assert audit.import_legacy_jsonl(tmp_path / "missing.jsonl", workspace_id="workspace") == 0
    destination = audit.export_jsonl(tmp_path / "audit.jsonl")
    with pytest.raises(FileExistsError, match="already exists"):
        audit.export_jsonl(destination)


def test_audit_imports_legacy_jsonl_idempotently_and_rolls_back_errors(tmp_path):
    """Legacy audit imports deduplicate identities and remain atomic on malformed input."""
    audit = SQLitePermissionAudit(tmp_path / "audit.db")
    source = tmp_path / "legacy.jsonl"
    source.write_text(
        json.dumps({"event_name": "first", "payload": {"decision": "allow"}})
        + "\n"
        + json.dumps({"record_id": "second", "timestamp_ns": 2})
        + "\n",
        encoding="utf-8",
    )

    assert audit.import_legacy_jsonl(source, workspace_id="workspace") == 2
    assert audit.import_legacy_jsonl(source, workspace_id="workspace") == 0
    source.write_text(json.dumps({"record_id": "third"}) + "\ninvalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        audit.import_legacy_jsonl(source, workspace_id="workspace")

    with sqlite3.connect(audit.path) as connection:
        assert (
            connection.execute(
                "SELECT record_id FROM permission_audit_records WHERE record_id = 'third'"
            ).fetchone()
            is None
        )


def test_audit_export_applies_payload_time_and_event_filters(tmp_path, monkeypatch):
    """Audit exports apply inclusive time bounds and all optional record filters."""
    timestamps = iter((10, 20, 30))
    monkeypatch.setattr(audit_module.time, "time_ns", lambda: next(timestamps))
    audit = SQLitePermissionAudit(tmp_path / "audit.db")
    audit.append("workspace", "permission.decided", {"session_id": "session", "decision": "allow"})
    audit.append("workspace", "permission.decided", {"session_id": "other", "decision": "deny"})
    audit.append("workspace", "permission.checked", {"session_id": "session", "decision": "allow"})

    destination = audit.export_jsonl(
        tmp_path / "filtered.jsonl",
        workspace_id="workspace",
        session_id="session",
        start_ns=10,
        end_ns=30,
        decision="allow",
        event_name="permission.decided",
    )
    records = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 1
    assert records[0]["timestamp_ns"] == 10
    assert records[0]["event_name"] == "permission.decided"
    assert records[0]["payload"] == {"decision": "allow", "session_id": "session"}

    empty = audit.export_jsonl(
        tmp_path / "empty.jsonl",
        start_ns=20,
        end_ns=20,
        session_id="session",
    )
    assert empty.read_text(encoding="utf-8") == ""

    forced = audit.export_jsonl(destination, force=True, event_name="permission.checked")
    assert len(forced.read_text(encoding="utf-8").splitlines()) == 1
