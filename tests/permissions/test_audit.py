"""Verify centralized SQLite permission audit persistence and export."""

import json
import sqlite3

import pytest

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
    destination = audit.export_jsonl(tmp_path / "audit.jsonl")
    with pytest.raises(FileExistsError, match="already exists"):
        audit.export_jsonl(destination)
