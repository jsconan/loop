"""Verify application directory reporting and operational exports."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from loop.application import ApplicationPaths
from loop.application.commands import ApplicationCommands
from loop.commands import CommandArgumentError, CommandContext
from loop.permissions import SQLitePermissionAudit


@pytest.fixture
def application(tmp_path):
    """Return application commands with bound isolated path references."""
    project = tmp_path / "project"
    project.mkdir()
    paths = ApplicationPaths(tmp_path / "config", tmp_path / "data", tmp_path / "state")
    workspace_paths = paths.for_workspace("workspace-id", project)
    return ApplicationCommands(paths, workspace_paths, "workspace-id"), paths


def test_dirs_prints_only_existing_global_and_workspace_paths(application):
    """Directory reporting omits paths that have not been created yet."""
    commands, paths = application
    interaction = Mock()
    paths.configuration_root.mkdir()
    paths.operational_log.parent.mkdir(parents=True)
    paths.operational_log.write_text("record\n", encoding="utf-8")

    commands.app(CommandContext("app", interaction))

    rows = interaction.table.call_args.args[0]
    assert {row["name"] for row in rows} == {
        "Configuration",
        "State",
        "Operational log",
        "Active workspace ID",
    }
    assert str(paths.configuration_root) == next(
        row["path"] for row in rows if row["name"] == "Configuration"
    )
    assert "workspace-id" == next(
        row["path"] for row in rows if row["name"] == "Active workspace ID"
    )
    assert all(Path(row["path"]).exists() for row in rows if row["name"] != "Active workspace ID")
    assert interaction.table.call_args.kwargs["columns"] == ("name", "path")
    assert commands.get_commands()[0].name == "app"


def test_logs_and_audit_export_create_files_and_filter_workspace(application, tmp_path):
    """Exports create new files and audit output is active-workspace scoped."""
    commands, paths = application
    interaction = Mock()
    context = CommandContext("app", interaction)
    log = paths.operational_log
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("record\n", encoding="utf-8")
    SQLitePermissionAudit(paths.permissions_audit).append(
        "workspace-id", "permission.decided", {"ok": True}
    )
    SQLitePermissionAudit(paths.permissions_audit).append(
        "other-workspace", "permission.decided", {"ok": False}
    )
    log_export = tmp_path / "exports" / "log.jsonl"
    audit_export = tmp_path / "exports" / "audit.jsonl"

    commands.app(context, "logs", str(log_export))
    commands.app(context, "audit", str(audit_export))

    exported_log = json.loads(log_export.read_text(encoding="utf-8"))
    assert exported_log["message"] == "record"
    assert exported_log["level"] == "UNKNOWN"
    audit_records = audit_export.read_text(encoding="utf-8").splitlines()
    assert len(audit_records) == 1
    assert "workspace-id" in audit_records[0]
    assert "other-workspace" not in audit_records[0]
    with pytest.raises(CommandArgumentError, match="already exists"):
        commands.app(context, "logs", str(log_export))
    with pytest.raises(CommandArgumentError, match="already exists"):
        commands.app(context, "audit", str(audit_export))


def test_audit_export_creates_empty_file_when_no_records(application, tmp_path):
    """Audit export succeeds with nested destinations when no records exist."""
    commands, _ = application
    context = CommandContext("app", Mock())
    target = tmp_path / "nested" / "audit.jsonl"

    commands.app(context, "audit", str(target))

    assert target.read_text(encoding="utf-8") == ""


def test_application_commands_validate_arguments_and_empty_log(application, tmp_path):
    """Operations reject invalid destinations and create an empty export for a missing log."""
    commands, _ = application
    context = CommandContext("app", Mock())

    with pytest.raises(CommandArgumentError, match="does not accept"):
        commands.app(context, "dirs", "unused")
    with pytest.raises(CommandArgumentError, match="requires"):
        commands.app(context, "audit")
    target = tmp_path / "empty.log"
    commands.app(context, "logs", str(target))
    assert target.read_bytes() == b""


def test_log_export_streams_archives_with_filters_and_force(application, tmp_path):
    """Rotated JSON logs export oldest-first across boundaries with exact filters."""
    commands, paths = application
    paths.operational_log.parent.mkdir(parents=True, exist_ok=True)
    (paths.operational_log.parent / "loop.log.2").write_text(
        '{"timestamp_ns":1,"level":"INFO","workspace_id":"workspace-id","event.name":"keep"}\n',
        encoding="utf-8",
    )
    (paths.operational_log.parent / "loop.log.1").write_text(
        "malformed\n"
        '{"timestamp_ns":2,"level":"ERROR","workspace_id":"workspace-id","event.name":"skip"}\n',
        encoding="utf-8",
    )
    paths.operational_log.write_text(
        '{"timestamp_ns":3,"level":"INFO","workspace_id":"workspace-id","event.name":"keep"}\n'
        '{"timestamp_ns":4,"level":"INFO","workspace_id":"other","event.name":"keep"}\n'
        '{"timestamp_ns":"unknown","level":"INFO","workspace_id":"workspace-id","event.name":"keep"}\n',
        encoding="utf-8",
    )
    target = tmp_path / "logs.jsonl"
    context = CommandContext("app", Mock())

    commands.app(context, "logs", str(target), "workspace-id", 1, 3, "info", "keep")
    assert [json.loads(line)["timestamp_ns"] for line in target.read_text().splitlines()] == [1, 3]
    commands.app(context, "logs", str(target), force=True)
    assert any(
        json.loads(line).get("message") == "malformed" for line in target.read_text().splitlines()
    )
    with pytest.raises(CommandArgumentError, match="audit-only"):
        commands.app(context, "logs", str(tmp_path / "bad"), session_id="session")
    empty = tmp_path / "empty-filtered"
    commands.app(context, "logs", str(empty), "workspace-id", 5, 6, "warning", "missing")
    assert empty.read_text(encoding="utf-8") == ""
    wrong_workspace = tmp_path / "wrong-workspace"
    commands.app(context, "logs", str(wrong_workspace), "absent")
    assert wrong_workspace.read_text(encoding="utf-8") == ""
    wrong_event = tmp_path / "wrong-event"
    commands.app(context, "logs", str(wrong_event), event_name="absent")
    assert wrong_event.read_text(encoding="utf-8") == ""


def test_log_export_filters_iso_timestamps_and_rejects_source_overwrite(application, tmp_path):
    """Older ISO timestamps remain filterable and exports cannot replace their own input."""
    commands, paths = application
    paths.operational_log.parent.mkdir(parents=True, exist_ok=True)
    paths.operational_log.write_text(
        '{"timestamp":"1970-01-01T00:00:01+00:00","level":"INFO"}\n'
        '{"timestamp":"invalid","level":"INFO"}\n',
        encoding="utf-8",
    )
    target = tmp_path / "filtered.jsonl"
    commands.app(CommandContext("app", Mock()), "logs", str(target), start_ns=1_000_000_000)
    assert json.loads(target.read_text())["timestamp_ns"] == 1_000_000_000
    with pytest.raises(CommandArgumentError, match="must not be"):
        commands.app(CommandContext("app", Mock()), "logs", str(paths.operational_log), force=True)


def test_audit_export_supports_all_filters_and_force(application, tmp_path):
    """Audit exports filter workspace, session, time, event, and decision without buffering."""
    commands, paths = application
    audit = SQLitePermissionAudit(paths.permissions_audit)
    audit.append("workspace-id", "permission.decided", {"session_id": "s", "decision": "allow"})
    target = tmp_path / "audit.jsonl"
    commands.app(
        CommandContext("app", Mock()),
        "audit",
        str(target),
        "workspace-id",
        0,
        2**63 - 1,
        event_name="permission.decided",
        session_id="s",
        decision="allow",
    )
    assert len(target.read_text(encoding="utf-8").splitlines()) == 1
    filtered = tmp_path / "filtered.jsonl"
    commands.app(
        CommandContext("app", Mock()),
        "audit",
        str(filtered),
        session_id="other",
        decision="deny",
    )
    assert filtered.read_text(encoding="utf-8") == ""
    decision_filtered = tmp_path / "decision-filtered.jsonl"
    commands.app(
        CommandContext("app", Mock()),
        "audit",
        str(decision_filtered),
        session_id="s",
        decision="deny",
    )
    assert decision_filtered.read_text(encoding="utf-8") == ""
    commands.app(CommandContext("app", Mock()), "audit", str(target), force=True)
    with pytest.raises(CommandArgumentError, match="severity"):
        commands.app(CommandContext("app", Mock()), "audit", str(tmp_path / "bad"), severity="info")
    with pytest.raises(CommandArgumentError, match="dirs"):
        commands.app(CommandContext("app", Mock()), "dirs", force=True)
