"""Verify application directory reporting and operational exports."""

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

    assert log_export.read_text(encoding="utf-8") == "record\n"
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
