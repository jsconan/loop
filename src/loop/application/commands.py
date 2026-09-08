"""Expose application paths and operational exports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from ..commands import CommandArgumentError, CommandContext, CommandRegistration
from ..completion import CommandCompletion, CompletionValue
from ..permissions import SQLitePermissionAudit
from .paths import ApplicationPaths, WorkspacePaths


class ApplicationCommands:
    """Expose application directories, logs, and permission audit exports.

    Args:
        paths (ApplicationPaths): Immutable global application paths.
        workspace_paths (WorkspacePaths): Immutable active-workspace storage paths.
        workspace_id (str): Active initialized workspace identifier.
    """

    _paths: ApplicationPaths
    _workspace_paths: WorkspacePaths
    _workspace_id: str

    def __init__(
        self,
        paths: ApplicationPaths,
        workspace_paths: WorkspacePaths,
        workspace_id: str,
    ) -> None:
        self._paths = paths
        self._workspace_paths = workspace_paths
        self._workspace_id = workspace_id

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return the application command registration.

        Returns:
            tuple[CommandRegistration, ...]: The ``/app`` command.
        """
        return (
            CommandRegistration(
                self.app,
                name="app",
                completion=CommandCompletion(
                    values=(
                        CompletionValue("dirs", "Show application directories."),
                        CompletionValue("audit", "Export audit logs."),
                        CompletionValue("logs", "Export application logs."),
                    ),
                    children={
                        "dirs": CommandCompletion(),
                        "audit": CommandCompletion(),
                        "logs": CommandCompletion(),
                    },
                ),
            ),
        )

    def app(
        self,
        context: CommandContext,
        action: Annotated[
            Literal["dirs", "logs", "audit"], Field(description="Application operation.")
        ] = "dirs",
        destination: Annotated[
            str | None, Field(description="New export path for logs or audit records.")
        ] = None,
        workspace_id: Annotated[str | None, Field(description="Workspace filter.")] = None,
        start_ns: Annotated[int | None, Field(description="Inclusive start timestamp.")] = None,
        end_ns: Annotated[int | None, Field(description="Inclusive end timestamp.")] = None,
        severity: Annotated[str | None, Field(description="Log severity filter.")] = None,
        event_name: Annotated[str | None, Field(description="Event-name filter.")] = None,
        session_id: Annotated[str | None, Field(description="Audit session filter.")] = None,
        decision: Annotated[str | None, Field(description="Audit decision filter.")] = None,
        force: Annotated[bool, Field(description="Allow replacing an export file.")] = False,
    ) -> None:
        """Print application directories or export operational records."""
        if action == "dirs":
            if (
                any(
                    value is not None
                    for value in (
                        destination,
                        workspace_id,
                        start_ns,
                        end_ns,
                        severity,
                        event_name,
                        session_id,
                        decision,
                    )
                )
                or force
            ):
                raise CommandArgumentError("The dirs operation does not accept a destination.")
            context.interaction.table(
                self._directories(),
                title="Application directories:",
                columns=("name", "path"),
            )
            return
        if destination is None:
            raise CommandArgumentError(f"The {action} operation requires a destination.")
        target = Path(destination).expanduser().resolve()
        try:
            if action == "logs":
                if session_id is not None or decision is not None:
                    raise CommandArgumentError("Log exports do not accept audit-only filters.")
                self._export_logs(
                    target,
                    workspace_id=workspace_id,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    severity=severity,
                    event_name=event_name,
                    force=force,
                )
            else:
                if severity is not None:
                    raise CommandArgumentError("Audit exports do not accept severity filters.")
                self._export_audit(
                    target,
                    workspace_id=workspace_id,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    event_name=event_name,
                    session_id=session_id,
                    decision=decision,
                    force=force,
                )
        except (FileExistsError, ValueError) as error:
            raise CommandArgumentError(str(error)) from error
        context.interaction.info(f"Exported {action} to {target}.")

    def _directories(self) -> tuple[dict[str, str], ...]:
        """Return resolved application and active workspace paths as table rows."""
        paths = self._paths
        return tuple(
            {"name": name, "path": str(path)}
            for name, path in (
                ("Configuration", paths.configuration_root),
                ("Data", paths.data_root),
                ("State", paths.state_root),
                ("User configuration", paths.user_configuration),
                ("Workspace registry", paths.workspace_catalog),
                ("Telemetry", paths.telemetry),
                ("Audit", paths.permissions_audit),
                ("Operational log", paths.operational_log),
                ("Active workspace ID", self._workspace_id),
                ("Workspace data", self._workspace_paths.data),
                ("Local override", self._workspace_paths.configuration),
            )
            if not isinstance(path, Path) or path.exists()
        )

    def _export_audit(self, destination: Path, **filters: object) -> None:
        """Export the current permissions audit without overwriting a destination."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        SQLitePermissionAudit(self._paths.permissions_audit).export_jsonl(
            destination,
            workspace_id=filters.pop("workspace_id") or self._workspace_id,
            **filters,
        )

    # pylint: disable-next=too-many-branches
    def _export_logs(
        self,
        destination: Path,
        *,
        workspace_id: str | None,
        start_ns: int | None,
        end_ns: int | None,
        severity: str | None,
        event_name: str | None,
        force: bool,
    ) -> None:
        """Stream matching rotated and active global log records in chronological order."""
        if destination.exists() and not force:
            raise FileExistsError(f"Log export already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_path = self._paths.operational_log
        archives = sorted(
            source_path.parent.glob(f"{source_path.name}.*"),
            key=lambda path: int(path.suffix[1:]) if path.suffix[1:].isdigit() else -1,
            reverse=True,
        )
        sources = [*archives, source_path]
        if destination in {path.resolve() for path in sources}:
            raise ValueError("Log export destination must not be an active or rotated log file.")
        with destination.open("w" if force else "x", encoding="utf-8") as output:
            for source_path in sources:
                if not source_path.is_file():
                    continue
                with source_path.open(encoding="utf-8") as source:
                    for line in source:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            record = {
                                "timestamp": None,
                                "timestamp_ns": None,
                                "workspace_id": None,
                                "level": "UNKNOWN",
                                "event.name": None,
                                "message": line.rstrip("\r\n"),
                            }
                        timestamp = record.get("timestamp_ns")
                        if not isinstance(timestamp, int) and isinstance(
                            record.get("timestamp"), str
                        ):
                            try:
                                timestamp = int(
                                    datetime.fromisoformat(record["timestamp"]).timestamp()
                                    * 1_000_000_000
                                )
                            except ValueError:
                                timestamp = None
                        record["timestamp_ns"] = timestamp
                        if start_ns is not None and (
                            not isinstance(timestamp, int) or timestamp < start_ns
                        ):
                            continue
                        if end_ns is not None and (
                            not isinstance(timestamp, int) or timestamp > end_ns
                        ):
                            continue
                        if workspace_id is not None and record.get("workspace_id") != workspace_id:
                            continue
                        if (
                            severity is not None
                            and record.get("level", "").lower() != severity.lower()
                        ):
                            continue
                        if event_name is not None and record.get("event.name") != event_name:
                            continue
                        output.write(
                            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                        )
