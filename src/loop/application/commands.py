"""Expose application paths and operational exports."""

from __future__ import annotations

import shutil
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
    ) -> None:
        """Print application directories or export operational records."""
        if action == "dirs":
            if destination is not None:
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
                self._export_logs(target)
            else:
                self._export_audit(target)
        except FileExistsError as error:
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

    def _export_audit(self, destination: Path) -> None:
        """Export the current permissions audit without overwriting a destination."""
        if destination.exists():
            raise FileExistsError(f"Audit export already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        SQLitePermissionAudit(self._paths.permissions_audit).export_jsonl(
            destination, workspace_id=self._workspace_id
        )

    def _export_logs(self, destination: Path) -> None:
        """Copy the current global operational log without overwriting a destination."""
        if destination.exists():
            raise FileExistsError(f"Log export already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_path = self._paths.operational_log
        with destination.open("xb") as output:
            if source_path.exists():
                with source_path.open("rb") as source:
                    shutil.copyfileobj(source, output)
