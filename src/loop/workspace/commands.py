"""Expose centralized workspace registry operations."""

from __future__ import annotations

import shlex
import threading
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ..commands import CommandArgumentError, CommandContext, CommandRegistration
from ..completion import (
    CommandCompletion,
    CompletionProviderRegistration,
    CompletionValue,
    SchemaCompletionProviderRegistration,
)
from .models import Workspace, WorkspaceSwitchRequested
from .repository import WorkspaceRepository


class WorkspaceAttachArguments(BaseModel):
    """Describe arguments accepted when attaching a workspace location.

    Args:
        name (str): Existing workspace directory to attach.
        workspace_id (str | None): Existing workspace identity to associate, or ``None`` to create
            an identity for the location.
    """

    name: Annotated[str, Field(description="Existing workspace directory to attach.")]
    workspace_id: Annotated[
        str | None,
        Field(description="Existing workspace identity to associate."),
        CommandCompletion(provider="workspace_ids"),
    ] = None


class WorkspaceReferenceArguments(BaseModel):
    """Describe the workspace path or identity accepted by registry operations.

    Args:
        name (str): Active workspace path or identity.
    """

    name: Annotated[
        str,
        Field(description="Active workspace path or identity."),
        CommandCompletion(provider="workspace_references"),
    ]


class WorkspaceRenameArguments(BaseModel):
    """Describe the new workspace name accepted by rename.

    Args:
        name (str): New human-readable workspace name.
    """

    name: Annotated[str, Field(description="New human-readable workspace name.")]


class WorkspaceCommands:
    """List, show, and rename centrally registered workspaces.

    Args:
        workspace (Workspace): Active initialized workspace.
        repository (WorkspaceRepository): Owner of durable workspace registry operations.
    """

    _workspace: Workspace
    _repository: WorkspaceRepository
    _switch_lock: threading.Lock

    def __init__(self, workspace: Workspace, repository: WorkspaceRepository) -> None:
        if workspace.id is None:
            raise ValueError("Workspace commands require an initialized workspace.")
        self._workspace = workspace
        self._repository = repository
        self._switch_lock = threading.Lock()

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return the workspace command registration.

        Returns:
            tuple[CommandRegistration, ...]: The ``/workspace`` command.
        """
        return (
            CommandRegistration(
                self.workspace,
                name="workspace",
                completion=CommandCompletion(
                    values=tuple(
                        CompletionValue(action, description)
                        for action, description in (
                            ("attach", "Attach a workspace location."),
                            ("forget", "Forget a workspace location."),
                            ("list", "List registered workspaces."),
                            ("rekey", "Assign a copied workspace a new identity."),
                            ("show", "Show the active workspace."),
                            ("switch", "Switch to another workspace."),
                            ("rename", "Rename the active workspace."),
                        )
                    ),
                    children={
                        action: CommandCompletion(schema_provider="workspace_arguments")
                        for action in (
                            "attach",
                            "forget",
                            "list",
                            "rekey",
                            "rename",
                            "show",
                            "switch",
                        )
                    },
                ),
            ),
        )

    def get_completion_providers(
        self,
    ) -> tuple[CompletionProviderRegistration | SchemaCompletionProviderRegistration, ...]:
        """Return workspace value and action-specific argument completion sources.

        Returns:
            tuple[CompletionProviderRegistration | SchemaCompletionProviderRegistration, ...]:
                Current workspace IDs, locations and argument schemas.
        """
        return (
            CompletionProviderRegistration("workspace_ids", self._workspace_ids),
            CompletionProviderRegistration("workspace_references", self._workspace_references),
            SchemaCompletionProviderRegistration("workspace_arguments", self._workspace_arguments),
        )

    def _workspace_ids(self) -> tuple[CompletionValue, ...]:
        """Return active workspace identities for attach completion."""
        return tuple(
            CompletionValue(workspace.id, f"{workspace.name}: {workspace.root}")
            for workspace in self._repository.list()
            if workspace.id is not None
        )

    def _workspace_references(self) -> tuple[CompletionValue, ...]:
        """Return active workspace identities and paths for registry operations."""
        values = []
        for workspace in self._repository.list():
            if workspace.id is not None:
                values.append(CompletionValue(workspace.id, f"{workspace.name}: {workspace.root}"))
            path = str(workspace.root)
            values.append(
                CompletionValue(
                    shlex.quote(path),
                    f"Workspace path: {workspace.name}",
                    display=path,
                )
            )
        return tuple(values)

    @staticmethod
    def _workspace_arguments(tokens: tuple[str, ...]) -> type[BaseModel] | None:
        """Return only the argument schema for the selected workspace operation."""
        if tokens == ("attach",):
            return WorkspaceAttachArguments
        if tokens == ("rename",):
            return WorkspaceRenameArguments
        if tokens in {("forget",), ("rekey",), ("switch",)}:
            return WorkspaceReferenceArguments
        return None

    # pylint: disable-next=too-many-branches
    def workspace(
        self,
        context: CommandContext,
        action: Annotated[
            Literal["attach", "forget", "list", "rekey", "show", "rename", "switch"],
            Field(description="Workspace registry operation."),
        ] = "show",
        name: Annotated[
            str | None, Field(description="Name, path, or workspace identifier for the operation.")
        ] = None,
        workspace_id: Annotated[
            str | None, Field(description="Existing identity to associate during attach.")
        ] = None,
    ) -> None:
        """List registered workspaces, show the active workspace, or rename it.

        Args:
            context (CommandContext): Interaction used to display results and confirm changes.
            action (Literal["attach", "forget", "list", "rekey", "show", "rename", "switch"]):
                Workspace operation. Defaults to ``"show"``.
            name (str | None): Name, path, or workspace identifier for the operation.
            workspace_id (str | None): Existing identity to associate during attach.

        Raises:
            CommandArgumentError: If arguments do not match the selected operation.
            WorkspaceSwitchRequested: If the user requests a different workspace.
        """
        if action in {"list", "show"} and (name is not None or workspace_id is not None):
            raise CommandArgumentError(f"The {action} operation does not accept a name.")
        if action != "attach" and workspace_id is not None:
            raise CommandArgumentError("Only attach accepts an existing workspace identity.")
        if action == "rename":
            if name is None or not name.strip():
                raise CommandArgumentError("The rename operation requires a non-empty name.")
            self._repository.rename(self._workspace.id, name.strip())
            context.interaction.info(f"Renamed workspace to {name.strip()}.")
            return
        if action in {"attach", "forget", "rekey", "switch"}:
            if name is None or not name.strip():
                raise CommandArgumentError(f"The {action} operation requires a path or identity.")
            value = name.strip()
            if action == "switch":
                if not self._switch_lock.acquire(  # pylint: disable=consider-using-with
                    blocking=False
                ):
                    raise CommandArgumentError("A workspace switch is already in progress.")
                try:
                    target = self._repository.resolve(value)
                except BaseException:
                    self._switch_lock.release()
                    raise
                raise WorkspaceSwitchRequested(target, self._switch_lock.release)
            if action == "attach":
                if workspace_id is not None and not context.interaction.confirm(
                    f"Associate {value} with workspace {workspace_id}?", default=False
                ):
                    context.interaction.info("Workspace attach cancelled.")
                    return
                attached = self._repository.attach(value, workspace_id)
                context.interaction.info(f"Attached workspace {attached.id} at {attached.root}.")
                return
            if not context.interaction.confirm(
                f"Confirm workspace {action} for {value}?", default=False
            ):
                context.interaction.info(f"Workspace {action} cancelled.")
                return
            if action == "forget":
                if not self._repository.forget(value):
                    raise CommandArgumentError(f"No active workspace location matches '{value}'.")
                context.interaction.info("Forgot workspace location; workspace data was retained.")
                return
            rekeyed = self._repository.rekey(value)
            context.interaction.info(f"Rekeyed workspace as {rekeyed.id}.")
            return
        workspaces = self._repository.list(self._workspace.id if action == "show" else None)
        rows = tuple(
            {
                "workspace_id": workspace.id,
                "name": workspace.name,
                "name_source": workspace.name_source,
                "canonical_path": str(workspace.root),
            }
            for workspace in workspaces
        )
        context.interaction.table(
            rows,
            title="Active workspace:" if action == "show" else "Registered workspaces:",
            columns=("workspace_id", "name", "name_source", "canonical_path"),
        )
