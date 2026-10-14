"""Expose session lifecycle operations as user commands."""

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ..commands import CommandArgumentError, CommandContext, CommandRegistration
from ..completion import (
    CommandCompletion,
    CompletionProviderRegistration,
    CompletionValue,
    SchemaCompletionProviderRegistration,
)
from .exporter import SessionExportFormat, export_session
from .models import SessionNameGenerator
from .session import Session
from .session_manager import SessionManager


class SessionExportArguments(BaseModel):
    """Describe the arguments offered for exporting the current session.

    Args:
        destination (str | None): Destination path, or ``None`` to generate a name in the current
            directory.
        format (SessionExportFormat): Markdown transcript or complete ZIP archive.
    """

    destination: Annotated[
        str | None,
        Field(description="Destination path, or omit to generate a name in the current directory."),
    ] = None
    format: Annotated[
        SessionExportFormat,
        Field(description="Export format: Markdown transcript or complete ZIP archive."),
    ] = "markdown"


class PersistedSessionExportArguments(BaseModel):
    """Describe the arguments offered for exporting a persisted session.

    Args:
        session_id (str): Exact persisted session ID to export.
        destination (str | None): Destination path, or ``None`` to generate a name in the current
            directory.
        format (SessionExportFormat): Markdown transcript or complete ZIP archive.
    """

    session_id: Annotated[
        str,
        Field(description="Exact persisted session ID to export."),
        CommandCompletion(provider="sessions"),
    ]
    destination: Annotated[
        str | None,
        Field(description="Destination path, or omit to generate a name in the current directory."),
    ] = None
    format: Annotated[
        SessionExportFormat,
        Field(description="Export format: Markdown transcript or complete ZIP archive."),
    ] = "markdown"


class SessionCommands:
    """Expose one session manager through interactive commands.

    Args:
        session_manager (SessionManager): Session lifecycle owner controlled by the commands.
        session_name_generator (SessionNameGenerator): Service used to automatically name the
            active session.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        session_name_generator: SessionNameGenerator,
    ) -> None:
        self._session_manager = session_manager
        self._session_name_generator = session_name_generator

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return session command registrations.

        Returns:
            tuple[CommandRegistration, ...]: Session lifecycle commands.
        """
        return (
            CommandRegistration(self.new, name="new"),
            CommandRegistration(self.rename, name="rename"),
            CommandRegistration(self.export, name="export"),
            CommandRegistration(
                self.session,
                name="session",
                completion=CommandCompletion(
                    values=(
                        CompletionValue("rename", "Rename the current session."),
                        CompletionValue("export", "Export the current session."),
                    ),
                    children={
                        "rename": CommandCompletion(),
                        "export": CommandCompletion(schema_provider="session_export_arguments"),
                    },
                ),
            ),
            CommandRegistration(
                self.sessions,
                name="sessions",
                completion=CommandCompletion(
                    values=(
                        CompletionValue("new", "Start a fresh session."),
                        CompletionValue("resume", "Restore a persisted session."),
                        CompletionValue("rename", "Rename a persisted session."),
                        CompletionValue("show", "Show the current session state."),
                        CompletionValue("export", "Export a persisted session."),
                    ),
                    children={
                        "new": CommandCompletion(),
                        "resume": CommandCompletion(provider="sessions"),
                        "rename": CommandCompletion(
                            provider="sessions",
                            next=CommandCompletion(),
                        ),
                        "show": CommandCompletion(),
                        "export": CommandCompletion(
                            schema_provider="persisted_session_export_arguments",
                        ),
                    },
                ),
            ),
            CommandRegistration(
                self.resume,
                name="resume",
                completion=CommandCompletion(provider="sessions"),
            ),
        )

    def get_completion_providers(
        self,
    ) -> tuple[CompletionProviderRegistration | SchemaCompletionProviderRegistration, ...]:
        """Return dynamic session value and argument-schema completion sources.

        Returns:
            tuple[CompletionProviderRegistration | SchemaCompletionProviderRegistration, ...]:
                Named session completion sources.
        """
        return (
            CompletionProviderRegistration("sessions", self._session_values),
            SchemaCompletionProviderRegistration(
                "session_export_arguments",
                self._session_export_arguments,
            ),
            SchemaCompletionProviderRegistration(
                "persisted_session_export_arguments",
                self._persisted_session_export_arguments,
            ),
        )

    @staticmethod
    def _session_export_arguments(tokens: tuple[str, ...]) -> type[BaseModel] | None:
        """Return the current-session export schema after its action is selected."""
        return SessionExportArguments if tokens == ("export",) else None

    @staticmethod
    def _persisted_session_export_arguments(tokens: tuple[str, ...]) -> type[BaseModel] | None:
        """Return the persisted-session export schema after its action is selected."""
        return PersistedSessionExportArguments if tokens == ("export",) else None

    def _session_values(self) -> tuple[CompletionValue, ...]:
        """Return persisted sessions in store order."""
        return tuple(
            CompletionValue(
                session.id,
                str(session.updated_at),
                display=session.name,
                sort_order=index,
            )
            for index, session in enumerate(self._session_manager.store.list())
        )

    def new(self, context: CommandContext) -> None:
        """Start a fresh unpersisted session."""
        self._session_manager.new_session()
        context.interaction.info("Started a new session.")

    def rename(
        self,
        context: CommandContext,
        name: Annotated[
            str | None,
            Field(description="New human-readable session name, or omit to generate one."),
        ] = None,
    ) -> None:
        """Rename the active session or generate a name automatically."""
        if name is None:
            self._session_manager.generate_session_name(self._session_name_generator)
            return
        try:
            self._session_manager.rename_session(name)
        except ValueError as error:
            raise CommandArgumentError(str(error)) from error
        context.interaction.info(f"Renamed session to '{self._session_manager.session.name}'.")

    def resume(
        self,
        context: CommandContext,
        session_id: Annotated[str, Field(description="Exact persisted session ID.")],
    ) -> None:
        """Resume a persisted session."""
        try:
            self._session_manager.load_session(session_id)
        except ValueError as error:
            raise CommandArgumentError(str(error)) from error
        session = self._session_manager.session
        context.interaction.info(f"Restoring session history for '{session.name}'...")
        self._session_manager.replay(interaction=context.interaction)
        context.interaction.info(f"Resumed session '{session.name}'.")

    def export(
        self,
        context: CommandContext,
        session_id: Annotated[
            str | None,
            Field(description="Exact persisted session ID, or omit for the active session."),
            CommandCompletion(provider="sessions"),
        ] = None,
        destination: Annotated[
            str | None,
            Field(
                description="Destination path, or omit to generate a name in the current directory."
            ),
        ] = None,
        format: Annotated[
            SessionExportFormat,
            Field(description="Export format: Markdown transcript or complete ZIP archive."),
        ] = "markdown",
    ) -> None:
        """Export the active or a selected persisted session to a file.

        Args:
            context (CommandContext): Interaction used for replacement confirmation and status.
            session_id (str | None): Exact persisted session ID, or ``None`` to export the active
                session.
            destination (str | None): File path, or ``None`` to generate one in the current
                directory from the session identifier and export format.
            format (SessionExportFormat): Export format. Defaults to ``"markdown"``.

        Raises:
            CommandArgumentError: If the session ID is unknown, the destination or export format
                is invalid, or writing the export fails.
        """
        session = self._session_manager.session
        if session_id is not None:
            try:
                session = self._session_manager.load_persisted_session(session_id)
            except ValueError as error:
                raise CommandArgumentError(str(error)) from error
        self._export_session(
            context,
            session,
            destination,
            format,
        )

    def _export_session(
        self,
        context: CommandContext,
        session: Session,
        destination: str | None,
        format: SessionExportFormat,
    ) -> None:
        """Write one selected session and ask before replacing any existing destination."""
        try:
            target = (
                _default_export_path(session, format)
                if destination is None
                else Path(destination).expanduser().resolve()
            )
            if target.is_dir():
                raise CommandArgumentError("Export destination must be a file.")
            store = self._session_manager.store
            store_path = getattr(store, "path", None)
            if store_path is not None and target == Path(store_path).expanduser().resolve():
                raise CommandArgumentError("Export destination must not be the session database.")

            overwrite = False
            while True:
                if target.exists() and not overwrite:
                    confirmed = context.interaction.confirm(
                        f"Replace the existing file at '{target}'?",
                        default=False,
                    )
                    if not confirmed:
                        context.interaction.info("Session export stopped.")
                        return
                    overwrite = True
                try:
                    export_session(
                        session,
                        store,
                        target,
                        format=format,
                        overwrite=overwrite,
                    )
                except FileExistsError:
                    if overwrite:
                        raise
                    continue
                break
        except CommandArgumentError:
            raise
        except (OSError, ValueError) as error:
            raise CommandArgumentError(str(error)) from error
        context.interaction.info(f"Exported session '{session.name or session.id}' to {target}.")

    def session(
        self,
        context: CommandContext,
        action: Annotated[
            Literal["rename", "export"] | None,
            Field(description="Current-session operation, or omit to show its state."),
        ] = None,
        destination: Annotated[
            str | None,
            Field(description="Export destination path, or omit to generate a name."),
        ] = None,
        name: Annotated[
            str | None,
            Field(description="New human-readable name for rename."),
        ] = None,
        format: Annotated[
            SessionExportFormat,
            Field(description="Export format: Markdown transcript or complete ZIP archive."),
        ] = "markdown",
    ) -> None:
        """Show, rename, or export the current session.

        Args:
            context (CommandContext): Interaction used to display session state or export status.
            action (Literal["rename", "export"] | None): Operation, or ``None`` to show status.
            destination (str | None): Export path, or positional rename value for compatibility.
            name (str | None): New human-readable name for rename.
            format (SessionExportFormat): Export format. Defaults to ``"markdown"``.

        Raises:
            CommandArgumentError: If arguments do not match the selected operation.
        """
        if action == "rename":
            if format != "markdown":
                raise CommandArgumentError("The rename operation does not accept an export format.")
            if destination is not None and name is not None:
                raise CommandArgumentError("The rename operation accepts only one session name.")
            self.rename(context, name if name is not None else destination)
            return
        if action == "export":
            if name is not None:
                raise CommandArgumentError(
                    "The export operation accepts a destination, not a name."
                )
            self.export(context, destination=destination, format=format)
            return
        if destination is not None or name is not None or format != "markdown":
            raise CommandArgumentError("The session status view does not accept arguments.")
        self._show_session(context)

    def sessions(
        self,
        context: CommandContext,
        action: Annotated[
            Literal["new", "resume", "rename", "show", "export"] | None,
            Field(description="Session operation, or omit to list persisted sessions."),
        ] = None,
        session_id: Annotated[
            str | None,
            Field(description="Exact persisted session ID for resume, rename, or export."),
        ] = None,
        destination: Annotated[
            str | None,
            Field(description="Export destination path, or omit to generate a name."),
        ] = None,
        name: Annotated[
            str | None,
            Field(description="New human-readable name for a persisted session."),
        ] = None,
        format: Annotated[
            SessionExportFormat,
            Field(description="Export format: Markdown transcript or complete ZIP archive."),
        ] = "markdown",
    ) -> None:
        """List, create, resume, rename, or export persisted sessions.

        Args:
            context (CommandContext): Interaction used to display, restore, or export sessions.
            action (Literal["new", "resume", "rename", "show", "export"] | None):
                Persisted-session operation, or ``None`` to list sessions.
            session_id (str | None): Exact persisted session ID required by resume, rename, export.
            destination (str | None): Export path, or positional rename value for compatibility.
            name (str | None): New human-readable name for a persisted session.
            format (SessionExportFormat): Export format. Defaults to ``"markdown"``.

        Raises:
            CommandArgumentError: If required session data is missing or arguments do not match
                the selected operation.
        """
        if action is None:
            if (
                session_id is not None
                or destination is not None
                or name is not None
                or format != "markdown"
            ):
                raise CommandArgumentError("The sessions list does not accept arguments.")
            self._list_sessions(context)
            return
        if action == "show":
            if (
                session_id is not None
                or destination is not None
                or name is not None
                or format != "markdown"
            ):
                raise CommandArgumentError("The show operation does not accept arguments.")
            self._show_session(context)
            return
        if action == "new":
            if (
                session_id is not None
                or destination is not None
                or name is not None
                or format != "markdown"
            ):
                raise CommandArgumentError("The new operation does not accept arguments.")
            self.new(context)
            return
        if action == "export":
            if session_id is None:
                raise CommandArgumentError("The export operation requires a session ID.")
            if name is not None:
                raise CommandArgumentError(
                    "The export operation accepts a destination, not a name."
                )
            try:
                session = self._session_manager.load_persisted_session(session_id)
            except ValueError as error:
                raise CommandArgumentError(str(error)) from error
            self._export_session(context, session, destination, format)
            return
        if session_id is None:
            raise CommandArgumentError(f"The {action} operation requires a session ID.")
        if action == "resume":
            if destination is not None or name is not None or format != "markdown":
                raise CommandArgumentError("The resume operation does not accept a destination.")
            self.resume(context, session_id)
            return
        if format != "markdown":
            raise CommandArgumentError("The rename operation does not accept an export format.")
        if destination is not None and name is not None:
            raise CommandArgumentError("The rename operation accepts only one session name.")
        rename = name if name is not None else destination
        if rename is None:
            raise CommandArgumentError("The rename operation requires a new name.")
        try:
            self._session_manager.rename_persisted_session(session_id, rename)
        except ValueError as error:
            raise CommandArgumentError(str(error)) from error
        context.interaction.info(f"Renamed session '{session_id}'.")

    def _list_sessions(self, context: CommandContext) -> None:
        """Display persisted sessions from oldest to newest."""
        context.interaction.table(
            reversed(self._session_manager.store.list()),
            title="Persisted sessions:",
            columns=("name", "updated_at", "message_count"),
        )

    def _show_session(self, context: CommandContext) -> None:
        """Display the active session's concise operational state."""
        session = self._session_manager.session
        recovery_state = self._session_manager.recovery_state
        context_usage = self._format_context(session.tokens, session.context_window)
        recovery = (
            "Ready" if recovery_state is None else f"Recovery required: {recovery_state.action}"
        )
        context.interaction.table(
            (
                {"field": "ID", "value": session.id},
                {"field": "Name", "value": session.name or "Untitled"},
                {"field": "Model", "value": session.model or "Backend default"},
                {"field": "Context", "value": context_usage},
                {"field": "Conversation items", "value": len(session.messages)},
                {"field": "Compactions", "value": len(session.compactions)},
                {"field": "Recovery", "value": recovery},
                {
                    "field": "Active skills",
                    "value": ", ".join(name for name, _ in session.active_skills) or "None",
                },
                {"field": "Workspace", "value": session.workspace_id or "Unassigned"},
                {
                    "field": "Instruction directory",
                    "value": session.instruction_working_directory or "Not recorded",
                },
                {
                    "field": "Persistence",
                    "value": "Unpersisted" if session.revision == 0 else "Persisted",
                },
            ),
            title="Current session:",
            columns=("field", "value"),
        )

    @staticmethod
    def _format_context(tokens: int, context_window: int | None) -> str:
        """Format current context occupancy with capacity when it is known."""
        if context_window is None:
            return f"{tokens:,} tokens (capacity unknown)"
        return f"{tokens:,} / {context_window:,} tokens ({tokens / context_window:.1%})"


def _default_export_path(session: Session, format: SessionExportFormat) -> Path:
    """Generate a safe, stable export filename in the current directory."""
    session_id = session.id
    if (
        len(session_id) > 100
        or not session_id.isascii()
        or any(not (character.isalnum() or character in "-_") for character in session_id)
    ):
        session_id = hashlib.sha256(session_id.encode("utf-8", errors="surrogatepass")).hexdigest()
    extension = "zip" if format == "archive" else "md"
    return Path.cwd() / f"loop-session-{session_id}.{extension}"
