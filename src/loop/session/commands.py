"""Expose session lifecycle operations as user commands."""

from typing import Annotated, Literal

from pydantic import Field

from ..commands import CommandArgumentError, CommandContext, CommandRegistration
from ..completion import CommandCompletion, CompletionProviderRegistration, CompletionValue
from .models import SessionNameGenerator
from .session_manager import SessionManager


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
            CommandRegistration(
                self.session,
                name="session",
                completion=CommandCompletion(
                    values=(CompletionValue("rename", "Rename the current session."),),
                    children={"rename": CommandCompletion()},
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
                    ),
                    children={
                        "new": CommandCompletion(),
                        "resume": CommandCompletion(provider="sessions"),
                        "rename": CommandCompletion(
                            provider="sessions",
                            next=CommandCompletion(),
                        ),
                        "show": CommandCompletion(),
                    },
                ),
            ),
            CommandRegistration(
                self.resume,
                name="resume",
                completion=CommandCompletion(provider="sessions"),
            ),
        )

    def get_completion_providers(self) -> tuple[CompletionProviderRegistration, ...]:
        """Return dynamic session completion sources.

        Returns:
            tuple[CompletionProviderRegistration, ...]: Named session completion source.
        """
        return (CompletionProviderRegistration("sessions", self._session_values),)

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

    def session(
        self,
        context: CommandContext,
        action: Annotated[
            Literal["rename"] | None,
            Field(description="Current-session operation, or omit to show its state."),
        ] = None,
        name: Annotated[
            str | None,
            Field(description="New human-readable session name, or omit to generate one."),
        ] = None,
    ) -> None:
        """Show the current session state or rename it."""
        if action == "rename":
            self.rename(context, name)
            return
        if name is not None:
            raise CommandArgumentError("The session status view does not accept arguments.")
        self._show_session(context)

    def sessions(
        self,
        context: CommandContext,
        action: Annotated[
            Literal["new", "resume", "rename", "show"] | None,
            Field(description="Session operation, or omit to list persisted sessions."),
        ] = None,
        session_id: Annotated[
            str | None,
            Field(description="Exact persisted session ID for resume or rename."),
        ] = None,
        name: Annotated[
            str | None,
            Field(description="New human-readable name for a persisted session."),
        ] = None,
    ) -> None:
        """List, create, resume, or rename persisted sessions."""
        if action is None:
            self._list_sessions(context)
            return
        if action == "show":
            if session_id is not None or name is not None:
                raise CommandArgumentError("The show operation does not accept arguments.")
            self._show_session(context)
            return
        if action == "new":
            if session_id is not None or name is not None:
                raise CommandArgumentError("The new operation does not accept arguments.")
            self.new(context)
            return
        if session_id is None:
            raise CommandArgumentError(f"The {action} operation requires a session ID.")
        if action == "resume":
            if name is not None:
                raise CommandArgumentError("The resume operation does not accept a name.")
            self.resume(context, session_id)
            return
        if name is None:
            raise CommandArgumentError("The rename operation requires a new name.")
        try:
            self._session_manager.rename_persisted_session(session_id, name)
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
