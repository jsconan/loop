"""Expose session lifecycle operations as user commands."""

from typing import Annotated

from pydantic import Field

from ..commands import CommandArgumentError, CommandContext, CommandRegistration
from ..completion import CommandCompletion
from .session_manager import SessionManager


class SessionCommands:
    """Expose one session manager through interactive commands.

    Args:
        session_manager (SessionManager): Session lifecycle owner controlled by the commands.
    """

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return session command registrations.

        Returns:
            tuple[CommandRegistration, ...]: Session lifecycle commands.
        """
        return (
            CommandRegistration(self.new, name="new"),
            CommandRegistration(self.rename, name="rename"),
            CommandRegistration(self.sessions, name="sessions"),
            CommandRegistration(
                self.resume,
                name="resume",
                completion=CommandCompletion(provider="sessions"),
            ),
        )

    def new(self, context: CommandContext) -> None:
        """Start a fresh unpersisted session."""
        self._session_manager.new_session()
        context.interaction.info("Started a new session.")

    def rename(
        self,
        context: CommandContext,
        name: Annotated[str, Field(description="New human-readable session name.")],
    ) -> None:
        """Rename the active session."""
        self._session_manager.rename_session(name)
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
        context.interaction.history(
            self._session_manager.messages,
            session.compactions,
        )
        context.interaction.token_usage(
            self._session_manager.model,
            self._session_manager.tokens,
            self._session_manager.context_window,
        )
        context.interaction.info(f"Resumed session '{session.name}'.")

    def sessions(self, context: CommandContext) -> None:
        """List persisted sessions."""
        context.interaction.table(
            self._session_manager.store.list(),
            title="Persisted sessions:",
            columns=("name", "updated_at", "message_count"),
        )
