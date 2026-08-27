"""Expose session lifecycle operations as user commands."""

from typing import Annotated

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
            CommandRegistration(self.sessions, name="sessions"),
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

    def sessions(self, context: CommandContext) -> None:
        """List persisted sessions."""
        context.interaction.table(
            self._session_manager.store.list(),
            title="Persisted sessions:",
            columns=("name", "updated_at", "message_count"),
        )
