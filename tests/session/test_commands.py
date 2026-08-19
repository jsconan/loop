"""Tests for session-owned user commands."""

from unittest.mock import Mock

from loop import (
    CommandManager,
    Interaction,
    MemorySessionStore,
    Message,
    Session,
    SessionManager,
)
from loop.session import SessionCommands


def test_session_commands_list_resume_rename_and_reset_sessions():
    """Session commands preserve lifecycle behavior and restoration output order."""
    interaction = Mock(spec=Interaction)
    store = MemorySessionStore()
    first = Session(name="First topic", name_source="user", model="served-model", tokens=1234)
    first.add_message(Message(role="user", content="Prior question"))
    first_id = store.save(first)
    sessions = SessionManager(session_store=store)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(sessions))

    manager.call("sessions")
    manager.call("resume", first_id)
    manager.call("rename", '"Renamed topic"')
    manager.call("new")

    assert interaction.table.call_args.kwargs["columns"] == (
        "name",
        "updated_at",
        "message_count",
    )
    interaction.history.assert_called_once_with(first.messages, first.compactions)
    interaction.token_usage.assert_called_once_with("served-model", 1234, None)
    assert store.load(first_id).name == "Renamed topic"
    assert sessions.session == Session()
    assert manager.commands[-1].completion.provider == "sessions"


def test_resume_reports_unknown_session_ids():
    """Resume translates missing persisted IDs into command argument warnings."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager()))

    manager.call("resume", "missing-id")

    assert "Session 'missing-id' was not found" in interaction.warning.call_args.args[0]
