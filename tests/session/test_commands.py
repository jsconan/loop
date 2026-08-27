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
    manager.register_provider(SessionCommands(sessions, Mock()))

    manager.call("sessions")
    manager.call("resume", first_id)
    manager.call("rename", '"Renamed topic"')
    manager.call("new")

    assert interaction.table.call_args.kwargs["columns"] == (
        "name",
        "updated_at",
        "message_count",
    )
    interaction.user.assert_called_once_with("Prior question")
    assert store.load(first_id).name == "Renamed topic"
    # Model is preserved across new session, but other state is reset
    assert sessions.session.model == "served-model"
    assert sessions.session.messages == []
    assert sessions.session.tokens == 0
    assert manager.commands[-1].completion.provider == "sessions"


def test_resume_reports_unknown_session_ids():
    """Resume requires an ID and translates unknown persisted IDs into argument warnings."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager(), Mock()))

    manager.call("resume")
    assert "Field required" in interaction.report.call_args.args[0].detail
    manager.call("resume", "missing-id")

    assert "Session 'missing-id' was not found" in interaction.report.call_args.args[0].detail


def test_rename_reports_invalid_session_names():
    """Rename translates invalid domain values into standard command argument feedback."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager(), Mock()))

    manager.call("rename", "''")

    assert "Session name cannot be empty" in interaction.report.call_args.args[0].detail


def test_rename_without_a_name_generates_a_session_name():
    """Rename without arguments uses the configured automatic naming service."""
    interaction = Mock(spec=Interaction)
    generator = Mock()
    generator.generate.return_value = "Generated topic"
    sessions = SessionManager(interaction=interaction)
    sessions.add_user_message("Describe this repository")
    sessions.add_message(Message(role="assistant", content="It is a conversation loop."))
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(sessions, generator))

    manager.call("rename")

    generator.generate.assert_called_once_with(
        "Describe this repository", "It is a conversation loop.", None
    )
    assert sessions.session.name == "Generated topic"
    assert sessions.session.name_source == "generated"
    assert interaction.info.call_args_list == [
        (("Generating a session name...",), {}),
        (("Session name: Generated topic",), {}),
    ]


def test_rename_without_a_name_reports_a_problem_when_generation_fails():
    """Rename reports automatic naming failures without rejecting the command arguments."""
    interaction = Mock(spec=Interaction)
    generator = Mock()
    generator.generate.side_effect = RuntimeError("Generator is unavailable")
    sessions = SessionManager(interaction=interaction)
    sessions.add_user_message("Describe this repository")
    sessions.add_message(Message(role="assistant", content="It is a conversation loop."))
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(sessions, generator))

    manager.call("rename")

    problem = interaction.report.call_args.args[0]
    assert problem.code == "session.name_generation_failed"
    assert problem.title == "Could not generate session name"
    assert problem.detail == "Could not generate the session name."
    assert problem.severity == "warning"
    assert problem.retryable is True
    assert problem.operation == "generate_session_name"
    interaction.warning.assert_not_called()
