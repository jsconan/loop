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
    store.save(Session(name="Second topic", name_source="user"))
    sessions = SessionManager(session_store=store)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(sessions, Mock()))

    manager.call("sessions")
    listed = interaction.table.call_args.args[0]
    manager.call("resume", first_id)
    manager.call("rename", '"Renamed topic"')
    manager.call("new")

    assert interaction.table.call_args.kwargs["columns"] == (
        "name",
        "updated_at",
        "message_count",
    )
    assert [session.name for session in listed] == ["First topic", "Second topic"]
    interaction.user.assert_called_once_with("Prior question")
    assert store.load(first_id).name == "Renamed topic"
    # Model is preserved across new session, but other state is reset
    assert sessions.session.model == "served-model"
    assert sessions.session.messages == []
    assert sessions.session.tokens == 0
    assert manager.commands[-1].completion.provider == "sessions"


def test_sessions_namespace_matches_shortcuts_and_renames_stored_sessions():
    """The session namespace exposes non-blocking lifecycle operations beside shortcuts."""
    interaction = Mock(spec=Interaction)
    store = MemorySessionStore()
    active = Session(name="Active", name_source="user", model="served-model")
    stored = Session(name="Stored", name_source="user", model="resumed-model")
    active_id = store.save(active)
    stored_id = store.save(stored)
    sessions = SessionManager(session=active, session_store=store)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(sessions, Mock()))

    manager.call("sessions")
    manager.call("sessions", f'rename {stored_id} "Renamed stored"')
    manager.call("sessions", f"resume {stored_id}")
    manager.call("sessions", "new")

    assert interaction.table.call_args.kwargs["columns"] == (
        "name",
        "updated_at",
        "message_count",
    )
    assert store.load(stored_id).name == "Renamed stored"
    assert active_id != sessions.session.id
    assert sessions.session.model == "resumed-model"
    commands = {command.name: command for command in manager.commands}
    assert tuple(value.value for value in commands["sessions"].completion.values) == (
        "new",
        "resume",
        "rename",
        "show",
    )
    assert commands["sessions"].completion.children["resume"].provider == "sessions"
    assert commands["sessions"].completion.children["rename"].provider == "sessions"


def test_session_command_and_sessions_show_display_current_operational_state():
    """Current-session commands display live state without persisting or restoring history."""
    interaction = Mock(spec=Interaction)
    active = Session(
        name="Active topic",
        name_source="user",
        model="served-model",
        tokens=1234,
        context_window=8192,
        workspace_id="workspace",
        instruction_working_directory="/project",
        active_skills=[("review", "/skills/review")],
        revision=2,
    )
    active.add_message(Message(role="user", content="Continue the work."))
    sessions = SessionManager(session=active)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(sessions, Mock()))

    manager.call("session")
    manager.call("sessions", "show")
    sessions.new_session()
    manager.call("session")

    shown = interaction.table.call_args_list
    assert shown[0].kwargs == {
        "title": "Current session:",
        "columns": ("field", "value"),
    }
    assert {row["field"]: row["value"] for row in shown[0].args[0]} == {
        "ID": active.id,
        "Name": "Active topic",
        "Model": "served-model",
        "Context": "1,234 / 8,192 tokens (15.1%)",
        "Conversation items": 1,
        "Compactions": 0,
        "Recovery": "Recovery required: query_model",
        "Active skills": "review",
        "Workspace": "workspace",
        "Instruction directory": "/project",
        "Persistence": "Persisted",
    }
    assert shown[1] == shown[0]
    assert {row["field"]: row["value"] for row in shown[2].args[0]} == {
        "ID": sessions.session.id,
        "Name": "Untitled",
        "Model": "served-model",
        "Context": "0 tokens (capacity unknown)",
        "Conversation items": 0,
        "Compactions": 0,
        "Recovery": "Ready",
        "Active skills": "None",
        "Workspace": "Unassigned",
        "Instruction directory": "Not recorded",
        "Persistence": "Unpersisted",
    }


def test_sessions_namespace_validates_action_arguments():
    """Namespaced session operations reject incomplete and incompatible arguments."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager(), Mock()))

    manager.call("sessions", "new unexpected")
    assert "new operation does not accept arguments" in interaction.report.call_args.args[0].detail
    manager.call("sessions", "resume")
    assert "resume operation requires a session ID" in interaction.report.call_args.args[0].detail
    manager.call("sessions", "resume session-id unexpected")
    assert "resume operation does not accept a name" in interaction.report.call_args.args[0].detail
    manager.call("sessions", "rename session-id")
    assert "rename operation requires a new name" in interaction.report.call_args.args[0].detail
    manager.call("sessions", 'rename missing-session "Renamed"')
    assert "Session 'missing-session' was not found" in interaction.report.call_args.args[0].detail
    manager.call("sessions", "show unexpected")
    assert "show operation does not accept arguments" in interaction.report.call_args.args[0].detail
    manager.call("session", "name=unexpected")
    assert (
        "session status view does not accept arguments"
        in interaction.report.call_args.args[0].detail
    )


def test_session_rename_alias_matches_explicit_and_generated_rename_commands():
    """The singular session namespace preserves explicit and generated renaming behavior."""
    interaction = Mock(spec=Interaction)
    generator = Mock()
    generator.generate.return_value = "Generated topic"
    sessions = SessionManager(interaction=interaction)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(sessions, generator))

    manager.call("session", 'rename "Renamed topic"')
    sessions.add_user_message("Describe this repository")
    sessions.add_message(Message(role="assistant", content="It is a conversation loop."))
    manager.call("session", "rename")

    assert generator.generate.call_args.args == (
        "Describe this repository",
        "It is a conversation loop.",
        None,
    )
    assert sessions.session.name == "Generated topic"
    assert interaction.info.call_args_list[0] == (("Renamed session to 'Renamed topic'.",), {})


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
