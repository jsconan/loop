"""Tests for session-owned user commands."""

import asyncio
import json
from threading import Event, get_ident
from unittest.mock import Mock
from zipfile import ZipFile

import pytest
from prompt_toolkit.document import Document

from loop import (
    CommandArgumentError,
    CommandCapability,
    CommandCompletionAdapter,
    CommandContext,
    CommandManager,
    CompletionManager,
    MemorySessionStore,
    Message,
    Session,
    SessionManager,
)
from loop.interaction import ConsoleInteraction
from loop.session import SessionCommands
from loop.session import commands as session_commands_module
from loop.session.store import SQLiteSessionStore


def test_session_commands_list_resume_rename_and_reset_sessions():
    """Session commands preserve lifecycle behavior and restoration output order."""
    interaction = Mock(spec=ConsoleInteraction)
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
    """The session namespace exposes lifecycle commands and the current-session export shortcut."""
    interaction = Mock(spec=ConsoleInteraction)
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
    assert "session" in commands
    assert "sessions" in commands
    assert "export" in commands


def test_active_run_rejects_commands_that_replace_or_rename_its_session():
    """Active runs keep session replacement and active-session rename at a safe boundary."""
    interaction = Mock(spec=ConsoleInteraction)
    store = MemorySessionStore()
    active = Session(id="active-session", name="Active", name_source="user")
    store.save(active)
    other = Session(id="other-session", name="Other", name_source="user")
    other_id = store.save(other)
    sessions = SessionManager(session=active, session_store=store)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(sessions, Mock()))
    manager.set_capability_provider(lambda: frozenset({CommandCapability.RUN_ACTIVE}))

    manager.call("new")
    manager.call("rename", '"Changed name"')
    manager.call("resume", other_id)
    manager.call("sessions", "new")
    manager.call("sessions", f"resume {other_id}")
    manager.call("session", 'rename "Changed name"')
    manager.call("sessions", f'rename {active.id} "Changed name"')

    assert sessions.session is active
    assert sessions.session.name == "Active"
    assert store.load(active.id).name == "Active"
    assert store.load(other_id).name == "Other"
    assert {"session", "sessions", "new", "rename", "resume"}.isdisjoint(
        command.name for command in manager.active_commands
    )


def test_sessions_export_completes_persisted_ids_after_the_export_action():
    """Session completion defers export arguments until an export action is selected."""
    interaction = Mock(spec=ConsoleInteraction)
    store = MemorySessionStore()
    store.save(Session(id="saved-session", name="Saved", name_source="user"))
    provider = SessionCommands(SessionManager(session_store=store), Mock())
    manager = CommandManager(interaction=interaction)
    manager.register_provider(provider)
    completer = CompletionManager(
        (CommandCompletionAdapter(lambda: manager.commands, providers=(provider,)),)
    )

    completions = list(completer.get_completions(Document("/sessions export saved"), Mock()))
    session_operations = list(completer.get_completions(Document("/session "), Mock()))
    sessions_operations = list(completer.get_completions(Document("/sessions "), Mock()))
    session_actions = list(completer.get_completions(Document("/session ex"), Mock()))
    sessions_actions = list(completer.get_completions(Document("/sessions ex"), Mock()))
    session_parameters = list(completer.get_completions(Document("/session export "), Mock()))
    session_positional_formats = list(
        completer.get_completions(Document("/session export output.md "), Mock())
    )
    session_formats = list(completer.get_completions(Document("/session export format="), Mock()))
    export_session_ids = list(completer.get_completions(Document("/export "), Mock()))
    export_parameters = list(completer.get_completions(Document("/export saved-session "), Mock()))
    export_formats = list(
        completer.get_completions(Document("/export saved-session format="), Mock())
    )
    sessions_start = list(completer.get_completions(Document("/sessions export "), Mock()))
    sessions_parameters = list(
        completer.get_completions(Document("/sessions export saved-session "), Mock())
    )
    sessions_positional_formats = list(
        completer.get_completions(Document("/sessions export saved-session output.md "), Mock())
    )
    sessions_formats = list(
        completer.get_completions(Document("/sessions export saved-session format="), Mock())
    )

    assert [completion.text for completion in completions] == ["saved-session"]
    assert {completion.text for completion in session_operations} == {"rename", "export"}
    assert {completion.text for completion in sessions_operations} == {
        "new",
        "resume",
        "rename",
        "show",
        "export",
    }
    assert [completion.text for completion in session_actions] == ["export"]
    assert [completion.text for completion in sessions_actions] == ["export"]
    assert {completion.text for completion in session_parameters} == {"destination=", "format="}
    assert {completion.text for completion in session_formats} == {
        "format=archive",
        "format=markdown",
    }
    assert {completion.text for completion in session_positional_formats} == {
        "archive",
        "markdown",
    }
    assert {completion.text for completion in export_session_ids} == {
        "saved-session",
        "destination=",
        "format=",
    }
    assert {completion.text for completion in export_parameters} == {"destination=", "format="}
    assert {completion.text for completion in export_formats} == {
        "format=archive",
        "format=markdown",
    }
    assert {completion.text for completion in sessions_start} == {
        "saved-session",
        "destination=",
        "format=",
    }
    assert {completion.text for completion in sessions_parameters} == {"destination=", "format="}
    assert {completion.text for completion in sessions_positional_formats} == {
        "archive",
        "markdown",
    }
    assert {completion.text for completion in sessions_formats} == {
        "format=archive",
        "format=markdown",
    }


def test_session_command_and_sessions_show_display_current_operational_state():
    """Current-session commands display live state without persisting or restoring history."""
    interaction = Mock(spec=ConsoleInteraction)
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
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager(), Mock()))

    manager.call("sessions", "new unexpected")
    assert "new operation does not accept arguments" in interaction.report.call_args.args[0].detail
    manager.call("sessions", "resume")
    assert "resume operation requires a session ID" in interaction.report.call_args.args[0].detail
    manager.call("sessions", "resume session-id unexpected")
    assert (
        "resume operation does not accept a destination"
        in interaction.report.call_args.args[0].detail
    )
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
    manager.call("session", "export session_id=saved")
    assert "Unknown parameter 'session_id'" in interaction.report.call_args.args[0].detail
    manager.call("session", "rename destination=first name=second")
    assert "rename operation accepts only one session name" in (
        interaction.report.call_args.args[0].detail
    )
    manager.call("session", "export name=unexpected")
    assert "export operation accepts a destination, not a name" in (
        interaction.report.call_args.args[0].detail
    )
    manager.call("sessions", "export saved-session name=unexpected")
    assert "export operation accepts a destination, not a name" in (
        interaction.report.call_args.args[0].detail
    )
    manager.call("sessions", "rename session-id destination=first name=second")
    assert "rename operation accepts only one session name" in (
        interaction.report.call_args.args[0].detail
    )
    manager.call("sessions", "format=archive")
    assert "sessions list does not accept arguments" in interaction.report.call_args.args[0].detail
    manager.call("sessions", "export")
    assert "export operation requires a session ID" in interaction.report.call_args.args[0].detail
    manager.call("sessions", 'rename session-id "Renamed" format=archive')
    assert "rename operation does not accept an export format" in (
        interaction.report.call_args.args[0].detail
    )
    manager.call("session", 'rename "Renamed" format=archive')
    assert "rename operation does not accept an export format" in (
        interaction.report.call_args.args[0].detail
    )


def test_session_rename_alias_matches_explicit_and_generated_rename_commands():
    """The singular session namespace preserves explicit and generated renaming behavior."""
    interaction = Mock(spec=ConsoleInteraction)
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
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager(), Mock()))

    manager.call("resume")
    assert "Field required" in interaction.report.call_args.args[0].detail
    manager.call("resume", "missing-id")

    assert "Session 'missing-id' was not found" in interaction.report.call_args.args[0].detail


def test_export_command_writes_a_markdown_transcript_for_the_active_session(tmp_path):
    """Export writes the active session as a readable transcript and reports its destination."""
    interaction = Mock(spec=ConsoleInteraction)
    active = Session(
        name="Current session",
        name_source="user",
        messages=[Message(role="user", content="Export this conversation.")],
    )
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager(session=active), Mock()))
    target = tmp_path / "current.md"

    manager.call("session", f"export {target}")

    assert "Export this conversation." in target.read_text(encoding="utf-8")
    interaction.info.assert_called_once_with(
        f"Exported session 'Current session' to {target.resolve()}."
    )
    interaction.prompt.assert_not_called()


def test_export_command_generates_a_markdown_filename_when_destination_is_omitted(
    tmp_path,
    monkeypatch,
):
    """An omitted destination writes a deterministic session-named file in the current folder."""
    interaction = Mock(spec=ConsoleInteraction)
    active = Session(
        id="active_session-id",
        name="Current session",
        name_source="user",
        messages=[Message(role="user", content="Export to the generated filename.")],
    )
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager(session=active), Mock()))
    monkeypatch.chdir(tmp_path)

    manager.call("session", "export")

    target = tmp_path / "loop-session-active_session-id.md"
    assert "Export to the generated filename." in target.read_text(encoding="utf-8")
    interaction.info.assert_called_once_with(f"Exported session 'Current session' to {target}.")


def test_export_alias_writes_the_current_session_transcript(tmp_path):
    """The root export command accepts a named destination for the active session."""
    interaction = Mock(spec=ConsoleInteraction)
    active = Session(
        id="active",
        name="Current session",
        name_source="user",
        messages=[Message(role="user", content="Export this conversation.")],
    )
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager(session=active), Mock()))
    target = tmp_path / "current session.md"

    manager.call("export", f'destination="{target}"')

    assert "Export this conversation." in target.read_text(encoding="utf-8")
    interaction.info.assert_called_once_with(
        f"Exported session 'Current session' to {target.resolve()}."
    )


def test_export_command_selects_a_persisted_session_without_resuming_it(tmp_path):
    """The root export accepts positional and named options for a saved session."""
    interaction = Mock(spec=ConsoleInteraction)
    store = MemorySessionStore()
    active = Session(id="active", name="Active", name_source="user")
    saved = Session(
        id="saved",
        name="Saved",
        name_source="user",
        messages=[Message(role="user", content="Export the selected conversation.")],
    )
    store.save(saved)
    sessions = SessionManager(session=active, session_store=store)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(sessions, Mock()))
    positional_target = tmp_path / "selected-positional.zip"
    named_target = tmp_path / "selected-named.zip"

    manager.call("export", f"saved {positional_target} archive")
    manager.call("export", f'saved destination="{named_target}" format=archive')

    assert sessions.session is active
    for target in (positional_target, named_target):
        with ZipFile(target) as archive:
            serialized = json.loads(archive.read("session.json"))
        assert serialized["messages"][0]["data"]["content"] == ("Export the selected conversation.")
    assert interaction.info.call_count == 2


@pytest.mark.parametrize("session_id", ["../escape", "é", chr(0xD800), "x" * 101])
def test_export_command_keeps_generated_filename_safe_for_unusual_session_ids(
    tmp_path,
    monkeypatch,
    session_id,
):
    """Generated filenames remain local when a session ID is unsafe or unusually long."""
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(
        SessionCommands(SessionManager(session=Session(id=session_id)), Mock())
    )
    monkeypatch.chdir(tmp_path)

    manager.call("session", "export")

    exports = list(tmp_path.glob("loop-session-*.md"))
    assert len(exports) == 1
    assert exports[0].parent == tmp_path
    assert len(exports[0].stem.removeprefix("loop-session-")) == 64


def test_export_command_prompts_before_replacing_generated_destination(tmp_path, monkeypatch):
    """An existing generated filename stays intact when yes-or-no confirmation is declined."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm.return_value = False
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager(session=Session(id="active")), Mock()))
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "loop-session-active.md"
    target.write_text("Keep this file.", encoding="utf-8")

    manager.call("session", "export")

    assert target.read_text(encoding="utf-8") == "Keep this file."
    interaction.confirm.assert_called_once_with(
        f"Replace the existing file at '{target}'?",
        default=False,
    )


def test_export_command_reads_saved_session_without_resuming_and_writes_archive(
    tmp_path,
    monkeypatch,
):
    """Export can archive a selected saved session without replacing the active session."""
    interaction = Mock(spec=ConsoleInteraction)
    store = MemorySessionStore()
    active = Session(id="active", name="Active", name_source="user")
    saved = Session(
        id="saved",
        name="Saved",
        name_source="user",
        messages=[Message(role="user", content="Saved conversation.")],
    )
    store.save(saved)
    sessions = SessionManager(session=active, session_store=store)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(sessions, Mock()))
    monkeypatch.chdir(tmp_path)

    manager.call("sessions", "export saved format=archive")

    target = tmp_path / "loop-session-saved.zip"
    assert sessions.session is active
    with ZipFile(target) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        serialized = json.loads(archive.read("session.json"))
    assert manifest["session_id"] == "saved"
    assert serialized["id"] == "saved"
    interaction.info.assert_called_once_with(f"Exported session 'Saved' to {target.resolve()}.")


def test_sessions_export_writes_a_saved_session_to_a_quoted_destination(tmp_path):
    """The plural command exports a selected session to a path containing spaces."""
    interaction = Mock(spec=ConsoleInteraction)
    store = MemorySessionStore()
    active = Session(id="active", name="Active", name_source="user")
    store.save(
        Session(
            id="saved",
            name="Saved",
            name_source="user",
            messages=[Message(role="user", content="Export the saved session.")],
        )
    )
    sessions = SessionManager(session=active, session_store=store)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(sessions, Mock()))
    target = tmp_path / "saved sessions.md"

    manager.call("sessions", f'export saved "{target}"')

    assert "Export the saved session." in target.read_text(encoding="utf-8")
    assert sessions.session is active


def test_export_command_preserves_existing_destination_when_replacement_is_declined(tmp_path):
    """A no response leaves an existing export untouched."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm.return_value = False
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager(), Mock()))
    target = tmp_path / "existing.md"
    target.write_text("Keep this file.", encoding="utf-8")

    manager.call("session", f"export {target}")

    assert target.read_text(encoding="utf-8") == "Keep this file."
    interaction.confirm.assert_called_once_with(
        f"Replace the existing file at '{target.resolve()}'?",
        default=False,
    )
    interaction.info.assert_called_once_with("Session export stopped.")


def test_async_export_dispatch_keeps_file_work_off_the_event_loop(tmp_path, monkeypatch):
    """A sync export command runs in a worker while the async owner continues scheduling."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.worker_calls_routed = True
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager(), Mock()))
    target = tmp_path / "async-export.md"
    release = Event()
    export_threads = []
    loop_released_worker = []
    original_export = session_commands_module.export_session

    def wait_then_export(session, store, destination, *, format="markdown", overwrite=False):
        """Pause the file operation until the async owner gets a scheduling turn."""
        export_threads.append(get_ident())
        loop_released_worker.append(release.wait(timeout=0.5))
        return original_export(session, store, destination, format=format, overwrite=overwrite)

    monkeypatch.setattr(session_commands_module, "export_session", wait_then_export)

    async def dispatch():
        """Let the event loop release the simulated blocking export operation."""
        owner_thread = get_ident()
        asyncio.get_running_loop().call_later(0.01, release.set)
        await manager.call_async("session", f"export {target}", interaction=interaction)
        assert export_threads[0] != owner_thread

    asyncio.run(dispatch())

    assert loop_released_worker == [True]
    assert target.is_file()


def test_export_command_replaces_existing_destination_after_confirmation(tmp_path):
    """A yes response authorizes replacing the destination and reports completion."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm.return_value = True
    session = Session(name="Replace session", name_source="user")
    sessions = SessionManager(session=session)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(sessions, Mock()))
    target = tmp_path / "existing.md"
    target.write_text("Old contents.", encoding="utf-8")

    manager.call("session", f"export {target}")

    assert "Old contents." not in target.read_text(encoding="utf-8")
    interaction.confirm.assert_called_once_with(
        f"Replace the existing file at '{target.resolve()}'?",
        default=False,
    )
    interaction.info.assert_called_once_with(
        f"Exported session 'Replace session' to {target.resolve()}."
    )


def test_export_command_prompts_if_the_destination_appears_during_export(tmp_path, monkeypatch):
    """A raced export collision still requires yes-or-no replacement confirmation."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm.return_value = True
    sessions = SessionManager(session=Session(name="Racing session", name_source="user"))
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(sessions, Mock()))
    target = tmp_path / "racing.md"
    original_export = session_commands_module.export_session
    attempts = 0

    def race_once(session, store, destination, *, format="markdown", overwrite=False):
        """Create one concurrent destination before allowing the next export attempt."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            target.write_text("Concurrent file.", encoding="utf-8")
            raise FileExistsError(target)
        return original_export(session, store, destination, format=format, overwrite=overwrite)

    monkeypatch.setattr(session_commands_module, "export_session", race_once)

    manager.call("session", f"export {target}")

    assert attempts == 2
    interaction.confirm.assert_called_once_with(
        f"Replace the existing file at '{target.resolve()}'?",
        default=False,
    )
    assert "Concurrent file." not in target.read_text(encoding="utf-8")


def test_export_command_reports_write_collisions_after_replacement_confirmation(
    tmp_path,
    monkeypatch,
):
    """An unresolved write collision after a yes response is reported as an export failure."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm.return_value = True
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager(), Mock()))
    target = tmp_path / "collision.md"
    target.write_text("Preserve this file.", encoding="utf-8")

    def fail_after_confirmation(*args, **kwargs):
        """Simulate a write collision that remains after replacement was authorized."""
        raise FileExistsError(target)

    monkeypatch.setattr(
        session_commands_module,
        "export_session",
        fail_after_confirmation,
    )

    manager.call("session", f"export {target}")

    assert target.name in interaction.report.call_args.args[0].detail
    assert target.read_text(encoding="utf-8") == "Preserve this file."
    interaction.confirm.assert_called_once_with(
        f"Replace the existing file at '{target.resolve()}'?",
        default=False,
    )


def test_export_command_rejects_directories_and_the_session_database(tmp_path):
    """Export validates file destinations and refuses to replace its session database."""
    interaction = Mock(spec=ConsoleInteraction)
    database = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(database, workspace_id="workspace")
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager(session_store=store), Mock()))
    directory = tmp_path / "directory"
    directory.mkdir()

    manager.call("session", f"export {directory}")
    assert "destination must be a file" in interaction.report.call_args.args[0].detail
    manager.call("session", f"export {database}")
    assert "must not be the session database" in interaction.report.call_args.args[0].detail
    interaction.prompt.assert_not_called()
    interaction.confirm.assert_not_called()


def test_export_command_reports_missing_saved_sessions_and_invalid_formats(tmp_path):
    """Export reports unavailable session IDs and unsupported direct-call formats."""
    interaction = Mock(spec=ConsoleInteraction)
    sessions = SessionManager()
    commands = SessionCommands(sessions, Mock())
    manager = CommandManager(interaction=interaction)
    manager.register_provider(commands)

    manager.call("sessions", "export missing")
    assert "Session 'missing' was not found" in interaction.report.call_args.args[0].detail
    manager.call("export", "missing")
    assert "Session 'missing' was not found" in interaction.report.call_args.args[0].detail

    with pytest.raises(CommandArgumentError, match="Unsupported session export format"):
        commands.export(
            CommandContext(name="export", interaction=interaction),
            destination=str(tmp_path / "invalid.md"),
            format="pdf",
        )


def test_rename_reports_invalid_session_names():
    """Rename translates invalid domain values into standard command argument feedback."""
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SessionCommands(SessionManager(), Mock()))

    manager.call("rename", "''")

    assert "Session name cannot be empty" in interaction.report.call_args.args[0].detail


def test_rename_without_a_name_generates_a_session_name():
    """Rename without arguments uses the configured automatic naming service."""
    interaction = Mock(spec=ConsoleInteraction)
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
    interaction = Mock(spec=ConsoleInteraction)
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
