"""Verify durable session working files across independent application processes."""

import json
import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

from loop import Interaction, SessionManager, SQLiteSessionStore
from loop.utils.path import private_directory_lock

pytestmark = pytest.mark.integration


def test_saved_working_files_survive_process_exit_and_resume(tmp_path):
    """A resumed session reopens its working files after the creating process has exited."""
    database = tmp_path / "storage" / "sessions.db"
    source = """
import json
import sys
from unittest.mock import Mock
from loop import Interaction, SessionManager, SQLiteSessionStore

manager = SessionManager(
    interaction=Mock(spec=Interaction),
    session=sys.argv[2] or None,
    session_store=SQLiteSessionStore(sys.argv[1], workspace_id="workspace"),
    workspace_id="workspace",
)
directory = manager.temporary_directory.get()
if not sys.argv[2]:
    (directory / "report.txt").write_text("retained", encoding="utf-8")
    manager.add_user_message("prepare a report")
print(json.dumps({
    "session_id": manager.session.id,
    "directory": str(directory),
    "content": (directory / "report.txt").read_text(encoding="utf-8"),
}))
"""

    def run(identifier):
        """Launch an isolated application process and collect its public storage result."""
        result = subprocess.run(
            [sys.executable, "-B", "-c", source, str(database), identifier],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        assert result.stderr == ""
        return json.loads(result.stdout)

    created = run("")
    resumed = run(created["session_id"])

    assert created["content"] == "retained"
    assert resumed == created


def test_native_ownership_serializes_one_session_and_releases_after_owner_death(tmp_path):
    """Cooperating processes exclude shared-session execution while independent sessions overlap."""
    database = tmp_path / "sessions.db"
    source = """
import json, signal, sys
from unittest.mock import Mock
from loop import Interaction, Session, SessionManager, SQLiteSessionStore, SessionExecutionConflictError
signal.alarm(15)
try:
    manager = SessionManager(
        interaction=Mock(spec=Interaction), session=Session(id=sys.argv[2]),
        session_store=SQLiteSessionStore(sys.argv[1], workspace_id="workspace"),
    )
    with manager.execution():
        print(json.dumps({"owned": True, "directory": str(manager.temporary_directory.get())}), flush=True)
        if sys.argv[3] == "hold":
            sys.stdin.readline()
except SessionExecutionConflictError:
    print(json.dumps({"owned": False}), flush=True)
"""
    owner = subprocess.Popen(
        [sys.executable, "-B", "-u", "-c", source, str(database), "shared", "hold"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        initial = json.loads(owner.stdout.readline())
        assert initial["owned"] is True

        def attempt(identifier):
            """Attempt bounded independent-process admission without waiting for the owner."""
            result = subprocess.run(
                [sys.executable, "-B", "-c", source, str(database), identifier, "attempt"],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            assert result.stderr == ""
            return json.loads(result.stdout)

        assert attempt("shared") == {"owned": False}
        assert attempt("independent")["owned"] is True
        owner.kill()
        stdout, stderr = owner.communicate(timeout=5)
        assert stdout == stderr == ""
        assert attempt("shared") == initial
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.communicate(timeout=5)


@pytest.mark.parametrize("crash", [False, True])
def test_concurrent_load_preserves_live_confirmation_and_recovers_abandoned_requests(
    tmp_path, crash
):
    """A rejected process cannot revoke live approval; owner death permits one recovery."""
    database = tmp_path / "sessions.db"
    source = """
import asyncio, json, signal, sys
from unittest.mock import Mock
from loop import Interaction, SessionManager, SQLiteSessionStore
signal.alarm(15)
manager = SessionManager(interaction=Mock(spec=Interaction),
    session_store=SQLiteSessionStore(sys.argv[1], workspace_id="workspace"))
with manager.execution():
    asyncio.run(manager.record_interaction_requested_async("confirmation", kind="confirm",
        session_id=None, run_id=None, operation_id=None))
    print(json.dumps({"id": manager.session.id}), flush=True)
    if sys.stdin.readline().strip() == "answer":
        asyncio.run(manager.record_interaction_settled_async("confirmation", "answered"))
"""
    contender_source = """
import sys
from unittest.mock import Mock
from loop import Interaction, SessionExecutionConflictError, SessionManager, SQLiteSessionStore
try:
    SessionManager(interaction=Mock(spec=Interaction), session=sys.argv[2],
        session_store=SQLiteSessionStore(sys.argv[1], workspace_id="workspace"))
except SessionExecutionConflictError:
    print("owned")
else:
    raise RuntimeError("Live confirmation was admitted for recovery")
"""
    owner = subprocess.Popen(
        [sys.executable, "-B", "-u", "-c", source, str(database)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        session_id = json.loads(owner.stdout.readline())["id"]
        store = SQLiteSessionStore(database, workspace_id="workspace")
        snapshot = store.load(session_id).serialize()
        contender = subprocess.run(
            [sys.executable, "-B", "-c", contender_source, str(database), session_id],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        assert contender.stdout == "owned\n" and contender.stderr == ""
        assert store.load(session_id).serialize() == snapshot
        if crash:
            owner.kill()
            stdout, stderr = owner.communicate(timeout=5)
        else:
            stdout, stderr = owner.communicate("answer\n", timeout=5)
            assert owner.returncode == 0
        assert stdout == stderr == ""
        restored = SessionManager(
            interaction=Mock(spec=Interaction), session=session_id, session_store=store
        )
        assert restored.session.pending_interactions == ()
        assert restored.session.events[-1].outcome == ("revoked" if crash else "answered")
        revision = restored.session.revision
        reopened = SessionManager(
            interaction=Mock(spec=Interaction), session=session_id, session_store=store
        )
        assert reopened.session.revision == revision
        assert sum(event.type == "interaction_settled" for event in reopened.session.events) == 1
        with reopened.execution():
            assert reopened.temporary_directory.get().is_dir()
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.communicate(timeout=5)


def test_ownership_descriptor_closes_on_exec(tmp_path):
    """Payload exec cannot inherit the harness ownership lock even with close_fds disabled."""
    with private_directory_lock(tmp_path, "owner") as descriptor:
        assert not os.get_inheritable(descriptor)
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                "import os, sys\ntry: os.fstat(int(sys.argv[1]))\nexcept OSError: sys.exit(0)\nsys.exit(1)",
                str(descriptor),
            ],
            check=False,
            close_fds=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0
        assert result.stdout == result.stderr == ""


def test_native_commands_file_tools_and_sqlite_survive_independent_resume(tmp_path):
    """A new harness process reads and edits command-created bytes and retains SQLite state."""
    database = tmp_path / "Library/Application Support/Loop é ;$'" / "sessions.db"
    workspace = tmp_path / "work"
    workspace.mkdir(mode=0o700)
    source = r"""
import json, shlex, sys, tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from loop import (ApprovalChoice, Interaction, PermissionManager, SessionManager,
                  SQLiteSessionStore, ToolRegistry)
from loop.execution import CommandExecutionService
from loop.execution.sandbox import UnavailableSandboxBackend, select_sandbox_backend
from loop.instructions import InstructionsManager, RuntimeEnvironment
from loop.tools.files import read_text_file, edit_text_file
from loop.tools.system import run_command
from loop.session.exporter import export_session
from loop.utils import cached_path
from zipfile import ZipFile
monitor = Mock()
monitor.denial.return_value = None
if sys.platform == "linux":
    # Fresh probe scratch must match the installed receiver's native system temporary root.
    tempfile.tempdir = "/tmp"
with patch("loop.execution.sandbox.macos.backend.SeatbeltDiagnosticService.monitor", return_value=monitor), \
     patch("loop.execution.sandbox.macos.backend._denial_from_log", return_value=None):
    backend = select_sandbox_backend()
    if isinstance(backend, UnavailableSandboxBackend):
        print(json.dumps({"unavailable": str(backend.capability_failure())}))
        sys.exit(0)
    interaction = Mock(spec=Interaction)
    interaction.prompt.side_effect = lambda message, **kwargs: (
        "deny" if "without the OS sandbox" in message else ApprovalChoice.ONCE
        if message == "Proceed?" else "approve")
    manager = SessionManager(interaction=interaction, session=sys.argv[3] or None,
        session_store=SQLiteSessionStore(sys.argv[1], workspace_id="workspace"))
    permissions = PermissionManager(Path(sys.argv[2]))
    permissions.bind_temporary_directory(manager.temporary_directory)
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(
        Path(sys.argv[2]), manager.temporary_directory))
    host = Mock()
    host.run_host_command.side_effect = AssertionError("unapproved host retry")
    registry = ToolRegistry([run_command, read_text_file, edit_text_file],
        permission_manager=permissions, execution_service=CommandExecutionService(backend, host))
    def sqlite_command(query, *, select=False):
        '''Use the installed Linux runtime's Python SQLite module or the macOS system CLI.'''
        if sys.platform != "linux":
            return 'sqlite3 "$TMPDIR/state.db" ' + shlex.quote(query)
        operation = (f"print(connection.execute({query!r}).fetchone()[0]);" if select else
                     f"connection.executescript({query!r});connection.commit();")
        program = (
            "import os,sqlite3;"
            "connection=sqlite3.connect(os.path.join(os.environ['TMPDIR'],'state.db'));"
            + operation + "connection.close()"
        )
        return "/usr/bin/python3 -c " + shlex.quote(program)

    def dispatch(name, **arguments):
        with manager.execution():
            result = json.loads(registry.call(name, json.dumps(arguments),
                interaction=interaction, instructions_manager=instructions))
        if not result["ok"]:
            raise RuntimeError(json.dumps(result))
        manager.add_tool_call("synthetic-" + str(len(manager.messages)), json.dumps(result),
                              "/workspace", (), succeeded=True)
        return result["result"]
    try:
        if not sys.argv[3]:
            dispatch("run_command", command='printf draft > "$TMPDIR/report.txt"; '
                + sqlite_command("CREATE TABLE state(value); INSERT INTO state VALUES ('retained SQL');"),
                read_only=True)
            manager.add_user_message("retain working state")
            diagnostics = dispatch("run_command", command='i=0; while [ "$i" -lt 2048 ]; '
                'do printf "%s\n" "$TMPDIR"; i=$((i+1)); done', read_only=True)
            cached = cached_path(diagnostics["stdout"]["handle"])[0].read_text()
            assert "$TMPDIR" in cached
            assert str(manager.temporary_directory.get()) not in cached
        else:
            assert dispatch("read_text_file", path="$TMPDIR/report.txt")["content"] == "draft"
            dispatch("edit_text_file", path="$TMPDIR/report.txt", old_content="draft",
                new_content="revised")
        content = dispatch("read_text_file", path="$TMPDIR/report.txt")["content"]
        state = dispatch("run_command", command=sqlite_command("SELECT value FROM state;", select=True),
            read_only=True)["stdout"]["content"].strip()
        persisted = manager.store.load(manager.session.id)
        native = str(manager.temporary_directory.get())
        assert native not in persisted.serialize()
        for kind, filename in (("markdown", "transcript.md"), ("archive", "transcript.zip")):
            destination = Path(sys.argv[2]) / filename
            export_session(persisted, manager.store, destination, format=kind, overwrite=True)
            if kind == "markdown":
                exported = destination.read_text()
            else:
                with ZipFile(destination) as archive:
                    exported = "".join(archive.read(name).decode() for name in archive.namelist())
            assert native not in exported
            assert "$TMPDIR" in exported
        print(json.dumps({"id": manager.session.id, "content": content, "state": state,
                          "directory": str(manager.temporary_directory.get())}))
    finally:
        registry.close()
        permissions.close()
"""

    def run(identifier):
        """Run a bounded independent harness with real native confinement and synthetic data."""
        result = subprocess.run(
            [sys.executable, "-B", "-c", source, str(database), str(workspace), identifier],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        value = json.loads(result.stdout)
        if "unavailable" in value:
            pytest.skip(value["unavailable"])
        return value

    created = run("")
    resumed = run(created["id"])
    assert created["content"] == "draft"
    assert resumed["content"] == "revised"
    assert created["state"] == resumed["state"] == "retained SQL"
    assert created["directory"] == resumed["directory"]
