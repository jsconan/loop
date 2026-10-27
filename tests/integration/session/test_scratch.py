"""Verify durable session working files across independent application processes."""

import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Event
from unittest.mock import Mock

import pytest

from loop import Interaction, SessionExecutionConflictError, SessionManager, SQLiteSessionStore
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


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "outcome", ["answer", "reject", "cancel", "crash", "failure", "registration-stop"]
)
def test_idle_export_confirmation_excludes_concurrent_recovery(tmp_path, mode, outcome):
    """Idle export confirmations retain ownership through answer, cancellation, or owner death."""
    database = tmp_path / "private" / "sessions.db"
    destination = tmp_path / "work" / "existing export.md"
    destination.parent.mkdir(mode=0o700)
    destination.write_text("original export")
    source = r"""
import asyncio,json,logging,shlex,signal,sys
from contextlib import nullcontext
from logging.handlers import BufferingHandler
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
from loop import (ConsoleInteraction,Loop,PermissionManager,SessionController,SessionManager,
                  SQLiteSessionStore,ToolRegistry)
from loop.session import LocalSessionNameGenerator
signal.alarm(20)
diagnostics=BufferingHandler(1000)
logging.getLogger("loop.operational").addHandler(diagnostics)
inputs=asyncio.Queue()
ui=MagicMock(spec=ConsoleInteraction)
controller=SessionController(ui)
manager=SessionManager(interaction=controller.interaction,
    session_store=SQLiteSessionStore(sys.argv[1],workspace_id="workspace"))
manager.rename_session("kept conversation")
register=manager.record_interaction_requested_async
async def register_then_stop(*args,**kwargs):
    await register(*args,**kwargs)
    if sys.argv[4]=="registration-stop":
        print(json.dumps({"id":manager.session.id}),flush=True)
        await asyncio.to_thread(sys.stdin.readline)
        controller.stop()
        await asyncio.sleep(0)
manager.record_interaction_requested_async=register_then_stop
async def prompt(*args,**kwargs):
    return await inputs.get()
async def confirm(*args,**kwargs):
    if sys.argv[4]=="registration-stop":
        return True
    print(json.dumps({"id":manager.session.id}),flush=True)
    answer=(await asyncio.to_thread(sys.stdin.readline)).strip()
    if answer=="cancel":
        controller.stop()
        await asyncio.Future()
    if answer=="failure":
        manager.store.save=MagicMock(side_effect=OSError("settlement unavailable"))
    return answer=="answer"
def info(message=""):
    if message.startswith("Exported session ") or message=="Session export stopped.":
        inputs.put_nowait(False)
ui.prompt_async.side_effect=prompt
ui.confirm_async.side_effect=confirm
ui.info.side_effect=info
workspace=Path(sys.argv[2]).parent
permissions=PermissionManager(workspace)
registry=ToolRegistry(permission_manager=permissions)
backend=SimpleNamespace(default_model="fixture",get_context_window=lambda _model:None)
loop=Loop.create_default(backend,interaction=controller.interaction,run_control=controller.control,
    session_manager=manager,working_directory=workspace,permission_manager=permissions,
    tool_registry=registry,session_name_generator=LocalSessionNameGenerator())
controller.bind_session_manager(manager)
controller.bind_command_manager(loop.command_manager)
inputs.put_nowait("/session export "+shlex.quote(sys.argv[2]))
try:
    with patch("loop.conversation.controller.patch_stdout",lambda **kwargs:nullcontext()):
        if sys.argv[3]=="sync":
            controller.run(lambda receive:loop.run(input_source=receive),route_idle_commands=True)
        else:
            asyncio.run(controller.run_async(
                lambda receive:loop.run_async(input_source=receive),route_idle_commands=True))
    if sys.argv[4]=="registration-stop":
        assert ui.confirm_async.await_count==0
except OSError as error:
    if str(error)!="settlement unavailable":
        raise
    failures=[record for record in diagnostics.buffer
        if getattr(record,"event.name",None)=="session.persistence.failed"]
    assert len(failures)==1
    assert getattr(failures[0],"error.type")=="session.persistence_failed"
    print("settlement failed",flush=True)
finally:
    loop.command_manager.close()
    registry.close()
    permissions.close()
"""
    contender_source = """
import sys
from unittest.mock import Mock
from loop import Interaction,SessionExecutionConflictError,SessionManager,SQLiteSessionStore
try:
    SessionManager(interaction=Mock(spec=Interaction),session=sys.argv[2],
        session_store=SQLiteSessionStore(sys.argv[1],workspace_id="workspace"))
except SessionExecutionConflictError:
    print("owned")
else:
    raise RuntimeError("Live idle confirmation was admitted for recovery")
"""
    owner = subprocess.Popen(
        [sys.executable, "-B", "-u", "-c", source, str(database), str(destination), mode, outcome],
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
        if outcome == "crash":
            owner.kill()
            stdout, stderr = owner.communicate(timeout=5)
        else:
            stdout, stderr = owner.communicate(outcome + "\n", timeout=10)
            assert owner.returncode == 0, stderr
        assert stdout == ("settlement failed\n" if outcome == "failure" else "")
        assert stderr == ""
        restored = SessionManager(
            interaction=Mock(spec=Interaction), session=session_id, session_store=store
        )
        expected = (
            "revoked"
            if outcome in {"cancel", "crash", "failure", "registration-stop"}
            else "answered"
        )
        assert restored.session.pending_interactions == ()
        assert restored.session.events[-1].outcome == expected
        assert sum(event.type == "interaction_settled" for event in restored.session.events) == 1
        revision = restored.session.revision
        reopened = SessionManager(
            interaction=Mock(spec=Interaction), session=session_id, session_store=store
        )
        assert reopened.session.revision == revision
        with reopened.execution():
            pass
        if outcome == "answer":
            assert "kept conversation" in destination.read_text()
        else:
            assert destination.read_text() == "original export"
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.communicate(timeout=5)


@pytest.mark.parametrize("selection", [False, True])
@pytest.mark.parametrize("late_value", [False, True])
def test_requester_cancellation_retains_ownership_through_revocation(
    tmp_path, selection, late_value
):
    """A cancelled owner closes its prompt and retains native exclusion until revocation commits."""
    database = tmp_path / "sessions.db"
    source = """
import asyncio,json,signal,sys
from contextlib import nullcontext
from unittest.mock import MagicMock,patch
from loop import ConsoleInteraction,SessionController,SessionManager,SQLiteSessionStore
signal.alarm(20)
async def main():
    ui=MagicMock(spec=ConsoleInteraction)
    manager=SessionManager(interaction=ui,
        session_store=SQLiteSessionStore(sys.argv[1],workspace_id="workspace"))
    controller=SessionController(ui)
    controller.bind_session_manager(manager)
    ready,cancelled=asyncio.Event(),asyncio.Event()
    owners=[]
    async def confirm(*args,**kwargs):
        ready.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            if sys.argv[3]=="True":
                return True
            raise
    async def conversation(receive):
        owners.append(asyncio.current_task())
        with manager.execution(allow_session_selection=sys.argv[2]=="True"):
            try:
                await controller.interaction.confirm_async("Approve?")
            except asyncio.CancelledError:
                pass
    settle=manager.record_interaction_settled_async
    async def settle_after_gate(*args,**kwargs):
        assert cancelled.is_set()
        print("settling",flush=True)
        await asyncio.to_thread(sys.stdin.readline)
        await settle(*args,**kwargs)
    ui.confirm_async.side_effect=confirm
    manager.record_interaction_settled_async=settle_after_gate
    with patch("loop.conversation.controller.patch_stdout",lambda **kwargs:nullcontext()):
        task=asyncio.create_task(controller.run_async(conversation))
        await ready.wait()
        print(json.dumps({"id":manager.session.id}),flush=True)
        await asyncio.to_thread(sys.stdin.readline)
        owners[0].cancel()
        await owners[0]
        await task
    assert manager.session.pending_interactions==()
    assert manager.session.events[-1].outcome=="revoked"
    print("done",flush=True)
asyncio.run(main())
"""
    owner = subprocess.Popen(
        [sys.executable, "-B", "-u", "-c", source, str(database), str(selection), str(late_value)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        session_id = json.loads(owner.stdout.readline())["id"]
        store = SQLiteSessionStore(database, workspace_id="workspace")
        snapshot = store.load(session_id).serialize()
        with pytest.raises(SessionExecutionConflictError):
            SessionManager(
                interaction=Mock(spec=Interaction), session=session_id, session_store=store
            )
        owner.stdin.write("cancel\n")
        owner.stdin.flush()
        assert owner.stdout.readline() == "settling\n"
        assert store.load(session_id).serialize() == snapshot
        with pytest.raises(SessionExecutionConflictError):
            SessionManager(
                interaction=Mock(spec=Interaction), session=session_id, session_store=store
            )
        stdout, stderr = owner.communicate("commit\n", timeout=10)
        assert owner.returncode == 0, stderr
        assert stdout == "done\n" and stderr == ""
        restored = SessionManager(
            interaction=Mock(spec=Interaction), session=session_id, session_store=store
        )
        assert restored.session.pending_interactions == ()
        assert restored.session.events[-1].outcome == "revoked"
        assert sum(event.type == "interaction_settled" for event in restored.session.events) == 1
        with restored.execution():
            pass
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.communicate(timeout=5)


def test_busy_database_admission_does_not_block_an_independent_store(tmp_path, monkeypatch):
    """A real SQLite writer wait cannot stall another store's admission or settlement."""
    database = tmp_path / "sessions.db"
    manager = SessionManager(
        interaction=Mock(spec=Interaction),
        session_store=SQLiteSessionStore(database, workspace_id="workspace"),
    )
    original = manager.temporary_directory.get()
    independent = SessionManager(interaction=Mock(spec=Interaction))
    waiting = Event()
    connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        """Observe the actual writer-lock request before SQLite waits for the owned writer."""
        connection = connect(*args, **kwargs)
        connection.set_trace_callback(
            lambda statement: waiting.set() if statement == "BEGIN IMMEDIATE" else None
        )
        return connection

    def admit(selected):
        """Enter and settle one real ownership lease against the selected store."""
        with selected.execution() as directory:
            assert directory.is_dir()
            return directory

    with closing(connect(database)) as writer, ThreadPoolExecutor(max_workers=2) as executor:
        writer.execute("BEGIN IMMEDIATE")
        monkeypatch.setattr(sqlite3, "connect", traced_connect)
        pending = executor.submit(admit, manager)
        try:
            assert waiting.wait(timeout=5)
            executor.submit(admit, independent).result(timeout=2)
        finally:
            writer.rollback()
        assert pending.result(timeout=5) == original
    with manager.execution() as directory:
        assert directory == original


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
from loop import (ApprovalChoice, Interaction, PermissionManager, Session, SessionManager,
                  SQLiteSessionStore, ToolRegistry)
from loop.execution import CommandExecutionService
from loop.execution.sandbox import UnavailableSandboxBackend, select_sandbox_backend
from loop.instructions import InstructionsManager, RuntimeEnvironment
from loop.tools.files import read_text_file, edit_text_file, write_text_file
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
    permissions.bind_temporary_directory(manager.temporary_directory,
        ordinary_names=manager.working_directory_allows_private_names)
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(
        Path(sys.argv[2]), manager.temporary_directory))
    host = Mock()
    host.run_host_command.side_effect = AssertionError("unapproved host retry")
    registry = ToolRegistry([run_command, read_text_file, edit_text_file, write_text_file],
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
                              "$WORKSPACE_ROOT", (), succeeded=True)
        return result["result"]
    try:
        if not sys.argv[3]:
            dispatch("run_command", command='printf draft > "$TMPDIR/report.txt"; '
                + sqlite_command("CREATE TABLE state(value); INSERT INTO state VALUES ('retained SQL');"),
                read_only=True)
            visible = instructions.virtual_paths.display(str(manager.temporary_directory.get()))
            script = ('set -eu\ncd "$WORKSPACE_ROOT"\nmkdir "$TMPDIR/.loop"\n'
                      'cat "$TMPDIR/report.txt"\n'
                      '/bin/sh -c \'printf child > "$TMPDIR/.loop/data"\'\n')
            dispatch("write_text_file", path=visible + "/roundtrip.sh", content=script)
            assert dispatch("run_command", command='/bin/sh "$TMPDIR/roundtrip.sh"',
                            read_only=True)["stdout"]["content"] == "draft"
            assert dispatch("read_text_file", path=visible + "/.loop/data")["content"] == "child"
            manager.add_user_message("retain working state")
            diagnostics = dispatch("run_command", command='i=0; while [ "$i" -lt 2048 ]; '
                'do printf "%s\n" "$TMPDIR"; i=$((i+1)); done', read_only=True)
            cached = cached_path(diagnostics["stdout"]["handle"])[0].read_text()
            visible = instructions.virtual_paths.display(str(manager.temporary_directory.get()))
            assert visible in cached
        else:
            assert dispatch("read_text_file", path="$TMPDIR/report.txt")["content"] == "draft"
            dispatch("edit_text_file", path="$TMPDIR/report.txt", old_content="draft",
                new_content="revised")
        content = dispatch("read_text_file", path="$TMPDIR/report.txt")["content"]
        state = dispatch("run_command", command=sqlite_command("SELECT value FROM state;", select=True),
            read_only=True)["stdout"]["content"].strip()
        persisted = manager.store.load(manager.session.id)
        native = str(manager.temporary_directory.get())
        visible = instructions.virtual_paths.display(native)
        outputs = [json.loads(item.output) for item in persisted.messages
                   if getattr(item, "output", None)]
        assert visible in json.dumps(outputs, ensure_ascii=False)
        for kind, filename in (("markdown", "transcript.md"), ("archive", "transcript.zip")):
            destination = Path(sys.argv[2]) / filename
            export_session(persisted, manager.store, destination, format=kind, overwrite=True)
            if kind == "markdown":
                assert visible in destination.read_text()
            else:
                with ZipFile(destination) as archive:
                    archived = Session.deserialize(archive.read("session.json").decode())
                archived_outputs = [json.loads(item.output) for item in archived.messages
                                    if getattr(item, "output", None)]
                assert visible in json.dumps(archived_outputs, ensure_ascii=False)
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
