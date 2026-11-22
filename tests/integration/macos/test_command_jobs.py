"""Qualify owned native command continuations through public macOS tool dispatch."""

import asyncio
import errno
import json
import os
import select
import shlex
import socket
import subprocess
import sys
import time
from array import array
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from unittest.mock import AsyncMock, Mock

import pytest

from loop import PermissionManager, SessionExecutionConflictError, SessionManager, ToolRegistry
from loop.execution import CommandExecutionService
from loop.execution.coordinator import LocalHostCommandExecutor
from loop.execution.sandbox.contracts import SandboxCapability
from loop.execution.sandbox.macos import backend as macos_backend
from loop.execution.sandbox.macos.backend import MacOSSeatbeltBackend
from loop.execution.sandbox.selection import select_sandbox_backend
from loop.instructions import InstructionsManager, RuntimeEnvironment
from loop.tooling import ToolRuntimeSettings
from loop.tools.system import (
    cancel_command_job,
    control_command_terminal,
    inspect_command_jobs,
    poll_command_job,
    read_command_output,
    run_command,
)
from loop.tools.web import read_cached_content
from loop.utils.content import encode_content_cursor

pytestmark = [pytest.mark.integration, pytest.mark.e2e, pytest.mark.macos]


def _registry(
    workspace: Path,
    *,
    terminal: bool = False,
    command_default_initial_wait_seconds: float = 10,
    command_job_runtime_limit_seconds: float = 3600,
) -> tuple[ToolRegistry, Mock, SessionManager]:
    """Compose public command tools with one real native backend and approval surface."""
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    interaction.prompt_async = AsyncMock(return_value="approve")
    sessions = SessionManager()
    permissions = PermissionManager(workspace)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    backend = select_sandbox_backend() if terminal else MacOSSeatbeltBackend()
    if terminal:
        assert SandboxCapability.TERMINAL in backend.capabilities
    registry = ToolRegistry(
        [
            run_command,
            inspect_command_jobs,
            poll_command_job,
            read_command_output,
            read_cached_content,
            cancel_command_job,
            control_command_terminal,
        ],
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, LocalHostCommandExecutor()),
        settings=ToolRuntimeSettings(
            command_default_initial_wait_seconds=command_default_initial_wait_seconds,
            command_job_runtime_limit_seconds=command_job_runtime_limit_seconds,
        ),
        session_manager=sessions,
    )
    return registry, interaction, sessions


def _call(registry: ToolRegistry, interaction: Mock, tool: str, **arguments) -> dict:
    """Decode one public dispatch result without invoking a model."""
    return json.loads(registry.call(tool, json.dumps(arguments), interaction=interaction))


def _read_output_page(
    registry: ToolRegistry,
    interaction: Mock,
    handle: str,
    *,
    stream: str = "stdout",
    cursor: str | None = None,
) -> dict:
    """Read one immediately available page from an owned command stream."""
    result = _call(
        registry,
        interaction,
        "read_command_output",
        handle=handle,
        stream=stream,
        cursor=cursor,
    )
    assert result["ok"], result
    return result["result"]


def _poll_until_complete(
    registry: ToolRegistry, interaction: Mock, handle: str, *, wait_seconds: float = 5
) -> dict:
    """Wait for job completion without coupling the wait to output arrival."""
    for _ in range(10):
        status = _call(
            registry,
            interaction,
            "poll_command_job",
            handle=handle,
            wait_seconds=wait_seconds,
        )
        assert status["ok"], status
        if status["result"]["status"] == "complete":
            return status["result"]
    pytest.fail("The owned native command did not settle.")


def _read_complete_output(
    registry: ToolRegistry, interaction: Mock, handle: str, *, stream: str = "stdout"
) -> str:
    """Read the completed stream in byte-cursor pages without a live ring."""
    cursor = None
    output = []
    while True:
        page = _read_output_page(registry, interaction, handle, stream=stream, cursor=cursor)
        output.append(page["content"])
        cursor = page.get("next_cursor")
        if cursor is None:
            return "".join(output)


def _terminal_result(registry: ToolRegistry, interaction: Mock, handle: str) -> tuple[dict, str]:
    """Wait for an owned terminal to settle, then read its stdout stream."""
    status = _poll_until_complete(registry, interaction, handle)
    return status, _read_complete_output(registry, interaction, handle)


def test_native_terminal_keeps_streaming_after_final_capture_limit(native_workspace):
    """A native terminal remains readable after output exceeds the native result cap."""
    registry, interaction, _sessions = _registry(native_workspace, terminal=True)
    output_limit = macos_backend.constants.MAX_OUTPUT_CHARS + 128
    command = (
        f"printf '%{output_limit}s' x | /usr/bin/tr ' ' x; "
        "read -r response; printf 'after-limit\\n'"
    )
    try:
        started = _call(
            registry,
            interaction,
            "run_command",
            command=command,
            cwd=str(native_workspace),
            interactive=True,
        )
        assert started["ok"], started
        handle = started["result"]["handle"]
        output = ""
        cursor = None
        deadline = time.monotonic() + 10
        while len(output) < output_limit and time.monotonic() < deadline:
            page = _read_output_page(registry, interaction, handle, cursor=cursor)
            output += page["content"]
            cursor = page.get("next_cursor")
            time.sleep(0.02)
        assert output == "x" * output_limit
        sent = _call(
            registry,
            interaction,
            "control_command_terminal",
            handle=handle,
            action="write",
            content="go\n",
        )
        assert sent["ok"], sent
        status = _poll_until_complete(registry, interaction, handle)
        page = _read_output_page(registry, interaction, handle, cursor=cursor)
        output += page["content"]
        assert "after-limit" in output
        assert status["stdout_capture_complete"]
        assert status["stdout_bytes"] == len(output.encode("utf-8"))
    finally:
        registry.close()


def test_native_command_continuation_retains_output_and_session_ownership(native_workspace):
    """The same native process supplies status and byte cursors isolated by session."""
    registry, interaction, sessions = _registry(native_workspace)
    other = native_workspace / "other-session"
    other.mkdir(mode=0o700)
    switched = InstructionsManager(runtime_environment=RuntimeEnvironment(native_workspace, other))
    try:
        started = _call(
            registry,
            interaction,
            "run_command",
            command="printf 'first\\n'; sleep 2; printf 'last\\n'",
            cwd=str(native_workspace),
            wait_seconds=0,
        )
        assert started["ok"], started
        handle = started["result"]["handle"]
        output_handle = started["result"]["output_handles"]["stdout"]
        future = encode_content_cursor(output_handle, 999999)
        invalid = _call(
            registry,
            interaction,
            "read_command_output",
            handle=handle,
            stream="stdout",
            cursor=future,
        )
        assert invalid["problem"]["code"] == "command_output.invalid_read"
        with pytest.raises(SessionExecutionConflictError):
            sessions.new_session()
        denied = json.loads(
            registry.call(
                "read_command_output",
                json.dumps({"handle": handle, "stream": "stdout"}),
                interaction=interaction,
                instructions_manager=switched,
            )
        )
        assert denied["problem"]["code"] == "command_job.unavailable"
        assert _call(registry, interaction, "inspect_command_jobs")["result"]["jobs"]
        final = _poll_until_complete(registry, interaction, handle)
        assert "stdout" not in final["result"].get("metadata", {})
        assert final["main_process"] == "exited"
        assert final["descendant_cleanup"] == "uncertain"
        assert final["stdout"]["content"] == "first\nlast\n"
        observed = final["stdout"]["content"]
        end_cursor = encode_content_cursor(output_handle, len(observed.encode("utf-8")))
        assert _read_output_page(registry, interaction, handle, cursor=end_cursor)["content"] == ""
        sessions.new_session()
    finally:
        registry.close()


def test_native_failed_command_exposes_a_callable_cached_output_continuation(native_workspace):
    """A failed command's published tool call reads the rest of its completed output."""
    registry, interaction, _sessions = _registry(native_workspace)
    count = macos_backend.constants.MAX_TOOL_CONTENT_BYTES
    expected = " " * count + "failure detail"
    try:
        failed = _call(
            registry,
            interaction,
            "run_command",
            command=f"printf '%{count}s%s' '' 'failure detail'; exit 7",
            cwd=str(native_workspace),
        )
        assert failed["problem"]["code"] == "process.nonzero_exit"
        output = failed["problem"]["metadata"]["stdout"]
        continuation = output["next_tool_call"]
        page = _call(registry, interaction, continuation["name"], **continuation["arguments"])
        assert page["ok"], page
        assert output["content"] + page["result"]["content"] == expected
        assert "next_cursor" not in page["result"]
    finally:
        registry.close()


def test_native_command_cancel_reports_uncertain_detached_cleanup(native_workspace):
    """Cancellation stops the owned command without claiming detached-child proof."""
    registry, interaction, _sessions = _registry(native_workspace)
    try:
        started = _call(
            registry,
            interaction,
            "run_command",
            command="printf 'ready\n'; sleep 20",
            cwd=str(native_workspace),
            wait_seconds=0,
        )
        assert started["ok"], started
        handle = started["result"]["handle"]
        deadline = time.monotonic() + 10
        cursor = None
        output = ""
        while "ready" not in output and time.monotonic() < deadline:
            page = _read_output_page(registry, interaction, handle, cursor=cursor)
            output += page["content"]
            cursor = page.get("next_cursor")
            time.sleep(0.02)
        assert "ready" in output, "The native command did not publish live output before cancel."
        cancelled = _call(registry, interaction, "cancel_command_job", handle=handle)
        assert cancelled["result"]["cancel_requested"]
        final = _poll_until_complete(registry, interaction, handle)
        assert final["result"]["code"] == "sandbox.cancelled"
        assert final["result"]["metadata"]["lifecycle"]["cancel_requested"]
        assert final["result"]["metadata"]["lifecycle"]["descendant_cleanup"] == "uncertain"
    finally:
        registry.close()


def test_native_command_default_wait_continues_same_process(native_workspace):
    """The configured initial wait yields a cursor while the original process continues."""
    registry, interaction, _sessions = _registry(
        native_workspace,
        command_default_initial_wait_seconds=0.1,
        command_job_runtime_limit_seconds=10,
    )
    try:
        started = _call(
            registry,
            interaction,
            "run_command",
            command="printf x >> continuation-count; sleep 1; printf finished",
            cwd=str(native_workspace),
            write_roots=[str(native_workspace)],
        )
        assert started["ok"], started
        job = started["result"]
        assert job["result"] is None and job["elapsed_seconds"] >= 0
        result, output = _terminal_result(registry, interaction, job["handle"])
        assert result["exit_code"] == 0
        assert "finished" in output
        assert (native_workspace / "continuation-count").read_text() == "x"
    finally:
        registry.close()


def test_native_command_hard_deadline_after_continuation(native_workspace):
    """A short wait preserves one command until its separate hard deadline stops it."""
    registry, interaction, _sessions = _registry(
        native_workspace,
        command_default_initial_wait_seconds=0.1,
        command_job_runtime_limit_seconds=2,
    )
    marker = native_workspace / "deadline-count"
    try:
        started = _call(
            registry,
            interaction,
            "run_command",
            command="printf x >> deadline-count; printf ready; sleep 20",
            cwd=str(native_workspace),
            write_roots=[str(native_workspace)],
        )
        assert started["ok"], started
        handle = started["result"]["handle"]
        initial_elapsed = started["result"]["elapsed_seconds"]
        assert initial_elapsed is not None and initial_elapsed >= 0
        final = _poll_until_complete(registry, interaction, handle)
        assert final["result"]["code"] == "sandbox.timed_out"
        assert final["result"]["metadata"]["lifecycle"]["descendant_cleanup"] == "uncertain"
        assert "ready" in _read_complete_output(registry, interaction, handle)
        assert marker.read_text() == "x"
    finally:
        registry.close()


def test_native_async_public_dispatch_retains_job_owner(native_workspace):
    """The asynchronous registry passes its selected session lease into a detached worker."""
    registry, interaction, _sessions = _registry(native_workspace)

    async def exercise() -> None:
        """Keep the owner loop alive until the original native attempt settles."""
        started = json.loads(
            await registry.call_async(
                "run_command",
                json.dumps(
                    {
                        "command": "printf async; sleep 0.3; printf done",
                        "cwd": str(native_workspace),
                        "wait_seconds": 0,
                    }
                ),
                interaction=interaction,
            )
        )
        assert started["ok"], started
        handle = started["result"]["handle"]
        for _ in range(8):
            page = json.loads(
                await registry.call_async(
                    "poll_command_job",
                    json.dumps({"handle": handle, "wait_seconds": 5}),
                    interaction=interaction,
                )
            )
            assert page["ok"], page
            if page["result"]["status"] == "complete":
                assert page["result"]["stdout"]["content"] == "asyncdone"
                return
        pytest.fail("The asynchronous native job did not settle.")

    try:
        asyncio.run(exercise())
    finally:
        registry.close()


def test_native_terminal_job_accepts_input_and_inner_pty(native_workspace):
    """An owned macOS terminal accepts input and permits a confined inner PTY."""
    registry, interaction, _sessions = _registry(native_workspace, terminal=True)
    runtime = Path(sys.base_prefix).resolve(strict=True)
    interpreter = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    command = (
        shlex.quote(str(interpreter))
        + " -B -c "
        + shlex.quote(
            "import os; print('TTY', os.isatty(0), os.tcgetpgrp(0) == os.getpgrp()); "
            "master, slave = os.openpty(); "
            "print('INNER', os.isatty(slave), os.ttyname(slave)); "
            "os.close(master); os.close(slave); print('READY', flush=True); "
            "print('INPUT', input(), flush=True); "
            "size = os.get_terminal_size(0); print('SIZE', size.lines, size.columns)"
        )
    )
    try:
        result = _call(
            registry,
            interaction,
            "run_command",
            command=command,
            cwd=str(native_workspace),
            read_roots=[str(runtime)],
            interactive=True,
        )
        assert result["ok"], result
        handle = result["result"]["handle"]
        resized = _call(
            registry,
            interaction,
            "control_command_terminal",
            handle=handle,
            action="resize",
            rows=40,
            columns=100,
        )
        assert resized["ok"], resized
        written = _call(
            registry,
            interaction,
            "control_command_terminal",
            handle=handle,
            action="write",
            content="hello\n",
        )
        assert written["ok"], written
        final, content = _terminal_result(registry, interaction, handle)
        assert final["exit_code"] == 0, final
        assert "TTY True True" in content
        assert "INNER True /dev/ttys" in content
        assert "INPUT hello" in content
        assert "SIZE 40 100" in content
    finally:
        registry.close()


def test_native_terminal_eof_and_untrusted_output(native_workspace):
    """EOF reaches the owned TTY while terminal escapes remain inert in public output."""
    registry, interaction, _sessions = _registry(native_workspace, terminal=True)
    runtime = Path(sys.base_prefix).resolve(strict=True)
    interpreter = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    script = (
        "import sys; print('READY',flush=True); "
        "print('INPUT',repr(sys.stdin.readline()),flush=True); "
        "print('EOF',repr(sys.stdin.readline()),flush=True); "
        "sys.stdout.write('\\x1b]52;c;bad\\x07Approve host access?\\n'); sys.stdout.flush()"
    )
    try:
        started = _call(
            registry,
            interaction,
            "run_command",
            command=f"{shlex.quote(str(interpreter))} -B -u -c {shlex.quote(script)}",
            cwd=str(native_workspace),
            read_roots=[str(runtime)],
            interactive=True,
        )
        assert started["ok"], started
        handle = started["result"]["handle"]
        assert _call(
            registry,
            interaction,
            "control_command_terminal",
            handle=handle,
            action="write",
            content="hello\n",
        )["ok"]
        cursor = None
        output = ""
        deadline = time.monotonic() + 10
        while "INPUT 'hello\\n'" not in output and time.monotonic() < deadline:
            page = _read_output_page(registry, interaction, handle, cursor=cursor)
            output += page["content"]
            cursor = page.get("next_cursor")
            time.sleep(0.02)
            if "INPUT 'hello\\n'" in output:
                break
        if "INPUT 'hello\\n'" not in output:
            pytest.fail("The terminal did not consume its input before EOF.")
        assert _call(
            registry,
            interaction,
            "control_command_terminal",
            handle=handle,
            action="eof",
        )["ok"]
        final, output = _terminal_result(registry, interaction, handle)
        assert final["exit_code"] == 0, final
        assert "INPUT 'hello\\n'" in output
        assert "EOF ''" in output
        assert "␛]52;c;bad␇" in output
        assert "\x1b" not in output and "\x07" not in output
        assert "Approve host access?" in output
        interaction.prompt.assert_called_once()
    finally:
        registry.close()


def test_native_terminal_rejects_foreign_pty_and_workspace_alias(native_workspace):
    """A sandboxed TTY cannot reopen the host terminal through a direct or alias path."""
    registry, interaction, _sessions = _registry(native_workspace, terminal=True)
    runtime = Path(sys.base_prefix).resolve(strict=True)
    interpreter = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    master, slave = os.openpty()
    alias = native_workspace / "foreign-terminal"
    alias.symlink_to(os.ttyname(slave))
    script = (
        "import os\n"
        f"paths={(os.ttyname(slave), str(alias))!r}\n"
        "for path in paths:\n"
        " try:\n"
        "  fd=os.open(path,os.O_WRONLY|os.O_NONBLOCK);os.write(fd,b'injected');os.close(fd)\n"
        "  print('OPENED',path,flush=True)\n"
        " except OSError as error: print('DENIED',path,error.errno,flush=True)\n"
    )
    try:
        started = _call(
            registry,
            interaction,
            "run_command",
            command=f"{shlex.quote(str(interpreter))} -B -u -c {shlex.quote(script)}",
            cwd=str(native_workspace),
            read_roots=[str(runtime)],
            interactive=True,
        )
        assert started["ok"], started
        final, output = _terminal_result(registry, interaction, started["result"]["handle"])
        assert final["exit_code"] == 0, final
        assert output.count("DENIED") == 2 and "OPENED" not in output, output
        if select.select([master], [], [], 0)[0]:
            try:
                assert b"injected" not in os.read(master, 4096)
            except OSError as error:
                assert error.errno == errno.EIO
    finally:
        os.close(master)
        os.close(slave)
        registry.close()


def test_native_terminal_shell_interrupt_and_close(native_workspace):
    """A live shell uses the same TTY, while interrupt and close target only its job."""
    registry, interaction, _sessions = _registry(native_workspace, terminal=True)
    try:
        shell = _call(
            registry,
            interaction,
            "run_command",
            command="exec /bin/sh -i",
            cwd=str(native_workspace),
            interactive=True,
        )
        assert shell["ok"], shell
        handle = shell["result"]["handle"]
        assert _call(
            registry,
            interaction,
            "control_command_terminal",
            handle=handle,
            action="write",
            content="printf SHELL_READY\\n\nexit\n",
        )["ok"]
        final, output = _terminal_result(registry, interaction, handle)
        assert final["exit_code"] == 0, final
        assert "SHELL_READY" in output

        runtime = Path(sys.base_prefix).resolve(strict=True)
        interpreter = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
        script = (
            "import signal; "
            "signal.signal(signal.SIGINT,lambda *_:print('INTERRUPTED',flush=True)); "
            "print('WAITING',flush=True); signal.pause()"
        )
        waiting = _call(
            registry,
            interaction,
            "run_command",
            command=f"{shlex.quote(str(interpreter))} -B -u -c {shlex.quote(script)}",
            cwd=str(native_workspace),
            read_roots=[str(runtime)],
            interactive=True,
        )
        assert waiting["ok"], waiting
        interrupt_handle = waiting["result"]["handle"]
        deadline = time.monotonic() + 10
        output = ""
        cursor = None
        while "WAITING" not in output and time.monotonic() < deadline:
            page = _read_output_page(registry, interaction, interrupt_handle, cursor=cursor)
            output += page["content"]
            cursor = page.get("next_cursor")
            time.sleep(0.02)
        assert "WAITING" in output, output
        interrupted = _call(
            registry,
            interaction,
            "control_command_terminal",
            handle=interrupt_handle,
            action="interrupt",
        )
        assert interrupted["ok"], interrupted
        final, output = _terminal_result(registry, interaction, interrupt_handle)
        assert final["exit_code"] == 0, final
        assert "INTERRUPTED" in output

        running = _call(
            registry,
            interaction,
            "run_command",
            command="printf CLOSE_READY; sleep 30",
            cwd=str(native_workspace),
            interactive=True,
        )
        assert running["ok"], running
        closed = _call(
            registry,
            interaction,
            "control_command_terminal",
            handle=running["result"]["handle"],
            action="close",
        )
        assert closed["ok"] and closed["result"]["closed"], closed
        assert closed["result"]["cancel_requested"]
    finally:
        registry.close()


def test_native_terminal_disconnect_stops_owned_main_process(native_workspace):
    """Closing the client retires an active terminal without leaving its main process alive."""
    registry, interaction, _sessions = _registry(native_workspace, terminal=True)
    marker = native_workspace / "terminal-main-pid"
    pid = None
    try:
        started = _call(
            registry,
            interaction,
            "run_command",
            command=f"echo $$ > {shlex.quote(str(marker))}; printf READY; exec sleep 20",
            cwd=str(native_workspace),
            write_roots=[str(native_workspace)],
            interactive=True,
        )
        assert started["ok"], started
        handle = started["result"]["handle"]
        cursor = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if marker.exists() and marker.stat().st_size:
                break
            page = _read_output_page(registry, interaction, handle, cursor=cursor)
            cursor = page.get("next_cursor")
            time.sleep(0.02)
        assert marker.exists(), "The owned terminal did not publish its process identity."
        pid = int(marker.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    finally:
        registry.close()
    assert pid is not None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("The terminal main process outlived client closure.")


def test_native_terminal_cannot_open_another_live_job(native_workspace, monkeypatch):
    """A second sandboxed job cannot write the first job's live terminal slave."""
    registry, interaction, _sessions = _registry(native_workspace, terminal=True)
    allocated = Queue()
    allocate_terminal = macos_backend._owned_terminal

    @contextmanager
    def observe_terminal():
        """Observe the actual slave from its owner before the confined job starts."""
        with allocate_terminal() as terminal:
            allocated.put(terminal.path)
            yield terminal

    monkeypatch.setattr(macos_backend, "_owned_terminal", observe_terminal)
    try:
        first = _call(
            registry,
            interaction,
            "run_command",
            command="read line; printf 'FIRST:%s' \"$line\"",
            cwd=str(native_workspace),
            interactive=True,
        )
        assert first["ok"], first
        handle = first["result"]["handle"]
        path = allocated.get(timeout=10)
        assert path.is_char_device()
        foreign = str(path)
        second = _call(
            registry,
            interaction,
            "run_command",
            command=(
                f"if ( : > {shlex.quote(foreign)} ) 2>/dev/null; "
                "then printf OTHER_OPENED; else printf OTHER_DENIED; fi"
            ),
            cwd=str(native_workspace),
            interactive=True,
        )
        assert second["ok"], second
        final, output = _terminal_result(registry, interaction, second["result"]["handle"])
        assert final["exit_code"] == 0 and "OTHER_DENIED" in output, (final, output)
        assert _call(
            registry,
            interaction,
            "control_command_terminal",
            handle=handle,
            action="write",
            content="done\n",
        )["ok"]
        final, output = _terminal_result(registry, interaction, handle)
        assert final["exit_code"] == 0, final
        assert "FIRST:done" in output
        assert "OTHER_OPENED" not in output
    finally:
        registry.close()


def test_native_terminal_keeps_host_signals_files_and_descriptors_separate(native_workspace):
    """PTY admission does not inherit bootstrap channels or grant host process authority."""
    registry, interaction, _sessions = _registry(native_workspace, terminal=True)
    runtime = Path(sys.base_prefix).resolve(strict=True)
    interpreter = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    private = native_workspace / ".ssh" / "terminal-secret"
    private.parent.mkdir(exist_ok=True)
    private.write_text("private-pty-canary", encoding="utf-8")
    host = subprocess.Popen(["/bin/sleep", "30"])
    script = (
        "import fcntl,os,signal,termios\n"
        "leaks=[]\n"
        "for fd in range(3,32):\n"
        " try: os.fstat(fd); leaks.append(fd)\n"
        " except OSError: pass\n"
        "print('LEAKS',leaks,flush=True)\n"
        f"try: open({str(private)!r},'rb').read(); print('PRIVATE_OPENED',flush=True)\n"
        "except OSError as error: print('PRIVATE_DENIED',error.errno,flush=True)\n"
        f"try: os.kill({host.pid},signal.SIGUSR1); print('SIGNAL_SENT',flush=True)\n"
        "except OSError as error: print('SIGNAL_DENIED',error.errno,flush=True)\n"
        "try: fcntl.ioctl(0,termios.TIOCSTI,b'x'); print('INJECTED',flush=True)\n"
        "except OSError as error: print('INJECT_DENIED',error.errno,flush=True)\n"
    )
    try:
        started = _call(
            registry,
            interaction,
            "run_command",
            command=f"{shlex.quote(str(interpreter))} -B -u -c {shlex.quote(script)}",
            cwd=str(native_workspace),
            read_roots=[str(runtime)],
            interactive=True,
        )
        assert started["ok"], started
        final, output = _terminal_result(registry, interaction, started["result"]["handle"])
        assert final["exit_code"] == 0, final
        assert "LEAKS []" in output, output
        assert "PRIVATE_DENIED" in output and "PRIVATE_OPENED" not in output
        assert "SIGNAL_DENIED" in output and "SIGNAL_SENT" not in output
        assert "INJECT_DENIED" in output and "INJECTED" not in output
        assert "private-pty-canary" not in output
        assert host.poll() is None
    finally:
        if host.poll() is None:
            host.terminate()
        host.wait(timeout=5)
        registry.close()


def test_native_terminal_host_peer_can_transfer_owned_pty_descriptor(
    native_workspace, local_socket_parent
):
    """An unsandboxed host peer can deliberately delegate its owned PTY descriptor."""
    registry, interaction, _sessions = _registry(native_workspace, terminal=True)
    runtime = Path(sys.base_prefix).resolve(strict=True)
    interpreter = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    marker = native_workspace / "terminal-socket-path"
    master, slave = os.openpty()
    script = (
        "import array,os,socket\n"
        "from pathlib import Path\n"
        "path=os.environ['LOOP_LOCAL_SOCKET_DIR']+'/terminal-fd.sock'\n"
        "server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)\n"
        "server.bind(path);server.listen(1)\n"
        f"Path({str(marker)!r}).write_text(path)\n"
        "print('LISTENING',flush=True)\n"
        "peer,_=server.accept()\n"
        "try:\n"
        " data,ancillary,_,_=peer.recvmsg(1,socket.CMSG_SPACE(4))\n"
        " received=[]\n"
        " for level,kind,payload in ancillary:\n"
        "  if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:\n"
        "   received.extend(array.array('i',payload))\n"
        " if received:\n"
        "  try: os.write(received[0],b'host-pty-injected');print('FD_WRITTEN',flush=True)\n"
        "  except OSError as error: print('FD_DENIED',error.errno,flush=True)\n"
        "  finally: [os.close(fd) for fd in received]\n"
        " else: print('FD_REJECTED',flush=True)\n"
        "except OSError as error: print('TRANSFER_DENIED',error.errno,flush=True)\n"
        "peer.close();server.close()\n"
    )
    try:
        started = _call(
            registry,
            interaction,
            "run_command",
            command=f"{shlex.quote(str(interpreter))} -B -u -c {shlex.quote(script)}",
            cwd=str(native_workspace),
            read_roots=[str(runtime)],
            write_roots=[str(native_workspace)],
            local_ipc=True,
            interactive=True,
        )
        assert started["ok"], started
        handle = started["result"]["handle"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if marker.exists() and marker.stat().st_size:
                break
            _read_output_page(registry, interaction, handle)
            time.sleep(0.02)
        assert marker.exists(), json.dumps(
            _call(registry, interaction, "poll_command_job", handle=handle, wait_seconds=0),
            indent=2,
        )
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(5)
            peer.connect(marker.read_text(encoding="utf-8"))
            peer.sendmsg([b"F"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array("i", [slave]))])
        final, output = _terminal_result(registry, interaction, handle)
        assert final["exit_code"] == 0, final
        assert "FD_WRITTEN" in output
        assert select.select([master], [], [], 1)[0]
        assert b"host-pty-injected" in os.read(master, 4096)
    finally:
        os.close(master)
        os.close(slave)
        registry.close()


@pytest.mark.skipif(
    os.environ.get("LOOP_NATIVE_LONG_COMMAND_TEST") != "extended-native-check",
    reason="Run the five-minute native duration control explicitly.",
)
@pytest.mark.parametrize("noisy", [False, True])
def test_native_command_continues_past_previous_default_deadline(native_workspace, noisy):
    """One retained command completes after five minutes without replay."""
    registry, interaction, _sessions = _registry(native_workspace)
    command = (
        "set -e; printf x >> count; "
        + ("printf begin; " if noisy else "")
        + "sleep 301; printf end"
    )
    try:
        started = _call(
            registry,
            interaction,
            "run_command",
            command=command,
            cwd=str(native_workspace),
            wait_seconds=1,
            write_roots=[str(native_workspace)],
        )
        assert started["ok"], started
        handle = started["result"]["handle"]
        deadline = time.monotonic() + 330
        final = None
        while time.monotonic() < deadline:
            page = _call(
                registry,
                interaction,
                "poll_command_job",
                handle=handle,
                wait_seconds=30,
            )
            assert page["ok"], page
            if page["result"]["status"] == "complete":
                final = page["result"]
                break
        assert final is not None and final["exit_code"] == 0, final
        assert _read_complete_output(registry, interaction, handle).endswith("end")
        assert (native_workspace / "count").read_text() == "x"
    finally:
        registry.close()


@pytest.mark.skipif(
    os.environ.get("LOOP_NATIVE_LONG_COMMAND_TEST") != "extended-native-check",
    reason="Run the five-minute native duration control explicitly.",
)
def test_native_terminal_survives_previous_default_deadline(native_workspace):
    """One interactive handle stays live beyond five minutes without replay or model polling."""
    registry, interaction, _sessions = _registry(native_workspace, terminal=True)
    try:
        started = _call(
            registry,
            interaction,
            "run_command",
            command="printf TERM_BEGIN; sleep 301; printf TERM_END",
            cwd=str(native_workspace),
            interactive=True,
        )
        assert started["ok"], started
        handle = started["result"]["handle"]
        final = None
        deadline = time.monotonic() + 330
        while time.monotonic() < deadline:
            page = _call(
                registry,
                interaction,
                "poll_command_job",
                handle=handle,
                wait_seconds=30,
            )
            assert page["ok"], page
            if page["result"]["status"] == "complete":
                final = page["result"]
                break
        assert final is not None and final["exit_code"] == 0, final
        output = _read_complete_output(registry, interaction, handle)
        assert "TERM_BEGIN" in output and "TERM_END" in output
    finally:
        registry.close()
