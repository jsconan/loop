"""Tests for the built-in system tools."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from typing import ClassVar
from unittest.mock import MagicMock, call

import pytest

from loop import (
    BUILTIN_TOOLS,
    ConsoleInteraction,
    PermissionConfiguration,
    PermissionManager,
    PolicyLimits,
    ToolContext,
    ToolRegistry,
)
from loop.constants import MAX_OUTPUT_CHARS
from loop.instructions import InstructionsManager, RuntimeEnvironment
from loop.tools.system import run_command as run_command_tool
from loop.utils import cached_path

# pylint: disable=unused-argument, redefined-outer-name

tool_registry: ToolRegistry


@pytest.fixture(autouse=True)
def fresh_tool_registry():
    """Provide an isolated built-in registry for each system-tool case."""
    global tool_registry  # pylint: disable=global-statement
    tool_registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(
            configuration=PermissionConfiguration(limits=PolicyLimits(allow_host_processes=True))
        ),
    )


def run_command(command, cwd="."):
    """Dispatch the context-aware command tool."""
    output = tool_registry.call(
        "run_command",
        json.dumps({"command": command, "cwd": cwd}),
        interaction=ConsoleInteraction(),
    )
    payload = json.loads(output)
    return payload["result"] if payload["ok"] else output


def problem(output: str):
    """Return the problem from a failed tool result envelope."""
    return json.loads(output)["problem"]


class ImmediateThread:
    """Run a thread target synchronously so stream behavior is deterministic."""

    instances: ClassVar[list[ImmediateThread]] = []

    def __init__(self, *, target, args, daemon):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.joined = False
        self.instances.append(self)

    def start(self):
        """Execute the target as soon as the thread is started."""
        self.target(*self.args)

    def join(self, timeout=None):
        """Record that command cleanup joined the reader."""
        self.joined = True

    def is_alive(self):
        """Report synchronous completion after start returns."""
        return False


@pytest.fixture
def authorized(monkeypatch):
    """Confirm command execution while retaining real process and thread behavior."""
    monkeypatch.setattr(PermissionManager, "request_permission", MagicMock(return_value=True))


@pytest.fixture
def confirmed(monkeypatch, authorized):
    """Confirm command execution and make stream readers synchronous."""
    ImmediateThread.instances = []
    monkeypatch.setattr("loop.tools.system.threading.Thread", ImmediateThread)


def python_command(script, *arguments):
    """Build a restricted command line for the current Python interpreter."""
    return " ".join(shlex.quote(value) for value in (sys.executable, "-c", script, *arguments))


def process_exists(pid):
    """Report whether an operating-system process still has the given identifier."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def make_process(*, stdout=("",), stderr=("",), returncode=0):
    """Create a context-managed process double with configured streams."""
    process = MagicMock()
    process.__enter__.return_value = process
    process.stdout.read.side_effect = stdout
    process.stderr.read.side_effect = stderr
    process.wait.return_value = returncode
    process.poll.return_value = returncode
    return process


def test_run_command_requires_an_affirmative_confirmation(monkeypatch):
    """A rejected confirmation cancels command execution."""
    popen = MagicMock()
    confirm = MagicMock(return_value=False)
    monkeypatch.setattr(PermissionManager, "request_permission", confirm)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", popen)

    assert problem(run_command("echo hello"))["code"] == "tool.denied"
    confirm.assert_called_once()
    assert "process.execute" in confirm.call_args.args[0]
    assert "echo hello" in confirm.call_args.args[0]
    popen.assert_not_called()


def test_run_command_preserves_both_streams_and_passes_safe_process_options(monkeypatch, confirmed):
    """A successful command preserves stdout whitespace and useful stderr warnings."""
    process = make_process(stdout=("hello world\n", ""), stderr=("warning\n", ""))
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", popen)

    result = run_command("echo hello")
    assert result["stdout"]["content"] == "hello world\n"
    assert result["stderr"]["content"] == "warning\n"
    assert result["stdout"]["capture_complete"] is True
    popen.assert_called_once()
    args, kwargs = popen.call_args
    assert args == (["echo", "hello"],)
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == os.path.realpath(".")
    assert set(kwargs["env"]) <= {"PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"}
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert kwargs["start_new_session"] is (os.name == "posix")
    assert kwargs["creationflags"] == 0
    assert all(reader.joined for reader in ImmediateThread.instances)


@pytest.mark.parametrize(
    "command",
    [
        "   ",
        'echo "unterminated',
        "echo trailing\\",
        'echo "value\\',
        "echo ok | grep ok",
        "echo ok > output",
        "echo ok && date",
    ],
)
def test_run_command_rejects_empty_malformed_or_shell_syntax(command):
    """Command planning rejects empty, malformed, and shell-language command text."""
    result = tool_registry.call(
        "run_command",
        json.dumps({"command": command}),
        interaction=ConsoleInteraction(),
    )

    assert problem(result)["code"] == "tool.planning_failed"


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        ("printf '%s' 'a|b'", ["printf", "%s", "a|b"]),
        (r"printf '%s' a\|b", ["printf", "%s", "a|b"]),
        ('printf "%s" "price: $5"', ["printf", "%s", "price: $5"]),
        ('printf "%s" "a\\"b"', ["printf", "%s", 'a"b']),
    ],
)
def test_run_command_preserves_quoted_or_escaped_shell_characters_as_argument_data(
    monkeypatch, confirmed, command, argv
):
    """Quoted and escaped shell characters remain literal values in the planned argv."""
    process = make_process()
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", popen)

    assert run_command(command)["stdout"]["content"] == ""

    assert popen.call_args.args == (argv,)


def test_run_command_fails_closed_without_an_authorized_process_target():
    """Direct execution cannot parse or execute an unplanned command."""
    result = run_command_tool(ToolContext(ConsoleInteraction(), "run_command"), "echo hello")

    assert result.code == "process.execution_failed"
    assert result.detail == "Authorized process target is missing."


def test_run_command_resolves_virtual_cwd_and_redacts_known_host_roots(
    monkeypatch, confirmed, tmp_path
):
    """Virtual working directories execute locally without returning their backing root."""
    process = make_process(stdout=(f"{tmp_path}/created.txt\n", ""))
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", popen)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "temporary")
    )

    result = tool_registry.call(
        "run_command",
        json.dumps({"command": "pwd", "cwd": "/workspace"}),
        interaction=ConsoleInteraction(),
        instructions_manager=instructions,
    )

    assert json.loads(result)["result"]["stdout"]["content"] == "/workspace/created.txt\n"
    assert popen.call_args.kwargs["cwd"] == str(tmp_path.resolve())


def test_run_command_resolves_virtual_path_arguments_before_execution(
    monkeypatch, confirmed, tmp_path
):
    """Virtual command arguments execute against the corresponding local paths."""
    process = make_process()
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", popen)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "temporary")
    )

    result = tool_registry.call(
        "run_command",
        json.dumps({"command": "git -C /workspace status", "cwd": "/workspace"}),
        interaction=ConsoleInteraction(),
        instructions_manager=instructions,
    )

    assert json.loads(result)["result"]["exit_code"] == 0
    assert popen.call_args.args == (["git", "-C", str(tmp_path.resolve()), "status"],)


def test_successful_run_command_invalidates_instruction_scope(monkeypatch, tmp_path, confirmed):
    """Successful shell operations request a conservative instruction refresh."""
    manager = MagicMock()
    manager.virtual_paths.resolve.side_effect = lambda value: value
    manager.virtual_paths.resolve_command.side_effect = lambda value: value
    manager.virtual_paths.metadata.side_effect = lambda value: value
    manager.virtual_paths.redact.side_effect = lambda value: value
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", subprocess.Popen)

    result = tool_registry.call(
        "run_command",
        json.dumps({"command": "printf ok", "cwd": str(tmp_path)}),
        interaction=ConsoleInteraction(),
        instructions_manager=manager,
    )

    assert json.loads(result)["result"]["stdout"]["content"] == "ok"
    manager.invalidate.assert_called_once_with(None)


def test_run_command_completes_a_normal_process_within_its_lifecycle_timeout(
    monkeypatch, authorized
):
    """A short real command completes normally under the shared lifecycle deadline."""
    tool_registry.settings.command_timeout = 0.5

    assert run_command(python_command("print('complete')"))["stdout"]["content"] == "complete\n"


def test_run_command_reports_exit_code_stdout_and_stderr(monkeypatch, confirmed):
    """A failed command exposes its exit code and both captured streams."""
    process = make_process(
        stdout=("some output\n", ""), stderr=("command not found\n", ""), returncode=127
    )
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))

    failure = problem(run_command("missing"))
    assert failure["code"] == "process.nonzero_exit"
    assert failure["metadata"]["exit_code"] == 127
    assert failure["metadata"]["stdout"]["content"] == "some output\n"
    assert failure["metadata"]["stderr"]["content"] == "command not found\n"


def test_run_command_caps_each_output_stream_while_draining_it(monkeypatch, confirmed):
    """Readers discard excess chunks but continue draining through end of stream."""
    process = make_process(
        stdout=("x" * MAX_OUTPUT_CHARS, "discarded", ""),
        stderr=("y" * (MAX_OUTPUT_CHARS + 1), "also discarded", ""),
        returncode=2,
    )
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))

    result = run_command("verbose-command")

    failure = problem(result)
    assert failure["code"] == "process.nonzero_exit"
    assert failure["metadata"]["stdout"]["captured_bytes"] == MAX_OUTPUT_CHARS
    assert failure["metadata"]["stdout"]["discarded_characters"] == 9
    assert (
        cached_path(failure["metadata"]["stdout"]["handle"])[0].read_text()
        == "x" * MAX_OUTPUT_CHARS
    )
    assert failure["metadata"]["stdout"]["next_cursor"]
    assert failure["metadata"]["stderr"]["captured_bytes"] == MAX_OUTPUT_CHARS
    assert failure["metadata"]["stderr"]["discarded_characters"] == 15
    assert process.stdout.read.call_count == 3
    assert process.stderr.read.call_count == 3


def test_run_command_drains_large_stdout_and_stderr_without_deadlocking(monkeypatch, authorized):
    """Concurrent readers drain both full pipes while retaining their independent caps."""
    tool_registry.settings.command_timeout = 3
    script = (
        "import os,sys;"
        f"os.write(1,b'x'*{MAX_OUTPUT_CHARS + 8192});"
        f"os.write(2,b'y'*{MAX_OUTPUT_CHARS + 8192});"
        "sys.exit(7)"
    )

    failure = problem(run_command(python_command(script)))

    assert failure["code"] == "process.nonzero_exit"
    assert failure["metadata"]["exit_code"] == 7
    for stream in ("stdout", "stderr"):
        assert failure["metadata"][stream]["captured_bytes"] == MAX_OUTPUT_CHARS
        assert failure["metadata"][stream]["discarded_characters"] == 8192
        assert failure["metadata"][stream]["truncated"] is True


def test_run_command_replaces_undecodable_output(monkeypatch, authorized):
    """Invalid UTF-8 output is represented with replacement characters instead of failing."""
    tool_registry.settings.command_timeout = 0.5

    assert (
        run_command(python_command("import os; os.write(1, b'ok\\xff')"))["stdout"]["content"]
        == "ok�"
    )


def test_run_command_surfaces_reader_failures(monkeypatch, confirmed):
    """A pipe read failure interrupts a live command and becomes an execution problem."""
    process = make_process(stdout=(OSError("pipe read failed"),))
    process.poll.return_value = None
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock())

    failure = problem(run_command("broken-reader"))

    assert failure["code"] == "process.execution_failed"
    assert failure["detail"] == "pipe read failed"
    assert process.wait.call_count == 1


def test_run_command_surfaces_reader_failures_observed_after_process_exit(monkeypatch, confirmed):
    """A reader failure racing with process exit still becomes an execution problem."""
    process = make_process(stdout=(OSError("late pipe read failed"),))

    class JoinThread(ImmediateThread):
        """Defer each reader target until its join operation."""

        def start(self):
            """Leave the reader pending until join."""

        def join(self, timeout=None):
            """Execute the reader target while modeling reader completion."""
            if not self.joined:
                self.target(*self.args)
                self.joined = True

    JoinThread.instances = []
    monkeypatch.setattr("loop.tools.system.threading.Thread", JoinThread)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock())

    failure = problem(run_command("late-broken-reader"))

    assert failure["code"] == "process.execution_failed"
    assert failure["detail"] == "late pipe read failed"


def test_run_command_kills_a_posix_process_group_after_timeout(monkeypatch, confirmed):
    """Timeout kills the entire POSIX process group before reporting it."""
    process = make_process()
    process.pid = 123
    process.wait.side_effect = subprocess.TimeoutExpired("sleep", 30)
    process.poll.return_value = 0
    killpg = MagicMock()
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)
    tool_registry.settings.command_timeout = 0.01

    assert problem(run_command("sleep 60"))["code"] == "process.timeout"
    assert len(process.wait.call_args_list) >= 2
    assert all(call_.kwargs["timeout"] >= 0 for call_ in process.wait.call_args_list)
    killpg.assert_called_once_with(123, 9)


def test_run_command_bounds_reaping_after_timeout(monkeypatch, confirmed):
    """A child that is not promptly reaped cannot extend timeout cleanup indefinitely."""
    process = make_process()
    process.wait.side_effect = subprocess.TimeoutExpired("sleep", 30)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock())
    tool_registry.settings.command_timeout = 0.01

    assert problem(run_command("sleep 60"))["code"] == "process.timeout"
    assert process.wait.call_count >= 2


def test_run_command_interrupts_pipe_descriptors_held_by_stuck_readers(monkeypatch, confirmed):
    """Cleanup interrupts stuck pipe reads without contending on text-stream locks."""
    process = make_process()
    process.wait.side_effect = subprocess.TimeoutExpired("sleep", 30)
    process.stdout.fileno.return_value = 10
    process.stderr.fileno.side_effect = OSError("already closed")
    close_descriptor = MagicMock()

    class StuckThread(ImmediateThread):
        """Model a reader that remains blocked after bounded joins."""

        def start(self):
            """Leave the modeled reader pending."""

        def is_alive(self):
            """Report that the modeled reader remains blocked."""
            return True

    StuckThread.instances = []
    monkeypatch.setattr("loop.tools.system.threading.Thread", StuckThread)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock())
    monkeypatch.setattr("loop.tools.system.os.close", close_descriptor)
    tool_registry.settings.command_timeout = 0.01

    assert problem(run_command("sleep 60"))["code"] == "process.timeout"
    assert call(10) in close_descriptor.call_args_list
    process.stdout.close.assert_not_called()
    process.stderr.close.assert_not_called()


def test_run_command_times_out_a_long_running_direct_child(monkeypatch, authorized):
    """The lifecycle deadline terminates a direct child that does not exit in time."""
    tool_registry.settings.command_timeout = 0.2
    started = time.monotonic()

    failure = problem(
        run_command(python_command("import time; print('partial', flush=True); time.sleep(30)"))
    )

    assert failure["code"] == "process.timeout"
    assert failure["detail"] == "Command did not complete within 0.2 seconds."
    assert failure["metadata"]["stdout"]["content"] == "partial\n"
    assert time.monotonic() - started < 0.5


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group behavior")
def test_run_command_times_out_and_cleans_a_descendant_holding_output_pipes(
    monkeypatch, tmp_path, authorized
):
    """An exited leader cannot let a pipe-owning descendant outlive the lifecycle deadline."""
    tool_registry.settings.command_timeout = 0.3
    pid_path = tmp_path / "descendant.pid"
    script = (
        "import pathlib,subprocess,sys;"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid),encoding='utf-8')"
    )
    started = time.monotonic()

    failure = problem(run_command(python_command(script, str(pid_path)), cwd=str(tmp_path)))

    assert failure["code"] == "process.timeout"
    assert time.monotonic() - started < 0.6
    descendant_pid = int(pid_path.read_text(encoding="utf-8"))
    for _ in range(50):
        if not process_exists(descendant_pid):
            break
        time.sleep(0.01)
    assert not process_exists(descendant_pid)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group behavior")
def test_run_command_cleanup_does_not_terminate_an_unrelated_process(monkeypatch, authorized):
    """Timeout cleanup remains scoped to the isolated command process group."""
    tool_registry.settings.command_timeout = 0.2
    unrelated = subprocess.Popen(  # pylint: disable=consider-using-with
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        assert problem(run_command(python_command("import time; time.sleep(30)")))["code"] == (
            "process.timeout"
        )
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=1)


def test_run_command_ignores_a_process_that_disappears_during_posix_cleanup(monkeypatch, confirmed):
    """A process-group lookup race does not replace the timeout result."""
    process = make_process()
    process.wait.side_effect = subprocess.TimeoutExpired("sleep", 30)
    process.poll.return_value = 0
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock(side_effect=ProcessLookupError))
    tool_registry.settings.command_timeout = 0.01

    assert problem(run_command("sleep 60"))["code"] == "process.timeout"


def test_run_command_kills_only_the_process_on_non_posix_systems(monkeypatch, confirmed):
    """Non-POSIX timeout handling uses the portable process kill method."""
    process = make_process()
    process.wait.side_effect = subprocess.TimeoutExpired("sleep", 30)
    process.poll.return_value = 0
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.name", "nt")
    tool_registry.settings.command_timeout = 0.01

    assert problem(run_command("sleep 60"))["code"] == "process.timeout"
    process.kill.assert_called_once_with()


def test_run_command_cleans_up_when_wait_raises(monkeypatch, confirmed):
    """Unexpected wait errors still kill the process and join stream readers."""
    process = make_process()
    process.wait.side_effect = [RuntimeError("wait failed"), 0]
    process.poll.return_value = None
    killpg = MagicMock()
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)

    assert problem(run_command("broken"))["detail"] == "wait failed"
    killpg.assert_called_once_with(process.pid, 9)
    assert all(reader.joined for reader in ImmediateThread.instances)


def test_run_command_cleans_up_readers_that_started_before_start_failure(monkeypatch, confirmed):
    """A partially started reader set is joined and the live process is killed."""
    process = make_process()
    process.poll.return_value = None

    class FailingSecondThread(ImmediateThread):
        """Fail the second thread start to simulate a reader start failure."""

        starts = 0

        def start(self):
            type(self).starts += 1
            if self.starts == 2:
                raise RuntimeError("thread failed")
            super().start()

    FailingSecondThread.instances = []
    monkeypatch.setattr("loop.tools.system.threading.Thread", FailingSecondThread)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock())

    assert problem(run_command("broken"))["detail"] == "thread failed"
    assert FailingSecondThread.instances[0].joined
    assert not FailingSecondThread.instances[1].joined


def test_run_command_rejects_missing_process_pipe_handles(monkeypatch, confirmed):
    """Missing pipe handles fail explicitly and still clean up the created process."""
    process = make_process()
    process.stdout = None
    killpg = MagicMock()
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)

    failure = problem(run_command("broken-pipes"))

    assert failure["code"] == "process.execution_failed"
    assert failure["detail"] == "Command process did not expose its output streams."
    process.stderr.close.assert_called_once_with()
    killpg.assert_called_once()


def test_run_command_reports_process_creation_errors(monkeypatch, confirmed):
    """Process creation failures become readable tool results."""
    monkeypatch.setattr(
        "loop.tools.system.subprocess.Popen", MagicMock(side_effect=PermissionError("denied"))
    )

    assert problem(run_command("restricted"))["detail"] == "denied"
