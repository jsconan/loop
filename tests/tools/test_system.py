"""Tests for the built-in system tools."""

import json
import os
import subprocess
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
from loop.tools.system import run_command as run_command_tool

# pylint: disable=unused-argument, redefined-outer-name

tool_registry = ToolRegistry(BUILTIN_TOOLS)


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


def run_command(command):
    """Dispatch the context-aware command tool."""
    return tool_registry.call(
        "run_command",
        json.dumps({"command": command}),
        interaction=ConsoleInteraction(),
    )


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

    def join(self):
        """Record that command cleanup joined the reader."""
        self.joined = True


@pytest.fixture
def confirmed(monkeypatch):
    """Confirm command execution and make stream readers synchronous."""
    ImmediateThread.instances = []
    monkeypatch.setattr(ConsoleInteraction, "confirm", MagicMock(return_value=True))
    monkeypatch.setattr("loop.tools.system.threading.Thread", ImmediateThread)


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
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", popen)

    assert '"error": "tool_call_denied"' in run_command("echo hello")
    confirm.assert_called_once()
    assert "process.execute" in confirm.call_args.args[0]
    assert "echo hello" in confirm.call_args.args[0]
    popen.assert_not_called()


def test_run_command_returns_stripped_stdout_and_passes_safe_process_options(
    monkeypatch, confirmed
):
    """A successful command returns stdout and configures all process pipes."""
    process = make_process(stdout=("hello world\n", ""))
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", popen)

    assert run_command("echo hello") == "hello world"
    popen.assert_called_once()
    args, kwargs = popen.call_args
    assert args == (["echo", "hello"],)
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == os.path.realpath(".")
    assert set(kwargs["env"]) <= {"PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"}
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    assert kwargs["text"] is True
    assert kwargs["start_new_session"] is (os.name == "posix")
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

    assert '"error": "operation_planning_failed"' in result


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

    assert run_command(command) == ""

    assert popen.call_args.args == (argv,)


def test_run_command_fails_closed_without_an_authorized_process_target():
    """Direct execution cannot parse or execute an unplanned command."""
    result = run_command_tool(ToolContext(ConsoleInteraction(), "run_command"), "echo hello")

    assert result == "Error running command: Authorized process target is missing."


def test_successful_run_command_invalidates_instruction_scope(monkeypatch, tmp_path, confirmed):
    """Successful shell operations request a conservative instruction refresh."""
    manager = MagicMock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", subprocess.Popen)

    result = tool_registry.call(
        "run_command",
        json.dumps({"command": "printf ok", "cwd": str(tmp_path)}),
        interaction=ConsoleInteraction(),
        instructions_manager=manager,
    )

    assert result == "ok"
    manager.invalidate.assert_called_once_with(None)


def test_run_command_reports_exit_code_stdout_and_stderr(monkeypatch, confirmed):
    """A failed command exposes its exit code and both captured streams."""
    process = make_process(
        stdout=("some output\n", ""), stderr=("command not found\n", ""), returncode=127
    )
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))

    assert run_command("missing") == (
        "Command failed with code 127. Output: some output Error: command not found"
    )


def test_run_command_caps_each_output_stream_while_draining_it(monkeypatch, confirmed):
    """Readers discard excess chunks but continue draining through end of stream."""
    process = make_process(
        stdout=("x" * MAX_OUTPUT_CHARS, "discarded", ""),
        stderr=("y" * (MAX_OUTPUT_CHARS + 1), "also discarded", ""),
        returncode=2,
    )
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))

    result = run_command("verbose-command")

    prefix = "Command failed with code 2. Output: "
    separator = " Error: "
    assert result.startswith(prefix + "x" * MAX_OUTPUT_CHARS + separator)
    assert result.endswith("y" * MAX_OUTPUT_CHARS)
    assert process.stdout.read.call_count == 3
    assert process.stderr.read.call_count == 3


def test_run_command_kills_a_posix_process_group_after_timeout(monkeypatch, confirmed):
    """Timeout kills the entire POSIX process group before reporting it."""
    process = make_process()
    process.pid = 123
    process.wait.side_effect = [subprocess.TimeoutExpired("sleep", 30), 0]
    process.poll.return_value = 0
    killpg = MagicMock()
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)

    assert run_command("sleep 60") == "Command timed out after 30 seconds."
    assert process.wait.call_args_list == [call(timeout=30), call()]
    killpg.assert_called_once_with(123, 9)


def test_run_command_ignores_a_process_that_disappears_during_posix_cleanup(monkeypatch, confirmed):
    """A process-group lookup race does not replace the timeout result."""
    process = make_process()
    process.wait.side_effect = [subprocess.TimeoutExpired("sleep", 30), 0]
    process.poll.return_value = 0
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock(side_effect=ProcessLookupError))

    assert run_command("sleep 60") == "Command timed out after 30 seconds."


def test_run_command_kills_only_the_process_on_non_posix_systems(monkeypatch, confirmed):
    """Non-POSIX timeout handling uses the portable process kill method."""
    process = make_process()
    process.wait.side_effect = [subprocess.TimeoutExpired("sleep", 30), 0]
    process.poll.return_value = 0
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.name", "nt")

    assert run_command("sleep 60") == "Command timed out after 30 seconds."
    process.kill.assert_called_once_with()


def test_run_command_cleans_up_when_wait_raises(monkeypatch, confirmed):
    """Unexpected wait errors still kill the process and join stream readers."""
    process = make_process()
    process.wait.side_effect = [RuntimeError("wait failed"), 0]
    process.poll.return_value = None
    killpg = MagicMock()
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)

    assert run_command("broken") == "Error running command: wait failed"
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

    assert run_command("broken") == "Error running command: thread failed"
    assert FailingSecondThread.instances[0].joined
    assert not FailingSecondThread.instances[1].joined


def test_run_command_reports_process_creation_errors(monkeypatch, confirmed):
    """Process creation failures become readable tool results."""
    monkeypatch.setattr(
        "loop.tools.system.subprocess.Popen", MagicMock(side_effect=PermissionError("denied"))
    )

    assert run_command("restricted") == "Error running command: denied"
