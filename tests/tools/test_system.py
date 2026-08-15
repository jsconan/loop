"""Tests for the built-in system tools."""

import json
import subprocess
from unittest.mock import MagicMock, call

import pytest

from loop import ConsoleInteraction, tool_registry
from loop.constants import MAX_OUTPUT_CHARS

# pylint: disable=unused-argument, redefined-outer-name


def run_command(command):
    """Dispatch the context-aware command tool."""
    return tool_registry.call(
        "run_command",
        json.dumps({"command": command}),
        interaction=ConsoleInteraction(),
    )


class ImmediateThread:
    """Run a thread target synchronously so stream behavior is deterministic."""

    instances = []

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
    confirm.assert_called_once_with(
        "⚙️ Agent wants to use 'run_command' for process.exec on 'echo hello'. Proceed?",
        default=False,
    )
    popen.assert_not_called()


def test_run_command_returns_stripped_stdout_and_passes_safe_process_options(
    monkeypatch, confirmed
):
    """A successful command returns stdout and configures all process pipes."""
    process = make_process(stdout=("hello world\n", ""))
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", popen)

    assert run_command("echo hello") == "hello world"
    popen.assert_called_once_with(
        "echo hello",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert all(reader.joined for reader in ImmediateThread.instances)


def test_successful_run_command_invalidates_instruction_scope(monkeypatch, tmp_path, confirmed):
    """Successful shell operations request a conservative instruction refresh."""
    manager = MagicMock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", subprocess.Popen)

    result = tool_registry.call(
        "run_command",
        json.dumps({"command": "printf ok"}),
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
    monkeypatch.setattr("loop.tools.system.os.killpg", killpg)

    assert run_command("sleep 60") == "Command timed out after 30 seconds."
    assert process.wait.call_args_list == [call(timeout=30), call()]
    killpg.assert_called_once_with(123, 9)


def test_run_command_ignores_a_process_that_disappears_during_posix_cleanup(monkeypatch, confirmed):
    """A process-group lookup race does not replace the timeout result."""
    process = make_process()
    process.wait.side_effect = [subprocess.TimeoutExpired("sleep", 30), 0]
    process.poll.return_value = 0
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.tools.system.os.killpg", MagicMock(side_effect=ProcessLookupError))

    assert run_command("sleep 60") == "Command timed out after 30 seconds."


def test_run_command_kills_only_the_process_on_non_posix_systems(monkeypatch, confirmed):
    """Non-POSIX timeout handling uses the portable process kill method."""
    process = make_process()
    process.wait.side_effect = [subprocess.TimeoutExpired("sleep", 30), 0]
    process.poll.return_value = 0
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.tools.system.os.name", "nt")

    assert run_command("sleep 60") == "Command timed out after 30 seconds."
    process.kill.assert_called_once_with()


def test_run_command_cleans_up_when_wait_raises(monkeypatch, confirmed):
    """Unexpected wait errors still kill the process and join stream readers."""
    process = make_process()
    process.wait.side_effect = [RuntimeError("wait failed"), 0]
    process.poll.return_value = None
    killpg = MagicMock()
    monkeypatch.setattr("loop.tools.system.subprocess.Popen", MagicMock(return_value=process))
    monkeypatch.setattr("loop.tools.system.os.killpg", killpg)

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
    monkeypatch.setattr("loop.tools.system.os.killpg", MagicMock())

    assert run_command("broken") == "Error running command: thread failed"
    assert FailingSecondThread.instances[0].joined
    assert not FailingSecondThread.instances[1].joined


def test_run_command_reports_process_creation_errors(monkeypatch, confirmed):
    """Process creation failures become readable tool results."""
    monkeypatch.setattr(
        "loop.tools.system.subprocess.Popen", MagicMock(side_effect=PermissionError("denied"))
    )

    assert run_command("restricted") == "Error running command: denied"
