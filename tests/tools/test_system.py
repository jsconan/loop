"""Tests for the built-in system tools."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, call

import pytest

from loop import (
    BUILTIN_TOOLS,
    Action,
    ApprovalChoice,
    ConsoleInteraction,
    Operation,
    PermissionConfiguration,
    PermissionManager,
    PolicyLimits,
    ProcessTarget,
    ToolContext,
    ToolRegistry,
)
from loop.constants import MAX_OUTPUT_CHARS
from loop.instructions import InstructionsManager, RuntimeEnvironment, Skill, SkillManager
from loop.sandbox import SandboxPlan, SandboxUnavailableError
from loop.tools.system import run_command as run_command_tool
from loop.utils import cached_path

# pylint: disable=unused-argument, redefined-outer-name

tool_registry: ToolRegistry
tool_instructions: InstructionsManager


@pytest.fixture(autouse=True)
def fresh_tool_registry(tmp_path, monkeypatch):
    """Provide an isolated built-in registry for each system-tool case."""
    command_directory = tmp_path / "commands"
    command_directory.mkdir()
    for name in ("echo", "git", "printf", "pwd", "sleep"):
        isolated_executable(command_directory, name)
    monkeypatch.setenv("PATH", str(command_directory))
    global tool_instructions, tool_registry  # pylint: disable=global-statement
    permissions = PermissionManager(tmp_path, configuration=PermissionConfiguration())
    tool_registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
    )
    tool_instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, permissions.temporary_directory)
    )


def run_command(command, cwd="."):
    """Dispatch the context-aware command tool."""
    resolved_cwd = "/workspace" if cwd == "." else tool_instructions.virtual_paths.display(cwd)
    output = tool_registry.call(
        "run_command",
        json.dumps({"command": command, "cwd": resolved_cwd}),
        interaction=ConsoleInteraction(),
        instructions_manager=tool_instructions,
    )
    payload = json.loads(output)
    return payload["result"] if payload["ok"] else output


def problem(output: str):
    """Return the problem from a failed tool result envelope."""
    return json.loads(output)["problem"]


def isolated_executable(tmp_path, name="tool"):
    """Create an executable fixture independent of host-installed commands."""
    path = tmp_path / name
    path.write_text("tool", encoding="utf-8")
    path.chmod(0o755)
    return path


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
    """Confirm command execution through a deterministic sandbox backend double."""
    monkeypatch.setattr(PermissionManager, "request_permission", MagicMock(return_value=True))
    backend = MagicMock(side_effect=lambda _plan, **_kwargs: make_process())
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", backend)
    return backend


@pytest.fixture
def confirmed(monkeypatch, authorized):
    """Confirm command execution and make stream readers synchronous."""
    ImmediateThread.instances = []
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
    monkeypatch.setattr(PermissionManager, "request_permission", confirm)
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", popen)

    assert problem(run_command("echo hello"))["code"] == "tool.denied"
    confirm.assert_called_once()
    assert "process.execute" in confirm.call_args.args[0]
    assert "echo hello" in confirm.call_args.args[0]
    popen.assert_not_called()


def test_run_command_preserves_both_streams_and_passes_safe_process_options(monkeypatch, confirmed):
    """A successful command preserves stdout whitespace and useful stderr warnings."""
    process = make_process(stdout=("hello world\n", ""), stderr=("warning\n", ""))
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", popen)

    result = run_command("echo hello")
    assert result["stdout"]["content"] == "hello world\n"
    assert result["stderr"]["content"] == "warning\n"
    assert result["stdout"]["capture_complete"] is True
    popen.assert_called_once()
    args, kwargs = popen.call_args
    plan = args[0]
    assert plan.argv[-2:] == (str(Path(shutil.which("echo")).resolve()), "hello")
    assert plan.cwd == tool_instructions.virtual_paths.roots["/workspace"].resolve()
    assert set(plan.environment) == {"HOME", "PATH", "TMPDIR", "TEMP", "TMP"}
    options = kwargs["popen_options"]
    assert options["stdout"] is subprocess.PIPE
    assert options["stderr"] is subprocess.PIPE
    assert options["text"] is True
    assert options["encoding"] == "utf-8"
    assert options["errors"] == "replace"
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
        instructions_manager=tool_instructions,
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
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", popen)

    assert run_command(command)["stdout"]["content"] == ""

    assert popen.call_args.args[0].argv[-len(argv) :] == (
        str(Path(shutil.which(argv[0])).resolve()),
        *argv[1:],
    )


def test_run_command_fails_closed_without_an_authorized_process_target():
    """Direct execution cannot parse or execute an unplanned command."""
    result = run_command_tool(ToolContext(ConsoleInteraction(), "run_command"), "echo hello")

    assert result.code == "process.execution_failed"
    assert result.detail == "Authorized process target is missing."


def test_run_command_rejects_a_stale_authorized_sandbox_policy(tmp_path):
    """Execution cannot reuse approval issued for a different containment plan."""
    command = str(isolated_executable(tmp_path))
    target = ProcessTarget(
        argv=(command,),
        cwd=str(tmp_path),
        workspace=str(tmp_path),
        readable_roots=(str(tmp_path),),
        writable_roots=(str(tmp_path),),
        sandbox_policy="0" * 64,
    )
    context = ToolContext(
        ConsoleInteraction(),
        "run_command",
        operations=(
            Operation(tool_id="run_command", action=Action.PROCESS_EXECUTE, target=target),
        ),
    )

    result = run_command_tool(context, command, str(tmp_path))

    assert result.code == "process.execution_failed"
    assert result.detail == "Authorized sandbox policy is stale."


def test_run_command_executes_an_authorized_plan_without_instruction_redaction(
    tmp_path, monkeypatch
):
    """Direct execution accepts a complete target without instruction path rendering."""
    command = str(isolated_executable(tmp_path))
    sandbox_plan = SandboxPlan.create((command,), tmp_path, tmp_path)
    target = ProcessTarget(
        argv=sandbox_plan.argv,
        cwd=str(sandbox_plan.cwd),
        workspace=str(sandbox_plan.workspace),
        read_only_roots=tuple(str(root) for root in sandbox_plan.read_only_roots),
        readable_roots=tuple(str(root) for root in sandbox_plan.readable_roots),
        writable_roots=tuple(str(root) for root in sandbox_plan.writable_roots),
        sandbox_policy=sandbox_plan.policy_digest,
    )
    monkeypatch.setattr(
        "loop.tools.system.spawn_sandboxed", lambda _plan, *, popen_options: make_process()
    )
    context = ToolContext(
        ConsoleInteraction(),
        "run_command",
        operations=(
            Operation(tool_id="run_command", action=Action.PROCESS_EXECUTE, target=target),
        ),
    )

    result = run_command_tool(context, command, str(tmp_path))

    assert result["exit_code"] == 0


def test_run_command_planning_requires_runtime_workspace_context():
    """Registry planning fails closed when no workspace authority is available."""
    result = tool_registry.call(
        "run_command",
        json.dumps({"command": "unresolved-command", "cwd": "."}),
        interaction=ConsoleInteraction(),
    )

    assert problem(result)["detail"] == "Sandbox planning requires a configured workspace root."


def test_host_execution_ceiling_denies_without_prompt_or_process(tmp_path, monkeypatch):
    """A closed host ceiling returns incompatibility without confirmation or execution."""
    command = isolated_executable(tmp_path)
    interaction = MagicMock()
    interaction.prompt.return_value = ApprovalChoice.ONCE
    permissions = PermissionManager(tmp_path, configuration=PermissionConfiguration())
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=permissions)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, permissions.temporary_directory)
    )
    monkeypatch.setattr(
        "loop.tools.system.spawn_sandboxed",
        MagicMock(side_effect=SandboxUnavailableError("unsupported command feature")),
    )
    host_spawn = MagicMock()
    monkeypatch.setattr("loop.tooling.tool_registry.spawn_host", host_spawn)

    output = registry.call(
        "run_command",
        json.dumps({"command": str(command), "cwd": "/workspace"}),
        interaction=interaction,
        instructions_manager=instructions,
    )

    assert problem(output)["code"] == "process.sandbox_incompatible"
    interaction.confirm.assert_not_called()
    host_spawn.assert_not_called()


def test_host_execution_requires_fresh_exact_confirmation_every_time(tmp_path, monkeypatch):
    """An open ceiling still confirms every host invocation with full authority disclosure."""
    command = isolated_executable(tmp_path)
    interaction = MagicMock()
    interaction.prompt.return_value = ApprovalChoice.ONCE
    interaction.confirm.return_value = True
    configuration = PermissionConfiguration(limits=PolicyLimits(allow_host_processes=True))
    permissions = PermissionManager(tmp_path, configuration=configuration)
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=permissions)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, permissions.temporary_directory)
    )
    monkeypatch.setattr(
        "loop.tools.system.spawn_sandboxed",
        MagicMock(side_effect=SandboxUnavailableError("unsupported command feature")),
    )
    host_spawn = MagicMock(side_effect=lambda _plan, **_kwargs: make_process())
    monkeypatch.setattr("loop.tooling.tool_registry.spawn_host", host_spawn)
    arguments = json.dumps({"command": str(command), "cwd": "/workspace"})

    first = registry.call(
        "run_command", arguments, interaction=interaction, instructions_manager=instructions
    )
    second = registry.call(
        "run_command", arguments, interaction=interaction, instructions_manager=instructions
    )

    assert json.loads(first)["result"]["boundary"] == "host"
    assert json.loads(second)["result"]["boundary"] == "host"
    assert interaction.confirm.call_count == 2
    prompt = interaction.confirm.call_args.args[0]
    assert "not sandboxed" in prompt
    assert "filesystem, credential, process, IPC, GUI, and network access" in prompt
    assert "unsupported command feature" in prompt
    assert "Canonical argv boundaries" in prompt
    assert host_spawn.call_count == 2


def test_host_rejection_creates_no_process(tmp_path, monkeypatch):
    """Rejecting the one-time host warning prevents host process creation."""
    command = isolated_executable(tmp_path)
    interaction = MagicMock()
    interaction.prompt.return_value = ApprovalChoice.ONCE
    interaction.confirm.return_value = False
    permissions = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(limits=PolicyLimits(allow_host_processes=True)),
    )
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=permissions)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, permissions.temporary_directory)
    )
    monkeypatch.setattr(
        "loop.tools.system.spawn_sandboxed",
        MagicMock(side_effect=SandboxUnavailableError("unsupported command feature")),
    )
    host_spawn = MagicMock()
    monkeypatch.setattr("loop.tooling.tool_registry.spawn_host", host_spawn)

    output = registry.call(
        "run_command",
        json.dumps({"command": str(command), "cwd": "/workspace"}),
        interaction=interaction,
        instructions_manager=instructions,
    )

    assert problem(output)["code"] == "process.host_rejected"
    host_spawn.assert_not_called()


def test_noninteractive_host_execution_fails_closed(tmp_path, monkeypatch):
    """An allow default and open ceiling cannot bypass interactive host confirmation."""
    command = isolated_executable(tmp_path)
    defaults = PermissionConfiguration().defaults
    defaults[Action.PROCESS_EXECUTE] = "allow"
    permissions = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(
            defaults=defaults,
            limits=PolicyLimits(allow_host_processes=True),
        ),
    )
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=permissions)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, permissions.temporary_directory)
    )
    monkeypatch.setattr(
        "loop.tools.system.spawn_sandboxed",
        MagicMock(side_effect=SandboxUnavailableError("unsupported command feature")),
    )
    host_spawn = MagicMock()
    monkeypatch.setattr("loop.tooling.tool_registry.spawn_host", host_spawn)

    output = registry.call(
        "run_command",
        json.dumps({"command": str(command), "cwd": "/workspace"}),
        instructions_manager=instructions,
    )

    assert problem(output)["code"] == "process.host_denied"
    host_spawn.assert_not_called()


@pytest.mark.parametrize(
    ("readable_roots", "writable_roots"),
    [((), ()), (("restricted",), ("restricted",))],
)
def test_command_planning_enforces_effective_filesystem_roots(
    tmp_path, readable_roots, writable_roots
):
    """Commands cannot regain workspace access excluded by filesystem limits."""
    command = isolated_executable(tmp_path)
    (tmp_path / "restricted").mkdir()
    permissions = PermissionManager(
        tmp_path,
        configuration=PermissionConfiguration(
            limits=PolicyLimits(
                readable_roots=readable_roots,
                writable_roots=writable_roots,
            )
        ),
    )
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=permissions)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, permissions.temporary_directory)
    )

    output = registry.call(
        "run_command",
        json.dumps({"command": str(command), "cwd": "/workspace"}),
        instructions_manager=instructions,
    )

    assert problem(output)["code"] == "tool.planning_failed"
    assert "outside readable roots" in problem(output)["detail"]


def test_run_command_resolves_virtual_cwd_and_redacts_known_host_roots(
    monkeypatch, confirmed, tmp_path
):
    """Virtual working directories execute locally without returning their backing root."""
    process = make_process(stdout=(f"{tmp_path}/created.txt\n", ""))
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", popen)
    (tmp_path / "temporary").mkdir()
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
    assert popen.call_args.args[0].cwd == tmp_path.resolve()


def test_run_command_resolves_virtual_path_arguments_before_execution(
    monkeypatch, confirmed, tmp_path
):
    """Virtual command arguments execute against the corresponding local paths."""
    process = make_process()
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", popen)
    (tmp_path / "temporary").mkdir()
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
    assert popen.call_args.args[0].argv[-4:] == (
        str(Path(shutil.which("git")).resolve()),
        "-C",
        str(tmp_path.resolve()),
        "status",
    )


def test_run_command_authorizes_complete_runtime_roots(monkeypatch, confirmed, tmp_path):
    """A subdirectory cwd retains workspace, temporary, and activated-skill authority."""
    workspace = tmp_path / "workspace"
    cwd = workspace / "nested"
    temporary = tmp_path / "temporary"
    skill_root = tmp_path / "skill"
    for directory in (cwd, temporary, skill_root):
        directory.mkdir(parents=True)
    skill_file = skill_root / "SKILL.md"
    skill_file.write_text(
        "---\nname: review\ndescription: Review files.\n---\n\nInstructions.\n",
        encoding="utf-8",
    )
    skills = SkillManager((Skill("review", "Review files.", skill_file),))
    skills.activate("review")
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(workspace, temporary), skill_manager=skills
    )
    captured = {}

    def spawn(plan, *, popen_options):
        """Capture the authorized plan while preserving lifecycle behavior."""
        captured["plan"] = plan
        return make_process()

    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", spawn)

    result = tool_registry.call(
        "run_command",
        json.dumps({"command": "pwd", "cwd": "/workspace/nested"}),
        interaction=ConsoleInteraction(),
        instructions_manager=instructions,
    )

    assert json.loads(result)["ok"] is True
    plan = captured["plan"]
    assert plan.workspace == workspace.resolve()
    assert plan.cwd == cwd.resolve()
    assert plan.temporary_directory == temporary.resolve()
    assert skill_root.resolve() in plan.read_only_roots


def test_successful_run_command_invalidates_instruction_scope(monkeypatch, tmp_path, confirmed):
    """Successful shell operations request a conservative instruction refresh."""
    manager = MagicMock()
    manager.virtual_paths.resolve.side_effect = lambda value: value
    manager.virtual_paths.resolve_command.side_effect = lambda value: value
    manager.virtual_paths.metadata.side_effect = lambda value: value
    manager.virtual_paths.redact.side_effect = lambda value: value
    manager.virtual_paths.roots = {"/workspace": tmp_path}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "loop.tools.system.spawn_sandboxed", MagicMock(return_value=make_process(stdout=("ok", "")))
    )

    result = tool_registry.call(
        "run_command",
        json.dumps({"command": "printf ok", "cwd": str(tmp_path)}),
        interaction=ConsoleInteraction(),
        instructions_manager=manager,
    )

    assert json.loads(result)["result"]["stdout"]["content"] == "ok"
    manager.invalidate.assert_called_once_with(None)


def test_run_command_completes_a_process_within_its_lifecycle_timeout(authorized):
    """A completed backend process returns normally under the shared lifecycle deadline."""
    tool_registry.settings.command_timeout = 0.5
    authorized.side_effect = None
    authorized.return_value = make_process(stdout=("complete\n", ""))

    assert run_command("printf complete")["stdout"]["content"] == "complete\n"


def test_run_command_reports_exit_code_stdout_and_stderr(monkeypatch, confirmed):
    """A failed command exposes its exit code and both captured streams."""
    process = make_process(
        stdout=("some output\n", ""), stderr=("command not found\n", ""), returncode=127
    )
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", MagicMock(return_value=process))

    failure = problem(run_command("printf missing"))
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
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", MagicMock(return_value=process))

    result = run_command("printf verbose-command")

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
    authorized.side_effect = None
    authorized.return_value = make_process(
        stdout=("x" * (MAX_OUTPUT_CHARS + 8192), ""),
        stderr=("y" * (MAX_OUTPUT_CHARS + 8192), ""),
        returncode=7,
    )

    failure = problem(run_command("printf large-output"))

    assert failure["code"] == "process.nonzero_exit"
    assert failure["metadata"]["exit_code"] == 7
    for stream in ("stdout", "stderr"):
        assert failure["metadata"][stream]["captured_bytes"] == MAX_OUTPUT_CHARS
        assert failure["metadata"][stream]["discarded_characters"] == 8192
        assert failure["metadata"][stream]["truncated"] is True


def test_run_command_replaces_undecodable_output(monkeypatch, authorized):
    """Invalid UTF-8 output is represented with replacement characters instead of failing."""
    tool_registry.settings.command_timeout = 0.5
    authorized.side_effect = None
    authorized.return_value = make_process(stdout=("ok�", ""))

    assert run_command("printf invalid-utf8")["stdout"]["content"] == "ok�"


def test_run_command_surfaces_reader_failures(monkeypatch, confirmed):
    """A pipe read failure interrupts a live command and becomes an execution problem."""
    process = make_process(stdout=(OSError("pipe read failed"),))
    process.poll.return_value = None
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock())

    failure = problem(run_command("printf broken-reader"))

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
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock())

    failure = problem(run_command("printf late-broken-reader"))

    assert failure["code"] == "process.execution_failed"
    assert failure["detail"] == "late pipe read failed"


def test_run_command_kills_a_posix_process_group_after_timeout(monkeypatch, confirmed):
    """Timeout kills the entire POSIX process group before reporting it."""
    process = make_process()
    process.pid = 123
    process.wait.side_effect = subprocess.TimeoutExpired("sleep", 30)
    process.poll.return_value = 0
    killpg = MagicMock()
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", MagicMock(return_value=process))
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
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", MagicMock(return_value=process))
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
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock())
    monkeypatch.setattr("loop.tools.system.os.close", close_descriptor)
    tool_registry.settings.command_timeout = 0.01

    assert problem(run_command("sleep 60"))["code"] == "process.timeout"
    assert call(10) in close_descriptor.call_args_list
    process.stdout.close.assert_not_called()
    process.stderr.close.assert_not_called()


def test_run_command_times_out_when_readers_remain_stuck_after_process_exit(
    tmp_path, monkeypatch, confirmed
):
    """Completed processes still time out when their output readers cannot finish."""
    process = make_process()

    class StuckThread(ImmediateThread):
        """Model a pipe reader that remains blocked after process completion."""

        def start(self):
            """Leave the modeled reader pending."""

        def is_alive(self):
            """Report that the modeled reader remains blocked."""
            return True

    StuckThread.instances = []
    monkeypatch.setattr("loop.tools.system.threading.Thread", StuckThread)
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock())
    tool_registry.settings.command_timeout = 0.01

    failure = problem(run_command(str(isolated_executable(tmp_path))))

    assert failure["code"] == "process.timeout"
    assert process.wait.called


def test_run_command_ignores_a_process_that_disappears_during_posix_cleanup(monkeypatch, confirmed):
    """A process-group lookup race does not replace the timeout result."""
    process = make_process()
    process.wait.side_effect = subprocess.TimeoutExpired("sleep", 30)
    process.poll.return_value = 0
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock(side_effect=ProcessLookupError))
    tool_registry.settings.command_timeout = 0.01

    assert problem(run_command("sleep 60"))["code"] == "process.timeout"


def test_run_command_cleans_up_when_wait_raises(monkeypatch, confirmed):
    """Unexpected wait errors still kill the process and join stream readers."""
    process = make_process()
    process.wait.side_effect = [RuntimeError("wait failed"), 0]
    process.poll.return_value = None
    killpg = MagicMock()
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)

    assert problem(run_command("printf broken"))["detail"] == "wait failed"
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
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", MagicMock())

    assert problem(run_command("printf broken"))["detail"] == "thread failed"
    assert FailingSecondThread.instances[0].joined
    assert not FailingSecondThread.instances[1].joined


def test_run_command_rejects_missing_process_pipe_handles(monkeypatch, confirmed):
    """Missing pipe handles fail explicitly and still clean up the created process."""
    process = make_process()
    process.stdout = None
    killpg = MagicMock()
    monkeypatch.setattr("loop.tools.system.spawn_sandboxed", MagicMock(return_value=process))
    monkeypatch.setattr("loop.utils.process.os.killpg", killpg)

    failure = problem(run_command("printf broken-pipes"))

    assert failure["code"] == "process.execution_failed"
    assert failure["detail"] == "Command process did not expose its output streams."
    process.stderr.close.assert_called_once_with()
    killpg.assert_called_once()


def test_run_command_reports_process_creation_errors(monkeypatch, confirmed):
    """Process creation failures become readable tool results."""
    monkeypatch.setattr(
        "loop.tools.system.spawn_sandboxed", MagicMock(side_effect=PermissionError("denied"))
    )

    assert problem(run_command("printf restricted"))["detail"] == "denied"
