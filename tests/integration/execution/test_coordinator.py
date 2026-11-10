"""Exercise owned native processes in disposable workspaces."""

import asyncio
import base64
import io
import json
import os
import shlex
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from rich.console import Console

from loop import (
    BUILTIN_TOOLS,
    Action,
    Decision,
    FileTarget,
    Interaction,
    Operation,
    PermissionManager,
    ToolRegistry,
)
from loop.execution import CommandExecutionService
from loop.execution.coordinator import (
    CommandExecutionCoordinator,
    HostCommandRequest,
    LocalHostCommandExecutor,
)
from loop.execution.sandbox import CommandProcessResult, SandboxOutcome, SandboxRequest
from loop.interaction import ConsoleInteraction
from loop.permissions import PermissionRule, PolicyScope
from loop.tooling import ToolRuntimeSettings
from loop.tooling.tool_registry import ToolExecutionPortal, ToolOwnerServiceProxy
from loop.workspace import Workspace, WorkspaceRepository
from tests.scheduling import receive_fifo_byte

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def controlled_execution_clock(monkeypatch):
    """Use one synthetic clock across the execution and process boundaries."""
    clock = SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda _delay: os.sched_yield())
    monkeypatch.setattr("loop.execution.facade.time", clock)
    monkeypatch.setattr("loop.execution.coordinator.time", clock)
    monkeypatch.setattr("loop.utils.process.time", clock)


def test_async_public_sandbox_command_reaches_native_backend(tmp_path):
    """The owner relay approves a routine tool command and starts the sandbox backend."""
    manager = PermissionManager(tmp_path)
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="ok"
    )
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=manager,
        execution_service=CommandExecutionService(backend, Mock()),
    )
    payload = json.dumps({"command": "printf ok", "cwd": str(tmp_path)})

    try:
        result = json.loads(asyncio.run(registry.call_async("run_command", payload)))
        assert result["ok"]
        assert result["result"]["boundary"] == "sandbox"
        assert result["result"]["stdout"]["content"] == "ok"
        backend.run.assert_called_once()
    finally:
        registry.close()
        manager.close()


@pytest.mark.parametrize("outcome", [SandboxOutcome.DENIED, SandboxOutcome.UNAVAILABLE])
def test_possibly_effective_sandbox_failure_needs_explicit_host_request(tmp_path, outcome):
    """A possibly effective failure cannot launch host work without a new explicit request."""
    marker = tmp_path / "host-marker"
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW, scope=PolicyScope.SESSION)
    interaction = Mock(spec=Interaction)
    interaction.supports_fresh_host_approval = True
    interaction.prompt.return_value = "approve"
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        outcome, detail="sandbox stopped", possible_effects=True
    )
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=manager,
        execution_service=CommandExecutionService(backend, LocalHostCommandExecutor()),
    )

    result = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "printf host > host-marker", "cwd": str(tmp_path)}),
            interaction=interaction,
        )
    )

    assert result["problem"]["code"] == f"sandbox.{outcome.value}"
    assert result["problem"]["metadata"]["sandbox_diagnostics"]["possible_effects"] is True
    assert "host_offer" not in result["problem"]["metadata"]
    assert not marker.exists()
    assert all(
        call.args[0] != "Run this exact command without the OS sandbox?"
        for call in interaction.prompt.call_args_list
    )
    explicit = json.loads(
        registry.call(
            "request_host_command",
            json.dumps(
                {
                    "command": "printf host > host-marker",
                    "cwd": str(tmp_path),
                    "reason": "Test a separate host request",
                }
            ),
            interaction=interaction,
        )
    )
    assert explicit["result"]["boundary"] == "host"
    assert marker.read_text(encoding="utf-8") == "host"
    assert any("may repeat" in call.args[0] for call in interaction.info.call_args_list)


@pytest.mark.parametrize("answer", ["approve", "deny"])
def test_explicit_host_tool_prompts_before_real_host_command(tmp_path, answer):
    """The public host tool starts one real child only after an explicit fresh decision."""
    marker = tmp_path / "marker"
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW, scope=PolicyScope.SESSION)
    interaction = Mock(spec=Interaction)
    interaction.supports_fresh_host_approval = True

    def decide(*_args, **_kwargs):
        """Observe the marker before answering the host prompt."""
        assert not marker.exists()
        return answer

    interaction.prompt.side_effect = decide
    service = CommandExecutionService(Mock(), LocalHostCommandExecutor())
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=manager, execution_service=service)
    payload = json.dumps(
        {"command": "printf approved > marker", "cwd": str(tmp_path), "reason": "Test host access"}
    )

    result = json.loads(registry.call("request_host_command", payload, interaction=interaction))

    interaction.prompt.assert_called_once()
    if answer == "approve":
        assert result["result"]["boundary"] == "host"
        assert marker.read_text() == "approved"
    else:
        assert result["problem"]["code"] == "host_command.denied"
        assert not marker.exists()


def test_yolo_public_host_tool_runs_exact_request_without_gate_or_prompt(tmp_path):
    """A synthetic real host child runs only for a separately issued eligible YOLO request."""
    marker = tmp_path / "marker"
    manager = PermissionManager(tmp_path, yolo=True)
    interaction = Mock(spec=Interaction)
    interaction.supports_fresh_host_approval = False
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=manager,
        execution_service=CommandExecutionService(Mock(), LocalHostCommandExecutor()),
    )
    payload = json.dumps(
        {"command": "printf run >> marker", "cwd": str(tmp_path), "reason": "Native YOLO check"}
    )

    first = json.loads(registry.call("request_host_command", payload, interaction=interaction))

    assert first["result"]["boundary"] == "host"
    assert marker.read_text(encoding="utf-8") == "run"
    interaction.prompt.assert_not_called()
    assert "Command:" in interaction.warning.call_args.args[0]
    assert manager.configuration.limits.allow_host_processes is False

    manager.set_yolo(False)
    second = json.loads(registry.call("request_host_command", payload, interaction=interaction))
    assert second["problem"]["code"] == "tool.denied"
    assert marker.read_text(encoding="utf-8") == "run"


@pytest.mark.skipif(
    os.environ.get("LOOP_NATIVE_HOST_TEST") != "disposable-linux-host",
    reason="Real unrestricted host-shell qualification requires a disposable Linux host.",
)
def test_trusted_host_shell_crosses_file_tool_deny_but_obeys_host_deny(tmp_path):
    """A trusted unrestricted shell reads synthetic data while a host deny blocks admission."""
    (tmp_path / "private").write_text("synthetic-host-private", encoding="utf-8")
    permissions = PermissionManager(tmp_path)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    permissions.add_rule(
        PermissionRule(
            decision=Decision.DENY,
            tool="read_text_file",
            action=Action.FILESYSTEM_READ,
        ),
        scope=PolicyScope.SESSION,
    )
    file_read = Operation(
        tool_id="read_text_file",
        action=Action.FILESYSTEM_READ,
        target=FileTarget(path=str(tmp_path / "private")),
    )
    assert permissions.evaluate((file_read,)).decision is Decision.DENY
    permissions.trust_host_command("cat private")
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(Mock(), LocalHostCommandExecutor()),
    )
    payload = json.dumps(
        {"command": "cat private", "cwd": str(tmp_path), "reason": "Native boundary check"}
    )

    permitted = json.loads(registry.call("request_host_command", payload))

    assert permitted["result"]["stdout"]["content"] == "synthetic-host-private"
    permissions.add_rule(
        PermissionRule(
            decision=Decision.DENY,
            tool="request_host_command",
            action=Action.PROCESS_EXECUTE,
        ),
        scope=PolicyScope.SESSION,
    )
    blocked = json.loads(registry.call("request_host_command", payload))
    assert blocked["problem"]["code"] == "tool.denied"
    permissions.close()


@pytest.mark.skipif(
    os.environ.get("LOOP_NATIVE_HOST_TEST") != "disposable-linux-host",
    reason="Real unrestricted host-shell qualification requires a disposable Linux host.",
)
def test_host_prompt_persists_grant_for_a_later_headless_session(tmp_path):
    """A real host prompt can persist a user grant that works after process restart."""
    catalog = tmp_path.parent / f"{tmp_path.name}-policy-data" / "workspaces.sqlite"
    workspace = WorkspaceRepository(catalog).initialize(Workspace.discover(tmp_path))
    arguments = {
        "configuration_path": catalog.parent / "workspaces" / workspace.id / "permissions.yaml",
        "user_configuration_path": catalog.parent / "user-permissions.yaml",
        "workspace_id": workspace.id,
        "namespace_catalog_path": catalog,
        "user_namespace_identity": str(catalog.parent),
    }
    permissions = PermissionManager(tmp_path, **arguments)
    permissions.initialize_policy_namespace("workspace", workspace.id, authorized=True)
    permissions.initialize_policy_namespace("user", str(catalog.parent), authorized=True)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    interaction = Mock(spec=Interaction, supports_fresh_host_approval=True)
    interaction.prompt.return_value = "trust_user"
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(Mock(), LocalHostCommandExecutor()),
    )
    payload = json.dumps(
        {"command": "printf persisted-host", "cwd": str(tmp_path), "reason": "Native grant check"}
    )

    first = json.loads(registry.call("request_host_command", payload, interaction=interaction))
    assert first["result"]["stdout"]["content"] == "persisted-host"
    interaction.prompt.assert_called_once()
    registry.close()
    permissions.close()

    reloaded = PermissionManager(tmp_path, **arguments)
    reloaded.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    resumed = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=reloaded,
        execution_service=CommandExecutionService(Mock(), LocalHostCommandExecutor()),
    )
    second = json.loads(resumed.call("request_host_command", payload))
    assert second["result"]["stdout"]["content"] == "persisted-host"
    resumed.close()
    reloaded.close()


def test_explicit_host_timeout_reports_effects_without_replay(tmp_path, monkeypatch):
    """A fake deadline preserves one real host process's effects without replay."""
    marker = tmp_path / "marker"
    ready = tmp_path / "ready"
    gate = tmp_path / "gate"
    os.mkfifo(ready)
    os.mkfifo(gate)
    observed = False

    def expired_after_entry():
        """Advance the process clock after the child reports its first effects."""
        nonlocal observed
        if not observed:
            assert receive_fifo_byte(ready) == b"r"
            observed = True
        return 200.0

    monkeypatch.setattr("loop.execution.facade.time", SimpleNamespace(monotonic=lambda: 100.0))
    monkeypatch.setattr("loop.execution.coordinator.time", SimpleNamespace(monotonic=lambda: 100.0))
    monkeypatch.setattr("loop.utils.process.time", SimpleNamespace(monotonic=expired_after_entry))
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW, scope=PolicyScope.SESSION)
    interaction = Mock(spec=Interaction)
    interaction.supports_fresh_host_approval = True
    interaction.prompt.return_value = "approve"
    backend = Mock()
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=manager,
        execution_service=CommandExecutionService(backend, LocalHostCommandExecutor()),
        settings=ToolRuntimeSettings(command_timeout=2),
    )
    try:
        result = json.loads(
            registry.call(
                "request_host_command",
                json.dumps(
                    {
                        "command": "printf x >> marker; printf ready; printf r > ready; read _ < gate",
                        "cwd": str(tmp_path),
                        "reason": "Verify timeout handling",
                    }
                ),
                interaction=interaction,
            )
        )
        assert result["problem"]["code"] == "host_command.timed_out"
        metadata = result["problem"]["metadata"]
        assert metadata["possible_effects"] is True
        assert metadata["stdout"]["content"] == "ready"
        assert "not retried" in metadata["recovery_hint"]
        assert "may repeat effects" in metadata["recovery_hint"]
        assert marker.read_text(encoding="utf-8") == "x"
        backend.run.assert_not_called()
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("command", "content", "size", "loss"),
    [
        ("printf '\\377'; printf '\\357\\277\\275' >&2", "�", 3, "unknown"),
        ("printf '\\303\\251'; printf '\\303\\251' >&2", "é", 2, "none"),
        ("printf 'a\\rb\\r\\nc\\n'; printf 'a\\rb\\r\\nc\\n' >&2", "a\rb\r\nc\n", 7, "none"),
    ],
)
@pytest.mark.parametrize("exit_code", [0, 7])
def test_explicit_host_output_reports_decoding_loss(
    tmp_path, command, content, size, loss, exit_code
):
    """Successful and failed host output preserve newlines and report decoding loss."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW, scope=PolicyScope.SESSION)
    interaction = Mock(spec=Interaction)
    interaction.supports_fresh_host_approval = True
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=manager,
        execution_service=CommandExecutionService(Mock(), LocalHostCommandExecutor()),
    )
    payload = json.dumps(
        {
            "command": f"{command}; exit {exit_code}",
            "cwd": str(tmp_path),
            "reason": "Inspect synthetic output bytes",
        }
    )

    result = json.loads(registry.call("request_host_command", payload, interaction=interaction))
    output = result["result"] if exit_code == 0 else result["problem"]["metadata"]
    if exit_code:
        assert result["problem"]["code"] == "process.nonzero_exit"

    for name in ("stdout", "stderr"):
        stream = output[name]
        assert stream["content"] == content
        assert stream["captured_bytes"] == size
        assert stream["capture_complete"] is True
        assert stream["decoding_loss"] == loss


@pytest.mark.parametrize("exit_code", [0, 7])
def test_explicit_host_output_preserves_path_like_content(tmp_path, exit_code):
    """Approved host streams and file-tool reads preserve identical synthetic path text."""
    workspace = str(tmp_path)
    content = (
        json.dumps(
            {
                "workspace": workspace,
                "sibling": workspace + "-extra/file",
                "url": "https://example.test" + workspace + "/file",
                "quoted": shlex.quote(workspace),
                "unicode": "é ☃",
                "encoded": base64.b64encode(workspace.encode()).decode(),
            },
            ensure_ascii=False,
        )
        + "\r\n"
    )
    (tmp_path / "path-content.json").write_text(content, encoding="utf-8", newline="")
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW, scope=PolicyScope.SESSION)
    interaction = Mock(spec=Interaction)
    interaction.supports_fresh_host_approval = True
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=manager,
        execution_service=CommandExecutionService(Mock(), LocalHostCommandExecutor()),
    )
    try:
        read = json.loads(
            registry.call(
                "read_text_file",
                json.dumps({"path": str(tmp_path / "path-content.json")}),
                interaction=interaction,
            )
        )
        source = f"printf %s {shlex.quote(content)}"
        result = json.loads(
            registry.call(
                "request_host_command",
                json.dumps(
                    {
                        "command": f"{source}; {source} >&2; exit {exit_code}",
                        "cwd": workspace,
                        "reason": "Inspect synthetic output bytes",
                    }
                ),
                interaction=interaction,
            )
        )
    finally:
        registry.close()
    output = result["result"] if exit_code == 0 else result["problem"]["metadata"]
    if exit_code:
        assert result["problem"]["code"] == "process.nonzero_exit"
    assert read["ok"] and read["result"]["content"] == content
    assert output["boundary"] == "host"
    assert all(output[name]["content"] == content for name in ("stdout", "stderr"))
    interaction.prompt.assert_called_once()


def test_explicit_host_tool_uses_async_owner_relay(tmp_path):
    """An asynchronous tool call relays a fresh decision before starting its real host child."""
    marker = tmp_path / "async-marker"
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW, scope=PolicyScope.SESSION)
    interaction = Mock(spec=Interaction)
    interaction.supports_fresh_host_approval = True
    interaction.prompt_async = AsyncMock(return_value="approve")
    interaction.confirm_async = AsyncMock()
    service = CommandExecutionService(Mock(), LocalHostCommandExecutor())
    registry = ToolRegistry(BUILTIN_TOOLS, permission_manager=manager, execution_service=service)
    payload = json.dumps(
        {"command": "printf async > async-marker", "cwd": str(tmp_path), "reason": "Test relay"}
    )

    async def run():
        """Invoke the synchronous host tool through the registry's worker relay."""
        return await registry.call_async("request_host_command", payload, interaction=interaction)

    result = json.loads(asyncio.run(run()))
    assert result["result"]["boundary"] == "host"
    assert marker.read_text() == "async"
    interaction.prompt_async.assert_awaited_once()


def test_explicit_host_tool_async_cancellation_precedes_approval(tmp_path):
    """A cancellation while an owner-relayed decision is pending prevents host launch."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW, scope=PolicyScope.SESSION)
    interaction = Mock(spec=Interaction)
    interaction.supports_fresh_host_approval = True
    interaction.confirm_async = AsyncMock()
    host = Mock()
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=manager,
        execution_service=CommandExecutionService(Mock(), host),
    )
    payload = json.dumps({"command": "printf host", "cwd": str(tmp_path), "reason": "Test relay"})

    async def run():
        """Release an affirmative answer only after signalling cancellation."""
        entered = asyncio.Event()
        release = asyncio.Event()
        cancelled = threading.Event()

        async def decide(*_args, **_kwargs):
            """Hold the owner-thread response until cancellation has been recorded."""
            entered.set()
            await release.wait()
            return "approve"

        interaction.prompt_async = AsyncMock(side_effect=decide)
        task = asyncio.create_task(
            registry.call_async(
                "request_host_command",
                payload,
                interaction=interaction,
                cancellation_event=cancelled,
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        cancelled.set()
        release.set()
        return await asyncio.wait_for(task, timeout=5)

    result = json.loads(asyncio.run(run()))
    assert not result["ok"]
    host.run_host_command.assert_not_called()


def request(tmp_path, **changes):
    """Build a bound sandbox command in the disposable workspace."""
    values = {
        "source": "printf ok",
        "cwd": tmp_path,
        "workspace": tmp_path,
        "read_roots": (tmp_path.parent,),
        "write_roots": (tmp_path,),
        "network": False,
        "environment": {"PATH": "/usr/bin:/bin"},
        "policy_version": "macos-v1",
        "deadline": 160.0,
        "workspace_id": "workspace-1",
    }
    values.update(changes)
    return SandboxRequest.create(**values)


def test_local_host_executor_captures_approved_shell_output(tmp_path):
    """The separate host supervisor returns bounded output and the shell exit code."""
    approved = HostCommandRequest.from_sandbox(request(tmp_path, source="printf host; exit 7"))

    result = LocalHostCommandExecutor().run_host_command(approved)

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 7
    assert result.stdout == "host"


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), float("-inf")])
def test_local_host_executor_rejects_nonfinite_deadline_before_real_launch(tmp_path, deadline):
    """Invalid deadlines cannot run a shell; a finite deadline permits the same command."""
    marker = tmp_path / "marker"
    approved = HostCommandRequest.from_sandbox(
        request(tmp_path, source=f"printf written > '{marker}'")
    )
    with pytest.raises(ValueError, match="Command deadline must be finite"):
        LocalHostCommandExecutor().run_host_command(replace(approved, deadline=deadline))
    assert not marker.exists()
    assert LocalHostCommandExecutor().run_host_command(approved).exit_code == 0
    assert marker.read_text() == "written"


def test_local_host_executor_times_out_and_cleans_up(tmp_path, process_boundary):
    """Executor and supervisor compose deadline handling with owned group cleanup."""
    boundary = process_boundary
    boundary.waitid.side_effect = None
    boundary.waitid.return_value = None
    approved = HostCommandRequest.from_sandbox(request(tmp_path, deadline=101))

    result = LocalHostCommandExecutor().run_host_command(approved)

    assert result.outcome is SandboxOutcome.TIMED_OUT
    assert result.possible_effects
    boundary.kill.assert_called_once_with(123, 9)
    assert boundary.process.wait.call_count == 1
    assert boundary.waitid.call_count > 1


def test_local_host_executor_completes_within_its_budget(tmp_path):
    """A real host child returns its output within a generous scheduling budget."""
    approved = HostCommandRequest.from_sandbox(
        request(tmp_path, source="printf finished", deadline=160.0)
    )
    result = LocalHostCommandExecutor().run_host_command(approved)

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.stdout == "finished"


def test_local_host_executor_timeout_closes_retained_pipe_without_thread_error(
    tmp_path, process_boundary, monkeypatch
):
    """Executor cleanup closes descendant-held pipes without scheduling real readers."""
    boundary = process_boundary

    def retained_thread(**kwargs):
        """Represent a descendant-held reader at the operating-system boundary."""
        thread = boundary.thread_factory(**kwargs)
        thread.is_alive.return_value = True
        return thread

    monkeypatch.setattr("loop.execution.coordinator.threading.Thread", retained_thread)
    approved = HostCommandRequest.from_sandbox(request(tmp_path, deadline=101))

    result = LocalHostCommandExecutor().run_host_command(approved)

    assert result.outcome is SandboxOutcome.TIMED_OUT
    boundary.kill.assert_called_once_with(123, 9)
    assert [call.args for call in boundary.close.call_args_list] == [(101,), (102,)]
    assert all(thread.join.call_count == 2 for thread in boundary.threads)


def test_session_host_approval_does_not_survive_a_replaced_read_root(tmp_path):
    """An exact session host rule cannot authorize a replacement directory at the same path."""
    outside = tmp_path.parent / "external-read"
    outside.mkdir()
    interaction = Mock(spec=Interaction)
    interaction.supports_fresh_host_approval = True
    host_answers = iter(("session", "deny"))
    interaction.prompt.side_effect = lambda prompt, **_kwargs: (
        next(host_answers) if prompt.startswith("Run this exact command") else "approve"
    )
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    host = Mock()
    host.run_host_command.return_value = "host-result"
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    first = request(tmp_path, read_roots=(outside,))
    offer = service.run_sandboxed(first, "run_command").offer
    assert service.retry_host_command(offer, first) == "host-result"

    outside.rename(tmp_path.parent / "old-external-read")
    outside.mkdir()
    replacement = request(tmp_path, read_roots=(outside,))
    offer = service.run_sandboxed(replacement, "run_command").offer

    assert service.retry_host_command(offer, replacement) is None
    assert host.run_host_command.call_count == 1
    assert (
        sum(
            call.args[0].startswith("Run this exact command")
            for call in interaction.prompt.call_args_list
        )
        == 2
    )


def test_host_retry_uses_owner_relay_without_blocking(tmp_path, monkeypatch):
    """Revision snapshots and remembered host consent work through the async owner relay."""
    interaction = ConsoleInteraction(
        console=Console(file=io.StringIO()), session=Mock(), choices_session=Mock(), markdown=False
    )
    interaction.info = Mock()
    interaction.prompt_async = AsyncMock(side_effect=("approve", "session"))
    interaction._choices_session.input.fileno.return_value = 0
    monkeypatch.setattr("loop.interaction.console.os.isatty", lambda fd: fd == 0)
    manager = PermissionManager(tmp_path, interaction=interaction)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    host = Mock()
    host.run_host_command.return_value = "host-result"
    approved = request(tmp_path)

    async def exercise():
        """Run the synchronous coordinator in a worker with owner-thread service relays."""
        portal = ToolExecutionPortal(asyncio.get_running_loop())
        service = CommandExecutionCoordinator(
            backend,
            host,
            ToolOwnerServiceProxy(manager, portal),
            ToolOwnerServiceProxy(interaction, portal),
        )

        def execute():
            """Offer and approve one host retry from the tool worker."""
            attempt = service.run_sandboxed(approved, "run_command")
            assert attempt.offer is not None
            return service.retry_host_command(attempt.offer, approved)

        return await asyncio.wait_for(asyncio.to_thread(execute), timeout=5)

    try:
        assert asyncio.run(exercise()) == "host-result"
        assert interaction.prompt_async.await_count == 2
        assert len(manager.host_command_rules()) == 1
        host.run_host_command.assert_called_once()
    finally:
        manager.close()
