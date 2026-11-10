"""Test the public native command tool boundary."""

from __future__ import annotations

import asyncio
import json
import platform
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from loop import BUILTIN_TOOLS, PermissionManager, ToolContext, ToolRegistration, ToolRegistry
from loop.constants import MAX_OUTPUT_CHARS, MAX_TOOL_RESULT_BYTES
from loop.execution import CommandExecutionService
from loop.execution.coordinator import HostCommandRequest, LocalHostCommandExecutor
from loop.execution.sandbox import (
    CommandProcessResult,
    SandboxCapability,
    SandboxOutcome,
    UnavailableSandboxBackend,
)
from loop.execution.sandbox.constants import (
    HOME_ENVIRONMENT_VARIABLE,
    LANG_ENVIRONMENT_VARIABLE,
    PATH_ENVIRONMENT_VARIABLE,
    TEMP_ENVIRONMENT_VARIABLE,
    TMP_ENVIRONMENT_VARIABLE,
    TMPDIR_ENVIRONMENT_VARIABLE,
    WORKSPACE_ROOT_ENVIRONMENT_VARIABLE,
    XDG_CACHE_HOME_ENVIRONMENT_VARIABLE,
)
from loop.execution.sandbox.linux import backend as linux
from loop.execution.sandbox.macos import backend as macos_backend
from loop.execution.sandbox.macos.backend import MacOSSeatbeltBackend
from loop.instructions import InstructionsManager, RuntimeEnvironment
from loop.permissions import (
    Action,
    CommandFinding,
    CommandInspection,
    CommandReviewStatus,
    Decision,
    FileTarget,
    Operation,
    PermissionRule,
    PolicyScope,
    ProcessBoundary,
    ProcessTarget,
)
from loop.tooling import get_tool_arguments_model
from loop.tools.system import _command_plan
from loop.tools.system import cancel_command_job as cancel_command_job_tool
from loop.tools.system import control_command_terminal as control_command_terminal_tool
from loop.tools.system import inspect_command_jobs as inspect_command_jobs_tool
from loop.tools.system import read_command_job as read_command_job_tool
from loop.tools.system import request_host_command as request_host_command_tool
from loop.tools.system import resolve_executable as resolve_executable_tool
from loop.tools.system import run_command as run_command_tool
from loop.utils import VirtualPath

from ..execution.sandbox.helpers import logical_private_name


@pytest.fixture(autouse=True)
def logical_policy_names(monkeypatch, tmp_path):
    """Keep fixture storage ancestry separate from protected policy components."""

    def logical(path):
        """Preserve protected components inside this case's logical roots."""
        return logical_private_name(path, tmp_path.parent)

    monkeypatch.setattr("loop.execution.facade.is_protected_external_read", logical)
    monkeypatch.setattr("loop.execution.sandbox.linux.paths.is_protected_external_read", logical)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.is_protected_external_read", logical)


class ApprovingInteraction:
    """Approve every explicit one-off native command prompt."""

    supports_fresh_host_approval = True

    def info(self, message):
        """Accept informational output."""

    def prompt(self, message, **kwargs):
        """Approve the requested command."""
        return "deny" if "without the OS sandbox" in message else "approve"


class ApprovingHostInteraction(ApprovingInteraction):
    """Approve both separately presented sandbox and host prompts."""

    def prompt(self, message, **kwargs):
        """Approve the fresh one-off choice shown in each prompt."""
        return "approve"


class RememberingHostInteraction(ApprovingInteraction):
    """Record the first exact host choice and deny any later host prompt."""

    host_prompts: int

    def __init__(self):
        self.host_prompts = 0

    def prompt(self, message, **kwargs):
        """Choose a session rule once and expose unexpected reapproval requests."""
        if "without the OS sandbox" in message:
            self.host_prompts += 1
            return "session" if self.host_prompts == 1 else "deny"
        return "approve"


class CountingHostCommandExecutor:
    """Record approved host requests without starting a real shell."""

    _executor: MagicMock
    requests: list[HostCommandRequest]

    def __init__(self):
        self._executor = MagicMock()
        self._executor.run_host_command.return_value = CommandProcessResult(
            SandboxOutcome.COMPLETED, exit_code=0, stdout="startup-recovery"
        )
        self.requests = []

    def run_host_command(self, request: HostCommandRequest) -> CommandProcessResult:
        """Record one approved host request and return the configured process result."""
        self.requests.append(request)
        return self._executor.run_host_command(request)


class RecordingHostOfferInteraction(ApprovingInteraction):
    """Record the offer and host-launch count before selecting a retry decision."""

    _host_answer: str
    _host_executor: CountingHostCommandExecutor
    information: list[str]
    host_prompts: list[str]
    launches_before_host_prompt: list[int]

    def __init__(self, host_executor: CountingHostCommandExecutor, host_answer: str):
        self._host_answer = host_answer
        self._host_executor = host_executor
        self.information = []
        self.host_prompts = []
        self.launches_before_host_prompt = []

    def info(self, message):
        """Capture the warning presented with a host retry offer."""
        self.information.append(message)

    def prompt(self, message, **kwargs):
        """Choose the configured host decision and approve unrelated prompts."""
        if message and "without the OS sandbox" in message:
            self.host_prompts.append(message)
            self.launches_before_host_prompt.append(len(self._host_executor.requests))
            return self._host_answer
        return super().prompt(message, **kwargs)


@pytest.fixture
def registry():
    """Provide a registry with the native permission authority."""
    return native_registry()


def native_registry(*, permission_manager=None):
    """Compose the real execution facade for public tool tests."""
    return ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permission_manager or PermissionManager(),
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )


def call(registry, command, cwd=".", interaction=None, instructions_manager=None, **grants):
    """Dispatch one command and decode its result envelope."""
    output = registry.call(
        "run_command",
        json.dumps({"command": command, "cwd": cwd, **grants}),
        interaction=interaction or ApprovingInteraction(),
        instructions_manager=instructions_manager,
    )
    payload = json.loads(output)
    return payload.get("result"), payload.get("problem")


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (inspect_command_jobs_tool, ()),
        (read_command_job_tool, ("a" * 32,)),
        (cancel_command_job_tool, ("a" * 32,)),
    ],
)
def test_command_job_tools_require_the_execution_service(tool, arguments):
    """Continuation tools fail safely when native command execution is unavailable."""
    result = tool(ToolContext(None, tool.__name__), *arguments)
    assert result.code == "command_job.unavailable"


def test_command_job_tools_delegate_to_current_execution_service():
    """Continuation calls forward exact handle, cursor, and wait to the bound owner."""
    service = MagicMock()
    service.inspect_command_jobs.return_value = {"jobs": []}
    service.read_command_job.return_value = {"events": []}
    service.cancel_command_job.return_value = {"cancel_requested": True}
    context = ToolContext(None, "inspect_command_jobs", execution_service=service)
    assert inspect_command_jobs_tool(context) == {"jobs": []}
    assert read_command_job_tool(context, "a" * 32, 3, 2) == {"events": []}
    assert cancel_command_job_tool(context, "a" * 32) == {"cancel_requested": True}
    service.read_command_job.assert_called_once_with(context, "a" * 32, 3, 2, False)
    service.cancel_command_job.assert_called_once_with(context, "a" * 32)


def test_terminal_control_tool_requires_and_uses_the_bound_execution_service():
    """Only the bound execution service receives exact transient terminal controls."""
    missing = control_command_terminal_tool(
        ToolContext(None, "control_command_terminal"), "a" * 32, "eof"
    )
    assert missing.code == "command_terminal.unavailable"
    service = MagicMock()
    service.control_command_terminal.return_value = {"terminal": True}
    context = ToolContext(None, "control_command_terminal", execution_service=service)
    assert control_command_terminal_tool(context, "a" * 32, "write", "text") == {"terminal": True}
    service.control_command_terminal.assert_called_once_with(
        context, "a" * 32, "write", "text", None, None
    )


@pytest.mark.parametrize("action", ["write", "eof", "resize"])
@pytest.mark.parametrize("spelling", ["None", "nOnE", "null", "NULL"])
def test_terminal_controls_preserve_literal_input_and_accept_unused_nulls(action, spelling):
    """The write action sends literal text while unused controls become absent."""
    service = MagicMock()
    service.control_command_terminal.return_value = {"terminal": True}
    registry = ToolRegistry([control_command_terminal_tool], execution_service=service)
    arguments = {
        "handle": "a" * 32,
        "action": action,
        "content": spelling,
        "rows": 24 if action == "resize" else spelling,
        "columns": 80 if action == "resize" else spelling,
    }
    result = json.loads(registry.call("control_command_terminal", json.dumps(arguments)))
    assert result["ok"]
    context, handle, selected, content, rows, columns = (
        service.control_command_terminal.call_args.args
    )
    assert isinstance(context, ToolContext)
    assert (handle, selected) == ("a" * 32, action)
    assert content == (spelling if action == "write" else None)
    assert (rows, columns) == ((24, 80) if action == "resize" else (None, None))


def lookup(registry, name, cwd=None, **context):
    """Dispatch one named host executable lookup through the public registry."""
    arguments = {"name": name}
    if cwd is not None:
        arguments["cwd"] = cwd
    return json.loads(registry.call("resolve_executable", json.dumps(arguments), **context))


def test_explicit_host_tool_requires_fresh_approval_for_each_run(tmp_path):
    """The host tool uses ordinary policy and asks again even after an approved host run."""
    backend = MagicMock()
    host = MagicMock()
    host.run_host_command.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="host-output"
    )
    permissions = PermissionManager(tmp_path)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    permissions.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW, scope=PolicyScope.SESSION)
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, host),
    )
    interaction = MagicMock()
    interaction.prompt.return_value = "approve"
    interaction.supports_fresh_host_approval = True
    payload = json.dumps({"command": "printf host", "cwd": str(tmp_path), "reason": "Need host"})

    first = json.loads(registry.call("request_host_command", payload, interaction=interaction))
    second = json.loads(registry.call("request_host_command", payload, interaction=interaction))

    assert first["result"]["boundary"] == "host"
    assert first["result"]["stdout"]["content"] == "host-output"
    assert second["result"]["exit_code"] == 0
    assert interaction.prompt.call_count == 2
    assert host.run_host_command.call_count == 2
    backend.run.assert_not_called()


def test_explicit_host_tool_uses_one_prompt_for_process_ask(tmp_path):
    """The host prompt alone decides an explicit request under generic process ask."""
    host = MagicMock()
    host.run_host_command.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    permissions = PermissionManager(tmp_path)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(MagicMock(), host),
    )
    interaction = MagicMock()
    interaction.supports_fresh_host_approval = True
    interaction.prompt.return_value = "approve"
    payload = json.dumps({"command": "printf host", "cwd": str(tmp_path), "reason": "Needed"})

    result = json.loads(registry.call("request_host_command", payload, interaction=interaction))

    assert result["ok"]
    interaction.prompt.assert_called_once()
    assert "without the OS sandbox" in interaction.prompt.call_args.args[0]
    host.run_host_command.assert_called_once()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_explicit_host_tool_enables_session_gate_and_approves_in_one_call(tmp_path, asynchronous):
    """A gate-only block can be cleared and the exact host request approved in one call."""
    host = MagicMock()
    host.run_host_command.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    permissions = PermissionManager(tmp_path)
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(MagicMock(), host),
    )
    interaction = MagicMock()
    interaction.supports_fresh_host_approval = True
    interaction.prompt.side_effect = ["enable", "approve"] if not asynchronous else ["approve"]
    interaction.prompt_async = AsyncMock(side_effect=["enable", "approve"])
    payload = json.dumps({"command": "printf host", "cwd": str(tmp_path), "reason": "Needed"})

    raw = (
        asyncio.run(registry.call_async("request_host_command", payload, interaction=interaction))
        if asynchronous
        else registry.call("request_host_command", payload, interaction=interaction)
    )

    assert json.loads(raw)["ok"], raw
    assert permissions.effective_configuration.limits.allow_host_processes
    assert interaction.prompt.call_count == (0 if asynchronous else 2)
    assert interaction.prompt_async.await_count == (2 if asynchronous else 0)
    gate_prompt = (
        interaction.prompt_async.await_args_list[0]
        if asynchronous
        else interaction.prompt.call_args_list[0]
    )
    assert gate_prompt.kwargs["choices"] == {
        "deny": "Deny",
        "enable": "Enable for this session",
    }
    assert gate_prompt.kwargs["index"] == {"deny": "N", "enable": "Y"}
    host.run_host_command.assert_called_once()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("change", ["before", "after"])
def test_explicit_host_gate_rechecks_policy_during_enablement(
    tmp_path, monkeypatch, asynchronous, change
):
    """A host deny introduced during eligibility review prevents execution."""
    host = MagicMock()
    permissions = PermissionManager(tmp_path)
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(MagicMock(), host),
    )
    interaction = MagicMock()
    interaction.supports_fresh_host_approval = True
    interaction.prompt.return_value = "enable"
    interaction.prompt_async = AsyncMock(return_value="enable")
    original = permissions.enable_host_gate_if_current

    def changed(operations, revision):
        """Change the policy immediately before or after gate enablement."""
        if change == "before":
            permissions.set_default(
                Action.PROCESS_EXECUTE, Decision.DENY, scope=PolicyScope.SESSION
            )
        enabled = original(operations, revision)
        if change == "after":
            permissions.set_default(
                Action.PROCESS_EXECUTE, Decision.DENY, scope=PolicyScope.SESSION
            )
        return enabled

    monkeypatch.setattr(permissions, "enable_host_gate_if_current", changed)
    payload = json.dumps({"command": "printf host", "cwd": str(tmp_path), "reason": "Needed"})

    raw = (
        asyncio.run(registry.call_async("request_host_command", payload, interaction=interaction))
        if asynchronous
        else registry.call("request_host_command", payload, interaction=interaction)
    )

    assert json.loads(raw)["problem"]["code"] == "tool.denied"
    host.run_host_command.assert_not_called()


def test_explicit_host_tool_trusted_grant_bypasses_file_tool_deny_only_when_eligible(tmp_path):
    """An exact trusted host shell is independent of file-tool policy but obeys host denies."""
    host = MagicMock()
    host.run_host_command.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="synthetic-secret"
    )
    permissions = PermissionManager(tmp_path)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    permissions.add_rule(
        PermissionRule(
            decision=Decision.DENY, tool="read_text_file", action=Action.FILESYSTEM_READ
        ),
        scope=PolicyScope.SESSION,
    )
    permissions.trust_host_command("cat private")
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(MagicMock(), host),
    )
    denied_file = Operation(
        tool_id="read_text_file",
        action=Action.FILESYSTEM_READ,
        target=FileTarget(path=str(tmp_path / "private")),
    )
    assert permissions.evaluate((denied_file,)).decision is Decision.DENY
    payload = json.dumps({"command": "cat private", "cwd": str(tmp_path), "reason": "Needed"})

    result = json.loads(registry.call("request_host_command", payload))

    assert result["result"]["stdout"]["content"] == "synthetic-secret"
    assert "sandbox restrictions did not apply" in result["result"]["authority_notice"]
    host.run_host_command.assert_called_once()
    permissions.add_rule(
        PermissionRule(
            decision=Decision.DENY, tool="request_host_command", action=Action.PROCESS_EXECUTE
        ),
        scope=PolicyScope.SESSION,
    )
    blocked = json.loads(registry.call("request_host_command", payload))
    assert blocked["problem"]["code"] == "tool.denied"
    host.run_host_command.assert_called_once()


def test_explicit_host_tool_without_service_fails_closed():
    """The host tool cannot run when the execution service is unavailable."""
    result = request_host_command_tool(ToolContext(None, "request_host_command"), "pwd", "Needed")
    assert result.code == "host_command.unavailable"


def test_renamed_host_tool_keeps_its_fresh_input_requirement(tmp_path):
    """Renaming the host tool cannot bypass the early interactive-input gate."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    host = MagicMock()
    registry = ToolRegistry(
        [ToolRegistration(request_host_command_tool, name="renamed_host")],
        permission_manager=manager,
        execution_service=CommandExecutionService(MagicMock(), host),
    )
    interaction = MagicMock()
    interaction.supports_fresh_host_approval = False
    payload = json.dumps({"command": "pwd", "cwd": str(tmp_path), "reason": "Needed"})

    result = json.loads(registry.call("renamed_host", payload, interaction=interaction))

    assert result["problem"]["code"] == "host_command.denied"
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize("mode", ["policy", "headless", "decline", "automatic"])
def test_explicit_host_tool_refuses_unapproved_launch(tmp_path, mode):
    """Host policy, absent or automatic input, and decline prevent any host launch."""
    backend = MagicMock()
    host = MagicMock()
    permissions = PermissionManager(tmp_path)
    if mode != "policy":
        permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
        if mode != "automatic":
            permissions.set_default(
                Action.PROCESS_EXECUTE, Decision.ALLOW, scope=PolicyScope.SESSION
            )
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, host),
    )
    interaction = MagicMock() if mode != "headless" else None
    if interaction is not None:
        interaction.prompt.return_value = "deny"
        interaction.supports_fresh_host_approval = mode != "automatic"
    payload = json.dumps({"command": "printf host", "cwd": str(tmp_path), "reason": "Need host"})

    result = json.loads(registry.call("request_host_command", payload, interaction=interaction))

    assert not result["ok"]
    if mode == "automatic":
        interaction.prompt.assert_not_called()
        assert not permissions.persistent_rules
    host.run_host_command.assert_not_called()
    backend.run.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_explicit_host_gate_denial_is_durably_audited(tmp_path, asynchronous):
    """A rejected host request leaves a minimized decision even before an offer exists."""
    audit_path = tmp_path / "audit.sqlite"
    permissions = PermissionManager(tmp_path, audit_path=audit_path, workspace_id="workspace")
    host = MagicMock()
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(MagicMock(), host),
    )
    payload = json.dumps(
        {"command": "printf synthetic-secret", "cwd": str(tmp_path), "reason": "Needed"}
    )

    raw = (
        asyncio.run(registry.call_async("request_host_command", payload))
        if asynchronous
        else registry.call("request_host_command", payload)
    )

    assert json.loads(raw)["problem"]["code"] == "tool.denied"
    host.run_host_command.assert_not_called()
    with closing(sqlite3.connect(audit_path)) as connection:
        rows = connection.execute(
            "SELECT event_name, payload_json FROM permission_audit_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "host_command.denied"
    assert json.loads(rows[0][1])["source"] == "policy"
    assert "synthetic-secret" not in rows[0][1]
    permissions.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("denial", ["workspace", "session", "rule"])
def test_explicit_host_tool_explains_the_effective_denial(tmp_path, asynchronous, denial):
    """A gate-only denial offers session eligibility without overriding a policy deny."""
    permissions = PermissionManager(tmp_path)
    if denial != "workspace":
        permissions.set_limit("allow_host_processes", True, scope=PolicyScope.WORKSPACE)
    if denial == "session":
        permissions.set_limit("allow_host_processes", False, scope=PolicyScope.SESSION)
    elif denial == "rule":
        permissions.add_rule(
            PermissionRule(
                decision=Decision.DENY,
                tool="request_host_command",
                action=Action.PROCESS_EXECUTE,
            ),
            scope=PolicyScope.SESSION,
        )
    host = MagicMock()
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(MagicMock(), host),
    )
    interaction = MagicMock()
    interaction.supports_fresh_host_approval = True
    interaction.prompt.return_value = "deny"
    interaction.prompt_async = AsyncMock(return_value="deny")
    payload = json.dumps({"command": "printf host", "cwd": str(tmp_path), "reason": "Need host"})

    output = (
        asyncio.run(registry.call_async("request_host_command", payload, interaction=interaction))
        if asynchronous
        else registry.call("request_host_command", payload, interaction=interaction)
    )

    result = json.loads(output)
    assert result["problem"]["code"] == "tool.denied"
    detail = result["problem"]["detail"]
    if denial == "rule":
        assert "/permissions limit set" not in detail
        interaction.prompt.assert_not_called()
        interaction.prompt_async.assert_not_awaited()
    else:
        assert "Host eligibility was declined" in detail
        if asynchronous:
            interaction.prompt_async.assert_awaited_once()
        else:
            interaction.prompt.assert_called_once()
    host.run_host_command.assert_not_called()


def test_explicit_host_tool_rejects_automatic_async_input_before_policy_prompt(tmp_path):
    """Async automatic input cannot cause a remembered approval or start a host process."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    host = MagicMock()
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=manager,
        execution_service=CommandExecutionService(MagicMock(), host),
    )
    interaction = MagicMock()
    interaction.supports_fresh_host_approval = False
    payload = json.dumps({"command": "printf host", "cwd": str(tmp_path), "reason": "Need host"})

    result = json.loads(
        asyncio.run(registry.call_async("request_host_command", payload, interaction=interaction))
    )

    assert result["problem"]["code"] == "host_command.denied"
    interaction.prompt.assert_not_called()
    assert not manager.persistent_rules
    host.run_host_command.assert_not_called()


def test_public_host_session_approval_reuses_across_fresh_scratch(tmp_path):
    """Public commands reuse one exact host approval and prompt for changed source."""
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="probe")
    host = MagicMock()
    host.run_host_command.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="host\n"
    )
    permissions = PermissionManager(tmp_path)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, host),
    )
    interaction = RememberingHostInteraction()

    first, first_problem = call(registry, "printf host", cwd=str(tmp_path), interaction=interaction)
    second, second_problem = call(
        registry, "printf host", cwd=str(tmp_path), interaction=interaction
    )
    changed, changed_problem = call(
        registry, "printf changed", cwd=str(tmp_path), interaction=interaction
    )

    assert first_problem is None and second_problem is None
    assert first["boundary"] == second["boundary"] == "host"
    assert first["stdout"]["content"] == second["stdout"]["content"] == "host\n"
    assert changed is None and changed_problem["code"] == "sandbox.unavailable"
    assert interaction.host_prompts == 2
    assert host.run_host_command.call_count == 2
    assert len(permissions.host_command_rules()) == 1


def scope_receipt_opener(tmp_path, monkeypatch):
    """Anchor owned fixture receipt reads without accessing the host root."""
    opener = macos_backend.open_selected_file

    def open_owned_file(root, path, identity):
        """Keep fixture receipt reads beneath the owned temporary directory."""
        if root == Path("/") and path.is_relative_to(tmp_path):
            return opener(tmp_path, path, identity)
        return opener(root, path, identity)

    monkeypatch.setattr(macos_backend, "open_selected_file", open_owned_file)


def homebrew_fixture(tmp_path, monkeypatch, receipt=None):
    """Create a disposable Homebrew-shaped executable and optional dependency receipt."""
    scope_receipt_opener(tmp_path, monkeypatch)
    prefix = tmp_path / "brew"
    package = prefix / "Cellar" / "sample" / "1.0"
    binaries = package / "bin"
    binaries.mkdir(parents=True)
    executable = binaries / "sampletool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    opt = prefix / "opt"
    opt.mkdir()
    (opt / "sample").symlink_to(package, target_is_directory=True)
    if receipt is not None:
        (package / "INSTALL_RECEIPT.json").write_text(receipt, encoding="utf-8")
    monkeypatch.setenv("PATH", str(binaries))
    return prefix, package, opt


def test_resolve_executable_reports_plain_and_symlinked_tools(tmp_path, monkeypatch, registry):
    """A host PATH lookup finds files and symlinks without running or reading them."""
    prefix = tmp_path / "tool-prefix"
    binaries = prefix / "bin"
    support = prefix / "lib"
    binaries.mkdir(parents=True)
    support.mkdir()
    plain = binaries / "plain"
    plain.write_text("#!/bin/sh\nprintf executed > marker\n", encoding="utf-8")
    plain.chmod(0o755)
    target = support / "linked-target"
    target.write_text("#!/bin/sh\nprintf executed > marker\n", encoding="utf-8")
    target.chmod(0o755)
    (binaries / "linked").symlink_to(target)
    monkeypatch.setenv("PATH", str(binaries))

    plain_result = lookup(registry, "plain")["result"]
    linked_result = lookup(registry, "linked")["result"]

    assert plain_result["sandbox_access"] == "external_read_grant_required"
    assert plain_result["candidate_read_roots"] == [str(binaries)]
    assert linked_result["resolved_path"] == str(target)
    assert linked_result["candidate_read_roots"] == [str(support)]
    assert plain_result["grant_reference"] != linked_result["grant_reference"]
    assert "both references as a list" in plain_result["grant_note"]
    assert "executed" not in json.dumps((plain_result, linked_result))
    assert not (tmp_path / "marker").exists()


def test_resolve_executable_accepts_project_tool_list(tmp_path, monkeypatch, registry):
    """Public lookup reports each requested tool independently without executing it."""
    for name in ("build-tool", "format-tool"):
        executable = tmp_path / name
        executable.write_text("#!/bin/sh\nprintf executed > marker\n", encoding="utf-8")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    result = lookup(registry, ["build-tool", "missing-tool", "format-tool"])["result"]

    first, missing, second = result["executables"]
    assert first["name"] == "build-tool" and first["grant_reference"]
    assert missing == {"name": "missing-tool", "status": "absent_from_host_path"}
    assert second["name"] == "format-tool" and second["grant_reference"]
    assert first["grant_reference"] != second["grant_reference"]
    assert not (tmp_path / "marker").exists()


def test_executable_lookup_without_a_bound_service_fails_closed():
    """A registry missing native execution state cannot issue a reusable grant."""
    registry = ToolRegistry([resolve_executable_tool])
    assert lookup(registry, "true")["problem"]["code"] == "executable.invalid_lookup"


def test_resolve_executable_distinguishes_absent_and_invalid_names(tmp_path, monkeypatch, registry):
    """A missing named tool is distinct from an unsafe or oversized lookup."""
    monkeypatch.setenv("PATH", str(tmp_path))
    assert lookup(registry, "missing")["result"] == {
        "name": "missing",
        "status": "absent_from_host_path",
    }
    assert lookup(registry, "../secret")["problem"]["code"] == "executable.invalid_lookup"
    monkeypatch.setenv("PATH", ":".join([str(tmp_path)] * 129))
    assert lookup(registry, "missing")["problem"]["code"] == "executable.invalid_lookup"


def test_resolve_executable_skips_nonexecutables_and_preserves_path_order(
    tmp_path, monkeypatch, registry
):
    """Only the first executable regular file on PATH is reported."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "tool").write_text("not executable", encoding="utf-8")
    selected = second / "tool"
    selected.write_text("selected", encoding="utf-8")
    selected.chmod(0o755)
    monkeypatch.setenv("PATH", f"{first}:{second}")

    assert lookup(registry, "tool")["result"]["path"] == str(selected)


def test_resolve_executable_does_not_suggest_a_shared_temporary_parent(monkeypatch, registry):
    """A symlink across separate temporary roots recommends only their file directories."""
    with (
        tempfile.TemporaryDirectory(prefix="lookup-search-") as first_name,
        tempfile.TemporaryDirectory(prefix="lookup-target-") as second_name,
    ):
        first = Path(first_name).resolve()
        second = Path(second_name).resolve()
        target = second / "tool"
        target.write_text("#!/bin/sh\n", encoding="utf-8")
        target.chmod(0o755)
        (first / "tool").symlink_to(target)
        monkeypatch.setenv("PATH", str(first))

        roots = lookup(registry, "tool")["result"]["candidate_read_roots"]

        assert roots == [str(second)]


@pytest.fixture
def reviewed_system_tool(tmp_path, monkeypatch):
    """Provide a reviewed executable without depending on installed host utilities."""
    binaries = tmp_path / "system-tools"
    binaries.mkdir()
    executable = binaries / "true"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(binaries))
    monkeypatch.setattr(
        MacOSSeatbeltBackend, "system_read_roots", property(lambda self: (binaries,))
    )
    return executable


def test_resolve_executable_reports_system_tool_as_default_read(registry, reviewed_system_tool):
    """A reviewed system executable needs no external read grant."""

    result = lookup(registry, "true")["result"]

    assert result["status"] == "installed_on_host"
    assert result["sandbox_access"] == "default_read"
    assert "candidate_read_roots" not in result


def test_homebrew_lookup_binds_only_package_and_declared_support_roots(
    tmp_path, monkeypatch, registry
):
    """Installation metadata binds versioned dependencies and their exact opt aliases."""
    prefix, package, opt = homebrew_fixture(
        tmp_path,
        monkeypatch,
        json.dumps({"runtime_dependencies": [{"full_name": "support"}]}),
    )
    support = prefix / "Cellar" / "support" / "2.0"
    support.mkdir(parents=True)
    (opt / "support").symlink_to(support, target_is_directory=True)

    result = lookup(registry, "sampletool")["result"]

    assert result["candidate_read_roots"] == [str(package), str(support)]
    assert result["grant_reference"]


def test_homebrew_lookup_accepts_installed_package_without_opt_alias(
    tmp_path, monkeypatch, registry
):
    """A versioned package can use its canonical root without an optional opt link."""
    _, package, opt = homebrew_fixture(tmp_path, monkeypatch)
    (opt / "sample").unlink()

    result = lookup(registry, "sampletool")["result"]

    assert result["candidate_read_roots"] == [str(package)]


def test_lookup_under_incomplete_cellar_layout_keeps_executable_directory(
    tmp_path, monkeypatch, registry
):
    """A path named Cellar without a version is treated as a plain installation."""
    cellar = tmp_path / "Cellar"
    cellar.mkdir()
    executable = cellar / "sampletool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(cellar))

    assert lookup(registry, "sampletool")["result"]["candidate_read_roots"] == [str(cellar)]


@pytest.mark.parametrize(
    "receipt",
    [
        "x" * 131_073,
        json.dumps({"runtime_dependencies": "wrong"}),
        json.dumps({"runtime_dependencies": [{}]}),
        json.dumps({"runtime_dependencies": [{"full_name": "../private"}]}),
        json.dumps({"runtime_dependencies": [{"full_name": "missing"}]}),
        "not json",
    ],
)
def test_homebrew_lookup_rejects_unsafe_or_unusable_dependency_metadata(
    tmp_path, monkeypatch, registry, receipt
):
    """An invalid package receipt never widens a sandboxed grant."""
    homebrew_fixture(tmp_path, monkeypatch, receipt)

    assert lookup(registry, "sampletool")["problem"]["code"] == "executable.invalid_lookup"


def test_homebrew_lookup_rejects_dependency_outside_its_package(tmp_path, monkeypatch, registry):
    """A receipt cannot turn an unrelated opt alias into a tool support grant."""
    _, _, opt = homebrew_fixture(
        tmp_path,
        monkeypatch,
        json.dumps({"runtime_dependencies": [{"full_name": "support"}]}),
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (opt / "support").symlink_to(unrelated, target_is_directory=True)

    assert lookup(registry, "sampletool")["problem"]["code"] == "executable.invalid_lookup"


def test_lookup_cannot_issue_a_grant_for_an_uninspectable_installation(
    tmp_path, monkeypatch, registry
):
    """A root identity failure leaves an installed tool without a usable grant."""
    _, package, _ = homebrew_fixture(tmp_path, monkeypatch)
    original = Path.lstat

    def inaccessible(path):
        """Simulate loss of metadata access at grant binding."""
        if path == package:
            raise OSError("unavailable")
        return original(path)

    monkeypatch.setattr(Path, "lstat", inaccessible)
    result = lookup(registry, "sampletool")["result"]
    assert result["sandbox_access"] == "external_grant_unavailable"
    assert "grant_reference" not in result


def test_homebrew_alias_change_invalidates_bound_reference(tmp_path, monkeypatch, registry):
    """Replacing a package opt alias cannot reuse a previous executable reference."""
    _, _, opt = homebrew_fixture(tmp_path, monkeypatch)
    reference = lookup(registry, "sampletool")["result"]["grant_reference"]
    (opt / "sample").unlink()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (opt / "sample").symlink_to(replacement, target_is_directory=True)

    output = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "sampletool", "executable_grant": reference}),
            interaction=ApprovingInteraction(),
        )
    )
    assert output["problem"]["code"] == "sandbox.unavailable"
    (opt / "sample").unlink()
    missing = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "sampletool", "executable_grant": reference}),
            interaction=ApprovingInteraction(),
        )
    )
    assert missing["problem"]["code"] == "sandbox.unavailable"


def test_lookup_reference_requires_unchanged_path_and_live_target(tmp_path, monkeypatch, registry):
    """A reference is rejected when its PATH snapshot, executable, or token is stale."""
    _, package, _ = homebrew_fixture(tmp_path, monkeypatch)
    reference = lookup(registry, "sampletool")["result"]["grant_reference"]

    def run(grant):
        """Send one public request using a previously returned reference."""
        return json.loads(
            registry.call(
                "run_command",
                json.dumps({"command": "sampletool", "executable_grant": grant}),
                interaction=ApprovingInteraction(),
            )
        )

    assert run("unknown")["problem"]["code"] == "sandbox.unavailable"
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert run(reference)["problem"]["code"] == "sandbox.unavailable"
    monkeypatch.setenv("PATH", str(package / "bin"))
    (package / "bin" / "sampletool").unlink()
    assert run(reference)["problem"]["code"] == "sandbox.unavailable"


def test_lookup_limits_live_references(tmp_path, monkeypatch, registry):
    """Old lookup references expire as the bounded service table fills."""
    homebrew_fixture(tmp_path, monkeypatch)
    first = lookup(registry, "sampletool")["result"]["grant_reference"]
    for _ in range(128):
        assert lookup(registry, "sampletool")["result"]["grant_reference"]
    output = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "sampletool", "executable_grant": first}),
            interaction=ApprovingInteraction(),
        )
    )
    assert output["problem"]["code"] == "sandbox.unavailable"


def test_lookup_never_grants_a_protected_external_tool(tmp_path, monkeypatch, registry):
    """An installed executable below a private directory is not requestable."""
    protected = tmp_path / ".ssh"
    protected.mkdir()
    executable = protected / "secret-tool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(protected))

    result = lookup(registry, "secret-tool")["result"]

    assert result["sandbox_access"] == "protected_external_path"
    assert "grant_reference" not in result


def test_external_lookup_reference_round_trips_through_public_virtual_boundary(
    tmp_path, monkeypatch
):
    """Request a bound tool grant while preserving command output without a host retry."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    installation = tmp_path / "installation"
    installation.mkdir()
    executable = installation / "sampletool"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(installation))
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, scratch))
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=0,
        stderr=f"loaded {installation}/sampletool",
    )
    host = MagicMock()
    service = CommandExecutionService(backend, host)
    interaction = MagicMock()
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(
            workspace, audit_path=tmp_path / "audit.db", workspace_id="workspace"
        ),
        execution_service=service,
    )

    looked_up = json.loads(
        registry.call(
            "resolve_executable",
            json.dumps({"name": "sampletool"}),
            instructions_manager=instructions,
        )
    )["result"]
    assert looked_up["sandbox_access"] == "external_read_grant_required"
    assert looked_up["path"] == "<external>"
    assert "candidate_read_roots" not in looked_up
    assert str(tmp_path) not in json.dumps(looked_up)
    payload = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "sampletool", "executable_grant": looked_up["grant_reference"]}),
            instructions_manager=instructions,
            interaction=interaction,
        )
    )
    assert payload["result"]["exit_code"] == 0
    assert payload["result"]["stderr"]["content"] == f"loaded {installation}/sampletool"
    assert backend.run.call_args.args[0].read_roots == (installation,)
    assert backend.run.call_args.args[0].automatic_tool_reads == ()
    assert "Read the installed sampletool tool folder" in interaction.info.call_args.args[0]
    interaction.prompt.assert_called_once()
    host.run.assert_not_called()
    with closing(sqlite3.connect(tmp_path / "audit.db")) as connection:
        row = connection.execute(
            "SELECT payload_json FROM permission_audit_records "
            "WHERE event_name = 'sandbox.permission_scope'"
        ).fetchone()
    assert json.loads(row[0])["automatic_tool_reads"] == []


def test_managed_toolchain_binds_named_tools_and_explicit_children(tmp_path, monkeypatch):
    """Unrelated workspace PATH symlinks do not grant their managed installation roots."""
    scope_receipt_opener(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    binaries = workspace / "bin"
    binaries.mkdir(parents=True)
    cellar = tmp_path / "brew" / "Cellar"
    roots = []
    for name in ("sampletool", "childtool"):
        package = cellar / name / "1.0"
        target = package / "bin" / name
        target.parent.mkdir(parents=True)
        target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        target.chmod(0o755)
        (package / "INSTALL_RECEIPT.json").write_text("{}", encoding="utf-8")
        roots.append(package)
        (binaries / name).symlink_to(target)
    monkeypatch.setenv("PATH", str(binaries))
    backend = MagicMock()
    backend.installed_tool_roots.side_effect = MacOSSeatbeltBackend.installed_tool_roots
    backend.managed_tool_root.side_effect = lambda root: (
        root.is_relative_to(cellar) and (root / "INSTALL_RECEIPT.json").is_file()
    )
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    host = MagicMock()
    interaction = MagicMock()
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, host),
    )

    result, problem = call(registry, "sampletool", cwd=str(workspace), interaction=interaction)

    assert problem is None
    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.automatic_tool_reads == (roots[0],)
    assert request.read_roots == (roots[0],)
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()

    direct, problem = call(
        registry, "sampletool; childtool", cwd=str(workspace), interaction=interaction
    )
    assert problem is None and direct["exit_code"] == 0
    assert set(backend.run.call_args.args[0].automatic_tool_reads) == set(roots)

    child = lookup(registry, "childtool", cwd=str(workspace))["result"]["grant_reference"]
    result, problem = call(
        registry,
        "sampletool",
        cwd=str(workspace),
        interaction=interaction,
        executable_grant=child,
    )
    assert problem is None and result["exit_code"] == 0
    request = backend.run.call_args.args[0]
    assert set(request.automatic_tool_reads) == set(roots)
    assert set(request.read_roots) == set(roots)
    interaction.prompt.assert_not_called()


def test_managed_toolchain_discovery_rejects_unsafe_or_unavailable_candidates(
    tmp_path, monkeypatch
):
    """A failed managed lookup adds no installation-root authority to a command."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    binaries = tmp_path / "tools"
    binaries.mkdir()
    executable = binaries / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(binaries))
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    service = CommandExecutionService(backend, MagicMock())
    monkeypatch.setattr(service, "resolve_executable", MagicMock(side_effect=ValueError("bad")))
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=service,
    )

    result, problem = call(registry, "tool", cwd=str(workspace))

    assert problem is None and result["exit_code"] == 0
    request = backend.run.call_args.args[0]
    assert request.read_roots == () and request.automatic_tool_reads == ()
    assert request.executable_identities[0][0] == "tool"


def test_public_extra_read_contexts_and_protected_root_fail_closed(tmp_path):
    """Public requests distinguish narrow and broad reads and reject protected roots prelaunch."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    protected = external / ".ssh"
    protected.mkdir()
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, None))
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = MagicMock()
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )

    for root, expected in (
        (str(external), "Read an additional folder"),
        (str(tmp_path), "Read a broad area"),
    ):
        output = json.loads(
            registry.call(
                "run_command",
                json.dumps({"command": "/usr/bin/true", "read_roots": [root]}),
                interaction=interaction,
                instructions_manager=instructions,
            )
        )
        assert "result" in output, output
        assert output["result"]["exit_code"] == 0
        assert expected in interaction.info.call_args.args[0]
        scope = json.loads(
            interaction.info.call_args.args[0].split("\nScope: ", 1)[1].split("\n", 1)[0]
        )
        assert root in scope["read_roots"]
    before = backend.run.call_count
    output = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "/usr/bin/true", "read_roots": [str(protected)]}),
            interaction=interaction,
            instructions_manager=instructions,
        )
    )
    assert output["problem"]["code"] == "sandbox.unavailable"
    assert backend.run.call_count == before


def test_homebrew_grants_run_without_prompt_but_manual_read_still_prompts(tmp_path, monkeypatch):
    """Bound tool roots run in the sandbox while an additional data root still asks."""
    _, package, _ = homebrew_fixture(tmp_path, monkeypatch)
    (package / "INSTALL_RECEIPT.json").write_text("{}", encoding="utf-8")
    second = package / "bin" / "othertool"
    second.write_text("#!/bin/sh\n", encoding="utf-8")
    second.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    extra = tmp_path / "data"
    extra.mkdir()
    backend = MagicMock()
    backend.installed_tool_roots.side_effect = MacOSSeatbeltBackend.installed_tool_roots
    backend.managed_tool_root.side_effect = lambda root: (
        root.is_relative_to(package.parent.parent) and (root / "INSTALL_RECEIPT.json").is_file()
    )
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = MagicMock()
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )
    tools = lookup(registry, ["sampletool", "othertool"])["result"]["executables"]
    first_ref = tools[0]["grant_reference"]
    second_ref = tools[1]["grant_reference"]

    for grants, manual, expected, roots in (
        (first_ref, None, None, (package,)),
        ([first_ref, second_ref], None, None, (package,)),
        (
            first_ref,
            [str(extra)],
            "Use installed tools and read an additional folder",
            (package, extra),
        ),
    ):
        output = json.loads(
            registry.call(
                "run_command",
                json.dumps(
                    {
                        "command": "sampletool",
                        "cwd": str(workspace),
                        "executable_grant": grants,
                        "read_roots": manual,
                    }
                ),
                interaction=interaction,
            )
        )
        assert "result" in output, output
        assert output["result"]["exit_code"] == 0
        if expected is None:
            interaction.info.assert_not_called()
            interaction.prompt.assert_not_called()
        else:
            assert expected in interaction.info.call_args.args[0]
            interaction.prompt.assert_called_once()
        assert backend.run.call_args.args[0].read_roots == roots
        assert backend.run.call_args.args[0].automatic_tool_reads == (package,)
        interaction.reset_mock()


def test_managed_tool_receipt_is_rechecked_before_native_launch(tmp_path, monkeypatch):
    """A stale package receipt cannot preserve prompt-free executable authority."""
    _, package, _ = homebrew_fixture(tmp_path, monkeypatch)
    receipt = package / "INSTALL_RECEIPT.json"
    receipt.write_text("{}", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backend = MagicMock()
    backend.installed_tool_roots.side_effect = MacOSSeatbeltBackend.installed_tool_roots
    backend.managed_tool_root.side_effect = lambda root: (
        root == package and (root / "INSTALL_RECEIPT.json").is_file()
    )
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )
    reference = lookup(registry, "sampletool")["result"]["grant_reference"]
    receipt.unlink()

    _, problem = call(registry, "sampletool", cwd=str(workspace), executable_grant=reference)

    assert problem["code"] == "sandbox.unavailable"
    backend.run.assert_not_called()


def test_unverified_cellar_name_needs_an_additional_read_decision(tmp_path, monkeypatch):
    """A Cellar-shaped directory without trusted provenance never receives automatic reads."""
    homebrew_fixture(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = MagicMock()
    interaction.prompt.return_value = "deny"
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )
    reference = lookup(registry, "sampletool")["result"]["grant_reference"]
    _, problem = call(
        registry,
        "sampletool",
        cwd=str(workspace),
        interaction=interaction,
        executable_grant=reference,
    )
    assert problem["code"] == "tool.denied"
    assert "Read the installed sampletool tool folder" in interaction.info.call_args.args[0]
    backend.run.assert_not_called()


def test_external_reference_requires_approval_even_for_unrelated_command(tmp_path, monkeypatch):
    """A bound arbitrary tool directory never becomes silent general read authority."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    executable = installation / "reviewtool"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    secret = installation / "unrelated.txt"
    secret.write_text("private marker", encoding="utf-8")
    monkeypatch.setenv("PATH", str(installation))
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, scratch))
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )
    result = json.loads(
        registry.call(
            "resolve_executable",
            json.dumps({"name": "reviewtool"}),
            instructions_manager=instructions,
        )
    )["result"]
    assert str(installation) not in json.dumps(result)
    source = f"/bin/cat {secret}"
    request = {"command": source, "read_only": True, "executable_grant": result["grant_reference"]}
    denied = MagicMock()
    denied.prompt.return_value = "deny"
    output = json.loads(
        registry.call(
            "run_command",
            json.dumps(request),
            instructions_manager=instructions,
            interaction=denied,
        )
    )
    assert output["problem"]["code"] == "tool.denied"
    denied.prompt.assert_called_once()
    backend.run.assert_not_called()

    approved = MagicMock()
    approved.prompt.return_value = "approve"
    output = json.loads(
        registry.call(
            "run_command",
            json.dumps(request),
            instructions_manager=instructions,
            interaction=approved,
        )
    )
    assert output["result"]["boundary"] == "sandbox"
    assert backend.run.call_args.args[0].automatic_tool_reads == ()
    assert backend.run.call_args.args[0].read_roots == (installation,)
    approved.prompt.assert_called_once()


def test_external_lookup_reference_rejects_changed_target(tmp_path, monkeypatch):
    """A changed executable cannot inherit approval from a prior host lookup."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    executable = installation / "sampletool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(installation))
    backend = MagicMock()
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )
    reference = lookup(registry, "sampletool")["result"]["grant_reference"]
    executable.rename(installation / "original-executable")
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    output = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "sampletool", "executable_grant": reference}),
            interaction=ApprovingInteraction(),
        )
    )
    assert output["problem"]["code"] == "sandbox.unavailable"
    backend.run.assert_not_called()


def test_run_command_explains_external_tool_read_authority(registry):
    """The public schema tells the agent how to request access to installed tools."""
    definition = next(item for item in registry.definitions() if item.name == "run_command")

    description = definition.parameters["properties"]["read_roots"]["description"]
    assert "executable and support files" in description
    assert "An identifiable scope is requested in this call" in description
    assert "when it cannot be identified" in description
    assert "Unused PATH entries grant lookup metadata only" in description
    assert "that exact file without its neighbors" in description


def test_run_command_explains_tool_temporary_directory_settings(registry):
    """The public schema explains session scratch without assuming a particular tool."""
    definition = next(item for item in registry.definitions() if item.name == "run_command")

    description = definition.parameters["properties"]["command"]["description"]
    assert f'"{VirtualPath.TEMPORARY}/..."' in description
    assert "Configure each tool's output, cache and environment updates" in description
    assert "Do not rerun a command that may have had effects" in description


def test_platform_command_capabilities_shape_live_model_definitions(monkeypatch):
    """Native support shapes model fields and places nullable guidance on each argument."""
    backend = MagicMock(capabilities=frozenset())
    service = CommandExecutionService(backend, MagicMock())
    registry = ToolRegistry(
        [
            ToolRegistration(run_command_tool, name="sandbox_shell"),
            ToolRegistration(control_command_terminal_tool, name="terminal_control"),
            read_command_job_tool,
        ],
        execution_service=service,
    )

    unavailable = {item.name: item for item in registry.definitions()}
    properties = unavailable["sandbox_shell"].parameters["properties"]
    assert "interactive" not in properties
    assert "egress_origins" not in properties
    assert "interactive" not in unavailable["sandbox_shell"].parameters["required"]
    assert "egress_origins" not in unavailable["sandbox_shell"].parameters["required"]
    assert "terminal_control" not in unavailable
    assert "read_command_job" in unavailable
    assert "until result is non-null, then stop" in unavailable["read_command_job"].description
    wait = unavailable["read_command_job"].parameters["properties"]["wait_seconds"]
    assert "use 30 to avoid empty polls" in wait["description"]
    assert "terminal_control" in registry.names

    monkeypatch.setattr(
        "loop.execution.facade.select_sandbox_backend",
        lambda: MagicMock(
            capabilities=frozenset({SandboxCapability.TERMINAL, SandboxCapability.MEDIATED_EGRESS})
        ),
    )
    service.refresh_backend()
    supported = {item.name: item for item in registry.view().definitions()}
    assert "interactive" in supported["sandbox_shell"].parameters["properties"]
    assert "egress_origins" in supported["sandbox_shell"].parameters["properties"]
    assert "Run sandboxed shell source" in supported["sandbox_shell"].description
    assert "JSON null" not in supported["sandbox_shell"].description
    for name in ("read_roots", "write_roots", "executable_grant", "egress_origins", "reason"):
        field = supported["sandbox_shell"].parameters["properties"][name]
        assert "JSON null (unquoted)" in field["description"]
        assert {"type": "null"} in field["anyOf"]
    assert (
        supported["sandbox_shell"]
        .parameters["properties"]["egress_origins"]["description"]
        .startswith("Exact HTTP(S) origins requested")
    )
    egress_description = supported["sandbox_shell"].parameters["properties"]["egress_origins"][
        "description"
    ]
    assert "The native mediator configures a proxy" in egress_description
    assert "macOS" not in egress_description
    assert "uv" not in egress_description
    assert (
        "Cannot combine with egress_origins"
        in supported["sandbox_shell"].parameters["properties"]["interactive"]["description"]
    )
    assert "terminal_control" in supported
    assert (
        "resize needs rows and columns"
        in supported["terminal_control"].parameters["properties"]["action"]["description"]
    )
    assert "interactive" not in unavailable["sandbox_shell"].parameters["properties"]


def test_run_command_accepts_opaque_shell_and_returns_bounded_result(monkeypatch, registry):
    """The public tool preserves shell source and returns a successful sandbox result."""
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="hello\n", stderr="warning\n"
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)

    result, problem = call(registry, "printf hello | cat; /bin/sh -c 'printf child'")

    assert problem is None
    assert result["exit_code"] == 0
    assert result["stdout"]["content"] == "hello\n"
    assert backend.run.call_args.args[0].source.endswith("'printf child'")
    environment = dict(backend.run.call_args.args[0].environment)
    assert environment.keys() == {
        "LANG",
        "PATH",
        "TMPDIR",
        "TMP",
        "TEMP",
        "WORKSPACE_ROOT",
        "XDG_CACHE_HOME",
        "HOME",
    }
    assert Path(environment["XDG_CACHE_HOME"]).parent == Path(environment["TMPDIR"])
    assert environment["HOME"] == environment["TMPDIR"]
    assert Path(environment["TMPDIR"]).is_dir()
    assert environment["TMP"] == environment["TEMP"] == environment["TMPDIR"]


@pytest.mark.parametrize("spelling", ["None", "nOnE", "null", "NULL"])
def test_command_optional_capability_null_spelling_keeps_narrow_sandbox_authority(
    tmp_path, monkeypatch, spelling
):
    """Optional null spellings add no scope and retain sandbox authorization."""
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    host = MagicMock()
    registry = ToolRegistry(
        [run_command_tool],
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(backend, host),
    )
    interaction = MagicMock()
    result, problem = call(
        registry,
        "printf ready",
        cwd=str(tmp_path),
        interaction=interaction,
        read_only=True,
        read_roots=spelling,
        write_roots=spelling,
        executable_grant=spelling,
        egress_origins=spelling,
    )
    assert problem is None
    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.read_roots == request.write_roots == request.egress_origins == ()
    assert request.network is False
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("spelling", ["None", "nOnE", "null", "NULL"])
def test_command_null_spelling_does_not_override_a_process_deny(tmp_path, asynchronous, spelling):
    """A process denial remains binding for both synchronous and asynchronous dispatch."""
    permissions = PermissionManager(tmp_path)
    permissions.set_default(Action.PROCESS_EXECUTE, Decision.DENY, scope=PolicyScope.SESSION)
    backend, host = MagicMock(), MagicMock()
    registry = ToolRegistry(
        [run_command_tool],
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, host),
    )
    arguments = json.dumps({"command": ":", "cwd": str(tmp_path), "read_roots": spelling})
    raw = (
        asyncio.run(registry.call_async("run_command", arguments))
        if asynchronous
        else registry.call("run_command", arguments)
    )
    assert json.loads(raw)["problem"]["code"] == "tool.denied"
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()


def test_command_null_spelling_never_rewrites_other_values():
    """Only exact top-level optional capability sentinels normalize to null."""
    model = get_tool_arguments_model(run_command_tool, "run_command")
    parsed = model.model_validate(
        {
            "command": "None",
            "cwd": "None",
            "read_roots": ["None"],
            "write_roots": ["None"],
            "executable_grant": ["None"],
            "egress_origins": ["None"],
            "reason": "None",
        }
    )
    assert parsed.command == parsed.cwd == parsed.reason == "None"
    assert parsed.read_roots == parsed.write_roots == ["None"]
    assert parsed.executable_grant == parsed.egress_origins == ["None"]
    with pytest.raises(ValueError):
        model.model_validate({"command": ":", "read_roots": " None "})


@pytest.mark.parametrize("approved", [False, True])
def test_command_null_spelling_preserves_scoped_read_approval(tmp_path, approved):
    """An explicit read root still requires its own approval alongside null spellings."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    host = MagicMock()
    registry = ToolRegistry(
        [run_command_tool],
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, host),
    )
    interaction = MagicMock()
    interaction.prompt.return_value = "approve" if approved else "deny"
    result, problem = call(
        registry,
        ":",
        cwd=str(workspace),
        interaction=interaction,
        read_only=True,
        read_roots=[str(external)],
        write_roots="None",
        executable_grant="None",
        egress_origins="None",
    )
    interaction.prompt.assert_called_once()
    if approved:
        assert problem is None and result["boundary"] == "sandbox"
        request = backend.run.call_args.args[0]
        assert request.read_roots == (external,)
        assert request.write_roots == request.egress_origins == ()
    else:
        assert result is None and problem["code"] == "tool.denied"
        backend.run.assert_not_called()
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize("field", ["executable_grant", "egress_origins"])
def test_command_null_spelling_does_not_authorize_an_invalid_explicit_capability(tmp_path, field):
    """An invalid real grant or origin remains refused even beside a null spelling."""
    backend = MagicMock(capabilities=frozenset({SandboxCapability.MEDIATED_EGRESS}))
    host = MagicMock()
    registry = ToolRegistry(
        [run_command_tool],
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(backend, host),
    )
    invalid = "not-a-valid-grant" if field == "executable_grant" else ["None"]
    result, problem = call(registry, ":", cwd=str(tmp_path), read_roots="None", **{field: invalid})
    assert result is None
    assert problem is not None
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize(
    "outcome", [SandboxOutcome.COMPLETED, SandboxOutcome.CANCELLED, SandboxOutcome.TIMED_OUT]
)
def test_command_working_files_survive_success_interruption_and_timeout(
    monkeypatch, registry, outcome
):
    """Finishing or interrupting an execution preserves its working files for the next command."""
    directories = []

    def execute(request):
        """Write through the bound TMPDIR and return the requested process outcome."""
        directory = Path(request.shell_environment()["TMPDIR"])
        directories.append(directory)
        file = directory / "draft"
        if file.exists():
            assert file.read_text() == "retained"
        file.write_text("retained")
        return CommandProcessResult(
            outcome, exit_code=0 if outcome is SandboxOutcome.COMPLETED else None
        )

    backend = MagicMock()
    backend.run.side_effect = execute
    monkeypatch.setattr(registry._execution_service, "_backend", backend)

    for _ in range(2):
        result, problem = call(registry, ":", read_only=True)
        if outcome is SandboxOutcome.COMPLETED:
            assert problem is None
            assert result["scratch"]["lifetime"] == "this session"
        else:
            assert problem["code"] == f"sandbox.{outcome.value}"
        assert (directories[-1] / "draft").read_text() == "retained"
    assert directories[0] == directories[1]


def test_read_only_command_uses_a_nonwritable_workspace_policy(monkeypatch, registry):
    """A default read-only command runs without approval or workspace writes."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    interaction = MagicMock()
    interaction.prompt.return_value = "approve"

    result, problem = call(registry, "cat file", interaction=interaction, read_only=True)

    assert problem is None
    assert result["boundary"] == "sandbox"
    assert result["scratch"]["path"] == VirtualPath.TEMPORARY
    assert result["scratch"]["lifetime"] == "this session"
    assert backend.run.call_args.args[0].write_roots == ()
    environment = dict(backend.run.call_args.args[0].environment)
    assert set(environment) == {
        LANG_ENVIRONMENT_VARIABLE,
        PATH_ENVIRONMENT_VARIABLE,
        TEMP_ENVIRONMENT_VARIABLE,
        TMP_ENVIRONMENT_VARIABLE,
        TMPDIR_ENVIRONMENT_VARIABLE,
        WORKSPACE_ROOT_ENVIRONMENT_VARIABLE,
        XDG_CACHE_HOME_ENVIRONMENT_VARIABLE,
        HOME_ENVIRONMENT_VARIABLE,
    }
    interaction.prompt.assert_not_called()


def test_known_read_command_defaults_to_read_only(monkeypatch, registry):
    """An exact inspection command receives no automatic write restriction."""
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    result, problem = call(registry, "git diff --staged")
    assert problem is None
    assert result["boundary"] == "sandbox"
    assert backend.run.call_args.args[0].write_roots


def test_read_only_cannot_request_git_write(tmp_path):
    """Contradictory read and Git write grants fail before execution."""
    registry = native_registry()
    _, problem = call(registry, "git add file", cwd=str(tmp_path), read_only=True, git_write=True)
    assert problem["code"] == "sandbox.unavailable"


def test_public_git_write_asks_before_launch_without_explicit_flag(monkeypatch, registry, tmp_path):
    """A direct Git mutation requests its metadata grant before any sandbox child starts."""
    (tmp_path / ".git").mkdir()
    backend = MagicMock()
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    interaction = MagicMock()
    interaction.prompt.return_value = "deny"

    _, problem = call(registry, "git add file", cwd=str(tmp_path), interaction=interaction)

    assert problem["code"] == "tool.denied"
    assert "change Git metadata" in interaction.info.call_args.args[0]
    interaction.prompt.assert_called_once()
    backend.run.assert_not_called()


def test_public_command_uses_composed_inspection_for_grant_and_review(monkeypatch, tmp_path):
    """One injected matcher drives both Git authority planning and the approval prompt."""

    def match_special(command: list[str]) -> CommandFinding | None:
        """Classify the chosen test command as needing Git metadata authority."""
        if command != ["printf", "special"]:
            return None
        return CommandFinding(
            policy_id="special",
            status=CommandReviewStatus.FRESH,
            context="Review the special command",
            reason="run the special command",
            requests_git_write=True,
        )

    (tmp_path / ".git").mkdir()
    manager = PermissionManager(
        command_inspection=CommandInspection(()).with_matcher(match_special)
    )
    registry = native_registry(permission_manager=manager)
    backend = MagicMock()
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    interaction = MagicMock()
    interaction.prompt.return_value = "deny"

    _, problem = call(registry, "printf special", cwd=str(tmp_path), interaction=interaction)

    assert problem["code"] == "tool.denied"
    assert "change Git metadata" in interaction.info.call_args.args[0]
    assert "Review the special command" in interaction.info.call_args.args[0]
    assert "run the special command" in interaction.info.call_args.args[0]
    backend.run.assert_not_called()


@pytest.mark.parametrize(
    "source",
    [
        "env rm -f victim",
        "env -i /bin/rm victim",
        "env -u HOME -- /bin/rm victim",
        "env --unset=HOME /bin/rm victim",
        "env -i /usr/bin/git clean -fd",
        "env -u HOME -- /usr/bin/git clean -fd",
        "env --unset=HOME /usr/bin/git clean -fd",
        "FOO=1 rm -f victim",
        ": > victim",
        "printf text > victim",
        "printf text &> victim",
        "printf text >& victim",
        "printf text 1>& victim",
        "printf text 2&> victim",
        "env git add victim",
        "FOO=1 git add victim; echo done",
    ],
)
def test_public_direct_destructive_commands_need_fresh_approval(
    monkeypatch, registry, tmp_path, source
):
    """Direct wrappers and truncation ask before launch and cannot reuse an earlier approval."""
    victim = tmp_path / "victim"
    victim.write_text("safe", encoding="utf-8")
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    interaction = MagicMock()
    interaction.prompt.side_effect = ["approve", "deny"]
    first, problem = call(registry, source, cwd=str(tmp_path), interaction=interaction)
    assert problem is None
    assert first["boundary"] == "sandbox"
    assert backend.run.call_count == 1
    _, problem = call(registry, source, cwd=str(tmp_path), interaction=interaction)
    assert problem["code"] == "tool.denied"
    assert backend.run.call_count == 1
    assert victim.read_text(encoding="utf-8") == "safe"
    assert interaction.prompt.call_count == 2
    assert all(
        "similar_session" not in item.kwargs["choices"]
        for item in interaction.prompt.call_args_list
    )


def test_git_grant_resolves_linked_worktree_metadata(monkeypatch, registry, tmp_path):
    """A linked worktree grant binds its real metadata directory for read and write."""
    gitdir = tmp_path / "main" / ".git" / "worktrees" / "linked"
    gitdir.mkdir(parents=True)
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    common = tmp_path / "main" / ".git"
    (common / "HEAD").write_text("ref: refs/heads/main\n")
    (gitdir / "commondir").write_text("../..\n")
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / ".git").write_text(f"gitdir: {gitdir}\n")
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    result, problem = call(registry, "git add file", cwd=str(linked), git_write=True)
    assert problem is None
    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.write_roots == (linked, gitdir, common)
    assert request.read_roots == (gitdir, common)


def test_git_grant_binds_absent_metadata_creation(monkeypatch, registry, tmp_path):
    """A fresh workspace can approve Git initialization without precreating `.git`."""
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    result, problem = call(registry, "git init", cwd=str(tmp_path), git_write=True)
    assert problem is None
    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.git_create is True
    assert request.write_roots == (tmp_path,)
    assert not (tmp_path / ".git").exists()


def test_git_creation_rejects_symlinked_metadata_directory(tmp_path):
    """A linked directory cannot replace the absent Git creation target."""
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / ".git").symlink_to(external, target_is_directory=True)
    registry = native_registry()
    _, problem = call(registry, "git init", cwd=str(tmp_path), git_write=True)
    assert problem["code"] == "sandbox.unavailable"


def test_linked_git_status_requests_metadata_read_grant(monkeypatch, registry, tmp_path):
    """A linked worktree status cannot silently read metadata outside the workspace."""
    gitdir = tmp_path / "main" / ".git" / "worktrees" / "linked"
    gitdir.mkdir(parents=True)
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / ".git").write_text(f"gitdir: {gitdir}\n")
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    interaction = MagicMock()
    interaction.prompt.return_value = "approve"
    _, problem = call(registry, "git status", cwd=str(linked), interaction=interaction)
    assert problem is None
    request = backend.run.call_args.args[0]
    assert request.write_roots == (linked,)
    assert request.read_roots == (gitdir,)
    assert "read files outside this workspace" in interaction.info.call_args.args[0]
    scope = json.loads(
        interaction.info.call_args.args[0].split("\nScope: ", 1)[1].split("\n", 1)[0]
    )
    assert str(gitdir) in scope["read_roots"]


@pytest.mark.parametrize("pointer", ["invalid", "gitdir: missing"])
def test_invalid_linked_worktree_grant_fails_before_approval(tmp_path, pointer):
    """An invalid linked pointer cannot silently grant an unrelated directory."""
    (tmp_path / ".git").write_text(pointer)
    registry = native_registry()
    _, problem = call(registry, "git add file", cwd=str(tmp_path), git_write=True)
    assert problem["code"] == "sandbox.unavailable"


def test_linked_worktree_grant_rejects_missing_head_and_symlink_pointer(tmp_path):
    """Git pointer names must be regular files and lead to usable metadata."""
    gitdir = tmp_path / "gitdir"
    gitdir.mkdir()
    pointer = tmp_path / ".git"
    pointer.write_text(f"gitdir: {gitdir}\n")
    registry = native_registry()
    _, problem = call(registry, "git add file", cwd=str(tmp_path), git_write=True)
    assert problem["code"] == "sandbox.unavailable"
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    pointer.rename(tmp_path / "git-pointer")
    pointer.symlink_to(tmp_path / "git-pointer")
    _, problem = call(registry, "git add file", cwd=str(tmp_path), git_write=True)
    assert problem["code"] == "sandbox.unavailable"


@pytest.mark.parametrize("common_pointer", ["missing", "no-head", "symlink"])
def test_linked_worktree_grant_rejects_invalid_common_metadata(tmp_path, common_pointer):
    """The shared Git metadata directory must be a real reviewed directory."""
    gitdir = tmp_path / "gitdir"
    gitdir.mkdir()
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / ".git").write_text(f"gitdir: {gitdir}\n")
    if common_pointer == "missing":
        (gitdir / "commondir").write_text("../missing\n")
    elif common_pointer == "no-head":
        (tmp_path / "common-empty").mkdir()
        (gitdir / "commondir").write_text("../common-empty\n")
    else:
        target = tmp_path / "common-target"
        target.mkdir()
        (target / "HEAD").write_text("ref: refs/heads/main\n")
        (gitdir / "commondir").symlink_to(target / "HEAD")
    registry = native_registry()
    _, problem = call(registry, "git add file", cwd=str(tmp_path), git_write=True)
    assert problem["code"] == "sandbox.unavailable"


def test_git_write_grant_uses_regular_metadata_directory(monkeypatch, registry, tmp_path):
    """A conventional repository keeps its Git grant under the workspace."""
    gitdir = tmp_path / ".git"
    gitdir.mkdir()
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    _, problem = call(registry, "git add file", cwd=str(tmp_path), git_write=True)
    assert problem is None
    assert backend.run.call_args.args[0].write_roots == (tmp_path, gitdir)


def test_run_command_preserves_nonzero_exit(monkeypatch, registry):
    """An ordinary shell nonzero exit does not offer host execution."""
    monkeypatch.setattr(
        registry._execution_service,
        "_backend",
        MagicMock(
            run=MagicMock(
                return_value=CommandProcessResult(
                    SandboxOutcome.COMPLETED,
                    exit_code=17,
                )
            )
        ),
    )

    _, problem = call(registry, "exit 17")

    assert problem["code"] == "process.nonzero_exit"
    assert problem["metadata"]["exit_code"] == 17
    assert "host_offer" not in problem["metadata"]


def test_run_command_displays_escaped_explanation_for_requested_network(tmp_path):
    """Public dispatch presents the model explanation and actual network scope to the user."""
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    host = MagicMock()
    interaction = MagicMock()
    interaction.prompt.return_value = "approve"
    manager = PermissionManager(tmp_path)
    manager.set_limit("deny_private_networks", False)
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=manager,
        execution_service=CommandExecutionService(backend, host),
    )
    reason = "Fetch dependencies\n\x1b[2J"

    result, problem = call(
        registry, ":", cwd=str(tmp_path), interaction=interaction, network=True, reason=reason
    )

    assert problem is None
    message = interaction.info.call_args.args[0]
    scope_text = message.split("\nScope: ", 1)[1].split("\n", 1)[0]
    explanation = message.split("\nAgent explanation: ", 1)[1]
    assert json.loads(scope_text)["network"] is True
    assert json.loads(explanation) == reason
    assert "\x1b" not in message
    assert "Fetch dependencies" not in json.dumps(result)
    backend.run.assert_called_once()
    host.run_host_command.assert_not_called()


def test_run_command_reports_unqualified_egress_without_prompt_or_host_offer(tmp_path):
    """Exact origin syntax cannot open broad networking or trigger host recovery."""
    backend = MagicMock(capabilities=frozenset())
    host = MagicMock()
    interaction = MagicMock()
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(backend, host),
    )

    result, problem = call(
        registry,
        "printf safe",
        cwd=str(tmp_path),
        interaction=interaction,
        egress_origins=["https://EXAMPLE.com"],
    )

    assert result is None
    assert problem["code"] == "sandbox.unavailable"
    assert "qualified sandboxed HTTP proxy route" in problem["detail"]
    interaction.prompt.assert_not_called()
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()


def test_command_planner_rejects_nonlist_egress_scope():
    """Malformed callers cannot bypass exact-origin normalization in the planner."""
    with pytest.raises(ValueError, match="origins must be a list"):
        _command_plan({"command": "true", "cwd": ".", "egress_origins": "http://example.com"})


def test_run_command_binds_empty_and_relative_path_entries_from_command_cwd(
    monkeypatch, registry, tmp_path
):
    """Public command binding resolves empty and relative PATH entries from its cwd."""
    workspace = tmp_path / "workspace"
    (workspace / "bin").mkdir(parents=True)
    external = tmp_path / "external-bin"
    external.mkdir()
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    monkeypatch.setenv("PATH", ":bin:../external-bin")

    result, problem = call(registry, ":", cwd=str(workspace))

    assert problem is None
    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.shell_environment()["PATH"] == ":bin:../external-bin"
    assert request.path_roots == (
        (workspace, workspace),
        (workspace / "bin", workspace / "bin"),
        (external, external),
    )


def test_relative_one_character_tool_requires_explicit_external_read(tmp_path, monkeypatch):
    """A one-character tool on relative PATH runs only after its bound read grant."""
    workspace = tmp_path / "workspace"
    cwd = workspace / "subdir"
    cwd.mkdir(parents=True)
    installation = tmp_path / "external-bin"
    installation.mkdir()
    executable = installation / "x"
    executable.write_text("#!/bin/sh\nprintf external-ok\n", encoding="utf-8")
    executable.chmod(0o755)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    monkeypatch.setenv("PATH", "../../external-bin:/usr/bin:/bin")
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, scratch))
    backend = MagicMock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=1, stderr="blocked"
    )
    host = MagicMock()
    interaction = MagicMock()
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, host),
    )

    found = lookup(registry, "x", cwd="subdir", instructions_manager=instructions)["result"]
    assert found["status"] == "installed_on_host"
    assert found["sandbox_access"] == "external_read_grant_required"
    assert "candidate_read_roots" not in found
    assert str(installation) not in json.dumps(found)

    _, problem = call(
        registry,
        "x",
        cwd="subdir",
        read_only=True,
        interaction=interaction,
        instructions_manager=instructions,
    )
    assert problem["code"] == "process.nonzero_exit"
    assert backend.run.call_args.args[0].read_roots == ()
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()

    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="external-ok"
    )
    result, problem = call(
        registry,
        "x",
        cwd="subdir",
        read_only=True,
        executable_grant=found["grant_reference"],
        interaction=interaction,
        instructions_manager=instructions,
    )
    assert problem is None
    assert result["stdout"]["content"] == "external-ok"
    assert backend.run.call_args.args[0].read_roots == (installation,)
    assert "Read the installed x tool folder" in interaction.info.call_args.args[0]
    interaction.prompt.assert_called_once()
    host.run_host_command.assert_not_called()


def test_resolve_executable_binds_relative_lookup_cwd_and_rejects_missing_cwd(
    tmp_path, registry, reviewed_system_tool
):
    """Executable lookup resolves workspace cwd and returns typed failures for missing cwd."""
    workspace = tmp_path / "workspace"
    subdirectory = workspace / "subdirectory"
    subdirectory.mkdir(parents=True)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, scratch))

    service_context = ToolContext(
        interaction=None,
        tool_name="resolve_executable",
        instructions_manager=instructions,
    )
    result = registry._execution_service.resolve_executable(service_context, "true", "subdirectory")
    missing_without_context = lookup(registry, "true", cwd=str(tmp_path / "missing"))
    missing_with_context = lookup(
        registry, "true", cwd="missing", instructions_manager=instructions
    )

    assert result["status"] == "installed_on_host"
    assert result["sandbox_access"] == "default_read"
    assert missing_without_context["problem"]["code"] == "executable.invalid_lookup"
    assert missing_with_context["problem"]["code"] == "executable.invalid_lookup"


def test_request_binding_error_redacts_external_paths_without_instructions(
    monkeypatch, registry, tmp_path
):
    """A late request-binding failure keeps external grant paths out of model output."""
    external = tmp_path / "external-read"
    external.mkdir()
    monkeypatch.setattr(
        "loop.execution.facade.SandboxRequest.create",
        MagicMock(side_effect=ValueError(f"request failed at {external}")),
    )

    _, problem = call(registry, ":", read_roots=[str(external)])

    assert problem["code"] == "sandbox.unavailable"
    assert str(external) not in problem["detail"]
    assert "<external>" in problem["detail"]


def test_run_command_preserves_system_path_in_redacted_output(tmp_path):
    """Approved system tool text stays readable while external roots remain redacted."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(tmp_path, scratch))
    backend = MagicMock(
        run=MagicMock(
            return_value=CommandProcessResult(
                SandboxOutcome.COMPLETED, exit_code=0, stdout="/usr/bin/true\n"
            )
        )
    )
    backend.system_read_roots = (Path("/usr/bin"),)
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(backend, MagicMock()),
    )
    payload = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "printf /usr/bin/true", "read_roots": ["/usr/bin"]}),
            instructions_manager=instructions,
            interaction=ApprovingInteraction(),
        )
    )

    assert payload["result"]["stdout"]["content"] == "/usr/bin/true\n"


def test_observed_denial_preserves_exit_without_host_retry(monkeypatch, registry):
    """A completed shell exit and its output remain authoritative over an observed denial."""
    backend = MagicMock(
        run=MagicMock(
            return_value=CommandProcessResult(
                SandboxOutcome.COMPLETED,
                exit_code=7,
                stderr="Operation not permitted: /outside/secret",
                observed_denial="file-read-data /outside/secret",
            )
        )
    )
    host = MagicMock()
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    monkeypatch.setattr(registry._execution_service, "_host_executor", host)

    _, problem = call(registry, "cat outside || :; exit 7")

    assert problem["code"] == "process.nonzero_exit"
    assert problem["metadata"]["exit_code"] == 7
    assert problem["metadata"]["stderr"]["content"] == "Operation not permitted: /outside/secret"
    assert problem["metadata"]["sandbox_diagnostics"]["observed_denial"] is None
    assert problem["metadata"]["sandbox_diagnostics"]["denial_candidates"][0] == {
        "observation": "file-read-data /outside/secret",
        "classification": "unclassified",
        "attribution": "unknown",
        "truncated": False,
    }
    assert "sandbox_diagnostic" not in problem["metadata"]
    assert "executable_hint" not in problem["metadata"]
    assert "host_offer" not in problem["metadata"]
    host.run_host_command.assert_not_called()


def test_denial_candidate_survives_bounded_command_output(monkeypatch, registry):
    """A large command result retains the candidate in its first model-facing response."""
    denial = "Sandbox: tool(1) deny(1) file-read-data /outside/secret"
    backend = MagicMock(
        run=MagicMock(
            return_value=CommandProcessResult(
                SandboxOutcome.COMPLETED,
                exit_code=7,
                stdout="x" * (MAX_TOOL_RESULT_BYTES * 2),
                stderr="y" * (MAX_TOOL_RESULT_BYTES * 2),
                observed_denial=denial,
                diagnostic_status="available",
                diagnostic_source="macos-kernel-log",
            )
        )
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)

    _, problem = call(registry, "exit 7")

    metadata = problem["metadata"]
    assert metadata["exit_code"] == 7
    assert metadata["sandbox_diagnostics"]["denial_candidates"][0]["observation"] == denial
    assert metadata["sandbox_diagnostics"]["denial_candidates"][0]["attribution"] == "unknown"
    assert metadata["stdout"]["truncated"] and metadata["stderr"]["truncated"]


def test_control_character_tails_keep_command_result_within_budget(monkeypatch, registry):
    """Both serialized tails retain useful endings without displacing exit diagnostics."""
    backend = MagicMock(
        run=MagicMock(
            return_value=CommandProcessResult(
                SandboxOutcome.COMPLETED,
                exit_code=7,
                stdout="\x01" * (MAX_TOOL_RESULT_BYTES * 2) + "stdout-end",
                stderr="\x02" * (MAX_TOOL_RESULT_BYTES * 2) + "stderr-end",
                observed_denial="Sandbox: fixture deny file-read-data /outside/secret",
                diagnostic_status="available",
                diagnostic_source="macos-kernel-log",
            )
        )
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)

    _, problem = call(registry, "exit 7")

    metadata = problem["metadata"]
    assert metadata["exit_code"] == 7
    assert metadata["sandbox_diagnostics"]["denial_candidates"]
    assert metadata["stdout"]["tail"]["content"].endswith("stdout-end")
    assert metadata["stderr"]["tail"]["content"].endswith("stderr-end")


@pytest.mark.parametrize("exit_code", [126, 127])
def test_executable_failure_hint_preserves_unknown_cause_and_effects(
    monkeypatch, registry, exit_code
):
    """A missing executable suggests grants without offering an unrelated host retry."""
    backend = MagicMock(
        run=MagicMock(
            return_value=CommandProcessResult(
                SandboxOutcome.COMPLETED,
                exit_code=exit_code,
                stderr="/bin/sh: tool: command not found\n",
                observed_denial="file-read-data /Library/Preferences/unrelated.plist",
            )
        )
    )
    host = MagicMock()
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    monkeypatch.setattr(registry._execution_service, "_host_executor", host)

    _, problem = call(registry, "tool --version", read_only=True)

    assert problem["code"] == "process.nonzero_exit"
    metadata = problem["metadata"]
    assert metadata["exit_code"] == exit_code
    assert "resolve_executable" in metadata["executable_hint"]
    assert "host_offer" not in metadata
    assert "indirect tools remain unidentified" in metadata["executable_hint"]
    assert "look up each one" not in metadata["executable_hint"]
    assert "caused" not in metadata["executable_hint"]
    assert "may already have had effects" in metadata["executable_hint"]
    host.run_host_command.assert_not_called()


def test_observed_denial_never_prompts_for_host_retry(monkeypatch, registry):
    """Even an approving interaction cannot turn a completed denial into a host run."""
    registry.permission_manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    backend = MagicMock(
        run=MagicMock(
            return_value=CommandProcessResult(
                SandboxOutcome.COMPLETED,
                exit_code=1,
                stderr="Operation not permitted: /outside/secret",
                observed_denial="file-read-data /outside/secret",
            )
        )
    )
    host = MagicMock(
        run_host_command=MagicMock(
            return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
        )
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    monkeypatch.setattr(registry._execution_service, "_host_executor", host)

    result, problem = call(registry, "cat outside", interaction=ApprovingHostInteraction())

    assert result is None
    assert problem["code"] == "process.nonzero_exit"
    assert problem["metadata"]["boundary"] == "sandbox"
    assert problem["metadata"]["stderr"]["content"] == "Operation not permitted: /outside/secret"
    assert problem["metadata"]["sandbox_diagnostics"]["observed_denial"] is None
    assert (
        problem["metadata"]["sandbox_diagnostics"]["denial_candidates"][0]["attribution"]
        == "unknown"
    )
    assert "sandbox_diagnostic" not in problem["metadata"]
    assert "host_offer" not in problem["metadata"]
    host.run_host_command.assert_not_called()


def test_boundary_failure_keeps_observed_diagnostic(monkeypatch, registry):
    """A sandbox boundary failure retains its own diagnostic without a host retry."""
    backend = MagicMock(
        run=MagicMock(
            return_value=CommandProcessResult(
                SandboxOutcome.UNAVAILABLE,
                detail="Sandbox capture failed.",
                possible_effects=True,
                observed_denial="file-read-data /outside/secret",
            )
        )
    )
    host = MagicMock()
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    monkeypatch.setattr(registry._execution_service, "_host_executor", host)

    _, problem = call(registry, "cat outside")

    assert problem["code"] == "sandbox.unavailable"
    assert problem["metadata"]["sandbox_diagnostic"] == "file-read-data /outside/secret"
    assert "host_offer" not in problem["metadata"]
    host.run_host_command.assert_not_called()


def test_run_command_reports_typed_sandbox_failure(monkeypatch, registry):
    """A backend launch failure is returned without host execution."""
    monkeypatch.setattr(
        registry._execution_service,
        "_backend",
        MagicMock(
            run=MagicMock(
                return_value=CommandProcessResult(
                    SandboxOutcome.UNAVAILABLE, detail="profile rejected"
                )
            )
        ),
    )

    _, problem = call(registry, "printf MUST_NOT_RUN")

    assert problem["code"] == "sandbox.unavailable"
    assert problem["detail"] == "profile rejected"
    assert 'Command: "printf MUST_NOT_RUN"' in problem["metadata"]["host_offer"]


@pytest.mark.parametrize(("host_answer", "approved"), [("deny", False), ("approve", True)])
def test_public_startup_capability_failure_requires_separate_host_approval(
    tmp_path, host_answer, approved
):
    """An unavailable native backend offers recovery but never starts host before approval."""
    backend = UnavailableSandboxBackend("Operational sandbox probe failed.")
    backend.run = MagicMock(wraps=backend.run)
    host = CountingHostCommandExecutor()
    interaction = RecordingHostOfferInteraction(host, host_answer)
    permissions = PermissionManager(tmp_path)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, host),
    )

    result, problem = call(
        registry,
        "printf startup-recovery",
        cwd=str(tmp_path),
        interaction=interaction,
    )

    assert backend.run.call_count == 0
    assert interaction.host_prompts == ["Run this exact command without the OS sandbox?"]
    assert interaction.launches_before_host_prompt == [0]
    assert len(interaction.information) == 1
    assert "The sandbox was unavailable." in interaction.information[0]
    assert str(tmp_path) not in interaction.information[0]
    assert json.dumps(str(tmp_path))[1:-1] not in interaction.information[0]
    if approved:
        assert problem is None
        assert result["boundary"] == "host"
        assert result["stdout"]["content"] == "startup-recovery"
        assert len(host.requests) == 1
        assert host.requests[0].source == "printf startup-recovery"
    else:
        assert result is None
        assert problem["code"] == "sandbox.unavailable"
        assert "host_offer" in problem["metadata"]
        assert host.requests == []


@pytest.mark.parametrize("outcome", [SandboxOutcome.DENIED, SandboxOutcome.UNAVAILABLE])
def test_run_command_does_not_retry_after_possible_effects(monkeypatch, registry, outcome):
    """A possibly effective attempt cannot produce an offer or launch a host retry."""
    registry.permission_manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    host = MagicMock()
    monkeypatch.setattr(registry._execution_service, "_host_executor", host)
    monkeypatch.setattr(
        registry._execution_service,
        "_backend",
        MagicMock(
            run=MagicMock(
                return_value=CommandProcessResult(
                    outcome, detail="sandbox stopped", possible_effects=True
                )
            )
        ),
    )

    _, problem = call(
        registry, "printf changed > file; cat secret", interaction=ApprovingHostInteraction()
    )

    assert problem["code"] == f"sandbox.{outcome.value}"
    assert problem["metadata"]["sandbox_diagnostics"]["possible_effects"] is True
    assert "request_host_command" in problem["metadata"]["recovery_hint"]
    assert "may repeat effects" in problem["metadata"]["recovery_hint"]
    assert "host_offer" not in problem["metadata"]
    host.run_host_command.assert_not_called()


def test_run_command_host_retry_requires_a_second_prompt(monkeypatch, registry):
    """A distinct approved host retry is marked as unrestricted in the public result."""
    registry.permission_manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    backend = MagicMock(
        run=MagicMock(
            return_value=CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="no profile")
        )
    )
    host = MagicMock(
        run_host_command=MagicMock(
            return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0, stdout="host")
        )
    )
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    monkeypatch.setattr(registry._execution_service, "_host_executor", host)
    interaction = ApprovingHostInteraction()

    result, problem = call(registry, "printf host", interaction=interaction)

    assert problem is None
    assert result["boundary"] == "host"
    assert result["stdout"]["content"] == "host"
    host.run_host_command.assert_called_once()


def test_run_command_keeps_host_timeout_distinct(monkeypatch, registry):
    """A separately approved host timeout is labeled as a host outcome."""
    registry.permission_manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    monkeypatch.setattr(
        registry._execution_service,
        "_backend",
        MagicMock(
            run=MagicMock(
                return_value=CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="no profile")
            )
        ),
    )
    monkeypatch.setattr(
        registry._execution_service,
        "_host_executor",
        MagicMock(
            run_host_command=MagicMock(return_value=CommandProcessResult(SandboxOutcome.TIMED_OUT))
        ),
    )

    _, problem = call(registry, "sleep 10", interaction=ApprovingHostInteraction())

    assert problem["code"] == "host_command.timed_out"
    assert problem["metadata"]["boundary"] == "host"


def test_run_command_timeout_does_not_offer_host_retry(monkeypatch, registry):
    """A timed-out sandbox attempt is not reclassified as a host offer."""
    monkeypatch.setattr(
        registry._execution_service,
        "_backend",
        MagicMock(run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.TIMED_OUT))),
    )

    _, problem = call(registry, "sleep 10")

    assert problem["code"] == "sandbox.timed_out"
    assert "host_offer" not in problem["metadata"]


def test_run_command_requires_bound_target_and_permission_manager(tmp_path):
    """Direct calls fail closed without registry-owned native authority."""
    context = ToolContext(ApprovingInteraction(), "run_command")
    assert run_command_tool(context, "printf no").code == "sandbox.unavailable"
    context = ToolContext(
        ApprovingInteraction(),
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action="process.execute",
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "printf no"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
    )
    assert (
        run_command_tool(context, "printf no").detail
        == "Native command execution service is unavailable."
    )


def test_run_command_fails_when_request_paths_are_invalid(tmp_path):
    """A missing working directory fails before approval or launch."""
    manager = MagicMock()
    context = ToolContext(
        ApprovingInteraction(),
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action="process.execute",
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", ":"),
                    cwd=str(Path(tmp_path) / "missing"),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=manager,
        execution_service=CommandExecutionService(
            MacOSSeatbeltBackend(), LocalHostCommandExecutor()
        ),
    )
    result = run_command_tool(context, ":")
    assert result.code == "sandbox.unavailable"
    manager.authorize_sandboxed_command.assert_not_called()


@pytest.mark.parametrize(
    "source",
    [
        "cat $WORKSPACE_ROOT/file",
        "printf '%s' '$WORKSPACE_ROOT/file'",
        "printf '%s' '/.loop-private-paths/r0/file'",
        "python -c 'print(\"$TMPDIR/example\")'",
    ],
)
def test_run_command_preserves_opaque_shell_source(monkeypatch, tmp_path, source):
    """Opaque source, including quoted and embedded paths, reaches the backend unchanged."""
    backend = MagicMock(
        run=MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    )
    execution_service = CommandExecutionService(backend, LocalHostCommandExecutor())
    permissions = PermissionManager()
    instructions = MagicMock(
        virtual_paths=VirtualPath(
            workspace=tmp_path, temporary_directory=permissions.temporary_directory
        )
    )
    context = ToolContext(
        ApprovingInteraction(),
        "run_command",
        instructions_manager=instructions,
        operations=(
            Operation(
                tool_id="run_command",
                action="process.execute",
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", source),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=permissions,
        execution_service=execution_service,
    )

    result = run_command_tool(context, source)

    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.execution_source == request.source == source


def test_run_command_honors_fresh_denial(monkeypatch, registry):
    """A denied native prompt starts no sandbox backend."""
    backend = MagicMock()
    monkeypatch.setattr(registry._execution_service, "_backend", backend)
    registry.permission_manager.authorize_sandboxed_command = MagicMock(return_value=False)

    _, problem = call(registry, "printf no")

    assert problem["code"] == "tool.denied"
    backend.run.assert_not_called()


def test_run_command_spills_an_oversized_backend_preview(monkeypatch, registry):
    """Oversized backend text retains a continuation handle in the public result."""
    monkeypatch.setattr(
        registry._execution_service,
        "_backend",
        MagicMock(
            run=MagicMock(
                return_value=CommandProcessResult(
                    SandboxOutcome.COMPLETED,
                    exit_code=0,
                    stdout="x" * MAX_OUTPUT_CHARS,
                    stdout_discarded=1,
                )
            )
        ),
    )
    result, _ = call(registry, "printf lots")
    assert result["stdout"]["truncated"] is True
    assert result["stdout"]["handle"]


@pytest.mark.parametrize("initial_selection", [False, True])
@pytest.mark.parametrize(
    "component",
    [
        "helper bytes",
        "helper mode",
        "record",
        "receipt",
        "lock",
        "code bytes",
        "code mode",
        "unknown inventory",
    ],
)
def test_linux_integrity_failure_never_uses_host_offer_or_session_rule(
    tmp_path, monkeypatch, initial_selection, component
):
    """Initial and per-command integrity failures cannot prompt, offer or reuse host authority."""
    from loop.execution.sandbox import selection

    record = tmp_path / "management.json"
    record.write_text("bound authority")
    native = linux.LinuxNativeBackend(record)
    monkeypatch.setattr(linux, "runtime_read_roots", lambda: ())
    monkeypatch.setattr(linux.LinuxNativeBackend, "file_creation_mask", 0o022)
    monkeypatch.setattr(linux.LinuxNativeBackend, "managed_tool_root", lambda *_: False)
    validator = MagicMock()
    monkeypatch.setattr(linux, "execution_authority", validator)
    exchanged = MagicMock()
    monkeypatch.setattr(linux, "exchange_installed_request", exchanged)
    host = MagicMock()
    host.run_host_command.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    permissions = PermissionManager(tmp_path)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    interaction = RememberingHostInteraction()
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=permissions,
        execution_service=CommandExecutionService(native, host),
    )

    def unavailable(_record, payload, _deadline, *, observer):
        """Allow an explicit host session decision before introducing integrity drift."""
        from loop.execution.sandbox import SandboxRequest

        assert observer is not None
        request = SandboxRequest.from_wire(payload)
        return CommandProcessResult(SandboxOutcome.UNAVAILABLE).to_wire(request.attempt_id)

    exchanged.side_effect = unavailable
    first, problem = call(registry, "printf ordinary", cwd=str(tmp_path), interaction=interaction)
    assert problem is None and first["boundary"] == "host"
    assert len(permissions.host_command_rules()) == 1
    assert interaction.host_prompts == 1
    host.run_host_command.reset_mock()
    exchanged.reset_mock()
    validator.side_effect = ValueError(f"Private installed {component} authority changed.")
    if initial_selection:
        # New application selection must preserve invalid authority, including an existing
        # host session rule; select under the public Linux composition route.
        native = linux.LinuxNativeBackend(record)
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            "loop.execution.sandbox.linux.backend.LinuxNativeBackend", lambda: native
        )
        selected = selection.select_sandbox_backend()
        registry = ToolRegistry(
            BUILTIN_TOOLS,
            permission_manager=permissions,
            execution_service=CommandExecutionService(selected, host),
        )
    result, problem = call(registry, "printf ordinary", cwd=str(tmp_path), interaction=interaction)
    assert result is None
    assert problem["code"] == ("sandbox.invalid" if initial_selection else "sandbox.stale")
    assert "host_offer" not in problem.get("metadata", {})
    assert "Private installed" not in problem["detail"]
    assert interaction.host_prompts == 1
    host.run_host_command.assert_not_called()
    exchanged.assert_not_called()


@pytest.mark.parametrize("delivered", [True, False])
def test_linux_public_timeout_retains_capture_or_reports_unknown_loss(
    tmp_path, monkeypatch, delivered
):
    """Public timeout streams retain authenticated text or honestly report unavailable capture."""
    from loop.execution.sandbox import SandboxRequest

    record = tmp_path / "management.json"
    record.write_text("bound authority")
    monkeypatch.setattr(linux, "execution_authority", MagicMock())
    monkeypatch.setattr(linux, "runtime_read_roots", lambda: ())
    monkeypatch.setattr(linux.LinuxNativeBackend, "file_creation_mask", 0o022)
    monkeypatch.setattr(linux.LinuxNativeBackend, "managed_tool_root", lambda *_: False)

    def exchange(_record, payload, _deadline, *, observer):
        """Return timeout output after native cleanup or a failed result-delivery budget."""
        assert observer is not None
        if not delivered:
            raise TimeoutError("delivery expired")
        request = SandboxRequest.from_wire(payload)
        return CommandProcessResult(
            SandboxOutcome.TIMED_OUT,
            stdout="distinct-out",
            stderr="distinct-err",
            stdout_discarded=None,
            stderr_discarded=None,
            possible_effects=True,
        ).to_wire(request.attempt_id)

    monkeypatch.setattr(linux, "exchange_installed_request", exchange)
    host = MagicMock()
    interaction = RememberingHostInteraction()
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(linux.LinuxNativeBackend(record), host),
    )
    result, problem = call(registry, "printf ordinary", cwd=str(tmp_path), interaction=interaction)
    assert result is None and problem["code"] == "sandbox.timed_out"
    assert "host_offer" not in problem.get("metadata", {})
    for stream, text in (("stdout", "distinct-out"), ("stderr", "distinct-err")):
        capture = problem["metadata"][stream]
        assert capture["content"] == (text if delivered else "")
        assert capture["capture_complete"] is False
        assert capture["discarded_characters"] is None
        assert capture["decoding_loss"] == "unknown"
    assert interaction.host_prompts == 0
    host.run_host_command.assert_not_called()
