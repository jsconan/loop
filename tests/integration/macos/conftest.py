"""Provide isolated fixtures for macOS execution, permission and search integration."""

import json
import os
import secrets
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop import PermissionManager, RuntimeEnvironment, Skill, SkillManager, ToolRegistry
from loop.execution import CommandExecutionService
from loop.execution.sandbox import (
    CommandProcessResult,
    SandboxRequest,
    UnavailableSandboxBackend,
    select_sandbox_backend,
    selection,
)
from loop.execution.sandbox.macos import backend as macos_backend
from loop.instructions import InstructionsManager
from loop.permissions.protection import is_protected_external_read
from loop.tooling import ToolRegistration
from loop.tools.files import search_text
from loop.tools.system import resolve_executable, run_command
from loop.utils import process as process_utils


@pytest.fixture(scope="session", autouse=True)
def require_macos():
    """Skip macOS native cases before any platform-specific fixture starts."""
    if sys.platform != "darwin":
        pytest.skip("macOS native integration requires macOS")


@pytest.fixture(scope="session")
def macos_local_volume(require_macos):
    """Own one disposable native socket filesystem across macOS integration cases."""
    if os.environ.get("LOOP_NATIVE_MACOS_TEST") != "owned-volume":
        pytest.skip("Set LOOP_NATIVE_MACOS_TEST=owned-volume for native macOS qualification")
    try:
        volume = macos_backend._mount_local_socket_volume(time.monotonic() + 30)
    except (OSError, TimeoutError, subprocess.TimeoutExpired) as error:
        pytest.exit(
            "Native macOS integration requires an unsandboxed parent with permission "
            f"to attach its owned RAM volume: {error}",
            returncode=4,
        )
    try:
        yield volume
    finally:
        macos_backend._eject_local_socket_volume(volume)


@pytest.fixture(autouse=True)
def local_socket_parent(monkeypatch, native_directory, macos_local_volume):
    """Keep native attempt-local sockets inside one registered disposable volume subtree."""
    safe = native_directory(macos_local_volume.mount, name="t" + secrets.token_hex(4))
    monkeypatch.setattr(macos_backend, "_PRIVATE_TEMP_ROOT", safe)
    monkeypatch.setattr(
        macos_backend,
        "_mount_local_socket_volume",
        lambda _deadline: macos_local_volume._replace(mount=safe),
    )
    monkeypatch.setattr(macos_backend, "_eject_local_socket_volume", lambda _volume: None)
    return safe


@pytest.fixture
def backend(monkeypatch, local_socket_parent, native_execution_clock):
    """Require a qualified native sandbox and always close its owned diagnostic service."""
    # OS enforcement stays real; diagnostic log polling is covered with doubles in units.
    monitor = Mock()
    monitor.register.return_value = None
    monitor.denial.return_value = None
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.SeatbeltDiagnosticService.monitor",
        lambda *args: monitor,
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend._denial_from_log", lambda *args: None)
    selected = select_sandbox_backend()
    if isinstance(selected, UnavailableSandboxBackend):
        failure = selected.capability_failure()
        if isinstance(failure, CommandProcessResult):
            pytest.fail(
                f"Native sandbox capability failed ({failure.outcome.value}): {failure.detail}"
            )
        pytest.skip(failure)
    try:
        yield selected
    finally:
        selected.close()


@pytest.fixture
def native_workspace(tmp_path, request, native_directory):
    """Build fake control files and secrets without reading any user or project data."""
    if is_protected_external_read(tmp_path):
        tmp_path = native_directory("/private/tmp") / "workspace"
        tmp_path.mkdir(mode=0o700)
    if getattr(request, "param", None) is not None:
        tmp_path = tmp_path / request.param
        tmp_path.mkdir()
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "id_ed25519").write_text("fake-private-canary", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("preserve-git", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("preserve-instructions", encoding="utf-8")
    (tmp_path / "ordinary").write_text("ordinary-data", encoding="utf-8")
    outside = tmp_path.parent / "outside"
    outside.mkdir()
    (outside / "secret").write_text("fake-outside-canary", encoding="utf-8")
    (outside / "write").write_text("preserve-outside", encoding="utf-8")
    (tmp_path / "external-link").symlink_to(outside / "secret")
    return tmp_path


@pytest.fixture
def native_execution_clock(monkeypatch):
    """Use a synthetic clock for native command execution and child supervision."""
    clock = SimpleNamespace(
        monotonic=lambda: 100.0,
        sleep=lambda _delay: os.sched_yield(),
        strftime=lambda _format, _value: "2000-01-01 00:00:00+0000",
        localtime=lambda: None,
    )
    monkeypatch.setattr("loop.execution.coordinator.time", clock)
    monkeypatch.setattr("loop.execution.facade.time", clock)
    monkeypatch.setattr(selection, "time", clock)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.time", clock)
    monkeypatch.setattr(process_utils, "time", clock)
    return clock


@pytest.fixture
def native_command(backend, native_workspace, native_execution_clock):
    """Run real Seatbelt requests against a fresh synthetic workspace."""

    def run(source, **changes):
        """Bind native authority explicitly without inheriting host environment or files."""
        values = {
            "source": source,
            "cwd": native_workspace,
            "workspace": native_workspace,
            "read_roots": (),
            "write_roots": (native_workspace,),
            "network": False,
            "environment": {"PATH": "/usr/bin:/bin", "LANG": "C"},
            "policy_version": backend.policy_version,
            "deadline": 110.0,
            "workspace_id": "native-execution-fixture",
        }
        values.update(changes)
        return backend.run(SandboxRequest.create(**values))

    return run


@pytest.fixture
def public_command(backend, native_workspace, native_execution_clock, request):
    """Compose public dispatch, permissions and native enforcement with a forbidden host retry."""
    host = Mock()
    host.run_host_command.side_effect = AssertionError("unapproved host launch")
    interaction = Mock()
    interaction.prompt.side_effect = lambda message, **kwargs: (
        "deny"
        if "without the OS sandbox" in message
        else "review"
        if "review" in kwargs.get("choices", {})
        else "approve"
    )
    policy_case = (
        request.node.callspec.params.get("policy_case")
        if getattr(request.node, "callspec", None)
        else None
    )
    if policy_case in {"default_saved", "custom_saved"}:
        policy_directory = native_workspace / ".loop"
        policy_directory.mkdir()
        payload = (
            "version: 1\nlimits:\n  readable_roots: [workspace]\n"
            if policy_case == "custom_saved"
            else "version: 1\n"
        )
        (policy_directory / "permissions.yaml").write_text(payload, encoding="utf-8")
    manager = PermissionManager(native_workspace)
    scratch = manager.temporary_directory
    skill_root = native_workspace.parent / "fixture-skill"
    skill_root.mkdir()
    skill_file = skill_root / "SKILL.md"
    skill_file.write_text(
        "---\nname: fixture\ndescription: Native fixture.\n---\nRead resources.\n"
    )
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(native_workspace, scratch),
        skill_manager=SkillManager([Skill("fixture", "Native fixture.", skill_file)]),
        agents_filenames=("AGENTS.md", "POLICY+.md", 'POLICY"name.md', "POLICY\nname.md"),
    )
    registry = ToolRegistry(
        [
            run_command,
            resolve_executable,
            search_text,
            ToolRegistration(run_command, name="renamed_command"),
        ],
        permission_manager=manager,
        execution_service=CommandExecutionService(backend, host),
    )

    def call(source=None, *, name="run_command", chosen_interaction=None, **arguments):
        """Dispatch through the model's virtual paths and return the public response."""
        if source is not None:
            arguments = {"command": source, "cwd": "$WORKSPACE_ROOT", **arguments}
        return json.loads(
            registry.call(
                name,
                json.dumps(arguments),
                interaction=chosen_interaction or interaction,
                instructions_manager=instructions,
            )
        )

    try:
        yield SimpleNamespace(
            call=call,
            host=host,
            interaction=interaction,
            registry=registry,
            manager=manager,
            backend=backend,
            instructions=instructions,
            scratch=scratch,
            skill_root=skill_root,
        )
    finally:
        registry.close()
        manager.close()


@pytest.fixture
def live_egress_clock(monkeypatch, public_command):
    """Use live monotonic deadlines for an owned native mediator and its command."""
    for name in (
        "loop.execution.coordinator.time",
        "loop.execution.facade.time",
        "loop.execution.sandbox.selection.time",
        "loop.execution.sandbox.macos.backend.time",
        "loop.utils.process.time",
    ):
        monkeypatch.setattr(name, time)
    return public_command
