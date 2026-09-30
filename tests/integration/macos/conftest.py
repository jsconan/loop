"""Provide isolated fixtures for macOS execution, permission and search integration."""

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop import PermissionManager, RuntimeEnvironment, ToolRegistry
from loop.execution import CommandExecutionService
from loop.execution.sandbox import SandboxRequest, UnavailableSandboxBackend, select_sandbox_backend
from loop.instructions import InstructionsManager
from loop.tooling import ToolRegistration
from loop.tools.files import search_text
from loop.tools.system import resolve_executable, run_command


@pytest.fixture
def backend(monkeypatch):
    """Require a qualified native sandbox and always close its owned diagnostic service."""
    # OS enforcement stays real; diagnostic log polling is covered with doubles in units.
    monitor = Mock()
    monitor.register.return_value = None
    monitor.denial.return_value = None
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.SeatbeltDiagnosticService.monitor", lambda *args: monitor
    )
    monkeypatch.setattr("loop.execution.sandbox.macos._denial_from_log", lambda *args: None)
    selected = select_sandbox_backend()
    if isinstance(selected, UnavailableSandboxBackend):
        pytest.skip(selected.capability_failure())
    try:
        yield selected
    finally:
        selected.close()


@pytest.fixture
def native_workspace(tmp_path):
    """Build fake control files and secrets without reading any user or project data."""
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
def native_command(backend, native_workspace):
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
            "deadline": time.monotonic() + 5,
            "workspace_id": "native-execution-fixture",
        }
        values.update(changes)
        return backend.run(SandboxRequest.create(**values))

    return run


@pytest.fixture
def public_command(backend, native_workspace):
    """Compose public dispatch, permissions and native enforcement with a forbidden host retry."""
    host = Mock()
    host.run_host_command.side_effect = AssertionError("unapproved host launch")
    interaction = Mock()
    interaction.prompt.side_effect = lambda message, **kwargs: (
        "deny" if "without the OS sandbox" in message else "approve"
    )
    scratch = native_workspace.parent / "instruction-scratch"
    scratch.mkdir()
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(native_workspace, scratch),
        agents_filenames=("AGENTS.md", "POLICY+.md"),
    )
    registry = ToolRegistry(
        [
            run_command,
            resolve_executable,
            search_text,
            ToolRegistration(run_command, name="renamed_command"),
        ],
        permission_manager=PermissionManager(native_workspace),
        execution_service=CommandExecutionService(backend, host),
    )

    def call(source=None, *, name="run_command", chosen_interaction=None, **arguments):
        """Dispatch through the model's virtual paths and return the public response."""
        if source is not None:
            arguments = {"command": source, "cwd": "/workspace", **arguments}
        return json.loads(
            registry.call(
                name,
                json.dumps(arguments),
                interaction=chosen_interaction or interaction,
                instructions_manager=instructions,
            )
        )

    try:
        yield SimpleNamespace(call=call, host=host, interaction=interaction, registry=registry)
    finally:
        registry.close()
