"""Tests for platform command-execution composition."""

import hashlib
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import loop.application.execution as execution_module
from loop.application import ApplicationPaths
from loop.application.execution import create_command_executor
from loop.application.secrets import ApplicationSecretAuthority
from loop.execution.command import SandboxCommandExecutor
from loop.execution.sandbox.macos import load_macos_runtime_release
from loop.permissions import PermissionManager
from loop.workspace import Workspace


def _inputs(tmp_path, *, initialized=True):
    """Return one isolated composition input graph."""
    project = tmp_path / "project"
    project.mkdir(parents=True)
    workspace = Workspace(
        project,
        project,
        "workspace" if initialized else None,
        "project" if initialized else None,
        "directory" if initialized else None,
        1 if initialized else None,
        1 if initialized else None,
    )
    paths = ApplicationPaths(tmp_path / "config", tmp_path / "data", tmp_path / "state")
    workspace_paths = paths.for_workspace("workspace", project)
    return workspace, paths, workspace_paths


def test_command_executor_rejects_uninitialized_and_unsupported_platforms(tmp_path):
    """Composition requires workspace identity and stays closed off supported macOS hosts."""
    workspace, paths, workspace_paths = _inputs(tmp_path, initialized=False)
    permissions = PermissionManager(tmp_path)
    with pytest.raises(ValueError, match="initialized"):
        create_command_executor(
            workspace,
            paths,
            workspace_paths,
            permissions,
            lambda: "agent",
            system="Darwin",
            machine="arm64",
        )

    workspace, paths, workspace_paths = _inputs(tmp_path / "supported")
    assert (
        create_command_executor(
            workspace,
            paths,
            workspace_paths,
            permissions,
            lambda: "agent",
            system="Linux",
            machine="amd64",
        )
        is None
    )


def test_command_executor_composes_lazy_macos_product_boundary(tmp_path, monkeypatch):
    """Supported Apple Silicon composition binds the release manifest and private state."""
    workspace, paths, workspace_paths = _inputs(tmp_path)
    manifest = load_macos_runtime_release()
    bootstrapper = MagicMock()
    backend = MagicMock()
    adapter = MagicMock()
    permission_adapter = MagicMock()
    monkeypatch.setattr(execution_module, "load_macos_runtime_release", lambda: manifest)
    monkeypatch.setattr(
        execution_module, "RuntimeBootstrapper", MagicMock(return_value=bootstrapper)
    )
    monkeypatch.setattr(execution_module, "MacosSandboxBackend", MagicMock(return_value=backend))
    monkeypatch.setattr(execution_module, "MacosProductAdapter", MagicMock(return_value=adapter))
    monkeypatch.setattr(
        execution_module,
        "ExecutionPermissionAdapter",
        MagicMock(return_value=permission_adapter),
    )
    identity = lambda: "agent"
    progress = MagicMock()

    executor = create_command_executor(
        workspace,
        paths,
        workspace_paths,
        MagicMock(),
        identity,
        progress=progress,
        system="Darwin",
        machine="aarch64",
    )

    assert isinstance(executor, SandboxCommandExecutor)
    assert executor.agent_run_id is identity
    assert executor.runtime_digest.startswith("sha256:")
    assert executor.supports_network_effects is True
    assert executor.supports_secret_exposures is False
    execution_module.RuntimeBootstrapper.assert_called_once()
    execution_module.MacosSandboxBackend.assert_called_once()
    backend_call = execution_module.MacosSandboxBackend.call_args
    assert backend_call.kwargs["state_root"] == Path("/private/tmp") / (
        f"loop-{os.getuid()}-{hashlib.sha256(workspace.id.encode()).hexdigest()[:12]}"
    )
    assert backend_call.kwargs["progress"] is progress
    execution_module.MacosProductAdapter.assert_called_once()

    authority = ApplicationSecretAuthority({("token", "process"): b"secret"})
    enabled = create_command_executor(
        workspace,
        paths,
        workspace_paths,
        MagicMock(),
        identity,
        system="Darwin",
        machine="arm64",
        secret_authority=authority,
    )
    assert enabled is not None
    assert enabled.supports_network_effects is True
    assert enabled.supports_secret_exposures is True


def test_command_executor_management_invalidates_cached_readiness() -> None:
    """Explicit cleanup forces the next command to prepare its runtime again."""
    adapter = MagicMock()
    adapter.manage_sandbox.return_value = True
    executor = SandboxCommandExecutor(
        MagicMock(adapter=adapter),
        MagicMock(),
        "workspace",
        "agent",
        "sha256:" + "a" * 64,
        runtime_resolver=MagicMock(return_value="sha256:" + "b" * 64),
    )
    executor.runtime_ready = True

    assert executor.manage_sandbox(delete=True) is True

    adapter.manage_sandbox.assert_called_once_with(delete=True)
    assert executor.runtime_ready is False
    assert executor.runtime_digest == ""

    executor.service.adapter = object()
    with pytest.raises(NotImplementedError, match="unavailable"):
        executor.manage_sandbox(delete=False)


def test_command_executor_exposes_sandbox_inspection() -> None:
    """Command execution facade exposes status and inventory without changing readiness."""
    adapter = MagicMock()
    adapter.sandbox_status.return_value = ("loop-one", "RUNNING")
    adapter.list_sandboxes.return_value = (("loop-one", "RUNNING"),)
    executor = SandboxCommandExecutor(
        MagicMock(adapter=adapter), MagicMock(), "workspace", "agent", "digest"
    )

    assert executor.sandbox_status() == ("loop-one", "RUNNING")
    assert executor.list_sandboxes() == (("loop-one", "RUNNING"),)
    assert executor.runtime_ready is True

    executor.service.adapter = object()
    with pytest.raises(NotImplementedError, match="status"):
        executor.sandbox_status()
    with pytest.raises(NotImplementedError, match="inventory"):
        executor.list_sandboxes()
