"""Verify first-use policy publication through public file-tool dispatch."""

import json
from unittest.mock import Mock

import pytest

from loop.application.paths import ApplicationPaths
from loop.execution import CommandExecutionService
from loop.permissions import PermissionManager
from loop.tooling import ToolRegistry
from loop.tools.catalog import BUILTIN_TOOLS
from loop.utils.yaml import dump_yaml, load_yaml
from loop.workspace import Workspace, WorkspaceRepository

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("backup", (False, True))
def test_pristine_policy_allows_tools_but_backup_requires_recovery(tmp_path, backup):
    """Registered first use works immediately while older policy evidence stays protected."""
    root = tmp_path / "project"
    root.mkdir()
    paths = ApplicationPaths(tmp_path / "config", tmp_path / "data", tmp_path / "state")
    workspace = WorkspaceRepository(paths.workspace_catalog).initialize(Workspace.discover(root))
    if backup:
        paths.user_permissions.parent.mkdir(parents=True)
        paths.user_permissions.with_name(paths.user_permissions.name + ".bak").write_text(
            "version: 1\nrules: []\n", encoding="utf-8"
        )
    manager = PermissionManager(
        root,
        configuration_path=paths.for_workspace(workspace.id, root).permissions,
        user_configuration_path=paths.user_permissions,
        protected_host_roots=(paths.configuration_root, paths.data_root, paths.state_root),
        workspace_id=workspace.id,
        namespace_catalog_path=paths.workspace_catalog,
        bootstrap_pristine_policy=True,
        user_namespace_identity=str(paths.data_root),
    )
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=manager,
        execution_service=Mock(spec=CommandExecutionService),
    )
    try:
        result = json.loads(
            registry.call(
                "list_folder",
                json.dumps({"path": str(root), "entry_type": "folders", "recursive": False}),
            )
        )
        if backup:
            assert result["problem"]["code"] == "tool.denied"
            assert "recovery required" in result["problem"]["detail"]
            assert not paths.user_permissions.exists()
        else:
            assert result["ok"] is True
            assert paths.user_permissions.is_file()
            assert paths.for_workspace(workspace.id, root).permissions.is_file()
    finally:
        registry.close()
        manager.close()


def test_marked_previous_policy_migrates_before_public_tool_dispatch(tmp_path):
    """A saved older policy cannot strand file tools behind repeated recovery denials."""
    root = tmp_path / "project"
    root.mkdir()
    paths = ApplicationPaths(tmp_path / "config", tmp_path / "data", tmp_path / "state")
    workspace = WorkspaceRepository(paths.workspace_catalog).initialize(Workspace.discover(root))
    policy_path = paths.for_workspace(workspace.id, root).permissions
    arguments = {
        "configuration_path": policy_path,
        "user_configuration_path": paths.user_permissions,
        "workspace_id": workspace.id,
        "namespace_catalog_path": paths.workspace_catalog,
        "bootstrap_pristine_policy": True,
        "user_namespace_identity": str(paths.data_root),
    }
    first = PermissionManager(root, **arguments)
    first.close()
    payload = load_yaml(policy_path.read_text(encoding="utf-8"))
    generation = payload["namespace_generation"]
    payload["version"] = 1
    payload["limits"].pop("deny_local_networks")
    original = dump_yaml(payload).encode("utf-8")
    policy_path.write_bytes(original)
    manager = PermissionManager(root, **arguments)
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=manager,
        execution_service=Mock(spec=CommandExecutionService),
    )
    try:
        manager.require_policy_ready()
        result = json.loads(
            registry.call(
                "list_folder",
                json.dumps({"path": str(root), "entry_type": "folders", "recursive": False}),
            )
        )
        assert result["ok"] is True
        saved = load_yaml(policy_path.read_text(encoding="utf-8"))
        assert saved["version"] == 2
        assert saved["limits"]["deny_local_networks"] is True
        assert (
            policy_path.with_name(f"{policy_path.name}.{generation}.schema-v1.bak").read_bytes()
            == original
        )
    finally:
        registry.close()
        manager.close()
