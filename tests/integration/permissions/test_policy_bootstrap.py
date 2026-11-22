"""Verify first-use policy publication through public file-tool dispatch."""

import asyncio
import json
from io import StringIO
from unittest.mock import AsyncMock, Mock

import pytest
from rich.console import Console

from loop.application.paths import ApplicationPaths
from loop.application.runtime import ApplicationRuntime
from loop.configuration import ApplicationSettings, BackendSettings, ConfigurationManager
from loop.execution import CommandExecutionService
from loop.interaction import ConsoleInteraction
from loop.permissions import PermissionManager
from loop.permissions.models import PERMISSION_POLICY_VERSION
from loop.tooling import ToolRegistry
from loop.tools.catalog import BUILTIN_TOOLS
from loop.utils.yaml import dump_yaml, load_yaml
from loop.workspace import Workspace, WorkspaceRepository

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("choice", ["c", "t", "T"])
@pytest.mark.parametrize("scope", ["workspace", "user"])
def test_startup_recovery_reports_the_selected_action(tmp_path, choice, scope):
    """Startup shortcuts report the observed preservation or archived replacement of policy."""
    root = tmp_path / "project"
    root.mkdir()
    paths = ApplicationPaths(tmp_path / "config", tmp_path / "data", tmp_path / "state")
    repository = WorkspaceRepository(paths.workspace_catalog)
    workspace = repository.initialize(Workspace.discover(root))
    workspace_paths = paths.for_workspace(workspace.id, root)
    manager = PermissionManager(
        root,
        configuration_path=workspace_paths.permissions,
        user_configuration_path=paths.user_permissions,
        workspace_id=workspace.id,
        namespace_catalog_path=paths.workspace_catalog,
        bootstrap_pristine_policy=True,
        user_namespace_identity=str(paths.data_root),
    )
    manager.close()
    policy = workspace_paths.permissions if scope == "workspace" else paths.user_permissions
    original = b"rules: [incomplete"
    policy.write_bytes(original)
    output = StringIO()
    interaction = ConsoleInteraction(
        console=Console(file=output, width=200),
        session=Mock(),
        choices_session=Mock(prompt_async=AsyncMock(return_value=choice)),
    )

    async def recover():
        """Compose the real startup runtime using the selected console response."""
        runtime = await ApplicationRuntime.create_interactively_async(
            workspace,
            paths,
            workspace_paths,
            ApplicationSettings(backend=BackendSettings(context_window=4096)),
            ConfigurationManager(paths.user_configuration, workspace_paths.configuration),
            repository,
            interaction,
        )
        await runtime.close_async()

    asyncio.run(recover())

    transcript = " ".join(output.getvalue().split())
    backups = list(policy.parent.glob(f"{policy.name}.*.bak"))
    if choice == "c":
        assert "the original is unchanged" in transcript
        assert policy.read_bytes() == original
        assert not backups
    else:
        assert load_yaml(policy.read_text(encoding="utf-8"))["rules"] == []
        assert [backup.read_bytes() for backup in backups] == [original]
        assert "the original is unchanged" not in transcript
        assert f"Original archived at {backups[0]}" in transcript


@pytest.mark.parametrize("backup", (False, True))
def test_missing_policy_allows_tools_and_preserves_existing_backups(tmp_path, backup):
    """Missing policies receive defaults while existing backups remain available for restoration."""
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
        assert result["ok"] is True
        assert paths.user_permissions.is_file()
        assert paths.for_workspace(workspace.id, root).permissions.is_file()
        if backup:
            assert paths.user_permissions.with_name(
                paths.user_permissions.name + ".bak"
            ).read_text() == ("version: 1\nrules: []\n")
    finally:
        registry.close()
        manager.close()


@pytest.mark.parametrize("version", [1, 5, 6, 9])
def test_marked_previous_policy_migrates_before_public_tool_dispatch(tmp_path, version):
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
    payload["version"] = version
    payload.pop("backend_option_revisions")
    if version < 8:
        payload.pop("backend_options")
        payload.pop("command_backend_options")
    read_path, denied_path = str(root / "runtime"), str(root / "withheld")
    if version == 6:
        payload["sandbox_paths"] = {read_path: "allow", denied_path: "deny"}
    elif version < 6:
        payload.pop("sandbox_paths")
    if version == 1:
        payload["defaults"].pop("filesystem.execute", None)
        payload["defaults"].pop("network.local", None)
        payload["defaults"].pop("network.broad", None)
        payload["defaults"].pop("network.ingress", None)
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
        assert saved["version"] == PERMISSION_POLICY_VERSION
        assert not saved["backend_options"] and not saved["command_backend_options"]
        assert saved["limits"]["deny_local_networks"] is (version == 1)
        assert saved["sandbox_paths"] == (
            {read_path: "read", denied_path: "deny"} if version == 6 else {}
        )
        assert (
            policy_path.with_name(
                f"{policy_path.name}.{generation}.schema-v{version}.bak"
            ).read_bytes()
            == original
        )
    finally:
        registry.close()
        manager.close()
