"""Verify explicit host CLI provisioning with a durable workspace registry."""

import os
import subprocess
import sys

from loop.application import ApplicationPaths
from loop.workspace import Workspace, WorkspaceRepository


def test_headless_policy_provisioning_requires_exact_identity(tmp_path):
    """A separate process fails on wrong identity and publishes both named scopes on request."""
    project = tmp_path / "project"
    project.mkdir()
    paths = ApplicationPaths(tmp_path / "config", tmp_path / "data", tmp_path / "state")
    workspace = WorkspaceRepository(paths.workspace_catalog).initialize(Workspace.discover(project))
    environment = {
        **os.environ,
        "LOOP_CONFIG_HOME": str(paths.configuration_root),
        "LOOP_DATA_HOME": str(paths.data_root),
        "LOOP_STATE_HOME": str(paths.state_root),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    def provision(scope: str, identity: str) -> subprocess.CompletedProcess[str]:
        """Invoke the real process entry point outside a sandbox."""
        return subprocess.run(
            [sys.executable, "-m", "loop.main", "--initialize-policy", scope, identity],
            cwd=project,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )

    incorrect = provision("workspace", "incorrect")
    assert incorrect.returncode != 0
    assert "exact namespace identity" in incorrect.stderr
    assert not paths.for_workspace(workspace.id, project).permissions.exists()

    initialized_workspace = provision("workspace", workspace.id)
    initialized_user = provision("user", str(paths.data_root))
    assert initialized_workspace.returncode == initialized_user.returncode == 0
    assert paths.for_workspace(workspace.id, project).permissions.is_file()
    assert paths.user_permissions.is_file()

    repeated = provision("workspace", workspace.id)
    assert repeated.returncode != 0
    assert "Existing policy or backup data" in repeated.stderr
