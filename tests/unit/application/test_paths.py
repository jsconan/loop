"""Verify immutable application and workspace storage paths."""

from pathlib import Path

import pytest

from loop.application import ApplicationPaths


def test_paths_discover_independent_absolute_environment_overrides(tmp_path: Path) -> None:
    """Every platform root accepts an independent absolute override."""
    paths = ApplicationPaths.discover(
        {
            "LOOP_CONFIG_HOME": str(tmp_path / "config"),
            "LOOP_DATA_HOME": str(tmp_path / "data"),
            "LOOP_STATE_HOME": str(tmp_path / "state"),
        }
    )

    assert paths == ApplicationPaths(tmp_path / "config", tmp_path / "data", tmp_path / "state")
    assert paths.user_configuration == tmp_path / "config" / "config.toml"
    assert paths.user_permissions == tmp_path / "config" / "permissions.yaml"
    assert paths.workspace_catalog == tmp_path / "data" / "workspaces.db"


def test_paths_discover_platform_defaults_when_overrides_are_absent() -> None:
    """Missing environment entries fall back to absolute platform locations."""
    paths = ApplicationPaths.discover({})

    assert paths.configuration_root.is_absolute()
    assert paths.data_root.is_absolute()
    assert paths.state_root.is_absolute()


@pytest.mark.parametrize("value", ["relative", "nested/path"])
def test_paths_reject_relative_roots(tmp_path: Path, value: str) -> None:
    """Application storage cannot be redirected through the working directory."""
    with pytest.raises(ValueError, match="LOOP_CONFIG_HOME"):
        ApplicationPaths.discover({"LOOP_CONFIG_HOME": value})
    with pytest.raises(ValueError, match="must be absolute"):
        ApplicationPaths(Path(value), tmp_path, tmp_path)


def test_paths_derive_fixed_workspace_storage(tmp_path: Path) -> None:
    """An initialized identity produces immutable UUID-scoped paths."""
    paths = ApplicationPaths(tmp_path, tmp_path, tmp_path)
    workspace = paths.for_workspace("workspace-id", tmp_path / "project")

    assert workspace.data == tmp_path / "workspaces" / "workspace-id"
    assert workspace.sessions == workspace.data / "sessions.db"
    assert workspace.permissions == workspace.data / "permissions.yaml"
    assert workspace.configuration == tmp_path / "project" / ".loop" / "config.toml"
    with pytest.raises(ValueError, match="path segment"):
        paths.for_workspace("../outside", tmp_path)
