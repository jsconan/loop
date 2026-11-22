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
    assert set(workspace.scratch_roots) == {
        Path.home().resolve() / ".loop-scratch",
        tmp_path / "scratch",
    }
    with pytest.raises(ValueError, match="path segment"):
        paths.for_workspace("../outside", tmp_path)


def test_workspace_scratch_falls_back_to_external_durable_data(tmp_path, monkeypatch):
    """A home checkout uses an independent application data directory."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    paths = ApplicationPaths(tmp_path, tmp_path / "data", tmp_path)
    assert paths.for_workspace("workspace", home).scratch_roots == (paths.data_root / "scratch",)
    assert paths.for_workspace("workspace", paths.session_scratch).scratch_roots == (
        paths.data_root / "scratch",
    )


@pytest.mark.parametrize("data_name", ["data", "data with spaces"])
def test_workspace_scratch_prefers_paths_without_whitespace(tmp_path, monkeypatch, data_name):
    """A safe workspace starts even when all external scratch paths contain spaces."""
    home = tmp_path / "home with spaces"
    monkeypatch.setenv("HOME", str(home))
    paths = ApplicationPaths(tmp_path, tmp_path / data_name, tmp_path)
    workspace = paths.for_workspace("workspace", tmp_path / "checkout")
    home_scratch = home / ".loop-scratch"
    data_scratch = paths.data_root / "scratch"
    if any(character.isspace() for character in str(data_scratch)):
        assert workspace.scratch_roots == (home_scratch, data_scratch)
    else:
        assert workspace.scratch_roots == (data_scratch, home_scratch)


def test_workspace_scratch_requires_storage_outside_the_checkout(tmp_path, monkeypatch):
    """A fresh generation cannot be placed under the checkout."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    paths = ApplicationPaths(tmp_path, tmp_path / "data", tmp_path)
    with pytest.raises(ValueError, match="private path outside the workspace"):
        paths.for_workspace("workspace", tmp_path)


def test_workspace_scratch_refuses_a_directory_identity_alias(tmp_path, monkeypatch):
    """Different path spellings cannot place working files inside the same checkout."""
    home = tmp_path / "home"
    home.mkdir()
    alias = tmp_path / "HOME"
    monkeypatch.setenv("HOME", str(home))
    stat = Path.stat

    def alias_stat(path, **options):
        """Expose the same home directory under its alternate spelling."""
        return stat(home if path == alias else path, **options)

    monkeypatch.setattr(Path, "stat", alias_stat)
    paths = ApplicationPaths(tmp_path, tmp_path / "data", tmp_path)
    assert paths.for_workspace("workspace", alias).scratch_roots == (paths.data_root / "scratch",)
