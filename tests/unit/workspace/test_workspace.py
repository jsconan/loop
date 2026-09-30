"""Verify workspace discovery and identity invariants."""

from pathlib import Path

import pytest

from loop.workspace import Workspace


def test_discover_uses_nearest_git_root_and_preserves_active_directory(tmp_path: Path) -> None:
    """Git discovery separates the workspace root from its nested active directory."""
    (tmp_path / ".git").mkdir()
    working = tmp_path / "src" / "package"
    working.mkdir(parents=True)

    workspace = Workspace.discover(working)

    assert workspace.root == tmp_path
    assert workspace.working_directory == working


def test_discover_uses_active_directory_outside_git(tmp_path: Path) -> None:
    """A non-Git directory forms its own workspace boundary."""
    assert Workspace.discover(tmp_path) == Workspace(tmp_path, tmp_path)


def test_workspace_rejects_invalid_location_and_partial_identity(tmp_path: Path) -> None:
    """Workspace values cannot mix roots or expose incomplete durable metadata."""
    with pytest.raises(ValueError, match="within its root"):
        Workspace(tmp_path / "root", tmp_path / "outside")
    with pytest.raises(ValueError, match="identity metadata must be complete"):
        Workspace(tmp_path, tmp_path, id="workspace")


def test_workspace_owns_live_working_directory_reference(tmp_path: Path) -> None:
    """The workspace exposes one stable holder for its active directory."""
    nested = tmp_path / "nested"
    nested.mkdir()
    workspace = Workspace(tmp_path, tmp_path)
    reference = workspace.working_directory

    workspace.working_directory.set(nested)

    assert workspace.working_directory == nested
    assert reference.get() == nested
