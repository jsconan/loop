"""Verify the shipped native runtime path presets."""

from pathlib import Path

import pytest

from loop.sandbox_catalog import (
    hardlink_source_paths,
    installed_paths,
    managed_directory,
    package_stores,
    system_paths,
    validate_sandbox_options,
)


def test_system_preset_includes_process_and_loader_paths():
    """Both native backends supply their shell and loader directories."""
    assert {Path("/bin"), Path("/usr")} <= set(system_paths("linux"))
    assert {Path("/bin"), Path("/usr/lib")} <= set(system_paths("darwin"))
    assert not system_paths("unsupported")
    assert set(package_stores("darwin")) == {
        Path("/opt/homebrew/Cellar"),
        Path("/usr/local/Cellar"),
    }
    assert not package_stores("linux")


def test_managed_directory_uses_the_application_identity(monkeypatch):
    """The native platform chooses the parent and casing of the configured application name."""
    monkeypatch.setattr("loop.constants.APP_NAME", "example")
    monkeypatch.setattr("loop.constants.APP_DISPLAY_NAME", "Example")
    assert managed_directory("linux") == Path("/etc/example")
    assert managed_directory("darwin") == Path("/Library/Example")


def test_hardlink_sources_use_account_home_for_selected_native_options(tmp_path, monkeypatch):
    """The project environment cannot redirect native metadata inspection."""
    from types import SimpleNamespace

    monkeypatch.setenv("HOME", str(tmp_path / "project"))
    monkeypatch.setattr(
        "loop.sandbox_catalog.pwd.getpwuid", lambda _: SimpleNamespace(pw_dir=str(tmp_path))
    )
    assert hardlink_source_paths(("shared_dependencies",), "linux") == (tmp_path,)
    assert not hardlink_source_paths(("shared_dependencies",), "darwin")
    assert not hardlink_source_paths((), "linux")

    def missing_account(_uid):
        """Return the ordinary account lookup failure for an unmapped owner."""
        raise KeyError("missing")

    monkeypatch.setattr("loop.sandbox_catalog.pwd.getpwuid", missing_account)
    with pytest.raises(ValueError, match="account home"):
        hardlink_source_paths(("shared_dependencies",), "linux")


@pytest.mark.parametrize("state", ["installed", "missing", "redirected"])
def test_installed_preset_requires_a_real_store(monkeypatch, state):
    """Only conventional present stores enable the corresponding layout."""
    store = Path("/opt/homebrew/Cellar")
    monkeypatch.setattr(Path, "is_dir", lambda path: path == store and state != "missing")
    monkeypatch.setattr(
        Path, "resolve", lambda path: Path("/elsewhere") if state == "redirected" else path
    )

    roots = installed_paths("darwin")

    assert Path("/usr/local/bin") in roots
    if state == "installed":
        assert {store, Path("/opt/homebrew/bin"), Path("/opt/homebrew/lib")} <= set(roots)
    else:
        assert store not in roots and Path("/opt/homebrew/bin") not in roots
    assert not installed_paths("linux")


def test_option_declarations_require_known_distinct_keys():
    """Only distinct backend-declared options can be saved or requested."""
    assert managed_directory("unsupported") is None
    assert validate_sandbox_options(("process_pools", "account_lookup")) == (
        "account_lookup",
        "process_pools",
    )
    for options in (("process_pools", "process_pools"), ("unknown",)):
        with pytest.raises(ValueError, match="Unknown or duplicate"):
            validate_sandbox_options(options)
