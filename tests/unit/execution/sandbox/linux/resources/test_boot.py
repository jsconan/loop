"""Verify the protected boot entry before invoking native recovery."""

import os
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import management
from loop.execution.sandbox.linux.resources import boot


@pytest.fixture
def boot_entry(tmp_path, monkeypatch):
    """Bind the boot entry and isolate root-only native recovery."""
    entry = tmp_path / "loop_boot.py"
    entry.write_text("boot")
    monkeypatch.setattr(boot, "__file__", str(entry))
    monkeypatch.setattr(boot.sys, "path", list(boot.sys.path))
    monkeypatch.setattr(boot.os, "getuid", lambda: 0)
    read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "0 0 4294967295"
            if path == Path("/proc/self/uid_map")
            else read_text(path, *args, **kwargs)
        ),
    )
    lstat = Path.lstat
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda path, *args, **kwargs: SimpleNamespace(
            st_uid=0, st_mode=lstat(path, *args, **kwargs).st_mode & ~0o6022
        ),
    )
    monkeypatch.setattr(boot.time, "monotonic", lambda: 10)
    recover = Mock()
    monkeypatch.setattr(management, "recover_boot_generation", recover)
    return entry, recover


def test_boot_entry_runs_authenticated_recovery(boot_entry):
    """A canonical protected root entry invokes recovery once with a bounded deadline."""
    entry, recover = boot_entry
    boot.main()
    recover.assert_called_once_with(entry.parent / "management.json", 70)


@pytest.mark.parametrize(
    "change", ["user", "mapping", "name", "alias", "owner", "writable", "symlink"]
)
def test_boot_entry_refuses_untrusted_authority(boot_entry, monkeypatch, change):
    """Invalid administrator or entry ancestry never invokes installed recovery."""
    entry, recover = boot_entry
    if change == "user":
        monkeypatch.setattr(boot.os, "getuid", lambda: 1000)
    elif change == "mapping":
        monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: "0 1000 1")
    elif change == "name":
        monkeypatch.setattr(boot, "__file__", str(entry.with_name("other.py")))
    elif change == "alias":
        monkeypatch.setattr(Path, "resolve", lambda *args, **kwargs: Path("/other"))
    elif change == "owner":
        monkeypatch.setattr(
            Path, "lstat", lambda *args, **kwargs: SimpleNamespace(st_uid=1000, st_mode=0o100444)
        )
    elif change == "writable":
        monkeypatch.setattr(
            Path, "lstat", lambda *args, **kwargs: SimpleNamespace(st_uid=0, st_mode=0o100666)
        )
    else:
        monkeypatch.setattr(
            Path, "lstat", lambda *args, **kwargs: SimpleNamespace(st_uid=0, st_mode=0o120444)
        )
    with pytest.raises((ValueError, PermissionError)):
        boot.main()
    recover.assert_not_called()


def test_boot_script_cannot_run_as_unprivileged_source(monkeypatch):
    """Running the package resource directly does not confer administrator authority."""
    monkeypatch.setattr(os, "getuid", lambda: 1000)
    with pytest.raises(PermissionError, match="initial-host"):
        runpy.run_path(boot.__file__, run_name="__main__")
