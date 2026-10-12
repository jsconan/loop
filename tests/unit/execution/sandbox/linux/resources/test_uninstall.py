"""Verify the exported installer-owned uninstaller rejects unreviewed or unsafe authority."""

import hashlib
import runpy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import management as linux_management
from loop.execution.sandbox.linux.resources import uninstall as module


@pytest.fixture
def export(tmp_path, monkeypatch):
    """Isolate retained code and original host authority without any administrator operation."""
    root = tmp_path.resolve()
    entry = root / "loop_uninstall.py"
    entry.write_bytes(b"entry")
    entry.chmod(0o555)
    record = root / "management.json"
    record.write_bytes(b"record")
    record.chmod(0o444)
    digest = hashlib.sha256(record.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "__file__", str(entry))
    monkeypatch.setattr(module.os, "getuid", lambda: 0)
    read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, **kw: (
            "0 0 4294967295" if str(path) == "/proc/self/uid_map" else read(path, **kw)
        ),
    )
    real = Path.lstat

    def details(path):
        """Isolate root ownership and canonical protected ancestry for disposable real files."""
        info = real(path)
        return SimpleNamespace(
            st_uid=0, st_mode=info.st_mode & ~0o6022, st_nlink=info.st_nlink, st_size=info.st_size
        )

    monkeypatch.setattr(Path, "lstat", details)
    monkeypatch.setattr(module.sys, "argv", [str(entry), digest])
    monkeypatch.setattr(module.sys, "path", list(module.sys.path))
    monkeypatch.setattr(module.time, "monotonic", lambda: 10)
    authority = Mock(digest=digest)
    bound = Mock(path=entry, open_verified=Mock(return_value=nullcontext()))
    authenticate = Mock(return_value=(authority, bound))
    retire = Mock()
    monkeypatch.setattr(linux_management, "uninstall_authority", authenticate)
    monkeypatch.setattr(linux_management, "uninstall_generation", retire)
    return entry, record, authenticate, retire


def test_export_independent_of_application_uses_exact_record(export, capsys):
    """The protected retained distribution supplies checked retirement after application removal."""
    _, record, authenticate, retire = export
    module.main()
    authenticate.assert_called_once_with(record)
    retire.assert_called_once_with(record, 15)
    assert "capacity released" in capsys.readouterr().out


@pytest.mark.parametrize(
    "change",
    [
        "uid",
        "mapping",
        "args",
        "digest",
        "name",
        "alias",
        "owner",
        "mode",
        "symlink",
        "record-type",
        "record-owner",
        "record-links",
        "record-mode",
        "record-size",
        "record-digest",
        "bound-digest",
        "bound-entry",
    ],
)
def test_export_refuses_unsafe_host_ancestry_and_changed_review(export, monkeypatch, change):
    """Require initial authority and exact protected decisions before removal."""
    entry, record, authenticate, retire = export
    if change == "uid":
        monkeypatch.setattr(module.os, "getuid", lambda: 1000)
    elif change == "mapping":
        monkeypatch.setattr(Path, "read_text", lambda *args, **kw: "0 1000 1")
    elif change == "args":
        monkeypatch.setattr(module.sys, "argv", [str(entry)])
    elif change == "digest":
        module.sys.argv[1] = "invalid"
    elif change == "name":
        monkeypatch.setattr(module, "__file__", str(entry.with_name("other.py")))
    elif change == "alias":
        monkeypatch.setattr(Path, "resolve", lambda *args, **kw: Path("/other"))
    elif change.startswith("bound"):
        authority, bound = authenticate.return_value
        if change == "bound-digest":
            authority.digest = "0" * 64
        else:
            bound.path = Path("/other")
    elif change == "record-digest":
        record.chmod(0o644)
        record.write_bytes(b"changed")
        record.chmod(0o444)
    else:
        original = Path.lstat

        def details(path):
            """Change exactly one requested security property at the relevant boundary."""
            info = original(path)
            values = vars(info).copy()
            if path == (record if change.startswith("record") else entry):
                if change.endswith("owner"):
                    values["st_uid"] = 1000
                elif change.endswith("links"):
                    values["st_nlink"] = 2
                elif change.endswith("size"):
                    values["st_size"] = 65537
                elif change.endswith("type"):
                    values["st_mode"] = 0o040755
                elif change == "symlink":
                    values["st_mode"] = 0o120777
                else:
                    values["st_mode"] |= 0o022
            return SimpleNamespace(**values)

        monkeypatch.setattr(Path, "lstat", details)
    with pytest.raises((PermissionError, ValueError)):
        module.main()
    retire.assert_not_called()


def test_export_script_entry_requires_root_without_elevation(monkeypatch):
    """Direct script invocation refuses ordinary callers instead of starting system elevation."""
    monkeypatch.setattr(module.os, "getuid", lambda: 1000)
    with pytest.raises(PermissionError, match="administrator"):
        runpy.run_path(module.__file__, run_name="__main__")
