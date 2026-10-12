"""Verify the privileged loader authenticates bounded bytes before package imports."""

import hashlib
import json
import re
import runpy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import setup as owner
from loop.execution.sandbox.linux.resources import setup_loader as module


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    """Provide authenticated disposable input and isolated root/terminal collaborators."""
    data = {
        "contents": {"nested/file.py": b"code".hex(), "nested/other.py": b"other".hex()},
        "runtime": ["nested/file.py"],
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data))
    argv = [
        "-c",
        str(path),
        hashlib.sha256(path.read_bytes()).hexdigest(),
        "install",
        "generation",
        "1000",
        "1000",
        "",
    ]
    monkeypatch.setattr(module, "sys", SimpleNamespace(argv=argv, path=[]))
    monkeypatch.setattr(module.os, "getuid", lambda: 0)
    read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **k: (
            "0 0 4294967295" if str(path) == "/proc/self/uid_map" else read_text(path, *a, **k)
        ),
    )
    temporary = module.tempfile.TemporaryDirectory

    @contextmanager
    def scratch(*, prefix, dir):
        """Redirect authenticated import output to the test's disposable root."""
        assert dir == "/run"
        with temporary(prefix=prefix, dir=tmp_path) as root:
            yield root

    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", scratch)
    install = Mock()
    monkeypatch.setattr(owner, "install_reviewed", install)
    return path, data, argv, install


def replace_payload(reviewed, payload):
    """Bind each malformed test input to its fresh exact digest."""
    path, _, argv, _ = reviewed
    path.write_text(json.dumps(payload))
    argv[2] = hashlib.sha256(path.read_bytes()).hexdigest()


def assert_loader_refusal(capsys, expected_message):
    """Assert the executable entry returns a sanitized refusal without a traceback."""
    assert module.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    errors = json.loads(captured.out)["errors"]
    assert errors
    assert re.search(expected_message, "\n".join(errors))


def test_loader_authenticates_and_removes_import_snapshot(reviewed):
    """Load only authenticated runtime modules and remove temporary imports."""
    assert module.main() == 0
    reviewed[3].assert_called_once_with(
        "install", "generation", 1000, 1000, "", {"nested/file.py": b"code"}
    )
    assert not Path(module.sys.path[0]).exists()


@pytest.mark.parametrize(
    "case",
    [
        "uid",
        "mapping",
        "args",
        "digest",
        "changed",
        "symlink",
        "nonregular",
        "links",
        "declared_size",
        "read_size",
    ],
)
def test_loader_refuses_before_import(reviewed, monkeypatch, case, capsys):
    """Missing root authority and unsafe or changed input cannot execute reviewed package code."""
    path, _, argv, install = reviewed
    if case == "uid":
        monkeypatch.setattr(module.os, "getuid", lambda: 1000)
    elif case == "mapping":
        monkeypatch.setattr(Path, "read_text", lambda *a, **k: "0 1000 1")
    elif case == "args":
        argv.pop()
    elif case == "digest":
        argv[2] = "invalid"
    elif case == "changed":
        path.write_bytes(b"changed")
    elif case == "symlink":
        link = path.with_name("link")
        link.symlink_to(path)
        argv[1] = str(link)
    elif case in {"nonregular", "links", "declared_size"}:
        info = path.stat()
        monkeypatch.setattr(
            module.os,
            "fstat",
            lambda fd: SimpleNamespace(
                st_mode=0o040755 if case == "nonregular" else info.st_mode,
                st_nlink=2 if case == "links" else 1,
                st_size=34 * 1024 * 1024 + 1 if case == "declared_size" else info.st_size,
            ),
        )
    elif case == "read_size":
        # A regular input that grows after metadata inspection still has an allocation bound.
        path.write_bytes(b"x" * (34 * 1024 * 1024 + 1))
        info = path.stat()
        monkeypatch.setattr(
            module.os,
            "fstat",
            lambda fd: SimpleNamespace(st_mode=info.st_mode, st_nlink=1, st_size=1),
        )
    assert_loader_refusal(capsys, "administrator|reviewed|Reviewed|symbolic|Too many")
    install.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"contents": [], "runtime": []},
        {"contents": {}, "runtime": {}},
        {"contents": {}, "runtime": []},
        {"contents": {str(i): "00" for i in range(257)}, "runtime": []},
        {"contents": {"x": "00" * (17 * 1024 * 1024 + 1)}, "runtime": []},
    ],
)
def test_loader_refuses_schema_and_inventory_bounds(reviewed, payload, capsys):
    """Malformed shapes and excessive authenticated import inventories remain rejected."""
    replace_payload(reviewed, payload)
    assert_loader_refusal(capsys, "schema|inventory|bound")
    reviewed[3].assert_not_called()


@pytest.mark.parametrize("name", ["bad space", "/absolute", "a/../b", "a//b", "a/./b"])
def test_loader_refuses_noncanonical_members(reviewed, name, capsys):
    """Exact authenticated bytes cannot authorize extraction outside canonical private paths."""
    replace_payload(reviewed, {"contents": {name: "00"}, "runtime": []})
    assert_loader_refusal(capsys, "import member")
    reviewed[3].assert_not_called()


@pytest.mark.parametrize("runtime", [["absent"], ["nested/file.py", "nested/file.py"]])
def test_loader_refuses_changed_runtime_inventory(reviewed, runtime, capsys):
    """Unknown and duplicate retained runtime members refuse before privileged imports."""
    payload = reviewed[1]
    payload["runtime"] = runtime
    replace_payload(reviewed, payload)
    assert_loader_refusal(capsys, "runtime inventory")
    reviewed[3].assert_not_called()


def test_loader_cleanup_after_owner_failure(reviewed, capsys):
    """Native owner refusal removes only temporary imports and preserves its failure."""
    reviewed[3].side_effect = ValueError("native owner refused")
    assert_loader_refusal(capsys, "native owner refused")
    assert not Path(module.sys.path[0]).exists()


def test_loader_direct_entry_requires_root(monkeypatch):
    """Direct script execution refuses ordinary users before state or package access."""
    monkeypatch.setattr(module.os, "getuid", lambda: 1000)
    with pytest.raises(SystemExit) as result:
        runpy.run_path(module.__file__, run_name="__main__")

    assert result.value.code == 1


@pytest.mark.parametrize(
    "failure",
    [
        None,
        ValueError(""),
        ExceptionGroup("group", [ValueError("same\nraw"), OSError("same"), OSError("other\x1b")]),
    ],
)
def test_loader_renders_bounded_distinct_failures(reviewed, capsys, failure):
    """The executable boundary returns sanitized distinct reasons without traceback output."""
    reviewed[3].side_effect = failure
    assert module.main() == (0 if failure is None else 1)
    captured = capsys.readouterr()
    assert captured.err == ""
    if failure is None:
        assert captured.out == ""
    else:
        payload = json.loads(captured.out)
        assert payload["errors"] == (
            ["Management refused."] if isinstance(failure, ValueError) else ["same", "other?"]
        )
