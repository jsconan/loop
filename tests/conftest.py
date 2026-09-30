"""Isolate both suites from user configuration and clean all disposable test data."""

import fcntl
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from weakref import WeakSet

import pytest

sys.dont_write_bytecode = True

from loop.permissions import PermissionManager
from loop.permissions import manager as permission_module
from loop.telemetry import get_telemetry, set_telemetry
from loop.utils import content as content_module

_OUTPUT_ROOT: str | None = None
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_REALPATH = os.path.realpath
_ABSPATH = os.path.abspath
_READLINK = os.readlink
_FCNTL = fcntl.fcntl
_OPEN = os.open


def _check_output(path, *, follow=True, directory=None):
    """Reject mutation paths, including descriptor-relative paths, outside test scratch."""
    if isinstance(path, int):
        return
    path = os.fsdecode(path)
    if not os.path.isabs(path) and directory is not None and directory >= 0:
        if sys.platform == "darwin":
            parent = os.fsdecode(_FCNTL(directory, 50, b"\0" * 1024).split(b"\0", 1)[0])
        else:
            parent = _READLINK(f"/proc/self/fd/{directory}")
        path = os.path.join(parent, path)
    absolute = _ABSPATH(path)
    if follow and absolute == _ABSPATH(os.devnull):
        return
    canonical = (
        _REALPATH(absolute)
        if follow
        else os.path.join(_REALPATH(os.path.dirname(absolute)), os.path.basename(absolute))
    )
    if canonical != _OUTPUT_ROOT and not canonical.startswith(_OUTPUT_ROOT + os.sep):
        raise AssertionError(f"Test output escapes disposable scratch: {canonical}")


def _isolated_open(path, flags, mode=0o777, *, dir_fd=None):
    """Check os.open's directory descriptor, which Python's open audit event omits."""
    if _OUTPUT_ROOT is not None and flags & _WRITE_FLAGS:
        _check_output(path, directory=dir_fd)
    return _OPEN(path, flags, mode, dir_fd=dir_fd)


def _audit_output(event, arguments):
    """Guard Python filesystem mutations while a test's isolated environment is active."""
    if _OUTPUT_ROOT is None:
        return
    if event == "open":
        path, mode, flags = arguments
        if (isinstance(flags, int) and flags & _WRITE_FLAGS) or (
            isinstance(mode, str) and any(value in mode for value in "wax+")
        ):
            _check_output(path)
    elif event in {
        "os.mkdir",
        "os.remove",
        "os.rmdir",
        "os.chmod",
        "os.truncate",
        "os.utime",
        "os.chown",
    }:
        directory = arguments[-1] if event != "os.truncate" else None
        _check_output(
            arguments[0],
            follow=event in {"os.chmod", "os.truncate", "os.utime", "os.chown"},
            directory=directory,
        )
    elif event in {"os.rename", "os.link"}:
        for index, path in enumerate(arguments[:2]):
            _check_output(path, follow=False, directory=arguments[index + 2])
    elif event == "os.symlink":
        _check_output(arguments[1], follow=False, directory=arguments[2])
    elif event == "sqlite3.connect":
        database = os.fsdecode(arguments[0])
        if database.startswith("file:"):
            database = database[5:].split("?", 1)[0]
        if database != ":memory:":
            _check_output(database)


sys.addaudithook(_audit_output)


@pytest.fixture
def tmp_path(monkeypatch):
    """Provide fresh scratch and remove it, including sibling fixtures, after each test."""
    # /tmp is a model-facing virtual alias; physical fixtures must have another prefix.
    parent = "/var/tmp" if Path(tempfile.gettempdir()).resolve() == Path("/tmp") else None
    with tempfile.TemporaryDirectory(prefix="loop-test-", dir=parent) as directory:
        workspace = Path(directory).resolve() / "workspace"
        workspace.mkdir()
        yield workspace
        monkeypatch.undo()


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    """Replace ambient inputs and redirect every application and temporary output root."""
    root = tmp_path.parent
    home = root / "home"
    temporary = root / "temporary"
    working = root / "cwd"
    working.mkdir()
    home.mkdir()
    temporary.mkdir()
    for name in tuple(os.environ):
        monkeypatch.delenv(name, raising=False)
    for name, value in {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "TMPDIR": str(temporary),
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "XDG_CONFIG_HOME": str(home / "config"),
        "XDG_DATA_HOME": str(home / "data"),
        "XDG_STATE_HOME": str(home / "state"),
        "XDG_CACHE_HOME": str(home / "cache"),
        "LOOP_CONFIG_HOME": str(home / "loop-config"),
        "LOOP_DATA_HOME": str(home / "loop-data"),
        "LOOP_STATE_HOME": str(home / "loop-state"),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    monkeypatch.chdir(working)
    monkeypatch.setattr(sys.modules[__name__], "_OUTPUT_ROOT", str(root))
    cache = root / "content-cache"
    cache.mkdir()
    monkeypatch.setattr(content_module, "_CACHE", SimpleNamespace(name=str(cache)))
    monkeypatch.setattr(content_module, "_SOURCES", {})
    monkeypatch.setattr(content_module, "_METADATA", {})
    monkeypatch.setattr(permission_module, "_LIVE_MANAGERS", WeakSet())
    monkeypatch.setattr(os, "open", _isolated_open)
    previous = get_telemetry()
    set_telemetry(None)
    yield
    current = get_telemetry()
    if current is not None and current is not previous:
        current.close()
    set_telemetry(previous)


@pytest.fixture(autouse=True)
def close_permission_managers(monkeypatch, isolated_environment):
    """Close temporary directories owned by permission managers created in each test."""
    managers: list[PermissionManager] = []
    initialize: Callable[..., None] = PermissionManager.__init__

    def tracked_initialize(manager: PermissionManager, *args: object, **kwargs: object) -> None:
        """Track only successfully initialized permission authorities for cleanup."""
        initialize(manager, *args, **kwargs)
        managers.append(manager)

    monkeypatch.setattr(PermissionManager, "__init__", tracked_initialize)
    yield
    for manager in reversed(managers):
        manager.close()
