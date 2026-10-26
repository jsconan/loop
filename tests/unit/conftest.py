"""Fail immediately when a unit test tries to use a real external boundary."""

import ctypes
import errno
import getpass
import os
import pwd
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Event, Lock
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import loop.utils.asyncio as asyncio_utils
from loop.permissions import PermissionPreset
from loop.tooling import ToolRegistry

_EXTERNAL_EVENTS = frozenset(
    {
        "subprocess.Popen",
        "os.system",
        "os.fork",
        "os.forkpty",
        "os.exec",
        "os.posix_spawn",
        "os.kill",
        "os.killpg",
        "socket.connect",
        "socket.bind",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.getnameinfo",
        "ctypes.dlopen",
    }
)
_BOUNDARIES_GUARDED = False


def _audit_external_resources(event, arguments):
    """Reject native calls even when a caller retained a pre-fixture function alias."""
    if _BOUNDARIES_GUARDED and event in _EXTERNAL_EVENTS:
        forbidden_resource()


sys.addaudithook(_audit_external_resources)


def forbidden_resource(*args, **kwargs):
    """Reject real launches, signals, network connections and DNS in unit tests."""
    raise AssertionError("Unit tests must replace external resource boundaries with scoped doubles")


@pytest.fixture(autouse=True)
def deny_external_resources(monkeypatch, isolated_environment):
    """Require explicit doubles instead of consuming host processes or network services."""
    monkeypatch.setattr(sys.modules[__name__], "_BOUNDARIES_GUARDED", True)
    monkeypatch.setattr(subprocess, "Popen", forbidden_resource)
    monkeypatch.setattr("builtins.input", forbidden_resource)
    monkeypatch.setattr(getpass, "getpass", forbidden_resource)
    for name in (
        "system",
        "fork",
        "forkpty",
        "posix_spawn",
        "posix_spawnp",
        "execv",
        "execve",
        "execl",
        "execlp",
        "execlpe",
        "execvp",
        "execvpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
    ):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, forbidden_resource)
    monkeypatch.setattr(os, "kill", forbidden_resource)
    if hasattr(os, "killpg"):
        monkeypatch.setattr(os, "killpg", forbidden_resource)
    monkeypatch.setattr(socket.socket, "connect", forbidden_resource)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden_resource)
    monkeypatch.setattr(socket.socket, "bind", forbidden_resource)
    monkeypatch.setattr(socket.socket, "listen", forbidden_resource)
    for name in (
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
        "gethostbyaddr",
        "getnameinfo",
    ):
        monkeypatch.setattr(socket, name, forbidden_resource)


@pytest.fixture(autouse=True)
def isolated_workers(monkeypatch, isolated_environment):
    """Own fallback worker admission and shutdown within each test."""
    with ThreadPoolExecutor(
        max_workers=asyncio_utils._WORKER_COUNT, thread_name_prefix="loop-test-io"
    ) as executor:
        monkeypatch.setattr(asyncio_utils, "_WORKER_EXECUTOR", executor)
        monkeypatch.setattr(
            asyncio_utils,
            "_WORKER_SLOTS",
            BoundedSemaphore(asyncio_utils._WORKER_COUNT + asyncio_utils._WORKER_QUEUE_SIZE),
        )
        yield


@pytest.fixture(autouse=True)
def close_tool_registries(monkeypatch, isolated_environment):
    """Close every successfully constructed tool runner before its scratch is removed."""
    registries = []
    initialize = ToolRegistry.__init__

    def tracked_initialize(registry, *args, **kwargs):
        """Retain successfully initialized registries for scoped cleanup."""
        initialize(registry, *args, **kwargs)
        registries.append(registry)

    monkeypatch.setattr(ToolRegistry, "__init__", tracked_initialize)
    yield
    for registry in reversed(registries):
        registry.close()


@pytest.fixture
def contended_lock():
    """Expose actual lock contention without waiting to prove an event's absence."""
    lock = Lock()
    contended = Event()

    class ObservedLock:
        """Signal a blocked acquisition before waiting for its current owner."""

        def __enter__(self):
            if not lock.acquire(blocking=False):
                contended.set()
                assert lock.acquire(timeout=5), "test lock owner was not released"
            return self

        def __exit__(self, *exc):
            lock.release()

    return SimpleNamespace(lock=ObservedLock(), contended=contended)


@pytest.fixture(autouse=True)
def isolated_prompt_sessions(monkeypatch):
    """Replace terminal input with fresh explicit doubles in portable unit tests."""

    def create_session(*args, **kwargs):
        """Fail unconfigured input immediately instead of touching the host terminal."""
        return SimpleNamespace(
            default_buffer=SimpleNamespace(text=""),
            prompt=Mock(side_effect=forbidden_resource),
            prompt_async=AsyncMock(side_effect=forbidden_resource),
        )

    monkeypatch.setattr("loop.interaction.console.PromptSession", create_session)


@pytest.fixture(autouse=True)
def isolated_native_accounts_and_acls(monkeypatch, isolated_environment):
    """Provide synthetic account and ACL boundaries without host directory services or libraries."""
    monkeypatch.setattr(
        pwd,
        "getpwuid",
        lambda uid: SimpleNamespace(
            pw_uid=uid, pw_gid=os.getgid(), pw_name="test-user", pw_dir=os.environ["HOME"]
        ),
    )
    monkeypatch.setattr(pwd, "getpwnam", forbidden_resource)
    native = SimpleNamespace(
        acl_get_fd_np=Mock(return_value=None), acl_get_entry=Mock(), acl_free=Mock()
    )
    monkeypatch.setattr(
        "loop.utils.path.ctypes",
        SimpleNamespace(
            **{**vars(ctypes), "CDLL": Mock(return_value=native), "get_errno": lambda: errno.ENOENT}
        ),
    )

    def absent_acl(*args, **kwargs):
        """Report a synthetic filesystem with no extended access grants."""
        raise OSError(errno.ENODATA, "no test ACL")

    monkeypatch.setattr(os, "getxattr", absent_acl, raising=False)


@pytest.fixture(scope="session")
def builtin_permission_catalog():
    """Read the shipped permission catalog once for collaborator doubles."""
    presets, failures = PermissionPreset.load_builtin_presets()
    assert not failures
    return presets


@pytest.fixture(autouse=True)
def isolated_permission_catalog(monkeypatch, builtin_permission_catalog):
    """Give catalog consumers fresh models without repeatedly parsing unchanged package YAML."""

    def load(cls):
        """Clone each catalog entry so mutations cannot reach the template or another case."""
        return tuple(preset.model_copy(deep=True) for preset in builtin_permission_catalog), ()

    monkeypatch.setattr(PermissionPreset, "load_builtin_presets", classmethod(load))
