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


@pytest.fixture
def monkeypatch(unit_patch):
    """Restore case-specific mocks before tearing down shared unit boundaries."""
    with pytest.MonkeyPatch.context() as patch:
        yield patch


@pytest.fixture
def unit_patch(isolated_environment):
    """Retain unit boundary guards when a case undoes its own mocks."""
    with pytest.MonkeyPatch.context() as patch:
        yield patch


@pytest.fixture(autouse=True)
def deny_external_resources(unit_patch, isolated_environment):
    """Require explicit doubles instead of consuming host processes or network services."""
    unit_patch.setattr(sys.modules[__name__], "_BOUNDARIES_GUARDED", True)
    unit_patch.setattr(subprocess, "Popen", forbidden_resource)
    unit_patch.setattr("builtins.input", forbidden_resource)
    unit_patch.setattr(getpass, "getpass", forbidden_resource)
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
            unit_patch.setattr(os, name, forbidden_resource)
    unit_patch.setattr(os, "kill", forbidden_resource)
    if hasattr(os, "killpg"):
        unit_patch.setattr(os, "killpg", forbidden_resource)
    unit_patch.setattr(socket.socket, "connect", forbidden_resource)
    unit_patch.setattr(socket.socket, "connect_ex", forbidden_resource)
    unit_patch.setattr(socket.socket, "bind", forbidden_resource)
    unit_patch.setattr(socket.socket, "listen", forbidden_resource)
    for name in (
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
        "gethostbyaddr",
        "getnameinfo",
    ):
        unit_patch.setattr(socket, name, forbidden_resource)


@pytest.fixture(autouse=True)
def isolated_workers(unit_patch, isolated_environment):
    """Own fallback worker admission and shutdown within each test."""
    with ThreadPoolExecutor(
        max_workers=asyncio_utils._WORKER_COUNT, thread_name_prefix="loop-test-io"
    ) as executor:
        unit_patch.setattr(asyncio_utils, "_WORKER_EXECUTOR", executor)
        unit_patch.setattr(
            asyncio_utils,
            "_WORKER_SLOTS",
            BoundedSemaphore(asyncio_utils._WORKER_COUNT + asyncio_utils._WORKER_QUEUE_SIZE),
        )
        yield


@pytest.fixture(autouse=True)
def close_tool_registries(unit_patch, isolated_environment):
    """Close every successfully constructed tool runner before its scratch is removed."""
    registries = []
    initialize = ToolRegistry.__init__

    def tracked_initialize(registry, *args, **kwargs):
        """Retain successfully initialized registries for scoped cleanup."""
        initialize(registry, *args, **kwargs)
        registries.append(registry)

    unit_patch.setattr(ToolRegistry, "__init__", tracked_initialize)
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
def isolated_prompt_sessions(unit_patch):
    """Replace terminal input with fresh explicit doubles in portable unit tests."""

    def create_session(*args, **kwargs):
        """Fail unconfigured input immediately instead of touching the host terminal."""
        return SimpleNamespace(
            default_buffer=SimpleNamespace(text=""),
            prompt=Mock(side_effect=forbidden_resource),
            prompt_async=AsyncMock(side_effect=forbidden_resource),
        )

    unit_patch.setattr("loop.interaction.console.PromptSession", create_session)


@pytest.fixture(autouse=True)
def isolated_native_accounts_and_acls(unit_patch, isolated_environment):
    """Provide synthetic account and ACL boundaries without host directory services or libraries."""
    unit_patch.setattr(
        pwd,
        "getpwuid",
        lambda uid: SimpleNamespace(
            pw_uid=uid, pw_gid=os.getgid(), pw_name="test-user", pw_dir=os.environ["HOME"]
        ),
    )
    unit_patch.setattr(pwd, "getpwnam", forbidden_resource)
    native = SimpleNamespace(
        acl_get_fd_np=Mock(return_value=None), acl_get_entry=Mock(), acl_free=Mock()
    )
    unit_patch.setattr(
        "loop.utils.path.ctypes",
        SimpleNamespace(
            **{**vars(ctypes), "CDLL": Mock(return_value=native), "get_errno": lambda: errno.ENOENT}
        ),
    )

    def absent_acl(*args, **kwargs):
        """Report a synthetic filesystem with no extended access grants."""
        raise OSError(errno.ENODATA, "no test ACL")

    unit_patch.setattr(os, "getxattr", absent_acl, raising=False)


@pytest.fixture(scope="session")
def builtin_permission_catalog():
    """Read the shipped permission catalog once for collaborator doubles."""
    presets, failures = PermissionPreset.load_builtin_presets()
    assert not failures
    return presets


@pytest.fixture(autouse=True)
def isolated_permission_catalog(unit_patch, builtin_permission_catalog):
    """Give catalog consumers fresh models without repeatedly parsing unchanged package YAML."""

    def load(cls):
        """Clone each catalog entry so mutations cannot reach the template or another case."""
        return tuple(preset.model_copy(deep=True) for preset in builtin_permission_catalog), ()

    unit_patch.setattr(PermissionPreset, "load_builtin_presets", classmethod(load))


@pytest.fixture(scope="session")
def clock_bindings():
    """Locate immutable module clock aliases once after collection."""
    import time

    return tuple(
        (module, alias, "module" if value is time else value.__name__)
        for name, module in tuple(sys.modules.items())
        if module is not None and name.startswith(("loop.", "tests.unit."))
        for alias, value in tuple(vars(module).items())
        if value is time
        or any(value is getattr(time, clock) for clock in ("monotonic", "perf_counter", "sleep"))
    )


@pytest.fixture(autouse=True)
def controlled_time(unit_patch, clock_bindings):
    """Keep application and case clocks synthetic and reject native sleep boundaries."""
    import time
    from datetime import UTC, datetime, timedelta
    from itertools import count

    native_time = time

    def reject_sleep(*args, **kwargs):
        """Require a controlled collaborator for every elapsed-time delay."""
        raise AssertionError("Unit tests must not use real sleeps or elapsed-time delays")

    boundary = SimpleNamespace(
        **{
            **vars(native_time),
            "time": lambda: 1_800_000_000.0,
            "time_ns": lambda: 1_800_000_000_000_000_000,
            "monotonic": lambda: 1000.0,
            "monotonic_ns": lambda: 1_000_000_000_000,
            "perf_counter": lambda: 1000.0,
            "perf_counter_ns": lambda: 1_000_000_000_000,
            "process_time": lambda: 0.0,
            "process_time_ns": lambda: 0,
            "thread_time": lambda: 0.0,
            "thread_time_ns": lambda: 0,
            "sleep": reject_sleep,
        }
    )
    for module, alias, kind in clock_bindings:
        unit_patch.setattr(module, alias, boundary if kind == "module" else getattr(boundary, kind))
    timestamps = count()

    class ClockDateTime(datetime):
        """Supply fixed timestamp inputs without consulting the host date or timezone."""

        @classmethod
        def now(cls, tz=None):
            """Return the fixed instant using UTC as the synthetic local timezone."""
            value = datetime(2026, 10, 1, 8, 0, tzinfo=UTC) + timedelta(
                microseconds=next(timestamps)
            )
            return value if tz is None else value.astimezone(tz)

    unit_patch.setattr("loop.utils.dates.datetime", ClockDateTime)
    unit_patch.setattr("loop.backend.openai.datetime", ClockDateTime)


@pytest.fixture(autouse=True)
def telemetry_writer(unit_patch):
    """Execute queued telemetry at explicit lifecycle checkpoints without native threads."""
    import queue
    import threading

    class WriterIdle(BaseException):
        """End a controlled writer turn when it has no queued input."""

    threads = []
    empty_reads = set()
    reads = [0]

    class Queue(queue.Queue):
        """Deliver queued records or an explicitly selected empty-queue response."""

        def get(self, block=True, timeout=None):
            index = reads[0]
            reads[0] += 1
            if index in empty_reads:
                raise queue.Empty
            try:
                return super().get(block=False)
            except queue.Empty:
                raise WriterIdle from None

    def run():
        """Deliver each admitted writer's currently queued work in submission order."""
        for thread in threads:
            if thread.running or thread.finished:
                continue
            thread.running = True
            try:
                thread.target()
            except WriterIdle:
                pass
            else:
                thread.finished = True
            finally:
                thread.running = False

    class Event:
        """Expose completion immediately, with an explicit writer delivery checkpoint."""

        def __init__(self):
            self.signalled = False

        def set(self):
            """Publish completion without signalling a native wait primitive."""
            self.signalled = True

        def is_set(self):
            """Return the synthetic completion state."""
            return self.signalled

        def wait(self, timeout=None):
            """Deliver queued work for a lifecycle request, without elapsed-time waiting."""
            if timeout != 0:
                run()
            return self.signalled

    class Thread:
        """Retain a writer callback until its owning lifecycle checkpoint."""

        def __init__(self, *, target, name, daemon):
            self.target = target
            self.running = False
            self.finished = False
            threads.append(self)

        def start(self):
            """Record admission; delivery is controlled by lifecycle requests."""

        def join(self, timeout=None):
            """Settle queued writer work without creating or joining an OS thread."""
            run()

        def is_alive(self):
            """Return whether the controlled writer has consumed its close request."""
            return not self.finished

    unit_patch.setattr(
        "loop.telemetry.telemetry.threading",
        SimpleNamespace(**{**vars(threading), "Thread": Thread, "Event": Event}),
    )
    unit_patch.setattr(
        "loop.telemetry.telemetry.queue", SimpleNamespace(**{**vars(queue), "Queue": Queue})
    )
    return SimpleNamespace(run=run, empty_reads=empty_reads)
