"""Bind CPython and loky spawn synchronization to an explicitly selected endpoint."""

import _multiprocessing
import errno
import json
import math
import os
import socket
import threading
import time

SLOT_COUNT = 512
SYSTEM_SYSCONF = os.sysconf
CLIENT_LOCK = threading.Lock()
CLIENT_SOCKET = None
CLIENT_STREAM = None


def disconnect():
    """Close this process's references when its transport is retired."""
    global CLIENT_SOCKET, CLIENT_STREAM
    if CLIENT_STREAM is not None:
        CLIENT_STREAM.close()
        CLIENT_STREAM = None
    if CLIENT_SOCKET is not None:
        CLIENT_SOCKET.close()
        CLIENT_SOCKET = None


def disconnect_child():
    """Keep forked interpreters from sharing the parent's control stream."""
    global CLIENT_LOCK
    disconnect()
    CLIENT_LOCK = threading.Lock()


os.register_at_fork(after_in_child=disconnect_child)


def request(operation, identifier, *arguments):
    """Send a single counter request over the attempt-owned endpoint."""
    global CLIENT_SOCKET, CLIENT_STREAM
    with CLIENT_LOCK:
        try:
            if CLIENT_SOCKET is None:
                CLIENT_SOCKET = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                CLIENT_SOCKET.settimeout(2)
                CLIENT_SOCKET.connect(os.environ["LOOP_SYNC_SOCKET"])
                CLIENT_STREAM = CLIENT_SOCKET.makefile("rb")
            CLIENT_SOCKET.sendall(json.dumps([operation, identifier, *arguments]).encode() + b"\n")
            payload = CLIENT_STREAM.readline(512)
        except (FileNotFoundError, ConnectionError, PermissionError) as error:
            disconnect()
            raise FileNotFoundError(errno.ENOENT, "Semaphore namespace retired") from error
        except OSError:
            disconnect()
            raise
    if not payload:
        raise FileNotFoundError(errno.ENOENT, "Semaphore namespace retired")
    code, result = json.loads(payload)
    if code == errno.EINVAL:
        raise ValueError("Invalid semaphore operation")
    if code:
        raise OSError(code, os.strerror(code))
    return result


def sem_unlink(name):
    """Remove one logical name from the current attempt."""
    request("unlink", name)


class SemLock:
    """Implement spawn and loky semaphore operations over the optional broker."""

    SEM_VALUE_MAX = (1 << 31) - 1
    _handle = None

    def __init__(self, kind, value, maxvalue, name, unlink):
        if kind not in (0, 1):
            raise ValueError("Invalid semaphore kind")
        handle = request("create", name, value, maxvalue)
        self._bind(handle, kind, maxvalue, None if unlink else name)
        if unlink:
            sem_unlink(name)

    def _bind(self, handle, kind, maxvalue, name):
        """Initialize process-local state for one broker handle."""
        self._handle = handle
        self._pid = os.getpid()
        self._thread = None
        self._acquires = 0
        self.handle = 0
        self.kind = kind
        self.maxvalue = maxvalue
        self.name = name

    @classmethod
    def _rebuild(cls, _handle, kind, maxvalue, name):
        """Open a named semaphore when unpickled by a spawned process."""
        item = cls.__new__(cls)
        item._bind(request("open", name), kind, maxvalue, name)
        return item

    def _after_fork(self):
        """Clear inherited recursion state while keeping fork usage disabled."""
        self._acquires = 0
        self._thread = None

    def _count(self):
        """Return the process-local acquisition count."""
        return self._acquires

    def _is_mine(self):
        """Report recursive ownership in the current thread."""
        return self._thread == threading.get_ident() and self._acquires > 0

    def _get_value(self):
        """Read a snapshot of the current counter."""
        return request("value", self._handle)

    def _is_zero(self):
        """Report whether the current counter is empty."""
        return not self._get_value()

    def acquire(self, block=True, timeout=None):
        """Acquire one token until the caller's deadline."""
        if self._pid != os.getpid():
            raise RuntimeError("The semaphore adapter supports spawn")
        if self.kind == 0 and self._is_mine():
            self._acquires += 1
            return True
        if timeout is not None and not math.isfinite(timeout):
            raise ValueError("Invalid semaphore timeout")
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        while True:
            if request("acquire", self._handle):
                self._thread = threading.get_ident()
                self._acquires += 1
                return True
            interval = 0.01 if deadline is None else min(0.01, deadline - time.monotonic())
            if not block or interval <= 0:
                return False
            time.sleep(interval)

    def release(self):
        """Return one token or decrement a recursive acquisition."""
        if self._pid != os.getpid():
            raise RuntimeError("The semaphore adapter supports spawn")
        if self.kind == 0:
            if not self._is_mine():
                raise AssertionError("Recursive semaphore belongs to another thread")
            if self._acquires > 1:
                self._acquires -= 1
                return
        request("release", self._handle)
        self._acquires -= 1

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()

    def __del__(self):
        if self._handle is not None and self._pid == os.getpid():
            try:
                request("detach", self._handle)
            except FileNotFoundError:
                pass


def sysconf(key):
    """Report the bounded slot capacity to joblib."""
    if key in ("SC_SEM_NSEMS_MAX", os.sysconf_names.get("SC_SEM_NSEMS_MAX")):
        return SLOT_COUNT
    return SYSTEM_SYSCONF(key)


def enable():
    """Install the explicit test-client binding for spawned processes."""
    _multiprocessing.SemLock = SemLock
    _multiprocessing.sem_unlink = sem_unlink
    os.sysconf = sysconf
