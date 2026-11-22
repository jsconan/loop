"""Exercise the CPython spawn interface and enforce the attempt's slot limit."""

import _multiprocessing
import ctypes
import errno
import multiprocessing
import os

from synchronization_client import SLOT_COUNT


def child(event, queue):
    """Notify the parent through standard synchronization and queue locks."""
    queue.put("spawned")
    event.set()


if __name__ == "__main__":
    context = multiprocessing.get_context("spawn")
    event = context.Event()
    queue = context.Queue()
    worker = context.Process(target=child, args=(event, queue))
    worker.start()
    assert event.wait(5)
    assert queue.get(timeout=5) == "spawned"
    worker.join(5)
    assert worker.exitcode == 0
    lock = context.RLock()
    assert lock.acquire()
    assert lock.acquire(block=False)
    lock.release()
    lock.release()
    del event, queue, lock
    active = []
    names = []
    try:
        for i in range(SLOT_COUNT):
            name = f"/loop-cap-{i}"
            active.append(_multiprocessing.SemLock(1, 0, 1, name, False))
            names.append(name)
        try:
            _multiprocessing.SemLock(1, 0, 1, "/loop-over-cap", False)
        except OSError as error:
            assert error.errno == errno.ENOSPC
        else:
            raise AssertionError("Unbounded named semaphore creation")
    finally:
        for name in names:
            _multiprocessing.sem_unlink(name)
        active.clear()
    with _multiprocessing.SemLock(1, 1, 1, "/loop-after-cap", True):
        pass
    libsystem = ctypes.CDLL(None, use_errno=True)
    libsystem.sem_open.argtypes = (ctypes.c_char_p, ctypes.c_int)
    libsystem.sem_open.restype = ctypes.c_void_p
    name = f"/loop-raw-{os.getpid()}".encode()
    opened = libsystem.sem_open(name, os.O_CREAT | os.O_EXCL, 0o600, 1)
    if opened not in (None, ctypes.c_void_p(-1).value):
        libsystem.sem_unlink(name)
        raise AssertionError("Raw POSIX semaphore unexpectedly available")
    assert ctypes.get_errno() in (errno.EPERM, errno.EACCES)
    print(
        f"CPython spawn, queue, recursive lock, {SLOT_COUNT} slots, and raw semaphore denial passed."
    )
