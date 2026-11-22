"""Serve attempt-local named counters over a bounded Unix socket protocol."""

from __future__ import annotations

import errno
import json
import re
import secrets
import selectors
import socket
import threading
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

_MAX_COUNTERS = 512
_MAX_PEERS = 64
_MAX_FRAME_BYTES = 512
_MAX_REFERENCES_PER_COUNTER = 4096
_MAX_VALUE = 2**31 - 1
_NAME = re.compile(r"/?[A-Za-z0-9_-]{1,128}")
_HANDLE = re.compile(r"[a-f0-9]{32}")
_WAIT_SECONDS = 0.05
_SHUTDOWN_SECONDS = 2.0


@dataclass
class _Counter:
    """Track a counter and the lifetime of its name and references."""

    value: int
    maximum: int
    name: str | None
    references: int = 1


@dataclass
class _Peer:
    """Own one bounded request buffer and only this connection's references."""

    frame: bytearray = field(default_factory=bytearray)
    references: dict[str, int] = field(default_factory=dict)


class SynchronizationBroker:
    """Run one in-memory synchronization namespace for an approved local IPC attempt.

    Args:
        endpoint (Path): Socket path in a fresh, backend-owned, mutation-protected directory.
    """

    _endpoint: Path
    _resources: ExitStack
    _selector: selectors.BaseSelector
    _listener: socket.socket
    _stopping: threading.Event
    _thread: threading.Thread
    _peers: dict[socket.socket, _Peer]
    _names: dict[str, str]
    _counters: dict[str, _Counter]
    _failure: OSError | None

    def __init__(self, endpoint: Path) -> None:
        self._endpoint = endpoint
        self._resources = ExitStack()
        self._stopping = threading.Event()
        self._peers = {}
        self._names = {}
        self._counters = {}
        self._failure = None

    def __enter__(self) -> Self:
        """Bind the private endpoint and start its owned worker.

        Returns:
            Self: The broker serving the bound endpoint.

        Raises:
            OSError: The listener or selector cannot be created or configured.
            RuntimeError: The worker cannot be started.
        """
        with ExitStack() as resources:
            self._selector = resources.enter_context(selectors.DefaultSelector())
            self._listener = resources.enter_context(
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            )
            self._listener.bind(str(self._endpoint))
            resources.callback(self._endpoint.unlink, missing_ok=True)
            self._listener.listen(_MAX_PEERS)
            self._listener.setblocking(False)
            self._selector.register(self._listener, selectors.EVENT_READ)
            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._thread.start()
            self._resources = resources.pop_all()
        return self

    def __exit__(self, *_error: object) -> None:
        """Stop the worker and close the listener and endpoint.

        Args:
            *_error (object): Context exception details, if any.

        Raises:
            OSError: The worker cannot be confirmed stopped, the service failed, or endpoint
                cleanup fails.
        """
        self._stopping.set()
        self._thread.join(_SHUTDOWN_SECONDS)
        try:
            if self._thread.is_alive():
                raise OSError("Synchronization broker shutdown could not be confirmed.")
            if self._failure is not None:
                raise OSError("Synchronization broker service failed.") from self._failure
        finally:
            self._resources.close()

    def _serve(self) -> None:
        """Service bounded reads without a thread or wait queue per client."""
        try:
            while not self._stopping.is_set():
                for event, _ in self._selector.select(_WAIT_SECONDS):
                    if self._stopping.is_set():
                        return
                    if event.fileobj == self._listener:
                        self._accept()
                    else:
                        self._receive(event.fileobj)
        except OSError as error:
            self._failure = error
        finally:
            self._listener.close()
            for connection in self._peers:
                connection.close()
            self._peers.clear()
            self._names.clear()
            self._counters.clear()

    def _accept(self) -> None:
        """Set up one nonblocking connection without exceeding the peer limit."""
        try:
            connection, _ = self._listener.accept()
        except (BlockingIOError, ConnectionAbortedError):
            return
        with ExitStack() as resources:
            resources.callback(connection.close)
            if len(self._peers) >= _MAX_PEERS:
                return
            connection.setblocking(False)
            self._selector.register(connection, selectors.EVENT_READ)
            self._peers[connection] = _Peer()
            resources.pop_all()

    def _receive(self, connection: socket.socket) -> None:
        """Process at most one bounded record and retire broken connections."""
        peer = self._peers[connection]
        try:
            data = connection.recv(_MAX_FRAME_BYTES + 1 - len(peer.frame))
        except BlockingIOError:
            return
        except OSError:
            data = b""
        if data:
            peer.frame.extend(data)
            if len(peer.frame) <= _MAX_FRAME_BYTES and b"\n" not in peer.frame:
                return
            if (
                len(peer.frame) <= _MAX_FRAME_BYTES
                and peer.frame.find(b"\n") == len(peer.frame) - 1
            ):
                reply = self._respond(peer)
                try:
                    connection.sendall(json.dumps(reply, separators=(",", ":")).encode() + b"\n")
                except OSError:
                    pass
                else:
                    peer.frame.clear()
                    return
        self._selector.unregister(connection)
        del self._peers[connection]
        connection.close()
        for handle, count in peer.references.items():
            self._drop(handle, count)

    def _respond(self, peer: _Peer) -> tuple[int, object]:
        """Map invalid frames and failed operations to bounded protocol statuses."""
        try:
            message = json.loads(peer.frame.decode("ascii"))
            if (
                not isinstance(message, list)
                or len(message) < 2
                or not isinstance(message[0], str)
                or not isinstance(message[1], str)
            ):
                raise ValueError("Invalid synchronization message.")
            operation, identifier, *arguments = message
            if operation in {"create", "open", "unlink"}:
                return 0, self._named(operation, identifier, arguments, peer.references)
            return 0, self._access(operation, identifier, arguments, peer.references)
        except (ValueError, RecursionError):
            return errno.EINVAL, None
        except OSError as error:
            return error.errno or errno.EIO, None

    def _named(
        self,
        operation: str,
        name: str,
        arguments: list[object],
        references: dict[str, int],
    ) -> str | None:
        """Resolve only protocol names and enforce object and reference capacity."""
        if _NAME.fullmatch(name) is None:
            raise ValueError("Invalid synchronization name.")
        handle = self._names.get(name)
        if operation == "create":
            value, maximum = arguments
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not isinstance(maximum, int)
                or isinstance(maximum, bool)
                or not 0 <= value <= maximum <= _MAX_VALUE
            ):
                raise ValueError("Invalid synchronization counter bounds.")
            if handle is not None:
                raise FileExistsError(errno.EEXIST, "Synchronization name already exists.")
            if len(self._counters) >= _MAX_COUNTERS:
                raise OSError(errno.ENOSPC, "Synchronization object capacity reached.")
            handle = secrets.token_hex(16)
            if handle in self._counters:
                raise OSError(errno.EAGAIN, "Synchronization identifier already exists.")
            self._names[name] = handle
            self._counters[handle] = _Counter(value, maximum, name)
            references[handle] = 1
            return handle
        if arguments:
            raise ValueError("Unexpected synchronization name arguments.")
        if handle is None:
            raise FileNotFoundError(errno.ENOENT, "Unknown synchronization name.")
        counter = self._counters[handle]
        if operation == "open":
            count = references.get(handle, 0)
            if count >= _MAX_REFERENCES_PER_COUNTER:
                raise OSError(errno.ENOSPC, "Synchronization reference capacity reached.")
            references[handle] = count + 1
            counter.references += 1
            return handle
        del self._names[name]
        counter.name = None
        self._drop(handle, 0)
        return None

    def _access(
        self,
        operation: str,
        handle: str,
        arguments: list[object],
        references: dict[str, int],
    ) -> int | bool:
        """Apply only counter operations owned by the peer's connection."""
        if arguments or _HANDLE.fullmatch(handle) is None:
            raise ValueError("Invalid synchronization handle arguments.")
        count = references.get(handle, 0)
        if count == 0:
            raise FileNotFoundError(errno.ENOENT, "Unknown synchronization handle.")
        counter = self._counters[handle]
        if operation == "detach":
            if count == 1:
                del references[handle]
            else:
                references[handle] = count - 1
            self._drop(handle, 1)
        elif operation == "acquire":
            if counter.value == 0:
                return False
            counter.value -= 1
        elif operation == "release":
            if counter.value == counter.maximum:
                raise ValueError("Synchronization counter limit reached.")
            counter.value += 1
        elif operation == "value":
            return counter.value
        else:
            raise ValueError("Unknown synchronization operation.")
        return True

    def _drop(self, handle: str, count: int) -> None:
        """Reclaim an unlinked object after its last connection releases it."""
        counter = self._counters[handle]
        counter.references -= count
        if counter.references == 0 and counter.name is None:
            del self._counters[handle]
