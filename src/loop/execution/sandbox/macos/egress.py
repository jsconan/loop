"""Mediate one macOS sandbox attempt through an exact Unix SOCKS endpoint."""

from __future__ import annotations

import ipaddress
import os
import select
import socket
import stat
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING, Self
from urllib.parse import urlsplit

from ..contracts import SandboxRequest, normalize_egress_origin

if TYPE_CHECKING:
    from ....permissions.manager import PermissionManager

AUTHORITY_SECONDS = 2.0
POLL_SECONDS = 0.1
MAX_CLIENTS = 64
MAX_PENDING_RESOLUTIONS = 64
_RESOLUTION_SLOTS = threading.BoundedSemaphore(MAX_PENDING_RESOLUTIONS)
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_ROUTE_BYTES = 4096
ROUTE_TOOL = "/sbin/route"
_FORWARDED_METHODS = frozenset({"GET", "HEAD", "POST"})
_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def _read_exact(connection: socket.socket, size: int) -> bytes:
    """Read a bounded SOCKS frame without accepting a truncated request."""
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(size - len(result))
        if not chunk:
            raise OSError("Incomplete SOCKS request.")
        result.extend(chunk)
    return bytes(result)


def _external_route(address: ipaddress.IPv4Address) -> bool:
    """Reject an externally numbered address routed to a local host interface."""
    try:
        for path in ("/sbin", ROUTE_TOOL):
            details = os.lstat(path)
            if details.st_uid != 0 or details.st_mode & 0o022 or stat.S_ISLNK(details.st_mode):
                return False
        result = subprocess.run(
            [ROUTE_TOOL, "-n", "get", "-inet", str(address)],
            capture_output=True,
            check=True,
            timeout=1.0,
        )
        if len(result.stdout) > MAX_ROUTE_BYTES:
            return False
        fields: dict[str, str] = {}
        for line in result.stdout.decode("ascii").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key.strip() in fields:
                    return False
                fields[key.strip()] = value.strip()
        flags = fields.get("flags", "").strip("<>").split(",")
        interface = fields.get("interface", "")
        return (
            fields.get("route to") == str(address)
            and bool(interface)
            and not interface.startswith("lo")
            and "UP" in flags
            and not {"LOCAL", "REJECT", "BLACKHOLE"}.intersection(flags)
        )
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError):
        return False


class MacOSEgressBroker:
    """Own a private Unix SOCKS listener and checked upstreams for one approved attempt.

    The broker lives in the application process. If that process exits, it loses every broker
    descriptor, while a runnable but stalled policy authority loses its two-second lease.

    Args:
        request (SandboxRequest): Exact approved origins and command deadline.
        path (Path): Socket path in an isolated, sandbox-nonwritable volume directory.
        authority (PermissionManager): Manager that approved the request.
        revision (int): Active policy revision captured with approval.
    """

    request: SandboxRequest
    path: Path
    authority: PermissionManager
    revision: int
    _listener: socket.socket | None
    _lock: threading.Lock
    _sockets: set[socket.socket]
    _workers: set[threading.Thread]
    _acceptor: threading.Thread | None
    _watcher: threading.Thread | None
    _active: bool
    _closing: bool
    _paused: bool
    _pause_deadline: float | None
    _bound: bool
    _last_heartbeat: float
    _deadline: float
    _closed: threading.Event
    _confirmed: bool
    _revoked_revision: int | None
    _revocation_confirmed: bool
    _interrupted: bool

    def __init__(
        self,
        request: SandboxRequest,
        path: Path,
        authority: PermissionManager,
        revision: int,
    ) -> None:
        if not request.egress_origins or request.network or revision < 0:
            raise ValueError("The macOS broker requires one mediated-only approved grant.")
        self.request = request
        self.path = path
        self.authority = authority
        self.revision = revision
        self._listener = None
        self._lock = threading.Lock()
        self._sockets = set()
        self._workers = set()
        self._acceptor = None
        self._watcher = None
        self._active = False
        self._closing = False
        self._paused = False
        self._pause_deadline = None
        self._bound = False
        self._last_heartbeat = 0.0
        self._deadline = request.deadline
        self._closed = threading.Event()
        self._confirmed = False
        self._revoked_revision = None
        self._revocation_confirmed = False
        self._interrupted = False

    def __enter__(self) -> Self:
        parent = self.path.parent.lstat()
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.getuid()
            or stat.S_IMODE(parent.st_mode) != 0o700
            or self.path.exists()
            or self.path.is_symlink()
        ):
            raise OSError("Unsafe macOS mediator socket directory.")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._listener = listener
            listener.bind(str(self.path))
            listener.listen(16)
            listener.setblocking(False)
            with self._lock:
                self._active = True
                self._last_heartbeat = time.monotonic()
            self.authority.bind_egress(self.revision, self)
            self._bound = True
            with self._lock:
                if not self._active:
                    raise OSError("Mediated egress authority ended during activation.")
            self._acceptor = threading.Thread(target=self._accept, daemon=True)
            self._watcher = threading.Thread(target=self._watch, daemon=True)
            self._acceptor.start()
            self._watcher.start()
            return self
        except BaseException:
            self.close()
            if self._bound:
                self.authority.unbind_egress(self)
                self._bound = False
            raise

    def __exit__(self, *_error: object) -> None:
        confirmed = self.close()
        if self._bound:
            self.authority.unbind_egress(self)
            self._bound = False
        if not confirmed:
            raise OSError("MacOS mediated egress closure could not be confirmed.")

    @property
    def interrupted(self) -> bool:
        """Report whether policy or mediator health interrupted this attempt.

        Returns:
            bool: True after authority loss or an active policy revision change.
        """
        with self._lock:
            return self._interrupted

    def pause(self) -> bool:
        """Block new connections before policy mutation.

        Returns:
            bool: Whether the admission barrier was installed.
        """
        with self._lock:
            if not self._active:
                return self._confirmed
            if not self._paused:
                self._paused = True
                self._pause_deadline = time.monotonic() + AUTHORITY_SECONDS
            return True

    def resume(self) -> bool:
        """Resume admission only while the original revision remains active.

        Returns:
            bool: Whether the unchanged attempt is again accepting connections.
        """
        with self._lock:
            if not self._active or not self._paused:
                return False
            if self._lease_current():
                self._paused = False
                self._pause_deadline = None
                return True
            self._interrupted = True
        self.close()
        return False

    def bind_runtime_deadline(self, deadline: float) -> bool:
        """Replace a live preparation ceiling with the launched command ceiling.

        Args:
            deadline (float): Bounded command deadline established at launch.

        Returns:
            bool: Whether the original lease remained live while binding.
        """
        with self._lock:
            if not self._active or not self._lease_current() or deadline <= time.monotonic():
                self._interrupted = True
                return False
            self._deadline = deadline
            return True

    def revoke_revision(self, revision: int) -> bool:
        """Close all covered transports and acknowledge one newer revision.

        Args:
            revision (int): Monotonic policy revision published by the manager.

        Returns:
            bool: Whether exact-revision closure completed within two seconds.
        """
        with self._lock:
            if self._revoked_revision is not None:
                return revision == self._revoked_revision and self._revocation_confirmed
            if revision <= self.revision:
                return False
            self._revoked_revision = revision
            self._interrupted = True
        deadline = time.monotonic() + AUTHORITY_SECONDS
        confirmed = self.close() and time.monotonic() <= deadline
        with self._lock:
            self._revocation_confirmed = confirmed
        return confirmed

    def allow_new(self) -> bool:
        """Check the active admission gate, revision lease, and command deadline.

        Returns:
            bool: Whether a fresh client and upstream may be admitted.
        """
        with self._lock:
            return self._active and not self._paused and self._lease_current()

    def active(self) -> bool:
        """Check whether existing transports retain their bounded authority.

        Returns:
            bool: Whether covered forwarding may continue.
        """
        with self._lock:
            return self._active and self._lease_current()

    def _lease_current(self) -> bool:
        """Evaluate a lease while the broker lock is held."""
        return (
            time.monotonic() < self._deadline
            and time.monotonic() - self._last_heartbeat < AUTHORITY_SECONDS
            and (self._pause_deadline is None or time.monotonic() < self._pause_deadline)
        )

    def close(self) -> bool:
        """Stop admission, close all transports, and settle bounded worker cleanup.

        Returns:
            bool: Whether listener and all covered sockets and workers closed.
        """
        with self._lock:
            if self._closing:
                waiting = True
                listener = None
                sockets = ()
                workers = ()
            else:
                waiting = False
                self._closing = True
                self._active = False
                listener = self._listener
                sockets = tuple(self._sockets)
                workers = tuple(self._workers)
        if waiting:
            return self._closed.wait(AUTHORITY_SECONDS) and self._confirmed
        if listener is not None:
            listener.close()
        for connection in sockets:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        deadline = time.monotonic() + AUTHORITY_SECONDS
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join(max(0.0, deadline - time.monotonic()))
        confirmed = all(
            worker is threading.current_thread() or not worker.is_alive() for worker in workers
        )
        with self._lock:
            self._confirmed = confirmed
            self._closed.set()
        return confirmed

    def _watch(self) -> None:
        """Renew the exact active revision or independently expire every transport."""
        try:
            while True:
                with self._lock:
                    if not self._active:
                        return
                try:
                    current = self.authority.try_egress_policy_revision()
                except OSError:
                    current = -1
                if current is not None:
                    with self._lock:
                        if current != self.revision or not self._lease_current():
                            self._interrupted = True
                            break
                        self._last_heartbeat = time.monotonic()
                if not self.active():
                    with self._lock:
                        self._interrupted = True
                    break
                time.sleep(POLL_SECONDS)
        except BaseException:
            with self._lock:
                self._interrupted = True
            raise
        finally:
            self.close()

    def _accept(self) -> None:
        """Accept only bounded local clients while the attempt lease remains current."""
        listener = self._listener
        if listener is None:
            return
        while self.active():
            worker = None
            client = None
            try:
                if not select.select([listener], [], [], POLL_SECONDS)[0]:
                    continue
                client, _ = listener.accept()
                if not self._track(client):
                    client.close()
                    continue
                with self._lock:
                    if (
                        not self._active
                        or self._paused
                        or not self._lease_current()
                        or len(self._workers) >= MAX_CLIENTS
                    ):
                        self._sockets.discard(client)
                        client.close()
                        continue
                    worker = threading.Thread(target=self._serve, args=(client,), daemon=True)
                    self._workers.add(worker)
                    worker.start()
            except (OSError, ValueError, RuntimeError):
                with self._lock:
                    if worker is not None:
                        self._workers.discard(worker)
                if client is not None:
                    self._release(client)
                break
        with self._lock:
            if self._active:
                self._interrupted = True
        self.close()

    def _track(self, connection: socket.socket) -> bool:
        """Retain one owned descriptor under the active admission barrier."""
        with self._lock:
            if not self._active or self._paused or len(self._sockets) >= MAX_CLIENTS * 2:
                return False
            self._sockets.add(connection)
            return True

    def _release(self, connection: socket.socket) -> None:
        """Remove and close one owned connection."""
        with self._lock:
            self._sockets.discard(connection)
        connection.close()

    def _origin(self, address_type: int, connection: socket.socket) -> str | None:
        """Match one SOCKS destination to exactly one approved HTTP(S) origin."""
        if address_type == 3:
            length = _read_exact(connection, 1)[0]
            if not 1 <= length <= 253:
                return None
            host = _read_exact(connection, length).decode("ascii").lower()
        elif address_type == 1:
            host = str(ipaddress.IPv4Address(_read_exact(connection, 4)))
        else:
            return None
        port = int.from_bytes(_read_exact(connection, 2), "big")
        matches = []
        for scheme in ("http", "https"):
            candidate = f"{scheme}://{host}"
            if port != (80 if scheme == "http" else 443):
                candidate += f":{port}"
            try:
                candidate = normalize_egress_origin(candidate)
            except ValueError:
                continue
            if candidate in self.request.egress_origins:
                matches.append(candidate)
        return matches[0] if len(matches) == 1 else None

    def _connect(self, origin: str) -> socket.socket | None:
        """Resolve each candidate and connect only to a pinned public IPv4 route."""
        if not self.allow_new():
            return None
        target = urlsplit(origin)
        if not _RESOLUTION_SLOTS.acquire(blocking=False):
            return None
        try:
            answers = socket.getaddrinfo(
                target.hostname,
                target.port or (443 if target.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except OSError:
            return None
        finally:
            _RESOLUTION_SLOTS.release()
        for family, kind, protocol, _, address in answers:
            if family != socket.AF_INET:
                continue
            try:
                numeric = ipaddress.IPv4Address(address[0])
            except ipaddress.AddressValueError:
                continue
            if not numeric.is_global or not _external_route(numeric) or not self.allow_new():
                continue
            try:
                upstream = socket.socket(family, kind, protocol)
            except OSError:
                continue
            if not self._track(upstream):
                upstream.close()
                return None
            try:
                upstream.settimeout(min(1.0, max(0.1, self._deadline - time.monotonic())))
                upstream.connect(address)
                upstream.settimeout(None)
                if not self.allow_new():
                    self._release(upstream)
                    return None
                return upstream
            except OSError:
                self._release(upstream)
        return None

    def _serve(self, client: socket.socket) -> None:
        """Handle one bounded SOCKS request and relay only approved HTTP(S) bytes."""
        upstream = None
        try:
            client.settimeout(AUTHORITY_SECONDS)
            version, count = _read_exact(client, 2)
            if version != 5 or not 1 <= count <= 16 or 0 not in _read_exact(client, count):
                client.sendall(b"\x05\xff")
                return
            client.sendall(b"\x05\x00")
            version, command, reserved, address_type = _read_exact(client, 4)
            if version != 5 or command != 1 or reserved != 0:
                return
            origin = self._origin(address_type, client)
            if origin is None or (upstream := self._connect(origin)) is None:
                client.sendall(b"\x05\x02\x00\x01\x00\x00\x00\x00\x00\x00")
                return
            client.sendall(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")
            if urlsplit(origin).scheme == "http":
                SocksHttpHandler(client, ("localhost", 0), self, origin, upstream)
            else:
                client.settimeout(None)
                self._relay(client, upstream)
        except (OSError, UnicodeError, ValueError):
            pass
        finally:
            if upstream is not None:
                self._release(upstream)
            self._release(client)
            with self._lock:
                self._workers.discard(threading.current_thread())

    def _relay(self, client: socket.socket, upstream: socket.socket) -> None:
        """Forward an opaque HTTPS tunnel only while its lease remains active."""
        while self.active():
            try:
                ready, _, _ = select.select([client, upstream], [], [], POLL_SECONDS)
                for source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    (upstream if source is client else client).sendall(data)
            except OSError:
                return


class SocksHttpHandler(BaseHTTPRequestHandler):
    """Parse one plaintext HTTP request on an already checked SOCKS connection.

    Args:
        request (socket.socket): Connected sandbox client.
        client_address (tuple[str, int]): Non-authoritative local display address.
        server (MacOSEgressBroker): Attempt broker that owns the pinned upstream.
        origin (str): Exact approved HTTP origin.
        upstream (socket.socket): Already checked numeric transport.
    """

    protocol_version = "HTTP/1.1"
    server: MacOSEgressBroker
    origin: str
    upstream: socket.socket

    def __init__(
        self,
        request: socket.socket,
        client_address: tuple[str, int],
        server: MacOSEgressBroker,
        origin: str,
        upstream: socket.socket,
    ) -> None:
        self.origin = origin
        self.upstream = upstream
        super().__init__(request, client_address, server)

    def log_message(self, format: str, *args: object) -> None:
        """Discard request paths and destinations from routine broker output.

        Args:
            format (str): Standard-library log template, intentionally ignored.
            *args (object): Standard-library arguments, intentionally ignored.
        """
        del format, args

    def do_GET(self) -> None:
        """Forward one checked plaintext GET request."""
        self._forward()

    def do_HEAD(self) -> None:
        """Forward one checked plaintext HEAD request."""
        self._forward()

    def do_POST(self) -> None:
        """Forward one checked plaintext POST request with a bounded body."""
        self._forward()

    def _forward(self) -> None:
        """Validate visible authority and headers before forwarding one request."""
        self.close_connection = True
        target = urlsplit(self.origin)
        lengths = self.headers.get_all("Content-Length", [])
        try:
            if (
                not self.path.startswith("/")
                or self.path.startswith("//")
                or "#" in self.path
                or self.command not in _FORWARDED_METHODS
                or len(self.headers.get_all("Host", [])) != 1
                or self.headers["Host"].casefold() != target.netloc.casefold()
                or len(lengths) > 1
                or self.headers.get("Transfer-Encoding") is not None
                or self.headers.get("Upgrade") is not None
                or self.headers.get("Expect") is not None
                or any("\r" in value or "\n" in value for value in self.headers.values())
                or lengths
                and (not lengths[0].isascii() or not lengths[0].isdecimal())
            ):
                raise ValueError("Unsupported or ambiguous plaintext request.")
            length = int(lengths[0]) if lengths else 0
            if length > MAX_BODY_BYTES or self.command == "POST" and not lengths:
                raise ValueError("Unsupported HTTP request body length.")
        except (TypeError, ValueError):
            self.send_error(403)
            return
        try:
            self.upstream.sendall(f"{self.command} {self.path} HTTP/1.1\r\n".encode("ascii"))
            connection_tokens = {
                token.strip().lower() for token in self.headers.get("Connection", "").split(",")
            }
            for name, value in self.headers.items():
                if name.lower() not in _HOP_HEADERS | connection_tokens | {
                    "host",
                    "content-length",
                }:
                    self.upstream.sendall(f"{name}: {value}\r\n".encode("latin-1"))
            self.upstream.sendall(f"Host: {target.netloc}\r\n".encode("ascii"))
            if lengths:
                self.upstream.sendall(f"Content-Length: {length}\r\n".encode("ascii"))
            self.upstream.sendall(b"Connection: close\r\n\r\n")
            remaining = length
            while remaining and self.server.active():
                data = self.rfile.read(min(65536, remaining))
                if not data:
                    return
                self.upstream.sendall(data)
                remaining -= len(data)
            if remaining:
                return
            while self.server.active():
                if select.select([self.upstream], [], [], POLL_SECONDS)[0]:
                    data = self.upstream.recv(65536)
                    if not data:
                        return
                    self.connection.sendall(data)
        except (OSError, ValueError, UnicodeError):
            return
