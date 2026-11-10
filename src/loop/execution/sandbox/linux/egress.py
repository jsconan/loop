"""Broker attempt-private HTTP proxy connections in the trusted Linux receiver."""

from __future__ import annotations

import array
import ctypes
import ipaddress
import json
import os
import select
import socket
import stat
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit

from ..contracts import SandboxRequest, normalize_egress_origin

MAX_CLIENTS = 64
MAX_BODY_BYTES = 16 * 1024 * 1024
AUTHORITY_SECONDS = 2.0
POLL_SECONDS = 0.1
ROUTE_TOOL = "/usr/bin/ip"
MAX_ROUTE_BYTES = 4096
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


def start_namespace_proxy(descriptor: int) -> int:
    """Start an attempt-private loopback FD handoff before replacing payload setup.

    Args:
        descriptor (int): One inherited connected Unix sequenced-packet socket endpoint.

    Returns:
        int: TCP port used only inside this attempt's private network namespace.

    Raises:
        OSError: The listener, fork, or inherited handoff cannot be established.
    """
    channel = socket.socket(fileno=descriptor)
    try:
        channel.settimeout(AUTHORITY_SECONDS)
        if channel.recv(1) != b"G":
            raise OSError("Mediated egress authority did not activate.")
        # Seal the trusted process before fork. The proxy child inherits nondumpability,
        # while execve may reset it only for the future untrusted shell in the parent.
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
        libc.prctl.restype = ctypes.c_int
        if libc.prctl(4, 0, 0, 0, 0) != 0 or libc.prctl(3, 0, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Mediated egress proxy identity could not be sealed.")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(16)
            port = listener.getsockname()[1]
            child = os.fork()
            if child == 0:
                try:
                    with open(os.devnull, "wb", buffering=0) as sink:
                        os.dup2(sink.fileno(), 1)
                        os.dup2(sink.fileno(), 2)
                    while True:
                        client, _ = listener.accept()
                        with client:
                            payload = array.array("i", [client.fileno()])
                            channel.sendmsg(
                                [b"C"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, payload)]
                            )
                except (OSError, ValueError):
                    pass
                finally:
                    os._exit(0)
            return port
        finally:
            listener.close()
    finally:
        channel.close()


class EgressBroker:
    """Own checked host transports for one isolated sandbox attempt.

    Args:
        request (SandboxRequest): Original approved origins and command deadline.
        channel (socket.socket): Inherited private Unix handoff endpoint from this attempt.
        The broker always excludes private addresses; private networking uses another grant.
    """

    request: SandboxRequest
    channel: socket.socket
    _lock: threading.Lock
    _sockets: set[socket.socket]
    _workers: set[threading.Thread]
    _reader: threading.Thread
    _active: bool
    _last_heartbeat: float
    _revision: int | None
    _lease_started: bool
    _gate_open: bool
    _paused: bool
    _closed: threading.Event
    _uncertain: bool

    def __init__(self, request: SandboxRequest, channel: socket.socket) -> None:
        if not request.egress_origins or request.network:
            raise ValueError("The broker requires one mediated-only attempt grant.")
        self.request = request
        self.channel = channel
        self._lock = threading.Lock()
        self._sockets = set()
        self._workers = set()
        self._active = True
        self._last_heartbeat = 0.0
        self._revision = None
        self._lease_started = False
        self._gate_open = False
        self._paused = False
        self._closed = threading.Event()
        self._uncertain = False
        self._reader = threading.Thread(target=self._receive, daemon=True)
        self._reader.start()

    def heartbeat(self, revision: int) -> bool:
        """Renew only the initially bound revision over the authenticated control socket.

        Args:
            revision (int): Monotonic approved policy revision from the application authority.

        Returns:
            bool: Whether this heartbeat matches the attempt's active revision.
        """
        with self._lock:
            if not self._active or revision < 0:
                return False
            if self._revision is None:
                self._revision = revision
            if revision != self._revision:
                return False
            self._last_heartbeat = time.monotonic()
            if not self._gate_open:
                try:
                    if self.channel.send(b"G") != 1:
                        return False
                except OSError:
                    return False
                self._gate_open = True
            return True

    def start_lease(self) -> None:
        """Start the two-second first-heartbeat deadline only after native spawn."""
        with self._lock:
            self._lease_started = True
            self._last_heartbeat = time.monotonic()

    def lease_expired(self) -> bool:
        """Report a missed independent authority deadline.

        Returns:
            bool: Whether two seconds passed without a valid heartbeat after launch.
        """
        with self._lock:
            return (
                self._lease_started and time.monotonic() - self._last_heartbeat >= AUTHORITY_SECONDS
            )

    def accept_revision(self, revision: int) -> bool:
        """Reject stale revision frames and close every covered route before acknowledgement.

        Args:
            revision (int): Exact new policy revision delivered by the active authority.

        Returns:
            bool: Whether the revision advanced and closure was confirmed.
        """
        with self._lock:
            current = self._revision
        if current is None or revision <= current:
            self.close()
            return False
        return self.close()

    def pause(self, revision: int) -> bool:
        """Block new connections while the approving policy is being updated.

        Args:
            revision (int): Exact currently approved revision.

        Returns:
            bool: Whether the admission barrier applies to this active attempt.
        """
        with self._lock:
            if not self._active or self._revision != revision:
                return False
            self._paused = True
            return True

    def resume(self, revision: int) -> bool:
        """Reopen admission only when the same active policy revision remains approved.

        Args:
            revision (int): Exact unchanged policy revision.

        Returns:
            bool: Whether new connections may resume.
        """
        with self._lock:
            if not self._active or self._revision != revision or not self._paused:
                return False
            self._paused = False
            return True

    def allow_new(self) -> bool:
        """Report whether this lease permits a new destination connection.

        Returns:
            bool: Active, unpaused admission before the command deadline.
        """
        with self._lock:
            return (
                self._active
                and self._gate_open
                and not self._paused
                and self._revision is not None
                and time.monotonic() < self.request.deadline
                and time.monotonic() - self._last_heartbeat < AUTHORITY_SECONDS
            )

    def active(self) -> bool:
        """Check the bound deadline and independent two-second authority lease.

        Returns:
            bool: Whether new or continuing transports still have authority.
        """
        with self._lock:
            return (
                self._active
                and self._gate_open
                and self._revision is not None
                and time.monotonic() < self.request.deadline
                and time.monotonic() - self._last_heartbeat < AUTHORITY_SECONDS
            )

    def close(self) -> bool:
        """Stop new handoffs and close every tracked client and upstream socket.

        Returns:
            bool: Whether all covered sockets and worker threads closed within two seconds.
        """
        with self._lock:
            if not self._active:
                already_closing = True
                sockets = ()
                workers = ()
            else:
                already_closing = False
                self._active = False
                sockets = tuple(self._sockets)
                workers = tuple(self._workers)
        if already_closing:
            return self._closed.wait(AUTHORITY_SECONDS) and not self._uncertain
        for connection in sockets:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        try:
            self.channel.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.channel.close()
        deadline = time.monotonic() + AUTHORITY_SECONDS
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join(max(0.0, deadline - time.monotonic()))
        with self._lock:
            self._uncertain = self._uncertain or any(worker.is_alive() for worker in workers)
            result = not self._uncertain
        self._closed.set()
        return result

    def connect(self, origin: str) -> socket.socket | None:
        """Resolve once, filter every address, and connect only to a selected numeric peer.

        Args:
            origin (str): Canonical approved destination authority.

        Returns:
            socket.socket | None: Tracked pinned upstream socket, or None on denial/failure.
        """
        if origin not in self.request.egress_origins or not self.allow_new():
            return None
        target = urlsplit(origin)
        if target.hostname is None:
            return None
        port = target.port or (443 if target.scheme == "https" else 80)
        try:
            answers = socket.getaddrinfo(target.hostname, port, type=socket.SOCK_STREAM)
        except OSError:
            return None
        for family, kind, protocol, _, address in answers:
            # IPv6 transition mechanisms can carry private IPv4 destinations through a
            # globally routed address. Qualify IPv4 mediation first; fail closed for AAAA.
            if family != socket.AF_INET:
                continue
            try:
                numeric = ipaddress.ip_address(address[0])
            except ValueError:
                continue
            if not numeric.is_global or not self._external_route(numeric):
                continue
            if not self.allow_new():
                return None
            try:
                upstream = socket.socket(family, kind, protocol)
            except OSError:
                continue
            if not self._track(upstream):
                upstream.close()
                return None
            try:
                upstream.settimeout(min(1.0, max(0.1, self.request.deadline - time.monotonic())))
                upstream.connect(address)
                upstream.settimeout(None)
                if not self.allow_new():
                    self.release(upstream)
                    return None
                return upstream
            except OSError:
                self.release(upstream)
        return None

    @staticmethod
    def _external_route(address: ipaddress.IPv4Address) -> bool:
        """Reject a globally addressed host-local route before transport creation."""
        try:
            for path in ("/usr", "/usr/bin", ROUTE_TOOL):
                details = os.lstat(path)
                if details.st_uid != 0 or details.st_mode & 0o022 or stat.S_ISLNK(details.st_mode):
                    return False
            result = subprocess.run(
                [
                    ROUTE_TOOL,
                    "-j",
                    "route",
                    "get",
                    str(address),
                ],
                capture_output=True,
                check=True,
                timeout=1.0,
            )
            if len(result.stdout) > MAX_ROUTE_BYTES:
                return False
            routes = json.loads(result.stdout)
            return (
                isinstance(routes, list)
                and len(routes) == 1
                and isinstance(routes[0], dict)
                and routes[0].get("type", "unicast") == "unicast"
                and routes[0].get("dev") != "lo"
                and isinstance(routes[0].get("dev"), str)
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            return False

    def release(self, connection: socket.socket) -> None:
        """Remove and close one owned transport after forwarding settles.

        Args:
            connection (socket.socket): Tracked client or upstream socket.
        """
        with self._lock:
            self._sockets.discard(connection)
        connection.close()

    def _track(self, connection: socket.socket) -> bool:
        """Admit a socket only while the bounded attempt remains active."""
        with self._lock:
            if not self._active or self._paused or len(self._sockets) >= MAX_CLIENTS * 2:
                return False
            self._sockets.add(connection)
            return True

    def _receive(self) -> None:
        """Receive only connected TCP descriptors from the attempt-private bridge."""
        while True:
            with self._lock:
                if not self._active:
                    break
            try:
                ready, _, _ = select.select([self.channel], [], [], POLL_SECONDS)
                if not ready:
                    continue
                data, controls, flags, _ = self.channel.recvmsg(
                    1, socket.CMSG_SPACE(array.array("i", [0]).itemsize), socket.MSG_CMSG_CLOEXEC
                )
            except (OSError, ValueError):
                break
            descriptors = array.array("i")
            for level, kind, payload in controls:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    descriptors.frombytes(
                        payload[: len(payload) - len(payload) % descriptors.itemsize]
                    )
            if (
                data != b"C"
                or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
                or len(descriptors) != 1
            ):
                for descriptor in descriptors:
                    os.close(descriptor)
                break
            try:
                client = socket.socket(fileno=descriptors[0])
            except OSError:
                try:
                    os.close(descriptors[0])
                except OSError:
                    pass
                break
            worker = None
            try:
                if (
                    not self.allow_new()
                    or client.family != socket.AF_INET
                    or client.type != socket.SOCK_STREAM
                    or client.getsockname()[0] != "127.0.0.1"
                    or client.getpeername()[0] != "127.0.0.1"
                    or not self._track(client)
                ):
                    client.close()
                    continue
                with self._lock:
                    if len(self._workers) >= MAX_CLIENTS:
                        self._sockets.discard(client)
                        client.close()
                        continue
                    worker = threading.Thread(target=self._serve, args=(client,), daemon=True)
                    self._workers.add(worker)
                worker.start()
            except (OSError, RuntimeError):
                with self._lock:
                    if worker is not None:
                        self._workers.discard(worker)
                self.release(client)
                break
        self.close()

    def _serve(self, client: socket.socket) -> None:
        """Run one bounded proxy parser and release its client even on failure."""
        try:
            client.settimeout(2.0)
            ProxyHandler(client, client.getpeername(), self)
        except (OSError, ValueError):
            pass
        finally:
            self.release(client)
            with self._lock:
                self._workers.discard(threading.current_thread())


class ProxyHandler(BaseHTTPRequestHandler):
    """Forward only visible HTTP requests or an explicitly approved opaque HTTPS tunnel."""

    protocol_version = "HTTP/1.1"
    server: EgressBroker

    def log_message(self, format: str, *args: object) -> None:
        """Discard request paths and destination names from routine receiver logs.

        Args:
            format (str): Standard-library log template, intentionally ignored.
            *args (object): Standard-library log arguments, intentionally ignored.
        """
        del format, args

    def do_CONNECT(self) -> None:
        """Connect one approved HTTPS origin and relay its opaque bytes."""
        try:
            target = urlsplit("https://" + self.path)
            if (
                target.port is None
                or target.path
                or target.query
                or target.fragment
                or target.username is not None
                or target.password is not None
                or len(self.headers.get_all("Host", [])) != 1
                or self.headers["Host"].casefold() != self.path.casefold()
                or self.headers.get("Content-Length") is not None
                or self.headers.get("Transfer-Encoding") is not None
                or self.headers.get("Upgrade") is not None
                or self.headers.get("Expect") is not None
            ):
                raise ValueError("CONNECT requires an explicit TCP port.")
            authority = self.path
            if target.port == 443:
                authority = self.path.rsplit(":", 1)[0]
            origin = normalize_egress_origin("https://" + authority)
        except ValueError:
            self.send_error(403)
            return
        upstream = self.server.connect(origin)
        if upstream is None:
            self.send_error(403)
            return
        try:
            self.send_response(200, "Connection Established")
            self.end_headers()
            self._relay(upstream)
        finally:
            self.server.release(upstream)

    def do_GET(self) -> None:
        """Forward an approved plaintext HTTP GET."""
        self._forward_http()

    def do_HEAD(self) -> None:
        """Forward an approved plaintext HTTP HEAD."""
        self._forward_http()

    def do_POST(self) -> None:
        """Forward an approved plaintext HTTP POST with a bounded body."""
        self._forward_http()

    def _forward_http(self) -> None:
        """Validate every visible request before opening an upstream connection."""
        try:
            target = urlsplit(self.path)
            authority = target.netloc
            if target.port == 80:
                authority = target.netloc.rsplit(":", 1)[0]
            origin = normalize_egress_origin(f"{target.scheme}://{authority}")
            lengths = self.headers.get_all("Content-Length", [])
            if (
                target.scheme != "http"
                or target.fragment
                or self.command not in _FORWARDED_METHODS
                or len(self.headers.get_all("Host", [])) != 1
                or self.headers["Host"].casefold() != target.netloc.casefold()
                or len(lengths) > 1
                or self.headers.get("Transfer-Encoding") is not None
                or self.headers.get("Upgrade") is not None
                or self.headers.get("Expect") is not None
                or any("\r" in value or "\n" in value for value in self.headers.values())
            ):
                raise ValueError("Unsupported or ambiguous HTTP request.")
            if lengths and not lengths[0].isascii() or lengths and not lengths[0].isdecimal():
                raise ValueError("Unsupported HTTP request body length.")
            length = int(lengths[0]) if lengths else 0
            if length > MAX_BODY_BYTES or (self.command == "POST" and not lengths):
                raise ValueError("Unsupported HTTP request body length.")
        except (ValueError, TypeError):
            self.send_error(403)
            return
        upstream = self.server.connect(origin)
        if upstream is None:
            self.send_error(403)
            return
        try:
            path = target.path or "/"
            if target.query:
                path += "?" + target.query
            upstream.sendall(f"{self.command} {path} HTTP/1.1\r\n".encode("ascii"))
            connection_tokens = {
                token.strip().lower() for token in self.headers.get("Connection", "").split(",")
            }
            for name, value in self.headers.items():
                if name.lower() not in _HOP_HEADERS | connection_tokens | {
                    "host",
                    "content-length",
                }:
                    upstream.sendall(f"{name}: {value}\r\n".encode("latin-1"))
            upstream.sendall(f"Host: {target.netloc}\r\n".encode("ascii"))
            if lengths:
                upstream.sendall(f"Content-Length: {length}\r\n".encode("ascii"))
            upstream.sendall(b"Connection: close\r\n\r\n")
            remaining = length
            while remaining and self.server.active():
                data = self.rfile.read(min(65536, remaining))
                if not data:
                    return
                upstream.sendall(data)
                remaining -= len(data)
            if remaining:
                return
            while self.server.active():
                ready, _, _ = select.select([upstream], [], [], POLL_SECONDS)
                if ready:
                    data = upstream.recv(65536)
                    if not data:
                        break
                    self.connection.sendall(data)
        except (OSError, ValueError, UnicodeError):
            return
        finally:
            self.server.release(upstream)

    def _relay(self, upstream: socket.socket) -> None:
        """Forward opaque CONNECT bytes only during the active attempt lease."""
        while self.server.active():
            try:
                ready, _, _ = select.select([upstream, self.connection], [], [], POLL_SECONDS)
                for source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    peer = self.connection if source is upstream else upstream
                    peer.sendall(data)
            except OSError:
                return
