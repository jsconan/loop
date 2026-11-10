"""Verify attempt-private egress brokering without opening native sockets."""

import array
import ipaddress
import json
import socket
import stat
from email.message import Message
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from loop.execution.sandbox.linux import egress

ORIGIN = "http://example.com"


@pytest.fixture
def broker(monkeypatch):
    """Construct one inert broker with a controlled clock and thread boundary."""
    clock = [10.0]
    monkeypatch.setattr(egress.time, "monotonic", lambda: clock[0])
    worker = Mock()
    worker.is_alive.return_value = False
    thread = Mock(return_value=worker)
    monkeypatch.setattr(egress.threading, "Thread", thread)
    monkeypatch.setattr(egress.socket, "MSG_CMSG_CLOEXEC", 0x40000000, raising=False)
    channel = Mock()
    channel.send.return_value = 1
    request = SimpleNamespace(egress_origins=(ORIGIN,), network=False, deadline=20.0)
    owner = egress.EgressBroker(request, channel)
    return owner, channel, clock, thread, monkeypatch


def test_broker_binds_first_heartbeat_and_pauses_new_routes(broker):
    """Open the namespace gate once and prevent new connections during policy activation."""
    owner, channel, clock, _thread, _patch = broker
    assert not owner.active()
    assert not owner.allow_new()
    assert not owner.lease_expired()
    owner.start_lease()
    assert owner.heartbeat(7)
    assert owner.heartbeat(7)
    assert not owner.heartbeat(6)
    assert not owner.heartbeat(-1)
    channel.send.assert_called_once_with(b"G")
    assert owner.active() and owner.allow_new()
    assert owner.pause(7)
    assert not owner.pause(8)
    assert owner.active() and not owner.allow_new()
    assert not owner.resume(8)
    assert owner.resume(7)
    assert owner.allow_new()
    clock[0] = 12.1
    assert owner.lease_expired()
    assert not owner.active()
    assert owner.close()
    assert owner.close()
    channel.close.assert_called_once()


@pytest.mark.parametrize("gate_failure", ["short-send", "closed-channel"])
def test_broker_refuses_an_unacknowledged_namespace_gate(broker, gate_failure):
    """A missing first gate byte cannot authorize the private proxy listener."""
    owner, channel, _clock, _thread, _patch = broker
    owner.start_lease()
    if gate_failure == "short-send":
        channel.send.return_value = 0
    else:
        channel.send.side_effect = OSError("closed")
    assert not owner.heartbeat(7)
    assert not owner.allow_new()
    owner.close()


def test_broker_close_reports_unsettled_worker_and_tolerates_closed_sockets(broker, monkeypatch):
    """Shutdown errors are bounded and a live worker prevents false closure confirmation."""
    owner, channel, _clock, _thread, _patch = broker
    client = Mock()
    client.shutdown.side_effect = OSError("already closed")
    owner._sockets.add(client)
    worker = Mock()
    worker.is_alive.return_value = True
    owner._workers.add(worker)
    channel.shutdown.side_effect = OSError("already closed")
    assert not owner.close()
    assert not owner.close()
    client.close.assert_called_once()
    worker.join.assert_called_once()
    monkeypatch.setattr(egress.threading, "current_thread", lambda: worker)
    second = egress.EgressBroker(
        SimpleNamespace(egress_origins=(ORIGIN,), network=False, deadline=20.0), Mock()
    )
    second._workers.add(worker)
    assert not second.close()


@pytest.mark.parametrize("revision", [7, 8])
def test_broker_revision_closes_every_tracked_route(broker, revision):
    """A newer revision confirms closure while a stale one fails closed."""
    owner, channel, _clock, _thread, _patch = broker
    owner.start_lease()
    assert owner.heartbeat(7)
    assert owner.accept_revision(revision) is (revision == 8)
    assert not owner.active()
    assert not owner.heartbeat(7)
    assert not owner.pause(7)
    assert not owner.resume(7)
    channel.close.assert_called_once()


def test_broker_refuses_missing_or_broad_grants(monkeypatch):
    """Never construct a mediator for no grant or unrestricted direct networking."""
    monkeypatch.setattr(egress.threading, "Thread", Mock())
    for origins, network in (((), False), ((ORIGIN,), True)):
        request = SimpleNamespace(egress_origins=origins, network=network)
        with pytest.raises(ValueError, match="mediated-only"):
            egress.EgressBroker(request, Mock())


def test_broker_selects_only_public_pinned_answer(broker, monkeypatch):
    """Skip forbidden resolver answers and connect to one checked numeric sockaddr."""
    owner, _channel, _clock, _thread, _patch = broker
    owner.start_lease()
    assert owner.heartbeat(7)
    answers = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.2", 80)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("64:ff9b::c0a8:401d", 80, 0, 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("11.0.0.2", 80)),
    ]
    monkeypatch.setattr(egress.socket, "getaddrinfo", Mock(return_value=answers))
    monkeypatch.setattr(egress.EgressBroker, "_external_route", staticmethod(lambda _address: True))
    upstream = Mock()
    monkeypatch.setattr(egress.socket, "socket", Mock(return_value=upstream))

    connected = owner.connect(ORIGIN)

    assert connected is upstream
    upstream.connect.assert_called_once_with(("11.0.0.2", 80))
    assert owner.connect("http://other.example") is None
    owner.release(upstream)
    upstream.close.assert_called_once()
    owner.close()


def test_broker_re_resolves_after_a_previous_allowed_connection(broker, monkeypatch):
    """A later forbidden DNS answer never reuses an earlier pinned address."""
    owner, _channel, _clock, _thread, _patch = broker
    owner.start_lease()
    assert owner.heartbeat(7)
    answers = [
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("11.0.0.2", 80))],
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.2", 80))],
    ]
    resolver = Mock(side_effect=answers)
    monkeypatch.setattr(egress.socket, "getaddrinfo", resolver)
    monkeypatch.setattr(egress.EgressBroker, "_external_route", staticmethod(lambda _address: True))
    upstream = Mock()
    monkeypatch.setattr(egress.socket, "socket", Mock(return_value=upstream))
    assert owner.connect(ORIGIN) is upstream
    owner.release(upstream)
    assert owner.connect(ORIGIN) is None
    assert resolver.call_count == 2
    upstream.connect.assert_called_once_with(("11.0.0.2", 80))
    owner.close()


@pytest.mark.parametrize("route", ["local", "unicast", "blackhole"])
def test_broker_route_probe_rejects_host_local_and_unusable_routes(monkeypatch, route):
    """Accept only a protected non-loopback unicast route for a public address."""
    details = SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | 0o755)
    monkeypatch.setattr(egress.os, "lstat", Mock(return_value=details))
    probe = Mock(
        return_value=SimpleNamespace(stdout=json.dumps([{"type": route, "dev": "eth0"}]).encode())
    )
    monkeypatch.setattr(egress.subprocess, "run", probe)
    assert egress.EgressBroker._external_route(ipaddress.ip_address("11.0.0.2")) is (
        route == "unicast"
    )
    probe.assert_called_once()


@pytest.mark.parametrize("unsafe", ["owner", "writable", "symlink", "output", "syntax", "tool"])
def test_broker_route_probe_fails_closed_on_untrusted_or_invalid_metadata(monkeypatch, unsafe):
    """Invalid route-tool ancestry, execution, or JSON cannot authorize a socket."""
    details = SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | 0o755)
    if unsafe == "owner":
        details.st_uid = 1000
    elif unsafe == "writable":
        details.st_mode |= 0o002
    elif unsafe == "symlink":
        details.st_mode = stat.S_IFLNK | 0o777
    monkeypatch.setattr(egress.os, "lstat", Mock(return_value=details))
    payload = b"x" * (egress.MAX_ROUTE_BYTES + 1) if unsafe == "output" else b"invalid"
    probe = Mock(return_value=SimpleNamespace(stdout=payload))
    if unsafe == "tool":
        probe.side_effect = OSError("missing route tool")
    monkeypatch.setattr(egress.subprocess, "run", probe)
    assert not egress.EgressBroker._external_route(ipaddress.ip_address("11.0.0.2"))


def _handler(method="GET", path=ORIGIN + "/", headers=None, body=b""):
    """Build a parser instance around byte streams without constructing a network service."""
    handler = object.__new__(egress.ProxyHandler)
    handler.command = method
    handler.path = path
    handler.headers = Message()
    for name, value in [("Host", "example.com")] if headers is None else headers:
        handler.headers.add_header(name, value)
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    handler.connection = Mock()
    handler.server = Mock()
    handler.send_error = Mock()
    handler.send_response = Mock()
    handler.end_headers = Mock()
    return handler


def test_visible_http_request_regenerates_authority_and_framing(monkeypatch):
    """A client's Connection tokens cannot strip the validated Host or body length."""
    upstream = Mock()
    upstream.recv.side_effect = [b"HTTP/1.1 200 OK\r\n\r\nbody", b""]
    handler = _handler(
        "POST",
        ORIGIN + "/submit?x=1",
        [
            ("Host", "example.com"),
            ("Connection", "Host, Content-Length"),
            ("Content-Length", "3"),
        ],
        b"abc",
    )
    handler.server.connect.return_value = upstream
    handler.server.active.return_value = True
    monkeypatch.setattr(egress.select, "select", lambda *_: ([upstream], [], []))

    handler.do_POST()

    sent = b"".join(call.args[0] for call in upstream.sendall.call_args_list)
    assert sent.startswith(b"POST /submit?x=1 HTTP/1.1\r\n")
    assert sent.count(b"Host: example.com\r\n") == 1
    assert sent.count(b"Content-Length: 3\r\n") == 1
    assert sent.endswith(b"\r\n\r\nabc")
    handler.connection.sendall.assert_called_once_with(b"HTTP/1.1 200 OK\r\n\r\nbody")
    handler.server.release.assert_called_once_with(upstream)
    handler.send_error.assert_not_called()


def test_visible_head_and_default_port_forward_only_checked_headers(monkeypatch):
    """HEAD on port 80 preserves the path and safe headers without a request body."""
    handler = _handler(
        "HEAD",
        "http://example.com:80/a?x=1",
        [("Host", "example.com:80"), ("X-Fixture", "safe"), ("Proxy-Authorization", "secret")],
    )
    upstream = Mock()
    upstream.recv.return_value = b""
    handler.server.connect.return_value = upstream
    handler.server.active.return_value = True
    monkeypatch.setattr(egress.select, "select", lambda *_: ([upstream], [], []))
    handler.do_HEAD()
    sent = b"".join(call.args[0] for call in upstream.sendall.call_args_list)
    assert b"HEAD /a?x=1 HTTP/1.1\r\n" in sent
    assert b"X-Fixture: safe\r\n" in sent
    assert b"Proxy-Authorization" not in sent
    handler.server.connect.assert_called_once_with(ORIGIN)


@pytest.mark.parametrize(
    ("method", "headers", "body", "fault"),
    [
        ("GET", [("Host", "example.com")], b"", "unapproved"),
        ("GET", [("Host", "example.com"), ("Content-Length", "1")], b"", "short-body"),
        ("POST", [("Host", "example.com"), ("Content-Length", "2")], b"x", "partial-body"),
        ("GET", [("Host", "example.com")], b"", "upstream-error"),
        ("GET", [("Host", "example.com")], b"", "idle-upstream"),
    ],
)
def test_visible_http_fails_closed_when_transport_or_body_is_incomplete(
    monkeypatch, method, headers, body, fault
):
    """No denied or incomplete HTTP request can continue forwarding another response."""
    handler = _handler(method, ORIGIN + "/", headers, body)
    if fault == "unapproved":
        handler.server.connect.return_value = None
        getattr(handler, "do_" + method)()
        handler.send_error.assert_called_once_with(403)
        return
    upstream = Mock()
    handler.server.connect.return_value = upstream
    if fault == "partial-body":
        handler.server.active.side_effect = [True, False]
    elif fault == "idle-upstream":
        handler.server.active.side_effect = [True, False]
        monkeypatch.setattr(egress.select, "select", lambda *_: ([], [], []))
    else:
        handler.server.active.return_value = True
        monkeypatch.setattr(egress.select, "select", lambda *_: ([upstream], [], []))
    if fault == "upstream-error":
        upstream.sendall.side_effect = OSError("route closed")
    getattr(handler, "do_" + method)()
    handler.server.release.assert_called_once_with(upstream)
    handler.connection.sendall.assert_not_called()


def test_https_connect_denies_unapproved_origin_after_validating_authority():
    """A well-formed CONNECT still needs its exact approved destination."""
    handler = _handler("CONNECT", "example.com:8443", [("Host", "example.com:8443")])
    handler.server.connect.return_value = None
    handler.do_CONNECT()
    handler.server.connect.assert_called_once_with("https://example.com:8443")
    handler.send_error.assert_called_once_with(403)


def test_opaque_tunnel_relays_only_while_lease_is_active(monkeypatch):
    """Forward one opaque chunk and stop on EOF without inspecting TLS contents."""
    handler = _handler()
    upstream = Mock()
    handler.server.active.side_effect = [True, True]
    polls = iter([([upstream], [], []), ([handler.connection], [], [])])
    monkeypatch.setattr(egress.select, "select", lambda *_: next(polls))
    upstream.recv.return_value = b"tls-data"
    handler.connection.recv.return_value = b""
    handler._relay(upstream)
    handler.connection.sendall.assert_called_once_with(b"tls-data")
    upstream.sendall.assert_not_called()


def test_opaque_tunnel_stops_on_transport_failure(monkeypatch):
    """A closed source or write error never keeps the CONNECT tunnel alive."""
    handler = _handler()
    upstream = Mock()
    handler.server.active.return_value = True
    monkeypatch.setattr(egress.select, "select", lambda *_: ([handler.connection], [], []))
    handler.connection.recv.return_value = b"tls-data"
    upstream.sendall.side_effect = OSError("closed")
    handler._relay(upstream)
    upstream.sendall.assert_called_once_with(b"tls-data")


def test_opaque_tunnel_does_not_poll_after_lease_expiry(monkeypatch):
    """An expired attempt cannot forward even a queued TLS chunk."""
    handler = _handler()
    handler.server.active.return_value = False
    polling = Mock()
    monkeypatch.setattr(egress.select, "select", polling)
    handler._relay(Mock())
    polling.assert_not_called()


def test_proxy_log_and_worker_errors_do_not_disclose_request_data(broker, monkeypatch):
    """The parser omits URLs from routine logs and releases failed client workers."""
    handler = _handler()
    handler.log_message("%s", "secret-path")
    owner, _channel, _clock, _thread, _patch = broker
    client = Mock()
    client.getpeername.return_value = ("127.0.0.1", 4000)
    parser = Mock(side_effect=ValueError("malformed"))
    monkeypatch.setattr(egress, "ProxyHandler", parser)
    owner._serve(client)
    parser.assert_called_once()
    client.close.assert_called_once()
    owner.close()


@pytest.mark.parametrize(
    ("method", "path", "headers"),
    [
        ("GET", "ftp://example.com/", [("Host", "example.com")]),
        ("GET", ORIGIN + "/#fragment", [("Host", "example.com")]),
        ("GET", ORIGIN + "/", []),
        ("GET", ORIGIN + "/", [("Host", "other.example")]),
        ("GET", ORIGIN + "/", [("Host", "example.com"), ("Host", "example.com")]),
        ("GET", ORIGIN + "/", [("Host", "example.com"), ("Transfer-Encoding", "chunked")]),
        ("GET", ORIGIN + "/", [("Host", "example.com"), ("Upgrade", "websocket")]),
        ("GET", ORIGIN + "/", [("Host", "example.com"), ("Expect", "100-continue")]),
        ("GET", ORIGIN + "/", [("Host", "example.com"), ("Content-Length", "-1")]),
        ("GET", ORIGIN + "/", [("Host", "example.com"), ("Content-Length", "1,2")]),
        ("POST", ORIGIN + "/", [("Host", "example.com")]),
    ],
)
def test_visible_http_rejects_ambiguous_or_unsupported_requests(method, path, headers):
    """Malformed authority, framing, and unsupported schemes never reach an upstream socket."""
    handler = _handler(method, path, headers)
    getattr(handler, "do_" + method)()
    handler.send_error.assert_called_once_with(403)
    handler.server.connect.assert_not_called()


@pytest.mark.parametrize("authority", ["example.com", "example.com:443/path", "other:443"])
def test_https_connect_requires_an_exact_approved_authority(authority):
    """Reject malformed CONNECT targets and other destinations before tunnel creation."""
    handler = _handler("CONNECT", authority)
    handler.server.connect.return_value = None
    handler.do_CONNECT()
    handler.send_error.assert_called_once_with(403)


def test_https_connect_accepts_default_port_and_closes_opaque_tunnel(monkeypatch):
    """Canonicalize standard port 443 and release the pinned tunnel after relay."""
    handler = _handler("CONNECT", "example.com:443", [("Host", "example.com:443")])
    upstream = Mock()
    handler.server.connect.return_value = upstream
    monkeypatch.setattr(egress.ProxyHandler, "_relay", Mock())
    handler.do_CONNECT()
    handler.server.connect.assert_called_once_with("https://example.com")
    handler.send_response.assert_called_once_with(200, "Connection Established")
    handler.server.release.assert_called_once_with(upstream)


def test_namespace_proxy_requires_first_authenticated_gate(monkeypatch):
    """A payload cannot open its local listener until the host broker sends its gate."""
    channel = Mock()
    channel.recv.return_value = b"X"
    monkeypatch.setattr(egress.socket, "socket", Mock(return_value=channel))
    with pytest.raises(OSError, match="did not activate"):
        egress.start_namespace_proxy(9)
    channel.close.assert_called_once()


def test_namespace_proxy_binds_only_private_loopback_after_gate(monkeypatch):
    """The admitted payload creates an attempt-private loopback listener and closes setup FDs."""
    channel = Mock()
    channel.recv.return_value = b"G"
    listener = Mock()
    listener.getsockname.return_value = ("127.0.0.1", 4567)
    factory = Mock(side_effect=[channel, listener])
    monkeypatch.setattr(egress.socket, "socket", factory)
    monkeypatch.setattr(egress.os, "fork", lambda: 123)
    libc = SimpleNamespace(prctl=Mock(return_value=0))
    monkeypatch.setattr(egress.ctypes, "CDLL", Mock(return_value=libc))
    assert egress.start_namespace_proxy(9) == 4567
    assert [call.args[0] for call in libc.prctl.call_args_list] == [4, 3]
    listener.bind.assert_called_once_with(("127.0.0.1", 0))
    listener.listen.assert_called_once_with(16)
    listener.close.assert_called_once()
    channel.close.assert_called_once()


def test_namespace_proxy_child_hands_only_accepted_fd_to_broker(monkeypatch):
    """The sidecar forwards one accepted loopback socket and exits on listener failure."""
    channel = Mock()
    channel.recv.return_value = b"G"
    listener = Mock()
    listener.getsockname.return_value = ("127.0.0.1", 4567)
    accepted = MagicMock()
    accepted.fileno.return_value = 91
    listener.accept.side_effect = [(accepted, None), OSError("listener closed")]
    monkeypatch.setattr(egress.socket, "socket", Mock(side_effect=[channel, listener]))
    monkeypatch.setattr(egress.os, "fork", lambda: 0)
    monkeypatch.setattr(
        egress.ctypes, "CDLL", Mock(return_value=SimpleNamespace(prctl=Mock(return_value=0)))
    )
    monkeypatch.setattr(egress.os, "dup2", Mock())

    def terminate(_status):
        """Expose the child exit without terminating pytest."""
        raise SystemExit(0)

    monkeypatch.setattr(egress.os, "_exit", terminate)
    with pytest.raises(SystemExit):
        egress.start_namespace_proxy(9)
    channel.sendmsg.assert_called_once()
    assert channel.sendmsg.call_args.args[0] == [b"C"]
    listener.close.assert_called_once()
    channel.close.assert_called_once()


@pytest.mark.parametrize("failure", ["set", "verify"])
def test_namespace_proxy_refuses_unsealed_child_before_fork(monkeypatch, failure):
    """A failed dumpability seal stops proxy creation before any untrusted handoff."""
    channel = Mock()
    channel.recv.return_value = b"G"
    factory = Mock(return_value=channel)
    monkeypatch.setattr(egress.socket, "socket", factory)
    libc = SimpleNamespace(prctl=Mock(side_effect=[-1] if failure == "set" else [0, 1]))
    monkeypatch.setattr(egress.ctypes, "CDLL", Mock(return_value=libc))
    fork = Mock()
    monkeypatch.setattr(egress.os, "fork", fork)
    with pytest.raises(OSError, match="identity could not be sealed"):
        egress.start_namespace_proxy(9)
    fork.assert_not_called()
    factory.assert_called_once_with(fileno=9)
    channel.close.assert_called_once()


@pytest.mark.parametrize(
    "case", ["dns", "unsupported-family", "invalid-answer", "socket", "connect", "paused"]
)
def test_broker_connection_failures_never_return_an_upstream(broker, monkeypatch, case):
    """DNS, route, socket, connect and policy changes fail before transport reuse."""
    owner, _channel, _clock, _thread, _patch = broker
    owner.start_lease()
    assert owner.heartbeat(7)
    if case == "paused":
        assert owner.pause(7)
        assert owner.connect(ORIGIN) is None
        owner.close()
        return
    if case == "dns":
        monkeypatch.setattr(egress.socket, "getaddrinfo", Mock(side_effect=OSError("dns")))
    else:
        family = socket.AF_UNIX if case == "unsupported-family" else socket.AF_INET
        address = "bad-ip" if case == "invalid-answer" else "11.0.0.2"
        monkeypatch.setattr(
            egress.socket,
            "getaddrinfo",
            Mock(return_value=[(family, socket.SOCK_STREAM, 6, "", (address, 80))]),
        )
    monkeypatch.setattr(egress.EgressBroker, "_external_route", staticmethod(lambda _: True))
    upstream = Mock()
    if case == "socket":
        monkeypatch.setattr(egress.socket, "socket", Mock(side_effect=OSError("socket")))
    else:
        monkeypatch.setattr(egress.socket, "socket", Mock(return_value=upstream))
    if case == "connect":
        upstream.connect.side_effect = OSError("refused")
    assert owner.connect(ORIGIN) is None
    if case == "connect":
        upstream.close.assert_called_once()
    owner.close()


@pytest.mark.parametrize(
    "case", ["missing-host", "route", "paused-after-dns", "capacity", "revoked-after-connect"]
)
def test_broker_rechecks_each_transport_before_admission(broker, monkeypatch, case):
    """Reject malformed origins, route changes, capacity, and a revoked pinned connection."""
    owner, _channel, _clock, _thread, _patch = broker
    owner.start_lease()
    assert owner.heartbeat(7)
    if case == "missing-host":
        owner.request.egress_origins = ("http://",)
        assert owner.connect("http://") is None
        owner.close()
        return
    monkeypatch.setattr(
        egress.socket,
        "getaddrinfo",
        Mock(return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("11.0.0.2", 80))]),
    )
    monkeypatch.setattr(
        egress.EgressBroker, "_external_route", staticmethod(lambda _: case != "route")
    )
    upstream = Mock()
    monkeypatch.setattr(egress.socket, "socket", Mock(return_value=upstream))
    if case == "paused-after-dns":
        monkeypatch.setattr(owner, "allow_new", Mock(side_effect=[True, False]))
    elif case == "capacity":
        monkeypatch.setattr(owner, "_track", Mock(return_value=False))
    elif case == "revoked-after-connect":
        monkeypatch.setattr(owner, "allow_new", Mock(side_effect=[True, True, False]))
    assert owner.connect(ORIGIN) is None
    if case in {"capacity", "revoked-after-connect"}:
        upstream.close.assert_called_once()
    owner.close()


def test_broker_refuses_new_socket_after_attempt_socket_limit(broker):
    """A saturated attempt never tracks another client or upstream socket."""
    owner, _channel, _clock, _thread, _patch = broker
    owner._sockets = {Mock() for _ in range(egress.MAX_CLIENTS * 2)}
    assert not owner._track(Mock())
    owner._sockets.clear()
    assert owner.close()


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "wrong-data",
        "truncated",
        "two-fds",
        "receive-error",
        "start-error",
        "idle",
        "ignored-control",
        "fd-ctor",
        "fd-close-fails",
        "not-admitted",
        "worker-limit",
        "peer-error",
    ],
)
def test_broker_handoff_accepts_only_one_private_tcp_descriptor(monkeypatch, case):
    """Close malformed FD handoffs and never admit a foreign or truncated descriptor."""
    clock = [10.0]
    monkeypatch.setattr(egress.time, "monotonic", lambda: clock[0])
    reader = {}
    workers = []

    def thread(*, target, daemon, args=()):
        """Keep native worker callbacks deterministic without starting real threads."""
        worker = Mock()
        worker.is_alive.return_value = False
        workers.append(worker)
        if not reader:
            reader["run"] = target
        elif case == "start-error":
            worker.start.side_effect = RuntimeError("cannot start")
        return worker

    monkeypatch.setattr(egress.threading, "Thread", thread)
    monkeypatch.setattr(egress.socket, "MSG_CMSG_CLOEXEC", 0x40000000, raising=False)
    channel = Mock()
    if case == "idle":
        polls = iter([([], [], []), ([channel], [], []), ([channel], [], [])])
        monkeypatch.setattr(egress.select, "select", lambda *_: next(polls))
    else:
        monkeypatch.setattr(egress.select, "select", lambda *_: ([channel], [], []))
    close_fd = Mock()
    if case == "fd-close-fails":
        close_fd.side_effect = OSError("already closed")
    monkeypatch.setattr(egress.os, "close", close_fd)
    descriptor = array.array("i", [91, 92] if case == "two-fds" else [91]).tobytes()
    controls = (
        [(0, 0, descriptor)]
        if case == "ignored-control"
        else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptor)]
    )
    data = b"X" if case == "wrong-data" else b"C"
    flags = socket.MSG_TRUNC if case == "truncated" else socket.MSG_CMSG_CLOEXEC
    channel.send.return_value = 1
    if case == "receive-error":
        channel.recvmsg.side_effect = OSError("handoff closed")
    else:
        channel.recvmsg.side_effect = [(data, controls, flags, None), (b"", [], 0, None)]
    accepted = Mock()
    accepted.family = socket.AF_INET
    accepted.type = socket.SOCK_STREAM
    accepted.getsockname.return_value = ("127.0.0.1", 41000)
    accepted.getpeername.return_value = ("127.0.0.1", 41001)
    if case == "peer-error":
        accepted.getpeername.side_effect = OSError("peer disappeared")
    factory = Mock(return_value=accepted)
    if case in {"fd-ctor", "fd-close-fails"}:
        factory.side_effect = OSError("bad descriptor")
    monkeypatch.setattr(egress.socket, "socket", factory)
    request = SimpleNamespace(egress_origins=(ORIGIN,), network=False, deadline=20.0)
    owner = egress.EgressBroker(request, channel)
    owner.start_lease()
    assert owner.heartbeat(7)
    if case == "not-admitted":
        assert owner.pause(7)
    elif case == "worker-limit":
        monkeypatch.setattr(egress, "MAX_CLIENTS", 1)
        occupied = Mock()
        occupied.is_alive.return_value = False
        owner._workers.add(occupied)

    reader["run"]()

    assert not owner.active()
    channel.close.assert_called_once()
    if case in {"valid", "idle"}:
        factory.assert_called_once_with(fileno=91)
        assert len(workers) == 2
        workers[1].start.assert_called_once()
        accepted.close.assert_called_once()
    elif case == "start-error" or case in {"not-admitted", "worker-limit", "peer-error"}:
        accepted.close.assert_called_once()
    elif case in {"wrong-data", "truncated", "two-fds"}:
        assert close_fd.call_count == (2 if case == "two-fds" else 1)
        factory.assert_not_called()
    elif case in {"fd-ctor", "fd-close-fails"}:
        factory.assert_called_once_with(fileno=91)
        close_fd.assert_called_once_with(91)
    elif case == "ignored-control":
        close_fd.assert_not_called()
        factory.assert_not_called()
    else:
        factory.assert_not_called()


@pytest.mark.parametrize("problem", ["family", "type", "local", "peer"])
def test_broker_handoff_rejects_nonprivate_or_nontcp_socket(monkeypatch, problem):
    """A passed descriptor cannot redirect broker traffic to a host or cross-session socket."""
    reader = {}
    monkeypatch.setattr(egress.time, "monotonic", lambda: 10.0)

    def thread(*, target, daemon):
        """Capture the broker reader without a native thread."""
        reader.setdefault("run", target)
        worker = Mock()
        worker.is_alive.return_value = False
        return worker

    monkeypatch.setattr(egress.threading, "Thread", thread)
    monkeypatch.setattr(egress.socket, "MSG_CMSG_CLOEXEC", 0x40000000, raising=False)
    channel = Mock()
    channel.send.return_value = 1
    channel.recvmsg.side_effect = [
        (b"C", [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [91]).tobytes())], 0, None),
        (b"", [], 0, None),
    ]
    monkeypatch.setattr(egress.select, "select", lambda *_: ([channel], [], []))
    accepted = Mock()
    accepted.family = socket.AF_UNIX if problem == "family" else socket.AF_INET
    accepted.type = socket.SOCK_DGRAM if problem == "type" else socket.SOCK_STREAM
    accepted.getsockname.return_value = ("192.168.1.2" if problem == "local" else "127.0.0.1", 4000)
    accepted.getpeername.return_value = ("192.168.1.3" if problem == "peer" else "127.0.0.1", 4001)
    monkeypatch.setattr(egress.socket, "socket", Mock(return_value=accepted))
    owner = egress.EgressBroker(
        SimpleNamespace(egress_origins=(ORIGIN,), network=False, deadline=20.0), channel
    )
    owner.start_lease()
    assert owner.heartbeat(7)
    reader["run"]()
    accepted.close.assert_called_once()


def test_closed_broker_reader_does_not_receive_another_descriptor(broker, monkeypatch):
    """A completed attempt cannot reenter FD handoff after closure."""
    owner, channel, _clock, thread, _patch = broker
    assert owner.close()
    polling = Mock()
    monkeypatch.setattr(egress.select, "select", polling)
    thread.call_args.kwargs["target"]()
    polling.assert_not_called()
    channel.close.assert_called_once()
