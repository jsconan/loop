"""Verify macOS attempt-owned mediation without opening native sockets."""

import ipaddress
import socket
import stat
import threading
import time
from email.message import Message
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.macos import egress

HTTP_ORIGIN = "http://example.com"
HTTPS_ORIGIN = "https://example.com"


@pytest.fixture
def broker(tmp_path, monkeypatch):
    """Start one fake listener under a controlled policy revision and clock."""
    tmp_path.chmod(0o700)
    clock = [10.0]
    monkeypatch.setattr(
        egress.time,
        "monotonic",
        lambda: clock[0],
    )
    listener = Mock()
    monkeypatch.setattr(egress.socket, "socket", Mock(return_value=listener))
    worker = Mock()
    worker.is_alive.return_value = False
    thread = Mock(return_value=worker)
    monkeypatch.setattr(egress.threading, "Thread", thread)
    authority = Mock()
    authority.try_egress_policy_revision.return_value = 7
    request = SimpleNamespace(
        egress_origins=(HTTP_ORIGIN, HTTPS_ORIGIN), network=False, deadline=30.0
    )
    owner = egress.MacOSEgressBroker(request, tmp_path / "proxy.sock", authority, 7)
    owner.__enter__()
    return owner, listener, authority, clock, thread


def test_broker_requires_one_scoped_approved_revision(tmp_path):
    """A broad, absent, or invalid grant cannot construct a mediator."""
    for origins, network, revision in (
        ((), False, 0),
        ((HTTP_ORIGIN,), True, 0),
        ((HTTP_ORIGIN,), False, -1),
    ):
        request = SimpleNamespace(egress_origins=origins, network=network)
        with pytest.raises(ValueError, match="mediated-only"):
            egress.MacOSEgressBroker(request, tmp_path / "proxy", Mock(), revision)


def test_broker_bounds_stalled_resolutions_across_closed_instances(tmp_path, monkeypatch):
    """A stalled resolver consumes one shared slot until it actually returns."""
    tmp_path.chmod(0o700)
    monkeypatch.setattr(egress, "_RESOLUTION_SLOTS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(egress, "AUTHORITY_SECONDS", 0.2)
    monkeypatch.setattr(egress, "POLL_SECONDS", 0.01)
    started = threading.Event()
    release = threading.Event()
    finish_watch = threading.Event()
    monkeypatch.setattr(egress.time, "sleep", lambda _seconds: finish_watch.wait(5))
    resolved = []

    def lookup(*_args, **_kwargs):
        """Hold the first synthetic DNS query until its broker has closed."""
        resolved.append(True)
        started.set()
        assert release.wait(5)
        return []

    monkeypatch.setattr(egress.socket, "getaddrinfo", lookup)
    listeners = [Mock() for _ in range(3)]
    clients = [Mock() for _ in listeners]
    ready = {listener: threading.Event() for listener in listeners}
    replies = [threading.Event() for _ in clients]
    for listener, client, reply in zip(listeners, clients, replies):
        client.recv.side_effect = [
            b"\x05\x01",
            b"\x00",
            b"\x05\x01\x00\x03",
            b"\x0b",
            b"example.com",
            b"\x00\x50",
        ]
        client.sendall.side_effect = lambda data, event=reply: (
            event.set() if data.startswith(b"\x05\x02") else None
        )
        listener.accept.side_effect = lambda listener=listener, client=client: (
            ready[listener].clear(),
            (client, None),
        )[1]
        ready[listener].set()

    def select_listeners(readers, _writers, _errors, timeout):
        """Signal only mocked listeners with a pending synthetic client."""
        listener = readers[0]
        return ([listener] if ready[listener].wait(timeout) else [], [], [])

    monkeypatch.setattr(egress.select, "select", select_listeners)
    monkeypatch.setattr(egress.socket, "socket", Mock(side_effect=listeners))
    authority = Mock()
    authority.try_egress_policy_revision.return_value = 7
    request = SimpleNamespace(
        egress_origins=(HTTP_ORIGIN,), network=False, deadline=time.monotonic() + 10
    )
    owners = [
        egress.MacOSEgressBroker(request, tmp_path / f"{n}.sock", authority, 7) for n in range(3)
    ]
    entered = []
    try:
        owners[0].__enter__()
        entered.append(owners[0])
        assert started.wait(5)
        assert owners[0].close() is False
        owners[1].__enter__()
        entered.append(owners[1])
        assert replies[1].wait(5)
        assert len(resolved) == 1
        release.set()
        assert replies[0].wait(5)
        owners[2].__enter__()
        entered.append(owners[2])
        assert replies[2].wait(5)
        assert len(resolved) == 2
    finally:
        release.set()
        try:
            for owner in reversed(entered):
                if owner is owners[0]:
                    with pytest.raises(OSError, match="closure could not be confirmed"):
                        owner.__exit__(None, None, None)
                else:
                    owner.__exit__(None, None, None)
        finally:
            finish_watch.set()


def test_broker_binds_and_settles_policy_lifecycle(broker):
    """Pause blocks new admissions and a newer revision closes every owned route."""
    owner, listener, authority, clock, _thread = broker
    authority.bind_egress.assert_called_once_with(7, owner)
    listener.bind.assert_called_once_with(str(owner.path))
    assert owner.allow_new() and owner.active() and not owner.interrupted
    assert owner.pause() and not owner.allow_new() and owner.active()
    assert owner.resume() and owner.allow_new()
    assert not owner.resume()
    assert not owner.revoke_revision(7)
    assert owner.revoke_revision(8)
    assert owner.revoke_revision(8)
    assert not owner.revoke_revision(9)
    assert owner.interrupted and not owner.active()
    assert owner.pause()
    owner.__exit__(None, None, None)
    authority.unbind_egress.assert_called_once_with(owner)
    assert owner.close()
    clock[0] = 31.0
    assert not owner.allow_new()


def test_broker_cannot_resume_an_expired_policy_lease(broker):
    """A delayed policy update cannot refresh expired mediator authority."""
    owner, _listener, _authority, clock, _thread = broker
    assert owner.pause()
    clock[0] = 12.1
    assert not owner.resume()
    assert owner.interrupted and not owner.active()
    owner.__exit__(None, None, None)


def test_broker_pause_deadline_expires_despite_healthy_heartbeat(broker):
    """A withheld revision ack closes covered tunnels even while health remains current."""
    owner, _listener, authority, clock, _thread = broker
    assert owner.pause()
    clock[0] = 11.0
    assert owner.pause()
    clock[0] = 12.1
    owner._last_heartbeat = clock[0]
    assert not owner.resume()
    assert owner.interrupted and not owner.active()
    authority.try_egress_policy_revision.assert_not_called()
    owner.__exit__(None, None, None)


def test_broker_binds_runtime_ceiling_only_before_preparation_expires(broker):
    """A live launched attempt retains its runtime beyond the setup ceiling."""
    owner, _listener, _authority, clock, _thread = broker
    clock[0] = 11.0
    assert owner.bind_runtime_deadline(40.0)
    clock[0] = 29.0
    owner._last_heartbeat = 29.0
    assert owner.active()
    clock[0] = 40.0
    assert not owner.active()
    owner.__exit__(None, None, None)


def test_broker_cannot_extend_a_spent_preparation_ceiling(broker):
    """A launch that crosses the preparation ceiling cannot reopen egress."""
    owner, _listener, _authority, clock, _thread = broker
    clock[0] = 30.0
    assert not owner.bind_runtime_deadline(50.0)
    assert owner.interrupted and not owner.active()
    owner.__exit__(None, None, None)


def test_broker_cannot_activate_after_concurrent_policy_revocation(tmp_path, monkeypatch):
    """A policy change during binding cannot resurrect a closed listener."""
    tmp_path.chmod(0o700)
    listener = Mock()
    monkeypatch.setattr(egress.socket, "socket", Mock(return_value=listener))
    authority = Mock()
    authority.bind_egress.side_effect = lambda _revision, lease: lease.revoke_revision(8)
    request = SimpleNamespace(egress_origins=(HTTP_ORIGIN,), network=False, deadline=30.0)
    owner = egress.MacOSEgressBroker(request, tmp_path / "proxy.sock", authority, 7)
    with pytest.raises(OSError, match="authority ended during activation"):
        owner.__enter__()
    assert not owner.active()
    authority.unbind_egress.assert_called_once_with(owner)
    listener.close.assert_called()


def test_broker_requires_timely_revocation_confirmation(broker, monkeypatch):
    """Late cleanup cannot acknowledge a policy revision as safely closed."""
    owner, _listener, _authority, clock, _thread = broker
    original_close = owner.close

    def late_close():
        """Advance the controlled clock while settling all covered transports."""
        clock[0] += 2.1
        return original_close()

    monkeypatch.setattr(owner, "close", late_close)
    assert not owner.revoke_revision(8)
    assert not owner.revoke_revision(8)
    owner.__exit__(None, None, None)


@pytest.mark.parametrize("case", ["directory", "existing", "bind", "authority", "thread"])
def test_broker_refuses_unsafe_or_failed_setup(tmp_path, monkeypatch, case):
    """An untrusted endpoint or failed setup closes its listener without a grant."""
    tmp_path.chmod(0o700)
    request = SimpleNamespace(egress_origins=(HTTP_ORIGIN,), network=False, deadline=30.0)
    authority = Mock()
    listener = Mock()
    monkeypatch.setattr(egress.socket, "socket", Mock(return_value=listener))
    thread = Mock()
    thread.return_value.is_alive.return_value = False
    monkeypatch.setattr(egress.threading, "Thread", thread)
    path = tmp_path / "proxy.sock"
    if case == "directory":
        tmp_path.chmod(0o755)
    elif case == "existing":
        path.write_bytes(b"not a socket")
    elif case == "bind":
        listener.bind.side_effect = OSError("bind failed")
    elif case == "authority":
        authority.bind_egress.side_effect = OSError("policy changed")
    else:
        thread.return_value.start.side_effect = RuntimeError("thread failed")
    owner = egress.MacOSEgressBroker(request, path, authority, 7)
    with pytest.raises((OSError, RuntimeError)):
        owner.__enter__()
    assert not owner.allow_new()
    if case not in {"directory", "existing"}:
        listener.close.assert_called_once()


def test_broker_close_reports_unsettled_workers(broker):
    """A live worker or uncertain closure cannot acknowledge policy revocation."""
    owner, listener, authority, _clock, _thread = broker
    client = Mock()
    client.shutdown.side_effect = OSError("already closed")
    worker = Mock()
    worker.is_alive.return_value = True
    owner._sockets.add(client)
    owner._workers.add(worker)
    assert not owner.revoke_revision(8)
    assert not owner.revoke_revision(8)
    with pytest.raises(OSError, match="closure could not be confirmed"):
        owner.__exit__(None, None, None)
    client.close.assert_called_once()
    listener.close.assert_called_once()
    worker.join.assert_called_once()
    authority.unbind_egress.assert_called_once_with(owner)


def test_broker_waits_for_concurrent_close(broker, monkeypatch):
    """A second closer waits outside the lock for the first closure verdict."""
    owner, _listener, _authority, _clock, _thread = broker
    owner._closing = True
    owner._confirmed = True
    owner._closed.set()
    assert owner.close()
    owner._closing = False
    owner.__exit__(None, None, None)


def test_broker_without_a_listener_closes_before_binding(tmp_path):
    """An unstarted broker has no descriptor or policy grant to retain."""
    request = SimpleNamespace(egress_origins=(HTTP_ORIGIN,), network=False, deadline=30.0)
    owner = egress.MacOSEgressBroker(request, tmp_path / "proxy.sock", Mock(), 7)
    assert owner.close()
    assert owner.pause()
    owner.__exit__(None, None, None)


def test_broker_close_skips_its_current_worker(broker, monkeypatch):
    """A worker ending its own route cannot join itself."""
    owner, *_ = broker
    worker = Mock()
    owner._workers.add(worker)
    monkeypatch.setattr(egress.threading, "current_thread", lambda: worker)
    assert owner.close()
    worker.join.assert_not_called()
    owner.__exit__(None, None, None)


@pytest.mark.parametrize("state", ["changed", "error", "expired"])
def test_broker_watchdog_closes_on_authority_loss(broker, monkeypatch, state):
    """Policy changes, verification errors and missed leases end covered transports."""
    owner, _listener, authority, clock, _thread = broker
    if state == "changed":
        authority.try_egress_policy_revision.return_value = 8
    elif state == "error":
        authority.try_egress_policy_revision.side_effect = OSError("managed policy gone")
    else:
        authority.try_egress_policy_revision.return_value = None
        clock[0] = 12.1
    monkeypatch.setattr(egress.time, "sleep", Mock())
    owner._watch()
    assert owner.interrupted and not owner.active()
    owner.__exit__(None, None, None)


def test_broker_watchdog_renews_matching_revision(broker, monkeypatch):
    """A current policy refreshes the independent lease before the next check."""
    owner, _listener, _authority, clock, _thread = broker
    clock[0] = 11.0
    active = Mock(side_effect=[True, False])
    monkeypatch.setattr(owner, "active", active)
    monkeypatch.setattr(egress.time, "sleep", Mock())
    owner._watch()
    assert owner._last_heartbeat == 11.0
    owner.__exit__(None, None, None)


def test_broker_watchdog_exits_after_prior_close(broker):
    """A settled broker never checks authority again."""
    owner, _listener, authority, _clock, _thread = broker
    owner.close()
    owner._watch()
    authority.try_egress_policy_revision.assert_not_called()
    owner.__exit__(None, None, None)


def test_broker_watchdog_closes_after_unexpected_authority_error(broker):
    """An unexpected manager error cannot leave live broker descriptors behind."""
    owner, listener, authority, _clock, _thread = broker
    authority.try_egress_policy_revision.side_effect = RuntimeError("authority failure")
    with pytest.raises(RuntimeError, match="authority failure"):
        owner._watch()
    assert owner.interrupted and not owner.active()
    listener.close.assert_called()
    owner.__exit__(None, None, None)


@pytest.mark.parametrize(
    "case",
    ["missing-listener", "idle", "accept-error", "rejected", "saturated", "worker", "start-error"],
)
def test_broker_accepts_only_bounded_owned_clients(broker, monkeypatch, case):
    """Listener failures and admission limits close clients without orphaned workers."""
    owner, listener, _authority, _clock, thread = broker
    if case == "missing-listener":
        owner._listener = None
        owner._accept()
        listener.close()
        owner.__exit__(None, None, None)
        return
    monkeypatch.setattr(owner, "active", Mock(side_effect=[True, False]))
    monkeypatch.setattr(
        egress.select,
        "select",
        Mock(return_value=([], [], []) if case == "idle" else ([listener], [], [])),
    )
    client = Mock()
    listener.accept.return_value = (client, "")
    if case == "accept-error":
        listener.accept.side_effect = OSError("closed")
    if case == "rejected":
        monkeypatch.setattr(owner, "_track", Mock(return_value=False))
    if case == "saturated":
        owner._workers = {Mock() for _ in range(egress.MAX_CLIENTS)}
        for worker in owner._workers:
            worker.is_alive.return_value = False
    if case == "start-error":
        thread.return_value.start.side_effect = RuntimeError("cannot start")
    owner._accept()
    assert not owner.allow_new()
    if case in {"rejected", "saturated", "start-error"}:
        client.close.assert_called()
    owner.__exit__(None, None, None)


def test_broker_acceptor_exits_after_normal_close(broker, monkeypatch):
    """A settled listener does not mark an ordinary shutdown as interrupted."""
    owner, _listener, _authority, _clock, _thread = broker

    def settled():
        """Close before a subsequent acceptor poll."""
        owner.close()
        return False

    monkeypatch.setattr(owner, "active", settled)
    owner._accept()
    assert not owner.interrupted
    owner.__exit__(None, None, None)


def test_broker_matches_only_one_canonical_socks_origin(broker):
    """A copied IP or ambiguous scheme cannot inherit a DNS-name approval."""
    owner, *_ = broker
    connection = Mock()
    connection.recv.side_effect = [b"\x0b", b"example.com", b"\x00\x50"]
    assert owner._origin(3, connection) == HTTP_ORIGIN
    connection.recv.side_effect = [b"\x01\x01\x01\x01", b"\x00\x50"]
    assert owner._origin(1, connection) is None
    assert owner._origin(4, connection) is None
    connection.recv.side_effect = [b"\x00"]
    assert owner._origin(3, connection) is None
    connection.recv.side_effect = [b"\x0b", b"example.com", b"\x01\xbb"]
    assert owner._origin(3, connection) == HTTPS_ORIGIN
    owner.__exit__(None, None, None)


def test_broker_refuses_ambiguous_and_invalid_socks_names(broker):
    """A host-port shared across schemes or malformed name is rejected."""
    owner, *_ = broker
    owner.request.egress_origins = ("http://example.com:8443", "https://example.com:8443")
    connection = Mock()
    connection.recv.side_effect = [b"\x0b", b"example.com", b"\x20\xfb"]
    assert owner._origin(3, connection) is None
    connection.recv.side_effect = [b"\x03", b"a!b", b"\x00\x50"]
    assert owner._origin(3, connection) is None
    owner.__exit__(None, None, None)


def test_bounded_socks_reads_refuse_eof():
    """SOCKS framing never treats partial reads as a complete request."""
    connection = Mock()
    connection.recv.side_effect = [b"a", b"b"]
    assert egress._read_exact(connection, 2) == b"ab"
    connection.recv.side_effect = [b"a", b""]
    with pytest.raises(OSError, match="Incomplete SOCKS"):
        egress._read_exact(connection, 2)


@pytest.mark.parametrize(
    "route,allowed",
    [
        ("route to: 11.0.0.2\ninterface: en0\nflags: <UP,GATEWAY>\n", True),
        ("route to: 11.0.0.2\ninterface: lo0\nflags: <UP,LOCAL>\n", False),
        ("route to: 11.0.0.2\ninterface: en0\nflags: <REJECT>\n", False),
        ("route to: 11.0.0.3\ninterface: en0\nflags: <UP>\n", False),
        ("route to: 11.0.0.2\ninterface: en0\ninterface: en1\nflags: <UP>\n", False),
    ],
)
def test_route_check_requires_one_external_kernel_route(monkeypatch, route, allowed):
    """Only an unambiguous, live nonlocal IPv4 route can support a connection."""
    monkeypatch.setattr(
        egress.os, "lstat", lambda _path: SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | 0o555)
    )
    monkeypatch.setattr(
        egress.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(stdout=route.encode("ascii"))),
    )
    assert egress._external_route(ipaddress.IPv4Address("11.0.0.2")) is allowed


def test_route_check_ignores_non_field_lines(monkeypatch):
    """Route statistics do not affect the checked destination and interface."""
    monkeypatch.setattr(
        egress.os,
        "lstat",
        lambda _path: SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | 0o555),
    )
    monkeypatch.setattr(
        egress.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=b"route to: 11.0.0.2\ninterface: en0\nflags: <UP>\nno colon here\n"
        ),
    )
    assert egress._external_route(ipaddress.IPv4Address("11.0.0.2"))


@pytest.mark.parametrize("case", ["ownership", "large", "encoding", "route-error"])
def test_route_check_fails_closed_on_untrusted_or_unreadable_routes(monkeypatch, case):
    """A route probe with invalid provenance or output cannot authorize transport."""
    mode = stat.S_IFREG | (0o666 if case == "ownership" else 0o555)
    monkeypatch.setattr(egress.os, "lstat", lambda _path: SimpleNamespace(st_uid=0, st_mode=mode))
    output = b"x" * (egress.MAX_ROUTE_BYTES + 1) if case == "large" else b"\xff"
    run = Mock(return_value=SimpleNamespace(stdout=output))
    if case == "route-error":
        run.side_effect = OSError("route unavailable")
    monkeypatch.setattr(egress.subprocess, "run", run)
    assert not egress._external_route(ipaddress.IPv4Address("11.0.0.2"))


@pytest.mark.parametrize(
    "case", ["dns", "private", "ipv6", "route", "socket", "connect", "revoked"]
)
def test_broker_connects_only_to_checked_numeric_public_routes(broker, monkeypatch, case):
    """DNS, route, socket and policy failures cannot return a reusable upstream."""
    owner, _listener, _authority, _clock, _thread = broker
    if case == "dns":
        monkeypatch.setattr(egress.socket, "getaddrinfo", Mock(side_effect=OSError("dns")))
    else:
        family = socket.AF_INET6 if case == "ipv6" else socket.AF_INET
        address = "192.168.1.2" if case == "private" else "11.0.0.2"
        monkeypatch.setattr(
            egress.socket,
            "getaddrinfo",
            Mock(return_value=[(family, socket.SOCK_STREAM, 6, "", (address, 80))]),
        )
    monkeypatch.setattr(egress, "_external_route", lambda _address: case != "route")
    upstream = Mock()
    if case == "socket":
        monkeypatch.setattr(egress.socket, "socket", Mock(side_effect=OSError("socket")))
    else:
        monkeypatch.setattr(egress.socket, "socket", Mock(return_value=upstream))
    if case == "connect":
        upstream.connect.side_effect = OSError("refused")
    if case == "revoked":
        monkeypatch.setattr(owner, "allow_new", Mock(side_effect=[True, True, False]))
    assert owner._connect(HTTP_ORIGIN) is None
    owner.__exit__(None, None, None)


def test_broker_connects_to_first_allowed_answer(broker, monkeypatch):
    """The broker ignores forbidden answers and pins the selected numeric address."""
    owner, _listener, _authority, _clock, _thread = broker
    monkeypatch.setattr(
        egress.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("11.0.0.2", 80)),
        ],
    )
    monkeypatch.setattr(egress, "_external_route", lambda _address: True)
    upstream = Mock()
    monkeypatch.setattr(egress.socket, "socket", Mock(return_value=upstream))
    assert owner._connect(HTTP_ORIGIN) is upstream
    upstream.connect.assert_called_once_with(("11.0.0.2", 80))
    owner._release(upstream)
    owner.__exit__(None, None, None)


@pytest.mark.parametrize("case", ["inactive", "invalid-answer", "post-dns", "capacity"])
def test_broker_refuses_stale_or_invalid_numeric_admission(broker, monkeypatch, case):
    """A stale gate or malformed answer cannot turn an origin into a transport."""
    owner, _listener, _authority, _clock, _thread = broker
    if case == "inactive":
        owner.pause()
        assert owner._connect(HTTP_ORIGIN) is None
        owner.__exit__(None, None, None)
        return
    address = "not-an-ip" if case == "invalid-answer" else "11.0.0.2"
    monkeypatch.setattr(
        egress.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 80))],
    )
    monkeypatch.setattr(egress, "_external_route", lambda _address: True)
    upstream = Mock()
    monkeypatch.setattr(egress.socket, "socket", Mock(return_value=upstream))
    if case == "post-dns":
        monkeypatch.setattr(owner, "allow_new", Mock(side_effect=[True, False]))
    if case == "capacity":
        monkeypatch.setattr(owner, "_track", Mock(return_value=False))
    assert owner._connect(HTTP_ORIGIN) is None
    owner.__exit__(None, None, None)


def test_broker_refuses_new_connections_when_saturated(broker):
    """A full attempt cannot add another client or upstream descriptor."""
    owner, *_ = broker
    owner._sockets = {Mock() for _ in range(egress.MAX_CLIENTS * 2)}
    assert not owner._track(Mock())
    owner._sockets.clear()
    owner.__exit__(None, None, None)


def test_http_handler_forwards_only_one_checked_request(monkeypatch):
    """One valid plaintext request is rewritten with a close boundary."""
    handler = object.__new__(egress.SocksHttpHandler)
    handler.origin = HTTP_ORIGIN
    handler.path = "/item?x=1"
    handler.command = "POST"
    handler.headers = Message()
    handler.headers.add_header("Host", "example.com")
    handler.headers.add_header("Content-Length", "4")
    handler.headers.add_header("X-Example", "ok")
    handler.rfile = BytesIO(b"body")
    handler.upstream = Mock()
    handler.upstream.recv.side_effect = [b"HTTP/1.1 200 OK\r\n\r\nreply", b""]
    handler.connection = Mock()
    handler.server = Mock()
    handler.server.active.return_value = True
    monkeypatch.setattr(egress.select, "select", lambda *_args: ([handler.upstream], [], []))
    handler._forward()
    sent = b"".join(call.args[0] for call in handler.upstream.sendall.call_args_list)
    assert sent.startswith(b"POST /item?x=1 HTTP/1.1\r\n")
    assert b"Host: example.com\r\n" in sent
    assert b"Connection: close\r\n\r\n" in sent
    assert b"body" in sent
    handler.connection.sendall.assert_called_once_with(b"HTTP/1.1 200 OK\r\n\r\nreply")
    assert handler.close_connection


@pytest.mark.parametrize(
    "case",
    [
        "path",
        "authority",
        "duplicate",
        "encoding",
        "length",
        "post",
        "transfer",
        "upgrade",
        "expect",
        "method",
    ],
)
def test_http_handler_rejects_ambiguous_visible_requests(case):
    """Malformed Host, unsupported framing and non-HTTP paths never reach upstream."""
    handler = object.__new__(egress.SocksHttpHandler)
    handler.origin = HTTP_ORIGIN
    handler.path = "/item"
    handler.command = "GET"
    handler.headers = Message()
    handler.headers.add_header("Host", "example.com")
    handler.upstream = Mock()
    handler.server = Mock()
    handler.send_error = Mock()
    if case == "path":
        handler.path = "http://example.com/item"
    elif case == "authority":
        handler.headers.replace_header("Host", "evil.example")
    elif case == "duplicate":
        handler.headers.add_header("Host", "example.com")
    elif case == "encoding":
        handler.headers.add_header("Content-Length", "1x")
    elif case == "length":
        handler.headers.add_header("Content-Length", str(egress.MAX_BODY_BYTES + 1))
    elif case == "post":
        handler.command = "POST"
    elif case == "transfer":
        handler.headers.add_header("Transfer-Encoding", "chunked")
    elif case == "upgrade":
        handler.headers.add_header("Upgrade", "websocket")
    elif case == "expect":
        handler.headers.add_header("Expect", "100-continue")
    else:
        handler.command = "PUT"
    handler._forward()
    handler.send_error.assert_called_once_with(403)
    handler.upstream.sendall.assert_not_called()


def test_http_handler_methods_delegate_and_hide_logs(monkeypatch):
    """Supported verbs use one validator and routine logs omit request content."""
    handler = object.__new__(egress.SocksHttpHandler)
    forward = Mock()
    monkeypatch.setattr(handler, "_forward", forward)
    handler.do_GET()
    handler.do_HEAD()
    handler.do_POST()
    handler.log_message("secret %s", "token")
    assert forward.call_count == 3


def test_http_handler_initializes_one_checked_stream(monkeypatch):
    """The parser binds the approved origin before inherited HTTP dispatch."""
    inherited = Mock()
    monkeypatch.setattr(egress.BaseHTTPRequestHandler, "__init__", inherited)
    request, upstream, server = Mock(), Mock(), Mock()
    handler = egress.SocksHttpHandler(request, ("localhost", 0), server, HTTP_ORIGIN, upstream)
    assert handler.origin == HTTP_ORIGIN and handler.upstream is upstream
    inherited.assert_called_once_with(request, ("localhost", 0), server)


def test_http_handler_stops_when_client_body_is_short(monkeypatch):
    """An incomplete POST is never followed by response forwarding."""
    handler = object.__new__(egress.SocksHttpHandler)
    handler.origin = HTTP_ORIGIN
    handler.path = "/"
    handler.command = "POST"
    handler.headers = Message()
    handler.headers.add_header("Host", "example.com")
    handler.headers.add_header("Content-Length", "4")
    handler.rfile = BytesIO(b"")
    handler.upstream = Mock()
    handler.server = Mock()
    handler.server.active.return_value = True
    handler._forward()
    assert not any(call.args[0] == b"body" for call in handler.upstream.sendall.call_args_list)


def test_http_handler_stops_when_authority_ends_during_body():
    """A policy transition prevents forwarding the rest of a POST body."""
    handler = object.__new__(egress.SocksHttpHandler)
    handler.origin = HTTP_ORIGIN
    handler.path = "/"
    handler.command = "POST"
    handler.headers = Message()
    handler.headers.add_header("Host", "example.com")
    handler.headers.add_header("Content-Length", "4")
    handler.rfile = BytesIO(b"body")
    handler.upstream = Mock()
    handler.server = Mock()
    handler.server.active.return_value = False
    handler._forward()
    assert b"body" not in b"".join(call.args[0] for call in handler.upstream.sendall.call_args_list)


def test_http_handler_waits_for_response_and_checks_authority(monkeypatch):
    """An idle upstream is polled without allowing an unbounded retained tunnel."""
    handler = object.__new__(egress.SocksHttpHandler)
    handler.origin = HTTP_ORIGIN
    handler.path = "/"
    handler.command = "GET"
    handler.headers = Message()
    handler.headers.add_header("Host", "example.com")
    handler.rfile = BytesIO()
    handler.upstream = Mock()
    handler.upstream.recv.return_value = b""
    handler.connection = Mock()
    handler.server = Mock()
    handler.server.active.side_effect = [True, False]
    monkeypatch.setattr(egress.select, "select", lambda *_args: ([], [], []))
    handler._forward()
    handler.upstream.recv.assert_not_called()


def test_http_handler_forwards_head_without_body(monkeypatch):
    """A HEAD request omits body framing and stops when the server closes."""
    handler = object.__new__(egress.SocksHttpHandler)
    handler.origin = HTTP_ORIGIN
    handler.path = "/"
    handler.command = "HEAD"
    handler.headers = Message()
    handler.headers.add_header("Host", "example.com")
    handler.rfile = BytesIO()
    handler.upstream = Mock()
    handler.upstream.recv.return_value = b""
    handler.connection = Mock()
    handler.server = Mock()
    handler.server.active.return_value = True
    monkeypatch.setattr(egress.select, "select", lambda *_args: ([handler.upstream], [], []))
    handler._forward()
    sent = b"".join(call.args[0] for call in handler.upstream.sendall.call_args_list)
    assert b"Content-Length" not in sent
    handler.connection.sendall.assert_not_called()


def test_http_handler_stops_on_forwarding_error(monkeypatch):
    """A transport failure cannot fall through to another HTTP request."""
    handler = object.__new__(egress.SocksHttpHandler)
    handler.origin = HTTP_ORIGIN
    handler.path = "/"
    handler.command = "GET"
    handler.headers = Message()
    handler.headers.add_header("Host", "example.com")
    handler.upstream = Mock()
    handler.upstream.sendall.side_effect = OSError("closed")
    handler.server = Mock()
    handler._forward()
    assert handler.close_connection


def test_socks_server_refuses_invalid_handshake(broker):
    """A client without SOCKS5 no-auth cannot reach an upstream transport."""
    owner, *_ = broker
    client = Mock()
    client.recv.side_effect = [b"\x05\x01", b"\x02"]
    owner._serve(client)
    client.sendall.assert_called_once_with(b"\x05\xff")
    owner.__exit__(None, None, None)


def test_socks_server_does_not_forward_ungranted_target(broker, monkeypatch):
    """A valid greeting still refuses an unauthorized destination."""
    owner, *_ = broker
    client = Mock()
    client.recv.side_effect = [
        b"\x05\x01",
        b"\x00",
        b"\x05\x01\x00\x03",
        b"\x0b",
        b"example.com",
        b"\x00\x50",
    ]
    monkeypatch.setattr(owner, "_connect", lambda _origin: None)
    owner._serve(client)
    assert client.sendall.call_args_list[-1].args[0][1] == 2
    owner.__exit__(None, None, None)


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_socks_server_dispatches_checked_http_or_https(broker, monkeypatch, scheme):
    """The protocol handler receives only a pinned approved upstream."""
    owner, *_ = broker
    client = Mock()
    port = b"\x00\x50" if scheme == "http" else b"\x01\xbb"
    client.recv.side_effect = [
        b"\x05\x01",
        b"\x00",
        b"\x05\x01\x00\x03",
        b"\x0b",
        b"example.com",
        port,
    ]
    upstream = Mock()
    monkeypatch.setattr(owner, "_connect", lambda _origin: upstream)
    http = Mock()
    relay = Mock()
    monkeypatch.setattr(egress, "SocksHttpHandler", http)
    monkeypatch.setattr(owner, "_relay", relay)
    owner._serve(client)
    if scheme == "http":
        http.assert_called_once()
    else:
        relay.assert_called_once_with(client, upstream)
    owner.__exit__(None, None, None)


def test_socks_server_rejects_invalid_command_and_truncated_frame(broker):
    """Malformed SOCKS command frames close without any network authority."""
    owner, *_ = broker
    client = Mock()
    client.recv.side_effect = [b"\x05\x01", b"\x00", b"\x05\x02\x00\x03"]
    owner._serve(client)
    client.sendall.assert_called_once_with(b"\x05\x00")
    client.recv.side_effect = [b"\x05\x01", b""]
    owner._serve(client)
    owner.__exit__(None, None, None)


def test_socks_server_rejects_unknown_destination_type(broker):
    """An unsupported SOCKS address type cannot reach a checked upstream."""
    owner, *_ = broker
    client = Mock()
    client.recv.side_effect = [b"\x05\x01", b"\x00", b"\x05\x01\x00\x04"]
    owner._serve(client)
    assert client.sendall.call_args_list[-1].args[0][1] == 2
    owner.__exit__(None, None, None)


def test_https_relay_forwards_both_directions_and_stops_on_error(broker, monkeypatch):
    """Opaque data crosses only the active tunnel's two owned descriptors."""
    owner, *_ = broker
    client, upstream = Mock(), Mock()
    client.recv.return_value = b"request"
    upstream.recv.return_value = b"reply"
    monkeypatch.setattr(owner, "active", Mock(side_effect=[True, False]))
    monkeypatch.setattr(egress.select, "select", lambda *_args: ([client, upstream], [], []))
    owner._relay(client, upstream)
    upstream.sendall.assert_called_once_with(b"request")
    client.sendall.assert_called_once_with(b"reply")
    monkeypatch.setattr(owner, "active", Mock(return_value=True))
    monkeypatch.setattr(egress.select, "select", Mock(side_effect=OSError("closed")))
    owner._relay(client, upstream)
    owner.__exit__(None, None, None)


def test_https_relay_stops_at_eof(broker, monkeypatch):
    """An opaque tunnel exits when either covered side closes."""
    owner, *_ = broker
    client, upstream = Mock(), Mock()
    client.recv.return_value = b""
    monkeypatch.setattr(egress.select, "select", lambda *_args: ([client], [], []))
    owner._relay(client, upstream)
    upstream.sendall.assert_not_called()
    owner.__exit__(None, None, None)
