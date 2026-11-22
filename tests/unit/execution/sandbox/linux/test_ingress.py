"""Verify sealed namespace ingress handoff and relay ownership without native effects."""

import array
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, mock_open

import pytest

from loop.execution.sandbox.linux import ingress


@pytest.fixture
def worker(monkeypatch):
    """Replace process, socket and thread boundaries while exercising the public worker entry."""
    channel = MagicMock()
    channel.__enter__.return_value = channel
    channel.recv.return_value = b"G"
    client = MagicMock(family=socket.AF_INET, type=socket.SOCK_STREAM)
    client.getsockname.return_value = ("127.0.0.1", 8080)
    client.getpeername.return_value = ("127.0.0.1", 30000)
    upstream = MagicMock()
    upstream.__enter__.return_value = upstream
    client.recv.side_effect = [b"request", b""]
    upstream.recv.side_effect = [b"response", b""]
    monkeypatch.setattr(ingress.socket, "MSG_CMSG_CLOEXEC", 0x40000000, raising=False)
    channel.recvmsg.side_effect = [
        (
            b"C\x00\x00\x00\x01",
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [10]).tobytes())],
            0x40000000,
            None,
        ),
        (b"", [], 0, None),
    ]
    factory = Mock(side_effect=lambda **kwargs: channel if kwargs["fileno"] == 9 else client)
    monkeypatch.setattr(ingress.socket, "socket", factory)
    connection = Mock(return_value=upstream)
    monkeypatch.setattr(ingress.socket, "create_connection", connection)
    selection = Mock(side_effect=lambda sockets, *_: ([sockets[0]], [], []))
    monkeypatch.setattr(ingress.select, "select", selection)
    libc = Mock()
    libc.prctl.return_value = 0
    monkeypatch.setattr(ingress.ctypes, "CDLL", Mock(return_value=libc))
    fork = Mock(return_value=0)
    monkeypatch.setattr(ingress.os, "fork", fork)
    monkeypatch.setattr(ingress.os, "dup2", Mock())
    close = Mock()
    monkeypatch.setattr(ingress.os, "close", close)
    terminate = Mock(side_effect=SystemExit(0))
    monkeypatch.setattr(ingress.os, "_exit", terminate)
    monkeypatch.setattr(ingress, "open", mock_open(), raising=False)

    def thread(*, target, args, daemon):
        """Execute owned callbacks deterministically without scheduling a real thread."""
        assert daemon
        return SimpleNamespace(start=lambda: target(*args))

    threads = Mock(side_effect=thread)
    monkeypatch.setattr(ingress.threading, "Thread", threads)
    return SimpleNamespace(
        channel=channel,
        client=client,
        upstream=upstream,
        connection=connection,
        selection=selection,
        libc=libc,
        fork=fork,
        close=close,
        threads=threads,
        terminate=terminate,
    )


@pytest.mark.parametrize("failure", ["none", "gate", "set_identity", "read_identity", "fork"])
def test_worker_seals_before_fork_and_closes_payload_descriptor(worker, failure):
    """Only an activated nondumpable worker is forked, and the future payload loses its channel."""
    worker.fork.return_value = 21
    if failure == "gate":
        worker.channel.recv.return_value = b"X"
    elif failure == "set_identity":
        worker.libc.prctl.side_effect = [1]
    elif failure == "read_identity":
        worker.libc.prctl.side_effect = [0, 1]
    elif failure == "fork":
        worker.fork.side_effect = OSError("fork refused")
    if failure == "none":
        ingress.start_namespace_ingress(9, 8080)
        worker.fork.assert_called_once()
        assert worker.libc.prctl.call_args_list[0].args == (4, 0, 0, 0, 0)
    else:
        with pytest.raises(OSError):
            ingress.start_namespace_ingress(9, 8080)
    worker.channel.__exit__.assert_called_once()
    worker.threads.assert_not_called()


@pytest.mark.parametrize(
    "direction", ["client", "upstream", "idle", "connect_error", "relay_error", "ack_error"]
)
def test_worker_relays_only_to_fixed_private_port_and_releases_handoff(worker, direction):
    """Close-on-exec is valid; accepted bytes never select the worker's upstream destination."""
    if direction == "upstream":
        worker.selection.side_effect = lambda sockets, *_: ([sockets[-1]], [], [])
    elif direction == "idle":
        worker.selection.side_effect = [
            ([], [], []),
            ([worker.channel], [], []),
            ([worker.client], [], []),
            ([worker.client], [], []),
            ([worker.channel], [], []),
        ]
    elif direction == "connect_error":
        worker.connection.side_effect = OSError("private service unavailable")
    elif direction == "relay_error":
        worker.client.recv.side_effect = OSError("closed")
    elif direction == "ack_error":
        worker.channel.send.side_effect = OSError("revoked")
    with pytest.raises(SystemExit) as ending:
        ingress.start_namespace_ingress(9, 8080)
    assert ending.value.code == 0
    worker.connection.assert_called_once_with(("127.0.0.1", 8080), timeout=0.5)
    if direction == "upstream":
        worker.client.sendall.assert_called_once_with(b"response")
    elif direction in {"client", "idle", "ack_error"}:
        worker.upstream.sendall.assert_called_once_with(b"request")
    worker.client.close.assert_called_once()
    worker.channel.send.assert_called_once_with(b"D\x00\x00\x00\x01")


@pytest.mark.parametrize(
    "case",
    [
        "short",
        "marker",
        "truncated",
        "two_fds",
        "unknown_control",
        "receive_error",
        "family",
        "type",
        "local",
        "peer",
        "peer_error",
        "thread_error",
        "shutdown_error",
        "capacity",
        "stopped",
    ],
)
def test_worker_rejects_malformed_handoff_and_settles_retained_clients(worker, monkeypatch, case):
    """Malformed authority closes all received handles; control loss closes active clients."""
    data = b"C\x00\x00\x00\x01"
    handles = [10]
    flags = 0
    level = socket.SOL_SOCKET
    if case == "short":
        data = b"C"
    elif case == "marker":
        data = b"X0001"
    elif case == "truncated":
        flags = socket.MSG_CTRUNC
    elif case == "two_fds":
        handles = [10, 11]
    elif case == "unknown_control":
        level = -1
    elif case == "family":
        worker.client.family = socket.AF_UNIX
    elif case == "type":
        worker.client.type = socket.SOCK_DGRAM
    elif case == "local":
        worker.client.getsockname.return_value = ("0.0.0.0", 8080)
    elif case == "peer":
        worker.client.getpeername.return_value = ("10.0.0.1", 30000)
    elif case == "peer_error":
        worker.client.getpeername.side_effect = OSError("closed")
    elif case == "capacity":
        monkeypatch.setattr(ingress, "_MAX_CLIENTS", 0)
    controls = [(level, socket.SCM_RIGHTS, array.array("i", handles).tobytes())]
    worker.channel.recvmsg.side_effect = [(data, controls, flags, None), (b"", [], 0, None)]
    if case == "receive_error":
        worker.channel.recvmsg.side_effect = OSError("closed")
    if case in {"thread_error", "shutdown_error", "stopped"}:
        worker.threads.side_effect = None
        worker.threads.return_value = Mock()
        if case == "thread_error":
            worker.threads.return_value.start.side_effect = RuntimeError("no thread")
        if case == "shutdown_error":
            worker.client.shutdown.side_effect = OSError("already closed")
        if case == "stopped":

            def exit_after_stop(_status):
                """Deliver the deferred worker only after the channel's EOF cancels it."""
                arguments = worker.threads.call_args.kwargs
                arguments["target"](*arguments["args"])
                raise SystemExit(0)

            worker.terminate.side_effect = exit_after_stop
    with pytest.raises(SystemExit):
        ingress.start_namespace_ingress(9, 8080)
    if case in {"short", "marker", "truncated", "two_fds"}:
        assert [call.args[0] for call in worker.close.call_args_list] == handles
    if case in {"thread_error", "shutdown_error", "stopped"}:
        worker.client.shutdown.assert_called_once_with(socket.SHUT_RDWR)
