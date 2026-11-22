"""Verify synchronization protocol, admission bounds, and broker lifecycle."""

import errno
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from loop.execution.sandbox.macos import synchronization
from loop.execution.sandbox.macos.synchronization import SynchronizationBroker

_HANDLE = "a" * 32


def frame(*values):
    """Encode one complete request in its wire form."""
    return json.dumps(values).encode("ascii") + b"\n"


@pytest.fixture
def transport(monkeypatch, tmp_path):
    """Drive the public broker through fake socket and worker boundaries."""
    listener, selector, thread = MagicMock(), MagicMock(), MagicMock()
    listener.__enter__.return_value = listener
    selector.__enter__.return_value = selector
    thread.is_alive.return_value = False
    stopping = Event()
    monkeypatch.setattr(synchronization.threading, "Event", lambda: stopping)
    monkeypatch.setattr(synchronization.socket, "socket", lambda *_: listener)
    monkeypatch.setattr(synchronization.selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(synchronization.secrets, "token_hex", lambda _: _HANDLE)

    def start(target, **kwargs):
        """Execute each queued readiness event synchronously."""
        thread.start.side_effect = target
        return thread

    monkeypatch.setattr(synchronization.threading, "Thread", start)

    def run(*events):
        """Supply readiness events and stop once the script is exhausted."""
        queue = iter(events)

        def select(_timeout):
            """Yield visible readiness or finish the worker."""
            event = next(queue, None)
            if event is None:
                stopping.set()
                return []
            if isinstance(event, OSError):
                raise event
            if callable(event) and not isinstance(event, MagicMock):
                stopping.set()
                event = event()
            return [(SimpleNamespace(fileobj=event), 1)]

        selector.select.side_effect = select
        return SynchronizationBroker(Path(tmp_path) / "owned.sock")

    return run, listener, selector, thread


def responses(connection):
    """Decode all replies visible on the fake connection."""
    return [tuple(json.loads(call.args[0])) for call in connection.sendall.call_args_list]


def test_named_counter_handles_and_unlinked_lifetime(transport):
    """Names admit later peers while unlinked handles retain connection-local lifetime."""
    run, listener, _, _ = transport
    owner, peer = MagicMock(), MagicMock()
    listener.accept.side_effect = [(owner, None), (peer, None)]
    owner.recv.side_effect = [
        frame("create", "lock", 1, 2),
        frame("create", "lock", 1, 2),
        frame("acquire", _HANDLE),
        frame("acquire", _HANDLE),
        frame("value", _HANDLE),
        frame("unlink", "lock"),
        frame("detach", _HANDLE),
        frame("value", _HANDLE),
        b"",
    ]
    peer.recv.side_effect = [
        frame("value", _HANDLE),
        frame("open", "lock"),
        frame("open", "lock"),
        frame("release", _HANDLE),
        frame("release", _HANDLE),
        frame("release", _HANDLE),
        frame("open", "lock"),
        frame("detach", _HANDLE),
        frame("value", _HANDLE),
        b"",
    ]
    with run(
        listener,
        owner,
        owner,
        owner,
        owner,
        owner,
        listener,
        *([peer] * 6),
        owner,
        owner,
        owner,
        owner,
        peer,
        peer,
        peer,
        peer,
    ):
        pass
    assert responses(owner) == [
        (0, _HANDLE),
        (errno.EEXIST, None),
        (0, True),
        (0, False),
        (0, 0),
        (0, None),
        (0, True),
        (errno.ENOENT, None),
    ]
    assert responses(peer) == [
        (errno.ENOENT, None),
        (0, _HANDLE),
        (0, _HANDLE),
        (0, True),
        (0, True),
        (errno.EINVAL, None),
        (errno.ENOENT, None),
        (0, True),
        (0, 2),
    ]
    owner.close.assert_called_once()
    peer.close.assert_called_once()


@pytest.mark.parametrize(
    "request_bytes",
    [
        b'{"create":"lock"}\n',
        frame("create"),
        frame(0, "lock"),
        frame("create", 0),
        frame("create", "..", 0, 1),
        frame("create", "lock", True, 1),
        frame("create", "lock", 0, True),
        frame("create", "lock", 0, "1"),
        frame("create", "lock", "0", 1),
        frame("create", "lock", 2, 1),
        frame("create", "lock", 0),
        frame("open", "lock", 0),
        frame("value", "lock"),
        frame("value", _HANDLE, 0),
        b"\xff\n",
    ],
)
def test_invalid_frames_neither_mutate_nor_kill_the_worker(transport, request_bytes):
    """Malformed frames receive a bounded status and the next good request works."""
    run, listener, _, _ = transport
    peer = MagicMock()
    listener.accept.return_value = (peer, None)
    peer.recv.side_effect = [request_bytes, frame("create", "lock", 0, 1)]
    with run(listener, peer, peer):
        pass
    assert responses(peer) == [(errno.EINVAL, None), (0, _HANDLE)]


def test_counter_and_peer_capacity(transport, monkeypatch):
    """A live peer cannot exceed the namespace bounds or starve closure."""
    run, listener, _, _ = transport
    monkeypatch.setattr(synchronization, "_MAX_PEERS", 1)
    monkeypatch.setattr(synchronization, "_MAX_COUNTERS", 1)
    monkeypatch.setattr(synchronization, "_MAX_REFERENCES_PER_COUNTER", 1)
    owner, denied = MagicMock(), MagicMock()
    listener.accept.side_effect = [(owner, None), (denied, None)]
    owner.recv.side_effect = [
        frame("create", "lock", 0, 1),
        frame("unknown", _HANDLE),
        frame("create", "other", 0, 1),
        frame("open", "lock"),
        frame("detach", _HANDLE),
        frame("unlink", "lock"),
        frame("create", "other", 0, 1),
    ]
    with run(listener, listener, *([owner] * 7)):
        pass
    assert responses(owner) == [
        (0, _HANDLE),
        (errno.EINVAL, None),
        (errno.ENOSPC, None),
        (errno.ENOSPC, None),
        (0, True),
        (0, None),
        (0, _HANDLE),
    ]
    denied.close.assert_called_once()


def test_identifier_collision_never_rebinds_an_existing_counter(transport):
    """Handle generation failure leaves an admitted handle bound to its original counter."""
    run, listener, _, _ = transport
    owner = MagicMock()
    listener.accept.return_value = (owner, None)
    owner.recv.side_effect = [
        frame("create", "lock", 1, 1),
        frame("create", "other", 0, 1),
        frame("value", _HANDLE),
    ]
    with run(listener, owner, owner, owner):
        pass
    assert responses(owner) == [(0, _HANDLE), (errno.EAGAIN, None), (0, 1)]


def test_stop_wins_over_pending_readiness(transport):
    """The broker refuses a pending connection once the attempt has ended."""
    run, listener, _, _ = transport
    with run(lambda: listener):
        pass
    listener.accept.assert_not_called()


@pytest.mark.parametrize("disrupted_call", ["accept", "recv", "send"])
def test_temporary_and_disconnected_peers(transport, disrupted_call):
    """Transient readiness and disconnected peers cannot stop the broker."""
    run, listener, _, _ = transport
    peer = MagicMock()
    events = [listener, peer]
    if disrupted_call == "accept":
        listener.accept.side_effect = [BlockingIOError(), (peer, None)]
        peer.recv.return_value = b""
        events.insert(0, listener)
    else:
        listener.accept.return_value = (peer, None)
        if disrupted_call == "recv":
            peer.recv.side_effect = [BlockingIOError(), ConnectionResetError()]
            events.append(peer)
        else:
            peer.recv.return_value = frame("create", "lock", 0, 1)
            peer.sendall.side_effect = BrokenPipeError()
    with run(*events):
        pass
    peer.close.assert_called_once()


@pytest.mark.parametrize("oversize", [False, True])
def test_fragment_and_extra_frame_retires_one_peer(transport, oversize):
    """A partial message is bounded and a disallowed framing pattern closes the peer."""
    run, listener, _, _ = transport
    owner = MagicMock()
    listener.accept.return_value = (owner, None)
    extra = b"*" * 512 if oversize else b"\n\n"
    owner.recv.side_effect = [b'["create"', extra]
    with run(listener, owner, owner):
        pass
    owner.sendall.assert_not_called()
    owner.close.assert_called_once()


@pytest.mark.parametrize("joined", [True, False])
def test_shutdown_reports_worker_failure(transport, joined):
    """Cleanup closes all descriptors and reports an unverified or failed worker."""
    run, listener, selector, thread = transport
    thread.is_alive.return_value = not joined
    reason = "service failed" if joined else "shutdown could not be confirmed"
    with pytest.raises(OSError, match=reason), run(OSError("listener failure")):
        pass
    selector.__exit__.assert_called_once()
    listener.__exit__.assert_called_once()
