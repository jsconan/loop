"""Verify bounded ordinary transport and original listener authority before content transmission."""

import socket
import stat
import struct
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import transport as linux_transport
from loop.execution.sandbox.linux.admission import GenerationAdmission


@pytest.fixture
def transport(monkeypatch):
    """Isolate native sockets and protected file authority without launching processes."""
    config = GenerationAdmission(
        "test", 1000, 1000, Path("/native/loop_receiver.py"), Mock(), Mock()
    )
    config.slice_unit.path = Path("/run/systemd/system/loopgen_test.slice")
    config.reservation.open_verified.return_value = nullcontext(4)
    config.slice_unit.open_verified.side_effect = lambda **kwargs: nullcontext(1)
    files = {name: Mock() for name in ("receiver", "socket_unit", "service_unit")}
    files["receiver"].path = config.receiver
    for fd, name in enumerate(files, 2):
        files[name].open_verified.side_effect = lambda fd=fd, **kwargs: nullcontext(fd)
    sources = config.unit_sources()
    contents = {
        1: sources["loopgen_test.slice"].encode(),
        3: sources["loopgen_test.socket"].encode(),
        4: sources["loopgen_test@.service"].encode(),
    }
    monkeypatch.setattr(linux_transport, "generation_limits", Mock(return_value=(29, 14680064)))
    monkeypatch.setattr(linux_transport.os, "pread", Mock(side_effect=lambda fd, *_: contents[fd]))
    for name in ("getuid", "geteuid", "getgid", "getegid"):
        monkeypatch.setattr(linux_transport.os, name, lambda: 1000)
    monkeypatch.setattr(linux_transport.time, "monotonic", lambda: 10)
    monkeypatch.setattr(linux_transport, "protected_path", Mock(side_effect=lambda p, **_: p))
    details = SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_mode=stat.S_IFSOCK | 0o600,
        st_uid=1000,
        st_gid=1000,
        st_nlink=1,
        st_mtime_ns=10,
        st_ctime_ns=10,
        st_atime=10,
        st_atime_ns=10,
    )
    metadata = Mock(return_value=details)
    monkeypatch.setattr(Path, "lstat", metadata)
    connection = Mock()
    connection.getsockopt.return_value = struct.pack("3i", 1, 0, 0)
    connection.recv.side_effect = [b'{"result":', b' "UNAVAILABLE"}\n', b""]
    factory = Mock()
    factory.return_value.__enter__ = Mock(return_value=connection)
    factory.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(linux_transport.socket, "socket", factory)
    return config, files, connection, metadata, details, contents


def test_transport_preserves_frame_halfclose_and_bounded_response(transport):
    """An exact authorized byte frame is sent once only after authenticated installed authority."""
    config, files, connection, *_ = transport
    payload = b"private\x00opaque request"
    assert (
        linux_transport.exchange_request(config, files, payload, 20)
        == b'{"result": "UNAVAILABLE"}\n'
    )
    connection.connect.assert_called_once_with("/run/loopgen_test.socket")
    connection.sendall.assert_called_once_with(
        b"LOOP\x01" + struct.pack("!I", len(payload)) + payload
    )
    connection.shutdown.assert_called_once_with(socket.SHUT_WR)
    assert [call.args for call in connection.settimeout.call_args_list] == [
        (10,),
        (10,),
        (15,),
        (15,),
        (15,),
    ]
    assert connection.recv.call_count == 3


@pytest.mark.parametrize("payload", [None, "text", b"", b"x" * 65537])
def test_transport_invalid_payload_never_connects(transport, payload):
    """Invalid or oversized content cannot reach the native listener."""
    config, files, connection, *_ = transport
    with pytest.raises(ValueError, match="bounded installed request"):
        linux_transport.exchange_request(config, files, payload, 20)
    connection.connect.assert_not_called()


@pytest.mark.parametrize("name", ["getuid", "geteuid", "getgid", "getegid"])
def test_transport_wrong_identity_never_connects(transport, monkeypatch, name):
    """Every real/effective UID/GID must match the installer-selected ordinary identity."""
    config, files, connection, *_ = transport
    monkeypatch.setattr(linux_transport.os, name, lambda: 0)
    with pytest.raises(ValueError, match="fixed ordinary generation identity"):
        linux_transport.exchange_request(config, files, b"secret", 20)
    connection.connect.assert_not_called()


@pytest.mark.parametrize("change", ["receiver", "slice", "socket", "service", "parent"])
def test_transport_changed_installed_authority_never_connects(transport, change):
    """Changed original receiver, exact unit bytes or canonical parent refuse before connection."""
    config, files, connection, _, _, contents = transport
    if change == "receiver":
        files["receiver"].path = Path("/other")
    elif change == "parent":
        linux_transport.protected_path.return_value = Path("/other")
        linux_transport.protected_path.side_effect = None
    else:
        contents[{"slice": 1, "socket": 3, "service": 4}[change]] = b"changed"
    with pytest.raises(ValueError, match="authority|aliased"):
        linux_transport.exchange_request(config, files, b"secret", 20)
    connection.connect.assert_not_called()


@pytest.mark.parametrize("change", ["type", "uid", "gid", "mode"])
def test_transport_invalid_socket_authority_never_connects(transport, change):
    """Only a fixed owner/group 0600 socket under protected ancestry may receive content."""
    config, files, connection, _, details, _ = transport
    if change == "type":
        details.st_mode = stat.S_IFLNK | 0o600
    elif change == "uid":
        details.st_uid = 0
    elif change == "gid":
        details.st_gid = 0
    else:
        details.st_mode |= 0o020
    with pytest.raises(ValueError, match="listener authority"):
        linux_transport.exchange_request(config, files, b"secret", 20)
    connection.connect.assert_not_called()


@pytest.mark.parametrize("peer", [(0, 0, 0), (2, 1000, 0), (2, 0, 1000)])
def test_transport_unprotected_peer_never_receives_content(transport, peer):
    """An ordinary impostor or invalid peer cannot receive an authorized frame."""
    config, files, connection, *_ = transport
    connection.getsockopt.return_value = struct.pack("3i", *peer)
    with pytest.raises(ValueError, match="protected system listener"):
        linux_transport.exchange_request(config, files, b"secret", 20)
    connection.sendall.assert_not_called()


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize(
    "field",
    ["st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_mtime_ns", "st_ctime_ns"],
)
def test_transport_socket_replacement_refuses_without_retry(transport, late, field):
    """Replacing the original socket inode before send or during exchange prevents success."""
    config, files, connection, metadata, details, _ = transport
    changed = SimpleNamespace(**vars(details))
    setattr(changed, field, getattr(changed, field) + 1)
    metadata.side_effect = [details, details if late else changed, changed]
    with pytest.raises(ValueError, match="socket changed"):
        linux_transport.exchange_request(config, files, b"secret", 20)
    assert connection.sendall.call_count == late
    assert connection.connect.call_count == 1


def test_transport_socket_access_time_changes_preserve_authority(transport):
    """Normal connection and response access-time changes cannot mimic authority replacement."""
    config, files, _, metadata, details, _ = transport
    connected = SimpleNamespace(**vars(details))
    connected.st_atime = 20
    connected.st_atime_ns = 20_000_000_000
    responded = SimpleNamespace(**vars(connected))
    responded.st_atime = 21
    responded.st_atime_ns = 21_000_000_000
    metadata.side_effect = [details, connected, responded]
    assert (
        linux_transport.exchange_request(config, files, b"request", 20)
        == b'{"result": "UNAVAILABLE"}\n'
    )


@pytest.mark.parametrize("blocks", [[b""], [b"x" * 65536, b"y"], [b"x" * 65536, b""]])
def test_transport_response_eof_and_exact_capacity(transport, blocks, monkeypatch):
    """Empty or overflowing responses refuse while an exact bounded response remains accepted."""
    config, files, connection, *_ = transport
    monkeypatch.setattr(linux_transport, "MAX_NATIVE_RESULT_BYTES", 65536)
    connection.recv.side_effect = blocks
    if blocks == [b"x" * 65536, b""]:
        assert linux_transport.exchange_request(config, files, b"request", 20) == b"x" * 65536
    else:
        with pytest.raises(ValueError, match="no response|exceeds"):
            linux_transport.exchange_request(config, files, b"request", 20)
    assert connection.sendall.call_count == 1


@pytest.mark.parametrize("operation", ["connect", "sendall", "shutdown", "recv"])
@pytest.mark.parametrize("category", [OSError, KeyboardInterrupt])
def test_transport_native_failure_closes_without_retry(transport, operation, category):
    """Native I/O failure or cancellation closes without interpreting effects or repeating."""
    config, files, connection, *_ = transport
    getattr(connection, operation).side_effect = category("native transport failed")
    with pytest.raises(category, match="native transport failed"):
        linux_transport.exchange_request(config, files, b"request", 20)
    linux_transport.socket.socket.return_value.__exit__.assert_called_once()
    assert connection.connect.call_count == 1
    assert connection.sendall.call_count <= 1


def test_transport_response_trickle_cannot_reset_deadline(transport, monkeypatch):
    """Each partial response consumes the original budget instead of extending it."""
    config, files, connection, *_ = transport
    clock = [10]
    monkeypatch.setattr(linux_transport.time, "monotonic", lambda: clock[0])

    def receive(size):
        """Return one byte while consuming the remaining original response budget."""
        clock[0] += 5
        return b"x"

    connection.recv.side_effect = receive
    with pytest.raises(TimeoutError, match="transport deadline"):
        linux_transport.exchange_request(config, files, b"request", 20)
    assert connection.recv.call_count == 3


def test_transport_expired_deadline_never_connects(transport):
    """An already expired request cannot start native transport preparation."""
    config, files, connection, *_ = transport
    with pytest.raises(TimeoutError, match="transport deadline"):
        linux_transport.exchange_request(config, files, b"request", 10)
    connection.connect.assert_not_called()


@pytest.mark.parametrize("deadline", [float("inf"), float("-inf"), float("nan")])
def test_transport_nonfinite_deadline_never_connects(transport, deadline):
    """Every exchange requires a finite deadline before native preparation or connection."""
    config, files, connection, *_ = transport
    with pytest.raises(ValueError, match="finite deadline"):
        linux_transport.exchange_request(config, files, b"request", deadline)
    connection.connect.assert_not_called()


def test_transport_late_installed_drift_never_returns_response(transport):
    """Changed installed authority after response EOF prevents a successful transport result."""
    config, files, connection, _, _, contents = transport

    def receive(size):
        """Change unit authority only after the frame was transmitted."""
        contents[1] = b"changed"
        connection.recv.side_effect = None
        connection.recv.return_value = b""
        return b"response"

    connection.recv.side_effect = receive
    with pytest.raises(ValueError, match="unit differs"):
        linux_transport.exchange_request(config, files, b"request", 20)
    connection.sendall.assert_called_once()


def test_transport_restores_large_attempt_bound_result(transport):
    """A result larger than its request ceiling passes bounded transport without losing Unicode."""
    from loop.execution.sandbox.contracts import CommandProcessResult, SandboxOutcome

    config, files, connection, *_ = transport
    attempt = "1248c892-42dc-4b68-a3fb-30d2a70a93a7"
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=17, stdout="😀" * 65536, stderr="ordinary failure"
    )
    wire = result.to_wire(attempt)
    connection.recv.side_effect = [*(wire[i : i + 4096] for i in range(0, len(wire), 4096)), b""]
    received = linux_transport.exchange_request(config, files, b"request", 20)
    assert CommandProcessResult.from_wire(received, attempt) == result
    assert connection.sendall.call_args.args[0] == b"LOOP\x01" + struct.pack("!I", 7) + b"request"


def test_transport_timeout_capture_uses_fixed_delivery_interval(transport, monkeypatch):
    """An authenticated timeout result can arrive after execution expires within a fixed bound."""
    from loop.execution.sandbox.contracts import CommandProcessResult, SandboxOutcome

    config, files, connection, *_ = transport
    now = [10.0]
    monkeypatch.setattr(linux_transport.time, "monotonic", lambda: now[0])
    attempt = "1248c892-42dc-4b68-a3fb-30d2a70a93a7"
    expected = CommandProcessResult(
        SandboxOutcome.TIMED_OUT,
        stdout="captured-out",
        stderr="captured-err",
        stdout_discarded=None,
        stderr_discarded=None,
        possible_effects=True,
    )

    def receive(_):
        """Consume the original execution deadline before receiving bounded timeout output."""
        now[0] = 21.0
        connection.recv.side_effect = [b""]
        return expected.to_wire(attempt)

    connection.recv.side_effect = receive
    result = linux_transport.exchange_request(config, files, b"request", 20)
    assert CommandProcessResult.from_wire(result, attempt) == expected
    assert connection.settimeout.call_args.args == (4.0,)
    connection.sendall.assert_called_once()
