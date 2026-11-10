"""Verify trusted mapped payload inputs and fail-closed irreversible handoff."""

import base64
import binascii
import io
import json
import os
import runpy
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.contracts import SandboxRequest
from loop.execution.sandbox.linux import bootstrap as linux_bootstrap
from loop.execution.sandbox.linux.resources import payload as module


@pytest.fixture
def payload(monkeypatch, tmp_path):
    """Isolate mapped native operations while preserving actual trusted envelope restoration."""
    entry = tmp_path / "loop_payload.py"
    entry.write_bytes(b"trusted installed entry")
    monkeypatch.setattr(module, "__file__", str(entry))
    monkeypatch.setattr(module.sys, "path", list(module.sys.path))
    monkeypatch.setattr(
        SandboxRequest,
        "from_wire",
        Mock(return_value=SimpleNamespace(terminal=False, egress_origins=(), local_network=True)),
    )
    record = {
        "path": "/trusted/control",
        "identity": [1, 2, 0o100444, 1, 4, 5, 6],
        "sha256": "a" * 64,
    }
    value = {
        "request": base64.b64encode(b"request").decode(),
        "compiled": base64.b64encode(b"policy").decode(),
        "profile": "payload_profile",
        "namespace": "generation",
        "uid": 1000,
        "tracking": [1, 2, "loopgen_native@instance.service"],
        **{role: record for role in ("apparmor", "bootstrap", "cache_lock", "reservation")},
    }
    monkeypatch.setattr(module.sys, "stdin", Mock(buffer=io.BytesIO(json.dumps(value).encode())))
    monkeypatch.setattr(linux_bootstrap, "enter_bootstrap", Mock())
    monkeypatch.setattr(linux_bootstrap, "execute_payload", Mock())
    monkeypatch.setattr(module, "os", Mock(O_RDONLY=os.O_RDONLY, O_CLOEXEC=os.O_CLOEXEC))
    monkeypatch.setattr(module.os, "open", Mock(return_value=8))
    monkeypatch.setattr(module.os, "dup2", Mock())
    monkeypatch.setattr(module.os, "close", Mock())
    return entry


def test_payload_enters_bound_setup_and_replaces_private_input(payload):
    """Use receiver-prepared controls and redirect command stdin to the null device."""
    module.main()
    SandboxRequest.from_wire.assert_called_once_with(b"request")
    linux_bootstrap.enter_bootstrap.assert_called_once()
    module.os.dup2.assert_called_once_with(8, 0)
    module.os.close.assert_called_once_with(8)
    args = linux_bootstrap.execute_payload.call_args
    assert args.args[0] == SandboxRequest.from_wire.return_value and args.args[4:] == (
        b"policy",
        "payload_profile",
        "generation",
        1000,
    )
    assert args.kwargs["reservation"].path == Path("/trusted/control")


@pytest.mark.parametrize("session_leader", [False, True])
def test_payload_attaches_only_its_receiver_owned_terminal(payload, monkeypatch, session_leader):
    """The inherited slave becomes stdin and the process's own controlling terminal."""
    SandboxRequest.from_wire.return_value = SimpleNamespace(
        terminal=True, egress_origins=(), local_network=True
    )
    module.os.isatty.side_effect = lambda fd: fd in (1, 2)
    module.os.getpid.return_value = 13
    module.os.getsid.return_value = 13 if session_leader else 12
    ioctl = Mock()
    monkeypatch.setattr(module.fcntl, "ioctl", ioctl)

    module.main()

    module.os.dup2.assert_called_once_with(1, 0)
    assert module.os.setsid.call_count == (not session_leader)
    ioctl.assert_called_once_with(0, module.termios.TIOCSCTTY, 0)
    module.os.open.assert_not_called()
    linux_bootstrap.execute_payload.assert_called_once()


@pytest.mark.parametrize("missing", [1, 2])
def test_payload_refuses_missing_receiver_terminal_slave(payload, missing):
    """A missing slave aborts before stdin binding or source execution."""
    SandboxRequest.from_wire.return_value = SimpleNamespace(
        terminal=True, egress_origins=(), local_network=True
    )
    module.os.isatty.side_effect = lambda fd: fd != missing
    with pytest.raises(ValueError, match="receiver-owned terminal"):
        module.main()
    module.os.dup2.assert_not_called()
    linux_bootstrap.execute_payload.assert_not_called()


@pytest.mark.parametrize(
    "case", ["valid", "missing", "boolean", "stdio", "not-socket", "unexpected"]
)
def test_payload_requires_one_bound_socket_only_for_mediated_attempts(payload, monkeypatch, case):
    """The protected handoff rejects absent, forged, or extra proxy descriptors."""
    value = json.loads(module.sys.stdin.buffer.getvalue())
    if case != "missing":
        value["egress_fd"] = True if case == "boolean" else 1 if case == "stdio" else 9
    monkeypatch.setattr(module.sys, "stdin", Mock(buffer=io.BytesIO(json.dumps(value).encode())))
    SandboxRequest.from_wire.return_value = SimpleNamespace(
        egress_origins=() if case == "unexpected" else ("http://example.com",),
        terminal=False,
        local_network=True,
    )
    module.os.fstat.return_value = SimpleNamespace(
        st_mode=(stat.S_IFREG if case == "not-socket" else stat.S_IFSOCK) | 0o600
    )
    if case == "valid":
        module.main()
        assert linux_bootstrap.execute_payload.call_args.kwargs["egress_descriptor"] == 9
        module.os.fstat.assert_called_once_with(9)
    else:
        with pytest.raises(ValueError, match="Mediated egress|Unexpected mediated"):
            module.main()
        linux_bootstrap.execute_payload.assert_not_called()


@pytest.mark.parametrize("case", ["valid", "missing", "unexpected", "malformed"])
def test_payload_requires_filter_exactly_for_local_deny(payload, monkeypatch, case):
    """The trusted setup envelope carries a filter only for a denied local request."""
    value = json.loads(module.sys.stdin.buffer.getvalue())
    if case in {"valid", "unexpected"}:
        value["local_network_filter"] = base64.b64encode(b"filter").decode()
    elif case == "malformed":
        value["local_network_filter"] = "!"
    monkeypatch.setattr(module.sys, "stdin", Mock(buffer=io.BytesIO(json.dumps(value).encode())))
    SandboxRequest.from_wire.return_value = SimpleNamespace(
        terminal=False,
        egress_origins=(),
        local_network=case == "unexpected",
    )
    if case == "valid":
        module.main()
        assert linux_bootstrap.execute_payload.call_args.kwargs["local_network_filter"] == b"filter"
    else:
        with pytest.raises((ValueError, binascii.Error)):
            module.main()
        linux_bootstrap.execute_payload.assert_not_called()


@pytest.mark.parametrize(
    "tracking",
    [None, [], [1, 2], [True, 2, "service"], [1, "2", "service"], [1, -1, "service"], [1, 2, 3]],
)
def test_payload_refuses_malformed_tracking(payload, monkeypatch, tracking):
    """Malformed trusted tracking data cannot reach bootstrap or execute a command."""
    value = json.loads(module.sys.stdin.buffer.getvalue())
    value["tracking"] = tracking
    monkeypatch.setattr(module.sys, "stdin", Mock(buffer=io.BytesIO(json.dumps(value).encode())))
    with pytest.raises(ValueError, match="Invalid generation tracking"):
        module.main()
    linux_bootstrap.enter_bootstrap.assert_not_called()
    linux_bootstrap.execute_payload.assert_not_called()


@pytest.mark.parametrize("failure", ["name", "alias", "entry", "dup", "execute"])
def test_payload_failure_never_continues_toward_command(payload, monkeypatch, failure):
    """Fixed-path, native transition and descriptor failures refuse without an alternate handoff."""
    if failure == "name":
        monkeypatch.setattr(module, "__file__", str(payload.with_name("other.py")))
    elif failure == "alias":
        monkeypatch.setattr(Path, "resolve", lambda *args, **kwargs: Path("/changed"))
    elif failure == "entry":
        linux_bootstrap.enter_bootstrap.side_effect = ValueError("unsafe setup")
    elif failure == "dup":
        module.os.dup2.side_effect = ValueError("unsafe stdin")
    else:
        linux_bootstrap.execute_payload.side_effect = ValueError("unsafe activation")
    with pytest.raises(ValueError, match="fixed canonical|unsafe"):
        module.main()
    assert linux_bootstrap.execute_payload.call_count == (failure == "execute")


def test_payload_script_entry_refuses_uninstalled_code():
    """A package source path cannot masquerade as the installed payload entry."""
    with pytest.raises(ValueError, match="fixed canonical"):
        runpy.run_path(module.__file__, run_name="__main__")


@pytest.mark.parametrize("failure", ["size", "duplicate"])
def test_payload_refuses_oversized_or_ambiguous_trusted_input(payload, monkeypatch, failure):
    """Writer/reader share one ceiling and ambiguous fields cannot reach native entry."""
    from loop.constants import MAX_NATIVE_SETUP_BYTES

    data = b" " * (MAX_NATIVE_SETUP_BYTES + 1) if failure == "size" else b'{"uid":1000,"uid":0}'
    monkeypatch.setattr(module.sys, "stdin", Mock(buffer=io.BytesIO(data)))
    with pytest.raises(ValueError, match="exceeds its bound|Duplicate trusted setup field"):
        module.main()
    SandboxRequest.from_wire.assert_not_called()
    linux_bootstrap.enter_bootstrap.assert_not_called()
    linux_bootstrap.execute_payload.assert_not_called()
