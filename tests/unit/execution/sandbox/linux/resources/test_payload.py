"""Verify trusted mapped payload inputs and fail-closed irreversible handoff."""

import base64
import io
import json
import os
import runpy
from pathlib import Path
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
    monkeypatch.setattr(SandboxRequest, "from_wire", Mock(return_value="restored request"))
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
    assert args.args[0] == "restored request" and args.args[4:] == (
        b"policy",
        "payload_profile",
        "generation",
        1000,
    )
    assert args.kwargs["reservation"].path == Path("/trusted/control")


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
