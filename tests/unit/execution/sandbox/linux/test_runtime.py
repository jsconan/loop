"""Verify Linux request admission, transport and confined execution."""

import errno
import io
import json
import os
import select
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from loop.execution.sandbox.contracts import NATIVE_REQUEST_FIELDS, SandboxOutcome, SandboxRequest
from loop.execution.sandbox.linux import host as linux_host
from loop.execution.sandbox.linux import runtime as linux_runtime

from .helpers import publish

_WIRE_REQUEST = json.dumps(
    {
        "schema": 1,
        "request": {
            **dict.fromkeys(
                set(NATIVE_REQUEST_FIELDS) - {"workspace_executable_identities", "egress_origins"}
            ),
            "policy_version": linux_runtime.NATIVE_POLICY_VERSION,
            "workspace": "/tmp",
            "read_roots": [],
            "write_roots": [],
            "executable_identities": [],
        },
    }
).encode()


@pytest.fixture(autouse=True)
def runtime_revision(installation, monkeypatch):
    """Include current native namespace semantics in each isolated retained installation."""
    record, arguments = installation
    boot = record.parent / linux_runtime.BOOT_FILENAME
    boot.write_bytes(b"boot entry")
    boot.chmod(0o444)
    arguments[-1][linux_runtime.BOOT_FILENAME] = linux_host.bind_host_file(boot)
    monkeypatch.setattr(linux_runtime, "discover_tool", Mock(return_value=arguments[6]))
    monkeypatch.setattr(linux_runtime, "require_fixed_bubblewrap", Mock())
    path = record.parent / "loop/execution/sandbox/linux/constants.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(Path(linux_runtime.__file__).with_name("constants.py").read_bytes())
    path.chmod(0o444)
    arguments[-1]["loop/execution/sandbox/linux/constants.py"] = linux_host.bind_host_file(path)
    seccomp = record.parent / linux_runtime._SECCOMP_ARTIFACT
    seccomp.write_bytes(Path(linux_runtime.__file__).with_name("seccomp.py").read_bytes())
    seccomp.chmod(0o444)
    arguments[-1][linux_runtime._SECCOMP_ARTIFACT] = linux_host.bind_host_file(seccomp)
    contracts = record.parent / linux_runtime._WIRE_ARTIFACT
    contracts.write_bytes(
        Path(linux_runtime.__file__).parents[1].joinpath("contracts.py").read_bytes()
    )
    contracts.chmod(0o444)
    arguments[-1][linux_runtime._WIRE_ARTIFACT] = linux_host.bind_host_file(contracts)


@pytest.mark.parametrize(
    "source",
    [
        None,
        b"",
        b"NATIVE_UNIX_HOST_SOCKET_ISOLATION_V1 = False\n",
        b"NATIVE_UNIX_HOST_SOCKET_ISOLATION_V1 = True\nNATIVE_UNIX_HOST_SOCKET_ISOLATION_V1 = True\n",
        b"if True:\n    NATIVE_UNIX_HOST_SOCKET_ISOLATION_V1 = True\n",
        b"NATIVE_UNIX_HOST_SOCKET_ISOLATION_V1 =\n",
        b"NATIVE_UNIX_HOST_SOCKET_ISOLATION_V1 = True\n",
        b"NATIVE_UNIX_HOST_SOCKET_ISOLATION_V1 = True\nNATIVE_SCOPED_SOCKET_FAMILIES_V1 = False\n",
        b"NATIVE_UNIX_HOST_SOCKET_ISOLATION_V1 = True\nif True:\n    NATIVE_SCOPED_SOCKET_FAMILIES_V1 = True\n",
        b"NATIVE_UNIX_HOST_SOCKET_ISOLATION_V1 = True\nNATIVE_SCOPED_SOCKET_FAMILIES_V1 = True\nNATIVE_SCOPED_SOCKET_FAMILIES_V1 = True\n",
        b"NATIVE_UNIX_HOST_SOCKET_ISOLATION_V1 = True\nNATIVE_SCOPED_SOCKET_FAMILIES_V1 = True\nimport other as NATIVE_SCOPED_SOCKET_FAMILIES_V1\n",
        b"NATIVE_UNIX_HOST_SOCKET_ISOLATION_V1 = True\nNATIVE_SCOPED_SOCKET_FAMILIES_V1 = True\nfrom other import *\n",
        b"NATIVE_UNIX_HOST_SOCKET_ISOLATION_V1 = True\nNATIVE_SCOPED_SOCKET_FAMILIES_V1 = True\ndef NATIVE_SCOPED_SOCKET_FAMILIES_V1(): pass\n",
    ],
)
def test_missing_unix_isolation_is_unavailable_before_transport(installation, monkeypatch, source):
    """An older or ambiguous protected receiver cannot silently run without Unix isolation."""
    record, arguments = installation
    artifact = arguments[-1].pop(linux_runtime._SECCOMP_ARTIFACT)
    if source is not None:
        artifact.path.chmod(0o644)
        artifact.path.write_bytes(source)
        artifact.path.chmod(0o444)
        arguments[-1][linux_runtime._SECCOMP_ARTIFACT] = linux_host.bind_host_file(artifact.path)
    else:
        artifact.path.unlink()
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    transport = Mock()
    monkeypatch.setattr(linux_runtime, "exchange_request", transport)
    with pytest.raises(linux_runtime.NativeRepairRequired, match="Unix socket isolation"):
        linux_runtime.exchange_installed_request(record, b"request", 50)
    transport.assert_not_called()


@pytest.mark.parametrize("case", ["oversized", "short-read"])
def test_unix_isolation_refuses_unbounded_or_incomplete_capability_reads(monkeypatch, case):
    """The retained capability is unavailable when its bounded read is incomplete."""
    size = linux_runtime.MAX_NATIVE_SETUP_BYTES + 1 if case == "oversized" else 10
    artifact = SimpleNamespace(identity=(0, 0, 0, 0, size), open_verified=Mock())
    artifact.open_verified.return_value = nullcontext(9)
    monkeypatch.setattr(linux_runtime.os, "pread", Mock(return_value=b"short"))
    installed = SimpleNamespace(artifacts={linux_runtime._SECCOMP_ARTIFACT: artifact})
    assert not linux_runtime._has_unix_isolation(installed, None)
    assert artifact.open_verified.call_count == (case == "short-read")


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "oversized",
        "short",
        "syntax",
        "nested",
        "multiple-targets",
        "not-tuple",
        "empty-schemas",
        "too-many",
        "wrong-type",
        "duplicate",
        "non-ascii",
        "empty-name",
        "long-name",
        "zero-schema",
        "large-schema",
        "unknown-optional",
        "unknown-absent",
    ],
)
def test_wire_capabilities_reject_malformed_protected_declarations(monkeypatch, case):
    """Malformed, incomplete or unbounded installed declarations never enable transport."""
    declarations = {
        "NATIVE_REQUEST_SCHEMAS": (1, 2),
        "NATIVE_REQUEST_FEATURES": ("feature",),
        "NATIVE_REQUEST_FIELDS": ("source", "execution_source"),
        "NATIVE_REQUEST_OPTIONAL_FIELDS": (),
        "NATIVE_REQUEST_SCHEMA2_ABSENT_FIELDS": ("execution_source",),
        "NATIVE_RESULT_SCHEMAS": (1,),
        "NATIVE_POLICY_CAPABILITIES": ("linux-native-v1", "unix-host-isolation-v1"),
    }
    if case == "not-tuple":
        declarations["NATIVE_REQUEST_SCHEMAS"] = [1]
    elif case == "empty-schemas":
        declarations["NATIVE_REQUEST_SCHEMAS"] = ()
    elif case == "too-many":
        declarations["NATIVE_REQUEST_SCHEMAS"] = tuple(range(1, 66))
    elif case == "wrong-type":
        declarations["NATIVE_REQUEST_SCHEMAS"] = (True,)
    elif case == "duplicate":
        declarations["NATIVE_REQUEST_SCHEMAS"] = (1, 1)
    elif case in {"non-ascii", "empty-name", "long-name"}:
        declarations["NATIVE_REQUEST_FEATURES"] = (
            {"non-ascii": "é", "empty-name": "", "long-name": "x" * 101}[case],
        )
    elif case in {"zero-schema", "large-schema"}:
        declarations["NATIVE_REQUEST_SCHEMAS"] = (0 if case == "zero-schema" else 65536,)
    elif case == "unknown-optional":
        declarations["NATIVE_REQUEST_OPTIONAL_FIELDS"] = ("unknown",)
    elif case == "unknown-absent":
        declarations["NATIVE_REQUEST_SCHEMA2_ABSENT_FIELDS"] = ("unknown",)
    source = "\n".join(f"{name} = {value!r}" for name, value in declarations.items())
    if case == "syntax":
        source += "\ninvalid =\n"
    elif case == "nested":
        source = source.replace(
            "NATIVE_RESULT_SCHEMAS = (1,)", "if True:\n    NATIVE_RESULT_SCHEMAS = (1,)"
        )
    elif case == "multiple-targets":
        source = source.replace(
            "NATIVE_RESULT_SCHEMAS = (1,)", "other = NATIVE_RESULT_SCHEMAS = (1,)"
        )
    data = source.encode()
    size = linux_runtime.MAX_NATIVE_SETUP_BYTES + 1 if case == "oversized" else len(data)
    artifact = SimpleNamespace(identity=(0, 0, 0, 0, size), open_verified=Mock())
    artifact.open_verified.return_value = nullcontext(9)
    monkeypatch.setattr(
        linux_runtime.os, "pread", Mock(return_value=data[:-1] if case == "short" else data)
    )
    installed = SimpleNamespace(
        artifacts={} if case == "missing" else {linux_runtime._WIRE_ARTIFACT: artifact}
    )
    with pytest.raises(linux_runtime.NativeFeatureUnavailable, match="wire capabilities"):
        linux_runtime._wire_capabilities(installed, None)


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff",
        b"[]",
        b"{}",
        b'{"schema":true,"request":{}}',
        b'{"schema":1,"schema":1,"request":{}}',
    ],
)
def test_wire_negotiation_rejects_invalid_envelopes_before_transport(installation, payload):
    """Malformed and duplicate-key request frames are never sent to the receiver."""
    _record, arguments = installation
    installed = SimpleNamespace(artifacts=arguments[-1])
    with pytest.raises(linux_runtime.NativeFeatureUnavailable, match="wire format"):
        linux_runtime._require_wire_capabilities(installed, payload, None)


@pytest.mark.parametrize("length", [1, 40000])
def test_wire_negotiation_accepts_real_supported_roundtrips(installation, tmp_path, length):
    """Both wire schemas negotiate and restore real bound request authority."""
    _record, arguments = installation
    request = SandboxRequest.create(
        source="x" * length,
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin"},
        policy_version=linux_runtime.NATIVE_POLICY_VERSION,
        deadline=time.monotonic() + 60,
        workspace_id="wire-roundtrip",
    )
    payload = request.to_wire()
    assert json.loads(payload)["schema"] == (1 if length == 1 else 2)
    linux_runtime._require_wire_capabilities(
        SimpleNamespace(artifacts=arguments[-1]), payload, None
    )
    assert SandboxRequest.from_wire(payload) == request


@pytest.mark.parametrize(
    "case",
    [
        "baseline",
        "schema2",
        "missing-optional-feature",
        "new-optional",
        "new-required",
        "new-request-schema",
        "new-feature",
        "new-result-schema",
        "unrelated-declaration",
        "alias-not-needed",
        "alias-identical",
        "alias-required",
        "schema2-extra-field",
        "extra-envelope",
        "missing-policy",
        "different-policy",
        "unsupported-schema",
        "unknown-request-schema",
        "missing-baseline-schema",
        "unsupported-result",
        "unknown-field",
        "missing-field",
        "missing-identities",
        "wrong-identities",
        "wrong-identity-row",
        "short-identity-row",
        "wrong-identity-spelling",
        "wrong-identity-target",
        "unavailable-feature",
        "undeclared",
        "ambiguous",
        "egress-feature",
        "terminal-feature",
        "local-feature",
        "live-output-feature",
    ],
)
def test_wire_negotiation_rejects_unsupported_requests_before_transport(
    installation, monkeypatch, case
):
    """Negotiate the encoded request and all possible results against retained capabilities."""
    record, arguments = installation
    artifact = arguments[-1][linux_runtime._WIRE_ARTIFACT]
    source = artifact.path.read_bytes()
    optional_fields = (
        b"NATIVE_REQUEST_OPTIONAL_FIELDS = (\n"
        b'    "workspace_executable_identities",\n'
        b'    "egress_origins",\n'
        b'    "stream_output",\n'
        b'    "runtime_timeout",\n'
        b'    "cleanup_proof",\n'
        b'    "terminal",\n'
        b'    "local_network",\n'
        b")"
    )
    assert optional_fields in source
    envelope = json.loads(_WIRE_REQUEST)
    if case in {"schema2", "schema2-extra-field", "unsupported-schema"}:
        envelope["schema"] = 2
        del envelope["request"]["execution_source"]
    if case == "schema2-extra-field":
        envelope["request"]["execution_source"] = "unexpected"
    if case == "extra-envelope":
        envelope["unexpected"] = True
    if case == "missing-policy":
        source = source.replace(b'"unix-host-isolation-v1"', b'"unrelated-policy"', 1)
    if case == "different-policy":
        envelope["request"]["policy_version"] = "other-native-v1"
    if case == "new-optional":
        source = source.replace(
            b"NATIVE_REQUEST_FIELDS = (", b'NATIVE_REQUEST_FIELDS = (\n    "new_optional",', 1
        )
        source = source.replace(
            optional_fields,
            optional_fields.replace(b")", b'    "new_optional",\n)'),
        )
    if case == "new-required":
        source = source.replace(
            b"NATIVE_REQUEST_FIELDS = (", b'NATIVE_REQUEST_FIELDS = (\n    "new_required",', 1
        )
    if case == "new-request-schema":
        source = source.replace(
            b"NATIVE_REQUEST_SCHEMAS = (1, 2)", b"NATIVE_REQUEST_SCHEMAS = (1, 2, 3)"
        )
    if case == "new-feature":
        source = source.replace(
            b'    "mediated-egress-v1",\n',
            b'    "mediated-egress-v1",\n    "future-v1",\n',
        )
    if case == "egress-feature":
        envelope["request"]["egress_origins"] = ["http://example.com"]
        source = source.replace(b'    "mediated-egress-v1",\n', b"", 1)
    if case == "terminal-feature":
        envelope["request"]["terminal"] = True
        source = source.replace(b'    "owned-terminal-v1",\n', b"", 1)
    if case == "local-feature":
        envelope["request"]["local_network"] = False
        source = source.replace(b'    "sandbox-local-network-v1",\n', b"", 1)
    if case == "live-output-feature":
        envelope["request"]["stream_output"] = True
        source = source.replace(b'    "live-output-beyond-capture-v1",\n', b"", 1)
    if case in {"alias-not-needed", "alias-identical", "alias-required"}:
        if case == "alias-required":
            envelope["request"]["executable_identities"] = [
                ["awk", "/usr/bin/awk", "/usr/bin/mawk", 1, 2]
            ]
        elif case == "alias-identical":
            envelope["request"]["executable_identities"] = [
                ["awk", "/usr/bin/awk", "/usr/bin/awk", 1, 2]
            ]
        source = source.replace(
            b'    "system-executable-alias-v1",\n',
            b"",
        )
    if case == "unsupported-schema":
        source = source.replace(
            b"NATIVE_REQUEST_SCHEMAS = (1, 2)", b"NATIVE_REQUEST_SCHEMAS = (1,)"
        )
    if case == "unknown-request-schema":
        envelope["schema"] = 3
    if case == "missing-baseline-schema":
        source = source.replace(
            b"NATIVE_REQUEST_SCHEMAS = (1, 2)", b"NATIVE_REQUEST_SCHEMAS = (2,)"
        )
    if case in {"new-result-schema", "unsupported-result"}:
        versions = b"(1, 2)" if case == "new-result-schema" else b"(1, 3)"
        source = source.replace(
            b"NATIVE_RESULT_SCHEMAS = (1,)", b"NATIVE_RESULT_SCHEMAS = " + versions
        )
    if case == "unknown-field":
        envelope["request"]["future_authority"] = True
    if case == "missing-field":
        del envelope["request"]["network"]
    if case == "missing-identities":
        del envelope["request"]["executable_identities"]
    elif case == "wrong-identities":
        envelope["request"]["executable_identities"] = None
    elif case == "wrong-identity-row":
        envelope["request"]["executable_identities"] = [None]
    elif case == "short-identity-row":
        envelope["request"]["executable_identities"] = [["awk"]]
    elif case == "wrong-identity-spelling":
        envelope["request"]["executable_identities"] = [["awk", None, "/bin/awk", 1, 2]]
    elif case == "wrong-identity-target":
        envelope["request"]["executable_identities"] = [["awk", "/bin/awk", None, 1, 2]]
    if case in {"unavailable-feature", "missing-optional-feature"}:
        if case == "unavailable-feature":
            envelope["request"]["workspace_executable_identities"] = []
        source = source.replace(
            b'    "workspace-executable-identities-v1",\n',
            b"",
        )
        if case == "missing-optional-feature":
            before_fields, marker, fields_source = source.partition(b"NATIVE_REQUEST_FIELDS = (")
            source = (
                before_fields
                + marker
                + fields_source.replace(b'    "workspace_executable_identities",\n', b"", 1)
            )
            source = source.replace(
                optional_fields,
                optional_fields.replace(b'    "workspace_executable_identities",\n', b""),
            )
    if case == "undeclared":
        source = source.replace(b"NATIVE_RESULT_SCHEMAS = (1,)", b"OLD_RESULT_SCHEMAS = (1,)")
    if case == "ambiguous":
        source += b"\nNATIVE_RESULT_SCHEMAS = (1,)\n"
    if case == "unrelated-declaration":
        source += b"\nOTHER = 1\n"
    artifact.path.chmod(0o644)
    artifact.path.write_bytes(source)
    artifact.path.chmod(0o444)
    arguments[-1][linux_runtime._WIRE_ARTIFACT] = linux_host.bind_host_file(artifact.path)
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    receipt = {**arguments[1], "receipt": linux_host.bind_host_file(record)}
    monkeypatch.setattr(linux_runtime, "installation_files", Mock(return_value=receipt))
    transport = Mock(return_value=b"response")
    monkeypatch.setattr(linux_runtime, "exchange_request", transport)
    payload = json.dumps(envelope).encode()
    if case in {
        "baseline",
        "schema2",
        "missing-optional-feature",
        "new-optional",
        "new-request-schema",
        "new-feature",
        "new-result-schema",
        "alias-not-needed",
        "alias-identical",
        "unrelated-declaration",
    }:
        assert linux_runtime.exchange_installed_request(record, payload, 50) == b"response"
        transport.assert_called_once_with(arguments[0], receipt, payload, 50)
    else:
        with pytest.raises(linux_runtime.NativeFeatureUnavailable):
            linux_runtime.exchange_installed_request(record, payload, 50)
        transport.assert_not_called()


def test_incomplete_installation_requires_refresh(installation, monkeypatch):
    """An installation without its required boot owner is refused before transport."""
    record, arguments = installation
    boot = arguments[-1].pop(linux_runtime.BOOT_FILENAME)
    boot.path.unlink()
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    transport = Mock()
    monkeypatch.setattr(linux_runtime, "exchange_request", transport)
    for operation in (
        lambda: linux_runtime.execution_authority(record),
        lambda: linux_runtime.exchange_installed_request(record, b"request", 50),
    ):
        with pytest.raises(linux_runtime.NativeRepairRequired, match="needs /sandbox refresh"):
            operation()
    transport.assert_not_called()


def test_changed_system_bubblewrap_requires_checked_repair(installation, monkeypatch):
    """Changed system helper bytes are detected before sending a command to the old copy."""
    record, _ = installation
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    linux_runtime.discover_tool.return_value = Mock(digest="different")
    transport = Mock()
    monkeypatch.setattr(linux_runtime, "exchange_request", transport)
    for operation in (
        lambda: linux_runtime.execution_authority(record),
        lambda: linux_runtime.exchange_installed_request(record, b"request", 50),
    ):
        with pytest.raises(linux_runtime.NativeRepairRequired, match="System Bubblewrap changed"):
            operation()
    transport.assert_not_called()


@pytest.mark.parametrize(
    "failure", [ValueError("old helper"), OSError("unavailable"), TimeoutError()]
)
def test_unfixed_installed_bubblewrap_refuses_before_transport(installation, monkeypatch, failure):
    """An installed helper without fixed setup semantics cannot enter a command."""
    record, _ = installation
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    linux_runtime.require_fixed_bubblewrap.side_effect = failure
    transport = Mock()
    monkeypatch.setattr(linux_runtime, "exchange_request", transport)
    for operation in (
        lambda: linux_runtime.execution_authority(record),
        lambda: linux_runtime.exchange_installed_request(record, b"request", 50),
    ):
        with pytest.raises(linux_runtime.NativeRepairRequired, match="fixed setup-symlink"):
            operation()
    transport.assert_not_called()


def test_system_bubblewrap_replacement_with_identical_bytes_remains_compatible(
    installation, monkeypatch
):
    """A system helper inode change alone does not force a privileged refresh."""
    record, arguments = installation
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    linux_runtime.discover_tool.return_value = Mock(digest=arguments[6].digest, identity=(99,))
    receipt = {**arguments[1], "receipt": linux_host.bind_host_file(record)}
    monkeypatch.setattr(linux_runtime, "installation_files", Mock(return_value=receipt))
    monkeypatch.setattr(linux_runtime, "exchange_request", Mock(return_value=b"response"))
    assert linux_runtime.exchange_installed_request(record, _WIRE_REQUEST, 50) == b"response"


def test_managed_bubblewrap_survives_system_and_manifest_updates(installation, monkeypatch):
    """Protected reviewed helper bytes remain usable when unused host bytes change."""
    record, arguments = installation
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    monkeypatch.setattr(linux_runtime, "is_managed_bubblewrap", Mock(return_value=True))
    linux_runtime.discover_tool.side_effect = AssertionError("system helper must not be selected")
    receipt = {**arguments[1], "receipt": linux_host.bind_host_file(record)}
    monkeypatch.setattr(linux_runtime, "installation_files", Mock(return_value=receipt))
    monkeypatch.setattr(linux_runtime, "exchange_request", Mock(return_value=b"response"))
    assert linux_runtime.exchange_installed_request(record, _WIRE_REQUEST, 50) == b"response"
    linux_runtime.discover_tool.assert_not_called()


@pytest.mark.parametrize("change", [None, "receipt", "record"])
def test_installed_admission_checks_original_receipt_and_late_record(
    installation, monkeypatch, change
):
    """Bind installed roles to saved originals and revalidate after receiving."""
    record, arguments = installation
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    receipt = {**arguments[1], "receipt": linux_host.bind_host_file(record)}
    if change == "receipt":
        receipt["receiver"] = arguments[6]
    monkeypatch.setattr(linux_runtime, "installation_files", Mock(return_value=receipt))

    def receive(*args):
        """Return opaque content and optionally simulate late original record loss."""
        if change == "record":
            record.unlink()
        return b"opaque request"

    received = Mock(side_effect=lambda *args: nullcontext((receive(*args), Mock())))
    monkeypatch.setattr(linux_runtime, "receive_request", received)
    if change:
        with (
            pytest.raises(
                ValueError if change == "receipt" else FileNotFoundError,
                match="Receipt admission differs" if change == "receipt" else "management.json",
            ),
            linux_runtime.receive_installed_request(record, 50),
        ):
            pass
        assert received.call_count == (change == "record")
    else:
        with linux_runtime.receive_installed_request(record, 50) as (payload, _):
            assert payload == b"opaque request"
        received.assert_called_once_with(arguments[0], receipt, 50)
    linux_runtime.installation_files.assert_called_once_with(
        record.parent / "receipt.json", deadline=50
    )


@pytest.mark.parametrize("change", [None, "receipt", "record", "deadline", "late_deadline"])
def test_installed_transport_retains_original_receipt_record_and_deadline(
    installation, monkeypatch, change
):
    """Ordinary exchange uses saved original authority and refuses drift or expired budgets."""
    record, arguments = installation
    publish(installation)
    receipt = {**arguments[1], "receipt": linux_host.bind_host_file(record)}
    if change == "receipt":
        receipt["receiver"] = arguments[6]
    monkeypatch.setattr(linux_runtime, "installation_files", Mock(return_value=receipt))
    clock = [10]
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: clock[0])

    def exchange(*args):
        """Return opaque response and optionally invalidate original late authority or budget."""
        if change == "record":
            record.unlink()
        if change == "late_deadline":
            clock[0] = 55
        return b"unavailable response"

    exchanged = Mock(side_effect=exchange)
    monkeypatch.setattr(linux_runtime, "exchange_request", exchanged)
    if change:
        with pytest.raises(
            FileNotFoundError
            if change == "record"
            else TimeoutError
            if "deadline" in change
            else ValueError,
            match="management.json"
            if change == "record"
            else "transport deadline"
            if "deadline" in change
            else "Receipt transport differs",
        ):
            linux_runtime.exchange_installed_request(
                record, _WIRE_REQUEST, 10 if change == "deadline" else 50
            )
        assert exchanged.call_count == (change in {"record", "late_deadline"})
    else:
        assert (
            linux_runtime.exchange_installed_request(record, _WIRE_REQUEST, 50)
            == b"unavailable response"
        )
        exchanged.assert_called_once_with(arguments[0], receipt, _WIRE_REQUEST, 50)


def test_installed_transport_forwards_cancellation_after_authenticating_authority(
    installation, monkeypatch
):
    """Cancellation and terminal control reach only the revalidated original socket."""
    record, arguments = installation
    publish(installation)
    receipt = {**arguments[1], "receipt": linux_host.bind_host_file(record)}
    monkeypatch.setattr(linux_runtime, "installation_files", Mock(return_value=receipt))
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    exchanged = Mock(return_value=b"cleanup result")
    monkeypatch.setattr(linux_runtime, "exchange_request", exchanged)
    cancellation_event = Event()
    observer = Mock()
    terminal_input = Mock(return_value=None)

    result = linux_runtime.exchange_installed_request(
        record,
        _WIRE_REQUEST,
        50,
        cancellation_event=cancellation_event,
        observer=observer,
        terminal_input=terminal_input,
    )

    assert result == b"cleanup result"
    exchanged.assert_called_once_with(
        arguments[0],
        receipt,
        _WIRE_REQUEST,
        50,
        cancellation_event=cancellation_event,
        observer=observer,
        terminal_input=terminal_input,
    )


def test_installed_transport_forwards_live_output_only_after_authority(installation, monkeypatch):
    """The retained receiver gets the exact observer after installation verification."""
    record, arguments = installation
    publish(installation)
    receipt = {**arguments[1], "receipt": linux_host.bind_host_file(record)}
    monkeypatch.setattr(linux_runtime, "installation_files", Mock(return_value=receipt))
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    exchanged = Mock(return_value=b"result")
    monkeypatch.setattr(linux_runtime, "exchange_request", exchanged)
    observer = Mock()

    assert (
        linux_runtime.exchange_installed_request(
            record, _WIRE_REQUEST, 50, output_observer=observer
        )
        == b"result"
    )
    exchanged.assert_called_once_with(
        arguments[0], receipt, _WIRE_REQUEST, 50, output_observer=observer
    )


def test_installed_transport_forwards_only_its_bound_egress_lease(installation, monkeypatch):
    """The authenticated receiver exchange receives the exact approved lease object."""
    record, arguments = installation
    publish(installation)
    receipt = {**arguments[1], "receipt": linux_host.bind_host_file(record)}
    monkeypatch.setattr(linux_runtime, "installation_files", Mock(return_value=receipt))
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    exchanged = Mock(return_value=b"mediated result")
    monkeypatch.setattr(linux_runtime, "exchange_request", exchanged)
    lease = Mock()
    assert (
        linux_runtime.exchange_installed_request(record, _WIRE_REQUEST, 50, egress_lease=lease)
        == b"mediated result"
    )
    exchanged.assert_called_once_with(arguments[0], receipt, _WIRE_REQUEST, 50, egress_lease=lease)


def test_installed_transport_refuses_deadline_expiring_after_record_check(
    installation, monkeypatch
):
    """The final record verification cannot return a response after its delivery budget."""
    from types import SimpleNamespace

    record, arguments = installation
    clock = [10]

    @contextmanager
    def verified(*, deadline):
        """Expire only after original record verification has completed."""
        yield
        clock[0] = deadline

    authority = Mock(open_verified=Mock(side_effect=verified))
    owner = SimpleNamespace(
        config=arguments[0], files=arguments[1], artifacts=arguments[-1], helper=arguments[6]
    )
    receipt = {**arguments[1], "receipt": authority}
    monkeypatch.setattr(linux_runtime, "read_generation", Mock(return_value=(owner, authority)))
    monkeypatch.setattr(linux_runtime, "installation_files", Mock(return_value=receipt))
    exchange = Mock(return_value=b"opaque response")
    monkeypatch.setattr(linux_runtime, "exchange_request", exchange)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: clock[0])
    with pytest.raises(TimeoutError, match="transport deadline"):
        linux_runtime.exchange_installed_request(record, _WIRE_REQUEST, 50)
    exchange.assert_called_once()
    authority.open_verified.assert_called_once_with(deadline=55)


@pytest.mark.parametrize("missing", [False, True, "conditional"])
def test_installed_authority_refuses_incompatible_namespace_revision(
    installation, monkeypatch, missing
):
    """A package upgrade cannot send new namespace semantics to an older retained receiver."""
    record, arguments = installation
    name = "loop/execution/sandbox/linux/constants.py"
    if missing is True:
        path = arguments[-1][name].path
        path.unlink()
        del arguments[-1][name]
    else:
        path = arguments[-1][name].path
        path.chmod(0o644)
        if missing == "conditional":
            source = path.read_bytes().replace(
                b'NATIVE_POLICY_VERSION = "linux-native-v1"',
                b'if True:\n    NATIVE_POLICY_VERSION = "linux-native-v1"',
            )
        else:
            source = b"old native revision"
        path.write_bytes(source)
        path.chmod(0o444)
        arguments[-1][name] = linux_host.bind_host_file(path)
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    transport = Mock()
    monkeypatch.setattr(linux_runtime, "exchange_request", transport)
    for operation in (
        lambda: linux_runtime.execution_authority(record),
        lambda: linux_runtime.exchange_installed_request(record, b"request", 50),
    ):
        with pytest.raises(linux_runtime.NativeFeatureUnavailable, match="native interface"):
            operation()
    transport.assert_not_called()


@pytest.mark.parametrize("change", ["comment", "addition", "unrelated_value"])
def test_installed_authority_refuses_unreviewed_constant_changes(installation, monkeypatch, change):
    """Any retained source edit requires a new checked native generation."""
    record, arguments = installation
    name = "loop/execution/sandbox/linux/constants.py"
    path = arguments[-1][name].path
    path.chmod(0o644)
    source = path.read_bytes()
    if change == "comment":
        source += b"\n# Retained installation comment.\n"
    elif change == "addition":
        source += b'\nEXTRA = "x"\n'
    else:
        previous = b'COMPILER_LOCK_FILENAME = "compiler.lock"'
        assert previous in source
        source = source.replace(previous, b'COMPILER_LOCK_FILENAME = "other.lock"')
    path.write_bytes(source)
    path.chmod(0o444)
    arguments[-1][name] = linux_host.bind_host_file(path)
    for name in ("loop_payload.py", "policies/root/bootstrap.bin"):
        path = record.parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"entry")
        path.chmod(0o444)
        arguments[-1][name] = linux_host.bind_host_file(path)
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    receipt = {**arguments[1], "receipt": linux_host.bind_host_file(record)}
    monkeypatch.setattr(linux_runtime, "installation_files", Mock(return_value=receipt))
    transport = Mock(return_value=b"response")
    monkeypatch.setattr(linux_runtime, "exchange_request", transport)

    with pytest.raises(linux_runtime.NativeFeatureUnavailable, match="native interface"):
        linux_runtime.execution_authority(record)
    with pytest.raises(linux_runtime.NativeFeatureUnavailable, match="native interface"):
        linux_runtime.exchange_installed_request(record, _WIRE_REQUEST, 50)
    transport.assert_not_called()


def test_installed_authority_refuses_changed_constant_value(installation, monkeypatch):
    """Retained constants with different values cannot silently change request semantics."""
    record, arguments = installation
    name = "loop/execution/sandbox/linux/constants.py"
    path = arguments[-1][name].path
    path.chmod(0o644)
    source = path.read_bytes().replace(b'"linux-native-v1"', b'"linux-native-v2"')
    path.write_bytes(source)
    path.chmod(0o444)
    arguments[-1][name] = linux_host.bind_host_file(path)
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)

    with pytest.raises(linux_runtime.NativeFeatureUnavailable, match="native interface"):
        linux_runtime.execution_authority(record)


@pytest.mark.parametrize(
    "source_change",
    [
        lambda source: source.replace(
            b"from .... import constants as app_constants",
            b"from other import constants as app_constants",
        ),
        lambda source: source + b"\napp_constants = object()\n",
        lambda source: source + b"\napp_constants.APP_NAME = 'other'\n",
        lambda source: source.replace(b"-private-paths", b"-other-paths"),
    ],
)
def test_installed_authority_refuses_changed_composed_interface(
    installation, monkeypatch, source_change
):
    """The interface reader does not execute or trust altered symbolic name expressions."""
    record, arguments = installation
    name = "loop/execution/sandbox/linux/constants.py"
    path = arguments[-1][name].path
    path.chmod(0o644)
    path.write_bytes(source_change(path.read_bytes()))
    path.chmod(0o444)
    arguments[-1][name] = linux_host.bind_host_file(path)
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)

    with pytest.raises(linux_runtime.NativeFeatureUnavailable, match="native interface"):
        linux_runtime.execution_authority(record)


@pytest.mark.parametrize(
    "suffix",
    [
        b'\nNATIVE_POLICY_VERSION = "linux-native-v1"\n',
        b"\nNATIVE_SCRATCH_PATH: object = None\n",
        b"\nfrom other import NATIVE_POLICY_VERSION\n",
        b"\nfrom other import *\n",
    ],
)
def test_installed_authority_refuses_ambiguous_interface(installation, monkeypatch, suffix):
    """Repeated interface declarations cannot silently select incompatible semantics."""
    record, arguments = installation
    name = "loop/execution/sandbox/linux/constants.py"
    path = arguments[-1][name].path
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + suffix)
    path.chmod(0o444)
    arguments[-1][name] = linux_host.bind_host_file(path)
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)

    with pytest.raises(linux_runtime.NativeFeatureUnavailable, match="native interface"):
        linux_runtime.execution_authority(record)


def test_installed_authority_refuses_annotated_retained_interface(installation, monkeypatch):
    """A source edit needs checked refresh even when one declaration looks equivalent."""
    record, arguments = installation
    name = "loop/execution/sandbox/linux/constants.py"
    path = arguments[-1][name].path
    path.chmod(0o644)
    source = path.read_bytes().replace(b"NATIVE_POLICY_VERSION =", b"NATIVE_POLICY_VERSION: str =")
    path.write_bytes(source)
    path.chmod(0o444)
    arguments[-1][name] = linux_host.bind_host_file(path)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)

    for name in ("loop_payload.py", "policies/root/bootstrap.bin"):
        entry = record.parent / name
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_bytes(b"entry")
        entry.chmod(0o444)
        arguments[-1][name] = linux_host.bind_host_file(entry)
    publish(installation)
    with pytest.raises(linux_runtime.NativeFeatureUnavailable, match="native interface"):
        linux_runtime.execution_authority(record)


def test_installed_authority_uses_reviewed_source_digest(installation, monkeypatch):
    """Compatibility follows the frozen reviewed source, not mutable module attributes."""
    record, arguments = installation
    for name in ("loop_payload.py", "policies/root/bootstrap.bin"):
        entry = record.parent / name
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_bytes(b"entry")
        entry.chmod(0o444)
        arguments[-1][name] = linux_host.bind_host_file(entry)
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    assert linux_runtime.execution_authority(record)
    monkeypatch.setattr(linux_runtime.linux_constants, "NATIVE_POLICY_VERSION", "changed")
    assert linux_runtime.execution_authority(record)
    monkeypatch.setattr(linux_runtime, "_REVIEWED_CONSTANTS_DIGEST", b"changed")
    with pytest.raises(linux_runtime.NativeFeatureUnavailable, match="native interface"):
        linux_runtime.execution_authority(record)


@pytest.fixture
def execution(installation, monkeypatch):
    """Bind a complete runtime while controlling privileged launch collaborators."""
    from types import SimpleNamespace

    from loop.utils.process import ProcessCapture, ProcessCaptureStatus

    record, arguments = installation
    for name, data in (
        ("cache.lock", b"test\n"),
        ("mask.empty", b""),
        ("loop_payload.py", b"entry"),
    ):
        path = record.parent / name
        path.write_bytes(data)
        path.chmod(0o444)
        arguments[-1][name] = linux_host.bind_host_file(path)
    bootstrap_path = record.parent / "policies/root/bootstrap.bin"
    bootstrap_path.parent.mkdir(parents=True)
    bootstrap_path.write_bytes(b"bootstrap")
    bootstrap_path.chmod(0o444)
    arguments[-1]["policies/root/bootstrap.bin"] = linux_host.bind_host_file(bootstrap_path)
    publish(installation)
    for name in ("getuid", "geteuid", "getgid", "getegid"):
        monkeypatch.setattr(linux_runtime.os, name, lambda: 1000)
    request = Mock(
        deadline=50,
        runtime_timeout=None,
        terminal=False,
        attempt_id="11111111-1111-4111-8111-111111111111",
        environment=(),
        policy_version=linux_runtime.NATIVE_POLICY_VERSION,
    )
    request.shell_environment.return_value = {"TMPDIR": "/tmp/private"}
    request.paths_are_current.return_value = True
    request.host_command_runtime_signature.return_value = "a" * 64
    monkeypatch.setattr(linux_runtime.SandboxRequest, "from_wire", Mock(return_value=request))
    monkeypatch.setattr(
        linux_runtime,
        "installation_files",
        Mock(return_value={**arguments[1], "apparmor": arguments[6]}),
    )
    original_bind = linux_runtime.bind_host_file
    monkeypatch.setattr(
        linux_runtime,
        "bind_host_file",
        lambda path, **_kwargs: (
            arguments[6] if path == Path("/usr/bin/python3") else original_bind(path, **_kwargs)
        ),
    )
    monkeypatch.setattr(
        linux_runtime,
        "namespace_arguments",
        Mock(return_value=nullcontext((("--unshare-pid",), (77,)))),
    )
    monkeypatch.setattr(linux_runtime, "runtime_read_roots", Mock(return_value=(Path("/usr"),)))
    monkeypatch.setattr(linux_runtime, "payload_profile", Mock(return_value="payload_profile"))
    monkeypatch.setattr(linux_runtime, "compile_payload", Mock(return_value=b"compiled"))
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    original_open, original_close = os.open, os.close
    monkeypatch.setattr(linux_runtime.os, "O_PATH", 0, raising=False)
    monkeypatch.setattr(
        linux_runtime.os,
        "open",
        lambda path, *args: (
            999
            if path in (record.parent, Path("/sys/kernel/security/apparmor"))
            else original_open(path, *args)
        ),
    )
    monkeypatch.setattr(
        linux_runtime.os, "close", lambda fd: None if fd == 999 else original_close(fd)
    )
    launch = Mock()
    monkeypatch.setattr(linux_runtime.subprocess, "Popen", launch)
    capture = ProcessCapture(
        ProcessCaptureStatus.COMPLETED,
        exit_code=17,
        stdout="LOOP_PAYLOAD_" + request.attempt_id + "\noutput",
        stderr="error",
        stdout_discarded=3,
    )
    supervisor = Mock(return_value=capture)
    monkeypatch.setattr(linux_runtime, "supervise_payload", supervisor)
    group = record.parent.parent / "tracking" / "loopgen_native@instance.service"
    group.mkdir(parents=True)
    membership = Mock(return_value=group)
    monkeypatch.setattr(linux_runtime, "admission_group", membership)
    guard = Mock(side_effect=lambda _: nullcontext(Event()))
    monkeypatch.setattr(linux_runtime, "guard_client_disconnect", guard)
    poller = Mock()
    poller.poll.return_value = []
    monkeypatch.setattr(linux_runtime.select, "poll", lambda: poller)
    return SimpleNamespace(
        record=record,
        connection=Mock(),
        arguments=arguments,
        request=request,
        launch=launch,
        capture=capture,
        supervisor=supervisor,
        membership=membership,
        group=group,
        guard=guard,
        poller=poller,
    )


def test_receiver_refuses_changed_system_helper_before_launch(execution):
    """A compatible older client cannot launch an outdated copied system helper."""
    linux_runtime.discover_tool.return_value = Mock(digest="different")
    result = linux_runtime.execute_installed_request(
        execution.record, b"snapshot", execution.connection
    )
    assert result.outcome is SandboxOutcome.INVALID
    assert result.detail == "System Bubblewrap changed; use /sandbox refresh."
    assert not result.possible_effects
    execution.launch.assert_not_called()


@pytest.mark.parametrize("case", ["emfile", "enfile", "io"])
def test_mount_descriptor_failures_refuse_before_payload(execution, case):
    """Descriptor exhaustion is reported before launch without a host recovery offer."""
    number = {"emfile": errno.EMFILE, "enfile": errno.ENFILE, "io": errno.EIO}[case]
    linux_runtime.namespace_arguments.side_effect = OSError(number, "descriptor preparation")
    if case == "io":
        with pytest.raises(OSError, match="descriptor preparation"):
            linux_runtime.execute_installed_request(
                execution.record, b"snapshot", execution.connection
            )
        return
    result = linux_runtime.execute_installed_request(
        execution.record, b"snapshot", execution.connection
    )
    assert result.outcome is SandboxOutcome.INVALID
    assert not result.possible_effects
    assert "open-file limit" in result.detail
    execution.launch.assert_not_called()


@pytest.mark.parametrize("case", ["emfile", "enfile", "io"])
def test_tracking_descriptor_failures_refuse_before_payload(execution, monkeypatch, case):
    """Tracking descriptor exhaustion is reported without starting the command."""
    original_open = os.open
    number = {"emfile": errno.EMFILE, "enfile": errno.ENFILE, "io": errno.EIO}[case]

    def open_tracking(path, *args, **kwargs):
        """Fail only when binding the authenticated tracking directory."""
        if path == execution.group.parent:
            raise OSError(number, "tracking descriptor")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(linux_runtime.os, "open", open_tracking)
    if case == "io":
        with pytest.raises(OSError, match="tracking descriptor"):
            linux_runtime.execute_installed_request(
                execution.record, b"snapshot", execution.connection
            )
        return
    result = linux_runtime.execute_installed_request(
        execution.record, b"snapshot", execution.connection
    )
    assert result.outcome is SandboxOutcome.INVALID
    assert not result.possible_effects
    assert "open-file limit" in result.detail
    execution.launch.assert_not_called()


@pytest.mark.parametrize("change", ["directory", "membership"])
def test_execute_installed_request_refuses_changed_tracking(execution, monkeypatch, change):
    """The cgroup source and receiver membership must remain bound through launch."""
    if change == "directory":
        from types import SimpleNamespace

        original = Path.stat

        def changed(path, *args, **kwargs):
            """Alter only the claimed cgroup directory identity."""
            if path == execution.group.parent:
                return SimpleNamespace(st_dev=0, st_ino=0)
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", changed)
    else:
        execution.membership.side_effect = [execution.group, execution.group.with_name("other")]
    with pytest.raises(ValueError, match="tracking"):
        linux_runtime.execute_installed_request(execution.record, b"snapshot", execution.connection)
    execution.launch.assert_not_called()


def test_execute_installed_request_guards_spawn_and_supervision(execution, monkeypatch):
    """The disconnect guard owns the entire launch and descendant-cleanup scope."""
    order = []
    spawn_returned = Event()

    @contextmanager
    def guard(descriptor):
        """Record when the authenticated socket is guarded."""
        assert descriptor is execution.connection
        order.append("guard-enter")
        yield spawn_returned
        order.append("guard-exit")

    execution.guard.side_effect = guard
    execution.launch.side_effect = lambda *args, **kwargs: order.append("spawn") or Mock()

    def supervise(*_args, **_kwargs):
        """Require the handoff signal before ordinary supervision begins."""
        assert spawn_returned.is_set()
        order.append("supervise")
        return execution.capture

    execution.supervisor.side_effect = supervise

    result = linux_runtime.execute_installed_request(
        execution.record, b"snapshot", execution.connection
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    assert execution.membership.call_count == 2
    execution.membership.assert_called_with(execution.arguments[0])
    assert order == ["guard-enter", "spawn", "supervise", "guard-exit"]


@pytest.mark.parametrize("prepared_in_time", [False, True])
def test_receiver_renews_runtime_only_after_preparation(execution, monkeypatch, prepared_in_time):
    """An authenticated receiver rejects slow setup and starts runtime at launch."""
    state = execution
    state.request.runtime_timeout = 30
    state.request.deadline = 100
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10 if prepared_in_time else 70)

    def renew(request, *, deadline):
        """Model immutable request replacement while preserving the fixture collaborator."""
        request.deadline = deadline
        return request

    monkeypatch.setattr(linux_runtime, "replace", renew)
    result = linux_runtime.execute_installed_request(state.record, b"snapshot", state.connection)

    if prepared_in_time:
        assert result.outcome is SandboxOutcome.COMPLETED
        assert result.cleanup_verified is True
        assert state.supervisor.call_args.args[2] == 40
        state.launch.assert_called_once()
    else:
        assert result.outcome is SandboxOutcome.TIMED_OUT
        assert "preparation" in result.detail
        state.launch.assert_not_called()


def test_receiver_refuses_preparation_expiring_at_final_spawn_check(execution, monkeypatch):
    """A deadline crossed during setup prevents the final native process launch."""
    state = execution
    state.request.runtime_timeout = 30
    state.request.deadline = 100
    clock = [10]
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: clock[0])

    def renew(request, *, deadline):
        """Model immutable request replacement for the existing fixture collaborator."""
        request.deadline = deadline
        return request

    def finish_preparation(_request, _descriptor):
        """Cross the preparation deadline just before the final launch check."""
        clock[0] = 70
        return False

    monkeypatch.setattr(linux_runtime, "replace", renew)
    monkeypatch.setattr(linux_runtime, "_cancelled_before_spawn", finish_preparation)

    result = linux_runtime.execute_installed_request(state.record, b"snapshot", state.connection)

    assert result.outcome is SandboxOutcome.TIMED_OUT
    assert "preparation" in result.detail
    state.launch.assert_not_called()


@pytest.mark.parametrize("confirmed", [True, False])
def test_execute_installed_request_binds_and_closes_private_egress_bridge(
    execution, monkeypatch, confirmed
):
    """Native launch passes only this attempt's bridge FD and checks closure before success."""
    execution.request.egress_origins = ("http://example.com",)
    bridge_host, bridge_child = MagicMock(), MagicMock()
    bridge_child.fileno.return_value = 88
    factory = Mock(return_value=(bridge_host, bridge_child))
    monkeypatch.setattr(linux_runtime.socket, "socketpair", factory)
    broker = Mock()
    broker.close.return_value = confirmed
    monkeypatch.setattr(linux_runtime, "EgressBroker", Mock(return_value=broker))
    result = linux_runtime.execute_installed_request(
        execution.record, b"snapshot", execution.connection
    )
    assert result.outcome is (SandboxOutcome.COMPLETED if confirmed else SandboxOutcome.INVALID)
    factory.assert_called_once_with(
        linux_runtime.socket.AF_UNIX, linux_runtime.socket.SOCK_SEQPACKET
    )
    assert 88 in execution.launch.call_args.kwargs["pass_fds"]
    execution.connection.sendall.assert_called_once_with(b"E")
    execution.supervisor.assert_called_once()
    assert execution.supervisor.call_args.kwargs["mediator"] is broker
    broker.start_lease.assert_called_once()
    assert broker.close.call_count >= 1
    if not confirmed:
        assert result.possible_effects


@pytest.mark.parametrize("entered", [False, True])
def test_execute_installed_request_preserves_partial_output_after_reader_failure(
    execution, entered
):
    """Incomplete capture retains output and never claims completion or safe replay."""
    from dataclasses import replace

    from loop.execution.sandbox.contracts import SandboxOutcome
    from loop.utils.process import ProcessCaptureStatus

    header = "LOOP_PAYLOAD_" + execution.request.attempt_id + "\n"
    execution.supervisor.return_value = replace(
        execution.capture,
        status=ProcessCaptureStatus.UNAVAILABLE,
        exit_code=None,
        stdout=(header if entered else "") + "partial",
        stderr="error",
        stdout_discarded=None,
    )

    result = linux_runtime.execute_installed_request(
        execution.record, b"snapshot", execution.connection
    )

    assert result.outcome is (SandboxOutcome.UNAVAILABLE if entered else SandboxOutcome.INVALID)
    assert result.possible_effects is True
    assert result.stdout == "partial" and result.stderr == "error"
    assert result.stdout_discarded is None
    assert result.detail == (
        "Native command output capture failed."
        if entered
        else "Trusted payload entry was not established."
    )


@pytest.mark.parametrize("error", [OSError("membership unavailable"), ValueError("unsafe scope")])
def test_execute_installed_request_refuses_unverified_disconnect_guard(execution, error):
    """A receiver outside its fixed kernel scope cannot start the process-exit guard."""
    execution.membership.side_effect = error
    with pytest.raises(type(error), match=str(error)):
        linux_runtime.execute_installed_request(execution.record, b"snapshot", execution.connection)
    execution.guard.assert_not_called()
    execution.launch.assert_not_called()


def test_execute_installed_request_cancels_while_waiting_to_compile(execution, monkeypatch):
    """A cancelled compiler wait returns without launching or claiming command effects."""
    from loop.execution.sandbox.contracts import SandboxOutcome

    check = Mock(return_value=True)
    monkeypatch.setattr(linux_runtime, "_cancelled_before_spawn", check)

    def compile_wait(*args, cancelled):
        """Observe the receiver's cancellation check during simulated lock contention."""
        if cancelled():
            raise linux_runtime.CompilationCancelled("Payload compilation cancelled.")
        return b"compiled"

    monkeypatch.setattr(linux_runtime, "compile_payload", compile_wait)
    result = linux_runtime.execute_installed_request(
        execution.record, b"snapshot", execution.connection
    )
    assert result.outcome is SandboxOutcome.CANCELLED
    assert result.possible_effects is False
    check.assert_called_once_with(execution.request, 0)
    execution.launch.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        None,
        "caller",
        "revision",
        "scratch",
        "cache",
        "mask",
        "entry",
        "nonempty",
        "receipt",
        "bootstrap",
        "write",
        "size",
        "stale",
        "expired",
        "late-expired",
        "no-header",
        "no-header-overflow",
        "unknown-discard",
        "overflow",
        "timeout",
        "pre-entry-timeout",
        "pre-entry-cancel",
    ],
)
def test_execute_installed_request_authenticates_launch_and_result(execution, monkeypatch, change):
    """A complete original runtime returns only a bounded attempt-bound result or fails closed."""
    from dataclasses import replace

    from loop.execution.sandbox.contracts import SandboxOutcome
    from loop.utils.process import ProcessCaptureStatus

    state = execution
    if change == "caller":
        monkeypatch.setattr(linux_runtime.os, "geteuid", lambda: 0)
    elif change == "revision":
        state.request.policy_version = "linux-native-unsupported"
    elif change == "scratch":
        state.request.shell_environment.return_value = {}
    elif change in {"cache", "mask", "entry", "bootstrap", "nonempty"}:
        name = {
            "cache": "cache.lock",
            "mask": "mask.empty",
            "entry": "loop_payload.py",
            "bootstrap": "policies/root/bootstrap.bin",
            "nonempty": "mask.empty",
        }[change]
        if change == "nonempty":
            path = state.arguments[-1][name].path
            path.chmod(0o644)
            path.write_bytes(b"unsafe")
            path.chmod(0o444)
            state.arguments[-1][name] = linux_host.bind_host_file(path)
        else:
            state.arguments[-1][name].path.unlink()
            if change == "bootstrap":
                (state.record.parent / "policies/root").rmdir()
                (state.record.parent / "policies").rmdir()
            del state.arguments[-1][name]
        state.record.unlink()
        publish((state.record, state.arguments))
    elif change == "receipt":
        linux_runtime.installation_files.return_value["receiver"] = state.arguments[6]
    elif change == "stale":
        state.request.paths_are_current.return_value = False
    elif change == "expired":
        monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 50)
    elif change == "late-expired":
        clock = [10]
        monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: clock[0])
        state.request.paths_are_current.side_effect = lambda: clock.__setitem__(0, 50) or True
    elif change in {"write", "size"}:
        if change == "size":
            linux_runtime.compile_payload.return_value = b"x" * 1_600_000
        else:
            stream = Mock()
            stream.write.return_value = 0
            monkeypatch.setattr(
                linux_runtime.tempfile, "TemporaryFile", lambda: nullcontext(stream)
            )
    elif change in {"no-header", "no-header-overflow"}:
        state.supervisor.return_value = replace(
            state.capture, stdout="forged" if change == "no-header" else "x" * 1_000_080
        )
    elif change == "unknown-discard":
        state.supervisor.return_value = replace(state.capture, stdout_discarded=None)
    elif change == "overflow":
        state.supervisor.return_value = replace(
            state.capture, stdout=state.capture.stdout + "x" * 1_000_000
        )
    elif change in {"pre-entry-timeout", "pre-entry-cancel"}:
        state.supervisor.return_value = replace(
            state.capture,
            status=ProcessCaptureStatus.TIMED_OUT
            if change == "pre-entry-timeout"
            else ProcessCaptureStatus.CANCELLED,
            exit_code=None,
            stdout="bootstrap-out",
            stderr="bootstrap-error",
            stdout_discarded=None,
            stderr_discarded=13,
        )
    elif change == "timeout":
        state.supervisor.return_value = replace(
            state.capture, status=ProcessCaptureStatus.TIMED_OUT, exit_code=None
        )
    failures = {
        "caller",
        "revision",
        "scratch",
        "cache",
        "mask",
        "entry",
        "nonempty",
        "receipt",
        "bootstrap",
        "write",
        "size",
        "stale",
        "expired",
        "late-expired",
    }
    if change in failures:
        with pytest.raises(TimeoutError if change in {"expired", "late-expired"} else ValueError):
            linux_runtime.execute_installed_request(state.record, b"snapshot", state.connection)
        state.launch.assert_not_called()
    else:
        result = linux_runtime.execute_installed_request(
            state.record, b"snapshot", state.connection
        )
        assert result.outcome is (
            SandboxOutcome.INVALID
            if change in {"no-header", "no-header-overflow"}
            else SandboxOutcome.TIMED_OUT
            if change in {"timeout", "pre-entry-timeout"}
            else SandboxOutcome.CANCELLED
            if change == "pre-entry-cancel"
            else SandboxOutcome.COMPLETED
        )
        assert result.possible_effects
        assert len(result.stdout) <= 1_000_000
        if change in {"pre-entry-timeout", "pre-entry-cancel"}:
            assert (result.stdout, result.stderr) == ("bootstrap-out", "bootstrap-error")
            assert result.stdout_discarded is None
            assert result.stderr_discarded == 13
            assert result.exit_code is None
        elif change == "no-header":
            assert (result.stdout, result.stderr) == ("forged", "error")
        elif change == "no-header-overflow":
            assert result.stdout == "x" * 1_000_000
            assert result.stdout_discarded == state.capture.stdout_discarded + 80
        elif change == "unknown-discard":
            assert result.stdout_discarded is None
        elif change == "overflow":
            assert result.stdout_discarded == 9
        elif change is None:
            assert (result.stdout, result.stderr, result.exit_code) == ("output", "error", 17)
        descriptors = state.launch.call_args.kwargs["pass_fds"]
        assert descriptors[:3] == (77, 999, 999)
        assert len(descriptors) == 4 and isinstance(descriptors[3], int)
        state.supervisor.assert_called_once_with(
            state.launch.return_value, state.arguments[0], 50, connection=0
        )


def test_installed_execution_streams_only_payload_output_after_verified_entry(
    execution, monkeypatch
):
    """The receiver sends bounded framed streams without exposing its entry marker."""
    state = execution
    monkeypatch.setattr(linux_runtime, "MAX_OUTPUT_CHARS", 8)
    state.request.stream_output = True
    state.connection.send.side_effect = lambda frame, _flags: len(frame)
    marker = "LOOP_PAYLOAD_" + state.request.attempt_id + "\n"

    def supervise(*_args, output_observer, send_lock, **_kwargs):
        """Feed split entry and both command streams through the receiver callback."""
        assert send_lock is not None
        output_observer("stderr", "warning")
        output_observer("stderr", "overflow")
        output_observer("stderr", "discarded")
        output_observer("stdout", marker[:10])
        output_observer("stdout", marker[10:] + "hello")
        return state.capture

    state.supervisor.side_effect = supervise

    result = linux_runtime.execute_installed_request(state.record, b"snapshot", state.connection)

    assert result.outcome is SandboxOutcome.COMPLETED
    frames = [call.args[0] for call in state.connection.send.call_args_list]
    assert frames == [
        b"O2" + (7).to_bytes(4, "big") + b"warning",
        b"O2" + (1).to_bytes(4, "big") + b"o",
        b"O1" + (5).to_bytes(4, "big") + b"hello",
    ]


def test_installed_execution_owns_terminal_master_and_attests_ready(execution, monkeypatch):
    """Only the receiver retains the PTY master and sends readiness after payload entry."""
    state = execution
    state.request.terminal = True
    state.request.stream_output = True
    state.connection.send.side_effect = lambda frame, _flags: len(frame)
    original_fdopen = os.fdopen
    opened = []

    def fdopen(fd, *args, **kwargs):
        """Provide closeable synthetic streams for only the newly owned terminal fds."""
        if fd in (12, 13, 14):
            stream = io.BytesIO() if fd == 13 else io.StringIO()
            opened.append(stream)
            return stream
        return original_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(linux_runtime.os, "openpty", Mock(return_value=(12, 13)))
    monkeypatch.setattr(linux_runtime.os, "fdopen", fdopen)
    monkeypatch.setattr(linux_runtime.os, "pipe", Mock(return_value=(14, 15)))
    close = linux_runtime.os.close
    monkeypatch.setattr(linux_runtime.os, "close", lambda fd: None if fd == 15 else close(fd))
    monkeypatch.setattr(linux_runtime.termios, "tcgetattr", Mock(return_value=[0, 0, 0, 0]))
    set_attributes = Mock()
    monkeypatch.setattr(linux_runtime.termios, "tcsetattr", set_attributes)
    resize = Mock()
    monkeypatch.setattr(linux_runtime.fcntl, "ioctl", resize)
    marker = "LOOP_PAYLOAD_" + state.request.attempt_id + "\n"

    def supervise(*_args, output_observer, terminal_master, **_kwargs):
        """Authenticate the entry and publish one terminal output fragment."""
        assert terminal_master == 12
        output_observer("stdout", marker + "ready")
        return state.capture

    state.supervisor.side_effect = supervise
    result = linux_runtime.execute_installed_request(state.record, b"snapshot", state.connection)

    assert result.outcome is SandboxOutcome.COMPLETED
    assert state.launch.call_args.kwargs["stdout"] is opened[1]
    assert state.launch.call_args.kwargs["stderr"] is opened[1]
    assert state.launch.call_args.kwargs["pass_fds"][-1] != 12
    assert state.connection.sendall.call_args_list[-1].args == (b"T",)
    assert b"ready" in state.connection.send.call_args.args[0]
    assert all(stream.closed for stream in opened)
    set_attributes.assert_called_once()
    resize.assert_called_once()


def test_installed_execution_does_not_stream_setup_output_before_payload_entry(execution):
    """A missing native entry marker withholds both streams from live readers."""
    state = execution
    state.request.stream_output = True
    marker = "LOOP_PAYLOAD_" + state.request.attempt_id + "\n"

    def supervise(*_args, output_observer, **_kwargs):
        """Present setup output without a verified trusted payload marker."""
        output_observer("stderr", "setup diagnostic")
        output_observer("stdout", "invalid entry" + "x" * len(marker))
        output_observer("stderr", "more setup")
        output_observer("stdout", "later")
        return state.capture

    state.supervisor.side_effect = supervise
    linux_runtime.execute_installed_request(state.record, b"snapshot", state.connection)
    state.connection.send.assert_not_called()


@pytest.mark.parametrize("delivery", ["partial", "blocked", "disconnected", "expired", "oversize"])
def test_installed_execution_bounds_live_output_delivery(execution, monkeypatch, delivery):
    """The receiver handles short writes and refuses stalled or oversized live frames."""
    state = execution
    state.request.stream_output = True
    marker = "LOOP_PAYLOAD_" + state.request.attempt_id + "\n"
    writes = []

    def send(frame, _flags):
        """Simulate one delivery variation on the authenticated socket."""
        writes.append(frame)
        if delivery == "blocked" and len(writes) == 1:
            raise BlockingIOError()
        if delivery == "disconnected":
            return 0
        return 2 if delivery == "partial" and len(writes) == 1 else len(frame)

    state.connection.send.side_effect = send
    selected = Mock(return_value=([], [state.connection], []))
    monkeypatch.setattr(linux_runtime.select, "select", selected)

    def supervise(*_args, output_observer, **_kwargs):
        """Publish only after authentication so delivery bounds are exercised."""
        output_observer("stdout", marker)
        if delivery == "expired":
            monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 51)
        output_observer("stdout", "x" * (32769 if delivery == "oversize" else 1))
        return state.capture

    state.supervisor.side_effect = supervise
    if delivery in {"disconnected", "expired", "oversize"}:
        expected = {"disconnected": OSError, "expired": TimeoutError, "oversize": ValueError}
        with pytest.raises(expected[delivery]):
            linux_runtime.execute_installed_request(state.record, b"snapshot", state.connection)
    else:
        assert (
            linux_runtime.execute_installed_request(
                state.record, b"snapshot", state.connection
            ).outcome
            is SandboxOutcome.COMPLETED
        )
        assert len(writes) == 2
    if delivery == "blocked":
        selected.assert_called_once()


@pytest.mark.parametrize(
    ("events", "control", "outcome"),
    [
        ([], None, "launch"),
        ([(0, select.POLLIN)], b"", "launch"),
        ([(0, select.POLLIN)], b"\x00", "cancel"),
        ([(0, select.POLLHUP)], None, "cancel"),
        ([(0, select.POLLIN | select.POLLHUP)], b"", "cancel"),
        ([(0, select.POLLIN)], b"x", "invalid"),
        ([(0, select.POLLERR)], None, "error"),
        ([(0, select.POLLIN | select.POLLERR)], None, "error"),
        ([(0, select.POLLNVAL)], None, "error"),
        ([(0, select.POLLPRI)], None, "error"),
        ([(1, select.POLLIN)], None, "error"),
        ([(0, select.POLLIN), (0, select.POLLIN)], None, "error"),
    ],
)
def test_execute_installed_request_observes_queued_cancellation_before_spawn(
    execution, monkeypatch, events, control, outcome
):
    """Do not launch after queued cancellation; preserve request half-close behavior."""
    execution.poller.poll.return_value = events
    read = Mock(return_value=control)
    monkeypatch.setattr(linux_runtime.os, "read", read)

    if outcome in {"invalid", "error"}:
        with pytest.raises(ValueError if outcome == "invalid" else OSError):
            linux_runtime.execute_installed_request(
                execution.record, b"snapshot", execution.connection
            )
        execution.launch.assert_not_called()
    else:
        result = linux_runtime.execute_installed_request(
            execution.record, b"snapshot", execution.connection
        )
        if outcome == "cancel":
            assert result.outcome is SandboxOutcome.CANCELLED
            assert result.possible_effects is False
            execution.launch.assert_not_called()
            execution.supervisor.assert_not_called()
        else:
            assert result.outcome is SandboxOutcome.COMPLETED
            execution.launch.assert_called_once()
    execution.poller.register.assert_called_once_with(
        0, select.POLLIN | select.POLLHUP | select.POLLERR
    )
    execution.poller.poll.assert_called_once_with(0)
    if control is None:
        read.assert_not_called()
    else:
        read.assert_called_once_with(0, 1)


@pytest.mark.parametrize("error", [BlockingIOError("socket unavailable"), OSError("socket failed")])
def test_execute_installed_request_refuses_unreadable_queued_control(execution, monkeypatch, error):
    """An unreadable control stream cannot authorize a payload launch."""
    execution.poller.poll.return_value = [(0, select.POLLIN)]
    monkeypatch.setattr(linux_runtime.os, "read", Mock(side_effect=error))

    with pytest.raises(type(error), match=str(error)):
        linux_runtime.execute_installed_request(execution.record, b"snapshot", execution.connection)
    execution.launch.assert_not_called()
    execution.supervisor.assert_not_called()


@pytest.mark.parametrize("unknown", [False, True])
def test_execute_installed_request_preserves_path_output_with_bounded_capture(execution, unknown):
    """Preserve native output text and capture loss metadata in both streams."""
    from dataclasses import replace

    state = execution
    text = "/.loop-private-paths/r10/file /.loop-private-paths/r1/file /.loop-private-paths"
    state.supervisor.return_value = replace(
        state.capture,
        stdout="LOOP_PAYLOAD_" + state.request.attempt_id + "\n" + text,
        stderr=text + "x" * 1_000_000,
        stderr_discarded=None if unknown else 3,
    )
    result = linux_runtime.execute_installed_request(state.record, b"snapshot", state.connection)
    assert result.stdout == text
    assert result.stderr.startswith(result.stdout)
    assert len(result.stderr) == 1_000_000
    assert (
        result.stderr_discarded is None
        if unknown
        else result.stderr_discarded == 3 + len(result.stdout)
    )


def test_execute_installed_request_returns_typed_path_preparation_refusal(execution):
    """Return an authenticated invalid result when path preparation refuses before launch."""
    state = execution
    linux_runtime.namespace_arguments.side_effect = ValueError(
        "Linux hardlinked file aliases protected or private data."
    )

    result = linux_runtime.execute_installed_request(state.record, b"snapshot", state.connection)

    assert result.outcome is SandboxOutcome.INVALID
    assert result.detail == "Linux hardlinked file aliases protected or private data."
    assert not result.possible_effects
    assert result.failure_context == "Native sandbox path preparation was refused."
    state.launch.assert_not_called()
    state.supervisor.assert_not_called()


@pytest.mark.parametrize("entry", [False, True])
def test_execution_authority_requires_complete_trusted_entry(installation, monkeypatch, entry):
    """Ordinary preflight authenticates loaded authority without reconstructing or launching it."""
    record, arguments = installation
    if entry:
        entry_path = record.parent / "loop_payload.py"
        entry_path.write_bytes(b"entry")
        entry_path.chmod(0o444)
        arguments[-1]["loop_payload.py"] = linux_host.bind_host_file(entry_path)
        path = record.parent / "policies/root/bootstrap.bin"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"bootstrap")
        path.chmod(0o444)
        arguments[-1]["policies/root/bootstrap.bin"] = linux_host.bind_host_file(path)
    publish(installation)
    if entry:
        linux_runtime.execution_authority(record)
    else:
        with pytest.raises(ValueError, match="no trusted payload"):
            linux_runtime.execution_authority(record)
