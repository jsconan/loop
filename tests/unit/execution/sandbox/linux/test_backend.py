"""Verify Linux runtime composition, typed results and selected-content isolation."""

import json
import os
import shlex
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.contracts import (
    CommandProcessResult,
    SandboxCapability,
    SandboxOutcome,
    SandboxRequest,
)
from loop.execution.sandbox.linux import backend as linux
from loop.execution.sandbox.linux import policy as linux_policy
from loop.execution.sandbox.linux.constants import NATIVE_ALIAS_PATH, NATIVE_SCRATCH_PATH
from loop.execution.sandbox.linux.policy import payload_profile
from loop.utils import VirtualPath
from loop.utils.process import ProcessCaptureStatus

from ..helpers import logical_private_name


@pytest.fixture
def backend(monkeypatch, tmp_path):
    """Isolate installed native collaborators and the shared monotonic budget."""
    monkeypatch.setattr(linux.time, "monotonic", lambda: 10)
    socket = tmp_path / "admission.socket"
    monkeypatch.setattr(linux, "execution_authority", Mock(return_value=socket))
    monkeypatch.setattr(Path, "is_socket", lambda path: path == socket)
    monkeypatch.setattr(linux, "exchange_installed_request", Mock())
    observer = Mock()
    observer.finish.return_value = ("", "unknown")
    monkeypatch.setattr(linux, "LinuxDenialObserver", Mock(return_value=observer))
    record = tmp_path / "management.json"
    record.write_text("installed")
    return linux.LinuxNativeBackend(record)


def test_backend_describes_namespace_scratch_without_rewriting_results(backend):
    """Namespace scratch maps to the logical session root."""
    request = Mock(spec=SandboxRequest)

    assert backend.output_path_roots(request) == (
        (str(NATIVE_SCRATCH_PATH), VirtualPath.TEMPORARY),
    )


def test_backend_declares_nested_seccomp_without_advertising_namespace_controls(backend):
    """The installed backend reports only the narrower qualified inner restriction."""
    assert SandboxCapability.INNER_SECCOMP in backend.capabilities
    assert "inner_user_namespace" not in backend.capabilities


def test_backend_qualification_uses_installed_run_contract(backend, monkeypatch):
    """Generic selection probes Linux through the ordinary installed receiver path."""
    expected = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=1)
    run = Mock(return_value=expected)
    monkeypatch.setattr(backend, "run", run)
    probe = Mock(spec=SandboxRequest)
    assert backend.qualification_probe(probe) is expected
    backend.complete_qualification(probe)
    run.assert_called_once_with(probe)


def test_qualification_failure_identifies_denied_loopback_setup(backend):
    """Report the known native loopback failure without exposing arbitrary stderr."""
    result = CommandProcessResult(
        SandboxOutcome.INVALID,
        stderr="bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\nprivate data",
    )

    failure = backend.qualification_failure(result, "generic failure")

    assert failure.outcome is SandboxOutcome.INVALID
    assert "RTM_NEWADDR: Operation not permitted" in failure.detail
    assert "AppArmor" in failure.detail
    assert "private data" not in failure.detail


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), float("-inf"), True, "20"])
def test_backend_rejects_invalid_deadline_before_exchange(backend, deadline):
    """A forged deadline cannot start native admission or transport."""
    request = Mock(spec=SandboxRequest)
    request.deadline = deadline
    assert backend.run(request).outcome is SandboxOutcome.INVALID
    linux.exchange_installed_request.assert_not_called()


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("uid", [0, 1000])
def test_trusted_file_owners_require_initial_host_namespace(backend, monkeypatch, initial, uid):
    """Mapped namespace root never gains host administrator authority."""
    monkeypatch.setattr(linux.os, "getuid", lambda: uid)
    monkeypatch.setattr(linux, "in_initial_host_user_namespace", lambda: initial)
    expected = tuple(dict.fromkeys((0, uid))) if initial else (uid,)
    assert backend.trusted_file_owner_ids == expected


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("directory", [False, True])
def test_trusted_system_path_uses_protected_host_authority(
    backend, monkeypatch, tmp_path, initial, directory
):
    """System selections require initial-host and protected path validation."""
    monkeypatch.setattr(linux, "in_initial_host_user_namespace", lambda: initial)
    protected = Mock(return_value=tmp_path)
    monkeypatch.setattr(linux, "protected_path", protected)

    assert backend.trusted_system_path(tmp_path, directory=directory) is initial
    if initial:
        protected.assert_called_once_with(tmp_path, directory=directory)
    else:
        protected.assert_not_called()


def test_trusted_system_path_rejects_changed_or_unverifiable_authority(
    backend, monkeypatch, tmp_path
):
    """Changed canonical identity is refused and host inspection errors propagate."""
    monkeypatch.setattr(linux, "in_initial_host_user_namespace", lambda: True)
    protected = Mock(return_value=tmp_path / "other")
    monkeypatch.setattr(linux, "protected_path", protected)
    assert not backend.trusted_system_path(tmp_path)
    protected.side_effect = PermissionError("system path veto")
    with pytest.raises(PermissionError, match="system path veto"):
        backend.trusted_system_path(tmp_path)


@pytest.mark.parametrize("needed", [False, True])
def test_system_executable_alias_requirement_uses_protected_identity(
    backend, monkeypatch, tmp_path, needed
):
    """Alias preflight carries the selected executable identity into the native checker."""
    binary = tmp_path / "tool"
    binary.write_text("tool", encoding="utf-8")
    details = binary.stat()
    monkeypatch.setattr(linux, "runtime_read_roots", lambda: (tmp_path,))
    monkeypatch.setattr(linux, "protected_path", lambda path, **_kwargs: path)
    check = Mock(return_value=((tmp_path / "alias", binary),) if needed else ())
    monkeypatch.setattr(linux, "_runtime_executable_aliases", check)
    assert backend.system_executable_requires_alias(binary, binary, tmp_path / "work", ()) is needed
    request, bindings = check.call_args.args
    assert request.executable_identities == (
        ("tool", binary, binary, details.st_dev, details.st_ino),
    )
    assert bindings == ((tmp_path, tmp_path), (tmp_path / "work", tmp_path / "work"))


@pytest.mark.parametrize("requested", [False, True])
def test_system_executable_outside_runtime_requires_requested_scope(
    backend, monkeypatch, tmp_path, requested
):
    """An external absolute system tool is unavailable unless its directory was requested."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    tool = external / "tool"
    tool.write_text("tool", encoding="utf-8")
    monkeypatch.setattr(linux, "runtime_read_roots", lambda: (runtime,))
    monkeypatch.setattr(linux, "protected_path", lambda path, **_kwargs: path)
    check = Mock(return_value=())
    monkeypatch.setattr(linux, "_runtime_executable_aliases", check)
    assert (
        backend.system_executable_requires_alias(
            tool, tool, tmp_path / "workspace", (external,) if requested else ()
        )
        is not requested
    )
    assert check.called is requested


@pytest.mark.parametrize("include_alternatives", [False, True])
def test_requested_external_system_tool_checks_intermediate_mounts(
    backend, monkeypatch, tmp_path, include_alternatives, logical_alias_private_names
):
    """Approved endpoints do not imply that an intermediate alternatives link is mounted."""
    from loop.execution.sandbox.linux import namespace as linux_namespace

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    target = vendor / "target"
    target.write_text("tool", encoding="utf-8")
    alternatives = tmp_path / "alternatives"
    alternatives.mkdir()
    intermediate = alternatives / "tool"
    intermediate.symlink_to(target)
    spelling = vendor / "tool"
    spelling.symlink_to(intermediate)
    monkeypatch.setattr(linux, "runtime_read_roots", lambda: (runtime,))
    monkeypatch.setattr(linux_namespace, "_SYSTEM_ALTERNATIVES", alternatives)

    def protected(path, **_kwargs):
        """Resolve the synthetic root-owned fixture without requiring real system paths."""
        return path.resolve(strict=True)

    monkeypatch.setattr(linux, "protected_path", protected)
    monkeypatch.setattr(linux_namespace, "protected_path", protected)
    roots = (vendor, alternatives) if include_alternatives else (vendor,)
    assert (
        backend.system_executable_requires_alias(spelling, target, tmp_path / "workspace", roots)
        is not include_alternatives
    )


def test_backend_properties_and_safe_read_candidates(backend, monkeypatch):
    """Preserve root provenance and give external tools only narrow candidates."""
    monkeypatch.setattr(linux, "in_initial_host_user_namespace", lambda: True)
    monkeypatch.setattr(linux, "runtime_read_roots", lambda: (Path("/runtime"),))
    protected = Mock(side_effect=lambda path, **kwargs: path)
    monkeypatch.setattr(linux, "protected_path", protected)
    assert backend.system_read_roots == (Path("/runtime"),)
    assert backend.policy_version == "linux-native-v1"
    assert backend.scratch_prefix == "loop-sandbox-"
    assert backend.managed_tool_root(Path("/runtime/bin"))
    assert backend.managed_tool_root(Path("/other"))
    protected.return_value = Path("/changed")
    protected.side_effect = None
    assert not backend.managed_tool_root(Path("/runtime/bin"))
    protected.side_effect = OSError("unsafe directory")
    assert not backend.managed_tool_root(Path("/runtime/bin"))
    assert backend.installed_tool_roots(Path("/other/bin/tool")) == ((Path("/other/bin"),), ())
    assert backend.capability_failure() is None
    linux.execution_authority.side_effect = ValueError("private authority details")
    failure = backend.capability_failure()
    assert failure.outcome is SandboxOutcome.STALE
    assert "private authority" not in failure.detail
    assert "refresh" in failure.detail


def test_managed_tool_root_refuses_mapped_user_namespace(backend, monkeypatch):
    """A mapped caller-owned directory cannot masquerade as a host-root installation."""
    monkeypatch.setattr(linux.sys, "platform", "linux")
    monkeypatch.setattr(linux, "in_initial_host_user_namespace", lambda: False)
    monkeypatch.setattr(linux, "protected_path", Mock(return_value=Path("/runtime/bin")))

    assert not backend.managed_tool_root(Path("/runtime/bin"))
    linux.protected_path.assert_not_called()


def test_initial_host_user_namespace_requires_full_identity_mapping(monkeypatch):
    """Reject missing or nested UID maps before trusting host-root ownership."""
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: "0 0 4294967295\n",
    )
    assert linux.in_initial_host_user_namespace()
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: "0 1000 1\n")
    assert not linux.in_initial_host_user_namespace()

    def missing(*_args, **_kwargs):
        """Model an unavailable proc mapping without relying on the host proc mount."""
        raise OSError("missing uid map")

    monkeypatch.setattr(Path, "read_text", missing)
    assert not linux.in_initial_host_user_namespace()


def test_backend_reads_live_file_creation_mask(backend, monkeypatch):
    """Read the process mask without mutating it and reject ambiguous status."""
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: "Name:\tpython\nUmask:\t0077\n",
    )
    assert backend.file_creation_mask == 0o077
    for value, message in (
        ("Name:\tpython\n", "unavailable"),
        ("Umask:\tinvalid\n", "invalid"),
        ("Umask:\t1000\n", "invalid"),
    ):
        monkeypatch.setattr(Path, "read_text", lambda *_args, value=value, **_kwargs: value)
        with pytest.raises(OSError, match=f"umask is {message}"):
            _ = backend.file_creation_mask


@pytest.mark.parametrize(
    "outcome", [item for item in SandboxOutcome if item is not SandboxOutcome.UNSUPPORTED]
)
def test_backend_restores_only_original_typed_result(backend, outcome, tmp_path):
    """Every accepted outcome preserves the original attempt and supervised fields."""
    request = Mock(terminal=False, deadline=20, attempt_id="1248c892-42dc-4b68-a3fb-30d2a70a93a7")
    request.to_wire.return_value = b"request"
    result = CommandProcessResult(
        outcome,
        exit_code=17 if outcome is SandboxOutcome.COMPLETED else None,
        stdout=str(NATIVE_SCRATCH_PATH / "selected/source"),
        stderr=str(NATIVE_ALIAS_PATH / "r0/diagnostic"),
    )
    linux.exchange_installed_request.return_value = result.to_wire(request.attempt_id)
    assert backend.run(request) == result
    linux.exchange_installed_request.assert_called_once_with(
        tmp_path / "management.json",
        b"request",
        20,
        observer=linux.LinuxDenialObserver.return_value,
    )


def test_backend_streaming_uses_only_original_authenticated_exchange(backend, tmp_path):
    """A live observer sets the optional wire flag and stays on the original transport."""
    request = SandboxRequest.create(
        source=":",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin"},
        policy_version="linux-v1",
        deadline=20,
        workspace_id="workspace",
    )
    observer = Mock()
    expected = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    linux.exchange_installed_request.return_value = expected.to_wire(request.attempt_id)

    assert backend.run(request, output_observer=observer) == expected
    arguments = linux.exchange_installed_request.call_args
    assert SandboxRequest.from_wire(arguments.args[1]).stream_output
    assert arguments.kwargs["output_observer"] is observer


def test_backend_terminal_requires_owned_input_and_output_channels(backend, tmp_path):
    """A direct interactive request cannot run with a missing or mismatched control channel."""
    request = SandboxRequest.create(
        source=":",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin"},
        policy_version="linux-v1",
        deadline=20,
        workspace_id="workspace",
        terminal=True,
    )
    input_frame = Mock(return_value=None)
    assert backend.run(request).outcome is SandboxOutcome.INVALID
    assert backend.run(request, terminal_input=input_frame).outcome is SandboxOutcome.INVALID
    assert (
        backend.run(replace(request, terminal=False), terminal_input=input_frame).outcome
        is SandboxOutcome.INVALID
    )
    linux.exchange_installed_request.assert_not_called()


def test_backend_terminal_forwards_only_owned_stream_controls(backend, tmp_path):
    """An interactive request keeps its input callback and output observer on one exchange."""
    request = SandboxRequest.create(
        source=":",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin"},
        policy_version="linux-v1",
        deadline=20,
        workspace_id="workspace",
        terminal=True,
    )
    input_frame = Mock(return_value=None)
    observer = Mock()
    expected = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    linux.exchange_installed_request.return_value = expected.to_wire(request.attempt_id)

    assert backend.run(request, terminal_input=input_frame, output_observer=observer) == expected
    arguments = linux.exchange_installed_request.call_args
    sent = SandboxRequest.from_wire(arguments.args[1])
    assert sent.terminal and sent.stream_output
    assert arguments.kwargs["terminal_input"] is input_frame
    assert arguments.kwargs["output_observer"] is observer


def test_backend_terminal_refuses_mediated_egress(backend, tmp_path):
    """A terminal cannot silently share the egress lease control channel."""
    request = SandboxRequest.create(
        source=":",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin"},
        policy_version="linux-v1",
        deadline=20,
        workspace_id="workspace",
        terminal=True,
        egress_origins=("https://example.com",),
    )
    assert (
        backend.run(request, terminal_input=Mock(), output_observer=Mock()).outcome
        is SandboxOutcome.UNSUPPORTED
    )
    linux.exchange_installed_request.assert_not_called()


def test_backend_marks_cleanup_only_after_authenticated_receiver_result(backend):
    """Only an explicit authenticated verdict proves cleanup, including on cancellation."""
    request = Mock(terminal=False, deadline=20, attempt_id="1248c892-42dc-4b68-a3fb-30d2a70a93a7")
    request.to_wire.return_value = b"request"
    native = CommandProcessResult(
        SandboxOutcome.CANCELLED,
        possible_effects=True,
        stdout_discarded=None,
        cleanup_verified=True,
    )
    linux.exchange_installed_request.return_value = native.to_wire(
        request.attempt_id, include_cleanup_proof=True
    )
    assert backend.run(request).cleanup_verified is True

    uncertain = CommandProcessResult(
        SandboxOutcome.TIMED_OUT, possible_effects=True, stdout_discarded=None
    )
    linux.exchange_installed_request.return_value = uncertain.to_wire(request.attempt_id)
    assert backend.run(request).cleanup_verified is None


def test_backend_requires_approving_authority_for_mediated_egress(backend):
    """A direct backend caller cannot start mediation without a bound policy revision."""
    request = Mock(terminal=False, deadline=20)
    request.egress_origins = ("http://example.com",)
    assert backend.run(request).outcome is SandboxOutcome.UNSUPPORTED
    linux.exchange_installed_request.assert_not_called()


def test_backend_passes_exact_mediated_lease_to_original_transport(backend, tmp_path):
    """The approved origin attempt carries its manager and revision only on its own exchange."""
    request = Mock(terminal=False, deadline=20, attempt_id="1248c892-42dc-4b68-a3fb-30d2a70a93a7")
    request.egress_origins = ("http://example.com",)
    request.to_wire.return_value = b"request"
    result = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    linux.exchange_installed_request.return_value = result.to_wire(request.attempt_id)
    authority = Mock()
    assert backend.run(request, policy_authority=authority, approved_revision=7) == result
    arguments = linux.exchange_installed_request.call_args
    assert arguments.args == (tmp_path / "management.json", b"request", 20)
    assert arguments.kwargs["egress_lease"].authority is authority
    assert arguments.kwargs["egress_lease"].revision == 7


@pytest.mark.parametrize("outcome", [SandboxOutcome.COMPLETED, SandboxOutcome.TIMED_OUT])
def test_backend_adds_only_local_observations_to_completed_results(backend, outcome):
    """Receiver results remain authoritative while local evidence stays unattributed."""
    request = Mock(terminal=False, deadline=20, attempt_id="1248c892-42dc-4b68-a3fb-30d2a70a93a7")
    request.to_wire.return_value = b"request"
    result = CommandProcessResult(
        outcome, exit_code=7 if outcome is SandboxOutcome.COMPLETED else None
    )
    linux.exchange_installed_request.return_value = result.to_wire(request.attempt_id)
    linux.LinuxDenialObserver.return_value.finish.return_value = ("apparmor-open /x", "available")

    actual = backend.run(request)

    assert actual.outcome == outcome
    if outcome is SandboxOutcome.COMPLETED:
        assert actual.observed_denial == "apparmor-open /x"
        assert actual.diagnostic_source == "linux-kernel-journal"
        assert actual.diagnostic_status == "available"
    else:
        assert actual.observed_denial == ""
        assert actual.diagnostic_source == actual.diagnostic_status == "unknown"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("apparmor-open /one", (("apparmor-open /one", False),)),
        (linux.DENIAL_RECORDS_PREFIX + '["one","two"]', (("one", False), ("two", False))),
        (linux.DENIAL_RECORDS_PREFIX + "bad", ()),
        (linux.DENIAL_RECORDS_PREFIX + "{}", ()),
        (linux.DENIAL_RECORDS_PREFIX + "[]", ()),
        (linux.DENIAL_RECORDS_PREFIX + '["x","x","x","x","x"]', ()),
        (linux.DENIAL_RECORDS_PREFIX + '[""]', ()),
        (linux.DENIAL_RECORDS_PREFIX + "[3]", ()),
        (linux.DENIAL_RECORDS_PREFIX + json.dumps(["x" * 501]), ()),
    ],
)
def test_backend_classifies_bounded_unverified_journal_records(backend, payload, expected):
    """Journal records have no startup baseline and malformed envelopes disclose nothing."""
    assert backend.classify_denial_candidates(payload) == (expected, "unavailable")


@pytest.mark.parametrize(
    ("diagnostic_status", "diagnostic_source", "observed_denial"),
    [
        ("available", "unknown", ""),
        ("unavailable", "unknown", ""),
        ("available", "macos-kernel-log", "file-read-data /outside"),
        ("unknown", "unknown", "file-read-data /outside"),
    ],
)
def test_backend_rejects_unsupported_diagnostic_provenance(
    backend, diagnostic_status, diagnostic_source, observed_denial
):
    """Linux cannot claim a macOS diagnostic source or unimplemented collection status."""
    request = Mock(terminal=False, deadline=20, attempt_id="1248c892-42dc-4b68-a3fb-30d2a70a93a7")
    request.to_wire.return_value = b"request"
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=1,
        diagnostic_status=diagnostic_status,
        diagnostic_source=diagnostic_source,
        observed_denial=observed_denial,
    )
    linux.exchange_installed_request.return_value = result.to_wire(request.attempt_id)

    outcome = backend.run(request)

    assert outcome.outcome is SandboxOutcome.INVALID
    assert outcome.possible_effects
    assert outcome.diagnostic_status == outcome.diagnostic_source == "unknown"
    assert outcome.observed_denial == ""
    assert outcome.stdout_discarded is outcome.stderr_discarded is None
    linux.exchange_installed_request.assert_called_once()


def test_backend_propagates_cancellation_through_installed_transport(backend, tmp_path):
    """A live cancellation signal uses the receiver control channel and cleanup acknowledgement."""
    request = Mock(terminal=False, deadline=20, attempt_id="1248c892-42dc-4b68-a3fb-30d2a70a93a7")
    request.to_wire.return_value = b"request"
    result = CommandProcessResult(SandboxOutcome.CANCELLED, possible_effects=True)
    linux.exchange_installed_request.return_value = result.to_wire(request.attempt_id)
    cancellation_event = Event()

    assert backend.run(request, cancellation_event=cancellation_event) == result
    linux.exchange_installed_request.assert_called_once_with(
        tmp_path / "management.json",
        b"request",
        20,
        cancellation_event=cancellation_event,
        observer=linux.LinuxDenialObserver.return_value,
    )


def test_backend_does_not_open_transport_for_pre_cancelled_request(backend):
    """A run cancelled before native admission cannot send its command frame."""
    cancellation_event = Event()
    cancellation_event.set()

    result = backend.run(
        Mock(terminal=False, deadline=20),
        cancellation_event=cancellation_event,
    )

    assert result.outcome is SandboxOutcome.CANCELLED
    linux.exchange_installed_request.assert_not_called()


def test_backend_does_not_open_transport_when_cancelled_during_serialization(backend):
    """Cancellation while building the request prevents any native transport or effects."""
    cancellation_event = Event()
    request = Mock(terminal=False, deadline=20)

    def serialize():
        """Cancel before the request bytes become available."""
        cancellation_event.set()
        return b"request"

    request.to_wire.side_effect = serialize

    result = backend.run(request, cancellation_event=cancellation_event)

    assert result == CommandProcessResult(SandboxOutcome.CANCELLED)
    request.to_wire.assert_called_once_with()
    linux.exchange_installed_request.assert_not_called()


@pytest.mark.parametrize(
    "error,outcome",
    [
        (ValueError, SandboxOutcome.INVALID),
        (TimeoutError, SandboxOutcome.TIMED_OUT),
        (OSError, SandboxOutcome.INVALID),
        (KeyboardInterrupt, SandboxOutcome.CANCELLED),
    ],
)
def test_backend_reports_pretransport_failures_without_possible_effects(backend, error, outcome):
    """A serialization failure cannot have launched a command or lost its output."""
    request = Mock(terminal=False, deadline=20)
    request.to_wire.side_effect = error("private host path")

    result = backend.run(request)

    assert result.outcome is outcome
    assert not result.possible_effects
    assert result.stdout_discarded == result.stderr_discarded == 0
    assert "private host path" not in result.detail
    linux.exchange_installed_request.assert_not_called()


@pytest.mark.parametrize("failure", ["expired", "stale", "timeout", "cancel", "invalid", "io"])
def test_backend_failure_never_retries_or_discloses_authority(backend, failure):
    """Expiry, stale grants and ambiguous exchanges remain typed and never replay source."""
    request = Mock(terminal=False, deadline=10 if failure == "expired" else 20)
    request.paths_are_current.return_value = failure != "stale"
    request.to_wire.return_value = b"request"
    error = {
        "timeout": TimeoutError,
        "cancel": KeyboardInterrupt,
        "invalid": ValueError,
        "io": OSError,
    }
    if failure in error:
        linux.exchange_installed_request.side_effect = error[failure]("private host path")
    result = backend.run(request)
    assert (
        result.outcome
        is {
            "expired": SandboxOutcome.TIMED_OUT,
            "stale": SandboxOutcome.STALE,
            "timeout": SandboxOutcome.TIMED_OUT,
            "cancel": SandboxOutcome.CANCELLED,
            "invalid": SandboxOutcome.INVALID,
            "io": SandboxOutcome.INVALID,
        }[failure]
    )
    assert "private host path" not in result.detail
    if failure in {"timeout", "cancel", "invalid", "io"}:
        assert result.possible_effects
        assert result.stdout_discarded is result.stderr_discarded is None
    assert linux.exchange_installed_request.call_count == (failure not in {"expired", "stale"})


@pytest.fixture
def selected(backend, monkeypatch, tmp_path):
    """Supply real disposable selected content and a bound fake installed executable."""
    monkeypatch.setattr(linux, "in_initial_host_user_namespace", lambda: True)
    root = tmp_path / "runtime"
    root.mkdir()
    executable = root / "rg"
    executable.write_bytes(b"fake installed helper")
    executable.chmod(0o755)
    details = executable.stat()
    monkeypatch.setattr(linux, "runtime_read_roots", lambda: (root,))
    monkeypatch.setattr(linux, "protected_path", lambda path, **kwargs: path)
    monkeypatch.setattr(
        linux,
        "bind_host_file",
        lambda path: SimpleNamespace(identity=(details.st_dev, details.st_ino)),
    )
    source = tmp_path / "selected"
    source.write_bytes(b"private selected content")
    descriptor = os.open(source, os.O_RDONLY)
    yield backend, executable, descriptor, source
    os.close(descriptor)


@pytest.mark.parametrize(
    "outcome",
    [
        SandboxOutcome.COMPLETED,
        SandboxOutcome.TIMED_OUT,
        SandboxOutcome.CANCELLED,
        SandboxOutcome.UNAVAILABLE,
        SandboxOutcome.UNSUPPORTED,
        SandboxOutcome.INVALID,
        SandboxOutcome.STALE,
    ],
)
def test_search_preserves_selected_snapshot_and_typed_outcome(selected, monkeypatch, outcome):
    """Search private selected copies while preserving caller descriptors and offsets."""
    backend, executable, descriptor, source = selected
    captured = []

    def run(request):
        """Inspect the prepared public request while its disposable inputs remain owned."""
        assert request.write_roots == () and request.network is False
        assert next(iter(request.workspace.iterdir())).read_bytes() == source.read_bytes()
        assert str(source) not in request.execution_source
        invocation = shlex.split(request.execution_source)
        selected = json.loads(invocation[-1])
        assert selected["arguments"] == ["--json", f"/dev/fd/{descriptor}"]
        assert selected["files"] == [
            [descriptor, str(NATIVE_SCRATCH_PATH / "selected" / str(descriptor))]
        ]
        assert invocation[:4] == ["cd", str(NATIVE_SCRATCH_PATH / "selected"), "&&", "exec"]
        assert request.executable_identities[0][2] == executable
        captured.append(request.workspace)
        return CommandProcessResult(
            outcome, exit_code=2 if outcome is SandboxOutcome.COMPLETED else None, stdout="output"
        )

    monkeypatch.setattr(backend, "run", run)
    if outcome in {SandboxOutcome.INVALID, SandboxOutcome.STALE}:
        with pytest.raises(RuntimeError, match="Native search authority"):
            backend.run_read_only_argv(
                executable, ("--json", f"/dev/fd/{descriptor}"), (descriptor,), 20
            )
    else:
        result = backend.run_read_only_argv(
            executable, ("--json", f"/dev/fd/{descriptor}"), (descriptor,), 20
        )
        if outcome in {SandboxOutcome.UNAVAILABLE, SandboxOutcome.UNSUPPORTED}:
            assert result is None
        else:
            assert result.status is ProcessCaptureStatus(outcome.value)
            assert result.stdout == "output"
    assert os.lseek(descriptor, 0, os.SEEK_CUR) == 0
    assert all(not path.exists() for path in captured)


def test_search_never_retries_unsupported_result_with_possible_effects(selected, monkeypatch):
    """A possibly executed search cannot trigger the in-process fallback."""
    backend, executable, descriptor, _ = selected
    monkeypatch.setattr(
        backend,
        "run",
        lambda _request: CommandProcessResult(SandboxOutcome.UNSUPPORTED, possible_effects=True),
    )
    with pytest.raises(RuntimeError, match="may already have had effects"):
        backend.run_read_only_argv(
            executable, ("--json", f"/dev/fd/{descriptor}"), (descriptor,), 20
        )


def test_search_reuses_exact_policy_with_disjoint_selected_backing(selected, monkeypatch, tmp_path):
    """Two native search preparations retain identical policy and different private snapshots."""
    backend, executable, descriptor, _ = selected
    workspaces, profiles = [], []
    case_root = tmp_path.parent

    def protected(candidate):
        """Model the owned physical case root as an ordinary logical runtime parent."""
        return logical_private_name(candidate, case_root)

    assert protected(case_root / "runtime/.ssh")
    monkeypatch.setattr(linux_policy, "is_protected_external_read", protected)

    def run(request):
        """Observe policy identity while the caller owns a fresh selected-file snapshot."""
        workspaces.append(request.workspace)
        profiles.append(
            payload_profile(
                request, request.workspace.parent, Path("/run/cache.lock"), (executable.parent,)
            )
        )
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)

    monkeypatch.setattr(backend, "run", run)
    for _ in range(2):
        assert (
            backend.run_read_only_argv(
                executable, (f"/dev/fd/{descriptor}",), (descriptor,), 20
            ).exit_code
            == 0
        )
    assert profiles[0] == profiles[1]
    assert workspaces[0] != workspaces[1]
    assert all(not path.exists() for path in workspaces)


@pytest.mark.parametrize(
    "failure",
    [
        "expired",
        "untrusted",
        "empty",
        "duplicate",
        "type",
        "range",
        "arguments",
        "nonregular",
        "linked",
        "short",
        "drift",
        "late",
        "create_timeout",
        "create_io",
        "lookup_exit",
    ],
)
def test_search_refuses_unsafe_inputs_or_prelaunch_failure(selected, monkeypatch, failure):
    """Unsafe descriptor authority and bounded preparation failures never expose host content."""
    backend, executable, descriptor, _source = selected
    descriptors = (descriptor,)
    arguments = (f"/dev/fd/{descriptor}",)
    deadline = 10 if failure == "expired" else 20
    if failure == "untrusted":
        monkeypatch.setattr(backend, "managed_tool_root", lambda root: False)
    elif failure in {"empty", "duplicate", "type", "range"}:
        descriptors = {
            "empty": (),
            "duplicate": (descriptor, descriptor),
            "type": (True,),
            "range": (4097,),
        }[failure]
    elif failure == "arguments":
        arguments = ("/unselected/path",)
    elif failure in {"nonregular", "linked", "drift"}:
        native_fstat = os.fstat
        real = native_fstat(descriptor)
        fields = {
            key: getattr(real, key)
            for key in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        }
        if failure == "nonregular":
            fields["st_mode"] = 0o040700
        elif failure == "linked":
            fields["st_nlink"] = 2
        else:
            fields["st_size"] += 1
        mock = Mock(return_value=SimpleNamespace(**fields))
        if failure == "drift":
            mock.side_effect = [real, SimpleNamespace(**fields)]
        monkeypatch.setattr(
            linux.os, "fstat", lambda fd: mock(fd) if fd == descriptor else native_fstat(fd)
        )
    elif failure == "short":
        monkeypatch.setattr(linux.os, "pread", Mock(return_value=b""))
    elif failure == "late":
        monkeypatch.setattr(linux.time, "monotonic", Mock(side_effect=[10, 20]))
    elif failure in {"create_timeout", "create_io"}:
        monkeypatch.setattr(
            linux.SandboxRequest,
            "create",
            Mock(side_effect=TimeoutError() if failure == "create_timeout" else OSError()),
        )
    monkeypatch.setattr(
        backend,
        "run",
        Mock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=127)),
    )
    if failure in {"drift", "short"}:
        with pytest.raises(RuntimeError, match="Native search preparation authority"):
            backend.run_read_only_argv(executable, arguments, descriptors, deadline)
    else:
        result = backend.run_read_only_argv(executable, arguments, descriptors, deadline)
        if failure in {"expired", "late", "create_timeout"}:
            assert result.status is ProcessCaptureStatus.TIMED_OUT
        else:
            assert result is None


@pytest.mark.parametrize("existing", [False, True])
def test_missing_setup_is_distinct_from_partial_authority(tmp_path, monkeypatch, existing):
    """Only absent setup can offer recovery; a partial installation is invalid."""
    record = tmp_path / "native" / "management.json"
    if existing:
        record.parent.mkdir()
        (record.parent / "loop_receiver.py").write_bytes(b"partial authority")
    monkeypatch.setattr(linux, "protected_path", lambda path, **_: path.resolve(strict=True))
    native = linux.LinuxNativeBackend(record)
    failure = native.capability_failure()
    assert failure.outcome is (SandboxOutcome.INVALID if existing else SandboxOutcome.UNAVAILABLE)


@pytest.mark.parametrize("error", [ValueError, OSError, FileNotFoundError])
def test_installed_authority_failures_are_invalid_before_first_success(backend, error):
    """Unavailable files within a present installation never imply fresh missing setup."""
    linux.execution_authority.side_effect = error("private installation details")
    failure = backend.capability_failure()
    assert failure.outcome is SandboxOutcome.INVALID
    assert "private installation details" not in failure.detail


@pytest.mark.parametrize(
    "reason",
    [
        "Installed Linux sandbox needs /sandbox refresh.",
        "System Bubblewrap changed; use /sandbox refresh.",
    ],
)
def test_checked_repair_is_reported_before_admission(backend, reason):
    """A mismatched installation or changed system helper gets a direct repair instruction."""
    linux.execution_authority.side_effect = linux.NativeRepairRequired(reason)
    failure = backend.capability_failure()
    assert failure.outcome is SandboxOutcome.INVALID
    assert failure.detail == reason
    linux.exchange_installed_request.assert_not_called()


def test_incompatible_installation_refusal_before_transport_has_no_effects(backend):
    """A direct request detects mismatched authority before any receiver can execute it."""
    request = Mock(terminal=False, deadline=20)
    request.paths_are_current.return_value = True
    request.to_wire.return_value = b"request"
    linux.exchange_installed_request.side_effect = linux.NativeRepairRequired(
        "Installed Linux sandbox needs /sandbox refresh."
    )
    result = backend.run(request)
    assert result.outcome is SandboxOutcome.INVALID
    assert result.detail == "Installed Linux sandbox needs /sandbox refresh."
    assert not result.possible_effects


def test_unsupported_wire_feature_is_unavailable_before_effects(backend):
    """A request rejected by retained capability negotiation never claims command effects."""
    request = Mock(terminal=False, deadline=20)
    request.paths_are_current.return_value = True
    request.to_wire.return_value = b"request"
    linux.exchange_installed_request.side_effect = linux.NativeFeatureUnavailable(
        "Installed Linux sandbox cannot support this request schema."
    )
    result = backend.run(request)
    assert result.outcome is SandboxOutcome.UNSUPPORTED
    assert result.detail == "Installed Linux sandbox cannot support this request schema."
    assert not result.possible_effects


def test_capability_discovery_reports_unsupported_wire_feature(backend):
    """Preflight returns a specific unavailable feature without suggesting host execution."""
    linux.execution_authority.side_effect = linux.NativeFeatureUnavailable("Unavailable feature.")
    result = backend.capability_failure()
    assert result.outcome is SandboxOutcome.UNSUPPORTED
    assert result.detail == "Unavailable feature."


def test_missing_admission_socket_is_not_ready(backend, monkeypatch):
    """Intact saved files alone cannot mark the backend ready without its listener."""
    monkeypatch.setattr(Path, "is_socket", lambda _: False)
    assert backend.capability_failure().outcome is SandboxOutcome.INVALID


@pytest.mark.parametrize("state", ["activating", "active"])
def test_capability_waits_for_authenticated_boot_without_retrying_command(
    backend, monkeypatch, tmp_path, state
):
    """A caller joins boot activation and only returns ready after restored authority validates."""
    monkeypatch.setattr(linux, "_RECORD", tmp_path / "management.json")
    linux.execution_authority.side_effect = [OSError("booting"), tmp_path / "admission.socket"]
    states = ["activating", "active"] if state == "activating" else ["active"]
    monkeypatch.setattr(linux, "boot_activation_state", Mock(side_effect=states))
    sleep = Mock()
    monkeypatch.setattr(linux.time, "sleep", sleep)
    assert backend.capability_failure() is None
    assert linux.execution_authority.call_count == 2
    assert sleep.call_count == (1 if state == "activating" else 0)


@pytest.mark.parametrize(
    "failure", ["inactive", "failed", "active_invalid", "state_error", "deadline"]
)
def test_capability_refuses_when_boot_does_not_safely_complete(
    backend, monkeypatch, tmp_path, failure
):
    """Unavailable, failed, ambiguous or expired activation remains a fail-closed result."""
    monkeypatch.setattr(linux, "_RECORD", tmp_path / "management.json")
    linux.execution_authority.side_effect = OSError("not ready")
    state = Mock(
        side_effect=OSError("system manager unavailable") if failure == "state_error" else None,
        return_value=(
            "active"
            if failure == "active_invalid"
            else "activating"
            if failure == "deadline"
            else failure
        ),
    )
    monkeypatch.setattr(linux, "boot_activation_state", state)
    if failure == "deadline":
        values = iter((10, 140))
        monkeypatch.setattr(linux.time, "monotonic", lambda: next(values))
    result = backend.capability_failure()
    assert result.outcome is SandboxOutcome.INVALID


def test_removed_observed_installation_is_stale(backend, tmp_path):
    """Removing previously authenticated authority cannot restore host retry eligibility."""
    assert backend.capability_failure() is None
    (tmp_path / "management.json").unlink()
    assert backend.capability_failure().outcome is SandboxOutcome.STALE


def test_missing_setup_under_aliased_ancestry_is_invalid(tmp_path, monkeypatch):
    """A missing record below a dangling or redirected ancestor cannot become a host offer."""
    record = tmp_path / "missing" / "management.json"

    def protect(path, **_):
        """Return redirected authority only for the nearest existing parent."""
        if path == record.parent:
            raise FileNotFoundError("missing")
        return path / "other"

    monkeypatch.setattr(linux, "protected_path", protect)
    assert linux.LinuxNativeBackend(record).capability_failure().outcome is SandboxOutcome.INVALID


@pytest.mark.parametrize(
    "change", [None, "alias", "mode", "bytes", "reservations", "reservation_alias"]
)
def test_uninstalled_scaffolding_requires_exact_safe_authority(tmp_path, monkeypatch, change):
    """Only exact idle scaffolding after uninstall can count as genuine missing setup."""
    from contextlib import nullcontext

    directory = tmp_path / "native"
    directory.mkdir()
    reservations = directory / "reservations"
    reservations.mkdir()
    if change == "reservations":
        (reservations / "retained").write_bytes(b"retained capacity")
    lock = directory / "management.lock"
    lock.write_bytes(linux.NATIVE_LOCK_SEEDS[lock.name])
    bound = Mock(identity=(1, 2, 0o444 | (0o200 if change == "mode" else 0)))
    bound.open_verified.return_value = nullcontext(4)
    monkeypatch.setattr(linux, "bind_host_file", Mock(return_value=bound))
    monkeypatch.setattr(
        linux.os, "pread", lambda *_: b"changed" if change == "bytes" else lock.read_bytes()
    )

    def protect(path, **_):
        """Resolve a supplied original path or simulate one changed canonical target."""
        if (change == "alias" and path == directory) or (
            change == "reservation_alias" and path == reservations
        ):
            return path / "other"
        return path

    monkeypatch.setattr(linux, "protected_path", protect)
    result = linux.LinuxNativeBackend(directory / "management.json").capability_failure()
    assert result.outcome is (
        SandboxOutcome.UNAVAILABLE if change is None else SandboxOutcome.INVALID
    )


def test_empty_protected_setup_directory_is_genuinely_unavailable(tmp_path, monkeypatch):
    """An empty authenticated setup directory has no partial authority to invalidate."""
    monkeypatch.setattr(linux, "protected_path", lambda path, **_: path)
    result = linux.LinuxNativeBackend(tmp_path / "management.json").capability_failure()
    assert result.outcome is SandboxOutcome.UNAVAILABLE
