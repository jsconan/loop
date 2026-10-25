"""Verify Linux runtime composition, typed results and selected-content isolation."""

import json
import os
import shlex
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.contracts import CommandProcessResult, SandboxOutcome, SandboxRequest
from loop.execution.sandbox.linux import backend as linux
from loop.execution.sandbox.linux.constants import NATIVE_ALIAS_PATH, NATIVE_SCRATCH_PATH
from loop.execution.sandbox.linux.policy import payload_profile
from loop.utils import VirtualPath
from loop.utils.process import ProcessCaptureStatus


@pytest.fixture
def backend(monkeypatch, tmp_path):
    """Isolate installed native collaborators and the shared monotonic budget."""
    monkeypatch.setattr(linux.time, "monotonic", lambda: 10)
    monkeypatch.setattr(linux, "execution_authority", Mock())
    monkeypatch.setattr(linux, "exchange_installed_request", Mock())
    record = tmp_path / "management.json"
    record.write_text("installed")
    return linux.LinuxNativeBackend(record)


def test_backend_describes_namespace_output_paths_without_rewriting_results(backend, tmp_path):
    """Namespace scratch and indexed aliases map to their logical command roots."""
    request = Mock(spec=SandboxRequest)
    request.aliases = tuple(
        (f"/skills/guide{index}", tmp_path / f"r{index}", tmp_path / f"skill{index}")
        for index in range(12)
    )

    roots = backend.output_path_roots(request)

    assert dict(roots) == {
        str(NATIVE_SCRATCH_PATH): VirtualPath.TEMPORARY,
        **{str(NATIVE_ALIAS_PATH / f"r{index}"): f"/skills/guide{index}" for index in range(12)},
    }
    assert [len(local) for local, _ in roots] == sorted(
        (len(local) for local, _ in roots), reverse=True
    )
    request.aliases = ()
    assert backend.output_path_roots(request) == (
        (str(NATIVE_SCRATCH_PATH), VirtualPath.TEMPORARY),
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


@pytest.mark.parametrize("outcome", list(SandboxOutcome))
def test_backend_restores_only_original_typed_result(backend, outcome, tmp_path):
    """Every accepted outcome preserves the original attempt and supervised fields."""
    request = Mock(deadline=20, attempt_id="1248c892-42dc-4b68-a3fb-30d2a70a93a7")
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
        tmp_path / "management.json", b"request", 20
    )


def test_backend_propagates_cancellation_through_installed_transport(backend, tmp_path):
    """A live cancellation signal uses the receiver control channel and cleanup acknowledgement."""
    request = Mock(deadline=20, attempt_id="1248c892-42dc-4b68-a3fb-30d2a70a93a7")
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
    )


def test_backend_does_not_open_transport_for_pre_cancelled_request(backend):
    """A run cancelled before native admission cannot send its command frame."""
    cancellation_event = Event()
    cancellation_event.set()

    result = backend.run(
        Mock(deadline=20),
        cancellation_event=cancellation_event,
    )

    assert result.outcome is SandboxOutcome.CANCELLED
    linux.exchange_installed_request.assert_not_called()


@pytest.mark.parametrize("failure", ["expired", "stale", "timeout", "cancel", "invalid", "io"])
def test_backend_failure_never_retries_or_discloses_authority(backend, failure):
    """Expiry, stale grants and ambiguous exchanges remain typed and never replay source."""
    request = Mock(deadline=10 if failure == "expired" else 20)
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
        if outcome is SandboxOutcome.UNAVAILABLE:
            assert result is None
        else:
            assert result.status is ProcessCaptureStatus(outcome.value)
            assert result.stdout == "output"
    assert os.lseek(descriptor, 0, os.SEEK_CUR) == 0
    assert all(not path.exists() for path in captured)


def test_search_reuses_exact_policy_with_disjoint_selected_backing(selected, monkeypatch):
    """Two native search preparations retain identical policy and different private snapshots."""
    backend, executable, descriptor, _ = selected
    workspaces, profiles = [], []

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
