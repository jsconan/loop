"""Verify bounded loaded-unit attestation and refusal paths."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import preflight as linux_preflight
from loop.utils.process import ProcessCapture, ProcessCaptureStatus


@pytest.fixture
def preflight(monkeypatch):
    """Isolate protected authority and native processes at their public boundaries."""

    @contextmanager
    def verified(**kwargs):
        """Model an authenticated input descriptor with scoped lifetime."""
        yield 99

    def authority(path, **kwargs):
        """Return a bound installed-input double."""
        return SimpleNamespace(path=Path(path), open_verified=Mock(side_effect=verified))

    state = SimpleNamespace(
        files={
            "socket_unit": authority("/installed/loop.socket"),
            "service_unit": authority("/installed/loop@.service"),
        },
        expectations={
            "loop.socket": {"Listen": "/run/loop.socket (Stream)"},
            "loop@1000.service": {"ExecStart": "fixed receiver", "User": "", "Group": ""},
        },
        result=None,
        output=None,
    )
    state.bind = Mock(side_effect=authority)
    state.launch = Mock(return_value=SimpleNamespace())

    def supervise(process, deadline, **limits):
        """Supply observed manager/probe output without native execution."""
        argv = state.launch.call_args.args[0]
        if state.result:
            return state.result
        name = argv[-1]
        kind = "socket" if name.endswith(".socket") else "service"
        expected = state.expectations[name] | {
            "Id": name,
            "LoadState": "loaded",
            "FragmentPath": str(state.files[kind + "_unit"].path),
            "DropInPaths": "",
            "NeedDaemonReload": "no",
            "Transient": "no",
        }
        if kind == "service":
            expected.update(ProtectControlGroups="yes", Delegate="no", KillMode="control-group")
        output = "\n".join(f"{key}={value}" for key, value in expected.items())
        return ProcessCapture(
            ProcessCaptureStatus.COMPLETED,
            0,
            state.output if state.output is not None else output,
            "",
        )

    state.supervise = Mock(side_effect=supervise)
    monkeypatch.setattr(linux_preflight, "bind_host_file", state.bind)
    monkeypatch.setattr(linux_preflight.subprocess, "Popen", state.launch)
    monkeypatch.setattr(linux_preflight, "supervise_process", state.supervise)
    monkeypatch.setattr(linux_preflight.time, "monotonic", lambda: 10)
    return state


def attest(state):
    """Attest the fixed installation through its public owner."""
    linux_preflight.attest_units(state.files, state.expectations, 100)


def test_attest_units_revalidates_actual_configuration(preflight):
    """Socket and concrete template instances must match protected fragments and live properties."""
    attest(preflight)
    assert preflight.launch.call_count == 2
    for call in preflight.launch.call_args_list:
        assert call.args[0][:4] == ["/usr/bin/systemctl", "--system", "--no-pager", "show"]
        assert call.kwargs["close_fds"] and call.kwargs["start_new_session"]
    assert all(bound.open_verified.call_count == 2 for bound in preflight.files.values())
    assert preflight.supervise.call_args.args[1] == 15


@pytest.mark.parametrize(
    "change",
    ["count", "roles", "fragment", "name", "missing", "key", "type", "newline", "nul", "weaken"],
)
def test_attest_units_rejects_invalid_expectations(preflight, change):
    """Request-like names, incomplete roles and weakened invariants cannot define unit authority."""
    service = preflight.expectations["loop@1000.service"]
    if change == "count":
        preflight.expectations.pop("loop.socket")
    elif change == "roles":
        preflight.expectations["other.socket"] = preflight.expectations.pop("loop@1000.service")
    elif change in ("fragment", "name"):
        preflight.expectations["bad/name.socket" if change == "name" else "other.socket"] = (
            preflight.expectations.pop("loop.socket")
        )
    elif change == "missing":
        service.pop("ExecStart")
    elif change == "key":
        service["bad,key"] = "value"
    elif change == "type":
        service["User"] = 1
    elif change in ("newline", "nul"):
        service["User"] = "\n" if change == "newline" else "\x00"
    else:
        service["Delegate"] = "yes"
    with pytest.raises(ValueError, match="Expected|roles|authority|expectations|invariants"):
        attest(preflight)


@pytest.mark.parametrize(
    "output", ["garbage", "Id=loop.socket\nId=loop.socket", "LoadState=not-found"]
)
def test_attest_units_rejects_ambiguous_or_drifted_manager_output(preflight, output):
    """Missing, duplicate and malformed manager properties fail closed."""
    preflight.output = output
    with pytest.raises(ValueError, match="properties|authority"):
        attest(preflight)


@pytest.mark.parametrize("extra", [False, True])
def test_attest_units_reports_only_fixed_mismatched_property_names(preflight, extra):
    """Drift diagnostics identify fixed fields without revealing values or unknown field names."""
    output = (
        "Listen=private-value\nId=loop.socket\nLoadState=loaded\n"
        "FragmentPath=/installed/loop.socket\nDropInPaths=\nNeedDaemonReload=no\nTransient=no"
    )
    preflight.output = output + ("\nPrivateField=private-value" if extra else "")
    with pytest.raises(ValueError) as failure:
        attest(preflight)
    assert str(failure.value) == (
        "Loaded unit differs from installed enforcement authority: Listen"
        + (", <unexpected>" if extra else "")
        + "."
    )


@pytest.mark.parametrize(
    "change", ["status", "exit", "stderr", "stdout_discarded", "stderr_discarded"]
)
def test_attest_units_refuses_failed_or_unbounded_manager_operation(preflight, change):
    """Manager failure, timeout, diagnostics and truncated output cannot attest loaded authority."""
    values = {"status": ProcessCaptureStatus.COMPLETED, "exit_code": 0, "stdout": "", "stderr": ""}
    values[{"exit": "exit_code"}.get(change, change)] = (
        ProcessCaptureStatus.TIMED_OUT
        if change == "status"
        else ("error" if change == "stderr" else 1)
    )
    preflight.result = ProcessCapture(**values)
    with pytest.raises(OSError, match="unit attestation"):
        attest(preflight)


def test_attest_units_stops_before_launch_when_budget_expired(preflight):
    """Expired preparation never launches a system-manager query."""
    with pytest.raises(TimeoutError, match="deadline"):
        linux_preflight.attest_units(preflight.files, preflight.expectations, 0)
    preflight.launch.assert_not_called()


@pytest.mark.parametrize("change", ["metadata", "argv"])
def test_attest_units_compares_live_command_authority_without_volatile_metadata(preflight, change):
    """Live timestamps and exit metadata do not hide changed fixed executable arguments."""
    command = (
        "{ path=/usr/bin/setpriv ; argv[]=/usr/bin/setpriv --clear-groups ; ignore_errors=no"
        " ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }"
    )
    preflight.expectations["loop@1000.service"]["ExecStart"] = command
    original = preflight.supervise.side_effect

    def live_result(*args, **kwargs):
        """Supply a real-shaped volatile manager property at its public process boundary."""
        result = original(*args, **kwargs)
        text = result.stdout.replace(
            "start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0",
            "start_time=[Sun 2026-10-04 14:00:00 CEST] ; stop_time=[n/a] ; "
            "pid=42 ; code=(null) ; status=0/0",
        )
        if change == "argv":
            text = text.replace("--clear-groups", "--keep-groups")
        return ProcessCapture(result.status, result.exit_code, text, result.stderr)

    preflight.supervise.side_effect = live_result
    if change == "argv":
        with pytest.raises(ValueError, match="Loaded unit differs"):
            attest(preflight)
    else:
        attest(preflight)


def test_attest_units_bounds_authority_hashing(preflight):
    """Initial bindings and both pinned validations use the same shortened preparation budget."""
    linux_preflight.attest_units(preflight.files, preflight.expectations, 12)
    assert preflight.bind.call_args.kwargs == {"deadline": 12}
    for bound in preflight.files.values():
        assert all(call.kwargs == {"deadline": 12} for call in bound.open_verified.call_args_list)
    assert all(call.args[1] == 12 for call in preflight.supervise.call_args_list)
