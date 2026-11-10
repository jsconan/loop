"""Verify native Seatbelt execution, authority binding and filesystem confinement."""

import errno
import json
import os
import plistlib
import secrets
import shlex
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from loop.control import RunControl
from loop.execution.sandbox import (
    SandboxOutcome,
    SandboxRequest,
    UnavailableSandboxBackend,
    select_sandbox_backend,
)
from loop.execution.sandbox.macos import backend as macos_backend
from loop.execution.sandbox.macos.backend import MacOSSeatbeltBackend
from loop.utils import process as process_utils
from tests.scheduling import await_condition, receive_fifo_byte

pytestmark = [pytest.mark.integration, pytest.mark.macos]


def test_owned_ram_device_pin_prevents_forced_detachment():
    """An open raw descriptor keeps an authenticated RAM image from being replaced."""
    marker = f"LoopOwned{secrets.token_hex(16)}"
    attached = subprocess.run(
        [
            "/usr/bin/hdiutil",
            "attach",
            "-nomount",
            "-plist",
            "-drivekey",
            f"{marker}=1",
            "ram://16384",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    entities = plistlib.loads(attached.stdout)["system-entities"]
    assert len(entities) == 1
    device = entities[0]["dev-entry"]
    descriptor = None
    try:
        descriptor = macos_backend._pin_local_socket_attachment(
            device, marker, time.monotonic() + 30
        )
        detached = subprocess.run(
            ["/usr/bin/hdiutil", "detach", "-force", device],
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert detached.returncode == errno.EBUSY, detached.stderr
        assert macos_backend._local_socket_attachment_matches(device, marker, time.monotonic() + 30)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        macos_backend._remove_local_socket_attachment(device, marker, time.monotonic() + 30)


def execute(backend, workspace, source, *, writable=True, scratch=None):
    """Run one approved native request bounded to the synthetic workspace."""
    clock = SimpleNamespace(
        monotonic=lambda: 100.0,
        sleep=lambda _delay: os.sched_yield(),
        strftime=lambda _format, _value: "2000-01-01 00:00:00+0000",
        localtime=lambda: None,
    )
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C"}
    if scratch is not None:
        environment["TMPDIR"] = str(scratch)
    request = SandboxRequest.create(
        source=source,
        cwd=workspace,
        workspace=workspace,
        read_roots=(),
        write_roots=(workspace,) if writable else (),
        network=False,
        environment=environment,
        policy_version=backend.policy_version,
        deadline=110.0,
        workspace_id="isolated-native-test",
        scratch_root=scratch,
    )
    with (
        patch("loop.execution.sandbox.macos.backend.time", clock),
        patch.object(process_utils, "time", clock),
    ):
        return backend.run(request)


def test_native_selection_fails_closed_when_diagnostic_monitor_is_unavailable(backend, monkeypatch):
    """Native selection fails closed when its required diagnostic monitor is unavailable."""

    def unavailable(*_args, **_kwargs):
        """Simulate an unavailable diagnostic stream after the native control succeeds."""
        raise OSError("synthetic diagnostic stream unavailable")

    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.SeatbeltDiagnosticService.monitor", unavailable
    )
    selected = select_sandbox_backend()
    assert isinstance(selected, UnavailableSandboxBackend)
    assert selected.capability_failure().outcome is SandboxOutcome.INVALID
    assert "capability probe failed" in selected.capability_failure().detail


def test_native_unavailable_diagnostics_do_not_start_an_unqualified_command(
    backend, native_workspace, monkeypatch
):
    """An explicit missing-monitor result cannot yet permit a real sandboxed command."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.SeatbeltDiagnosticService.monitor",
        lambda _self, _deadline: None,
    )
    unqualified = MacOSSeatbeltBackend()
    try:
        result = execute(unqualified, native_workspace, "printf changed > diagnostic-marker")
    finally:
        unqualified.close()
    assert result.outcome is SandboxOutcome.INVALID
    assert result.detail == "Seatbelt denial stream is unavailable."
    assert not result.possible_effects
    assert not (native_workspace / "diagnostic-marker").exists()


def test_native_qualified_backend_enforces_without_diagnostic_monitor(
    backend, native_workspace, monkeypatch
):
    """A monitor-free enforcement probe permits only commands that retain Seatbelt isolation."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.SeatbeltDiagnosticService.monitor",
        lambda _self, _deadline: None,
    )
    selected = select_sandbox_backend()
    assert isinstance(selected, MacOSSeatbeltBackend)
    try:
        allowed = execute(selected, native_workspace, "printf allowed > diagnostic-allowed")
        denied_read = execute(selected, native_workspace, "cat .ssh/id_ed25519")
        denied_write = execute(selected, native_workspace, "printf changed > ../outside/write")
    finally:
        selected.close()

    assert allowed.outcome is SandboxOutcome.COMPLETED and allowed.exit_code == 0
    assert (native_workspace / "diagnostic-allowed").read_text() == "allowed"
    assert denied_read.outcome is SandboxOutcome.COMPLETED and denied_read.exit_code != 0
    assert "fake-private-canary" not in denied_read.stdout
    assert denied_write.outcome is SandboxOutcome.COMPLETED and denied_write.exit_code != 0
    assert (native_workspace.parent / "outside" / "write").read_text() == "preserve-outside"
    assert all(
        result.diagnostic_status == "unavailable" and not result.observed_denial
        for result in (allowed, denied_read, denied_write)
    )


def test_native_selection_with_private_temporary_ancestry(
    backend, native_workspace, native_directory, monkeypatch
):
    """Qualify a private temporary root without exposing private data or outside writes."""
    from loop.execution.sandbox import selection

    private = native_workspace.parent / "Library" / "é ; scratch"
    private.mkdir(mode=0o700, parents=True)
    scratch = private / "command"
    scratch.mkdir(mode=0o700)
    sibling = private / "private-sibling"
    sibling.write_text("private-sibling-canary", encoding="utf-8")
    monkeypatch.setattr(selection.tempfile, "tempdir", str(private))
    monkeypatch.setattr(macos_backend, "_PRIVATE_TEMP_ROOT", native_directory("/private/tmp"))
    selected = select_sandbox_backend()
    assert isinstance(selected, MacOSSeatbeltBackend), selected.capability_failure()
    try:
        allowed = execute(
            selected, native_workspace, "printf allowed > probe-allowed", scratch=scratch
        )
        denied_read = execute(selected, native_workspace, "cat .ssh/id_ed25519", scratch=scratch)
        denied_write = execute(
            selected, native_workspace, "printf changed > ../outside/write", scratch=scratch
        )
        implicit = execute(
            selected,
            native_workspace,
            'printf owned > "$TMPDIR/report"; cat "$TMPDIR/report"; '
            f"if cat {shlex.quote(str(sibling))} >/dev/null 2>&1; then exit 9; fi",
        )
    finally:
        selected.close()

    assert allowed.outcome is SandboxOutcome.COMPLETED and allowed.exit_code == 0
    assert (native_workspace / "probe-allowed").read_text() == "allowed"
    assert denied_read.outcome is SandboxOutcome.COMPLETED and denied_read.exit_code != 0
    assert "fake-private-canary" not in denied_read.stdout
    assert denied_write.outcome is SandboxOutcome.COMPLETED and denied_write.exit_code != 0
    assert (native_workspace.parent / "outside" / "write").read_text() == "preserve-outside"
    assert implicit.outcome is SandboxOutcome.COMPLETED and implicit.exit_code == 0
    assert implicit.stdout == "owned"
    assert sibling.read_text(encoding="utf-8") == "private-sibling-canary"


def test_native_audit_veto_of_diagnostic_spawn_prevents_command(backend, native_workspace):
    """A real Python audit veto of the log process cannot launch the sandboxed child."""
    program = """
import os, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from loop.execution import coordinator
from loop.execution.sandbox import SandboxOutcome, SandboxRequest, UnavailableSandboxBackend, select_sandbox_backend
from loop.execution.sandbox import selection
from loop.execution.sandbox.macos.backend import MacOSSeatbeltBackend
from loop.execution.sandbox.macos import backend as macos_backend
from loop.execution.coordinator import CommandExecutionCoordinator
from loop.utils import process
clock = SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda _: os.sched_yield(),
    strftime=lambda _format, _value: '2000-01-01 00:00:00+0000', localtime=lambda: None)
coordinator.time = selection.time = macos_backend.time = process.time = clock
workspace = Path(sys.argv[1])
def audit(event, args):
    if event == 'subprocess.Popen' and args[0] == '/usr/bin/log':
        raise PermissionError('synthetic audit veto')
sys.addaudithook(audit)
backend = MacOSSeatbeltBackend()
try:
    request = SandboxRequest.create(source='printf changed > audit-marker', cwd=workspace,
        workspace=workspace, read_roots=(), write_roots=(workspace,), network=False,
        environment={'PATH': '/usr/bin:/bin', 'LANG': 'C'}, policy_version=backend.policy_version,
        deadline=110.0, workspace_id='native-audit-veto')
    result = backend.run(request)
    if result.outcome is not SandboxOutcome.INVALID or 'synthetic audit veto' not in result.detail:
        raise RuntimeError(repr(result))
    if result.possible_effects or (workspace / 'audit-marker').exists():
        raise RuntimeError('A vetoed command started')
    permissions = Mock()
    def approve(_request, **kwargs):
        '''Bind the mock approval to the revision checked before admission.'''
        kwargs['approved_revision'].append(1)
        return True
    permissions.authorize_sandboxed_command.side_effect = approve
    permissions.sandbox_policy_revision_current.return_value = True
    attempt = CommandExecutionCoordinator(backend, Mock(), permissions, None).run_sandboxed(request, 'run_command')
    if attempt.offer is not None or attempt.result.outcome is not SandboxOutcome.INVALID:
        raise RuntimeError('A diagnostic audit veto created a host retry offer')
finally:
    backend.close()
selected = select_sandbox_backend()
if not isinstance(selected, UnavailableSandboxBackend):
    raise RuntimeError('A diagnostic audit veto was ignored during qualification')
if selected.capability_failure().outcome is not SandboxOutcome.INVALID:
    raise RuntimeError('A diagnostic audit veto could produce an unrestricted host offer')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, str(native_workspace)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_native_qualified_backend_honors_later_diagnostic_audit_veto(backend, native_workspace):
    """A previously qualified backend cannot treat a later real audit veto as log absence."""
    program = """
import os, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from loop.execution import coordinator
from loop.execution.coordinator import CommandExecutionCoordinator
from loop.execution.sandbox import SandboxOutcome, SandboxRequest, select_sandbox_backend
from loop.execution.sandbox import selection
from loop.execution.sandbox.macos.backend import MacOSSeatbeltBackend
from loop.execution.sandbox.macos import backend as macos_backend
from loop.execution.sandbox.macos.backend import SeatbeltDiagnosticService
from loop.utils import process
clock = SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda _: os.sched_yield(),
    strftime=lambda _format, _value: '2000-01-01 00:00:00+0000', localtime=lambda: None)
coordinator.time = selection.time = macos_backend.time = process.time = clock
workspace = Path(sys.argv[1])
original = SeatbeltDiagnosticService.monitor
SeatbeltDiagnosticService.monitor = lambda self, deadline: None
try:
    selected = select_sandbox_backend()
finally:
    SeatbeltDiagnosticService.monitor = original
if not isinstance(selected, MacOSSeatbeltBackend):
    raise RuntimeError('Monitor-free qualification failed')
def audit(event, args):
    if event == 'subprocess.Popen' and args[0] == '/usr/bin/log':
        raise PermissionError('synthetic audit veto')
sys.addaudithook(audit)
try:
    request = SandboxRequest.create(source='printf changed > later-audit-marker', cwd=workspace,
        workspace=workspace, read_roots=(), write_roots=(workspace,), network=False,
        environment={'PATH': '/usr/bin:/bin', 'LANG': 'C'}, policy_version=selected.policy_version,
        deadline=110.0, workspace_id='native-later-audit-veto')
    permissions = Mock()
    def approve(_request, **kwargs):
        '''Bind the mock approval to the revision checked before admission.'''
        kwargs['approved_revision'].append(1)
        return True
    permissions.authorize_sandboxed_command.side_effect = approve
    permissions.sandbox_policy_revision_current.return_value = True
    attempt = CommandExecutionCoordinator(selected, Mock(), permissions, None).run_sandboxed(request, 'run_command')
    if attempt.result.outcome is not SandboxOutcome.INVALID or attempt.offer is not None:
        raise RuntimeError('A qualified backend ignored the diagnostic audit veto')
    if 'synthetic audit veto' not in attempt.result.detail or (workspace / 'later-audit-marker').exists():
        raise RuntimeError('The veto was lost or the sandboxed child started')
finally:
    selected.close()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, str(native_workspace)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_native_commands_preserve_output_and_nonzero_exit(backend, native_workspace):
    """Real shell exit status and both streams survive native enforcement."""
    result = execute(backend, native_workspace, "printf output; printf error >&2; exit 7")
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 7
    assert result.stdout == "output"
    assert result.stderr == "error"


def test_native_cancellation_during_launch_audit_prevents_command_effects(
    backend, native_workspace, monkeypatch
):
    """Cancellation during launch auditing leaves no marker; an uncancelled control succeeds."""
    clock = SimpleNamespace(
        monotonic=lambda: 100.0,
        sleep=lambda _delay: os.sched_yield(),
        strftime=lambda _format, _value: "2000-01-01 00:00:00+0000",
        localtime=lambda: None,
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.time", clock)
    monkeypatch.setattr(process_utils, "time", clock)
    marker = native_workspace / "audit-cancellation-marker"
    request = SandboxRequest.create(
        source=f"printf started > {marker.name}",
        cwd=native_workspace,
        workspace=native_workspace,
        read_roots=(),
        write_roots=(native_workspace,),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
        policy_version=backend.policy_version,
        deadline=130.0,
        workspace_id="isolated-native-test",
    )
    cancellation_event = Event()

    def cancel_during_audit(event_name, **_attributes):
        """Request cancellation when the backend records its impending launch."""
        if event_name == "sandbox.launching":
            cancellation_event.set()

    with monkeypatch.context() as patch:
        patch.setattr("loop.execution.sandbox.macos.backend.telemetry_audit", cancel_during_audit)
        result = backend.run(request, cancellation_event=cancellation_event)

    assert cancellation_event.is_set()
    assert result.outcome is SandboxOutcome.CANCELLED and not result.possible_effects
    assert not marker.exists()
    result = backend.run(request)
    assert result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0, result
    assert marker.read_text() == "started"


@pytest.mark.parametrize("action", ["interrupt", "stop"])
@pytest.mark.parametrize("retained_pipe", [False, True])
def test_native_cancellation_drains_running_commands_and_retained_pipes(
    backend, native_workspace, action, retained_pipe, monkeypatch
):
    """Cancellation cleans up native children and retained pipes without elapsed-time waits."""
    ready = native_workspace / "ready"
    gate = native_workspace / "gate"
    os.mkfifo(ready)
    os.mkfifo(gate)
    source = (
        "(read _ < gate) & printf ready; printf started > cancellation-marker; printf r > ready"
        if retained_pipe
        else "printf ready; printf started > cancellation-marker; printf r > ready; read _ < gate"
    )
    clock = SimpleNamespace(
        monotonic=lambda: 100.0,
        sleep=lambda _delay: os.sched_yield(),
        strftime=lambda _format, _value: "2000-01-01 00:00:00+0000",
        localtime=lambda: None,
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.time", clock)
    monkeypatch.setattr(process_utils, "time", clock)
    native_select = process_utils.select

    class NonblockingQueue:
        """Observe real kqueue events without a timed kernel wait."""

        def __init__(self):
            self.queue = native_select.kqueue()

        def control(self, changes, maximum, _timeout):
            """Poll the native queue and yield when no event is ready."""
            events = self.queue.control(changes, maximum, 0)
            if changes is None and not events:
                os.sched_yield()
            return events

        def close(self):
            """Close the real native event queue."""
            self.queue.close()

    monkeypatch.setattr(
        process_utils,
        "select",
        SimpleNamespace(**{**vars(native_select), "kqueue": NonblockingQueue}),
    )
    observations = Queue()

    def observe_supervision(process, deadline, **kwargs):
        """Register an independent native exit observer before public supervision starts."""
        if not retained_pipe:
            return process_utils.supervise_process(process, deadline, **kwargs)
        queue = native_select.kqueue()
        try:
            try:
                queue.control(
                    [
                        native_select.kevent(
                            process.pid,
                            filter=native_select.KQ_FILTER_PROC,
                            flags=native_select.KQ_EV_ADD | native_select.KQ_EV_ONESHOT,
                            fflags=native_select.KQ_NOTE_EXIT,
                        )
                    ],
                    0,
                    0,
                )
            except ProcessLookupError:
                observations.put(None)
            else:
                observations.put(queue)
            return process_utils.supervise_process(process, deadline, **kwargs)
        finally:
            queue.close()

    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.supervise_process", observe_supervision
    )
    request = SandboxRequest.create(
        source=source,
        cwd=native_workspace,
        workspace=native_workspace,
        read_roots=(),
        write_roots=(native_workspace,),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
        policy_version=backend.policy_version,
        deadline=130.0,
        workspace_id="isolated-native-test",
    )
    control = RunControl()
    with control.execution(), ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            backend.run, request, cancellation_event=control.cancellation_event
        )
        try:
            assert receive_fifo_byte(ready) == b"r"
            assert (native_workspace / "cancellation-marker").read_text() == "started"
            if retained_pipe:
                observer = observations.get()
                assert observer is None or observer.control(None, 1, None), "Leader did not exit"
            getattr(control, action)()
            result = future.result()
            assert result.outcome is SandboxOutcome.CANCELLED, result
            assert result.stdout == "ready" and result.exit_code is None
        finally:
            control.interrupt()
    assert control.cancelled
    assert control.cancellation_event.is_set() == (action == "stop")


@pytest.mark.parametrize("target", [".ssh/id_ed25519", "external-link"])
def test_native_reads_cannot_disclose_private_or_external_files(backend, native_workspace, target):
    """Real protected reads and symlink aliases never disclose synthetic private content."""
    result = execute(backend, native_workspace, f"cat {shlex.quote(target)}", writable=False)
    assert result.outcome in {SandboxOutcome.COMPLETED, SandboxOutcome.DENIED}
    assert result.exit_code != 0
    assert "fake-private-canary" not in result.stdout
    assert "fake-outside-canary" not in result.stdout


@pytest.mark.parametrize("target", ["AGENTS.md", ".git/config", "../outside/write"])
def test_native_writes_preserve_control_and_external_files(backend, native_workspace, target):
    """Native attempts cannot overwrite protected controls or files outside the workspace."""
    path = native_workspace / target
    original = path.read_text(encoding="utf-8")
    result = execute(backend, native_workspace, f"printf changed > {shlex.quote(target)}")
    assert result.exit_code != 0
    assert path.read_text(encoding="utf-8") == original


def test_native_read_only_request_cannot_mutate_workspace(backend, native_workspace):
    """Read-only native authority keeps ordinary workspace content unchanged."""
    result = execute(backend, native_workspace, "printf changed > ordinary", writable=False)
    assert result.exit_code != 0
    assert (native_workspace / "ordinary").read_text(encoding="utf-8") == "ordinary-data"


def test_native_hardlink_preflight_preserves_outside_inode(backend, native_workspace):
    """A writable hardlink alias fails closed before a real command can modify its target."""
    outside = native_workspace.parent / "outside" / "write"
    (native_workspace / "hardlink").hardlink_to(outside)
    result = execute(backend, native_workspace, "printf changed > hardlink")
    assert result.outcome is SandboxOutcome.INVALID
    assert outside.read_text(encoding="utf-8") == "preserve-outside"


def test_shell_pipeline_and_nested_child(native_command, native_workspace):
    """Shell pipelines and workspace executables remain available within native authority."""
    script = native_workspace / "script.sh"
    script.write_text("#!/bin/sh\nprintf workspace-ok\n", encoding="utf-8")
    script.chmod(0o755)
    result = native_command("printf 'hello\n' | tr a-z A-Z; /bin/sh -c './script.sh'")
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 0
    assert result.stdout == "HELLO\nworkspace-ok"


def test_nested_permissive_sandbox_cannot_remove_parent_confinement(native_command):
    """Launching another sandbox never grants access denied by the command's parent policy."""
    result = native_command(
        "/usr/bin/sandbox-exec -p '(version 1)(allow default)' /bin/cat ../outside/secret"
    )

    assert result.outcome is SandboxOutcome.COMPLETED, result
    assert result.exit_code != 0
    assert "fake-outside-canary" not in result.stdout
    assert "fake-outside-canary" not in result.stderr


@pytest.mark.parametrize(
    "profile",
    [
        "(version 1)(allow default)",
        "(version 1)(allow default)(deny network*)",
    ],
    ids=["permissive", "narrower-network"],
)
def test_nested_seatbelt_profile_is_unavailable_inside_native_sandbox(native_command, profile):
    """Valid permissive and narrower profiles cannot be applied inside the outer sandbox."""
    control = subprocess.run(
        ["/usr/bin/sandbox-exec", "-p", profile, "/bin/sh", "-c", "printf profile-valid"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert control.returncode == 0, control.stderr
    assert control.stdout == "profile-valid"

    result = native_command(
        f"/usr/bin/sandbox-exec -p {shlex.quote(profile)} "
        "/bin/sh -c 'printf unexpected-inner-sandbox'"
    )

    assert result.outcome is SandboxOutcome.COMPLETED, result
    assert result.exit_code != 0
    assert "sandbox_apply: Operation not permitted" in result.stderr
    assert "unexpected-inner-sandbox" not in result.stdout


@pytest.mark.parametrize(
    "source",
    [
        "cat ../outside/secret",
        "/bin/sh -c 'cat ../outside/secret'",
        "cat external-link",
        "printf bad > ../outside/write",
        "/bin/sh -c 'printf bad > ../outside/write'",
        "printf bad > external-link",
        "ln ../outside/write newlink",
        "ln .git/config newlink",
    ],
)
def test_outside_alias_and_child_escape_remain_confined(native_command, native_workspace, source):
    """Outside aliases, nested children and hardlink creation cannot escape native authority."""
    result = native_command(source)
    assert result.exit_code != 0
    assert "fake-outside-canary" not in result.stdout
    outside = native_workspace.parent / "outside"
    assert (outside / "write").read_text() == "preserve-outside"
    assert (outside / "secret").read_text() == "fake-outside-canary"
    assert not (native_workspace / "newlink").exists()


@pytest.mark.parametrize(
    "relative",
    [
        ".git/config",
        ".git/new-control",
        ".loop/control",
        ".loop/permissions.yaml",
        ".loop/permissions.yaml.lock",
        ".loop/permissions.yaml.reset-pending",
        "linked/.git",
        "nested/.git/config",
        "AGENTS.md",
    ],
)
def test_protected_control_write_or_creation(native_command, native_workspace, relative):
    """Control files and linked-worktree pointers resist overwrite and creation."""
    target = native_workspace / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    if not exists and relative != ".git/new-control":
        target.write_text("preserve")
        exists = True
    original = target.read_text() if exists else None
    result = native_command(f"printf bad > {shlex.quote(relative)}")
    assert result.exit_code != 0
    assert target.read_text() == original if exists else not target.exists()


def test_workspace_root_cannot_be_renamed(native_command, native_workspace):
    """An approved workspace cannot be moved out from under its authority binding."""
    result = native_command("mv ../workspace ../moved")
    assert result.exit_code != 0
    assert native_workspace.is_dir()
    assert not (native_workspace.parent / "moved").exists()


def test_symlinked_authority_root_is_rejected(native_command, native_workspace):
    """A symlink root fails closed before a command can produce its marker."""
    alias = native_workspace.parent / "workspace-alias"
    alias.symlink_to(native_workspace, target_is_directory=True)
    with pytest.raises(ValueError):
        native_command("printf marker > ordinary", workspace=alias, cwd=alias)
    assert (native_workspace / "ordinary").read_text() == "ordinary-data"


def test_detached_child_retains_native_confinement(native_command, native_workspace):
    """A child that starts its own session still cannot disclose the outside canary."""
    source = "/usr/bin/perl -MPOSIX -e " + shlex.quote(
        "my $pid=fork(); die unless defined $pid; "
        'if (!$pid) { setsid(); system("/bin/cat", "../outside/secret"); exit($? >> 8); } '
        "waitpid($pid, 0); exit($? >> 8);"
    )
    result = native_command(source)
    assert result.exit_code != 0
    assert "fake-outside-canary" not in result.stdout
    assert (native_workspace.parent / "outside" / "secret").read_text() == "fake-outside-canary"


@pytest.mark.parametrize("code", [0, 7, 17, 23, 65, 71, 126, 127])
def test_ordinary_exit_and_forged_stderr(native_command, code):
    """Preserve ordinary exit status despite forged permission or launcher text."""
    result = native_command(f"printf 'Operation not permitted; sandbox_apply' >&2; exit {code}")
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == code
    assert result.stderr == "Operation not permitted; sandbox_apply"


def test_missing_executable_preserves_shell_status(native_command):
    """An absent executable returns the shell's ordinary missing-command exit."""
    result = native_command("loop-command-that-does-not-exist")
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 127


def test_explicit_external_executable_read_grant(native_command, native_workspace):
    """An external executable runs only with its explicitly bound readable root."""
    external = native_workspace.parent / "external"
    external.mkdir()
    tool = external / "tool"
    tool.write_text("#!/bin/sh\nprintf external-ok\n")
    tool.chmod(0o755)
    denied = native_command(shlex.quote(str(tool)))
    assert denied.exit_code != 0
    assert denied.stdout == ""
    approved = native_command(shlex.quote(str(tool)), read_roots=(external,))
    assert approved.exit_code == 0
    assert approved.stdout == "external-ok"


@pytest.mark.parametrize(
    "source,expected",
    [
        ("cat ../outside/secret >/dev/null 2>&1 || true", 0),
        ("printf changed > effect; cat ../outside/secret", 1),
    ],
)
def test_handled_denial_and_partial_effects(native_command, native_workspace, source, expected):
    """Denied reads preserve shell control flow and prior authorized workspace effects."""
    result = native_command(source)
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == expected
    assert "fake-outside-canary" not in result.stdout
    if "effect" in source:
        assert (native_workspace / "effect").read_text() == "changed"


def test_git_grant_preserves_nested_controls(native_command, native_workspace):
    """An explicit top-level Git write grant never unlocks nested Git or application controls."""
    nested = native_workspace / "nested" / ".git"
    nested.mkdir(parents=True)
    (nested / "config").write_text("preserve-nested")
    loop = native_workspace / ".loop"
    loop.mkdir()
    (loop / "control").write_text("preserve-loop")
    writes = (native_workspace, native_workspace / ".git")
    granted = native_command("printf granted > .git/config", write_roots=writes)
    assert granted.exit_code == 0
    assert (native_workspace / ".git" / "config").read_text() == "granted"
    for target in ("nested/.git/config", ".loop/control"):
        result = native_command(f"printf bad > {target}", write_roots=writes)
        assert result.exit_code != 0
    assert (nested / "config").read_text() == "preserve-nested"
    assert (loop / "control").read_text() == "preserve-loop"


def test_network_authority_is_explicit(native_command):
    """A disposable loopback endpoint is reachable only after an explicit egress grant."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        source = "/usr/bin/perl -MSocket -e " + shlex.quote(
            'socket(S, PF_INET, SOCK_STREAM, getprotobyname("tcp")) or die $!; '
            f'connect(S, sockaddr_in({port}, inet_aton("127.0.0.1"))) or die $!; '
            'print "connected";'
        )
        denied = native_command(source)
        assert denied.exit_code != 0
        assert denied.stdout == ""
        approved = native_command(source, network=True)
        assert approved.exit_code == 0
        assert approved.stdout == "connected"


def test_offline_macos_cannot_use_host_loopback_as_local_network(native_command):
    """Seatbelt offline policy refuses even an ordinary server and client in one attempt."""
    source = "/usr/bin/perl -MSocket -e " + shlex.quote(
        'socket(S, PF_INET, SOCK_STREAM, getprotobyname("tcp")) or die $!; '
        'bind(S, sockaddr_in(0, inet_aton("127.0.0.1"))) or die $!; '
        "listen(S, 1) or die $!; "
        'socket(C, PF_INET, SOCK_STREAM, getprotobyname("tcp")) or die $!; '
        'connect(C, getsockname(S)) or die $!; print "local";'
    )
    result = native_command(source)
    assert result.exit_code != 0
    assert result.stdout == ""


def test_attempt_local_unix_socket_stays_inside_private_path(
    native_command, native_workspace, native_directory
):
    """A Unix server and client work while host sockets and IP services stay unreachable."""
    host_path = native_workspace / "host.sock"
    host_datagram_path = native_directory("/private/tmp") / "host-dgram.sock"
    with (
        socket.socket(socket.AF_UNIX) as host_unix,
        socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as host_datagram,
        socket.socket() as host_tcp,
    ):
        host_unix.bind(str(host_path))
        host_unix.listen(4)
        host_datagram.bind(str(host_datagram_path))
        host_datagram.sendto(b"live", str(host_datagram_path))
        assert host_datagram.recvfrom(4)[0] == b"live"
        host_tcp.bind(("127.0.0.1", 0))
        host_tcp.listen(4)
        with socket.socket(socket.AF_UNIX) as control:
            control.connect(str(host_path))
        with socket.socket() as control:
            control.connect(host_tcp.getsockname())
        source = f"""
use Socket;
use JSON::PP;
my $dir = $ENV{{LOOP_LOCAL_SOCKET_DIR}} // die "missing local directory";
my $own = "$dir/server.sock";
socket(my $server, PF_UNIX, SOCK_STREAM, 0) or die $!;
bind($server, pack_sockaddr_un($own)) or die $!;
listen($server, 1) or die $!;
socket(my $client, PF_UNIX, SOCK_STREAM, 0) or die $!;
connect($client, pack_sockaddr_un($own)) or die $!;
defined accept(my $peer, $server) or die $!;
send($client, "local", 0) or die $!;
my $received = "";
defined recv($peer, $received, 5, 0) or die $!;
my %checks = (local => $received eq "local" ? JSON::PP::true : JSON::PP::false);
my $udp = "$dir/datagram.sock";
socket(my $datagram_server, PF_UNIX, SOCK_DGRAM, 0) or die $!;
bind($datagram_server, pack_sockaddr_un($udp)) or die $!;
socket(my $datagram_client, PF_UNIX, SOCK_DGRAM, 0) or die $!;
send($datagram_client, "udp", 0, pack_sockaddr_un($udp)) or die $!;
my $packet = "";
defined recv($datagram_server, $packet, 3, 0) or die $!;
$checks{{datagram}} = $packet eq "udp" ? JSON::PP::true : JSON::PP::false;
my $alias = "$dir/alias.sock";
symlink({json.dumps(str(host_path))}, $alias) or die $!;
my $datagram_alias = "$dir/datagram-alias.sock";
symlink({json.dumps(str(host_datagram_path))}, $datagram_alias) or die $!;
for my $case (
    ["host-unix", PF_UNIX, pack_sockaddr_un({json.dumps(str(host_path))})],
    ["alias", PF_UNIX, pack_sockaddr_un($alias)],
    ["host-tcp", PF_INET, pack_sockaddr_in({host_tcp.getsockname()[1]}, inet_aton("127.0.0.1"))],
    ["external", PF_INET, pack_sockaddr_in(443, inet_aton("1.1.1.1"))],
) {{
    my ($label, $family, $address) = @$case;
    my $probe;
    $checks{{$label}} = socket($probe, $family, SOCK_STREAM, 0)
        && connect($probe, $address) ? JSON::PP::false : JSON::PP::true;
    close($probe) if defined $probe;
}}
socket(my $host_datagram, PF_UNIX, SOCK_DGRAM, 0) or die $!;
$checks{{host_datagram}} = send($host_datagram, "x", 0,
    pack_sockaddr_un({json.dumps(str(host_datagram_path))}))
    ? JSON::PP::false : JSON::PP::true;
$checks{{datagram_alias}} = send($host_datagram, "x", 0,
    pack_sockaddr_un($datagram_alias))
    ? JSON::PP::false : JSON::PP::true;
print encode_json(\\%checks);
"""
        result = native_command("/usr/bin/perl -e " + shlex.quote(source))
    assert result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0, result.stderr
    assert all(json.loads(result.stdout).values()), result.stdout


def test_host_may_connect_inbound_to_macos_sandbox_listener(native_command, native_workspace):
    """An external host peer can initiate traffic into a sandbox listener without an escape."""
    rendezvous = native_workspace / "listener-path"
    source = (
        "use Socket; "
        'my $path = "$ENV{LOOP_LOCAL_SOCKET_DIR}/inbound.sock"; '
        "socket(my $server, PF_UNIX, SOCK_STREAM, 0) or die $!; "
        "bind($server, pack_sockaddr_un($path)) or die $!; listen($server, 1) or die $!; "
        f"open(my $ready, '>', {json.dumps(str(rendezvous))}) or die $!; "
        "print $ready $path; close($ready); alarm 5; "
        "defined accept(my $peer, $server) or die $!; "
        "my $data = ''; defined recv($peer, $data, 4, 0) or die $!; print $data;"
    )
    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(native_command, "/usr/bin/perl -e " + shlex.quote(source))
        await_condition(rendezvous.exists)
        with socket.socket(socket.AF_UNIX) as host:
            host.connect(rendezvous.read_text(encoding="utf-8"))
            host.sendall(b"host")
        result = pending.result(timeout=10)
    assert result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0, result
    assert result.stdout == "host"


def test_host_socket_cannot_be_moved_into_attempt_local_path(
    native_command, native_workspace, native_directory
):
    """A sandbox cannot relocate a host listener into its allowed Unix socket directory."""
    host_directory = native_directory("/private/tmp")
    host_path = host_directory / "host.sock"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(host_path))
        listener.listen(1)
        source = (
            "use Socket; "
            f"my $host = {json.dumps(str(host_path))}; "
            'my $local = "$ENV{LOOP_LOCAL_SOCKET_DIR}/relocated.sock"; '
            "my $moved = rename($host, $local); "
            "socket(my $s, PF_UNIX, SOCK_STREAM, 0) or die $!; "
            "my $connected = connect($s, pack_sockaddr_un($local)); "
            'print "moved=$moved connected=$connected error=", 0+$!, "\\n";'
        )
        result = native_command(
            "/usr/bin/perl -e " + shlex.quote(source),
            write_roots=(native_workspace, host_directory),
        )
    assert result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0, result
    assert "connected=1" not in result.stdout, result.stdout


@pytest.mark.parametrize("destination", ["child", "root"])
def test_host_socket_directory_cannot_be_moved_into_local_volume(
    native_command, native_workspace, native_directory, destination
):
    """Moving a host listener's parent cannot turn it into an allowed local endpoint."""
    host_directory = native_directory("/private/tmp")
    service_directory = host_directory / "service"
    service_directory.mkdir()
    host_path = service_directory / "host.sock"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(host_path))
        listener.listen(1)
        suffix = "/service" if destination == "child" else ""
        source = (
            "use Socket; "
            f"my $source = {json.dumps(str(service_directory))}; "
            f'my $target = "$ENV{{LOOP_LOCAL_SOCKET_DIR}}{suffix}"; '
            "my $moved = rename($source, $target) ? 1 : 0; "
            "socket(my $s, PF_UNIX, SOCK_STREAM, 0) or die $!; "
            "my $connected = connect($s, pack_sockaddr_un(qq($target/host.sock))) ? 1 : 0; "
            'print "moved=$moved connected=$connected\\n";'
        )
        result = native_command(
            "/usr/bin/perl -e " + shlex.quote(source),
            write_roots=(native_workspace, host_directory),
        )
    assert result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0, result
    assert result.stdout == "moved=0 connected=0\n"
    assert host_path.exists()


def test_local_socket_volume_rejects_a_writable_ancestor(
    native_command, native_workspace, local_socket_parent
):
    """A broad write grant cannot include the private socket volume from above."""
    result = native_command(
        "printf should-not-run",
        write_roots=(native_workspace, local_socket_parent.parent),
    )
    assert result.outcome is SandboxOutcome.UNAVAILABLE
    assert "shares the local socket filesystem" in result.detail
    assert result.stdout == ""


def test_local_network_limit_denies_macos_unix_bind(native_command):
    """A user local-network deny removes the attempt-local Unix socket authority."""
    source = (
        "use Socket; my $s; "
        "if (!socket($s, PF_UNIX, SOCK_STREAM, 0)) { print 0+$!, qq(\\n); exit 0 } "
        "my $path = qq($ENV{TMPDIR}/denied.sock); "
        "print bind($s, pack_sockaddr_un($path)) ? 0 : 0+$!, qq(\\n);"
    )
    result = native_command("/usr/bin/perl -e " + shlex.quote(source), local_network=False)
    assert result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0, result
    assert result.stdout == f"{errno.EPERM}\n"


def test_attempt_local_socket_path_is_not_shared_with_a_later_command(native_command):
    """A later attempt cannot reach a live service at the prior attempt's socket path."""
    first = native_command('printf %s "$LOOP_LOCAL_SOCKET_DIR"')
    assert first.outcome is SandboxOutcome.COMPLETED and first.exit_code == 0, first
    old_directory = Path(first.stdout)
    assert not old_directory.exists()
    old_directory.mkdir(mode=0o700)
    path = old_directory / "old.sock"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(path))
        listener.listen(4)
        with socket.socket(socket.AF_UNIX) as control:
            control.connect(str(path))
        source = (
            "use Socket; socket(my $s, PF_UNIX, SOCK_STREAM, 0) or die $!; "
            f"connect($s, pack_sockaddr_un({json.dumps(str(path))})) "
            "and die 'connected'; print 0+$!, qq(\\n);"
        )
        second = native_command("/usr/bin/perl -e " + shlex.quote(source))
    assert second.outcome is SandboxOutcome.COMPLETED and second.exit_code == 0, second
    assert second.stdout == f"{errno.EPERM}\n"
    path.unlink()
    old_directory.rmdir()


def test_unix_ipc_outside_authority_is_denied(native_command, native_workspace):
    """A native command cannot connect to an outside Unix-domain service."""
    path = native_workspace.parent / "outside" / "service.sock"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(path))
        listener.listen(1)
        source = "/usr/bin/perl -MSocket -e " + shlex.quote(
            "socket(S, PF_UNIX, SOCK_STREAM, 0) or die $!; "
            f'connect(S, sockaddr_un("{path}")) or die $!; print "connected";'
        )
        result = native_command(source)
        assert result.exit_code != 0
        assert result.stdout == ""


def test_host_unix_socket_in_writable_workspace_is_denied_without_network(
    native_command, native_workspace
):
    """A writable workspace does not itself authorize a host Unix-domain service."""
    path = native_workspace / "service.sock"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(path))
        listener.listen(1)
        source = "/usr/bin/perl -MSocket -e " + shlex.quote(
            "socket(S, PF_UNIX, SOCK_STREAM, 0) or die $!; "
            f'connect(S, sockaddr_un("{path}")) or die $!; print "connected";'
        )
        result = native_command(source)
        assert result.exit_code != 0
        assert result.stdout == ""
        approved = native_command(source, network=True)
        assert approved.exit_code == 0
        assert approved.stdout == "connected"


@pytest.mark.parametrize(
    "relative",
    [
        ".ssh/key",
        ".aws/credentials",
        ".gnupg/key",
        ".config/key",
        ".loop/control",
        "Library/private",
        "AGENTS.md",
        ".git/config",
        ".agents/skills/SKILL.md",
        "POLICY+.md",
    ],
)
def test_native_parent_rename_cannot_relocate_protected_data(
    native_command, native_workspace, relative
):
    """Prevent protected descendants from escaping through a renamed parent."""
    parent = native_workspace / "protected-parent"
    target = parent / relative
    target.parent.mkdir(parents=True)
    target.write_text("preserve-relocation")
    source = "/usr/bin/perl -e " + shlex.quote(
        'rename "protected-parent", "$ENV{TMPDIR}/moved" or die $!; '
        f'open F, ">", "$ENV{{TMPDIR}}/moved/{relative}" or die $!; print F "changed"'
    )
    result = native_command(source, protected_instruction_names=("POLICY+.md",))
    assert result.exit_code != 0
    assert target.read_text() == "preserve-relocation"


def test_ordinary_directory_rename_remains_available(native_command, native_workspace):
    """Protected ancestor restrictions do not forbid ordinary workspace directory renames."""
    parent = native_workspace / "ordinary-parent"
    parent.mkdir()
    (parent / "data").write_text("ordinary")
    result = native_command("mv ordinary-parent ordinary-moved")
    assert result.exit_code == 0
    assert (native_workspace / "ordinary-moved" / "data").read_text() == "ordinary"


def test_selected_descriptor_helper_cannot_read_outside_files(
    backend, native_workspace, monkeypatch
):
    """Give a read-only helper the selected descriptor without outside read authority."""
    clock = SimpleNamespace(
        monotonic=lambda: 100.0,
        sleep=lambda _delay: os.sched_yield(),
        strftime=lambda _format, _value: "2000-01-01 00:00:00+0000",
        localtime=lambda: None,
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.time", clock)
    monkeypatch.setattr(process_utils, "time", clock)
    helper = native_workspace.parent / "attack-helper"
    helper.write_text('#!/bin/sh\n/bin/cat "$1"\n/bin/cat "$2"\n')
    helper.chmod(0o755)
    selected = native_workspace / "selected.txt"
    selected.write_text("selected-data\n")
    descriptor = os.open(selected, os.O_RDONLY)
    try:
        result = backend.run_read_only_argv(
            helper,
            (f"/dev/fd/{descriptor:05d}", str(native_workspace.parent / "outside" / "secret")),
            (descriptor,),
            110.0,
        )
    finally:
        os.close(descriptor)
    assert result is not None
    assert result.exit_code != 0
    assert "selected-data" in result.stdout
    assert "fake-outside-canary" not in result.stdout


@pytest.mark.parametrize("path", [".ssh/key", ".aws/credentials", ".gnupg/key"])
def test_extra_read_root_preserves_private_restrictions(native_command, native_workspace, path):
    """Granting an external readable directory does not grant its private credential descendants."""
    extra = native_workspace.parent / "extra"
    target = extra / path
    target.parent.mkdir(parents=True)
    target.write_text("extra-private-canary")
    result = native_command(f"cat ../extra/{path}", read_roots=(extra,), write_roots=())
    assert result.exit_code != 0
    assert "extra-private-canary" not in result.stdout


def test_session_ancestor_metadata_does_not_expose_private_siblings(
    native_command, native_workspace, native_directory
):
    """Temporary private storage permits exact ancestor traversal without sibling metadata."""
    private = native_workspace / "Library"
    scratch = private / "session" / "files"
    scratch.mkdir(mode=0o700, parents=True)
    sibling = private / "unrelated"
    sibling.write_text("private-sibling-canary")
    source = 'printf scratch > "$TMPDIR/report"; cat "$TMPDIR/report"; stat -f %HT ' + shlex.quote(
        str(private)
    )
    changes = {
        "environment": {"PATH": "/usr/bin:/bin", "LANG": "C", "TMPDIR": str(scratch)},
        "scratch_root": scratch,
        "write_roots": (),
    }
    allowed = native_command(source, **changes)
    assert allowed.exit_code == 0, allowed
    assert allowed.stdout == "scratchDirectory\n"
    for command in ("stat -f %HT " + shlex.quote(str(sibling)), "ls " + shlex.quote(str(private))):
        denied = native_command(command, **changes)
        assert denied.exit_code != 0, denied
        assert denied.stdout == ""
    assert sibling.read_text() == "private-sibling-canary"


def test_detached_descendant_retains_old_descriptor_after_generation_retirement(
    backend, native_workspace, native_directory, monkeypatch
):
    """Detached macOS writers remain confined but are not revoked by ownership or retirement."""
    import json
    from unittest.mock import Mock

    from loop import Interaction, SessionManager

    manager = SessionManager(interaction=Mock(spec=Interaction))
    original = manager.temporary_directory.get()
    gate, done = original / "gate", original / "done"
    os.mkfifo(gate)
    os.mkfifo(done)
    gate_fd = os.open(gate, os.O_RDWR | os.O_NONBLOCK)
    done_fd = os.open(done, os.O_RDWR | os.O_NONBLOCK)
    program = r"""
pipe(R, W) or die;
my $pid = fork(); defined($pid) or die;
if ($pid) { close W; sysread(R, my $ready, 1); exit; }
close R;
POSIX::setsid() >= 0 or die;
my $second = fork(); defined($second) or die;
if ($second) { POSIX::_exit(0); }
open(my $retained, ">", "$ENV{TMPDIR}/late") or die;
open(my $gate, "<", "$ENV{TMPDIR}/gate") or die;
open(my $done, ">", "$ENV{TMPDIR}/done") or die;
syswrite(W, "R"); close W;
close STDIN; close STDOUT; close STDERR;
sysread($gate, my $release, 1);
my $escaped = open(my $outside, ">", "$ENV{TMPDIR}/../escape");
print $retained ($escaped ? "escaped" : "confined"); close $retained;
syswrite($done, "D"); close $done;
POSIX::_exit(0);
"""
    clock = SimpleNamespace(
        monotonic=lambda: 100.0,
        sleep=lambda _delay: os.sched_yield(),
        strftime=lambda _format, _value: "2000-01-01 00:00:00+0000",
        localtime=lambda: None,
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.time", clock)
    monkeypatch.setattr(process_utils, "time", clock)
    request = SandboxRequest.create(
        source="/usr/bin/perl -MPOSIX -e " + shlex.quote(program),
        cwd=native_workspace,
        workspace=native_workspace,
        read_roots=(),
        write_roots=(),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C", "TMPDIR": str(original)},
        policy_version=backend.policy_version,
        deadline=110.0,
        workspace_id="detached-working-generation",
        scratch_root=original,
    )
    try:
        result = backend.run(request)
        assert result.outcome is SandboxOutcome.COMPLETED, result
        assert result.exit_code == 0, result
        assert result.stdout == result.stderr == "", json.dumps(result.__dict__)
        with (original / "late").open() as retained:
            manager.discard_working_directory()
            assert manager.temporary_directory.get() != original
            assert os.write(gate_fd, b"G") == 1

            def completed():
                """Observe the detached writer's completion without a timed wait."""
                try:
                    value = os.read(done_fd, 1)
                except BlockingIOError:
                    return False
                if value:
                    assert value == b"D"
                    return True
                return False

            await_condition(completed)
            assert retained.read() == "confined"
    finally:
        os.close(gate_fd)
        os.close(done_fd)
    assert not original.exists()
    assert not original.parent.exists()
    assert not (original.parent / "escape").exists()
