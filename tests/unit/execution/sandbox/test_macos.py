"""Test the macOS Seatbelt process backend and its fail-closed outcomes."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from loop.execution.sandbox import SandboxOutcome, SandboxRequest
from loop.execution.sandbox.macos import MacOSSeatbeltBackend
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry
from loop.utils import ProcessCapture, ProcessCaptureStatus


@pytest.fixture(autouse=True)
def fake_darwin_temp(monkeypatch, tmp_path, request):
    """Keep unit tests independent of the host's Darwin cache directory."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.platform", SimpleNamespace(system=lambda: "Darwin")
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.time",
        SimpleNamespace(**{**vars(time), "sleep": lambda seconds: None}),
    )
    home = tmp_path.with_name(tmp_path.name + "-home")
    home.mkdir()
    if getattr(request, "param", None) != "real_home":
        monkeypatch.setattr("loop.execution.sandbox.macos._trusted_home", lambda: home)
    monitor = MagicMock()
    monitor.denial.return_value = None
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.SeatbeltDiagnosticService.monitor",
        lambda self, deadline: monitor,
    )
    if getattr(request, "param", True):
        monkeypatch.setattr(
            "loop.execution.sandbox.macos.SeatbeltDiagnosticService.darwin_temp",
            lambda self, deadline: tmp_path,
        )


def request(root: Path, **changes) -> SandboxRequest:
    """Create a current Seatbelt request rooted in one disposable workspace."""
    values = {
        "source": "printf ok",
        "cwd": root,
        "workspace": root,
        "read_roots": (),
        "write_roots": (root,),
        "network": False,
        "environment": {"PATH": os.environ["PATH"], "LANG": "C"},
        "policy_version": "macos-seatbelt-v1",
        "deadline": time.monotonic() + 30,
        "workspace_id": "test",
    }
    values.update(changes)
    return SandboxRequest.create(**values)


class Process:
    """Provide the subprocess surface used by the backend capture loop."""

    def __init__(self, stdout="ok", stderr="", returncode=0, wait_error=None, started=True):
        self.stdout = (
            io.StringIO(("LOOP_SHELL_STARTED\n" if started else "") + stdout)
            if stdout is not None
            else None
        )
        self.stderr = io.StringIO(stderr) if stderr is not None else None
        self.returncode = returncode
        self.pid = 123
        self.wait_error = wait_error

    def wait(self, timeout=None):
        """Return completion or raise the configured timeout."""
        if self.wait_error is not None:
            raise self.wait_error
        return self.returncode


class LogProcess:
    """Expose bounded kernel-log bytes through a real readable pipe."""

    def __init__(
        self,
        records=b"",
        *,
        running=False,
        stdout=True,
        wait_error=None,
        stuck_after_kill=False,
    ):
        self.records = records
        self.running = running
        self.has_stdout = stdout
        self.killed = False
        self.wait_error = wait_error
        self.stuck_after_kill = stuck_after_kill
        self.wait_timeouts = []
        if stdout:
            read_fd, write_fd = os.pipe()
            os.write(write_fd, records)
            self.stdout = os.fdopen(read_fd, "rb")
            self._write_fd = write_fd
        else:
            self.stdout = None
            self._write_fd = None
        if not running or not stuck_after_kill:
            self.finish()

    def clone(self):
        """Create a fresh pipe for a later bounded log snapshot."""
        return LogProcess(
            self.records,
            running=self.running,
            stdout=self.has_stdout,
            wait_error=self.wait_error,
            stuck_after_kill=self.stuck_after_kill,
        )

    def poll(self):
        """Report whether the simulated log query still runs."""
        return None if self.running and (self.stuck_after_kill or not self.killed) else 0

    def kill(self):
        """Record termination of a simulated running log query."""
        self.killed = True
        if not self.stuck_after_kill:
            self.finish()

    def wait(self, timeout=None):
        """Complete the simulated log query."""
        self.wait_timeouts.append(timeout)
        if self.wait_error is not None:
            raise self.wait_error
        return 0

    def finish(self):
        """Close the producer after its data has been consumed."""
        if self._write_fd is not None:
            os.close(self._write_fd)
            self._write_fd = None


class CloseErrorStream:
    """Close a pipe and report a diagnostic close failure."""

    def __init__(self, stream):
        self._stream = stream

    @property
    def closed(self):
        """Report whether the underlying pipe is closed."""
        return self._stream.closed

    def fileno(self):
        """Return the underlying pipe descriptor for select and os.read."""
        return self._stream.fileno()

    def close(self):
        """Close the pipe before raising the configured I/O failure."""
        self._stream.close()
        raise OSError("diagnostic pipe close failed")


class PollErrorLogProcess(LogProcess):
    """Simulate a diagnostic child whose status cannot be queried."""

    def __init__(self):
        super().__init__()
        self.stdout = CloseErrorStream(self.stdout)
        self.kill_attempts = 0

    def poll(self):
        """Raise the process-status error used to exercise best-effort cleanup."""
        raise OSError("diagnostic process status failed")

    def kill(self):
        """Reject termination after the process-status query has failed."""
        self.kill_attempts += 1
        raise OSError("diagnostic process termination failed")


def kernel_record(message, *, process_id=0, image="/kernel", sender="/Sandbox.kext/Sandbox"):
    """Encode one structured macOS kernel log record."""
    return (
        json.dumps(
            {
                "processID": process_id,
                "processImagePath": image,
                "senderImagePath": sender,
                "eventMessage": message,
            }
        ).encode()
        + b"\n"
    )


def run_with_log(
    monkeypatch,
    tmp_path,
    log_process,
    *,
    output="Permission denied",
    log_error=None,
    expire_after_wait=False,
    later_records=None,
    returncode=1,
):
    """Run the public backend with one completed child and one log query."""
    command = Process(stdout="", stderr=output, returncode=returncode)
    launches = 0

    def launch(*args, **kwargs):
        """Return the command once and a fresh log stream on each diagnostic retry."""
        nonlocal launches
        launches += 1
        if launches == 1:
            return command
        if log_error is not None:
            raise log_error
        if launches == 2:
            return log_process
        return LogProcess(later_records) if later_records is not None else log_process.clone()

    popen = MagicMock(side_effect=launch)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.secrets.token_hex",
        MagicMock(side_effect=["a" * 32, "b" * 32]),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    value = request(tmp_path)
    if expire_after_wait:

        def expire(timeout=None):
            """Move the request deadline past before denial inspection."""
            object.__setattr__(value, "deadline", time.monotonic() - 1)
            return 1

        command.wait = expire
    result = MacOSSeatbeltBackend().run(value)
    return result, popen


def test_backend_rejects_unsupported_policy(tmp_path):
    """A policy mismatch fails before launcher preparation."""
    result = MacOSSeatbeltBackend().run(request(tmp_path, policy_version="other"))
    assert result.outcome is SandboxOutcome.UNAVAILABLE


def test_backend_rejects_stale_initial_identity(monkeypatch, tmp_path):
    """A changed approved root fails before profile generation."""
    value = request(tmp_path)
    monkeypatch.setattr(SandboxRequest, "paths_are_current", lambda self: False)
    result = MacOSSeatbeltBackend().run(value)
    assert result.outcome is SandboxOutcome.STALE
    assert "path identity changed" in result.detail


def test_backend_expired_budget_is_timeout_before_native_launch(monkeypatch, tmp_path):
    """An expired request remains a timeout without attempting native initialization."""
    value = request(tmp_path, deadline=100.0)
    monkeypatch.setattr("loop.execution.sandbox.macos.time.monotonic", lambda: 101.0)
    popen = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    result = MacOSSeatbeltBackend().run(value)
    assert result.outcome is SandboxOutcome.TIMED_OUT
    assert result.detail == "Command deadline expired."
    popen.assert_not_called()


def test_backend_builds_profile_and_runs_child(monkeypatch, tmp_path):
    """A successful launcher check starts the fixed shell with a private TMPDIR."""
    probe = MagicMock(returncode=0, stderr="")
    process = Process()
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run", MagicMock(return_value=probe)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())

    result = MacOSSeatbeltBackend().run(request(tmp_path, network=True))

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.stdout == "ok"
    argv = popen.call_args.args[0]
    assert argv[-2:] == ["loop-shell", "printf ok"]
    assert argv[3:5] == ["/bin/sh", "-c"]
    assert "(allow network*)" in argv[2]
    assert popen.call_args.kwargs["close_fds"] is True
    assert Path(popen.call_args.kwargs["env"]["TMPDIR"]).is_relative_to(
        Path(tempfile.gettempdir()).resolve()
    )


def test_backend_audits_launch_with_the_approved_attempt_id(monkeypatch, tmp_path):
    """Native launch records correlate to the same approved request without shell text."""
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    approved = request(tmp_path, source="printf private-command")
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    try:
        assert MacOSSeatbeltBackend().run(approved).outcome is SandboxOutcome.COMPLETED
        assert telemetry.close(1)
    finally:
        set_telemetry(None)
    records = [item for item in adapter.records if item.event_name.startswith("sandbox.launch")]
    assert [item.event_name for item in records] == ["sandbox.launching", "sandbox.launched"]
    assert all(item.attributes["attempt_id"] == approved.attempt_id for item in records)
    assert all("private-command" not in str(item.attributes) for item in records)


def test_backend_reads_only_alias_symlink_and_runs_translated_source(monkeypatch, tmp_path):
    """Alias traversal is narrow while the shell receives the approved translation."""
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    alias = private / "r0"
    alias.symlink_to(tmp_path, target_is_directory=True)
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    value = request(
        tmp_path,
        source="cat /workspace/file",
        execution_source=f"cat {alias}/file",
        aliases=(("/workspace", alias, tmp_path),),
    )

    result = MacOSSeatbeltBackend().run(value)

    assert result.outcome is SandboxOutcome.COMPLETED
    arguments = popen.call_args.args[0]
    assert arguments[-1] == f"cat {alias}/file"
    profile = arguments[2]
    assert f'(subpath "{private}")' in profile
    assert f'(literal "{alias}")' in profile
    assert f'(subpath "{alias}")' not in profile


def test_backend_granted_git_metadata_keeps_nested_git_protected(monkeypatch, tmp_path):
    """An approved root Git grant leaves nested repository metadata denied."""
    git = tmp_path / ".git"
    git.mkdir()
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(request(tmp_path, write_roots=(tmp_path, git)))
    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    assert "/.+/\\.git" in profile


def test_backend_allows_approved_fresh_git_metadata_only_at_workspace_root(monkeypatch, tmp_path):
    """A missing top-level Git directory can be created without opening nested metadata."""
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(request(tmp_path, git_create=True))
    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    assert "/.+/\\.git" in profile
    assert f'"^{tmp_path}/.*[.]git(/.*)?$"' not in profile


def test_backend_read_only_profile_has_no_workspace_write_grant(monkeypatch, tmp_path):
    """The OS profile denies workspace writes for an auto-approved read request."""
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(request(tmp_path, write_roots=()))
    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    write_rule = next(
        line for line in profile.splitlines() if line.startswith("(allow file-write* ")
    )
    assert str(tmp_path) not in write_rule


def test_backend_rejects_scratch_without_its_own_prefix(tmp_path):
    """Seatbelt alone enforces its scratch ownership convention before any launch."""
    with tempfile.TemporaryDirectory(prefix="loop-other-") as directory:
        bound = request(
            tmp_path,
            environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(Path(directory).resolve())},
        )
        result = MacOSSeatbeltBackend().run(bound)

    assert result.outcome is SandboxOutcome.INVALID
    assert "Unsafe command scratch directory" in result.detail


def test_backend_protects_instruction_files_even_with_workspace_write_grant(monkeypatch, tmp_path):
    """Native write policy carves instructions and skills out of writable workspace roots."""
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())

    assert MacOSSeatbeltBackend().run(request(tmp_path)).outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    assert "(AGENTS\\.md|SKILL\\.md)" in profile
    assert "/(.*/)?\\.agents/skills(/.*)?$" in profile
    assert "/(.*/)?\\.gitignore$" in profile
    assert "/(.*/)?\\.agentignore$" in profile
    assert profile.index("(deny file-write* ") > profile.index("(allow file-write* ")


def test_backend_protects_configured_instruction_name_at_every_depth(monkeypatch, tmp_path):
    """A custom instruction basename is anchored and escaped in the native deny policy."""
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())

    bound = request(tmp_path, protected_instruction_names=("POLICY+.md",))
    assert MacOSSeatbeltBackend().run(bound).outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    assert r"AGENTS\.md|SKILL\.md|POLICY\+\.md" in profile
    assert f"{re.escape(str(tmp_path))}/(.*/)?(" in profile
    assert r"POLICY\+\.md)$" in profile


def test_backend_denies_private_reads_inside_workspace_and_extra_roots(monkeypatch, tmp_path):
    """Workspace and extra read grants retain protected-path denials."""
    workspace = tmp_path / "work"
    workspace.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())

    result = MacOSSeatbeltBackend().run(request(workspace, read_roots=(extra,)))

    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    assert f'(subpath "{workspace}")' in profile
    assert f'(subpath "{extra}")' in profile
    names = r"(\.ssh|\.aws|\.gnupg|\.config|\.loop|Library)"
    for root in (workspace, extra):
        assert f"{re.escape(str(root))}/(.*/)?{names}(/.*)?$" in profile
    assert profile.index("(deny file-read*") > profile.index("(allow file-read*")


def test_backend_guards_protected_entries_and_their_ancestors_against_rename(monkeypatch, tmp_path):
    """Protected entries cannot be moved out of their pathname policy by renaming a parent."""
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("instructions")
    private_parent = tmp_path / "private-parent"
    private = private_parent / ".ssh"
    private.mkdir(parents=True)
    (private / "id_ed25519").write_text("canary")
    git_parent = tmp_path / "git-parent"
    git = git_parent / ".git"
    git.mkdir(parents=True)
    (git / "config").write_text("canary")
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    rename_rules = [
        line for line in profile.splitlines() if line.startswith("(deny file-write-unlink ")
    ]
    assert any(r"(\.ssh|\.aws|\.gnupg|\.config|\.loop|Library)" in line for line in rename_rules)
    for protected in (nested, private_parent, private, git_parent, git):
        assert any(f'(literal "{protected}")' in line for line in rename_rules)
    assert all(f'(literal "{ordinary}")' not in line for line in rename_rules)


def test_backend_limits_baseline_reads_to_system_managed_roots(monkeypatch, tmp_path):
    """Writable installed-tool trees cannot receive recursive default reads."""
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())

    backend = MacOSSeatbeltBackend()
    assert backend.policy_version == "macos-seatbelt-v1"
    assert backend.scratch_prefix == "loop-seatbelt-"
    result = backend.run(request(tmp_path))

    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    names = r"(\.ssh|\.aws|\.gnupg|\.config|\.loop|Library)"
    for root in backend.system_read_roots:
        assert f"{re.escape(str(root))}/(.*/)?{names}(/.*)?$" in profile
    assert '(subpath "/System/Library")' in profile
    assert '(subpath "/System")' not in profile
    for root in ("/usr", "/usr/local", "/Library", "/opt/homebrew"):
        assert f'(subpath "{root}")' not in profile


def test_backend_auto_trusts_only_receipted_managed_package_roots(monkeypatch, tmp_path):
    """Homebrew auto-read eligibility belongs to the native platform adapter."""
    cellar = tmp_path / "Cellar"
    package = cellar / "sample" / "1.0"
    package.mkdir(parents=True)
    monkeypatch.setattr("loop.execution.sandbox.macos._MANAGED_TOOLCHAIN_PREFIXES", (cellar,))

    assert not MacOSSeatbeltBackend.managed_tool_root(package)
    (package / "INSTALL_RECEIPT.json").write_text("{}", encoding="utf-8")
    assert MacOSSeatbeltBackend.managed_tool_root(package)
    assert not MacOSSeatbeltBackend.managed_tool_root(tmp_path)


def test_trusted_argv_search_passes_only_selected_descriptors_under_read_only_policy(
    monkeypatch, tmp_path
):
    """A selected helper launches through Seatbelt without shell, network, or workspace writes."""
    executable = tmp_path / "rg"
    executable.write_bytes(b"verified helper")
    executable.chmod(0o755)
    popen = MagicMock()
    capture = ProcessCapture(ProcessCaptureStatus.COMPLETED, exit_code=0, stdout="matched")
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.supervise_process", lambda *_args, **_kwargs: capture
    )
    backend = MacOSSeatbeltBackend()

    result = backend.run_read_only_argv(
        executable, ("--json", "needle", "/dev/fd/3"), (3,), time.monotonic() + 5
    )

    assert result is capture
    argv = popen.call_args.args[0]
    assert argv[0] == "/usr/bin/sandbox-exec"
    assert argv[-3:] == ["--json", "needle", "/dev/fd/3"]
    assert popen.call_args.kwargs["pass_fds"] == (3,)
    assert "(deny default" in argv[2]
    assert "(allow network*)" not in argv[2]


def test_trusted_argv_search_fails_closed_on_expiry_tamper_and_copy_error(monkeypatch, tmp_path):
    """An expired request or unsafe copy cannot reach a native child launch."""
    executable = tmp_path / "rg"
    executable.write_bytes(b"verified helper")
    executable.chmod(0o755)
    popen = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    backend = MacOSSeatbeltBackend()

    with monkeypatch.context() as scoped:
        scoped.setattr("loop.execution.sandbox.macos.platform.system", lambda: "Linux")
        assert backend.run_read_only_argv(executable, (), (), time.monotonic() + 5) is None

    assert (
        backend.run_read_only_argv(executable, (), (), time.monotonic() - 1).status
        is ProcessCaptureStatus.TIMED_OUT
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(SandboxRequest, "paths_are_current", lambda _self: False)
        assert backend.run_read_only_argv(executable, (), (), time.monotonic() + 5) is None
    original_open = Path.open

    def denied_open(path, *args, **kwargs):
        """Fail only the copied helper destination before a native launch."""
        if path.name == "helper":
            raise OSError("denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_open)
    assert backend.run_read_only_argv(executable, (), (), time.monotonic() + 5) is None
    popen.assert_not_called()


def test_trusted_argv_search_stops_after_slow_helper_copy(monkeypatch, tmp_path):
    """A short fixed-argv deadline prevents launch after helper preparation runs long."""
    executable = tmp_path / "rg"
    executable.write_bytes(b"verified helper")
    executable.chmod(0o755)
    clock = MagicMock(return_value=time.monotonic())
    monkeypatch.setattr("loop.execution.sandbox.macos.time.monotonic", clock)
    original_open = os.open

    def slow_open(path, *args, **kwargs):
        """Delay the copied helper input in a disposable fixture."""
        if path == executable:
            clock.return_value += 0.04
        return original_open(path, *args, **kwargs)

    launch = MagicMock()
    monkeypatch.setattr(os, "open", slow_open)
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", launch)
    started = clock.return_value
    result = MacOSSeatbeltBackend().run_read_only_argv(executable, (), (), started + 0.01)
    assert result.status is ProcessCaptureStatus.TIMED_OUT
    assert time.monotonic() - started < 0.5
    launch.assert_not_called()


def test_trusted_argv_search_rejects_symlink_and_hardlink_helpers(monkeypatch, tmp_path):
    """Fixed-argv preparation never copies an aliased executable into private state."""
    executable = tmp_path / "rg"
    executable.write_bytes(b"verified helper")
    executable.chmod(0o755)
    symlink = tmp_path / "symlink-rg"
    symlink.symlink_to(executable)
    launch = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", launch)
    backend = MacOSSeatbeltBackend()
    assert backend.run_read_only_argv(symlink, (), (), time.monotonic() + 5) is None
    assert backend.run_read_only_argv(tmp_path, (), (), time.monotonic() + 5) is None
    hardlink = tmp_path / "hardlink-rg"
    hardlink.hardlink_to(executable)
    assert backend.run_read_only_argv(executable, (), (), time.monotonic() + 5) is None
    launch.assert_not_called()


def test_backend_deadline_during_granted_tree_walk_prevents_launch(monkeypatch, tmp_path):
    """A slow granted-tree scan times out before any native command starts."""
    for index in range(10):
        (tmp_path / f"file-{index}").write_text("ordinary", encoding="utf-8")
    clock = MagicMock(return_value=time.monotonic())
    monkeypatch.setattr("loop.execution.sandbox.macos.time.monotonic", clock)
    original_scandir = os.scandir

    def slow_scandir(path):
        """Delay entry enumeration only for the granted workspace."""
        if not isinstance(path, int) and Path(path) == tmp_path:
            clock.return_value += 0.04
        return original_scandir(path)

    launch = MagicMock()
    monkeypatch.setattr(os, "scandir", slow_scandir)
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", launch)
    started = clock.return_value
    result = MacOSSeatbeltBackend().run(request(tmp_path, deadline=started + 0.01))
    assert result.outcome is SandboxOutcome.TIMED_OUT
    assert time.monotonic() - started < 0.5
    launch.assert_not_called()


def test_backend_pipe_failure_does_not_wait_past_deadline(monkeypatch, tmp_path):
    """A malformed sandbox child remains unavailable when reaping times out."""
    process = Process(stdout=None, stderr=None, wait_error=subprocess.TimeoutExpired("sh", 0))
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.Popen", MagicMock(return_value=process)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    assert result.outcome is SandboxOutcome.UNAVAILABLE


@pytest.mark.parametrize("path_value", [":/usr/bin", "/usr/bin:", "relative:/usr/bin"])
def test_backend_binds_empty_and_relative_search_paths(tmp_path, path_value):
    """Empty and relative PATH entries bind to cwd instead of failing closed."""
    relative = tmp_path / "relative"
    relative.mkdir()

    bound = request(tmp_path, environment={"PATH": path_value, "LANG": "C"})

    if path_value.startswith(":"):
        assert bound.path_roots[0] == (tmp_path, tmp_path)
    if path_value.endswith(":"):
        assert bound.path_roots[-1] == (tmp_path, tmp_path)
    if path_value.startswith("relative:"):
        assert bound.path_roots[0] == (relative, relative)


@pytest.mark.parametrize("search_root", ["home", "temporary", "private", "darwin_tmp"])
def test_backend_omits_private_search_root(monkeypatch, tmp_path, search_root):
    """A private PATH entry does not block safe commands or gain lookup access."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    home = tmp_path.with_name(tmp_path.name + "-home")
    private = home / ".ssh"
    private.mkdir()
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setattr(
        "loop.execution.sandbox.macos._trusted_home",
        lambda: home if search_root != "temporary" else temporary / "child",
    )
    selected = {
        "home": home,
        "temporary": temporary,
        "private": private,
        "darwin_tmp": Path("/private/tmp"),
    }[search_root]
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(
        request(tmp_path, environment={"PATH": f"{selected}:/usr/bin", "LANG": "C"})
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    metadata = next(
        (line for line in profile.splitlines() if line.startswith("(allow file-read-metadata ")),
        "",
    )
    if search_root == "darwin_tmp":
        data = next(line for line in profile.splitlines() if line.startswith("(allow file-read* "))
        assert '(literal "/private/tmp")' not in data
    else:
        assert f'(literal "{selected}")' not in metadata
    assert popen.call_args.kwargs["env"]["PATH"] == f"{selected}:/usr/bin"


def test_backend_omits_private_search_root_through_symlink(monkeypatch, tmp_path):
    """A PATH symlink into a private root gains no lookup access."""
    alias = tmp_path / "search"
    alias.symlink_to(tmp_path.with_name(tmp_path.name + "-home"), target_is_directory=True)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )

    popen = MagicMock(return_value=Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(
        request(tmp_path, environment={"PATH": str(alias), "LANG": "C"})
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    metadata = next(
        (line for line in profile.splitlines() if line.startswith("(allow file-read-metadata ")),
        "",
    )
    assert f'(literal "{alias}")' not in metadata


@pytest.mark.parametrize("fake_darwin_temp", ["real_home"], indirect=True)
def test_backend_fails_closed_when_account_home_is_unavailable(monkeypatch, tmp_path):
    """Missing trusted account home invalidates native preparation before launch."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.pwd.getpwuid",
        MagicMock(side_effect=KeyError("no account")),
    )
    popen = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.INVALID
    assert "home directory is unavailable" in result.detail
    popen.assert_not_called()


@pytest.mark.parametrize("fake_darwin_temp", ["real_home"], indirect=True)
@pytest.mark.parametrize("home_case", ["valid", "wrong_owner", "missing"])
def test_backend_validates_account_home_without_ambient_home(monkeypatch, tmp_path, home_case):
    """Trusted account lookup accepts a real home and rejects unsafe or absent homes."""

    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    home = tmp_path.with_name(tmp_path.name + "-account")
    if home_case != "missing":
        home.mkdir()
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.pwd.getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(home)),
    )
    if home_case == "wrong_owner":
        original_stat = Path.stat

        def changed_owner(path, *args, **kwargs):
            """Return a foreign UID only for the controlled account home."""
            details = original_stat(path, *args, **kwargs)
            if path == home:
                values = list(details)
                values[4] = os.getuid() + 1
                return os.stat_result(values)
            return details

        monkeypatch.setattr(Path, "stat", changed_owner)
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(
        request(tmp_path, environment={"PATH": str(home), "LANG": "C"})
    )
    if home_case == "valid":
        assert result.outcome is SandboxOutcome.COMPLETED
        popen.assert_called_once()
    else:
        assert result.outcome is SandboxOutcome.INVALID
        assert "home directory" in result.detail
        popen.assert_not_called()


def test_backend_external_path_grants_only_metadata_even_with_hardlink(monkeypatch, tmp_path):
    """An unused search directory exposes no file data or hardlink target."""
    workspace = tmp_path / "work"
    workspace.mkdir()
    search = tmp_path / "search"
    search.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("fake-private")
    os.link(outside, search / "alias")
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(
        request(workspace, environment={"PATH": str(search), "LANG": "C"})
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    metadata = next(
        line for line in profile.splitlines() if line.startswith("(allow file-read-metadata ")
    )
    data = next(line for line in profile.splitlines() if line.startswith("(allow file-read* "))
    assert f'(literal "{search}")' in metadata
    assert str(search) not in data


@pytest.mark.parametrize("config_kind", ["regular", "symlink", "hardlink"])
def test_backend_grants_only_safe_python_environment_config(monkeypatch, tmp_path, config_kind):
    """A host Python search root grants its exact config or fails on an alias."""
    workspace = tmp_path / "work"
    workspace.mkdir()
    external = tmp_path / "venv"
    search = external / "bin"
    search.mkdir(parents=True)
    config = external / "pyvenv.cfg"
    if config_kind == "symlink":
        secret = external / "secret"
        secret.write_text("private")
        config.symlink_to(secret)
    else:
        config.write_text("home = /usr/local/bin")
        if config_kind == "hardlink":
            os.link(config, external / "alias")
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(
        request(workspace, read_roots=(search,), environment={"PATH": str(search), "LANG": "C"})
    )
    if config_kind == "regular":
        assert result.outcome is SandboxOutcome.COMPLETED
        assert f'(literal "{config}")' in popen.call_args.args[0][2]
        assert f'(subpath "{external}")' not in popen.call_args.args[0][2]
    else:
        assert result.outcome is SandboxOutcome.INVALID
        assert "Unsafe Python environment configuration" in result.detail
        popen.assert_not_called()


def test_backend_records_tagged_kernel_denial_without_losing_shell_exit(monkeypatch, tmp_path):
    """A tagged child file denial remains distinct from the shell exit status."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: cat(123) deny(1) file-read-data /outside/secret\n{tag}")
    result, popen = run_with_log(
        monkeypatch,
        tmp_path,
        LogProcess(records, running=True),
        output="/outside/secret: Permission denied",
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 1
    assert result.stderr == "/outside/secret: Permission denied"
    assert "/outside/secret" in result.observed_denial
    assert popen.call_args_list[1].args[0][0] == "/usr/bin/log"


def test_backend_preserves_nonstandard_exit_after_real_denial(monkeypatch, tmp_path):
    """A handled kernel denial and matching stderr cannot override exit code seven."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: cat(123) deny(1) file-read-data /outside/secret\n{tag}")
    result, _ = run_with_log(
        monkeypatch,
        tmp_path,
        LogProcess(records),
        output="/outside/secret: Operation not permitted",
        returncode=7,
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 7
    assert "/outside/secret" in result.observed_denial


def test_backend_bounds_denial_log_cleanup_without_masking_shell_result(monkeypatch, tmp_path):
    """A log child that ignores termination cannot extend command cleanup indefinitely."""
    monkeypatch.setattr("loop.execution.sandbox.macos._DENIAL_LOG_ATTEMPTS", 1)
    monkeypatch.setattr("loop.execution.sandbox.macos._DENIAL_LOG_WAIT_SECONDS", 0.005)
    monkeypatch.setattr("loop.execution.sandbox.macos._DENIAL_STREAM_SHUTDOWN_SECONDS", 0.01)
    log_process = LogProcess(
        running=True,
        wait_error=subprocess.TimeoutExpired("/usr/bin/log", 0.01),
        stuck_after_kill=True,
    )
    started = time.monotonic()

    try:
        result, _ = run_with_log(
            monkeypatch,
            tmp_path,
            log_process,
            output="Permission denied",
            returncode=23,
        )
    finally:
        log_process.finish()

    assert time.monotonic() - started < 0.5
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 23
    assert result.observed_denial == ""
    assert log_process.killed
    assert log_process.wait_timeouts == [0.01]
    assert log_process.stdout.closed


def test_backend_keeps_shell_result_when_log_poll_and_pipe_close_fail(monkeypatch, tmp_path):
    """Diagnostic cleanup errors cannot replace a completed shell result."""
    monkeypatch.setattr("loop.execution.sandbox.macos._DENIAL_LOG_ATTEMPTS", 1)
    process = PollErrorLogProcess()

    result, _ = run_with_log(
        monkeypatch,
        tmp_path,
        process,
        output="Permission denied",
        returncode=19,
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 19
    assert result.observed_denial == ""
    assert process.kill_attempts == 1
    assert process.stdout.closed
    assert process.wait_timeouts == [1.0]


def test_backend_attributes_denial_after_kernel_log_delay(monkeypatch, tmp_path):
    """A denial arriving after the first snapshot still produces a host offer outcome."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: cat(123) deny(1) file-read-data /outside/secret\n{tag}")

    result, popen = run_with_log(
        monkeypatch,
        tmp_path,
        LogProcess(),
        later_records=records,
        output="/outside/secret: Permission denied",
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    assert "/outside/secret" in result.observed_denial
    assert popen.call_count == 3


def test_backend_prioritizes_network_denial_over_incidental_file_read(monkeypatch, tmp_path):
    """A later network denial supplies the actionable diagnostic for a child."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: Python(123) deny(1) file-read-data /venv/pyvenv.cfg\n{tag}")
    records += kernel_record(f"Sandbox: Python(123) deny(1) network-outbound remote:*:443\n{tag}")
    result, _ = run_with_log(
        monkeypatch, tmp_path, LogProcess(records), output="socket.connect: Permission denied"
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    assert "network-outbound" in result.observed_denial


def test_backend_does_not_attribute_unrelated_network_permission_text(monkeypatch, tmp_path):
    """A generic permission phrase does not tie a network denial to command failure."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: Python(123) deny(1) network-outbound remote:*:443\n{tag}")

    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess(records))

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "output", ["", "OSError: [Errno 9] Bad file descriptor", "Permission denied"]
)
def test_backend_preserves_nonzero_after_incidental_read_denial(monkeypatch, tmp_path, output):
    """A Python startup probe cannot turn an unrelated exit into a host offer."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: Python(123) deny(1) file-read-data /venv/pyvenv.cfg\n{tag}")

    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess(records), output=output)

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 1


def test_backend_attributes_a_denied_symlink_target(monkeypatch, tmp_path):
    """A reported alias path can identify the kernel's denied canonical target."""
    target = tmp_path / "secret"
    target.write_text("private")
    (tmp_path / "alias").symlink_to(target)
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: cat(123) deny(1) file-read-data {target}\n{tag}")

    result, _ = run_with_log(
        monkeypatch, tmp_path, LogProcess(records), output="alias: Permission denied"
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    assert str(target) in result.observed_denial


def test_backend_rejects_relative_denial_path(monkeypatch, tmp_path):
    """A kernel record without an absolute denied path cannot justify host retry."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: cat(123) deny(1) file-read-data relative\n{tag}")

    result, _ = run_with_log(
        monkeypatch, tmp_path, LogProcess(records), output="relative: Permission denied"
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 1


def test_backend_ignores_unresolvable_stderr_path(monkeypatch, tmp_path):
    """A malformed child path cannot turn an ordinary exit into a host offer."""
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        """Simulate a candidate path that fails canonicalization."""
        if path.name == "loop":
            raise RuntimeError("symlink loop")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: cat(123) deny(1) file-read-data /outside/secret\n{tag}")

    result, _ = run_with_log(
        monkeypatch, tmp_path, LogProcess(records), output="loop: Permission denied"
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 1


def test_backend_prefers_write_over_read_and_keeps_higher_priority(monkeypatch, tmp_path):
    """Later lower-priority log records do not replace an actionable write denial."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: bash(123) deny(1) file-write-data /outside/write\n{tag}")
    records += kernel_record(f"Sandbox: cat(123) deny(1) file-read-data /outside/write\n{tag}")
    result, _ = run_with_log(
        monkeypatch, tmp_path, LogProcess(records), output="/outside/write: Permission denied"
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    assert "/outside/write" in result.observed_denial


def test_backend_skips_log_inspection_after_deadline(monkeypatch, tmp_path):
    """A spent request deadline cannot start a fresh log query."""
    result, popen = run_with_log(monkeypatch, tmp_path, None, expire_after_wait=True)
    assert result.outcome is SandboxOutcome.COMPLETED
    assert popen.call_count == 1


def test_backend_bounds_log_bytes(monkeypatch, tmp_path):
    """Oversized log output cannot grow diagnostic memory without limit."""
    monkeypatch.setattr("loop.execution.sandbox.macos._LOG_LIMIT", 8)
    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess(b"long record without newline"))
    assert result.outcome is SandboxOutcome.COMPLETED


def test_backend_handles_log_query_without_readable_output(monkeypatch, tmp_path):
    """An idle query returns an ordinary nonzero result on its bounded timeout."""
    monkeypatch.setattr("loop.execution.sandbox.macos.select.select", lambda *args: ([], [], []))
    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess())
    assert result.outcome is SandboxOutcome.COMPLETED


def test_backend_preserves_result_on_log_read_error(monkeypatch, tmp_path):
    """A diagnostic read failure cannot turn an ordinary exit into a denial."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.select.select", MagicMock(side_effect=OSError("closed"))
    )
    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess())
    assert result.outcome is SandboxOutcome.COMPLETED


@pytest.mark.parametrize(
    "records",
    [
        b"not json\n",
        b"42\n",
        b'{"processID":0,"processImagePath":"/kernel","senderImagePath":null}\n',
        b'{"processID":0,"processImagePath":"/kernel","senderImagePath":"/Sandbox.kext/Sandbox","eventMessage":42}\n',
        kernel_record("Sandbox: cat(123) deny(1) file-read-data /outside\nWRONG"),
        kernel_record(
            "Sandbox: cat(123) deny(1) file-read-data /outside\nLOOP_SBX_" + "b" * 32,
            process_id=123,
        ),
        kernel_record(
            "Sandbox: cat(123) deny(1) file-read-data /outside\nLOOP_SBX_" + "b" * 32,
            image="/usr/bin/logger",
        ),
        kernel_record(
            "Sandbox: cat(123) deny(1) file-read-data /outside\nLOOP_SBX_" + "b" * 32,
            sender="/usr/bin/logger",
        ),
        kernel_record("Sandbox: cat(123) deny(1) file-write-data /dev/tty\nLOOP_SBX_" + "b" * 32),
        kernel_record(
            "Sandbox: uv(123) deny(1) file-read-data "
            "/Library/Preferences/Logging/com.apple.diagnosticd.filter.plist\nLOOP_SBX_" + "b" * 32
        ),
        kernel_record(
            "Sandbox: Python(123) deny(1) network-outbound /private/var/run/syslog\nLOOP_SBX_"
            + "b" * 32
        ),
        kernel_record(
            "Sandbox: Python(123) deny(1) network-outbound /var/run/syslog\nLOOP_SBX_" + "b" * 32
        ),
        kernel_record("Sandbox: cat(123) deny(1) sysctl-read kern.bootargs\nLOOP_SBX_" + "b" * 32),
        kernel_record("Sandbox: cat(123) deny(1) file-read-metadata /tmp\nLOOP_SBX_" + "b" * 32),
        kernel_record("Sandbox: cat(123) allow file-read-data /outside\nLOOP_SBX_" + "b" * 32),
        kernel_record("Sandbox: cat(123) deny(1) file-read-data /outside"),
    ],
)
def test_backend_does_not_infer_denial_from_untrusted_or_irrelevant_logs(
    monkeypatch, tmp_path, records
):
    """Only tagged kernel records for relevant OS operations classify a failure."""
    result, _ = run_with_log(
        monkeypatch, tmp_path, LogProcess(records), output="Operation not permitted"
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 1
    assert not result.observed_denial


def test_backend_preserves_nonzero_when_log_query_fails(monkeypatch, tmp_path):
    """A broken log reader cannot invent a sandbox denial or host retry."""
    result, _ = run_with_log(monkeypatch, tmp_path, None, log_error=OSError("log unavailable"))
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 1


def test_backend_preserves_nonzero_when_log_pipe_is_missing(monkeypatch, tmp_path):
    """An unavailable log stream leaves an ordinary command failure intact."""
    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess(stdout=False))
    assert result.outcome is SandboxOutcome.COMPLETED


@pytest.mark.parametrize("fake_darwin_temp", [False], indirect=True)
@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "root_mode",
        "root_owner",
        "entry_directory",
        "entry_hardlink",
        "entry_owner",
        "object_exists",
        "root_changed",
    ],
)
def test_backend_validates_darwin_compiler_cache(monkeypatch, tmp_path, case):
    """Compiler cache exceptions reject unsafe entries before child launch."""
    user_temp = tmp_path / "darwin-user-temp"
    user_temp.mkdir(mode=0o700)
    (user_temp / "unrelated").write_text("safe")
    if case == "valid":
        (user_temp / "xcrun_db").write_text("cache")
    if case == "root_mode":
        user_temp.chmod(0o755)
    if case in {"entry_directory", "entry_hardlink", "entry_owner"}:
        target = user_temp / "xcrun_db-ab"
        if case == "entry_directory":
            target.mkdir()
        else:
            target.write_text("cache")
        if case == "entry_hardlink":
            os.link(target, user_temp / "other")
    if case == "object_exists":
        (user_temp / "main-abcdef.o").write_text("object")
    if case in {"root_owner", "entry_owner"}:
        actual_uid = os.getuid()
        values = [actual_uid + 1] if case == "root_owner" else [actual_uid, actual_uid + 1]
        monkeypatch.setattr("loop.execution.sandbox.macos.os.getuid", MagicMock(side_effect=values))
    if case == "root_changed":
        original_lstat = Path.lstat
        calls = 0

        def changed_lstat(path):
            """Return a changed inode on the second cache-root inspection."""
            nonlocal calls
            details = original_lstat(path)
            if path == user_temp:
                calls += 1
                if calls == 2:
                    details = list(details)
                    details[1] += 1
                    return os.stat_result(details)
            return details

        monkeypatch.setattr(Path, "lstat", changed_lstat)
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stdout=str(user_temp), stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    if case == "valid":
        assert result.outcome is SandboxOutcome.COMPLETED
        assert "xcrun_db" in popen.call_args.args[0][2]
    else:
        assert result.outcome is SandboxOutcome.INVALID
        popen.assert_not_called()


@pytest.mark.parametrize("fake_darwin_temp", [False], indirect=True)
def test_backend_reuses_trusted_darwin_temp_path_but_revalidates_entries(monkeypatch, tmp_path):
    """Cached getconf output never bypasses per-command compiler-cache validation."""
    user_temp = tmp_path / "darwin-user-temp"
    user_temp.mkdir(mode=0o700)
    probe = MagicMock(returncode=0, stdout=str(user_temp), stderr="")
    run = MagicMock(return_value=probe)
    launch = MagicMock(return_value=Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.run", run)
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", launch)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    approved = request(tmp_path)

    backend = MacOSSeatbeltBackend()
    assert backend.run(approved).outcome is SandboxOutcome.COMPLETED
    (user_temp / "xcrun_db-new").mkdir()
    assert backend.run(approved).outcome is SandboxOutcome.INVALID

    assert run.call_count == 2
    launch.assert_called_once()


@pytest.mark.parametrize("fake_darwin_temp", [False], indirect=True)
def test_backend_bounds_getconf_by_remaining_deadline(monkeypatch, tmp_path):
    """A stalled compiler-cache lookup has a finite timeout and starts no shell."""
    lookup = MagicMock(side_effect=subprocess.TimeoutExpired("getconf", 0.01))
    launch = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.run", lookup)
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", launch)
    result = MacOSSeatbeltBackend().run(request(tmp_path, deadline=time.monotonic() + 0.1))
    assert result.outcome is SandboxOutcome.TIMED_OUT
    assert 0 < lookup.call_args.kwargs["timeout"] <= 0.1
    launch.assert_not_called()


def test_backend_rejects_bad_profile_before_script(monkeypatch, tmp_path):
    """A launcher validation error is typed unavailable and starts no command."""
    probe = MagicMock(returncode=65, stderr="bad profile")
    popen = MagicMock()
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run", MagicMock(return_value=probe)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.UNAVAILABLE
    assert "exit 65" in result.detail
    popen.assert_not_called()


def test_backend_probes_parent_once_and_launches_every_profile(monkeypatch, tmp_path):
    """A verified parent reuses readiness while each command gets a Seatbelt child."""
    probe = MagicMock(returncode=0, stderr="")
    run = MagicMock(return_value=probe)
    launch = MagicMock(side_effect=[Process(), Process()])
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.run", run)
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", launch)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    approved = request(tmp_path)
    backend = MacOSSeatbeltBackend()

    assert backend.run(approved).outcome is SandboxOutcome.COMPLETED
    assert backend.run(approved).outcome is SandboxOutcome.COMPLETED

    run.assert_called_once()
    assert launch.call_count == 2


@pytest.mark.parametrize("failure", [PermissionError("denied"), subprocess.TimeoutExpired([], 1)])
def test_backend_classifies_preparation_and_probe_errors(monkeypatch, tmp_path, failure):
    """Preparation errors remain unavailable and probe expiry remains a timeout."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run", MagicMock(side_effect=failure)
    )
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    assert result.outcome is (
        SandboxOutcome.TIMED_OUT
        if isinstance(failure, subprocess.TimeoutExpired)
        else SandboxOutcome.UNAVAILABLE
    )


def test_backend_cancels_before_child_launch(monkeypatch, tmp_path):
    """An interrupted readiness check never starts a shell child."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run", MagicMock(side_effect=KeyboardInterrupt)
    )
    launch = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", launch)

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.CANCELLED
    launch.assert_not_called()


def test_backend_rejects_hardlinked_writable_file(tmp_path):
    """The mandatory preflight rejects preexisting hardlinks before launch."""
    source = tmp_path / "one"
    source.write_text("safe")
    os.link(source, tmp_path / "two")
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    assert "hardlinked file" in result.detail


@pytest.mark.parametrize("root_kind", ["workspace", "extra"])
def test_backend_rejects_hardlinked_read_root_even_when_read_only(tmp_path, root_kind):
    """Read-only grants cannot expose a protected inode through a hardlink alias."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    granted = workspace if root_kind == "workspace" else extra
    private = granted / ".ssh"
    private.mkdir()
    source = private / "id_ed25519"
    source.write_text("fake-credential")
    os.link(source, granted / "public-alias")

    result = MacOSSeatbeltBackend().run(
        request(
            workspace,
            read_roots=(extra,) if root_kind == "extra" else (),
            write_roots=(),
        )
    )

    assert result.outcome is SandboxOutcome.INVALID
    assert "hardlinked file" in result.detail


@pytest.mark.parametrize("root_kind", ["workspace", "extra"])
@pytest.mark.parametrize("name", [".aws", ".AWS"])
def test_backend_rejects_protected_directory_as_read_root(tmp_path, root_kind, name):
    """A protected directory selected as a grant root never becomes readable."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    private = (workspace if root_kind == "workspace" else tmp_path) / name
    private.mkdir()
    result = MacOSSeatbeltBackend().run(
        request(
            private if root_kind == "workspace" else workspace,
            read_roots=(private,) if root_kind == "extra" else (),
            write_roots=(),
        )
    )
    assert result.outcome is SandboxOutcome.INVALID
    assert "Protected directory cannot be a read root" in result.detail


def test_backend_rejects_symlink_replacement_at_preflight(monkeypatch, tmp_path):
    """A writable root replaced by a symlink is rejected by the launch preflight."""
    value = request(tmp_path)
    real = tmp_path.with_name(tmp_path.name + "-real")
    tmp_path.rename(real)
    tmp_path.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(SandboxRequest, "paths_are_current", lambda self: True)
    result = MacOSSeatbeltBackend().run(value)
    assert "not a real directory" in result.detail


def test_backend_walks_nested_directories(monkeypatch, tmp_path):
    """Preflight traverses nested writable directories before launch."""
    (tmp_path / "nested").mkdir()
    (tmp_path / "regular.txt").write_text("safe")
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=65, stderr="stop")),
    )
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    assert "exit 65" in result.detail


def test_backend_detects_root_replacement_during_walk(monkeypatch, tmp_path):
    """A writable root replaced during its walk fails before launcher validation."""
    original_scandir = os.scandir

    class ReplacingScan:
        """Replace the workspace pathname when its directory scan closes."""

        def __init__(self, path):
            self.path = Path(path) if not isinstance(path, int) else None
            self.scan = original_scandir(path)

        def __enter__(self):
            return self.scan.__enter__()

        def __exit__(self, *args):
            result = self.scan.__exit__(*args)
            if self.path == tmp_path:
                old = tmp_path.with_name(tmp_path.name + "-old")
                tmp_path.rename(old)
                tmp_path.mkdir()
            return result

    monkeypatch.setattr("loop.execution.sandbox.macos.os.scandir", ReplacingScan)
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    assert "changed during preflight" in result.detail


@pytest.mark.parametrize("states", [(True, False), (True, True, False), (True, True, True, False)])
def test_backend_revalidates_identity_around_launcher(monkeypatch, tmp_path, states):
    """Identity replacement through final monitor setup prevents shell execution."""
    value = request(tmp_path)
    current = iter(states)
    monkeypatch.setattr(SandboxRequest, "paths_are_current", lambda self: next(current))
    probe = MagicMock(returncode=0, stderr="")
    popen = MagicMock()
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run", MagicMock(return_value=probe)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.subprocess.Popen", popen)
    result = MacOSSeatbeltBackend().run(value)
    assert result.outcome is SandboxOutcome.STALE
    assert "path identity changed" in result.detail
    popen.assert_not_called()


def test_backend_times_out_when_capture_reader_stays_alive(monkeypatch, tmp_path):
    """A reader that misses the request deadline yields a timed-out result."""

    class StuckThread:
        """Model a pipe reader that never completes."""

        def __init__(self, **kwargs):
            pass

        def start(self):
            """Leave the simulated reader pending."""

        def join(self, timeout=None):
            """Leave the simulated reader pending."""

        def is_alive(self):
            """Report that capture remains pending."""
            return True

    process = Process()
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.Popen", MagicMock(return_value=process)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.threading.Thread", StuckThread)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    assert result.outcome is SandboxOutcome.TIMED_OUT


@pytest.mark.parametrize(
    ("process", "outcome"),
    [
        (Process(stdout=None), SandboxOutcome.UNAVAILABLE),
        (
            Process(
                stdout="",
                stderr="sandbox-exec: sandbox_apply: Operation not permitted",
                returncode=71,
                started=False,
            ),
            SandboxOutcome.UNAVAILABLE,
        ),
        (
            Process(stdout="", stderr="sandbox-exec: forged failure", returncode=71),
            SandboxOutcome.COMPLETED,
        ),
        (
            Process(stdout="", stderr="sandbox-exec: forged failure", returncode=65),
            SandboxOutcome.COMPLETED,
        ),
        (Process(stdout="", stderr="ordinary failure", returncode=65), SandboxOutcome.COMPLETED),
        (
            Process(wait_error=subprocess.TimeoutExpired(["sandbox-exec"], 1)),
            SandboxOutcome.TIMED_OUT,
        ),
        (Process(wait_error=KeyboardInterrupt()), SandboxOutcome.CANCELLED),
    ],
)
def test_backend_classifies_child_capture_failures(monkeypatch, tmp_path, process, outcome):
    """Missing pipes, nested launcher failure, and timeout each fail closed."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.subprocess.Popen", MagicMock(return_value=process)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos._denial_from_log", lambda *args: None)
    monkeypatch.setattr("loop.execution.sandbox.macos.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    assert result.outcome is outcome
    if process.stdout is None:
        assert result.possible_effects
    if isinstance(process.wait_error, subprocess.TimeoutExpired):
        assert result.stdout == "ok"
