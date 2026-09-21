"""Test the isolated fixed-argv process primitive without starting child processes."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.execution.infrastructure import (
    InfrastructureProcessCommand,
    InfrastructureProcessRunner,
    ProcessCancelledError,
    VerifiedDirectory,
    identify_host_directory,
    identify_host_executable,
    sealed_environment,
    validate_installed_executable,
)
from loop.execution.infrastructure import process as process_module
from loop.execution.runtime.bootstrap import InstalledRuntime
from loop.execution.runtime.lease import create_lease
from loop.execution.runtime.models import (
    AcquisitionKind,
    Artifact,
    ArtifactRole,
    InstallLayout,
    PlatformSelector,
)


class _Process:
    """Provide deterministic in-memory pipes for the process primitive."""

    pid = 123
    returncode = 0

    def __init__(self) -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(b"stdout")
        self.stderr = io.BytesIO(b"stderr")

    def poll(self) -> int:
        """Report immediate completion."""
        return 0

    def wait(self) -> int:
        """Report successful reaping."""
        return 0


def _installed_executable(tmp_path: Path):
    """Create one real lease-bound executable identity without running it."""
    root = tmp_path / "runtime"
    path = root / "artifacts" / "tool" / ("a" * 64) / "tool"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"tool")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = Artifact(
        artifact_id="tool",
        version="1",
        role=ArtifactRole.NERDCTL,
        platform=PlatformSelector(os="linux", architecture="amd64"),
        source="https://example.test/tool",
        size=4,
        digest="b" * 64,
        acquisition=AcquisitionKind.FILE,
        media_type="application/octet-stream",
        layout=InstallLayout(files=("tool",), executables=("tool",), identities={"tool": identity}),
        sbom="https://example.test/sbom",
        notices="https://example.test/notices",
    )
    runtime = InstalledRuntime(
        "manifest",
        "e" * 64,
        {"tool": path.parent},
        create_lease(root, "manifest", "e" * 64, 60),
        {"tool": artifact},
        root,
    )
    return runtime.executable("tool", "tool")


def test_runner_forwards_exact_sealed_invocation_and_bounds_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The primitive adds no argv or environment and returns bounded independent pipes."""
    executable = SimpleNamespace(path=tmp_path / "tool")
    monkeypatch.setattr(process_module, "validate_installed_executable", lambda value: None)
    observed: dict[str, object] = {}

    def popen(argv: tuple[str, ...], **kwargs: object) -> _Process:
        """Capture the exact child boundary without creating a process."""
        observed.update(argv=argv, **kwargs)
        return _Process()

    monkeypatch.setattr(process_module.subprocess, "Popen", popen)
    command = InfrastructureProcessCommand(
        (str(executable.path), "fixed"),
        {"LANG": "C"},
        tmp_path,
        "runtime.inspect",
        executable,  # type: ignore[arg-type]
    )
    result = InfrastructureProcessRunner(8).run(command, stdin=b"input")
    assert observed["argv"] == command.argv
    assert observed["env"] == {"LANG": "C"}
    assert observed["shell"] is False
    assert result.stdout == b"stdout"
    assert result.stderr == b"stderr"
    assert result.stdout_truncated is False
    assert result.stderr_truncated is False


def test_runner_rejects_unclassified_system_commands_and_invalid_bounds(tmp_path: Path) -> None:
    """Only codesign may bypass manifest provenance and all buffer bounds are positive."""
    with pytest.raises(ValueError, match="positive"):
        InfrastructureProcessRunner(0)
    command = InfrastructureProcessCommand(
        ("/bin/sh", "-c", "true"), {}, tmp_path, "unclassified", None
    )
    with pytest.raises(ValueError, match="not classified"):
        InfrastructureProcessRunner().run(command)
    with pytest.raises(ValueError, match="bounds"):
        InfrastructureProcessRunner(1).run(
            InfrastructureProcessCommand(
                ("/usr/bin/codesign", "--help"), {}, tmp_path, "codesign", None
            ),
            stdin=b"too large",
        )


def test_sealed_environment_uses_only_declared_paths_and_fixed_entries() -> None:
    """A child environment is rebuilt from declared bundle directories and explicit values."""
    executable = SimpleNamespace(path_directories=("/private/runtime/bin",))
    environment = sealed_environment(executable, {"LIMA_HOME": "/private/lima"})  # type: ignore[arg-type]
    assert environment == {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/private/runtime/bin:/usr/bin:/bin",
        "LIMA_HOME": "/private/lima",
    }
    assert sealed_environment(executable)["PATH"] == "/private/runtime/bin:/usr/bin:/bin"  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="environment"):
        sealed_environment(executable, {"BAD-KEY": "value"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonabsolute"):
        sealed_environment(SimpleNamespace(path_directories=("relative",)))  # type: ignore[arg-type]


def test_runner_wraps_spawn_reader_and_broken_pipe_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch, pipe, and reader failures remain bounded and reap any created process."""
    executable = SimpleNamespace(path=tmp_path / "tool")
    command = InfrastructureProcessCommand(
        (str(executable.path),),
        {},
        tmp_path,
        "runtime.test",
        executable,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(process_module, "validate_installed_executable", lambda value: None)
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("spawn")),
    )
    with pytest.raises(RuntimeError, match="could not start"):
        InfrastructureProcessRunner().run(command)

    class _BrokenStream(io.BytesIO):
        """Raise an OS error while draining."""

        def read(self, _size: int = -1) -> bytes:
            """Reject the read."""
            raise OSError("read")

    process = _Process()
    process.stdout = _BrokenStream()
    monkeypatch.setattr(process_module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(process_module.os, "killpg", lambda *_args: None)
    with pytest.raises(RuntimeError, match="reader failed"):
        InfrastructureProcessRunner().run(command)

    class _BrokenInput(io.BytesIO):
        """Raise a broken pipe but permit the child result to complete."""

        def write(self, _data: bytes) -> int:
            """Reject input."""
            raise BrokenPipeError

    process = _Process()
    process.stdin = _BrokenInput()
    monkeypatch.setattr(process_module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    assert InfrastructureProcessRunner().run(command, stdin=b"x").exit_code == 0

    class _OsErrorInput(io.BytesIO):
        """Raise a non-broken-pipe OS error."""

        def write(self, _data: bytes) -> int:
            """Reject input with a generic OS failure."""
            raise OSError("input")

    process = _Process()
    process.stdin = _OsErrorInput()
    monkeypatch.setattr(process_module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    with pytest.raises(RuntimeError, match="could not start"):
        InfrastructureProcessRunner().run(command, stdin=b"x")

    class _ChunkedStream:
        """Return multiple chunks to cover post-limit draining."""

        def __init__(self) -> None:
            self.values = iter((b"1234", b"5", b""))

        def read(self, _size: int) -> bytes:
            """Return the next deterministic chunk."""
            return next(self.values)

        def close(self) -> None:
            """Provide the parent-owned stream cleanup interface."""

    process = _Process()
    process.stdout = _ChunkedStream()  # type: ignore[assignment]
    monkeypatch.setattr(process_module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    assert InfrastructureProcessRunner(4).run(command).stdout == b"1234"

    process = _Process()
    process.poll = lambda: None  # type: ignore[method-assign]
    monkeypatch.setattr(process_module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        process_module, "_wait_for_process", lambda *_args: (_ for _ in ()).throw(TimeoutError())
    )
    monkeypatch.setattr(
        process_module.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    with pytest.raises(RuntimeError, match="deadline"):
        InfrastructureProcessRunner().run(command)


def test_command_and_provenance_validation_reject_every_changed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid commands, leases, paths, descriptors, content, and modes fail closed."""
    executable = _installed_executable(tmp_path)
    validate_installed_executable(executable)
    invalid = InfrastructureProcessCommand((), {}, tmp_path, "", executable)
    with pytest.raises(ValueError, match="command"):
        InfrastructureProcessRunner().run(invalid)
    monkeypatch.setattr(process_module, "validate_installed_executable", lambda value: None)
    mismatch = InfrastructureProcessCommand(("/wrong",), {}, tmp_path, "x", executable)
    with pytest.raises(ValueError, match="inconsistent"):
        InfrastructureProcessRunner().run(mismatch)
    monkeypatch.undo()

    malformed = executable.__class__(
        executable.artifact_id,
        "f" * 64,
        executable.path,
        executable.runtime_root,
        executable.artifact_root,
        executable.lease,
        executable.expected_identity,
        executable.path_directories,
    )
    with pytest.raises(ValueError, match="active private runtime"):
        validate_installed_executable(malformed)
    executable.lease.path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        validate_installed_executable(executable)
    executable.lease.path.write_text(
        json.dumps(
            {
                "owner_id": executable.lease.owner_id,
                "manifest_digest": executable.lease.manifest_digest,
                "artifact_set_digest": executable.lease.artifact_set_digest,
                "created": executable.lease.created,
                "expiry": time.time() - 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not active"):
        validate_installed_executable(executable)


def test_provenance_rejects_missing_components_and_changed_content(tmp_path: Path) -> None:
    """Descriptor traversal and final content identity changes cannot authorize launch."""
    executable = _installed_executable(tmp_path)
    executable.path.write_bytes(b"changed")
    executable.path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    with pytest.raises(ValueError, match="identity changed"):
        validate_installed_executable(executable)
    executable.path.unlink()
    with pytest.raises(ValueError, match="provenance"):
        validate_installed_executable(executable)

    fresh = _installed_executable(tmp_path / "fresh")
    moved_artifacts = fresh.runtime_root / "artifacts-moved"
    (fresh.runtime_root / "artifacts").rename(moved_artifacts)
    with pytest.raises(ValueError, match="provenance"):
        validate_installed_executable(fresh)


def test_host_authorities_reject_links_modes_and_changed_identity(tmp_path: Path) -> None:
    """Host executable and cwd authorities fail closed on ambiguous or changed objects."""
    executable_path = tmp_path / "tool"
    executable_path.write_bytes(b"tool")
    executable_path.chmod(0o700)
    executable = identify_host_executable(executable_path)
    directory = identify_host_directory(tmp_path)

    process_module.validate_verified_executable(executable)
    descriptor = process_module.open_verified_directory(directory)
    process_module.os.close(descriptor)
    executable_path.write_bytes(b"changed")
    executable_path.chmod(0o700)
    with pytest.raises(ValueError, match="identity changed"):
        process_module.validate_verified_executable(executable)
    executable_path.chmod(0o600)
    with pytest.raises(ValueError, match="executable regular file"):
        identify_host_executable(executable_path)
    with pytest.raises(ValueError, match="absolute"):
        identify_host_executable(Path("relative"))
    linked = tmp_path / "linked"
    linked.symlink_to(executable_path)
    with pytest.raises(ValueError, match="could not be verified"):
        identify_host_executable(linked)
    with pytest.raises(ValueError, match="identity changed"):
        process_module.open_verified_directory(
            VerifiedDirectory(tmp_path, directory.device, directory.inode + 1)
        )


def test_host_command_validation_binds_executable_and_cwd(tmp_path: Path) -> None:
    """The common primitive accepts only the exact verified host executable and cwd."""
    executable_path = tmp_path / "tool"
    executable_path.write_bytes(b"tool")
    executable_path.chmod(0o700)
    executable = identify_host_executable(executable_path)
    directory = identify_host_directory(tmp_path)
    command = InfrastructureProcessCommand(
        (str(executable_path),), {}, tmp_path, "host.execute", executable, directory
    )

    process_module.validate_infrastructure_command(command)
    with pytest.raises(ValueError, match="cwd identity"):
        process_module.validate_infrastructure_command(
            InfrastructureProcessCommand(
                command.argv,
                {},
                tmp_path,
                command.operation,
                executable,
                VerifiedDirectory(tmp_path / "other", directory.device, directory.inode),
            )
        )
    with pytest.raises(ValueError, match="executable identity"):
        process_module.validate_infrastructure_command(
            InfrastructureProcessCommand(
                ("/wrong",), {}, tmp_path, command.operation, executable, directory
            )
        )


def test_runner_revalidates_cwd_reports_start_and_cleans_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified cwd is checked around launch and host cleanup targets the process group."""
    executable_path = tmp_path / "tool"
    executable_path.write_bytes(b"tool")
    executable_path.chmod(0o700)
    command = InfrastructureProcessCommand(
        (str(executable_path),),
        {},
        tmp_path,
        "host.execute",
        identify_host_executable(executable_path),
        identify_host_directory(tmp_path),
    )
    process = _Process()
    starts: list[str] = []
    kills: list[int] = []
    monkeypatch.setattr(process_module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(process_module.os, "killpg", lambda pid, _signal: kills.append(pid))

    result = InfrastructureProcessRunner().run(
        command, started=lambda: starts.append("started"), cleanup_descendants=True
    )

    assert result.exit_code == 0
    assert starts == ["started"]
    assert kills == [process.pid]


def test_runner_closes_changed_cwd_and_maps_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cwd races fail before start and cancellation kills the owned process group."""
    executable_path = tmp_path / "tool"
    executable_path.write_bytes(b"tool")
    executable_path.chmod(0o700)
    command = InfrastructureProcessCommand(
        (str(executable_path),),
        {},
        tmp_path,
        "host.execute",
        identify_host_executable(executable_path),
        identify_host_directory(tmp_path),
    )
    process = _Process()
    monkeypatch.setattr(process_module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(process_module.os, "killpg", lambda *_args: None)
    real_open = process_module.open_verified_directory
    opens = 0

    def changed_cwd(directory):
        """Permit the pre-launch open and reject the post-launch identity check."""
        nonlocal opens
        opens += 1
        if opens == 2:
            raise ValueError("changed")
        return real_open(directory)

    monkeypatch.setattr(process_module, "open_verified_directory", changed_cwd)
    with pytest.raises(ValueError, match="changed"):
        InfrastructureProcessRunner().run(command)

    class _RunningProcess(_Process):
        """Remain live until best-effort cancellation reaps the process."""

        def poll(self):
            """Report a running child."""
            return

    running = _RunningProcess()
    monkeypatch.setattr(process_module, "open_verified_directory", real_open)
    monkeypatch.setattr(process_module.subprocess, "Popen", lambda *_args, **_kwargs: running)
    with pytest.raises(ProcessCancelledError, match="cancelled"):
        InfrastructureProcessRunner().run(command, cancellation=lambda: True)

    mismatched_directory = VerifiedDirectory(
        tmp_path, command.cwd_identity.device, command.cwd_identity.inode + 1
    )
    with pytest.raises(ValueError, match="identity changed"):
        InfrastructureProcessRunner().run(
            InfrastructureProcessCommand(
                command.argv,
                command.environment,
                command.cwd,
                command.operation,
                command.executable,
                mismatched_directory,
            )
        )

    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("spawn")),
    )
    with pytest.raises(process_module.ProcessSpawnError, match="could not start"):
        InfrastructureProcessRunner().run(command)


def test_runner_wall_deadline_polls_before_timing_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live classified process is polled until its wall deadline and then killed."""
    executable = SimpleNamespace(path=tmp_path / "tool")
    command = InfrastructureProcessCommand(
        (str(executable.path),),
        {},
        tmp_path,
        "runtime.test",
        executable,  # type: ignore[arg-type]
    )

    class _RunningProcess(_Process):
        """Remain live until deadline cleanup."""

        def poll(self):
            """Report a running child."""
            return

    process = _RunningProcess()
    times = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(process_module, "validate_installed_executable", lambda value: None)
    monkeypatch.setattr(process_module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(process_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(process_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(process_module.os, "killpg", lambda *_args: None)

    with pytest.raises(process_module.ProcessTimedOutError, match="deadline"):
        InfrastructureProcessRunner().run(command, deadline_seconds=1)


def test_runner_reaps_a_started_child_when_start_observation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed start audit cannot leak an already-created classified child."""
    executable = SimpleNamespace(path=tmp_path / "tool")
    command = InfrastructureProcessCommand(
        (str(executable.path),),
        {},
        tmp_path,
        "runtime.test",
        executable,  # type: ignore[arg-type]
    )
    process = _Process()
    kills: list[int] = []
    monkeypatch.setattr(process_module, "validate_installed_executable", lambda value: None)
    monkeypatch.setattr(process_module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(process_module.os, "killpg", lambda pid, _signal: kills.append(pid))

    with pytest.raises(RuntimeError, match="observer failed"):
        InfrastructureProcessRunner().run(
            command,
            started=lambda: (_ for _ in ()).throw(RuntimeError("audit unavailable")),
        )

    assert kills == [process.pid]
