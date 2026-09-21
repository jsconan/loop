"""Exercise attached OCI pipe and PTY mechanics with real local child processes."""

from __future__ import annotations

import hashlib
import io
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from loop.execution.infrastructure import InfrastructureProcessCommand, sealed_environment
from loop.execution.runtime.bootstrap import InstalledRuntime
from loop.execution.runtime.lease import create_lease
from loop.execution.runtime.models import (
    AcquisitionKind,
    Artifact,
    ArtifactRole,
    InstallLayout,
    PlatformSelector,
)
from loop.execution.sandbox.oci import process as process_module
from loop.execution.sandbox.oci.process import OciProcessSession


def _command(tmp_path: Path, body: str) -> InfrastructureProcessCommand:
    """Create one leased local transport command for attached-process checks."""
    root = tmp_path / "runtime"
    path = root / "artifacts" / "transport" / ("a" * 64) / "transport"
    path.parent.mkdir(parents=True)
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = Artifact(
        artifact_id="transport",
        version="1",
        role=ArtifactRole.LIMA,
        platform=PlatformSelector(os="macos", architecture="arm64"),
        source="https://example.test/transport",
        size=path.stat().st_size,
        digest="b" * 64,
        acquisition=AcquisitionKind.FILE,
        media_type="application/octet-stream",
        layout=InstallLayout(
            files=("transport",),
            executables=("transport",),
            identities={"transport": identity},
        ),
        sbom="https://example.test/sbom",
        notices="https://example.test/notices",
    )
    runtime = InstalledRuntime(
        "manifest",
        "e" * 64,
        {"transport": path.parent},
        create_lease(root, "manifest", "e" * 64, 60),
        {"transport": artifact},
        root,
    )
    executable = runtime.executable("transport", "transport")
    return InfrastructureProcessCommand(
        (str(path),),
        sealed_environment(executable),
        tmp_path,
        "integration.oci_process",
        executable,
    )


def _read_all(session: OciProcessSession) -> bytes:
    """Read queued output until the process has had time to terminate."""
    output = bytearray()
    while frame := session.read(timeout=0.05):
        output.extend(frame.data)
    return bytes(output)


def test_attached_pipe_preserves_stream_identity_and_stdin_eof(tmp_path: Path) -> None:
    """Pipe sessions keep streams distinct and deliver input before terminal completion."""
    session = OciProcessSession(
        _command(
            tmp_path,
            "import sys\ndata = sys.stdin.buffer.read()\n"
            "sys.stdout.buffer.write(b'out:' + data)\n"
            "sys.stderr.buffer.write(b'err')\n",
        ),
        False,
        16,
        5,
    )
    session.write(b"value")
    session.close_stdin()
    assert session.wait() == 0
    frames = []
    while frame := session.read(timeout=0.01):
        frames.append(frame)
    assert {frame.stream for frame in frames} == {"stdout", "stderr"}
    assert b"out:value" in b"".join(frame.data for frame in frames)


def test_attached_pty_merges_output_accepts_resize_and_delivers_eof(tmp_path: Path) -> None:
    """PTY sessions merge output, accept resize, and terminate after canonical EOF."""
    session = OciProcessSession(
        _command(
            tmp_path,
            "import sys\ndata = sys.stdin.buffer.read()\n"
            "sys.stdout.buffer.write(b'pty:' + data)\nsys.stdout.buffer.flush()\n",
        ),
        True,
        16,
        5,
    )
    session.resize(100, 40)
    session.write(b"value")
    session.close_stdin()
    assert session.wait() == 0
    assert b"pty:value" in _read_all(session)


def test_pty_detach_sequence_is_forwarded_without_flow_control(tmp_path: Path) -> None:
    """PTY transport forwards the upstream detach sequence as opaque bytes."""
    session = OciProcessSession(
        _command(
            tmp_path,
            "import sys, tty\ntty.setraw(sys.stdin.fileno())\n"
            "sys.stdout.buffer.write(b'ready')\nsys.stdout.buffer.flush()\n"
            "value = sys.stdin.buffer.read(2)\n"
            "raise SystemExit(0 if value == b'\\x10\\x11' else 1)\n",
        ),
        True,
        4,
        5,
    )
    assert session.read(1) is not None
    session.detach()


def test_failed_detach_closes_the_local_management_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected detach cannot leave its local management process alive."""
    session = OciProcessSession(
        _command(tmp_path, "import time\ntime.sleep(10)\n"),
        False,
        4,
        5,
    )
    monkeypatch.setattr(session, "wait", lambda: 1)
    with pytest.raises(RuntimeError, match="detach failed"):
        session.detach()
    assert session._closed.is_set()


def test_silent_pty_session_closes_without_blocking_on_its_reader(tmp_path: Path) -> None:
    """Closing a silent PTY cannot deadlock against its concurrent descriptor read."""
    session = OciProcessSession(
        _command(tmp_path, "import time\ntime.sleep(10)\n"),
        True,
        4,
        5,
    )
    assert session._master_fd is not None
    closer = threading.Thread(target=session.close, daemon=True)
    closer.start()
    closer.join(timeout=1)
    assert not closer.is_alive()


def test_silent_pipe_session_closes_without_buffered_reader_deadlock(tmp_path: Path) -> None:
    """Closing silent pipes cannot deadlock against buffered stream reader locks."""
    session = OciProcessSession(
        _command(tmp_path, "import time\ntime.sleep(10)\n"),
        False,
        4,
        5,
    )
    closer = threading.Thread(target=session.close, daemon=True)
    closer.start()
    closer.join(timeout=1)
    assert not closer.is_alive()


def test_session_bounds_terminal_operations_timeout_and_cancellation(tmp_path: Path) -> None:
    """Invalid bounds and every terminal control path fail closed and remain idempotent."""
    command = _command(tmp_path, "import time\ntime.sleep(10)\n")
    for values in ((False, 0, 1), (False, 1, 0)):
        with pytest.raises(ValueError, match="bounds"):
            OciProcessSession(command, *values)
    with pytest.raises(ValueError, match="dimensions"):
        OciProcessSession(command, False, 1, 1, 80, 24)
    for columns, rows in (
        (None, 24),
        (80, None),
        (0, 24),
        (80, 0),
        (16385, 24),
        (80, 16385),
    ):
        with pytest.raises(ValueError, match="dimensions"):
            process_module._set_pty_size(0, columns, rows)
    with pytest.raises(ValueError, match="bounds"):
        OciProcessSession(
            InfrastructureProcessCommand(command.argv, command.environment, command.cwd, "x", None),
            False,
            1,
            1,
        )
    with pytest.raises(ValueError, match="inconsistent"):
        OciProcessSession(
            InfrastructureProcessCommand(
                ("/bin/sh", "-c", "true"),
                command.environment,
                command.cwd,
                command.operation,
                command.executable,
            ),
            False,
            1,
            1,
        )
    session = OciProcessSession(command, False, 4, 5)
    assert session.read(0.01) is None
    with pytest.raises(ValueError, match="dimensions"):
        session.resize(0, 1)
    with pytest.raises(RuntimeError, match="does not own"):
        session.resize(80, 24)
    with pytest.raises(ValueError, match="input exceeds"):
        session.write(b"x" * (1024 * 1024 + 1))
    session.close_stdin()
    session.close_stdin()
    with pytest.raises(RuntimeError, match="stdin is closed"):
        session.write(b"later")
    session.close()
    session.close()

    cancelled = OciProcessSession(command, False, 4, 5)
    with pytest.raises(RuntimeError, match="cancelled"):
        cancelled.wait(lambda: True)
    timed = OciProcessSession(command, False, 4, 0.01)
    with pytest.raises(RuntimeError, match="deadline"):
        timed.wait()


def test_close_after_completion_does_not_signal_a_reaped_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing a completed session cannot signal a subsequently reused process-group ID."""
    session = OciProcessSession(_command(tmp_path, "pass\n"), False, 4, 5)
    assert session.wait() == 0
    monkeypatch.setattr(
        process_module.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(AssertionError("signalled reaped process")),
    )
    session.close()


def test_session_wraps_spawn_stdin_resize_and_descriptor_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OS failures close the owned group and surface only typed session errors."""
    command = _command(tmp_path, "import time\ntime.sleep(10)\n")
    session = OciProcessSession(command, False, 4, 5)

    class _BrokenInput:
        """Reject every input write."""

        def write(self, _data: bytes) -> None:
            """Raise a broken-pipe error."""
            raise BrokenPipeError

        def flush(self) -> None:
            """Provide the pipe interface."""

        def close(self) -> None:
            """Provide the owned-stream cleanup interface."""

    original_input = session._process.stdin
    assert original_input is not None
    original_input.close()
    session._process.stdin = _BrokenInput()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="stdin failed"):
        session.write(b"data")

    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("spawn")),
    )
    with pytest.raises(RuntimeError, match="could not start"):
        OciProcessSession(command, False, 4, 5)
    with pytest.raises(RuntimeError, match="could not start"):
        OciProcessSession(command, True, 4, 5)


def test_pty_and_reader_failure_injection_remains_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PTY EOF, resize, reader, saturation, expiry, and close races cannot leak a child."""
    command = _command(tmp_path, "import time\ntime.sleep(10)\n")
    session = OciProcessSession(command, True, 4, 5)
    monkeypatch.setattr(
        process_module.os,
        "write",
        lambda *_args: (_ for _ in ()).throw(OSError("closed")),
    )
    session.close_stdin()
    monkeypatch.undo()
    monkeypatch.setattr(
        process_module.fcntl,
        "ioctl",
        lambda *_args: (_ for _ in ()).throw(OSError("resize")),
    )
    with pytest.raises(RuntimeError, match="resize failed"):
        session.resize(80, 24)

    expired = OciProcessSession(command, False, 4, 5)
    expired._deadline = 0
    with pytest.raises(RuntimeError, match="no longer active"):
        expired.read()

    reader = OciProcessSession(command, False, 4, 5)

    class _BrokenSource:
        """Raise one deterministic reader error."""

        def fileno(self) -> int:
            """Reject descriptor access."""
            raise OSError("read")

    reader._read_stream("stdout", _BrokenSource())
    assert reader._closed.is_set()
    with pytest.raises(RuntimeError, match="reader failed"):
        reader.wait()

    saturated = OciProcessSession(command, False, 1, 5)
    saturated._frames.put(process_module.OciStreamFrame(0, "stdout", b"full"))
    saturated_source = tmp_path / "saturated-output"
    saturated_source.write_bytes(b"more")
    with saturated_source.open("rb") as source:
        saturated._read_stream("stdout", source)
    with pytest.raises(RuntimeError, match="backpressure"):
        saturated.wait()

    no_pipe = OciProcessSession(command, False, 4, 5)
    assert no_pipe._process.stdin is not None
    no_pipe._process.stdin.close()
    no_pipe._process.stdin = None
    no_pipe.write(b"ignored")
    no_pipe.close_stdin()
    no_pipe.close()
    no_pipe._read_stream("stdout", io.BytesIO(b"ignored"))

    eio = OciProcessSession(command, False, 4, 5)
    eio._process.poll = lambda: 0  # type: ignore[method-assign]
    monkeypatch.setattr(
        process_module.os,
        "read",
        lambda *_args: (_ for _ in ()).throw(OSError(process_module.errno.EIO, "eof")),
    )
    eio._read_stream("pty", 123)
    monkeypatch.undo()
    del eio._process.poll
    eio.close()

    raced = OciProcessSession(command, True, 4, 5)
    descriptor = raced._master_fd
    original_close = process_module.os.close

    def close_race(value: int) -> None:
        """Report an already-closed PTY master only."""
        if value == descriptor:
            raise OSError("closed")
        original_close(value)

    monkeypatch.setattr(process_module.os, "close", close_race)
    monkeypatch.setattr(
        process_module.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    raced.close()


def test_output_pressure_closes_a_real_attached_process(tmp_path: Path) -> None:
    """Unread real output exceeding the queue bound cancels the process group."""
    command = _command(
        tmp_path,
        "import sys\n"
        "for _ in range(1024):\n"
        " sys.stdout.buffer.write(b'x' * 65536)\n"
        " sys.stdout.flush()\n",
    )
    session = OciProcessSession(command, False, 1, 5)
    time.sleep(0.2)
    with pytest.raises(RuntimeError, match="backpressure"):
        session.wait()
