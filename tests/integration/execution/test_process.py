"""Exercise real pipe, timeout, cancellation, and process-group behavior separately."""

from __future__ import annotations

import hashlib
import stat
import sys
from pathlib import Path

import pytest

from loop.execution.infrastructure import (
    InfrastructureProcessCommand,
    InfrastructureProcessRunner,
    sealed_environment,
)
from loop.execution.runtime.bootstrap import InstalledRuntime
from loop.execution.runtime.lease import create_lease
from loop.execution.runtime.models import (
    AcquisitionKind,
    Artifact,
    ArtifactRole,
    InstallLayout,
    PlatformSelector,
)


def _executable(tmp_path: Path, body: str):
    """Create one leased private executable for process integration checks."""
    root = tmp_path / "runtime"
    path = root / "artifacts" / "tool" / ("a" * 64) / "tool"
    path.parent.mkdir(parents=True)
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = Artifact(
        artifact_id="tool",
        version="1",
        role=ArtifactRole.NERDCTL,
        platform=PlatformSelector(os="linux", architecture="amd64"),
        source="https://example.test/tool",
        size=path.stat().st_size,
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


def _command(tmp_path: Path, body: str) -> InfrastructureProcessCommand:
    """Build one exact integration command."""
    executable = _executable(tmp_path, body)
    return InfrastructureProcessCommand(
        (str(executable.path),),
        sealed_environment(executable),
        tmp_path,
        "integration.process",
        executable,
    )


def test_process_primitive_drains_bounded_pipes_and_forwards_stdin(tmp_path: Path) -> None:
    """Real child pipes cannot deadlock and retain only the configured output prefix."""
    command = _command(
        tmp_path,
        "import sys\ndata = sys.stdin.buffer.read()\nsys.stdout.buffer.write(data * 1000)\n"
        "sys.stderr.buffer.write(b'error')\n",
    )
    result = InfrastructureProcessRunner(32).run(command, stdin=b"x")
    assert result.exit_code == 0
    assert result.stdout == b"x" * 32
    assert result.stdout_truncated is True
    assert result.stderr == b"error"


@pytest.mark.parametrize(("cancel", "message"), [(False, "deadline"), (True, "cancelled")])
def test_process_primitive_kills_the_group_on_timeout_or_cancellation(
    tmp_path: Path, cancel: bool, message: str
) -> None:
    """Timeout and cancellation both terminate and reap the entire owned process group."""
    command = _command(tmp_path, "import time\ntime.sleep(10)\n")
    with pytest.raises(RuntimeError, match=message):
        InfrastructureProcessRunner().run(
            command,
            deadline_seconds=0.02,
            cancellation=(lambda: cancel),
        )
