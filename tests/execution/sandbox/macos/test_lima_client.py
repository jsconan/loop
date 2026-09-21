"""Test the macOS Lima client as an isolated fixed-command compiler."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.execution.infrastructure import InfrastructureProcessResult
from loop.execution.sandbox.macos import LimaClient
from loop.execution.sandbox.macos import lima as lima_module


class _Process:
    """Record commands without creating child processes."""

    output_limit = 4096

    def __init__(self, results: list[InfrastructureProcessResult] | None = None) -> None:
        self.commands: list[tuple[object, dict[str, object]]] = []
        self.results = results or [InfrastructureProcessResult(0, b"ok", b"", False, False)]

    def run(self, command: object, **kwargs: object) -> InfrastructureProcessResult:
        """Return the next deterministic process result."""
        self.commands.append((command, kwargs))
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


class _Context:
    """Expose validated private Lima command state."""

    def __init__(self, tmp_path: Path) -> None:
        self.state_root = tmp_path / "state"
        self.lima_home = self.state_root / "lima"
        self.lima_home.mkdir(parents=True)
        (self.state_root / "home").mkdir()
        self.instance_name = "loop-managed-1"
        self.health_nonce = "a" * 64
        self.executable = SimpleNamespace(
            artifact_id="lima",
            path=tmp_path / "limactl",
            path_directories=(str(tmp_path),),
        )
        self.validations = 0

    def validate(self) -> None:
        """Record immediate context revalidation."""
        self.validations += 1


def test_lima_client_compiles_explicit_lifecycle_guest_copy_and_session_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each public method owns exactly one fixed limactl shape and sealed environment."""
    process = _Process()
    client = LimaClient(process, tmp_path / "app")  # type: ignore[arg-type]
    assert client.output_limit == 4096
    context = _Context(tmp_path)
    source = context.state_root / "archive.tar"
    source.write_bytes(b"data")
    client.create(context, b"vmType: vz")  # type: ignore[arg-type]
    client.start(context)  # type: ignore[arg-type]
    client.list(context)  # type: ignore[arg-type]
    client.stop(context)  # type: ignore[arg-type]
    client.stop(context, force=True)  # type: ignore[arg-type]
    client.delete(context)  # type: ignore[arg-type]
    client.health(context)  # type: ignore[arg-type]
    client.copy_to_guest(context, source, "/tmp/archive.tar")  # type: ignore[arg-type]
    destination = context.state_root / "copied.tar"
    client.copy_from_guest(  # type: ignore[arg-type]
        context, "/tmp/archive.tar", destination
    )
    client.write_broker_file(  # type: ignore[arg-type]
        context,
        "/run/user/1000/loop/brokers/lease/config",
        b"secret-data",
    )
    client.remove_broker_paths(  # type: ignore[arg-type]
        context,
        (
            "/home/loop/.config/cni/net.d/90-loop-network.conflist",
            "/run/user/1000/loop/brokers/lease",
        ),
    )
    session = object()
    session_arguments = []

    def fake_session(*args):
        """Record the PTY-aware Lima session command."""
        session_arguments.append(args)
        return session

    monkeypatch.setattr(lima_module, "OciProcessSession", fake_session)
    assert (
        client.open_guest_session(  # type: ignore[arg-type]
            context,
            ("/usr/local/bin/nerdctl", "attach", "container"),
            operation="oci.attach",
            pty=True,
            queue_limit=4,
            deadline_seconds=5,
        )
        is session
    )
    arguments = [call[0].argv[1:] for call in process.commands]
    assert "chmod 700" in arguments[9][6]
    assert "chmod 644" in arguments[9][6]
    assert arguments[0] == ("--tty=false", "create", "--name", context.instance_name, "-")
    assert arguments[3] == ("--tty=false", "stop", context.instance_name)
    assert arguments[4] == ("--tty=false", "stop", "--force", context.instance_name)
    assert arguments[7][1:4] == ("copy", "--backend=scp", str(source))
    assert arguments[8][1:4] == (
        "copy",
        "--backend=scp",
        f"{context.instance_name}:/tmp/archive.tar",
    )
    assert process.commands[9][1]["stdin"] == b"secret-data"
    assert arguments[10][-2:] == (
        "/home/loop/.config/cni/net.d/90-loop-network.conflist",
        "/run/user/1000/loop/brokers/lease",
    )
    assert session_arguments[0][0].argv[1] == "--tty=true"
    assert session_arguments[0][0].environment["TERM"] == "xterm-256color"
    assert all(
        call[0].environment["LIMA_HOME"] == str(context.lima_home) for call in process.commands
    )


def test_lima_client_rejects_unsafe_context_guest_copy_and_signature_shapes(
    tmp_path: Path,
) -> None:
    """Unsafe paths, names, guest argv, health data, and signature outcomes fail closed."""
    bad_home = tmp_path / "bad-home"
    bad_home.write_text("x")
    with pytest.raises(ValueError, match="private directory"):
        LimaClient(_Process(), bad_home)  # type: ignore[arg-type]
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    linked_home = tmp_path / "linked-home"
    linked_home.symlink_to(real_home, target_is_directory=True)
    with pytest.raises(ValueError, match="private directory"):
        LimaClient(_Process(), linked_home)  # type: ignore[arg-type]
    client = LimaClient(_Process(), tmp_path / "app")  # type: ignore[arg-type]
    context = _Context(tmp_path)
    context.health_nonce = "bad"
    with pytest.raises(ValueError, match="nonce"):
        client.health(context)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="guest command"):
        client.guest_command(context, (), operation="guest")  # type: ignore[arg-type]
    command = client.guest_command(context, ("/bin/printf", ""), operation="guest")
    assert command.argv[-2:] == ("/bin/printf", "")
    with pytest.raises(ValueError, match="copy source"):
        client.copy_to_guest(context, tmp_path / "missing", "/tmp/x")  # type: ignore[arg-type]
    source = context.state_root / "source"
    source.write_text("x")
    with pytest.raises(ValueError, match="destination"):
        client.copy_to_guest(context, source, "../escape")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source"):
        client.copy_from_guest(context, "../escape", context.state_root / "copy")  # type: ignore[arg-type]
    outside = tmp_path / "outside"
    with pytest.raises(ValueError, match="outside"):
        client.copy_from_guest(context, "/tmp/source", outside / "copy")  # type: ignore[arg-type]
    existing = context.state_root / "existing"
    existing.write_text("x")
    with pytest.raises(ValueError, match="must not already exist"):
        client.copy_from_guest(context, "/tmp/source", existing)  # type: ignore[arg-type]
    context.instance_name = "--evil"
    with pytest.raises(ValueError, match="context"):
        client.list(context)  # type: ignore[arg-type]
    context.instance_name = "loop-managed-1"
    context.lima_home = tmp_path / "foreign"
    context.lima_home.mkdir()
    with pytest.raises(ValueError, match="outside"):
        client.list(context)  # type: ignore[arg-type]

    context.lima_home = context.state_root / "lima"
    for path in (
        "/run/user/1000/loop/brokers",
        "/run/user/0/loop/brokers/lease/file",
        "/run/user/1000/loop/brokers/../escape",
        "/home/loop/.config/cni/net.d/config.conflist",
    ):
        with pytest.raises(ValueError, match="path"):
            client.write_broker_file(context, path, b"x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="content"):
        client.write_broker_file(  # type: ignore[arg-type]
            context, "/run/user/1000/loop/brokers/lease/file", b""
        )
    with pytest.raises(ValueError, match="cleanup paths"):
        client.remove_broker_paths(  # type: ignore[arg-type]
            context,
            (
                "/run/user/1000/loop/brokers/lease",
                "/run/user/1000/loop/brokers/lease",
            ),
        )

    wrong = SimpleNamespace(artifact_id="tool", path=tmp_path / "tool")
    with pytest.raises(ValueError, match="requires"):
        client.verify_codesign(wrong)  # type: ignore[arg-type]
    failed = InfrastructureProcessResult(1, b"", b"", False, False)
    with pytest.raises(RuntimeError, match="signature verification"):
        LimaClient(_Process([failed]), tmp_path / "verify").verify_codesign(  # type: ignore[arg-type]
            SimpleNamespace(artifact_id="lima", path=tmp_path / "limactl")
        )
    ok = InfrastructureProcessResult(0, b"", b"", False, False)
    with pytest.raises(RuntimeError, match="entitlement inspection"):
        LimaClient(_Process([ok, failed]), tmp_path / "entitlements").verify_codesign(  # type: ignore[arg-type]
            SimpleNamespace(artifact_id="lima", path=tmp_path / "limactl")
        )
    assert (
        LimaClient(_Process([ok, ok]), tmp_path / "codesign").verify_codesign(  # type: ignore[arg-type]
            SimpleNamespace(artifact_id="lima", path=tmp_path / "limactl")
        )
        is ok
    )
