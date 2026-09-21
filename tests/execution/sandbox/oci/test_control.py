"""Test common typed OCI control records."""

from __future__ import annotations

import pytest

from loop.execution.contracts import TerminalMode
from loop.execution.infrastructure import InfrastructureProcessResult
from loop.execution.runtime.models import PlatformSelector
from loop.execution.sandbox.oci import GuestOciRuntimeEndpoint, OciSupervisor
from loop.execution.sandbox.oci.control import OciAttemptBindings, OciSignal
from loop.execution.sandbox.oci.spec import OciExecutionSpec, OciMount, OciResourceLimits


def _endpoint(**changes: object) -> GuestOciRuntimeEndpoint:
    """Build one valid guest endpoint with selected field replacements."""
    values = {
        "socket_path": "/proc/123/root/run/containerd/containerd.sock",
        "state_path": "/home/loop/.local/share/containerd",
        "content_path": ("/home/loop/.local/share/containerd/io.containerd.content.v1.content"),
        "namespace": "loop-private",
        "content_store_identity": "sha256:" + "b" * 64,
        "platform": PlatformSelector(os="linux", architecture="arm64"),
        "artifact_set_digest": "a" * 64,
        "owner_uid": 1000,
        "network_namespace_path": ("/proc/123/root/run/user/1000/containerd-rootless/netns"),
        "network_namespace_identity": "4:4026533000",
        "nerdctl_version": "nerdctl version 2.2.0",
        "containerd_version": "containerd github.com/containerd/containerd/v2 v2.2.0 " + "a" * 40,
        "runc_version": "runc version 1.3.3",
        "buildkit_version": "buildkitd github.com/moby/buildkit v0.25.2 abcdef0",
    }
    values.update(changes)
    return GuestOciRuntimeEndpoint(**values)  # type: ignore[arg-type]


def _spec(
    *,
    detached: bool = False,
    attach_stdin: bool = False,
    terminal: TerminalMode = TerminalMode.PIPE,
) -> OciExecutionSpec:
    """Build one closed foreground OCI specification."""
    return OciExecutionSpec(
        request_id="request",
        container_name="container-1",
        image_reference="registry.test/image@sha256:" + "a" * 64,
        image_digest="sha256:" + "a" * 64,
        image_manifest_digest="sha256:" + "b" * 64,
        image_config_digest="sha256:" + "c" * 64,
        argv=("/bin/sh", "-c", "printf safe"),
        cwd="/workspace",
        environment=(("HOME", "/home/agent"),),
        labels=(("io.loop.request", "request"),),
        workspace_mount=OciMount("generation", "/workspace", True),
        limits=OciResourceLimits(1024, 32, 10000),
        network_mode="none",
        detached=detached,
        attach_stdin=attach_stdin,
        terminal=terminal,
        terminal_columns=100 if terminal is TerminalMode.PTY else None,
        terminal_rows=40 if terminal is TerminalMode.PTY else None,
    )


def test_guest_endpoint_accepts_only_bound_private_non_root_evidence() -> None:
    """Relative, escaping, root-owned, malformed, and unversioned endpoints fail closed."""
    assert _endpoint().owner_uid == 1000
    invalid = (
        {"owner_uid": 0},
        {"socket_path": "relative"},
        {"state_path": "/home/loop/../root"},
        {"state_path": "/var/lib/containerd"},
        {"content_path": "/other/content"},
        {"namespace": ""},
        {"content_store_identity": "bad"},
        {"content_store_identity": "sha256:" + "g" * 64},
        {"artifact_set_digest": "bad"},
        {"artifact_set_digest": "g" * 64},
        {"network_namespace_path": "/run/user/1000/containerd-rootless/netns"},
        {"network_namespace_path": ("/proc/999/root/run/user/1000/containerd-rootless/netns")},
        {"network_namespace_path": ("/proc/123/root/run/user/1001/containerd-rootless/netns")},
        {"network_namespace_identity": "bad"},
        {"nerdctl_version": ""},
    )
    for changes in invalid:
        with pytest.raises(ValueError, match="invalid"):
            _endpoint(**changes)


class _Transport:
    """Record exact OCI-owned commands without creating a process."""

    def __init__(self) -> None:
        self.commands: list[tuple[tuple[str, ...], str]] = []
        self.sessions: list[
            tuple[tuple[str, ...], str, bool, int, float, int | None, int | None]
        ] = []

    def run_oci_command(
        self, argv: tuple[str, ...], operation: str, cancellation: object
    ) -> InfrastructureProcessResult:
        """Record one bounded command and return a successful result."""
        del cancellation
        self.commands.append((argv, operation))
        return InfrastructureProcessResult(0, b"ok", b"", False, False)

    def open_oci_session(
        self,
        argv: tuple[str, ...],
        operation: str,
        use_pty: bool,
        queue_limit: int,
        deadline_seconds: float,
        terminal_columns: int | None = None,
        terminal_rows: int | None = None,
    ) -> object:
        """Record one attached command and return an opaque session."""
        self.sessions.append(
            (
                argv,
                operation,
                use_pty,
                queue_limit,
                deadline_seconds,
                terminal_columns,
                terminal_rows,
            )
        )
        return self


def test_supervisor_exposes_only_explicit_fixed_shape_operations() -> None:
    """Every supported operation compiles an exact private-runtime command."""
    transport = _Transport()
    supervisor = OciSupervisor(_endpoint(), transport)  # type: ignore[arg-type]
    image = "registry.test/image@sha256:" + "a" * 64
    assert supervisor.version().stdout == b"ok"
    assert supervisor.pull_image(image).stdout == b"ok"
    assert supervisor.inspect_image(image).stdout == b"ok"
    assert supervisor.remove_image("loop.local/sandbox@sha256:" + "b" * 64).stdout == b"ok"
    for operation in (
        supervisor.inspect_container,
        supervisor.inspect_container_state,
        supervisor.remove_container,
        supervisor.wait_container,
        supervisor.stop_container,
        supervisor.kill_container,
        supervisor.pause_container,
        supervisor.unpause_container,
    ):
        assert operation("container-1").stdout == b"ok"
    assert supervisor.signal_container("container-1", OciSignal.INTERRUPT).stdout == b"ok"
    assert supervisor.start_attached("container-1") is transport
    assert (
        supervisor.run_job(
            _spec(detached=True),
            "/run/loop/snapshots/generation",
            queue_limit=7,
            deadline_seconds=9,
        )
        is transport
    )
    assert (
        supervisor.attach("container-1", pty=True, queue_limit=7, deadline_seconds=9) is transport
    )
    assert supervisor.run_attempt(_spec(), "/run/loop/snapshots/generation") is transport
    assert (
        supervisor.run_attempt(_spec(terminal=TerminalMode.PTY), "/run/loop/snapshots/generation")
        is transport
    )
    assert len(transport.commands) == 13
    assert transport.commands[0][0][-3:] == ("version", "--format", "{{json .}}")
    assert transport.commands[1][0][-3:-1] == ("--platform", "linux/arm64")
    assert transport.commands[4][0][-3:] == (
        "--format",
        "{{json .}}",
        "container-1",
    )
    assert any(
        command[0][-4:] == ("kill", "--signal", "INT", "container-1")
        for command in transport.commands
    )
    pipe_create = transport.sessions[-2][0]
    pty_create = transport.sessions[-1][0]
    assert "--pull=never" in pty_create
    assert pty_create[pty_create.index("--userns") + 1] == "host"
    assert pty_create[pty_create.index("--user") + 1] == "1000:1000"
    assert "--security-opt" in pty_create
    assert pty_create[pty_create.index("--ulimit") + 1] == "nofile=1024:1024"
    assert "--storage-opt" not in pty_create
    assert "/var/tmp:rw,nosuid,nodev,exec,mode=1777,size=536870912" in pty_create
    assert "/tmp:rw,nosuid,nodev,exec,mode=1777,size=67108864" in pty_create
    assert (
        "/home/agent:rw,nosuid,nodev,noexec,mode=700,uid=1000,gid=1000,size=16777216" in pty_create
    )
    assert any(
        value.startswith("/cache:rw,nosuid,nodev,noexec,") and value.endswith("size=268435456")
        for value in pty_create
    )
    assert "-i" in pty_create
    assert "-t" in pty_create
    assert "-i" not in pipe_create
    assert "-t" not in pipe_create
    assert transport.sessions[-1][2] is True
    assert "/run/loop/snapshots/generation" in " ".join(pty_create)
    assert transport.sessions[-1][1:] == ("attempt_run", True, 128, 30.0, 100, 40)
    assert transport.sessions[-3][0][-2:] == ("attach", "container-1")
    assert transport.sessions[-3][1:] == ("container_attach", True, 7, 9, 80, 24)


def test_supervisor_rejects_untyped_values_and_invalid_session_bounds() -> None:
    """Untrusted options and invalid attach bounds fail before reaching transport."""
    transport = _Transport()
    supervisor = OciSupervisor(_endpoint(), transport)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="private immutable reference"):
        supervisor.remove_image("registry.test/image@sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="typed value"):
        supervisor.inspect_container("--help")
    with pytest.raises(ValueError, match="mount source"):
        supervisor.run_attempt(_spec(), "/run/loop/snapshots/../bad")
    with pytest.raises(ValueError, match="lifetime"):
        supervisor.run_attempt(_spec(detached=True), "/run/loop/snapshots/generation")
    with pytest.raises(ValueError, match="bounds"):
        supervisor.run_attempt(_spec(), "/run/loop/snapshots/generation", queue_limit=0)
    with pytest.raises(ValueError, match="bounds"):
        supervisor.run_job(_spec(detached=True), "/run/loop/snapshots/generation", queue_limit=0)
    with pytest.raises(ValueError, match="lifetime"):
        supervisor.run_job(_spec(), "/run/loop/snapshots/generation")
    with pytest.raises(ValueError, match="bounds"):
        supervisor.attach("container", queue_limit=0)
    with pytest.raises(ValueError, match="bounds"):
        supervisor.start_attached("container", deadline_seconds=0)
    with pytest.raises(ValueError, match="bounds"):
        supervisor.create_attempt(_spec(), "/run/loop/snapshots/generation", queue_limit=0)
    with pytest.raises(ValueError, match="bounds"):
        supervisor.start_attempt(_spec(), deadline_seconds=0)
    assert transport.commands == []
    assert transport.sessions == []


def test_supervisor_compiles_broker_bindings_and_fixed_management_operations() -> None:
    """Broker resources become only reviewed nerdctl flags and fixed lifecycle commands."""
    transport = _Transport()
    supervisor = OciSupervisor(_endpoint(), transport)  # type: ignore[arg-type]
    image = "registry.test/image@sha256:" + "a" * 64
    bindings = OciAttemptBindings(
        "loop-network",
        "10.240.1.10",
        (("api.example", "10.240.1.1"),),
        ("10.240.1.1",),
        "/run/user/1000/loop/brokers/lease/environment",
        (("/run/user/1000/loop/brokers/lease/secret-0", "/run/secrets/token"),),
        "/run/user/1000/loop/brokers/lease/start-gate",
    )
    assert (
        supervisor.run_attempt(
            _spec(),
            "/run/loop/snapshots/generation",
            bindings=bindings,
        )
        is transport
    )
    assert (
        supervisor.create_attempt(
            _spec(),
            "/run/loop/snapshots/generation",
            bindings=bindings,
        ).stdout
        == b"ok"
    )
    assert supervisor.start_attempt(_spec(attach_stdin=True)) is transport
    argv = transport.sessions[-2][0]
    start_argv = transport.sessions[-1][0]
    assert "--interactive" in start_argv
    assert ("--network", "loop-network") == argv[argv.index("--network") :][:2]
    assert "api.example:10.240.1.1" in argv
    assert "10.240.1.1" in argv
    assert "/run/user/1000/loop/brokers/lease/environment" in argv
    assert "/run/secrets/token,readonly" in " ".join(argv)
    assert "dst=/run/loop/start-gate,readonly" in " ".join(argv)
    assert "loop-start-gate" in argv

    assert supervisor.start_broker("loop-broker", image, "/run/broker.json").stdout == b"ok"
    assert ("--user", "0:0") == transport.commands[-1][0][
        transport.commands[-1][0].index("--user") :
    ][:2]
    broker_argv = transport.commands[-1][0]
    assert broker_argv.count("--network") == 1
    assert broker_argv[broker_argv.index("--ulimit") + 1] == "nofile=1024:1024"
    assert "ns:/proc/123/root/run/user/1000/containerd-rootless/netns" in broker_argv
    assert "host" not in broker_argv
    assert transport.commands[-1][1:] == ("broker_start",)
    assert supervisor.inspect_container_pid("loop-broker").stdout == b"ok"
    assert transport.commands[-1][0][-5:] == (
        "container",
        "inspect",
        "--format",
        "{{.State.Pid}}",
        "loop-broker",
    )
    assert supervisor.inspect_container_hosts_path("loop-command").stdout == b"ok"
    assert transport.commands[-1][0][-5:] == (
        "container",
        "inspect",
        "--format",
        "{{.HostsPath}}",
        "loop-command",
    )
    assert (
        supervisor.initialize_network(
            "loop-initializer",
            "loop-network",
            "10.240.1.11",
            image,
        ).stdout
        == b"ok"
    )
    secret_only = OciAttemptBindings(None, None, environment_file="/run/secret-env")
    supervisor.run_attempt(
        _spec(),
        "/run/loop/snapshots/generation",
        bindings=secret_only,
    )
    assert "none" in transport.sessions[-1][0]
    network_only = OciAttemptBindings("loop-network", "10.240.1.10")
    supervisor.run_attempt(
        _spec(),
        "/run/loop/snapshots/generation",
        bindings=network_only,
    )
    assert "--env-file" not in transport.sessions[-1][0]
