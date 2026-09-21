"""Test the macOS typed guest OCI control plane."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.execution.sandbox.macos import MacosControlPlane, load_macos_runtime_candidate
from loop.execution.sandbox.oci.control import OciSignal
from loop.execution.sandbox.oci.spec import OciExecutionSpec, OciMount, OciResourceLimits


def _tool_arguments() -> tuple[str, ...]:
    """Return one complete validated external-tool build argument set."""
    return (
        "UV_VERSION=0.12.17",
        "UV_ORIGIN=https://releases.astral.sh/github/uv/releases/download/0.12.17",
        "UV_MAX_BYTES=26214400",
        "UV_AARCH64_SHA256=" + "d" * 64,
        "UV_X86_64_SHA256=" + "e" * 64,
    )


class _Instance:
    """Record mandatory running-instance reattestation."""

    def __init__(self) -> None:
        self.calls = 0

    def attest_running(self) -> None:
        """Record one live attestation."""
        self.calls += 1


class _Runner:
    """Record sealed runner invocations at the public boundary."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.session = SimpleNamespace()
        self.pid_result = None
        self.hosts_path_result = None
        self.namespace_result = None
        self.namespace_identity = b"4:4026533000\n"
        self.attestation = SimpleNamespace(
            exit_code=0,
            stdout=(
                b"owner_uid=1000\n"
                b"socket_path=/proc/123/root/run/containerd/containerd.sock\n"
                b"socket_is_socket=true\n"
                b"socket_uid=1000\n"
                b"state_path=/home/loop/.local/share/containerd\n"
                b"state_exists=true\n"
                b"content_path=/home/loop/.local/share/containerd/io.containerd.content.v1.content\n"
                b"content_exists=true\n"
                b"rootful_socket_exists=false\n"
                b"network_namespace_path=/proc/123/root/run/user/1000/"
                b"containerd-rootless/netns\n"
                b"network_namespace_identity=4:4026533000\n"
                b"entered_network_namespace_identity=4:4026533000\n"
                b"nerdctl_version=nerdctl version 2.2.0\n"
                b"containerd_version=containerd github.com/containerd/containerd/v2 v2.2.0 "
                + b"a"
                * 40
                + b"\n"
                b"runc_version=runc version 1.3.3\n"
                b"buildkit_version=buildkitd github.com/moby/buildkit v0.25.2 abcdef0\n"
            ),
            stdout_truncated=False,
            stderr_truncated=False,
        )

    def run_guest(
        self,
        context,
        argv,
        *,
        operation,
        deadline_seconds=120.0,
        cancellation=lambda: False,
    ):
        """Record one bounded invocation."""
        self.calls.append((context, argv, operation, deadline_seconds, cancellation))
        if operation == "oci.attest":
            return self.attestation
        if operation == "oci.container_pid_inspect":
            if self.pid_result is not None:
                return self.pid_result
            return SimpleNamespace(
                exit_code=0,
                stdout=b"456\n",
                stderr=b"",
                stdout_truncated=False,
                stderr_truncated=False,
            )
        if operation == "oci.container_hosts_path_inspect":
            if self.hosts_path_result is not None:
                return self.hosts_path_result
            return SimpleNamespace(
                exit_code=0,
                stdout=(
                    b"/home/loop/.local/share/nerdctl/runtime/etchosts/loop-private/"
                    b"container/hosts\n"
                ),
                stderr=b"",
                stdout_truncated=False,
                stderr_truncated=False,
            )
        if operation == "broker.harden_hosts":
            return SimpleNamespace(
                exit_code=0,
                stdout=b"444\n",
                stderr=b"",
                stdout_truncated=False,
                stderr_truncated=False,
            )
        if operation == "broker.release_hosts":
            return SimpleNamespace(
                exit_code=0,
                stdout=b"600\n",
                stderr=b"",
                stdout_truncated=False,
                stderr_truncated=False,
            )
        if operation == "broker.container_network_namespace":
            if self.namespace_result is not None:
                return self.namespace_result
            return SimpleNamespace(
                exit_code=0,
                stdout=self.namespace_identity,
                stderr=b"",
                stdout_truncated=False,
                stderr_truncated=False,
            )
        return SimpleNamespace(
            exit_code=0,
            stdout=b"ok",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
        )

    def open_guest_session(
        self,
        context,
        argv,
        *,
        operation,
        pty,
        queue_limit,
        deadline_seconds,
        terminal_columns=None,
        terminal_rows=None,
    ):
        """Record one attached invocation."""
        self.calls.append(
            (
                context,
                argv,
                operation,
                pty,
                queue_limit,
                deadline_seconds,
                terminal_columns,
                terminal_rows,
            )
        )
        return self.session

    def write_broker_file(self, context, path, content):
        """Record trusted stdin staging."""
        self.calls.append((context, path, content, "write"))
        return SimpleNamespace(
            exit_code=0,
            stdout=b"ok",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
        )

    def copy_to_guest(self, context, source, destination):
        """Record one private source transfer."""
        self.calls.append((context, source, destination, "copy"))
        return SimpleNamespace(
            exit_code=0,
            stdout=b"",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
        )

    def remove_broker_paths(self, context, paths):
        """Record exact broker-path cleanup."""
        self.calls.append((context, paths, "remove_paths"))
        return SimpleNamespace(
            exit_code=0,
            stdout=b"ok",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
        )


def _candidate():
    """Load the repository-controlled runtime candidate."""
    return load_macos_runtime_candidate(
        Path(__file__).parents[4] / "scripts/runtime-candidates/macos-arm64-v1.json"
    )


def _spec(*, detached: bool = False) -> OciExecutionSpec:
    """Build one closed foreground attempt specification."""
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
    )


def test_control_plane_compiles_every_bounded_operation_and_reattests() -> None:
    """Every non-attached operation has one exact namespace-independent command shape."""
    runner = _Runner()
    instance = _Instance()
    context = SimpleNamespace(artifact_set_digest="a" * 64)
    plane = MacosControlPlane(runner, instance, context)  # type: ignore[arg-type]
    plane.attest_runtime(_candidate())
    results = (
        plane.version(),
        plane.pull_image("registry.test/image@sha256:" + "a" * 64),
        plane.inspect_image("registry.test/image@sha256:" + "a" * 64),
        plane.remove_image("loop.local/sandbox@sha256:" + "a" * 64),
        plane.inspect_container("container-1"),
        plane.inspect_container_state("container-1"),
        plane.remove_container("container-1"),
        plane.wait_container("container-1"),
        plane.stop_container("container-1"),
        plane.kill_container("container-1"),
        plane.signal_container("container-1", OciSignal.HANGUP),
        plane.pause_container("container-1"),
        plane.unpause_container("container-1"),
    )
    assert all(result.stdout == b"ok" for result in results)
    assert instance.calls == len(results) + 1
    assert runner.calls[0][0] is context
    assert all(call[1][0] == "/usr/local/bin/nerdctl" for call in runner.calls[1:])


def test_control_plane_builds_image_from_private_closed_inputs(tmp_path: Path) -> None:
    """Image construction transfers one trusted archive and reattests after cleanup."""
    runner = _Runner()
    instance = _Instance()
    context = SimpleNamespace(artifact_set_digest="a" * 64, state_root=tmp_path)
    plane = MacosControlPlane(runner, instance, context)  # type: ignore[arg-type]
    endpoint = plane.attest_runtime(_candidate())
    archive = tmp_path / "context.tar"
    archive.write_bytes(b"source")

    result = plane.build_sandbox_image(
        archive,
        "loop.local/sandbox:" + "b" * 24,
        "docker.io/library/debian@sha256:" + "c" * 64,
        "20260918T000000Z",
        "bash=1.0 git=2.0",
        _tool_arguments(),
        ("bash", "git"),
    )

    assert result.exit_code == 0
    assert result.stdout == b"ok"
    assert result.stderr == b""
    assert result.stdout_truncated is False
    assert result.stderr_truncated is False
    assert result.platform == endpoint.platform
    operations = [call[2] for call in runner.calls if len(call) == 5]
    assert operations[-3:] == [
        "sandbox_image.prepare_source",
        "sandbox_image.build",
        "oci.version",
    ]
    build = next(
        call for call in runner.calls if len(call) >= 3 and call[2] == "sandbox_image.build"
    )
    assert build[1][0:3] == ("/bin/sh", "-c", build[1][2])
    assert "--secret id=loop-ca,src=/etc/ssl/certs/ca-certificates.crt" in build[1][2]
    assert 'nsenter -t "$pid" -m -U --preserve-credentials' in build[1][2]
    assert '--net="/run/user/$uid/containerd-rootless/netns"' in build[1][2]
    assert 'nsenter -t "$pid" -n -m -U' not in build[1][2]
    assert build[1][10] == "docker.io/library/debian@sha256:" + "c" * 64
    assert build[1][11] == "20260918T000000Z"
    assert build[1][13] == " ".join(_tool_arguments())


def test_control_plane_rejects_untrusted_image_build_inputs(tmp_path: Path) -> None:
    """Sources outside private state and open image arguments never reach the guest."""
    runner = _Runner()
    context = SimpleNamespace(artifact_set_digest="a" * 64, state_root=tmp_path / "private")
    context.state_root.mkdir()
    plane = MacosControlPlane(runner, _Instance(), context)  # type: ignore[arg-type]
    plane.attest_runtime(_candidate())
    outside = tmp_path / "outside.tar"
    outside.write_bytes(b"source")
    with pytest.raises(ValueError, match="outside private Lima state"):
        plane.build_sandbox_image(
            outside,
            "loop.local/sandbox:" + "b" * 24,
            "docker.io/library/debian@sha256:" + "c" * 64,
            "20260918T000000Z",
            "bash=1.0",
            _tool_arguments(),
            ("bash",),
        )

    archive = context.state_root / "context.tar"
    archive.write_bytes(b"source")
    with pytest.raises(ValueError, match="build values are invalid"):
        plane.build_sandbox_image(
            archive,
            "loop.local/custom:latest",
            "debian:latest",
            "today",
            "bash",
            ("UV_VERSION=latest",),
            ("bash;id",),
        )

    directory = context.state_root / "directory.tar"
    directory.mkdir()
    with pytest.raises(ValueError, match="private regular file"):
        plane.build_sandbox_image(
            directory,
            "loop.local/sandbox:" + "b" * 24,
            "docker.io/library/debian@sha256:" + "c" * 64,
            "20260918T000000Z",
            "bash=1.0",
            _tool_arguments(),
            ("bash",),
        )


@pytest.mark.parametrize(
    "tools",
    (
        ("broken",),
        _tool_arguments() + ("UV_VERSION=0.12.17",),
        tuple(
            value.replace("UV_VERSION=0.12.17", "UV_VERSION=latest") for value in _tool_arguments()
        ),
        tuple(
            value.replace(
                "UV_ORIGIN=https://releases.astral.sh/github/uv/releases/download/0.12.17",
                "UV_ORIGIN=https://example.test/uv",
            )
            for value in _tool_arguments()
        ),
        tuple(
            value.replace("UV_MAX_BYTES=26214400", "UV_MAX_BYTES=none")
            for value in _tool_arguments()
        ),
        tuple(
            value.replace("UV_MAX_BYTES=26214400", "UV_MAX_BYTES=0") for value in _tool_arguments()
        ),
        tuple(
            value.replace("UV_MAX_BYTES=26214400", "UV_MAX_BYTES=104857601")
            for value in _tool_arguments()
        ),
        tuple(
            "UV_AARCH64_SHA256=bad" if value.startswith("UV_AARCH64_SHA256=") else value
            for value in _tool_arguments()
        ),
        tuple(
            "UV_X86_64_SHA256=bad" if value.startswith("UV_X86_64_SHA256=") else value
            for value in _tool_arguments()
        ),
    ),
)
def test_control_plane_rejects_untrusted_tool_build_arguments(
    tmp_path: Path, tools: tuple[str, ...]
) -> None:
    """External tool arguments cannot widen the reviewed download boundary."""
    runner = _Runner()
    context = SimpleNamespace(artifact_set_digest="a" * 64, state_root=tmp_path)
    plane = MacosControlPlane(runner, _Instance(), context)  # type: ignore[arg-type]
    plane.attest_runtime(_candidate())
    archive = tmp_path / "context.tar"
    archive.write_bytes(b"source")

    with pytest.raises(ValueError, match="build values are invalid"):
        plane.build_sandbox_image(
            archive,
            "loop.local/sandbox:" + "b" * 24,
            "docker.io/library/debian@sha256:" + "c" * 64,
            "20260918T000000Z",
            "bash=1.0",
            tools,
            ("bash",),
        )


@pytest.mark.parametrize("failure", ["prepare", "copy", "version"])
def test_control_plane_image_build_fails_closed_at_management_boundaries(
    tmp_path: Path, failure: str
) -> None:
    """Source preparation, transfer, and post-build reattestation are mandatory."""

    class _FailingRunner(_Runner):
        def run_guest(self, context, argv, *, operation, **kwargs):
            result = super().run_guest(context, argv, operation=operation, **kwargs)
            if (failure == "prepare" and operation == "sandbox_image.prepare_source") or (
                failure == "version" and operation == "oci.version"
            ):
                result.exit_code = 1
            return result

        def copy_to_guest(self, context, source, destination):
            result = super().copy_to_guest(context, source, destination)
            if failure == "copy":
                result.stderr_truncated = True
            return result

    runner = _FailingRunner()
    context = SimpleNamespace(artifact_set_digest="a" * 64, state_root=tmp_path)
    plane = MacosControlPlane(runner, _Instance(), context)  # type: ignore[arg-type]
    plane.attest_runtime(_candidate())
    archive = tmp_path / "context.tar"
    archive.write_bytes(b"source")

    message = "reattested" if failure == "version" else "could not be"
    with pytest.raises(RuntimeError, match=message):
        plane.build_sandbox_image(
            archive,
            "loop.local/sandbox:" + "b" * 24,
            "docker.io/library/debian@sha256:" + "c" * 64,
            "20260918T000000Z",
            "bash=1.0",
            _tool_arguments(),
            ("bash",),
        )
    if failure == "copy":
        assert any(call[2] == "sandbox_image.cleanup_source" for call in runner.calls)


def test_control_plane_preserves_attached_settings_and_rejects_wrong_shapes() -> None:
    """Attach methods preserve session settings and require prior runtime attestation."""
    runner = _Runner()
    instance = _Instance()
    plane = MacosControlPlane(runner, instance, SimpleNamespace(artifact_set_digest="a" * 64))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="attested guest runtime"):
        plane.version()
    with pytest.raises(ValueError, match="attested guest runtime"):
        plane.attach("container-1")
    plane.attest_runtime(_candidate())
    with pytest.raises(ValueError, match="attempt source"):
        plane.run_attempt(_spec(), "../bad")
    assert (
        plane.run_attempt(
            _spec(),
            "/home/loop/.local/share/loop/workspaces/attempts/attempt/merged",
        )
        is runner.session
    )
    assert "/home/loop/.local/share/loop/workspaces/attempts/attempt/merged" in " ".join(
        runner.calls[-1][1]
    )
    session = plane.attach(
        "container-1",
        pty=True,
        queue_limit=7,
        deadline_seconds=9,
    )
    assert session is runner.session
    assert runner.calls[-1][1][-2:] == ("attach", "container-1")
    assert runner.calls[-1][2:] == (
        "oci.container_attach",
        True,
        7,
        9,
        80,
        24,
    )
    assert plane.start_attached("container-1") is runner.session
    assert runner.calls[-1][1][-3:] == ("start", "--attach", "container-1")
    assert (
        plane.run_job(
            _spec(detached=True),
            "/home/loop/.local/share/loop/workspaces/attempts/attempt/merged",
        )
        is runner.session
    )
    assert runner.calls[-1][2] == "oci.job_run"
    with pytest.raises(ValueError, match="durable workspace source"):
        plane.run_job(_spec(detached=True), "/tmp/unmanaged")
    assert instance.calls == 5


def test_control_plane_attests_rootless_runtime_and_rejects_bad_evidence() -> None:
    """Only exact non-root private runtime evidence can mint a guest endpoint."""
    runner = _Runner()
    instance = _Instance()
    plane = MacosControlPlane(runner, instance, SimpleNamespace(artifact_set_digest="a" * 64))  # type: ignore[arg-type]
    endpoint = plane.attest_runtime(_candidate())
    assert endpoint.owner_uid == 1000
    assert endpoint.namespace == "loop-private"
    assert endpoint.platform.architecture == "arm64"
    assert endpoint.content_path.startswith(endpoint.state_path + "/")
    assert endpoint.network_namespace_identity == "4:4026533000"
    assert instance.calls == 1

    for result in (
        SimpleNamespace(exit_code=1, stdout=b"", stdout_truncated=False, stderr_truncated=False),
        SimpleNamespace(exit_code=0, stdout=b"bad", stdout_truncated=False, stderr_truncated=False),
        SimpleNamespace(
            exit_code=0, stdout=b"\xff", stdout_truncated=False, stderr_truncated=False
        ),
        SimpleNamespace(
            exit_code=0,
            stdout=runner.attestation.stdout.replace(b"owner_uid=1000", b"owner_uid=nope"),
            stdout_truncated=False,
            stderr_truncated=False,
        ),
        SimpleNamespace(
            exit_code=0,
            stdout=runner.attestation.stdout.replace(b"socket_uid=1000", b"socket_uid=1001"),
            stdout_truncated=False,
            stderr_truncated=False,
        ),
        SimpleNamespace(
            exit_code=0,
            stdout=runner.attestation.stdout.replace(
                b"entered_network_namespace_identity=4:4026533000",
                b"entered_network_namespace_identity=4:4026533001",
            ),
            stdout_truncated=False,
            stderr_truncated=False,
        ),
        SimpleNamespace(
            exit_code=0,
            stdout=runner.attestation.stdout.replace(b"owner_uid=1000", b"unexpected=1000"),
            stdout_truncated=False,
            stderr_truncated=False,
        ),
        SimpleNamespace(
            exit_code=0,
            stdout=runner.attestation.stdout.replace(b"2.2.0", b"9.9.9", 1),
            stdout_truncated=False,
            stderr_truncated=False,
        ),
        SimpleNamespace(
            exit_code=0,
            stdout=runner.attestation.stdout.replace(
                b"containerd github.com/containerd/containerd/v2 v2.2.0 " + b"a" * 40,
                b"containerd github.com/containerd/containerd/v2 9.9.9",
            ),
            stdout_truncated=False,
            stderr_truncated=False,
        ),
        SimpleNamespace(
            exit_code=0,
            stdout=runner.attestation.stdout.replace(b"runc version 1.3.3", b"runc version 9.9.9"),
            stdout_truncated=False,
            stderr_truncated=False,
        ),
    ):
        runner.attestation = result
        with pytest.raises(ValueError):
            plane.attest_runtime(_candidate())


def test_control_plane_routes_broker_state_network_and_rootlesskit_operations() -> None:
    """Broker operations use only attested OCI, Lima stdin, and RootlessKit API shapes."""
    runner = _Runner()
    instance = _Instance()
    context = SimpleNamespace(artifact_set_digest="a" * 64)
    plane = MacosControlPlane(runner, instance, context)  # type: ignore[arg-type]
    plane.attest_runtime(_candidate())
    image = "registry.test/image@sha256:" + "a" * 64
    assert plane.start_broker("loop-broker", image, "/run/broker.json").stdout == b"ok"
    assert (
        plane.create_attempt(
            _spec(),
            "/home/loop/.local/share/loop/workspaces/attempts/request/merged",
        ).stdout
        == b"ok"
    )
    assert plane.start_attempt(_spec()) is runner.session
    assert plane.container_uses_rootless_network_namespace("loop-broker") is True
    assert plane.harden_container_hosts("loop-command").stdout == b"444\n"
    assert (
        plane.open_container_start_gate("/run/user/1000/loop/brokers/lease/start-gate").stdout
        == b"ok"
    )
    assert plane.release_container_hosts("loop-command").stdout == b"600\n"
    runner.namespace_identity = b"4:4026533001\n"
    assert plane.container_uses_rootless_network_namespace("loop-broker") is False
    assert plane.write_broker_file("/run/user/1000/loop/brokers/lease/file", b"x").stdout == b"ok"
    assert plane.remove_broker_paths(("/run/user/1000/loop/brokers/lease",)).stdout == b"ok"
    assert plane.harden_network_namespace().stdout == b"ok"
    assert (
        plane.initialize_network("loop-lease", "lb0123456789", "/run/netns", "/run/cni.json").stdout
        == b"ok"
    )
    assert (
        plane.remove_network_attachment(
            "loop-lease", "lb0123456789", "/run/netns", "/run/cni.json"
        ).stdout
        == b"ok"
    )
    assert plane.remove_broker_bridge("br0123456789").stdout == b"ok"
    assert plane.publish_broker_port("127.0.0.1", 18080).stdout == b"ok"
    assert plane.remove_broker_port(7).stdout == b"ok"
    assert plane.list_broker_ports().stdout == b"ok"
    commands = [call[1] for call in runner.calls if len(call) > 2 and isinstance(call[1], tuple)]
    assert any("add-ports" in command for command in commands)
    assert any("remove-ports" in command for command in commands)
    assert any("list-ports" in command for command in commands)
    detached_network_operations = {
        call[2]: call[1][2]
        for call in runner.calls
        if call[2]
        in {
            "broker.harden_network",
            "broker.cni_add",
            "broker.cni_del",
            "broker.remove_bridge",
        }
    }
    assert set(detached_network_operations) == {
        "broker.harden_network",
        "broker.cni_add",
        "broker.cni_del",
        "broker.remove_bridge",
    }
    assert all(
        '--net="/run/user/$1/containerd-rootless/netns"' in script
        and 'nsenter -t "$pid" -n' not in script
        for script in detached_network_operations.values()
    )
    with pytest.raises(ValueError, match="publication endpoint"):
        plane.publish_broker_port("0.0.0.0", 80)
    with pytest.raises(ValueError, match="publication identity"):
        plane.remove_broker_port(0)
    with pytest.raises(ValueError, match="bridge identity"):
        plane.remove_broker_bridge("eth0")


@pytest.mark.parametrize(
    "pid_result",
    (
        SimpleNamespace(
            exit_code=1,
            stdout=b"",
            stderr=b"bad",
            stdout_truncated=False,
            stderr_truncated=False,
        ),
        SimpleNamespace(
            exit_code=0,
            stdout=b"0\n",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
        ),
        SimpleNamespace(
            exit_code=0,
            stdout=b"bad\n",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
        ),
    ),
)
def test_control_plane_rejects_malformed_container_pid_evidence(pid_result) -> None:
    """Missing, nonpositive, and nonnumeric task PIDs fail namespace attestation."""
    runner = _Runner()
    runner.pid_result = pid_result
    plane = MacosControlPlane(
        runner,
        _Instance(),
        SimpleNamespace(artifact_set_digest="a" * 64),
    )  # type: ignore[arg-type]
    plane.attest_runtime(_candidate())
    with pytest.raises(ValueError, match="PID evidence"):
        plane.container_uses_rootless_network_namespace("loop-broker")


@pytest.mark.parametrize(
    "namespace_result",
    (
        SimpleNamespace(
            exit_code=1,
            stdout=b"",
            stderr=b"bad",
            stdout_truncated=False,
            stderr_truncated=False,
        ),
        SimpleNamespace(
            exit_code=0,
            stdout=b"\xff",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
        ),
        SimpleNamespace(
            exit_code=0,
            stdout=b"bad\n",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
        ),
    ),
)
def test_control_plane_rejects_malformed_namespace_identity(namespace_result) -> None:
    """Unavailable, non-ASCII, and malformed namespace identities fail closed."""
    runner = _Runner()
    runner.namespace_result = namespace_result
    plane = MacosControlPlane(
        runner,
        _Instance(),
        SimpleNamespace(artifact_set_digest="a" * 64),
    )  # type: ignore[arg-type]
    plane.attest_runtime(_candidate())
    with pytest.raises(ValueError, match="namespace evidence"):
        plane.container_uses_rootless_network_namespace("loop-broker")


@pytest.mark.parametrize(
    "hosts_path_result",
    (
        SimpleNamespace(
            exit_code=1,
            stdout=b"",
            stderr=b"bad",
            stdout_truncated=False,
            stderr_truncated=False,
        ),
        SimpleNamespace(
            exit_code=0,
            stdout=b"\xff",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
        ),
        SimpleNamespace(
            exit_code=0,
            stdout=b"/etc/hosts\n",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
        ),
    ),
)
def test_control_plane_rejects_untrusted_generated_hosts_paths(hosts_path_result) -> None:
    """Unavailable, malformed, or non-private generated hosts state cannot be hardened."""
    runner = _Runner()
    runner.hosts_path_result = hosts_path_result
    plane = MacosControlPlane(
        runner,
        _Instance(),
        SimpleNamespace(artifact_set_digest="a" * 64),
    )  # type: ignore[arg-type]
    plane.attest_runtime(_candidate())
    with pytest.raises(ValueError, match="hosts-path evidence"):
        plane.harden_container_hosts("loop-command")


def test_control_plane_rejects_invalid_created_attempt_source() -> None:
    """Two-phase network launch accepts only managed snapshot or overlay sources."""
    plane = MacosControlPlane(
        _Runner(),
        _Instance(),
        SimpleNamespace(artifact_set_digest="a" * 64),
    )  # type: ignore[arg-type]
    plane.attest_runtime(_candidate())
    with pytest.raises(ValueError, match="attempt source"):
        plane.create_attempt(_spec(), "/tmp/unmanaged")
