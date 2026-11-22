"""Exercise approved host-loopback publication through actual native command dispatch."""

import json
import os
import secrets
import shlex
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop import Action, Decision, PermissionManager, ToolRegistry, constants
from loop.execution import CommandExecutionService
from loop.instructions import InstructionsManager, RuntimeEnvironment
from loop.tools.system import run_command

pytestmark = pytest.mark.integration
_NATIVE_PLATFORM = "MACOS" if sys.platform == "darwin" else "LINUX"
_NATIVE_MODE = "owned-volume" if sys.platform == "darwin" else "installed"
_NATIVE_ENABLED = os.environ.get(f"LOOP_NATIVE_{_NATIVE_PLATFORM}_TEST") == _NATIVE_MODE
_NATIVE_INTERFACE = os.environ.get("LOOP_NATIVE_INGRESS_INTERFACE")


@pytest.fixture
def publication(native_directory, native_mount_point, monkeypatch):
    """Own one real command service, policy authority and exact publication port."""
    if not _NATIVE_ENABLED:
        pytest.skip(f"Requires LOOP_NATIVE_{_NATIVE_PLATFORM}_TEST={_NATIVE_MODE}")
    workspace = native_directory("/private/tmp" if sys.platform == "darwin" else "/tmp")
    token_hex = secrets.token_hex

    def owned_token(size):
        """Register the exact future mount without changing its native random identity."""
        value = token_hex(size)
        if size == 4:
            native_mount_point(
                Path("/Volumes") / f"{constants.APP_DISPLAY_NAME}IPC-{os.getpid()}-{value}"
            )
        return value

    if sys.platform == "darwin":
        monkeypatch.setattr(
            "loop.execution.sandbox.macos.backend.secrets",
            SimpleNamespace(token_hex=owned_token),
        )
    service = CommandExecutionService.for_host()
    manager = PermissionManager(workspace)
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry([run_command], permission_manager=manager, execution_service=service)
    scratch = manager.temporary_directory
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, scratch))
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]

    def call(source, **arguments):
        """Dispatch the approved opaque service source through the public tool."""
        return json.loads(
            registry.call(
                "run_command",
                json.dumps(
                    {"command": source, "ingress_port": port, "wait_seconds": 25, **arguments}
                ),
                interaction=interaction,
                instructions_manager=instructions,
            )
        )

    try:
        yield call, manager, interaction, scratch, port, registry
    finally:
        (scratch / "stop").touch()
        registry.close()
        manager.close()


def service_source():
    """Serve synthetic HTTP from the foreground identity using only attempt-local transport."""
    if sys.platform == "darwin":
        source = r"""
use Socket; use IO::Select; alarm 25;
socket(my $server, PF_UNIX, SOCK_STREAM, 0) or die $!;
my $path = $ENV{LOOP_INGRESS_SOCKET};
bind($server, pack_sockaddr_un($path)) or die $!; listen($server, 8) or die $!;
open(my $ready, '>', "$ENV{TMPDIR}/ready.tmp") or die $!; print $ready $path; close $ready;
rename("$ENV{TMPDIR}/ready.tmp", "$ENV{TMPDIR}/ready") or die $!;
my $readers = IO::Select->new($server);
while (!-e "$ENV{TMPDIR}/stop") {
    if (-e "$ENV{TMPDIR}/swap" && !-e "$ENV{TMPDIR}/swapped") {
        open(my $target, '<', "$ENV{TMPDIR}/swap") or die $!; my $host = <$target>; close $target;
        unlink($path) or die $!; symlink($host, $path) or die $!;
        open(my $done, '>', "$ENV{TMPDIR}/swapped") or die $!; close $done;
    }
    next unless $readers->can_read(0.05);
    defined accept(my $client, $server) or die $!;
    my $data = ''; sysread($client, $data, 4096);
    syswrite($client, "HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\ningress-ok");
    close $client;
}
print "stopped\n";
"""
        return "exec /usr/bin/perl -e " + shlex.quote(source)
    source = """
import os, socket, select, pathlib, signal, array
signal.alarm(25)
# The trusted worker may receive the host FD; both agent generations must still be denied.
left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
try:
    with open('/dev/null', 'rb') as descriptor:
        left.sendmsg([b'x'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                              array.array('i', [descriptor.fileno()]))])
        try:
            right.recvmsg(1, socket.CMSG_SPACE(4))
        except PermissionError:
            pass
        else:
            raise AssertionError('agent received a descriptor')
        child = os.fork()
        if child == 0:
            try:
                right.recvmsg(1, socket.CMSG_SPACE(4))
            except PermissionError:
                os._exit(0)
            os._exit(3)
        assert os.waitpid(child, 0)[1] == 0, 'child lost descriptor denial'
finally:
    left.close(); right.close()
scratch = pathlib.Path(os.environ['TMPDIR'])
with socket.socket() as server:
    server.bind(('127.0.0.1', int(os.environ['LOOP_INGRESS_PORT'])))
    server.listen(8)
    (scratch/'ready').write_text('ready')
    while not (scratch/'stop').exists():
        if not select.select([server],[],[],0.05)[0]: continue
        client,_=server.accept()
        with client:
            client.recv(4096)
            client.sendall(b'HTTP/1.1 200 OK\\r\\nContent-Length: 10\\r\\nConnection: close\\r\\n\\r\\ningress-ok')
print('stopped')
"""
    return "exec /usr/bin/python3 -c " + shlex.quote(source)


def exchange(port):
    """Read one actual loopback HTTP response without using proxy configuration."""
    with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
        client.settimeout(2)
        client.sendall(b"GET / HTTP/1.0\r\n\r\n")
        result = b""
        while chunk := client.recv(4096):
            result += chunk
        return result


def closed(port, address="127.0.0.1"):
    """Observe refusal on an owned publication address."""
    with socket.socket() as client:
        client.settimeout(0.5)
        return client.connect_ex((address, port)) != 0


def wait_for_service_marker(marker, future):
    """Wait for a native service milestone or report its earlier command result."""
    deadline = time.monotonic() + 25
    while not marker.exists():
        if future.done():
            pytest.fail(f"Service exited before {marker.name}: {future.result()}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pytest.fail(f"Service did not publish {marker.name} within 25 seconds")
        wait((future,), timeout=min(0.05, remaining))


@pytest.mark.parametrize("ending", ["complete", "revoke", "cancel"])
def test_publication_is_exact_approved_and_closed_with_its_authority(publication, ending):
    """Approval precedes bind; a live exact route ends on completion or policy revision."""
    call, manager, interaction, scratch, port, registry = publication

    def approve(*_args, **_kwargs):
        """Reject any publication that precedes the human decision."""
        assert closed(port)
        return "approve"

    interaction.prompt.side_effect = approve
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(call, service_source())
        try:
            wait_for_service_marker(scratch / "ready", future)
            assert b"ingress-ok" in exchange(port)
            interaction.prompt.assert_called_once()
            assert manager.configuration.defaults[Action.PROCESS_EXECUTE] is Decision.ASK
            assert not manager.direct_host_execution_enabled
            if _NATIVE_INTERFACE is not None:
                assert closed(port, _NATIVE_INTERFACE)
            if ending in {"revoke", "cancel"}:
                # Keep one accepted connection idle so shutdown must also close an active relay.
                with socket.create_connection(("127.0.0.1", port), timeout=2) as active:
                    active.settimeout(3)
                    if ending == "revoke":
                        manager.set_default(Action.NETWORK_INGRESS, Decision.DENY)
                    else:
                        registry.close()
                    deadline = time.monotonic() + 3
                    while not closed(port):
                        assert time.monotonic() < deadline, "Revoked publication remained open"
                        time.sleep(0.02)
                    try:
                        assert active.recv(1) == b""
                    except ConnectionResetError:
                        pass
            else:
                (scratch / "stop").touch()
            result = future.result(timeout=8)
            assert result["ok"] is (ending == "complete"), result
            assert closed(port)
        finally:
            (scratch / "stop").touch()


def test_denied_publication_never_binds_or_launches(publication):
    """Ingress deny cannot be overridden by an egress allow or routine process approval."""
    call, manager, interaction, scratch, port, _registry = publication
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW)
    manager.set_default(Action.NETWORK_REQUEST, Decision.ALLOW)
    manager.set_default(Action.NETWORK_INGRESS, Decision.DENY)
    result = call('printf forbidden > "$TMPDIR/launched"')
    assert not result["ok"], result
    assert closed(port) and not (scratch / "launched").exists()
    interaction.prompt.assert_not_called()


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="macOS binds publication before the foreground Unix service starts",
)
def test_macos_ingress_waits_for_foreground_service(publication):
    """One early client disconnects while the same authorized service remains available to start."""
    call, _, _, scratch, port, _registry = publication
    source = (
        'printf waiting > "$TMPDIR/waiting"; '
        'while test ! -f "$TMPDIR/start"; do /bin/sleep 0.05; done; ' + service_source()
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(call, source)
        try:
            wait_for_service_marker(scratch / "waiting", future)
            with socket.create_connection(("127.0.0.1", port), timeout=2) as early:
                assert early.recv(1) == b""
            (scratch / "start").touch()
            wait_for_service_marker(scratch / "ready", future)
            assert b"ingress-ok" in exchange(port)
        finally:
            (scratch / "start").touch()
            (scratch / "stop").touch()
        result = future.result(timeout=8)
        assert result["ok"], result
    assert closed(port)


@pytest.mark.parametrize("other_attempt", [False, True])
@pytest.mark.skipif(
    sys.platform != "darwin", reason="macOS authenticates foreground Unix service endpoints"
)
def test_macos_payload_endpoint_swap_never_connects_unrelated_service(
    publication, other_attempt, native_directory
):
    """A payload-created symlink cannot turn approved ingress into a host or cross-attempt relay."""
    call, _, _, scratch, port, _registry = publication
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(call, service_source())
        holder = None
        try:
            wait_for_service_marker(scratch / "ready", future)
            endpoint = Path((scratch / "ready").read_text())
            if other_attempt:
                holder = executor.submit(
                    call,
                    'printf "%s" "$LOOP_LOCAL_SOCKET_DIR" > "$TMPDIR/holding.tmp"; '
                    '/bin/mv "$TMPDIR/holding.tmp" "$TMPDIR/holding"; '
                    'while test ! -f "$TMPDIR/release"; do /bin/sleep 0.05; done',
                    ingress_port=None,
                    local_ipc=True,
                )
                wait_for_service_marker(scratch / "holding", holder)
                unrelated = Path((scratch / "holding").read_text())
                assert unrelated.parent == endpoint.parent.parent
                assert unrelated != endpoint.parent
            else:
                unrelated = native_directory("/private/tmp")
            host_path = unrelated / "other.sock"
            with socket.socket(socket.AF_UNIX) as host:
                host.bind(str(host_path))
                host.listen(2)
                with socket.socket(socket.AF_UNIX) as control:
                    control.connect(str(host_path))
                    host.accept()[0].close()
                (scratch / "swap").write_text(str(host_path))
                wait_for_service_marker(scratch / "swapped", future)
                try:
                    response = exchange(port)
                except OSError:
                    response = b""
                assert b"ingress-ok" not in response
                host.settimeout(0.2)
                with pytest.raises(TimeoutError):
                    host.accept()
            host_path.unlink(missing_ok=True)
        finally:
            (scratch / "stop").touch()
            try:
                result = future.result(timeout=8)
                assert not result["ok"], result
                assert result["problem"]["code"] == "sandbox.unavailable", result
                assert "ingress connector failed" in result["problem"]["detail"], result
                assert "cleanup could not be confirmed" not in result["problem"]["detail"], result
            finally:
                if holder is not None:
                    (scratch / "release").touch()
                    held = holder.result(timeout=8)
                    assert held["ok"], held
    assert closed(port)


def test_macos_descendant_service_is_explicitly_unavailable(publication):
    """A child cannot impersonate the retained foreground identity or receive client bytes."""
    if sys.platform != "darwin":
        pytest.skip("macOS requires the live foreground Unix service identity")
    call, _, _, scratch, port, _registry = publication
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(call, service_source().removeprefix("exec ") + " & wait")
        try:
            wait_for_service_marker(scratch / "ready", future)
            try:
                response = exchange(port)
            except OSError:
                response = b""
            assert b"ingress-ok" not in response
            result = future.result(timeout=8)
            assert not result["ok"], result
            assert "descendant services are unsupported" in json.dumps(result)
            assert closed(port)
        finally:
            (scratch / "stop").touch()


def test_ingress_does_not_authorize_payload_connections_to_host_service(publication):
    """A host client reaches publication while payload-origin traffic cannot reach a host listener."""
    call, _, _, scratch, port, _registry = publication
    with socket.socket() as forbidden:
        forbidden.bind(("127.0.0.1", 0))
        forbidden.listen(4)
        forbidden_port = forbidden.getsockname()[1]
        with socket.create_connection(forbidden.getsockname(), timeout=1):
            forbidden.accept()[0].close()
        if sys.platform == "darwin":
            probe = (
                "use Socket; my $s; if (socket($s,PF_INET,SOCK_STREAM,0)) { "
                f'connect($s,pack_sockaddr_in({forbidden_port},inet_aton("127.0.0.1"))) '
                'and die "host service reached"; }'
            )
            command = "/usr/bin/perl -e " + shlex.quote(probe)
        else:
            probe = (
                "import socket; s=socket.socket(); s.settimeout(1); "
                f"assert s.connect_ex(('127.0.0.1',{forbidden_port})) != 0, 'host service reached'; s.close()"
            )
            command = "/usr/bin/python3 -c " + shlex.quote(probe)
        control = subprocess.run(
            shlex.split(command), check=False, capture_output=True, text=True, timeout=2
        )
        assert control.returncode != 0 and "host service reached" in control.stderr
        forbidden.accept()[0].close()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(call, command + " && " + service_source())
            try:
                wait_for_service_marker(scratch / "ready", future)
                assert b"ingress-ok" in exchange(port)
                forbidden.settimeout(0.2)
                with pytest.raises(TimeoutError):
                    forbidden.accept()
            finally:
                (scratch / "stop").touch()
                result = future.result(timeout=8)
                assert result["ok"], json.dumps(result)
    assert closed(port)
