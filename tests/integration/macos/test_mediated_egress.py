"""Verify macOS public mediation against owned, forbidden network endpoints."""

import json
import os
import secrets
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple
from urllib.request import ProxyHandler, build_opener

import pytest

from loop import Action, Decision, PermissionManager, ToolRegistry
from loop.execution import CommandExecutionService
from loop.execution.sandbox.macos import egress
from loop.permissions import PolicyScope
from loop.permissions.managed import ManagedPolicyStore
from loop.tools.system import run_command
from tests.integration.linux.egress_fixture import prepare

pytestmark = [pytest.mark.integration, pytest.mark.e2e, pytest.mark.macos]
_FIXTURE_GATEWAY = os.environ.get("LOOP_MACOS_EGRESS_FIXTURE_GATEWAY")
_DNS_FIXTURE_GATEWAY = os.environ.get("LOOP_MACOS_EGRESS_DNS_GATEWAY")
_QUALIFICATION_UV = shutil.which("uv")


class _LifecycleProbe(NamedTuple):
    """Identify one externally controlled broker and its owned fixture stream."""

    worker: subprocess.Popen
    socket_path: str
    workspace: Path


@pytest.fixture
def routed_egress_fixture():
    """Require the owned synthetic public endpoint and its native host route."""
    gateway = _FIXTURE_GATEWAY
    if gateway is None:
        pytest.skip("Requires the owned routed macOS egress fixture")
    route = subprocess.run(
        ["/sbin/route", "-n", "get", "-inet", "11.0.0.2"],
        capture_output=True,
        text=True,
        check=True,
    )
    fields = dict(line.strip().split(":", 1) for line in route.stdout.splitlines() if ":" in line)
    assert fields.get("route to", "").strip() == "11.0.0.2"
    assert fields.get("gateway", "").strip() == gateway
    assert not fields.get("interface", "").strip().startswith("lo")
    with build_opener(ProxyHandler({})).open("http://11.0.0.2:18080/", timeout=2) as response:
        assert response.read() == b"egress-fixture-ok\n"
    return "11.0.0.2"


@pytest.fixture
def owned_dns_fixture(routed_egress_fixture):
    """Require a verified temporary resolver mapping to the owned VM DNS listener."""
    if _DNS_FIXTURE_GATEWAY is None:
        pytest.skip("Requires the owned macOS resolver fixture")
    assert _DNS_FIXTURE_GATEWAY == _FIXTURE_GATEWAY
    resolver = Path("/private/etc/resolver/loop-egress-owned.test")
    directory = resolver.parent.lstat()
    file = resolver.lstat()
    assert stat.S_ISDIR(directory.st_mode) and directory.st_uid == 0
    assert stat.S_ISREG(file.st_mode) and file.st_uid == 0 and file.st_nlink == 1
    assert directory.st_mode & 0o022 == 0 and file.st_mode & 0o022 == 0
    assert resolver.read_text() == f"nameserver {_DNS_FIXTURE_GATEWAY}\n"
    name = "host-probe-" + secrets.token_hex(8) + ".loop-egress-owned.test"
    addresses = socket.getaddrinfo(name, 18080, type=socket.SOCK_STREAM)
    assert any(address[4][0] == routed_egress_fixture for address in addresses)
    service_name = "service-probe-" + secrets.token_hex(8) + ".loop-egress-owned.test"
    service = subprocess.Popen(
        ["/usr/bin/dns-sd", "-G", "v4", service_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 3
        while service_name.encode() not in _owned_dns_queries() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert service_name.encode() in _owned_dns_queries()
    finally:
        service.terminate()
        service.wait(timeout=3)
    return routed_egress_fixture


def _owned_dns_queries() -> bytes:
    """Read only the disposable fixture's DNS observation file over its checked route."""
    with build_opener(ProxyHandler({})).open(
        "http://11.0.0.2:18080/dns-queries.log", timeout=2
    ) as response:
        return response.read()


@pytest.fixture
def owned_lifecycle_probe(macos_local_volume, local_socket_parent, native_workspace):
    """Start one native mediator in another Loop process and own its forced cleanup."""
    script = Path(__file__).with_name("egress_lifecycle_probe.py")
    worker = subprocess.Popen(
        [
            sys.executable,
            str(script),
            str(native_workspace),
            str(local_socket_parent),
            macos_local_volume.device,
            str(macos_local_volume.device_id),
            macos_local_volume.name,
            macos_local_volume.volume_uuid,
        ],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        ready = native_workspace / "ready"
        deadline = time.monotonic() + 12
        while not ready.exists() and time.monotonic() < deadline:
            if worker.poll() is not None:
                raise AssertionError(worker.stderr.read())
            time.sleep(0.02)
        assert ready.exists(), "The owned native tunnel did not become ready."
        assert int((native_workspace / "payload-pid").read_text()) > 0
        proxy = (native_workspace / "proxy-url").read_text()
        assert proxy.startswith("socks5h://localhost")
        path = proxy.removeprefix("socks5h://localhost")
        assert Path(path).is_relative_to(local_socket_parent)
        yield _LifecycleProbe(worker, path, native_workspace)
    finally:
        if worker.poll() is None:
            os.kill(worker.pid, signal.SIGKILL)
        worker.wait(timeout=5)


def _owned_tunnel_open(worker_pid: int) -> bool:
    """Inspect only an owned broker process's fixture-bound TCP descriptor."""
    result = subprocess.run(
        [
            "/usr/sbin/lsof",
            "-nP",
            "-a",
            "-p",
            str(worker_pid),
            "-iTCP@11.0.0.2:18443",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and "ESTABLISHED" in result.stdout


def _start_slow_stream(public, executor, workspace: Path, origin: str, filename: str):
    """Start one approved HTTPS tunnel and wait for its first owned fixture bytes."""
    stream = workspace / filename
    future = executor.submit(
        public.call,
        f"/usr/bin/curl -kfsS --no-buffer --max-time 12 {origin}/slow "
        f'> "$WORKSPACE_ROOT/{filename}"',
        egress_origins=[origin],
    )
    deadline = time.monotonic() + 8
    while (
        not stream.exists() or "started" not in stream.read_text()
    ) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert stream.exists() and "started" in stream.read_text()
    return future, stream


def test_public_macos_curl_http_and_redirect(live_egress_clock, routed_egress_fixture):
    """A public origin and same-origin redirect work through the scoped broker."""
    origin = f"http://{routed_egress_fixture}:18080"
    for path, expected in (("/", "egress-fixture-ok\n"), ("/redirect", "redirect-ok\n")):
        result = live_egress_clock.call(
            f"/usr/bin/curl -fsSL --max-time 8 {origin}{path}",
            egress_origins=[origin],
        )
        assert result["ok"], json.dumps(result)
        assert result["result"]["stdout"]["content"] == expected
    live_egress_clock.host.run_host_command.assert_not_called()


def test_macos_ca_bundle_read_does_not_open_other_ssl_files(live_egress_clock):
    """Only the two required system TLS files are readable from the SSL directory."""
    origin = "https://11.0.0.2:18443"
    for path in ("/private/etc/ssl/cert.pem", "/private/etc/ssl/openssl.cnf"):
        required = live_egress_clock.call(
            f"/usr/bin/head -c 1 {path} >/dev/null", egress_origins=[origin]
        )
        assert required["ok"], json.dumps(required)
    for path in ("/private/etc/ssl/x509v3.cnf", "/etc/ssl/x509v3.cnf"):
        other = live_egress_clock.call(
            f"/usr/bin/head -c 1 {path} >/dev/null 2>&1", egress_origins=[origin]
        )
        assert not other["ok"], json.dumps(other)
        assert other["problem"]["code"] == "process.nonzero_exit"


def test_public_macos_resolver_ipc_cannot_query_owned_dns_fixture(
    live_egress_clock, owned_dns_fixture
):
    """Agent-origin libc and DNSService lookups emit no query to the owned resolver."""
    del owned_dns_fixture
    name = "sandbox-probe-" + secrets.token_hex(8) + ".loop-egress-owned.test"
    before = _owned_dns_queries()
    source = "/usr/bin/perl -MSocket -e " + shlex.quote(
        f'print gethostbyname("{name}") ? qq(resolved) : qq(unresolved)'
    )
    result = live_egress_clock.call(source, egress_origins=["https://11.0.0.2:18443"])
    assert result["ok"], json.dumps(result)
    assert result["result"]["stdout"]["content"] == "unresolved"
    modern = "/usr/bin/perl -MSocket -e " + shlex.quote(
        f'my ($err,@addresses)=Socket::getaddrinfo("{name}", "80"); '
        "print $err ? qq(unresolved) : qq(resolved)"
    )
    getaddrinfo = live_egress_clock.call(modern, egress_origins=["https://11.0.0.2:18443"])
    assert getaddrinfo["ok"], json.dumps(getaddrinfo)
    assert getaddrinfo["result"]["stdout"]["content"] == "unresolved"
    dns_service = live_egress_clock.call(
        f"/usr/bin/dns-sd -G v4 {name} >/dev/null 2>&1 & "
        'child=$!; /bin/sleep 1; kill "$child" 2>/dev/null; '
        'wait "$child" 2>/dev/null; true',
        egress_origins=["https://11.0.0.2:18443"],
    )
    assert dns_service["ok"], json.dumps(dns_service)
    assert _owned_dns_queries() == before
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_curl_https_and_git(live_egress_clock, routed_egress_fixture):
    """Curl and Git reuse the scoped Unix SOCKS path for owned HTTPS content."""
    origin = f"https://{routed_egress_fixture}:18443"
    curl = live_egress_clock.call(
        f"/usr/bin/curl -kfsS --max-time 8 {origin}/", egress_origins=[origin]
    )
    assert curl["ok"], json.dumps(curl)
    assert curl["result"]["stdout"]["content"] == "egress-https-ok\n"
    git = live_egress_clock.call(
        f"/usr/bin/git -c http.sslVerify=false ls-remote {origin}/repo.git HEAD",
        egress_origins=[origin],
    )
    assert git["ok"], json.dumps(git)
    assert "HEAD" in git["result"]["stdout"]["content"]
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_yolo_uses_only_the_scoped_mediator(live_egress_clock, routed_egress_fixture):
    """YOLO supplies eligible approval without disabling the default private-network limit."""
    origin = f"http://{routed_egress_fixture}:18080"
    live_egress_clock.manager.set_yolo(True)
    result = live_egress_clock.call(
        f"/usr/bin/curl -fsS --max-time 8 {origin}/", egress_origins=[origin]
    )
    assert result["ok"], json.dumps(result)
    assert result["result"]["stdout"]["content"] == "egress-fixture-ok\n"
    live_egress_clock.interaction.prompt.assert_not_called()
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_local_network_deny_retains_only_mediation(
    live_egress_clock, routed_egress_fixture
):
    """A local-network hard deny removes Unix bind while preserving approved egress."""
    origin = f"http://{routed_egress_fixture}:18080"
    live_egress_clock.manager.set_limit("deny_local_networks", True, scope=PolicyScope.SESSION)
    probe = "/usr/bin/perl -MSocket -e " + shlex.quote(
        "socket(my $s, PF_UNIX, SOCK_STREAM, 0) or die $!; "
        "my $path = qq($ENV{TMPDIR}/denied.sock); "
        "bind($s, pack_sockaddr_un($path)) and die 'local-bind'; "
        "print qq(local-denied\\n);"
    )
    result = live_egress_clock.call(
        f"{probe}; /usr/bin/curl -fsS --max-time 8 {origin}/",
        egress_origins=[origin],
    )
    assert result["ok"], json.dumps(result)
    assert result["result"]["stdout"]["content"] == "local-denied\negress-fixture-ok\n"
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_local_ipc_and_mediation_share_one_attempt(
    live_egress_clock, routed_egress_fixture
):
    """Qualified attempt-local Unix exchange and scoped external egress coexist."""
    origin = f"http://{routed_egress_fixture}:18080"
    probe = "/usr/bin/perl -MSocket -e " + shlex.quote(
        "socket(my $s, PF_UNIX, SOCK_STREAM, 0) or die $!; "
        "my $path = qq($ENV{LOOP_LOCAL_SOCKET_DIR}/exchange.sock); "
        "bind($s, pack_sockaddr_un($path)) or die $!; listen($s, 1) or die $!; "
        "socket(my $c, PF_UNIX, SOCK_STREAM, 0) or die $!; "
        "connect($c, pack_sockaddr_un($path)) or die $!; "
        "defined accept(my $p, $s) or die $!; send($c, qq(local), 0) or die $!; "
        "my $data = qq(); defined recv($p, $data, 5, 0) or die $!; print qq($data\\n);"
    )
    result = live_egress_clock.call(
        f"{probe}; /usr/bin/curl -fsS --max-time 8 {origin}/",
        egress_origins=[origin],
    )
    assert result["ok"], json.dumps(result)
    assert result["result"]["stdout"]["content"] == "local\negress-fixture-ok\n"
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_different_origins_stay_scoped_in_parallel(
    live_egress_clock, routed_egress_fixture
):
    """Concurrent HTTP and HTTPS grants keep separate attempt-owned destinations."""
    http = f"http://{routed_egress_fixture}:18080"
    https = f"https://{routed_egress_fixture}:18443"
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            live_egress_clock.call,
            f"/usr/bin/curl -fsS --max-time 8 {http}/",
            egress_origins=[http],
        )
        second = executor.submit(
            live_egress_clock.call,
            f"/usr/bin/curl -kfsS --max-time 8 {https}/",
            egress_origins=[https],
        )
        assert first.result(timeout=12)["result"]["stdout"]["content"] == "egress-fixture-ok\n"
        assert second.result(timeout=12)["result"]["stdout"]["content"] == "egress-https-ok\n"
    live_egress_clock.host.run_host_command.assert_not_called()


@pytest.mark.parametrize("managed_deny", ["network", "local"])
def test_public_macos_managed_limits_reach_native_mediation(
    live_egress_clock, native_workspace, routed_egress_fixture, managed_deny
):
    """A validated managed network deny blocks egress; local deny keeps only the broker."""
    parent = native_workspace.parent / "system-egress"
    directory = parent / "loop"
    directory.mkdir(parents=True)
    parent.chmod(0o755)
    directory.chmod(0o755)
    identity = "e" * 32
    (parent / "loop.managed.expect.json").write_text(
        json.dumps({"version": 1, "identity": identity, "minimum_revision": 1}),
        encoding="utf-8",
    )
    (directory / "managed-policy.json").write_text(
        json.dumps(
            {
                "version": 1,
                "identity": identity,
                "revision": 1,
                "rules": (
                    [{"id": "deny-egress", "decision": "deny", "action": "network.request"}]
                    if managed_deny == "network"
                    else []
                ),
                "deny_local_networks": managed_deny == "local",
            }
        ),
        encoding="utf-8",
    )
    (parent / "loop.managed.expect.json").chmod(0o644)
    (directory / "managed-policy.json").chmod(0o644)
    manager = PermissionManager(
        native_workspace,
        managed_policy_store=ManagedPolicyStore(
            directory, trust_anchor=native_workspace.parent, owner_uid=os.getuid()
        ),
    )
    registry = ToolRegistry(
        [run_command],
        permission_manager=manager,
        execution_service=CommandExecutionService(
            live_egress_clock.backend, live_egress_clock.host
        ),
    )
    origin = f"http://{routed_egress_fixture}:18080"
    try:
        result = json.loads(
            registry.call(
                "run_command",
                json.dumps(
                    {
                        "command": f"/usr/bin/curl -fsS --max-time 8 {origin}/",
                        "cwd": "$WORKSPACE_ROOT",
                        "egress_origins": [origin],
                    }
                ),
                interaction=live_egress_clock.interaction,
                instructions_manager=live_egress_clock.instructions,
            )
        )
        if managed_deny == "network":
            assert not result["ok"], json.dumps(result)
            live_egress_clock.interaction.prompt.assert_not_called()
        else:
            assert result["ok"], json.dumps(result)
            assert result["result"]["stdout"]["content"] == "egress-fixture-ok\n"
        live_egress_clock.host.run_host_command.assert_not_called()
    finally:
        registry.close()


def test_public_macos_denies_ungranted_redirect(live_egress_clock, routed_egress_fixture):
    """A checked HTTP redirect cannot create a new ungranted HTTPS tunnel."""
    origin = f"http://{routed_egress_fixture}:18080"
    result = live_egress_clock.call(
        f"/usr/bin/curl -kfsSL --max-time 8 {origin}/redirect-forbidden",
        egress_origins=[origin],
    )
    assert result["problem"]["code"] == "process.nonzero_exit", json.dumps(result)
    live_egress_clock.host.run_host_command.assert_not_called()


@pytest.mark.parametrize("public_answer", [True, False])
def test_public_macos_mediator_checks_each_resolved_address(
    live_egress_clock, routed_egress_fixture, monkeypatch, public_answer
):
    """Mixed DNS chooses only a routed public address; private-only DNS is denied."""
    del routed_egress_fixture
    original = socket.getaddrinfo

    def answers(host, port, *args, **kwargs):
        """Supply a controlled mixed or forbidden response to the real broker."""
        if host == "loop-egress-mixed.invalid":
            addresses = ["192.168.64.29"]
            if public_answer:
                addresses.append("11.0.0.2")
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
                for address in addresses
            ]
        return original(host, port, *args, **kwargs)

    monkeypatch.setattr(egress.socket, "getaddrinfo", answers)
    origin = "http://loop-egress-mixed.invalid:18080"
    result = live_egress_clock.call(
        f"/usr/bin/curl -fsS --max-time 8 {origin}/", egress_origins=[origin]
    )
    if public_answer:
        assert result["ok"], json.dumps(result)
        assert result["result"]["stdout"]["content"] == "egress-fixture-ok\n"
    else:
        assert result["problem"]["code"] == "process.nonzero_exit", json.dumps(result)
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_mediator_rechecks_dns_on_a_later_connection(
    live_egress_clock, native_workspace, routed_egress_fixture, monkeypatch
):
    """A forbidden rebinding cannot reuse authority from an earlier HTTP connection."""
    del routed_egress_fixture
    current = ["11.0.0.2"]
    original = socket.getaddrinfo

    def answers(host, port, *args, **kwargs):
        """Move the synthetic name from an owned public route to a private target."""
        if host == "loop-egress-rebind.invalid":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (current[0], port))]
        return original(host, port, *args, **kwargs)

    monkeypatch.setattr(egress.socket, "getaddrinfo", answers)
    origin = "http://loop-egress-rebind.invalid:18080"
    started = native_workspace / "first-egress-done"
    release = native_workspace / "release-second-egress"
    source = (
        f"/usr/bin/curl -fsS --max-time 8 {origin}/ > first-response; "
        'touch "$WORKSPACE_ROOT/first-egress-done"; '
        'while test ! -e "$WORKSPACE_ROOT/release-second-egress"; do /bin/sleep 0.1; done; '
        f"/usr/bin/curl -fsS --max-time 8 {origin}/ > second-response"
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(live_egress_clock.call, source, egress_origins=[origin])
        try:
            deadline = time.monotonic() + 8
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert started.exists()
            current[0] = "192.168.64.29"
        finally:
            release.write_text("continue", encoding="utf-8")
        result = running.result(timeout=10)
    assert (native_workspace / "first-response").read_text() == "egress-fixture-ok\n"
    assert result["problem"]["code"] == "process.nonzero_exit", json.dumps(result)
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_policy_change_closes_an_active_https_tunnel(
    live_egress_clock, native_workspace, routed_egress_fixture
):
    """A new exact policy revision closes the old scoped stream before acknowledgement."""
    origin = f"https://{routed_egress_fixture}:18443"
    stream = native_workspace / "revoked-stream"
    source = (
        f"/usr/bin/curl -kfsS --no-buffer --max-time 12 {origin}/slow "
        '> "$WORKSPACE_ROOT/revoked-stream"'
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(live_egress_clock.call, source, egress_origins=[origin])
        deadline = time.monotonic() + 8
        while (
            not stream.exists() or "started" not in stream.read_text()
        ) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert stream.exists() and "started" in stream.read_text()
        live_egress_clock.manager.set_yolo(True)
        result = running.result(timeout=8)
    assert result["problem"]["code"] == "sandbox.unavailable", json.dumps(result)
    assert "finished" not in stream.read_text()
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_missing_revision_ack_closes_tunnel_despite_healthy_heartbeats(
    live_egress_clock, native_workspace, routed_egress_fixture, monkeypatch
):
    """A healthy revision heartbeat cannot extend a withheld closure acknowledgement."""
    origin = f"https://{routed_egress_fixture}:18443"
    stream = native_workspace / "unacknowledged-stream"
    original_revision = live_egress_clock.manager.egress_policy_revision()
    monkeypatch.setattr(
        live_egress_clock.manager,
        "try_egress_policy_revision",
        lambda: original_revision,
    )
    original_revoke = egress.MacOSEgressBroker.revoke_revision

    def withheld_ack(broker, revision):
        """Delay only the revision reply while health checks continue."""
        time.sleep(2.2)
        return original_revoke(broker, revision)

    monkeypatch.setattr(egress.MacOSEgressBroker, "revoke_revision", withheld_ack)
    source = (
        f"/usr/bin/curl -kfsS --no-buffer --max-time 12 {origin}/slow "
        '> "$WORKSPACE_ROOT/unacknowledged-stream"'
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(live_egress_clock.call, source, egress_origins=[origin])
        deadline = time.monotonic() + 8
        while (
            not stream.exists() or "started" not in stream.read_text()
        ) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert stream.exists() and "started" in stream.read_text()
        with pytest.raises(OSError, match="transition is uncertain"):
            live_egress_clock.manager.set_yolo(True)
        result = running.result(timeout=8)
    assert result["problem"]["code"] == "sandbox.unavailable", json.dumps(result)
    assert "finished" not in stream.read_text()
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_parallel_load_keeps_unchanged_policy_tunnels(
    live_egress_clock, native_workspace, routed_egress_fixture
):
    """Six healthy HTTPS streams survive one unchanged-policy admission pause."""
    origin = f"https://{routed_egress_fixture}:18443"
    live_egress_clock.manager.set_yolo(True)
    with ThreadPoolExecutor(max_workers=6) as executor:
        running = [
            executor.submit(
                live_egress_clock.call,
                f"/usr/bin/curl -kfsS --no-buffer --max-time 15 {origin}/slow "
                f'> "$WORKSPACE_ROOT/load-stream-{index}"',
                egress_origins=[origin],
            )
            for index in range(6)
        ]
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if all(
                (native_workspace / f"load-stream-{index}").exists()
                and "started" in (native_workspace / f"load-stream-{index}").read_text()
                for index in range(6)
            ):
                break
            time.sleep(0.02)
        assert all(
            (native_workspace / f"load-stream-{index}").exists()
            and "started" in (native_workspace / f"load-stream-{index}").read_text()
            for index in range(6)
        )
        with live_egress_clock.manager.policy_update():
            pass
        for future in running:
            result = future.result(timeout=15)
            assert result["ok"], json.dumps(result)
    assert all(
        "finished" in (native_workspace / f"load-stream-{index}").read_text() for index in range(6)
    )
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_other_manager_write_waits_for_reload(
    live_egress_clock, native_workspace, routed_egress_fixture
):
    """Another manager's persisted write affects a live tunnel only after reload."""
    origin = f"https://{routed_egress_fixture}:18443"
    with ThreadPoolExecutor(max_workers=1) as executor:
        first, stream = _start_slow_stream(
            live_egress_clock, executor, native_workspace, origin, "other-manager-stream"
        )
        other = PermissionManager(native_workspace)
        other.set_default(Action.FILESYSTEM_READ, Decision.DENY)
        other.close()
        assert first.result(timeout=12)["ok"]
        assert "finished" in stream.read_text()
        stream.unlink()
        second, stream = _start_slow_stream(
            live_egress_clock, executor, native_workspace, origin, "other-manager-stream"
        )
        live_egress_clock.manager.reload()
        result = second.result(timeout=5)
    assert result["problem"]["code"] == "sandbox.unavailable", json.dumps(result)
    assert "finished" not in stream.read_text()


def test_public_macos_direct_policy_edit_waits_for_reload(
    live_egress_clock, native_workspace, routed_egress_fixture
):
    """A direct policy-file edit changes a live tunnel only after accepted reload."""
    origin = f"https://{routed_egress_fixture}:18443"
    live_egress_clock.manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first, stream = _start_slow_stream(
            live_egress_clock, executor, native_workspace, origin, "direct-edit-stream"
        )
        policy = native_workspace / ".loop" / "permissions.yaml"
        policy.write_text(policy.read_text() + "\n# owned external edit\n")
        assert first.result(timeout=12)["ok"]
        assert "finished" in stream.read_text()
        stream.unlink()
        second, stream = _start_slow_stream(
            live_egress_clock, executor, native_workspace, origin, "direct-edit-stream"
        )
        live_egress_clock.manager.reload()
        result = second.result(timeout=5)
    assert result["problem"]["code"] == "sandbox.unavailable", json.dumps(result)
    assert "finished" not in stream.read_text()


def test_public_macos_supervisor_death_closes_owned_tunnel(
    routed_egress_fixture, owned_lifecycle_probe
):
    """An external controller observes tunnel and broker closure after Loop SIGKILL."""
    del routed_egress_fixture
    worker = owned_lifecycle_probe.worker
    assert _owned_tunnel_open(worker.pid), "The owned HTTPS tunnel was not established."
    started = time.monotonic()
    os.kill(worker.pid, signal.SIGKILL)
    closed = False
    while time.monotonic() - started < 2.0:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            try:
                client.connect(owned_lifecycle_probe.socket_path)
            except OSError:
                broker_closed = True
            else:
                broker_closed = False
        if broker_closed and not _owned_tunnel_open(worker.pid):
            closed = True
            break
        time.sleep(0.02)
    assert closed, "Loop death left broker admission or the fixture tunnel live after 2 s."


@pytest.mark.parametrize("pause_seconds", [0.4, 2.2])
def test_public_macos_heartbeat_bound_survives_short_pause_and_expires_long_one(
    routed_egress_fixture, owned_lifecycle_probe, pause_seconds
):
    """A runnable mediator renews after a short pause and closes after a spent lease."""
    del routed_egress_fixture
    worker = owned_lifecycle_probe.worker
    assert _owned_tunnel_open(worker.pid)
    os.kill(worker.pid, signal.SIGSTOP)
    try:
        time.sleep(pause_seconds)
    finally:
        os.kill(worker.pid, signal.SIGCONT)
    result_file = owned_lifecycle_probe.workspace / "result"
    deadline = time.monotonic() + 12
    while not result_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert result_file.exists(), "The resumed native mediator did not settle its stream."
    result = json.loads(result_file.read_text())
    if pause_seconds < 2:
        assert result["ok"], json.dumps(result)
        assert "finished" in (owned_lifecycle_probe.workspace / "stream").read_text()
    else:
        assert result["problem"]["code"] == "sandbox.unavailable", json.dumps(result)
        assert "finished" not in (owned_lifecycle_probe.workspace / "stream").read_text()


def test_public_macos_mediator_refuses_a_private_socks_destination(live_egress_clock):
    """A scoped proxy call reaches the broker but cannot connect to a private address."""
    public = live_egress_clock
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        result = public.call(
            f"/usr/bin/curl --silent --show-error --max-time 3 http://127.0.0.1:{port}/",
            egress_origins=["http://example.com"],
        )
        assert result["problem"]["code"] == "process.nonzero_exit", json.dumps(result)
        assert "SOCKS5" in result["problem"]["metadata"]["stderr"]["content"]
        listener.settimeout(0.1)
        with pytest.raises(TimeoutError):
            listener.accept()
    public.host.run_host_command.assert_not_called()


@pytest.mark.parametrize("family,host", [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")])
def test_public_macos_mediator_does_not_open_direct_host_loopback(live_egress_clock, family, host):
    """An approved origin cannot turn proxy bypass into IPv4 or IPv6 host access."""
    public = live_egress_clock
    with socket.socket(family) as listener:
        listener.bind((host, 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        display = f"[{host}]" if family == socket.AF_INET6 else host
        result = public.call(
            f"/usr/bin/curl --noproxy '*' --silent --show-error --max-time 3 "
            f"'http://{display}:{port}/'",
            egress_origins=["https://example.com"],
        )
        assert result["problem"]["code"] == "process.nonzero_exit", json.dumps(result)
        listener.settimeout(0.1)
        with pytest.raises(TimeoutError):
            listener.accept()
    public.host.run_host_command.assert_not_called()


def test_public_macos_mediator_does_not_open_direct_public_route(
    live_egress_clock, routed_egress_fixture
):
    """A routed public fixture is reachable only through its approved mediator."""
    origin = f"http://{routed_egress_fixture}:18080"
    result = live_egress_clock.call(
        f"/usr/bin/curl --noproxy '*' -fsS --max-time 3 {origin}/",
        egress_origins=[origin],
    )
    assert result["problem"]["code"] == "process.nonzero_exit", json.dumps(result)
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_mediator_denies_direct_udp_and_ipv6(live_egress_clock):
    """Mediation never grants direct datagram DNS or external IPv6 transports."""
    source = "/usr/bin/perl -MSocket -e " + shlex.quote(
        "socket(my $udp, PF_INET, SOCK_DGRAM, 0) or die $!; "
        "connect($udp, pack_sockaddr_in(53, inet_aton('1.1.1.1'))) "
        "and die 'direct-udp'; print qq(udp=).(0+$!).qq(\\n); "
        "socket(my $v6, PF_INET6, SOCK_STREAM, 0) or die $!; "
        "connect($v6, Socket::pack_sockaddr_in6(443, "
        "Socket::inet_pton(AF_INET6, '2001:4860:4860::8888'))) "
        "and die 'direct-ipv6'; print qq(ipv6=).(0+$!).qq(\\n);"
    )
    result = live_egress_clock.call(source, egress_origins=["https://example.com"])
    assert result["ok"], json.dumps(result)
    assert result["result"]["stdout"]["content"] == "udp=1\nipv6=1\n"
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_mediator_does_not_inherit_host_socket(live_egress_clock):
    """A host-owned inheritable descriptor cannot bypass the native proxy boundary."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.set_inheritable(True)
        descriptor = listener.fileno()
        identity = os.fstat(descriptor)
        source = "/usr/bin/perl -e " + shlex.quote(
            f"my $fd={descriptor}; "
            "if (open(my $s, '<&='.$fd)) { my @i=stat($s); "
            "print qq($i[0]:$i[1]); } else { print qq(absent); }"
        )
        result = live_egress_clock.call(source, egress_origins=["https://example.com"])
        assert result["ok"], json.dumps(result)
        assert result["result"]["stdout"]["content"] != (f"{identity.st_dev}:{identity.st_ino}")
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_copied_proxy_path_cannot_cross_attempts(live_egress_clock, native_workspace):
    """A concurrent attempt cannot use the first attempt's live broker path."""
    copied = native_workspace / "copied-proxy"
    release = native_workspace / "release-proxy"
    origin = "http://example.com"
    source = (
        'printf %s "$http_proxy" > "$WORKSPACE_ROOT/copied-proxy"; '
        'while test ! -e "$WORKSPACE_ROOT/release-proxy"; do /bin/sleep 0.1; done'
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(live_egress_clock.call, source, egress_origins=[origin])
        try:
            deadline = time.monotonic() + 8
            while not copied.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert copied.exists()
            result = live_egress_clock.call(
                f'/usr/bin/curl -fsS --max-time 3 -x "$(cat {copied})" {origin}/',
                egress_origins=[origin],
            )
            assert result["problem"]["code"] == "process.nonzero_exit", json.dumps(result)
        finally:
            release.write_text("done", encoding="utf-8")
        assert running.result(timeout=8)["ok"]
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_uv_unix_socks_client_is_unavailable(
    live_egress_clock, native_workspace, routed_egress_fixture, monkeypatch
):
    """An installed uv without pathname-SOCKS support fails against an owned endpoint."""
    if _QUALIFICATION_UV is None:
        pytest.skip("uv is not installed")
    uv = Path(_QUALIFICATION_UV)
    python = Path(sys.executable).resolve()
    monkeypatch.setenv("PATH", os.pathsep.join((str(uv.parent), str(python.parent), os.defpath)))
    (native_workspace / "pyproject.toml").write_text(
        '[project]\nname = "egress-client"\nversion = "0.1.0"\n'
        'requires-python = ">=3.9"\ndependencies = ["loop-egress-fixture==1.0.0"]\n'
        "[tool.uv]\npackage = false\n",
        encoding="utf-8",
    )
    origin = f"http://{routed_egress_fixture}:18080"
    lookup = json.loads(
        live_egress_clock.registry.call(
            "resolve_executable",
            json.dumps({"name": [uv.name, python.name]}),
            interaction=live_egress_clock.interaction,
            instructions_manager=live_egress_clock.instructions,
        )
    )
    assert lookup["ok"], json.dumps(lookup)
    grants = [item["grant_reference"] for item in lookup["result"]["executables"]]
    result = live_egress_clock.call(
        'UV_CACHE_DIR="$TMPDIR/uv-cache" UV_PYTHON_INSTALL_DIR="$TMPDIR/uv-python" '
        'UV_PROJECT_ENVIRONMENT="$TMPDIR/uv-env" '
        f"UV_PYTHON_DOWNLOADS=never UV_PYTHON={shlex.quote(str(python))} "
        f"{shlex.quote(str(uv))} --no-cache sync --index-url {origin}/simple",
        egress_origins=[origin],
        executable_grant=grants,
    )
    assert result["problem"]["code"] == "process.nonzero_exit", json.dumps(result)
    assert "socks" in result["problem"]["metadata"]["stderr"]["content"].lower(), json.dumps(result)
    live_egress_clock.host.run_host_command.assert_not_called()


def _offline_uv_project(live_egress_clock, native_workspace, monkeypatch):
    """Prepare an owned uv project and executable grants for an offline sync."""
    if _QUALIFICATION_UV is None:
        pytest.skip("uv is not installed")
    uv = Path(_QUALIFICATION_UV)
    python = Path(sys.executable).resolve()
    monkeypatch.setenv("PATH", os.pathsep.join((str(uv.parent), str(python.parent), os.defpath)))
    (native_workspace / "pyproject.toml").write_text(
        '[project]\nname = "egress-client"\nversion = "0.1.0"\n'
        'requires-python = ">=3.9"\ndependencies = ["loop-egress-fixture==1.0.0"]\n'
        "[tool.uv]\npackage = false\n",
        encoding="utf-8",
    )
    lookup = json.loads(
        live_egress_clock.registry.call(
            "resolve_executable",
            json.dumps({"name": [uv.name, python.name]}),
            interaction=live_egress_clock.interaction,
            instructions_manager=live_egress_clock.instructions,
        )
    )
    assert lookup["ok"], json.dumps(lookup)
    grants = [item["grant_reference"] for item in lookup["result"]["executables"]]
    wheels = native_workspace / "wheels"
    wheels.mkdir()
    command = (
        'UV_CACHE_DIR="$TMPDIR/uv-cache" UV_PYTHON_INSTALL_DIR="$TMPDIR/uv-python" '
        'UV_PROJECT_ENVIRONMENT="$TMPDIR/uv-env" '
        f"UV_PYTHON_DOWNLOADS=never UV_PYTHON={shlex.quote(str(python))} "
        f"{shlex.quote(str(uv))} --no-cache sync --offline --no-index "
        '--find-links "$WORKSPACE_ROOT/wheels" '
        '&& "$TMPDIR/uv-env/bin/python" '
        '-c "import loop_egress_fixture; print(loop_egress_fixture.VALUE)"'
    )
    return wheels, command, grants


def test_public_macos_uv_syncs_an_owned_wheel_offline(
    live_egress_clock, native_workspace, monkeypatch
):
    """A native sandbox installs a local wheel without general or mediated network access."""
    wheels, command, grants = _offline_uv_project(live_egress_clock, native_workspace, monkeypatch)
    package_fixture = native_workspace.parent / "package-fixture"
    prepare(package_fixture)
    wheel_name = "loop_egress_fixture-1.0.0-py3-none-any.whl"
    shutil.copyfile(package_fixture / "wheels" / wheel_name, wheels / wheel_name)
    result = live_egress_clock.call(command, executable_grant=grants)
    assert result["ok"], json.dumps(result)
    assert result["result"]["boundary"] == "sandbox"
    assert result["result"]["stdout"]["content"] == "ok\n"
    assert live_egress_clock.manager.sandbox_network_limit("run_command") is not None
    live_egress_clock.host.run_host_command.assert_not_called()


def test_public_macos_curl_download_supports_offline_uv_sync(
    live_egress_clock, native_workspace, routed_egress_fixture, monkeypatch
):
    """An approved wheel download supports offline uv sync without a broad network grant."""
    wheels, command, grants = _offline_uv_project(live_egress_clock, native_workspace, monkeypatch)
    wheel = wheels / "loop_egress_fixture-1.0.0-py3-none-any.whl"
    origin = f"http://{routed_egress_fixture}:18080"
    prompts_before = live_egress_clock.interaction.prompt.call_count
    info_before = live_egress_clock.interaction.info.call_count
    download = live_egress_clock.call(
        f'/usr/bin/curl -fsS --max-time 8 -o "$WORKSPACE_ROOT/wheels/{wheel.name}" '
        f"{origin}/wheels/{wheel.name}",
        egress_origins=[origin],
    )
    assert download["ok"], json.dumps(download)
    assert download["result"]["boundary"] == "sandbox"
    assert wheel.is_file()
    assert live_egress_clock.interaction.prompt.call_count == prompts_before + 1
    assert any(
        origin in call.args[0]
        for call in live_egress_clock.interaction.info.call_args_list[info_before:]
    )

    offline = live_egress_clock.call(command, executable_grant=grants)
    assert offline["ok"], json.dumps(offline)
    assert offline["result"]["boundary"] == "sandbox"
    assert offline["result"]["stdout"]["content"] == "ok\n"
    assert live_egress_clock.manager.sandbox_network_limit("run_command") is not None
    live_egress_clock.host.run_host_command.assert_not_called()
