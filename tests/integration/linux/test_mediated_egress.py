"""Qualify public Linux mediated HTTP against an owned isolated endpoint."""

import fcntl
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

from loop import Action, Decision, PermissionManager, ToolRegistry
from loop.constants import NATIVE_INSTALLATION_DIRECTORY
from loop.execution import CommandExecutionService
from loop.execution.sandbox.linux.backend import LinuxNativeBackend
from loop.execution.sandbox.linux.runtime import execution_authority
from loop.execution.sandbox.linux.transport import EgressLease
from loop.interaction.interaction import Interaction
from loop.tools.system import run_command

pytestmark = [pytest.mark.integration, pytest.mark.e2e, pytest.mark.linux]
_ORIGIN = os.environ.get("LOOP_NATIVE_EGRESS_FIXTURE_ORIGIN")
_HTTPS_ORIGIN = os.environ.get("LOOP_NATIVE_EGRESS_HTTPS_FIXTURE_ORIGIN")
_MIXED_DNS_ORIGIN = os.environ.get("LOOP_NATIVE_EGRESS_MIXED_DNS_ORIGIN")
_PRIVATE_DNS_ORIGIN = os.environ.get("LOOP_NATIVE_EGRESS_PRIVATE_DNS_ORIGIN")
_REBIND_DNS_ORIGIN = os.environ.get("LOOP_NATIVE_EGRESS_REBIND_DNS_ORIGIN")
_REBIND_TEST = os.environ.get("LOOP_NATIVE_EGRESS_REBIND_TEST")
_PRIVATE_IP = os.environ.get("LOOP_NATIVE_EGRESS_PRIVATE_IP")
_SUPERVISOR_TEST = os.environ.get("LOOP_NATIVE_EGRESS_SUPERVISOR_TEST")
_SUSPEND_TEST = os.environ.get("LOOP_NATIVE_EGRESS_SUSPEND_TEST")
_UV = (
    os.environ.get("LOOP_NATIVE_EGRESS_UV")
    or shutil.which("uv")
    or str(Path.home() / ".local/bin/uv")
)


class RefusingHost:
    """Reject any accidental host fallback during public egress qualification."""

    def run_host_command(self, request):
        """Reject an unrequested host execution boundary."""
        raise AssertionError("Mediated egress offered host execution.")


@pytest.fixture
def registry(scratch_root, native_directory):
    """Provide one YOLO manager and public tool registry on a qualified native host."""
    if not _ORIGIN:
        pytest.skip("Requires an owned isolated LOOP_NATIVE_EGRESS_FIXTURE_ORIGIN")
    execution_authority(NATIVE_INSTALLATION_DIRECTORY / "management.json")
    workspace = native_directory(scratch_root, name="mediated-egress-workspace")
    backend = LinuxNativeBackend()
    manager = PermissionManager(workspace, yolo=True)
    tools = ToolRegistry(
        [run_command],
        permission_manager=manager,
        execution_service=CommandExecutionService(backend, RefusingHost()),
    )
    try:
        yield tools, workspace, manager
    finally:
        tools.close()


def _call(
    registry,
    workspace: Path,
    command: str,
    origins: list[str],
    *,
    read_only: bool = True,
    read_roots: list[str] | None = None,
) -> dict:
    """Dispatch one exact command through the public tool boundary."""
    return json.loads(
        registry.call(
            "run_command",
            json.dumps(
                {
                    "command": command,
                    "cwd": str(workspace),
                    "read_only": read_only,
                    "read_roots": read_roots or [],
                    "egress_origins": origins,
                }
            ),
        )
    )


def test_owned_http_get_uses_mediator_without_host_execution(registry):
    """Fetch from an owned separately routed namespace under the private-network hard limit."""
    tools, workspace, _manager = registry
    result = _call(tools, workspace, f"curl -fsS --max-time 8 {_ORIGIN}/", [_ORIGIN])
    assert result["ok"], result
    assert result["result"]["boundary"] == "sandbox"
    assert result["result"]["stdout"]["content"] == "egress-fixture-ok\n"


def test_local_deny_preserves_only_attempt_mediator(registry):
    """Egress-only commands reach the broker but no local, host, alias or direct route."""
    tools, workspace, manager = registry
    manager.set_limit("deny_local_networks", True)
    with socket.socket() as host_service, socket.socket() as alternate_service:
        host_service.bind(("127.0.0.1", 0))
        host_service.listen(1)
        host_port = host_service.getsockname()[1]
        alternate_service.bind(("127.0.0.1", 0))
        alternate_service.listen(1)
        alternate_port = alternate_service.getsockname()[1]
        source = f"""
import errno, json, os, socket, urllib.parse, urllib.request
checks = {{}}
checks['mediated'] = urllib.request.urlopen('{_ORIGIN}/', timeout=8).read() == b'egress-fixture-ok\\n'
proxy = urllib.parse.urlsplit(os.environ['http_proxy'])
host_port = {host_port} if {host_port} != proxy.port else {alternate_port}
with socket.socket() as server:
    try:
        server.bind(('127.0.0.1', 0))
        checks['listener'] = False
    except OSError as error:
        checks['listener'] = error.errno == errno.EPERM
for label, family, kind in (
    ('udp', socket.AF_INET, socket.SOCK_DGRAM),
    ('ipv6', socket.AF_INET6, socket.SOCK_STREAM),
    ('unix', socket.AF_UNIX, socket.SOCK_STREAM),
):
    try:
        socket.socket(family, kind)
        checks[label] = False
    except OSError as error:
        checks[label] = error.errno == errno.EPERM
for label, address in (
    ('host', ('127.0.0.1', host_port)),
    ('proxy-alias', ('127.0.0.2', proxy.port)),
    ('external-direct', ('11.0.0.2', 18080)),
):
    try:
        with socket.create_connection(address, timeout=0.25):
            checks[label] = False
    except OSError:
        checks[label] = True
print(json.dumps(checks, sort_keys=True))
"""
        result = _call(tools, workspace, "python3 -I -c " + shlex.quote(source), [_ORIGIN])
    assert result["ok"], result
    checks = json.loads(result["result"]["stdout"]["content"])
    assert all(checks.values()), checks


def test_local_and_mediated_egress_share_one_isolated_attempt(registry):
    """Combined authority permits owned loopback TCP and only approved external HTTP."""
    tools, workspace, _manager = registry
    source = f"""
import socket, urllib.request
with socket.socket() as server:
    server.bind(('127.0.0.1', 0))
    server.listen(1)
    with socket.socket() as client:
        client.connect(server.getsockname())
        with server.accept()[0] as peer:
            client.sendall(b'local')
            print(peer.recv(5).decode())
print(urllib.request.urlopen('{_ORIGIN}/', timeout=8).read().decode().strip())
"""
    result = _call(tools, workspace, "python3 -I -c " + shlex.quote(source), [_ORIGIN])
    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "local\negress-fixture-ok\n"


def test_mediator_child_cannot_be_inspected_or_duplicated_by_payload(registry):
    """The unfiltered trusted proxy is sealed against ptrace and descriptor theft."""
    tools, workspace, manager = registry
    manager.set_limit("deny_local_networks", True)
    source = f"""
import ctypes, errno, json, os, pathlib, urllib.request
assert urllib.request.urlopen('{_ORIGIN}/', timeout=8).read() == b'egress-fixture-ok\\n'
own = os.getpid()
parent = os.getppid()
children = set()
for owner in (own, parent):
    try:
        data = pathlib.Path(f'/proc/{{owner}}/task/{{owner}}/children').read_text()
        children.update(int(value) for value in data.split())
    except OSError:
        pass
children.discard(own)
checks = {{'proxy-found': len(children) == 1}}
if children:
    proxy = min(children)
    libc = ctypes.CDLL(None, use_errno=True)
    checks['ptrace'] = libc.ptrace(16, proxy, 0, 0) == -1 and ctypes.get_errno() in (errno.EPERM, errno.EACCES)
    checks['proc-fd'] = True
    for number in range(3, 12):
        try:
            descriptor = os.open(f'/proc/{{proxy}}/fd/{{number}}', os.O_RDONLY)
        except OSError:
            continue
        else:
            os.close(descriptor)
            checks['proc-fd'] = False
    pidfd = os.pidfd_open(proxy)
    try:
        checks['pidfd-getfd'] = libc.syscall(438, pidfd, 3, 0) == -1 and ctypes.get_errno() in (errno.EPERM, errno.EACCES)
    finally:
        os.close(pidfd)
print(json.dumps(checks, sort_keys=True))
"""
    result = _call(tools, workspace, "python3 -I -c " + shlex.quote(source), [_ORIGIN])
    assert result["ok"], result
    checks = json.loads(result["result"]["stdout"]["content"])
    assert all(checks.values()), checks


def test_interactive_scoped_approval_uses_the_same_public_dispatch(registry):
    """A normal user approval permits the scoped request in its original tool call."""
    _tools, workspace, _manager = registry
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "approve"
    manager = PermissionManager(workspace, interaction=interaction)
    tools = ToolRegistry(
        [run_command],
        permission_manager=manager,
        execution_service=CommandExecutionService(LinuxNativeBackend(), RefusingHost()),
    )
    try:
        result = _call(tools, workspace, f"curl -fsS --max-time 8 {_ORIGIN}/", [_ORIGIN])
        assert result["ok"], result
        assert result["result"]["boundary"] == "sandbox"
        assert interaction.prompt.call_count >= 1
    finally:
        tools.close()


def test_owned_https_connect_uses_approved_opaque_tunnel(registry):
    """Open a pinned TLS tunnel to an owned HTTPS endpoint through public dispatch."""
    if not _HTTPS_ORIGIN:
        pytest.skip("Requires an owned isolated HTTPS fixture")
    tools, workspace, _manager = registry
    result = _call(tools, workspace, f"curl -kfsS --max-time 8 {_HTTPS_ORIGIN}/", [_HTTPS_ORIGIN])
    assert result["ok"], result
    assert result["result"]["boundary"] == "sandbox"
    assert result["result"]["stdout"]["content"] == "egress-https-ok\n"


def test_uv_sync_uses_the_owned_scoped_package_index(registry):
    """Sync a synthetic dependency through the mediated HTTP package index."""
    if not Path(_UV).is_file():
        pytest.skip("Requires a native uv client executable")
    tools, workspace, _manager = registry
    project = workspace / "uv-project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "egress-client"\nversion = "0.1.0"\nrequires-python = ">=3.12"\n'
        'dependencies = ["loop-egress-fixture==1.0.0"]\n[tool.uv]\npackage = false\n',
        encoding="utf-8",
    )
    command = (
        f'UV_PROJECT_ENVIRONMENT="$TMPDIR/uv-env" {shlex.quote(_UV)} --no-cache sync '
        '--project "$WORKSPACE_ROOT/uv-project" '
        f"--python /usr/bin/python3 --index-url {_ORIGIN}/simple "
        '&& "$TMPDIR/uv-env/bin/python" '
        '-c "import loop_egress_fixture; print(loop_egress_fixture.VALUE)"'
    )
    result = _call(
        tools, workspace, command, [_ORIGIN], read_only=False, read_roots=[str(Path(_UV).parent)]
    )
    assert result["ok"], json.dumps(result, indent=2)
    assert result["result"]["stdout"]["content"] == "ok\n"


def test_git_ls_remote_uses_the_owned_scoped_https_repository(registry):
    """Read an owned Git advertisement through the approved TLS CONNECT target."""
    if not _HTTPS_ORIGIN:
        pytest.skip("Requires an owned isolated HTTPS Git fixture")
    tools, workspace, _manager = registry
    result = _call(
        tools,
        workspace,
        f"GIT_SSL_NO_VERIFY=true git ls-remote {_HTTPS_ORIGIN}/repo.git HEAD",
        [_HTTPS_ORIGIN],
        read_only=False,
    )
    assert result["ok"], result
    assert result["result"]["stdout"]["content"].endswith("\tHEAD\n")


@pytest.mark.parametrize("client", ["uv", "git"])
def test_ungranted_package_or_git_destination_is_denied(registry, client):
    """An approved origin cannot be reused for a different package or Git server."""
    if client == "git" and not _HTTPS_ORIGIN:
        pytest.skip("Requires an owned isolated HTTPS Git fixture")
    tools, workspace, _manager = registry
    if client == "uv":
        if not Path(_UV).is_file():
            pytest.skip("Requires a native uv client executable")
        project = workspace / "denied-uv-project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[project]\nname = "egress-denied"\nversion = "0.1.0"\n'
            'requires-python = ">=3.12"\ndependencies = ["loop-egress-fixture==1.0.0"]\n'
            "[tool.uv]\npackage = false\n",
            encoding="utf-8",
        )
        command = (
            f'{shlex.quote(_UV)} --no-cache sync --project "$WORKSPACE_ROOT/denied-uv-project" '
            f"--python /usr/bin/python3 --index-url http://11.0.0.3:18080/simple"
        )
        grants = [_ORIGIN]
    else:
        command = "GIT_SSL_NO_VERIFY=true git ls-remote https://11.0.0.3:18443/repo.git HEAD"
        grants = [_HTTPS_ORIGIN]
    result = _call(
        tools,
        workspace,
        command,
        grants,
        read_only=False,
        read_roots=[str(Path(_UV).parent)] if client == "uv" else None,
    )
    assert not result["ok"], result
    assert result["problem"]["code"] == "process.nonzero_exit"


def test_mixed_dns_answers_pin_only_the_allowed_address(registry):
    """Skip a private first answer and connect to the owned public-classified endpoint."""
    if not _MIXED_DNS_ORIGIN:
        pytest.skip("Requires an owned mixed-address DNS fixture")
    tools, workspace, _manager = registry
    result = _call(
        tools, workspace, f"curl -fsS --max-time 8 {_MIXED_DNS_ORIGIN}/", [_MIXED_DNS_ORIGIN]
    )
    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "egress-fixture-ok\n"


def test_dns_with_no_allowed_answer_is_denied(registry):
    """Return a proxy denial when every resolver result is a private host address."""
    if not _PRIVATE_DNS_ORIGIN:
        pytest.skip("Requires an owned private-address DNS fixture")
    tools, workspace, _manager = registry
    result = _call(
        tools,
        workspace,
        f"curl -sS --max-time 4 -o /dev/null -w '%{{http_code}}' {_PRIVATE_DNS_ORIGIN}/",
        [_PRIVATE_DNS_ORIGIN],
    )
    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "403"


def test_dns_rebinding_rechecks_each_new_connection(registry):
    """A public first answer cannot authorize a later private resolver answer."""
    if _REBIND_TEST != "disposable-linux-host" or not _REBIND_DNS_ORIGIN or not _PRIVATE_IP:
        pytest.skip("Requires an owned disposable resolver-rebinding fixture")
    tools, workspace, _manager = registry
    script = Path(__file__).with_name("egress_fixture_setup.sh")
    environment = {**os.environ, "LOOP_NATIVE_EGRESS_PRIVATE_IP": _PRIVATE_IP}
    subprocess.run(["sh", str(script), "rebind-public"], check=True, env=environment)
    try:
        first = _call(
            tools,
            workspace,
            f"curl -fsS --max-time 8 {_REBIND_DNS_ORIGIN}/",
            [_REBIND_DNS_ORIGIN],
        )
        assert first["ok"], first
        subprocess.run(["sh", str(script), "rebind-private"], check=True, env=environment)
        second = _call(
            tools,
            workspace,
            f"curl -sS --max-time 4 -o /dev/null -w '%{{http_code}}' {_REBIND_DNS_ORIGIN}/",
            [_REBIND_DNS_ORIGIN],
        )
        assert second["ok"], second
        assert second["result"]["stdout"]["content"] == "403"
    finally:
        subprocess.run(["sh", str(script), "rebind-public"], check=True, env=environment)


def test_plaintext_redirect_is_checked_again_at_its_visible_target(registry):
    """Follow a same-origin HTTP redirect through a second validated proxy request."""
    tools, workspace, _manager = registry
    result = _call(tools, workspace, f"curl -fsSL --max-time 8 {_ORIGIN}/redirect", [_ORIGIN])
    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "redirect-ok\n"


def test_https_redirect_to_new_unapproved_tunnel_is_denied(registry):
    """Recheck each new CONNECT destination when the client follows an opaque redirect."""
    if not _HTTPS_ORIGIN:
        pytest.skip("Requires an owned isolated HTTPS redirect fixture")
    tools, workspace, _manager = registry
    result = _call(
        tools,
        workspace,
        f"curl -kfsSL --max-time 8 {_HTTPS_ORIGIN}/redirect-forbidden",
        [_HTTPS_ORIGIN],
    )
    assert not result["ok"], result
    assert result["problem"]["code"] == "process.nonzero_exit"


@pytest.mark.parametrize(
    ("target", "grants", "at_policy"),
    [
        ("http://11.0.0.1:18080", ["http://11.0.0.1:18080"], False),
        ("http://192.168.64.29:18080", ["http://192.168.64.29:18080"], True),
        ("http://169.254.1.1:18080", ["http://169.254.1.1:18080"], True),
    ],
)
def test_forbidden_route_is_denied_even_with_origin_approval(registry, target, grants, at_policy):
    """Reject host-local, private, and link-local targets at their effective policy boundary."""
    tools, workspace, _manager = registry
    result = _call(
        tools,
        workspace,
        f"curl -sS --max-time 4 -o /dev/null -w '%{{http_code}}' {shlex.quote(target)}",
        grants,
    )
    if at_policy:
        assert result["problem"]["code"] == "tool.denied"
    else:
        assert result["ok"], result
        assert result["result"]["stdout"]["content"] == "403"


def test_direct_socket_cannot_reach_approved_external_target(registry):
    """Keep the payload network namespace isolated even after approving mediation."""
    tools, workspace, _manager = registry
    result = _call(
        tools,
        workspace,
        f"curl --noproxy '*' -fsS --connect-timeout 2 --max-time 4 {_ORIGIN}/",
        [_ORIGIN],
    )
    assert not result["ok"], result
    assert result["problem"]["code"] == "process.nonzero_exit"


def test_direct_udp_dns_and_unapproved_proxy_stay_unreachable(registry):
    """A mediated HTTP grant leaves direct UDP and unrelated proxy endpoints isolated."""
    tools, workspace, _manager = registry
    command = (
        "python3 -c 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); "
        's.sendto(b"probe", ("11.0.0.2", 53))\''
    )
    udp = _call(tools, workspace, command, [_ORIGIN])
    assert not udp["ok"], udp
    assert udp["problem"]["code"] == "process.nonzero_exit"
    proxy = _call(
        tools,
        workspace,
        f"curl -fsS --max-time 2 -x http://127.0.0.1:1 {_ORIGIN}/",
        [_ORIGIN],
    )
    assert not proxy["ok"], proxy
    assert proxy["problem"]["code"] == "process.nonzero_exit"


def test_inherited_host_socket_is_closed_before_mediated_payload(registry):
    """A parent-owned socket does not become an alternate payload egress path."""
    tools, workspace, _manager = registry
    parent, peer = socket.socketpair()
    try:
        descriptor = fcntl.fcntl(parent.fileno(), fcntl.F_DUPFD, 100)
        try:
            os.set_inheritable(descriptor, True)
            result = _call(
                tools,
                workspace,
                f"test ! -e /proc/self/fd/{descriptor} && curl -fsS --max-time 8 {_ORIGIN}/",
                [_ORIGIN],
            )
            assert result["ok"], result
            assert result["result"]["stdout"]["content"] == "egress-fixture-ok\n"
        finally:
            os.close(descriptor)
    finally:
        parent.close()
        peer.close()


@pytest.mark.parametrize(
    "command",
    [
        "curl --noproxy '*' -fsS --connect-timeout 2 --max-time 4 http://127.0.0.1:18080/",
        "curl --noproxy '*' -g -fsS --connect-timeout 2 --max-time 4 'http://[2001:db8::2]:18080/'",
        "curl --noproxy '*' -fsS --connect-timeout 2 --max-time 4 http://11.0.0.2:18080/",
    ],
)
def test_direct_ipv4_ipv6_and_host_loopback_stay_unreachable(registry, command):
    """Proxy approval never grants the workload direct IP or host-loopback access."""
    tools, workspace, _manager = registry
    result = _call(tools, workspace, command, [_ORIGIN])
    assert not result["ok"], result
    assert result["problem"]["code"] == "process.nonzero_exit"


def test_simultaneous_attempts_keep_different_origin_grants(registry):
    """An active public HTTP grant does not open a concurrent HTTPS attempt's proxy."""
    if not _HTTPS_ORIGIN:
        pytest.skip("Requires an owned isolated HTTPS fixture")
    tools, workspace, _manager = registry
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            _call, tools, workspace, f"curl -fsS --max-time 8 {_ORIGIN}/", [_ORIGIN]
        )
        second = executor.submit(
            _call,
            tools,
            workspace,
            f"curl -kfsS --max-time 8 {_HTTPS_ORIGIN}/",
            [_HTTPS_ORIGIN],
        )
        assert first.result(timeout=12)["result"]["stdout"]["content"] == "egress-fixture-ok\n"
        assert second.result(timeout=12)["result"]["stdout"]["content"] == "egress-https-ok\n"


def test_other_managers_persisted_change_waits_for_active_reload(registry):
    """A separate policy write leaves the first manager's tunnel active until reload."""
    if not _HTTPS_ORIGIN:
        pytest.skip("Requires an owned isolated streaming HTTPS fixture")
    tools, workspace, manager = registry
    stream = workspace / "persisted-stream"
    command = (
        f"curl -kfsS --no-buffer --max-time 12 {_HTTPS_ORIGIN}/slow "
        '> "$WORKSPACE_ROOT/persisted-stream"'
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            _call, tools, workspace, command, [_HTTPS_ORIGIN], read_only=False
        )
        deadline = time.monotonic() + 5
        while (not stream.exists() or "started" not in stream.read_text()) and (
            time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert stream.exists() and "started" in stream.read_text()
        other = PermissionManager(workspace)
        other.set_default(Action.FILESYSTEM_READ, Decision.DENY)
        other.close()
        assert running.result(timeout=12)["ok"]
        assert "finished" in stream.read_text()

    stream.unlink()
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            _call, tools, workspace, command, [_HTTPS_ORIGIN], read_only=False
        )
        deadline = time.monotonic() + 5
        while (not stream.exists() or "started" not in stream.read_text()) and (
            time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert stream.exists() and "started" in stream.read_text()
        manager.reload()
        assert not running.result(timeout=5)["ok"]
        assert "finished" not in stream.read_text()


def test_direct_policy_edit_waits_for_active_reload(registry):
    """A changed policy file takes effect only after this manager accepts its revision."""
    if not _HTTPS_ORIGIN:
        pytest.skip("Requires an owned isolated streaming HTTPS fixture")
    tools, workspace, manager = registry
    manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    stream = workspace / "edited-stream"
    command = (
        f"curl -kfsS --no-buffer --max-time 12 {_HTTPS_ORIGIN}/slow "
        '> "$WORKSPACE_ROOT/edited-stream"'
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            _call, tools, workspace, command, [_HTTPS_ORIGIN], read_only=False
        )
        deadline = time.monotonic() + 5
        while (not stream.exists() or "started" not in stream.read_text()) and (
            time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert stream.exists() and "started" in stream.read_text()
        policy_file = workspace / ".loop" / "permissions.yaml"
        policy_file.write_text(policy_file.read_text() + "\n# synthetic external edit\n")
        assert running.result(timeout=12)["ok"]

    stream.unlink()
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            _call, tools, workspace, command, [_HTTPS_ORIGIN], read_only=False
        )
        deadline = time.monotonic() + 5
        while (not stream.exists() or "started" not in stream.read_text()) and (
            time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert stream.exists() and "started" in stream.read_text()
        manager.reload()
        assert not running.result(timeout=5)["ok"]
        assert "finished" not in stream.read_text()


def test_copied_proxy_url_cannot_cross_sandbox_attempts(registry):
    """A second private network namespace cannot use a concurrent attempt's proxy URL."""
    tools, workspace, _manager = registry
    copied = workspace / "proxy-url"
    source = 'printf %s "$http_proxy" > "$WORKSPACE_ROOT/proxy-url"; sleep 4'
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(_call, tools, workspace, source, [_ORIGIN], read_only=False)
        deadline = time.monotonic() + 3
        while not copied.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert copied.exists()
        second = _call(
            tools,
            workspace,
            f'curl -fsS --max-time 2 -x "$(cat {shlex.quote(str(copied))})" {_ORIGIN}/',
            [],
        )
        assert not second["ok"], second
        assert second["problem"]["code"] == "process.nonzero_exit"
        assert first.result(timeout=10)["ok"]


def test_expired_attempt_proxy_url_cannot_reopen_egress(registry):
    """An old attempt's loopback address has no authority in a later namespace."""
    tools, workspace, _manager = registry
    copied = workspace / "expired-proxy-url"
    first = _call(
        tools,
        workspace,
        'printf %s "$http_proxy" > "$WORKSPACE_ROOT/expired-proxy-url"',
        [_ORIGIN],
        read_only=False,
    )
    assert first["ok"], first
    second = _call(
        tools,
        workspace,
        f'curl -fsS --max-time 2 -x "$(cat {shlex.quote(str(copied))})" {_ORIGIN}/',
        [],
    )
    assert not second["ok"], second
    assert second["problem"]["code"] == "process.nonzero_exit"


def test_representative_parallel_load_keeps_healthy_leases(registry):
    """Six active streams survive heartbeat and unchanged-policy pause/resume."""
    if not _HTTPS_ORIGIN:
        pytest.skip("Requires an owned isolated streaming HTTPS fixture")
    tools, workspace, manager = registry
    with ThreadPoolExecutor(max_workers=6) as executor:
        running = [
            executor.submit(
                _call,
                tools,
                workspace,
                f"curl -kfsS --no-buffer --max-time 15 {_HTTPS_ORIGIN}/slow",
                [_HTTPS_ORIGIN],
            )
            for _ in range(6)
        ]
        deadline = time.monotonic() + 6
        while len(manager._egress_revokers) < 6 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(manager._egress_revokers) == 6
        manager.set_yolo(True)
        for future in running:
            outcome = future.result(timeout=15)
            assert outcome["ok"], outcome
            assert outcome["result"]["stdout"]["content"] == "started\nfinished\n"


def test_revision_ack_timeout_does_not_report_success(registry, monkeypatch):
    """A missing exact revision acknowledgement never reports confirmed closure."""
    if not _HTTPS_ORIGIN:
        pytest.skip("Requires an owned isolated streaming HTTPS fixture")
    tools, workspace, manager = registry
    stream = workspace / "unacknowledged-stream"
    monkeypatch.setattr(EgressLease, "revision_ack", lambda _self, _revision, _confirmed: None)
    command = (
        f"curl -kfsS --no-buffer --max-time 12 {_HTTPS_ORIGIN}/slow "
        '> "$WORKSPACE_ROOT/unacknowledged-stream"'
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            _call, tools, workspace, command, [_HTTPS_ORIGIN], read_only=False
        )
        deadline = time.monotonic() + 5
        while (not stream.exists() or "started" not in stream.read_text()) and (
            time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert stream.exists() and "started" in stream.read_text()
        start = time.monotonic()
        with pytest.raises(OSError, match="transition is uncertain"):
            manager.set_yolo(False)
        assert 1.9 <= time.monotonic() - start < 3.5
        outcome = running.result(timeout=5)
        assert not outcome["ok"], outcome
        assert "finished" not in stream.read_text()


@pytest.mark.parametrize(("pause_seconds", "survives"), [(0.7, True), (2.7, False)])
def test_authority_process_suspend_and_resume(registry, pause_seconds, survives):
    """Short scheduler stalls preserve a lease; a missed lease closes it independently."""
    if _SUSPEND_TEST != "disposable-linux-host":
        pytest.skip("Requires an explicitly disposable process-suspension test host")
    if not _HTTPS_ORIGIN:
        pytest.skip("Requires an owned isolated streaming HTTPS fixture")
    tools, workspace, _manager = registry
    stream = workspace / "suspended-stream"
    command = (
        f"curl -kfsS --no-buffer --max-time 12 {_HTTPS_ORIGIN}/slow "
        '> "$WORKSPACE_ROOT/suspended-stream"'
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            _call, tools, workspace, command, [_HTTPS_ORIGIN], read_only=False
        )
        deadline = time.monotonic() + 5
        while (not stream.exists() or "started" not in stream.read_text()) and (
            time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert stream.exists() and "started" in stream.read_text()
        wake = subprocess.Popen(
            ["/bin/sh", "-c", f"sleep {pause_seconds}; kill -CONT {os.getpid()}"],
            start_new_session=True,
        )
        os.kill(os.getpid(), signal.SIGSTOP)
        assert wake.wait(timeout=3) == 0
        outcome = running.result(timeout=12)
        assert outcome["ok"] is survives, outcome
        assert ("finished" in stream.read_text()) is survives


def test_policy_change_closes_an_active_https_tunnel(registry):
    """A local active-policy revision closes a streaming tunnel before reporting success."""
    if not _HTTPS_ORIGIN:
        pytest.skip("Requires an owned isolated streaming HTTPS fixture")
    tools, workspace, manager = registry
    stream = workspace / "stream"
    command = (
        f'curl -kfsS --no-buffer --max-time 12 {_HTTPS_ORIGIN}/slow > "$WORKSPACE_ROOT/stream"'
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            _call, tools, workspace, command, [_HTTPS_ORIGIN], read_only=False
        )
        deadline = time.monotonic() + 5
        while (not stream.exists() or "started" not in stream.read_text()) and (
            time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert stream.exists() and "started" in stream.read_text()
        start = time.monotonic()
        manager.set_yolo(False)
        assert time.monotonic() - start < 3
        outcome = running.result(timeout=5)
        assert not outcome["ok"], outcome
        assert "finished" not in stream.read_text()


def test_policy_change_closes_two_active_tunnels_without_deadlock(registry):
    """Concurrent revision acknowledgements cannot wait on the owner's policy lock."""
    if not _HTTPS_ORIGIN:
        pytest.skip("Requires an owned isolated streaming HTTPS fixture")
    tools, workspace, manager = registry
    streams = [workspace / "parallel-stream-one", workspace / "parallel-stream-two"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        running = [
            executor.submit(
                _call,
                tools,
                workspace,
                f'curl -kfsS --no-buffer --max-time 12 {_HTTPS_ORIGIN}/slow > "$WORKSPACE_ROOT/{stream.name}"',
                [_HTTPS_ORIGIN],
                read_only=False,
            )
            for stream in streams
        ]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not all(
            stream.exists() and "started" in stream.read_text() for stream in streams
        ):
            time.sleep(0.02)
        assert all(stream.exists() and "started" in stream.read_text() for stream in streams)
        start = time.monotonic()
        manager.set_yolo(False)
        assert time.monotonic() - start < 3
        for future, stream in zip(running, streams):
            assert not future.result(timeout=5)["ok"]
            assert "finished" not in stream.read_text()


def test_missed_authority_heartbeat_closes_an_active_tunnel(registry, monkeypatch):
    """The protected receiver expires an abandoned authority lease independently."""
    if not _HTTPS_ORIGIN:
        pytest.skip("Requires an owned isolated streaming HTTPS fixture")
    tools, workspace, _manager = registry
    stream = workspace / "heartbeat-stream"
    monkeypatch.setattr(EgressLease, "_watch", lambda _self: None)
    command = (
        f"curl -kfsS --no-buffer --max-time 12 {_HTTPS_ORIGIN}/slow "
        '> "$WORKSPACE_ROOT/heartbeat-stream"'
    )
    started = time.monotonic()
    outcome = _call(tools, workspace, command, [_HTTPS_ORIGIN], read_only=False)
    assert time.monotonic() - started < 5
    assert not outcome["ok"], outcome
    assert stream.exists() and "started" in stream.read_text()
    assert "finished" not in stream.read_text()


def test_receiver_death_stops_an_existing_tunnel(registry):
    """A killed protected supervisor cannot leave its independent mediator connected."""
    if _SUPERVISOR_TEST != "disposable-linux-host":
        pytest.skip("Requires an explicitly disposable Linux supervisor test host")
    if not _HTTPS_ORIGIN:
        pytest.skip("Requires an owned isolated streaming HTTPS fixture")
    tools, workspace, _manager = registry
    stream = workspace / "supervisor-stream"
    command = (
        f"curl -kfsS --no-buffer --max-time 12 {_HTTPS_ORIGIN}/slow "
        '> "$WORKSPACE_ROOT/supervisor-stream"'
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            _call, tools, workspace, command, [_HTTPS_ORIGIN], read_only=False
        )
        deadline = time.monotonic() + 5
        while (not stream.exists() or "started" not in stream.read_text()) and (
            time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert stream.exists() and "started" in stream.read_text()
        listed = subprocess.run(
            [
                "systemctl",
                "list-units",
                "--type=service",
                "--state=running",
                "--plain",
                "--no-legend",
                "loopgen_native_*@*.service",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        units = [line.split()[0] for line in listed.stdout.splitlines() if line.strip()]
        assert len(units) == 1, units
        try:
            killed_at = time.monotonic()
            subprocess.run(
                ["sudo", "-n", "systemctl", "kill", "--kill-whom=main", "--signal=KILL", units[0]],
                capture_output=True,
                text=True,
                check=True,
            )
            outcome = running.result(timeout=5)
            assert time.monotonic() - killed_at < 2.5
            assert not outcome["ok"], outcome
            assert "finished" not in stream.read_text()
        finally:
            subprocess.run(
                ["sudo", "-n", "systemctl", "reset-failed", units[0]],
                capture_output=True,
                text=True,
                check=True,
            )
