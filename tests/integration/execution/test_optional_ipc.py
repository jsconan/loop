"""Observe real optional macOS IPC resources through the public command tool."""

import json
import os
import plistlib
import secrets
import shlex
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop import Action, Decision, PermissionManager, ToolRegistry, constants
from loop.execution import CommandExecutionService
from loop.execution.sandbox.macos.backend import MacOSSeatbeltBackend
from loop.instructions import InstructionsManager, RuntimeEnvironment
from loop.tools.system import cancel_command_job, poll_command_job, run_command
from tests.scheduling import await_condition

pytestmark = [pytest.mark.integration, pytest.mark.macos]
_NATIVE_ENABLED = os.environ.get("LOOP_NATIVE_MACOS_TEST") == "owned-volume"


def attachments():
    """Snapshot disk images without modifying unrelated mounted filesystems."""
    result = subprocess.run(
        ["/usr/bin/hdiutil", "info", "-plist"], capture_output=True, check=True, timeout=10
    )
    return {
        (
            item["image-path"],
            tuple(entity.get("dev-entry", "") for entity in item.get("system-entities", [])),
        )
        for item in plistlib.loads(result.stdout).get("images", [])
    }


@pytest.fixture
def optional_ipc_command(native_directory, native_mount_point, monkeypatch):
    """Own a fresh actual application boundary without a shared volume fixture."""
    if sys.platform != "darwin" or not _NATIVE_ENABLED:
        pytest.skip("Requires macOS with LOOP_NATIVE_MACOS_TEST=owned-volume")
    monkeypatch.setattr(MacOSSeatbeltBackend, "installed_read_roots", staticmethod(lambda: ()))
    workspace = native_directory("/private/tmp")
    token_hex = secrets.token_hex

    def owned_token(size):
        """Register each freshly generated native mount name without changing its randomness."""
        value = token_hex(size)
        if size == 4:
            native_mount_point(
                Path("/Volumes") / f"{constants.APP_DISPLAY_NAME}IPC-{os.getpid()}-{value}"
            )
        return value

    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.secrets", SimpleNamespace(token_hex=owned_token)
    )
    before = attachments()
    service = CommandExecutionService.for_host()
    assert attachments() == before
    manager = PermissionManager(workspace)
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW)
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    registry = ToolRegistry(
        [run_command, cancel_command_job, poll_command_job],
        permission_manager=manager,
        execution_service=service,
    )
    scratch = manager.temporary_directory
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, scratch))

    def invoke(name, **arguments):
        """Invoke a public command tool in the real application boundary."""
        return json.loads(
            registry.call(
                name,
                json.dumps(arguments),
                interaction=interaction,
                instructions_manager=instructions,
            )
        )

    def call(source, **arguments):
        """Execute one public command with the real native service and owned scratch."""
        return invoke("run_command", command=source, cwd="$WORKSPACE_ROOT", **arguments)

    try:
        yield call, manager, interaction, scratch, before, invoke
    finally:
        registry.close()
        manager.close()
        assert attachments() == before


def test_offline_and_denied_ipc_create_no_attachment(optional_ipc_command):
    """Selection and offline execution stay mount-free, and approval precedes IPC setup."""
    call, manager, interaction, scratch, before, _ = optional_ipc_command
    result = call("printf offline-ok")
    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "offline-ok"
    assert attachments() == before
    combined = call('printf forbidden > "$TMPDIR/launched"', local_ipc=True, network=True)
    assert not combined["ok"], combined
    assert combined["problem"]["code"] == "sandbox.unsupported", combined
    assert not (scratch / "launched").exists()
    interaction.prompt.assert_not_called()
    assert attachments() == before
    manager.set_default(Action.NETWORK_LOCAL, Decision.DENY)
    refused = call("printf forbidden", local_ipc=True)
    assert not refused["ok"]
    interaction.prompt.assert_not_called()
    assert attachments() == before
    manager.set_default(Action.NETWORK_LOCAL, Decision.ASK)

    def approve(*_args, **_kwargs):
        """Observe that no native socket resource exists while the decision is pending."""
        assert attachments() == before
        return "approve"

    interaction.prompt.side_effect = approve
    source = "/usr/bin/perl -MSocket -e " + shlex.quote(
        'my $p="$ENV{LOOP_LOCAL_SOCKET_DIR}/s"; '
        "socket(my $s,PF_UNIX,SOCK_STREAM,0) or die $!; "
        "bind($s,pack_sockaddr_un($p)) or die $!; listen($s,1) or die $!; "
        "socket(my $c,PF_UNIX,SOCK_STREAM,0) or die $!; "
        "connect($c,pack_sockaddr_un($p)) or die $!; "
        'defined accept(my $a,$s) or die $!; print "local-ok";'
    )
    result = call(source, local_ipc=True)
    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "local-ok"
    interaction.prompt.assert_called_once()
    assert attachments() == before


def test_overlapping_ipc_attempts_share_and_release_one_attachment(optional_ipc_command):
    """Concurrent public attempts share one checked volume until the final lease ends."""
    call, _, _, scratch, before, _ = optional_ipc_command
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                call,
                f'printf ready > "$TMPDIR/ready{index}"; '
                f'while test ! -f "$TMPDIR/stop{index}"; do /bin/sleep 0.05; done; printf done',
                local_ipc=True,
                wait_seconds=15,
            )
            for index in range(2)
        ]
        try:
            await_condition(lambda: all((scratch / f"ready{i}").exists() for i in range(2)))
            assert len(attachments() - before) == 1
            (scratch / "stop0").touch()
            first = futures[0].result(timeout=10)
            assert first["ok"], first
            assert len(attachments() - before) == 1
        finally:
            for index in range(2):
                (scratch / f"stop{index}").touch()
        second = futures[1].result(timeout=10)
        assert second["ok"], second
    assert attachments() == before


def test_synchronization_endpoint_is_scoped_and_retires_on_cancel(optional_ipc_command):
    """A live attempt has one protected endpoint and cancellation closes connected peers."""
    call, _, _, scratch, before, invoke = optional_ipc_command
    source = (
        'printf "%s" "$LOOP_SYNC_SOCKET" > "$TMPDIR/endpoint"; '
        'if /bin/mv "$LOOP_SYNC_SOCKET" "$LOOP_LOCAL_SOCKET_DIR/replaced" 2>/dev/null; then '
        'printf vulnerable > "$TMPDIR/moved"; fi; '
        "/usr/bin/perl -MSocket -MIO::Handle -e "
        + shlex.quote(
            "socket(my $s,PF_UNIX,SOCK_STREAM,0) or die $!; "
            "connect($s,pack_sockaddr_un($ENV{LOOP_SYNC_SOCKET})) or die $!; "
            'print $s qq{["create","shared",1,2]\n}; '
            r'$s->flush; <$s> =~ /^\[0,"[0-9a-f]{32}"\]$/ or die "invalid create response"; '
            'open(my $f,">",$ENV{TMPDIR}."/ready") or die $!; close $f; '
            "sleep 60;"
        )
    )
    started = call(source, local_ipc=True, wait_seconds=0)
    assert started["ok"], started
    handle = started["result"]["handle"]
    try:
        await_condition(lambda: (scratch / "ready").exists())
        endpoint = Path((scratch / "endpoint").read_text())
        assert endpoint.is_socket()
        assert not (scratch / "moved").exists()
        outsider = "/usr/bin/perl -MSocket -e " + shlex.quote(
            "socket(my $s,PF_UNIX,SOCK_STREAM,0) or die $!; "
            "connect($s,pack_sockaddr_un($ARGV[0])) and exit 3; "
            "exit 0;"
        )
        other = call(f"{outsider} {shlex.quote(str(endpoint))}", local_ipc=True, wait_seconds=10)
        assert other["ok"] and other["result"]["exit_code"] == 0, other
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(15)
            connection.connect(str(endpoint))
            connection.sendall(b'["open","shared"]\n')
            with connection.makefile("rb") as stream:
                reply = json.loads(stream.readline())
                assert reply[0] == 0
                cancelled = invoke("cancel_command_job", handle=handle)
                assert cancelled["ok"], cancelled
                assert stream.readline() == b"", "the broker retained a connected peer"
        finished = invoke("poll_command_job", handle=handle, wait_seconds=15)
        assert finished["ok"] and finished["result"]["result"]["code"] == "sandbox.cancelled"
        assert not endpoint.exists()
    finally:
        invoke("cancel_command_job", handle=handle)
    assert attachments() == before


def test_spawn_client_reuses_released_counter_capacity(optional_ipc_command):
    """Stock spawned processes use only the explicitly supplied client binding."""
    call, _, _, _, _, _ = optional_ipc_command
    binding = Path(__file__).resolve().parents[1] / "macos/fixtures/synchronization"
    runtime = Path(sys.base_prefix).resolve(strict=True)
    interpreter = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    command = (
        f"PYTHONPATH={shlex.quote(str(binding))} {shlex.quote(str(interpreter))} -B "
        f"{shlex.quote(str(binding / 'stdlib_workload.py'))}"
    )
    result = call(command, local_ipc=True, read_roots=[str(binding), str(runtime)], wait_seconds=30)
    assert result["ok"] and result["result"]["exit_code"] == 0, result
    assert "512 slots, and raw semaphore denial passed." in result["result"]["stdout"]["content"]


def test_explicit_private_account_binding_keeps_native_lookup_denied(optional_ipc_command):
    """Python can retain private HOME locally while host account lookup remains offline."""
    call, _, _, _, _, _ = optional_ipc_command
    binding = Path(__file__).resolve().parents[1] / "macos/fixtures/synchronization"
    runtime = Path(sys.base_prefix).resolve(strict=True)
    interpreter = shlex.quote(str(Path(getattr(sys, "_base_executable", sys.executable)).resolve()))
    adapted = """
import os
from pathlib import Path
expected = Path(os.environ['HOME'])
os.environ.clear()
assert Path.home() == expected
print('private account')
"""
    native = """
import os
import pwd
try:
    pwd.getpwuid(os.getuid())
except KeyError:
    print('host account denied')
else:
    raise AssertionError('host account lookup unexpectedly succeeded')
"""
    command = (
        f"LOOP_TEST_PRIVATE_ACCOUNT=1 PYTHONPATH={shlex.quote(str(binding))} "
        f"{interpreter} -B -c {shlex.quote(adapted)} && "
        f"{interpreter} -IB -c {shlex.quote(native)}"
    )
    result = call(command, read_roots=[str(binding), str(runtime)], wait_seconds=30)
    assert result["ok"], result
    assert result["result"]["stdout"]["content"].splitlines() == [
        "private account",
        "host account denied",
    ]
