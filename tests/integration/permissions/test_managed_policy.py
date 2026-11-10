"""Check the actual platform administrator-policy trust anchor and optional installation."""

import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from loop import (
    Action,
    Decision,
    Operation,
    PermissionManager,
    ProcessBoundary,
    ProcessTarget,
    ToolRegistry,
)
from loop.commands import CommandManager
from loop.configuration import ConfigurationCommands, ConfigurationManager
from loop.execution import CommandExecutionService
from loop.execution.sandbox.linux.backend import LinuxNativeBackend
from loop.interaction import ConsoleInteraction
from loop.permissions.managed import ManagedPolicyStore
from loop.permissions.models import SessionTarget
from loop.tools.system import run_command
from loop.utils.path import has_extended_access_acl

_SYSTEM_DIRECTORIES = {
    "darwin": Path("/Library/Loop"),
    "linux": Path("/etc/loop"),
}
_NATIVE_IDENTITY = "0123456789abcdef0123456789abcdef"
_NATIVE_POLICY_OPT_IN = os.environ.get("LOOP_NATIVE_MANAGED_POLICY_TEST")


@pytest.mark.skipif(sys.platform not in _SYSTEM_DIRECTORIES, reason="No system policy anchor")
def test_actual_platform_anchor_is_root_protected_and_optional():
    """The fixed system ancestry is root controlled even when no policy is installed."""
    store = ManagedPolicyStore()
    assert store.directory == _SYSTEM_DIRECTORIES[sys.platform]
    for path in (*reversed(store.directory.parent.parents), store.directory.parent):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            details = os.fstat(descriptor)
            assert details.st_uid == 0
            assert not details.st_mode & 0o022
            assert not has_extended_access_acl(descriptor)
        finally:
            os.close(descriptor)
    if not os.path.lexists(store.directory) and not os.path.lexists(store.expectation_path):
        assert store.load() is None


@pytest.mark.skipif(
    sys.platform not in _SYSTEM_DIRECTORIES or _NATIVE_POLICY_OPT_IN != "owned-root-fixture",
    reason="Exact owned root policy fixture must be provisioned separately",
)
def test_native_root_policy_prohibits_yolo_activation(tmp_path):
    """A root-owned policy blocks activation without the test modifying its source."""
    store = ManagedPolicyStore()
    assert store.directory == _SYSTEM_DIRECTORIES[sys.platform]
    snapshot = store.load()
    assert snapshot is not None
    assert snapshot.policy.identity == _NATIVE_IDENTITY
    assert snapshot.policy.revision == 1
    assert snapshot.policy.forbid_yolo is True
    managed_files = (store.expectation_path, store.directory / "managed-policy.json")
    original = tuple(path.read_bytes() for path in managed_files)
    for path in (store.directory, *managed_files):
        details = path.stat()
        assert details.st_uid == 0
        assert stat.S_IMODE(details.st_mode) == (0o755 if path.is_dir() else 0o644)
    for path in managed_files:
        probe = subprocess.run(
            [sys.executable, "-c", "import os,sys; os.open(sys.argv[1],os.O_WRONLY)", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert probe.returncode != 0
        assert "PermissionError" in probe.stderr

    saved = ConfigurationManager(tmp_path / "saved" / "config.toml")
    saved.initialize()
    saved.load({})
    saved.set("app.yolo", True, scope="user")
    startup = PermissionManager(tmp_path, yolo=saved.effective.app.yolo)
    assert startup.yolo is False
    startup.require_policy_ready()
    assert startup.confirmation_decision() is None

    configuration = ConfigurationManager(tmp_path / "current" / "config.toml")
    configuration.initialize()
    configuration.load({})
    before = configuration.path.read_bytes()
    interaction = Mock(spec=ConsoleInteraction)
    permissions = PermissionManager(tmp_path)

    def apply(path, settings):
        """Apply the configuration through the real permission manager."""
        assert path == "app.yolo"
        permissions.set_yolo(settings.app.yolo)
        return "applied now"

    provider = ConfigurationCommands(
        configuration,
        apply,
        validate_yolo=permissions.validate_yolo_activation,
        yolo_active=lambda: permissions.yolo,
        yolo_gate=permissions.suspend_yolo_decisions,
    )
    commands = CommandManager(providers=(provider,), interaction=interaction)
    try:
        for scope in ("session", "user"):
            assert commands.handle_user_command(f"/config set app.yolo true scope={scope}")
            assert "prohibits YOLO" in interaction.report.call_args.args[0].detail
            assert not permissions.yolo
            assert not configuration.effective.app.yolo
            assert not configuration.saved_yolo
            assert configuration.path.read_bytes() == before
        interaction.prompt.assert_not_called()
        with pytest.raises(ValueError, match="prohibits YOLO"):
            permissions.set_yolo(True)

    finally:
        commands.close()
        permissions.close()
        startup.close()
    assert tuple(path.read_bytes() for path in managed_files) == original
    assert store.load() == snapshot


@pytest.mark.skipif(sys.platform != "linux", reason="Disposable Linux root fixture required")
def test_native_root_policy_is_authoritative_for_yolo_and_host(tmp_path):
    """A provisioned root-owned policy denies user and YOLO requests at the real source."""
    if _NATIVE_POLICY_OPT_IN != "disposable-linux-host":
        pytest.skip("Root-owned disposable policy fixture not explicitly provisioned")
    store = ManagedPolicyStore()
    snapshot = store.load()
    assert snapshot is not None
    assert snapshot.policy.identity == _NATIVE_IDENTITY
    assert snapshot.policy.revision == 1
    assert snapshot.policy.forbid_host
    assert snapshot.policy.deny_local_networks
    assert any(rule.id == "native-deny-session" for rule in snapshot.policy.rules)
    assert any(rule.id == "native-deny-egress" for rule in snapshot.policy.rules)
    for path in (store.expectation_path, store.directory / "managed-policy.json"):
        details = path.stat()
        assert details.st_uid == 0
        assert stat.S_IMODE(details.st_mode) == 0o644
    write_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sys; os.open(sys.argv[1], os.O_WRONLY)",
            str(store.directory / "managed-policy.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert write_probe.returncode != 0
    assert "PermissionError" in write_probe.stderr

    manager = PermissionManager(tmp_path, yolo=True)
    session = (
        Operation(
            tool_id="test",
            action=Action.SESSION_MUTATE,
            target=SessionTarget(identifier="native"),
        ),
    )
    host = (
        Operation(
            tool_id="request_host_command",
            action=Action.PROCESS_EXECUTE,
            target=ProcessTarget(
                argv=("/bin/sh", "-c", "printf native"),
                cwd=str(tmp_path),
                boundary=ProcessBoundary.HOST,
            ),
        ),
    )
    assert manager.evaluate(session).sources == ("managed:deny:native-deny-session",)
    assert manager.authorize(session).decision is Decision.DENY
    assert manager.evaluate(host).sources == ("managed:host_forbidden",)
    assert "active identity=" in manager.describe("managed")

    class RefusingHost:
        """Fail if a denied mediated request incorrectly crosses to host execution."""

        def run_host_command(self, request):
            """Reject unrequested host execution."""
            raise AssertionError("Managed egress denial offered host execution.")

    registry = ToolRegistry(
        [run_command],
        permission_manager=manager,
        execution_service=CommandExecutionService(LinuxNativeBackend(), RefusingHost()),
    )
    try:
        result = json.loads(
            registry.call(
                "run_command",
                json.dumps(
                    {
                        "command": "printf managed-deny",
                        "cwd": str(tmp_path),
                        "egress_origins": ["http://11.0.0.2:18080"],
                    }
                ),
            )
        )
        assert not result["ok"], result
        assert result["problem"]["code"] == "tool.denied"
        local = json.loads(
            registry.call(
                "run_command",
                json.dumps(
                    {
                        "command": "/usr/bin/python3 -I -c "
                        + shlex.quote(
                            "import errno,socket; "
                            "exec('try:\\n socket.socket()\\nexcept OSError as error:\\n "
                            "print(error.errno)')"
                        ),
                        "cwd": str(tmp_path),
                        "read_only": True,
                    }
                ),
            )
        )
        assert local["ok"], local
        assert local["result"]["stdout"]["content"] == "1\n"
    finally:
        registry.close()
