"""Exercise native commands with production-shaped protected application and session storage."""

import json
import os
import shlex
import sys
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from loop import SessionManager, SQLiteSessionStore
from loop.application.paths import ApplicationPaths
from loop.execution import CommandExecutionService
from loop.execution.sandbox import UnavailableSandboxBackend, select_sandbox_backend
from loop.execution.sandbox.macos.backend import MacOSSeatbeltBackend
from loop.instructions import InstructionsManager, RuntimeEnvironment
from loop.interaction import Interaction
from loop.permissions import Action, ApprovalChoice, Decision, PermissionManager, SandboxPathAccess
from loop.tooling import ToolRegistry
from loop.tools import BUILTIN_TOOLS
from loop.utils.yaml import dump_yaml, load_yaml
from loop.workspace import Workspace, WorkspaceRepository

pytestmark = pytest.mark.integration
_NATIVE_PLATFORM = "MACOS" if sys.platform == "darwin" else "LINUX"
_NATIVE_MODE = "owned-volume" if sys.platform == "darwin" else "installed"
_NATIVE_ENABLED = os.environ.get(f"LOOP_NATIVE_{_NATIVE_PLATFORM}_TEST") == _NATIVE_MODE


@pytest.fixture(params=[False, True])
def application_command(native_directory, request, monkeypatch):
    """Own real durable storage and public tools with ordinary ask policy and host disabled."""
    if not _NATIVE_ENABLED:
        pytest.skip(
            f"Set LOOP_NATIVE_{_NATIVE_PLATFORM}_TEST={_NATIVE_MODE} for native qualification"
        )
    root = native_directory("/private/tmp" if sys.platform == "darwin" else "/tmp")
    checkout = root / "checkout"
    checkout.mkdir()
    home = root / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    data = root / "Library/Application Support/loop"
    paths = ApplicationPaths(data, data, data)
    workspace = WorkspaceRepository(paths.workspace_catalog).initialize(
        Workspace.discover(checkout)
    )
    storage = paths.for_workspace(workspace.id, checkout)
    store = SQLiteSessionStore(
        storage.sessions, workspace_id=workspace.id, scratch_roots=storage.scratch_roots
    )
    sessions = SessionManager(session_store=store, workspace_id=workspace.id)
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "approve"
    manager = PermissionManager(
        checkout,
        configuration_path=storage.permissions,
        user_configuration_path=paths.user_permissions,
        protected_host_roots=(
            paths.configuration_root,
            paths.data_root,
            paths.state_root,
            paths.session_scratch,
        ),
        workspace_id=workspace.id,
        namespace_catalog_path=paths.workspace_catalog,
        bootstrap_pristine_policy=True,
        user_namespace_identity=str(paths.data_root),
    )
    manager.set_limit("deny_local_networks", True)
    manager.bind_temporary_directory(
        sessions.temporary_directory,
        ordinary_names=sessions.working_directory_allows_private_names
        if request.param
        else lambda: False,
    )
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(checkout, sessions.temporary_directory)
    )
    backend = MacOSSeatbeltBackend() if sys.platform == "darwin" else select_sandbox_backend()
    # Scope assertions in this fixture use only owned installations.
    monkeypatch.setattr(MacOSSeatbeltBackend, "installed_read_roots", staticmethod(lambda: ()))
    assert not isinstance(backend, UnavailableSandboxBackend), backend.capability_failure()
    host = Mock()
    host.run_host_command.side_effect = AssertionError("Unexpected host execution")
    registry = ToolRegistry(
        BUILTIN_TOOLS,
        permission_manager=manager,
        execution_service=CommandExecutionService(backend, host),
        session_manager=sessions,
    )

    def call(name, **arguments):
        """Dispatch through the real virtual-path and command permission boundaries."""
        return json.loads(
            registry.call(
                name,
                json.dumps(arguments),
                interaction=interaction,
                instructions_manager=instructions,
            )
        )

    try:
        yield paths, manager, interaction, call
    finally:
        registry.close()
        manager.close()


def test_native_enrolled_workspace_runs_ordinary_commands_without_unneeded_approval(
    application_command,
):
    """An enrolled workspace with no shared dependencies respects automatic process approval."""
    _, manager, interaction, call = application_command
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW)

    result = call("run_command", command="printf ordinary", write_roots="scratch")

    assert result["ok"], result
    assert result["result"]["boundary"] == "sandbox"
    assert result["result"]["stdout"]["content"] == "ordinary"
    interaction.prompt.assert_not_called()


def test_native_session_scratch_is_writable_but_control_state_is_not(application_command):
    """App-data protection permits private TMPDIR while denying controls and sibling sessions."""
    paths, manager, interaction, call = application_command
    scratch = manager.temporary_directory
    controls = (
        paths.user_permissions,
        scratch.parent / "control",
        paths.data_root / "sibling/value",
    )
    for path in controls[1:]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("control-canary")
    snapshots = {path: path.read_bytes() for path in controls}
    source = "printf remembered-ok"
    interaction.prompt.return_value = "session"
    for _ in range(2):
        result = call("run_command", command=source, cwd="$WORKSPACE_ROOT")
        assert result["ok"], result
        assert result["result"]["stdout"]["content"] == "remembered-ok", result
    assert interaction.prompt.call_count == 1
    assert {"approve", "session", "workspace", "user"} <= set(
        interaction.prompt.call_args.kwargs["choices"]
    )
    interaction.prompt.return_value = "approve"
    result = call(
        "run_command",
        command='printf scratch-ok > "$TMPDIR/report"; cat "$TMPDIR/report"',
        cwd="$WORKSPACE_ROOT",
    )
    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "scratch-ok", result
    direct = call("read_text_file", path="$TMPDIR/report")
    assert direct["ok"], direct
    for path, content in snapshots.items():
        for command in (
            f"cat {shlex.quote(str(path))}",
            f"printf changed > {shlex.quote(str(path))}",
        ):
            denied = call("run_command", command=command, cwd="$WORKSPACE_ROOT")
            assert not denied["ok"], denied
            assert "control-canary" not in json.dumps(denied)
        assert path.read_bytes() == content
    (scratch / ".env").write_text("scratch-data")
    private_name = call("run_command", command='cat "$TMPDIR/.env"', cwd="$WORKSPACE_ROOT")
    assert private_name["ok"] is manager.temporary_directory_allows_private_names
    direct_private_name = call("read_text_file", path="$TMPDIR/.env")
    assert direct_private_name["ok"] is private_name["ok"]
    policy = manager.configuration_path
    payload = load_yaml(policy.read_text())
    payload["limits"]["hard_denied_paths"] = [str(scratch / "report")]
    policy.write_text(dump_yaml(payload))
    manager.reload()
    denied = call("run_command", command='cat "$TMPDIR/report"', cwd="$WORKSPACE_ROOT")
    assert not denied["ok"], denied
    assert "scratch-ok" not in json.dumps(denied)
    assert not call("read_text_file", path="$TMPDIR/report")["ok"]
    assert (scratch / "report").read_text() == "scratch-ok"


def test_native_attempt_outputs_share_session_retention_without_sharing_names(application_command):
    """Independent commands can save reports in scratch without modifying a checkout report."""
    paths, _, interaction, call = application_command
    checkout = paths.data_root.parents[2] / "checkout"
    coverage = checkout / ".coverage"
    coverage.write_text("checkout-report")
    interaction.prompt.return_value = "approve"
    outputs = []
    for value in ("first", "second"):
        result = call(
            "run_command",
            command=(
                'case "$LOOP_OUTPUT_DIR" in "$TMPDIR"/output-*) ;; *) exit 3 ;; esac; '
                f'printf {value} > "$LOOP_OUTPUT_DIR/.coverage"'
            ),
            cwd="$WORKSPACE_ROOT",
        )
        assert result["ok"], result
        outputs.append(result["result"]["scratch"]["output_path"])
        for path, expected in zip(outputs, ("first", "second")):
            direct = call("read_text_file", path=f"{path}/.coverage")
            assert direct["ok"], direct
            assert expected in json.dumps(direct["result"])
    assert len(set(outputs)) == 2
    result = call("run_command", command='rm "$WORKSPACE_ROOT/.coverage"')
    assert not result["ok"], result
    assert coverage.read_text() == "checkout-report"


def test_native_short_scratch_supports_unquoted_child_paths(application_command):
    """Physical app-data whitespace cannot change the private command or artifact path."""
    paths, manager, interaction, call = application_command
    checkout = paths.data_root.parents[2] / "checkout"
    source = checkout / "child.sh"
    source.write_text('printf child-ok > "$1"\n')
    (checkout / ".env").write_text("checkout-secret")
    interaction.prompt.return_value = "approve"
    scratch = manager.temporary_directory
    assert scratch.parent.parent == paths.session_scratch
    assert not scratch.is_relative_to(checkout)
    assert (scratch.stat().st_mode & 0o777) == 0o700
    result = call("run_command", command="sh child.sh $TMPDIR/child-output")
    assert result["ok"], result
    artifact = call("read_text_file", path="$TMPDIR/child-output")
    assert artifact["ok"] and "child-ok" in json.dumps(artifact["result"]), artifact
    denied = call("run_command", command="cat .env")
    assert not denied["ok"], denied
    assert "checkout-secret" not in json.dumps(denied)


def test_native_write_path_approval_is_reusable_without_implicit_writes(application_command):
    """Saving a command binds its external write scope and leaves other commands subject to review."""
    paths, manager, interaction, call = application_command
    output = paths.data_root.parents[2] / "external output"
    output.mkdir()
    child = output / "read-only"
    child.mkdir()
    filename = shlex.quote(str(output / "value"))
    (output / "value").write_text("first")
    secret = output / ".env"
    secret.write_text("write-denial-canary")
    new_directory = output / "created"
    command = f"mkdir {shlex.quote(str(new_directory))}"
    interaction.prompt.return_value = "workspace"
    result = call("run_command", command=command, write_roots=[str(output)])
    assert result["ok"], result
    assert new_directory.is_dir()
    assert interaction.prompt.call_args.kwargs["index"]["workspace"] == "W"
    assert not manager.sandbox_path_rules(ApprovalChoice.WORKSPACE)
    assert [rule.source for rule in manager.sandboxed_command_rules(ApprovalChoice.WORKSPACE)] == [
        command
    ]
    interaction.prompt.return_value = "approve"
    result = call("run_command", command=f"cat {filename}", read_roots=[str(output)])
    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "first"
    assert interaction.prompt.call_count == 2
    result = call("run_command", command=f"printf rejected > {filename}")
    assert not result["ok"], result
    denied = call(
        "run_command",
        command=f"printf rejected > {shlex.quote(str(secret))}",
        write_roots=[str(output)],
    )
    assert not denied["ok"], denied
    assert secret.read_text() == "write-denial-canary"
    assert (output / "value").read_text() == "first"
    manager.update_sandbox_paths(ApprovalChoice.SESSION, {str(child): SandboxPathAccess.READ_ONLY})
    denied = call("run_command", command=command, write_roots=[str(output)])
    assert not denied["ok"], denied
    assert "read-only" in json.dumps(denied)


@pytest.mark.linux
@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux private shared memory")
def test_native_posix_semaphores_live_only_in_the_attempt(application_command):
    """Python spawn can reopen local semaphores without seeing a host or prior attempt name."""
    from _multiprocessing import SemLock, sem_unlink

    paths, _, interaction, call = application_command
    checkout = paths.data_root.parents[2] / "checkout"
    name = f"/loop-test-{uuid4().hex}"
    host = SemLock(1, 0, 1, name, False)
    interaction.prompt.return_value = "approve"
    (checkout / "check_ipc.py").write_text(
        """import multiprocessing
import os
import sys
from _multiprocessing import SemLock, sem_unlink

if __name__ == '__main__':
    name, mode = sys.argv[1:]
    try:
        SemLock._rebuild(0, 1, 1, name)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('semaphore visible across attempts')
    try:
        sem_unlink(name)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('foreign semaphore was removed')
    if mode == 'create':
        quota = os.statvfs('/dev/shm')
        assert quota.f_blocks * quota.f_frsize == 8 * 1024 * 1024
        local = SemLock(1, 1, 1, name, False)
        assert local.acquire(False)
        context = multiprocessing.get_context('spawn')
        event = context.Event()
        worker = context.Process(target=event.set)
        worker.start()
        assert event.wait(5)
        worker.join(5)
        assert worker.exitcode == 0
    print('attempt-private')
"""
    )
    try:
        for mode in ("create", "reopen"):
            result = call("run_command", command=f"/usr/bin/python3 check_ipc.py {name} {mode}")
            assert result["ok"], json.dumps(result, indent=2)
            assert result["result"]["stdout"]["content"].strip() == "attempt-private"
        assert not host.acquire(False)
    finally:
        sem_unlink(name)


def test_native_path_policy_supports_wrappers_and_preserves_hard_denials(
    application_command, monkeypatch
):
    """Stored read scopes admit indirect tools and their aliases under the private-data guard."""
    paths, manager, interaction, call = application_command
    checkout = paths.data_root.parents[2] / "checkout"
    runtime = checkout.parent / "runtime"
    commands = runtime / "bin"
    package = runtime / "packages" / "fixture"
    commands.mkdir(parents=True)
    package.mkdir(parents=True)
    tool = package / "verify"
    tool.write_text("#!/bin/sh\nprintf 'fixture-runtime\\n'\n")
    tool.chmod(0o755)
    (commands / "fixture-tool").symlink_to(Path("../packages/fixture/verify"))
    (package / "public").write_text("ordinary-data")
    (package / "complete-alias").hardlink_to(package / "public")
    secret = package / "withheld"
    secret.write_text("configured-secret")
    (package / ".env").write_text("built-in-secret")
    (checkout / "shortcut").symlink_to(secret)
    wrapper = checkout / "check.sh"
    wrapper.write_text("#!/bin/sh\nrun_other_tool() { fixture-tool; }\nrun_other_tool\n")
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(commands), os.defpath)))
    manager.update_sandbox_paths(ApprovalChoice.WORKSPACE, {str(runtime): Decision.ALLOW})
    manager.update_sandbox_paths(ApprovalChoice.USER, {str(secret): Decision.DENY})
    interaction.prompt.return_value = "approve"
    quoted = shlex.quote(str(package))

    result = call(
        "run_command",
        command=(
            "./check.sh && "
            f"cat {quoted}/complete-alias && "
            f"if cat {quoted}/withheld {quoted}/.env shortcut; then exit 21; fi; "
            f"if printf changed > {quoted}/public; then exit 22; fi"
        ),
    )

    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "fixture-runtime\nordinary-data"
    assert "secret" not in json.dumps(result)
    assert (package / "public").read_text() == "ordinary-data"
    assert secret.read_text() == "configured-secret"
    interaction.prompt.assert_called_once()
    assert "read files outside" not in interaction.info.call_args.args[0]
    assert not manager.direct_host_execution_enabled

    assert manager.update_sandbox_paths(ApprovalChoice.WORKSPACE, {str(runtime): None})
    refused = call("run_command", command="./check.sh")
    assert not refused["ok"], refused
    assert "fixture-runtime" not in json.dumps(refused)
