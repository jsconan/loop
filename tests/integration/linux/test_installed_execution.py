"""Qualify current installed Linux timeout delivery and resistance to recursive admission."""

import base64
import errno
import fcntl
import hashlib
import http.server
import importlib.metadata
import json
import os
import pwd
import re
import select
import shlex
import shutil
import signal
import socket
import ssl
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from packaging.requirements import Requirement

from loop import (
    ApprovalChoice,
    PermissionManager,
    PolicyScope,
    RuntimeEnvironment,
    SessionManager,
    SQLiteSessionStore,
    ToolRegistry,
)
from loop.constants import (
    MAX_NATIVE_RESULT_BYTES,
    NATIVE_INSTALLATION_DIRECTORY,
    NATIVE_RESULT_DELIVERY_SECONDS,
    NATIVE_UNIT_PREFIX,
)
from loop.execution import CommandExecutionService
from loop.execution.sandbox import (
    SandboxCapability,
    SandboxOutcome,
    SandboxRequest,
    select_sandbox_backend,
)
from loop.execution.sandbox.constants import CPYTHON_EXECUTABLE_NAME
from loop.execution.sandbox.contracts import CommandProcessResult
from loop.execution.sandbox.linux.backend import LinuxNativeBackend
from loop.execution.sandbox.linux.catalog import runtime_catalog
from loop.execution.sandbox.linux.constants import (
    NATIVE_CANCEL_FRAME,
    NATIVE_CONVENTIONAL_TMP_PATH,
    NATIVE_POLICY_VERSION,
    NATIVE_REQUEST_MAGIC,
    NATIVE_SCRATCH_PATH,
    REQUEST_LENGTH_FORMAT,
    SESSION_CONVENTIONAL_TMP_NAME,
    SYSTEM_PYTHON_PATH,
    SYSTEM_RUNTIME_DIRECTORY,
    SYSTEMD_SOCKET_UNIT_SUFFIX,
)
from loop.execution.sandbox.linux.host import bind_host_file, runtime_read_roots
from loop.execution.sandbox.linux.installation import generation_limits
from loop.execution.sandbox.linux.policy import payload_profile
from loop.execution.sandbox.linux.records import read_generation
from loop.execution.sandbox.linux.runtime import execution_authority
from loop.instructions import InstructionsManager, Skill, SkillManager
from loop.permissions.audit import SQLitePermissionAudit
from loop.permissions.protection import PRIVATE_READ_COMPONENTS
from loop.tooling import ToolRegistration, ToolRuntimeSettings
from loop.tools.files import read_text_file, search_text, write_text_file
from loop.tools.system import (
    cancel_command_job,
    control_command_terminal,
    inspect_command_jobs,
    read_command_job,
    resolve_executable,
    run_command,
)
from loop.utils import PathHolder, VirtualPath
from loop.utils.yaml import load_yaml

_ALIAS_DIRECTORY = os.environ.get("LOOP_NATIVE_SOCKET_ALIAS_DIRECTORY")
_ALTERNATIVE_CANARY = os.environ.get("LOOP_NATIVE_ALTERNATIVE_CANARY")
_SYSTEM_ALIAS_FIXTURE = os.environ.get("LOOP_NATIVE_SYSTEM_ALIAS_FIXTURE")
_CAPACITY_TEST_OPT_IN = os.environ.get("LOOP_NATIVE_CAPACITY_TEST") == "disposable-linux-host"
_INNER_ADVERSARIAL_OPT_IN = (
    os.environ.get("LOOP_NATIVE_INNER_ADVERSARIAL_TEST") == "disposable-linux-host"
)
_RUNTIME_FIXTURE = os.environ.get("LOOP_NATIVE_RUNTIME_FIXTURE")
_RUNTIME_FIXTURE_MARKER = b"Loop private-runtime qualification fixture v1\n"
_QUALIFICATION_UV = shutil.which("uv")
_SYSTEM_MAKE = shutil.which("make", path=os.defpath)
_SYSTEM_BWRAP = shutil.which("bwrap")
_SYSTEM_CAT = shutil.which("cat", path=os.defpath)
_SYSTEM_PERL = shutil.which("perl", path=os.defpath)
_SYSTEM_SH = shutil.which("sh", path=os.defpath)
_SYSTEM_PRINTF = shutil.which("printf", path=os.defpath)
_SYSTEM_LN = shutil.which("ln", path=os.defpath)
_HOST_HOME = Path.home().resolve()
_PUBLIC_ROOT_MARKER = b"Loop installed native qualification fixture v1\n"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.linux,
    pytest.mark.skipif(sys.platform != "linux", reason="Requires installed Linux native authority"),
    pytest.mark.skipif(
        os.environ.get("LOOP_NATIVE_LINUX_TEST") != "installed",
        reason="Set LOOP_NATIVE_LINUX_TEST=installed for native Linux qualification",
    ),
]


class RefusingHost:
    """Fail any unintended host execution during native public qualification."""

    def run_host_command(self, request):
        """Reject a host request that should never have been offered or authorized."""
        raise AssertionError("Native qualification attempted host execution")


class Interaction:
    """Select the requested sandbox decision and reject every host offer."""

    prompts: list[str]
    infos: list[str]
    configuration_path: Path | None
    deny_sandbox: bool
    retarget: tuple[Path, Path] | None

    def __init__(self):
        self.prompts = []
        self.infos = []
        self.configuration_path = None
        self.deny_sandbox = False
        self.retarget = None

    def prompt(self, message=None, **kwargs):
        """Record each permission decision and reject unintended host recovery."""
        self.prompts.append(str(message))
        assert "without the OS sandbox" not in str(message)
        if self.deny_sandbox:
            return "deny"
        if "review" in kwargs.get("choices", {}):
            return "review"
        if self.retarget is not None:
            link, target = self.retarget
            self.retarget = None
            link.unlink()
            link.symlink_to(target)
        return ApprovalChoice.ONCE if message == "Proceed?" else "approve"

    def info(self, message):
        """Accept the permission service's informational text."""
        self.infos.append(str(message))


@pytest.fixture(scope="module")
def installation():
    """Require authenticated compatible installed authority without provisioning or repairing it."""
    record = NATIVE_INSTALLATION_DIRECTORY / "management.json"
    if not record.exists():
        pytest.skip(
            "Current native installation must be explicitly provisioned before qualification"
        )
    execution_authority(record)
    generation = json.loads(record.read_text())["config"]["generation"]
    backend = select_sandbox_backend()
    assert isinstance(backend, LinuxNativeBackend), backend.capability_failure()
    assert backend.capability_failure() is None
    return backend, generation


def test_yolo_scoped_external_read_uses_installed_sandbox_without_saved_grant(
    installation, public_workspace_root, native_directory
):
    """An installed sandbox applies one YOLO scope without a prompt or durable approval."""
    backend, _generation = installation
    workspace = native_directory(public_workspace_root[0], name="yolo-scoped-workspace")
    outside = native_directory(public_workspace_root[2], name="yolo-scoped-external")
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    permissions = PermissionManager(workspace, yolo=True)
    interaction = Interaction()
    registry = ToolRegistry(
        [run_command],
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, RefusingHost()),
    )
    arguments = json.dumps(
        {
            "command": f"cat {shlex.quote(str(canary))}",
            "cwd": str(workspace),
            "read_only": True,
            "read_roots": [str(outside)],
        }
    )
    try:
        allowed = json.loads(registry.call("run_command", arguments, interaction=interaction))
        assert allowed["ok"], allowed
        assert allowed["result"]["boundary"] == "sandbox"
        assert allowed["result"]["stdout"]["content"] == "outside-canary"
        assert interaction.prompts == []

        permissions.set_yolo(False)
        interaction.deny_sandbox = True
        denied = json.loads(registry.call("run_command", arguments, interaction=interaction))
        assert denied["problem"]["code"] == "tool.denied", denied
        assert interaction.prompts == ["Allow this command?"]
    finally:
        registry.close()


def test_selection_with_an_alternate_private_temporary_root(
    installation, public_workspace_root, native_directory, monkeypatch
):
    """Qualify the real Linux backend when client and receiver temporary roots differ."""
    alternate = native_directory(public_workspace_root[0], name="loop-native-test-alternate-tmp")
    monkeypatch.setattr(tempfile, "tempdir", str(alternate))

    backend = select_sandbox_backend()

    assert isinstance(backend, LinuxNativeBackend), backend.capability_failure()
    assert backend.capability_failure() is None


def test_retained_receiver_imports_without_client_diagnostics(installation, tmp_path):
    """A fresh protected runtime does not require the client-only observer module."""
    catalog, _executables, _policies, _attachment = runtime_catalog(Path("/native"), "native_test")
    assert "loop/execution/sandbox/linux/diagnostics.py" not in catalog
    for name, data in catalog.items():
        if name.startswith("loop/"):
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    source = (
        "import sys; "
        f"sys.path.insert(0, {str(tmp_path)!r}); "
        "import loop.execution.sandbox.linux.runtime; "
        "import loop.execution.sandbox.linux.transport; "
        "assert 'loop.execution.sandbox.linux.diagnostics' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-B", "-c", source], check=True, timeout=15)


@pytest.fixture
def private_runtime_fixture():
    """Require explicitly provisioned fake private data under a protected runtime tree."""
    if not _RUNTIME_FIXTURE:
        pytest.skip("Set LOOP_NATIVE_RUNTIME_FIXTURE to the owned private-runtime fixture")
    root = Path(_RUNTIME_FIXTURE)
    assert root.parent == Path("/usr/local/share")
    assert root.name.startswith("loop-native-private-fixture-")
    assert root.resolve(strict=True) == root
    marker = bind_host_file(root / "marker")
    with marker.open_verified() as descriptor:
        assert os.pread(descriptor, len(_RUNTIME_FIXTURE_MARKER) + 1, 0) == _RUNTIME_FIXTURE_MARKER
    return root


@pytest.fixture(scope="module")
def public_workspace_root(request):
    """Provide stable, private fixture roots so repeated runs reuse exact native policies."""
    parent = (
        _HOST_HOME
        if getattr(request, "param", None) == "home"
        else Path(tempfile.gettempdir()).resolve(strict=True)
    )
    parent_info = parent.stat()
    private_parent = parent_info.st_uid == os.getuid() and not parent_info.st_mode & 0o022
    sticky_shared_parent = (
        parent_info.st_uid == 0
        and bool(parent_info.st_mode & stat.S_ISVTX)
        and bool(parent_info.st_mode & stat.S_IWOTH)
    )
    if not stat.S_ISDIR(parent_info.st_mode) or not (private_parent or sticky_shared_parent):
        pytest.fail("Installed native qualification requires a private or sticky temporary parent")

    suffix = str(os.getuid())
    roots = (
        parent / f"loop-native-public-{suffix}",
        parent / f"loop-native-public-permissions-{suffix}",
        parent / f"loop-native-public-external-{suffix}",
    )
    for root in roots:
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            metadata = root.lstat()
            marker = root / ".loop-native-public-fixture"
            assert stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.getuid()
            assert stat.S_IMODE(metadata.st_mode) == 0o700
            assert marker.is_file() and not marker.is_symlink()
            assert marker.read_bytes() == _PUBLIC_ROOT_MARKER
            assert stat.S_IMODE(marker.stat().st_mode) == 0o600
        else:
            marker = root / ".loop-native-public-fixture"
            marker.write_bytes(_PUBLIC_ROOT_MARKER)
            marker.chmod(0o600)

        marker = root / ".loop-native-public-fixture"
        metadata = root.lstat()
        assert stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.getuid()
        assert stat.S_IMODE(metadata.st_mode) == 0o700
        for child in root.iterdir():
            if child == marker:
                continue
            assert child.name.startswith("loop-native-test-"), (
                f"Unexpected qualification fixture entry: {child.name}"
            )
            if child.is_symlink() or not child.is_dir():
                child.unlink()
            else:
                shutil.rmtree(child)

    yield roots

    for root in roots:
        marker = root / ".loop-native-public-fixture"
        for child in root.iterdir():
            if child == marker:
                continue
            assert child.name.startswith("loop-native-test-"), (
                f"Unexpected qualification fixture entry: {child.name}"
            )
            if child.is_symlink() or not child.is_dir():
                child.unlink()
            else:
                shutil.rmtree(child)
        marker.unlink()
        root.rmdir()


def _test_directory_name(_node_id):
    """Reuse a bounded path after each case's owned directory has been removed."""
    return "loop-native-test-case"


@pytest.fixture
def public(installation, public_workspace_root, request, native_directory, environment_patch):
    """Own the stable workspace fixture and isolated permission roots for one test."""
    _, generation = installation
    workspace_root, permissions_root, _external_root = public_workspace_root
    environment_patch.setattr(tempfile, "tempdir", "/tmp")
    if getattr(request.node, "callspec", None) and request.node.callspec.params.get("alias"):
        if not _ALIAS_DIRECTORY:
            pytest.skip(
                "Set LOOP_NATIVE_SOCKET_ALIAS_DIRECTORY to run "
                "same-filesystem socket-alias qualification"
            )
        parent = Path(_ALIAS_DIRECTORY).resolve(strict=True)
        socket_path = Path("/run") / ("loopgen_" + generation + ".socket")
        assert parent.stat().st_uid == os.getuid() and parent.stat().st_mode & 0o777 == 0o700
        assert parent.stat().st_dev == socket_path.stat().st_dev
        workspace = native_directory(parent, name="loop-native-test-alias")
    else:
        workspace = native_directory(workspace_root, name="loop-native-test-workspace")
    permissions_directory = native_directory(
        permissions_root,
        name=_test_directory_name(request.node.nodeid),
    )
    interaction = Interaction()
    interaction.configuration_path = permissions_directory / f"{workspace.name}.yaml"
    yield interaction, workspace, generation


@pytest.fixture
def public_registry(public):
    """Provide isolated inputs for public tools executed by the installed native client."""
    interaction, workspace, generation = public
    assert interaction.configuration_path is not None
    audit_directory = interaction.configuration_path.parent
    audit_path = audit_directory / f"{workspace.name}-permission-audit.sqlite"
    return interaction, workspace, generation, audit_path


def invoke_tools(
    public,
    calls,
    *,
    audit_path=None,
    workspace_id=None,
    permission_workspace=None,
    model_paths=False,
    seconds=2,
    expected_search_errors=0,
    deny_sandbox=False,
    fresh_scratch=False,
    scratch_path=None,
    retarget=None,
    disable_runtime_discovery=False,
    skill_root=None,
    allow_general_network=False,
    deny_local_network=False,
):
    """Run public tool calls in one isolated process with the installed native receiver."""
    interaction, workspace, _ = public
    assert interaction.configuration_path is not None
    client = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve())],
        input=json.dumps(
            {
                "workspace": str(permission_workspace or workspace),
                "configuration_path": str(interaction.configuration_path),
                "audit_path": str(audit_path) if audit_path is not None else None,
                "workspace_id": workspace_id,
                "calls": calls,
                "model_paths": model_paths,
                "seconds": seconds,
                "deny_sandbox": deny_sandbox,
                "fresh_scratch": fresh_scratch,
                "scratch_path": str(scratch_path) if scratch_path is not None else None,
                "retarget": [str(path) for path in retarget] if retarget is not None else None,
                "disable_runtime_discovery": disable_runtime_discovery,
                "skill_root": str(skill_root) if skill_root is not None else None,
                "allow_general_network": allow_general_network,
                "deny_local_network": deny_local_network,
            }
        ),
        capture_output=True,
        text=True,
        env={
            key: value for key, value in os.environ.items() if key not in {"TMPDIR", "TMP", "TEMP"}
        },
        check=False,
        timeout=max(15, seconds * len(calls) + 10),
    )
    assert client.returncode == 0, (
        f"Installed public client exited {client.returncode}.\n"
        f"stdout:\n{client.stdout}\nstderr:\n{client.stderr}"
    )
    if expected_search_errors:
        assert re.fullmatch(
            r"(?:Could not search text \[err_[a-f0-9]+\]\n){" + str(expected_search_errors) + "}",
            client.stderr,
        ), client.stderr
    else:
        assert not client.stderr, client.stderr
    value = json.loads(client.stdout)
    interaction.prompts.extend(value["prompts"])
    interaction.infos.extend(value["infos"])
    assert not any("without the OS sandbox" in prompt for prompt in interaction.prompts)
    return value["results"]


def invoke_many(
    public,
    calls,
    *,
    permission_workspace=None,
    allow_general_network=False,
    deny_local_network=False,
):
    """Run several public commands in one client with their requested workspace authority."""
    _interaction, workspace, _ = public
    return invoke_tools(
        public,
        [
            {
                "tool": "run_command",
                "arguments": {"command": command, "cwd": str(workspace), **grants},
            }
            for command, grants in calls
        ],
        permission_workspace=permission_workspace,
        allow_general_network=allow_general_network,
        deny_local_network=deny_local_network,
    )


def invoke(public, command, *, allow_general_network=False, deny_local_network=False, **grants):
    """Run one public client call outside pytest's process-local temporary-root replacement."""
    return invoke_many(
        public,
        [(command, grants)],
        allow_general_network=allow_general_network,
        deny_local_network=deny_local_network,
    )[0]


def invoke_registered(
    public_registry,
    calls,
    *,
    model_paths=False,
    deny_sandbox=False,
    disable_runtime_discovery=False,
):
    """Call registered native tools together so opaque grants stay bound to one service."""
    interaction, workspace, generation, audit_path = public_registry
    return invoke_tools(
        (interaction, workspace, generation),
        calls,
        audit_path=audit_path,
        workspace_id="native-public-review",
        model_paths=model_paths,
        deny_sandbox=deny_sandbox,
        disable_runtime_discovery=disable_runtime_discovery,
    )


def native_request(
    backend, workspace, scratch, source, *, read_roots=(), network=False, seconds=15
):
    """Build one short-lived request for direct installed-backend qualification."""
    return SandboxRequest.create(
        source=source,
        cwd=workspace,
        workspace=workspace,
        read_roots=read_roots,
        write_roots=(workspace,),
        network=network,
        environment={"PATH": os.environ["PATH"], "LC_ALL": "C", "TMPDIR": str(scratch)},
        policy_version=backend.policy_version,
        deadline=time.monotonic() + seconds,
        workspace_id="native-installed-review",
    )


def public_client():
    """Run registered public calls with authenticated native authority and no host fallback."""
    value = json.loads(sys.stdin.read())
    if value.get("disable_runtime_discovery"):
        from loop.execution import facade

        facade.discover_workspace_runtimes = lambda *_args: ()
        facade.discover_workspace_interpreters = lambda *_args: ()
    backend = LinuxNativeBackend()
    failure = backend.capability_failure()
    assert failure is None, failure
    interaction = Interaction()
    interaction.deny_sandbox = value.get("deny_sandbox", False)
    if value.get("retarget") is not None:
        interaction.retarget = tuple(Path(path) for path in value["retarget"])
    instructions = None
    if value.get("model_paths"):
        scratch = (
            Path(value["scratch_path"])
            if value.get("scratch_path")
            else Path(value["configuration_path"]).parent / "instruction-scratch"
        )
        scratch.mkdir(mode=0o700, exist_ok=True)
        skill_root = value.get("skill_root")
        skill_manager = (
            SkillManager([Skill("fixture", "Native fixture.", Path(skill_root) / "SKILL.md")])
            if skill_root is not None
            else None
        )
        instructions = InstructionsManager(
            runtime_environment=RuntimeEnvironment(Path(value["workspace"]), scratch),
            skill_manager=skill_manager,
        )
        if skill_root is not None:
            instructions.activate_skill("fixture")
    permission_manager_arguments = {}
    if value.get("audit_path") is not None:
        permission_manager_arguments = {
            "audit_path": value["audit_path"],
            "workspace_id": value["workspace_id"],
        }
    permissions = PermissionManager(
        Path(value["workspace"]),
        configuration_path=value["configuration_path"],
        **permission_manager_arguments,
    )
    background_sessions = (
        SessionManager(interaction=interaction)
        if any(
            call["arguments"].get("background") or call["arguments"].get("interactive")
            for call in value["calls"]
        )
        else None
    )
    if background_sessions is not None:
        permissions.bind_temporary_directory(background_sessions.temporary_directory)
    if value.get("allow_general_network"):
        permissions.set_limit("deny_private_networks", False, scope=PolicyScope.SESSION)
    if value.get("deny_local_network"):
        permissions.set_limit("deny_local_networks", True, scope=PolicyScope.SESSION)
    if instructions is not None:
        permissions.bind_temporary_directory(
            PathHolder(scratch), ordinary_names=(lambda: True) if value["fresh_scratch"] else None
        )
    registry = ToolRegistry(
        [
            run_command,
            inspect_command_jobs,
            read_command_job,
            cancel_command_job,
            control_command_terminal,
            resolve_executable,
            read_text_file,
            search_text,
            write_text_file,
            ToolRegistration(run_command, name="renamed_command"),
        ],
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, RefusingHost()),
        settings=ToolRuntimeSettings(command_timeout=value.get("seconds", 2)),
        session_manager=background_sessions,
    )
    try:
        results = []
        for call in value["calls"]:
            arguments = dict(call["arguments"])
            for name, source in call.get("argument_sources", {}).items():
                result = results[source["result_index"]]
                result = result.get("result", result)
                arguments[name] = result[source["result_key"]]
            results.append(
                json.loads(
                    registry.call(
                        call["tool"],
                        json.dumps(arguments),
                        interaction=interaction,
                        instructions_manager=instructions,
                    )
                )
            )
        print(
            json.dumps(
                {"results": results, "prompts": interaction.prompts, "infos": interaction.infos}
            )
        )
    finally:
        registry.close()


def require_drained(generation):
    """Verify all owned service cgroups empty within a separate bounded observer interval."""
    scope = Path("/sys/fs/cgroup") / ("loopgen_" + generation + ".slice")
    deadline = time.monotonic() + 2
    while True:
        members = []
        for path in scope.rglob("cgroup.procs"):
            try:
                members.extend(path.read_text().split())
            except FileNotFoundError:
                pass  # systemd can remove a now-empty service cgroup during observation.
        if not members:
            return
        assert time.monotonic() < deadline, "Admitted descendants remain after delivery"
        time.sleep(0.01)


def profile_inventory(generation):
    """Observe enforcing kernel names and fingerprints under protected generation authority.

    Raw policy exports require administrator privilege. Ordinary qualification observes the
    kernel's readable compiled fingerprints; the protected receiver still authenticates exact
    compiled bytes before every admission.
    """
    installed, authority = read_generation(NATIVE_INSTALLATION_DIRECTORY / "management.json")
    assert installed.config.generation == generation
    directory = Path("/sys/kernel/security/apparmor/policy/namespaces") / generation
    with authority.open_verified():
        details = directory.stat()
        assert (details.st_dev, details.st_ino) == installed.namespace
        observed = {}
        for entry in (directory / "profiles").iterdir():
            assert entry.stat().st_uid == 0
            name = (entry / "name").read_text().strip()
            assert name == "bootstrap" or re.fullmatch(r"payload_[a-f0-9]{64}_[a-f0-9]{48}", name)
            assert (entry / "mode").read_text().strip() == "enforce"
            fingerprint = (entry / "sha256").read_bytes().strip()
            assert re.fullmatch(rb"[a-f0-9]{64}", fingerprint)
            assert name not in observed
            observed[name] = fingerprint
        assert "bootstrap" in observed
        return observed


def assert_equivalent_profiles(before, after):
    """Compare request policy identities while allowing distinct service lifetime suffixes."""
    assert before["bootstrap"] == after["bootstrap"]
    assert {name[:72] for name in before if name != "bootstrap"} == {
        name[:72] for name in after if name != "bootstrap"
    }


def test_public_skill_root_shell_uses_physical_path_without_rewriting(public, native_directory):
    """File-tool skill roots remain distinct from literal shell paths and native read grants."""
    skill = native_directory(public[1], name="loop-native-test-skill")
    (skill / "SKILL.md").write_text(
        "---\nname: fixture\ndescription: Native fixture.\n---\nRead resources.\n", encoding="utf-8"
    )
    (skill / "neighbor").write_text("ordinary-skill-data", encoding="utf-8")
    results = invoke_tools(
        public,
        [
            {"tool": "read_text_file", "arguments": {"path": "/skills/fixture/neighbor"}},
            {
                "tool": "run_command",
                "arguments": {"command": f"cat {shlex.quote(str(skill / 'neighbor'))}", "cwd": "."},
            },
            {
                "tool": "run_command",
                "arguments": {
                    "command": "printf '%s\\n' '/skills/fixture/neighbor'; "
                    "cat /skills/fixture/neighbor",
                    "cwd": ".",
                },
            },
        ],
        model_paths=True,
        skill_root=skill,
        seconds=15,
    )
    assert results[0]["ok"], json.dumps(results[0], indent=2)
    assert results[0]["result"]["content"] == "ordinary-skill-data"
    assert (
        results[1]["ok"] and results[1]["result"]["stdout"]["content"] == "ordinary-skill-data"
    ), results
    assert results[2]["problem"]["code"] == "process.nonzero_exit", results[2]
    assert results[2]["problem"]["metadata"]["stdout"]["content"] == "/skills/fixture/neighbor\n"
    require_drained(public[2])


def test_public_external_skill_root_respects_filesystem_limit(public, native_directory):
    """An activated external skill does not bypass the workspace filesystem hard limit."""
    skill = native_directory(public[1].parent, name="loop-native-test-external-skill")
    (skill / "SKILL.md").write_text(
        "---\nname: fixture\ndescription: Native fixture.\n---\nRead resources.\n", encoding="utf-8"
    )
    (skill / "neighbor").write_text("outside-skill-data", encoding="utf-8")
    result = invoke_tools(
        public,
        [{"tool": "read_text_file", "arguments": {"path": "/skills/fixture/neighbor"}}],
        model_paths=True,
        skill_root=skill,
        seconds=15,
    )[0]
    assert result["problem"]["code"] == "tool.denied", result
    assert "filesystem_root" in result["problem"]["detail"]
    assert "outside-skill-data" not in json.dumps(result)


def test_public_virtual_commands_reuse_policy_and_keep_protected_paths(public, native_directory):
    """Normal model aliases reuse exact policy and preserve renamed-tool and private boundaries."""
    interaction, workspace, generation = public
    (workspace / "ordinary").write_text("ordinary-data")
    (workspace / ".ssh").mkdir()
    secret = workspace / ".ssh/key"
    secret.write_text("preserve-private")
    outside = native_directory(workspace.parent, name="loop-native-test-outside")
    (outside / "secret").write_text("outside-secret")
    call = {
        "tool": "run_command",
        "arguments": {
            "command": 'cat "$WORKSPACE_ROOT/ordinary"; printf "%s" "$WORKSPACE_ROOT"',
            "cwd": "$WORKSPACE_ROOT",
            "read_only": True,
        },
    }
    (first,) = invoke_tools(public, [call], model_paths=True)
    assert first["ok"], first
    baseline = profile_inventory(generation)
    renamed = {**call, "tool": "renamed_command"}
    repeated = invoke_tools(public, [call, renamed], model_paths=True)
    assert all(
        result["ok"] and result["result"]["stdout"]["content"] == f"ordinary-data{workspace}"
        for result in repeated
    ), repeated
    assert_equivalent_profiles(baseline, profile_inventory(generation))
    assert not interaction.prompts
    denied = invoke_tools(
        public,
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": 'cat "$WORKSPACE_ROOT/.ssh/key"; cat ../loop-native-test-outside/secret',
                    "cwd": "$WORKSPACE_ROOT",
                    "read_only": True,
                },
            },
            {
                "tool": "run_command",
                "arguments": {
                    "command": 'printf bad > "$WORKSPACE_ROOT/.ssh/key"',
                    "cwd": "$WORKSPACE_ROOT",
                },
            },
        ],
        model_paths=True,
    )
    assert all(not result["ok"] for result in denied), denied
    assert "outside-secret" not in json.dumps(denied)
    assert secret.read_text() == "preserve-private"
    require_drained(generation)


def test_public_native_search_reuses_policy_and_preserves_dialect_bounds(public, monkeypatch):
    """Trusted ripgrep handles literal/regex queries with private selected-file authority."""
    interaction, workspace, generation = public
    trusted = Path("/usr/bin/rg")
    assert trusted.is_file(), "Native search qualification requires distribution ripgrep"
    bind_host_file(trusted.resolve(strict=True))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    (workspace / "selected").write_text("first needle\nsecond needle\n")
    calls = [
        {
            "tool": "search_text",
            "arguments": {"path": "$WORKSPACE_ROOT/selected", "query": "needle"},
        }
    ]
    (first,) = invoke_tools(public, calls, model_paths=True)
    assert first["ok"], first
    baseline = profile_inventory(generation)
    literal, regex, parse_error = invoke_tools(
        public,
        [
            *calls,
            {
                "tool": "search_text",
                "arguments": {
                    "path": "$WORKSPACE_ROOT/selected",
                    "query": "needle$",
                    "regex": True,
                },
            },
            {
                "tool": "search_text",
                "arguments": {
                    "path": "$WORKSPACE_ROOT/selected",
                    "query": "(?<=first )needle",
                    "regex": True,
                },
            },
        ],
        model_paths=True,
        expected_search_errors=1,
    )
    assert literal["ok"] and regex["ok"], (literal, regex)
    assert len(literal["result"]["matches"]) == len(regex["result"]["matches"]) == 2
    assert not parse_error["ok"] and "regex parse error" in json.dumps(parse_error)
    assert_equivalent_profiles(baseline, profile_inventory(generation))
    (workspace / ".git").mkdir()
    (workspace / ".git/key").write_text("private-canary")
    outside_canary = interaction.configuration_path.parent / "outside-search-canary"
    outside_canary.write_text("outside-canary")
    (workspace / "outside-link").symlink_to(outside_canary)
    private, outside = invoke_tools(
        public,
        [
            {
                "tool": "search_text",
                "arguments": {"path": "$WORKSPACE_ROOT/.git/key", "query": "private-canary"},
            },
            {
                "tool": "search_text",
                "arguments": {"path": "$WORKSPACE_ROOT/outside-link", "query": "anything"},
            },
        ],
        model_paths=True,
    )
    assert not private["ok"] and not outside["ok"], json.dumps((private, outside))
    assert "outside-canary" not in json.dumps((private, outside))
    (workspace / "selected").write_text("needle\n" * 20_000)
    (bounded,) = invoke_tools(
        public,
        [
            {
                "tool": "search_text",
                "arguments": {
                    "path": "$WORKSPACE_ROOT/selected",
                    "query": "needle",
                    "max_results": 1,
                },
            }
        ],
        model_paths=True,
    )
    assert bounded["ok"] and len(bounded["result"]["matches"]) == 1
    assert bounded["result"]["truncated"]
    require_drained(generation)


def test_public_timeout_preserves_both_streams_and_descendant_cleanup(public):
    """Retain timeout output and drain detached children without a host retry."""
    source = "/usr/bin/python3 -I -c " + shlex.quote(
        "import os,sys,time;print('timeout-out',flush=True);"
        "print('timeout-err',file=sys.stderr,flush=True);"
        "pid=os.fork();os.setsid() if pid==0 else None;time.sleep(30)"
    )
    started = time.monotonic()
    result = invoke(public, source, read_only=True)
    elapsed = time.monotonic() - started
    assert result["problem"]["code"] == "sandbox.timed_out"
    metadata = result["problem"]["metadata"]
    assert "host_offer" not in metadata
    assert metadata["sandbox_diagnostics"]["possible_effects"] is True
    assert "not retried" in metadata["recovery_hint"]
    assert "may repeat effects" in metadata["recovery_hint"]
    assert "request_host_command" not in metadata["recovery_hint"]
    assert "timeout-out" in metadata["stdout"]["content"]
    assert "timeout-err" in metadata["stderr"]["content"]
    assert metadata["lifecycle"]["descendant_cleanup"] == "verified_absent"
    for name in ("stdout", "stderr"):
        stream = metadata[name]
        assert stream["capture_complete"] == (stream["discarded_characters"] == 0)
    assert elapsed <= 2 + NATIVE_RESULT_DELIVERY_SECONDS + 2
    require_drained(public[2])


def test_public_background_job_streams_one_process_and_verifies_cleanup(public):
    """An admitted Linux job keeps its authenticated process, ordered cursor, and cgroup proof."""
    _interaction, workspace, generation = public
    calls = [
        {
            "tool": "run_command",
            "arguments": {
                "command": "printf 'first\\n'; sleep 0.3; printf 'last\\n'",
                "cwd": str(workspace),
                "background": True,
            },
        },
        {
            "tool": "read_command_job",
            "arguments": {"wait_seconds": 5},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        },
        {
            "tool": "read_command_job",
            "arguments": {"wait_seconds": 5},
            "argument_sources": {
                "handle": {"result_index": 0, "result_key": "handle"},
                "cursor": {"result_index": 1, "result_key": "next_cursor"},
            },
        },
        {
            "tool": "read_command_job",
            "arguments": {"wait_seconds": 5},
            "argument_sources": {
                "handle": {"result_index": 0, "result_key": "handle"},
                "cursor": {"result_index": 2, "result_key": "next_cursor"},
            },
        },
        {
            "tool": "read_command_job",
            "arguments": {"cursor": 999999},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        },
    ]
    results = invoke_tools(public, calls, seconds=5)
    assert all(item["ok"] for item in results[:-1]), results
    assert results[-1]["problem"]["code"] == "command_job.invalid_read"
    pages = [item["result"] for item in results[1:-1]]
    output = "".join(
        event["content"]
        for page in pages
        for event in page["events"]
        if event["stream"] == "stdout"
    )
    assert output == "first\nlast\n"
    assert "content" not in pages[-1]["result"]["stdout"]
    assert pages[-1]["result"]["lifecycle"]["descendant_cleanup"] == "verified_absent"
    require_drained(generation)


@pytest.mark.parametrize("interactive", [False, True])
def test_public_background_cancellation_waits_for_cgroup_cleanup(public, interactive):
    """Cancellation of a pipe or PTY job waits for verified descendant absence."""
    _interaction, workspace, generation = public
    results = invoke_tools(
        public,
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": "printf 'ready\\n'; sleep 20",
                    "cwd": str(workspace),
                    "background": not interactive,
                    "interactive": interactive,
                },
            },
            {
                "tool": "read_command_job",
                "arguments": {"wait_seconds": 5},
                "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
            },
            {
                "tool": "cancel_command_job",
                "arguments": {},
                "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
            },
            {
                "tool": "read_command_job",
                "arguments": {"wait_seconds": 5},
                "argument_sources": {
                    "handle": {"result_index": 0, "result_key": "handle"},
                    "cursor": {"result_index": 1, "result_key": "next_cursor"},
                },
            },
        ],
        seconds=30,
    )
    assert all(item["ok"] for item in results), results
    assert "ready" in "".join(event["content"] for event in results[1]["result"]["events"])
    assert results[2]["result"]["cancel_requested"]
    final = results[3]["result"]["result"]
    assert final["code"] == "sandbox.cancelled"
    assert final["metadata"]["lifecycle"]["cancel_requested"]
    assert final["metadata"]["lifecycle"]["descendant_cleanup"] == "verified_absent"
    require_drained(generation)


def test_public_process_created_inner_pty_is_private(public):
    """A sandboxed process can allocate a fresh inner terminal without host terminal access."""
    command = "/usr/bin/python3 -I -c " + shlex.quote(
        "import os; master, slave = os.openpty(); "
        "print(os.isatty(slave), os.ttyname(slave)); os.close(master); os.close(slave)"
    )
    result = invoke(public, command, read_only=True)
    assert result["ok"], result
    assert result["result"]["stdout"]["content"].startswith("True /dev/pts/")
    require_drained(public[2])


@pytest.mark.parametrize("adversarial", [False, True])
def test_public_inner_seccomp_restricts_children_without_losing_outer_boundary(
    public, public_workspace_root, native_directory, adversarial
):
    """One public command can run a useful narrower child under the unchanged outer policy."""
    if adversarial and not _INNER_ADVERSARIAL_OPT_IN:
        pytest.skip("Namespace and mount adversarial probes require a disposable Linux host")
    _interaction, workspace, generation = public
    outside = native_directory(public_workspace_root[2], name="loop-native-test-inner-outside")
    (outside / "secret").write_text("outside-canary", encoding="utf-8")
    (outside / "write").write_text("preserve-outside", encoding="utf-8")
    private = workspace / ".ssh"
    private.mkdir()
    (private / "key").write_text("private-canary", encoding="utf-8")
    control = workspace / ".loop" / "control"
    control.parent.mkdir()
    control.write_text("preserve-control", encoding="utf-8")
    source = Path(__file__).with_name("inner_sandbox_fixture.py").read_text(encoding="utf-8")
    script = workspace / "inner_sandbox_fixture.py"
    script.write_text(source, encoding="utf-8")
    bwrap = workspace / "inner-bwrap"
    if adversarial:
        assert _SYSTEM_BWRAP is not None
        shutil.copy2(_SYSTEM_BWRAP, bwrap)
        bwrap.chmod(0o755)
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer,
    ):
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        peer.bind(str(outside / "peer.sock"))
        peer.listen(1)
        command = " ".join(
            shlex.quote(value)
            for value in (
                "/usr/bin/python3",
                "-I",
                "-B",
                str(script),
                str(workspace),
                str(outside),
                str(control),
                str(os.getpid()),
                str(listener.getsockname()[1]),
                "adversarial" if adversarial else "safe",
                str(bwrap),
            )
        )
        result = invoke_tools(
            public,
            [{"tool": "run_command", "arguments": {"command": command, "cwd": str(workspace)}}],
            seconds=30,
        )[0]
    assert result["ok"] and result["result"]["exit_code"] == 0, json.dumps(result, indent=2)
    checks = json.loads(result["result"]["stdout"]["content"])
    assert checks["filter_before"] >= 1
    assert checks["filter_after"] > checks["filter_before"]
    assert checks["outer_unix_errno"] == errno.EPERM
    assert checks["cross_session_unix_errno"] == errno.EPERM
    assert checks["outer_loopback_errno"] in {
        errno.EPERM,
        errno.ENETUNREACH,
        errno.ECONNREFUSED,
        errno.EHOSTUNREACH,
    }
    assert checks["profile_unchanged"]
    assert checks["child_output"] == "child-ok\n"
    assert checks["child_socket_denied"]
    assert checks["allowed_content"] == "inner-work"
    assert checks["inner_socket_errno"] == errno.EPERM
    assert checks["descriptor_receive_errno"] == errno.EPERM
    assert checks["unexpected_descriptors"] == []
    assert all(
        checks[name] != 0
        for name in (
            "private_read_errno",
            "outside_read_errno",
            "outside_write_errno",
            "control_write_errno",
            "host_signal_errno",
        )
    )
    assert (outside / "write").read_text(encoding="utf-8") == "preserve-outside"
    assert control.read_text(encoding="utf-8") == "preserve-control"
    if adversarial:
        assert checks["userns_errno"] in {errno.EACCES, errno.EPERM}, checks
        assert checks["namespace_handle_errno"] in {0, errno.EACCES, errno.EPERM}
        assert (
            checks["setns_errno"] in {errno.EACCES, errno.EPERM}
            if checks["namespace_handle_errno"] == 0
            else checks["setns_errno"] is None
        )
        assert checks["mount_errno"] in {errno.EACCES, errno.EPERM}, checks
        assert checks["remount_errno"] in {errno.EACCES, errno.EPERM}, checks
        assert checks["bwrap_version"].startswith("bubblewrap ")
        assert checks["bwrap_refused"]
        assert checks["cgroup_write_errno"] != 0
        assert checks["profile_change_errno"] in {errno.EACCES, errno.EPERM}, checks
        assert checks["detached_started"]
        assert result["result"]["lifecycle"]["descendant_cleanup"] == "verified_absent"
    require_drained(generation)


def test_public_attempt_private_loopback(public):
    """One offline sandbox attempt exchanges TCP data over its own isolated loopback."""
    source = (
        "import socket; "
        "server=socket.socket(); server.bind(('127.0.0.1',0)); server.listen(1); "
        "client=socket.socket(); client.connect(server.getsockname()); "
        "peer,_=server.accept(); client.sendall(b'local-ok'); "
        "print(peer.recv(8).decode()); peer.close(); client.close(); server.close()"
    )
    result = invoke(public, "/usr/bin/python3 -I -c " + shlex.quote(source), read_only=True)
    assert result["ok"] and result["result"]["stdout"]["content"] == "local-ok\n", result
    require_drained(public[2])


def test_public_attempt_local_protocol_and_host_boundaries(public):
    """Private IPv4/IPv6 TCP and UDP work while host, Unix and external paths stay isolated."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as host_service:
        host_service.bind(("127.0.0.1", 0))
        host_service.listen(1)
        host_port = host_service.getsockname()[1]
        source = f"""
import errno, json, socket
checks = {{}}
for label, family, address in (
    ("ipv4", socket.AF_INET, "127.0.0.1"),
    ("ipv6", socket.AF_INET6, "::1"),
):
    for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
        with socket.socket(family, kind) as server:
            server.bind((address, 0))
            with socket.socket(family, kind) as client:
                if kind == socket.SOCK_STREAM:
                    server.listen(1)
                    client.connect(server.getsockname())
                    with server.accept()[0] as peer:
                        client.sendall(b"local")
                        checks[label + "-tcp"] = peer.recv(5) == b"local"
                else:
                    client.sendto(b"local", server.getsockname())
                    checks[label + "-udp"] = server.recvfrom(5)[0] == b"local"
for label, family, address in (
    ("host", socket.AF_INET, ("127.0.0.1", {host_port})),
    ("alias", socket.AF_INET, ("127.1", {host_port})),
    ("external", socket.AF_INET, ("1.1.1.1", 80)),
):
    try:
        with socket.create_connection(address, timeout=0.25):
            checks[label] = False
    except OSError:
        checks[label] = True
try:
    socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    checks["unix-path-and-abstract"] = False
except OSError as error:
    checks["unix-path-and-abstract"] = error.errno == errno.EPERM
print(json.dumps(checks, sort_keys=True))
"""
        result = invoke(public, "/usr/bin/python3 -I -c " + shlex.quote(source), read_only=True)
    assert result["ok"], result
    checks = json.loads(result["result"]["stdout"]["content"])
    assert all(checks.values()), checks
    require_drained(public[2])


def test_user_local_network_deny_blocks_addressable_sockets(public):
    """A user limit disables IP and Unix listeners without a new model argument."""
    _interaction, _workspace, generation = public
    source = (
        "import errno,socket; "
        "checks=[]; "
        "families=((socket.AF_INET,socket.SOCK_STREAM),"
        "(socket.AF_INET,socket.SOCK_DGRAM),"
        "(socket.AF_INET6,socket.SOCK_STREAM),"
        "(socket.AF_UNIX,socket.SOCK_STREAM)); "
        "exec('for family,kind in families:\\n try:\\n  socket.socket(family,kind)\\n  "
        "checks.append(0)\\n except OSError as error:\\n  checks.append(error.errno)'); "
        "print(checks)"
    )
    result = invoke(
        public,
        "/usr/bin/python3 -I -c " + shlex.quote(source),
        read_only=True,
        deny_local_network=True,
    )
    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == f"{[errno.EPERM] * 4}\n"
    require_drained(generation)


def test_public_interactive_shell_reuses_one_owned_terminal(public):
    """A public terminal handles shell input and output without replay or host execution."""
    _interaction, workspace, generation = public
    calls = [
        {
            "tool": "run_command",
            "arguments": {
                "command": "exec /bin/sh -i",
                "cwd": str(workspace),
                "interactive": True,
            },
        },
        {
            "tool": "control_command_terminal",
            "arguments": {"action": "write", "content": "printf nested-ok\nexit\n"},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        },
    ]
    for index in range(2, 10):
        sources = {"handle": {"result_index": 0, "result_key": "handle"}}
        if index > 2:
            sources["cursor"] = {"result_index": index - 1, "result_key": "next_cursor"}
        calls.append(
            {
                "tool": "read_command_job",
                "arguments": {"wait_seconds": 5},
                "argument_sources": sources,
            }
        )
    results = invoke_tools(public, calls, seconds=15)
    assert all(item["ok"] for item in results), results
    assert results[0]["result"]["terminal"]
    output = "".join(event["content"] for item in results[2:] for event in item["result"]["events"])
    assert "nested-ok" in output, json.dumps(results, indent=2)
    assert results[-1]["result"]["result"]["lifecycle"]["descendant_cleanup"] == ("verified_absent")
    require_drained(generation)


def test_public_terminal_streams_after_native_capture_limit(public):
    """An installed terminal continues delivering output after final capture fills."""
    _interaction, workspace, generation = public
    source = "import sys; sys.stdout.write('x' * 1010000 + 'after-limit'); sys.stdout.flush()"
    calls = [
        {
            "tool": "run_command",
            "arguments": {
                "command": "/usr/bin/python3 -I -c " + shlex.quote(source),
                "cwd": str(workspace),
                "interactive": True,
            },
        }
    ]
    for index in range(1, 161):
        sources = {"handle": {"result_index": 0, "result_key": "handle"}}
        arguments = {"wait_seconds": 5}
        if index == 1:
            arguments["cursor"] = 500
        else:
            sources["cursor"] = {"result_index": index - 1, "result_key": "next_cursor"}
        calls.append(
            {"tool": "read_command_job", "arguments": arguments, "argument_sources": sources}
        )
    results = invoke_tools(public, calls, seconds=30)
    assert all(item["ok"] for item in results), results
    output = "".join(event["content"] for item in results[1:] for event in item["result"]["events"])
    assert "after-limit" in output
    final = results[-1]["result"]["result"]
    assert final["stdout"]["discarded_characters"] > 0
    assert final["lifecycle"]["descendant_cleanup"] == "verified_absent"
    require_drained(generation)


def test_installed_model_schema_advertises_qualified_execution_capabilities(installation):
    """The selected backend exposes supported controls and declares nested seccomp separately."""
    backend = select_sandbox_backend()
    assert isinstance(backend, LinuxNativeBackend)
    service = CommandExecutionService(backend, RefusingHost())
    assert SandboxCapability.INNER_SECCOMP in service.backend_capabilities
    registry = ToolRegistry([run_command, control_command_terminal], execution_service=service)
    try:
        definitions = {item.name: item for item in registry.definitions()}
        properties = definitions["run_command"].parameters["properties"]
        assert "interactive" in properties
        assert "egress_origins" in properties
        assert "control_command_terminal" in definitions
    finally:
        registry.close()


def test_public_terminal_tty_resize_eof_and_untrusted_controls(public):
    """A TTY program receives size and EOF while escape output stays inert in public results."""
    _interaction, workspace, generation = public
    script = (
        "import fcntl,os,struct,sys,termios; "
        "print('TTY',os.isatty(0),os.isatty(1),os.isatty(2),flush=True); "
        "print('INPUT',repr(sys.stdin.readline()),flush=True); "
        "print('SIZE',struct.unpack('HHHH',fcntl.ioctl(0,termios.TIOCGWINSZ,"
        "b'\\0'*8))[:2],flush=True); "
        "print('EOF',repr(sys.stdin.readline()),flush=True); "
        "sys.stdout.write('\\x1b]52;c;bad\\x07\\x1b[31mpermission approved\\x1b[0m'); "
        "sys.stdout.flush()"
    )
    calls = [
        {
            "tool": "run_command",
            "arguments": {
                "command": "/usr/bin/python3 -u -I -c " + shlex.quote(script),
                "cwd": str(workspace),
                "interactive": True,
            },
        },
        {
            "tool": "control_command_terminal",
            "arguments": {"action": "write", "content": "hello\n"},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        },
        {
            "tool": "control_command_terminal",
            "arguments": {"action": "resize", "rows": 37, "columns": 91},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        },
        {
            "tool": "control_command_terminal",
            "arguments": {"action": "eof"},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        },
    ]
    for index in range(4, 12):
        sources = {"handle": {"result_index": 0, "result_key": "handle"}}
        if index > 4:
            sources["cursor"] = {"result_index": index - 1, "result_key": "next_cursor"}
        calls.append(
            {
                "tool": "read_command_job",
                "arguments": {"wait_seconds": 5},
                "argument_sources": sources,
            }
        )
    results = invoke_tools(public, calls, seconds=15)
    assert all(item["ok"] for item in results), results
    output = "".join(event["content"] for item in results[4:] for event in item["result"]["events"])
    assert "TTY True True True" in output, output
    assert "INPUT 'hello\\n'" in output, output
    assert "SIZE (37, 91)" in output, output
    assert "EOF ''" in output, output
    assert "␛]52;c;bad␇" in output and "permission approved" in output
    assert "\x1b" not in output and "\x07" not in output
    assert results[-1]["result"]["result"]["lifecycle"]["descendant_cleanup"] == ("verified_absent")
    require_drained(generation)


def test_public_terminal_interrupt_and_explicit_close(public):
    """Interrupt reaches only the owned foreground process and close settles cleanup."""
    _interaction, workspace, generation = public
    calls = [
        {
            "tool": "run_command",
            "arguments": {
                "command": "exec /bin/sh -i",
                "cwd": str(workspace),
                "interactive": True,
            },
        },
        {
            "tool": "control_command_terminal",
            "arguments": {"action": "write", "content": "sleep 30\n"},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        },
        {
            "tool": "control_command_terminal",
            "arguments": {"action": "interrupt"},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        },
        {
            "tool": "control_command_terminal",
            "arguments": {"action": "write", "content": "printf recovered\nexit\n"},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        },
    ]
    for index in range(4, 9):
        sources = {"handle": {"result_index": 0, "result_key": "handle"}}
        if index > 4:
            sources["cursor"] = {"result_index": index - 1, "result_key": "next_cursor"}
        calls.append(
            {
                "tool": "read_command_job",
                "arguments": {"wait_seconds": 5},
                "argument_sources": sources,
            }
        )
    calls.append(
        {
            "tool": "control_command_terminal",
            "arguments": {"action": "close"},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        }
    )
    results = invoke_tools(public, calls, seconds=35)
    assert all(item["ok"] for item in results), results
    output = "".join(
        event["content"] for item in results[4:9] for event in item["result"]["events"]
    )
    assert "recovered" in output, results
    assert results[-1]["result"]["closed"]
    require_drained(generation)


def test_public_terminal_close_cancels_running_job(public):
    """An explicit close cancels one long-lived terminal and verifies its native cleanup."""
    _interaction, workspace, generation = public
    calls = [
        {
            "tool": "run_command",
            "arguments": {"command": "exec /bin/sh -i", "cwd": str(workspace), "interactive": True},
        },
        {
            "tool": "control_command_terminal",
            "arguments": {"action": "write", "content": "sleep 30\n"},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        },
        {
            "tool": "read_command_job",
            "arguments": {"wait_seconds": 1},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        },
        {
            "tool": "control_command_terminal",
            "arguments": {"action": "close"},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        },
    ]
    results = invoke_tools(public, calls, seconds=35)
    assert all(item["ok"] for item in results), results
    assert results[2]["result"]["status"] in {"running", "complete"}
    assert results[3]["result"]["closed"]
    assert results[3]["result"]["cancel_requested"]
    require_drained(generation)


def test_public_terminal_cannot_write_host_pty_or_spoof_approval(public):
    """The private terminal cannot open a host PTY, reach control, or become a prompt."""
    interaction, workspace, generation = public
    master, slave = os.openpty()
    try:
        host_slave = os.ttyname(slave)
        script = (
            "import os\n"
            "print('OWN',os.isatty(0),os.ttyname(0),flush=True)\n"
            f"for path in {(host_slave, '/run/loopgen_' + generation + '.socket')!r}:\n"
            " try:\n"
            "  fd=os.open(path,os.O_WRONLY|os.O_NONBLOCK)\n"
            "  os.write(fd,b'host-pty-injected')\n"
            "  os.close(fd)\n"
            "  print('OPENED',path,flush=True)\n"
            " except OSError as error: print('DENIED',path,error.errno,flush=True)\n"
            "print('\\x1b]52;c;fake-approval\\x07Approve host access?',flush=True)\n"
        )
        calls = [
            {
                "tool": "run_command",
                "arguments": {
                    "command": "/usr/bin/python3 -u -I -c " + shlex.quote(script),
                    "cwd": str(workspace),
                    "interactive": True,
                },
            }
        ]
        for index in range(1, 8):
            sources = {"handle": {"result_index": 0, "result_key": "handle"}}
            if index > 1:
                sources["cursor"] = {"result_index": index - 1, "result_key": "next_cursor"}
            calls.append(
                {
                    "tool": "read_command_job",
                    "arguments": {"wait_seconds": 5},
                    "argument_sources": sources,
                }
            )
        results = invoke_tools(public, calls, seconds=15)
        assert all(item["ok"] for item in results), results
        output = "".join(
            event["content"] for item in results[1:] for event in item["result"]["events"]
        )
        assert "OWN True" in output and output.count("DENIED") == 2, output
        assert "OPENED" not in output and "\x1b" not in output
        assert "␛]52;c;fake-approval␇" in output
        assert not any("fake-approval" in prompt for prompt in interaction.prompts)
        ready = select.select([master], [], [], 0)[0]
        if ready:
            try:
                assert b"host-pty-injected" not in os.read(master, 4096)
            except OSError as error:
                assert error.errno == errno.EIO
    finally:
        os.close(master)
        os.close(slave)
    require_drained(generation)


def test_public_terminal_cannot_inject_into_another_live_session(public):
    """Separate public clients cannot write through another live session's PTY path."""
    _interaction, workspace, generation = public
    first_script = (
        "import os,time\n"
        "from pathlib import Path\n"
        "Path('first-terminal-path').write_text(os.ttyname(0))\n"
        "print('first-ready',flush=True)\n"
        "while not Path('first-finish').exists(): time.sleep(0.02)\n"
        "print('first-finished',flush=True)\n"
    )
    first_calls = [
        {
            "tool": "run_command",
            "arguments": {
                "command": "/usr/bin/python3 -u -I -c " + shlex.quote(first_script),
                "cwd": str(workspace),
                "interactive": True,
            },
        },
        {
            "tool": "read_command_job",
            "arguments": {"wait_seconds": 30},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        },
        {
            "tool": "read_command_job",
            "arguments": {"wait_seconds": 30},
            "argument_sources": {
                "handle": {"result_index": 0, "result_key": "handle"},
                "cursor": {"result_index": 1, "result_key": "next_cursor"},
            },
        },
    ]
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(invoke_tools, public, first_calls, seconds=35)
        marker = workspace / "first-terminal-path"
        deadline = time.monotonic() + 10
        while not marker.exists() and not first.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        try:
            assert marker.exists(), first.result() if first.done() else "First terminal not ready"
            other_path = marker.read_text()
            second_script = (
                "import os\n"
                f"path={other_path!r}\n"
                "try:\n"
                " fd=os.open(path,os.O_WRONLY|os.O_NONBLOCK)\n"
                " os.write(fd,b'cross-injected')\n"
                " os.close(fd)\n"
                " print('OTHER_OPENED',flush=True)\n"
                "except OSError as error: print('OTHER_DENIED',error.errno,flush=True)\n"
            )
            second_calls = [
                {
                    "tool": "run_command",
                    "arguments": {
                        "command": "/usr/bin/python3 -u -I -c " + shlex.quote(second_script),
                        "cwd": str(workspace),
                        "interactive": True,
                    },
                }
            ]
            for index in range(1, 7):
                sources = {"handle": {"result_index": 0, "result_key": "handle"}}
                if index > 1:
                    sources["cursor"] = {"result_index": index - 1, "result_key": "next_cursor"}
                second_calls.append(
                    {
                        "tool": "read_command_job",
                        "arguments": {"wait_seconds": 5},
                        "argument_sources": sources,
                    }
                )
            second = invoke_tools(public, second_calls, seconds=15)
            assert all(item["ok"] for item in second), second
            second_output = "".join(
                event["content"] for item in second[1:] for event in item["result"]["events"]
            )
            assert "OTHER_DENIED" in second_output or "OTHER_OPENED" in second_output
        finally:
            (workspace / "first-finish").write_text("done")
        first_results = first.result(timeout=35)
    assert all(item["ok"] for item in first_results), first_results
    first_output = "".join(
        event["content"] for item in first_results[1:] for event in item["result"]["events"]
    )
    assert "first-finished" in first_output and "cross-injected" not in first_output
    require_drained(generation)


def test_public_terminal_preserves_file_network_signal_and_descriptor_isolation(public):
    """PTY mode retains protected-file, host IPC, network, signal, and descriptor boundaries."""
    _interaction, workspace, generation = public
    private = workspace / ".ssh"
    private.mkdir()
    (private / "key").write_text("private-terminal-canary")
    unix_path = workspace / "host-control.sock"
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as network_listener,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as ipc_listener,
    ):
        network_listener.bind(("127.0.0.1", 0))
        network_listener.listen(1)
        network_listener.settimeout(0.1)
        ipc_listener.bind(str(unix_path))
        ipc_listener.listen(1)
        ipc_listener.settimeout(0.1)
        host_child = subprocess.Popen([sys.executable, "-I", "-c", "import time;time.sleep(30)"])
        try:
            script = (
                "import os,signal,socket\n"
                "print('TTY',os.isatty(0),flush=True)\n"
                "leaks=[]\n"
                "for name in os.listdir('/proc/self/fd'):\n"
                " try:\n"
                "  if int(name)>2: leaks.append(os.readlink('/proc/self/fd/'+name))\n"
                " except (OSError,ValueError): pass\n"
                "print('LEAKS',repr(leaks),flush=True)\n"
                f"for path in {(str(private / 'key'),)!r}:\n"
                " try:\n"
                "  open(path,'rb').read(1);print('READ',path,flush=True)\n"
                " except OSError as error: print('READ_DENIED',error.errno,flush=True)\n"
                "try:\n"
                " fd=os.open('/usr/local/libexec/loop/native/management.json',os.O_WRONLY)\n"
                " os.close(fd);print('WRITE_OPENED',flush=True)\n"
                "except OSError as error: print('WRITE_DENIED',error.errno,flush=True)\n"
                f"for family,address in {((int(socket.AF_INET), ('127.0.0.1', network_listener.getsockname()[1])), (int(socket.AF_UNIX), str(unix_path)))!r}:\n"
                " try:\n"
                "  client=socket.socket(family,socket.SOCK_STREAM);client.settimeout(1)\n"
                "  client.connect(address);print('CONNECTED',family,flush=True);client.close()\n"
                " except OSError as error: print('CONNECT_DENIED',family,error.errno,flush=True)\n"
                f"try: os.kill({host_child.pid},signal.SIGUSR1);print('SIGNAL_SENT',flush=True)\n"
                "except OSError as error: print('SIGNAL_DENIED',error.errno,flush=True)\n"
            )
            calls = [
                {
                    "tool": "run_command",
                    "arguments": {
                        "command": "/usr/bin/python3 -u -I -c " + shlex.quote(script),
                        "cwd": str(workspace),
                        "interactive": True,
                    },
                }
            ]
            for index in range(1, 9):
                sources = {"handle": {"result_index": 0, "result_key": "handle"}}
                if index > 1:
                    sources["cursor"] = {"result_index": index - 1, "result_key": "next_cursor"}
                calls.append(
                    {
                        "tool": "read_command_job",
                        "arguments": {"wait_seconds": 5},
                        "argument_sources": sources,
                    }
                )
            results = invoke_tools(public, calls, seconds=20)
            assert all(item["ok"] for item in results), results
            output = "".join(
                event["content"] for item in results[1:] for event in item["result"]["events"]
            )
            assert "TTY True" in output and output.count("READ_DENIED") == 1, output
            assert "WRITE_DENIED" in output and "WRITE_OPENED" not in output, output
            assert output.count("CONNECT_DENIED") == 2 and "CONNECTED" not in output, output
            assert "SIGNAL_DENIED" in output and "SIGNAL_SENT" not in output, output
            leaks = next(line for line in output.splitlines() if line.startswith("LEAKS "))
            assert "socket:[" not in leaks and "loopgen_" not in leaks, leaks
            assert "private-terminal-canary" not in output
            assert host_child.poll() is None
            with pytest.raises(socket.timeout):
                network_listener.accept()
            with pytest.raises(socket.timeout):
                ipc_listener.accept()
        finally:
            if host_child.poll() is None:
                host_child.terminate()
            host_child.wait(timeout=5)
    require_drained(generation)


@pytest.mark.skipif(
    os.environ.get("LOOP_NATIVE_LONG_COMMAND_TEST") != "extended-native-check",
    reason="Run the minute-long native duration control explicitly.",
)
def test_public_terminal_continues_same_process_past_old_deadline(public):
    """A quiet interactive command survives the former minute limit without replay."""
    _interaction, workspace, generation = public
    calls = [
        {
            "tool": "run_command",
            "arguments": {"command": "exec /bin/sh -i", "cwd": str(workspace), "interactive": True},
        },
        {
            "tool": "control_command_terminal",
            "arguments": {"action": "write", "content": "sleep 62\nprintf long-done\nexit\n"},
            "argument_sources": {"handle": {"result_index": 0, "result_key": "handle"}},
        },
    ]
    for index in range(2, 10):
        sources = {"handle": {"result_index": 0, "result_key": "handle"}}
        if index > 2:
            sources["cursor"] = {"result_index": index - 1, "result_key": "next_cursor"}
        calls.append(
            {
                "tool": "read_command_job",
                "arguments": {"wait_seconds": 30},
                "argument_sources": sources,
            }
        )
    started = time.monotonic()
    results = invoke_tools(public, calls, seconds=70)
    assert time.monotonic() - started >= 60
    assert all(item["ok"] for item in results), results
    output = "".join(event["content"] for item in results[2:] for event in item["result"]["events"])
    assert "long-done" in output
    assert results[-1]["result"]["result"]["lifecycle"]["descendant_cleanup"] == ("verified_absent")
    require_drained(generation)


@pytest.mark.skipif(
    os.environ.get("LOOP_NATIVE_LONG_COMMAND_TEST") != "extended-native-check",
    reason="Run the five-minute native duration control explicitly.",
)
@pytest.mark.parametrize("noisy", [False, True])
def test_public_foreground_survives_previous_default_deadline(public, noisy):
    """One installed public call completes after five minutes without replay or polling."""
    _interaction, workspace, generation = public
    command = "printf begin; sleep 301; printf end" if noisy else "sleep 301; printf end"

    results = invoke_tools(
        public,
        [{"tool": "run_command", "arguments": {"command": command, "cwd": str(workspace)}}],
        seconds=330,
    )

    assert len(results) == 1 and results[0]["ok"], results
    assert results[0]["result"]["stdout"]["content"].endswith("end")
    assert results[0]["result"]["lifecycle"]["descendant_cleanup"] == "verified_absent"
    require_drained(generation)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_silent_kernel_denial_is_reported_without_replay(public, exit_code):
    """A readable kernel denial becomes uncertain evidence for exactly one attempt."""
    _interaction, workspace, _generation = public
    journal = shutil.which("journalctl", path=os.defpath)
    if journal is None:
        pytest.skip("The host has no journal reader")
    marker = "lp" + os.urandom(5).hex()
    script = f"import ctypes,os; ctypes.CDLL(None).prctl(15, b'{marker}'); os.listdir('/')"
    source = (
        f"printf x >> count; /usr/bin/python3 -c {shlex.quote(script)} "
        f">/dev/null 2>&1; sleep 1; exit {exit_code}"
    )
    result = invoke(public, source)
    logs = subprocess.run(
        [journal, "--no-pager", "--quiet", "--boot", "--dmesg", "--lines=2000", "--output=json"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if not any(marker in json.loads(line).get("MESSAGE", "") for line in logs.stdout.splitlines()):
        pytest.skip("The host did not expose this kernel denial to the ordinary user")
    output = result["result"] if result["ok"] else result["problem"]["metadata"]
    assert output["exit_code"] == exit_code
    assert output["stderr"]["content"] == ""
    diagnostic = output["sandbox_diagnostics"]
    assert diagnostic["collection"] in {"available", "unknown"}, diagnostic
    if diagnostic["collection"] == "available":
        assert diagnostic["source"] == "linux-kernel-journal"
        assert any(
            item["observation"] == "apparmor-open /" and item["attribution"] == "unknown"
            for item in diagnostic["denial_candidates"]
        )
    else:
        assert diagnostic["source"] == "unknown"
        assert diagnostic["denial_candidates"] == []
    assert diagnostic["cause"] == "unknown"
    assert (workspace / "count").read_text() == "x"


def test_concurrent_kernel_denial_does_not_cross_attempts(installation, public, native_directory):
    """Simultaneous requests never receive each other's kernel observations."""
    backend, _generation = installation
    _interaction, workspace, _generation = public
    marker = "lp" + os.urandom(5).hex()
    script = f"import ctypes,os; ctypes.CDLL(None).prctl(15,b'{marker}'); os.listdir('/')"
    denied = native_request(
        backend,
        workspace,
        native_directory(Path("/tmp")),
        "printf x > denied-ready; while test ! -f other-ready; do sleep 0.01; done; "
        f"/usr/bin/python3 -c {shlex.quote(script)} >/dev/null 2>&1; sleep 1; exit 7",
    )
    other = native_request(
        backend,
        workspace,
        native_directory(Path("/tmp")),
        "printf x > other-ready; while test ! -f denied-ready; do sleep 0.01; done; sleep 1; exit 7",
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        denied_future = executor.submit(backend.run, denied)
        other_future = executor.submit(backend.run, other)
        denied_result, other_result = denied_future.result(), other_future.result()
    assert denied_result.exit_code == other_result.exit_code == 7
    journal = shutil.which("journalctl", path=os.defpath)
    if journal is None:
        pytest.skip("The host has no journal reader")
    logs = subprocess.run(
        [journal, "--no-pager", "--quiet", "--boot", "--dmesg", "--lines=2000", "--output=json"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if not any(marker in json.loads(line).get("MESSAGE", "") for line in logs.stdout.splitlines()):
        pytest.skip("The host did not expose this kernel denial to the ordinary user")
    candidates, _ = backend.classify_denial_candidates(denied_result.observed_denial)
    other_candidates, _ = backend.classify_denial_candidates(other_result.observed_denial)
    if denied_result.diagnostic_status == "available":
        assert ("apparmor-open /", False) in candidates
    else:
        assert denied_result.diagnostic_status == "unknown"
        assert denied_result.observed_denial == ""
    assert ("apparmor-open /", False) not in other_candidates, (
        denied_result.observed_denial,
        other_result.observed_denial,
    )


@pytest.mark.parametrize("exit_code", [7, 127])
def test_effectful_native_denial_is_not_replayed(
    public, public_workspace_root, request, native_directory, exit_code
):
    """A real protected read and forged denial text cannot replay an effectful command."""
    if _SYSTEM_CAT is None:
        pytest.skip("System file tool is required")
    _interaction, workspace, generation = public
    _workspace_root, _permissions_root, external_root = public_workspace_root
    outside = native_directory(external_root, name=_test_directory_name(request.node.nodeid))
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    (workspace / "outside-link").symlink_to(canary)
    assert subprocess.check_output([_SYSTEM_CAT, str(canary)], text=True) == "outside-canary"
    source = (
        "printf x >> count; "
        f'if {shlex.quote(_SYSTEM_CAT)} "$WORKSPACE_ROOT/outside-link" >/dev/null 2>&1; '
        f"then exit 9; fi; printf 'Operation not permitted' >&2; exit {exit_code}"
    )

    result = invoke(public, source)

    assert result["problem"]["code"] == "process.nonzero_exit", result
    metadata = result["problem"]["metadata"]
    assert metadata["exit_code"] == exit_code
    assert metadata["sandbox_diagnostics"]["possible_effects"] is True
    assert metadata["sandbox_diagnostics"]["cause"] == "unknown"
    diagnostics = metadata["sandbox_diagnostics"]
    assert diagnostics["collection"] in {"available", "unknown", "unavailable"}
    assert bool(diagnostics["denial_candidates"]) == (diagnostics["collection"] == "available")
    if diagnostics["collection"] == "available":
        assert diagnostics["source"] == "linux-kernel-journal"
    assert all(
        "Operation not permitted" not in item["observation"] and item["attribution"] == "unknown"
        for item in diagnostics["denial_candidates"]
    )
    assert metadata["sandbox_diagnostics"]["observations_may_be_incomplete"] is True
    assert "host_offer" not in metadata
    if exit_code == 127:
        assert "indirect tools remain unidentified" in metadata["executable_hint"]
        assert "may already have had effects" in metadata["executable_hint"]
    else:
        assert "executable_hint" not in metadata
    assert (workspace / "count").read_text(encoding="utf-8") == "x"
    assert canary.read_text(encoding="utf-8") == "outside-canary"
    require_drained(generation)


def test_public_external_python_library_runs_from_a_hardlinked_read_root(
    public, public_workspace_root, request, native_directory
):
    """Keep cache-linked library files readable through their approved read-only root."""
    _workspace_root, _permissions_root, external_root = public_workspace_root
    fixture = native_directory(
        external_root,
        name=_test_directory_name(request.node.nodeid),
    )
    library = fixture / "python-library"
    library.mkdir()
    cache = fixture / "cache.py"
    cache.write_text("value = 'copied-library'\n")
    module = library / "owned_module.py"
    module.hardlink_to(cache)
    code = (
        "import sys;sys.path.insert(0,"
        + repr(str(library))
        + ");import owned_module;print(owned_module.value)"
    )
    command = "/usr/bin/python3 -I -B -c " + shlex.quote(code)
    accepted = invoke(public, command, read_only=True, read_roots=[str(library)])
    assert accepted["result"]["exit_code"] == 0, accepted
    assert accepted["result"]["stdout"]["content"] == "copied-library\n"
    assert module.stat().st_nlink == 2
    assert cache.read_text() == "value = 'copied-library'\n"
    require_drained(public[2])


def test_public_writable_workspace_hardlink_cannot_modify_outside_inode(public, tmp_path):
    """Writable workspaces can read hardlinked files, but their aliases cannot mutate outside."""
    outside_directory = tmp_path.parent / "outside-hardlink"
    outside_directory.mkdir()
    outside = outside_directory / "outside-canary"
    outside.write_text("preserve-outside")
    alias = public[1] / "outside-alias"
    alias.hardlink_to(outside)
    result = invoke(public, "printf changed > " + shlex.quote(str(alias)))
    assert result["problem"]["code"] == "process.nonzero_exit", result
    assert result["problem"]["metadata"]["stderr"]["content"]
    assert outside.read_text() == "preserve-outside"
    assert alias.read_text() == "preserve-outside"
    alias.unlink()
    require_drained(public[2])


@pytest.mark.parametrize("layout", ["dense", "scattered"])
def test_public_ls_runs_with_large_hardlinked_virtualenv(public, native_directory, layout):
    """Keep the public list command usable when Bubblewrap needs grouped hardlink mounts."""
    library = public[1] / ".venv/site-packages/linked-package"
    library.mkdir(parents=True)
    cache = native_directory(Path("/tmp"))
    for index in range(2_001):
        source = cache / f"module-{index}.py"
        source.write_text("value = 1\n")
        directory = library if layout == "dense" else library / str(index)
        directory.mkdir(exist_ok=True)
        (directory / source.name).hardlink_to(source)

    listed = invoke(public, "ls .")
    assert listed["result"]["exit_code"] == 0, listed
    written = invoke(public, "printf allowed > ordinary")
    assert written["result"]["exit_code"] == 0, written
    assert (public[1] / "ordinary").read_text() == "allowed"
    linked = library / ("0/" if layout == "scattered" else "") / "module-0.py"
    code = (
        "from pathlib import Path; import errno,runpy; p=Path("
        + repr(str(linked))
        + "); assert p.read_text()=='value = 1\\n'; assert runpy.run_path(str(p))['value']==1;\n"
    )
    code += "for action in (lambda:p.write_text('bad'),lambda:p.chmod(0o600),lambda:p.unlink(),lambda:p.rename(p.with_name('moved'))):\n try: action()\n except OSError as error: assert error.errno in (errno.EPERM,errno.EACCES,errno.EROFS,errno.EBUSY)\n else: raise RuntimeError('linked inode was mutable')\n"
    verified = invoke(public, "/usr/bin/python3 -I -c " + shlex.quote(code))
    assert verified["ok"], verified
    assert (cache / "module-0.py").read_text() == "value = 1\n"
    if layout == "scattered":
        started = time.monotonic()
        (expired,) = invoke_tools(
            public,
            [
                {
                    "tool": "run_command",
                    "arguments": {
                        "command": "sleep 1; printf late > deadline-marker",
                        "cwd": str(public[1]),
                    },
                }
            ],
            seconds=0.1,
        )
        assert expired["problem"]["code"] == "sandbox.timed_out", expired
        assert "host_offer" not in expired["problem"]["metadata"]
        assert not (public[1] / "deadline-marker").exists()
        assert time.monotonic() - started < 0.1 + NATIVE_RESULT_DELIVERY_SECONDS + 2
    require_drained(public[2])


def test_public_external_write_grant_stays_in_approved_directory(public, native_directory):
    """Installed namespace permits one external write and blocks sibling and control paths."""
    external = native_directory(public[1].parent, name="external-write")
    sibling = native_directory(public[1].parent, name="external-sibling")
    control = public[1] / ".loop"
    control.mkdir()
    protected = control / "policy"
    protected.write_text("unchanged", encoding="utf-8")
    grants = {"write_roots": [str(external)]}

    approved = invoke(
        public, "printf approved > " + shlex.quote(str(external / "allowed")), **grants
    )
    assert approved["ok"], approved
    assert (external / "allowed").read_text(encoding="utf-8") == "approved"

    outside = invoke(public, "printf denied > " + shlex.quote(str(sibling / "blocked")), **grants)
    assert outside["problem"]["code"] == "process.nonzero_exit", outside
    assert not (sibling / "blocked").exists()

    control_write = invoke(public, 'printf denied > "$WORKSPACE_ROOT"/.loop/policy', **grants)
    assert control_write["problem"]["code"] == "process.nonzero_exit", control_write
    assert protected.read_text(encoding="utf-8") == "unchanged"
    require_drained(public[2])


def test_public_read_only_command_can_create_files_in_private_scratch(public):
    """Keep ordinary session scratch writable while the workspace remains read-only."""
    source = (
        "import os,pathlib; root=pathlib.Path(os.environ['TMPDIR']); "
        "target=root/'pytest-fixture'/'workspace'/'project'/'output'; "
        "target.mkdir(parents=True); "
        "(target/'created').write_text('private scratch',encoding='utf-8'); "
        "print('SCRATCH_WRITE=ok')"
    )
    command = "/usr/bin/python3 -I -c " + shlex.quote(source)

    result = invoke(public, command, read_only=True)

    assert result["result"]["exit_code"] == 0, result
    assert result["result"]["stdout"]["content"] == "SCRATCH_WRITE=ok\n"
    assert not public[0].prompts
    require_drained(public[2])


@pytest.mark.parametrize("public_workspace_root", ["home"], indirect=True)
def test_public_conventional_temporary_directory_is_session_private(
    public, native_directory, monkeypatch
):
    """Retain isolated temporary data without bypassing workspace or cross-session controls."""
    interaction, workspace, generation = public
    monkeypatch.setenv("PATH", os.defpath)
    host_directory = native_directory(NATIVE_CONVENTIONAL_TMP_PATH)
    host_canary = host_directory / "canary"
    host_canary.write_text("host-temporary-canary\n", encoding="utf-8")
    database = interaction.configuration_path.parent / "sessions.sqlite"
    workspace_id = "native-temporary-session"
    store = SQLiteSessionStore(database, workspace_id=workspace_id)
    manager = SessionManager(
        interaction=interaction, session_store=store, workspace_id=workspace_id
    )
    scratch = manager.temporary_directory.get()
    assert manager.working_directory_allows_private_names()
    manager.add_user_message("retain a synthetic report")
    other_manager = SessionManager(
        interaction=interaction,
        session_store=SQLiteSessionStore(database, workspace_id=workspace_id),
        workspace_id=workspace_id,
    )
    other = other_manager.temporary_directory.get()
    other_canary = other / "canary"
    other_canary.write_text("other-session-canary\n", encoding="utf-8")
    workspace_file = workspace / "read-only-source"
    workspace_file.write_text("workspace-canary", encoding="utf-8")
    private_name = PRIVATE_READ_COMPONENTS[0]
    private_file = workspace / private_name / "key"
    private_file.parent.mkdir()
    private_file.write_text("workspace-private-canary\n", encoding="utf-8")
    if _SYSTEM_LN is None:
        pytest.skip("A system link utility is required for native link controls")
    temporary = NATIVE_CONVENTIONAL_TMP_PATH
    report = temporary / "report"
    workspace_link = temporary / "workspace-link"
    hardlink = temporary / "workspace-hardlink"
    private_link = temporary / "private-link"
    link_command = shlex.quote(_SYSTEM_LN)
    child = f"{VirtualPath.TEMPORARY}/{SESSION_CONVENTIONAL_TMP_NAME}/report"
    (first,) = invoke_tools(
        public,
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": (
                        f"set -e; printf 'session-data\\n' > {shlex.quote(str(report))}; "
                        f'printf private > "{VirtualPath.TEMPORARY}/outside-subtree"; '
                        f"{link_command} -s {shlex.quote(str(workspace_file))} "
                        f"{shlex.quote(str(workspace_link))}; "
                        f"if printf changed > {shlex.quote(str(workspace_link))} 2>/dev/null; "
                        "then exit 17; fi; "
                        f"if {link_command} {shlex.quote(str(workspace_file))} "
                        f"{shlex.quote(str(hardlink))} 2>/dev/null; then exit 18; fi; "
                        f"{link_command} -s {shlex.quote(str(private_file))} "
                        f"{shlex.quote(str(private_link))}; "
                        f"if IFS= read -r leak 2>/dev/null < {shlex.quote(str(private_link))}; "
                        "then exit 19; fi; "
                        f"printf 'fresh\\n' > {shlex.quote(str(temporary / private_name))}"
                    ),
                    "cwd": VirtualPath.WORKSPACE,
                    "read_only": True,
                },
            }
        ],
        model_paths=True,
        fresh_scratch=True,
        scratch_path=scratch,
    )
    assert first["ok"], json.dumps(first, indent=2)
    assert host_canary.read_text(encoding="utf-8") == "host-temporary-canary\n"
    baseline = profile_inventory(generation)

    (isolated,) = invoke_tools(
        public,
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": (
                        f"set -e; test ! -e {shlex.quote(str(report))}; "
                        f"printf other-session > {shlex.quote(str(report))}"
                    ),
                    "cwd": VirtualPath.WORKSPACE,
                    "read_only": True,
                },
            }
        ],
        model_paths=True,
        fresh_scratch=True,
        scratch_path=other,
    )
    assert isolated["ok"], isolated
    assert_equivalent_profiles(baseline, profile_inventory(generation))

    resumed = SessionManager(
        interaction=interaction,
        session=manager.session.id,
        session_store=SQLiteSessionStore(database, workspace_id=workspace_id),
        workspace_id=workspace_id,
    )
    assert resumed.temporary_directory.get() == scratch
    assert resumed.working_directory_allows_private_names()
    later, read = invoke_tools(
        public,
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": (
                        f"set -e; IFS= read -r value < {shlex.quote(str(report))}; "
                        f"test ! -e {shlex.quote(str(temporary / 'outside-subtree'))}; "
                        f"IFS= read -r own < {shlex.quote(str(temporary / private_name))}; "
                        'test "$own" = fresh; '
                        f"if IFS= read -r leak 2>/dev/null < "
                        f"{shlex.quote(str(temporary / host_directory.name / 'canary'))}; "
                        "then exit 9; fi; "
                        f"if IFS= read -r leak 2>/dev/null < {shlex.quote(str(other_canary))}; "
                        "then exit 9; fi; "
                        'printf "%s" "$value"'
                    ),
                    "cwd": VirtualPath.WORKSPACE,
                    "read_only": True,
                },
            },
            {"tool": "read_text_file", "arguments": {"path": child}},
        ],
        model_paths=True,
        fresh_scratch=True,
        scratch_path=resumed.temporary_directory.get(),
    )
    assert later["ok"] and later["result"]["stdout"]["content"] == "session-data"
    assert read["ok"] and read["result"]["content"] == "session-data\n"
    assert other_canary.read_text(encoding="utf-8") == "other-session-canary\n"
    assert workspace_file.read_text(encoding="utf-8") == "workspace-canary"
    assert private_file.read_text(encoding="utf-8") == "workspace-private-canary\n"
    injected_link = scratch / SESSION_CONVENTIONAL_TMP_NAME / "preexisting-hardlink"
    os.link(workspace_file, injected_link)
    try:
        (refused,) = invoke_tools(
            public,
            [
                {
                    "tool": "run_command",
                    "arguments": {
                        "command": f"printf unsafe > {shlex.quote(str(temporary / injected_link.name))}",
                        "cwd": VirtualPath.WORKSPACE,
                        "read_only": True,
                    },
                }
            ],
            model_paths=True,
            fresh_scratch=True,
            scratch_path=scratch,
        )
        assert not refused["ok"]
        assert refused["problem"]["code"] == "sandbox.invalid", refused
        assert "Session scratch contains a shared inode" in refused["problem"]["detail"]
        assert refused["problem"]["metadata"]["sandbox_diagnostics"]["possible_effects"] is False
    finally:
        injected_link.unlink()
    assert workspace_file.read_text(encoding="utf-8") == "workspace-canary"
    require_drained(generation)
    resumed.discard_working_directory()
    assert not scratch.exists()
    assert other_canary.read_text(encoding="utf-8") == "other-session-canary\n"


def test_public_conventional_temporary_preserves_overlapping_workspace(public, native_directory):
    """An overlapping workspace keeps its grant while unrelated host temporary files stay private."""
    interaction, workspace, generation = public
    host_directory = native_directory(NATIVE_CONVENTIONAL_TMP_PATH)
    outside = host_directory / "outside"
    (result,) = invoke_tools(
        public,
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": (
                        f'set -e; printf workspace > "{VirtualPath.WORKSPACE}/output"; '
                        f'printf scratch > "{VirtualPath.TEMPORARY}/report"; '
                        f"if printf escaped > {shlex.quote(str(outside))} 2>/dev/null; "
                        "then exit 9; fi; printf overlap-ok"
                    ),
                    "cwd": VirtualPath.WORKSPACE,
                },
            }
        ],
        model_paths=True,
        fresh_scratch=True,
    )
    assert result["ok"] and result["result"]["stdout"]["content"] == "overlap-ok", result
    assert (workspace / "output").read_text(encoding="utf-8") == "workspace"
    assert not outside.exists()
    scratch = interaction.configuration_path.parent / "instruction-scratch"
    assert (scratch / "report").read_text(encoding="utf-8") == "scratch"
    assert not (scratch / SESSION_CONVENTIONAL_TMP_NAME).exists()
    require_drained(generation)


def test_nested_workspace_uses_real_scratch_mirror_without_broadening_access(public):
    """A nested workspace has a real namespace path while private data stays protected."""
    interaction, _workspace, generation = public
    scratch = interaction.configuration_path.parent / "instruction-scratch"
    scratch.mkdir(mode=0o700)
    project = scratch / "project"
    project.mkdir()
    (project / "input").write_text("workspace-data", encoding="utf-8")
    (project / "relative").symlink_to("input")
    (project / "absolute").symlink_to(project / "input")
    (project / ".ssh").mkdir()
    (project / ".ssh" / "secret").write_text("private-data", encoding="utf-8")
    mirror = NATIVE_SCRATCH_PATH / project.relative_to(scratch)
    command = (
        f'set -eu; pwd -P; printf \'%s\\n\' "$WORKSPACE_ROOT" "$TMPDIR"; '
        f"cd {shlex.quote(str(mirror))}; pwd -P; "
        f"{shlex.quote(str(SYSTEM_PYTHON_PATH))} -I -c 'import os; print(os.getcwd())'; "
        "cat input relative; printf written > output; "
        "if cat absolute 2>/dev/null; then exit 9; fi; "
        "if cat .ssh/secret 2>/dev/null; then exit 10; fi"
    )

    (result,) = invoke_tools(
        public,
        [{"tool": "run_command", "arguments": {"command": command, "cwd": VirtualPath.WORKSPACE}}],
        permission_workspace=project,
        scratch_path=scratch,
        model_paths=True,
    )

    assert result["ok"], result
    assert result["result"]["stdout"]["content"].splitlines() == [
        str(project),
        str(project),
        str(NATIVE_SCRATCH_PATH),
        str(mirror),
        str(mirror),
        "workspace-dataworkspace-data",
    ]
    assert (project / "output").read_text(encoding="utf-8") == "written"
    require_drained(generation)


@pytest.mark.parametrize("public_workspace_root", ["home"], indirect=True)
def test_bound_session_scratch_works_without_global_temporary_directories(public):
    """Home-based read-only requests share scratch across tools and children without host access."""
    interaction, workspace, generation = public
    parent = interaction.configuration_path.parent
    scratch = parent / "instruction-scratch"
    scratch.mkdir(mode=0o700)
    other = parent / "other-session"
    other.mkdir(mode=0o700)
    (other / "secret").write_text("other-session-canary")
    (parent / "host-private").write_text("host-private-canary")
    (scratch / "escape").symlink_to(other / "secret")
    (workspace / "input").write_text("workspace-data")
    program = (
        "import errno,os,pathlib,subprocess,tempfile\n"
        "s=pathlib.Path(os.environ['TMPDIR']); w=pathlib.Path(os.environ['WORKSPACE_ROOT'])\n"
        "assert os.environ['HOME']==str(s) and pathlib.Path.home()==s\n"
        "assert pathlib.Path.cwd()==w\n"
        "assert not any(pathlib.Path(p).exists() for p in ('/tmp','/var/tmp','/usr/tmp'))\n"
        "assert tempfile.gettempdir()==str(s)\n"
        "assert (w/'input').read_text()=='workspace-data'\n"
        "with tempfile.NamedTemporaryFile() as f:\n"
        " f.write(b'temporary-data'); f.flush()\n"
        " assert pathlib.Path(f.name).parent==s\n"
        " assert subprocess.check_output(['/bin/cat',f.name])==b'temporary-data'\n"
        "for p in (s/'escape',s/'../other-session/secret',s/'../host-private'):\n"
        " try: p.read_text()\n"
        " except OSError as e: assert e.errno in (errno.EACCES,errno.ENOENT)\n"
        " else: raise AssertionError('outside read allowed')\n"
        "try: (w/'forbidden-write').write_text('bad')\n"
        "except OSError as e: assert e.errno in (errno.EACCES,errno.EROFS)\n"
        "else: raise AssertionError('workspace write allowed')\n"
        "print('scratch-ok')\n"
    )
    results = invoke_tools(
        public,
        [
            {
                "tool": "write_text_file",
                "arguments": {
                    "path": "$TMPDIR/probe.sh",
                    "content": 'set -eu\ncd "$WORKSPACE_ROOT"\n'
                    'printf script-data > "$TMPDIR/report.txt"\n'
                    "/usr/bin/python3 -c " + shlex.quote(program) + "\n",
                },
            },
            {
                "tool": "run_command",
                "arguments": {
                    "command": '/bin/sh "$TMPDIR/probe.sh"',
                    "cwd": "$WORKSPACE_ROOT",
                    "read_only": True,
                },
            },
            {"tool": "read_text_file", "arguments": {"path": "$TMPDIR/report.txt"}},
        ],
        model_paths=True,
    )
    assert all(result["ok"] for result in results), json.dumps(results, indent=2)
    assert results[1]["result"]["stdout"]["content"] == "scratch-ok\n"
    assert results[2]["result"]["content"] == "script-data"
    assert (other / "secret").read_text() == "other-session-canary"
    assert (parent / "host-private").read_text() == "host-private-canary"
    assert not (workspace / "forbidden-write").exists()
    require_drained(generation)


def test_public_writable_command_cannot_change_policy_coordination_files(public):
    """Sandboxed commands cannot alter policy, its lock, or reset recovery evidence."""
    workspace = public[1]
    directory = workspace / ".loop"
    directory.mkdir()
    targets = [
        directory / "permissions.yaml",
        directory / "permissions.yaml.lock",
        directory / "permissions.yaml.reset-pending",
    ]
    for target in targets:
        target.write_text("preserve-policy-canary", encoding="utf-8")
    commands = [("printf changed > " + shlex.quote(str(target)), {}) for target in targets]
    commands += [("rm " + shlex.quote(str(target)), {}) for target in targets]
    commands.append(("printf writable > ordinary", {}))

    results = invoke_many(public, commands)

    for (command, _), result in zip(commands[:-1], results[:-1], strict=True):
        assert not result["ok"], (command, result)
        assert result["problem"]["code"] == "process.nonzero_exit", result
    assert all(target.read_text(encoding="utf-8") == "preserve-policy-canary" for target in targets)
    assert results[-1]["result"]["exit_code"] == 0, results[-1]
    assert (workspace / "ordinary").read_text(encoding="utf-8") == "writable"
    require_drained(public[2])


def test_public_writable_command_cannot_change_private_workspace_entries(public):
    """Private entries reject overwrite and creation while an ordinary write remains available."""
    components = (".ssh", ".aws", ".gnupg", ".config", ".loop", "Library", ".AWS")
    targets = []
    calls = []
    for component in components:
        directory = public[1] / component
        directory.mkdir()
        existing = directory / "credential"
        existing.write_text("preserve-private-canary", encoding="utf-8")
        created = directory / "new-entry"
        targets.append((component, existing, created))
        calls.extend(
            ("printf changed > " + shlex.quote(str(target)), {}) for target in (existing, created)
        )
    calls.append(("printf writable > ordinary", {}))

    results = invoke_many(public, calls)
    for result in results[:-1]:
        assert result["problem"]["code"] == "process.nonzero_exit", result
        assert result["problem"]["metadata"]["exit_code"] != 0

    for component, existing, created in targets:
        assert existing.read_text(encoding="utf-8") == "preserve-private-canary", component
        assert not created.exists(), component
    assert results[-1]["result"]["exit_code"] == 0, results[-1]
    assert (public[1] / "ordinary").read_text(encoding="utf-8") == "writable"
    require_drained(public[2])


def test_public_runtime_roots_deny_private_data_and_executables(public, private_runtime_fixture):
    """System runtime reads do not expose private components or execute their contents."""
    root = private_runtime_fixture
    relatives = (".ssh/key", "nested/.AWS/key", ".config/key", ".loop/key", "Library/key")
    calls = [
        ("cat " + shlex.quote(str(root / relative)), {"read_only": True}) for relative in relatives
    ]
    calls.append((shlex.quote(str(root / ".ssh/probe")), {"read_only": True}))
    calls.append(("cat " + shlex.quote(str(root / "neighbor")), {"read_only": True}))

    results = invoke_many(public, calls)

    for result in results[:-1]:
        assert result["problem"]["code"] == "process.nonzero_exit", result
        assert "private-runtime-canary" not in json.dumps(result)
    assert results[-1]["result"]["stdout"]["content"] == "ordinary-runtime-canary"
    for relative in relatives:
        assert (root / relative).read_text() == "private-runtime-canary"
    require_drained(public[2])


def test_public_session_scratch_denies_new_private_names(public):
    """Reused scratch enforces private-name restrictions without granting ambient private data."""
    calls = [
        (
            (
                f'mkdir -p "$TMPDIR/nested"; mkdir "$TMPDIR/nested/{component}"; '
                f'printf fixture > "$TMPDIR/nested/{component}/key"; '
                f'cat "$TMPDIR/nested/{component}/key"'
            ),
            {"read_only": True},
        )
        for component in (".ssh", ".AWS", ".config", ".loop", "Library")
    ]
    calls.append(('printf ordinary > "$TMPDIR/result"; cat "$TMPDIR/result"', {"read_only": True}))
    results = invoke_many(public, calls)
    for result in results[:-1]:
        assert result["problem"]["code"] == "process.nonzero_exit", result
        assert result["problem"]["metadata"]["stdout"]["content"] == ""
    assert results[-1]["result"]["stdout"]["content"] == "ordinary"
    require_drained(public[2])


@pytest.mark.parametrize("relative", [".SSH/key", "nested/.Aws/key", ".git/.CONFIG/key"])
def test_public_private_reads_are_case_insensitive_at_every_depth(public, relative):
    """Keep mixed-case private credentials unreadable beneath workspace and protected roots."""
    target = public[1] / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("private-read-canary", encoding="utf-8")

    result = invoke(public, "cat " + shlex.quote(relative), read_only=True)

    assert result["problem"]["code"] == "process.nonzero_exit", result
    assert "private-read-canary" not in json.dumps(result)
    assert target.read_text(encoding="utf-8") == "private-read-canary"
    require_drained(public[2])


@pytest.mark.parametrize("mode", ["existing", "create"])
def test_public_git_write_grants_only_top_level_metadata(public, mode):
    """Approved top-level Git writes work while nested and application controls stay denied."""
    workspace = public[1]
    nested_git = workspace / "nested" / ".git"
    nested_git.mkdir(parents=True)
    (nested_git / "config").write_text("preserve-nested-git", encoding="utf-8")
    loop_control = workspace / ".loop" / "control"
    loop_control.parent.mkdir()
    loop_control.write_text("preserve-loop-control", encoding="utf-8")
    instruction = workspace / "AGENTS.md"
    instruction.write_text("preserve-instructions", encoding="utf-8")

    if mode == "existing":
        top_level_git = workspace / ".git"
        top_level_git.mkdir()
        config = top_level_git / "config"
        config.write_text("before", encoding="utf-8")
        allowed = "printf granted > .git/config"
    else:
        allowed = "git init --quiet"

    controls = {
        "nested/.git/config": "preserve-nested-git",
        ".loop/control": "preserve-loop-control",
        "AGENTS.md": "preserve-instructions",
    }
    results = invoke_many(
        public,
        [(allowed, {"git_write": True})]
        + [
            ("printf changed > " + shlex.quote(relative), {"git_write": True})
            for relative in controls
        ],
        permission_workspace=workspace,
    )
    assert results[0]["result"]["exit_code"] == 0, results[0]
    if mode == "existing":
        assert config.read_text(encoding="utf-8") == "granted"
    else:
        assert (workspace / ".git" / "HEAD").is_file()
    assert public[0].prompts
    for denied in results[1:]:
        assert denied["problem"]["code"] == "process.nonzero_exit", denied
        assert denied["problem"]["metadata"]["exit_code"] != 0
    for relative, original in controls.items():
        assert (workspace / relative).read_text(encoding="utf-8") == original
    require_drained(public[2])


@pytest.mark.parametrize(
    ("prepared", "environment_name", "retarget", "synchronize", "selection"),
    [
        (False, ".venv", False, False, "project"),
        (True, ".venv", False, False, "project"),
        (True, ".venv", False, False, "project-no-bytecode"),
        (True, ".venv", False, False, "project-cd"),
        (True, ".venv", False, False, "project-scratch"),
        (True, "chosen-environment", False, False, "project"),
        (True, "chosen env's", False, False, "project"),
        (True, ".venv", True, False, "project"),
        (True, ".venv", False, True, "project"),
        (False, ".venv", False, False, "direct"),
        (True, ".venv", False, False, "direct"),
        (False, ".venv", False, False, "script"),
        (True, ".venv", False, False, "script"),
        (True, ".venv", False, False, "script-virtual"),
        (True, ".venv", False, False, "script-whole-quoted"),
        (False, ".venv", False, False, "standard-direct"),
        (True, ".venv", False, False, "standard-direct"),
    ],
)
def test_public_workspace_runtime_is_prepared_in_original_call(
    public_registry,
    public_workspace_root,
    request,
    native_directory,
    monkeypatch,
    prepared,
    environment_name,
    retarget,
    synchronize,
    selection,
):
    """A selected workspace runtime works off PATH after scoped approval."""
    if (_QUALIFICATION_UV is None and selection != "script-virtual") or _SYSTEM_SH is None:
        pytest.skip("The qualification runner and system shell are required")
    configuration = Path(sys.prefix) / "pyvenv.cfg"
    if not configuration.is_file():
        pytest.skip("The qualification interpreter has no environment metadata")
    metadata = {
        key.strip(): value.strip()
        for line in configuration.read_text(encoding="utf-8").splitlines()
        for key, separator, value in [line.partition("=")]
        if separator
    }
    home = Path(metadata.get("home", ""))
    if not home.is_absolute() or metadata.get("implementation") != "CPython":
        pytest.skip("The qualification environment has no supported runtime metadata")
    try:
        interpreter = (Path(sys.prefix) / "bin/python").resolve(strict=True)
        runtime = home.resolve(strict=True).parent
        account_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    except (KeyError, OSError, RuntimeError):
        pytest.skip("The qualification runtime is unavailable")
    if not runtime.is_relative_to(account_home) or not interpreter.is_relative_to(runtime):
        pytest.skip("The qualification runtime is not user-local")
    interaction, workspace, generation, _ = public_registry
    _, _, external_root = public_workspace_root
    outside = native_directory(external_root, name=_test_directory_name(request.node.nodeid))
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    private = workspace / ".ssh"
    private.mkdir()
    secret = private / "key"
    secret.write_text("private-canary", encoding="utf-8")
    environment = workspace / environment_name
    commands = environment / "bin"
    if selection == "standard-direct":
        created = subprocess.run(
            [str(home / interpreter.name), "-m", "venv", "--without-pip", str(environment)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert created.returncode == 0, created.stderr
        (commands / "python").unlink()
    else:
        commands.mkdir(parents=True)
    (commands / "python").symlink_to(interpreter)
    selected_identity = (commands / "python").lstat()
    selected_link = (
        selected_identity.st_dev,
        selected_identity.st_ino,
        os.readlink(commands / "python"),
    )
    if selection != "standard-direct":
        (environment / "pyvenv.cfg").write_text(
            configuration.read_text(encoding="utf-8"), encoding="utf-8"
        )
    project = '[project]\nname = "fixture"\nversion = "0.0.0"\n'
    if synchronize:
        parts = metadata.get("version_info", "").split(".")
        if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
            pytest.skip("The qualification environment has no supported version metadata")
        version = ".".join(parts[:2])
        (workspace / ".python-version").write_text(version + "\n", encoding="utf-8")
        project += f'requires-python = ">={version}"\n[tool.uv]\nexclude-newer = "7 days"\n'
    (workspace / "pyproject.toml").write_text(project, encoding="utf-8")
    program = (
        "import os,pathlib,sys\n"
        f"for target in ({str(canary)!r}, {str(secret)!r}):\n"
        " try: pathlib.Path(target).read_text()\n"
        " except OSError: pass\n"
        " else: raise SystemExit(9)\n"
        "pathlib.Path(os.environ['TMPDIR'], 'entered').write_text('entered')\n"
        "print(sys.prefix)\n"
        "print(sys.base_prefix)\n"
        "print(sys.executable)\n"
        f"print(pathlib.Path({str(interpreter)!r}).exists())\n"
    )
    assert (
        subprocess.run([str(interpreter), "-c", program], timeout=10, check=False).returncode == 9
    )
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(
            (str(Path(_QUALIFICATION_UV).parent), os.defpath)
            if _QUALIFICATION_UV is not None
            else (os.defpath,)
        ),
    )
    if selection in {"script", "script-virtual", "script-whole-quoted"}:
        script = (
            commands / "pytest"
            if selection in {"script-virtual", "script-whole-quoted"}
            else workspace / "fixture-task"
        )
        script.write_text(f"#!{commands / 'python'}\n" + program, encoding="utf-8")
        script.chmod(0o755)
        command = (
            'PYTHONDONTWRITEBYTECODE=1 "$WORKSPACE_ROOT"/.venv/bin/pytest'
            if selection == "script-virtual"
            else 'PYTHONDONTWRITEBYTECODE=1 "$WORKSPACE_ROOT/.venv/bin/pytest"'
            if selection == "script-whole-quoted"
            else "./fixture-task"
        )
    elif selection in {"direct", "standard-direct"}:
        command = "./.venv/bin/python -c "
    else:
        command = (
            ""
            if environment_name == ".venv"
            else f"UV_PROJECT_ENVIRONMENT={shlex.quote(environment_name)} "
        ) + (
            f"{shlex.quote(Path(_QUALIFICATION_UV).name)} run "
            f"{'' if synchronize or selection in {'project-no-bytecode', 'project-cd', 'project-scratch'} else '--no-sync '}python -c "
        )
        if selection in {"project-no-bytecode", "project-cd", "project-scratch"}:
            pytest_script = commands / "pytest"
            pytest_script.write_text(f"#!{commands / 'python'}\n" + program, encoding="utf-8")
            pytest_script.chmod(0o755)
            command = f"PYTHONDONTWRITEBYTECODE=1 {shlex.quote(Path(_QUALIFICATION_UV).name)} --no-cache run pytest"
            if selection == "project-cd":
                command = 'cd "$WORKSPACE_ROOT" && ' + command
            elif selection == "project-scratch":
                scratch = interaction.configuration_path.parent / "instruction-scratch"
                scratch.mkdir(mode=0o700, exist_ok=True)
                (scratch / "task.sh").write_text(
                    '#!/bin/sh\ncd "$WORKSPACE_ROOT"\n' + command + "\n", encoding="utf-8"
                )
                command = 'sh "$TMPDIR/task.sh"'
    arguments = {
        "command": command
        if selection
        in {
            "script",
            "script-virtual",
            "script-whole-quoted",
            "project-no-bytecode",
            "project-cd",
            "project-scratch",
        }
        else command + shlex.quote(program),
        "cwd": VirtualPath.WORKSPACE,
        "read_only": not synchronize
        and selection not in {"project-no-bytecode", "project-cd", "project-scratch"},
    }
    assert interaction.configuration_path is not None
    marker = interaction.configuration_path.parent / "instruction-scratch/entered"
    if not prepared:
        (prior,) = invoke_registered(
            public_registry,
            [{"tool": "run_command", "arguments": arguments}],
            model_paths=True,
            disable_runtime_discovery=True,
        )
        assert prior["problem"]["code"] == "process.nonzero_exit", json.dumps(prior, indent=2)
        assert len(interaction.prompts) == int(selection != "project")
        assert (commands / "python").resolve(strict=True) == interpreter
        assert canary.read_text(encoding="utf-8") == "outside-canary"
        assert secret.read_text(encoding="utf-8") == "private-canary"
        require_drained(generation)
        return
    if retarget:
        (stale,) = invoke_tools(
            (interaction, workspace, generation),
            [{"tool": "run_command", "arguments": arguments}],
            model_paths=True,
            retarget=(commands / "python", Path(_SYSTEM_SH)),
        )
        assert stale["problem"]["code"] == "sandbox.stale", stale
        assert len(interaction.prompts) == 1
        assert not marker.exists()
        require_drained(generation)
        return
    (denied,) = invoke_registered(
        public_registry,
        [{"tool": "run_command", "arguments": arguments}],
        model_paths=True,
        deny_sandbox=True,
    )
    assert denied["problem"]["code"] == "tool.denied", denied
    assert not marker.exists()
    (allowed,) = invoke_registered(
        public_registry, [{"tool": "run_command", "arguments": arguments}], model_paths=True
    )
    assert allowed["ok"], json.dumps(allowed, indent=2)
    assert allowed["result"]["stdout"]["content"].splitlines() == [
        str(environment),
        str(runtime),
        str(commands / "python"),
        "True",
    ]
    assert len(interaction.prompts) == 2
    assert marker.read_text(encoding="utf-8") == "entered"
    final_identity = (commands / "python").lstat()
    assert (
        final_identity.st_dev,
        final_identity.st_ino,
        os.readlink(commands / "python"),
    ) == selected_link
    assert any(str(runtime) in message for message in interaction.infos)
    assert canary.read_text(encoding="utf-8") == "outside-canary"
    assert secret.read_text(encoding="utf-8") == "private-canary"
    require_drained(generation)


def test_piped_make_prepares_its_selected_tool(public, monkeypatch):
    """A piped build prepares its selected tool while private files stay confined."""
    if _QUALIFICATION_UV is None or _SYSTEM_MAKE is None:
        pytest.skip("A managed tool and system build tool are required")
    interaction, workspace, generation = public
    (workspace / "Makefile").write_text(
        f".PHONY: test\ntest:\n\t{shlex.quote(Path(_QUALIFICATION_UV).name)} --version\n",
        encoding="utf-8",
    )
    private = workspace / ".ssh"
    private.mkdir()
    (private / "canary").write_text("private-canary", encoding="utf-8")
    monkeypatch.setenv("PATH", os.pathsep.join((str(Path(_QUALIFICATION_UV).parent), os.defpath)))
    result, denied = invoke_tools(
        public,
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": 'cd "$WORKSPACE_ROOT" && make test 2>&1 | tail -200',
                    "cwd": "$WORKSPACE_ROOT",
                },
            },
            {
                "tool": "run_command",
                "arguments": {
                    "command": 'cat "$WORKSPACE_ROOT/.ssh/canary"',
                    "cwd": "$WORKSPACE_ROOT",
                },
            },
        ],
        model_paths=True,
        seconds=20,
    )
    assert result["ok"], result
    assert "uv " in result["result"]["stdout"]["content"]
    assert not denied["ok"] and "private-canary" not in json.dumps(denied)
    assert interaction.prompts
    require_drained(generation)


def test_public_uv_runs_a_workspace_virtualenv_with_its_user_local_runtime(public, monkeypatch):
    """Run approved tools, nested session storage and pytest through the installed sandbox."""
    uv = _QUALIFICATION_UV
    if uv is None or _SYSTEM_SH is None:
        pytest.skip("The qualification tool and system shell are required")

    project_venv = Path(sys.prefix)
    config = project_venv / "pyvenv.cfg"
    if not config.is_file():
        pytest.skip("The qualification project has no active Python virtual environment")
    metadata = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            metadata[key.strip().casefold()] = value.strip()
    if metadata.get("implementation", "").casefold() != "cpython":
        pytest.skip("The qualification virtual environment does not use CPython")
    home_value = metadata.get("home", "")
    interpreter_name = (project_venv / "bin" / CPYTHON_EXECUTABLE_NAME).resolve(strict=True).name
    try:
        configured_home = Path(home_value).resolve(strict=True)
        interpreter = configured_home / interpreter_name
        account_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
        runtime_root = configured_home.parent
        version = metadata["version_info"].split(".")
        stdlib = runtime_root / "lib" / f"python{version[0]}.{version[1]}"
    except (KeyError, OSError, RuntimeError, ValueError):
        pytest.skip("The qualification Python runtime cannot be discovered")
    if (
        configured_home.name != "bin"
        or not runtime_root.is_relative_to(account_home)
        or not interpreter.is_file()
        or not stdlib.is_dir()
    ):
        pytest.skip("The active CPython runtime is not user-local")

    interaction, workspace, _generation = public
    venv_bin = workspace / ".venv/bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / CPYTHON_EXECUTABLE_NAME).symlink_to(interpreter)
    (workspace / ".venv/pyvenv.cfg").write_text(
        "\n".join(
            (
                f"home = {configured_home}",
                "implementation = CPython",
                f"version_info = {metadata['version_info']}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "pyproject.toml").write_text(
        "\n".join(
            (
                "[project]",
                'name = "native-runtime-probe"',
                'version = "0.0.0"',
                f'requires-python = ">={version[0]}.{version[1]}"',
                "dependencies = []",
                "",
                "[tool.uv]",
                "package = false",
                "",
            )
        ),
        encoding="utf-8",
    )
    site_packages = workspace / f".venv/lib/python{version[0]}.{version[1]}/site-packages"
    requirements = list(importlib.metadata.requires("loop") or ()) + [
        "pytest",
        "pytest-cov",
        "coverage",
        "pluggy",
        "iniconfig",
        "packaging",
        "pygments",
    ]
    copied = set()
    while requirements:
        requirement = Requirement(requirements.pop())
        name = requirement.name.lower().replace("_", "-")
        if name in copied or (
            requirement.marker and not requirement.marker.evaluate({"extra": ""})
        ):
            continue
        copied.add(name)
        distribution = importlib.metadata.distribution(name)
        requirements.extend(distribution.requires or ())
        for entry in distribution.files or ():
            relative = Path(entry)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.suffix in {".pyc", ".pth"}
            ):
                continue
            source = Path(distribution.locate_file(entry))
            if source.is_file():
                target = site_packages / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    pytest_entry = venv_bin / "pytest"
    pytest_entry.write_text(
        f"#!{venv_bin / CPYTHON_EXECUTABLE_NAME}\nimport sys\nimport pytest\nsys.exit(pytest.main())\n",
        encoding="utf-8",
    )
    pytest_entry.chmod(0o755)
    (workspace / "sample.py").write_text("def choose(value):\n    return 1 if value else 2\n")
    (workspace / "test_sample.py").write_text(
        "from sample import choose\n\n\ndef test_positive():\n    assert choose(True) == 1\n"
        "\n\ndef test_negative():\n    assert choose(False) == 2\n"
    )
    base_path = os.pathsep.join((str(Path(uv).parent), os.defpath))
    runner_name = Path(uv).name
    runner = shlex.quote(runner_name)
    for path_value in (base_path, os.pathsep.join((str(venv_bin), base_path))):
        monkeypatch.setenv("PATH", path_value)
        (lookup,) = invoke_tools(
            public,
            [
                {
                    "tool": "resolve_executable",
                    "arguments": {
                        "name": [runner_name, "fixture-missing-tool"],
                        "cwd": "$WORKSPACE_ROOT",
                    },
                }
            ],
            model_paths=True,
        )
        assert lookup["ok"], lookup
        present, missing = lookup["result"]["executables"]
        assert present["name"] == runner_name and present["status"] == "installed_on_host"
        assert missing == {"name": "fixture-missing-tool", "status": "absent_from_host_path"}
    (without_runtime,) = invoke_tools(
        public,
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": f"test ! -e {shlex.quote(str(interpreter))} && printf hidden",
                    "cwd": "$WORKSPACE_ROOT",
                    "read_only": True,
                },
            }
        ],
        model_paths=True,
    )
    assert without_runtime["ok"] and without_runtime["result"]["stdout"]["content"] == "hidden"
    assert not interaction.prompts

    (automatic,) = invoke_tools(
        public,
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": (
                        f'cd "$WORKSPACE_ROOT" && PYTHONDONTWRITEBYTECODE=1 {runner} '
                        "--no-cache run --no-sync pytest -p no:cacheprovider -q test_sample.py 2>&1"
                    ),
                    "cwd": "$WORKSPACE_ROOT",
                    "read_only": True,
                },
            }
        ],
        model_paths=True,
        seconds=30,
    )
    assert automatic["ok"], (
        json.dumps(automatic, indent=2),
        interaction.infos,
        interaction.prompts,
    )
    assert "2 passed" in automatic["result"]["stdout"]["content"]
    assert any(str(runtime_root) in message for message in interaction.infos)

    def run_with_interpreter(command, *, deny_sandbox=False, fresh_scratch=False, seconds=20):
        """Bind the selected interpreter and review its runtime for this exact command."""
        selected, result = invoke_tools(
            public,
            [
                {
                    "tool": "resolve_executable",
                    "arguments": {"name": CPYTHON_EXECUTABLE_NAME, "cwd": "$WORKSPACE_ROOT"},
                },
                {
                    "tool": "run_command",
                    "arguments": {
                        "command": command,
                        "cwd": "$WORKSPACE_ROOT",
                        "read_only": True,
                    },
                    "argument_sources": {
                        "executable_grant": {"result_index": 0, "result_key": "grant_reference"}
                    },
                },
            ],
            model_paths=True,
            seconds=seconds,
            deny_sandbox=deny_sandbox,
            fresh_scratch=fresh_scratch,
        )
        assert selected["ok"] and selected["result"]["grant_reference"]
        return result

    denied = run_with_interpreter('printf denied > "$TMPDIR/denied-marker"', deny_sandbox=True)
    assert denied["problem"]["code"] == "tool.denied"
    assert not (
        interaction.configuration_path.parent / "instruction-scratch/denied-marker"
    ).exists()

    result = run_with_interpreter(
        f'COVERAGE_FILE="{VirtualPath.TEMPORARY}/.coverage" {runner} '
        "run --no-sync pytest -p no:cacheprovider --cov=sample --cov-branch "
        "--cov-report=term-missing --cov-fail-under=100"
    )
    assert result["ok"], json.dumps(result, indent=2)
    assert result["result"]["exit_code"] == 0
    output = result["result"]["stdout"]["content"]
    assert "2 passed" in output and "100%" in output
    assert not (workspace / ".coverage").exists()
    assert interaction.prompts
    assert any(str(runtime_root) in message for message in interaction.infos)
    launcher = workspace / "project-launcher"
    launcher.write_text(
        f'#!{_SYSTEM_SH}\nexec {shlex.quote(CPYTHON_EXECUTABLE_NAME)} -I -B "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    (workspace / "runtime_probe.py").write_text("print('generic-runtime')\n", encoding="utf-8")
    generic = run_with_interpreter(
        f'{shlex.quote("./" + launcher.name)} "{VirtualPath.WORKSPACE}/runtime_probe.py"'
    )
    assert generic["ok"], generic
    assert generic["result"]["stdout"]["content"] == "generic-runtime\n"
    custom_bin = workspace / "alternate-environment/bin"
    custom_bin.mkdir(parents=True)
    custom_interpreter = custom_bin / "fixture-interpreter"
    custom_interpreter.symlink_to(interpreter)
    (custom_bin.parent / "pyvenv.cfg").write_text(
        (venv_bin.parent / "pyvenv.cfg").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with monkeypatch.context() as custom_environment:
        custom_environment.setenv("PATH", os.pathsep.join((str(custom_bin), base_path)))
        selected, custom = invoke_tools(
            public,
            [
                {
                    "tool": "resolve_executable",
                    "arguments": {"name": custom_interpreter.name, "cwd": VirtualPath.WORKSPACE},
                },
                {
                    "tool": "run_command",
                    "arguments": {
                        "command": f"{shlex.quote(custom_interpreter.name)} -I -B -c "
                        + shlex.quote("print('custom-interpreter-ok')"),
                        "cwd": VirtualPath.WORKSPACE,
                        "read_only": True,
                    },
                    "argument_sources": {
                        "executable_grant": {"result_index": 0, "result_key": "grant_reference"}
                    },
                },
            ],
            model_paths=True,
            seconds=20,
        )
    assert selected["ok"] and selected["result"]["grant_reference"]
    assert custom["ok"], custom
    assert custom["result"]["stdout"]["content"] == "custom-interpreter-ok\n"
    ruff = project_venv / "bin/ruff"
    assert ruff.is_file(), "The qualification environment needs its installed Ruff executable"
    shutil.copyfile(ruff, venv_bin / "ruff")
    (venv_bin / "ruff").chmod(0o755)
    lint = run_with_interpreter(
        f"{runner} run --no-sync ruff check --no-cache sample.py test_sample.py"
    )
    assert lint["ok"] and lint["result"]["exit_code"] == 0, json.dumps(lint)
    assert not (workspace / ".ruff_cache").exists()

    shutil.copytree(
        Path(__file__).resolve().parents[3] / "src/loop",
        site_packages / "loop",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (workspace / "outside").mkdir()
    (workspace / "nested_session.py").write_text(
        "import errno, os, pathlib, tempfile\n"
        "import loop\n"
        "from loop import SessionManager\n"
        "from loop.utils.path import ensure_private_directory\n"
        "w = pathlib.Path(os.environ['WORKSPACE_ROOT'])\n"
        "s = pathlib.Path(tempfile.gettempdir())\n"
        "assert pathlib.Path(loop.__file__).is_relative_to(w)\n"
        "try: os.open('/', os.O_RDONLY | os.O_DIRECTORY)\n"
        "except OSError as e: assert e.errno in (errno.EACCES, errno.EPERM)\n"
        "else: raise AssertionError('root listing was allowed')\n"
        "fd = os.open('/', os.O_PATH | os.O_DIRECTORY)\n"
        "os.close(fd)\n"
        "alias = s / 'outside-link'\n"
        "alias.symlink_to(w / 'outside', target_is_directory=True)\n"
        "try: ensure_private_directory(alias / 'escaped')\n"
        "except ValueError: pass\n"
        "else: raise AssertionError('symlink traversal was allowed')\n"
        "assert not (w / 'outside' / 'escaped').exists()\n"
        "manager = SessionManager()\n"
        "with manager.execution():\n"
        " p = manager.temporary_directory.get() / 'report'\n"
        " assert p.is_relative_to(s)\n"
        " p.write_text('nested-session', encoding='utf-8')\n"
        " assert p.read_text(encoding='utf-8') == 'nested-session'\n"
        "print('nested-session-ok')\n",
        encoding="utf-8",
    )
    nested = run_with_interpreter(
        f'{runner} run --no-sync python -I -B "{VirtualPath.WORKSPACE}/nested_session.py"'
    )
    assert nested["ok"], nested
    assert nested["result"]["stdout"]["content"] == "nested-session-ok\n"

    project = Path(__file__).resolve().parents[3]
    for relative in (
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/unit/__init__.py",
        "tests/unit/conftest.py",
        "tests/unit/test_isolation.py",
        "tests/unit/execution/__init__.py",
        "tests/unit/execution/sandbox/__init__.py",
        "tests/unit/execution/sandbox/helpers.py",
        "tests/unit/execution/sandbox/test_contracts.py",
        "tests/unit/execution/sandbox/linux/__init__.py",
        "tests/unit/execution/sandbox/linux/conftest.py",
        "tests/unit/execution/sandbox/linux/helpers.py",
        "tests/unit/execution/sandbox/linux/test_backend.py",
        "tests/unit/execution/sandbox/linux/test_installation.py",
        "tests/unit/execution/sandbox/linux/test_management.py",
        "tests/unit/execution/sandbox/linux/test_paths.py",
        "tests/unit/execution/sandbox/linux/test_policy.py",
        "tests/unit/execution/sandbox/linux/test_setup.py",
    ):
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / relative, target)
    nested_tests = run_with_interpreter(
        f"{runner} run --no-sync pytest -p no:cacheprovider -q -x --tb=short "
        "tests/unit/test_isolation.py::test_environment_and_current_directory_are_disposable "
        "tests/unit/execution/sandbox/test_contracts.py::"
        "test_request_binds_external_read_alias_and_rejects_retargeting "
        "tests/unit/execution/sandbox/linux/test_installation.py::"
        "test_bound_lifecycle_retires_only_its_original_record "
        "tests/unit/execution/sandbox/linux/test_installation.py::"
        "test_generation_publication_checks_logical_receiver_syntax "
        "tests/unit/execution/sandbox/linux/test_installation.py::"
        "test_generation_publication_exact_reuse_and_partial_completion "
        "tests/unit/execution/sandbox/linux/test_installation.py::"
        "test_drain_refuses_changed_service_before_closure "
        "tests/unit/execution/sandbox/linux/test_installation.py::"
        "test_drain_rejects_invalid_logical_receiver_before_closure "
        "tests/unit/execution/sandbox/linux/test_installation.py::"
        "test_drain_refuses_unknown_saved_template "
        "tests/unit/execution/sandbox/linux/test_installation.py::"
        "test_attachment_removal_withdraws_helper_before_exact_unload_and_retains_capacity "
        "tests/unit/execution/sandbox/linux/test_installation.py::"
        "test_attachment_removal_rejects_invalid_fixed_authority "
        "tests/unit/execution/sandbox/linux/test_installation.py::"
        "test_attachment_removal_refuses_unproved_literal_compilation "
        "tests/unit/execution/sandbox/linux/test_installation.py::"
        "test_reclamation_refuses_unsafe_fixed_inventory_before_deletion\\[mount\\] "
        "tests/unit/execution/sandbox/linux/test_management.py::"
        "test_original_record_status_export_and_uninstall "
        "tests/unit/execution/sandbox/linux/test_management.py::"
        "test_unrecorded_stage_is_cleaned_from_reviewed_bytes\\[clean\\] "
        "tests/unit/execution/sandbox/linux/test_paths.py::"
        "test_mount_arguments_preserves_grant_roles_protection_and_private_masks "
        "tests/unit/execution/sandbox/linux/test_paths.py::"
        "test_python_runtime_mounts_bind_only_a_selected_workspace_interpreter "
        "tests/unit/execution/sandbox/linux/test_paths.py::"
        "test_python_runtime_mounts_reject_unsafe_automatic_read_authority\\[protected\\] "
        "tests/unit/execution/sandbox/linux/test_policy.py::"
        "test_payload_policy_allows_only_the_bound_external_executable_file "
        "tests/unit/execution/sandbox/linux/test_setup.py::"
        "test_install_reviewed_fixed_scaffolding_and_repeat "
        "tests/unit/execution/sandbox/linux/test_backend.py::"
        "test_search_reuses_exact_policy_with_disjoint_selected_backing",
        fresh_scratch=True,
    )
    assert nested_tests["ok"], json.dumps(nested_tests, indent=2)
    assert "31 passed" in nested_tests["result"]["stdout"]["content"]

    shutil.copytree(
        project / "src",
        workspace / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        project / "tests/unit",
        workspace / "tests/unit",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(project / "tests/scheduling.py", workspace / "tests/scheduling.py")
    shutil.copyfile(project / "pyproject.toml", workspace / "pyproject.toml")
    full_suite = run_with_interpreter(
        f'COVERAGE_FILE="{VirtualPath.TEMPORARY}/.coverage" '
        f"PYTHONDONTWRITEBYTECODE=1 {runner} run --no-sync pytest "
        "-o addopts= -p no:cacheprovider -q -x --tb=short tests/unit "
        "--cov=loop --cov-branch --cov-report=term-missing:skip-covered --cov-fail-under=100 "
        '> "$TMPDIR/unit-report" 2>&1; status=$?; tail -n 40 "$TMPDIR/unit-report"; exit "$status"',
        fresh_scratch=True,
        seconds=180,
    )
    assert full_suite["ok"], json.dumps(full_suite, indent=2)
    assert re.search(
        r"(?m)^TOTAL\s+\d+\s+0\s+\d+\s+0\s+100%$",
        full_suite["result"]["stdout"]["content"],
    )
    assert not (workspace / ".coverage").exists()
    require_drained(public[2])


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("descendant", [False, True])
@pytest.mark.parametrize("network", [False, True])
def test_payload_cannot_submit_broader_request_over_pathname_ipc(
    public, tmp_path, read_only, alias, descendant, network, native_directory
):
    """Block direct and hardlink-alias requests from escalating receiver authority."""
    _, workspace, generation = public
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    listener = socket.socket()
    try:
        (workspace / ".ssh").mkdir()
        protected = workspace / ".ssh" / "canary"
        protected.write_text("keep-protected")
        secret = outside / "secret"
        secret.write_text("outside-secret")
        changed = outside / "changed"
        scratch = native_directory(Path("/tmp"))
        with listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            listener.settimeout(0.2)
            network_code = (
                "import socket;socket.create_connection(('127.0.0.1',"
                + str(listener.getsockname()[1])
                + "),timeout=1).sendall(b'escalated')"
            )
            broader = SandboxRequest.create(
                source="printf changed > .ssh/canary; cat "
                + shlex.quote(str(secret))
                + "; printf changed > "
                + shlex.quote(str(changed))
                + "; /usr/bin/python3 -I -c "
                + shlex.quote(network_code),
                cwd=workspace,
                workspace=workspace,
                read_roots=(outside,),
                write_roots=(workspace, outside),
                network=True,
                environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)},
                policy_version=NATIVE_POLICY_VERSION,
                deadline=time.monotonic() + 10,
                workspace_id="recursive-admission-negative",
            )
            payload = broader.to_wire()
            frame = b"LOOP\x01" + struct.pack("!I", len(payload)) + payload
            socket_path = Path("/run") / ("loopgen_" + generation + ".socket")
            if alias:
                exposed = workspace / "socket-alias"
                # This explicitly authorized fixture changes the listener's link count. Keep
                # ordinary test output guards closed to mutation of the original socket path.
                subprocess.run(
                    [
                        "/usr/bin/python3",
                        "-I",
                        "-B",
                        "-c",
                        "import os,sys;os.link(sys.argv[1],sys.argv[2])",
                        str(socket_path),
                        str(exposed),
                    ],
                    check=True,
                    timeout=5,
                )
                socket_path = exposed
            code = (
                "import os,socket,sys\n"
                + (
                    "pid=os.fork();\n"
                    "if pid:\n"
                    " _,status=os.waitpid(pid,0);sys.exit(os.waitstatus_to_exitcode(status))\n"
                    if descendant
                    else ""
                )
                + "try:\n s=socket.socket(socket.AF_UNIX);s.settimeout(1);s.connect("
                + repr(str(socket_path))
                + ");s.sendall("
                + repr(frame)
                + ");s.shutdown(socket.SHUT_WR);data=s.recv(4096);print('RESPONSE',data)\n"
                + "except OSError as error: print('REFUSED',error.errno)\n"
            )
            value = invoke(
                public,
                "/usr/bin/python3 -I -c " + shlex.quote(code),
                read_only=read_only,
                network=network,
                allow_general_network=network,
            )
            assert "result" in value and value["result"]["exit_code"] == 0, value
            output = value["result"]["stdout"]["content"]
            assert "REFUSED" in output or "RESPONSE" in output, "IPC attempt did not run"
            assert "outside-secret" not in output
            assert protected.read_text() == "keep-protected"
            assert not changed.exists()
            with pytest.raises(TimeoutError):
                listener.accept()
            require_drained(generation)
    finally:
        listener.close()
        for path in outside.iterdir():
            path.unlink()
        outside.rmdir()


def test_unix_socket_creation_is_unavailable_while_connected_pairs_and_pipes_work(public):
    """Unix host sockets and datagram pairs are denied, but connected pairs and pipes work."""
    code = (
        "import errno,os,socket\n"
        "for make in (lambda:socket.socket(socket.AF_UNIX),"
        "lambda:socket.socketpair(socket.AF_UNIX,socket.SOCK_DGRAM)):\n"
        " try: make()\n"
        " except OSError as e: assert e.errno == errno.EPERM\n"
        " else: raise AssertionError('Unix socket creation succeeded')\n"
        "for kind in (socket.SOCK_STREAM,socket.SOCK_SEQPACKET):\n"
        " a,b=socket.socketpair(socket.AF_UNIX,kind);a.sendall(b'pair');assert b.recv(4)==b'pair'\n"
        "r,w=os.pipe();os.write(w,b'pipe');print(os.read(r,4).decode())\n"
    )
    result = invoke(public, "/usr/bin/python3 -I -c " + shlex.quote(code))
    assert result["result"]["exit_code"] == 0, result
    assert result["result"]["stdout"]["content"] == "pipe\n"
    require_drained(public[2])


def test_public_large_command_negotiates_compact_wire_schema(public):
    """A real oversized duplicate source uses the compact request without replay."""
    result = invoke(public, "printf compact" + " " * 40000, read_only=True)
    assert result["ok"], json.dumps(result, indent=2)
    assert result["result"]["exit_code"] == 0
    assert result["result"]["stdout"]["content"] == "compact"
    require_drained(public[2])


@pytest.mark.parametrize("network", [False, True])
@pytest.mark.parametrize("kind", [socket.SOCK_STREAM, socket.SOCK_DGRAM, socket.SOCK_SEQPACKET])
def test_host_unix_socket_in_writable_workspace_is_unreachable(public, network, kind):
    """Host sockets and the datagram socket-pair bypass are denied under either network grant."""
    _interaction, workspace, generation = public
    endpoint = workspace / "host.sock"
    with socket.socket(socket.AF_UNIX, kind) as listener:
        listener.bind(str(endpoint))
        if kind != socket.SOCK_DGRAM:
            listener.listen()
        listener.settimeout(0.2)
        code = (
            "import errno,socket\n"
            "for make in (lambda:socket.socket(socket.AF_UNIX," + str(kind) + "),"
            "lambda:socket.socketpair(socket.AF_UNIX,socket.SOCK_DGRAM)):\n"
            " try:\n"
            "  value=make();s=value[0] if isinstance(value,tuple) else value\n"
            "  s.connect('host.sock');s.send(b'canary')\n"
            " except OSError as e: assert e.errno == errno.EPERM\n"
            " else: raise AssertionError('host socket connected')\n"
            "print('denied')\n"
        )
        if kind == socket.SOCK_STREAM:
            code += (
                "import ctypes\n"
                "a,b=socket.socketpair()\n"
                "for attempt in (lambda:a.connect('host.sock'),"
                "lambda:a.sendto(b'canary','host.sock'),"
                "lambda:a.sendmsg([b'canary'],[],0,'host.sock')):\n"
                " try: attempt()\n"
                " except OSError as e: assert e.errno == errno.EISCONN\n"
                " else: raise AssertionError('stream pair reached host')\n"
                "libc=ctypes.CDLL(None,use_errno=True);addr=ctypes.create_string_buffer(16)\n"
                "assert libc.connect(a.fileno(),addr,16)==-1 and ctypes.get_errno()==errno.EINVAL\n"
                "try: a.connect('host.sock')\n"
                "except OSError as e: assert e.errno == errno.EISCONN\n"
                "else: raise AssertionError('disconnected stream reached host')\n"
                "b.close()\n"
                "try: a.connect('host.sock')\n"
                "except OSError as e: assert e.errno == errno.EISCONN\n"
                "else: raise AssertionError('closed peer reached host')\n"
            )
        if kind == socket.SOCK_SEQPACKET:
            code += (
                "import ctypes\n"
                "libc=ctypes.CDLL(None,use_errno=True);addr=ctypes.create_string_buffer(16)\n"
                "a,b=socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET)\n"
                "try: a.connect('host.sock')\n"
                "except OSError as e: assert e.errno == errno.EISCONN\n"
                "else: raise AssertionError('sequenced pair reached host')\n"
                "assert a.sendto(b'pair','host.sock')==4 and b.recv(4)==b'pair'\n"
                "assert a.sendmsg([b'pair'],[],0,'host.sock')==4 and b.recv(4)==b'pair'\n"
                "assert libc.connect(a.fileno(),addr,16)==-1 and ctypes.get_errno()==errno.EINVAL\n"
                "b.close()\n"
                "try: a.connect('host.sock')\n"
                "except OSError as e: assert e.errno == errno.EISCONN\n"
                "else: raise AssertionError('sequenced pair reconnected')\n"
            )
        result = invoke(
            public,
            "/usr/bin/python3 -I -c " + shlex.quote(code),
            network=network,
            allow_general_network=network,
        )
        assert result["ok"], json.dumps(result, indent=2)
        assert result["result"]["exit_code"] == 0, result
        assert result["result"]["stdout"]["content"] == "denied\n"
        with pytest.raises(TimeoutError):
            listener.recv(10) if kind == socket.SOCK_DGRAM else listener.accept()
    require_drained(generation)


@pytest.mark.parametrize("network", [False, True])
def test_host_vsock_listener_is_unreachable(public, network):
    """A host VSOCK listener remains unreachable regardless of the network grant."""
    listener = None
    try:
        listener = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        listener.bind((socket.VMADDR_CID_ANY, socket.VMADDR_PORT_ANY))
        listener.listen(2)
        listener.settimeout(0.2)
        port = listener.getsockname()[1]
        with socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM) as control:
            control.settimeout(1)
            control.connect((socket.VMADDR_CID_LOCAL, port))
            with listener.accept()[0] as accepted:
                control.sendall(b"control")
                assert accepted.recv(7) == b"control"
    except OSError as error:
        if listener is not None:
            listener.close()
        if error.errno in (
            errno.EAFNOSUPPORT,
            errno.EPROTONOSUPPORT,
            errno.ENODEV,
            errno.EPERM,
            errno.EADDRNOTAVAIL,
            errno.ENETUNREACH,
            errno.ECONNREFUSED,
        ):
            pytest.skip(f"Host VSOCK positive control is unavailable: {error}")
        raise
    with listener:
        code = (
            "import errno,socket\n"
            "try: socket.socket(socket.AF_VSOCK,socket.SOCK_STREAM)\n"
            "except OSError as e: assert e.errno == errno.EPERM\n"
            "else: raise AssertionError('host VSOCK is available')\n"
            "print('denied')\n"
        )
        result = invoke(
            public,
            "/usr/bin/python3 -I -c " + shlex.quote(code),
            network=network,
            allow_general_network=network,
        )
        assert result["ok"] and result["result"]["stdout"]["content"] == "denied\n", result
        with pytest.raises(TimeoutError):
            listener.accept()
    require_drained(public[2])


@pytest.mark.parametrize("network", [False, True])
def test_host_netlink_usersock_is_unreachable(public, network):
    """Host userspace netlink IPC remains denied even with an approved network grant."""
    with socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, socket.NETLINK_USERSOCK) as listener:
        listener.bind((0, 0))
        listener.settimeout(0.2)
        port = listener.getsockname()[0]
        with socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, socket.NETLINK_USERSOCK) as control:
            control.bind((0, 0))
            control.sendto(b"control", (port, 0))
            assert listener.recv(7) == b"control"
        code = (
            "import errno,socket\n"
            "try: socket.socket(socket.AF_NETLINK,socket.SOCK_RAW,socket.NETLINK_USERSOCK)\n"
            "except OSError as e: assert e.errno == errno.EPERM\n"
            "else: raise AssertionError('host netlink IPC is available')\n"
            "route=socket.socket(socket.AF_NETLINK,socket.SOCK_RAW,socket.NETLINK_ROUTE)\n"
            "route.bind((0,0));route.close();print('denied')\n"
        )
        result = invoke(
            public,
            "/usr/bin/python3 -I -c " + shlex.quote(code),
            network=network,
            allow_general_network=network,
        )
        assert result["ok"] and result["result"]["stdout"]["content"] == "denied\n", result
        with pytest.raises(TimeoutError):
            listener.recv(7)
    require_drained(public[2])


def test_payload_cannot_trace_namespace_reaper(public):
    """The namespace reaper retains enforcement and does not expose its setup file."""
    code = (
        "import ctypes,errno,os\n"
        "status=open('/proc/1/status').read()\n"
        "assert 'Seccomp:\\t2' in status and 'CapEff:\\t0000000000000000' in status\n"
        "libc=ctypes.CDLL(None,use_errno=True)\n"
        "assert libc.ptrace(16,1,0,0)==-1 and ctypes.get_errno()==errno.EPERM\n"
        "try: target=os.readlink('/proc/1/fd/0')\n"
        "except FileNotFoundError: target=''\n"
        "if target.startswith('/tmp/#') and target.endswith(' (deleted)'):\n"
        " for flags in (os.O_RDONLY,os.O_WRONLY,os.O_RDWR):\n"
        "  try: fd=os.open('/proc/1/fd/0',flags)\n"
        "  except OSError as e: assert e.errno in (errno.EACCES,errno.EPERM)\n"
        "  else: os.close(fd);raise AssertionError('reaper setup file is accessible')\n"
        "print('confined')\n"
    )
    result = invoke(public, "/usr/bin/python3 -I -c " + shlex.quote(code))
    assert result["result"]["exit_code"] == 0, result
    assert result["result"]["stdout"]["content"] == "confined\n"
    require_drained(public[2])


def test_near_deadline_success_and_large_bounded_delivery(public):
    """A successful child retains both bounded streams even when delivery consumes cleanup time."""
    code = (
        "import sys,time;time.sleep(1);sys.stdout.write('x'*1000000);sys.stderr.write('y'*1000000)"
    )
    result = invoke(public, "/usr/bin/python3 -I -c " + shlex.quote(code), read_only=True)
    assert result["result"]["exit_code"] == 0
    for name in ("stdout", "stderr"):
        stream = result["result"][name]
        assert stream["captured_bytes"] == 1000000
        assert stream["capture_complete"] is True
        assert stream["discarded_characters"] == 0
    require_drained(public[2])


@pytest.mark.parametrize("exit_code", [0, 7])
def test_public_output_preserves_carriage_returns(public, exit_code):
    """Installed command output preserves CR, CRLF and LF on success and failure."""
    source = "printf 'a\\rb\\r\\nc\\n'; printf 'a\\rb\\r\\nc\\n' >&2"
    result = invoke(public, f"{source}; exit {exit_code}", read_only=True)
    output = result["result"] if exit_code == 0 else result["problem"]["metadata"]
    if exit_code:
        assert result["problem"]["code"] == "process.nonzero_exit"
    for name in ("stdout", "stderr"):
        assert output[name]["content"] == "a\rb\r\nc\n"
        assert output[name]["captured_bytes"] == 7
    require_drained(public[2])


def test_public_command_and_file_tool_preserve_path_like_content(public):
    """Linux command streams and file reads retain identical path-like synthetic data."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    _, workspace, generation = public
    physical = str(workspace)
    content = (
        json.dumps(
            {
                "workspace": physical,
                "sibling": physical + "-extra/file",
                "url": "https://example.test" + physical + "/file",
                "quoted": shlex.quote(physical),
                "unicode": "é ☃",
                "encoded": base64.b64encode(physical.encode()).decode(),
            },
            ensure_ascii=False,
        )
        + "\r\n"
    )
    source = workspace / "path-content.json"
    source.write_text(content, encoding="utf-8", newline="")
    logical = f"{VirtualPath.WORKSPACE}/{source.name}"
    command = f'{shlex.quote(_SYSTEM_CAT)} "{logical}"'
    quoted_source = "printf '%s\\n' '$WORKSPACE_ROOT/quoted'; " + shlex.join(
        ("/bin/sh", "-c", "printf %s '$TMPDIR/embedded'")
    )
    read, success, failure, literal, quoted = invoke_tools(
        public,
        [
            {"tool": "read_text_file", "arguments": {"path": logical}},
            {
                "tool": "run_command",
                "arguments": {"command": command, "cwd": VirtualPath.WORKSPACE, "read_only": True},
            },
            {
                "tool": "run_command",
                "arguments": {
                    "command": f"{command}; {command} >&2; exit 7",
                    "cwd": VirtualPath.WORKSPACE,
                    "read_only": True,
                },
            },
            {
                "tool": "run_command",
                "arguments": {
                    "command": f"printf %s {shlex.quote(content)}",
                    "cwd": VirtualPath.WORKSPACE,
                    "read_only": True,
                },
            },
            {
                "tool": "run_command",
                "arguments": {
                    "command": quoted_source,
                    "cwd": VirtualPath.WORKSPACE,
                    "read_only": True,
                },
            },
        ],
        model_paths=True,
    )
    assert read["ok"] and read["result"]["path"] == logical
    assert read["result"]["content"] == content
    assert success["ok"] and success["result"]["stdout"]["content"] == content
    assert failure["problem"]["code"] == "process.nonzero_exit"
    assert "executable_hint" not in failure["problem"]["metadata"]
    assert all(
        failure["problem"]["metadata"][stream]["content"] == content
        for stream in ("stdout", "stderr")
    )
    assert literal["ok"] and literal["result"]["stdout"]["content"] == content
    assert quoted["ok"] and quoted["result"]["stdout"]["content"] == (
        "$WORKSPACE_ROOT/quoted\n$TMPDIR/embedded"
    )
    require_drained(generation)


def test_public_system_hostname_and_default_trust_match_host(public):
    """Approved hostname clients retain system hosts/resolver data and default TLS anchors."""
    code = (
        "import hashlib,json,socket,ssl;from pathlib import Path;"
        "print(json.dumps({'localhost':sorted(set(x[4][0] for x in "
        "socket.getaddrinfo('localhost',80))),"
        "'certificates':len(ssl.create_default_context().get_ca_certs()),"
        "'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() "
        "for p in (Path('/etc/hosts'),Path('/etc/resolv.conf'))}}))"
    )
    direct = json.loads(subprocess.check_output(["/usr/bin/python3", "-I", "-c", code], text=True))
    assert direct["certificates"] > 0
    result = invoke(
        public,
        "/usr/bin/python3 -I -c " + shlex.quote(code),
        network=True,
        allow_general_network=True,
    )
    assert result["ok"], result
    assert json.loads(result["result"]["stdout"]["content"]) == direct
    require_drained(public[2])


@pytest.mark.parametrize("transport", ["tcp", "udp"])
def test_public_dns_client_uses_owned_network_endpoint(public, transport):
    """DNS over TCP works; clients requiring descriptor-receiving UDP calls are refused."""
    dig = shutil.which("dig", path=os.defpath)
    if dig is None:
        pytest.skip("The system dig client is unavailable")
    kind = socket.SOCK_STREAM if transport == "tcp" else socket.SOCK_DGRAM
    with socket.socket(socket.AF_INET, kind) as server:
        server.bind(("127.0.0.1", 0))
        server.settimeout(5)
        if transport == "tcp":
            server.listen()

        def answer():
            """Return one bounded synthetic DNS answer to the owned client."""
            if transport == "tcp":
                connection, _ = server.accept()
                connection.settimeout(2)
                header = b""
                while len(header) < 2:
                    data = connection.recv(2 - len(header))
                    if not data:
                        raise ValueError("DNS client closed its request early")
                    header += data
                size = int.from_bytes(header, "big")
                request = b""
                while len(request) < size:
                    data = connection.recv(size - len(request))
                    if not data:
                        raise ValueError("DNS client closed its request early")
                    request += data
            else:
                request, address = server.recvfrom(512)
            assert len(request) >= 12 and request[4:6] == b"\x00\x01"
            response = (
                request[:2]
                + b"\x81\x80\x00\x01\x00\x01\x00\x00\x00\x00"
                + request[12:]
                + b"\xc0\x0c\x00\x01\x00\x01\x00\x00\x00\x01\x00\x04\x7f\x00\x00\x2a"
            )
            if transport == "tcp":
                with connection:
                    connection.sendall(len(response).to_bytes(2, "big") + response)
            else:
                server.sendto(response, address)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(answer)
            command = shlex.join(
                (
                    dig,
                    "@127.0.0.1",
                    "-p",
                    str(server.getsockname()[1]),
                    "+short",
                    "+noedns",
                    "+tries=1",
                    "+time=2",
                    *(("+tcp",) if transport == "tcp" else ()),
                    "loop-test.invalid",
                    "A",
                )
            )
            result = invoke(
                public, command, network=True, read_only=True, allow_general_network=True
            )
            future.result(timeout=5)
        if transport == "tcp":
            assert result["ok"] and result["result"]["stdout"]["content"] == "127.0.0.42\n", result
        else:
            assert result["problem"]["code"] == "process.nonzero_exit", result
            assert result["problem"]["metadata"]["exit_code"] == 9
            assert "permission denied" in result["problem"]["metadata"]["stdout"]["content"]
    require_drained(public[2])


def test_public_default_verified_tls_to_owned_fixture(
    public, public_workspace_root, request, native_directory
):
    """Verify an owned HTTPS chain, hostname and offline boundary without changing host trust."""
    workspace = public[1]
    _workspace_root, _permissions_root, external_root = public_workspace_root
    private = native_directory(
        external_root,
        name=_test_directory_name(request.node.nodeid),
    )
    ca = private / "ca.pem"
    key = private / "server.key"
    certificate = private / "server.pem"
    extensions = private / "extensions"
    extensions.write_text(
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\nsubjectAltName=DNS:localhost\n"
        "subjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n"
    )
    commands = (
        [
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=Loop owned fixture CA",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "subjectKeyIdentifier=hash",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
            "-keyout",
            str(private / "ca.key"),
            "-out",
            str(ca),
        ],
        [
            "req",
            "-new",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-subj",
            "/CN=localhost",
            "-keyout",
            str(key),
            "-out",
            str(private / "server.csr"),
        ],
        [
            "x509",
            "-req",
            "-days",
            "1",
            "-in",
            str(private / "server.csr"),
            "-CA",
            str(ca),
            "-CAkey",
            str(private / "ca.key"),
            "-CAcreateserial",
            "-extfile",
            str(extensions),
            "-out",
            str(certificate),
        ],
    )
    for arguments in commands:
        subprocess.run(
            ["/usr/bin/openssl", *arguments], capture_output=True, check=True, timeout=10
        )
    trusted = workspace / "owned-ca.pem"
    trusted.write_bytes(ca.read_bytes())

    class Handler(http.server.BaseHTTPRequestHandler):
        """Return fixed fixture bytes without logging request data."""

        def do_GET(self):
            """Serve the bounded owned response."""
            body = b"loop-owned-https"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            """Discard the fixture server's ordinary access log."""

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:

        def source(hostname):
            """Build a default verified client for this owned fixture port."""
            url = f"https://{hostname}:{server.server_port}/"
            return (
                "import hashlib,ssl,urllib.request;context=ssl.create_default_context();"
                "assert context.check_hostname and context.verify_mode==ssl.CERT_REQUIRED;"
                f"response=urllib.request.urlopen({url!r},context=context,timeout=1);"
                "print(hashlib.sha256(response.read(65536)).hexdigest())"
            )

        code = source("localhost")
        direct = subprocess.check_output(
            ["/usr/bin/python3", "-I", "-c", code],
            text=True,
            env={**os.environ, "SSL_CERT_FILE": str(trusted)},
            timeout=5,
        ).strip()
        assert direct == hashlib.sha256(b"loop-owned-https").hexdigest()
        prefix = "SSL_CERT_FILE=" + shlex.quote(str(trusted)) + " /usr/bin/python3 -I -c "
        result = invoke(
            public, prefix + shlex.quote(code), network=True, allow_general_network=True
        )
        assert result["ok"], result
        assert result["result"]["stdout"]["content"].strip() == direct
        wrong_name = invoke(
            public,
            prefix + shlex.quote(source("127.0.0.1")),
            network=True,
            allow_general_network=True,
        )
        assert not wrong_name["ok"]
        assert "CERTIFICATE_VERIFY_FAILED" in wrong_name["problem"]["metadata"]["stderr"]["content"]
        untrusted = invoke(
            public,
            "/usr/bin/python3 -I -c " + shlex.quote(code),
            network=True,
            allow_general_network=True,
        )
        assert not untrusted["ok"]
        assert "CERTIFICATE_VERIFY_FAILED" in untrusted["problem"]["metadata"]["stderr"]["content"]
        offline = invoke(public, prefix + shlex.quote(code), network=False)
        assert not offline["ok"]
        assert direct not in json.dumps(offline)
        protected = invoke(
            public, "cat " + shlex.quote(str(key)), network=True, allow_general_network=True
        )
        assert not protected["ok"]
        assert "PRIVATE KEY" not in json.dumps(protected)
        require_drained(public[2])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_ordinary_approved_broader_request_is_executable(
    installation, public_workspace_root, request, native_directory, environment_patch
):
    """Approved outside access and network requests provide a valid escalation control."""
    from loop.execution.sandbox import SandboxOutcome

    backend, generation = installation
    workspace_root, _permissions_root, _external_root = public_workspace_root
    fixture_name = _test_directory_name(request.node.nodeid)
    workspace = native_directory(workspace_root, name=fixture_name)
    outside = native_directory(workspace_root, name=fixture_name + "-outside")
    unrelated = native_directory(workspace_root, name=fixture_name + "-unrelated")
    (outside / "secret").write_text("broader-control")
    (unrelated / "secret").write_text("unrelated-canary")
    (workspace / ".ssh").mkdir()
    (workspace / ".ssh/key").write_text("private-canary")
    environment_patch.setattr(tempfile, "tempdir", "/tmp")
    scratch = native_directory(Path("/tmp"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(2)
        code = (
            "import socket;socket.create_connection(('127.0.0.1',"
            + str(listener.getsockname()[1])
            + "),timeout=1).sendall(b'approved')"
        )
        request = SandboxRequest.create(
            source="cat "
            + shlex.quote(str(outside / "secret"))
            + "; printf approved > "
            + shlex.quote(str(outside / "changed"))
            + "; /usr/bin/python3 -I -c "
            + shlex.quote(code),
            cwd=workspace.resolve(),
            workspace=workspace.resolve(),
            read_roots=(outside.resolve(),),
            write_roots=(workspace.resolve(), outside.resolve()),
            network=True,
            environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(Path(scratch).resolve())},
            policy_version=backend.policy_version,
            deadline=time.monotonic() + 10,
            workspace_id="approved-broader-control",
        )
        result = backend.run(request)
        assert result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0
        assert result.stdout == "broader-control"
        assert (outside / "changed").read_text() == "approved"
        with listener.accept()[0] as connection:
            assert connection.recv(8) == b"approved"
        baseline = profile_inventory(generation)
        reordered = SandboxRequest.create(
            source=(
                f"if cat {shlex.quote(str(unrelated / 'secret'))} >/dev/null 2>&1; then exit 31; fi; "
                "if printf bad > .ssh/key 2>/dev/null; then exit 32; fi; "
                f"printf ordered > {shlex.quote(str(outside / 'ordered'))}; "
                "printf workspace > ordered-workspace; printf ordered-ok"
            ),
            cwd=workspace.resolve(),
            workspace=workspace.resolve(),
            read_roots=(outside.resolve(),),
            write_roots=(outside.resolve(), workspace.resolve()),
            network=True,
            environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(Path(scratch).resolve())},
            policy_version=backend.policy_version,
            deadline=time.monotonic() + 10,
            workspace_id="approved-broader-control",
        )
        reordered_result = backend.run(reordered)
        assert reordered_result.outcome is SandboxOutcome.COMPLETED, reordered_result
        assert reordered_result.exit_code == 0 and reordered_result.stdout == "ordered-ok"
        assert (outside / "ordered").read_text() == "ordered"
        assert (workspace / "ordered-workspace").read_text() == "workspace"
        assert (workspace / ".ssh/key").read_text() == "private-canary"
        assert (unrelated / "secret").read_text() == "unrelated-canary"
        assert_equivalent_profiles(baseline, profile_inventory(generation))
        require_drained(generation)


def test_public_automatic_discovery_preserves_a_scoped_reference(
    public_registry, public_workspace_root, request, native_directory, monkeypatch
):
    """A full lookup cache retains a scoped grant without exposing an unrelated directory."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None or _SYSTEM_PRINTF is None:
        pytest.skip("System shell and file tools are required")
    interaction, _workspace, generation, _ = public_registry
    _, _, external_root = public_workspace_root
    name = _test_directory_name(request.node.nodeid)
    tools = native_directory(external_root, name=name)
    unrelated = native_directory(external_root, name=name + "-other")
    support = tools / "support"
    support.write_text("approved-support", encoding="utf-8")
    secret = unrelated / "secret"
    secret.write_text("outside-private-canary", encoding="utf-8")
    tool = tools / "fixture-builder"
    tool.write_text(
        f"#!{_SYSTEM_SH}\n{shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(support))}\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(secret))} >/dev/null 2>&1; "
        "then exit 9; fi\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(tools), os.defpath)))
    lookup = {
        "tool": "resolve_executable",
        "arguments": {"name": tool.name, "cwd": VirtualPath.WORKSPACE},
    }
    command = {
        "tool": "run_command",
        "arguments": {"command": tool.name, "cwd": VirtualPath.WORKSPACE, "read_only": True},
        "argument_sources": {
            "executable_grant": {"result_index": 0, "result_key": "grant_reference"}
        },
    }

    results = invoke_registered(public_registry, [lookup] * 128 + [command], model_paths=True)

    assert all(item["ok"] and item["result"]["grant_reference"] for item in results[:-1])
    assert results[-1]["ok"], results[-1]
    assert results[-1]["result"]["stdout"]["content"] == "approved-support"
    assert secret.read_text(encoding="utf-8") == "outside-private-canary"
    assert len(interaction.prompts) == 1
    require_drained(generation)


@pytest.mark.parametrize("absolute_awk", [False, True])
def test_public_make_runs_unrelated_system_tools_in_one_call(public, absolute_awk):
    """One build request runs separate Perl and awk programs inside the Linux sandbox."""
    if (
        _SYSTEM_MAKE is None
        or not Path("/usr/bin/perl").is_file()
        or not Path("/usr/bin/awk").is_file()
    ):
        pytest.skip("System make, Perl and awk are required")
    interaction, workspace, generation = public
    awk = "/usr/bin/awk" if absolute_awk else "awk"
    (workspace / "Makefile").write_text(
        f".PHONY: test\ntest:\n\t@perl -e 'print \"perl\"'\n\t@{awk} 'BEGIN {{ print 23 }}'\n",
        encoding="utf-8",
    )
    result = invoke(public, f"{_SYSTEM_MAKE} test", read_only=True)
    assert result["ok"], json.dumps(result, indent=2)
    assert result["result"]["stdout"]["content"] == "perl23\n"
    assert not interaction.prompts
    require_drained(generation)


def test_absolute_system_link_within_runtime_does_not_require_path_entry(public, monkeypatch):
    """A system symlink that stays inside the runtime works with an unrelated PATH."""
    python = Path("/usr/bin/python3")
    if _SYSTEM_MAKE is None or not python.is_symlink():
        pytest.skip("Requires system make and a system Python symlink")
    interaction, workspace, generation = public
    (workspace / "Makefile").write_text(
        ".PHONY: test\ntest:\n\t@/usr/bin/python3 -c 'assert True'\n", encoding="utf-8"
    )
    monkeypatch.setenv("PATH", "/loop-no-such-path")
    result = invoke(public, f"{_SYSTEM_MAKE} test", read_only=True)
    assert result["ok"], json.dumps(result, indent=2)
    assert result["result"]["exit_code"] == 0
    assert not interaction.prompts
    require_drained(generation)


def test_selected_system_alternative_does_not_expose_other_entries(public):
    """The selected system link works without mounting its unrelated alternatives neighbors."""
    if not _ALTERNATIVE_CANARY:
        pytest.skip("Set LOOP_NATIVE_ALTERNATIVE_CANARY to the owned system-link fixture")
    canary = Path(_ALTERNATIVE_CANARY)
    assert canary.parent == Path("/etc/alternatives")
    assert canary.name.startswith("loop-native-alternative-canary-")
    assert canary.is_symlink() and canary.is_file() and canary.lstat().st_uid == 0
    _interaction, workspace, generation = public
    probe = workspace / "alternative-probe"
    probe.write_text(
        "#!/usr/bin/python3\nimport errno,os\n"
        f"path={str(canary)!r}\n"
        "for operation in (os.lstat,os.readlink):\n"
        " try: operation(path)\n"
        " except OSError as error: assert error.errno in (errno.ENOENT,errno.EACCES,errno.EPERM)\n"
        " else: raise AssertionError('unrelated alternative visible')\n"
        "try: entries=os.listdir(os.path.dirname(path))\n"
        "except OSError as error: assert error.errno in (errno.ENOENT,errno.EACCES,errno.EPERM)\n"
        "else: assert os.path.basename(path) not in entries\n",
        encoding="utf-8",
    )
    probe.chmod(0o755)
    (workspace / "Makefile").write_text(
        ".PHONY: test\ntest:\n\t@awk 'BEGIN { print 23 }'\n\t@./alternative-probe\n",
        encoding="utf-8",
    )
    result = invoke(public, f"{_SYSTEM_MAKE} test", read_only=True)
    assert result["ok"], json.dumps(result, indent=2)
    assert result["result"]["stdout"]["content"] == "23\n"
    assert canary.is_symlink()
    require_drained(generation)


def test_selected_system_alternative_preserves_approved_directory(public):
    """An explicitly approved alternatives directory remains usable without alias conflict."""
    interaction, workspace, generation = public
    (workspace / "Makefile").write_text(
        ".PHONY: test\ntest:\n\t@awk 'BEGIN { print 23 }'\n", encoding="utf-8"
    )
    result = invoke(
        public, f"{_SYSTEM_MAKE} test", read_only=True, read_roots=["/etc/alternatives"]
    )
    assert result["ok"], json.dumps(result, indent=2)
    assert result["result"]["stdout"]["content"] == "23\n"
    assert len(interaction.prompts) == 1
    require_drained(generation)


def test_system_executable_with_intermediate_directory_alias(public, monkeypatch):
    """A selected system tool works through a protected intermediate directory link."""
    if not _SYSTEM_ALIAS_FIXTURE:
        pytest.skip("Set LOOP_NATIVE_SYSTEM_ALIAS_FIXTURE to the owned system-link fixture")
    root = Path(_SYSTEM_ALIAS_FIXTURE)
    assert root.parent.parent == Path("/usr/local/lib")
    assert root.parent.name.startswith("loop-native-system-alias-") and root.name == "bin"
    executable = root / "loop-parent-tool"
    assert executable.is_symlink() and executable.lstat().st_uid == 0
    assert executable.resolve(strict=True).is_file()
    interaction, workspace, generation = public
    (workspace / "Makefile").write_text(
        ".PHONY: test\ntest:\n\t@loop-parent-tool\n", encoding="utf-8"
    )
    monkeypatch.setenv("PATH", os.pathsep.join((str(root), "/usr/bin", "/bin")))
    result = invoke(public, f"{_SYSTEM_MAKE} test", read_only=True)
    assert result["ok"], json.dumps(result, indent=2)
    assert result["result"]["stdout"]["content"] == "parent-chain"
    assert not interaction.prompts
    require_drained(generation)


def test_mismatched_receiver_alias_capability_refuses_before_effects(
    installation, public, native_directory, monkeypatch
):
    """A mismatched installed alias contract is refused before the selected command runs."""
    from loop.execution.sandbox.linux import runtime as linux_runtime

    backend, generation = installation
    _, workspace, _ = public
    spelling = Path("/usr/bin/awk")
    if not spelling.is_symlink() or spelling.readlink().parent != Path("/etc/alternatives"):
        pytest.skip("Requires a system awk selected through /etc/alternatives")
    resolved = spelling.resolve(strict=True)
    details = resolved.stat()
    scratch = native_directory(Path("/tmp"))
    request = SandboxRequest.create(
        source="printf entered > entered",
        cwd=workspace,
        workspace=workspace,
        read_roots=(),
        write_roots=(workspace,),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)},
        policy_version=backend.policy_version,
        deadline=time.monotonic() + 15,
        workspace_id="alias-capability-refusal",
        executable_identities=(("awk", spelling, resolved, details.st_dev, details.st_ino),),
    )
    retained_capabilities = linux_runtime._wire_capabilities

    def without_system_aliases(installed, deadline):
        """Represent an authenticated receiver without system-alias support."""
        capabilities = retained_capabilities(installed, deadline)
        capabilities["NATIVE_REQUEST_FEATURES"] = ("workspace-executable-identities-v1",)
        return capabilities

    monkeypatch.setattr(linux_runtime, "_wire_capabilities", without_system_aliases)
    result = backend.run(request)
    assert result.outcome is SandboxOutcome.UNSUPPORTED and not result.possible_effects, result
    assert "cannot bind executable aliases" in result.detail
    assert not (workspace / "entered").exists()
    require_drained(generation)


@pytest.mark.parametrize("source", ["make test", 'cd "$WORKSPACE_ROOT"; make test 2>&1'])
@pytest.mark.parametrize(
    ("policy_case", "retarget"),
    [
        ("new", False),
        ("new", True),
        ("default_saved", False),
        ("custom_saved", False),
    ],
)
def test_public_make_prepares_a_literal_indirect_tool_in_one_call(
    public_registry,
    public_workspace_root,
    request,
    native_directory,
    monkeypatch,
    policy_case,
    retarget,
    source,
):
    """A literal recipe asks once unless a custom legacy ceiling also needs review."""
    if _SYSTEM_MAKE is None or _SYSTEM_SH is None:
        pytest.skip("System build tool and shell are required")
    interaction, workspace, generation, _ = public_registry
    if policy_case != "new":
        payload = (
            "version: 1\nlimits:\n  readable_roots: [workspace]\n"
            if policy_case == "custom_saved"
            else "version: 1\n"
        )
        interaction.configuration_path.write_text(payload, encoding="utf-8")
    _, _, external_root = public_workspace_root
    root = native_directory(external_root, name=_test_directory_name(request.node.nodeid))
    outside = native_directory(
        external_root, name=_test_directory_name(request.node.nodeid) + "-outside"
    )
    canary = outside / "secret"
    canary.write_text("outside-private-canary", encoding="utf-8")
    target = root / "original-tool"
    target.write_text(
        f"#!{_SYSTEM_SH}\n"
        f"if cat {shlex.quote(str(canary))} >/dev/null 2>&1; then exit 9; fi\n"
        "printf indirect-ready",
        encoding="utf-8",
    )
    target.chmod(0o755)
    tool = root / "fixture-indirect-tool"
    tool.symlink_to(target)
    (workspace / "Makefile").write_text(
        ".PHONY: test\ntest:\n\t@fixture-indirect-tool\n", encoding="utf-8"
    )
    workspace_bin = workspace / ".venv/bin"
    workspace_bin.mkdir(parents=True)
    monkeypatch.setenv("PATH", os.pathsep.join((str(workspace_bin), str(root), os.defpath)))

    (result,) = invoke_tools(
        (interaction, workspace, generation),
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": source,
                    "cwd": VirtualPath.WORKSPACE,
                    "read_only": True,
                },
            }
        ],
        model_paths=True,
        retarget=(tool, Path(_SYSTEM_SH)) if retarget else None,
    )

    if retarget:
        assert result["problem"]["code"] == "sandbox.stale", result
        assert len(interaction.prompts) == 1
        require_drained(generation)
        return
    assert result["ok"], json.dumps(
        {"result": result, "prompts": interaction.prompts, "infos": interaction.infos}, indent=2
    )
    assert result["result"]["stdout"]["content"] == "indirect-ready"
    assert canary.read_text(encoding="utf-8") == "outside-private-canary"
    assert len(interaction.prompts) == (2 if policy_case == "custom_saved" else 1)
    if policy_case == "custom_saved":
        policy = load_yaml(interaction.configuration_path.read_text(encoding="utf-8"))
        assert policy["limit_classifications"]["readable_roots"] == "approval"
        assert policy["limits"]["readable_roots"] == ["workspace"]
    assert any(str(root) in message for message in interaction.infos)
    require_drained(generation)


def test_public_make_prepares_workspace_venv_runtime_in_one_call(public, monkeypatch):
    """A literal recipe selects and authorizes its workspace interpreter before launch."""
    if _SYSTEM_MAKE is None:
        pytest.skip("System make is required")
    interaction, workspace, generation = public
    environment = workspace / ".venv"
    created = subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    (workspace / "Makefile").write_text(
        '.PHONY: test\ntest:\n\t@./.venv/bin/python -c "print(1)"\n'
    )
    monkeypatch.setenv("PATH", os.defpath)
    (result,) = invoke_tools(
        public,
        [{"tool": "run_command", "arguments": {"command": "make test", "read_only": True}}],
        model_paths=True,
    )
    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "1\n"
    assert len(interaction.prompts) <= 1
    require_drained(generation)


def test_public_make_requires_a_scoped_indirect_tool_grant(
    public_registry, public_workspace_root, request, native_directory, monkeypatch
):
    """An indirect tool needs its own grant and cannot read an unrelated private directory."""
    if any(item is None for item in (_SYSTEM_MAKE, _SYSTEM_CAT, _SYSTEM_SH, _SYSTEM_PRINTF)):
        pytest.skip("System build and file tools are required")
    interaction, workspace, generation, _ = public_registry
    _, _, external_root = public_workspace_root
    name = _test_directory_name(request.node.nodeid)
    tool_root = native_directory(external_root, name=name)
    unrelated = native_directory(external_root, name=name + "-other")
    canary = unrelated / "canary"
    canary.write_text("outside-make-canary", encoding="utf-8")
    tool = tool_root / (name + "-builder")
    tool.write_text(
        f"#!{_SYSTEM_SH}\n{shlex.quote(_SYSTEM_PRINTF)} indirect-ok\n", encoding="utf-8"
    )
    tool.chmod(0o755)
    (workspace / "Makefile").write_text(
        f"all:\n\t@{shlex.quote(tool.name)}\n"
        f"\t@if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))} "
        ">/dev/null 2>&1; then exit 9; fi\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PATH", os.pathsep.join((str(tool_root), os.defpath)))
    command = {
        "tool": "run_command",
        "arguments": {
            "command": shlex.quote(_SYSTEM_MAKE),
            "cwd": VirtualPath.WORKSPACE,
            "read_only": True,
        },
    }
    (denied,) = invoke_registered(public_registry, [command], model_paths=True)
    assert not denied["ok"]
    assert denied["problem"]["code"] == "process.nonzero_exit"
    assert tool.name in denied["problem"]["metadata"]["stderr"]["content"]
    assert not interaction.prompts

    lookup, allowed = invoke_registered(
        public_registry,
        [
            {
                "tool": "resolve_executable",
                "arguments": {"name": tool.name, "cwd": VirtualPath.WORKSPACE},
            },
            {
                **command,
                "argument_sources": {
                    "executable_grant": {"result_index": 0, "result_key": "grant_reference"}
                },
            },
        ],
        model_paths=True,
    )
    assert lookup["ok"] and lookup["result"]["grant_reference"]
    assert allowed["ok"], allowed
    assert allowed["result"]["stdout"]["content"] == "indirect-ok"
    assert "outside-make-canary" not in json.dumps(allowed)
    assert canary.read_text(encoding="utf-8") == "outside-make-canary"
    messages = [message for message in interaction.infos if f"installed {tool.name}" in message]
    assert len(messages) == 1
    scope = json.loads(messages[0].split("\nScope: ", 1)[1].split("\n", 1)[0])
    assert scope["read_roots"] == [str(tool_root)]
    require_drained(generation)


@pytest.mark.parametrize("decision", ["deny", "approve"])
@pytest.mark.parametrize("selection", ["path", "relative", "absolute", "parent", "grant"])
def test_public_selected_executable_follows_directory_alias_into_approved_root(
    public_registry,
    public_workspace_root,
    request,
    native_directory,
    monkeypatch,
    selection,
    decision,
):
    """An approved installation remains usable through its selected executable's alias."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    interaction, workspace, generation, _ = public_registry
    _, _, external_root = public_workspace_root
    name = _test_directory_name(request.node.nodeid)
    installed = native_directory(external_root, name=name + "-installed")
    links = native_directory(external_root, name=name + "-links")
    outside = native_directory(external_root, name=name + "-outside")
    (installed / "support").write_text("approved-support", encoding="utf-8")
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    sibling = links / "sibling"
    sibling.write_text("sibling-canary", encoding="utf-8")
    executable = installed / "actual-executable"
    executable.write_text(
        f"#!{_SYSTEM_SH}\n{shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(installed / 'support'))}\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))} >/dev/null 2>&1; "
        "then exit 9; fi\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(sibling))} >/dev/null 2>&1; "
        "then exit 10; fi\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    alias = links / "runtime"
    alias.symlink_to(installed, target_is_directory=True)
    lookup = workspace / "lookup"
    lookup.mkdir()
    target = alias / executable.name
    (lookup / "fixture-task").symlink_to(
        os.path.relpath(target, lookup) if selection == "relative" else target
    )
    monkeypatch.setenv("PATH", os.pathsep.join((str(lookup), os.defpath)))
    command = {
        "path": "fixture-task",
        "relative": "./lookup/fixture-task",
        "absolute": shlex.quote(str(lookup / "fixture-task")),
        "parent": "./lookup/../lookup/fixture-task",
        "grant": "fixture-task",
    }[selection]
    arguments = {"command": command, "cwd": VirtualPath.WORKSPACE, "read_only": True}
    if decision == "deny":
        (denied,) = invoke_registered(
            public_registry,
            [{"tool": "run_command", "arguments": arguments}],
            model_paths=True,
            deny_sandbox=selection != "parent",
        )
        assert not denied["ok"], denied
        assert denied["problem"]["code"] == (
            "process.nonzero_exit" if selection == "parent" else "tool.denied"
        )
        assert len(interaction.prompts) == (0 if selection == "parent" else 1)
        assert canary.read_text(encoding="utf-8") == "outside-canary"
        assert sibling.read_text(encoding="utf-8") == "sibling-canary"
        require_drained(generation)
        return
    if selection == "grant":
        lookup_result, allowed = invoke_registered(
            public_registry,
            [
                {"tool": "resolve_executable", "arguments": {"name": "fixture-task"}},
                {
                    "tool": "run_command",
                    "arguments": arguments,
                    "argument_sources": {
                        "executable_grant": {"result_index": 0, "result_key": "grant_reference"}
                    },
                },
            ],
            model_paths=True,
        )
        assert lookup_result["ok"]
    else:
        (allowed,) = invoke_registered(
            public_registry,
            [
                {
                    "tool": "run_command",
                    "arguments": arguments | {"read_roots": [str(installed)]}
                    if selection == "parent"
                    else arguments,
                }
            ],
            model_paths=True,
        )
    assert allowed["ok"], allowed
    assert allowed["result"]["stdout"]["content"] == "approved-support"
    assert len(interaction.prompts) == 1
    assert canary.read_text(encoding="utf-8") == "outside-canary"
    assert sibling.read_text(encoding="utf-8") == "sibling-canary"
    require_drained(generation)


@pytest.mark.parametrize("alias_mode", ["file", "directory"])
def test_public_indirect_path_lookup_uses_an_approved_root_without_tool_lookup(
    public_registry, public_workspace_root, request, native_directory, monkeypatch, alias_mode
):
    """A PATH alias reaches an approved tool while unrelated files remain inaccessible."""
    if any(item is None for item in (_SYSTEM_SH, _SYSTEM_CAT, _SYSTEM_PRINTF)):
        pytest.skip("System shell and file tools are required")
    interaction, workspace, generation, _ = public_registry
    _, _, external_root = public_workspace_root
    name = _test_directory_name(request.node.nodeid)
    installed = native_directory(external_root, name=name + "-installed")
    lookup = native_directory(external_root, name=name + "-lookup")
    links = native_directory(external_root, name=name + "-links")
    outside = native_directory(external_root, name=name + "-outside")
    (installed / "support").write_text("approved-support", encoding="utf-8")
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    sibling = links / "sibling"
    sibling.write_text("sibling-canary", encoding="utf-8")
    tool = installed / "actual-executable"
    tool.write_text(
        f"#!{_SYSTEM_SH}\n{shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(installed / 'support'))}\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))} >/dev/null 2>&1; "
        "then exit 9; fi\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(sibling))} >/dev/null 2>&1; "
        "then exit 10; fi\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    alias = links / "runtime"
    alias.symlink_to(installed, target_is_directory=True)
    name = "fixture-task" if alias_mode == "file" else "@fixture-加工"
    (lookup / name).symlink_to(tool if alias_mode == "file" else alias / tool.name)
    launcher = workspace / "launcher"
    launcher.write_text(f"#!{_SYSTEM_SH}\n{shlex.quote(name)}\n", encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(lookup), os.defpath)))
    arguments = {"command": "./launcher", "cwd": VirtualPath.WORKSPACE, "read_only": True}
    (denied,) = invoke_registered(
        public_registry, [{"tool": "run_command", "arguments": arguments}], model_paths=True
    )
    assert not denied["ok"]
    assert not interaction.prompts
    (allowed,) = invoke_registered(
        public_registry,
        [{"tool": "run_command", "arguments": arguments | {"read_roots": [str(installed)]}}],
        model_paths=True,
    )
    assert allowed["ok"], allowed
    assert allowed["result"]["stdout"]["content"] == "approved-support"
    assert len(interaction.prompts) == 1
    assert canary.read_text(encoding="utf-8") == "outside-canary"
    assert sibling.read_text(encoding="utf-8") == "sibling-canary"
    require_drained(generation)


@pytest.mark.parametrize(
    "invocation",
    ["direct", "forwarded-path", "forwarded-displaced", "literal-direct", "literal-fixed"],
)
def test_workspace_selected_executable_retarget_is_stale(
    public, public_workspace_root, request, native_directory, monkeypatch, invocation
):
    """A selected executable changed during approval is refused before native launch."""
    if _SYSTEM_SH is None:
        pytest.skip("System shell is required")
    interaction, workspace, generation = public
    _, _, external_root = public_workspace_root
    directory = native_directory(external_root, name=_test_directory_name(request.node.nodeid))
    original = directory / "original"
    replacement = directory / "replacement"
    for executable in (original, replacement):
        shutil.copyfile(_SYSTEM_SH, executable)
        executable.chmod(0o755)
    selected = workspace / "fixture-tool"
    selected.symlink_to(original)
    earlier = workspace / "earlier"
    earlier.mkdir()
    competing = earlier / "fixture-tool"
    if invocation == "forwarded-displaced":
        competing.symlink_to(earlier / "missing")
    launcher = workspace / "fixture-launcher"
    launcher.write_text(
        "#!/bin/sh\nexec ./fixture-tool -c 'printf touched > marker'\n"
        if invocation == "literal-fixed"
        else '#!/bin/sh\nexec ./fixture-tool "$@"\n'
        if invocation == "literal-direct"
        else '#!/bin/sh\nexec \t"$@"\n'
        if invocation == "forwarded-path"
        else '#!/bin/sh\nexec "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(earlier), str(workspace), os.defpath)))

    positive = invoke(public, "printf positive > ordinary")
    assert positive["ok"], positive
    assert (workspace / "ordinary").read_text(encoding="utf-8") == "positive"
    interaction.prompts.clear()
    (result,) = invoke_tools(
        public,
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": (
                        "./fixture-launcher ignored"
                        if invocation == "literal-fixed"
                        else "./fixture-launcher -c 'printf touched > marker'"
                        if invocation == "literal-direct"
                        else "./fixture-launcher fixture-tool -c 'printf touched > marker'"
                        if invocation != "direct"
                        else "./fixture-tool -c 'printf touched > marker'"
                    ),
                    "cwd": str(workspace),
                },
            }
        ],
        retarget=(competing if invocation == "forwarded-displaced" else selected, replacement),
    )
    assert result["problem"]["code"] == "sandbox.stale", result
    assert len(interaction.prompts) == 1
    assert not (workspace / "marker").exists()
    require_drained(generation)


@pytest.mark.parametrize(
    "invocation",
    [
        "direct",
        "direct-file",
        "forwarded",
        "forwarded-chain",
        "forwarded-path",
        "forwarded-path-chain",
        "forwarded-path-earlier",
        "literal-forward",
        "literal-fixed",
        "literal-options",
        "literal-unicode",
        "literal-chain",
        "literal-direct-override",
        "literal-path-override",
        "forwarded-path-override",
        "not-forwarded-chain",
        "outside-relative",
        "path-override",
        "path-override-later",
        "path-assignment",
        "quoted-assignment",
    ],
)
def test_workspace_executable_link_requests_installation_scope(
    public_registry, public_workspace_root, request, native_directory, monkeypatch, invocation
):
    """A workspace-selected binary needs approval for its linked installation only."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    interaction, workspace, generation, _ = public_registry
    _, _, external_root = public_workspace_root
    name = _test_directory_name(request.node.nodeid)
    installed = native_directory(external_root, name=name + "-installed")
    links = native_directory(external_root, name=name + "-links")
    outside = native_directory(external_root, name=name + "-outside")
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    private = workspace / ".ssh"
    private.mkdir()
    secret = private / "key"
    secret.write_text("private-canary", encoding="utf-8")
    program = installed / "actual"
    shutil.copyfile(_SYSTEM_SH, program)
    program.chmod(0o755)
    source = (
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))} >/dev/null 2>&1; "
        "then exit 9; fi; "
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(secret))} >/dev/null 2>&1; "
        "then exit 10; fi; printf selected-ok"
    )
    assert subprocess.run([str(program), "-c", source], timeout=10, check=False).returncode == 9
    alias = links / "runtime"
    alias.symlink_to(installed, target_is_directory=True)
    selected = workspace / ("fixture-加工" if invocation == "literal-unicode" else "fixture-tool")
    selected.symlink_to(program if invocation == "direct-file" else alias / program.name)
    launcher = workspace / "fixture-launcher"
    launcher.write_text(
        f"#!/bin/sh\nexec ./fixture-tool -c {shlex.quote(source)}\n"
        if invocation == "literal-fixed"
        else '#!/bin/sh\nexec ./fixture-tool -e "$@"\n'
        if invocation == "literal-options"
        else '#!/bin/sh\nexec ./fixture-加工 "$@"\n'
        if invocation == "literal-unicode"
        else '#!/bin/sh\nexec ./fixture-tool "$@"\n'
        if invocation in {"literal-forward", "literal-direct-override"}
        else '#!/bin/sh\nexec ./fixture-forwarder "$@"\n'
        if invocation == "literal-chain"
        else '#!/bin/sh\nexec fixture-tool "$@"\n'
        if invocation == "literal-path-override"
        else '#!/bin/sh\nexec "$1"\n'
        if invocation == "not-forwarded-chain"
        else '#!/bin/sh\nexec \t"$@"\n'
        if invocation == "forwarded"
        else '#!/bin/sh\nexec "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    forwarder = workspace / "fixture-forwarder"
    forwarder.write_text(
        '#!/bin/sh\nexec ./fixture-tool "$@"\n'
        if invocation == "literal-chain"
        else '#!/bin/sh\nexec "$@"\n',
        encoding="utf-8",
    )
    forwarder.chmod(0o755)
    earlier = workspace / "earlier"
    earlier.mkdir()
    if invocation == "forwarded-path-earlier":
        competing = earlier / "fixture-tool"
        competing.write_text('#!/bin/sh\nprintf "earlier-ok"\n', encoding="utf-8")
        competing.chmod(0o755)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join((str(earlier), str(workspace), os.defpath))
        if invocation
        in {
            "path-override",
            "path-override-later",
            "path-assignment",
            "forwarded-path",
            "forwarded-path-chain",
            "forwarded-path-earlier",
            "literal-path-override",
            "forwarded-path-override",
        }
        else os.defpath,
    )
    invocation_source = (
        "PATH=/nonexistent ./fixture-launcher fixture-tool"
        if invocation == "forwarded-path-override"
        else "PATH=/nonexistent ./fixture-launcher"
        if invocation in {"literal-path-override", "literal-direct-override"}
        else "./fixture-launcher"
        if invocation
        in {
            "literal-forward",
            "literal-fixed",
            "literal-options",
            "literal-chain",
            "literal-unicode",
        }
        else "'FOO=bar' ./fixture-tool"
        if invocation == "quoted-assignment"
        else ":; PATH=/nonexistent fixture-tool"
        if invocation == "path-override-later"
        else "PATH=/nonexistent; fixture-tool"
        if invocation == "path-assignment"
        else "PATH=/nonexistent fixture-tool"
        if invocation == "path-override"
        else shlex.quote(os.path.relpath(alias / program.name, workspace))
        if invocation == "outside-relative"
        else "./fixture-tool"
        if invocation in {"direct", "direct-file"}
        else "./fixture-launcher ./fixture-forwarder ./fixture-tool"
        if invocation in {"forwarded-chain", "not-forwarded-chain"}
        else "./fixture-launcher fixture-tool"
        if invocation in {"forwarded-path", "forwarded-path-earlier"}
        else "./fixture-launcher fixture-forwarder fixture-tool"
        if invocation == "forwarded-path-chain"
        else "./fixture-launcher ./fixture-tool"
    )
    arguments = {
        "command": f"{invocation_source} -c {shlex.quote(source)}",
        "cwd": VirtualPath.WORKSPACE,
        "read_only": True,
    }
    (denied,) = invoke_registered(
        public_registry,
        [{"tool": "run_command", "arguments": arguments}],
        model_paths=True,
        deny_sandbox=True,
    )
    if invocation == "not-forwarded-chain":
        assert denied["ok"], denied
        assert denied["result"]["stdout"]["content"] == ""
        assert not interaction.prompts
        require_drained(generation)
        return
    if invocation == "forwarded-path-earlier":
        assert denied["ok"], denied
        assert denied["result"]["stdout"]["content"] == "earlier-ok"
        assert not interaction.prompts
        require_drained(generation)
        return
    if invocation in {
        "outside-relative",
        "path-override",
        "path-override-later",
        "path-assignment",
        "quoted-assignment",
        "literal-path-override",
        "forwarded-path-override",
    }:
        assert denied["problem"]["code"] == "process.nonzero_exit", denied
        assert denied["problem"]["metadata"]["exit_code"] not in {9, 10}
        assert not interaction.prompts
        require_drained(generation)
        return
    assert denied["problem"]["code"] == "tool.denied", denied
    (allowed,) = invoke_registered(
        public_registry,
        [{"tool": "run_command", "arguments": arguments}],
        model_paths=True,
    )
    assert allowed["ok"], json.dumps(allowed, indent=2)
    assert allowed["result"]["stdout"]["content"] == "selected-ok"
    assert len(interaction.prompts) == 2
    assert canary.read_text(encoding="utf-8") == "outside-canary"
    assert secret.read_text(encoding="utf-8") == "private-canary"
    require_drained(generation)


@pytest.mark.parametrize(
    ("engine", "invocation"),
    [
        (engine, invocation)
        for engine in ("shell", "perl")
        for invocation in ("direct", "launcher", "launcher-name", "launcher-nested")
    ]
    + [
        ("shell", "direct-identified"),
        ("shell", "direct-identified-workspace"),
        ("shell", "direct-quoted"),
        ("shell", "direct-literal-dollar"),
        ("shell", "direct-dynamic-dollar"),
        ("shell", "direct-wrapped"),
        ("shell", "forward"),
        ("shell", "forward-name"),
    ],
)
def test_workspace_script_uses_approved_off_path_interpreter(
    public_registry,
    public_workspace_root,
    request,
    native_directory,
    monkeypatch,
    engine,
    invocation,
):
    """An invoked script can reach its approved off-PATH interpreter without exposing canaries."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None or _SYSTEM_PERL is None:
        pytest.skip("System shell, Perl and file tools are required")
    interaction, workspace, generation, _ = public_registry
    _, _, external_root = public_workspace_root
    name = _test_directory_name(request.node.nodeid)
    identified = invocation in {
        "direct",
        "direct-identified",
        "direct-identified-workspace",
        "direct-quoted",
        "direct-literal-dollar",
        "direct-wrapped",
        "forward",
        "forward-name",
    }
    installed = native_directory(external_root, name=name + "-installed")
    links = native_directory(external_root, name=name + "-links")
    outside = native_directory(external_root, name=name + "-outside")
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    private = workspace / ".ssh"
    private.mkdir()
    secret = private / "key"
    secret.write_text("private-canary", encoding="utf-8")
    interpreter = installed / "actual"
    shutil.copyfile(_SYSTEM_SH if engine == "shell" else _SYSTEM_PERL, interpreter)
    interpreter.chmod(0o755)
    alias = links / "runtime"
    alias.symlink_to(installed, target_is_directory=True)
    environment = workspace / "off-path-tools"
    environment.mkdir()
    selected = environment / "fixture-engine"
    selected.symlink_to(alias / interpreter.name)
    dollar = invocation in {"direct-literal-dollar", "direct-dynamic-dollar"}
    script_root = workspace / "$NAME" if dollar else workspace
    script_root.mkdir(exist_ok=True)
    script = script_root / ("fixture task" if invocation == "direct-quoted" else "fixture-task")
    content = f"#!{selected}\n" + (
        "printf offpath-ok\n" if engine == "shell" else 'print "offpath-ok";\n'
    )
    if invocation in {
        "direct-identified",
        "direct-identified-workspace",
        "direct-literal-dollar",
        "direct-wrapped",
    }:
        content = (
            f"#!{selected}\n"
            f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))} >/dev/null 2>&1; "
            "then exit 9; fi\n"
            f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(secret))} >/dev/null 2>&1; "
            "then exit 10; fi\n"
            "printf offpath-ok\n"
        )
    if invocation == "launcher-nested":
        nested = workspace / "fixture-intermediate"
        nested.write_text(content, encoding="utf-8")
        nested.chmod(0o755)
        content = f"#!{nested}\n"
    script.write_text(content, encoding="utf-8")
    script.chmod(0o755)
    launcher = workspace / "fixture-launcher"
    target = "./$1" if invocation in {"launcher-name", "launcher-nested", "forward-name"} else "$1"
    suffix = "" if invocation in {"forward", "forward-name"} else "; :"
    launcher.write_text(f'#!/bin/sh\nexec "{target}"{suffix}\n', encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", os.defpath)
    monkeypatch.delenv("NAME", raising=False)
    invocation_source = (
        "'./$NAME/fixture-task'"
        if invocation == "direct-literal-dollar"
        else "exec ./fixture-task; :"
        if invocation == "direct-wrapped"
        else '"./$NAME/fixture-task"'
        if invocation == "direct-dynamic-dollar"
        else "'./fixture task'"
        if invocation == "direct-quoted"
        else "./fixture-task"
        if invocation == "direct" or identified and not invocation.startswith("forward")
        else "./fixture-launcher fixture-task"
        if invocation in {"launcher-name", "launcher-nested", "forward-name"}
        else "./fixture-launcher ./fixture-task"
    )
    command = invocation_source
    if invocation not in {
        "direct-identified",
        "direct-identified-workspace",
        "direct-literal-dollar",
        "direct-dynamic-dollar",
    }:
        command += (
            " || exit $?; "
            f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))} >/dev/null 2>&1; "
            "then exit 9; fi; "
            f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(secret))} >/dev/null 2>&1; "
            "then exit 10; fi"
        )
    arguments = {"command": command, "cwd": VirtualPath.WORKSPACE, "read_only": True}
    if invocation == "direct-identified-workspace":
        arguments["read_roots"] = [VirtualPath.WORKSPACE]
    (denied,) = invoke_registered(
        public_registry,
        [{"tool": "run_command", "arguments": arguments}],
        model_paths=True,
        deny_sandbox=identified or invocation == "direct-dynamic-dollar",
    )
    assert not denied["ok"]
    if invocation == "direct-dynamic-dollar":
        assert denied["problem"]["code"] == "process.nonzero_exit"
        assert not interaction.prompts
        require_drained(generation)
        return
    if identified:
        assert denied["problem"]["code"] == "tool.denied"
        assert len(interaction.prompts) == 1
    else:
        assert not interaction.prompts
    (allowed,) = invoke_registered(
        public_registry,
        [
            {
                "tool": "run_command",
                "arguments": arguments
                if identified
                else arguments | {"read_roots": [str(installed)]},
            }
        ],
        model_paths=True,
    )
    assert allowed["ok"], json.dumps(allowed, indent=2)
    assert allowed["result"]["stdout"]["content"] == "offpath-ok"
    assert len(interaction.prompts) == (2 if identified else 1)
    assert canary.read_text(encoding="utf-8") == "outside-canary"
    assert secret.read_text(encoding="utf-8") == "private-canary"
    require_drained(generation)


@pytest.mark.parametrize(
    "invocation", ["direct", "launcher", "direct-quoted", "forward", "executable"]
)
def test_workspace_script_uses_approved_off_path_virtual_environment(
    public_registry, public_workspace_root, request, native_directory, monkeypatch, invocation
):
    """An existing virtual environment works off PATH with an approved external runtime."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    runtime = Path(sys.base_prefix).resolve(strict=True)
    interpreter = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    if not interpreter.is_relative_to(runtime) or not interpreter.is_file():
        pytest.skip("The active interpreter is not contained in its base runtime")
    if any(runtime.is_relative_to(root) for root in runtime_read_roots()):
        pytest.skip("An external Python runtime is required for this approval test")
    interaction, workspace, generation, _ = public_registry
    _, _, external_root = public_workspace_root
    name = _test_directory_name(request.node.nodeid)
    links = native_directory(external_root, name=name + "-links")
    outside = native_directory(external_root, name=name + "-outside")
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    private = workspace / ".ssh"
    private.mkdir()
    secret = private / "key"
    secret.write_text("private-canary", encoding="utf-8")
    alias = links / "runtime"
    alias.symlink_to(runtime, target_is_directory=True)
    environment = workspace / "chosen-environment"
    commands = environment / "arbitrary-commands"
    commands.mkdir(parents=True)
    selected = commands / "fixture-engine"
    selected.symlink_to(alias / interpreter.relative_to(runtime))
    (environment / "pyvenv.cfg").write_text(
        f"home = {interpreter.parent}\ninclude-system-site-packages = false\n", encoding="utf-8"
    )
    script = workspace / ("fixture task" if invocation == "direct-quoted" else "fixture-task")
    script.write_text(
        f"#!{selected}\nimport sys\nprint('runtime-ok', sys.prefix)\n", encoding="utf-8"
    )
    script.chmod(0o755)
    launcher = workspace / "fixture-launcher"
    suffix = "" if invocation == "forward" else "; :"
    launcher.write_text(f'#!/bin/sh\nexec "$1"{suffix}\n', encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", os.defpath)
    invocation_source = (
        "./chosen-environment/arbitrary-commands/fixture-engine -c "
        + shlex.quote("import sys; print('runtime-ok', sys.prefix)")
        if invocation == "executable"
        else "'./fixture task'"
        if invocation == "direct-quoted"
        else "./fixture-task"
        if invocation == "direct"
        else "./fixture-launcher ./fixture-task"
    )
    command = (
        f"{invocation_source} || exit $?; "
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))} >/dev/null 2>&1; "
        "then exit 9; fi; "
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(secret))} >/dev/null 2>&1; "
        "then exit 10; fi"
    )
    arguments = {"command": command, "cwd": VirtualPath.WORKSPACE, "read_only": True}
    (denied,) = invoke_registered(
        public_registry,
        [{"tool": "run_command", "arguments": arguments}],
        model_paths=True,
        deny_sandbox=invocation != "launcher",
    )
    assert not denied["ok"]
    if invocation != "launcher":
        assert denied["problem"]["code"] == "tool.denied", json.dumps(denied, indent=2)
        assert len(interaction.prompts) == 1
    else:
        assert not interaction.prompts
    (allowed,) = invoke_registered(
        public_registry,
        [
            {
                "tool": "run_command",
                "arguments": arguments
                if invocation != "launcher"
                else arguments | {"read_roots": [str(runtime)]},
            }
        ],
        model_paths=True,
    )
    assert allowed["ok"], json.dumps(allowed, indent=2)
    assert allowed["result"]["stdout"]["content"] == f"runtime-ok {environment}\n"
    assert len(interaction.prompts) == (1 if invocation == "launcher" else 2)
    assert canary.read_text(encoding="utf-8") == "outside-canary"
    assert secret.read_text(encoding="utf-8") == "private-canary"
    require_drained(generation)


def test_indirect_alias_order_reuses_remembered_native_approval(
    installation, public, public_workspace_root, request, native_directory, monkeypatch
):
    """Reordered PATH entries reuse approval while outside files remain inaccessible."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    interaction, workspace, generation = public
    _, _, external_root = public_workspace_root
    name = _test_directory_name(request.node.nodeid)
    installed = native_directory(external_root, name=name + "-installed")
    lookup = native_directory(external_root, name=name + "-lookup")
    links = native_directory(external_root, name=name + "-links")
    outside = native_directory(external_root, name=name + "-outside")
    scratch_base = native_directory(external_root, name=name + "-scratch")
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    for tool_name in ("second", "first"):
        package = installed / tool_name
        package.mkdir()
        executable = package / "actual"
        executable.write_text(f"#!{_SYSTEM_SH}\nprintf {tool_name}\n", encoding="utf-8")
        executable.chmod(0o755)
        alias = links / tool_name
        alias.symlink_to(package, target_is_directory=True)
        (lookup / tool_name).symlink_to(alias / executable.name)
    launcher = workspace / "launcher"
    launcher.write_text(
        f"#!{_SYSTEM_SH}\nfirst\nsecond\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))} >/dev/null 2>&1; "
        "then exit 9; fi\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(lookup), os.defpath)))
    prompts = []

    def remember(message, **_kwargs):
        """Remember this fixture's sandbox scope in the current session."""
        prompts.append(message)
        return "session"

    interaction.prompt = remember
    backend, installed_generation = installation
    assert installed_generation == generation
    monkeypatch.setattr(tempfile, "tempdir", str(scratch_base))
    permissions = PermissionManager(workspace)
    registry = ToolRegistry(
        [run_command],
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, RefusingHost()),
    )
    arguments = {
        "command": "./launcher",
        "cwd": str(workspace),
        "read_only": True,
        "read_roots": [str(installed)],
    }
    original_scandir = os.scandir
    try:
        first = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        first_profiles = profile_inventory(generation)

        def reversed_scandir(path):
            """Reverse only the selected PATH directory without modifying its entries."""
            entries = original_scandir(path)
            if isinstance(path, int) or Path(path) != lookup:
                return entries
            with entries:
                return nullcontext(iter(reversed(list(entries))))

        monkeypatch.setattr("loop.execution.facade.os.scandir", reversed_scandir)
        second = json.loads(
            registry.call("run_command", json.dumps(arguments), interaction=interaction)
        )
        second_profiles = profile_inventory(generation)
        assert first["ok"] and second["ok"], (first, second)
        assert first["result"]["stdout"]["content"] == "firstsecond"
        assert second["result"]["stdout"]["content"] == "firstsecond"
        assert len(prompts) == 1
        assert_equivalent_profiles(first_profiles, second_profiles)
        assert canary.read_text(encoding="utf-8") == "outside-canary"
        require_drained(generation)
    finally:
        registry.close()
        permissions.close()


def test_indirect_path_alias_does_not_skip_an_unapproved_earlier_executable(
    public_registry, public_workspace_root, request, native_directory, monkeypatch
):
    """An approved later alias does not override the executable selected first on PATH."""
    if _SYSTEM_SH is None:
        pytest.skip("A system shell is required")
    interaction, workspace, generation, _ = public_registry
    _, _, external_root = public_workspace_root
    name = _test_directory_name(request.node.nodeid)
    earlier = native_directory(external_root, name=name + "-earlier")
    installed = native_directory(external_root, name=name + "-installed")
    lookup = native_directory(external_root, name=name + "-lookup")
    links = native_directory(external_root, name=name + "-links")
    for directory, filename, output in (
        (earlier, "fixture-task", "first-choice"),
        (installed, "actual", "later-choice"),
    ):
        executable = directory / filename
        executable.write_text(f"#!{_SYSTEM_SH}\nprintf {output}\n", encoding="utf-8")
        executable.chmod(0o755)
    alias = links / "runtime"
    alias.symlink_to(installed, target_is_directory=True)
    (lookup / "fixture-task").symlink_to(alias / "actual")
    launcher = workspace / "launcher"
    launcher.write_text(f"#!{_SYSTEM_SH}\nfixture-task\n", encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(earlier), str(lookup), os.defpath)))
    arguments = {"command": "./launcher", "cwd": VirtualPath.WORKSPACE, "read_only": True}
    (denied,) = invoke_registered(
        public_registry,
        [{"tool": "run_command", "arguments": arguments | {"read_roots": [str(installed)]}}],
        model_paths=True,
    )
    assert not denied["ok"]
    assert denied["problem"]["metadata"]["stdout"]["content"] == ""
    (allowed,) = invoke_registered(
        public_registry,
        [
            {
                "tool": "run_command",
                "arguments": arguments | {"read_roots": [str(earlier), str(installed)]},
            }
        ],
        model_paths=True,
    )
    assert allowed["ok"], allowed
    assert allowed["result"]["stdout"]["content"] == "first-choice"
    assert len(interaction.prompts) == 2
    require_drained(generation)


@pytest.mark.parametrize("metadata_target", ["workspace", "outside", "private"])
def test_public_unrelated_metadata_does_not_block_workspace_executable(
    public, public_workspace_root, request, native_directory, monkeypatch, metadata_target
):
    """Unidentified metadata neither blocks a generic tool nor exposes protected files."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    interaction, workspace, generation = public
    tools = workspace / "bin"
    tools.mkdir()
    executable = tools / "fixture-processor"
    executable.write_text(f"#!{_SYSTEM_SH}\nprintf fixture-ok\n", encoding="utf-8")
    executable.chmod(0o755)
    (workspace / "unrelated").write_text("ordinary metadata", encoding="utf-8")
    private = workspace / ".ssh"
    private.mkdir()
    (private / "key").write_text("private-canary", encoding="utf-8")
    outside = native_directory(
        public_workspace_root[2], name=_test_directory_name(request.node.nodeid) + "-outside"
    )
    canary = outside / "canary"
    canary.write_text("outside-canary", encoding="utf-8")
    metadata = workspace / "pyvenv.cfg"
    metadata.symlink_to(
        workspace / "unrelated"
        if metadata_target == "workspace"
        else canary
        if metadata_target == "outside"
        else private / "key"
    )
    monkeypatch.setenv("PATH", os.pathsep.join((str(tools), os.defpath)))
    command = (
        f"fixture-processor; if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(canary))} "
        ">/dev/null 2>&1; then exit 9; fi"
    )
    if metadata_target != "workspace":
        command += (
            f"; if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(metadata))} "
            ">/dev/null 2>&1; then exit 10; fi"
        )

    (result,) = invoke_tools(
        public,
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": command,
                    "cwd": "$WORKSPACE_ROOT",
                    "read_only": True,
                },
            }
        ],
        model_paths=True,
    )

    assert result["ok"], result
    assert result["result"]["stdout"]["content"] == "fixture-ok"
    assert not interaction.prompts
    assert canary.read_text(encoding="utf-8") == "outside-canary"
    assert (private / "key").read_text(encoding="utf-8") == "private-canary"
    require_drained(generation)


def test_public_executable_lookup_and_exact_file_authority(
    public_registry, public_workspace_root, request, native_directory, monkeypatch
):
    """Show exact authority to the user while keeping model paths and executable access narrow."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    interaction, workspace, _generation, _audit_path = public_registry
    _workspace_root, _permissions_root, external_root = public_workspace_root
    tool_directory = native_directory(
        external_root,
        name=_test_directory_name(request.node.nodeid),
    )
    executable = tool_directory / "owned-tool"
    sibling = tool_directory / "sibling-secret"
    sibling.write_text("external-sibling-canary", encoding="utf-8")
    executable.write_text(
        f"#!{_SYSTEM_SH}\n"
        f"if {shlex.quote(_SYSTEM_CAT)} {shlex.quote(str(sibling))} >/dev/null 2>&1; then\n"
        "  printf sibling-visible\n"
        "else\n"
        "  printf exact-file-only\n"
        "fi\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tool_directory) + os.pathsep + os.environ["PATH"])

    (batch,) = invoke_registered(
        public_registry,
        [
            {
                "tool": "resolve_executable",
                "arguments": {
                    "name": [executable.name, "fixture-missing-tool"],
                    "cwd": "$WORKSPACE_ROOT",
                },
            }
        ],
        model_paths=True,
    )
    assert batch["ok"], batch
    present, missing = batch["result"]["executables"]
    assert present["name"] == executable.name and present["grant_reference"]
    assert missing == {"name": "fixture-missing-tool", "status": "absent_from_host_path"}

    lookup, explicit, exact = invoke_registered(
        public_registry,
        [
            {
                "tool": "resolve_executable",
                "arguments": {"name": executable.name, "cwd": "$WORKSPACE_ROOT"},
            },
            {
                "tool": "run_command",
                "arguments": {
                    "command": executable.name,
                    "cwd": "$WORKSPACE_ROOT",
                    "read_only": True,
                },
                "argument_sources": {
                    "executable_grant": {"result_index": 0, "result_key": "grant_reference"}
                },
            },
            {
                "tool": "run_command",
                "arguments": {
                    "command": executable.name,
                    "cwd": "$WORKSPACE_ROOT",
                    "read_only": True,
                },
            },
        ],
        model_paths=True,
    )
    lookup = lookup.get("result", lookup)
    assert lookup["sandbox_access"] == "external_read_grant_required", lookup
    assert lookup.get("grant_reference"), lookup
    explicit_execution = explicit.get("result", explicit)
    assert explicit_execution.get("exit_code") == 0, explicit
    assert explicit_execution["stdout"]["content"] == "sibling-visible"
    exact_execution = exact.get("result", exact)
    assert exact_execution.get("exit_code") == 0, exact
    assert exact_execution["stdout"]["content"] == "exact-file-only"
    messages = [
        message
        for message in interaction.infos
        if "Read the installed owned-tool tool folder" in message
    ]
    assert len(messages) == 1
    scope = json.loads(messages[0].split("\nScope: ", 1)[1].split("\n", 1)[0])
    assert scope["workspace"] == str(workspace)
    assert scope["read_roots"] == [str(tool_directory)]
    assert all(
        str(tool_directory) not in json.dumps(result) for result in (lookup, explicit, exact)
    )
    require_drained(public_registry[2])


def test_public_audit_correlates_native_attempt_without_command_content(
    public_registry,
):
    """Record the correlated sandbox attempt without retaining opaque shell source."""
    _interaction, workspace, _generation, audit_path = public_registry
    token = "audit-private-command-content-7bf0d835"
    (result,) = invoke_registered(
        public_registry,
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": f"printf {token} >/dev/null",
                    "cwd": str(workspace),
                    "read_only": True,
                },
            }
        ],
    )
    assert result.get("result", result).get("exit_code") == 0, result
    exported = SQLitePermissionAudit(audit_path).export_jsonl(
        audit_path.with_name(f"{workspace.name}-audit-export.jsonl"),
        workspace_id="native-public-review",
    )
    content = exported.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in content.splitlines()]
    assert any(
        row["event_name"] == "sandbox.permission_scope" and row["payload"].get("attempt_id")
        for row in rows
    )
    assert token not in content
    require_drained(public_registry[2])


def test_installed_payload_closes_inheritable_host_descriptors(
    installation, public, native_directory
):
    """Close ambient host descriptors before any opaque payload can inspect them."""
    backend, generation = installation
    _interaction, workspace, _generation = public
    scratch = native_directory(Path("/tmp"))
    outside_directory = native_directory(workspace.parent)
    outside = outside_directory / "descriptor-canary"
    outside.write_text("host-only", encoding="utf-8")
    with outside.open("rb") as stream:
        descriptor = fcntl.fcntl(stream.fileno(), fcntl.F_DUPFD, 100)
        try:
            os.set_inheritable(descriptor, True)
            request = native_request(
                backend,
                workspace,
                scratch,
                f"test ! -e /proc/self/fd/{descriptor}",
            )
            result = backend.run(request)
        finally:
            os.close(descriptor)
    assert result.outcome is SandboxOutcome.COMPLETED, result
    assert result.exit_code == 0, result
    require_drained(generation)


def test_installed_payload_inherits_enforcement_and_cannot_replace_its_profile(
    installation, public, native_directory
):
    """Keep nested shells confined and refuse attempts to replace the active payload profile."""
    backend, generation = installation
    _interaction, workspace, _generation = public
    scratch = native_directory(Path("/tmp"))
    nested = backend.run(
        native_request(
            backend,
            workspace,
            scratch,
            "cat /proc/self/attr/current; /bin/sh -c 'cat /proc/self/attr/current'",
        )
    )
    labels = nested.stdout.splitlines()
    assert nested.outcome is SandboxOutcome.COMPLETED, nested
    assert nested.exit_code == 0, nested
    assert len(labels) == 2 and labels[0] == labels[1]
    assert "payload_" in labels[0] and labels[0].endswith("(enforce)")

    code = (
        "import ctypes;lib=ctypes.CDLL('libapparmor.so.1',use_errno=True);"
        "lib.aa_change_profile.argtypes=[ctypes.c_char_p];"
        "lib.aa_change_profile.restype=ctypes.c_int;"
        "result=lib.aa_change_profile(b'unconfined');"
        "raise OSError(ctypes.get_errno(),'Unapproved profile change denied') "
        "if result != 0 else RuntimeError('Profile replacement unexpectedly succeeded')"
    )
    replacement = backend.run(
        native_request(
            backend,
            workspace,
            scratch,
            "/usr/bin/python3 -I -c " + shlex.quote(code),
        )
    )
    assert replacement.outcome is SandboxOutcome.COMPLETED, replacement
    assert replacement.exit_code != 0
    assert "Unapproved profile change denied" in replacement.stderr
    require_drained(generation)


def test_public_default_hard_network_limit_rejects_coarse_grant(public):
    """Default private-network policy refuses broad egress without prompting or launching."""
    marker = public[1] / "network-not-launched"

    result = invoke(public, "printf unsafe > network-not-launched", network=True)

    assert result["problem"]["code"] == "sandbox.unavailable", result
    assert "private-network limit" in result["problem"]["detail"]
    assert not marker.exists()
    assert not public[0].prompts
    require_drained(public[2])


def test_public_network_grant_keeps_udp_offline_until_approved(public):
    """Keep UDP offline until an explicit public network grant is approved."""
    with socket.socket(type=socket.SOCK_DGRAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(0.2)
        source = "/usr/bin/python3 -I -c " + shlex.quote(
            "import socket;target=('127.0.0.1',"
            + str(listener.getsockname()[1])
            + ");socket.socket(type=socket.SOCK_DGRAM).sendto(b'granted',target)"
        )
        denied = invoke(public, source, read_only=True)
        assert denied.get("ok") is True, denied
        assert denied["result"]["exit_code"] == 0, denied
        with pytest.raises(TimeoutError):
            listener.recvfrom(32)

        approved = invoke(public, source, read_only=True, network=True, allow_general_network=True)
        assert approved["result"]["exit_code"] == 0, approved
        listener.settimeout(2)
        assert listener.recvfrom(32)[0] == b"granted"
    require_drained(public[2])


@pytest.mark.parametrize(
    ("family", "host"), [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")]
)
@pytest.mark.parametrize("kind", [socket.SOCK_STREAM, socket.SOCK_DGRAM])
def test_public_network_grant_preserves_internet_sockets(public, family, host, kind):
    """Approved IPv4 and IPv6 TCP and UDP reach an owned host listener."""
    listener = None
    try:
        listener = socket.socket(family, kind)
        listener.bind((host, 0))
    except OSError as error:
        if listener is not None:
            listener.close()
        if error.errno in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL, errno.EPROTONOSUPPORT):
            pytest.skip(f"Host address family is unavailable: {error}")
        raise
    with listener:
        listener.settimeout(2)
        if kind == socket.SOCK_STREAM:
            listener.listen()
        address = listener.getsockname()
        code = (
            f"import socket;s=socket.socket({family},{kind});"
            f"s.settimeout(1);s.connect({address!r});s.sendall(b'network-control')"
        )
        result = invoke(
            public,
            "/usr/bin/python3 -I -c " + shlex.quote(code),
            network=True,
            allow_general_network=True,
        )
        assert result["ok"] and result["result"]["exit_code"] == 0, result
        if kind == socket.SOCK_STREAM:
            with listener.accept()[0] as connection:
                assert connection.recv(32) == b"network-control"
        else:
            assert listener.recv(32) == b"network-control"
    require_drained(public[2])


def test_public_git_read_follows_linked_worktree_metadata_grant(
    public, public_workspace_root, request, native_directory
):
    """Read linked-worktree metadata only through its separately named external grant."""
    workspace = public[1]
    _workspace_root, _permissions_root, external_root = public_workspace_root
    repository = native_directory(
        external_root,
        name=_test_directory_name(request.node.nodeid),
    )
    linked = workspace / "linked-worktree"
    subprocess.run(
        ["/usr/bin/git", "init", "--quiet", "--initial-branch=main", str(repository)],
        check=True,
        capture_output=True,
        timeout=10,
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "-c",
            "user.name=Loop fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "worktree",
            "add",
            "--quiet",
            "--detach",
            str(linked),
            "HEAD",
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )
    metadata = repository / ".git"
    result = invoke(
        public,
        "/usr/bin/git status --porcelain",
        cwd=str(linked),
        read_roots=[str(metadata)],
        read_only=True,
    )
    assert result["result"]["exit_code"] == 0, result
    assert result["result"]["stdout"]["content"] == ""
    require_drained(public[2])


def test_installed_reordered_read_roots_reuse_policy_without_widening(
    installation, public, public_workspace_root, request, native_directory
):
    """Reordered read grants reuse one policy while unrelated data and writes stay denied."""
    if _SYSTEM_SH is None or _SYSTEM_CAT is None:
        pytest.skip("System shell and file tools are required")
    backend, generation = installation
    workspace = public[1]
    name = _test_directory_name(request.node.nodeid)
    first, second, outside = (
        native_directory(public_workspace_root[2], name=name + suffix)
        for suffix in ("-earlier", "-installed", "-outside")
    )
    for directory, content in ((first, "alpha"), (second, "beta"), (outside, "outside-canary")):
        (directory / "data").write_text(content, encoding="utf-8")
    (first / ".ssh").mkdir()
    (first / ".ssh/key").write_text("private-canary", encoding="utf-8")
    scratch = native_directory(Path("/tmp"))
    cat = shlex.quote(_SYSTEM_CAT)
    first_file, second_file, outside_file = (
        shlex.quote(str(directory / "data")) for directory in (first, second, outside)
    )
    private_file = shlex.quote(str(first / ".ssh/key"))
    source = (
        f"{cat} {first_file} {second_file}; "
        f"if {cat} {outside_file} >/dev/null 2>&1; then exit 31; fi; "
        f"if {cat} {private_file} >/dev/null 2>&1; then exit 32; fi; "
        f"if printf changed > {first_file} 2>/dev/null; then exit 33; fi; "
        "if printf changed > workspace-marker 2>/dev/null; then exit 34; fi"
    )
    host = subprocess.run(
        [_SYSTEM_SH, "-c", source], cwd=workspace, capture_output=True, text=True, check=False
    )
    assert host.returncode == 31

    def bound(roots, command=source):
        """Bind a read-only request to one explicit ordered grant set."""
        return SandboxRequest.create(
            source=command,
            cwd=workspace,
            workspace=workspace,
            read_roots=roots,
            write_roots=(),
            network=False,
            environment={"PATH": os.defpath, "LC_ALL": "C", "TMPDIR": str(scratch)},
            scratch_root=scratch,
            policy_version=backend.policy_version,
            deadline=time.monotonic() + 15,
            workspace_id="ordered-roots-native-test",
        )

    first_request = bound((first, second))
    first_result = backend.run(first_request)
    baseline = profile_inventory(generation)
    reordered = backend.run(bound((second, first)))
    assert first_result.outcome is SandboxOutcome.COMPLETED and first_result.exit_code == 0, (
        first_result
    )
    assert reordered.outcome is SandboxOutcome.COMPLETED and reordered.exit_code == 0, reordered
    assert first_result.stdout == reordered.stdout == "alphabeta"
    assert_equivalent_profiles(baseline, profile_inventory(generation))
    assert (first / "data").read_text(encoding="utf-8") == "alpha"
    assert (first / ".ssh/key").read_text(encoding="utf-8") == "private-canary"
    assert (outside / "data").read_text(encoding="utf-8") == "outside-canary"
    assert not (workspace / "workspace-marker").exists()

    narrow_request = bound(
        (second,),
        f"{cat} {second_file}; if {cat} {first_file} >/dev/null 2>&1; then exit 41; fi",
    )
    narrow = backend.run(narrow_request)
    assert narrow.outcome is SandboxOutcome.COMPLETED and narrow.exit_code == 0, narrow
    assert narrow.stdout == "beta"
    cache = NATIVE_INSTALLATION_DIRECTORY / "cache.lock"
    roots = runtime_read_roots()
    first_profile = payload_profile(first_request, scratch, cache, roots)
    narrow_profile = payload_profile(narrow_request, scratch, cache, roots)
    assert any(name.startswith(first_profile + "_") for name in baseline)
    assert narrow_profile != first_profile and any(
        name.startswith(narrow_profile + "_") for name in profile_inventory(generation)
    )
    require_drained(generation)


def test_installed_requests_reuse_policy_without_sharing_private_scratch(
    installation, public, native_directory
):
    """Reuse identical policy across isolated sequential and concurrent scratch mounts."""
    backend, generation = installation
    _interaction, workspace, _generation = public
    cache_lock = NATIVE_INSTALLATION_DIRECTORY / "cache.lock"
    runtime_roots = runtime_read_roots()

    def foreign_path_checks(path):
        """Return read/write probes for one other request's host scratch path."""
        direct = shlex.quote(str(path))
        traversed = shlex.quote(f"{NATIVE_SCRATCH_PATH}/../../{str(path).lstrip('/')}")
        return "".join(
            f"if /usr/bin/cat {candidate} >/dev/null 2>&1; then exit 31; fi; "
            f"if printf intruder > {candidate} 2>/dev/null; then exit 32; fi; "
            for candidate in (direct, traversed)
        )

    first_scratch = native_directory(Path("/tmp"))
    second_scratch = native_directory(Path("/tmp"))

    first = native_request(
        backend,
        workspace,
        first_scratch,
        'cd "$WORKSPACE_ROOT"; printf first > "$TMPDIR/value"; /usr/bin/cat "$TMPDIR/value"',
    )
    second_source = foreign_path_checks(first_scratch / "value") + (
        'printf second > "$TMPDIR/value"; /usr/bin/cat "$TMPDIR/value"'
    )
    second = native_request(backend, workspace, second_scratch, second_source)

    assert payload_profile(first, first_scratch, cache_lock, runtime_roots) == payload_profile(
        second, second_scratch, cache_lock, runtime_roots
    )
    first_result = backend.run(first)
    second_result = backend.run(second)
    assert first_result.outcome is SandboxOutcome.COMPLETED, (
        f"{first_result}\n{first_result.stderr}"
    )
    assert second_result.outcome is SandboxOutcome.COMPLETED, (
        f"{second_result}\n{second_result.stderr}"
    )
    assert first_result.stdout == "first"
    assert second_result.stdout == "second"
    assert (first_scratch / "value").read_text(encoding="utf-8") == "first"
    assert (second_scratch / "value").read_text(encoding="utf-8") == "second"

    concurrent_scratch = tuple(native_directory(Path("/tmp")) for _ in range(2))

    def concurrent_request(label, own_scratch, other_scratch):
        """Create one distinct scratch request that probes the peer's private path."""
        peer = "right" if label == "left" else "left"
        source = (
            'cd "$WORKSPACE_ROOT"; '
            f'printf {label} > "$TMPDIR/value"; printf ready > {label}-ready; '
            f"while test ! -f {peer}-ready; do /usr/bin/sleep 0.01; done; "
            + foreign_path_checks(other_scratch / "value")
            + f"printf isolated-{label}"
        )
        return native_request(backend, workspace, own_scratch, source)

    requests = (
        concurrent_request("left", concurrent_scratch[0], concurrent_scratch[1]),
        concurrent_request("right", concurrent_scratch[1], concurrent_scratch[0]),
    )
    assert payload_profile(requests[0], concurrent_scratch[0], cache_lock, runtime_roots) == (
        payload_profile(requests[1], concurrent_scratch[1], cache_lock, runtime_roots)
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        left = executor.submit(backend.run, requests[0])
        deadline = time.monotonic() + 5
        ready = workspace / "left-ready"
        while not ready.exists() and not left.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), left.result() if left.done() else "First payload did not enter"
        right = executor.submit(backend.run, requests[1])
        concurrent_results = (left.result(), right.result())
    for result in concurrent_results:
        assert result.outcome is SandboxOutcome.COMPLETED, (
            f"{result.outcome.value}: {result.detail}; {result.failure_context}; "
            f"exit={result.exit_code}; stderr={result.stderr!r}"
        )
    assert tuple(result.stdout for result in concurrent_results) == (
        "isolated-left",
        "isolated-right",
    ), concurrent_results
    assert (concurrent_scratch[0] / "value").read_text(encoding="utf-8") == "left"
    assert (concurrent_scratch[1] / "value").read_text(encoding="utf-8") == "right"
    require_drained(generation)


def test_native_selected_snapshots_overlap_without_shared_access(
    installation, public, native_directory, monkeypatch
):
    """Overlapping selected-file requests keep snapshots read-only and peer scratch inaccessible."""
    backend, generation = installation

    def temporary_directory(suffix=None, prefix=None, dir=None):
        """Register the real backend's disposable scratch before its standard cleanup owns it."""
        return str(
            native_directory(
                Path(dir or tempfile.gettempdir()),
                name=(prefix or "loop-native-test-") + uuid4().hex + (suffix or ""),
            )
        )

    monkeypatch.setattr(tempfile, "mkdtemp", temporary_directory)
    workspace = public[1]
    (workspace / ".ssh").mkdir()
    unselected = workspace / ".ssh/key"
    unselected.write_text("unselected-private-canary")
    executable = Path("/usr/bin/python3").resolve(strict=True)
    inputs = []
    for label in ("left", "right"):
        path = workspace / label
        path.write_text(label)
        inputs.append(path)
    requests = []
    original_run = backend.run

    def observe(request):
        """Observe owned preparation while delegating every request to real native enforcement."""
        requests.append(request)
        return original_run(request)

    monkeypatch.setattr(backend, "run", observe)
    code = (
        "UNSELECTED="
        + repr(str(unselected))
        + "\n"
        + r"""
import json,os,time
from pathlib import Path
fd=int(sys.argv[1].rsplit('/',1)[1])
label=os.read(fd,128).decode()
try: Path(UNSELECTED).read_text()
except OSError: pass
else: raise RuntimeError('Unselected private source was exposed')
scratch=Path(os.environ['TMPDIR'])
selected=scratch/'selected'/str(fd)
for change in (lambda: selected.write_text('bad'), lambda: selected.chmod(0o600), lambda: selected.unlink(), lambda: selected.rename(selected.with_name('moved'))):
 try: change()
 except OSError: pass
 else: raise RuntimeError('Selected snapshot was writable')
(scratch/'ready').write_text('ready')
while not (scratch/'peer.json').exists(): time.sleep(0.01)
peer=json.loads((scratch/'peer.json').read_text())
for value in peer:
 for action in (lambda: Path(value).read_bytes(), lambda: Path(value).write_text('bad')):
  try: action()
  except OSError: pass
  else: raise RuntimeError('Peer state was accessible')
print(label)
"""
    )
    code = "import sys\n" + code
    descriptors = [os.open(path, os.O_RDONLY) for path in inputs]

    def run(index):
        """Launch a protected system Python through the selected-descriptor native owner."""
        fd = descriptors[index]
        return backend.run_read_only_argv(
            executable, ("-I", "-c", code, f"/dev/fd/{fd}"), (fd,), time.monotonic() + 15
        )

    def ready(index, future):
        """Require actual payload entry before admitting the next overlapping request."""
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not future.done():
            if len(requests) > index:
                scratch = Path(requests[index].shell_environment()["TMPDIR"])
                if (scratch / "ready").exists():
                    return scratch
            time.sleep(0.01)
        pytest.fail(
            f"Native selected-file payload did not enter: {future.result() if future.done() else 'deadline'}"
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            left = executor.submit(run, 0)
            first = ready(0, left)
            baseline = profile_inventory(generation)
            right = executor.submit(run, 1)
            second = ready(1, right)
            for own, peer, peer_fd in (
                (first, second, descriptors[1]),
                (second, first, descriptors[0]),
            ):
                (own / "peer.json").write_text(
                    json.dumps([str(peer / "ready"), str(peer / "selected" / str(peer_fd))])
                )
            results = (left.result(), right.result())
        assert all(result is not None and result.exit_code == 0 for result in results), results
        assert tuple(result.stdout for result in results) == ("left\n", "right\n")
        assert_equivalent_profiles(baseline, profile_inventory(generation))
        assert all(path.read_text() == label for path, label in zip(inputs, ("left", "right")))
        assert not first.exists() and not second.exists()
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    require_drained(generation)


def test_native_scratch_hardlink_stays_read_only_at_both_spellings(
    installation, public, native_directory
):
    """The stable scratch mirror cannot write an outside inode through a preexisting hardlink."""
    backend, generation = installation
    workspace = public[1]
    scratch = native_directory(Path("/tmp"))
    outside = workspace / "outside-scratch-link"
    outside.write_text("preserve")
    os.link(outside, scratch / "linked")
    source = 'cat "$TMPDIR/linked"; if printf bad > "$TMPDIR/linked" 2>/dev/null; then exit 31; fi'
    result = backend.run(native_request(backend, workspace, scratch, source))
    assert result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0, result
    assert result.stdout == "preserve" and outside.read_text() == "preserve"
    require_drained(generation)


def test_installed_request_rejects_nonfinite_deadlines_before_entry(
    installation, public, native_directory
):
    """Invalid deadlines cannot enter the receiver; a finite control runs and drains."""
    backend, generation = installation
    workspace = public[1]
    scratch = native_directory(Path("/tmp"))
    marker = workspace / "deadline-marker"
    request = native_request(backend, workspace, scratch, "printf allowed > deadline-marker")

    for deadline in (float("nan"), float("inf"), float("-inf")):
        result = backend.run(replace(request, deadline=deadline))
        assert result.outcome is SandboxOutcome.INVALID, result
        assert not marker.exists()
    require_drained(generation)

    result = backend.run(request)
    assert result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0, result
    assert marker.read_text() == "allowed"
    require_drained(generation)


def test_installed_request_cancelled_during_serialization_does_not_execute(
    installation, public, native_directory, monkeypatch
):
    """Cancellation leaves no marker and drains; an uncancelled control writes the marker."""
    backend, generation = installation
    workspace = public[1]
    scratch = native_directory(Path("/tmp"))
    marker = workspace / "serialization-cancellation-marker"
    request = native_request(backend, workspace, scratch, f"printf allowed > {marker.name}")
    cancelled = threading.Event()
    serialize = SandboxRequest.to_wire

    def cancel_during_serialization(value):
        """Set the run cancellation signal after building the valid native request."""
        payload = serialize(value)
        cancelled.set()
        return payload

    with monkeypatch.context() as patch:
        patch.setattr(SandboxRequest, "to_wire", cancel_during_serialization)
        result = backend.run(request, cancellation_event=cancelled)

    assert result.outcome is SandboxOutcome.CANCELLED and not result.possible_effects
    assert not marker.exists()
    require_drained(generation)

    result = backend.run(request)
    assert result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0, result
    assert marker.read_text() == "allowed"
    require_drained(generation)


def test_installed_request_keeps_an_idle_cancellation_channel_open(
    installation, public, native_directory
):
    """Admit and complete a request without requiring EOF on its cancellation channel."""
    backend, generation = installation
    scratch = native_directory(Path("/tmp"))
    request = native_request(backend, public[1], scratch, "printf ready", seconds=15)

    result = backend.run(request, cancellation_event=threading.Event())

    assert result.outcome is SandboxOutcome.COMPLETED, result
    assert result.exit_code == 0 and result.stdout == "ready"
    require_drained(generation)


@pytest.mark.parametrize("detached_descendant", [False, True])
def test_installed_request_cancels_and_drains_admitted_source(
    installation, public, native_directory, detached_descendant
):
    """An explicit cancellation byte stops running source and detached namespace descendants."""
    backend, generation = installation
    workspace = public[1]
    scratch = native_directory(Path("/tmp"))
    source = (
        "/usr/bin/python3 -I -c "
        + shlex.quote(
            "import os,time\n"
            "if os.fork() == 0:\n"
            " os.setsid()\n"
            " print('ready',end='',flush=True)\n"
            " with open('cancellation-marker','w') as marker: marker.write('started')\n"
            "time.sleep(30)\n"
        )
        if detached_descendant
        else "printf ready; printf started > cancellation-marker; sleep 30"
    )
    request = native_request(backend, workspace, scratch, source, seconds=30)
    cancelled = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(backend.run, request, cancellation_event=cancelled)
        try:
            deadline = time.monotonic() + 10
            marker = workspace / "cancellation-marker"
            while not marker.exists() and not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.exists(), future.result() if future.done() else "No payload entry"
            started = time.monotonic()
            cancelled.set()
            result = future.result(timeout=5)
            assert time.monotonic() - started < 5
            assert result.outcome is SandboxOutcome.CANCELLED, result
            assert result.stdout == "ready" and result.exit_code is None
            require_drained(generation)
        finally:
            cancelled.set()


def test_installed_request_disconnect_cancels_admitted_source(
    installation, public, native_directory
):
    """Cancel an admitted client request and verify the tracked command is drained."""
    backend, generation = installation
    _interaction, workspace, _generation = public
    scratch = native_directory(Path("/tmp"))
    request = native_request(
        backend,
        workspace,
        scratch,
        "printf started > disconnect-marker; sleep 20",
        seconds=30,
    )
    record = NATIVE_INSTALLATION_DIRECTORY / "management.json"
    code = (
        "from pathlib import Path;import sys;"
        "from loop.execution.sandbox.linux.backend import LinuxNativeBackend;"
        "from loop.execution.sandbox.contracts import SandboxRequest;"
        "print(LinuxNativeBackend(Path(sys.argv[1])).run("
        "SandboxRequest.from_wire(sys.stdin.buffer.read())).outcome.value,flush=True)"
    )
    client = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", code, str(record)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            key: value for key, value in os.environ.items() if key not in {"TMPDIR", "TMP", "TEMP"}
        },
    )
    try:
        assert client.stdin is not None
        client.stdin.write(request.to_wire())
        client.stdin.close()
        deadline = time.monotonic() + 10
        marker = workspace / "disconnect-marker"
        while not marker.exists() and client.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if client.poll() is not None:
            assert client.stdout is not None and client.stderr is not None
            pytest.fail(
                f"Native client exited {client.returncode} before payload entry: "
                f"stdout={client.stdout.read()!r}; stderr={client.stderr.read()!r}"
            )
        assert marker.exists(), "native request did not enter its payload"
        client.send_signal(signal.SIGINT)
        assert client.wait(timeout=5) == 0
        assert client.stdout is not None and client.stdout.read().strip() == b"cancelled"
        assert client.stderr is not None and client.stderr.read() == b""
    finally:
        if client.poll() is None:
            client.kill()
            client.wait(timeout=5)
        if client.stdout is not None:
            client.stdout.close()
        if client.stderr is not None:
            client.stderr.close()
    require_drained(generation)


def test_installed_terminal_client_disconnect_drains_owned_pty(
    installation, public, native_directory
):
    """Losing a client with a live PTY cancels its authenticated receiver and descendants."""
    backend, generation = installation
    workspace = public[1]
    scratch = native_directory(Path("/tmp"))
    request = replace(
        native_request(
            backend,
            workspace,
            scratch,
            "printf started > terminal-disconnect-marker; exec /bin/sh -i",
            seconds=30,
        ),
        terminal=True,
        stream_output=True,
    )
    record = NATIVE_INSTALLATION_DIRECTORY / "management.json"
    code = (
        "from pathlib import Path;import sys;"
        "from loop.execution.sandbox.linux.backend import LinuxNativeBackend;"
        "from loop.execution.sandbox.contracts import SandboxRequest;"
        "print(LinuxNativeBackend(Path(sys.argv[1])).run("
        "SandboxRequest.from_wire(sys.stdin.buffer.read()),"
        "terminal_input=lambda:None,output_observer=lambda *_:None).outcome.value,flush=True)"
    )
    client = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", code, str(record)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            key: value for key, value in os.environ.items() if key not in {"TMPDIR", "TMP", "TEMP"}
        },
    )
    try:
        assert client.stdin is not None
        client.stdin.write(request.to_wire())
        client.stdin.close()
        deadline = time.monotonic() + 10
        marker = workspace / "terminal-disconnect-marker"
        while not marker.exists() and client.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "terminal request did not enter its payload"
        client.send_signal(signal.SIGINT)
        assert client.wait(timeout=5) == 0
        assert client.stdout is not None and client.stdout.read().strip() == b"cancelled"
        assert client.stderr is not None and client.stderr.read() == b""
    finally:
        if client.poll() is None:
            client.kill()
            client.wait(timeout=5)
        if client.stdout is not None:
            client.stdout.close()
        if client.stderr is not None:
            client.stderr.close()
    require_drained(generation)


def test_installed_receiver_rejects_retargeted_workspace_selection(
    installation, public, public_workspace_root, request, native_directory, monkeypatch
):
    """The receiver rejects a selected executable changed after request serialization."""
    if _SYSTEM_SH is None:
        pytest.skip("System shell is required")
    backend, generation = installation
    _, workspace, _ = public
    _, _, external_root = public_workspace_root
    directory = native_directory(external_root, name=_test_directory_name(request.node.nodeid))
    # Reuse the approved read-root spelling exercised by the hardlinked library control.
    directory = directory / "python-library"
    directory.mkdir()
    original = directory / "original"
    replacement = directory / "replacement"
    for executable in (original, replacement):
        shutil.copyfile(_SYSTEM_SH, executable)
        executable.chmod(0o755)
    selected = workspace / "fixture-tool"
    selected.symlink_to(original)
    details = original.stat()
    identity = ("./fixture-tool", selected, original, details.st_dev, details.st_ino)
    scratch = native_directory(Path("/tmp"))
    monkeypatch.setenv("PATH", os.defpath)

    def bound(source):
        """Capture the same selected executable for a distinct disposable command."""
        return SandboxRequest.create(
            source=source,
            cwd=workspace,
            workspace=workspace,
            read_roots=(directory,),
            write_roots=(),
            network=False,
            environment={"PATH": os.defpath, "LC_ALL": "C", "TMPDIR": str(scratch)},
            policy_version=backend.policy_version,
            deadline=time.monotonic() + 20,
            workspace_id="selected-executable-native-test",
            scratch_root=scratch,
            workspace_executable_identities=(identity,),
        )

    positive = backend.run(bound('printf positive > "$TMPDIR/receiver-positive"'))
    assert positive.outcome is SandboxOutcome.COMPLETED, positive
    assert (scratch / "receiver-positive").read_text(encoding="utf-8") == "positive"
    stale = bound('printf touched > "$TMPDIR/receiver-stale-marker"')
    payload = stale.to_wire()
    selected.unlink()
    selected.symlink_to(replacement)
    socket_path = SYSTEM_RUNTIME_DIRECTORY / (
        NATIVE_UNIT_PREFIX + generation + SYSTEMD_SOCKET_UNIT_SUFFIX
    )
    frame = NATIVE_REQUEST_MAGIC + struct.pack(REQUEST_LENGTH_FORMAT, len(payload)) + payload
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(20)
        connection.connect(str(socket_path))
        connection.sendall(frame)
        assert connection.recv(MAX_NATIVE_RESULT_BYTES) == b""
    assert not (scratch / "receiver-stale-marker").exists()
    require_drained(generation)


@pytest.mark.parametrize(("cancel", "halfclose"), [(False, True), (True, False), (True, True)])
def test_installed_receiver_observes_queued_cancellation_before_spawn(
    installation, public, native_directory, environment_patch, cancel, halfclose
):
    """Queued control preserves execution ordering and identifies the actual response sender."""
    backend, generation = installation
    workspace = public[1]
    environment_patch.setattr(tempfile, "tempdir", "/tmp")
    scratch = native_directory(Path("/tmp"))
    marker = workspace / "queued-control-marker"
    request = native_request(
        backend, workspace, scratch, "printf entered > queued-control-marker; printf control"
    )
    socket_path = SYSTEM_RUNTIME_DIRECTORY / (
        NATIVE_UNIT_PREFIX + generation + SYSTEMD_SOCKET_UNIT_SUFFIX
    )
    original = socket_path.lstat()
    assert stat.S_ISSOCK(original.st_mode)
    assert (original.st_uid, original.st_gid) == (os.getuid(), os.getgid())
    identity = (
        original.st_dev,
        original.st_ino,
        original.st_mode,
        original.st_uid,
        original.st_gid,
        original.st_nlink,
        original.st_mtime_ns,
        original.st_ctime_ns,
    )
    payload = request.to_wire()
    frame = NATIVE_REQUEST_MAGIC + struct.pack(REQUEST_LENGTH_FORMAT, len(payload)) + payload
    require_drained(generation)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(20)
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            connection.connect(str(socket_path))
            peer = struct.unpack(
                "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            assert peer[0] > 0 and peer[1:] == (0, 0)
            current = socket_path.lstat()
            assert identity == (
                current.st_dev,
                current.st_ino,
                current.st_mode,
                current.st_uid,
                current.st_gid,
                current.st_nlink,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            connection.sendall(frame + (NATIVE_CANCEL_FRAME if cancel else b""))
            if halfclose:
                connection.shutdown(socket.SHUT_WR)
            response = bytearray()
            sender = None
            while True:
                block, ancillary, flags, _ = connection.recvmsg(
                    4096, socket.CMSG_SPACE(struct.calcsize("3i"))
                )
                if not block:
                    break
                assert not flags & socket.MSG_CTRUNC
                assert len(ancillary) == 1
                level, kind, credentials = ancillary[0]
                assert (level, kind) == (socket.SOL_SOCKET, socket.SCM_CREDENTIALS)
                actual = struct.unpack("3i", credentials)
                assert actual[0] > 0 and actual[1:] == (os.getuid(), os.getgid())
                assert actual[0] != peer[0]
                if sender is None:
                    sender = actual
                assert actual == sender
                response.extend(block)
                assert len(response) <= MAX_NATIVE_RESULT_BYTES
        result = CommandProcessResult.from_wire(bytes(response), request.attempt_id)
        if cancel:
            assert result.outcome is SandboxOutcome.CANCELLED
            assert result.possible_effects is False
            assert not marker.exists()
        else:
            assert result.outcome is SandboxOutcome.COMPLETED
            assert result.stdout == "control"
            assert marker.read_text() == "entered"
        require_drained(generation)
    finally:
        deadline = request.deadline + NATIVE_RESULT_DELIVERY_SECONDS
        while True:
            try:
                require_drained(generation)
                break
            except AssertionError:
                if time.monotonic() >= deadline:
                    raise


@pytest.mark.parametrize("victim", ["receiver", "launcher"])
def test_installed_process_death_drains_its_running_command(
    installation, public, native_directory, environment_patch, victim
):
    """Receiver or launcher death drains the owned namespace without losing effect status."""
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        pytest.skip("Receiver identity qualification requires Linux pidfd signaling")
    backend, generation = installation
    workspace = public[1]
    environment_patch.setattr(tempfile, "tempdir", "/tmp")
    scratch = native_directory(Path("/tmp"))
    nonce = uuid4().hex
    marker_prefix = "receiver-death-"
    marker = workspace / (marker_prefix + nonce)
    payload = (
        "import os,signal,sys,time\n"
        "signal.alarm(15)\n"
        "if os.fork()==0:\n"
        " os.setsid()\n"
        " signal.alarm(15)\n"
        " signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        f" open({marker_prefix!r}+sys.argv[1],'w').write(sys.argv[1])\n"
        "time.sleep(12)\n"
    )
    source = (
        f"{shlex.quote(str(SYSTEM_PYTHON_PATH))} -I -c {shlex.quote(payload)} {shlex.quote(nonce)}"
    )
    request = native_request(backend, workspace, scratch, source, seconds=25)
    record = NATIVE_INSTALLATION_DIRECTORY / "management.json"
    installed, _ = read_generation(record)
    scope = Path("/sys/fs/cgroup") / ("loopgen_" + generation + ".slice")
    code = (
        "from pathlib import Path;import json,sys;"
        "from loop.execution.sandbox.linux.backend import LinuxNativeBackend;"
        "from loop.execution.sandbox.contracts import SandboxRequest;"
        "r=LinuxNativeBackend(Path(sys.argv[1])).run("
        "SandboxRequest.from_wire(sys.stdin.buffer.read()));"
        "print(json.dumps({'outcome':r.outcome.value,'exit_code':r.exit_code,"
        "'possible_effects':r.possible_effects,"
        "'detail':r.detail}),"
        "flush=True)"
    )
    require_drained(generation)
    client = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", code, str(record)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            key: value for key, value in os.environ.items() if key not in {"TMPDIR", "TMP", "TEMP"}
        },
    )
    try:
        assert client.stdin is not None
        client.stdin.write(request.to_wire())
        client.stdin.close()
        deadline = time.monotonic() + 10
        while client.poll() is None and time.monotonic() < deadline:
            if marker.exists() and marker.read_text() == nonce:
                break
            time.sleep(0.02)
        if not marker.exists() and client.poll() is not None:
            assert client.stdout is not None and client.stderr is not None
            pytest.fail(
                f"Native client exited {client.returncode} before payload entry: "
                f"stdout={client.stdout.read()!r}; stderr={client.stderr.read()!r}"
            )
        assert marker.exists() and marker.read_text() == nonce, "Native payload did not enter"

        def process_arguments(pid):
            """Read an owned process argument vector for cgroup-bound identity checks."""
            try:
                path = Path("/proc") / str(pid)
                if path.stat().st_uid != os.getuid():
                    return ()
                return tuple(path.joinpath("cmdline").read_bytes().split(b"\0"))
            except FileNotFoundError:
                return ()

        groups = []
        for file in scope.glob("*/cgroup.procs"):
            try:
                members = {int(value) for value in file.read_text().splitlines()}
            except FileNotFoundError:
                continue
            payloads = {pid for pid in members if nonce.encode() in process_arguments(pid)}
            receivers = {
                pid
                for pid in members
                if os.fsencode(record.parent / "loop_receiver.py") in process_arguments(pid)
            }
            if payloads and receivers:
                groups.append((file, payloads, receivers))
        assert len(groups) == 1, "Cannot uniquely identify the owned native service"
        file, payloads, receivers = groups[0]
        assert len(receivers) == 1, "Cannot uniquely identify the owned receiver"
        receiver = next(iter(receivers))
        target = receiver
        if victim == "launcher":
            launchers = []
            for pid in {int(value) for value in file.read_text().splitlines()}:
                try:
                    process = Path("/proc") / str(pid)
                    executable = process.joinpath("exe").stat()
                    parent = int(process.joinpath("stat").read_text().rsplit(") ", 1)[1].split()[1])
                    if (executable.st_dev, executable.st_ino) == installed.helper.identity[
                        :2
                    ] and parent == receiver:
                        launchers.append(pid)
                except FileNotFoundError:
                    continue
            assert len(launchers) == 1, "Cannot uniquely identify the owned namespace launcher"
            target = launchers[0]
        original = file.parent.stat()
        descriptor = os.pidfd_open(target)
        try:
            current = file.parent.stat()
            members = {int(value) for value in file.read_text().splitlines()}
            assert (current.st_dev, current.st_ino) == (original.st_dev, original.st_ino)
            assert receiver in members and target in members and payloads & members
            assert os.fsencode(record.parent / "loop_receiver.py") in process_arguments(receiver)
            assert any(nonce.encode() in process_arguments(pid) for pid in payloads & members)
            if victim == "launcher":
                process = Path("/proc") / str(target)
                executable = process.joinpath("exe").stat()
                assert (executable.st_dev, executable.st_ino) == installed.helper.identity[:2]
                parent = int(process.joinpath("stat").read_text().rsplit(") ", 1)[1].split()[1])
                assert parent == receiver
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        finally:
            os.close(descriptor)

        assert client.wait(timeout=8) == 0
        assert client.stdout is not None and client.stderr is not None
        result = json.loads(client.stdout.read())
        assert result["possible_effects"] is True, result
        if victim == "receiver":
            assert result["outcome"] == "invalid", result
        else:
            assert result["outcome"] == "completed" and result["exit_code"] == -signal.SIGKILL, (
                result
            )
        assert client.stderr.read() == b""
        require_drained(generation)
        control = backend.run(native_request(backend, workspace, scratch, "printf control"))
        assert control.outcome is SandboxOutcome.COMPLETED and control.stdout == "control"
        require_drained(generation)
    finally:
        failure = sys.exc_info()[1]
        cleanup_errors = []
        try:
            if client.poll() is None:
                client.kill()
                client.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as error:
            cleanup_errors.append(error)
        if client.stdout is not None:
            client.stdout.close()
        if client.stderr is not None:
            client.stderr.close()
        while True:
            try:
                require_drained(generation)
                break
            except AssertionError as error:
                if time.monotonic() < request.deadline + NATIVE_RESULT_DELIVERY_SECONDS:
                    continue
                cleanup_errors.append(error)
                break
        if cleanup_errors:
            if failure is None:
                raise ExceptionGroup("Native service cleanup failed", cleanup_errors)
            for error in cleanup_errors:
                failure.add_note(f"Native service cleanup also failed: {error}")


def test_compiler_contention_refuses_before_payload_entry(installation, public, native_directory):
    """Do not execute request source while another authenticated compiler owns the lock."""
    backend, generation = installation
    _interaction, workspace, _generation = public
    scratch = native_directory(Path("/tmp"))
    marker = workspace / "compiler-contention-marker"
    record = NATIVE_INSTALLATION_DIRECTORY / "management.json"
    compiler = bind_host_file(record.parent / "compiler.lock")
    with compiler.open_verified() as descriptor:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = backend.run(
            native_request(
                backend,
                workspace,
                scratch,
                f"printf entered > {shlex.quote(str(marker))}",
                seconds=1,
            )
        )
    assert result.outcome in {SandboxOutcome.INVALID, SandboxOutcome.TIMED_OUT}, result
    assert not marker.exists()
    require_drained(generation)


@pytest.mark.skipif(
    not _CAPACITY_TEST_OPT_IN,
    reason="Set LOOP_NATIVE_CAPACITY_TEST=disposable-linux-host for profile lifecycle stress",
)
def test_installed_profile_lifecycle_reclaims_completed_variants(
    installation, public, native_directory
):
    """Many distinct completed policies stay bounded and continue executing without repair."""
    backend, generation = installation
    _interaction, workspace, _generation = public
    installed, authority = read_generation(NATIVE_INSTALLATION_DIRECTORY / "management.json")
    capacity, _ = generation_limits(installed.config.reservation, generation)

    def inventory():
        """Observe enforcing kernel fingerprints bound to the original installation."""
        with authority.open_verified():
            return profile_inventory(generation)

    initial = inventory()
    scratch = native_directory(Path("/tmp"))
    marker = workspace / "profile-capacity-marker"
    for index in range(3 * capacity + 1):
        extra = native_directory(
            workspace.parent, name=f"loop-native-test-profile-capacity-read-{index}"
        )
        marker.unlink(missing_ok=True)
        request = native_request(
            backend,
            workspace,
            scratch,
            f"printf entered > {shlex.quote(str(marker))}",
            read_roots=(extra,),
        )
        profile = payload_profile(
            request, scratch, installed.artifacts["cache.lock"].path, runtime_read_roots()
        )
        result = backend.run(request)
        observed = inventory()
        assert result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0, result
        assert marker.read_text() == "entered"
        assert observed["bootstrap"] == initial["bootstrap"]
        assert any(name.startswith(profile + "_") for name in observed)
        assert len(observed) <= 2
    require_drained(generation)


@pytest.mark.skipif(
    not _CAPACITY_TEST_OPT_IN,
    reason="Set LOOP_NATIVE_CAPACITY_TEST=disposable-linux-host for profile lifecycle stress",
)
def test_public_profile_lifecycle_handles_many_workspaces(installation, public, native_directory):
    """Public dispatch handles many distinct workspaces without prompts or profile exhaustion."""
    _, generation = installation
    interaction, workspace, _ = public
    installed, _ = read_generation(NATIVE_INSTALLATION_DIRECTORY / "management.json")
    capacity, _ = generation_limits(installed.config.reservation, generation)
    for index in range(3 * capacity + 1):
        current = native_directory(workspace.parent, name=f"loop-native-test-public-{index}")
        result = invoke((interaction, current, generation), "printf public-ready", read_only=True)
        assert result["ok"], result
        assert result["result"]["stdout"]["content"] == "public-ready"
        assert len(profile_inventory(generation)) <= 2
    assert not interaction.prompts
    require_drained(generation)


@pytest.mark.skipif(
    not _CAPACITY_TEST_OPT_IN,
    reason="Set LOOP_NATIVE_CAPACITY_TEST=disposable-linux-host for profile lifecycle stress",
)
def test_installed_profile_capacity_waits_for_live_request(installation, public, native_directory):
    """A concurrent request waits for a live profile slot without replay or lost confinement."""
    backend, generation = installation
    _, workspace, _ = public
    installed, _ = read_generation(NATIVE_INSTALLATION_DIRECTORY / "management.json")
    capacity, _ = generation_limits(installed.config.reservation, generation)
    directories = [
        native_directory(workspace.parent, name=f"loop-native-test-live-capacity-{index}")
        for index in range(capacity + 1)
    ]
    scratches = [native_directory(Path("/tmp")) for _ in directories]
    source = (
        "if printf forbidden > AGENTS.md 2>/dev/null; then exit 17; fi; "
        "printf ready > ready; while [ ! -e release ]; do sleep 0.2; done; printf finished"
    )
    requests = [
        native_request(backend, directory, scratch, source, seconds=120)
        for directory, scratch in zip(directories, scratches)
    ]
    with ThreadPoolExecutor(max_workers=capacity + 1) as executor:
        futures = [executor.submit(backend.run, request) for request in requests]
        try:
            deadline = time.monotonic() + 75
            while time.monotonic() < deadline:
                ready = [
                    index for index, root in enumerate(directories) if (root / "ready").exists()
                ]
                if len(ready) >= capacity or any(future.done() for future in futures):
                    break
                time.sleep(0.1)
            assert len(ready) == capacity, [
                (index, future.result()) for index, future in enumerate(futures) if future.done()
            ]
            assert not any(future.done() for future in futures)
            waiting = next(index for index in range(len(directories)) if index not in ready)
            (directories[ready[0]] / "release").write_text("release")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not (directories[waiting] / "ready").exists():
                if futures[waiting].done():
                    break
                time.sleep(0.1)
            assert (directories[waiting] / "ready").exists(), futures[waiting].result()
            assert all(not (root / "AGENTS.md").exists() for root in directories)
        finally:
            for directory in directories:
                (directory / "release").write_text("release")
        results = [future.result(timeout=30) for future in futures]
    assert all(
        result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0 for result in results
    ), results
    assert all(result.stdout == "finished" for result in results)
    require_drained(generation)


if __name__ == "__main__":
    public_client()
