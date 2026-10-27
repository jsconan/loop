"""Qualify current installed Linux timeout delivery and resistance to recursive admission."""

import fcntl
import hashlib
import http.server
import importlib.metadata
import json
import os
import pwd
import re
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
from importlib.resources import files
from pathlib import Path
from uuid import uuid4

import pytest

from loop import ApprovalChoice, PermissionManager, RuntimeEnvironment, ToolRegistry
from loop.constants import NATIVE_INSTALLATION_DIRECTORY, NATIVE_RESULT_DELIVERY_SECONDS
from loop.execution import CommandExecutionService
from loop.execution.sandbox import SandboxOutcome, SandboxRequest, select_sandbox_backend
from loop.execution.sandbox.linux import LinuxNativeBackend
from loop.execution.sandbox.linux.catalog import runtime_catalog
from loop.execution.sandbox.linux.constants import NATIVE_POLICY_VERSION, NATIVE_SCRATCH_PATH
from loop.execution.sandbox.linux.host import bind_host_file, runtime_read_roots
from loop.execution.sandbox.linux.installation import generation_limits
from loop.execution.sandbox.linux.policy import payload_profile
from loop.execution.sandbox.linux.records import read_generation
from loop.execution.sandbox.linux.runtime import execution_authority
from loop.instructions import InstructionsManager
from loop.permissions.audit import SQLitePermissionAudit
from loop.tooling import ToolRegistration, ToolRuntimeSettings
from loop.tools.files import read_text_file, search_text, write_text_file
from loop.tools.system import resolve_executable, run_command
from loop.utils import PathHolder

_ALIAS_DIRECTORY = os.environ.get("LOOP_NATIVE_SOCKET_ALIAS_DIRECTORY")
_CAPACITY_TEST_OPT_IN = os.environ.get("LOOP_NATIVE_CAPACITY_TEST") == "disposable-linux-host"
_RUNTIME_FIXTURE = os.environ.get("LOOP_NATIVE_RUNTIME_FIXTURE")
_RUNTIME_FIXTURE_MARKER = b"Loop private-runtime qualification fixture v1\n"
_QUALIFICATION_UV = shutil.which("uv")
_HOST_HOME = Path.home().resolve()
_PUBLIC_ROOT_MARKER = b"Loop installed native qualification fixture v1\n"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.linux,
    pytest.mark.skipif(sys.platform != "linux", reason="Requires installed Linux native authority"),
]


class RefusingHost:
    """Fail any unintended host execution during native public qualification."""

    def run_host_command(self, request):
        """Reject a host request that should never have been offered or authorized."""
        raise AssertionError("Native qualification attempted host execution")


class Interaction:
    """Approve explicit sandbox grants and reject every host offer."""

    prompts: list[str]
    infos: list[str]
    configuration_path: Path | None

    def __init__(self):
        self.prompts = []
        self.infos = []
        self.configuration_path = None

    def prompt(self, message=None, **kwargs):
        """Record each permission decision and reject unintended host recovery."""
        self.prompts.append(str(message))
        assert "without the OS sandbox" not in str(message)
        return ApprovalChoice.ONCE if message == "Proceed?" else "approve"

    def info(self, message):
        """Accept the permission service's informational text."""
        self.infos.append(str(message))


@pytest.fixture(scope="module")
def installation():
    """Require explicitly installed current runtime bytes without provisioning or repairing them."""
    record = NATIVE_INSTALLATION_DIRECTORY / "management.json"
    if not record.exists():
        pytest.skip(
            "Current native installation must be explicitly provisioned before qualification"
        )
    execution_authority(record)
    generation = json.loads(record.read_text())["config"]["generation"]
    contents, _, _, _ = runtime_catalog(record.parent, generation)
    for name, data in contents.items():
        if name.startswith("loop/"):
            assert (record.parent / name).read_bytes() == data, f"Installed runtime differs: {name}"
    assert (record.parent / "loop_receiver.py").read_bytes() == files(
        "loop.execution.sandbox.linux"
    ).joinpath("resources/receiver.py").read_bytes()
    backend = select_sandbox_backend()
    assert isinstance(backend, LinuxNativeBackend), backend.capability_failure()
    assert backend.capability_failure() is None
    return backend, generation


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


def _test_directory_name(node_id):
    """Return a stable, bounded workspace component for one collected test case."""
    return "loop-native-test-" + hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:32]


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


def invoke_many(public, calls, *, permission_workspace=None):
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
    )


def invoke(public, command, **grants):
    """Run one public client call outside pytest's process-local temporary-root replacement."""
    return invoke_many(public, [(command, grants)])[0]


def invoke_registered(public_registry, calls):
    """Call registered native tools together so opaque grants stay bound to one service."""
    interaction, workspace, generation, audit_path = public_registry
    return invoke_tools(
        (interaction, workspace, generation),
        calls,
        audit_path=audit_path,
        workspace_id="native-public-review",
    )


def native_request(
    backend, workspace, scratch, source, *, read_roots=(), network=False, seconds=15, aliases=()
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
        aliases=aliases,
    )


def public_client():
    """Run registered public calls with authenticated native authority and no host fallback."""
    value = json.loads(sys.stdin.read())
    backend = LinuxNativeBackend()
    failure = backend.capability_failure()
    assert failure is None, failure
    interaction = Interaction()
    instructions = None
    if value.get("model_paths"):
        scratch = Path(value["configuration_path"]).parent / "instruction-scratch"
        scratch.mkdir(mode=0o700, exist_ok=True)
        instructions = InstructionsManager(
            runtime_environment=RuntimeEnvironment(Path(value["workspace"]), scratch)
        )
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
    if instructions is not None:
        permissions.bind_temporary_directory(PathHolder(scratch))
    registry = ToolRegistry(
        [
            run_command,
            resolve_executable,
            read_text_file,
            search_text,
            write_text_file,
            ToolRegistration(run_command, name="renamed_command"),
        ],
        permission_manager=permissions,
        execution_service=CommandExecutionService(backend, RefusingHost()),
        settings=ToolRuntimeSettings(command_timeout=value.get("seconds", 2)),
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
            assert name == "bootstrap" or re.fullmatch(r"payload_[a-f0-9]{64}", name)
            assert (entry / "mode").read_text().strip() == "enforce"
            fingerprint = (entry / "sha256").read_bytes().strip()
            assert re.fullmatch(rb"[a-f0-9]{64}", fingerprint)
            assert name not in observed
            observed[name] = fingerprint
        assert "bootstrap" in observed
        return observed


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
        result["ok"] and result["result"]["stdout"]["content"] == "ordinary-data$WORKSPACE_ROOT"
        for result in repeated
    ), repeated
    assert profile_inventory(generation) == baseline
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
    assert profile_inventory(generation) == baseline
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
    assert "timeout-out" in metadata["stdout"]["content"]
    assert "timeout-err" in metadata["stderr"]["content"]
    for name in ("stdout", "stderr"):
        stream = metadata[name]
        assert stream["capture_complete"] == (stream["discarded_characters"] == 0)
    assert elapsed <= 2 + NATIVE_RESULT_DELIVERY_SECONDS + 2
    require_drained(public[2])


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


def test_public_uv_runs_a_workspace_virtualenv_with_its_user_local_runtime(public, monkeypatch):
    """Run uv and a workspace Python entry point through the installed public tool."""
    uv = _QUALIFICATION_UV
    if uv is None:
        pytest.skip("uv is not available on the qualification PATH")

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
    interpreter_name = (project_venv / "bin/python").resolve(strict=True).name
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
    (venv_bin / "python").symlink_to(interpreter)
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
    for name in (
        "pytest",
        "pytest-cov",
        "coverage",
        "pluggy",
        "iniconfig",
        "packaging",
        "pygments",
    ):
        distribution = importlib.metadata.distribution(name)
        for entry in distribution.files or ():
            relative = Path(entry)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix == ".pyc":
                continue
            source = Path(distribution.locate_file(entry))
            if source.is_file():
                target = site_packages / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    pytest_entry = venv_bin / "pytest"
    pytest_entry.write_text(
        f"#!{venv_bin / 'python'}\n"
        "import sys\nfrom pytest import console_main\nsys.exit(console_main())\n",
        encoding="utf-8",
    )
    pytest_entry.chmod(0o755)
    (workspace / "sample.py").write_text("def choose(value):\n    return 1 if value else 2\n")
    (workspace / "test_sample.py").write_text(
        "from sample import choose\n\n\ndef test_positive():\n    assert choose(True) == 1\n"
        "\n\ndef test_negative():\n    assert choose(False) == 2\n"
    )
    monkeypatch.setenv(
        "PATH", os.pathsep.join((str(venv_bin), str(Path(uv).parent), os.environ["PATH"]))
    )
    (result,) = invoke_tools(
        public,
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": "uv run pytest -p no:cacheprovider --cov=sample --cov-branch --cov-report=term-missing --cov-fail-under=100",
                    "cwd": "$WORKSPACE_ROOT",
                    "read_only": True,
                },
            }
        ],
        model_paths=True,
        seconds=20,
    )
    assert result["ok"], result
    assert result["result"]["exit_code"] == 0
    output = result["result"]["stdout"]["content"]
    assert "2 passed" in output and "100%" in output
    assert not (workspace / ".coverage").exists()
    assert not interaction.prompts
    ruff = project_venv / "bin/ruff"
    assert ruff.is_file(), "The qualification environment needs its installed Ruff executable"
    shutil.copyfile(ruff, venv_bin / "ruff")
    (venv_bin / "ruff").chmod(0o755)
    (lint,) = invoke_tools(
        public,
        [
            {
                "tool": "run_command",
                "arguments": {
                    "command": "uv run ruff check sample.py test_sample.py",
                    "cwd": "$WORKSPACE_ROOT",
                    "read_only": True,
                },
            }
        ],
        model_paths=True,
        seconds=20,
    )
    assert lint["ok"] and lint["result"]["exit_code"] == 0, json.dumps(lint)
    assert not (workspace / ".ruff_cache").exists()
    assert not interaction.prompts
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
                "import os,socket,sys;"
                + (
                    "pid=os.fork();\n"
                    "if pid:\n"
                    " _,status=os.waitpid(pid,0);sys.exit(os.waitstatus_to_exitcode(status))\n"
                    if descendant
                    else ""
                )
                + "s=socket.socket(socket.AF_UNIX);s.settimeout(1);\ntry:\n s.connect("
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


def test_permitted_local_pathname_ipc_remains_usable(public):
    """Ordinary local Unix stream IPC between confined children remains available."""
    code = (
        "import os,socket; s=socket.socket(socket.AF_UNIX);s.bind('ordinary.socket');"
        "s.listen();pid=os.fork();\nif pid==0:\n c=socket.socket(socket.AF_UNIX);"
        "c.connect('ordinary.socket');c.sendall(b'local-ipc');os._exit(0)\n"
        "c,_=s.accept();print(c.recv(9).decode());os.waitpid(pid,0)"
    )
    result = invoke(public, "/usr/bin/python3 -I -c " + shlex.quote(code))
    assert result["result"]["stdout"]["content"] == "local-ipc\n"
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
    result = invoke(public, "/usr/bin/python3 -I -c " + shlex.quote(code), network=True)
    assert result["ok"], result
    assert json.loads(result["result"]["stdout"]["content"]) == direct
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
        result = invoke(public, prefix + shlex.quote(code), network=True)
        assert result["ok"], result
        assert result["result"]["stdout"]["content"].strip() == direct
        wrong_name = invoke(public, prefix + shlex.quote(source("127.0.0.1")), network=True)
        assert not wrong_name["ok"]
        assert "CERTIFICATE_VERIFY_FAILED" in wrong_name["problem"]["metadata"]["stderr"]["content"]
        untrusted = invoke(public, "/usr/bin/python3 -I -c " + shlex.quote(code), network=True)
        assert not untrusted["ok"]
        assert "CERTIFICATE_VERIFY_FAILED" in untrusted["problem"]["metadata"]["stderr"]["content"]
        offline = invoke(public, prefix + shlex.quote(code), network=False)
        assert not offline["ok"]
        assert direct not in json.dumps(offline)
        protected = invoke(public, "cat " + shlex.quote(str(key)), network=True)
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
    (outside / "secret").write_text("broader-control")
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
        require_drained(generation)


def test_public_executable_lookup_and_exact_file_authority(
    public_registry, public_workspace_root, request, native_directory, monkeypatch
):
    """Keep exact PATH executable authority narrow while explicit folder grants stay visible."""
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
        "#!/bin/sh\n"
        f"if /usr/bin/cat {shlex.quote(str(sibling))} >/dev/null 2>&1; then\n"
        "  printf sibling-visible\n"
        "else\n"
        "  printf exact-file-only\n"
        "fi\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tool_directory) + os.pathsep + os.environ["PATH"])

    lookup, explicit, exact = invoke_registered(
        public_registry,
        [
            {
                "tool": "resolve_executable",
                "arguments": {"name": executable.name, "cwd": str(workspace)},
            },
            {
                "tool": "run_command",
                "arguments": {
                    "command": executable.name,
                    "cwd": str(workspace),
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
                    "cwd": str(workspace),
                    "read_only": True,
                },
            },
        ],
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
    assert any(
        "Read the installed owned-tool tool folder" in message for message in interaction.infos
    )
    assert all(
        str(tool_directory) not in message for message in interaction.prompts + interaction.infos
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

        approved = invoke(public, source, read_only=True, network=True)
        assert approved["result"]["exit_code"] == 0, approved
        listener.settimeout(2)
        assert listener.recvfrom(32)[0] == b"granted"
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

    def aliases():
        """Keep each request's host symlink private while binding the same workspace authority."""
        parent = native_directory(Path("/tmp"))
        alias = parent / "r0"
        alias.symlink_to(workspace)
        return (("$WORKSPACE_ROOT", alias, workspace),)

    first_aliases, second_aliases = aliases(), aliases()
    first = native_request(
        backend,
        workspace,
        first_scratch,
        f"cd {shlex.quote(str(first_aliases[0][1]))}/; "
        + 'printf first > "$TMPDIR/value"; /usr/bin/cat "$TMPDIR/value"',
        aliases=first_aliases,
    )
    second_source = foreign_path_checks(first_scratch / "value") + (
        'printf second > "$TMPDIR/value"; /usr/bin/cat "$TMPDIR/value"'
    )
    second = native_request(
        backend,
        workspace,
        second_scratch,
        f"cd {shlex.quote(str(second_aliases[0][1]))}/; " + second_source,
        aliases=second_aliases,
    )

    assert payload_profile(first, first_scratch, cache_lock, runtime_roots) == payload_profile(
        second, second_scratch, cache_lock, runtime_roots
    )
    first_result = backend.run(first)
    second_result = backend.run(second)
    assert first_result.outcome is SandboxOutcome.COMPLETED, first_result
    assert second_result.outcome is SandboxOutcome.COMPLETED, second_result
    assert first_result.stdout == "first"
    assert second_result.stdout == "second"
    assert (first_scratch / "value").read_text(encoding="utf-8") == "first"
    assert (second_scratch / "value").read_text(encoding="utf-8") == "second"

    concurrent_scratch = tuple(native_directory(Path("/tmp")) for _ in range(2))

    def concurrent_request(label, own_scratch, other_scratch):
        """Create one distinct scratch request that probes the peer's private path."""
        peer = "right" if label == "left" else "left"
        own_aliases = aliases()
        source = (
            f"cd {shlex.quote(str(own_aliases[0][1]))}/; "
            f'printf {label} > "$TMPDIR/value"; printf ready > {label}-ready; '
            f"while test ! -f {peer}-ready; do /usr/bin/sleep 0.01; done; "
            + foreign_path_checks(other_scratch / "value")
            + f"printf isolated-{label}"
        )
        return native_request(backend, workspace, own_scratch, source, aliases=own_aliases)

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
        assert profile_inventory(generation) == baseline
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
        "from loop.execution.sandbox.linux import LinuxNativeBackend;"
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
    reason="Set LOOP_NATIVE_CAPACITY_TEST=disposable-linux-host to consume the installed profile cache",
)
def test_installed_profile_capacity_refuses_source_before_entry(
    installation, public, native_directory
):
    """Refuse payload entry when the add-only installed profile cache is exhausted."""
    backend, generation = installation
    _interaction, workspace, _generation = public
    installed, authority = read_generation(NATIVE_INSTALLATION_DIRECTORY / "management.json")
    capacity, _ = generation_limits(installed.config.reservation, generation)

    def inventory():
        """Observe enforcing kernel fingerprints bound to the original installation."""
        with authority.open_verified():
            return profile_inventory(generation)

    initial = inventory()
    available = capacity - (len(initial) - 1)
    assert 0 < available <= capacity, "Capacity qualification needs unused profile slots"
    scratch = native_directory(Path("/tmp"))
    marker = workspace / "profile-capacity-marker"
    current = initial
    first_extra = None
    for index in range(available + 1):
        extra = native_directory(
            workspace.parent, name=f"loop-native-test-profile-capacity-read-{index}"
        )
        first_extra = first_extra or extra
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
        assert profile not in current
        result = backend.run(request)
        observed = inventory()
        if index < available:
            assert result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0, result
            assert marker.read_text() == "entered"
            assert set(observed) == set(current) | {profile}
            assert all(observed[name] == data for name, data in current.items())
            current = observed
        else:
            assert len(current) == capacity + 1
            assert result.outcome is SandboxOutcome.INVALID, result
            assert not marker.exists()
            assert observed == current
    assert len(current) == capacity + 1
    reusable = native_request(
        backend,
        workspace,
        scratch,
        f"printf reused > {shlex.quote(str(marker))}",
        read_roots=(first_extra,),
    )
    result = backend.run(reusable)
    assert result.outcome is SandboxOutcome.COMPLETED and result.exit_code == 0, result
    assert marker.read_text() == "reused"
    assert inventory() == current
    require_drained(generation)


if __name__ == "__main__":
    public_client()
