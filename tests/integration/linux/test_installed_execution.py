"""Qualify current installed Linux timeout delivery and resistance to recursive admission."""

import hashlib
import http.server
import json
import os
import pwd
import shlex
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import tempfile
import threading
import time
from importlib.resources import files
from pathlib import Path

import pytest

from loop import PermissionManager, ToolRegistry
from loop.constants import NATIVE_INSTALLATION_DIRECTORY, NATIVE_RESULT_DELIVERY_SECONDS
from loop.execution import CommandExecutionService
from loop.execution.sandbox import SandboxRequest, select_sandbox_backend
from loop.execution.sandbox.linux import LinuxNativeBackend
from loop.execution.sandbox.linux.catalog import runtime_catalog
from loop.execution.sandbox.linux.runtime import execution_authority
from loop.tooling import ToolRuntimeSettings
from loop.tools.system import run_command

_ALIAS_DIRECTORY = os.environ.get("LOOP_NATIVE_SOCKET_ALIAS_DIRECTORY")

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

    def __init__(self):
        self.prompts = []

    def prompt(self, message=None, **kwargs):
        """Record each permission decision and reject unintended host recovery."""
        self.prompts.append(str(message))
        assert "without the OS sandbox" not in str(message)
        return "approve"

    def info(self, message):
        """Accept the permission service's informational text."""


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
def public(installation, tmp_path, request, native_directory, monkeypatch):
    """Own public fixtures while a real client uses the installed receiver's system scratch root."""
    _, generation = installation
    workspace = tmp_path.resolve()
    monkeypatch.setattr(tempfile, "tempdir", "/tmp")
    if getattr(request.node, "callspec", None) and request.node.callspec.params.get("alias"):
        assert _ALIAS_DIRECTORY, (
            "Alias qualification requires an explicitly provisioned same-filesystem directory"
        )
        parent = Path(_ALIAS_DIRECTORY).resolve(strict=True)
        socket_path = Path("/run") / ("loopgen_" + generation + ".socket")
        assert parent.stat().st_uid == os.getuid() and parent.stat().st_mode & 0o777 == 0o700
        assert parent.stat().st_dev == socket_path.stat().st_dev
        workspace = native_directory(parent)
    yield Interaction(), workspace, generation


def invoke(public, command, **grants):
    """Run the actual public client outside pytest's process-local temporary-root replacement."""
    interaction, workspace, _ = public
    client = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve())],
        input=json.dumps(
            {
                "workspace": str(workspace),
                "arguments": {
                    "command": command,
                    "cwd": str(workspace),
                    **grants,
                },
            }
        ),
        capture_output=True,
        text=True,
        env={
            key: value for key, value in os.environ.items() if key not in {"TMPDIR", "TMP", "TEMP"}
        },
        check=True,
        timeout=15,
    )
    assert not client.stderr, client.stderr
    value = json.loads(client.stdout)
    interaction.prompts.extend(value["prompts"])
    assert not any("without the OS sandbox" in prompt for prompt in interaction.prompts)
    return value["result"]


def public_client():
    """Run a registered public call with native selection and host fallback disabled."""
    value = json.loads(sys.stdin.read())
    backend = select_sandbox_backend()
    assert isinstance(backend, LinuxNativeBackend), backend.capability_failure()
    interaction = Interaction()
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(Path(value["workspace"])),
        execution_service=CommandExecutionService(backend, RefusingHost()),
        settings=ToolRuntimeSettings(command_timeout=2),
    )
    try:
        result = json.loads(
            registry.call(
                "run_command",
                json.dumps(value["arguments"]),
                interaction=interaction,
            )
        )
        print(json.dumps({"result": result, "prompts": interaction.prompts}))
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


def test_public_external_python_library_runs_from_a_hardlinked_read_root(public, tmp_path):
    """Keep cache-linked library files readable through their approved read-only root."""
    library = tmp_path.parent / "python-library"
    library.mkdir()
    cache = tmp_path.parent / "cache.py"
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


def test_public_writable_workspace_hardlink_cannot_modify_outside_inode(
    public, tmp_path, tmp_path_factory
):
    """Writable workspaces can read hardlinked files, but their aliases cannot mutate outside."""
    outside = tmp_path_factory.mktemp("outside-hardlink") / "outside-canary"
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


def test_public_ls_runs_with_large_hardlinked_virtualenv(public, tmp_path_factory):
    """Keep the public list command usable when Bubblewrap needs grouped hardlink mounts."""
    library = public[1] / ".venv/site-packages/linked-package"
    library.mkdir(parents=True)
    cache = tmp_path_factory.mktemp("virtualenv-hardlink-cache")
    for index in range(2_001):
        source = cache / f"module-{index}.py"
        source.write_text("value = 1\n")
        (library / source.name).hardlink_to(source)

    listed = invoke(public, "ls .")
    assert listed["result"]["exit_code"] == 0, listed
    written = invoke(public, "printf allowed > ordinary")
    assert written["result"]["exit_code"] == 0, written
    assert (public[1] / "ordinary").read_text() == "allowed"
    require_drained(public[2])


def test_public_read_only_command_can_create_files_in_private_scratch(public):
    """Keep nested private-name paths writable in scratch when the workspace is read-only."""
    source = (
        "import os,pathlib; root=pathlib.Path(os.environ['TMPDIR']); "
        "target=root/'pytest-fixture'/'workspace'/'project'/'.loop'; "
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


def test_public_uv_runs_a_workspace_virtualenv_with_its_user_local_runtime(public, monkeypatch):
    """Run uv and a workspace Python entry point through the installed public tool."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not available on the qualification PATH")

    project = Path(__file__).resolve().parents[3]
    project_venv = project / ".venv"
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
    alias = workspace.parent / f"{workspace.name}-python-runtime"
    alias.symlink_to(runtime_root, target_is_directory=True)
    venv_bin = workspace / ".venv/bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(alias / "bin" / interpreter_name)
    (workspace / ".venv/pyvenv.cfg").write_text(
        "\n".join(
            (
                f"home = {alias / 'bin'}",
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
    pytest_entry = venv_bin / "pytest"
    pytest_entry.write_text(
        f"#!{venv_bin / 'python'}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "venv = Path(__file__).parent.parent.resolve()\n"
        "print('WORKSPACE_VENV=' + str(Path(sys.prefix).resolve() == venv))\n",
        encoding="utf-8",
    )
    pytest_entry.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(venv_bin), os.environ["PATH"])))

    result = invoke(public, "uv run pytest")

    assert result.get("exit_code") == 0, result
    assert result["stdout"]["content"] == "WORKSPACE_VENV=True\n"
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
                policy_version="linux-native-v1",
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


def test_public_default_verified_tls_to_owned_fixture(public, native_directory):
    """Verify an owned HTTPS chain, hostname and offline boundary without changing host trust."""
    workspace = public[1]
    with tempfile.TemporaryDirectory(
        prefix="loop-owned-https-", dir=native_directory(Path("/tmp"))
    ) as directory:
        private = Path(directory)
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
            assert (
                "CERTIFICATE_VERIFY_FAILED"
                in wrong_name["problem"]["metadata"]["stderr"]["content"]
            )
            untrusted = invoke(public, "/usr/bin/python3 -I -c " + shlex.quote(code), network=True)
            assert not untrusted["ok"]
            assert (
                "CERTIFICATE_VERIFY_FAILED" in untrusted["problem"]["metadata"]["stderr"]["content"]
            )
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
    installation, tmp_path, native_directory, monkeypatch
):
    """Approved outside access and network requests provide a valid escalation control."""
    from loop.execution.sandbox import SandboxOutcome

    backend, generation = installation
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret").write_text("broader-control")
    monkeypatch.setattr(tempfile, "tempdir", "/tmp")
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


if __name__ == "__main__":
    public_client()
