"""Qualify privileged Linux sandbox management on an explicitly disposable native host."""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

ROOT = Path("/usr/local/libexec/loop/native")
_MANAGEMENT_OPT_IN = os.environ.get("LOOP_NATIVE_MANAGEMENT_TEST")
_MANAGEMENT_WHEEL = os.environ.get("LOOP_NATIVE_MANAGEMENT_WHEEL")
pytestmark = [pytest.mark.integration, pytest.mark.e2e, pytest.mark.linux]


class RefusingHost:
    """Reject an unrestricted fallback during management readiness checks."""

    def run_host_command(self, request):
        """Fail if a public command attempts to run outside the native sandbox."""
        raise AssertionError("Management qualification attempted host execution")


class ApprovingInteraction:
    """Approve the one bounded sandbox command used as a post-management readiness probe."""

    def prompt(self, message=None, **kwargs):
        """Approve the native sandbox command without offering host recovery."""
        require("without the OS sandbox" not in str(message), "host recovery was offered")
        return "approve"

    def info(self, message):
        """Accept public tool information without modifying the fixture."""


_BASELINE_PATHS = (
    Path("/usr/bin/bwrap"),
    Path("/usr/sbin/apparmor_parser"),
    Path("/usr/bin/systemctl"),
    Path("/usr/bin/true"),
)


def require(condition, message):
    """Stop without mutating protected fixture state when a check differs."""
    if not condition:
        raise ValueError(message)


def baseline():
    """Snapshot protected host helper and system utility bytes before management changes."""
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in _BASELINE_PATHS}


def require_baseline(expected):
    """Reject changes to any captured host helper or system utility."""
    require(baseline() == expected, "host tool baseline changed")


def public_call(workspace, source, *, read_only=True, seconds=5):
    """Dispatch a real public command with host execution forbidden."""
    from loop import PermissionManager, ToolRegistry
    from loop.execution import CommandExecutionService
    from loop.execution.sandbox import select_sandbox_backend
    from loop.tooling import ToolRuntimeSettings
    from loop.tools.system import run_command

    backend = select_sandbox_backend()
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(backend, RefusingHost()),
        settings=ToolRuntimeSettings(command_timeout=seconds),
    )
    try:
        result = json.loads(
            registry.call(
                "run_command",
                json.dumps(
                    {
                        "command": source,
                        "cwd": str(workspace),
                        "read_only": read_only,
                    }
                ),
                interaction=ApprovingInteraction(),
            )
        )
    finally:
        registry.close()
    return result


def public_probe(workspace):
    """Require ordinary installed public execution after an administrator operation."""
    result = public_call(workspace, "printf management-ready")
    output = result.get("result", result)
    require(output.get("exit_code") == 0, "public sandbox command failed after management")
    require(
        output["stdout"]["content"] == "management-ready",
        "public command output differs",
    )


def absent(generations):
    """Verify owned policy, units, listener and recursive tracking have all disappeared."""
    namespaces = Path("/sys/kernel/security/apparmor/policy/namespaces")
    for generation in generations:
        prefix = "loopgen_" + generation
        for path in [
            namespaces / generation,
            Path("/run") / (prefix + ".socket"),
            Path("/sys/fs/cgroup") / (prefix + ".slice"),
            *(
                Path("/run/systemd/system") / (prefix + suffix)
                for suffix in (".slice", ".socket", "@.service")
            ),
        ]:
            require(
                not path.exists() and not path.is_symlink(), "owned authority remains: " + str(path)
            )
        for profile in (namespaces.parent / "profiles").iterdir():
            require((profile / "name").read_text().strip() != prefix, "owned global policy remains")
        units = subprocess.check_output(
            [
                "/usr/bin/systemctl",
                "list-units",
                "--all",
                "--plain",
                "--no-legend",
                prefix + "@*.service",
            ],
            text=True,
            timeout=5,
        )
        for line in units.splitlines():
            fields = dict(
                item.split("=", 1)
                for item in subprocess.check_output(
                    [
                        "/usr/bin/systemctl",
                        "show",
                        line.split()[0],
                        "--property=ActiveState,MainPID,ControlPID,ControlGroup,FragmentPath",
                    ],
                    text=True,
                    timeout=5,
                ).splitlines()
            )
            require(
                fields.get("ActiveState") == "inactive"
                and fields.get("MainPID") == fields.get("ControlPID") == "0"
                and not fields.get("ControlGroup")
                and not fields.get("FragmentPath"),
                "retired instance authority remains",
            )


def live_client():
    """Keep an authorized detached descendant active while the parent tests management refusal."""
    value = json.loads(sys.stdin.read())
    print(
        json.dumps(
            public_call(Path(value["workspace"]), value["source"], read_only=False, seconds=90)
        )
    )


def live_removal_refusal(app, interaction, workspace, generation):
    """Challenge removal and new admission while an actual detached child remains confined."""
    private = workspace / ".ssh"
    private.mkdir()
    (private / "canary").write_text("preserve-private")
    outside = workspace.parent / "outside-canary"
    outside.write_text("preserve-outside")
    ready, release = workspace / "live-ready", workspace / "live-release"
    completed = workspace / "live-complete"
    code = """import os,time
from pathlib import Path
pid=os.fork()
if pid:
 _,status=os.waitpid(pid,0)
 if status: raise RuntimeError('Detached child violated confinement')
 Path('live-complete').write_text('completed')
 print('live-complete')
else:
 os.setsid()
 Path('live-ready').write_text('ready')
 while not Path('live-release').exists():
  for path,writing in [(Path('.ssh/canary'),False),(Path(__OUTSIDE__),True)]:
   try:
    if writing: path.write_text('bad')
    else: path.read_text()
   except OSError: pass
   else: os._exit(31)
  time.sleep(0.02)
 os._exit(0)
""".replace("__OUTSIDE__", repr(str(outside)))
    client = subprocess.Popen(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--live-client"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        client.stdin.write(
            json.dumps(
                {
                    "workspace": str(workspace),
                    "source": "/usr/bin/python3 -I -c " + shlex.quote(code),
                }
            )
        )
        client.stdin.close()
        client.stdin = None
        deadline = time.monotonic() + 10
        while not ready.exists() and client.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        require(ready.exists() and client.poll() is None, "detached native child did not enter")
        record = ROOT / "management.json"
        original = record.read_bytes()
        interaction.info(
            "Qualification: approve this uninstall; it must refuse the live detached child without killing it."
        )
        app.command_manager.handle_user_command("/sandbox uninstall")
        require(record.read_bytes() == original, "live removal changed original authority")
        namespace = Path("/sys/kernel/security/apparmor/policy/namespaces") / generation
        require(
            namespace.is_dir()
            and all(
                (entry / "mode").read_text().strip() == "enforce"
                for entry in (namespace / "profiles").iterdir()
            ),
            "live policy changed or stopped enforcing",
        )
        require(client.poll() is None, "management killed the admitted child")
        refused = public_call(workspace, "printf admitted > new-admission", read_only=False)
        require(
            not refused.get("ok") and not (workspace / "new-admission").exists(),
            "closed listener admitted new source",
        )
        require(
            (private / "canary").read_text() == "preserve-private"
            and outside.read_text() == "preserve-outside",
            "canary changed while removal was refused",
        )
        release.write_text("release")
        stdout, stderr = client.communicate(timeout=10)
        result = json.loads(stdout)
        require(
            client.returncode == 0 and not stderr and completed.read_text() == "completed",
            "live client or confined source did not finish after release: "
            + repr((client.returncode, stdout, stderr)),
        )
        # Closing admission removes the original socket. The client must reject the result
        # against that changed transport authority even though the confined source finished.
        require(
            not result.get("ok")
            and result["problem"]["code"] == "sandbox.invalid"
            and result["problem"]["metadata"]["boundary"] == "sandbox"
            and result["problem"]["metadata"]["stdout"]["capture_complete"] is False,
            "closed-listener delivery did not fail closed: " + json.dumps(result),
        )
        for path in (Path("/sys/fs/cgroup") / ("loopgen_" + generation + ".slice")).rglob(
            "cgroup.procs"
        ):
            require(not path.read_text().strip(), "descendants survived live removal refusal")
    finally:
        release.write_text("release")
        if client.poll() is None:
            client.terminate()
            client.communicate(timeout=10)
        ready.unlink(missing_ok=True)
        release.unlink(missing_ok=True)
        completed.unlink(missing_ok=True)
        (private / "canary").unlink()
        private.rmdir()
        outside.unlink()
    print("LIVE_DETACHED_CHILD_REMOVAL_AND_CONCURRENT_ADMISSION_REFUSAL=PASS", flush=True)


def partial_removal_fixture(interaction, generation):
    """Simulate lost ephemeral authority only after authenticated idle drain, preserving file bindings."""
    from loop.execution.sandbox.linux.host import bind_host_file
    from loop.execution.sandbox.linux.management import uninstall_authority

    authority, _ = uninstall_authority(ROOT / "management.json")
    source = r"""import hashlib,os,stat,subprocess,sys,time
from pathlib import Path
root=Path('/usr/local/libexec/loop/native')
if os.getuid()!=0 or Path('/proc/self/uid_map').read_text().split()!=['0','0','4294967295']: raise PermissionError('initial root required')
for path in (root,*root.parents):
 d=path.lstat()
 if not stat.S_ISDIR(d.st_mode) or d.st_uid!=0 or d.st_mode&0o6022: raise ValueError('protected ancestry differs')
record=root/'management.json'
if hashlib.sha256(record.read_bytes()).hexdigest()!=sys.argv[1]: raise ValueError('reviewed authority changed')
sys.path.insert(0,str(root))
from loop.execution.sandbox.linux.records import read_generation
from loop.execution.sandbox.linux.installation import drained_generation,generation_policy_cache,remove_generation_policy
installed,authority=read_generation(record)
if installed.config.generation!=sys.argv[2]: raise ValueError('reviewed generation changed')
policies=generation_policy_cache(installed.config,installed.namespace,installed.policies,installed.helper,installed.compiled)
remove_generation_policy(installed.config,installed.files,installed.lock,installed.namespace,policies,time.monotonic()+5)
with drained_generation(installed.config,installed.files,installed.lock,time.monotonic()+5):
 with authority.open_verified(),installed.helper.open_verified():
  command=('loopgen_'+installed.config.generation).encode()+b'\0'
  with Path('/sys/kernel/security/apparmor/.remove').open('wb',buffering=0) as output:
   if output.write(command)!=len(command): raise OSError('incomplete owned attachment removal')
  subprocess.run(['/usr/bin/systemctl','--system','--no-pager','stop','--',installed.config.slice_unit.path.name],check=True,timeout=5)
  for bound in (installed.config.slice_unit,installed.files['socket_unit'],installed.files['service_unit']):
   with bound.open_verified(): bound.path.unlink()
  subprocess.run(['/usr/bin/systemctl','--system','--no-pager','daemon-reload'],check=True,timeout=5)
 with installed.helper.open_verified(),authority.open_verified(): pass
print('AUTHENTICATED_IDLE_PARTIAL_REMOVAL_WITH_ORIGINAL_HELPER=PASS')
"""
    sudo = bind_host_file(Path("/usr/bin/sudo"), system_elevation=True)
    python = bind_host_file(Path("/usr/bin/python3"))
    interaction.info(
        "Qualification-only disappearance of this idle installation's owned policy and tracking. Admission is closed and drain is proven first; the original helper and record remain for ordinary checked retry. No unrelated state is changed."
    )
    require(
        interaction.confirm("Run this exact administrator partial-state fixture?", default=False)
        is True,
        "partial-state fixture declined",
    )
    with authority.open_verified(), sudo.open_verified(), python.open_verified():
        subprocess.run(
            [
                str(sudo.path),
                "--",
                str(python.path),
                "-I",
                "-S",
                "-B",
                "-c",
                source,
                authority.digest,
                generation,
            ],
            check=True,
            timeout=30,
        )
    require(
        (ROOT / "management.json").exists(), "partial fixture removed original management authority"
    )
    require(
        (ROOT / "helper").stat().st_mode & 0o777 == 0o555,
        "partial fixture changed original helper mode",
    )


def main():
    """Exercise real registered app lifecycle followed by exported application-absent uninstall."""
    require(
        os.getuid() > 0 and os.getgid() > 0 and sys.stdin.isatty() and sys.stdout.isatty(),
        "ordinary private terminal required",
    )
    incoming = Path(os.environ["LOOP_NATIVE_MANAGEMENT_OUTPUT"]).resolve(strict=True)
    wheel = Path(os.environ["LOOP_NATIVE_MANAGEMENT_WHEEL"]).resolve(strict=True)
    # Import the full pip-installed distribution, not an isolated retained catalog or mocked app.
    import loop
    from loop import Loop
    from loop.backend import OpenAIBackend
    from loop.execution.sandbox.linux.host import bind_host_file
    from loop.execution.sandbox.linux.management import (
        generation_status,
        installation_status,
        uninstall_authority,
    )
    from loop.execution.sandbox.linux.setup import manage_setup
    from loop.interaction import ConsoleInteraction

    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("loop/") and not name.endswith("/"):
                require(
                    (Path(loop.__file__).parent.parent / name).read_bytes() == archive.read(name),
                    "installed wheel bytes differ",
                )
    initial_scaffold = None
    if ROOT.exists() or ROOT.is_symlink():
        require(
            not ROOT.is_symlink()
            and {path.name for path in ROOT.iterdir()}
            == {"management.lock", "compiler.lock", "reservations"},
            "remove the existing installation through the main app before qualification",
        )
        reservations = ROOT / "reservations"
        require(
            not reservations.is_symlink()
            and reservations.is_dir()
            and reservations.stat().st_uid == 0
            and not reservations.stat().st_mode & 0o022
            and not list(reservations.iterdir()),
            "original reservations are not empty inert scaffolding",
        )
        initial_scaffold = (
            (ROOT.stat().st_dev, ROOT.stat().st_ino),
            (reservations.stat().st_dev, reservations.stat().st_ino),
            tuple(bind_host_file(ROOT / name) for name in ("management.lock", "compiler.lock")),
        )
    original_baseline = baseline()
    try:
        manage_setup(
            "install",
            lambda text: (_ for _ in ()).throw(
                ValueError("library unexpectedly requested consent")
            ),
        )
    except ValueError as error:
        require(
            str(error) == "Administrator management requires the main app in a real terminal.",
            "library refusal differs",
        )
    else:
        raise ValueError("library attempted elevation")
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "from loop.execution.sandbox.linux.setup import manage_setup\ntry: manage_setup('install', lambda text: True, main_app=True)\nexcept ValueError as error:\n if str(error) != 'Administrator management requires the main app in a real terminal.': raise\nelse: raise ValueError('headless elevation attempted')\nprint('PRODUCTION_INSTALLED_APP_HEADLESS_REFUSAL=PASS')",
        ],
        stdin=subprocess.DEVNULL,
        check=True,
        timeout=10,
    )
    with zipfile.ZipFile(wheel) as archive:
        loader_source = archive.read(
            "loop/execution/sandbox/linux/resources/setup_loader.py"
        ).decode()
    interaction = ConsoleInteraction(markdown=False)
    (incoming / "workspace").mkdir()
    app = Loop.create_default(
        OpenAIBackend(
            base_url="http://127.0.0.1:1", api_key="unused-native-check", default_model="unused"
        ),
        interaction=interaction,
        working_directory=incoming / "workspace",
        main_app=True,
    )
    require(
        sum(command.name == "sandbox" for command in app.command_manager.commands) == 1,
        "sandbox registration differs",
    )
    app.command_manager.handle_user_command("/sandbox status")
    interaction.info("Qualification: decline this first install (answer no).")
    app.command_manager.handle_user_command("/sandbox install")
    if initial_scaffold is None:
        require(not ROOT.exists(), "declined setup changed protected state")
    else:
        require(
            {path.name for path in ROOT.iterdir()}
            == {"management.lock", "compiler.lock", "reservations"}
            and (ROOT.stat().st_dev, ROOT.stat().st_ino) == initial_scaffold[0]
            and (reservations.stat().st_dev, reservations.stat().st_ino) == initial_scaffold[1]
            and not list(reservations.iterdir())
            and tuple(bind_host_file(bound.path) for bound in initial_scaffold[2])
            == initial_scaffold[2],
            "declined setup changed original inert scaffolding",
        )
    print(
        "PRODUCTION_INSTALLED_APP_SINGLE_REGISTRATION_LIBRARY_REFUSAL_AND_DECLINED_SETUP=PASS",
        flush=True,
    )
    interaction.info(
        "Qualification: review and approve this exact fresh install if it matches the reviewed scope."
    )
    app.command_manager.handle_user_command("/sandbox install")
    first = json.loads((ROOT / "management.json").read_bytes())["config"]["generation"]
    require(
        "authenticated" in generation_status(ROOT / "management.json")
        and installation_status().startswith("Installed."),
        "installed authority or concise status differs",
    )
    locks = [bind_host_file(ROOT / name) for name in ("management.lock", "compiler.lock")]
    directory_id = (ROOT.stat().st_dev, ROOT.stat().st_ino)
    reservations_id = ((ROOT / "reservations").stat().st_dev, (ROOT / "reservations").stat().st_ino)
    if initial_scaffold is not None:
        require(
            (directory_id, reservations_id, tuple(locks)) == initial_scaffold,
            "fresh install replaced original shared scaffold authority",
        )
    generations = [first]
    public_probe(incoming / "workspace")
    live_removal_refusal(app, interaction, incoming / "workspace", first)
    print("PRODUCTION_INSTALLED_APP_PUBLIC_NATIVE_COMMAND=PASS", flush=True)
    app.command_manager.handle_user_command("/sandbox refresh")
    second = json.loads((ROOT / "management.json").read_bytes())["config"]["generation"]
    require(second != first, "refresh did not replace original generation")
    generations.append(second)
    absent([first])
    public_probe(incoming / "workspace")
    print("PRODUCTION_INSTALLED_APP_CHECKED_SAME_LOCATION_REFRESH=PASS", flush=True)
    partial_removal_fixture(interaction, second)
    app.command_manager.handle_user_command("/sandbox uninstall")
    absent(generations)
    require(
        not (ROOT / "management.json").exists() and not list((ROOT / "reservations").iterdir()),
        "app uninstall retained capacity",
    )
    app.command_manager.handle_user_command("/sandbox install")
    third = json.loads((ROOT / "management.json").read_bytes())["config"]["generation"]
    require(third not in generations, "reinstallation identity reused")
    generations.append(third)
    public_probe(incoming / "workspace")
    authority, entry = uninstall_authority(ROOT / "management.json")
    # Remove only the ordinary pip-installed application. Its protected retained owner is untouched.
    subprocess.run(
        [sys.executable, "-I", "-m", "pip", "--isolated", "uninstall", "--yes", "loop"],
        check=True,
        timeout=60,
    )
    require(not Path(loop.__file__).exists(), "application package remains installed")
    argv = [
        "/usr/bin/sudo",
        "--",
        "/usr/bin/python3",
        "-I",
        "-S",
        "-B",
        str(entry.path),
        authority.digest,
    ]
    interaction.info(
        "Exported app-absent idle uninstall; record SHA256 "
        + authority.digest
        + "\nCommand: "
        + shlex.join(argv)
    )
    require(
        interaction.confirm("Run this exact exported administrator uninstall?", default=False)
        is True,
        "exported uninstall declined; authority retained",
    )
    with authority.open_verified(), entry.open_verified():
        subprocess.run(argv, check=True, timeout=120)
    absent(generations)
    require(not list((ROOT / "reservations").iterdir()), "exported uninstall retained capacity")
    print(
        "PRODUCTION_INSTALLED_APP_UNINSTALL_REINSTALL_AND_PACKAGE_ABSENT_EXPORTED_UNINSTALL=PASS",
        flush=True,
    )
    # Qualification-only cleanup of the exact original inert scaffolding, not a recovery API.
    cleanup = """import hashlib,os,stat,sys
from pathlib import Path
root=Path('/usr/local/libexec/loop/native')
if os.getuid()!=0 or Path('/proc/self/uid_map').read_text().split()!=['0','0','4294967295']: raise PermissionError('initial root required')
expected=__EXPECTED__
if (root.stat().st_dev,root.stat().st_ino)!=__ROOT__ or set(p.name for p in root.iterdir())!={'management.lock','compiler.lock','reservations'}: raise ValueError('exact inert root changed')
r=root/'reservations'
if (r.stat().st_dev,r.stat().st_ino)!=__RESERVATIONS__ or list(r.iterdir()): raise ValueError('reservation authority remains or changed')
for path in (root,r,*root.parents):
 s=path.lstat()
 if not stat.S_ISDIR(s.st_mode) or s.st_uid!=0 or s.st_mode&0o6022: raise ValueError('inert ancestry changed')
for name,(identity,digest) in expected.items():
 p=root/name;s=p.lstat();actual=(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if actual!=identity or s.st_uid!=0 or s.st_mode&0o6222 or hashlib.sha256(p.read_bytes()).hexdigest()!=digest: raise ValueError('original shared lock changed')
for name in expected: (root/name).unlink()
r.rmdir();root.rmdir()
print('AUTHENTICATED_EXACT_COMPLETED_SETUP_SCAFFOLDING_CLEANUP=PASS')
"""
    cleanup = (
        cleanup.replace(
            "__EXPECTED__",
            repr({bound.path.name: (bound.identity, bound.digest) for bound in locks}),
        )
        .replace("__ROOT__", repr(directory_id))
        .replace("__RESERVATIONS__", repr(reservations_id))
    )
    bad_input = incoming / "altered-reviewed-input.json"
    bad_input.write_bytes(b"altered reviewed bytes")
    guard = (
        "import sys\nscope={'__name__':'qualification_checked_loader'}\nexec(compile("
        + repr(loader_source)
        + ",'exact-reviewed-loader','exec'),scope)\nprevious=sys.argv\ntry:\n sys.argv="
        + repr(
            [
                "-c",
                str(bad_input),
                "a" * 64,
                "install",
                "native_negative_guard",
                str(os.getuid()),
                str(os.getgid()),
                "",
            ]
        )
        + "\n try: scope['_execute_reviewed_setup']()\n except ValueError as error:\n  if str(error)!='Reviewed setup input changed or exceeds its bound.': raise\n else: raise ValueError('changed review bytes reached imports')\nfinally: sys.argv=previous\nprint('PRODUCTION_PRIVILEGED_REVIEWED_BYTES_DRIFT_REFUSAL_BEFORE_IMPORT=PASS')\n"
    )
    cleanup = guard + cleanup
    argv = ["/usr/bin/sudo", "--", "/usr/bin/python3", "-I", "-S", "-B", "-c", cleanup]
    interaction.info(
        "Qualification-only altered-byte loader refusal followed by removal of the original inert fixed root, two immutable locks and empty reservations. No partial state adoption or changes to unrelated state."
    )
    require(
        interaction.confirm("Run this exact administrator fixture cleanup?", default=False) is True,
        "inert fixture retained for explicit cleanup",
    )
    subprocess.run(argv, check=True, timeout=120)
    require_baseline(original_baseline)
    require(not ROOT.exists(), "inert native root remains")
    (incoming / "generations.json").write_text(json.dumps(generations))
    print("PRODUCTION_INSTALLED_MAIN_APP_SETUP_REFRESH_UNINSTALL_NATIVE_BATCH=PASS", flush=True)


@pytest.mark.skipif(
    _MANAGEMENT_OPT_IN != "disposable-linux-host",
    reason="Set LOOP_NATIVE_MANAGEMENT_TEST=disposable-linux-host before pytest starts",
)
def test_installed_app_management_requires_real_consent_and_authentication(tmp_path, request):
    """Exercise decline, installation, refresh, removal, and exported uninstall on a disposable VM."""
    if len(request.session.items) != 1:
        pytest.fail("Run the destructive management lifecycle test by itself")
    if not _MANAGEMENT_WHEEL:
        pytest.fail("LOOP_NATIVE_MANAGEMENT_WHEEL must identify the reviewed application wheel")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        pytest.fail("Run this opt-in lifecycle test with pytest -s from a real terminal")
    output = tmp_path / "management-output"
    output.mkdir()
    environment = {
        **{key: value for key, value in os.environ.items() if key not in {"TMPDIR", "TMP", "TEMP"}},
        "LOOP_NATIVE_MANAGEMENT_OUTPUT": str(output),
        "LOOP_NATIVE_MANAGEMENT_WHEEL": _MANAGEMENT_WHEEL,
    }
    subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--management-lifecycle"],
        env=environment,
        check=True,
        timeout=900,
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--live-client"]:
        live_client()
    elif sys.argv[1:] == ["--management-lifecycle"]:
        main()
    else:
        raise SystemExit("Use the opt-in pytest integration test to run this lifecycle.")
