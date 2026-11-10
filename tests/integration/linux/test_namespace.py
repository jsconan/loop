"""Verify selected system data in the real production namespace without loading native policy."""

import errno
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from loop.execution.sandbox.contracts import SandboxRequest
from loop.execution.sandbox.linux.constants import NATIVE_SCRATCH_PATH
from loop.execution.sandbox.linux.host import bind_host_file, runtime_read_files
from loop.execution.sandbox.linux.namespace import namespace_arguments

pytestmark = [pytest.mark.integration, pytest.mark.linux]
_BWRAP = os.environ.get("LOOP_NATIVE_BWRAP")

_SYSTEM_DATA = """
import hashlib,json,socket,ssl
from pathlib import Path
print(json.dumps({
    'localhost': sorted(set(item[4][0] for item in socket.getaddrinfo('localhost',80))),
    'certificates': len(ssl.create_default_context().get_ca_certs()),
    'files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path('/etc/hosts'),Path('/etc/resolv.conf'))}
}))
"""


def test_namespace_preparation_ignores_unrelated_environment_metadata(tmp_path):
    """Real descriptor binding keeps private masks when unrelated metadata is a symlink."""
    workspace = (tmp_path / "workspace").resolve()
    scratch = (tmp_path / "scratch").resolve()
    workspace.mkdir()
    scratch.mkdir(mode=0o700)
    tools = workspace / "bin"
    tools.mkdir()
    executable = tools / "fixture-processor"
    executable.write_text("fixture executable", encoding="utf-8")
    executable.chmod(0o755)
    (workspace / "unrelated").write_text("ordinary metadata", encoding="utf-8")
    (workspace / "pyvenv.cfg").symlink_to(workspace / "unrelated")
    private = workspace / ".ssh"
    private.mkdir()
    (private / "key").write_text("private-canary", encoding="utf-8")
    details = executable.stat()
    request = SandboxRequest.create(
        source=executable.name,
        cwd=workspace,
        workspace=workspace,
        read_roots=(),
        write_roots=(),
        network=False,
        environment={"PATH": str(tools) + ":/usr/bin:/bin", "TMPDIR": str(scratch)},
        policy_version="namespace-fixture",
        deadline=time.monotonic() + 10,
        workspace_id="namespace-fixture",
        scratch_root=scratch,
        executable_identities=(
            (executable.name, executable, executable, details.st_dev, details.st_ino),
        ),
    )
    mask = tmp_path / "mask.empty"
    mask.write_bytes(b"")
    mask.chmod(0o444)

    with (
        mask.open("rb") as empty,
        namespace_arguments(request, scratch, empty_mask=empty.fileno()) as (
            arguments,
            descriptors,
        ),
    ):
        assert any(
            arguments[index : index + 2] == ("--tmpfs", str(private))
            for index in range(len(arguments) - 1)
        )
        assert descriptors
    assert (private / "key").read_text(encoding="utf-8") == "private-canary"


def test_dense_private_masks_respect_descriptor_limit(tmp_path):
    """Descriptor exhaustion fails closed and releases all sources."""
    code = """import errno, os, resource, sys, time
from pathlib import Path
from loop.execution.sandbox.contracts import SandboxRequest
from loop.execution.sandbox.linux.namespace import namespace_arguments
root=Path(sys.argv[1])
workspace=root/'workspace'; workspace.mkdir()
scratch=root/'scratch'; scratch.mkdir(mode=0o700)
for index in range(40):
    parent=workspace/str(index); parent.mkdir(); (parent/'.aws').write_text('private')
mask=root/'mask'; mask.write_bytes(b''); mask.chmod(0o444)
request=SandboxRequest.create(source='true', cwd=workspace, workspace=workspace,
    read_roots=(), write_roots=(workspace,), network=False,
    environment={'PATH':'/usr/bin:/bin', 'TMPDIR':str(scratch)},
    policy_version='namespace-fixture', deadline=time.monotonic()+30,
    workspace_id='dense-fixture', scratch_root=scratch)
resource.setrlimit(resource.RLIMIT_NOFILE, (120,120))
with mask.open('rb') as empty:
    before=len(os.listdir('/proc/self/fd'))
    try:
        with namespace_arguments(request,scratch,empty_mask=empty.fileno()):
            raise RuntimeError('descriptor limit was not enforced')
    except OSError as error:
        if error.errno != errno.EMFILE: raise
    if len(os.listdir('/proc/self/fd')) != before: raise RuntimeError('descriptor leak')
print('descriptor-boundary-preserved')
"""
    project = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=40,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(project / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "descriptor-boundary-preserved\n"


@pytest.fixture
def helper():
    """Require an explicitly selected root-protected native user-namespace qualification helper."""
    if not _BWRAP:
        pytest.skip("Set LOOP_NATIVE_BWRAP to the protected native qualification helper")
    return bind_host_file(Path(_BWRAP))


@pytest.mark.parametrize("network", [False, True])
def test_system_data_matches_host_with_read_only_mounts(helper, tmp_path, network):
    """Real namespace resolution and default CA bundles match host data without writable mounts."""
    host = json.loads(
        subprocess.check_output(
            ["/usr/bin/python3", "-I", "-c", _SYSTEM_DATA], text=True, timeout=5
        )
    )
    before = {str(path): authority.digest for path, authority in runtime_read_files()}
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    scratch = (Path(tempfile.gettempdir()) / "command-scratch").resolve()
    scratch.mkdir(mode=0o700)
    mask = tmp_path / "mask.empty"
    mask.write_bytes(b"")
    mask.chmod(0o444)
    request = SandboxRequest.create(
        source="true",
        cwd=workspace,
        workspace=workspace,
        read_roots=(),
        write_roots=(workspace,),
        network=network,
        environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)},
        policy_version="system-data-fixture",
        deadline=time.monotonic() + 10,
        workspace_id="fixture",
    )
    probe = (
        _SYSTEM_DATA
        + """
import errno
rows = [line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines()]
assert any(row[4]=='/etc/resolv.conf' and 'ro' in row[5].split(',') for row in rows)
try:
    with open('/etc/resolv.conf','ab'):
        pass
except OSError as error:
    print(error.errno)
else:
    raise RuntimeError('Resolver write was admitted')
"""
    )
    with (
        helper.open_verified(),
        mask.open("rb") as empty,
        namespace_arguments(request, scratch, empty_mask=empty.fileno()) as (
            arguments,
            descriptors,
        ),
    ):
        result = subprocess.run(
            [str(helper.path), *arguments, "/usr/bin/python3", "-I", "-c", probe],
            pass_fds=descriptors,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    lines = result.stdout.splitlines()
    assert len(lines) == 2, result.stdout
    assert not result.stderr, result.stderr
    assert json.loads(lines[0]) == host
    assert host["certificates"] > 0
    assert int(lines[1]) in (errno.EACCES, errno.EROFS)
    assert {str(path): authority.digest for path, authority in runtime_read_files()} == before


def test_namespace_scratch_symlink_keeps_read_only_mounts(helper, tmp_path):
    """Real scratch symlinks preserve restrictive overlays and private masks."""
    scratch = Path(tempfile.gettempdir()) / "private-scratch"
    scratch.mkdir(mode=0o700)
    selected = scratch / "selected"
    selected.mkdir()
    (selected / "input").write_text("selected-data")
    outside = tmp_path / "outside"
    outside.write_text("preserve-outside")
    os.link(outside, scratch / "linked")
    (scratch / ".ssh").mkdir()
    (scratch / ".ssh/key").write_text("private-canary")
    (scratch / ".aws").write_text("private-file-canary")
    (scratch / ".config").write_text("second-private-canary")
    (scratch / "selected-link").symlink_to("selected", target_is_directory=True)
    mask = tmp_path / "mask.empty"
    mask.write_bytes(b"")
    with mask.open("rb") as empty:
        request = SandboxRequest.create(
            source="native namespace control",
            cwd=selected,
            workspace=selected,
            read_roots=(),
            write_roots=(),
            network=False,
            environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)},
            policy_version="namespace-fixture",
            deadline=time.monotonic() + 10,
            workspace_id="fixture",
        )
        code = """import errno,os,sys
from pathlib import Path
scratch=Path(sys.argv[1]);alias=Path(sys.argv[2])
assert alias.resolve()==scratch/'selected'
assert (alias/'input').read_text()=='selected-data'
for path in (scratch/'selected/input',alias/'input',scratch/'linked'):
 for action in (lambda: path.write_text('bad'),lambda: path.chmod(0o600)):
  try: action()
  except OSError as error: assert error.errno in (errno.EACCES,errno.EPERM,errno.EROFS)
  else: raise RuntimeError('Read-only mount was writable')
assert not (scratch/'.ssh/key').exists()
assert (scratch/'.aws').read_bytes()==b''
assert (scratch/'.config').read_bytes()==b''
(scratch/'ordinary').write_text('allowed')
assert (scratch/'ordinary').read_text()=='allowed'
print('namespace-boundaries-preserved')
"""
        with (
            helper.open_verified(),
            namespace_arguments(request, scratch, empty_mask=empty.fileno()) as (
                arguments,
                descriptors,
            ),
        ):
            result = subprocess.run(
                [
                    str(helper.path),
                    *arguments,
                    "/usr/bin/python3",
                    "-I",
                    "-c",
                    code,
                    str(NATIVE_SCRATCH_PATH),
                    str(NATIVE_SCRATCH_PATH / "selected-link"),
                ],
                pass_fds=descriptors,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            )
    assert result.returncode == 0 and not result.stderr, result
    assert result.stdout == "namespace-boundaries-preserved\n"
    assert outside.read_text() == "preserve-outside"
    assert (selected / "input").read_text() == "selected-data"
    assert (scratch / ".ssh/key").read_text() == "private-canary"
    assert (scratch / ".aws").read_text() == "private-file-canary"
    assert (scratch / ".config").read_text() == "second-private-canary"


def test_bound_workspace_source_survives_path_replacement(helper, tmp_path):
    """A replaced host path cannot make a pinned workspace mount expose the new directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "canary").write_text("bound-original")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "canary").write_text("unapproved-replacement")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    mask = tmp_path / "mask.empty"
    mask.write_bytes(b"")
    mask.chmod(0o444)
    request = SandboxRequest.create(
        source="true",
        cwd=workspace,
        workspace=workspace,
        read_roots=(),
        write_roots=(),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)},
        policy_version="namespace-fixture",
        deadline=time.monotonic() + 10,
        workspace_id="source-replacement",
        scratch_root=scratch,
    )
    with (
        helper.open_verified(),
        mask.open("rb") as empty,
        namespace_arguments(request, scratch, empty_mask=empty.fileno()) as (
            arguments,
            descriptors,
        ),
    ):
        workspace.rename(tmp_path / "bound-original")
        replacement.rename(workspace)
        result = subprocess.run(
            [
                str(helper.path),
                *arguments,
                "/usr/bin/python3",
                "-I",
                "-c",
                "from pathlib import Path; print(Path('canary').read_text())",
            ],
            pass_fds=descriptors,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    assert result.returncode == 0 and not result.stderr, result
    assert result.stdout == "bound-original\n"
    assert (workspace / "canary").read_text() == "unapproved-replacement"
