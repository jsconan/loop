"""Verify selected system data in the real production namespace without loading native policy."""

import errno
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from loop.execution.sandbox.contracts import SandboxRequest
from loop.execution.sandbox.linux.constants import NATIVE_ALIAS_PATH, NATIVE_SCRATCH_PATH
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


def test_namespace_virtual_alias_and_selected_scratch_keep_read_only_mounts(helper, tmp_path):
    """Real namespace aliases preserve pinned targets and restrictive scratch overlays."""
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
    mappings = tmp_path / "mappings"
    mappings.mkdir(mode=0o700)
    alias = mappings / "r0"
    alias.symlink_to(selected)
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
            aliases=(("$WORKSPACE_ROOT", alias, selected),),
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
                    str(NATIVE_ALIAS_PATH / "r0"),
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
