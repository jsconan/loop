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
