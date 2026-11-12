"""Check protected host tool/library identities and pinned library loading on real Linux."""

import ctypes
import os
import subprocess
import time
from pathlib import Path

import pytest

from loop.execution.sandbox.linux.host import (
    apparmor_library,
    bind_host_file,
    compile_policy_source,
    discover_tool,
)
from loop.execution.sandbox.linux.mountinfo import read_descriptor_mount_ids, read_mountinfo
from loop.execution.sandbox.linux.policy import runtime_policy_sources

pytestmark = [pytest.mark.integration, pytest.mark.linux]


def test_compiler_preserves_native_binary_output():
    """The supervised compiler returns exactly the parser's binary policy bytes."""
    _, source = runtime_policy_sources(Path("/usr/local/libexec/loop/native"), "native_" + "a" * 32)
    parser = discover_tool("apparmor_parser")
    arguments = [
        str(parser.path),
        "--config-file=/dev/null",
        "--base=/etc/apparmor.d",
        "--jobs=0",
        "--stdout",
        "--skip-kernel-load",
        "--skip-cache",
        "--warn=all",
        "--Werror",
    ]
    options = {
        "input": source,
        "capture_output": True,
        "timeout": 5,
        "env": {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"},
    }
    expected = subprocess.run(arguments, check=True, **options).stdout
    if expected[:1] == b"\x01" and expected[2:12] == b"\0" * 6 + b"\x28\xb5\x2f\xfd":
        expected = subprocess.run(
            [*arguments, "--zstd-compress-level=0"], check=True, **options
        ).stdout
    assert compile_policy_source(source, time.monotonic() + 5) == expected


def test_live_mount_inventory_matches_the_open_root_descriptor():
    """The live kernel inventory identifies the mount holding the current filesystem root."""
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        identifiers = read_descriptor_mount_ids(descriptor)
        rows = read_mountinfo()
    finally:
        os.close(descriptor)

    assert len(identifiers) == 1 and len(identifiers[0]) == 1
    matched = [row for row in rows if row and [row[0]] == identifiers[0]]
    assert len(matched) == 1 and matched[0][4] == "/"


def test_protected_apparmor_library_loads_from_verified_inode():
    """The real installed native transition API is accessible through its prepared descriptor."""
    bound = apparmor_library()
    with bound.open_verified() as descriptor:
        library = ctypes.CDLL(f"/proc/self/fd/{descriptor}", use_errno=True)
        assert library.aa_change_profile is not None
    with pytest.raises(OSError) as caught:
        os.fstat(descriptor)
    assert caught.value.errno == 9


@pytest.mark.parametrize("name", ["bwrap", "apparmor_parser"])
def test_management_tools_have_stable_protected_identities(name):
    """Installed management tools retain their exact inode/byte identity across discovery."""
    first = discover_tool(name)
    assert first == discover_tool(name)
    with first.open_verified() as descriptor:
        assert os.fstat(descriptor).st_uid == 0


def test_unprotected_native_input_is_refused(tmp_path):
    """An ordinary caller's native library or tool cannot become trusted setup input."""
    path = tmp_path / "untrusted"
    path.write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="Unsafe native host authority"):
        bind_host_file(path)
