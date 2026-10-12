"""Check protected host tool/library identities and pinned library loading on real Linux."""

import ctypes
import os

import pytest

from loop.execution.sandbox.linux.host import apparmor_library, bind_host_file, discover_tool

pytestmark = [pytest.mark.integration, pytest.mark.linux]


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
