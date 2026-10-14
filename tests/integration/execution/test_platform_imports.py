"""Verify app imports keep native sandbox implementations platform isolated."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import loop

pytestmark = pytest.mark.integration


def test_app_import_avoids_other_platform_sandbox_modules(tmp_path):
    """Importing the application loads no native implementation for another host."""
    source = """
import sys
import loop
import loop.execution.sandbox
from loop.utils import VirtualPath
from loop.utils.virtual_path import VirtualPath as OwnedVirtualPath
assert VirtualPath is OwnedVirtualPath
foreign = 'linux' if sys.platform == 'darwin' else 'macos'
assert not any(
    name == f'loop.execution.sandbox.{foreign}'
    or name.startswith(f'loop.execution.sandbox.{foreign}_')
    for name in sys.modules
)
"""
    subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(loop.__file__).resolve().parent.parent)},
        check=True,
        capture_output=True,
    )


def test_retained_contract_import_avoids_filesystem_ignore_dependency(tmp_path):
    """Retained contracts use VirtualPath in its owning module without application dependencies."""
    from importlib.resources import files

    modules = (
        "constants.py",
        "utils/virtual_path.py",
        "utils/process.py",
        "utils/hashing.py",
        "utils/json.py",
        "permissions/protection.py",
        "execution/sandbox/contracts.py",
        "execution/sandbox/constants.py",
    )
    for prefix in ("", "utils/", "permissions/", "execution/", "execution/sandbox/"):
        barrel = tmp_path / "loop" / prefix / "__init__.py"
        barrel.parent.mkdir(parents=True, exist_ok=True)
        barrel.write_text("")
    for name in modules:
        (tmp_path / "loop" / name).write_bytes(files("loop").joinpath(name).read_bytes())
    source = f"""
import sys
sys.path.insert(0, {str(tmp_path)!r})
from loop.execution.sandbox.contracts import SandboxRequest
from loop.utils.virtual_path import VirtualPath
assert VirtualPath.__module__ == 'loop.utils.virtual_path'
assert (VirtualPath.WORKSPACE, VirtualPath.TEMPORARY, VirtualPath.SKILLS) == (
    '/workspace', '/tmp', '/skills'
)
assert 'pathspec' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", source],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
