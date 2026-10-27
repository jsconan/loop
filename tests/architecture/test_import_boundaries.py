"""Check repository-wide import and dependency boundaries."""

import ast
import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import loop


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
    """Retained contracts import from their owner without application dependencies."""
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
    '$WORKSPACE_ROOT', '$TMPDIR', '/skills'
)
assert 'pathspec' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", source],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


def test_sandbox_contract_has_no_host_execution_dependency():
    """Sandbox contract imports no host launcher or application retry authority."""
    source = files("loop.execution.sandbox").joinpath("contracts.py")
    tree = ast.parse(source.read_text())
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    names.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    assert not any("host" in name or "application" in name for name in names)
