"""Check isolated retained and privileged package imports against the real native catalog."""

import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest

from loop.execution.sandbox.linux.catalog import runtime_catalog

pytestmark = [
    pytest.mark.integration,
    pytest.mark.linux,
    pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux host tools"),
]


def test_catalog_imports_without_application_or_site_packages(tmp_path):
    """Use reviewed distribution bytes for execution and setup imports."""
    contents, _, _, _ = runtime_catalog(Path("/usr/local/libexec/loop/native"), "native_test")
    package = files("loop")
    for name in ("catalog.py", "setup.py"):
        contents["loop/execution/sandbox/linux/" + name] = package.joinpath(
            "execution/sandbox/linux/" + name
        ).read_bytes()
    for name, data in contents.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    result = subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from loop.execution.sandbox.linux import setup, management, runtime; "
                "import runpy; runpy.run_path(sys.argv[1]+'/loop_receiver.py'); "
                "runpy.run_path(sys.argv[1]+'/loop_uninstall.py')"
            ),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
