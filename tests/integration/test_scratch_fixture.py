"""Verify pytest scratch admission through independent test processes."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
_SETFACL = shutil.which("setfacl")


def run_scratch_probe(root, *, temporary=False):
    """Run one isolated case with a selected root or inherited temporary directory."""
    environment = os.environ.copy()
    if temporary:
        environment.pop("LOOP_TEST_SCRATCH_ROOT", None)
        environment["TMPDIR"] = str(root)
    else:
        environment["LOOP_TEST_SCRATCH_ROOT"] = str(root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "tests/unit/test_isolation.py::test_environment_and_current_directory_are_disposable",
            "-q",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_interrupted_shebang_fixture_never_creates_workspace_files(tmp_path):
    """A killed test with whitespace in scratch leaves its workspace untouched."""
    root = tmp_path / "private scratch"
    root.mkdir(mode=0o700)
    case = root / "test_interrupted.py"
    case.write_text(
        "import os\n"
        "def test_interrupt(shebang_root):\n"
        "    (shebang_root / 'marker').write_text('fixture')\n"
        "    os._exit(17)\n",
        encoding="utf-8",
    )
    project = Path(__file__).resolve().parents[2]
    existing = set(project.glob("loop-native-test-*"))
    environment = os.environ | {
        "LOOP_TEST_SCRATCH_ROOT": str(root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(filter(None, (str(project), os.environ.get("PYTHONPATH")))),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-c",
            str(project / "pyproject.toml"),
            "-p",
            "tests.conftest",
            "-p",
            "tests.unit.conftest",
            "-q",
            str(case),
        ],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 17, result.stdout + result.stderr
    assert set(project.glob("loop-native-test-*")) == existing


def test_case_monkeypatch_does_not_leave_a_deleted_temporary_directory(tmp_path):
    """Later Linux fixtures reuse clean roots after a case patches its temporary directory."""
    root = tmp_path / "nested-scratch"
    root.mkdir(mode=0o700)
    case_directory = tmp_path / "brackets [x]"
    case_directory.mkdir()
    case = case_directory / "test_fixture_lifetime.py"
    case.write_text(
        "import tempfile\n"
        "from pathlib import Path\n"
        "import pytest\n"
        "from loop.execution.sandbox import SandboxRequest\n"
        "from loop.execution.sandbox.linux import policy\n"
        "from tests.integration.linux.test_installed_execution import public, public_workspace_root\n"
        "\n"
        "@pytest.fixture(scope='module')\n"
        "def installation():\n"
        "    return None, 'fixture-generation'\n"
        "\n"
        "@pytest.fixture(scope='module')\n"
        "def shared():\n"
        "    return Path(tempfile.gettempdir()).resolve(strict=True)\n"
        "\n"
        "@pytest.fixture(scope='module')\n"
        "def observed():\n"
        "    return {}\n"
        "\n"
        "def fingerprint(public, public_workspace_root, external, monkeypatch):\n"
        "    anchor = public_workspace_root[0].parent\n"
        "    protected = policy.is_protected_external_read\n"
        "    def logical(path):\n"
        "        if path.is_relative_to(anchor):\n"
        "            path = Path('/fixture') / path.relative_to(anchor)\n"
        "        return protected(path)\n"
        "    monkeypatch.setattr(policy, 'is_protected_external_read', logical)\n"
        "    workspace = public[1]\n"
        "    scratch = public[0].configuration_path.parent / 'instruction-scratch'\n"
        "    scratch.mkdir(mode=0o700)\n"
        "    request = SandboxRequest.create(source='true', cwd=workspace, workspace=workspace,\n"
        "        read_roots=(external,), write_roots=(), network=False,\n"
        "        environment={'TMPDIR': str(scratch), 'PATH': '/usr/bin:/bin'},\n"
        "        scratch_root=scratch, policy_version='fixture', deadline=1000, workspace_id='fixture')\n"
        "    return policy.payload_profile(request, scratch, Path('/fixture/cache.lock'), (Path('/usr'),))\n"
        "\n"
        "def test_patched(monkeypatch, tmp_path):\n"
        "    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))\n"
        "\n"
        "def test_later(shared, public, public_workspace_root, native_directory, observed, monkeypatch):\n"
        "    assert shared.is_dir()\n"
        "    assert public[1].is_dir()\n"
        "    assert public[2] == 'fixture-generation'\n"
        "    parent = public[0].configuration_path.parent\n"
        "    external = native_directory(public_workspace_root[2], name=parent.name + '-installed')\n"
        "    (parent / 'prior-marker').write_text('private')\n"
        "    (external / 'prior-marker').write_text('private')\n"
        "    observed.update(parent=parent, external=external,\n"
        "        profile=fingerprint(public, public_workspace_root, external, monkeypatch))\n"
        "\n"
        "def test_root_is_still_available(public, public_workspace_root, native_directory, observed, monkeypatch):\n"
        "    assert public[1].is_dir()\n"
        "    parent = public[0].configuration_path.parent\n"
        "    external = native_directory(public_workspace_root[2], name=parent.name + '-installed')\n"
        "    assert parent == observed['parent'] and external == observed['external']\n"
        "    assert not (parent / 'prior-marker').exists()\n"
        "    assert not (external / 'prior-marker').exists()\n"
        "    assert fingerprint(public, public_workspace_root, external, monkeypatch) == observed['profile']\n",
        encoding="utf-8",
    )
    project = Path(__file__).resolve().parents[2]
    environment = os.environ | {
        "LOOP_TEST_SCRATCH_ROOT": str(root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(filter(None, (str(project), os.environ.get("PYTHONPATH")))),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-c",
            str(project / "pyproject.toml"),
            "-p",
            "tests.conftest",
            "-q",
            case.name,
        ],
        cwd=case_directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "warnings summary" not in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("kind", ["private", "alias", "shared", "file", "missing", "relative"])
def test_selected_scratch_root_requires_private_existing_directory(tmp_path, kind):
    """Accept a private root or alias and reject unsafe or missing roots before cases run."""
    root = tmp_path / "scratch-root"
    if kind == "file":
        root.write_text("not a directory", encoding="utf-8")
    elif kind == "alias":
        target = tmp_path / "actual-root"
        target.mkdir(mode=0o700)
        root.symlink_to(target, target_is_directory=True)
    elif kind != "missing":
        root.mkdir(mode=0o700)
    if kind == "shared":
        root.chmod(0o755)

    result = run_scratch_probe(Path("relative") if kind == "relative" else root)

    assert result.stderr == ""
    if kind in {"private", "alias"}:
        assert result.returncode == 0, result.stdout + result.stderr
        assert list(root.iterdir()) == []
    else:
        assert result.returncode != 0
        assert {
            "shared": "mode 0700 and no extended access list",
            "file": "mode 0700 and no extended access list",
            "missing": "cannot be resolved",
            "relative": "must be an absolute existing directory",
        }[kind] in result.stdout


@pytest.mark.parametrize("component", [".ssh", "Library", "space @ é name"])
def test_selected_scratch_root_accepts_private_path_spellings(tmp_path, component):
    """Owned private directories work even beneath protected names or unusual characters."""
    root = tmp_path / component / "scratch"
    root.mkdir(parents=True, mode=0o700)

    result = run_scratch_probe(root)

    assert result.returncode == 0, result.stdout
    assert result.stderr == ""
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("component", ["Library", ".local", "space @ é name"])
def test_default_scratch_accepts_inherited_temporary_directory(tmp_path, component):
    """Inherited temporary storage may use private names, spaces and Unicode."""
    root = tmp_path / component / "temporary"
    root.mkdir(parents=True, mode=0o700)

    result = run_scratch_probe(root, temporary=True)

    assert result.stderr == ""
    assert result.returncode == 0, result.stdout
    assert list(root.iterdir()) == []


@pytest.mark.skipif(sys.platform != "darwin", reason="Requires native macOS extended ACLs")
def test_selected_scratch_root_rejects_extended_acl(tmp_path):
    """A 0700 directory with an extended ACL is refused until the ACL is removed."""
    root = tmp_path / "scratch-root"
    root.mkdir(mode=0o700)
    added = subprocess.run(
        ["/bin/chmod", "+a", "user:nobody allow read,search", str(root)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert added.returncode == 0, added.stderr
    try:
        assert root.stat().st_mode & 0o777 == 0o700
        rejected = run_scratch_probe(root)
        assert rejected.returncode != 0
        assert rejected.stderr == ""
        assert "mode 0700 and no extended access list" in rejected.stdout
    finally:
        subprocess.run(["/bin/chmod", "-N", str(root)], check=True, timeout=5)

    accepted = run_scratch_probe(root)
    assert accepted.stderr == ""
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert list(root.iterdir()) == []


@pytest.mark.skipif(
    sys.platform != "linux" or _SETFACL is None,
    reason="Requires Linux and setfacl",
)
@pytest.mark.parametrize("kind", ["access", "default"])
def test_selected_scratch_root_rejects_linux_acl(tmp_path, kind):
    """Access and inheritable default ACLs are refused even with mode 0700."""
    root = tmp_path / "scratch-root"
    root.mkdir(mode=0o700)
    specification = "u:65534:r-x,m::---" if kind == "access" else "d:u:65534:r-x"
    added = subprocess.run(
        [_SETFACL, "-m", specification, str(root)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert added.returncode == 0, added.stderr
    try:
        assert root.stat().st_mode & 0o777 == 0o700
        rejected = run_scratch_probe(root)
        assert rejected.returncode != 0
        assert rejected.stderr == ""
        assert "mode 0700 and no extended access list" in rejected.stdout
    finally:
        subprocess.run([_SETFACL, "-b", "-k", str(root)], check=True, timeout=5)

    accepted = run_scratch_probe(root)
    assert accepted.stderr == ""
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert list(root.iterdir()) == []
