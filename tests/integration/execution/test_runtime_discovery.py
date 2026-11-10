"""Verify runtime discovery respects native host audit vetoes."""

import os
import subprocess
import sys

import pytest

from loop.utils.path import has_extended_access_acl

pytestmark = pytest.mark.integration


def test_scratch_script_audit_veto_prevents_execution(tmp_path):
    """A host audit hook can reject inspection of the selected scratch script."""
    source = """
import os, sys
from pathlib import Path
from unittest.mock import Mock
from loop.execution import CommandExecutionService
from loop.permissions import Action, Operation, PermissionManager, ProcessBoundary, ProcessTarget
from loop.tooling import ToolContext
workspace = Path(sys.argv[1])
permissions = PermissionManager(workspace)
(permissions.temporary_directory / "veto.sh").write_text("#!/bin/sh\\ntrue\\n")
backend = Mock()
type(backend).trusted_system_path = lambda self, path: True
backend.system_read_roots = ()
command = 'sh "$TMPDIR/veto.sh"'
context = ToolContext(Mock(), 'run_command', operations=(Operation(tool_id='run_command',
    action=Action.PROCESS_EXECUTE, target=ProcessTarget(argv=('/bin/sh', '-c', command),
    cwd=str(workspace), boundary=ProcessBoundary.SANDBOXED)),), permission_manager=permissions)
def audit(event, args):
    if event == 'open' and args[0] == 'veto.sh':
        raise PermissionError('scratch audit veto')
sys.addaudithook(audit)
result = CommandExecutionService(backend, Mock()).run_command(context, command)
if result.code != 'sandbox.unavailable' or result.detail != 'Command preparation failed.':
    raise RuntimeError(f'audit veto was not reported: {result}')
backend.run.assert_not_called()
print('scratch-audit-veto-preserved')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", source, str(tmp_path)],
        env={**os.environ, "PATH": os.defpath},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "scratch-audit-veto-preserved\n"
    assert result.stderr == ""


def test_makefile_audit_veto_prevents_execution(tmp_path):
    """An actual audit hook can reject makefile inspection before any sandbox launch."""
    (tmp_path / "Makefile").write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    source = """
import sys
from pathlib import Path
from unittest.mock import Mock
from loop.execution import CommandExecutionService
from loop.permissions import Action, Operation, PermissionManager, ProcessBoundary, ProcessTarget
from loop.tooling import ToolContext

workspace = Path(sys.argv[1])
backend = Mock()
backend.system_read_roots = ()
context = ToolContext(
    Mock(), 'run_command',
    operations=(Operation(tool_id='run_command', action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(argv=('/bin/sh', '-c', 'make test'), cwd=str(workspace),
            boundary=ProcessBoundary.SANDBOXED)),),
    permission_manager=PermissionManager(workspace),
)
vetoed = []
def audit(event, args):
    if event == 'open' and args[0] == 'Makefile':
        vetoed.append(True)
        raise PermissionError('makefile audit veto')
sys.addaudithook(audit)
result = CommandExecutionService(backend, Mock()).run_command(context, 'make test')
if not vetoed or result.code != 'sandbox.unavailable' or result.detail != 'Command preparation failed.':
    raise RuntimeError(f'audit veto was not reported: {result}')
backend.run.assert_not_called()
print('audit-veto-preserved')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", source, str(tmp_path)],
        env={**os.environ, "PATH": os.defpath},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "audit-veto-preserved\n"
    assert result.stderr == ""


@pytest.mark.parametrize("exception", ["PermissionError", "OSError", "ValueError", "RuntimeError"])
@pytest.mark.parametrize("operation", ["runtime", "direct", "make", "search"])
def test_runtime_receipt_audit_veto_prevents_execution(tmp_path, exception, operation):
    """Real audit hooks can veto receipt reads before command or native search launch."""
    source = r"""
import builtins, os, sys
from pathlib import Path
from unittest.mock import Mock
from loop.execution import CommandExecutionService
from loop.execution import facade
from loop.execution.sandbox.macos.backend import MacOSSeatbeltBackend
from loop.permissions import Action, Operation, PermissionManager, ProcessBoundary, ProcessTarget
from loop.tooling import ToolContext

base = Path(sys.argv[1])
workspace = base / 'workspace'
commands = workspace / '.venv/bin'
commands.mkdir(parents=True)
package = base / 'Cellar/python/1'
(package / 'bin').mkdir(parents=True)
selected = package / 'bin/python3.14'
selected.write_text('fixture')
selected.chmod(0o755)
receipt = package / 'INSTALL_RECEIPT.json'
receipt.write_text('{"runtime_dependencies": []}')
alias = base / 'opt/python'
alias.parent.mkdir()
alias.symlink_to(package, target_is_directory=True)
(commands / 'python').symlink_to(alias / 'bin/python3.14')
(commands.parent / 'pyvenv.cfg').write_text(
    f'home = {alias / "bin"}\nimplementation = CPython\nversion_info = 3.14\n')
command = './.venv/bin/python -c pass'
if sys.argv[3] in ('direct', 'make'):
    os.environ['PATH'] = str(alias / 'bin') + os.pathsep + os.defpath
    command = 'python3.14 -c pass'
    if sys.argv[3] == 'make':
        (workspace / 'Makefile').write_text('.PHONY: test\ntest:\n\tpython3.14 -c pass\n')
        command = 'make test'
        MacOSSeatbeltBackend.trusted_system_path = lambda *args, **kwargs: True
backend = MacOSSeatbeltBackend()
backend.capability_failure = Mock(return_value=None)
backend.run = Mock()
backend.run_read_only_argv = Mock()
facade.ripgrep_path = lambda: str(selected)
context = ToolContext(Mock(), 'run_command', operations=(Operation(tool_id='run_command',
    action=Action.PROCESS_EXECUTE, target=ProcessTarget(argv=('/bin/sh', '-c', command),
    cwd=str(workspace), boundary=ProcessBoundary.SANDBOXED)),),
    permission_manager=PermissionManager(workspace))
vetoed = []
def audit(event, args):
    if event == 'open' and args[0] == str(receipt):
        vetoed.append(True)
        raise getattr(builtins, sys.argv[2])('fixture audit veto')
sys.addaudithook(audit)
try:
    service = CommandExecutionService(backend, Mock())
    result = (service.run_command(context, command) if sys.argv[3] != 'search'
        else service.run_trusted_search((), (), 10.0))
except Exception as exc:
    if type(exc) is not getattr(builtins, sys.argv[2]) or str(exc) != 'fixture audit veto':
        raise
else:
    if sys.argv[3] == 'search' or result.code != 'sandbox.unavailable':
        raise RuntimeError(f'audit veto was not reported: {result}')
if not vetoed:
    raise RuntimeError('receipt was not checked')
backend.run.assert_not_called()
backend.run_read_only_argv.assert_not_called()
print('receipt-audit-veto-preserved')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", source, str(tmp_path), exception, operation],
        env={**os.environ, "PATH": os.defpath},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "receipt-audit-veto-preserved\n"
    assert result.stderr == ""


@pytest.mark.skipif(sys.platform != "darwin", reason="Requires native macOS extended ACLs")
def test_system_path_acl_probe_detects_a_real_file_acl(tmp_path):
    """The descriptor-level ACL probe detects and clears a native file access list."""
    selected = tmp_path / "selected"
    selected.write_text("fixture")
    descriptor = os.open(selected, os.O_RDONLY | os.O_NOFOLLOW)
    added = False
    try:
        assert not has_extended_access_acl(descriptor)
        subprocess.run(
            ["/bin/chmod", "+a", "user:nobody allow write", str(selected)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        added = True
        assert has_extended_access_acl(descriptor)
        subprocess.run(
            ["/bin/chmod", "-N", str(selected)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        added = False
        assert not has_extended_access_acl(descriptor)
    finally:
        if added:
            subprocess.run(["/bin/chmod", "-N", str(selected)], check=True, timeout=5)
        os.close(descriptor)
