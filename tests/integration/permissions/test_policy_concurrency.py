"""Exercise concurrent permission policy updates using owned files and threads."""

import asyncio
import errno
import fcntl
import os
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from loop.permissions import (
    Action,
    ApprovalChoice,
    Decision,
    NetworkTarget,
    Operation,
    PermissionConfiguration,
    PermissionConfigurationError,
    PermissionLoadPolicy,
    PermissionManager,
    PermissionRule,
    PolicyScope,
    UserPermissionConfiguration,
)
from loop.utils.yaml import load_yaml
from loop.workspace import Workspace, WorkspaceRepository

pytestmark = pytest.mark.integration
_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"


def test_async_revision_checks_leave_policy_evaluation_worker_available():
    """Final checks cannot occupy every worker while evaluation owns the policy lock."""
    program = """
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from loop.permissions import Action, FileTarget, Operation, PermissionManager

async def exercise(manager, executor, root):
    async def run_blocking(function, *args):
        return await asyncio.get_running_loop().run_in_executor(executor, function, *args)

    manager.set_async_blocking_runner(run_blocking)
    revision = manager.egress_policy_revision()
    operation = Operation(
        tool_id="demo",
        action=Action.FILESYSTEM_READ,
        target=FileTarget(path=str(root / "file")),
    )
    checks = [
        asyncio.create_task(manager._sandbox_policy_revision_current_async(revision))
        for _ in range(2)
    ]
    evaluation = asyncio.create_task(manager.evaluate_with_revision_async((operation,)))
    first, second, snapshot = await asyncio.gather(*checks, evaluation)
    assert first and second and snapshot[1] == revision

root = Path(sys.argv[1]).resolve()
manager = PermissionManager(root)
try:
    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(exercise(manager, executor, root))
finally:
    manager.close()
"""
    with TemporaryDirectory() as directory:
        result = subprocess.run(
            [sys.executable, "-c", program, directory],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    assert result.returncode == 0, result.stderr


def policy_entry_probe(manager, monkeypatch):
    """Observe when a chosen worker enters a public policy transaction."""
    owner = []
    order = []
    attempted = []
    entered = threading.Event()
    original = manager.policy_update

    @contextmanager
    def observed():
        """Record the contender's transaction attempt and eventual entry."""
        selected = bool(owner and owner[0] == threading.get_ident() and not entered.is_set())
        if selected:
            attempted.append(True)
            entered.set()
        with original():
            if selected:
                order.append("acquired")
            yield

    monkeypatch.setattr(manager, "policy_update", observed)
    return SimpleNamespace(owner=owner, order=order, attempted=attempted, entered=entered)


def test_policy_writer_rejects_changes_from_another_process(tmp_path):
    """A manager cannot overwrite a newer revision published by a separate process."""
    path = tmp_path / "permissions.yaml"
    stale = PermissionManager(tmp_path, configuration_path=path)
    program = """
import sys
from pathlib import Path
from loop.permissions import Action, Decision, PermissionManager
path = Path(sys.argv[1])
manager = PermissionManager(path.parent, configuration_path=path)
manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
manager.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    with pytest.raises(OSError, match="changed.*reload"):
        stale.set_default(Action.NETWORK_REQUEST, Decision.DENY)
    assert (
        PermissionManager(tmp_path, configuration_path=path).configuration.defaults[
            Action.FILESYSTEM_READ
        ]
        is Decision.DENY
    )
    stale.close()


def test_policy_lock_coordinates_processes_and_releases_on_exit(tmp_path):
    """A separate process cannot write while ownership is held, and exit releases ownership."""
    path = tmp_path / "permissions.yaml"
    lock = path.with_name(path.name + ".lock")
    program = """
import sys
from pathlib import Path
from loop.permissions import Action, Decision, PermissionManager
path = Path(sys.argv[1])
manager = PermissionManager(path.parent, configuration_path=path)
try:
    manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
except BlockingIOError:
    print('busy')
else:
    print('committed')
manager.close()
"""
    with lock.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [sys.executable, "-c", program, str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "busy"
        assert not path.exists()
    crash = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
os._exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", crash, str(lock)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [sys.executable, "-c", program, str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "committed"


@pytest.mark.parametrize("boundary", ["lock", "stage"])
def test_policy_audit_reentrant_mutation_cannot_overwrite_a_nested_update(tmp_path, boundary):
    """A real reentrant audit callback cannot cause an older policy candidate to win."""
    path = tmp_path / "permissions.yaml"
    program = """
import sys
from pathlib import Path
from loop.permissions import Action, Decision, PermissionManager
path = Path(sys.argv[1])
boundary = sys.argv[2]
manager = PermissionManager(path.parent, configuration_path=path)
watched = str(path.with_name(path.name + '.lock') if boundary == 'lock'
              else path.with_suffix('.yaml.tmp'))
armed = True
nested = None
def audit(event, args):
    global armed, nested
    if armed and event == 'open' and str(args[0]) == watched:
        armed = False
        try:
            manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
            nested = 'committed'
        except OSError:
            nested = 'refused'
sys.addaudithook(audit)
try:
    manager.set_default(Action.NETWORK_REQUEST, Decision.DENY)
except OSError:
    outer = 'refused'
else:
    outer = 'committed'
loaded = PermissionManager(path.parent, configuration_path=path)
if boundary == 'lock':
    if (nested, outer) != ('committed', 'refused'): raise RuntimeError((nested, outer))
    if loaded.configuration.defaults[Action.FILESYSTEM_READ] is not Decision.DENY: raise RuntimeError('lost nested change')
else:
    if (nested, outer) != ('refused', 'committed'): raise RuntimeError((nested, outer))
    if loaded.configuration.defaults[Action.NETWORK_REQUEST] is not Decision.DENY: raise RuntimeError('lost outer change')
manager.close()
loaded.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(path), boundary],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX filesystem link identities")
@pytest.mark.parametrize("stage_kind", ["symlink", "hardlink", "fifo"])
def test_policy_save_refuses_unsafe_staging_file(tmp_path, stage_kind):
    """An unsafe staging entry cannot overwrite outside data or block a policy save."""
    workspace = tmp_path / "nested"
    workspace.mkdir()
    canary = tmp_path / "outside-canary"
    canary.write_text("synthetic-canary", encoding="utf-8")
    path = workspace / "permissions.yaml"
    stage = path.with_suffix(".yaml.tmp")
    if stage_kind == "symlink":
        stage.symlink_to(canary)
    elif stage_kind == "hardlink":
        stage.hardlink_to(canary)
    else:
        os.mkfifo(stage, 0o600)
    manager = PermissionManager(workspace, configuration_path=path)
    try:
        with pytest.raises(OSError) as raised:
            manager.add_rule(PermissionRule(id="deny", decision=Decision.DENY))
        if stage_kind == "symlink":
            assert raised.value.errno in (errno.ELOOP, errno.EMLINK)
        elif stage_kind == "hardlink":
            assert str(raised.value) == "Unsafe policy staging file."
        else:
            assert raised.value.errno == errno.ENXIO
        assert canary.read_text(encoding="utf-8") == "synthetic-canary"
        assert not path.exists()
        assert not manager.persistent_rules
    finally:
        manager.close()


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="Requires no-follow staging")
@pytest.mark.parametrize("boundary", ["open", "fdopen", "truncate", "rename"])
def test_policy_save_honors_audit_denial_without_committing(tmp_path, boundary):
    """Real audit hooks can refuse staging or publication without committing a policy."""
    path = tmp_path / "permissions.yaml"
    stage = path.with_suffix(".yaml.tmp")
    stage.write_text("existing", encoding="utf-8")
    stage.chmod(0o600)
    program = """
import os, sys
from pathlib import Path
from loop.permissions import Decision, PermissionManager, PermissionRule
path = Path(sys.argv[1])
stage = str(path.with_suffix('.yaml.tmp'))
boundary = sys.argv[2]
manager = PermissionManager(path.parent, configuration_path=path)
def audit(event, args):
    if boundary == 'open' and event == 'open' and args[0] == stage:
        raise PermissionError('policy audit denied')
    if boundary == 'fdopen' and event == 'open' and isinstance(args[0], int) and args[1] == 'w':
        raise PermissionError('policy audit denied')
    if boundary == 'truncate' and event == 'os.truncate' and isinstance(args[0], int):
        raise PermissionError('policy audit denied')
    if boundary == 'rename' and event == 'os.rename' and args[0] == stage:
        raise PermissionError('policy audit denied')
sys.addaudithook(audit)
try:
    manager.add_rule(PermissionRule(id='deny', decision=Decision.DENY))
except PermissionError as error:
    if str(error) != 'policy audit denied': raise
else:
    raise RuntimeError('Audit denial was not honored')
finally:
    manager.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(path), boundary],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    if boundary != "rename":
        assert stage.read_text(encoding="utf-8") == "existing"
    assert not path.exists()


def test_policy_save_honors_audit_denial_based_on_active_policy(tmp_path):
    """An audit hook can inspect active policy and veto a save without changing it."""
    path = tmp_path / "permissions.yaml"
    program = """
import sys
from pathlib import Path
from loop.permissions import Action, Decision, PermissionManager
path = Path(sys.argv[1])
stage = str(path.with_suffix('.yaml.tmp'))
manager = PermissionManager(path.parent, configuration_path=path)
manager.set_default(Action.NETWORK_REQUEST, Decision.DENY)
before = path.read_bytes()
def audit(event, args):
    if event == 'open' and str(args[0]) == stage:
        with manager.policy_update():
            if manager.configuration.defaults[Action.NETWORK_REQUEST] is Decision.DENY:
                raise PermissionError('denial based on active policy')
sys.addaudithook(audit)
try:
    manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
except PermissionError as error:
    if str(error) != 'denial based on active policy': raise
else:
    raise RuntimeError('Audit denial was not honored')
if path.read_bytes() != before:
    raise RuntimeError('Audit denial changed persisted policy')
if manager.configuration.defaults[Action.FILESYSTEM_READ] is not Decision.ALLOW:
    raise RuntimeError('Audit denial changed active policy')
manager.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("automatic", [False, True])
def test_policy_reload_honors_file_not_found_audit_denial(tmp_path, automatic):
    """A read veto that looks like a missing file remains a policy load error."""
    path = tmp_path / "permissions.yaml"
    program = """
import sys
from pathlib import Path
from loop.permissions import (Action, Decision, PermissionConfigurationError,
                              PermissionLoadPolicy, PermissionManager)
path = Path(sys.argv[1])
automatic = sys.argv[2] == 'true'
manager = PermissionManager(
    path.parent, configuration_path=path,
    load_policy=PermissionLoadPolicy.AUTO if automatic else PermissionLoadPolicy.ERROR,
)
manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
def audit(event, args):
    if event == 'open' and str(args[0]) == str(path):
        raise FileNotFoundError('policy audit denied')
sys.addaudithook(audit)
try:
    manager.reload()
except PermissionConfigurationError as error:
    if not isinstance(error.__cause__, FileNotFoundError): raise
    if str(error.__cause__) != 'policy audit denied': raise
else:
    raise RuntimeError('Audit denial was not honored')
if manager.configuration.defaults[Action.FILESYSTEM_READ] is not Decision.DENY:
    raise RuntimeError('Audit denial changed active policy')
manager.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(path), str(automatic).lower()],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env={**os.environ, "PYTHONPATH": str(_SOURCE_ROOT)},
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("scope", ["workspace", "user"])
@pytest.mark.parametrize("yolo", [False, True])
def test_policy_reset_honors_audit_denial_of_marker_creation(tmp_path, scope, yolo):
    """A veto on reset-marker creation leaves the existing policy untouched."""
    program = """
import sys
from pathlib import Path
from unittest.mock import Mock
from loop.permissions import Action, Decision, PermissionManager
root = Path(sys.argv[1])
scope = sys.argv[2]
yolo = sys.argv[3] == 'true'
path = root / ('permissions.yaml' if scope == 'workspace' else 'user.yaml')
marker = path.with_name(path.name + '.reset-pending')
interaction = Mock()
manager = PermissionManager(root, configuration_path=root / 'permissions.yaml',
                            user_configuration_path=root / 'user.yaml',
                            yolo=yolo, interaction=interaction)
manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
before = path.read_bytes() if path.exists() else None
def audit(event, args):
    if event == 'open' and str(args[0]) == str(marker):
        raise PermissionError('reset marker audit denied')
sys.addaudithook(audit)
try:
    manager.reset_configuration(scope)
except PermissionError as error:
    if str(error) != 'reset marker audit denied': raise
else:
    raise RuntimeError('Audit denial was not honored')
if (path.read_bytes() if path.exists() else None) != before or marker.exists():
    raise RuntimeError('Audit denial changed the policy or created a marker')
interaction.prompt.assert_not_called()
manager.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), scope, str(yolo).lower()],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env={**os.environ, "PYTHONPATH": str(_SOURCE_ROOT)},
    )

    assert result.returncode == 0, result.stderr


def test_first_use_refuses_an_audit_veto_on_backup_listing(tmp_path):
    """A denied directory listing cannot hide recovery evidence and publish defaults."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    catalog = tmp_path / "private" / "workspaces.sqlite"
    selected = WorkspaceRepository(catalog).initialize(Workspace.discover(workspace))
    path = catalog.parent / "workspaces" / selected.id / "permissions.yaml"
    path.parent.mkdir(parents=True)
    backup = path.with_name(path.name + ".old.bak")
    backup.write_text("version: 1\ndefaults:\n  filesystem.read: deny\n", encoding="utf-8")
    program = """
import sys
from pathlib import Path
from loop.permissions import PermissionManager
workspace, workspace_id, catalog, path = sys.argv[1:]
catalog, path = Path(catalog), Path(path)
def audit(event, args):
    if event == 'os.scandir' and str(args[0]) == str(path.parent):
        raise PermissionError('policy backup listing denied')
sys.addaudithook(audit)
manager = PermissionManager(
    workspace_root=Path(workspace), configuration_path=path,
    user_configuration_path=catalog.parent / 'permissions.yaml',
    workspace_id=workspace_id, namespace_catalog_path=catalog,
    user_namespace_identity=str(catalog.parent), bootstrap_pristine_policy=True,
)
if 'Policy backup evidence cannot be inspected' not in manager.describe():
    raise RuntimeError('Backup listing veto did not require recovery')
if path.exists(): raise RuntimeError('Published defaults despite hidden backup')
manager.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(workspace), selected.id, str(catalog), str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env={**os.environ, "PYTHONPATH": str(_SOURCE_ROOT)},
    )
    assert result.returncode == 0, result.stderr
    assert not path.exists()
    assert backup.read_text(encoding="utf-8").endswith("filesystem.read: deny\n")


@pytest.mark.parametrize(
    "interruption", ["before_archive", "after_archive", "crash", "published", "marker_remove"]
)
def test_policy_reset_interruption_preserves_recoverable_state(tmp_path, interruption):
    """Reset failures and process exits cannot silently discard the prior policy."""
    path = tmp_path / "permissions.yaml"
    manager = PermissionManager(tmp_path, configuration_path=path)
    manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    previous = path.read_bytes()
    manager.close()
    program = """
import os, sys
from pathlib import Path
from loop.permissions import PermissionManager
path = Path(sys.argv[1])
stage = path.with_suffix('.yaml.tmp')
marker = path.with_name(path.name + '.reset-pending')
mode = sys.argv[2]
manager = PermissionManager(path.parent, configuration_path=path)
if mode == 'published':
    replace = os.replace
    def exit_after_publication(source, destination, **kwargs):
        result = replace(source, destination, **kwargs)
        if str(source) == str(stage): os._exit(74)
        return result
    os.replace = exit_after_publication
def audit(event, args):
    if mode == 'before_archive' and event == 'os.rename' and str(args[0]) == str(path):
        raise PermissionError('archive refused')
    if mode == 'marker_remove' and event == 'os.remove' and str(args[0]) == str(marker):
        raise PermissionError('marker removal refused')
    if event == 'open' and str(args[0]) == str(stage):
        if mode == 'after_archive': raise PermissionError('reset refused')
        if mode == 'crash': os._exit(73)
sys.addaudithook(audit)
try:
    manager.reset_configuration("workspace")
except PermissionError as error:
    expected = {'before_archive': 'archive refused', 'marker_remove': 'marker removal refused'}.get(mode, 'reset refused')
    if str(error) != expected: raise
else:
    raise RuntimeError('Expected an interruption')
finally:
    manager.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(path), interruption],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env={**os.environ, "PYTHONPATH": str(_SOURCE_ROOT)},
    )

    assert result.returncode == {"crash": 73, "published": 74}.get(interruption, 0), result.stderr
    archives = tuple(tmp_path.glob("permissions.yaml.*.bak"))
    marker = path.with_name(f"{path.name}.reset-pending")
    assert marker.exists()
    with pytest.raises(PermissionConfigurationError, match="reset is pending"):
        PermissionManager(tmp_path, configuration_path=path, load_policy=PermissionLoadPolicy.AUTO)
    if interruption == "before_archive":
        assert not archives
        assert path.read_bytes() == previous
        marker.unlink()
        assert (
            PermissionManager(tmp_path, configuration_path=path).configuration.defaults[
                Action.FILESYSTEM_READ
            ]
            is Decision.DENY
        )
    elif interruption in {"published", "marker_remove"}:
        assert len(archives) == 1
        assert archives[0].read_bytes() == previous
        marker.unlink()
        assert (
            PermissionManager(tmp_path, configuration_path=path).configuration.defaults[
                Action.FILESYSTEM_READ
            ]
            is Decision.ALLOW
        )
    else:
        assert len(archives) == 1
        assert archives[0].read_bytes() == previous
        assert not path.exists()
        archives[0].replace(path)
        marker.unlink()
        assert (
            PermissionManager(tmp_path, configuration_path=path).configuration.defaults[
                Action.FILESYSTEM_READ
            ]
            is Decision.DENY
        )


@pytest.mark.parametrize("scope", ["workspace", "user"])
@pytest.mark.parametrize(
    ("phase", "exit_code", "pending", "published"),
    [
        ("before_marker", 70, False, False),
        ("marked", 71, True, False),
        ("archived", 72, True, False),
        ("cancelled", 79, True, False),
        ("published", 73, True, True),
        ("cleaned", 74, False, True),
    ],
)
def test_namespaced_reset_process_death_keeps_policy_or_requires_recovery(
    tmp_path, scope, phase, exit_code, pending, published
):
    """Each process-death boundary preserves old policy or leaves explicit recovery evidence."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    catalog = tmp_path / "private" / "workspaces.sqlite"
    location = WorkspaceRepository(catalog).initialize(Workspace.discover(workspace))
    workspace_path = catalog.parent / "workspaces" / location.id / "permissions.yaml"
    user_path = catalog.parent / "permissions.yaml"
    arguments = {
        "workspace_root": workspace,
        "configuration_path": workspace_path,
        "user_configuration_path": user_path,
        "workspace_id": location.id,
        "namespace_catalog_path": catalog,
        "user_namespace_identity": str(catalog.parent),
        "bootstrap_pristine_policy": True,
    }
    manager = PermissionManager(**arguments)
    manager.set_default(Action.FILESYSTEM_READ, Decision.DENY)
    manager.trust_host_command("printf private", ApprovalChoice.USER)
    manager.close()
    path = workspace_path if scope == "workspace" else user_path
    previous = path.read_bytes()
    program = """
import os, sys
from pathlib import Path
from loop.permissions import PermissionManager
workspace, workspace_id, catalog, workspace_path, user_path, scope, phase = sys.argv[1:]
catalog = Path(catalog)
path = Path(workspace_path if scope == 'workspace' else user_path)
marker = path.with_name(path.name + '.reset-pending')
stage = path.with_suffix(path.suffix + '.tmp')
manager = PermissionManager(
    workspace_root=Path(workspace), configuration_path=Path(workspace_path),
    user_configuration_path=Path(user_path), workspace_id=workspace_id,
    namespace_catalog_path=catalog, user_namespace_identity=str(catalog.parent),
    bootstrap_pristine_policy=True,
)
opened = os.open
replaced = os.replace
unlinked = os.unlink
def controlled_open(candidate, flags, *args, **kwargs):
    if phase == 'cancelled' and str(candidate) == str(stage):
        raise KeyboardInterrupt('controlled cancellation')
    descriptor = opened(candidate, flags, *args, **kwargs)
    if phase == 'marked' and str(candidate) == str(marker): os._exit(71)
    return descriptor
def controlled_replace(source, destination, **kwargs):
    result = replaced(source, destination, **kwargs)
    if phase == 'archived' and str(source) == str(path): os._exit(72)
    if phase == 'published' and str(source) == str(stage): os._exit(73)
    return result
def controlled_unlink(candidate, *args, **kwargs):
    result = unlinked(candidate, *args, **kwargs)
    if phase == 'cleaned' and str(candidate) == str(marker): os._exit(74)
    return result
def audit(event, args):
    if phase == 'before_marker' and event == 'open' and str(args[0]) == str(marker):
        os._exit(70)
sys.addaudithook(audit)
os.open = controlled_open
os.replace = controlled_replace
os.unlink = controlled_unlink
try:
    manager.reset_configuration(scope)
except KeyboardInterrupt:
    os._exit(79)
raise RuntimeError('Expected controlled process death')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(workspace),
            location.id,
            str(catalog),
            str(workspace_path),
            str(user_path),
            scope,
            phase,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env={**os.environ, "PYTHONPATH": str(_SOURCE_ROOT)},
    )
    assert result.returncode == exit_code, result.stderr
    marker = path.with_name(path.name + ".reset-pending")
    archives = tuple(path.parent.glob(f"{path.name}.*.bak"))
    assert marker.exists() is pending
    assert len(archives) == (0 if phase in {"before_marker", "marked"} else 1)
    if archives:
        assert archives[0].read_bytes() == previous
    if phase in {"marked", "before_marker"}:
        assert path.read_bytes() == previous
    elif phase in {"archived", "cancelled"}:
        assert not path.exists()
    else:
        assert path.exists()
    with closing(sqlite3.connect(catalog)) as connection:
        identity = location.id if scope == "workspace" else str(catalog.parent)
        assert connection.execute(
            "SELECT status FROM policy_namespaces WHERE scope=? AND identity=?",
            (scope, identity),
        ).fetchone() == ("initialized",)
    restarted = PermissionManager(**arguments)
    if pending:
        assert "reset is pending" in restarted.describe()
    elif scope == "workspace":
        expected = Decision.ALLOW if published else Decision.DENY
        assert restarted.configuration.defaults[Action.FILESYSTEM_READ] is expected
    else:
        assert bool(restarted.trusted_host_command_rules()) is not published
    restarted.close()


@pytest.mark.parametrize("scope", ["workspace", "user"])
@pytest.mark.parametrize(
    ("phase", "exit_code", "status"),
    [
        ("before_reservation", 75, None),
        ("reserved", 76, "reserved_new"),
        ("published", 77, "reserved_new"),
        ("initialized", 78, "initialized"),
    ],
)
def test_namespaced_first_use_recovers_process_death_at_publication_boundaries(
    tmp_path, scope, phase, exit_code, status
):
    """An interrupted first-use ledger reservation resumes without losing its identity."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    catalog = tmp_path / "private" / "workspaces.sqlite"
    location = WorkspaceRepository(catalog).initialize(Workspace.discover(workspace))
    workspace_path = catalog.parent / "workspaces" / location.id / "permissions.yaml"
    user_path = catalog.parent / "permissions.yaml"
    arguments = {
        "workspace_root": workspace,
        "configuration_path": workspace_path,
        "user_configuration_path": user_path,
        "workspace_id": location.id,
        "namespace_catalog_path": catalog,
        "user_namespace_identity": str(catalog.parent),
    }
    if scope == "user":
        manager = PermissionManager(**arguments)
        manager.initialize_policy_namespace("workspace", location.id, authorized=True)
        manager.close()
    path = workspace_path if scope == "workspace" else user_path
    program = """
import os, sys
from pathlib import Path
from loop.permissions import PermissionManager
workspace, workspace_id, catalog, workspace_path, user_path, scope, phase = sys.argv[1:]
if phase == 'before_reservation': os._exit(75)
catalog = Path(catalog)
path = Path(workspace_path if scope == 'workspace' else user_path)
stage = path.with_suffix(path.suffix + '.tmp')
opened = os.open
replaced = os.replace
def controlled_open(candidate, flags, *args, **kwargs):
    if phase == 'reserved' and str(candidate) == str(stage): os._exit(76)
    return opened(candidate, flags, *args, **kwargs)
def controlled_replace(source, destination, **kwargs):
    result = replaced(source, destination, **kwargs)
    if phase == 'published' and str(source) == str(stage): os._exit(77)
    return result
os.open = controlled_open
os.replace = controlled_replace
manager = PermissionManager(
    workspace_root=Path(workspace), configuration_path=Path(workspace_path),
    user_configuration_path=Path(user_path), workspace_id=workspace_id,
    namespace_catalog_path=catalog, user_namespace_identity=str(catalog.parent),
    bootstrap_pristine_policy=True,
)
os._exit(78)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(workspace),
            location.id,
            str(catalog),
            str(workspace_path),
            str(user_path),
            scope,
            phase,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env={**os.environ, "PYTHONPATH": str(_SOURCE_ROOT)},
    )
    assert result.returncode == exit_code, result.stderr
    identity = location.id if scope == "workspace" else str(catalog.parent)
    if scope == "workspace" and phase == "before_reservation":
        row = None
    else:
        with closing(sqlite3.connect(catalog)) as connection:
            row = connection.execute(
                "SELECT status,generation FROM policy_namespaces WHERE scope=? AND identity=?",
                (scope, identity),
            ).fetchone()
    assert (row[0] if row else None) == status
    assert path.exists() is (phase in {"published", "initialized"})
    resumed = PermissionManager(**arguments, bootstrap_pristine_policy=True)
    assert path.exists()
    with closing(sqlite3.connect(catalog)) as connection:
        settled = connection.execute(
            "SELECT status,generation FROM policy_namespaces WHERE scope=? AND identity=?",
            (scope, identity),
        ).fetchone()
    assert settled[0] == "initialized"
    if row is not None:
        assert settled[1] == row[1]
    resumed.close()


@pytest.mark.parametrize("scope", [ApprovalChoice.WORKSPACE, ApprovalChoice.USER])
def test_async_persistent_approval_is_visible_with_its_saved_revision(tmp_path, monkeypatch, scope):
    """A real policy observer sees the saved approval before its caller resumes."""
    path = tmp_path / "permissions.yaml"
    interaction = Mock()
    interaction.prompt_async = AsyncMock(return_value=scope)
    manager = PermissionManager(
        tmp_path,
        configuration_path=path if scope is ApprovalChoice.WORKSPACE else None,
        user_configuration_path=path if scope is ApprovalChoice.USER else None,
        interaction=interaction,
    )
    target = Operation(
        tool_id="test",
        action=Action.NETWORK_REQUEST,
        target=NetworkTarget(url="https://example.com", origin="https://example.com"),
    )
    initial, initial_revision = manager.evaluate_with_revision((target,))
    saved = threading.Event()
    release = threading.Event()
    original = manager._persist_configuration_candidate

    def pause_completed_write(policy_path, configuration, revision):
        """Hold the caller after the real worker has saved the policy."""
        written = original(policy_path, configuration, revision)
        saved.set()
        if not release.wait(timeout=5):
            raise TimeoutError("The controlled policy observer did not finish.")
        return written

    monkeypatch.setattr(manager, "_persist_configuration_candidate", pause_completed_write)
    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            pending = executor.submit(lambda: asyncio.run(manager.authorize_async((target,))))
            assert saved.wait(timeout=5)
            observed, revision = executor.submit(manager.evaluate_with_revision, (target,)).result(
                timeout=5
            )
            model = (
                PermissionConfiguration
                if scope is ApprovalChoice.WORKSPACE
                else UserPermissionConfiguration
            )
            persisted = model.model_validate(load_yaml(path.read_text()))
            release.set()
            result = pending.result(timeout=5)
            assert initial.decision is Decision.ASK
            assert observed.decision is Decision.ALLOW
            assert revision > initial_revision
            assert len(persisted.rules) == 1
            assert result.decision is Decision.ALLOW
            assert result.installed_rule_ids == (persisted.rules[0].id,)
        finally:
            release.set()
            executor.shutdown(wait=True)
            manager.close()


def test_concurrent_scoped_rules_cannot_share_an_identifier(tmp_path, monkeypatch):
    """A workspace write and session addition cannot publish duplicate rule IDs."""
    manager = PermissionManager(tmp_path)
    probe = policy_entry_probe(manager, monkeypatch)
    original = manager._persist_configuration
    persistence_started = threading.Event()
    release = threading.Event()
    session_started = threading.Event()
    session_finished = threading.Event()
    rule = PermissionRule(id="shared-id", decision=Decision.DENY)

    def paused_persist(path, configuration):
        """Hold the workspace update while its policy transaction is owned."""
        persistence_started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("The controlled policy write was not released.")
        original(path, configuration)

    monkeypatch.setattr(manager, "_persist_configuration", paused_persist)

    def add_session():
        """Attempt the same rule identifier in session scope."""
        probe.owner.append(threading.get_ident())
        session_started.set()
        try:
            manager.add_rule(rule, scope=PolicyScope.SESSION)
        finally:
            session_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            workspace = executor.submit(manager.add_rule, rule, scope=PolicyScope.WORKSPACE)
            assert persistence_started.wait(timeout=5)
            session = executor.submit(add_session)
            session.add_done_callback(lambda _future: probe.entered.set())
            assert session_started.wait(timeout=5)
            probe.entered.wait()
            assert probe.attempted
            assert not session_finished.is_set()
            probe.order.append("released")
            release.set()
            workspace.result(timeout=5)
            with pytest.raises(ValueError, match="already exists"):
                session.result(timeout=5)
            assert probe.order == ["released", "acquired"]
            assert [item.id for item in manager.persistent_rules] == ["shared-id"]
            assert manager.session_rules == ()
        finally:
            release.set()
            executor.shutdown(wait=True)
            manager.close()


def test_preset_preview_cannot_mix_old_values_with_a_new_revision(tmp_path, monkeypatch):
    """A policy mutation cannot interleave between preview contents and their revision."""
    manager = PermissionManager(tmp_path)
    manager.set_default(Action.PROCESS_EXECUTE, Decision.ALLOW, scope=PolicyScope.SESSION)
    probe = policy_entry_probe(manager, monkeypatch)
    original = manager._defaults_for_scope
    snapshot_started = threading.Event()
    release = threading.Event()
    mutation_started = threading.Event()
    mutation_finished = threading.Event()

    def paused_defaults(scope):
        """Hold a preview after its old defaults have been copied."""
        defaults = original(scope)
        snapshot_started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("The controlled preview was not released.")
        return defaults

    monkeypatch.setattr(manager, "_defaults_for_scope", paused_defaults)

    def mutate():
        """Change the same scoped default while the preview is in progress."""
        probe.owner.append(threading.get_ident())
        mutation_started.set()
        manager.set_default(Action.PROCESS_EXECUTE, Decision.DENY, scope=PolicyScope.SESSION)
        mutation_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            pending = executor.submit(
                manager.preview_preset_replacement, "locked", scope=PolicyScope.SESSION
            )
            assert snapshot_started.wait(timeout=5)
            mutation = executor.submit(mutate)
            mutation.add_done_callback(lambda _future: probe.entered.set())
            assert mutation_started.wait(timeout=5)
            probe.entered.wait()
            assert probe.attempted
            assert not mutation_finished.is_set()
            probe.order.append("released")
            release.set()
            preview = pending.result(timeout=5)
            mutation.result(timeout=5)
            assert probe.order == ["released", "acquired"]
            assert preview.removed_defaults[Action.PROCESS_EXECUTE] is Decision.ALLOW
            with pytest.raises(ValueError, match="stale"):
                manager.replace_preset(preview)
        finally:
            release.set()
            executor.shutdown(wait=True)
            manager.close()


def test_workspace_update_cannot_restore_a_concurrently_revoked_host_limit(tmp_path, monkeypatch):
    """A stale unrelated policy write cannot overwrite a completed host revocation."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.WORKSPACE)
    probe = policy_entry_probe(manager, monkeypatch)
    original = PermissionConfiguration.model_copy
    copied = threading.Event()
    release = threading.Event()
    revocation_started = threading.Event()
    revoked = threading.Event()
    update_thread = []

    def paused_copy(configuration, *args, **kwargs):
        """Hold an unrelated update immediately after it copies the active policy."""
        updated = original(configuration, *args, **kwargs)
        if update_thread and threading.get_ident() == update_thread[0] and not copied.is_set():
            copied.set()
            if not release.wait(timeout=5):
                raise TimeoutError("The controlled policy update was not released.")
        return updated

    monkeypatch.setattr(PermissionConfiguration, "model_copy", paused_copy)

    def update_default():
        """Write an unrelated default while another worker revokes host authority."""
        update_thread.append(threading.get_ident())
        manager.set_default(Action.FILESYSTEM_READ, Decision.DENY, scope=PolicyScope.WORKSPACE)

    def revoke():
        """Publish a workspace host-process revocation."""
        probe.owner.append(threading.get_ident())
        revocation_started.set()
        manager.set_limit("allow_host_processes", False, scope=PolicyScope.WORKSPACE)
        revoked.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            update = executor.submit(update_default)
            assert copied.wait(timeout=5)
            revocation = executor.submit(revoke)
            revocation.add_done_callback(lambda _future: probe.entered.set())
            assert revocation_started.wait(timeout=5)
            probe.entered.wait()
            assert probe.attempted
            assert not revoked.is_set()
            probe.order.append("released")
            release.set()
            update.result(timeout=5)
            revocation.result(timeout=5)
            assert probe.order == ["released", "acquired"]
            assert revoked.is_set()
            reloaded = PermissionManager(tmp_path)
            try:
                assert reloaded.configuration.defaults[Action.FILESYSTEM_READ] is Decision.DENY
                assert not reloaded.configuration.limits.allow_host_processes
            finally:
                reloaded.close()
        finally:
            release.set()
            executor.shutdown(wait=True)
            manager.close()


@pytest.mark.parametrize("method", ["matching_host_command_rule", "remember_host_command_rule"])
def test_owner_host_rule_relay_does_not_wait_for_policy_transaction(tmp_path, monkeypatch, method):
    """A policy worker waiting for its owner relay cannot deadlock a host-rule relay."""
    import asyncio

    from loop.execution.sandbox.contracts import SandboxRequest
    from loop.tooling.tool_registry import ToolExecutionPortal, ToolOwnerServiceProxy

    manager = PermissionManager(tmp_path)
    request = SandboxRequest.create(
        source="printf safe",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
        policy_version="test",
        deadline=160.0,
        workspace_id="test",
    )
    if method == "matching_host_command_rule":
        manager.remember_host_command_rule(request)
    locked = threading.Event()
    lookup_started = threading.Event()
    relay_completed = threading.Event()
    timed_out = threading.Event()
    original = getattr(manager, method)

    def marked_lookup(*args):
        """Start the worker's owner relay while this owner callback runs."""
        lookup_started.set()
        return original(*args)

    monkeypatch.setattr(manager, method, marked_lookup)

    async def exercise():
        """Run actual worker-to-owner relays while the policy transaction is held."""
        loop = asyncio.get_running_loop()
        portal = ToolExecutionPortal(loop)
        proxy = ToolOwnerServiceProxy(manager, portal)

        def hold_policy_and_relay():
            """Keep a policy transaction open until an owner interaction completes."""
            with manager.policy_update():
                locked.set()
                if not lookup_started.wait(timeout=5):
                    raise TimeoutError("The host lookup did not start.")
                loop.call_soon_threadsafe(relay_completed.set)
                if not relay_completed.wait(timeout=2):
                    timed_out.set()

        with ThreadPoolExecutor(max_workers=2) as executor:
            worker = executor.submit(hold_policy_and_relay)
            assert await asyncio.to_thread(locked.wait, 5)
            lookup = executor.submit(lambda: getattr(proxy, method)(request))
            await asyncio.wrap_future(lookup)
            await asyncio.wrap_future(worker)
        assert relay_completed.is_set()
        assert not timed_out.is_set()

    try:
        asyncio.run(exercise())
    finally:
        manager.close()


@pytest.mark.parametrize("change_policy", [False, True])
def test_revision_bound_host_relay_keeps_owner_available_for_policy_worker(
    tmp_path, monkeypatch, change_policy
):
    """A revision-bound host commit waits without blocking a policy worker's owner callback."""
    import asyncio

    from loop.execution.sandbox.contracts import SandboxRequest
    from loop.tooling.tool_registry import ToolExecutionPortal, ToolOwnerServiceProxy

    async def forbidden_runner(*args):
        """Nested relay preparation must not need a potentially occupied worker pool."""
        raise AssertionError("Unexpected nested worker admission")

    manager = PermissionManager(tmp_path, async_blocking_runner=forbidden_runner)
    request = SandboxRequest.create(
        source="printf safe",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
        policy_version="test",
        deadline=160.0,
        workspace_id="test",
    )
    _, revision = manager.evaluate_with_revision(())
    locked = threading.Event()
    preparation_started = threading.Event()
    callback_completed = threading.Event()
    original = manager._prepare_host_command_rule

    def prepare(candidate):
        """Signal that the owner has begun preparing the host rule."""
        preparation_started.set()
        return original(candidate)

    monkeypatch.setattr(manager, "_prepare_host_command_rule", prepare)

    async def exercise():
        """Drive both real relays while a separate thread owns the policy lock."""
        loop = asyncio.get_running_loop()
        proxy = ToolOwnerServiceProxy(manager, ToolExecutionPortal(loop))

        def policy_worker():
            """Await an owner callback before releasing the policy transaction."""
            with manager.policy_update():
                locked.set()
                if not preparation_started.wait(timeout=5):
                    raise TimeoutError("The host preparation did not start.")
                loop.call_soon_threadsafe(callback_completed.set)
                if not callback_completed.wait(timeout=5):
                    raise TimeoutError("The owner callback did not complete.")
                if change_policy:
                    manager.set_default(
                        Action.PROCESS_EXECUTE, Decision.DENY, scope=PolicyScope.SESSION
                    )

        with ThreadPoolExecutor(max_workers=2) as executor:
            worker = executor.submit(policy_worker)
            assert await asyncio.to_thread(locked.wait, 5)
            commit = executor.submit(proxy.remember_host_command_rule_if_current, request, revision)
            rule = await asyncio.wait_for(asyncio.wrap_future(commit), timeout=5)
            await asyncio.wrap_future(worker)
        assert callback_completed.is_set()
        if change_policy:
            assert rule is None
            assert manager.host_command_rules() == ()
        else:
            assert rule is not None
            assert manager.host_command_rules() == (rule,)

    try:
        asyncio.run(exercise())
    finally:
        manager.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sandbox_rule_revocation_during_lookup_prevents_authorization(
    tmp_path, monkeypatch, asynchronous
):
    """A rule revoked after lookup begins cannot authorize the pending sandbox attempt."""
    import asyncio
    from unittest.mock import Mock

    from loop.execution.sandbox.contracts import SandboxRequest
    from loop.permissions import ApprovalChoice

    interaction = Mock()
    interaction.prompt.return_value = "session"
    manager = PermissionManager(tmp_path, interaction=interaction)
    request = SandboxRequest.create(
        source="cat file",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(tmp_path.parent,),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin"},
        policy_version="test",
        deadline=160.0,
        workspace_id="test",
    )
    assert manager.authorize_sandboxed_command(request, tool_id="run_command")
    rule_id = manager.sandboxed_command_rules(ApprovalChoice.SESSION)[0].id
    original = manager.sandboxed_command_rules
    lookup_started = threading.Event()
    release = threading.Event()
    worker_thread = []

    def paused_rules(scope):
        """Hold a detached old rule snapshot while the user revokes its authority."""
        rules = original(scope)
        if scope is ApprovalChoice.SESSION and threading.get_ident() in worker_thread:
            lookup_started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("The sandbox rule lookup was not released.")
        return rules

    monkeypatch.setattr(manager, "sandboxed_command_rules", paused_rules)

    def authorize():
        """Resolve the existing rule on the worker without requesting another prompt."""
        worker_thread.append(threading.get_ident())
        if asynchronous:
            return asyncio.run(
                manager.authorize_sandboxed_command_async(request, tool_id="run_command")
            )
        return manager.authorize_sandboxed_command(request, tool_id="run_command")

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            pending = executor.submit(authorize)
            assert lookup_started.wait(timeout=5)
            assert manager.remove_sandboxed_command_rule(ApprovalChoice.SESSION, rule_id)
            release.set()
            assert not pending.result(timeout=5)
            interaction.prompt.assert_called_once()
        finally:
            release.set()
            executor.shutdown(wait=True)
            manager.close()


def test_session_preset_rejects_host_rule_change_during_validation(tmp_path, monkeypatch):
    """A host rule added during preview validation makes session replacement stale."""

    from loop.execution.sandbox.contracts import SandboxRequest

    manager = PermissionManager(tmp_path)
    request = SandboxRequest.create(
        source="printf safe",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
        policy_version="test",
        deadline=160.0,
        workspace_id="test",
    )
    preview = manager.preview_preset_replacement("locked", scope=PolicyScope.SESSION)
    original = manager._scope_revision

    def change_host_rule(scope):
        """Publish another host rule after computing the comparison revision."""
        revision = original(scope)
        manager.remember_host_command_rule(request)
        return revision

    monkeypatch.setattr(manager, "_scope_revision", change_host_rule)
    try:
        with pytest.raises(ValueError, match="stale"):
            manager.replace_preset(preview)
        assert len(manager.host_command_rules()) == 1
    finally:
        manager.close()


def test_session_reset_invalidates_host_rule_being_prepared(tmp_path, monkeypatch):
    """Even an empty session reset invalidates an approval still being prepared."""

    from loop.execution.sandbox.contracts import SandboxRequest

    manager = PermissionManager(tmp_path)
    request = SandboxRequest.create(
        source="printf safe",
        cwd=tmp_path,
        workspace=tmp_path,
        read_roots=(),
        write_roots=(tmp_path,),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
        policy_version="test",
        deadline=160.0,
        workspace_id="test",
    )
    started = threading.Event()
    release = threading.Event()
    original = SandboxRequest.host_command_runtime_signature

    def paused_signature(candidate):
        """Pause after approval preparation starts but before it is published."""
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("Host-rule preparation was not released.")
        return original(candidate)

    monkeypatch.setattr(SandboxRequest, "host_command_runtime_signature", paused_signature)
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            pending = executor.submit(manager.remember_host_command_rule, request)
            assert started.wait(timeout=5)
            assert not manager.reset_session()
            release.set()
            with pytest.raises(ValueError, match="changed while preparing"):
                pending.result(timeout=5)
            assert manager.host_command_rules() == ()
        finally:
            release.set()
            executor.shutdown(wait=True)
            manager.close()
