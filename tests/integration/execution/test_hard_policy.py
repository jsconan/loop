"""Qualify production native hard denials and ordinary link behavior on owned data."""

import os
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from loop import PermissionConfiguration, PermissionManager
from loop.execution.coordinator import CommandExecutionCoordinator
from loop.execution.sandbox import (
    SandboxOutcome,
    SandboxRequest,
    UnavailableSandboxBackend,
    select_sandbox_backend,
)
from loop.execution.sandbox.macos.backend import MacOSSeatbeltBackend
from loop.permissions import PolicyLimits
from loop.utils.yaml import dump_yaml
from tests.scheduling import receive_fifo_byte

pytestmark = pytest.mark.integration
_NATIVE_PLATFORM = "MACOS" if sys.platform == "darwin" else "LINUX"
_NATIVE_MODE = "owned-volume" if sys.platform == "darwin" else "installed"
_NATIVE_ENABLED = os.environ.get(f"LOOP_NATIVE_{_NATIVE_PLATFORM}_TEST") == _NATIVE_MODE


class RefusingHost:
    """Reject implicit host execution during native qualification."""

    def run_host_command(self, request):
        """Fail if any native outcome triggers an unrestricted process."""
        raise AssertionError("Native qualification attempted host execution")


@pytest.fixture
def native_hard_policy(native_directory):
    """Select the installed native boundary and own every canary and process lifetime."""
    if not _NATIVE_ENABLED:
        pytest.skip(
            f"Set LOOP_NATIVE_{_NATIVE_PLATFORM}_TEST={_NATIVE_MODE} for native qualification"
        )
    root = native_directory("/private/tmp" if sys.platform == "darwin" else "/tmp")
    workspace = root / "workspace"
    workspace.mkdir()
    # Offline hard-policy cases need no socket transport or filesystem attachment.
    backend = MacOSSeatbeltBackend() if sys.platform == "darwin" else select_sandbox_backend()
    assert not isinstance(backend, UnavailableSandboxBackend), backend.capability_failure()
    policy = root / "policy.yaml"
    policy.write_text(
        dump_yaml(
            PermissionConfiguration(
                limits=PolicyLimits(hard_denied_paths=(str(workspace / "designated"),))
            ).model_dump(mode="json")
        )
    )
    manager = PermissionManager(workspace, yolo=True, configuration_path=policy)
    coordinator = CommandExecutionCoordinator(backend, RefusingHost(), manager, None)

    def run(source, **changes):
        """Launch opaque source through real coordinator admission and native confinement."""
        values = {
            "source": source,
            "workspace": workspace,
            "cwd": workspace,
            "read_roots": (),
            "write_roots": (workspace,),
            "environment": {
                "PATH": "/usr/bin:/bin",
                "LANG": "C",
                "TMPDIR": str(manager.temporary_directory),
            },
            "scratch_root": manager.temporary_directory,
            "scratch_data": True,
            "network": False,
            "local_network": False,
            "deadline": time.monotonic() + 30,
            "policy_version": backend.policy_version,
            "workspace_id": "hard-policy-fixture",
        }
        values.update(changes)
        attempt = coordinator.run_sandboxed(SandboxRequest.create(**values), "run_command")
        assert attempt.offer is None
        assert attempt.result is not None
        return attempt.result

    try:
        yield workspace, run, manager
    finally:
        backend.close()
        manager.close()


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.production",
        "designated",
        ".env.public.template/.env.local",
        "parent/.env",
        "parent\nname/.ssh/key",
        ".ssh/new\nline",
    ],
)
def test_native_hard_names_deny_direct_child_symlink_link_and_mutation(native_hard_policy, name):
    """A broad approved workspace grant cannot disclose or mutate hard data through children."""
    workspace, run, _manager = native_hard_policy
    secret = workspace / name
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("hard-canary")
    (workspace / "symbolic").symlink_to(secret)
    quoted = shlex.quote(name)
    for source in (
        f"cat {quoted}",
        f"sh -c {shlex.quote('cat ' + quoted)}",
        "cat symbolic",
        f"ln {quoted} alias && cat alias",
        f"mv {quoted} moved && cat moved",
        f"printf changed > {quoted}",
    ):
        result = run(source)
        assert "hard-canary" not in result.stdout, result
        assert result.exit_code is not None, result.detail
        assert result.exit_code != 0, result
        assert secret.read_text() == "hard-canary"
        assert not (workspace / "alias").exists()
        assert not (workspace / "moved").exists()


def test_native_templates_and_benign_links_keep_ordinary_development_path(native_hard_policy):
    """Public template directories, complete aliases, new links and copy fallback remain usable."""
    workspace, run, _manager = native_hard_policy
    ordinary = workspace / "ordinary"
    ordinary.write_text("ordinary-canary")
    os.link(ordinary, workspace / "existing")
    template = workspace / ".env.public.template"
    template.mkdir()
    (template / "data").write_text("template-canary")
    result = run(
        "cat existing .env.public.template/data; "
        "cp ordinary copied && ln copied linked && cat linked; "
        "printf public > .env.future.example && cat .env.future.example"
    )
    assert result.outcome is SandboxOutcome.COMPLETED, result.detail
    assert result.exit_code == 0, result
    assert result.stdout == "ordinary-canarytemplate-canaryordinary-canarypublic"


@pytest.mark.parametrize("existing", [False, True])
def test_native_alias_cannot_mutate_an_approved_read_only_source(native_hard_policy, existing):
    """An approved external read root never supplies write authority through a workspace alias."""
    workspace, run, _manager = native_hard_policy
    outside = workspace.parent / "outside"
    outside.mkdir()
    source = outside / "ordinary"
    source.write_text("read-only-canary")
    if existing:
        os.link(source, workspace / "alias")
    command = "printf changed > alias"
    if not existing:
        command = f"ln {shlex.quote(str(source))} alias && " + command
    result = run(command, read_roots=(outside,))
    if result.outcome is SandboxOutcome.INVALID:
        assert existing and not result.possible_effects, result
        assert "read/write scopes" in result.detail
    else:
        assert result.exit_code is not None, result.detail
        assert result.exit_code != 0, result
    assert source.read_text() == "read-only-canary"


@pytest.mark.parametrize("kind", ["outside", "protected"])
def test_native_incomplete_and_protected_aliases_refuse_before_effects(native_hard_policy, kind):
    """Admission does not expose an unapproved outside inode or a known hard-resource alias."""
    workspace, run, _manager = native_hard_policy
    source = workspace.parent / "outside" if kind == "outside" else workspace / ".env"
    source.write_text("hard-canary")
    os.link(source, workspace / "alias")
    result = run("printf started > marker; cat alias")
    assert result.outcome is SandboxOutcome.INVALID, result
    assert not result.possible_effects, result.detail
    assert "hard-canary" not in result.stdout
    assert not (workspace / "marker").exists()


def test_native_hard_designation_in_runtime_baseline_refuses_before_effects(native_hard_policy):
    """Trusted runtime authority cannot waive an exact hard designation inside that baseline."""
    workspace, run, manager = native_hard_policy
    manager.configuration_path.write_text(
        dump_yaml(
            PermissionConfiguration(
                limits=PolicyLimits(hard_denied_paths=("/usr/bin",))
            ).model_dump(mode="json")
        )
    )
    manager.reload()
    result = run("printf started > marker")
    assert result.outcome is SandboxOutcome.INVALID, result
    assert not result.possible_effects, result
    assert not (workspace / "marker").exists()


def test_native_future_hard_names_and_parent_relocation_remain_denied(native_hard_policy):
    """Native rules protect future secret names and prevent moving a protected parent."""
    workspace, run, _manager = native_hard_policy
    parent = workspace / "parent"
    parent.mkdir()
    (parent / ".env").write_text("hard-canary")
    for source in (
        "printf secret > .env.future",
        "printf secret > designated",
        "printf ordinary > ordinary; mv ordinary .env.future",
        "mv parent relocated; cat relocated/.env",
    ):
        result = run(source)
        assert result.exit_code != 0, result
        assert "hard-canary" not in result.stdout
        assert not (workspace / ".env.future").exists()
        assert not (workspace / "designated").exists()
        assert (parent / ".env").read_text() == "hard-canary"


@pytest.mark.parametrize("name", [".awſ", ".ſsh", "parent\nname/.ssh"])
def test_native_future_casefolded_private_names_remain_denied(native_hard_policy, name):
    """Unicode spellings classified as private cannot be created after native admission."""
    workspace, run, _manager = native_hard_policy
    quoted = shlex.quote(name)
    result = run(f"mkdir -p {quoted} && printf hard-canary > {quoted}/key; cat {quoted}/key")
    assert result.exit_code is not None, result.detail
    assert result.exit_code != 0, result
    assert "hard-canary" not in result.stdout
    assert not (workspace / name).exists()


def test_native_inherited_host_descriptor_is_not_payload_authority(native_hard_policy):
    """An inheritable trusted-parent descriptor cannot carry a hard secret into shell children."""
    workspace, run, _manager = native_hard_policy
    secret = workspace / ".env"
    secret.write_text("hard-canary")
    descriptor = os.open(secret, os.O_RDONLY)
    os.set_inheritable(descriptor, True)
    try:
        assert os.read(descriptor, 100) == b"hard-canary"
        os.lseek(descriptor, 0, os.SEEK_SET)
        result = run(f"cat /dev/fd/{descriptor}")
        assert result.exit_code is not None, result.detail
        assert result.exit_code != 0
        assert "hard-canary" not in result.stdout
    finally:
        os.close(descriptor)


def test_native_detached_session_preserves_hard_read_and_write_denials(native_hard_policy):
    """A child in a new native session retains the parent's hard-resource restrictions."""
    workspace, run, _manager = native_hard_policy
    secret = workspace / ".env"
    secret.write_text("hard-canary")
    program = (
        "my $pid=fork(); die unless defined $pid; "
        "if (!$pid) { setsid() >= 0 or die; "
        "system('/bin/sh', '-c', 'cat .env; printf changed > .env'); exit($? >> 8); } "
        "waitpid($pid, 0); exit($? >> 8);"
    )
    result = run("/usr/bin/perl -MPOSIX -e " + shlex.quote(program))
    assert result.exit_code is not None, result.detail
    assert result.exit_code != 0
    assert "hard-canary" not in result.stdout
    assert secret.read_text() == "hard-canary"


def test_native_cross_root_copy_and_shared_publication_preserve_hard_denials(native_hard_policy):
    """An approved external source can be published while its hard neighbor stays inaccessible."""
    workspace, run, _manager = native_hard_policy
    outside = workspace.parent / "outside"
    outside.mkdir()
    (outside / "ordinary").write_text("approved-source")
    (outside / ".env").write_text("hard-canary")
    ready, release = workspace / "ready", workspace / "release"
    os.mkfifo(ready)
    os.mkfifo(release)
    source = shlex.quote(str(outside / "ordinary"))
    writer = (
        f"printf R > ready; read line < release; ln {source} published || cp {source} published"
    )
    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(run, writer, read_roots=(outside,))
        try:
            assert receive_fifo_byte(ready) == b"R"
            reader = run(
                f"cat {shlex.quote(str(outside / '.env'))}; printf reader", read_roots=(outside,)
            )
            assert reader.exit_code == 0, reader.detail
            assert reader.stdout == "reader"
        finally:
            descriptor = os.open(release, os.O_WRONLY | os.O_NONBLOCK)
            try:
                os.write(descriptor, b"go\n")
            finally:
                os.close(descriptor)
        completed = future.result(timeout=40)
    assert completed.exit_code == 0, completed.detail
    # Read-only source authority and Linux bind boundaries can require copy fallback.
    assert (workspace / "published").read_text() == "approved-source"
    published = run("cat published", read_roots=(outside,))
    assert published.exit_code == 0, published.detail
    assert published.stdout == "approved-source"
    denied = run(f"cat {shlex.quote(str(outside / '.env'))}", read_roots=(outside,))
    assert denied.exit_code != 0
    assert "hard-canary" not in denied.stdout


def test_native_policy_tightening_cancels_owned_attempt_and_guards_fresh_launch(native_hard_policy):
    """A trusted hard-policy reload cancels active execution and binds the fresh native grant."""
    workspace, run, manager = native_hard_policy
    (workspace / "ordinary").write_text("previously-readable")
    ready = workspace / "ready"
    os.mkfifo(ready)
    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(run, "printf R > ready; while :; do sleep 1; done")
        try:
            assert receive_fifo_byte(ready) == b"R"
            policy = PermissionConfiguration(
                limits=PolicyLimits(
                    hard_denied_paths=(str(workspace / "designated"), str(workspace / "ordinary"))
                )
            )
            manager.configuration_path.write_text(dump_yaml(policy.model_dump(mode="json")))
            manager.reload()
            result = future.result(timeout=40)
            assert result.outcome is SandboxOutcome.CANCELLED, result
            assert result.possible_effects
            # Cancellation is not evidence that macOS detached descendants were retired.
            if sys.platform == "darwin":
                assert result.cleanup_verified is not True
        finally:
            if not future.done():
                manager.reload()
    current = run("cat ordinary")
    assert current.exit_code is not None, current.detail
    assert current.exit_code != 0
    assert "previously-readable" not in current.stdout
    unrelated = run("printf still-usable")
    assert unrelated.exit_code == 0, unrelated.detail
    assert unrelated.stdout == "still-usable"


def test_native_hard_symlink_name_cannot_discard_requested_classification(native_hard_policy):
    """A pre-existing protected spelling that resolves into public data refuses the grant."""
    workspace, run, _manager = native_hard_policy
    ordinary = workspace / "ordinary"
    ordinary.write_text("protected-through-name")
    (workspace / ".env").symlink_to(ordinary)
    result = run("printf started > marker; cat .env")
    assert result.outcome is SandboxOutcome.INVALID, result.detail
    assert not result.possible_effects
    assert not (workspace / "marker").exists()
    assert "protected-through-name" not in result.stdout
