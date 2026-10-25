"""Verify native Seatbelt execution, authority binding and filesystem confinement."""

import os
import shlex
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from loop.control import RunControl
from loop.execution.sandbox import SandboxOutcome, SandboxRequest

pytestmark = [pytest.mark.integration, pytest.mark.macos]


def execute(backend, workspace, source, *, writable=True):
    """Run one approved native request bounded to the synthetic workspace."""
    request = SandboxRequest.create(
        source=source,
        cwd=workspace,
        workspace=workspace,
        read_roots=(),
        write_roots=(workspace,) if writable else (),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
        policy_version=backend.policy_version,
        deadline=time.monotonic() + 5,
        workspace_id="isolated-native-test",
    )
    return backend.run(request)


def test_native_commands_preserve_output_and_nonzero_exit(backend, native_workspace):
    """Real shell exit status and both streams survive native enforcement."""
    result = execute(backend, native_workspace, "printf output; printf error >&2; exit 7")
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 7
    assert result.stdout == "output"
    assert result.stderr == "error"


@pytest.mark.parametrize("action", ["interrupt", "stop"])
@pytest.mark.parametrize("retained_pipe", [False, True])
def test_native_cancellation_drains_running_commands_and_retained_pipes(
    backend, native_workspace, action, retained_pipe
):
    """Run interruption and shutdown promptly clean up native children and captured output."""
    source = (
        "sleep 30 & printf ready; printf started > cancellation-marker"
        if retained_pipe
        else "printf ready; printf started > cancellation-marker; sleep 30"
    )
    request = SandboxRequest.create(
        source=source,
        cwd=native_workspace,
        workspace=native_workspace,
        read_roots=(),
        write_roots=(native_workspace,),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
        policy_version=backend.policy_version,
        deadline=time.monotonic() + 30,
        workspace_id="isolated-native-test",
    )
    control = RunControl()
    with control.execution(), ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            backend.run, request, cancellation_event=control.cancellation_event
        )
        try:
            marker = native_workspace / "cancellation-marker"
            deadline = time.monotonic() + 10
            while not marker.exists() and not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.exists(), future.result() if future.done() else "No payload entry"
            started = time.monotonic()
            getattr(control, action)()
            result = future.result(timeout=5)
            assert time.monotonic() - started < 5
            assert result.outcome is SandboxOutcome.CANCELLED, result
            assert result.stdout == "ready" and result.exit_code is None
        finally:
            control.interrupt()
    assert control.cancelled
    assert control.cancellation_event.is_set() == (action == "stop")


@pytest.mark.parametrize("target", [".ssh/id_ed25519", "external-link"])
def test_native_reads_cannot_disclose_private_or_external_files(backend, native_workspace, target):
    """Real protected reads and symlink aliases never disclose synthetic private content."""
    result = execute(backend, native_workspace, f"cat {shlex.quote(target)}", writable=False)
    assert result.outcome in {SandboxOutcome.COMPLETED, SandboxOutcome.DENIED}
    assert result.exit_code != 0
    assert "fake-private-canary" not in result.stdout
    assert "fake-outside-canary" not in result.stdout


@pytest.mark.parametrize("target", ["AGENTS.md", ".git/config", "../outside/write"])
def test_native_writes_preserve_control_and_external_files(backend, native_workspace, target):
    """Native attempts cannot overwrite protected controls or files outside the workspace."""
    path = native_workspace / target
    original = path.read_text(encoding="utf-8")
    result = execute(backend, native_workspace, f"printf changed > {shlex.quote(target)}")
    assert result.exit_code != 0
    assert path.read_text(encoding="utf-8") == original


def test_native_read_only_request_cannot_mutate_workspace(backend, native_workspace):
    """Read-only native authority keeps ordinary workspace content unchanged."""
    result = execute(backend, native_workspace, "printf changed > ordinary", writable=False)
    assert result.exit_code != 0
    assert (native_workspace / "ordinary").read_text(encoding="utf-8") == "ordinary-data"


def test_native_hardlink_preflight_preserves_outside_inode(backend, native_workspace):
    """A writable hardlink alias fails closed before a real command can modify its target."""
    outside = native_workspace.parent / "outside" / "write"
    (native_workspace / "hardlink").hardlink_to(outside)
    result = execute(backend, native_workspace, "printf changed > hardlink")
    assert result.outcome is SandboxOutcome.INVALID
    assert outside.read_text(encoding="utf-8") == "preserve-outside"


def test_shell_pipeline_and_nested_child(native_command, native_workspace):
    """Shell pipelines and workspace executables remain available within native authority."""
    script = native_workspace / "script.sh"
    script.write_text("#!/bin/sh\nprintf workspace-ok\n", encoding="utf-8")
    script.chmod(0o755)
    result = native_command("printf 'hello\n' | tr a-z A-Z; /bin/sh -c './script.sh'")
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 0
    assert result.stdout == "HELLO\nworkspace-ok"


@pytest.mark.parametrize(
    "source",
    [
        "cat ../outside/secret",
        "/bin/sh -c 'cat ../outside/secret'",
        "cat external-link",
        "printf bad > ../outside/write",
        "/bin/sh -c 'printf bad > ../outside/write'",
        "printf bad > external-link",
        "ln ../outside/write newlink",
        "ln .git/config newlink",
    ],
)
def test_outside_alias_and_child_escape_remain_confined(native_command, native_workspace, source):
    """Outside aliases, nested children and hardlink creation cannot escape native authority."""
    result = native_command(source)
    assert result.exit_code != 0
    assert "fake-outside-canary" not in result.stdout
    outside = native_workspace.parent / "outside"
    assert (outside / "write").read_text() == "preserve-outside"
    assert (outside / "secret").read_text() == "fake-outside-canary"
    assert not (native_workspace / "newlink").exists()


@pytest.mark.parametrize(
    "relative",
    [
        ".git/config",
        ".git/new-control",
        ".loop/control",
        "linked/.git",
        "nested/.git/config",
        "AGENTS.md",
    ],
)
def test_protected_control_write_or_creation(native_command, native_workspace, relative):
    """Control files and linked-worktree pointers resist overwrite and creation."""
    target = native_workspace / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    if not exists and relative != ".git/new-control":
        target.write_text("preserve")
        exists = True
    original = target.read_text() if exists else None
    result = native_command(f"printf bad > {shlex.quote(relative)}")
    assert result.exit_code != 0
    assert target.read_text() == original if exists else not target.exists()


def test_workspace_root_cannot_be_renamed(native_command, native_workspace):
    """An approved workspace cannot be moved out from under its authority binding."""
    result = native_command("mv ../workspace ../moved")
    assert result.exit_code != 0
    assert native_workspace.is_dir()
    assert not (native_workspace.parent / "moved").exists()


def test_symlinked_authority_root_is_rejected(native_command, native_workspace):
    """A symlink root fails closed before a command can produce its marker."""
    alias = native_workspace.parent / "workspace-alias"
    alias.symlink_to(native_workspace, target_is_directory=True)
    with pytest.raises(ValueError):
        native_command("printf marker > ordinary", workspace=alias, cwd=alias)
    assert (native_workspace / "ordinary").read_text() == "ordinary-data"


def test_detached_child_retains_native_confinement(native_command, native_workspace):
    """A child that starts its own session still cannot disclose the outside canary."""
    source = "/usr/bin/perl -MPOSIX -e " + shlex.quote(
        "my $pid=fork(); die unless defined $pid; "
        'if (!$pid) { setsid(); system("/bin/cat", "../outside/secret"); exit($? >> 8); } '
        "waitpid($pid, 0); exit($? >> 8);"
    )
    result = native_command(source)
    assert result.exit_code != 0
    assert "fake-outside-canary" not in result.stdout
    assert (native_workspace.parent / "outside" / "secret").read_text() == "fake-outside-canary"


@pytest.mark.parametrize("code", [0, 7, 17, 23, 65, 71, 126, 127])
def test_ordinary_exit_and_forged_stderr(native_command, code):
    """Preserve ordinary exit status despite forged permission or launcher text."""
    result = native_command(f"printf 'Operation not permitted; sandbox_apply' >&2; exit {code}")
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == code
    assert result.stderr == "Operation not permitted; sandbox_apply"


def test_missing_executable_preserves_shell_status(native_command):
    """An absent executable returns the shell's ordinary missing-command exit."""
    result = native_command("loop-command-that-does-not-exist")
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 127


def test_explicit_external_executable_read_grant(native_command, native_workspace):
    """An external executable runs only with its explicitly bound readable root."""
    external = native_workspace.parent / "external"
    external.mkdir()
    tool = external / "tool"
    tool.write_text("#!/bin/sh\nprintf external-ok\n")
    tool.chmod(0o755)
    denied = native_command(shlex.quote(str(tool)))
    assert denied.exit_code != 0
    assert denied.stdout == ""
    approved = native_command(shlex.quote(str(tool)), read_roots=(external,))
    assert approved.exit_code == 0
    assert approved.stdout == "external-ok"


@pytest.mark.parametrize(
    "source,expected",
    [
        ("cat ../outside/secret >/dev/null 2>&1 || true", 0),
        ("printf changed > effect; cat ../outside/secret", 1),
    ],
)
def test_handled_denial_and_partial_effects(native_command, native_workspace, source, expected):
    """Denied reads preserve shell control flow and prior authorized workspace effects."""
    result = native_command(source)
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == expected
    assert "fake-outside-canary" not in result.stdout
    if "effect" in source:
        assert (native_workspace / "effect").read_text() == "changed"


def test_git_grant_preserves_nested_controls(native_command, native_workspace):
    """An explicit top-level Git write grant never unlocks nested Git or application controls."""
    nested = native_workspace / "nested" / ".git"
    nested.mkdir(parents=True)
    (nested / "config").write_text("preserve-nested")
    loop = native_workspace / ".loop"
    loop.mkdir()
    (loop / "control").write_text("preserve-loop")
    writes = (native_workspace, native_workspace / ".git")
    granted = native_command("printf granted > .git/config", write_roots=writes)
    assert granted.exit_code == 0
    assert (native_workspace / ".git" / "config").read_text() == "granted"
    for target in ("nested/.git/config", ".loop/control"):
        result = native_command(f"printf bad > {target}", write_roots=writes)
        assert result.exit_code != 0
    assert (nested / "config").read_text() == "preserve-nested"
    assert (loop / "control").read_text() == "preserve-loop"


def test_network_authority_is_explicit(native_command):
    """A disposable loopback endpoint is reachable only after an explicit egress grant."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        source = "/usr/bin/perl -MSocket -e " + shlex.quote(
            'socket(S, PF_INET, SOCK_STREAM, getprotobyname("tcp")) or die $!; '
            f'connect(S, sockaddr_in({port}, inet_aton("127.0.0.1"))) or die $!; '
            'print "connected";'
        )
        denied = native_command(source)
        assert denied.exit_code != 0
        assert denied.stdout == ""
        approved = native_command(source, network=True)
        assert approved.exit_code == 0
        assert approved.stdout == "connected"


def test_unix_ipc_outside_authority_is_denied(native_command, native_workspace):
    """A native command cannot connect to an outside Unix-domain service."""
    path = native_workspace.parent / "outside" / "service.sock"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(path))
        listener.listen(1)
        source = "/usr/bin/perl -MSocket -e " + shlex.quote(
            "socket(S, PF_UNIX, SOCK_STREAM, 0) or die $!; "
            f'connect(S, sockaddr_un("{path}")) or die $!; print "connected";'
        )
        result = native_command(source)
        assert result.exit_code != 0
        assert result.stdout == ""


@pytest.mark.parametrize(
    "relative",
    [
        ".ssh/key",
        ".aws/credentials",
        ".gnupg/key",
        ".config/key",
        ".loop/control",
        "Library/private",
        "AGENTS.md",
        ".git/config",
        ".agents/skills/SKILL.md",
        "POLICY+.md",
    ],
)
def test_native_parent_rename_cannot_relocate_protected_data(
    native_command, native_workspace, relative
):
    """Prevent protected descendants from escaping through a renamed parent."""
    parent = native_workspace / "protected-parent"
    target = parent / relative
    target.parent.mkdir(parents=True)
    target.write_text("preserve-relocation")
    source = "/usr/bin/perl -e " + shlex.quote(
        'rename "protected-parent", "$ENV{TMPDIR}/moved" or die $!; '
        f'open F, ">", "$ENV{{TMPDIR}}/moved/{relative}" or die $!; print F "changed"'
    )
    result = native_command(source, protected_instruction_names=("POLICY+.md",))
    assert result.exit_code != 0
    assert target.read_text() == "preserve-relocation"


def test_ordinary_directory_rename_remains_available(native_command, native_workspace):
    """Protected ancestor restrictions do not forbid ordinary workspace directory renames."""
    parent = native_workspace / "ordinary-parent"
    parent.mkdir()
    (parent / "data").write_text("ordinary")
    result = native_command("mv ordinary-parent ordinary-moved")
    assert result.exit_code == 0
    assert (native_workspace / "ordinary-moved" / "data").read_text() == "ordinary"


def test_selected_descriptor_helper_cannot_read_outside_files(backend, native_workspace):
    """Give a read-only helper the selected descriptor without outside read authority."""
    helper = native_workspace.parent / "attack-helper"
    helper.write_text('#!/bin/sh\n/bin/cat "$1"\n/bin/cat "$2"\n')
    helper.chmod(0o755)
    selected = native_workspace / "selected.txt"
    selected.write_text("selected-data\n")
    descriptor = os.open(selected, os.O_RDONLY)
    try:
        result = backend.run_read_only_argv(
            helper,
            (f"/dev/fd/{descriptor:05d}", str(native_workspace.parent / "outside" / "secret")),
            (descriptor,),
            time.monotonic() + 5,
        )
    finally:
        os.close(descriptor)
    assert result is not None
    assert result.exit_code != 0
    assert "selected-data" in result.stdout
    assert "fake-outside-canary" not in result.stdout


@pytest.mark.parametrize("path", [".ssh/key", ".aws/credentials", ".gnupg/key"])
def test_extra_read_root_preserves_private_restrictions(native_command, native_workspace, path):
    """Granting an external readable directory does not grant its private credential descendants."""
    extra = native_workspace.parent / "extra"
    target = extra / path
    target.parent.mkdir(parents=True)
    target.write_text("extra-private-canary")
    result = native_command(f"cat ../extra/{path}", read_roots=(extra,), write_roots=())
    assert result.exit_code != 0
    assert "extra-private-canary" not in result.stdout


def test_session_ancestor_metadata_does_not_expose_private_siblings(
    native_command, native_workspace
):
    """Temporary private storage permits exact ancestor traversal without sibling metadata."""
    private = native_workspace / "Library"
    scratch = private / "session" / "files"
    scratch.mkdir(mode=0o700, parents=True)
    sibling = private / "unrelated"
    sibling.write_text("private-sibling-canary")
    alias = native_workspace / "session-alias"
    alias.symlink_to(scratch, target_is_directory=True)
    source = 'printf scratch > "$TMPDIR/report"; cat "$TMPDIR/report"; stat -f %HT ' + shlex.quote(
        str(private)
    )
    changes = {
        "environment": {"PATH": "/usr/bin:/bin", "LANG": "C", "TMPDIR": str(scratch)},
        "aliases": (("$TMPDIR", alias, scratch),),
        "write_roots": (),
    }
    allowed = native_command(source, **changes)
    assert allowed.exit_code == 0, allowed
    assert allowed.stdout == "scratchDirectory\n"
    for command in ("stat -f %HT " + shlex.quote(str(sibling)), "ls " + shlex.quote(str(private))):
        denied = native_command(command, **changes)
        assert denied.exit_code != 0, denied
        assert denied.stdout == ""
    assert sibling.read_text() == "private-sibling-canary"


def test_detached_descendant_retains_old_descriptor_after_generation_retirement(
    backend, native_workspace
):
    """Detached macOS writers remain confined but are not revoked by ownership or retirement."""
    import json
    from unittest.mock import Mock

    from loop import Interaction, SessionManager

    manager = SessionManager(interaction=Mock(spec=Interaction))
    original = manager.temporary_directory.get()
    aliases = native_workspace.parent / "aliases"
    aliases.mkdir(mode=0o700)
    alias = aliases / "session"
    alias.symlink_to(original, target_is_directory=True)
    program = r"""
pipe(R, W) or die;
my $pid = fork(); defined($pid) or die;
if ($pid) { close W; sysread(R, my $ready, 1); exit; }
close R;
POSIX::setsid() >= 0 or die;
my $second = fork(); defined($second) or die;
if ($second) { POSIX::_exit(0); }
open(my $retained, ">", "$ENV{TMPDIR}/late") or die;
syswrite(W, "R"); close W;
close STDIN; close STDOUT; close STDERR;
select(undef, undef, undef, 0.4);
my $escaped = open(my $outside, ">", "$ENV{TMPDIR}/../escape");
print $retained ($escaped ? "escaped" : "confined"); close $retained;
POSIX::_exit(0);
"""
    request = SandboxRequest.create(
        source="/usr/bin/perl -MPOSIX -e " + shlex.quote(program),
        cwd=native_workspace,
        workspace=native_workspace,
        read_roots=(),
        write_roots=(),
        network=False,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C", "TMPDIR": str(original)},
        policy_version=backend.policy_version,
        deadline=time.monotonic() + 5,
        workspace_id="detached-working-generation",
        aliases=(("$TMPDIR", alias, original),),
    )
    result = backend.run(request)
    assert result.outcome is SandboxOutcome.COMPLETED, result
    assert result.exit_code == 0, result
    assert result.stdout == result.stderr == "", json.dumps(result.__dict__)
    with (original / "late").open() as retained:
        manager.discard_working_directory()
        assert manager.temporary_directory.get() != original
        deadline = time.monotonic() + 3
        content = ""
        while time.monotonic() < deadline and not content:
            retained.seek(0)
            content = retained.read()
            if not content:
                time.sleep(0.01)
        assert content == "confined"
    assert not original.exists()
    assert not original.parent.exists()
    assert not (original.parent / "escape").exists()
