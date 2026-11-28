"""Verify bound Linux path inspection, private masks and protected ancestor preparation."""

import errno
import os
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.contracts import SandboxRequest
from loop.execution.sandbox.linux import paths as linux_paths
from loop.execution.sandbox.linux.paths import (
    bind_paths,
    mount_arguments,
    prepare_paths,
)

from ..helpers import logical_private_name

_TEST_NATIVE_SCRATCH = Path("/test-native-scratch")


@pytest.fixture
def bound(monkeypatch, tmp_path):
    """Bind real test paths and a separate simulated inner namespace destination."""
    workspace, scratch = tmp_path / "workspace", tmp_path / "scratch"
    workspace.mkdir()
    scratch.mkdir(mode=0o700)
    monkeypatch.setattr(
        "loop.execution.sandbox.contracts.tempfile.gettempdir", lambda: str(tmp_path)
    )
    monkeypatch.setattr("loop.execution.sandbox.linux.paths.time.monotonic", lambda: 0)
    monkeypatch.setattr(linux_paths, "NATIVE_SCRATCH_PATH", _TEST_NATIVE_SCRATCH)
    monkeypatch.setattr(
        linux_paths,
        "is_protected_external_read",
        lambda path: logical_private_name(path, tmp_path.parent),
    )

    def create(**changes):
        """Bind a fresh request without launching source or inspecting shell effects."""
        values = {
            "source": "printf must-not-run",
            "cwd": workspace,
            "workspace": workspace,
            "read_roots": (),
            "write_roots": (workspace,),
            "network": False,
            "environment": {"PATH": str(workspace), "TMPDIR": str(scratch)},
            "policy_version": "path-test",
            "deadline": 1000,
            "workspace_id": "path-test",
        }
        values.update(changes)
        return SandboxRequest.create(**values)

    return create, workspace, scratch


def _external_executable_request(create, scratch, tools):
    """Create a request whose PATH contains one external executable identity."""
    tools.mkdir()
    executable = tools / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    details = executable.stat()
    request = create(
        environment={"PATH": str(tools), "TMPDIR": str(scratch)},
        executable_identities=(("tool", executable, executable, details.st_dev, details.st_ino),),
    )
    return request, executable


def test_prepare_paths_preserves_sorted_protection_private_masks_and_ancestor_writes(bound):
    """Nested controls and private files produce anchors without changing unrelated bytes."""
    create, workspace, scratch = bound
    protected = workspace / "nested/AGENTS.md"
    protected.parent.mkdir()
    protected.write_text("instructions")
    ordinary = protected.parent / "ordinary"
    ordinary.write_text("unchanged")
    skills = workspace / ".agents/skills"
    skills.mkdir(parents=True)
    (skills / "entry").write_text("protected")
    private = workspace / "nested/.Aws"
    private.write_text("private")
    config = scratch / ".Config"
    config.mkdir()
    (config / "secret").write_text("private")
    custom = workspace / "CUSTOM.md"
    custom.write_text("custom")
    git = workspace / ".git"
    git.mkdir()
    (git / "config").write_text("git")
    result = prepare_paths(create(protected_instruction_names=("CUSTOM.md",)), scratch)
    assert result == (
        tuple(sorted((protected, skills, custom, git))),
        tuple(sorted((private, config))),
        tuple(sorted((protected.parent, skills, skills.parent, git, config))),
        (),
    )
    assert ordinary.read_text() == "unchanged"
    assert not (workspace / "must-not-run").exists()


def test_prepare_paths_uses_only_the_mask_for_a_denied_control_file(bound):
    """A hard-denied control file has no native mount carrying its original data."""
    create, workspace, scratch = bound
    instructions = workspace / "AGENTS.md"
    instructions.write_text("denied-instructions")
    assert prepare_paths(create(hard_denied_paths=(instructions,)), scratch) == (
        (),
        (instructions,),
        (),
        (),
    )


@pytest.mark.parametrize("read_only", [False, True])
def test_prepare_paths_records_read_only_hardlinks_for_granted_roots(bound, tmp_path, read_only):
    """Writable hardlinks get mounts; links within an already read-only root need none."""
    create, workspace, scratch = bound
    external = tmp_path / "external"
    external.mkdir()
    outside = external / "outside"
    outside.write_text("original")
    root = external if read_only else workspace
    os.link(outside, root / "alias")
    request = create(read_roots=(external,), write_roots=() if read_only else (workspace,))
    expected_mounts = () if read_only else (root / "alias",)
    assert prepare_paths(request, scratch) == ((), (), (), expected_mounts)
    assert outside.read_text() == "original"


@pytest.mark.parametrize(
    "case,refusal",
    [
        ("approved", None),
        ("public-storage", None),
        ("owner-only-traversal", None),
        ("several-installations", None),
        ("scoped-sources", None),
        ("separate-home-filesystem", None),
        ("missing-revision", "Unapproved Linux hardlink source"),
        ("wrong-account", "Unapproved Linux hardlink source"),
        ("shared-write-home", "owner-controlled account directory"),
        ("denied-home", "owner-controlled account directory"),
        ("writable-parent", "unsafe account ancestry"),
        ("writable-peer", "names outside the verified scope"),
        ("denied-file", "names outside the verified scope"),
        ("protected-file", "names outside the verified scope"),
        ("protected-storage", "names outside the verified scope"),
        ("third-name", "names outside the verified scope"),
        ("peer-file-mount", "names outside the verified scope"),
        ("workspace-file-mount", "shared dependency has a mounted alias"),
        ("mounted-account-alias", "source has a mounted alias"),
        ("mounted-home-alias", "source has a mounted alias"),
        ("bad-mountinfo", "Ambiguous Linux hardlink source mount authority"),
        ("changed-mountinfo", "source mounts changed"),
    ],
)
def test_account_peers_check_native_policy_without_becoming_a_mount(
    bound, tmp_path, monkeypatch, case, refusal
):
    """A complete approved group supplies one immutable workspace entry; uncertain peers refuse."""
    create, workspace, scratch = bound
    account = tmp_path / "account"
    account.mkdir(mode=0o755)
    storage = account / "storage"
    storage.mkdir(mode=0o700)
    source = storage / ("AGENTS.md" if case == "protected-file" else "package")
    source.write_text("dependency")
    source.chmod(0o644)
    alias = workspace / "dependency"
    alias.hardlink_to(source)
    if case == "third-name":
        (tmp_path / "other").hardlink_to(source)
    alternate = account / "other-installation"
    if case in {"several-installations", "scoped-sources", "protected-storage"}:
        alternate.mkdir(mode=0o755)
        (alternate / "package").hardlink_to(source)
    if case in {"public-storage", "writable-peer"}:
        storage.chmod(0o755)
    if case == "shared-write-home":
        account.chmod(0o770)
    if case == "writable-parent":
        tmp_path.chmod(0o770)
    if case in {"writable-peer", "owner-only-traversal"}:
        source.chmod(0o664)
    denial = {"denied-home": (account,), "denied-file": (source,)}.get(case, ())
    request = create(
        hard_denied_paths=denial,
        protected_host_roots=(alternate,) if case == "protected-storage" else (),
        permission_revision=None if case == "missing-revision" else 2,
    ).with_backend_options((), (storage, alternate) if case == "scoped-sources" else (account,))
    mount = {
        "peer-file-mount": source,
        "workspace-file-mount": alias,
        "mounted-account-alias": account,
        "separate-home-filesystem": account,
        "mounted-home-alias": tmp_path,
    }.get(case, Path("/"))
    mount_root = "/subtree" if case in {"mounted-home-alias", "mounted-account-alias"} else "/"
    fields = ["1", "0", "1:0", mount_root, str(mount), "rw", "-", "ext4", "device", "rw"]
    first = [[]] if case == "bad-mountinfo" else [fields]
    changed = [[*fields[:4], str(workspace / "late-mount"), *fields[5:]]]
    mountinfo = Mock(side_effect=[first, changed if case == "changed-mountinfo" else first])
    monkeypatch.setattr(linux_paths, "read_mountinfo", mountinfo)
    resolved = tmp_path / "another-account" if case == "wrong-account" else account
    monkeypatch.setattr(linux_paths, "hardlink_source_paths", lambda *_: (resolved,))
    if refusal is None:
        assert prepare_paths(request, scratch) == ((), (), (), (alias,))
    else:
        with pytest.raises(ValueError, match=refusal):
            prepare_paths(request, scratch)
    assert source.stat().st_nlink == (
        3
        if case in {"several-installations", "scoped-sources", "protected-storage", "third-name"}
        else 2
    )


@pytest.mark.parametrize("concealed", [False, True])
def test_cache_is_unused_when_all_visible_links_are_accounted_for(
    bound, tmp_path, monkeypatch, concealed
):
    """Ordinary and masked files work without inspecting an unrelated unsafe cache."""
    create, workspace, scratch = bound
    cache = tmp_path / ".cache"
    cache.mkdir()
    cache.chmod(0o777)
    directory = workspace / "retired"
    directory.mkdir()
    (directory / "first").write_text("data")
    peer = (tmp_path if concealed else directory) / "second"
    peer.hardlink_to(directory / "first")
    request = create(
        hard_denied_paths=(directory,) if concealed else (),
        permission_revision=1,
    ).with_backend_options((), (cache,))
    mountinfo = Mock(side_effect=AssertionError("Unexpected cache inspection."))
    monkeypatch.setattr(linux_paths, "read_mountinfo", mountinfo)
    monkeypatch.setattr(linux_paths, "hardlink_source_paths", lambda *_: (cache,))

    _, private, _, immutable = prepare_paths(request, scratch)
    assert (directory in private) is concealed
    assert (directory / "first" in immutable) is not concealed
    mountinfo.assert_not_called()


@pytest.mark.parametrize("authority", ["read", "write", "workspace", "descendant"])
def test_runtime_duplicate_does_not_reclassify_system_hardlinks_as_data(
    bound, tmp_path, monkeypatch, authority
):
    """Only exact authenticated read-only runtime duplicates avoid the data alias census."""
    create, _, scratch = bound
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    selected = runtime / "child" if authority == "descendant" else runtime
    selected.mkdir(exist_ok=True)
    outside = tmp_path / "outside"
    outside.write_text("system-tool")
    (selected / "tool").hardlink_to(outside)
    changes = {"read_roots": (selected,)}
    if authority == "write":
        changes["write_roots"] = (selected,)
    elif authority == "workspace":
        changes.update(workspace=selected, cwd=selected, write_roots=())
    request = create(**changes)
    with pytest.raises(ValueError, match="Incomplete hardlink aliases"):
        prepare_paths(request, scratch)
    if authority == "read":
        assert prepare_paths(request, scratch, runtime_roots=(runtime,)) == ((), (), (), ())
    else:
        with pytest.raises(ValueError, match="Incomplete hardlink aliases"):
            prepare_paths(request, scratch, runtime_roots=(runtime,))


@pytest.mark.parametrize("source", ["instruction", "private"])
def test_prepare_paths_rejects_hardlinks_to_protected_or_private_files(bound, source):
    """A hardlink cannot expose protected instructions or private workspace data by alias."""
    create, workspace, scratch = bound
    protected = workspace / "AGENTS.md" if source == "instruction" else workspace / ".ssh"
    protected.write_text("restricted")
    os.link(protected, workspace / "alias")
    with pytest.raises(ValueError, match="hardlinked file aliases protected or private data"):
        prepare_paths(create(), scratch)


@pytest.mark.parametrize("name", ["AGENTS.md", ".git", ".ssh", ".loop/alias"])
def test_prepare_paths_rejects_protected_symlinks_without_following_them(bound, tmp_path, name):
    """Protected and private aliases, including descendants, fail before mount preparation."""
    create, workspace, scratch = bound
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = workspace / name
    alias.parent.mkdir(exist_ok=True)
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="protected or private entry is a symlink"):
        prepare_paths(create(), scratch)


def test_prepare_paths_skips_ordinary_symlinks_and_preserves_bound_read_aliases(bound, tmp_path):
    """Ordinary links are not traversed; shared approved alias identities remain required."""
    create, workspace, scratch = bound
    external = tmp_path / "external"
    external.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "AGENTS.md"
    secret.write_text("outside")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    alias = workspace / "approved"
    alias.symlink_to(external, target_is_directory=True)
    request = create(read_roots=(external,), read_aliases=(alias,))
    assert prepare_paths(request, scratch) == ((), (), (), ())
    alias.unlink()
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="approved path authority changed"):
        prepare_paths(request, scratch)
    with (
        pytest.raises(ValueError, match="alias target is outside the mounted view"),
        mount_arguments(request, scratch),
    ):
        pytest.fail("A retargeted alias must not reach mount construction.")


@pytest.mark.parametrize("git_create", [False, True])
def test_prepare_paths_limits_git_authority_to_top_level_metadata(bound, git_create):
    """Git write/create grants keep nested Git, instructions and controls protected."""
    create, workspace, scratch = bound
    git = workspace / ".git"
    if git_create:
        request = create(git_create=True)
        assert prepare_paths(request, scratch) == ((), (), (), ())
        return
    nested = git / "nested/.git"
    nested.mkdir(parents=True)
    (nested / "config").write_text("nested")
    (git / "config").write_text("writable")
    instruction = git / "AGENTS.md"
    instruction.write_text("protected")
    request = create(write_roots=(workspace, git))
    protected, private, anchors, hardlinked = prepare_paths(request, scratch)
    assert protected == tuple(sorted((nested, instruction)))
    assert private == ()
    assert anchors == tuple(sorted((git, nested.parent, nested)))
    assert hardlinked == ()


@pytest.mark.parametrize("unsafe", ["binding", "mode", "owner", "type"])
def test_prepare_paths_refuses_unsafe_scratch_before_scanning(bound, monkeypatch, tmp_path, unsafe):
    """Only the bound ordinary-owned private directory may supply command scratch."""
    create, _, scratch = bound
    request = create()
    if unsafe == "binding":
        request = replace(
            request,
            environment=tuple({**request.shell_environment(), "TMPDIR": "/unbound"}.items()),
        )
    elif unsafe == "mode":
        scratch.chmod(0o755)
    elif unsafe == "owner":
        monkeypatch.setattr("loop.execution.sandbox.linux.paths.os.getuid", lambda: -1)
    else:
        scratch = tmp_path / "file-scratch"
        scratch.write_text("not a directory")
        request = replace(
            request,
            environment=tuple({**request.shell_environment(), "TMPDIR": str(scratch)}.items()),
        )
    with pytest.raises(ValueError, match="scratch is not a bound private directory"):
        prepare_paths(request, scratch)


@pytest.mark.parametrize("stale_after", [False, True])
def test_prepare_paths_rechecks_shared_authority_before_and_after_inspection(
    bound, monkeypatch, stale_after
):
    """A stale shared request never becomes a mount preparation result."""
    create, _, scratch = bound
    request = create()
    responses = iter((True, False) if stale_after else (False,))
    monkeypatch.setattr(SandboxRequest, "paths_are_current", lambda self: next(responses))
    with pytest.raises(ValueError, match="approved path authority changed"):
        prepare_paths(request, scratch)


@pytest.mark.parametrize("phase", ["initial", "directory", "entry", "recheck", "final"])
def test_prepare_paths_enforces_the_deadline_at_every_scan_boundary(bound, monkeypatch, phase):
    """Expired preparation never returns mount authority, including empty or wide directories."""
    create, workspace, scratch = bound
    if phase == "entry":
        (workspace / "input").write_text("input")
    request = create()
    expiry_call = {"initial": 1, "directory": 2, "entry": 3, "recheck": 4, "final": 6}[phase]
    calls = 0

    def clock():
        """Expire exactly at the selected observable scan boundary."""
        nonlocal calls
        calls += 1
        return request.deadline if calls == expiry_call else 0

    monkeypatch.setattr("loop.execution.sandbox.linux.paths.time.monotonic", clock)
    with pytest.raises(TimeoutError, match="deadline expired|exceeded the command deadline"):
        prepare_paths(request, scratch)


@pytest.mark.parametrize("unsafe", ["file", "symlink", "private"])
def test_prepare_paths_rejects_unsafe_grant_roots(bound, tmp_path, unsafe):
    """Root validation refuses malformed direct callers and canonical private data grants."""
    create, _, scratch = bound
    root = tmp_path / (".ssh" if unsafe == "private" else "outside")
    if unsafe == "file":
        root.write_text("not a directory")
    else:
        root.mkdir()
    if unsafe == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    request = replace(create(), read_roots=(root,))
    diagnostic = "includes private data" if unsafe == "private" else "not a real directory"
    with pytest.raises(ValueError, match=diagnostic):
        prepare_paths(request, scratch)


def test_prepare_paths_refuses_unsafe_private_entry_types(bound):
    """A private FIFO cannot be passed as a regular-file or directory mask target."""
    create, workspace, scratch = bound
    os.mkfifo(workspace / ".ssh")
    with pytest.raises(ValueError, match="private entry has an unsafe type"):
        prepare_paths(create(), scratch)


@pytest.mark.parametrize("change", ["descriptor", "pathname", "root"])
def test_prepare_paths_rejects_directory_drift_and_closes_scan_descriptors(
    bound, monkeypatch, change
):
    """Changed directory metadata or final root identity fails with every scan FD closed."""
    create, workspace, scratch = bound
    request = create()
    original_fstat = os.fstat
    original_lstat = Path.lstat
    original_close = os.close
    opened, closed = set(), set()
    calls = 0

    def fstat(descriptor):
        """Supply one changed descriptor identity at the final directory check."""
        nonlocal calls
        calls += 1
        opened.add(descriptor)
        details = original_fstat(descriptor)
        if change == "descriptor" and calls == 2:
            values = list(details)
            values[1] += 1
            return os.stat_result(values)
        return details

    root_calls = 0

    def lstat(path, *args, **kwargs):
        """Model pathname drift only at a selected post-scan identity comparison."""
        nonlocal root_calls
        details = original_lstat(path, *args, **kwargs)
        if path == workspace and opened:
            root_calls += 1
            selected = 1 if change == "pathname" else 3
            if change != "descriptor" and root_calls == selected:
                workspace.rename(workspace.with_name("old-workspace"))
                workspace.mkdir()
                details = original_lstat(path, *args, **kwargs)
        return details

    def close(descriptor):
        """Observe closure at the public OS descriptor boundary."""
        closed.add(descriptor)
        original_close(descriptor)

    monkeypatch.setattr("loop.execution.sandbox.linux.paths.os.fstat", fstat)
    monkeypatch.setattr("loop.execution.sandbox.linux.paths.os.close", close)
    monkeypatch.setattr(Path, "lstat", lstat)
    diagnostic = {
        "root": "path root changed",
        "descriptor": "Alias inventory directory changed",
        "pathname": "Alias inventory directory changed",
    }[change]
    with pytest.raises(ValueError, match=diagnostic):
        prepare_paths(request, scratch)
    assert opened == closed


def test_prepare_paths_rechecks_earlier_directories_after_the_complete_walk(bound, monkeypatch):
    """A protected ancestor replaced after its own scan cannot become a returned mount source."""
    create, workspace, scratch = bound
    nested = workspace / "nested"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("protected")
    request = create()
    original = Path.lstat
    observations = 0

    def lstat(path, *args, **kwargs):
        """Replace the prior ancestor at the final whole-tree identity check."""
        nonlocal observations
        if path == nested:
            observations += 1
            if observations == 2:
                nested.rename(workspace / "old-nested")
                nested.mkdir()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="Alias inventory entry changed"):
        prepare_paths(request, scratch)


def test_prepare_paths_classifies_scratch_again_after_live_membership_change(bound, monkeypatch):
    """A permitted scratch retry also refreshes its protected and private classification."""
    create, _, scratch = bound
    directory = scratch.stat()
    original = os.scandir
    first = True

    @contextmanager
    def scandir(descriptor):
        """Publish a new scratch policy entry just after its initial scan."""
        nonlocal first
        with original(descriptor) as entries:
            yield entries
        if (
            isinstance(descriptor, int)
            and os.path.samestat(os.fstat(descriptor), directory)
            and first
        ):
            first = False
            (scratch / ".aws").mkdir()

    monkeypatch.setattr(os, "scandir", scandir)
    assert prepare_paths(create(), scratch) == ((), (scratch / ".aws",), (scratch / ".aws",), ())


@pytest.fixture
def binding(monkeypatch, bound):
    """Observe portable descriptor ownership while modeling the Linux O_PATH capability."""
    monkeypatch.setattr(os, "O_PATH", 0, raising=False)
    original_open = os.open
    descriptors = []

    def open_path(path, flags, *args, **kwargs):
        """Record mount descriptors separately from directory scanning descriptors."""
        descriptor = original_open(path, flags, *args, **kwargs)
        if not flags & os.O_DIRECTORY:
            descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", open_path)
    return *bound, descriptors


@pytest.mark.parametrize("caller_failure", [False, True])
def test_bind_paths_pins_complete_sources_and_closes_them_on_exit(
    binding, tmp_path, caller_failure
):
    """All live roots and protection targets are pinned once and owned only by the context."""
    create, workspace, scratch, opened = binding
    protected = workspace / "nested/AGENTS.md"
    protected.parent.mkdir()
    protected.write_text("instructions")
    private_file = workspace / ".ssh"
    private_file.write_text("credential")
    private_directory = scratch / ".aws"
    private_directory.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    request = create(read_roots=(workspace, external))
    expected = {
        workspace,
        scratch,
        external,
        protected,
        protected.parent,
        private_file,
        private_directory,
    }

    def inspect():
        """Require live inode matches and simulate an optional trusted caller failure."""
        with bind_paths(request, scratch) as (inventories, descriptors):
            assert inventories == (
                (protected,),
                tuple(sorted((private_file, private_directory))),
                tuple(sorted((protected.parent, private_directory))),
                (),
            )
            assert set(descriptors) == expected
            assert len(set(descriptors.values())) == len(expected) == len(opened)
            for path, descriptor in descriptors.items():
                assert os.fstat(descriptor).st_ino == path.lstat().st_ino
                assert not os.get_inheritable(descriptor)
            if caller_failure:
                raise RuntimeError("caller failed")

    if caller_failure:
        with pytest.raises(RuntimeError, match="caller failed"):
            inspect()
    else:
        inspect()
    for descriptor in opened:
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF
    assert private_file.read_text() == "credential"


def test_bind_paths_refuses_missing_linux_descriptor_support(bound, monkeypatch):
    """Unsupported platforms fail before path preparation rather than using pathname binds."""
    create, _, scratch = bound
    monkeypatch.delattr(os, "O_PATH", raising=False)
    with (
        pytest.raises(OSError, match="requires O_PATH") as error,
        bind_paths(create(), scratch),
    ):
        pytest.fail("unsupported platform yielded mount sources")
    assert error.value.errno == errno.ENOTSUP


@pytest.mark.parametrize("unsafe", ["fifo", "symlink", "hardlink", "ancestor_symlink"])
def test_bind_paths_refuses_unsafe_post_scan_sources(binding, monkeypatch, tmp_path, unsafe):
    """Post-scan special files, linked files and noncanonical source paths fail with FDs closed."""
    create, workspace, scratch, opened = binding
    protected = workspace / "nested/AGENTS.md"
    protected.parent.mkdir()
    protected.write_text("instructions")
    outside = tmp_path / "outside"
    outside.write_text("outside")
    request = create()
    original_open = os.open

    def open_path(path, flags, *args, **kwargs):
        """Change the later protected source after the scan and initial root bindings."""
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == scratch and not flags & os.O_DIRECTORY:
            if unsafe == "ancestor_symlink":
                protected.parent.rename(workspace / "old-nested")
                protected.parent.symlink_to(workspace / "old-nested", target_is_directory=True)
            else:
                protected.unlink()
                if unsafe == "fifo":
                    os.mkfifo(protected)
                elif unsafe == "symlink":
                    protected.symlink_to(outside)
                else:
                    os.link(outside, protected)
        return descriptor

    monkeypatch.setattr(os, "open", open_path)
    with (
        pytest.raises(ValueError, match="mount source is not a real directory or single-link file"),
        bind_paths(request, scratch),
    ):
        pytest.fail("unsafe source yielded a binding")
    for descriptor in opened:
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("change", ["opening", "pathname", "descriptor", "open_failure"])
def test_bind_paths_refuses_source_drift_and_open_errors(binding, monkeypatch, change):
    """Changed descriptors/pathnames or a failed later open never leak prior mount descriptors."""
    create, workspace, scratch, opened = binding
    request = create()
    original_fstat, original_lstat, original_open = os.fstat, Path.lstat, os.open
    observations = 0

    def fstat(descriptor):
        """Change identity at opening or at the whole-binding descriptor recheck."""
        nonlocal observations
        details = original_fstat(descriptor)
        if descriptor in opened:
            observations += 1
            selected = 1 if change == "opening" else 3
            if change in ("opening", "descriptor") and observations == selected:
                fields = {
                    name: getattr(details, name)
                    for name in (
                        "st_dev",
                        "st_ino",
                        "st_mode",
                        "st_uid",
                        "st_gid",
                        "st_nlink",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                }
                return SimpleNamespace(**{**fields, "st_ino": details.st_ino + 1})
        return details

    def lstat(path, *args, **kwargs):
        """Replace a previously bound pathname at the final source recheck."""
        if change == "pathname" and path == workspace and len(opened) == 2:
            workspace.rename(workspace.with_name("old-workspace"))
            workspace.mkdir()
        return original_lstat(path, *args, **kwargs)

    def open_path(path, flags, *args, **kwargs):
        """Fail a later mount open after a prior descriptor was acquired."""
        if change == "open_failure" and path == scratch and not flags & os.O_DIRECTORY:
            raise PermissionError(errno.EACCES, "controlled source open denial")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "fstat", fstat)
    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(os, "open", open_path)
    if change == "open_failure":
        expected, diagnostic = PermissionError, "controlled source open denial"
    else:
        expected = ValueError
        diagnostic = (
            "source changed during binding" if change == "opening" else "bound mount source changed"
        )
    with pytest.raises(expected, match=diagnostic), bind_paths(request, scratch):
        pytest.fail("changed source yielded a binding")
    assert opened
    for descriptor in opened:
        with pytest.raises(OSError) as error:
            original_fstat(descriptor)
        assert error.value.errno == errno.EBADF


def test_bind_paths_rechecks_shared_request_after_descriptor_binding(binding, monkeypatch):
    """Authority changes after scanning still refuse entry and close every pinned root."""
    create, _, scratch, opened = binding
    request = create()
    responses = iter((True, True, False))
    monkeypatch.setattr(SandboxRequest, "paths_are_current", lambda self: next(responses))
    with (
        pytest.raises(ValueError, match="approved path authority changed"),
        bind_paths(request, scratch),
    ):
        pytest.fail("stale request yielded a binding")
    for descriptor in opened:
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("phase", ["source", "recheck", "yield"])
def test_bind_paths_enforces_deadline_during_binding_and_before_yield(binding, monkeypatch, phase):
    """Binding expiry closes held descriptors and never hands preparation to a caller."""
    create, _, scratch, opened = binding
    request = create()
    calls = 0

    def clock():
        """Expire after path scanning at a selected binding boundary."""
        nonlocal calls
        calls += 1
        return request.deadline if calls == {"source": 10, "recheck": 11, "yield": 13}[phase] else 0

    monkeypatch.setattr("loop.execution.sandbox.linux.paths.time.monotonic", clock)
    with (
        pytest.raises(TimeoutError, match="exceeded the command deadline"),
        bind_paths(request, scratch),
    ):
        pytest.fail("expired preparation yielded a binding")
    assert opened
    for descriptor in opened:
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("read_only", [False, True])
def test_mount_arguments_preserves_grant_roles_protection_and_private_masks(binding, read_only):
    """Preserve sibling writes, apply protections last and inherit no credentials."""
    create, workspace, scratch, opened = binding
    instruction = workspace / "nested/AGENTS.md"
    instruction.parent.mkdir()
    instruction.write_text("instructions")
    private = workspace / ".ssh"
    private.mkdir()
    (private / "AGENTS.md").write_text("private instructions")
    request = create(write_roots=() if read_only else (workspace,))
    with mount_arguments(request, scratch) as (arguments, inherited, _):
        operation = "--ro-bind-fd" if read_only else "--bind-fd"
        workspace_index = arguments.index(str(workspace))
        assert arguments[workspace_index - 2] == operation
        workspace_fd = int(arguments[workspace_index - 1])
        nested_index = arguments.index(str(instruction.parent))
        assert arguments[nested_index - 2] == operation
        protected_index = arguments.index(str(instruction))
        assert protected_index > nested_index
        assert arguments[protected_index - 2] == "--ro-bind-fd"
        assert arguments[-5:-3] == ("--dir", str(_TEST_NATIVE_SCRATCH))
        assert arguments[-3] == "--bind-fd"
        assert arguments[-1] == str(_TEST_NATIVE_SCRATCH)
        assert int(arguments[-2]) in inherited
        assert str(private / "AGENTS.md") not in arguments
        assert set(inherited) < set(opened)
        assert workspace_fd in inherited
        assert all(os.fstat(descriptor).st_ino != private.stat().st_ino for descriptor in inherited)
    for descriptor in opened:
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF
    assert (private / "AGENTS.md").read_text() == "private instructions"


def test_mount_arguments_binds_only_a_private_session_temporary_subdirectory(binding, monkeypatch):
    """The conventional temporary path receives a pinned session-owned subtree."""
    create, _, scratch, _ = binding
    destination = Path("/unit-conventional-temp")
    monkeypatch.setattr(linux_paths, "NATIVE_CONVENTIONAL_TMP_PATH", destination, raising=False)
    request = create(scratch_root=scratch, scratch_data=True)

    with mount_arguments(request, scratch) as (arguments, inherited, _):
        index = len(arguments) - 1 - arguments[::-1].index(str(destination))
        assert arguments[index - 2] == "--bind-fd"
        descriptor = int(arguments[index - 1])
        assert descriptor in inherited
        backing = scratch / linux_paths.SESSION_CONVENTIONAL_TMP_NAME
        assert backing.stat().st_ino == os.fstat(descriptor).st_ino
        assert backing.stat().st_mode & 0o777 == 0o700


def test_conventional_temporary_mount_rejects_existing_hardlinks(binding, monkeypatch, tmp_path):
    """The conventional alias cannot bypass the session's shared-inode restriction."""
    create, _, scratch, _ = binding
    destination = Path("/unit-conventional-temp")
    monkeypatch.setattr(linux_paths, "NATIVE_CONVENTIONAL_TMP_PATH", destination)
    subtree = scratch / linux_paths.SESSION_CONVENTIONAL_TMP_NAME
    subtree.mkdir(mode=0o700)
    peers = tmp_path / "peers"
    peers.mkdir()
    outside = peers / "outside"
    outside.write_text("outside-data")
    os.link(outside, subtree / "link")
    request = create(scratch_root=scratch, scratch_data=True)

    with (
        pytest.raises(ValueError, match="Session scratch contains a shared inode"),
        mount_arguments(request, scratch),
    ):
        pytest.fail("A shared inode became accessible through the conventional alias")


def test_conventional_temporary_refuses_substituted_scratch_before_creation(binding, monkeypatch):
    """A scratch descriptor that no longer matches the grant is never mutated."""
    create, _, scratch, _ = binding
    monkeypatch.setattr(
        linux_paths, "NATIVE_CONVENTIONAL_TMP_PATH", Path("/unit-conventional-temp")
    )
    monkeypatch.setattr(SandboxRequest, "directory_matches_descriptor", lambda *_: False)
    request = create(scratch_root=scratch, scratch_data=True)
    with (
        pytest.raises(ValueError, match="changed before temporary preparation"),
        mount_arguments(request, scratch),
    ):
        pytest.fail("An unbound scratch descriptor was used")
    assert not (scratch / linux_paths.SESSION_CONVENTIONAL_TMP_NAME).exists()


@pytest.mark.parametrize("error", [False, True])
def test_conventional_temporary_checks_acl_read_outcomes(binding, monkeypatch, error):
    """An empty ACL is accepted while unexpected inspection errors fail closed."""
    create, _, scratch, _ = binding
    monkeypatch.setattr(
        linux_paths, "NATIVE_CONVENTIONAL_TMP_PATH", Path("/unit-conventional-temp")
    )
    request = create(scratch_root=scratch, scratch_data=True)
    original = os.getxattr

    def getxattr(path, name, *args, **kwargs):
        """Return the selected outcome only for the temporary subtree."""
        subtree = scratch / linux_paths.SESSION_CONVENTIONAL_TMP_NAME
        if (
            isinstance(path, int)
            and subtree.exists()
            and os.fstat(path).st_ino == subtree.stat().st_ino
        ):
            if error:
                raise OSError(errno.EIO, "ACL inspection failed")
            return b""
        return original(path, name, *args, **kwargs)

    monkeypatch.setattr(os, "getxattr", getxattr)
    expected = pytest.raises(OSError, match="ACL inspection failed") if error else nullcontext()
    with expected, mount_arguments(request, scratch):
        assert not error


def test_conventional_temporary_rechecks_source_after_scanning(binding, monkeypatch):
    """A changed temporary subtree is refused before any namespace arguments are yielded."""
    from contextlib import contextmanager

    create, _, scratch, _ = binding
    monkeypatch.setattr(
        linux_paths, "NATIVE_CONVENTIONAL_TMP_PATH", Path("/unit-conventional-temp")
    )
    original = linux_paths.bind_paths

    @contextmanager
    def changed(*args, **kwargs):
        """Substitute the selected inode after the real binding scan."""
        with original(*args, **kwargs) as result:
            subtree = scratch / linux_paths.SESSION_CONVENTIONAL_TMP_NAME
            subtree.rename(scratch / "replaced")
            subtree.mkdir(mode=0o700)
            yield result

    monkeypatch.setattr(linux_paths, "bind_paths", changed)
    with (
        pytest.raises(ValueError, match="temporary subtree changed during binding"),
        mount_arguments(create(scratch_root=scratch, scratch_data=True), scratch),
    ):
        pytest.fail("A changed temporary subtree was mounted")


def test_conventional_temporary_rechecks_source_after_acl_inspection(binding, monkeypatch):
    """Metadata changed during ACL inspection cannot become a private mount source."""
    create, _, scratch, _ = binding
    monkeypatch.setattr(
        linux_paths, "NATIVE_CONVENTIONAL_TMP_PATH", Path("/unit-conventional-temp")
    )
    subtree = scratch / linux_paths.SESSION_CONVENTIONAL_TMP_NAME

    def change_acl_target(_path, _name):
        """Change the inode authority while ACLs are being read."""
        subtree.chmod(0o750)
        return b""

    monkeypatch.setattr(os, "getxattr", change_acl_target)
    with (
        pytest.raises(ValueError, match="temporary subtree changed during inspection"),
        mount_arguments(create(scratch_root=scratch, scratch_data=True), scratch),
    ):
        pytest.fail("A changed temporary subtree was accepted")


def test_mount_arguments_rejects_stale_authority_before_temporary_creation(binding, monkeypatch):
    """Refuse stale grants before creating any temporary subtree."""
    create, _, scratch, _ = binding
    monkeypatch.setattr(SandboxRequest, "paths_are_current", lambda self: False)
    request = create(scratch_root=scratch, scratch_data=True)
    with (
        pytest.raises(ValueError, match="approved path authority changed"),
        mount_arguments(request, scratch),
    ):
        pytest.fail("Stale authority produced mount arguments")
    assert not (scratch / linux_paths.SESSION_CONVENTIONAL_TMP_NAME).exists()


@pytest.mark.parametrize("restricted", [False, True])
def test_conventional_temporary_mount_is_omitted_when_authority_overlaps(
    binding, monkeypatch, restricted
):
    """Restricted scratch or an existing grant at the destination keeps the mount unavailable."""
    create, workspace, scratch, _ = binding
    destination = Path("/unit-conventional-temp") if restricted else workspace
    monkeypatch.setattr(linux_paths, "NATIVE_CONVENTIONAL_TMP_PATH", destination)
    request = create(scratch_root=scratch, scratch_data=not restricted)

    assert not linux_paths.conventional_tmp_available(request, scratch)
    with mount_arguments(request, scratch):
        assert not (scratch / linux_paths.SESSION_CONVENTIONAL_TMP_NAME).exists()


def test_conventional_temporary_respects_protected_runtime_paths(binding, monkeypatch):
    """A bound runtime keeps its existing namespace view."""
    create, _, scratch, _ = binding
    destination = Path("/unit-conventional-temp")
    monkeypatch.setattr(linux_paths, "NATIVE_CONVENTIONAL_TMP_PATH", destination)
    roots = (destination / "selected-runtime",)
    request = create(scratch_root=scratch, scratch_data=True)
    assert not linux_paths.conventional_tmp_available(request, scratch, runtime_roots=roots)
    with mount_arguments(request, scratch, runtime_roots=roots):
        assert not (scratch / linux_paths.SESSION_CONVENTIONAL_TMP_NAME).exists()


@pytest.mark.parametrize("kind", ["symlink", "file", "shared", "access-acl", "default-acl"])
def test_conventional_temporary_subtree_refuses_unsafe_existing_authority(
    binding, monkeypatch, tmp_path, kind
):
    """A substituted, nonprivate or ACL-bearing subtree cannot become a mount source."""
    create, _, scratch, _ = binding
    monkeypatch.setattr(
        linux_paths, "NATIVE_CONVENTIONAL_TMP_PATH", Path("/unit-conventional-temp")
    )
    subtree = scratch / linux_paths.SESSION_CONVENTIONAL_TMP_NAME
    if kind == "symlink":
        subtree.symlink_to(tmp_path, target_is_directory=True)
    elif kind == "file":
        subtree.write_text("preserve", encoding="utf-8")
    else:
        subtree.mkdir(mode=0o700)
        if kind == "shared":
            subtree.chmod(0o755)
        if kind.endswith("-acl"):
            acl_name = "system.posix_acl_" + ("access" if kind == "access-acl" else "default")
            original = os.getxattr

            def getxattr(path, name, *args, **kwargs):
                """Expose only the selected synthetic ACL on the owned subtree."""
                if (
                    isinstance(path, int)
                    and os.fstat(path).st_ino == subtree.stat().st_ino
                    and name == acl_name
                ):
                    return b"extended-acl"
                return original(path, name, *args, **kwargs)

            monkeypatch.setattr(os, "getxattr", getxattr)
    request = create(scratch_root=scratch, scratch_data=True)
    if kind in {"symlink", "file"}:
        with pytest.raises(OSError) as error, mount_arguments(request, scratch):
            pytest.fail("An unsafe temporary subtree was mounted")
        assert error.value.errno in (errno.ELOOP, errno.ENOTDIR)
    else:
        expected = "extended access list" if kind.endswith("-acl") else "private real directory"
        with pytest.raises(ValueError, match=expected), mount_arguments(request, scratch):
            pytest.fail("An unsafe temporary subtree was mounted")


def test_mount_arguments_binds_hardlinks_read_only_after_writable_roots(binding, tmp_path):
    """A hardlinked leaf is readable but read-only inside an otherwise writable workspace."""
    create, workspace, scratch, _ = binding
    peers = tmp_path / "peers"
    peers.mkdir()
    outside = peers / "outside"
    outside.write_text("outside-data")
    alias = workspace / "alias"
    os.link(outside, alias)
    request = create(read_roots=(peers,))
    with mount_arguments(request, scratch) as (arguments, inherited, _):
        workspace_index = arguments.index(str(workspace))
        alias_index = arguments.index(str(alias))
        assert arguments[workspace_index - 2] == "--bind-fd"
        assert arguments[alias_index - 2] == "--ro-bind-fd"
        descriptor = int(arguments[alias_index - 1])
        assert descriptor in inherited
        assert os.fstat(descriptor).st_ino == alias.stat().st_ino
    assert alias.read_text() == "outside-data"
    assert outside.read_text() == "outside-data"


def test_large_hardlink_sets_use_containing_directory_mounts(binding, tmp_path, monkeypatch):
    """Group dense hardlink sets to stay within Bubblewrap's argument limit."""
    create, workspace, scratch, _ = binding
    peers = tmp_path / "peers"
    peers.mkdir()
    monkeypatch.setattr("loop.execution.sandbox.linux.paths._MAX_HARDLINK_MOUNTS", 2)
    package = workspace / ".venv/site-packages/package"
    package.mkdir(parents=True)
    for index in range(3):
        outside = peers / f"outside-{index}"
        outside.write_text(f"module-{index}")
        os.link(outside, package / f"module-{index}.py")
    nested = package / "data"
    nested.mkdir()
    outside = peers / "outside-nested"
    outside.write_text("nested-module")
    os.link(outside, nested / "module.py")
    request = create(read_roots=(peers,))
    with mount_arguments(request, scratch) as (arguments, inherited, _):
        workspace_index = arguments.index(str(workspace))
        package_index = arguments.index(str(package))
        assert arguments[workspace_index - 2] == "--bind-fd"
        assert arguments[package_index - 2] == "--ro-bind-fd"
        assert int(arguments[package_index - 1]) in inherited
        assert str(package / "module-0.py") not in arguments


def test_large_hardlink_sets_group_at_a_shared_ancestor_when_needed(bound, tmp_path, monkeypatch):
    """Escalate grouping only enough to keep scattered hardlink mounts within their bound."""
    create, workspace, scratch = bound
    peers = tmp_path / "peers"
    peers.mkdir()
    monkeypatch.setattr("loop.execution.sandbox.linux.paths._MAX_HARDLINK_MOUNTS", 2)
    site_packages = workspace / ".venv/site-packages"
    for index in range(3):
        package = site_packages / f"package-{index}"
        package.mkdir(parents=True)
        outside = peers / f"outside-{index}"
        outside.write_text(f"module-{index}")
        os.link(outside, package / "module.py")
    assert prepare_paths(create(read_roots=(peers,)), scratch)[3] == (site_packages,)


def test_scattered_hardlink_grouping_stops_when_its_deadline_expires(bound, tmp_path, monkeypatch):
    """Expiry while classifying scattered links cannot return late mount authority."""
    create, workspace, scratch = bound
    peers = tmp_path / "peers"
    peers.mkdir()
    monkeypatch.setattr("loop.execution.sandbox.linux.paths._MAX_HARDLINK_MOUNTS", 2)
    links = set()
    for index in range(3):
        directory = workspace / "group" / str(index)
        directory.mkdir(parents=True)
        outside = peers / f"outside-{index}"
        outside.write_text("canary")
        link = directory / "linked"
        os.link(outside, link)
        links.add(link)
    request = create(read_roots=(peers,))
    clock = {"now": 0}
    original = Path.is_relative_to

    def relative(path, other):
        """Let the preparation budget expire when a linked leaf is classified."""
        result = original(path, other)
        if path in links:
            clock["now"] = request.deadline
        return result

    monkeypatch.setattr(Path, "is_relative_to", relative)
    monkeypatch.setattr("loop.execution.sandbox.linux.paths.time.monotonic", lambda: clock["now"])
    with pytest.raises(TimeoutError, match="deadline expired|exceeded the command deadline"):
        prepare_paths(request, scratch)


def test_hardlink_grouping_preserves_nested_writable_roots(bound, tmp_path, monkeypatch):
    """A group containing another write root cannot silently make that root read-only."""
    create, workspace, scratch = bound
    peers = tmp_path / "peers"
    peers.mkdir()
    monkeypatch.setattr("loop.execution.sandbox.linux.paths._MAX_HARDLINK_MOUNTS", 2)
    group = workspace / "group"
    nested = group / "writable"
    nested.mkdir(parents=True)
    for index in range(3):
        directory = group / str(index)
        directory.mkdir()
        outside = peers / f"outside-{index}"
        outside.write_text("canary")
        os.link(outside, directory / "linked")
    value = create(write_roots=(workspace, nested), read_roots=(group, peers))
    with pytest.raises(ValueError, match="exceed the bounded mount limit"):
        prepare_paths(value, scratch)


def test_hardlink_grouping_updates_disjoint_ancestor_membership(bound, tmp_path, monkeypatch):
    """Several narrow groups meet the cap without making their shared parent read-only."""
    create, workspace, scratch = bound
    peers = tmp_path / "peers"
    peers.mkdir()
    monkeypatch.setattr("loop.execution.sandbox.linux.paths._MAX_HARDLINK_MOUNTS", 3)
    leaves = []
    for group, size in (("a", 2), ("b", 2), ("c", 1)):
        for index in range(size):
            directory = workspace / group / str(index)
            directory.mkdir(parents=True)
            outside = peers / f"outside-{group}-{index}"
            outside.write_text("canary")
            leaf = directory / "linked"
            os.link(outside, leaf)
            leaves.append(leaf)
    assert prepare_paths(create(read_roots=(peers,)), scratch)[3] == (
        workspace / "a",
        workspace / "b",
        leaves[-1],
    )


def test_large_root_hardlink_sets_fail_before_exceeding_mount_limit(bound, tmp_path, monkeypatch):
    """Root-level aliases cannot force the trusted helper beyond its bounded mount count."""
    create, workspace, scratch = bound
    peers = tmp_path / "peers"
    peers.mkdir()
    monkeypatch.setattr("loop.execution.sandbox.linux.paths._MAX_HARDLINK_MOUNTS", 2)
    for index in range(3):
        outside = peers / f"outside-{index}"
        outside.write_text(f"module-{index}")
        os.link(outside, workspace / f"alias-{index}")
    with pytest.raises(ValueError, match="exceed the bounded mount limit"):
        prepare_paths(create(read_roots=(peers,)), scratch)


def test_mount_arguments_preserves_nested_write_grants_inside_readonly_workspace(binding):
    """Limit nested writes to their subtree and keep instruction files protected."""
    create, workspace, scratch, _ = binding
    nested = workspace / "nested"
    nested.mkdir()
    instruction = nested / "AGENTS.md"
    instruction.write_text("instructions")
    request = create(read_roots=(nested,), write_roots=(nested,))
    with mount_arguments(request, scratch) as (arguments, _, _):
        workspace_index = arguments.index(str(workspace))
        nested_index = arguments.index(str(nested))
        instruction_index = arguments.index(str(instruction))
        assert workspace_index < nested_index < instruction_index
        assert arguments[workspace_index - 2] == "--ro-bind-fd"
        assert arguments[nested_index - 2] == "--bind-fd"
        assert arguments[instruction_index - 2] == "--ro-bind-fd"


def test_mount_arguments_keeps_exact_file_parent_mutable_for_filename_policy(binding):
    """Pin the containing directory while leaving the future approved file unmounted."""
    create, workspace, scratch, _ = binding
    parent = workspace / "output"
    parent.mkdir()
    coverage = parent / ".coverage"
    control = parent / "AGENTS.md"
    control.write_text("protected")
    request = create(write_roots=(), write_files=(coverage,))

    with mount_arguments(request, scratch) as (arguments, descriptors, _):
        workspace_index = arguments.index(str(workspace))
        parent_index = arguments.index(str(parent))
        control_index = arguments.index(str(control))
        assert workspace_index < parent_index < control_index
        assert arguments[workspace_index - 2] == "--ro-bind-fd"
        assert arguments[parent_index - 2] == "--bind-fd"
        assert int(arguments[parent_index - 1]) in descriptors
        assert arguments[control_index - 2] == "--ro-bind-fd"
        assert str(coverage) not in arguments
        assert not coverage.exists()


def test_prepare_paths_refuses_exact_file_scope_inside_writable_scratch(bound):
    """A scratch-wide filename rule must not expose siblings on an exact-file parent."""
    create, _, scratch = bound
    workspace = scratch / "nested-workspace"
    workspace.mkdir()
    request = create(
        workspace=workspace, cwd=workspace, write_roots=(), write_files=(workspace / ".coverage",)
    )

    with pytest.raises(ValueError, match="cannot overlap the writable scratch view"):
        prepare_paths(request, scratch)


@pytest.mark.parametrize("unsafe", ["missing", "contents", "writable", "hardlink", "directory"])
def test_mount_arguments_refuses_unsafe_private_file_control(binding, tmp_path, unsafe):
    """File masks require a separate empty single-link read-only regular inode."""
    create, workspace, scratch, _ = binding
    (workspace / ".ssh").write_text("private")
    mask = tmp_path / "mask"
    if unsafe == "directory":
        mask.mkdir()
    else:
        mask.write_text("contents" if unsafe == "contents" else "")
    if unsafe == "hardlink":
        os.link(mask, tmp_path / "mask-link")
    descriptor = os.open(mask, os.O_RDWR if unsafe == "writable" else os.O_RDONLY)
    try:
        supplied = None if unsafe == "missing" else descriptor
        diagnostic = (
            "require an empty control inode" if unsafe == "missing" else "control inode is unsafe"
        )
        with (
            pytest.raises(ValueError, match=diagnostic),
            mount_arguments(create(), scratch, empty_mask=supplied),
        ):
            pytest.fail("unsafe control inode produced mount arguments")
        assert os.fstat(descriptor).st_ino == mask.stat().st_ino
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("caller_failure", [False, True])
def test_mount_arguments_owns_only_its_private_mask_duplicate(binding, tmp_path, caller_failure):
    """Close mask duplicates on exit and retain ownership of the original descriptor."""
    create, workspace, scratch, _ = binding
    for name in (".ssh", ".aws"):
        (workspace / name).write_text("credential")
    mask = tmp_path / "mask"
    mask.write_bytes(b"")
    descriptor = os.open(mask, os.O_RDONLY)
    try:
        with (
            (
                pytest.raises(RuntimeError, match="controlled caller failure")
                if caller_failure
                else nullcontext()
            ),
            mount_arguments(create(), scratch, empty_mask=descriptor) as (arguments, inherited, _),
        ):
            duplicates = set()
            for name in (".ssh", ".aws"):
                index = arguments.index(str(workspace / name))
                assert arguments[index - 2] == "--ro-bind-fd"
                duplicates.add(int(arguments[index - 1]))
            assert len(duplicates) == 1
            duplicate = duplicates.pop()
            assert duplicate != descriptor and duplicate in inherited
            assert not os.get_inheritable(duplicate)
            assert os.fstat(duplicate).st_ino == mask.stat().st_ino
            if caller_failure:
                raise RuntimeError("controlled caller failure")
        with pytest.raises(OSError) as error:
            os.fstat(duplicate)
        assert error.value.errno == errno.EBADF
        assert os.fstat(descriptor).st_ino == mask.stat().st_ino
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("root", ["filesystem", "home"])
def test_mount_arguments_refuses_whole_filesystem_and_home_views(
    binding, monkeypatch, tmp_path, root
):
    """Broad grant roots cannot turn the request fragment into a host filesystem view."""
    create, _, scratch, opened = binding
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    request = replace(create(), read_roots=(Path("/") if root == "filesystem" else home,))
    with (
        pytest.raises(ValueError, match="whole filesystem or home"),
        mount_arguments(request, scratch),
    ):
        pytest.fail("broad host root produced mount arguments")
    assert not opened


@pytest.mark.parametrize("reserved", ["NATIVE_SCRATCH_PATH", "NATIVE_ALIAS_PATH"])
def test_mount_arguments_rejects_reserved_namespace_overlap(bound, monkeypatch, reserved):
    """A request grant cannot replace or contain a private namespace destination."""
    create, workspace, scratch = bound
    monkeypatch.setattr(linux_paths, reserved, workspace / "private-view")

    with (
        pytest.raises(ValueError, match="namespace destination overlaps"),
        mount_arguments(create(), scratch),
    ):
        pytest.fail("An overlapping mount authority was accepted")


def test_mount_arguments_keeps_selected_workspace_read_only_in_scratch_mirror(binding):
    """A workspace below scratch stays read-only through both pinned mount spellings."""
    create, _, scratch, _ = binding
    selected = scratch / "selected"
    selected.mkdir()
    (selected / "ordinary").write_text("selected")
    request = create(cwd=selected, workspace=selected, write_roots=())
    with mount_arguments(request, scratch) as (arguments, inherited, _):
        for path in (selected, _TEST_NATIVE_SCRATCH / "selected"):
            index = arguments.index(str(path))
            assert arguments[index - 2] == "--ro-bind-fd"
            assert int(arguments[index - 1]) in inherited
        index = arguments.index(str(_TEST_NATIVE_SCRATCH))
        assert arguments[index + 1] == "--bind-fd"


def test_mount_arguments_mirrors_scratch_hardlinks_and_private_masks(binding, tmp_path):
    """The stable scratch view retains read-only linked leaves and masked private entries."""
    create, _, scratch, _ = binding
    peers = tmp_path / "peers"
    peers.mkdir()
    outside = peers / "outside"
    outside.write_text("preserve")
    os.link(outside, scratch / "linked")
    (scratch / ".ssh").mkdir()
    (scratch / ".aws").write_text("private")
    mask = tmp_path / "mask.empty"
    mask.write_bytes(b"")
    with (
        mask.open("rb") as empty,
        mount_arguments(create(read_roots=(peers,)), scratch, empty_mask=empty.fileno()) as (
            arguments,
            inherited,
            _,
        ),
    ):
        for name in ("linked", ".aws"):
            path = _TEST_NATIVE_SCRATCH / name
            index = arguments.index(str(path))
            assert arguments[index - 2] == "--ro-bind-fd"
            assert int(arguments[index - 1]) in inherited
        index = arguments.index(str(_TEST_NATIVE_SCRATCH / ".ssh"))
        assert arguments[index - 1] == "--tmpfs"


@pytest.mark.parametrize("skip", [False, True])
def test_mount_arguments_pins_external_executables_once_by_file_identity(binding, tmp_path, skip):
    """Mount an external executable by descriptor unless its approved root already supplies it."""
    create, _, scratch, _ = binding
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    alias = tools / "alias"
    alias.symlink_to(executable)
    details = executable.stat()
    request = create(
        environment={"PATH": str(tools), "TMPDIR": str(scratch)},
        executable_identities=(
            ("tool", executable, executable, details.st_dev, details.st_ino),
            ("alias", alias, executable, details.st_dev, details.st_ino),
        ),
    )

    with mount_arguments(
        request,
        scratch,
        skip_executable_roots=(tools,) if skip else (),
    ) as (arguments, inherited, _):
        if skip:
            assert str(executable) not in arguments
        else:
            index = arguments.index(str(executable))
            descriptor = int(arguments[index - 1])
            assert arguments[index - 2] == "--ro-bind-fd"
            assert descriptor in inherited
            assert arguments.count(str(executable)) == 1


def test_mount_arguments_rejects_an_external_executable_without_a_path_root(
    binding, monkeypatch, tmp_path
):
    """An executable identity cannot mount from a directory absent from the bound PATH."""
    create, _, scratch, _ = binding
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    details = executable.stat()
    request = create(environment={"PATH": str(tmp_path / "empty"), "TMPDIR": str(scratch)})
    request = replace(
        request,
        executable_identities=(("tool", executable, executable, details.st_dev, details.st_ino),),
    )
    monkeypatch.setattr(SandboxRequest, "paths_are_current", lambda _self: True)

    with (
        pytest.raises(ValueError, match="outside the mounted PATH roots"),
        mount_arguments(request, scratch),
    ):
        pytest.fail("An external executable without a PATH root was mounted")


def test_mount_arguments_rejects_an_unsafe_external_executable_path_root(binding, tmp_path):
    """A writable external PATH directory cannot supply an automatic executable mount."""
    create, _, scratch, _ = binding
    tools = tmp_path / "tools"
    request, _ = _external_executable_request(create, scratch, tools)
    tools.chmod(0o777)

    with (
        pytest.raises(ValueError, match="executable PATH root is unsafe"),
        mount_arguments(request, scratch),
    ):
        pytest.fail("An unsafe executable PATH root was mounted")


def test_mount_arguments_requires_path_descriptors_for_external_executables(
    binding, monkeypatch, tmp_path
):
    """External executable binding refuses when O_PATH cannot pin its inode."""
    create, workspace, scratch, _ = binding
    tools = tmp_path / "tools"
    request, _ = _external_executable_request(create, scratch, tools)
    descriptors = {workspace: 3, scratch: 4, tools: 5}
    monkeypatch.setattr(
        linux_paths,
        "bind_paths",
        lambda *_args, **_kwargs: nullcontext((((), (), (), ()), descriptors)),
    )
    monkeypatch.delattr(linux_paths.os, "O_PATH", raising=False)

    with (
        pytest.raises(OSError, match="executable pinning requires O_PATH"),
        mount_arguments(request, scratch),
    ):
        pytest.fail("An executable was accepted without a pinned descriptor")


def test_mount_arguments_rejects_a_changed_external_executable_path(binding, monkeypatch, tmp_path):
    """A changed PATH spelling fails before its target receives a namespace mount."""
    create, _, scratch, _ = binding
    tools = tmp_path / "tools"
    request, executable = _external_executable_request(create, scratch, tools)
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        """Return a different target only for the bound executable spelling."""
        if path == executable:
            return tools / "replacement"
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(SandboxRequest, "paths_are_current", lambda _self: True)

    with (
        pytest.raises(ValueError, match="executable path changed during binding"),
        mount_arguments(request, scratch),
    ):
        pytest.fail("A changed executable spelling was mounted")


def test_mount_arguments_rejects_an_external_executable_changed_after_open(binding, tmp_path):
    """A multiply linked executable cannot pass descriptor identity validation."""
    create, _, scratch, _ = binding
    tools = tmp_path / "tools"
    executable = tools / "tool"
    tools.mkdir()
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    os.link(executable, tmp_path / "tool-alias")
    details = executable.stat()
    request = create(
        environment={"PATH": str(tools), "TMPDIR": str(scratch)},
        executable_identities=(("tool", executable, executable, details.st_dev, details.st_ino),),
    )

    with (
        pytest.raises(ValueError, match="executable changed during binding"),
        mount_arguments(request, scratch),
    ):
        pytest.fail("A multiply linked executable passed identity validation")


def test_mount_arguments_rechecks_mask_before_yield(binding, monkeypatch, tmp_path):
    """An empty mask changed during construction fails before arguments leave the context."""
    create, workspace, scratch, _ = binding
    (workspace / ".ssh").write_text("private")
    mask = tmp_path / "mask"
    mask.write_bytes(b"")
    descriptor = os.open(mask, os.O_RDONLY)
    original_fstat = os.fstat
    mask_calls = 0

    def fstat(current):
        """Mutate the independently controlled mask at its final identity recheck."""
        nonlocal mask_calls
        details = original_fstat(current)
        if details.st_ino == mask.stat().st_ino:
            mask_calls += 1
            if mask_calls == 2:
                mask.write_text("changed mask")
                details = original_fstat(current)
        return details

    monkeypatch.setattr(os, "fstat", fstat)
    try:
        with (
            pytest.raises(ValueError, match="control inode changed"),
            mount_arguments(create(), scratch, empty_mask=descriptor),
        ):
            pytest.fail("changed mask produced mount arguments")
    finally:
        os.close(descriptor)


def test_mount_arguments_rejects_final_shared_authority_drift(binding, monkeypatch):
    """Reject identity drift after descriptor preparation and before yielding the mount."""
    create, _, scratch, _ = binding
    request = create()
    responses = iter((True, True, True, True, False))
    monkeypatch.setattr(SandboxRequest, "paths_are_current", lambda self: next(responses))
    with (
        pytest.raises(ValueError, match="approved path authority changed"),
        mount_arguments(request, scratch),
    ):
        pytest.fail("stale authority produced mount arguments")


def test_mount_arguments_preserves_external_path_symlink_metadata(bound, monkeypatch, tmp_path):
    """A PATH spelling outside grants uses its pinned canonical directory and original alias."""
    create, _workspace, scratch = bound
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    alias = tmp_path / "commands"
    alias.symlink_to(runtime, target_is_directory=True)
    monkeypatch.setattr(os, "O_PATH", os.O_RDONLY, raising=False)
    request = create(environment={"PATH": str(alias), "TMPDIR": str(scratch)})
    with mount_arguments(request, scratch) as (arguments, _, _):
        index = arguments.index("--symlink")
        assert arguments[index : index + 3] == ("--symlink", str(runtime), str(alias))
        assert arguments[-5:-3] == ("--dir", str(_TEST_NATIVE_SCRATCH))
        assert arguments[-3] == "--bind-fd"
        assert arguments[-1] == str(_TEST_NATIVE_SCRATCH)


def test_mount_arguments_skips_path_roots_supplied_by_runtime_mount(bound, monkeypatch, tmp_path):
    """A runtime-covered PATH directory needs no duplicate request mount or alias."""
    create, _workspace, scratch = bound
    runtime = tmp_path / "runtime"
    canonical = runtime / "tool-version/bin"
    canonical.mkdir(parents=True)
    alias = runtime / "tool/bin"
    alias.parent.symlink_to(canonical.parent, target_is_directory=True)
    monkeypatch.setattr(os, "O_PATH", os.O_RDONLY, raising=False)
    request = create(environment={"PATH": str(alias), "TMPDIR": str(scratch)})

    with mount_arguments(request, scratch, skip_path_roots=(alias,)) as (arguments, _, _):
        assert str(canonical.parent) not in arguments
        assert str(alias) not in arguments


def test_mount_arguments_requires_skipped_path_roots_to_be_bound(bound, tmp_path):
    """A caller cannot omit mounts for PATH spellings absent from request authority."""
    create, _workspace, scratch = bound
    request = create()

    with (
        pytest.raises(ValueError, match="not bound to this request"),
        mount_arguments(request, scratch, skip_path_roots=(tmp_path / "outside",)),
    ):
        pytest.fail("an unbound PATH root was omitted")


def test_session_scratch_under_private_ancestry_keeps_masks_and_hardlink_guards(bound, tmp_path):
    """Only the exact scratch grant bypasses ancestor rejection and its contents remain guarded."""
    create, _, _ = bound
    scratch = tmp_path / "Library" / "storage" / "session"
    scratch.mkdir(parents=True, mode=0o700)
    private = scratch / ".ssh"
    private.mkdir()
    (private / "key").write_text("secret")
    value = create(
        environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)},
        scratch_root=scratch,
    )
    _, masked, _, _ = prepare_paths(value, scratch)
    assert private in masked
    (scratch / "public-alias").hardlink_to(private / "key")
    with pytest.raises(ValueError, match="aliases protected or private"):
        prepare_paths(value, scratch)


def test_fresh_session_data_scan_allows_private_names_without_exposing_workspace(bound):
    """Fresh scratch data is ordinary while identically named workspace entries stay masked."""
    create, workspace, scratch = bound
    (workspace / ".loop").mkdir()
    (scratch / ".loop").mkdir()
    (scratch / "library").write_text("data")
    source = create(scratch_root=scratch, scratch_data=True)
    _, private, _, _ = prepare_paths(source, scratch)
    assert workspace / ".loop" in private
    assert scratch / ".loop" not in private
    assert scratch / "library" not in private
    (scratch / "shared").hardlink_to(scratch / "library")
    with pytest.raises(ValueError, match="shared inode"):
        prepare_paths(create(scratch_root=scratch, scratch_data=True), scratch)


def test_prepare_paths_refuses_environment_value_root(bound, tmp_path):
    """A grant rooted inside environment data cannot hide its protected ancestor component."""
    create, _workspace, scratch = bound
    values = tmp_path / ".env.production"
    values.mkdir()
    with pytest.raises(ValueError, match="environment values"):
        prepare_paths(create(read_roots=(values,)), scratch)


def test_prepare_paths_checks_alias_authority_in_one_overlapping_walk(bound, monkeypatch):
    """An inner read grant does not repeat classification or bypass the outer protection."""
    create, workspace, scratch = bound
    inner = workspace / "nested"
    inner.mkdir()
    instruction = inner / "AGENTS.md"
    instruction.write_text("instructions")
    request = create(read_roots=(inner,))
    check = SandboxRequest.denies_path
    classified = []

    def denies_path(value, path):
        """Record policy classification of each physical filesystem entry."""
        classified.append(path)
        return check(value, path)

    monkeypatch.setattr(SandboxRequest, "denies_path", denies_path)
    assert prepare_paths(request, scratch) == ((instruction,), (), (inner,), ())
    assert classified.count(instruction) == 1


@pytest.mark.parametrize(
    "required",
    [
        "unused",
        "cwd",
        "scratch",
        "read",
        "write",
        "executable",
        "extra-alias",
        "visible",
        "visible-complete",
    ],
)
def test_prepare_paths_hides_links_only_inside_denied_directories(
    bound, tmp_path, monkeypatch, required
):
    """Unaccounted links stay refused unless a native directory mask already conceals them."""
    create, workspace, scratch = bound
    environment = workspace / "tools" / "retired"
    packages = environment / "packages"
    packages.mkdir(parents=True)
    (environment / "bin").mkdir()
    cached = tmp_path / "cache"
    cached.write_text("package data")
    (packages / "data").hardlink_to(cached)
    (packages / ".env").write_text("private")
    if required == "unused":
        (environment / "legacy-tool").symlink_to(cached)
    changes = {
        "write_roots": (workspace,),
        "hard_denied_paths": () if required == "visible" else (environment,),
    }
    if required == "cwd":
        changes["cwd"] = packages
    elif required == "scratch":
        scratch = environment / "temp"
        scratch.mkdir(mode=0o700)
        changes["environment"] = {"PATH": str(workspace), "TMPDIR": str(scratch)}
        changes["scratch_root"] = scratch
    elif required == "read":
        changes["read_roots"] = (packages,)
    elif required == "write":
        changes["write_roots"] = (environment,)
    elif required == "executable":
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        interpreter = runtime / "tool"
        interpreter.write_text("interpreter")
        interpreter.chmod(0o755)
        spelling = environment / "bin/tool"
        spelling.symlink_to(interpreter)
        changes["read_roots"] = (runtime,)
        changes["environment"] = {"PATH": str(spelling.parent), "TMPDIR": str(scratch)}
        detail = interpreter.stat()
        changes["workspace_executable_identities"] = (
            (spelling.name, spelling, interpreter, detail.st_dev, detail.st_ino),
        )
    elif required in {"extra-alias", "visible-complete"}:
        (workspace / "unmasked-link").hardlink_to(cached)
        if required == "visible-complete":
            cached.unlink()
    request = create(**changes)
    if required == "unused":
        assert prepare_paths(request, scratch) == (
            (),
            (environment,),
            tuple(sorted((environment, environment.parent))),
            (),
        )
    elif required in {"extra-alias", "visible"}:
        with pytest.raises(ValueError, match="Incomplete hardlink aliases"):
            prepare_paths(request, scratch)
    elif required == "visible-complete":
        with pytest.raises(ValueError, match="aliases protected or private data"):
            prepare_paths(request, scratch)
    else:
        with pytest.raises(ValueError, match="A required Linux sandbox path"):
            prepare_paths(request, scratch)
