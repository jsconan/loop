"""Verify bound Linux path inspection, private masks and protected ancestor preparation."""

import errno
import os
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.contracts import SandboxRequest
from loop.execution.sandbox.linux import paths as linux_paths
from loop.execution.sandbox.linux.constants import NATIVE_SCRATCH_PATH
from loop.execution.sandbox.linux.paths import (
    bind_paths,
    mount_arguments,
    prepare_paths,
    python_runtime_mounts,
)


@pytest.fixture
def bound(monkeypatch, tmp_path):
    """Create a real private scratch and workspace with controlled time and temp authority."""
    workspace, scratch = tmp_path / "workspace", tmp_path / "scratch"
    workspace.mkdir()
    scratch.mkdir(mode=0o700)
    monkeypatch.setattr(
        "loop.execution.sandbox.contracts.tempfile.gettempdir", lambda: str(tmp_path)
    )
    monkeypatch.setattr("loop.execution.sandbox.linux.paths.time.monotonic", lambda: 0)

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


def _python_environment(tmp_path):
    """Create a workspace venv that references a versioned user-local CPython install."""
    home = tmp_path / "home"
    home.mkdir()
    workspace = tmp_path / "workspace"
    venv_bin = workspace / ".venv/bin"
    venv_bin.mkdir(parents=True)
    runtime_parent = home / "runtime-store"
    runtime_parent.mkdir()
    runtime_alias = runtime_parent / "cpython-3.14"
    runtime_root = runtime_parent / "cpython-3.14.1"
    runtime_root.mkdir()
    (runtime_root / "bin").mkdir()
    (runtime_root / "lib/python3.14").mkdir(parents=True)
    interpreter = runtime_root / "bin/python3.14"
    interpreter.write_text("interpreter", encoding="utf-8")
    interpreter.chmod(0o755)
    runtime_alias.symlink_to(runtime_root, target_is_directory=True)
    (venv_bin / "python").symlink_to(runtime_alias / "bin/python3.14")
    (venv_bin.parent / "pyvenv.cfg").write_text(
        f"home = {runtime_alias / 'bin'}\nimplementation = CPython\nversion_info = 3.14\n",
        encoding="utf-8",
    )
    return home, workspace, venv_bin, runtime_root, runtime_alias


def _uv_identity(path):
    """Create an exact file identity row for the named uv runtime dependency."""
    path.write_text("uv", encoding="utf-8")
    details = path.stat()
    return (("uv", path, path, details.st_dev, details.st_ino),)


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


def test_python_runtime_mounts_bind_only_a_user_local_runtime_referenced_by_uv(
    tmp_path, monkeypatch
):
    """Discover the safe runtime named by a workspace venv only for a bound uv executable."""
    home, workspace, venv_bin, runtime_root, runtime_alias = _python_environment(tmp_path)
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.paths.pwd.getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(home)),
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    identity = _uv_identity(tools / "uv")
    config = venv_bin.parent / "pyvenv.cfg"
    config.write_text(config.read_text() + "unrecognized line\nother = value\n", encoding="utf-8")

    assert python_runtime_mounts(workspace, ((venv_bin, venv_bin),), identity) == (
        (runtime_root, runtime_alias),
    )
    assert python_runtime_mounts(workspace, ((venv_bin, venv_bin),), ()) == ()
    assert (
        python_runtime_mounts(workspace, ((venv_bin, venv_bin),), identity, system_roots=(home,))
        == ()
    )


def test_python_runtime_mounts_accept_a_canonical_python_home(tmp_path, monkeypatch):
    """A venv already pointing at the runtime's canonical path needs no alias."""
    home, workspace, venv_bin, runtime_root, _runtime_alias = _python_environment(tmp_path)
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.paths.pwd.getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(home)),
    )
    (venv_bin / "python").unlink()
    (venv_bin / "python").symlink_to(runtime_root / "bin/python3.14")
    (venv_bin.parent / "pyvenv.cfg").write_text(
        f"home = {runtime_root / 'bin'}\nimplementation = CPython\nversion_info = 3.14\n",
        encoding="utf-8",
    )
    tools = tmp_path / "tools"
    tools.mkdir()

    assert python_runtime_mounts(
        workspace, ((venv_bin, venv_bin),), _uv_identity(tools / "uv")
    ) == ((runtime_root, runtime_root),)


@pytest.mark.parametrize(
    "metadata",
    [
        "home = /runtime/bin\nimplementation = PyPy\nversion_info = 3.14\n",
        "home = /runtime/bin\nimplementation = CPython\nversion_info = unknown\n",
        "implementation = CPython\nversion_info = 3.14\n",
    ],
)
def test_python_runtime_mounts_skip_non_cpython_or_incomplete_venv_metadata(
    tmp_path, monkeypatch, metadata
):
    """Do not infer runtime authority from unsupported or incomplete venv metadata."""
    home, workspace, venv_bin, _, _ = _python_environment(tmp_path)
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.paths.pwd.getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(home)),
    )
    (venv_bin.parent / "pyvenv.cfg").write_text(metadata, encoding="utf-8")
    tools = tmp_path / "tools"
    tools.mkdir()

    assert (
        python_runtime_mounts(workspace, ((venv_bin, venv_bin),), _uv_identity(tools / "uv")) == ()
    )


def test_python_runtime_mounts_skip_missing_and_unrelated_path_environments(tmp_path, monkeypatch):
    """Search only workspace PATH venvs and ignore absent or unrelated configurations."""
    home, workspace, venv_bin, _, _ = _python_environment(tmp_path)
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.paths.pwd.getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(home)),
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    identity = _uv_identity(tools / "uv")
    config = venv_bin.parent / "pyvenv.cfg"
    config.unlink()

    assert python_runtime_mounts(workspace, ((venv_bin, venv_bin),), identity) == ()
    assert python_runtime_mounts(workspace, ((tools, tools),), identity) == ()
    assert python_runtime_mounts(workspace, ((tmp_path / "bin", tmp_path / "bin"),), identity) == ()


@pytest.mark.parametrize(
    ("failure", "raises"),
    [
        ("relative-home", True),
        ("non-bin-home", True),
        ("missing-home", False),
        ("missing-interpreter", False),
        ("missing-stdlib", False),
        ("external-interpreter", False),
        ("external-stdlib", False),
        ("home-root", True),
        ("workspace-root", True),
        ("outside-user-home", True),
        ("unsafe-directory", True),
        ("foreign-group-directory", True),
        ("unsafe-interpreter", True),
        ("non-executable-interpreter", False),
        ("directory-interpreter", False),
        ("protected-alias", True),
    ],
)
def test_python_runtime_mounts_validate_the_selected_interpreter_and_tree(
    tmp_path, monkeypatch, failure, raises
):
    """Only a safe CPython tree referenced by the workspace venv becomes read authority."""
    home, workspace, venv_bin, runtime_root, runtime_alias = _python_environment(tmp_path)
    configured_home = runtime_alias / "bin"
    interpreter = runtime_root / "bin/python3.14"
    config = venv_bin.parent / "pyvenv.cfg"
    if failure in {"relative-home", "non-bin-home", "missing-home"}:
        home_value = (
            "relative/bin"
            if failure == "relative-home"
            else str(runtime_root / "lib")
            if failure == "non-bin-home"
            else str(home / "missing/bin")
        )
        config.write_text(
            f"home = {home_value}\nimplementation = CPython\nversion_info = 3.14\n",
            encoding="utf-8",
        )
    elif failure == "missing-interpreter":
        (venv_bin / "python").unlink()
    elif failure == "missing-stdlib":
        (runtime_root / "lib/python3.14").rmdir()
    elif failure == "external-interpreter":
        external = home / "other/python3.14"
        external.parent.mkdir()
        external.write_text("interpreter", encoding="utf-8")
        external.chmod(0o755)
        (venv_bin / "python").unlink()
        (venv_bin / "python").symlink_to(external)
    elif failure == "external-stdlib":
        stdlib = runtime_root / "lib/python3.14"
        stdlib.rmdir()
        external = home / "external-stdlib"
        external.mkdir()
        stdlib.symlink_to(external, target_is_directory=True)
    elif failure == "home-root":
        (home / "bin").mkdir()
        (home / "lib/python3.14").mkdir(parents=True)
        (home / "bin/python3.14").write_text("interpreter", encoding="utf-8")
        (home / "bin/python3.14").chmod(0o755)
        configured_home = home / "bin"
        (venv_bin / "python").unlink()
        (venv_bin / "python").symlink_to(configured_home / "python3.14")
        config.write_text(
            f"home = {configured_home}\nimplementation = CPython\nversion_info = 3.14\n",
            encoding="utf-8",
        )
    elif failure == "workspace-root":
        runtime_root.rename(workspace / "cpython-runtime")
        runtime_root = workspace / "cpython-runtime"
        runtime_alias.unlink()
        runtime_alias.symlink_to(runtime_root, target_is_directory=True)
        home = workspace
    elif failure == "outside-user-home":
        runtime_root.rename(tmp_path / "external-python")
        runtime_root = tmp_path / "external-python"
        runtime_alias.unlink()
        runtime_alias.symlink_to(runtime_root, target_is_directory=True)
    elif failure in {"unsafe-directory", "foreign-group-directory"}:
        runtime_root.chmod(0o777 if failure == "unsafe-directory" else 0o775)
        if failure == "foreign-group-directory":
            monkeypatch.setattr("loop.execution.sandbox.linux.paths.os.getgid", lambda: -1)
    elif failure == "unsafe-interpreter":
        interpreter.chmod(0o777)
    elif failure == "non-executable-interpreter":
        interpreter.chmod(0o644)
    elif failure == "directory-interpreter":
        interpreter.unlink()
        interpreter.mkdir()
        (venv_bin / "python").unlink()
        (venv_bin / "python").symlink_to(runtime_alias / "bin/python3.14")
    elif failure == "protected-alias":
        alias = home / ".config/python"
        alias.parent.mkdir()
        alias.symlink_to(runtime_root, target_is_directory=True)
        (venv_bin / "python").unlink()
        (venv_bin / "python").symlink_to(alias / "bin/python3.14")
        config.write_text(
            f"home = {alias / 'bin'}\nimplementation = CPython\nversion_info = 3.14\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.paths.pwd.getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(home)),
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    identity = _uv_identity(tools / "uv")

    if raises:
        with pytest.raises(ValueError):
            python_runtime_mounts(workspace, ((venv_bin, venv_bin),), identity)
    else:
        assert python_runtime_mounts(workspace, ((venv_bin, venv_bin),), identity) == ()


@pytest.mark.parametrize("failure", ["lookup", "file", "world-write", "group-write"])
def test_python_runtime_mounts_reject_an_untrusted_account_home(tmp_path, monkeypatch, failure):
    """Refuse runtime discovery when the account home cannot be safely bound."""
    home, workspace, venv_bin, _, _ = _python_environment(tmp_path)
    if failure == "lookup":
        monkeypatch.setattr(
            "loop.execution.sandbox.linux.paths.pwd.getpwuid",
            Mock(side_effect=KeyError("missing account")),
        )
    elif failure == "file":
        fake_home = tmp_path / "not-home"
        fake_home.write_text("not a directory", encoding="utf-8")
        monkeypatch.setattr(
            "loop.execution.sandbox.linux.paths.pwd.getpwuid",
            lambda _uid: SimpleNamespace(pw_dir=str(fake_home)),
        )
    else:
        home.chmod(0o777 if failure == "world-write" else 0o775)
        monkeypatch.setattr(
            "loop.execution.sandbox.linux.paths.pwd.getpwuid",
            lambda _uid: SimpleNamespace(pw_dir=str(home)),
        )
        if failure == "group-write":
            monkeypatch.setattr("loop.execution.sandbox.linux.paths.os.getgid", lambda: -1)
    tools = tmp_path / "tools"
    tools.mkdir()

    with pytest.raises(ValueError, match="home is unavailable|home is unsafe"):
        python_runtime_mounts(
            workspace,
            ((venv_bin, venv_bin),),
            _uv_identity(tools / "uv"),
        )


@pytest.mark.parametrize("failure", ["link", "descriptor", "changed", "utf8", "oversized"])
def test_python_runtime_mounts_reject_changed_or_invalid_venv_metadata(
    tmp_path, monkeypatch, failure
):
    """Bind bounded venv metadata without accepting a replacement or malformed encoding."""
    home, workspace, venv_bin, _, _ = _python_environment(tmp_path)
    monkeypatch.setattr(linux_paths.pwd, "getpwuid", lambda _uid: SimpleNamespace(pw_dir=str(home)))
    config = venv_bin.parent / "pyvenv.cfg"
    tools = tmp_path / "tools"
    tools.mkdir()
    if failure == "link":
        copy = tmp_path / "pyvenv-copy"
        os.link(config, copy)
    elif failure == "descriptor":
        original_fstat = linux_paths.os.fstat

        def fstat(descriptor):
            """Change the opened config identity before parsing it."""
            details = original_fstat(descriptor)
            return SimpleNamespace(st_dev=details.st_dev, st_ino=details.st_ino + 1)

        monkeypatch.setattr(linux_paths.os, "fstat", fstat)
    elif failure == "changed":
        original_fstat = linux_paths.os.fstat
        calls = 0

        def fstat(descriptor):
            """Change the metadata timestamp between its read and final check."""
            nonlocal calls
            calls += 1
            details = original_fstat(descriptor)
            return SimpleNamespace(
                st_dev=details.st_dev,
                st_ino=details.st_ino,
                st_size=details.st_size,
                st_mtime_ns=details.st_mtime_ns + (calls > 1),
            )

        monkeypatch.setattr(linux_paths.os, "fstat", fstat)
    elif failure == "utf8":
        config.write_bytes(b"\xff")
    else:
        config.write_bytes(b"x" * (64 * 1024 + 1))
    expected = "unsafe|changed|invalid" if failure in {"link", "oversized"} else "changed|invalid"
    if failure == "utf8":
        expected = "invalid"

    with pytest.raises(ValueError, match=expected):
        python_runtime_mounts(
            workspace,
            ((venv_bin, venv_bin),),
            _uv_identity(tools / "uv"),
        )


@pytest.mark.parametrize("unsafe", ["protected", "config-mode", "duplicate"])
def test_python_runtime_mounts_reject_unsafe_automatic_read_authority(
    tmp_path, monkeypatch, unsafe
):
    """Reject private or ambiguous venv metadata before granting a runtime directory read."""
    home, workspace, venv_bin, _runtime_root, runtime_alias = _python_environment(tmp_path)
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.paths.pwd.getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(home)),
    )
    config = venv_bin.parent / "pyvenv.cfg"
    if unsafe == "protected":
        private_parent = home / ".config"
        private_parent.mkdir()
        protected_root = private_parent / "cpython-3.14.1"
        protected_root.mkdir()
        (protected_root / "bin").mkdir()
        (protected_root / "lib/python3.14").mkdir(parents=True)
        (protected_root / "bin/python3.14").write_text("interpreter", encoding="utf-8")
        (protected_root / "bin/python3.14").chmod(0o755)
        runtime_alias.unlink()
        runtime_alias.symlink_to(protected_root, target_is_directory=True)
        (venv_bin / "python").unlink()
        (venv_bin / "python").symlink_to(runtime_alias / "bin/python3.14")
        config.write_text(
            f"home = {runtime_alias / 'bin'}\nimplementation = CPython\nversion_info = 3.14\n",
            encoding="utf-8",
        )
    elif unsafe == "config-mode":
        config.chmod(0o666)
    else:
        config.write_text(config.read_text() + "home = /duplicate\n", encoding="utf-8")
    uv = tmp_path / "uv"
    uv.write_text("uv", encoding="utf-8")
    details = uv.stat()

    with pytest.raises(ValueError, match="unsafe|ambiguous|outside the safe"):
        python_runtime_mounts(
            workspace,
            ((venv_bin, venv_bin),),
            (("uv", uv, uv, details.st_dev, details.st_ino),),
        )


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


@pytest.mark.parametrize("read_only", [False, True])
def test_prepare_paths_records_read_only_hardlinks_for_granted_roots(bound, tmp_path, read_only):
    """Writable hardlinks get mounts; links within an already read-only root need none."""
    create, workspace, scratch = bound
    external = tmp_path / "external"
    external.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("original")
    root = external if read_only else workspace
    os.link(outside, root / "alias")
    request = create(read_roots=(external,), write_roots=() if read_only else (workspace,))
    expected_mounts = () if read_only else (root / "alias",)
    assert prepare_paths(request, scratch) == ((), (), (), expected_mounts)
    assert outside.read_text() == "original"


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
    with pytest.raises(TimeoutError, match="exceeded the command deadline"):
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
            return SimpleNamespace(
                st_dev=details.st_dev,
                st_ino=details.st_ino + 1,
                st_mtime_ns=details.st_mtime_ns,
                st_ctime_ns=details.st_ctime_ns,
            )
        return details

    root_calls = 0

    def lstat(path, *args, **kwargs):
        """Model pathname drift only at a selected post-scan identity comparison."""
        nonlocal root_calls
        details = original_lstat(path, *args, **kwargs)
        if path == workspace and opened:
            root_calls += 1
            selected = 1 if change == "pathname" else 2
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
    diagnostic = "path root changed" if change == "root" else "scanned directory changed"
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
    with pytest.raises(ValueError, match="scanned directory changed"):
        prepare_paths(request, scratch)


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
        return request.deadline if calls == {"source": 9, "recheck": 10, "yield": 12}[phase] else 0

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
    with mount_arguments(request, scratch) as (arguments, inherited):
        operation = "--ro-bind-fd" if read_only else "--bind-fd"
        workspace_index = arguments.index(str(workspace))
        assert arguments[workspace_index - 2] == operation
        workspace_fd = int(arguments[workspace_index - 1])
        nested_index = arguments.index(str(instruction.parent))
        assert arguments[nested_index - 2] == operation
        protected_index = arguments.index(str(instruction))
        assert protected_index > nested_index
        assert arguments[protected_index - 2] == "--ro-bind-fd"
        assert arguments[-5:-3] == ("--dir", str(NATIVE_SCRATCH_PATH))
        assert arguments[-3] == "--bind-fd"
        assert arguments[-1] == str(NATIVE_SCRATCH_PATH)
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


def test_mount_arguments_binds_hardlinks_read_only_after_writable_roots(binding, tmp_path):
    """A hardlinked leaf is readable but read-only inside an otherwise writable workspace."""
    create, workspace, scratch, _ = binding
    outside = tmp_path / "outside"
    outside.write_text("outside-data")
    alias = workspace / "alias"
    os.link(outside, alias)
    request = create()
    with mount_arguments(request, scratch) as (arguments, inherited):
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
    monkeypatch.setattr("loop.execution.sandbox.linux.paths._MAX_HARDLINK_MOUNTS", 2)
    package = workspace / ".venv/site-packages/package"
    package.mkdir(parents=True)
    for index in range(3):
        outside = tmp_path / f"outside-{index}"
        outside.write_text(f"module-{index}")
        os.link(outside, package / f"module-{index}.py")
    nested = package / "data"
    nested.mkdir()
    outside = tmp_path / "outside-nested"
    outside.write_text("nested-module")
    os.link(outside, nested / "module.py")
    request = create()
    with mount_arguments(request, scratch) as (arguments, inherited):
        workspace_index = arguments.index(str(workspace))
        package_index = arguments.index(str(package))
        assert arguments[workspace_index - 2] == "--bind-fd"
        assert arguments[package_index - 2] == "--ro-bind-fd"
        assert int(arguments[package_index - 1]) in inherited
        assert str(package / "module-0.py") not in arguments


def test_large_hardlink_sets_group_at_a_shared_ancestor_when_needed(bound, tmp_path, monkeypatch):
    """Escalate grouping only enough to keep scattered hardlink mounts within their bound."""
    create, workspace, scratch = bound
    monkeypatch.setattr("loop.execution.sandbox.linux.paths._MAX_HARDLINK_MOUNTS", 2)
    site_packages = workspace / ".venv/site-packages"
    for index in range(3):
        package = site_packages / f"package-{index}"
        package.mkdir(parents=True)
        outside = tmp_path / f"outside-{index}"
        outside.write_text(f"module-{index}")
        os.link(outside, package / "module.py")
    assert prepare_paths(create(), scratch)[3] == (site_packages,)


def test_scattered_hardlink_grouping_stops_when_its_deadline_expires(bound, tmp_path, monkeypatch):
    """Expiry while classifying scattered links cannot return late mount authority."""
    create, workspace, scratch = bound
    monkeypatch.setattr("loop.execution.sandbox.linux.paths._MAX_HARDLINK_MOUNTS", 2)
    links = set()
    for index in range(3):
        directory = workspace / "group" / str(index)
        directory.mkdir(parents=True)
        outside = tmp_path / f"outside-{index}"
        outside.write_text("canary")
        link = directory / "linked"
        os.link(outside, link)
        links.add(link)
    request = create()
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
    with pytest.raises(TimeoutError, match="exceeded the command deadline"):
        prepare_paths(request, scratch)


def test_hardlink_grouping_preserves_nested_writable_roots(bound, tmp_path, monkeypatch):
    """A group containing another write root cannot silently make that root read-only."""
    create, workspace, scratch = bound
    monkeypatch.setattr("loop.execution.sandbox.linux.paths._MAX_HARDLINK_MOUNTS", 2)
    group = workspace / "group"
    nested = group / "writable"
    nested.mkdir(parents=True)
    for index in range(3):
        directory = group / str(index)
        directory.mkdir()
        outside = tmp_path / f"outside-{index}"
        outside.write_text("canary")
        os.link(outside, directory / "linked")
    value = create(write_roots=(workspace, nested), read_roots=(group,))
    with pytest.raises(ValueError, match="exceed the bounded mount limit"):
        prepare_paths(value, scratch)


def test_hardlink_grouping_updates_disjoint_ancestor_membership(bound, tmp_path, monkeypatch):
    """Several narrow groups meet the cap without making their shared parent read-only."""
    create, workspace, scratch = bound
    monkeypatch.setattr("loop.execution.sandbox.linux.paths._MAX_HARDLINK_MOUNTS", 3)
    leaves = []
    for group, size in (("a", 2), ("b", 2), ("c", 1)):
        for index in range(size):
            directory = workspace / group / str(index)
            directory.mkdir(parents=True)
            outside = tmp_path / f"outside-{group}-{index}"
            outside.write_text("canary")
            leaf = directory / "linked"
            os.link(outside, leaf)
            leaves.append(leaf)
    assert prepare_paths(create(), scratch)[3] == (workspace / "a", workspace / "b", leaves[-1])


def test_large_root_hardlink_sets_fail_before_exceeding_mount_limit(bound, tmp_path, monkeypatch):
    """Root-level aliases cannot force the trusted helper beyond its bounded mount count."""
    create, workspace, scratch = bound
    monkeypatch.setattr("loop.execution.sandbox.linux.paths._MAX_HARDLINK_MOUNTS", 2)
    for index in range(3):
        outside = tmp_path / f"outside-{index}"
        outside.write_text(f"module-{index}")
        os.link(outside, workspace / f"alias-{index}")
    with pytest.raises(ValueError, match="exceed the bounded mount limit"):
        prepare_paths(create(), scratch)


def test_mount_arguments_preserves_nested_write_grants_inside_readonly_workspace(binding):
    """Limit nested writes to their subtree and keep instruction files protected."""
    create, workspace, scratch, _ = binding
    nested = workspace / "nested"
    nested.mkdir()
    instruction = nested / "AGENTS.md"
    instruction.write_text("instructions")
    request = create(read_roots=(nested,), write_roots=(nested,))
    with mount_arguments(request, scratch) as (arguments, _):
        workspace_index = arguments.index(str(workspace))
        nested_index = arguments.index(str(nested))
        instruction_index = arguments.index(str(instruction))
        assert workspace_index < nested_index < instruction_index
        assert arguments[workspace_index - 2] == "--ro-bind-fd"
        assert arguments[nested_index - 2] == "--bind-fd"
        assert arguments[instruction_index - 2] == "--ro-bind-fd"


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
            mount_arguments(create(), scratch, empty_mask=descriptor) as (arguments, inherited),
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
    with mount_arguments(request, scratch) as (arguments, inherited):
        for path in (selected, NATIVE_SCRATCH_PATH / "selected"):
            index = arguments.index(str(path))
            assert arguments[index - 2] == "--ro-bind-fd"
            assert int(arguments[index - 1]) in inherited
        index = arguments.index(str(NATIVE_SCRATCH_PATH))
        assert arguments[index + 1] == "--bind-fd"


def test_mount_arguments_mirrors_scratch_hardlinks_and_private_masks(binding, tmp_path):
    """The stable scratch view retains read-only linked leaves and masked private entries."""
    create, _, scratch, _ = binding
    outside = tmp_path / "outside"
    outside.write_text("preserve")
    os.link(outside, scratch / "linked")
    (scratch / ".ssh").mkdir()
    (scratch / ".aws").write_text("private")
    mask = tmp_path / "mask.empty"
    mask.write_bytes(b"")
    with (
        mask.open("rb") as empty,
        mount_arguments(create(), scratch, empty_mask=empty.fileno()) as (arguments, inherited),
    ):
        for name in ("linked", ".aws"):
            path = NATIVE_SCRATCH_PATH / name
            index = arguments.index(str(path))
            assert arguments[index - 2] == "--ro-bind-fd"
            assert int(arguments[index - 1]) in inherited
        index = arguments.index(str(NATIVE_SCRATCH_PATH / ".ssh"))
        assert arguments[index - 1] == "--tmpfs"


def test_mount_arguments_maps_virtual_alias_targets_inside_scratch(binding, tmp_path):
    """A namespace alias to private scratch cannot expose the original host spelling."""
    create, _, scratch, _ = binding
    parent = tmp_path / "mappings"
    parent.mkdir(mode=0o700)
    alias = parent / "r0"
    alias.symlink_to(scratch)
    request = create(aliases=(("$TMPDIR", alias, scratch),))
    with mount_arguments(request, scratch) as (arguments, _):
        assert arguments[-3:] == ("--symlink", str(NATIVE_SCRATCH_PATH), "/.loop-private-paths/r0")


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
    ) as (arguments, inherited):
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


@pytest.mark.parametrize("mounted", [False, True])
def test_mount_arguments_validates_alias_targets_and_preserves_original_links(
    binding, tmp_path, mounted
):
    """Pin external alias parents read-only and preserve original link inodes."""
    create, workspace, scratch, _ = binding
    external = tmp_path / "external"
    external.mkdir()
    parent = tmp_path / "aliases"
    parent.mkdir(mode=0o700)
    alias = parent / "workspace"
    alias.symlink_to(external, target_is_directory=True)
    interior = workspace / "approved-read"
    interior.symlink_to(external, target_is_directory=True)
    request = create(
        aliases=(("$WORKSPACE_ROOT", alias, external),),
        read_roots=(external,) if mounted else (),
        read_aliases=(interior,) if mounted else (),
    )
    if not mounted:
        with (
            pytest.raises(ValueError, match="alias target is outside"),
            mount_arguments(request, scratch),
        ):
            pytest.fail("unmounted alias target produced mount arguments")
    else:
        with mount_arguments(request, scratch) as (arguments, _):
            assert str(parent) in arguments
            index = arguments.index(str(parent))
            assert arguments[index - 2] == "--ro-bind-fd"
            assert ("--symlink", str(external), "/.loop-private-paths/r0") == arguments[-3:]
            assert str(interior) not in arguments
        assert interior.resolve() == external


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
    responses = iter((True, True, True, False))
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
    with mount_arguments(request, scratch) as (arguments, _):
        index = arguments.index("--symlink")
        assert arguments[index : index + 3] == ("--symlink", str(runtime), str(alias))
        assert arguments[-5:-3] == ("--dir", str(NATIVE_SCRATCH_PATH))
        assert arguments[-3] == "--bind-fd"
        assert arguments[-1] == str(NATIVE_SCRATCH_PATH)


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

    with mount_arguments(request, scratch, skip_path_roots=(alias,)) as (arguments, _):
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
    alias = tmp_path / "scratch-alias"
    alias.symlink_to(scratch, target_is_directory=True)
    private = scratch / ".ssh"
    private.mkdir()
    (private / "key").write_text("secret")
    value = create(
        environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)},
        aliases=(("$TMPDIR", alias, scratch),),
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
