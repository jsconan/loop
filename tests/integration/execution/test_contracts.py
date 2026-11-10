"""Exercise host approval identities at native filesystem boundaries."""

import tempfile
from pathlib import Path

import pytest

from loop.execution.sandbox import SandboxRequest

pytestmark = pytest.mark.integration


def request(tmp_path, **changes):
    """Create an approved request rooted in a disposable workspace."""
    values = {
        "source": "printf ok",
        "cwd": tmp_path,
        "workspace": tmp_path,
        "read_roots": (),
        "write_roots": (tmp_path,),
        "network": False,
        "environment": {"PATH": "/usr/bin:/bin"},
        "policy_version": "macos-v1",
        "deadline": 160.0,
        "workspace_id": "workspace-1",
    }
    values.update(changes)
    return SandboxRequest.create(**values)


def test_generated_environment_rejects_symlink_escape(tmp_path):
    """Generated paths cannot resolve through a symlink outside private scratch."""
    with tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as directory:
        scratch = Path(directory).resolve()
        alias = scratch / "cache"
        alias.symlink_to(tmp_path, target_is_directory=True)

        with pytest.raises(ValueError, match="Generated command environment"):
            request(
                tmp_path,
                environment={"TMPDIR": str(scratch), "CACHE": str(alias)},
                generated_environment=("TMPDIR", "CACHE"),
            )


def test_generated_environment_detects_symlink_retargeting(tmp_path):
    """A generated path retargeted outside scratch invalidates the request."""
    with tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as directory:
        scratch = Path(directory).resolve()
        inside = scratch / "inside"
        inside.mkdir()
        alias = scratch / "cache"
        alias.symlink_to(inside, target_is_directory=True)
        approved = request(
            tmp_path,
            environment={"TMPDIR": str(scratch), "CACHE": str(alias)},
            generated_environment=("TMPDIR", "CACHE"),
        )
        assert approved.paths_are_current()

        alias.unlink()
        alias.symlink_to(tmp_path, target_is_directory=True)

        assert not approved.paths_are_current()


def test_host_runtime_signature_normalizes_generated_scratch(tmp_path):
    """Fresh generated scratch roots can reuse the same host rule."""
    with (
        tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as first,
        tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as second,
    ):
        requests = [
            request(
                tmp_path,
                source="ls $TMPDIR",
                environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(Path(path).resolve())},
                generated_environment=("TMPDIR",),
            )
            for path in (first, second)
        ]
        assert all(item.paths_are_current() for item in requests)
        assert (
            requests[0].host_command_runtime_signature()
            == requests[1].host_command_runtime_signature()
        )


@pytest.mark.parametrize("kind", ["workspace", "cwd", "read", "write", "path", "read_alias"])
def test_host_runtime_signature_rejects_replaced_bound_directories(tmp_path, kind):
    """Rebinding an approved directory cannot reuse the preceding host authority."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    directory = (
        workspace
        if kind == "workspace"
        else workspace / "bound"
        if kind == "cwd"
        else tmp_path / "bound"
    )
    if directory != workspace:
        directory.mkdir(mode=0o700)
    values = {"cwd": workspace, "write_roots": (workspace,), "read_roots": ()}
    if kind == "workspace":
        pass
    elif kind == "cwd":
        values["cwd"] = directory
    elif kind == "read":
        values["read_roots"] = (directory,)
    elif kind == "write":
        values["write_roots"] = (workspace, directory)
    elif kind == "path":
        values["environment"] = {"PATH": str(directory)}
    else:
        target = tmp_path / "tool-root"
        target.mkdir()
        alias = directory / "tool"
        alias.symlink_to(target, target_is_directory=True)
        values["read_roots"] = (target,)
        values["read_aliases"] = (alias,)

    first = request(workspace, **values)
    original_signature = first.host_command_runtime_signature()
    directory.rename(directory.with_name("old-bound"))
    directory.mkdir(mode=0o700)
    if kind == "read_alias":
        alias.symlink_to(target, target_is_directory=True)
    second = request(workspace, **values)

    assert not first.paths_are_current()
    assert second.paths_are_current()
    assert second.host_command_runtime_signature() != original_signature


@pytest.mark.parametrize(
    "scope", ["workspace", "cwd", "read", "write", "path", "read_alias_parent"]
)
def test_host_runtime_signature_binds_authorized_directory_identity(tmp_path, scope):
    """An authorized directory remains bound to its original identity."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    parent = (
        workspace
        if scope == "workspace"
        else workspace / "bound"
        if scope == "cwd"
        else tmp_path / "bound"
    )
    if parent != workspace:
        parent.mkdir(mode=0o700)
    alias = parent / "workspace"
    alias.symlink_to(workspace, target_is_directory=True)
    values = {
        "cwd": parent if scope == "cwd" else workspace,
        "read_roots": (parent,) if scope == "read" else (),
        "write_roots": (workspace, parent) if scope == "write" else (workspace,),
    }
    if scope == "path":
        values["environment"] = {"PATH": str(parent)}
    if scope == "read_alias_parent":
        target = tmp_path / "tool-root"
        target.mkdir()
        read_alias = parent / "tool"
        read_alias.symlink_to(target, target_is_directory=True)
        values["read_roots"] = (target,)
        values["read_aliases"] = (read_alias,)

    first = request(workspace, **values)
    original_signature = first.host_command_runtime_signature()
    parent.rename(parent.with_name("old-bound"))
    parent.mkdir(mode=0o700)
    alias.symlink_to(workspace, target_is_directory=True)
    if scope == "read_alias_parent":
        read_alias.symlink_to(target, target_is_directory=True)
    second = request(workspace, **values)

    assert not first.paths_are_current()
    assert second.paths_are_current()
    assert second.host_command_runtime_signature() != original_signature


@pytest.mark.parametrize("scope", ["read", "write", "path", "read_alias_parent", "scratch_root"])
def test_host_runtime_signature_binds_generated_scratch_when_also_authorized(tmp_path, scope):
    """Generated scratch remains identity-bound when it also grants independent authority."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with (
        tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as directory,
        tempfile.TemporaryDirectory() as holding,
    ):
        scratch = Path(directory).resolve()
        values = {
            "environment": {"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)},
            "generated_environment": ("TMPDIR",),
            "read_roots": (scratch,) if scope == "read" else (),
            "write_roots": (workspace, scratch) if scope == "write" else (workspace,),
        }
        if scope == "path":
            values["environment"]["PATH"] = str(scratch)
        if scope == "read_alias_parent":
            target = tmp_path / "tool-root"
            target.mkdir()
            alias = scratch / "tool"
            alias.symlink_to(target, target_is_directory=True)
            values["read_roots"] = (target,)
            values["read_aliases"] = (alias,)
        if scope == "scratch_root":
            values["scratch_root"] = scratch

        first = request(workspace, **values)
        original_signature = first.host_command_runtime_signature()
        scratch.rename(Path(holding) / "old")
        scratch.mkdir(mode=0o700)
        if scope == "read_alias_parent":
            alias.symlink_to(target, target_is_directory=True)
        second = request(workspace, **values)

        assert not first.paths_are_current()
        assert second.paths_are_current()
        assert second.host_command_runtime_signature() != original_signature


@pytest.mark.parametrize("overlap", ["ancestor", "descendant"])
def test_host_runtime_signature_binds_overlapping_generated_directory(tmp_path, overlap):
    """Overlapping read authority binds generated directories in either direction."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    parent = tmp_path / "authority"
    parent.mkdir(mode=0o700)
    generated = parent / "generated"
    generated.mkdir(mode=0o700)
    read_root = parent if overlap == "ancestor" else generated / "read-root"
    if overlap == "descendant":
        read_root.mkdir(mode=0o700)
    scratch = generated
    values = {
        "environment": {"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)},
        "generated_environment": ("TMPDIR",),
        "read_roots": (read_root,),
        "scratch_root": scratch,
    }
    first = request(workspace, **values)
    original_signature = first.host_command_runtime_signature()

    old = parent / "old-generated"
    generated.rename(old)
    generated.mkdir(mode=0o700)
    if overlap == "descendant":
        (old / "read-root").rename(read_root)
    second = request(workspace, **values)

    assert not first.paths_are_current()
    assert second.paths_are_current()
    assert second.host_command_runtime_signature() != original_signature


def test_host_runtime_signature_normalizes_replaced_pure_generated_scratch(tmp_path):
    """A fresh generated scratch remains reusable when it has no independent authority."""
    with (
        tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as directory,
        tempfile.TemporaryDirectory() as holding,
    ):
        scratch = Path(directory).resolve()
        values = {
            "environment": {"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)},
            "generated_environment": ("TMPDIR",),
        }
        first = request(tmp_path, **values)
        original_signature = first.host_command_runtime_signature()
        scratch.rename(Path(holding) / "old")
        scratch.mkdir(mode=0o700)
        second = request(tmp_path, **values)

        assert not first.paths_are_current()
        assert second.paths_are_current()
        assert second.host_command_runtime_signature() == original_signature


def test_host_runtime_signature_binds_replaced_read_alias(tmp_path):
    """A replaced executable read symlink cannot reuse the old host authority."""
    target = tmp_path / "tool-root"
    target.mkdir()
    alias = tmp_path / "tool-alias"
    alias.symlink_to(target, target_is_directory=True)
    values = {"read_roots": (target,), "read_aliases": (alias,)}
    first = request(tmp_path, **values)
    original_signature = first.host_command_runtime_signature()
    alias.rename(alias.with_name("old-tool-alias"))
    alias.symlink_to(target, target_is_directory=True)
    second = request(tmp_path, **values)

    assert not first.paths_are_current()
    assert second.paths_are_current()
    assert second.host_command_runtime_signature() != original_signature


def test_host_runtime_signature_binds_replaced_read_alias_target(tmp_path):
    """A read alias cannot silently bind to a replacement target directory."""
    root = tmp_path / "tool-root"
    root.mkdir()
    target = root / "version"
    target.mkdir()
    alias = tmp_path / "tool-alias"
    alias.symlink_to(target, target_is_directory=True)
    values = {"read_roots": (root,), "read_aliases": (alias,)}
    first = request(tmp_path, **values)
    original_signature = first.host_command_runtime_signature()
    target.rename(root / "old-version")
    target.mkdir()
    second = request(tmp_path, **values)

    assert not first.paths_are_current()
    assert second.paths_are_current()
    assert second.host_command_runtime_signature() != original_signature
