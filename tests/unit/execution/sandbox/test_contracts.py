"""Verify exact sandbox request identities and typed outcome boundaries."""

import ast
import tempfile
import time
from pathlib import Path

import pytest

from loop.execution.sandbox import (
    CommandProcessResult,
    DirectoryIdentity,
    SandboxOutcome,
    SandboxRequest,
    resolve_host_executable,
)


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
        "deadline": time.monotonic() + 60,
        "workspace_id": "workspace-1",
    }
    values.update(changes)
    return SandboxRequest.create(**values)


def test_request_binds_exact_environment_and_directory_identity(tmp_path):
    """A request retains sanitized environment and detects a replaced approved root."""
    approved = request(tmp_path, environment={"Z": "last", "A": "first"})

    assert approved.environment == (("A", "first"), ("Z", "last"))
    assert approved.shell_environment() == {"A": "first", "Z": "last"}
    assert approved.paths_are_current()

    tmp_path.rename(tmp_path.with_name("old-workspace"))
    tmp_path.mkdir()
    assert not approved.paths_are_current()


def test_request_binds_and_validates_active_instruction_names(tmp_path):
    """Native policy names are bound to the request and reject path or regex injection."""
    first = request(tmp_path, protected_instruction_names=("REVIEW.md",))
    second = request(tmp_path, protected_instruction_names=("NOTES.md",))

    assert first.protected_instruction_names == ("REVIEW.md",)
    assert first.host_command_runtime_signature() != second.host_command_runtime_signature()
    for invalid in ("../REVIEW.md", "a/b.md", "a\\b.md", "", ".", "a\x00b"):
        with pytest.raises(ValueError, match="plain filenames"):
            request(tmp_path, protected_instruction_names=(invalid,))


def test_sandbox_command_signature_normalizes_scratch_and_binds_scope(tmp_path):
    """Sandbox rule signatures ignore private scratch paths but bind their policy identity."""
    with (
        tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as first_directory,
        tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as second_directory,
    ):
        first_scratch = Path(first_directory).resolve()
        second_scratch = Path(second_directory).resolve()
        first = request(
            tmp_path,
            generated_environment=("TMPDIR", "XDG_CACHE_HOME", "RUFF_CACHE_DIR", "COVERAGE_FILE"),
            environment={
                "TMPDIR": str(first_scratch),
                "XDG_CACHE_HOME": str(first_scratch / "cache"),
                "RUFF_CACHE_DIR": str(first_scratch / "ruff-cache"),
                "COVERAGE_FILE": str(first_scratch / ".coverage"),
            },
        )
        second = request(
            tmp_path,
            generated_environment=("TMPDIR", "XDG_CACHE_HOME", "RUFF_CACHE_DIR", "COVERAGE_FILE"),
            environment={
                "TMPDIR": str(second_scratch),
                "XDG_CACHE_HOME": str(second_scratch / "cache"),
                "RUFF_CACHE_DIR": str(second_scratch / "ruff-cache"),
                "COVERAGE_FILE": str(second_scratch / ".coverage"),
            },
        )

        assert first.sandbox_command_signature("workspace-1") == second.sandbox_command_signature(
            "workspace-1"
        )
        assert first.sandbox_command_signature("workspace-1") != first.sandbox_command_signature(
            "user"
        )
        user_chosen = request(
            tmp_path,
            environment=dict(first.environment),
        )
        assert first.sandbox_command_signature(
            "workspace-1"
        ) != user_chosen.sandbox_command_signature("workspace-1")
        with pytest.raises(ValueError, match="Generated command environment"):
            request(
                tmp_path,
                environment={"TMPDIR": str(first_scratch), "CACHE": str(tmp_path / "cache")},
                generated_environment=("CACHE",),
            )


def test_host_command_signature_separates_virtual_rule_from_runtime_authority(tmp_path):
    """Virtual rules remain portable while real changes stay bound in session memory."""
    with (
        tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as first_directory,
        tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as second_directory,
    ):
        first_scratch = Path(first_directory).resolve()
        second_scratch = Path(second_directory).resolve()
        first = request(
            tmp_path,
            environment={
                "PATH": "/usr/bin:/bin",
                "TMPDIR": str(first_scratch),
                "XDG_CACHE_HOME": str(first_scratch / "cache"),
            },
            generated_environment=("TMPDIR", "XDG_CACHE_HOME"),
        )
        second = request(
            tmp_path,
            environment={
                "PATH": "/usr/bin:/bin",
                "TMPDIR": str(second_scratch),
                "XDG_CACHE_HOME": str(second_scratch / "cache"),
            },
            generated_environment=("TMPDIR", "XDG_CACHE_HOME"),
        )
        assert first.host_command_signature() == second.host_command_signature()
        assert first.host_command_runtime_signature() == second.host_command_runtime_signature()
        user_selected_first = request(
            tmp_path, environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(first_scratch)}
        )
        user_selected_second = request(
            tmp_path, environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(second_scratch)}
        )
        assert (
            user_selected_first.host_command_runtime_signature()
            != user_selected_second.host_command_runtime_signature()
        )
        assert (
            first.host_command_signature()
            != request(
                tmp_path,
                source="printf changed",
                environment=dict(first.environment),
                generated_environment=("TMPDIR", "XDG_CACHE_HOME"),
            ).host_command_signature()
        )
        assert (
            first.host_command_runtime_signature()
            != request(
                tmp_path,
                environment={**dict(first.environment), "USER_VALUE": "changed"},
                generated_environment=("TMPDIR", "XDG_CACHE_HOME"),
            ).host_command_runtime_signature()
        )
        assert (
            first.host_command_signature()
            != request(
                tmp_path,
                environment=dict(first.environment),
                generated_environment=("TMPDIR", "XDG_CACHE_HOME"),
                network=True,
            ).host_command_signature()
        )


def test_host_command_signature_normalizes_verified_virtual_aliases(tmp_path):
    """Fresh private aliases for one virtual root do not invalidate exact reuse."""
    with (
        tempfile.TemporaryDirectory(prefix="loop-vpath-") as first_directory,
        tempfile.TemporaryDirectory(prefix="loop-vpath-") as second_directory,
    ):
        first_alias = Path(first_directory).resolve() / "r0"
        second_alias = Path(second_directory).resolve() / "r0"
        first_alias.symlink_to(tmp_path, target_is_directory=True)
        second_alias.symlink_to(tmp_path, target_is_directory=True)
        first = request(
            tmp_path,
            source="cat /workspace/file",
            execution_source=f"cat {first_alias}/file",
            aliases=(("/workspace", first_alias, tmp_path),),
        )
        second = request(
            tmp_path,
            source="cat /workspace/file",
            execution_source=f"cat {second_alias}/file",
            aliases=(("/workspace", second_alias, tmp_path),),
        )
        assert first.host_command_signature() == second.host_command_signature()
        assert first.host_command_runtime_signature() == second.host_command_runtime_signature()


def test_request_binds_only_private_loop_scratch(tmp_path):
    """A live private scratch is identity-bound without a platform-specific name."""
    with tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as directory:
        scratch = Path(directory).resolve(strict=True)
        approved = request(tmp_path, environment={"TMPDIR": str(scratch)})
        assert approved.paths_are_current()
    assert not approved.paths_are_current()
    with tempfile.TemporaryDirectory(prefix="loop-other-") as directory:
        assert request(tmp_path, environment={"TMPDIR": str(Path(directory).resolve())})
    with pytest.raises(ValueError, match="private real directory"):
        request(tmp_path, environment={"TMPDIR": str(tmp_path)})


def test_request_binds_search_directories_and_missing_entries(tmp_path, monkeypatch):
    """A PATH alias or newly created search directory invalidates prior approval."""
    search = tmp_path / "search"
    search.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(search, target_is_directory=True)
    missing = tmp_path / "missing"
    approved = request(tmp_path, environment={"PATH": f"{alias}:{missing}"})
    assert approved.path_roots == ((alias, search),)
    assert approved.paths_are_current()
    missing.mkdir()
    assert not approved.paths_are_current()
    missing.rmdir()
    alias.unlink()
    alias.symlink_to(tmp_path, target_is_directory=True)
    assert not approved.paths_are_current()
    alias.unlink()
    alias.symlink_to(search, target_is_directory=True)
    original_is_dir = Path.is_dir

    def inaccessible(path):
        """Simulate a search alias that becomes uninspectable at launch."""
        if path == alias:
            raise OSError("search alias inaccessible")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", inaccessible)
    assert not approved.paths_are_current()


def test_request_binds_empty_and_relative_search_directories_to_cwd(tmp_path):
    """Resolve empty and relative PATH entries from the command's working directory."""
    workspace = tmp_path / "workspace"
    cwd = workspace / "nested"
    local_bin = cwd / "bin"
    external_bin = tmp_path / "external-bin"
    local_bin.mkdir(parents=True)
    external_bin.mkdir()

    approved = request(
        workspace,
        cwd=cwd,
        environment={"PATH": ":bin:../../external-bin"},
    )

    assert approved.path_roots == (
        (cwd, cwd),
        (local_bin, local_bin),
        (external_bin, external_bin),
    )
    assert approved.paths_are_current()


def test_request_rejects_relative_cwd_for_search_roots(tmp_path):
    """Reject a relative request cwd before it can define PATH authority."""
    with pytest.raises(ValueError, match="absolute working directory"):
        request(tmp_path, cwd=Path("."), environment={"PATH": "bin"})


def test_request_ignores_missing_relative_search_root_until_it_appears(tmp_path):
    """A missing relative PATH directory adds no authority and invalidates if created."""
    approved = request(tmp_path, environment={"PATH": "missing-bin"})
    assert approved.path_roots == ()

    (tmp_path / "missing-bin").mkdir()

    assert not approved.paths_are_current()


def test_public_directory_and_executable_binding_reject_invalid_directories(tmp_path):
    """Directory identities and executable lookup require valid real working directories."""
    with pytest.raises(ValueError, match="must be absolute"):
        DirectoryIdentity.capture(Path("relative"))

    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="working directory is unavailable"):
        resolve_host_executable("x", "/usr/bin:/bin", missing)

    file_path = tmp_path / "file"
    file_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a directory"):
        resolve_host_executable("x", "/usr/bin:/bin", file_path)


def test_request_paths_fail_closed_when_path_roots_cannot_be_revalidated(tmp_path, monkeypatch):
    """A failed PATH identity recheck makes a bound request stale."""
    approved = request(tmp_path)

    def unavailable(_environment, _cwd):
        """Simulate PATH identity metadata becoming unavailable before launch."""
        raise ValueError("unavailable")

    monkeypatch.setattr(
        "loop.execution.sandbox.contracts.path_search_roots",
        unavailable,
    )

    assert not approved.paths_are_current()


def test_request_binds_private_alias_and_translated_source(tmp_path):
    """A translated command is bound to a private symlink and its canonical target."""
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    alias = private / "r0"
    alias.symlink_to(tmp_path, target_is_directory=True)
    approved = request(
        tmp_path,
        source="cat /workspace/file",
        execution_source=f"cat {alias}/file",
        aliases=(("/workspace", alias, tmp_path),),
    )
    assert approved.execution_source == f"cat {alias}/file"
    assert approved.paths_are_current()
    alias.unlink()
    alias.symlink_to(private, target_is_directory=True)
    assert not approved.paths_are_current()
    alias.unlink()
    assert not approved.paths_are_current()


def test_request_binds_external_read_alias_and_rejects_retargeting(tmp_path, monkeypatch):
    """An approved directory symlink keeps its inode and canonical read target until launch."""
    installation = tmp_path / "installation"
    installation.mkdir()
    alias = tmp_path / "opt-tool"
    alias.symlink_to(installation, target_is_directory=True)
    approved = request(tmp_path, read_roots=(installation,), read_aliases=(alias,))
    assert approved.paths_are_current()
    alias.rename(tmp_path / "original-alias")
    assert not approved.paths_are_current()
    alias.symlink_to(installation, target_is_directory=True)
    assert not approved.paths_are_current()

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="Invalid executable read alias"):
        request(tmp_path, read_roots=(other,), read_aliases=(alias,))
    alias.unlink()
    alias.write_text("not a link")
    with pytest.raises(ValueError, match="Invalid executable read alias"):
        request(tmp_path, read_roots=(installation,), read_aliases=(alias,))

    original_lstat = Path.lstat

    def unavailable(path):
        """Model an alias disappearing between authorization and revalidation."""
        if path == alias:
            raise OSError("missing")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", unavailable)
    assert not approved.paths_are_current()


def test_request_revalidates_exact_executable_before_launch(tmp_path):
    """A binary replaced after approval cannot run under its old grant identity."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    executable = binaries / "sampletool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    details = executable.stat()
    identity = ("sampletool", executable, executable, details.st_dev, details.st_ino)
    approved = request(
        tmp_path, environment={"PATH": str(binaries)}, executable_identities=(identity,)
    )
    assert approved.paths_are_current()
    with pytest.raises(ValueError, match="Executable identity changed"):
        request(tmp_path, environment={"PATH": "/usr/bin:/bin"}, executable_identities=(identity,))
    executable.rename(binaries / "original-executable")
    assert not approved.paths_are_current()
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    assert not approved.paths_are_current()


def test_automatic_tool_reads_require_bound_executable_and_approved_root(tmp_path):
    """A prompt-free tool root cannot be fabricated without its executable binding."""
    installation = tmp_path / "installation"
    installation.mkdir()
    executable = installation / "sampletool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    details = executable.stat()
    identity = ("sampletool", executable, executable, details.st_dev, details.st_ino)
    with pytest.raises(ValueError, match="Automatic tool reads require"):
        request(tmp_path, read_roots=(installation,), automatic_tool_reads=(installation,))
    with pytest.raises(ValueError, match="Automatic tool reads require"):
        request(
            tmp_path,
            environment={"PATH": str(installation)},
            executable_identities=(identity,),
            automatic_tool_reads=(installation,),
        )
    approved = request(
        tmp_path,
        read_roots=(installation,),
        environment={"PATH": str(installation)},
        executable_identities=(identity,),
        automatic_tool_reads=(installation,),
    )
    assert approved.automatic_tool_reads == (installation,)


def test_request_rejects_unsafe_aliases(tmp_path):
    """Unbound, unsafe, and nonprivate symlinks cannot extend native authority."""
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    alias = private / "r0"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        request(tmp_path, execution_source="")
    for item in (
        ("workspace", alias, tmp_path),
        ("/workspace", alias, private),
        ("/workspace", Path("relative"), tmp_path),
    ):
        with pytest.raises(ValueError):
            request(tmp_path, aliases=(item,))
    private.chmod(0o755)
    with pytest.raises(ValueError):
        request(tmp_path, aliases=(("/workspace", alias, tmp_path),))


@pytest.mark.parametrize(
    "change",
    [
        {"source": ""},
        {"policy_version": ""},
        {"workspace_id": ""},
        {"deadline": 0},
        {"environment": {"BAD=NAME": "value"}},
        {"environment": {"PATH": "bad\x00value"}},
    ],
)
def test_request_rejects_invalid_authority(tmp_path, change):
    """Malformed command authority never produces an approvable request."""
    with pytest.raises(ValueError):
        request(tmp_path, **change)


def test_request_rejects_nonabsolute_symlink_and_outside_cwd(tmp_path):
    """Directory grants require real absolute roots and a cwd inside the workspace."""
    alias = tmp_path.with_name("alias")
    alias.symlink_to(tmp_path)
    outside = tmp_path.with_name("outside")
    outside.mkdir()
    regular = tmp_path / "regular"
    regular.write_text("data")
    ancestor_alias = tmp_path.with_name("ancestor-alias")
    ancestor_alias.symlink_to(tmp_path, target_is_directory=True)

    for change in (
        {"cwd": Path("relative")},
        {"write_roots": (alias,)},
        {"read_roots": (regular,)},
        {"read_roots": (ancestor_alias,)},
        {"cwd": outside},
    ):
        with pytest.raises(ValueError):
            request(tmp_path, **change)


def test_request_rejects_missing_root_and_missing_after_approval(tmp_path):
    """Missing roots fail preparation and disappearing approved roots fail revalidation."""
    missing = tmp_path / "missing"
    with pytest.raises(OSError):
        request(tmp_path, read_roots=(missing,))

    present = tmp_path / "present"
    present.mkdir()
    approved = request(tmp_path, read_roots=(present,))
    present.rmdir()
    assert not approved.paths_are_current()


def test_missing_git_creation_target_is_bound_until_launch(tmp_path):
    """A fresh Git grant permits absence but fails if metadata appears before launch."""
    approved = request(tmp_path, git_create=True)
    assert approved.paths_are_current()
    (tmp_path / ".git").mkdir()
    assert not approved.paths_are_current()
    with pytest.raises(ValueError, match="absent"):
        request(tmp_path, git_create=True)
    (tmp_path / ".git").rmdir()
    (tmp_path / ".git").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        request(tmp_path, git_create=True)
    (tmp_path / ".git").unlink()
    with pytest.raises(ValueError, match="workspace write"):
        request(tmp_path, git_create=True, write_roots=())


def test_sandbox_results_distinguish_exit_status_from_boundary_failure():
    """Only completed commands carry an exit code, including nonzero shell exits."""
    assert CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=17).exit_code == 17
    with pytest.raises(ValueError):
        CommandProcessResult(SandboxOutcome.DENIED, exit_code=1)
    with pytest.raises(ValueError):
        CommandProcessResult(SandboxOutcome.COMPLETED)


def test_sandbox_contract_has_no_host_execution_dependency():
    """Sandbox contract imports no host launcher or application retry authority."""
    source = Path(__file__).parents[4] / "src/loop/execution/sandbox/contracts.py"
    tree = ast.parse(source.read_text())
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    names.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    assert not any("host" in name or "application" in name for name in names)
