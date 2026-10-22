"""Verify exact sandbox request identities and typed outcome boundaries."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from loop import constants
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


def test_request_binds_and_validates_file_creation_mask(tmp_path):
    """The caller umask is retained and malformed permission masks are rejected."""
    first = request(tmp_path, file_creation_mask=0o022)
    second = request(tmp_path, file_creation_mask=0o077)

    assert first.file_creation_mask == 0o022
    assert first.host_command_runtime_signature() != second.host_command_runtime_signature()
    for invalid in (True, -1, 0o1000, "022"):
        with pytest.raises(ValueError, match="file creation mask"):
            request(tmp_path, file_creation_mask=invalid)


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


def test_request_wire_roundtrip_preserves_all_authority_and_correlation(tmp_path):
    """Canonical snapshots restore exact source, grants, identities, deadline and attempt ID."""
    approved = request(
        tmp_path,
        source="printf 'é\\n'",
        execution_source="printf translated",
        read_context="Read tools",
        protected_instruction_names=("RULES.md",),
    )
    wire = approved.to_wire()
    assert json.loads(wire)["schema"] == 1
    restored = SandboxRequest.from_wire(wire)
    assert restored == approved
    assert restored.protected_paths == approved.protected_paths
    assert restored.shell_environment() == approved.shell_environment()
    assert restored.host_command_runtime_signature() == approved.host_command_runtime_signature()


def test_request_wire_roundtrip_keeps_alias_and_executable_inodes(tmp_path, monkeypatch):
    """Executable and virtual/read alias identities survive restoration without broader reads."""
    workspace = tmp_path / "workspace"
    tool = tmp_path / "tool"
    private = tmp_path / "private"
    for path in (workspace, tool, private):
        path.mkdir(mode=0o700)
    binary = tool / "runner"
    binary.write_text("tool")
    binary.chmod(0o755)
    read_alias = workspace / "read_alias"
    read_alias.symlink_to(tool, target_is_directory=True)
    alias = private / "workspace"
    alias.symlink_to(workspace, target_is_directory=True)
    details = binary.stat()
    approved = request(
        workspace,
        read_roots=(tool,),
        environment={"PATH": str(tool)},
        aliases=(("/workspace", alias, workspace),),
        read_aliases=(read_alias,),
        executable_identities=(("runner", binary, binary, details.st_dev, details.st_ino),),
        automatic_tool_reads=(tool,),
    )
    assert SandboxRequest.from_wire(approved.to_wire()) == approved


@pytest.mark.parametrize("change", ["deadline", "stale", "size"])
def test_request_wire_encoding_refuses_unsafe_authority(tmp_path, change):
    """Nonfinite deadlines, replaced directories and oversized snapshots refuse serialization."""
    approved = request(
        tmp_path,
        deadline=float("inf") if change == "deadline" else time.monotonic() + 60,
        source="x" * 65536 if change == "size" else "printf ok",
    )
    if change == "stale":
        tmp_path.rename(tmp_path.with_name("old-wire-root"))
        tmp_path.mkdir()
    with pytest.raises(ValueError, match="stale|admission bound"):
        approved.to_wire()


@pytest.mark.parametrize(
    "data", [None, "text", b"", b"x" * 65537, b"\xff", b"[" * 1100 + b"]" * 1100, b"{"]
)
def test_request_wire_invalid_bytes_refuse_before_restoration(data):
    """Wrong type, bounds, malformed UTF-8/JSON and excessive nesting cannot restore authority."""
    with pytest.raises(ValueError):
        SandboxRequest.from_wire(data)


@pytest.mark.parametrize(
    "envelope",
    [
        None,
        [],
        {},
        {"schema": True, "request": {}},
        {"schema": 2, "request": {}},
        {"schema": 3, "request": {}},
        {"schema": 1, "request": []},
        {"schema": 1, "request": {}},
    ],
)
def test_request_wire_unknown_or_ambiguous_schema_refuses(envelope):
    """Only the exact version-one envelope and complete shared request fields are accepted."""
    with pytest.raises(ValueError, match="request schema"):
        SandboxRequest.from_wire(json.dumps(envelope).encode())


def test_request_wire_duplicate_keys_refuse(tmp_path):
    """Duplicate authority fields are rejected instead of choosing the last value."""
    data = request(tmp_path).to_wire()
    data = data.replace(b'"network":false', b'"network":false,"network":true')
    with pytest.raises(ValueError, match="Duplicate sandbox request field"):
        SandboxRequest.from_wire(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("deadline", True),
        ("deadline", "20"),
        ("deadline", float("inf")),
        ("deadline", float("nan")),
        ("network", 1),
        ("git_create", "false"),
        ("source", None),
        ("source", "x\x00y"),
        ("read_context", []),
        ("attempt_id", "INVALID"),
        ("attempt_id", "00000000-0000-0000-0000-00000000000A"),
        ("file_creation_mask", True),
        ("file_creation_mask", -1),
        ("file_creation_mask", 0o1000),
        ("read_roots", {}),
        ("cwd", 1),
        ("environment", ["PATH"]),
        ("environment", [["PATH"]]),
        ("environment", [["PATH", 0]]),
        ("aliases", [["/workspace", 0, "/workspace"]]),
        ("executable_identities", [["python", "/bin/python", "/bin/python", True, 1]]),
        ("executable_identities", [["python", "/bin/python", "/bin/python", -1, 1]]),
    ],
)
def test_request_wire_invalid_typed_fields_refuse(tmp_path, field, value):
    """Request field types and row widths cannot be coerced into broader execution authority."""
    envelope = json.loads(request(tmp_path).to_wire())
    envelope["request"][field] = value
    with pytest.raises(ValueError):
        SandboxRequest.from_wire(json.dumps(envelope).encode())


@pytest.mark.parametrize("change", ["directory", "snapshot", "environment", "alias"])
def test_request_wire_changed_original_snapshot_refuses(tmp_path, change):
    """Restoration never recaptures a new directory or normalizes changed snapshot authority."""
    approved = request(tmp_path)
    envelope = json.loads(approved.to_wire())
    if change == "directory":
        tmp_path.rename(tmp_path.with_name("original-wire-root"))
        tmp_path.mkdir()
    elif change == "snapshot":
        envelope["request"]["_identities"][0]["inode"] += 1
    elif change == "environment":
        envelope["request"]["environment"].append(["PATH", "/usr/bin:/bin"])
    else:
        envelope["request"]["_alias_identities"] = [[str(tmp_path), 1, 1, str(tmp_path)]]
    with pytest.raises(ValueError, match="snapshot differs"):
        SandboxRequest.from_wire(json.dumps(envelope).encode())


@pytest.mark.parametrize("late", [False, True])
def test_request_wire_restoration_never_resets_expired_deadline(tmp_path, monkeypatch, late):
    """The original monotonic budget applies before and after filesystem restoration."""
    approved = request(tmp_path, deadline=20)
    wire = approved.to_wire()
    clock = iter([10, 20] if late else [20])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    from loop.execution.sandbox.contracts import SandboxRequestExpired

    with pytest.raises(SandboxRequestExpired, match="restoration deadline") as failure:
        SandboxRequest.from_wire(wire)
    assert failure.value.request == approved


@pytest.mark.parametrize("change", ["schema", "boolean", "identity"])
def test_expired_request_does_not_hide_malformed_or_stale_authority(tmp_path, monkeypatch, change):
    """Expired budgets cannot turn corrupted request authority into an authenticated timeout."""
    approved = request(tmp_path, deadline=20)
    envelope = json.loads(approved.to_wire())
    if change == "schema":
        envelope["schema"] = True
    elif change == "boolean":
        envelope["request"]["network"] = 1
    else:
        envelope["request"]["_directory_identities"] = []
    monkeypatch.setattr(time, "monotonic", lambda: 21)
    with pytest.raises(ValueError, match="schema|boolean|snapshot differs"):
        SandboxRequest.from_wire(json.dumps(envelope).encode())


def test_request_wire_preserves_private_generated_environment(tmp_path):
    """Private scratch provenance and generated environment names remain exact after restoration."""
    with tempfile.TemporaryDirectory(prefix="loop-wire-") as directory:
        scratch = Path(directory).resolve()
        approved = request(
            tmp_path,
            environment={"TMPDIR": str(scratch), "XDG_CACHE_HOME": str(scratch / "cache")},
            generated_environment=("TMPDIR", "XDG_CACHE_HOME"),
        )
        assert SandboxRequest.from_wire(approved.to_wire()) == approved


@pytest.mark.parametrize(
    "name,path_value",
    [("x" * 128, ""), ("bad/name", ""), ("tool", "x" * 16385), ("tool", ":" * 128)],
)
def test_executable_wire_lookup_rejects_unbounded_or_unsafe_names(tmp_path, name, path_value):
    """Executable identities in restored requests retain bounded simple-name and PATH validation."""
    with pytest.raises(ValueError, match="simple name|PATH is too large"):
        resolve_host_executable(name, path_value, tmp_path)


def test_executable_lookup_skips_nonexecutable_and_nonregular_candidates(tmp_path):
    """A PATH entry cannot bind a directory or a nonexecutable file as command authority."""
    (tmp_path / "tool").mkdir()
    assert resolve_host_executable("tool", str(tmp_path), tmp_path) is None
    (tmp_path / "tool").rmdir()
    (tmp_path / "tool").write_text("not executable")
    assert resolve_host_executable("tool", str(tmp_path), tmp_path) is None


def test_sandbox_results_distinguish_exit_status_from_boundary_failure():
    """Only completed commands carry an exit code, including nonzero shell exits."""
    assert CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=17).exit_code == 17
    with pytest.raises(ValueError):
        CommandProcessResult(SandboxOutcome.DENIED, exit_code=1)
    with pytest.raises(ValueError):
        CommandProcessResult(SandboxOutcome.COMPLETED)


@pytest.mark.parametrize("outcome", list(SandboxOutcome))
def test_result_wire_preserves_typed_outcomes_and_attempt(outcome):
    """Each terminal outcome retains exact output, effects and its original attempt binding."""
    attempt = "1248c892-42dc-4b68-a3fb-30d2a70a93a7"
    result = CommandProcessResult(
        outcome,
        exit_code=-9 if outcome is SandboxOutcome.COMPLETED else None,
        stdout="output\x00é😀",
        stderr="failure\n",
        stdout_discarded=None,
        stderr_discarded=17,
        possible_effects=True,
    )
    assert CommandProcessResult.from_wire(result.to_wire(attempt), attempt) == result


@pytest.mark.parametrize(
    "change",
    [
        "envelope",
        "schema",
        "schema_bool",
        "attempt",
        "result",
        "missing",
        "extra",
        "outcome_type",
        "outcome",
        "stdout_type",
        "stdout_bound",
        "detail_bound",
        "discard_type",
        "discard_negative",
        "discard_bound",
        "exit_type",
        "exit_bound",
        "effects",
        "inconsistent",
    ],
)
def test_result_wire_rejects_ambiguous_or_unbounded_results(change):
    """Malformed field types, bounds, attempt substitution and exit inconsistencies refuse."""
    attempt = "1248c892-42dc-4b68-a3fb-30d2a70a93a7"
    envelope = json.loads(CommandProcessResult(SandboxOutcome.UNAVAILABLE).to_wire(attempt))
    value = envelope["result"]
    if change == "envelope":
        envelope = []
    elif change == "schema":
        envelope["schema"] = 2
    elif change == "schema_bool":
        envelope["schema"] = True
    elif change == "attempt":
        envelope["attempt_id"] = "other"
    elif change == "result":
        envelope["result"] = []
    elif change == "missing":
        del value["detail"]
    elif change == "extra":
        value["other"] = "unknown"
    else:
        key, replacement = {
            "outcome_type": ("outcome", 1),
            "outcome": ("outcome", "unknown"),
            "stdout_type": ("stdout", []),
            "stdout_bound": ("stdout", "x" * 1_000_001),
            "detail_bound": ("detail", "x" * 2001),
            "discard_type": ("stdout_discarded", True),
            "discard_negative": ("stderr_discarded", -1),
            "discard_bound": ("stderr_discarded", 1 << 63),
            "exit_type": ("exit_code", True),
            "exit_bound": ("exit_code", 1 << 31),
            "effects": ("possible_effects", 1),
            "inconsistent": ("exit_code", 0),
        }[change]
        value[key] = replacement
    with pytest.raises(ValueError):
        CommandProcessResult.from_wire(json.dumps(envelope).encode(), attempt)


@pytest.mark.parametrize("data", [b"", "text", b"\xff", b'{"schema":1,"schema":1}', b"[" * 2000])
def test_result_wire_rejects_invalid_json_and_envelope(data):
    """Nonbytes, empty, duplicate, invalid UTF-8 and deeply nested input cannot supply a result."""
    with pytest.raises(ValueError):
        CommandProcessResult.from_wire(data, "1248c892-42dc-4b68-a3fb-30d2a70a93a7")


@pytest.mark.parametrize("attempt", [1, "bad", "1248C892-42DC-4B68-A3FB-30D2A70A93A7"])
def test_result_wire_requires_canonical_attempt_identity(attempt):
    """The expected attempt must itself be a canonical UUID before any result is accepted."""
    with pytest.raises(ValueError):
        CommandProcessResult.from_wire(b"{}", attempt)


def test_result_wire_bounds_bytes_and_maximum_unicode_output(monkeypatch):
    """Worst-case JSON escaping fits both full character limits and the explicit wire ceiling."""
    attempt = "1248c892-42dc-4b68-a3fb-30d2a70a93a7"
    monkeypatch.setattr(constants, "MAX_OUTPUT_CHARS", 2)
    monkeypatch.setattr(constants, "MAX_NATIVE_RESULT_BYTES", 24 * 2 + 131072)
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="😀😀", stderr="😀😀"
    )
    assert CommandProcessResult.from_wire(result.to_wire(attempt), attempt) == result
    with pytest.raises(ValueError, match="bounded sandbox result envelope"):
        CommandProcessResult.from_wire(b"x" * (24 * 2 + 131073), attempt)


def test_result_wire_encoding_refuses_changed_restoration(monkeypatch):
    """Encoding cannot emit an envelope whose restored value differs from its source result."""
    result = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    monkeypatch.setattr(CommandProcessResult, "from_wire", lambda *args: None)
    with pytest.raises(ValueError, match="differs after encoding"):
        result.to_wire("1248c892-42dc-4b68-a3fb-30d2a70a93a7")
