"""Verify exact sandbox request identities and typed outcome boundaries."""

import json
import os
import tempfile
import time
from dataclasses import fields, replace
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
from loop.execution.sandbox.contracts import (
    NATIVE_REQUEST_FIELDS,
    normalize_egress_origin,
    resolve_selected_executable,
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


def test_native_wire_field_declaration_matches_request():
    """Protocol capability discovery describes every serialized request field exactly once."""
    assert len(NATIVE_REQUEST_FIELDS) == len(set(NATIVE_REQUEST_FIELDS))
    assert set(NATIVE_REQUEST_FIELDS) == {field.name for field in fields(SandboxRequest)}


def test_cleanup_evidence_requires_a_negotiated_exact_result_wire_field(tmp_path):
    """Legacy results omit proof while a new receiver can attest completed cleanup."""
    approved = request(tmp_path)
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, possible_effects=True, cleanup_verified=True
    )
    encoded = result.to_wire(approved.attempt_id)
    assert "cleanup_verified" not in json.loads(encoded)["result"]
    assert CommandProcessResult.from_wire(encoded, approved.attempt_id).cleanup_verified is None
    attested = result.to_wire(approved.attempt_id, include_cleanup_proof=True)
    assert CommandProcessResult.from_wire(attested, approved.attempt_id).cleanup_verified is True
    malformed = json.loads(attested)
    malformed["result"]["cleanup_verified"] = "yes"
    with pytest.raises(ValueError, match="cleanup evidence"):
        CommandProcessResult.from_wire(json.dumps(malformed).encode(), approved.attempt_id)
    with pytest.raises(ValueError, match="cleanup evidence"):
        CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0, cleanup_verified="yes")


def test_runtime_budget_is_optional_exact_and_positive_on_native_wire(tmp_path):
    """The receiver restores only a finite positive launch-renewed runtime ceiling."""
    approved = request(tmp_path, runtime_timeout=30.0)
    encoded = approved.to_wire()
    assert SandboxRequest.from_wire(encoded) == approved
    assert json.loads(encoded)["request"]["runtime_timeout"] == 30.0
    assert "runtime_timeout" not in json.loads(request(tmp_path).to_wire())["request"]
    for invalid in (0, -1, float("inf"), True, "30"):
        with pytest.raises(ValueError, match="runtime budget"):
            request(tmp_path, runtime_timeout=invalid)
        payload = json.loads(encoded)
        payload["request"]["runtime_timeout"] = invalid
        with pytest.raises(ValueError, match="runtime budget"):
            SandboxRequest.from_wire(json.dumps(payload).encode())


def test_cleanup_proof_is_optional_exact_boolean_on_request_wire(tmp_path):
    """Only an explicit validated request can ask the receiver for cleanup evidence."""
    approved = request(tmp_path, cleanup_proof=True)
    encoded = approved.to_wire()
    assert json.loads(encoded)["request"]["cleanup_proof"] is True
    assert SandboxRequest.from_wire(encoded) == approved
    assert "cleanup_proof" not in json.loads(request(tmp_path).to_wire())["request"]
    for invalid in (1, "true", None):
        with pytest.raises(TypeError, match="cleanup proof"):
            request(tmp_path, cleanup_proof=invalid)
        payload = json.loads(encoded)
        payload["request"]["cleanup_proof"] = invalid
        with pytest.raises(ValueError, match="boolean authority"):
            SandboxRequest.from_wire(json.dumps(payload).encode())


def test_interactive_terminal_is_an_exact_optional_request_capability(tmp_path):
    """Only a negotiated Boolean request can ask a receiver to allocate an owned PTY."""
    approved = request(tmp_path, terminal=True)
    encoded = approved.to_wire()
    assert json.loads(encoded)["request"]["terminal"] is True
    assert SandboxRequest.from_wire(encoded) == approved
    assert "terminal" not in json.loads(request(tmp_path).to_wire())["request"]
    for invalid in (1, "true", None):
        with pytest.raises(TypeError, match="terminal flag"):
            request(tmp_path, terminal=invalid)
        payload = json.loads(encoded)
        payload["request"]["terminal"] = invalid
        with pytest.raises(ValueError, match="boolean authority"):
            SandboxRequest.from_wire(json.dumps(payload).encode())


def test_local_network_deny_is_exact_and_preserves_default_approval(tmp_path):
    """An explicit local deny survives transport without changing ordinary approval identity."""
    ordinary = request(tmp_path)
    denied = request(tmp_path, local_network=False)
    assert "local_network" not in json.loads(ordinary.to_wire())["request"]
    assert json.loads(denied.to_wire())["request"]["local_network"] is False
    assert SandboxRequest.from_wire(denied.to_wire()) == denied
    assert ordinary.sandbox_command_signature("workspace") != denied.sandbox_command_signature(
        "workspace"
    )
    for invalid in (0, "false", None):
        with pytest.raises(TypeError, match="local-network flag"):
            request(tmp_path, local_network=invalid)
        payload = json.loads(denied.to_wire())
        payload["request"]["local_network"] = invalid
        with pytest.raises(ValueError, match="boolean authority"):
            SandboxRequest.from_wire(json.dumps(payload).encode())
    with pytest.raises(ValueError, match="general network grant"):
        request(tmp_path, network=True, local_network=False)


@pytest.mark.parametrize(
    ("value", "canonical"),
    [
        ("https://EXAMPLE.com", "https://example.com"),
        ("http://example.com:8080", "http://example.com:8080"),
        ("https://[2001:db8::1]", "https://[2001:db8::1]"),
        ("http://127.0.0.1", "http://127.0.0.1"),
    ],
)
def test_egress_origin_normalizes_exact_http_authorities(value, canonical):
    """Origin grants use one stable HTTP authority without request content."""
    assert normalize_egress_origin(value) == canonical


@pytest.mark.parametrize(
    "value",
    [
        None,
        "https://x/after",
        "https://user:x@host",
        "https://host?x=1",
        "https://host#part",
        "https://host:0",
        "https://host:443",
        "https://host:99999",
        "https://001.002.003.004",
        "https://host.",
        "https://a..b",
        "https://-bad",
        "https://host%25alias",
        "ftp://host",
        "https://é.example",
        "https://" + "x" * 256,
        " https://host",
    ],
)
def test_egress_origin_rejects_ambiguous_or_unsupported_spelling(value):
    """Malformed origins never become a broader grant through URL parsing."""
    with pytest.raises(ValueError):
        normalize_egress_origin(value)


def test_egress_grant_binds_wire_and_remembered_signature(tmp_path):
    """A mediated grant survives wire restoration without reusing a plain command approval."""
    origin = "https://example.com"
    scoped = request(tmp_path, egress_origins=(origin,))
    plain = request(tmp_path)
    assert SandboxRequest.from_wire(scoped.to_wire()) == scoped
    assert "egress_origins" not in json.loads(plain.to_wire())["request"]
    assert scoped.sandbox_command_signature("workspace") != plain.sandbox_command_signature(
        "workspace"
    )
    assert scoped.host_command_runtime_signature() != plain.host_command_runtime_signature()
    with pytest.raises(ValueError):
        request(tmp_path, network=True, egress_origins=(origin,))
    with pytest.raises(ValueError):
        request(tmp_path, egress_origins=(origin, origin))


def test_native_output_stream_flag_round_trips_without_changing_approval(tmp_path):
    """Output delivery mode is validated and does not expand command authority."""
    streamed = request(tmp_path, stream_output=True)
    plain = request(tmp_path)

    assert SandboxRequest.from_wire(streamed.to_wire()) == streamed
    assert "stream_output" not in json.loads(plain.to_wire())["request"]
    assert streamed.sandbox_command_signature("workspace") == plain.sandbox_command_signature(
        "workspace"
    )
    with pytest.raises(TypeError, match="streaming flag"):
        request(tmp_path, stream_output="yes")
    envelope = json.loads(streamed.to_wire())
    envelope["request"]["stream_output"] = 1
    with pytest.raises(ValueError, match="boolean authority"):
        SandboxRequest.from_wire(json.dumps(envelope).encode())


def test_request_binds_exact_environment_and_directory_identity(tmp_path):
    """A request retains sanitized environment and detects a replaced approved root."""
    approved = request(tmp_path, environment={"Z": "last", "A": "first"})

    assert approved.environment == (("A", "first"), ("Z", "last"))
    assert approved.shell_environment() == {"A": "first", "Z": "last"}
    assert approved.paths_are_current()

    tmp_path.rename(tmp_path.with_name("old-workspace"))
    tmp_path.mkdir()
    assert not approved.paths_are_current()


def test_request_binds_workspace_selected_executable_through_wire(tmp_path):
    """A selected workspace link retains its target identity through native admission."""
    installation = tmp_path.parent / "selected-installation"
    installation.mkdir()
    executable = installation / "fixture"
    executable.write_text("binary", encoding="utf-8")
    executable.chmod(0o755)
    selected = tmp_path / "selected"
    selected.symlink_to(executable)
    details = executable.stat()
    identity = ("./selected", selected, executable, details.st_dev, details.st_ino)
    approved = request(
        tmp_path,
        read_roots=(installation,),
        workspace_executable_identities=(identity,),
    )
    wire = approved.to_wire()
    assert json.loads(wire)["request"]["workspace_executable_identities"]
    assert SandboxRequest.from_wire(wire) == approved
    assert approved.paths_are_current()
    plain = request(tmp_path, read_roots=(installation,))
    assert "workspace_executable_identities" not in json.loads(plain.to_wire())["request"]
    assert approved.sandbox_command_signature("session") != plain.sandbox_command_signature(
        "session"
    )
    assert approved.host_command_runtime_signature() != plain.host_command_runtime_signature()

    replacement = installation / "replacement"
    replacement.write_text("different", encoding="utf-8")
    replacement.chmod(0o755)
    selected.unlink()
    selected.symlink_to(replacement)
    assert not approved.paths_are_current()
    with pytest.raises(ValueError):
        SandboxRequest.from_wire(wire)


def test_selected_executable_lookup_handles_bounded_path_and_path_search(tmp_path):
    """Selected names use the bound PATH and oversized paths are not resolved."""
    executable = tmp_path / "fixture"
    executable.write_text("binary", encoding="utf-8")
    executable.chmod(0o755)
    assert resolve_selected_executable("fixture", str(tmp_path), tmp_path) == (
        executable,
        executable,
    )
    assert resolve_selected_executable("./" + "x" * constants.MAX_PATH_LENGTH, "", tmp_path) is None


def test_workspace_selected_executable_resolution_failure_is_stale(tmp_path, monkeypatch):
    """A resolution failure refuses request binding and invalidates a captured request."""
    installation = tmp_path.parent / "selected-installation"
    installation.mkdir()
    executable = installation / "fixture"
    executable.write_text("binary", encoding="utf-8")
    executable.chmod(0o755)
    selected = tmp_path / "selected"
    selected.symlink_to(executable)
    details = executable.stat()
    identity = ("./selected", selected, executable, details.st_dev, details.st_ino)
    approved = request(
        tmp_path, read_roots=(installation,), workspace_executable_identities=(identity,)
    )

    def unavailable(*_args):
        """Model a symbolic-link loop introduced after selection."""
        raise RuntimeError("unavailable")

    monkeypatch.setattr("loop.execution.sandbox.contracts.resolve_selected_executable", unavailable)
    assert not approved.paths_are_current()
    with pytest.raises(ValueError, match="Selected workspace executable is unavailable"):
        request(tmp_path, read_roots=(installation,), workspace_executable_identities=(identity,))


@pytest.mark.parametrize(
    "failure",
    [
        "outside-spelling",
        "traversal",
        "missing-root",
        "wrong-inode",
        "hardlink",
        "nonexec",
        "limit",
    ],
)
def test_workspace_selected_executable_rejects_unbound_authority(tmp_path, failure):
    """A selected executable cannot add authority beyond its validated workspace and root."""
    installation = tmp_path.parent / "selected-installation"
    installation.mkdir()
    executable = installation / "fixture"
    executable.write_text("binary", encoding="utf-8")
    executable.chmod(0o755)
    selected = tmp_path / "selected"
    selected.symlink_to(executable)
    details = executable.stat()
    if failure == "outside-spelling":
        selected = installation / "other"
        selected.symlink_to(executable)
        name = str(selected)
    elif failure == "traversal":
        selected = tmp_path / ".." / installation.name / "fixture"
        name = "../" + installation.name + "/fixture"
    else:
        name = "./selected"
    if failure == "hardlink":
        os.link(executable, installation / "hardlink")
    elif failure == "nonexec":
        executable.chmod(0o644)
    identity = (
        name,
        selected,
        executable,
        details.st_dev,
        details.st_ino + (failure == "wrong-inode"),
    )
    identities = (identity,) * (constants.MAX_COMMAND_PATH_COUNT + 1 if failure == "limit" else 1)
    with pytest.raises(ValueError):
        request(
            tmp_path,
            read_roots=() if failure == "missing-root" else (installation,),
            workspace_executable_identities=identities,
        )


def test_directory_descriptor_must_match_the_approved_path(tmp_path):
    """Only the originally approved open directory remains valid after replacement."""
    approved = request(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    regular = tmp_path / "regular"
    regular.write_text("file", encoding="utf-8")
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    other_descriptor = os.open(other, os.O_RDONLY | os.O_DIRECTORY)
    file_descriptor = os.open(regular, os.O_RDONLY)
    try:
        assert approved.directory_matches_descriptor(tmp_path, descriptor)
        assert not approved.directory_matches_descriptor(other, other_descriptor)
        assert not approved.directory_matches_descriptor(tmp_path, other_descriptor)
        assert not approved.directory_matches_descriptor(tmp_path, file_descriptor)
        tmp_path.rename(tmp_path.with_name("old-approved"))
        tmp_path.mkdir()
        assert not approved.directory_matches_descriptor(tmp_path, descriptor)
    finally:
        os.close(other_descriptor)
        os.close(file_descriptor)
        os.close(descriptor)
    assert not approved.directory_matches_descriptor(tmp_path, descriptor)


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


@pytest.mark.parametrize("suffix", ["cache-b", "cache-a/"])
def test_generated_environment_signatures_bind_relative_values(tmp_path, suffix):
    """Different generated paths within one scratch do not share remembered authority."""
    with tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as directory:
        scratch = Path(directory).resolve()
        values = {"TMPDIR": str(scratch), "XDG_CACHE_HOME": str(scratch / "cache-a")}
        first = request(tmp_path, environment=values, generated_environment=tuple(values))
        values["XDG_CACHE_HOME"] = f"{scratch}/{suffix}"
        second = request(tmp_path, environment=values, generated_environment=tuple(values))

        assert first.host_command_runtime_signature() != second.host_command_runtime_signature()
        assert first.sandbox_command_signature("session") != second.sandbox_command_signature(
            "session"
        )


@pytest.mark.parametrize("spelling", ["trailing", "dot"])
def test_generated_environment_accepts_equivalent_scratch_spellings(tmp_path, spelling):
    """Equivalent TMPDIR spellings and normally joined child paths remain valid."""
    with (
        tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as first_directory,
        tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as second_directory,
    ):
        requests = []
        for directory in (first_directory, second_directory):
            scratch = Path(directory).resolve()
            scratch_value = (
                f"{scratch}/" if spelling == "trailing" else f"{scratch.parent}/./{scratch.name}"
            )
            requests.append(
                request(
                    tmp_path,
                    environment={"TMPDIR": scratch_value, "CACHE": str(scratch / "cache")},
                    generated_environment=("TMPDIR", "CACHE"),
                )
            )
        assert all(item.paths_are_current() for item in requests)
        host_matches = (
            requests[0].host_command_runtime_signature()
            == requests[1].host_command_runtime_signature()
        )
        sandbox_matches = requests[0].sandbox_command_signature("session") == requests[
            1
        ].sandbox_command_signature("session")
        assert host_matches is (spelling == "trailing")
        assert sandbox_matches is (spelling == "trailing")


def test_generated_environment_rejects_missing_scratch(tmp_path):
    """Generated values require a bound TMPDIR."""
    with pytest.raises(ValueError, match="Generated command environment"):
        request(tmp_path, environment={"CACHE": str(tmp_path)}, generated_environment=("CACHE",))


def test_generated_environment_rejects_missing_value(tmp_path):
    """Every declared generated variable must have a value."""
    with tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as directory:
        scratch = Path(directory).resolve()
        with pytest.raises(ValueError, match="Generated command environment"):
            request(
                tmp_path,
                environment={"TMPDIR": str(scratch)},
                generated_environment=("CACHE",),
            )


@pytest.mark.parametrize("error", [OSError, RuntimeError])
def test_generated_environment_refuses_uninspectable_paths(tmp_path, monkeypatch, error):
    """A generated path that cannot be resolved cannot establish containment."""
    with tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as directory:
        scratch = Path(directory).resolve()
        value = scratch / "cache"
        original = Path.resolve

        def resolve(path, *args, **kwargs):
            """Reject the generated child while allowing normal scratch validation."""
            if path == value:
                raise error("uninspectable generated path")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", resolve)
        with pytest.raises(ValueError, match="Generated command environment"):
            request(
                tmp_path,
                environment={"TMPDIR": str(scratch), "CACHE": str(value)},
                generated_environment=("CACHE",),
            )


@pytest.mark.parametrize("escape", ["parent", "resolved"])
def test_generated_environment_rejects_escape(tmp_path, monkeypatch, escape):
    """Generated values cannot lexically or through a symlink leave private scratch."""
    with tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as directory:
        scratch = Path(directory).resolve()
        if escape == "parent":
            value = scratch / ".." / "outside"
        else:
            value = scratch / "link"
            original = Path.resolve

            def resolve(path, *args, **kwargs):
                """Represent a generated child resolving beyond scratch."""
                return tmp_path if path == value else original(path, *args, **kwargs)

            monkeypatch.setattr(Path, "resolve", resolve)
        with pytest.raises(ValueError, match="Generated command environment"):
            request(
                tmp_path,
                environment={"TMPDIR": str(scratch), "CACHE": str(value)},
                generated_environment=("TMPDIR", "CACHE"),
            )


def test_generated_environment_detects_changed_symlink_containment(tmp_path, monkeypatch):
    """A generated path that changes to escape scratch invalidates the request."""
    with tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as directory:
        scratch = Path(directory).resolve()
        alias = scratch / "cache"
        target = [scratch / "within"]
        original = Path.resolve

        def resolve(path, *args, **kwargs):
            """Represent changing a generated path's resolved destination."""
            return target[0] if path == alias else original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", resolve)
        approved = request(
            tmp_path,
            environment={"TMPDIR": str(scratch), "CACHE": str(alias)},
            generated_environment=("TMPDIR", "CACHE"),
        )
        assert approved.paths_are_current()
        target[0] = tmp_path
        assert not approved.paths_are_current()


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


def test_command_signatures_bind_exact_executed_source(tmp_path):
    """Distinct actual shell sources cannot share remembered execution authority."""
    first = request(tmp_path, source="printf shown", execution_source="printf first")
    second = replace(first, execution_source="printf second")
    assert first.host_command_runtime_signature() != second.host_command_runtime_signature()
    assert first.sandbox_command_signature("workspace") != second.sandbox_command_signature(
        "workspace"
    )


@pytest.mark.parametrize("scope", ["workspace", "cwd", "read", "write", "path"])
def test_host_runtime_signature_binds_captured_directory_identity(tmp_path, monkeypatch, scope):
    """Changing a captured authority identity changes the host rule signature."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bound = workspace if scope == "workspace" else workspace / "bound"
    if bound != workspace:
        bound.mkdir()
    values = {"cwd": workspace, "read_roots": (), "write_roots": (workspace,)}
    if scope == "cwd":
        values["cwd"] = bound
    elif scope == "read":
        values["read_roots"] = (bound,)
    elif scope == "write":
        values["write_roots"] = (workspace, bound)
    elif scope == "path":
        values["environment"] = {"PATH": str(bound)}
    first = request(workspace, **values)
    capture = DirectoryIdentity.capture

    def changed_identity(cls, path):
        """Supply a different identity for the selected authority."""
        identity = capture(path)
        if path == bound:
            return DirectoryIdentity(path, identity.device, identity.inode + 1)
        return identity

    monkeypatch.setattr(DirectoryIdentity, "capture", classmethod(changed_identity))
    second = request(workspace, **values)

    assert first.host_command_runtime_signature() != second.host_command_runtime_signature()


@pytest.mark.parametrize("overlap", ["ancestor", "descendant"])
def test_host_runtime_signature_binds_overlapping_ephemeral_identity(
    tmp_path, monkeypatch, overlap
):
    """Generated directories retain identity when an explicit authority overlaps them."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with tempfile.TemporaryDirectory(prefix="loop-generated-") as directory:
        ephemeral = Path(directory).resolve()
        child = ephemeral / "child"
        child.mkdir()
        authority = ephemeral.parent if overlap == "ancestor" else child
        values = {"read_roots": (authority,)}
        values.update(
            environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(ephemeral)},
            generated_environment=("TMPDIR",),
        )
        first = request(workspace, **values)
        capture = DirectoryIdentity.capture

        def changed_identity(cls, path):
            """Represent a changed identity without depending on native inode allocation."""
            identity = capture(path)
            if path == ephemeral:
                return DirectoryIdentity(path, identity.device, identity.inode + 1)
            return identity

        monkeypatch.setattr(DirectoryIdentity, "capture", classmethod(changed_identity))
        second = request(workspace, **values)
        assert first.host_command_runtime_signature() != second.host_command_runtime_signature()


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


def test_request_binds_external_read_alias_and_rejects_retargeting(tmp_path, monkeypatch):
    """An approved directory symlink keeps its inode and canonical read target until launch."""
    installation = tmp_path / "installation"
    installation.mkdir()
    alias = tmp_path / "opt-tool"
    alias.symlink_to(installation, target_is_directory=True)
    approved = request(tmp_path, read_roots=(installation,), read_aliases=(alias,))
    assert approved.paths_are_current()
    assert approved.read_alias_targets == (installation,)
    alias.rename(tmp_path / "original-alias")
    assert not approved.paths_are_current()
    assert approved.read_alias_targets == (installation,)
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


def test_request_rejects_empty_execution_source(tmp_path):
    """An empty execution source cannot replace an approved command."""
    with pytest.raises(ValueError, match="cannot be empty"):
        request(tmp_path, execution_source="")


@pytest.mark.parametrize(
    "change",
    [
        {"source": ""},
        {"policy_version": ""},
        {"workspace_id": ""},
        {"deadline": 0},
        {"deadline": float("inf")},
        {"deadline": float("-inf")},
        {"deadline": float("nan")},
        {"deadline": True},
        {"deadline": "20"},
        {"deadline": 10**1000},
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


def test_request_wire_compacts_only_identical_large_sources(tmp_path):
    """A large identical source is transmitted once without changing its approved authority."""
    source = '"' * (constants.MAX_NATIVE_REQUEST_BYTES // 4)
    approved = request(tmp_path, source=source)
    wire = approved.to_wire()
    envelope = json.loads(wire)
    assert envelope["schema"] == 2
    assert envelope["request"]["source"] == source
    assert "execution_source" not in envelope["request"]
    assert len(wire) <= constants.MAX_NATIVE_REQUEST_BYTES
    assert SandboxRequest.from_wire(wire) == approved

    distinct = replace(approved, execution_source=source + " ")
    with pytest.raises(ValueError, match="admission bound"):
        distinct.to_wire()


@pytest.mark.parametrize("change", ["extra_source", "unneeded", "identity"])
def test_request_wire_compact_schema_refuses_ambiguous_or_changed_authority(tmp_path, change):
    """Compact requests reject extra source fields, noncanonical encoding and changed identities."""
    source = '"' * (constants.MAX_NATIVE_REQUEST_BYTES // 4)
    approved = request(tmp_path, source=source)
    envelope = json.loads(approved.to_wire())
    if change == "extra_source":
        envelope["request"]["execution_source"] = "unexpected"
    elif change == "unneeded":
        envelope["request"]["source"] = "short"
    else:
        envelope["request"]["_identities"][0]["inode"] += 1
    with pytest.raises(ValueError, match="schema|snapshot differs"):
        SandboxRequest.from_wire(json.dumps(envelope).encode())


def test_request_wire_roundtrip_keeps_read_alias_and_executable_inodes(tmp_path):
    """Executable and read-alias identities survive restoration without broader reads."""
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
    details = binary.stat()
    approved = request(
        workspace,
        read_roots=(tool,),
        environment={"PATH": str(tool)},
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
        source="x" * 65536 if change == "size" else "printf ok",
    )
    if change == "deadline":
        approved = replace(approved, deadline=float("inf"))
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
    """Only the exact supported envelopes and their required request fields are accepted."""
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
        ("unexpected_field", [["root", 0, "target"]]),
        ("executable_identities", [["python", "/bin/python", "/bin/python", True, 1]]),
        ("executable_identities", [["python", "/bin/python", "/bin/python", -1, 1]]),
        ("workspace_executable_identities", [["tool", "/tmp/tool", "/tmp/tool", True, 1]]),
    ],
)
def test_request_wire_invalid_typed_fields_refuse(tmp_path, field, value):
    """Request field types and row widths cannot be coerced into broader execution authority."""
    envelope = json.loads(request(tmp_path).to_wire())
    envelope["request"][field] = value
    with pytest.raises(ValueError):
        SandboxRequest.from_wire(json.dumps(envelope).encode())


@pytest.mark.parametrize("change", ["directory", "snapshot", "environment", "unknown"])
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
        envelope["request"]["unexpected"] = [[str(tmp_path), 1, 1, str(tmp_path)]]
    with pytest.raises(ValueError, match="schema|snapshot differs"):
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


@pytest.mark.parametrize(
    "outcome", [item for item in SandboxOutcome if item is not SandboxOutcome.UNSUPPORTED]
)
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


def test_client_only_unsupported_outcome_is_not_serialized():
    """The client-only outcome cannot be serialized into the receiver result schema."""
    result = CommandProcessResult(SandboxOutcome.UNSUPPORTED)
    with pytest.raises(ValueError, match="client-only outcome"):
        result.to_wire("1248c892-42dc-4b68-a3fb-30d2a70a93a7")


@pytest.mark.parametrize("status", ["available", "unavailable"])
def test_result_wire_preserves_diagnostic_availability(status):
    """A diagnostic result retains availability independently of the outcome."""
    attempt = "1248c892-42dc-4b68-a3fb-30d2a70a93a7"
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=0,
        diagnostic_status=status,
        observed_denial="file-read-data /outside" if status == "available" else "",
        diagnostic_source="macos-kernel-log" if status == "available" else "unknown",
    )
    encoded = result.to_wire(attempt)
    assert json.loads(encoded)["schema"] == 2
    assert CommandProcessResult.from_wire(encoded, attempt) == result


def test_result_wire_keeps_schema_one_for_unknown_diagnostic_availability():
    """The installed receiver's version-one result remains compatible and unchanged."""
    attempt = "1248c892-42dc-4b68-a3fb-30d2a70a93a7"
    result = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    encoded = result.to_wire(attempt)
    assert encoded == (
        b'{"schema":1,"attempt_id":"1248c892-42dc-4b68-a3fb-30d2a70a93a7",'
        b'"result":{"outcome":"unavailable","exit_code":null,"stdout":"","stderr":"",'
        b'"stdout_discarded":0,"stderr_discarded":0,"detail":"","observed_denial":"",'
        b'"possible_effects":false,"failure_context":"Native sandbox execution could not start."}}'
    )
    envelope = json.loads(encoded)
    assert envelope["schema"] == 1
    assert "diagnostic_status" not in envelope["result"]
    assert "diagnostic_source" not in envelope["result"]
    assert CommandProcessResult.from_wire(json.dumps(envelope).encode(), attempt) == result


def test_result_rejects_invalid_diagnostic_availability():
    """An unrecognized local diagnostic state cannot enter a result."""
    with pytest.raises(ValueError, match="Invalid sandbox diagnostic status"):
        CommandProcessResult(SandboxOutcome.UNAVAILABLE, diagnostic_status="invented")
    with pytest.raises(ValueError, match="Invalid sandbox diagnostic source"):
        CommandProcessResult(SandboxOutcome.UNAVAILABLE, diagnostic_source="invented")
    with pytest.raises(ValueError, match="Invalid sandbox diagnostic status"):
        CommandProcessResult(SandboxOutcome.UNAVAILABLE, diagnostic_status=[])
    with pytest.raises(ValueError, match="Invalid sandbox diagnostic source"):
        CommandProcessResult(SandboxOutcome.UNAVAILABLE, diagnostic_source={})
    with pytest.raises(ValueError, match="Diagnostic provenance requires an observed denial"):
        CommandProcessResult(SandboxOutcome.UNAVAILABLE, diagnostic_source="macos-kernel-log")
    with pytest.raises(ValueError, match="An observed denial requires"):
        CommandProcessResult(
            SandboxOutcome.UNAVAILABLE,
            observed_denial="file-read-data /outside",
            diagnostic_status="unavailable",
        )
    with pytest.raises(ValueError, match="Known diagnostic provenance requires"):
        CommandProcessResult(
            SandboxOutcome.UNAVAILABLE,
            observed_denial="file-read-data /outside",
            diagnostic_source="macos-kernel-log",
        )


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


def test_request_binds_durable_session_scratch(tmp_path):
    """Durable scratch outside the system temporary root requires an explicit binding."""
    scratch = tmp_path / "Library" / "app" / "scratch" / "session"
    scratch.mkdir(parents=True, mode=0o700)
    environment = {"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)}
    with pytest.raises(ValueError, match="private real directory"):
        request(tmp_path, environment=environment)
    approved = request(tmp_path, environment=environment, scratch_root=scratch)
    assert approved.session_scratch
    assert SandboxRequest.from_wire(approved.to_wire()).session_scratch
    scratch.rename(scratch.with_name("previous-session"))
    scratch.mkdir(mode=0o700)
    assert not approved.paths_are_current()


def test_explicit_session_data_binding_survives_transport_and_binds_approval(tmp_path):
    """Fresh-data permission is explicit, identity-bound and part of remembered authority."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    scratch = tmp_path / "session"
    scratch.mkdir(mode=0o700)
    values = {
        "environment": {"TMPDIR": str(scratch)},
        "scratch_root": scratch,
        "scratch_data": True,
    }
    approved = request(workspace, **values)
    assert approved.session_scratch
    assert SandboxRequest.from_wire(approved.to_wire()) == approved
    restricted = request(workspace, **{**values, "scratch_data": False})
    assert approved.sandbox_command_signature("session") != restricted.sandbox_command_signature(
        "session"
    )
    assert approved.host_command_runtime_signature() != restricted.host_command_runtime_signature()
    scratch.rename(scratch.with_name("retired"))
    scratch.mkdir(mode=0o700)
    assert not approved.paths_are_current()


def test_bound_session_scratch_restores_without_global_temp_discovery(tmp_path, monkeypatch):
    """Authenticated scratch restores when the setup filesystem has no global temporary root."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    scratch = tmp_path / "session"
    scratch.mkdir(mode=0o700)
    values = {"environment": {"TMPDIR": str(scratch)}, "write_roots": ()}
    values.update(scratch_root=scratch, scratch_data=True)

    def unavailable():
        """Reject discovery probes in the request-restoration filesystem view."""
        raise FileNotFoundError("No usable temporary directory found")

    monkeypatch.setattr(tempfile, "gettempdir", unavailable)
    approved = request(workspace, **values)
    assert approved.session_scratch
    assert SandboxRequest.from_wire(approved.to_wire()) == approved
    with pytest.raises(FileNotFoundError, match="No usable temporary directory"):
        request(workspace, environment={"TMPDIR": str(scratch)}, write_roots=())


@pytest.mark.parametrize("change", ["unbound", "overlap", "wrong-root", "no-root", "not-bool"])
def test_session_data_binding_rejects_ambiguous_authority(tmp_path, change):
    """Environment values and overlapping roots cannot independently grant scratch-data access."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    changes = {
        "environment": {"TMPDIR": str(scratch)},
        "scratch_root": scratch,
        "scratch_data": True,
    }
    if change == "unbound":
        changes["environment"] = {}
    elif change == "overlap":
        changes["read_roots"] = (scratch,)
    elif change == "wrong-root":
        changes["scratch_root"] = tmp_path / "other"
    elif change == "no-root":
        changes["scratch_root"] = None
    else:
        changes["scratch_data"] = "true"
    with pytest.raises(ValueError, match="explicit and disjoint"):
        request(tmp_path, **changes)
