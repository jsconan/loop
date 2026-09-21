"""Test closed OCI specification compilation and inspection attestation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loop.execution.contracts import (
    Capability,
    DirectExecutionRequest,
    ExecutionLease,
    ExecutionMode,
    ShellExecutionRequest,
    TerminalMode,
)
from loop.execution.results import InfrastructureFailure
from loop.execution.runtime.models import (
    AcquisitionKind,
    Artifact,
    ArtifactRole,
    OciImageIdentity,
    PlatformSelector,
)
from loop.execution.sandbox.oci import OciSandboxAdapter, OciSpecCompiler
from loop.execution.sandbox.oci.spec import OciResourceLimits, OciSpecificationError


def _image() -> Artifact:
    """Build one complete immutable OCI execution profile."""
    return Artifact(
        artifact_id="core",
        version="1",
        role=ArtifactRole.OCI_PROFILE,
        platform=PlatformSelector(os="linux", architecture="amd64"),
        source="registry.test/core@sha256:" + "a" * 64,
        digest="sha256:" + "a" * 64,
        acquisition=AcquisitionKind.OCI,
        media_type="application/vnd.oci.image.manifest.v1+json",
        oci_identity=OciImageIdentity(
            index_digest="sha256:" + "a" * 64,
            manifest_digest="sha256:" + "b" * 64,
            config_digest="sha256:" + "c" * 64,
        ),
        sbom="https://example.test/sbom",
        notices="https://example.test/notices",
    )


def _request(**changes: object) -> ShellExecutionRequest:
    """Build one authorized sandbox shell request."""
    lease = ExecutionLease(
        lease_id="lease",
        workspace_id="workspace",
        agent_run_id="agent",
        policy_version="policy",
        runtime_digest="sha256:" + "b" * 64,
        expires_at_ns=1,
        capabilities=frozenset({Capability.WORKSPACE_READ, Capability.PROCESS_SPAWN}),
    )
    values = {
        "request_id": "request",
        "lease": lease,
        "script": "printf safe",
        "environment": (("Z", "last"), ("A", "first")),
    }
    values.update(changes)
    return ShellExecutionRequest(**values)


def _adapter() -> OciSandboxAdapter:
    """Build one selected private-runtime OCI preparation boundary."""
    return OciSandboxAdapter(
        OciSpecCompiler(_image(), OciResourceLimits(1024, 32, 10000)),
        "loop-private",
        "linux",
        "amd64",
    )


def _evidence(prepared) -> bytes:
    """Load the checked-in golden schema-v1 evidence for the fixed test attempt."""
    assert prepared.spec.request_id == "request"
    return Path(__file__).with_name("fixtures").joinpath("inspection-v1.json").read_bytes()


def test_compiler_creates_deterministic_shell_spec_without_host_paths():
    """Shell source remains opaque guest input while settings stay deterministic and confined."""
    spec = _adapter().prepare(_request(), "generation").spec

    assert spec.argv == ("/bin/sh", "-c", "printf safe")
    assert spec.environment == (
        ("A", "first"),
        ("GIT_OPTIONAL_LOCKS", "0"),
        ("HOME", "/home/agent"),
        ("PATH", "/tools/bin:/usr/local/bin:/usr/bin:/bin"),
        ("TMPDIR", "/tmp"),
        ("XDG_CONFIG_HOME", "/tmp/.config"),
        ("Z", "last"),
    )
    assert spec.workspace_mount.source_id == "generation"
    assert spec.workspace_mount.read_only
    assert spec.image_reference.startswith("registry.test/")
    assert "/Users/" not in repr(spec)
    assert spec.terminal is TerminalMode.PIPE
    assert not spec.attach_stdin

    input_spec = _adapter().prepare(_request(stdin=b"bounded\n"), "generation").spec
    assert input_spec.attach_stdin


def test_terminal_contract_requires_dimensions_only_for_pty() -> None:
    """Reject ambiguous terminal requests before compiling an OCI invocation."""
    with pytest.raises(ValueError, match="requires initial"):
        _request(terminal=TerminalMode.PTY)
    with pytest.raises(ValueError, match="cannot carry"):
        _request(terminal_columns=80)
    request = _request(
        terminal=TerminalMode.PTY,
        terminal_columns=100,
        terminal_rows=40,
    )
    spec = _adapter().prepare(request, "generation").spec
    assert (spec.terminal, spec.terminal_columns, spec.terminal_rows) == (
        TerminalMode.PTY,
        100,
        40,
    )


def test_compiler_preserves_direct_argv_and_marks_durable_ownership():
    """Direct guest argv stays exact and durable jobs are explicitly detached."""
    request = DirectExecutionRequest(
        request_id="direct",
        lease=_request().lease,
        argv=("/usr/bin/tool", "argument"),
        mode=ExecutionMode.DURABLE_JOB,
    )
    spec = _adapter().prepare(request, "generation").spec

    assert spec.argv == request.argv
    assert spec.detached


def test_compiler_allows_only_a_writable_attempt_mount_for_a_write_lease():
    """A granted workspace write uses the isolated attempt overlay rather than the host."""
    request = _request().model_copy(
        update={
            "lease": _request().lease.model_copy(
                update={"capabilities": frozenset({Capability.WORKSPACE_WRITE})}
            )
        }
    )
    spec = _adapter().prepare(request, "generation").spec

    assert not spec.workspace_mount.read_only


@pytest.mark.parametrize(
    ("changes", "generation"),
    [
        ({"environment": (("BAD-NAME", "x"),)}, "generation"),
        ({"environment": (("A", "x"), ("A", "y"))}, "generation"),
        ({"request_id": "bad/value"}, "generation"),
        ({}, "bad/value"),
    ],
)
def test_compiler_rejects_adversarial_identity_and_environment_values(changes, generation):
    """Model-controlled values cannot become labels, mounts, or OCI option syntax."""
    with pytest.raises(OciSpecificationError):
        _adapter().prepare(_request(**changes), generation)


def test_compiler_rejects_capabilities_not_yet_owned_by_effect_brokers():
    """Network and secret authority cannot appear without an effect broker."""
    request = _request().model_copy(
        update={
            "lease": _request().lease.model_copy(
                update={"capabilities": frozenset({Capability.NETWORK_CONNECT})}
            )
        }
    )
    with pytest.raises(OciSpecificationError, match="unavailable"):
        _adapter().prepare(request, "generation")

    unsupported = _request().model_copy(
        update={
            "lease": _request().lease.model_copy(
                update={
                    "capabilities": frozenset(
                        {
                            Capability.WORKSPACE_READ,
                            Capability.PROCESS_SPAWN,
                            Capability.IPC_CONNECT,
                        }
                    )
                }
            )
        }
    )
    with pytest.raises(OciSpecificationError, match="unsupported"):
        _adapter().prepare(unsupported, "generation")


def test_compiler_rejects_missing_image_identity_invalid_limits_and_nul_arguments():
    """The compiler refuses unpinned images, lease drift, missing controls, and unsafe argv."""
    unpinned = _image().model_copy(update={"oci_identity": None})
    with pytest.raises(OciSpecificationError):
        OciSpecCompiler(unpinned, OciResourceLimits(1, 1, 1))
    with pytest.raises(OciSpecificationError):
        OciResourceLimits(0, 1, 1)
    with pytest.raises(OciSpecificationError):
        OciResourceLimits(1, 1, 1, 0)
    with pytest.raises(OciSpecificationError):
        OciResourceLimits(1, 1, 1, persistent_write_bytes=0)
    mismatched = _request().model_copy(
        update={"lease": _request().lease.model_copy(update={"runtime_digest": "sha256:wrong"})}
    )
    with pytest.raises(OciSpecificationError, match="lease"):
        _adapter().prepare(mismatched, "generation")
    request = DirectExecutionRequest(
        request_id="direct",
        lease=_request().lease,
        argv=("/bin/tool", "\x00"),
    )
    with pytest.raises(OciSpecificationError):
        _adapter().prepare(request, "generation")


def test_attestation_accepts_exact_schema_and_fails_closed_for_every_mismatch():
    """Only exact image, mount, seccomp, controls, network, and socket evidence attests."""
    adapter = _adapter()
    prepared = adapter.prepare(_request(), "generation")
    attestation = adapter.attest(prepared, _evidence(prepared))
    assert attestation.policy_digest == "sha256:" + "d" * 64

    payload = json.loads(_evidence(prepared))
    payload["config"]["management_sockets"] = ["/run/containerd.sock"]
    assert isinstance(adapter.attest(prepared, json.dumps(payload).encode()), InfrastructureFailure)
    payload = json.loads(_evidence(prepared))
    del payload["config"]["seccomp_digest"]
    assert isinstance(adapter.attest(prepared, json.dumps(payload).encode()), InfrastructureFailure)
    payload = json.loads(_evidence(prepared))
    payload["config"]["mounts"].append(payload["config"]["mounts"][0])
    assert isinstance(adapter.attest(prepared, json.dumps(payload).encode()), InfrastructureFailure)
    payload = json.loads(_evidence(prepared))
    payload["unexpected"] = True
    assert isinstance(adapter.attest(prepared, json.dumps(payload).encode()), InfrastructureFailure)
    assert isinstance(adapter.attest(prepared, b"not-json"), InfrastructureFailure)
