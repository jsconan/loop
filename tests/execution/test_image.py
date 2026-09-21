"""Test the checked-in single sandbox image definition and readiness models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from loop.execution.runtime.image import (
    RuntimeImage,
    SandboxImageDefinition,
    SandboxImageReadiness,
    image_has_safe_defaults,
    load_sandbox_containerfile,
    load_sandbox_image_definition,
)
from loop.execution.runtime.models import OciImageIdentity, PlatformSelector


def _definition_payload() -> dict:
    """Return one mutable copy of the packaged image definition."""
    return load_sandbox_image_definition().model_dump(mode="json")


def _inspection(**config_updates: object) -> bytes:
    """Return minimal native inspection defaults for one safe image."""
    config = {
        "User": "agent:agent",
        "WorkingDir": "/workspace",
        "Env": ["PATH=/tools/bin:/usr/local/bin:/usr/bin:/bin"],
    }
    config.update(config_updates)
    return json.dumps(
        {"ImageConfig": {"os": "linux", "architecture": "arm64", "config": config}}
    ).encode()


def test_packaged_image_is_complete_locked_and_deterministic() -> None:
    """The one packaged image covers both architectures and every promised tool family."""
    definition = load_sandbox_image_definition()

    assert definition.environment.path == "/tools/bin:/usr/local/bin:/usr/bin:/bin"
    assert definition.environment.uid == definition.environment.gid == 1000
    assert definition.tools["uv"].version == "0.12.17"
    assert set(definition.tools["uv"].sha256) == {"linux/amd64", "linux/arm64"}
    assert {"python3", "uv", "uvx", "node", "gcc", "cargo", "go", "git", "rg"} <= set(
        definition.commands
    )
    assert "=" in definition.package_arguments
    assert definition.tool_arguments == (
        "UV_VERSION=0.12.17",
        "UV_ORIGIN=https://releases.astral.sh/github/uv/releases/download/0.12.17",
        "UV_MAX_BYTES=26214400",
        "UV_AARCH64_SHA256=d636d1b678e9e7f367ecb22b46bd1cabbed234d6bc3b4d96365d2b507f72f86c",
        "UV_X86_64_SHA256=fa82fd8dde8e8eefdecada6aa0889666556cfceb690d06e0c3bca49eb3070a63",
    )
    assert definition.digest == load_sandbox_image_definition().digest
    assert definition.source_version == load_sandbox_image_definition().source_version
    assert b"ARG BASE_IMAGE=" + definition.base_image.encode() in load_sandbox_containerfile()
    assert b"https://snapshot.debian.org" in load_sandbox_containerfile()
    assert b"type=secret,id=loop-ca" in load_sandbox_containerfile()
    assert b"ARG UV_VERSION\n" in load_sandbox_containerfile()
    assert b"releases.astral.sh" not in load_sandbox_containerfile()
    assert b'--max-redirect=0 --quota="$UV_MAX_BYTES"' in load_sandbox_containerfile()
    payload = json.dumps(_definition_payload()).encode()
    assert load_sandbox_image_definition(payload).digest == definition.digest


def test_readiness_contains_only_source_and_local_image_identity() -> None:
    """The durable readiness model excludes signatures, SBOMs, and publication evidence."""
    identity = OciImageIdentity(
        index_digest="sha256:" + "a" * 64,
        manifest_digest="sha256:" + "b" * 64,
        config_digest="sha256:" + "c" * 64,
    )
    readiness = SandboxImageReadiness(
        schema_version=1,
        source_version="sha256:" + "d" * 64,
        image=RuntimeImage(
            reference=f"loop.local/sandbox@{identity.index_digest}",
            platform=PlatformSelector(os="linux", architecture="arm64"),
            identity=identity,
        ),
    )

    assert set(readiness.model_dump()) == {"schema_version", "source_version", "image"}


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update(platforms=["linux/arm64", "linux/arm64"]), "exactly"),
        (lambda payload: payload.update(unexpected=True), "Extra inputs"),
        (lambda payload: payload["environment"].update(uid=1001), "reviewed non-root"),
        (
            lambda payload: payload["packages"].update(bash="latest"),
            "exact whitespace-free",
        ),
        (lambda payload: payload["packages"].update({"": "1"}), "exact whitespace-free"),
        (lambda payload: payload["packages"].update(extra=""), "exact whitespace-free"),
        (
            lambda payload: payload["packages"].update(extra="1 bad"),
            "exact whitespace-free",
        ),
        (
            lambda payload: payload["commands"].update(git="/host/bin/git"),
            "deterministic PATH",
        ),
        (lambda payload: payload["commands"].update({"": "/usr/bin/x"}), "deterministic PATH"),
        (
            lambda payload: payload["commands"].update({"bad/name": "/usr/bin/x"}),
            "deterministic PATH",
        ),
        (
            lambda payload: payload["commands"].update(extra="/usr/bin/"),
            "deterministic PATH",
        ),
        (
            lambda payload: payload.update(tools={"other": payload["tools"]["uv"]}),
            "reviewed external tool",
        ),
        (
            lambda payload: payload["tools"]["uv"].update(origin="http://example.test"),
            "String should match pattern",
        ),
        (
            lambda payload: payload["tools"]["uv"].update(version="0.12.16"),
            "origin must match",
        ),
        (
            lambda payload: payload["tools"]["uv"].update(sha256={"linux/arm64": "bad"}),
            "platform SHA-256",
        ),
    ),
)
def test_definition_rejects_open_or_ambiguous_inventory(mutation, message: str) -> None:
    """Mutable packages, duplicate platforms, extra fields, and host paths fail closed."""
    payload = _definition_payload()
    mutation(payload)

    with pytest.raises(ValidationError, match=message):
        SandboxImageDefinition.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "payload",
    (
        b"not-json",
        json.dumps({"ImageConfig": {}}).encode(),
        _inspection(User="root"),
        _inspection(WorkingDir="/root"),
        _inspection(Env=[]),
        _inspection(Entrypoint=["/bin/sh"]),
        _inspection(ExposedPorts={"80/tcp": {}}),
        _inspection(Volumes={"/host": {}}),
    ),
)
def test_image_defaults_reject_privileged_or_ambiguous_configuration(payload: bytes) -> None:
    """Readiness rejects malformed, root, executable, network, and volume defaults."""
    assert not image_has_safe_defaults(
        payload,
        PlatformSelector(os="linux", architecture="arm64"),
    )


def test_image_defaults_accept_the_exact_non_root_environment() -> None:
    """Readiness accepts only the declared non-root workspace environment."""
    assert image_has_safe_defaults(
        _inspection(),
        PlatformSelector(os="linux", architecture="arm64"),
    )
