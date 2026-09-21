"""Test local construction and reuse of the single managed macOS sandbox image."""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from loop.execution.infrastructure import InfrastructureProcessResult
from loop.execution.runtime import load_sandbox_image_definition
from loop.execution.runtime.image import SandboxImageDefinition
from loop.execution.runtime.models import PlatformSelector
from loop.execution.sandbox.macos.images import MacosSandboxImageBuilder, _atomic_private_write

_INDEX = "sha256:" + "a" * 64
_MANIFEST = "sha256:" + "b" * 64
_CONFIG = "sha256:" + "c" * 64


def _inspection(index: str = _INDEX, manifest: str = _MANIFEST) -> bytes:
    """Return one exact native nerdctl inspection document with safe defaults."""
    return json.dumps(
        {
            "Image": {"Target": {"digest": index}},
            "IndexDesc": {"digest": index},
            "Index": {
                "schemaVersion": 2,
                "manifests": [
                    {
                        "digest": manifest,
                        "platform": {"os": "linux", "architecture": "arm64"},
                    }
                ],
            },
            "ManifestDesc": {"digest": manifest},
            "Manifest": {"schemaVersion": 2, "config": {"digest": _CONFIG}},
            "ImageConfigDesc": {"digest": _CONFIG},
            "ImageConfig": {
                "os": "linux",
                "architecture": "arm64",
                "config": {
                    "User": "agent:agent",
                    "WorkingDir": "/workspace",
                    "Env": ["PATH=/tools/bin:/usr/local/bin:/usr/bin:/bin"],
                },
            },
            "size": 1,
        },
        separators=(",", ":"),
    ).encode()


def _inventory(definition: SandboxImageDefinition | None = None) -> bytes:
    """Return complete locked package, command, and process evidence."""
    definition = definition or load_sandbox_image_definition()
    lines = [
        (
            f"IDENTITY\t{definition.environment.uid}\t{definition.environment.gid}"
            f"\t{definition.environment.path}"
        )
    ]
    lines.extend(
        f"PACKAGE\t{name}\t{version}" for name, version in sorted(definition.packages.items())
    )
    lines.extend(f"COMMAND\t{name}\t{path}" for name, path in sorted(definition.commands.items()))
    return ("\n".join(lines) + "\n").encode()


def _framed(
    *,
    inspection: bytes | None = None,
    cleanup: str = "complete",
    inventory: bytes | None = None,
) -> bytes:
    """Frame native descriptor, inventory, and builder-cleanup evidence."""
    return b"".join(
        (
            inspection or _inspection(),
            b"\n--LOOP-INVENTORY--\n",
            inventory if inventory is not None else _inventory(),
            b"\n--LOOP-CLEANUP--\n",
            cleanup.encode(),
            b"\n",
        )
    )


class _ControlPlane:
    """Record image construction, inspection, and removal through a management double."""

    def __init__(self, output: bytes | None = None) -> None:
        self.output = output or _framed()
        self.builds: list[tuple[object, ...]] = []
        self.context_members: set[str] = set()
        self.inspect_result = InfrastructureProcessResult(0, _inspection(), b"", False, False)
        self.removals: list[str] = []
        self.remove_result = InfrastructureProcessResult(0, b"", b"", False, False)

    def build_sandbox_image(self, archive: Path, *values: object) -> object:
        """Capture the trusted source archive before temporary cleanup."""
        with tarfile.open(archive) as context:
            self.context_members = set(context.getnames())
        self.builds.append(values)
        return SimpleNamespace(
            exit_code=0,
            stdout=self.output,
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
            platform=PlatformSelector(os="linux", architecture="arm64"),
        )

    def inspect_image(self, reference: str) -> InfrastructureProcessResult:
        """Return configured cache reinspection evidence."""
        assert reference.startswith("loop.local/sandbox@sha256:")
        return self.inspect_result

    def remove_image(self, reference: str) -> InfrastructureProcessResult:
        """Record one bounded private image reclamation."""
        self.removals.append(reference)
        return self.remove_result


def _builder(
    tmp_path: Path,
    plane: _ControlPlane,
    definition: SandboxImageDefinition | None = None,
) -> MacosSandboxImageBuilder:
    """Construct one isolated builder with fake managed-runtime identity."""
    return MacosSandboxImageBuilder(
        plane,
        tmp_path / "images",
        definition,  # type: ignore[arg-type]
    )


def test_builder_constructs_records_and_reattests_cached_image(tmp_path: Path) -> None:
    """One build publishes a minimal readiness record and warm reuse only reinspects it."""
    plane = _ControlPlane()
    builder = _builder(tmp_path, plane)
    assert builder.definition == load_sandbox_image_definition()

    installed = builder.prepare()
    reused = builder.prepare()

    assert installed == reused
    assert installed.image.reference == f"loop.local/sandbox@{_INDEX}"
    assert installed.readiness.image.identity.manifest_digest == _MANIFEST
    assert installed.readiness_path.stat().st_mode & 0o777 == 0o600
    assert set(json.loads(installed.readiness_path.read_bytes())) == {
        "schema_version",
        "source_version",
        "image",
    }
    assert plane.context_members == {"Containerfile", "inventory.json"}
    assert len(plane.builds) == 1


def test_builder_accepts_platform_qualified_debian_binary_names(tmp_path: Path) -> None:
    """Debian architecture suffixes preserve exact versions under the attested platform."""
    inventory = _inventory().replace(b"PACKAGE\tbzip2\t", b"PACKAGE\tbzip2:arm64\t")
    installed = _builder(tmp_path, _ControlPlane(_framed(inventory=inventory))).prepare()
    assert installed.image.platform.architecture == "arm64"


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (b"broken", "malformed"),
        (_framed(cleanup="remaining"), "cleanup was incomplete"),
        (_framed(inventory=b"IDENTITY\t1000\t1000\t/bad\n"), "process identity"),
        (
            _framed(inspection=_inspection().replace(b'"User":"agent:agent"', b'"User":"root"')),
            "defaults are unsafe",
        ),
    ],
)
def test_builder_rejects_incomplete_or_untrusted_build_evidence(
    tmp_path: Path, output: bytes, message: str
) -> None:
    """No image becomes active from malformed, unsafe, or incompletely cleaned evidence."""
    with pytest.raises(RuntimeError, match=message):
        _builder(tmp_path, _ControlPlane(output)).prepare()


def test_builder_rejects_tampered_readiness_and_changed_cached_image(tmp_path: Path) -> None:
    """Private readiness shape and live descriptor/default identities fail closed on reuse."""
    plane = _ControlPlane()
    builder = _builder(tmp_path, plane)
    installed = builder.prepare()
    readiness = json.loads(installed.readiness_path.read_bytes())
    readiness["unexpected"] = True
    installed.readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    with pytest.raises(RuntimeError, match="readiness is invalid"):
        builder.prepare()

    installed.readiness_path.write_bytes(installed.readiness.model_dump_json().encode())
    installed.readiness_path.chmod(0o600)
    plane.inspect_result = InfrastructureProcessResult(
        0, _inspection(manifest="sha256:" + "f" * 64), b"", False, False
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        builder.prepare()

    plane.inspect_result = InfrastructureProcessResult(
        0,
        _inspection().replace(b'"User":"agent:agent"', b'"User":"root"'),
        b"",
        False,
        False,
    )
    with pytest.raises(RuntimeError, match="defaults are unsafe"):
        builder.prepare()


def test_builder_rebuilds_after_interrupted_or_missing_first_build(tmp_path: Path) -> None:
    """A failed first build leaves no readiness and a missing cached image rebuilds cleanly."""
    plane = _ControlPlane()
    original = plane.build_sandbox_image
    attempts = 0

    def interrupted(archive: Path, *values: object) -> object:
        """Fail only the first transient build attempt."""
        nonlocal attempts
        attempts += 1
        result = original(archive, *values)
        if attempts == 1:
            result.exit_code = 1
            result.stderr = b"interrupted"
        return result

    plane.build_sandbox_image = interrupted  # type: ignore[method-assign]
    builder = _builder(tmp_path, plane)
    with pytest.raises(RuntimeError, match="image build failed"):
        builder.prepare()
    assert not (tmp_path / "images" / "active.json").exists()
    builder.prepare()
    plane.inspect_result = InfrastructureProcessResult(1, b"", b"missing", False, False)
    builder.prepare()
    plane.inspect_result = InfrastructureProcessResult(0, b"", b"", False, False)
    builder.prepare()
    assert attempts == 4


def test_builder_rejects_symlink_non_directory_and_public_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Image state and readiness records stay private regular files."""
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="private directory"):
        MacosSandboxImageBuilder(
            _ControlPlane(),
            link,  # type: ignore[arg-type]
        )

    regular = tmp_path / "regular"
    regular.write_text("not a directory", encoding="utf-8")
    with pytest.raises((FileExistsError, ValueError)):
        MacosSandboxImageBuilder(
            _ControlPlane(),
            regular,  # type: ignore[arg-type]
        )

    builder = _builder(tmp_path, _ControlPlane())
    installed = builder.prepare()
    installed.readiness_path.chmod(0o644)
    with pytest.raises(RuntimeError, match="readiness is invalid"):
        builder.prepare()

    replaced = tmp_path / "replaced"
    real_is_dir = Path.is_dir
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda path: False if path == replaced else real_is_dir(path),
    )
    with pytest.raises(ValueError, match="private directory"):
        MacosSandboxImageBuilder(_ControlPlane(), replaced)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("inventory", "message"),
    [
        (b"\xff", "malformed"),
        (b"", "incomplete"),
        (
            _inventory().replace(b"PACKAGE\tbash\t", b"PACKAGE\tbash\t0\nPACKAGE\tbash\t"),
            "malformed",
        ),
        (_inventory().replace(b"PACKAGE\tbash\t5.2.15-2+b13\n", b""), "package inventory"),
        (_inventory().replace(b"COMMAND\tbash\t/usr/bin/bash\n", b""), "command inventory"),
        (_inventory() + b"unexpected\n", "malformed"),
    ],
)
def test_builder_rejects_malformed_or_incomplete_inventory(
    tmp_path: Path, inventory: bytes, message: str
) -> None:
    """Inventory parsing rejects corrupt, duplicate, and missing declarations."""
    with pytest.raises(RuntimeError, match=message):
        _builder(tmp_path, _ControlPlane(_framed(inventory=inventory))).prepare()


def test_builder_retains_only_active_and_rollback_records(tmp_path: Path) -> None:
    """Activation rotates one rollback and reclaims the previously retained image."""
    first = _builder(tmp_path, _ControlPlane()).prepare()
    original = load_sandbox_image_definition()
    second_definition = original.model_copy(
        update={"packages": {**original.packages, "extra-one": "1.0"}}
    )
    second_index = "sha256:" + "d" * 64
    second_manifest = "sha256:" + "e" * 64
    second_plane = _ControlPlane(
        _framed(
            inspection=_inspection(second_index, second_manifest),
            inventory=_inventory(second_definition),
        )
    )
    second = _builder(tmp_path, second_plane, second_definition).prepare()
    third_definition = original.model_copy(
        update={"packages": {**original.packages, "extra-two": "2.0"}}
    )
    third_index = "sha256:" + "f" * 64
    third_plane = _ControlPlane(
        _framed(inspection=_inspection(third_index), inventory=_inventory(third_definition))
    )
    third = _builder(tmp_path, third_plane, third_definition).prepare()

    assert (
        json.loads((tmp_path / "images" / "active.json").read_bytes())["image"]["reference"]
        == third.image.reference
    )
    assert (
        json.loads((tmp_path / "images" / "rollback.json").read_bytes())["image"]["reference"]
        == second.image.reference
    )
    assert third_plane.removals == [first.image.reference]


def test_rebuild_removes_a_duplicate_rollback_record(tmp_path: Path) -> None:
    """Rebuilding the same absent image does not retain an identical rollback record."""
    plane = _ControlPlane()
    builder = _builder(tmp_path, plane)
    active = builder.prepare()
    _atomic_private_write(
        tmp_path / "images" / "rollback.json",
        active.readiness.model_dump_json().encode(),
    )
    plane.inspect_result = InfrastructureProcessResult(1, b"", b"missing", False, False)

    builder.prepare()

    assert not (tmp_path / "images" / "rollback.json").exists()


def test_builder_cleans_new_image_when_rotation_cannot_reclaim_old_rollback(
    tmp_path: Path,
) -> None:
    """A failed bounded-retention cleanup leaves the prior active record unchanged."""
    original = load_sandbox_image_definition()
    active = _builder(tmp_path, _ControlPlane()).prepare()
    rollback = active.readiness.model_copy(
        update={
            "image": active.image.model_copy(
                update={"reference": "loop.local/sandbox@sha256:" + "9" * 64}
            )
        }
    )
    _atomic_private_write(
        tmp_path / "images" / "rollback.json",
        rollback.model_dump_json().encode(),
    )
    changed = original.model_copy(update={"packages": {**original.packages, "extra": "1.0"}})
    failed_plane = _ControlPlane(_framed(inventory=_inventory(changed)))
    failed_plane.remove_result = InfrastructureProcessResult(1, b"", b"busy", False, False)

    with pytest.raises(RuntimeError, match="could not be reclaimed"):
        _builder(tmp_path, failed_plane, changed).prepare()

    assert failed_plane.removals[-1] == f"loop.local/sandbox@{_INDEX}"
    assert (
        json.loads((tmp_path / "images" / "active.json").read_bytes())["source_version"]
        == original.source_version
    )


def test_atomic_private_write_removes_temporary_file_on_stream_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupted readiness writes leave neither active nor temporary data."""

    def fail_fdopen(*args: object, **kwargs: object) -> object:
        """Simulate a stream-open failure after temporary creation."""
        del args, kwargs
        raise OSError("write failed")

    monkeypatch.setattr(os, "fdopen", fail_fdopen)
    destination = tmp_path / "readiness.json"
    with pytest.raises(OSError, match="write failed"):
        _atomic_private_write(destination, b"readiness")
    assert not destination.exists()
    assert not list(tmp_path.glob(".readiness.json.*"))
