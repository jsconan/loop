"""Verify persisted Linux generation record authentication and retry evidence."""

import json
import os

import pytest

from loop.execution.sandbox.linux import host as linux_host
from loop.execution.sandbox.linux import records as linux_records

from .helpers import publish


@pytest.fixture
def boot_record(installation, monkeypatch, tmp_path):
    """Prepare a current-boot record beside an independently authenticated installation."""
    record, arguments = installation
    authority = publish(installation)
    root = tmp_path / "boot"
    root.mkdir()
    boot_id = tmp_path / "boot-id"
    boot_id.write_text("boot-identity\n")
    monkeypatch.setattr(linux_records, "SYSTEM_RUNTIME_DIRECTORY", root)
    monkeypatch.setattr(linux_records, "PROC_BOOT_ID_PATH", boot_id)
    config, files = arguments[:2]
    units = {
        "slice_unit": config.slice_unit,
        "socket_unit": files["socket_unit"],
        "service_unit": files["service_unit"],
    }
    value = {
        "schema": 1,
        "generation": config.generation,
        "record": authority.digest,
        "boot": "boot-identity",
        "namespace": [101, 202],
        "files": {name: bound.to_record() for name, bound in units.items()},
    }
    return record, arguments, root / "loopgen_test.boot", value


def test_boot_record_rebinds_recreated_volatile_units(boot_record):
    """A protected boot record binds exact replacement unit bytes and namespace identities."""
    record, arguments, marker, value = boot_record
    for role, original in (
        ("slice_unit", arguments[0].slice_unit),
        ("socket_unit", arguments[1]["socket_unit"]),
        ("service_unit", arguments[1]["service_unit"]),
    ):
        replacement = original.path.with_name(original.path.name + ".new")
        replacement.write_bytes(original.path.read_bytes())
        replacement.chmod(0o444)
        os.replace(replacement, original.path)
        value["files"][role] = linux_host.bind_host_file(original.path).to_record()
    marker.write_text(json.dumps(value))
    marker.chmod(0o444)
    recovered, _ = linux_records.read_generation(record)
    assert recovered.namespace == (101, 202)
    assert recovered.config.slice_unit == linux_host.bind_host_file(arguments[0].slice_unit.path)
    assert recovered.files["socket_unit"] == linux_host.bind_host_file(
        arguments[1]["socket_unit"].path
    )


@pytest.mark.parametrize(
    "change",
    [
        "mode",
        "schema",
        "schema_type",
        "generation",
        "record",
        "boot",
        "namespace",
        "namespace_type",
        "files",
        "unit",
    ],
)
def test_boot_record_rejects_changed_authority(boot_record, change):
    """Malformed, stale or widened boot metadata cannot rebind the original installation."""
    record, _, marker, value = boot_record
    if change == "schema":
        value["schema"] = True
    elif change == "schema_type":
        value["schema"] = "1"
    elif change == "generation":
        value["generation"] = "foreign"
    elif change == "record":
        value["record"] = "0" * 64
    elif change == "boot":
        value["boot"] = "previous-boot"
    elif change == "namespace":
        value["namespace"] = [1, True]
    elif change == "namespace_type":
        value["namespace"] = [1, "2"]
    elif change == "files":
        value["files"].pop("slice_unit")
    elif change == "unit":
        value["files"]["slice_unit"]["sha256"] = "0" * 64
    marker.write_text(json.dumps(value))
    marker.chmod(0o644 if change == "mode" else 0o444)
    with pytest.raises(ValueError, match="boot|Boot"):
        linux_records.read_generation(record)


def test_boot_recovery_binds_only_root_verified_exact_units(boot_record, monkeypatch):
    """Only initial-host root can adopt exact unit bytes while rebuilding boot state."""
    record, arguments, _, _ = boot_record
    mapping = record.parent.parent / "uid-map"
    mapping.write_text("0 0 4294967295\n")
    monkeypatch.setattr(linux_records, "PROC_UID_MAP_PATH", mapping)
    monkeypatch.setattr(linux_records.os, "getuid", lambda: 1000)
    with pytest.raises(PermissionError, match="Boot recovery"):
        linux_records.read_generation(record, partial=True, boot_recovery=True)
    monkeypatch.setattr(linux_records.os, "getuid", lambda: 0)
    original = arguments[1]["socket_unit"]
    replacement = original.path.with_name(original.path.name + ".new")
    replacement.write_bytes(original.path.read_bytes())
    replacement.chmod(0o444)
    os.replace(replacement, original.path)
    recovered, _ = linux_records.read_generation(record, partial=True, boot_recovery=True)
    assert recovered.files["socket_unit"] == linux_host.bind_host_file(original.path)


def test_partial_generation_retains_original_artifact_bindings(installation):
    """Checked teardown accepts a missing owned file without adopting a changed survivor."""
    record, arguments = installation
    publish(installation)
    original = arguments[-1]["loop_uninstall.py"]
    original.path.unlink()
    owner, _ = linux_records.read_generation(record, partial=True)
    assert owner.artifacts["loop_uninstall.py"] == original
    changed = arguments[-1]["bootstrap.bin"].path
    changed.chmod(0o644)
    changed.write_bytes(b"changed")
    changed.chmod(0o444)
    with pytest.raises(ValueError, match="Prepared native host authority changed"):
        linux_records.read_generation(record, partial=True)


def test_partial_generation_refuses_changed_unreferenced_artifact(installation):
    """Partial teardown rebinds artifacts that do not supply reconstructed policy bytes."""
    record, arguments = installation
    publish(installation)
    original = arguments[-1]["loop_uninstall.py"]
    original.path.chmod(0o755)
    original.path.write_bytes(b"changed uninstaller")
    original.path.chmod(original.identity[2] & 0o777)
    with pytest.raises(ValueError, match="Saved partial installation authority changed"):
        linux_records.read_generation(record, partial=True)


def test_partial_generation_reads_missing_policy_as_removed(installation):
    """A missing bound policy artifact becomes empty only during checked teardown."""
    record, arguments = installation
    publish(installation)
    arguments[-1]["bootstrap.bin"].path.unlink()
    owner, _ = linux_records.read_generation(record, partial=True)
    assert owner.policies[""]["bootstrap"] == b""


@pytest.mark.parametrize("change", ["shared_lock", "withdrawn_helper"])
def test_partial_generation_revalidates_shared_and_withdrawn_authority(installation, change):
    """Changed shared locks refuse retry while an exact withdrawn helper remains identifiable."""
    record, arguments = installation
    publish(installation)
    path = arguments[2].path if change == "shared_lock" else arguments[6].path
    if change == "shared_lock":
        path.chmod(0o644)
        path.write_bytes(b"changed")
        path.chmod(0o444)
        with pytest.raises(ValueError, match="Shared management authority changed"):
            linux_records.read_generation(record, partial=True)
    else:
        path.chmod(0o444)
        owner, _ = linux_records.read_generation(record, partial=True)
        assert owner.helper == arguments[6]


def test_generation_record_refuses_expired_deadline_before_binding(installation, monkeypatch):
    """Record reconstruction stops before opening authority after its deadline expires."""
    record, _ = installation
    publish(installation)
    monkeypatch.setattr(linux_records.time, "monotonic", lambda: 50)
    with pytest.raises(TimeoutError, match="record deadline"):
        linux_records.read_generation(record, deadline=50)
