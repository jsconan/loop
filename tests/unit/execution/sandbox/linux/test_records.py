"""Verify persisted Linux generation record authentication and retry evidence."""

import pytest

from loop.execution.sandbox.linux import records as linux_records

from .helpers import publish


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
