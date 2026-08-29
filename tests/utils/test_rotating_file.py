"""Tests for owner-private rotating text files."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from loop.utils.rotating_file import PrivateRotatingTextFile


def test_private_rotating_text_file_rotates_complete_utf8_text_and_keeps_private_archives(tmp_path):
    """Rotation uses encoded size, keeps complete appends, and secures every retained file."""
    output = PrivateRotatingTextFile(
        tmp_path / "nested" / "audit.jsonl", max_bytes=3, backup_count=2
    )

    output.append("é\n")
    output.append("b\n")
    output.append("c\n")
    output.append("d\n")

    paths = tuple(
        (tmp_path / "nested" / "audit.jsonl").with_name(f"audit.jsonl{suffix}")
        for suffix in ("", ".1", ".2")
    )
    assert [path.read_text(encoding="utf-8") for path in paths] == ["d\n", "c\n", "b\n"]
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths)
    assert (tmp_path / "nested").stat().st_mode & 0o777 == 0o700


def test_private_rotating_text_file_keeps_an_oversized_append_intact(tmp_path):
    """An oversized append rotates before writing and is never split."""
    path = tmp_path / "audit.jsonl"
    output = PrivateRotatingTextFile(path, max_bytes=3, backup_count=1)

    output.append("old\n")
    output.append("oversized\n")

    assert path.read_text(encoding="utf-8") == "oversized\n"
    assert path.with_name("audit.jsonl.1").read_text(encoding="utf-8") == "old\n"


def test_private_rotating_text_file_can_disable_rotation(tmp_path):
    """Missing size limits and zero backups retain all appended text in the active file."""
    path = tmp_path / "audit.jsonl"

    PrivateRotatingTextFile(path).append("first\n")
    PrivateRotatingTextFile(path, max_bytes=1, backup_count=0).append("second\n")

    assert path.read_text(encoding="utf-8") == "first\nsecond\n"
    assert not path.with_name("audit.jsonl.1").exists()


@pytest.mark.parametrize("kwargs", ({"max_bytes": -1}, {"backup_count": -1}))
def test_private_rotating_text_file_rejects_negative_limits(tmp_path, kwargs):
    """Configuration rejects limits that cannot describe a valid retention policy."""
    with pytest.raises(ValueError, match="must be non-negative"):
        PrivateRotatingTextFile(tmp_path / "audit.jsonl", **kwargs)


def test_private_rotating_text_file_propagates_rotation_failure(tmp_path, monkeypatch):
    """Append exposes storage failures for its owning boundary to handle."""
    path = tmp_path / "audit.jsonl"
    output = PrivateRotatingTextFile(path, max_bytes=1, backup_count=1)
    output.append("first\n")
    monkeypatch.setattr(Path, "replace", Mock(side_effect=OSError("unavailable")))

    with pytest.raises(OSError, match="unavailable"):
        output.append("second\n")
