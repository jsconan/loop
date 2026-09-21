"""Test durable bounded workspace-publication content staging."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pytest

from loop.execution.vfs import ContentReference, StagedContentStore


def test_content_is_staged_opened_reused_and_reclaimed(tmp_path: Path) -> None:
    """Immutable bytes remain restart-readable until their owner explicitly reclaims them."""
    root = tmp_path / "staging"
    store = StagedContentStore(root)
    first = store.stage("attempt", io.BytesIO(b"payload"), maximum_bytes=7)
    second = store.stage("attempt", io.BytesIO(b"payload"), maximum_bytes=7)

    assert first == second
    with StagedContentStore(root).open(first) as stream:
        assert stream.read() == b"payload"
    store.discard("attempt")
    store.discard("attempt")
    with pytest.raises(FileNotFoundError):
        store.open(first)


def test_staging_rejects_invalid_bounds_streams_and_references(tmp_path: Path) -> None:
    """Malformed identities, oversized data, text streams, and forged references fail closed."""
    store = StagedContentStore(tmp_path / "staging")
    with pytest.raises(ValueError, match="bounds"):
        store.stage("", io.BytesIO(), maximum_bytes=0)
    with pytest.raises(ValueError, match="byte limit"):
        store.stage("large", io.BytesIO(b"ab"), maximum_bytes=1)
    with pytest.raises(TypeError, match="non-bytes"):
        store.stage("text", io.StringIO("text"), maximum_bytes=10)  # type: ignore[arg-type]
    forged = ContentReference(digest="sha256:" + "0" * 64, size=0, reference="foreign")
    with pytest.raises(ValueError, match="reference"):
        store.open(forged)
    with pytest.raises(ValueError, match="identity"):
        store.discard("")


def test_staging_rejects_root_or_bucket_replacement(tmp_path: Path) -> None:
    """Symlinked buckets and replacement roots cannot redirect trusted recovery content."""
    root = tmp_path / "staging"
    store = StagedContentStore(root)
    bucket = root / hashlib.sha256(b"attempt").hexdigest()
    target = tmp_path / "target"
    target.mkdir()
    constructor_alias = tmp_path / "constructor-alias"
    constructor_alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="private directory"):
        StagedContentStore(constructor_alias)
    bucket.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="bucket"):
        store.stage("attempt", io.BytesIO(), maximum_bytes=0)
    bucket.unlink()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    root.rename(tmp_path / "old")
    replacement.rename(root)
    with pytest.raises(ValueError, match="root changed"):
        store.discard("attempt")


def test_staging_rejects_no_progress_and_bucket_aliases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stalled write or aliased open/discard bucket cannot cross the private store boundary."""
    store = StagedContentStore(tmp_path / "staging")
    original_write = os.write
    monkeypatch.setattr(os, "write", lambda descriptor, data: 0)
    with pytest.raises(OSError, match="no progress"):
        store.stage("stalled", io.BytesIO(b"value"), maximum_bytes=5)
    monkeypatch.setattr(os, "write", original_write)

    content = store.stage("attempt", io.BytesIO(b"value"), maximum_bytes=5)
    bucket = store.directory / hashlib.sha256(b"attempt").hexdigest()
    target = tmp_path / "target"
    bucket.rename(target)
    bucket.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="bucket"):
        store.open(content)
    with pytest.raises(ValueError, match="bucket"):
        store.discard("attempt")


def test_staging_validates_opened_object_size_and_kind(tmp_path: Path) -> None:
    """A replaced or size-mismatched staged object is rejected before publication reads it."""
    store = StagedContentStore(tmp_path / "staging")
    content = store.stage("attempt", io.BytesIO(b"value"), maximum_bytes=5)
    _, bucket, digest = content.reference.split(":")
    object_path = store.directory / bucket / digest
    object_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="object"):
        store.open(content)
    object_path.unlink()
    os.mkdir(object_path)
    with pytest.raises(ValueError, match="object"):
        store.open(content)
