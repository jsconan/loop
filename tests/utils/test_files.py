"""Tests for general filesystem mutation utilities."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from loop.utils.files import is_binary_file, write_text_atomically
from loop.utils.hashing import sha256_digest


def test_is_binary_file_detects_nul_bytes_within_its_bounded_probe(tmp_path):
    """Binary detection distinguishes NUL-marked prefixes from ordinary UTF-8 text."""
    binary = tmp_path / "binary.dat"
    binary.write_bytes(b"text\0binary")
    text = tmp_path / "text.txt"
    text.write_text("ordinary text", encoding="utf-8")

    assert is_binary_file(binary) is True
    assert is_binary_file(text, probe_bytes=4) is False


def test_write_text_atomically_creates_and_replaces_expected_files(tmp_path):
    """Atomic text writes exclusively create new files and replace approved content."""
    target = tmp_path / "target.txt"

    write_text_atomically(target, "created", expected_digest=None)
    digest = sha256_digest(target.read_bytes())
    write_text_atomically(target, "replaced", expected_digest=digest)

    assert target.read_text(encoding="utf-8") == "replaced"


def test_write_text_atomically_rejects_changed_creation_and_replacement_targets(tmp_path):
    """Atomic text writes fail closed when destination state differs from the precondition."""
    target = tmp_path / "target.txt"
    target.write_text("existing", encoding="utf-8")

    with pytest.raises(RuntimeError, match="creation was cancelled"):
        write_text_atomically(target, "new", expected_digest=None)
    with pytest.raises(RuntimeError, match="replacement was cancelled"):
        write_text_atomically(target, "new", expected_digest="wrong")

    assert target.read_text(encoding="utf-8") == "existing"


def test_write_text_atomically_treats_a_symbolic_link_as_an_existing_creation_target(tmp_path):
    """Exclusive creation cannot replace a symbolic link, including a dangling one."""
    target = tmp_path / "target.txt"
    target.symlink_to(tmp_path / "missing.txt")

    with pytest.raises(RuntimeError, match="creation was cancelled"):
        write_text_atomically(target, "new", expected_digest=None)

    assert target.is_symlink()


def test_write_text_atomically_cleans_up_staged_content_after_commit_failure(
    tmp_path, monkeypatch
):
    """A failed atomic commit leaves the destination intact and removes its staging file."""
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    digest = sha256_digest(target.read_bytes())
    monkeypatch.setattr(Path, "replace", MagicMock(side_effect=OSError("commit failed")))

    with pytest.raises(OSError, match="commit failed"):
        write_text_atomically(target, "new", expected_digest=digest)

    assert list(tmp_path.iterdir()) == [target]
    assert target.read_text(encoding="utf-8") == "old"
