"""Tests for skill-domain models."""

import pytest

from loop.instructions.models import (
    CapturedInstruction,
    InstructionSection,
    LiveInstructionSource,
)
from loop.utils.hashing import sha256_digest


def test_instruction_section_reports_encoded_size_and_digest():
    """Instruction sections derive stable byte size and digest metadata from content."""
    section = InstructionSection("agents", "héllo", "/project/AGENTS.md")

    assert section.size_bytes == 6
    assert section.digest == "3c48591d8d098a4538f5e013dfcf406e948eac4d3277b10bf614e295d6068179"


def test_captured_instruction_relocates_provenance_but_keeps_captured_content(tmp_path):
    """Immutable provenance rebases its source metadata without rereading changed content."""
    original = tmp_path / "original"
    moved = tmp_path / "moved"
    internal = original / "AGENTS.md"
    external = tmp_path / "shared.md"
    internal.parent.mkdir()
    internal.write_text("policy", encoding="utf-8")
    external.write_text("shared", encoding="utf-8")
    reference = CapturedInstruction.capture(
        internal, "policy", workspace_id="workspace", workspace_root=original
    )
    outside = CapturedInstruction.capture(
        external, "shared", workspace_id="workspace", workspace_root=original
    )
    moved.mkdir()
    (moved / "AGENTS.md").write_text("current", encoding="utf-8")

    assert reference.workspace_relative_path == "AGENTS.md"
    assert reference.resolve("workspace", moved) == moved / "AGENTS.md"
    assert reference.content() == "policy"
    assert reference.content_digest == sha256_digest(reference.content())
    assert outside.workspace_id is None
    assert outside.resolve("workspace", moved) == external
    assert outside.content() == "shared"
    with pytest.raises(ValueError, match="different workspace"):
        reference.resolve("other", moved)


def test_captured_instruction_always_retains_content_and_rejects_digest_mismatch(tmp_path):
    """Captures remain readable after source loss and reject inconsistent provenance."""
    path = tmp_path / "missing.md"
    saved = CapturedInstruction.capture(path, "saved")
    capture = CapturedInstruction.capture(path, "lost")

    assert saved.content() == "saved"
    assert capture.content() == "lost"
    assert capture.content_digest == sha256_digest(capture.content())
    with pytest.raises(ValueError, match="does not match"):
        CapturedInstruction(None, None, path, "wrong", "content")


def test_live_instruction_source_observes_changes_and_creates_fresh_captures(tmp_path):
    """Live sources read current content and capture a new digest after file changes."""
    original = tmp_path / "original"
    moved = tmp_path / "moved"
    path = original / "AGENTS.md"
    original.mkdir()
    moved.mkdir()
    path.write_text("first", encoding="utf-8")
    source = LiveInstructionSource.from_path(
        path, workspace_id="workspace", workspace_root=original
    )

    first = source.capture("workspace", original)
    (moved / "AGENTS.md").write_text("second", encoding="utf-8")
    second = source.capture("workspace", moved)
    external_path = tmp_path / "shared.md"
    external_path.write_text("shared", encoding="utf-8")
    external = LiveInstructionSource.from_path(
        external_path, workspace_id="workspace", workspace_root=original
    )
    plain = LiveInstructionSource.from_path(external_path)

    assert source.content("workspace", moved) == "second"
    assert isinstance(first, CapturedInstruction)
    assert first.content() == "first"
    assert second.content() == "second"
    assert first.content_digest != second.content_digest
    assert second.workspace_relative_path == "AGENTS.md"
    assert external.capture("workspace", moved).content() == "shared"
    assert plain.captured_absolute_path == external_path
    with pytest.raises(ValueError, match="different workspace"):
        source.resolve("other", moved)
