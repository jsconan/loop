"""Tests for skill-domain models."""

import pytest

from loop.instructions.models import InstructionReference, InstructionSection


def test_instruction_section_reports_encoded_size_and_digest():
    """Instruction sections derive stable byte size and digest metadata from content."""
    section = InstructionSection("agents", "héllo", "/project/AGENTS.md")

    assert section.size_bytes == 6
    assert section.digest == "3c48591d8d098a4538f5e013dfcf406e948eac4d3277b10bf614e295d6068179"


def test_instruction_reference_relocates_internal_paths_and_retains_external_paths(tmp_path):
    """Durable provenance rebases only sources owned by the matching workspace."""
    original = tmp_path / "original"
    moved = tmp_path / "moved"
    internal = original / "AGENTS.md"
    external = tmp_path / "shared.md"
    internal.parent.mkdir()
    internal.write_text("policy", encoding="utf-8")
    external.write_text("shared", encoding="utf-8")
    reference = InstructionReference.capture(
        internal, "policy", workspace_id="workspace", workspace_root=original, snapshot=True
    )
    outside = InstructionReference.capture(
        external, "shared", workspace_id="workspace", workspace_root=original
    )
    moved.mkdir()
    (moved / "AGENTS.md").write_text("current", encoding="utf-8")

    assert reference.workspace_relative_path == "AGENTS.md"
    assert reference.resolve("workspace", moved) == moved / "AGENTS.md"
    assert reference.content("workspace", moved) == "current"
    assert outside.workspace_id is None
    assert outside.resolve("workspace", moved) == external
    assert outside.content("workspace", moved) == "shared"
    with pytest.raises(ValueError, match="different workspace"):
        reference.resolve("other", moved)


def test_instruction_reference_uses_snapshot_or_reports_missing_source(tmp_path):
    """Missing sources recover only when capture explicitly retained a snapshot."""
    path = tmp_path / "missing.md"
    snapshot = InstructionReference.capture(path, "saved", snapshot=True)
    live = InstructionReference.capture(path, "lost")

    assert snapshot.content("workspace", tmp_path) == "saved"
    with pytest.raises(FileNotFoundError):
        live.content("workspace", tmp_path)
