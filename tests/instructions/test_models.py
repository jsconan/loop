"""Tests for skill-domain models."""

from loop.instructions.models import InstructionSection


def test_instruction_section_reports_encoded_size_and_digest():
    """Instruction sections derive stable byte size and digest metadata from content."""
    section = InstructionSection("agents", "héllo", "/project/AGENTS.md")

    assert section.size_bytes == 6
    assert section.digest == "3c48591d8d098a4538f5e013dfcf406e948eac4d3277b10bf614e295d6068179"
