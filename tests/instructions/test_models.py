"""Tests for skill-domain models."""

import pytest

from loop.instructions.models import AgentPolicy, InstructionSection


def test_agent_policy_loads_renders_and_digests_the_bundled_contract():
    """The default policy is versioned, delimited, and exposes stable provenance."""
    policy = AgentPolicy.default()

    assert policy.version == "1"
    assert policy.source == "loop.instructions/agent_policy.md"
    assert "You are Loop" in policy.content
    assert policy.render().startswith(
        '<agent_policy version="1" source="loop.instructions/agent_policy.md">'
    )
    assert policy.render().endswith("</agent_policy>")
    assert len(policy.digest) == 64


@pytest.mark.parametrize("field", ["content", "version", "source"])
def test_agent_policy_rejects_empty_required_fields(field):
    """Every policy identity and content field must contain meaningful text."""
    values = {"content": "Policy.", "version": "1", "source": "tests"}
    values[field] = "  "

    with pytest.raises(ValueError, match=field):
        AgentPolicy(**values)


def test_agent_policy_escapes_wrapper_metadata():
    """Policy provenance cannot break the model-facing composition wrapper."""
    rendered = AgentPolicy("Policy.", '1&"', "a<b").render()

    assert 'version="1&amp;&quot;"' in rendered
    assert 'source="a&lt;b"' in rendered


def test_instruction_section_reports_encoded_size_and_digest():
    """Instruction sections derive stable byte size and digest metadata from content."""
    section = InstructionSection("agents", "héllo", "/project/AGENTS.md")

    assert section.size_bytes == 6
    assert section.digest == "3c48591d8d098a4538f5e013dfcf406e948eac4d3277b10bf614e295d6068179"
