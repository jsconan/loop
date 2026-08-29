"""Tests for immutable agent definitions."""

import pytest

from loop import Agent, AgentIdentity, AgentInstructions, ToolRegistry


def test_agent_instructions_load_render_and_digest_the_bundled_contract():
    """Default instructions are versioned, delimited, and expose stable provenance."""
    instructions = AgentInstructions.default()

    assert instructions.version == "1"
    assert instructions.source == "loop.agent/instructions.md"
    assert "an agent that collaborates" in instructions.content
    assert instructions.render().startswith(
        '<agent_instructions version="1" source="loop.agent/instructions.md">'
    )
    assert instructions.render().endswith("</agent_instructions>")
    assert len(instructions.digest) == 64


@pytest.mark.parametrize("field", ["content", "version", "source"])
def test_agent_instructions_reject_empty_required_fields(field):
    """Every instruction provenance and content field must contain meaningful text."""
    values = {"content": "Policy.", "version": "1", "source": "tests"}
    values[field] = "  "

    with pytest.raises(ValueError, match=field):
        AgentInstructions(**values)


def test_agent_instructions_escape_wrapper_metadata():
    """Instruction provenance cannot break the model-facing composition wrapper."""
    rendered = AgentInstructions("Policy.", '1&"', "a<b").render()

    assert 'version="1&amp;&quot;"' in rendered
    assert 'source="a&lt;b"' in rendered


def test_agent_exposes_its_identity_instructions_and_tools():
    """An agent contains only reusable model-facing configuration."""
    instructions = AgentInstructions("Review carefully.", "1", "tests")
    tools = ToolRegistry()

    agent = Agent(AgentIdentity("Reviewer", "Reviews code."), instructions, tools)

    assert agent.name == "Reviewer"
    assert agent.identity.description == "Reviews code."
    assert agent.instructions is instructions
    assert agent.tools is tools
    assert "Review carefully." in agent.render()


def test_agent_renders_only_explicit_identity_placeholders():
    """Identity rendering safely substitutes its fields without general string formatting."""
    instructions = AgentInstructions(
        "You are {{name}}. {{description}} Keep {other} literal.", "1", "tests"
    )
    agent = Agent(AgentIdentity("A&B"), instructions)

    rendered = agent.render()

    assert "You are A&amp;B." in rendered
    assert "{{description}}" not in rendered
    assert "{other}" in rendered


def test_agent_renders_identity_placeholders_only_in_instruction_content():
    """Identity placeholders leave instruction provenance metadata unchanged."""
    instructions = AgentInstructions("You are {{name}}.", "{{description}}", "tests/{{name}}")
    agent = Agent(AgentIdentity("Reviewer", "Reviews code."), instructions)

    rendered = agent.render()

    assert '<agent_instructions version="{{description}}" source="tests/{{name}}">' in rendered
    assert "You are Reviewer." in rendered


def test_agent_rejects_an_empty_identity():
    """Agent construction rejects identities that cannot be presented or traced."""
    with pytest.raises(ValueError, match="must not be empty"):
        Agent("  ")
