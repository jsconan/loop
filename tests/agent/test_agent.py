"""Tests for agent capability configuration."""

from unittest.mock import Mock

import pytest

from loop import Agent, InstructionsManager, PermissionManager, ToolRegistry


def test_agent_exposes_its_identity_and_capabilities(tmp_path):
    """Agent accessors preserve the independently configured runtime capabilities."""
    backend = Mock()
    instructions = InstructionsManager()
    tools = ToolRegistry()
    permissions = PermissionManager(tmp_path)

    agent = Agent(
        "Reviewer",
        backend=backend,
        instructions_manager=instructions,
        tool_registry=tools,
        permission_manager=permissions,
    )

    assert agent.name == "Reviewer"
    assert agent.backend is backend
    assert agent.instructions_manager is instructions
    assert agent.tool_registry is tools
    assert agent.permission_manager is permissions


def test_agent_rejects_an_empty_identity(tmp_path):
    """Agent construction rejects identities that cannot be presented or traced."""
    with pytest.raises(ValueError, match="must not be empty"):
        Agent(
            "  ",
            backend=Mock(),
            instructions_manager=InstructionsManager(),
            tool_registry=ToolRegistry(),
            permission_manager=PermissionManager(tmp_path),
        )
