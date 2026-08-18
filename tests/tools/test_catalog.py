"""Tests for explicit composition of the built-in tool catalog."""

import importlib
from unittest.mock import Mock

from loop import BUILTIN_TOOLS, Interaction, PermissionManager, ToolRegistry
from loop.tools import create_default_tool_registry


def test_importing_tools_does_not_mutate_an_existing_registry():
    """Importing or reloading built-ins never registers them into unrelated containers."""
    registry = ToolRegistry()

    importlib.reload(importlib.import_module("loop.tools"))

    assert registry.names == []


def test_builtin_manifest_has_every_tool_in_deterministic_order():
    """The manifest exposes the complete standard capability set in stable order."""
    assert [function.__name__ for function in BUILTIN_TOOLS] == [
        "get_current_datetime",
        "list_folder",
        "read_text_file",
        "write_text_file",
        "delete_path",
        "manage_skills",
        "run_command",
        "fetch_content",
        "read_cached_content",
    ]


def test_default_registry_factory_returns_isolated_configured_registries():
    """Each factory call registers all built-ins with independently injected runtime state."""
    interaction = Mock(spec=Interaction)
    permissions = PermissionManager(interaction=interaction)

    configured = create_default_tool_registry(
        interaction=interaction,
        permission_manager=permissions,
    )
    independent = create_default_tool_registry()

    assert (
        configured.names
        == independent.names
        == sorted(function.__name__ for function in BUILTIN_TOOLS)
    )
    assert configured is not independent
    assert configured.interaction is interaction
    assert configured.permission_manager is permissions
    assert independent.permission_manager is not permissions
