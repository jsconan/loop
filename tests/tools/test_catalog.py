"""Tests for explicit composition of the built-in tool catalog."""

import importlib
from unittest.mock import Mock

from loop import BUILTIN_TOOLS, Interaction, PermissionManager, ToolRegistry
from loop.tools import create_default_tool_registry

files_module = importlib.import_module("loop.tools.files")


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
        "search_text",
        "write_text_file",
        "edit_text_file",
        "delete_path",
        "manage_skills",
        "run_command",
        "fetch_content",
        "read_cached_content",
    ]


def test_default_registry_factory_returns_isolated_configured_registries(monkeypatch):
    """Each factory call registers all built-ins with independently injected runtime state."""
    monkeypatch.setattr(files_module, "ripgrep_path", Mock(return_value="rg"))
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


def test_default_registry_omits_text_search_when_ripgrep_is_unavailable(monkeypatch):
    """The built-in catalog excludes text search and reports its missing executable."""
    interaction = Mock(spec=Interaction)
    monkeypatch.setattr(
        files_module,
        "ripgrep_path",
        Mock(side_effect=FileNotFoundError("Install ripgrep.")),
    )

    registry = create_default_tool_registry(interaction=interaction)

    assert "search_text" not in registry.names
    assert registry.names == sorted(
        function.__name__ for function in BUILTIN_TOOLS if function.__name__ != "search_text"
    )
    interaction.warning.assert_called_once_with("Install ripgrep.")
