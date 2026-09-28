"""Tests for explicit composition of the built-in tool catalog."""

import importlib
import json
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
        "search_text",
        "write_text_file",
        "edit_text_file",
        "delete_path",
        "activate_skill",
        "manage_skills",
        "resolve_executable",
        "run_command",
        "fetch_content",
        "read_cached_content",
    ]


def test_default_registry_factory_returns_isolated_configured_registries(monkeypatch):
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


def test_default_registry_keeps_text_search_without_ripgrep(monkeypatch):
    """Text search remains available without an external helper on PATH."""
    interaction = Mock(spec=Interaction)
    monkeypatch.setenv("PATH", "/missing")

    registry = create_default_tool_registry(interaction=interaction)

    assert "search_text" in registry.names
    interaction.warning.assert_not_called()


def test_default_registry_search_does_not_launch_workspace_path_program(tmp_path, monkeypatch):
    """Registration, replacement, and direct dispatch cannot run a project rg on the host."""
    from loop import CommandManager, ConsoleInteraction, InstructionsManager
    from loop.tooling import ToolCommands

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "source.txt"
    source.write_text("needle\n", encoding="utf-8")
    binary = workspace / "bin"
    binary.mkdir()
    fake = binary / "rg"
    marker = tmp_path / "outside"
    fake.write_text(f"#!/bin/sh\nprintf bypass > {marker}\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary))
    monkeypatch.setattr(PermissionManager, "request_permission", Mock(return_value=True))
    registry = create_default_tool_registry(permission_manager=PermissionManager(workspace))
    fake.write_text(f"#!/bin/sh\nprintf replaced > {marker}\n", encoding="utf-8")
    output = registry.call(
        "search_text",
        json.dumps({"path": str(source), "query": "needle"}),
        interaction=ConsoleInteraction(),
    )
    assert json.loads(output)["result"]["matches"][0]["text"] == "needle"
    interaction = Mock(spec=Interaction)
    commands = CommandManager(interaction=interaction)
    commands.register_provider(ToolCommands(registry, InstructionsManager()))
    commands.call("call", f'search_text "{source}" needle')
    interaction.tool_result.assert_called_once()
    assert not marker.exists()


def test_default_registry_search_falls_back_when_installed_rg_is_absent(tmp_path, monkeypatch):
    """The public search tool keeps literal and regex matches when rg is unavailable."""
    from loop import ConsoleInteraction

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("x" * 2000 + "\n€ needle\n", encoding="utf-8")
    second.write_text("needle\n", encoding="utf-8")
    monkeypatch.setattr(
        "loop.execution.facade.ripgrep_path",
        Mock(side_effect=FileNotFoundError("rg absent")),
    )
    registry = create_default_tool_registry(permission_manager=PermissionManager(tmp_path))
    interaction = ConsoleInteraction()
    literal = json.loads(
        registry.call(
            "search_text",
            json.dumps({"path": str(tmp_path), "query": "needle"}),
            interaction=interaction,
        )
    )
    expression = json.loads(
        registry.call(
            "search_text",
            json.dumps({"path": str(tmp_path), "query": r"(?<=€ )needle", "regex": True}),
            interaction=interaction,
        )
    )
    assert [(match["path"], match["line"]) for match in literal["result"]["matches"]] == [
        ("first.txt", 2),
        ("second.txt", 1),
    ]
    assert expression["result"]["matches"][0]["column"] == 3
    registry.close()


def test_default_registry_closes_its_injected_execution_service():
    """Registry ownership releases the application-selected native service once."""
    service = Mock()
    registry = create_default_tool_registry(execution_service=service)

    registry.close()

    service.close.assert_called_once()
