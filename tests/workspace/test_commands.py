"""Verify centralized workspace registry commands."""

from unittest.mock import Mock

import pytest
from prompt_toolkit.document import Document

from loop import CommandCompletionAdapter, CompletionManager
from loop.commands import CommandArgumentError, CommandContext
from loop.utils import PathHolder
from loop.workspace import Workspace, WorkspaceCommands, WorkspaceRepository


@pytest.fixture
def commands(tmp_path):
    """Return commands for one initialized isolated workspace."""
    catalog = PathHolder(tmp_path / "app" / "workspaces.db")
    repository = WorkspaceRepository(catalog)
    workspace = repository.initialize(Workspace.discover(tmp_path / "project"))
    return WorkspaceCommands(workspace, repository)


def test_workspace_commands_show_list_and_rename_active_workspace(commands):
    """Registry commands display locations and persist explicit user names."""
    interaction = Mock()
    context = CommandContext("workspace", interaction)

    commands.workspace(context)
    commands.workspace(context, "list")
    commands.workspace(context, "rename", "My project")
    commands.workspace(context)

    assert interaction.table.call_args.args[0][0]["name"] == "My project"
    assert interaction.table.call_args.args[0][0]["name_source"] == "user"
    assert commands.get_commands()[0].name == "workspace"


def test_workspace_command_completion_uses_positional_operations(commands):
    """Workspace completion offers operations without inaccurate named parameters."""
    manager = CompletionManager((CommandCompletionAdapter(lambda: commands.get_commands()),))

    values = manager.get_completions(Document("/workspace "), Mock())

    assert [value.text for value in values] == ["list", "rename", "show"]


def test_workspace_commands_reject_invalid_names_and_arguments(commands):
    """Registry commands reject empty renames and names on read operations."""
    context = CommandContext("workspace", Mock())

    with pytest.raises(CommandArgumentError, match="does not accept"):
        commands.workspace(context, "show", "unused")
    with pytest.raises(CommandArgumentError, match="non-empty"):
        commands.workspace(context, "rename", " ")


def test_workspace_commands_require_initialized_identity(tmp_path):
    """Registry commands cannot operate before identity resolution."""
    with pytest.raises(ValueError, match="initialized"):
        WorkspaceCommands(Workspace.discover(tmp_path), Mock())
