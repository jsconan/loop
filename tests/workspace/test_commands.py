"""Verify centralized workspace registry commands."""

from unittest.mock import Mock

import pytest
from prompt_toolkit.document import Document

from loop import CommandCompletionAdapter, CompletionManager
from loop.commands import CommandArgumentError, CommandContext
from loop.utils import PathHolder
from loop.workspace import (
    Workspace,
    WorkspaceCommands,
    WorkspaceRepository,
    WorkspaceSwitchRequested,
)


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

    assert [value.text for value in values] == [
        "attach",
        "forget",
        "list",
        "rekey",
        "rename",
        "show",
        "switch",
    ]


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


def test_workspace_lifecycle_commands_confirm_mutations_and_report_conflicts(commands, tmp_path):
    """Attach is immediate while forget and rekey support cancellation and recovery errors."""
    interaction = Mock()
    interaction.confirm.side_effect = [False, True, True, True]
    context = CommandContext("workspace", interaction)
    attached = tmp_path / "attached"
    attached.mkdir()

    commands.workspace(context, "attach", str(attached))
    commands.workspace(context, "forget", str(attached))
    commands.workspace(context, "forget", str(attached))
    commands.workspace(context, "rekey", str(attached))

    assert "cancelled" in interaction.info.call_args_list[1].args[0]
    assert "retained" in interaction.info.call_args_list[2].args[0]
    assert "Rekeyed" in interaction.info.call_args_list[3].args[0]
    with pytest.raises(CommandArgumentError, match="requires"):
        commands.workspace(context, "attach")
    with pytest.raises(CommandArgumentError, match="No active"):
        commands.workspace(context, "forget", str(tmp_path / "missing"))


def test_workspace_switch_resolves_target_before_signalling_rebuild(commands, tmp_path):
    """Switch requests carry a resolved identity and reject unknown targets."""
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(WorkspaceSwitchRequested) as request:
        commands.workspace(CommandContext("workspace", Mock()), "switch", str(target))
    assert request.value.workspace.root == target
    with pytest.raises(ValueError, match="Unknown"):
        commands.workspace(CommandContext("workspace", Mock()), "switch", "missing-id")
