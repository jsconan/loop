"""Verify centralized workspace registry commands."""

import shlex
from unittest.mock import Mock

import pytest
from prompt_toolkit.document import Document

from loop import CommandCompletionAdapter, CompletionManager
from loop.commands import CommandArgumentError, CommandContext
from loop.interaction import ConsoleInteraction
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
    interaction = Mock(spec=ConsoleInteraction)
    context = CommandContext("workspace", interaction)

    commands.workspace(context)
    commands.workspace(context, "list")
    commands.workspace(context, "rename", "My project")
    commands.workspace(context)

    assert interaction.table.call_args.args[0][0]["name"] == "My project"
    assert interaction.table.call_args.args[0][0]["name_source"] == "user"
    assert commands.get_commands()[0].name == "workspace"


def test_workspace_command_completion_shows_arguments_for_selected_operations(commands):
    """Workspace completion reveals only the selected operation's arguments and values."""
    manager = CompletionManager(
        (
            CommandCompletionAdapter(
                lambda: commands.get_commands(),
                providers=(commands,),
            ),
        )
    )

    actions = manager.get_completions(Document("/workspace "), Mock())
    attach = manager.get_completions(Document("/workspace attach "), Mock())
    attach_ids = manager.get_completions(Document("/workspace attach workspace_id="), Mock())
    attach_after_name = manager.get_completions(Document("/workspace attach project "), Mock())
    rename = manager.get_completions(Document("/workspace rename "), Mock())
    forget = manager.get_completions(Document("/workspace forget "), Mock())
    show = manager.get_completions(Document("/workspace show "), Mock())
    completion_sources = commands.get_completion_providers()
    workspace_id = completion_sources[0].provider()[0].value
    workspace_references = {value.value for value in completion_sources[1].provider()}

    assert {value.text for value in actions} == {
        "attach",
        "forget",
        "list",
        "rekey",
        "rename",
        "show",
        "switch",
    }
    assert {value.text for value in attach} == {"name=", "workspace_id="}
    assert {value.text for value in attach_ids} == {f"workspace_id={workspace_id}"}
    assert {value.text for value in attach_after_name} == {workspace_id}
    assert {value.text for value in rename} == {"name="}
    assert {value.text for value in forget} == workspace_references
    assert list(show) == []


def test_workspace_commands_reject_invalid_names_and_arguments(commands):
    """Registry commands reject empty renames and names on read operations."""
    context = CommandContext("workspace", Mock())

    with pytest.raises(CommandArgumentError, match="does not accept"):
        commands.workspace(context, "show", "unused")
    with pytest.raises(CommandArgumentError, match="non-empty"):
        commands.workspace(context, "rename", " ")
    with pytest.raises(CommandArgumentError, match="Only attach"):
        commands.workspace(context, "rename", "name", "identity")


def test_workspace_commands_require_initialized_identity(tmp_path):
    """Registry commands cannot operate before identity resolution."""
    with pytest.raises(ValueError, match="initialized"):
        WorkspaceCommands(Workspace.discover(tmp_path), Mock())


def test_workspace_lifecycle_commands_confirm_mutations_and_report_conflicts(commands, tmp_path):
    """Attach is immediate while forget and rekey support cancellation and recovery errors."""
    interaction = Mock(spec=ConsoleInteraction)
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


def test_workspace_attach_forwards_an_explicit_existing_identity(tmp_path):
    """The command surface can explicitly associate a path with a chosen identity."""
    workspace = Workspace(tmp_path, tmp_path, "active", "active", "directory", 1, 1)
    repository = Mock()
    repository.attach.return_value = workspace
    _commands = WorkspaceCommands(workspace, repository)

    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm.side_effect = [False, True]
    context = CommandContext("workspace", interaction)

    _commands.workspace(context, "attach", str(tmp_path), "chosen")
    repository.attach.assert_not_called()
    _commands.workspace(context, "attach", str(tmp_path), "chosen")

    repository.attach.assert_called_once_with(str(tmp_path), "chosen")


def test_workspace_completion_skips_locations_without_persisted_identity(tmp_path):
    """Workspace completion omits an ID candidate when a location has no identity."""
    active = Workspace(tmp_path, tmp_path, "active", "Active", "directory", 1, 1)
    repository = Mock()
    repository.list.return_value = (Workspace.discover(tmp_path),)
    commands = WorkspaceCommands(active, repository)
    providers = commands.get_completion_providers()

    assert providers[0].provider() == ()
    assert {value.value for value in providers[1].provider()} == {str(tmp_path.resolve())}


def test_workspace_path_completion_quotes_paths_with_spaces(tmp_path):
    """Workspace path candidates preserve spaces when inserted into command input."""
    root = tmp_path / "Project Root"
    root.mkdir()
    workspace = Workspace(root, root, "workspace-id", "Project Root", "directory", 1, 1)
    repository = Mock()
    repository.list.return_value = (workspace,)
    commands = WorkspaceCommands(workspace, repository)
    references = commands.get_completion_providers()[1].provider()

    path = next(value for value in references if value.description.startswith("Workspace path:"))

    assert path.value == shlex.quote(str(root.resolve()))
    assert path.display == str(root.resolve())


def test_workspace_switch_resolves_target_before_signalling_rebuild(commands, tmp_path):
    """Switch requests carry a resolved identity and reject unknown targets."""
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(WorkspaceSwitchRequested) as request:
        commands.workspace(CommandContext("workspace", Mock()), "switch", str(target))
    assert request.value.workspace.root == target
    with pytest.raises(CommandArgumentError, match="already in progress"):
        commands.workspace(CommandContext("workspace", Mock()), "switch", str(target))
    request.value.complete()
    request.value.complete()
    with pytest.raises(ValueError, match="Unknown"):
        commands.workspace(CommandContext("workspace", Mock()), "switch", "missing-id")
