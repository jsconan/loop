"""Tests for interactive application-configuration commands."""

from unittest.mock import Mock, call

from prompt_toolkit.document import Document

from loop.commands import CommandManager
from loop.completion import CommandCompletionAdapter, CompletionManager
from loop.configuration import ConfigurationCommands, ConfigurationManager


def test_config_set_applies_a_session_override(tmp_path):
    """The direct set form changes the active snapshot and calls its runtime applicator."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    apply = Mock(return_value="applied now")
    interaction = Mock()
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply),), interaction=interaction
    )

    manager.call("config", "set loop.debug true scope=session")

    assert configuration.effective.loop.debug is True
    assert configuration.source_for("loop.debug") == "session"
    apply.assert_called_once_with("loop.debug", configuration.effective)
    interaction.info.assert_called_once_with("Updated loop.debug for session: applied now")


def test_config_get_rejects_secret_entries(tmp_path):
    """The read-only get form directs secret edits to the masked secret operation."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    interaction = Mock()
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),), interaction=interaction
    )

    manager.call("config", "get backend.api_key")

    interaction.table.assert_not_called()
    assert (
        "Secret fields are not displayed with 'get'" in interaction.report.call_args.args[0].detail
    )


def test_config_get_displays_non_secret_entries(tmp_path):
    """The read-only get form presents an individual non-secret configuration field."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    interaction = Mock()
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),), interaction=interaction
    )

    manager.call("config", "get loop.debug")

    entries = interaction.table.call_args.args[0]
    assert [entry.path for entry in entries] == ["loop.debug"]


def test_config_without_arguments_displays_only_non_secret_entries(tmp_path):
    """The read-only config command omits secret configuration entries."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    interaction = Mock()
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),), interaction=interaction
    )

    manager.call("config", "")

    entries = interaction.table.call_args.args[0]
    assert len(entries) == len(configuration.entries) - 1
    assert all(not entry.secret for entry in entries)


def test_config_set_applies_a_file_override(tmp_path):
    """The direct set form persists the selected scope without a separate command."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    apply = Mock(return_value="applied now")
    interaction = Mock()
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply),), interaction=interaction
    )

    manager.call("config", "set loop.stream false scope=file")

    assert configuration.effective.loop.stream is False
    assert configuration.path.exists()
    assert apply.called


def test_config_secret_prompts_before_selecting_scope_and_applies_value(tmp_path):
    """The secret form reads masked input before selecting its destination scope."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    apply = Mock(return_value="backend replaced")
    interaction = Mock()
    interaction.prompt.side_effect = ["new-api-key", "session"]
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply),), interaction=interaction
    )

    manager.call("config", "secret backend.api_key")

    assert configuration.effective.backend.api_key.get_secret_value() == "new-api-key"
    assert interaction.prompt.call_args_list == [
        call("Enter backend.api_key:", secret=True),
        call(
            "Choose configuration scope:",
            choices={
                "file": "Save to .loop/config.toml",
                "session": "This session only",
                "cancel": "Cancel",
            },
            index={"file": "f", "session": "s", "cancel": "c"},
        ),
    ]
    apply.assert_called_once_with("backend.api_key", configuration.effective)
    interaction.info.assert_called_once_with(
        "Updated backend.api_key for session: backend replaced"
    )


def test_config_set_rejects_secret_entries_and_secret_rejects_plain_entries(tmp_path):
    """Normal set and secret actions accept only their matching entry types."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    interaction = Mock()
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),), interaction=interaction
    )

    manager.call("config", "set backend.api_key exposed scope=session")
    manager.call("config", "secret loop.debug scope=session")
    manager.call("config", "secret backend.api_key exposed scope=session")
    manager.call("config", "set unknown.path ignored scope=session")
    manager.call("config", "secret")

    assert configuration.effective.backend.api_key.get_secret_value() == "local-api-key"
    assert interaction.report.call_count == 5
    interaction.prompt.assert_not_called()


def test_config_secret_cancellation_leaves_the_existing_value_unchanged(tmp_path):
    """Cancelling masked input or the later scope selection changes no configuration."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    interaction = Mock()
    interaction.prompt.side_effect = [False, "new-api-key", "cancel"]
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),), interaction=interaction
    )

    manager.call("config", "secret backend.api_key")
    manager.call("config", "secret backend.api_key")

    assert configuration.effective.backend.api_key.get_secret_value() == "local-api-key"
    interaction.info.assert_called_once_with("Secret backend.api_key was not changed.")


def test_config_reset_and_invalid_direct_forms_report_argument_errors(tmp_path):
    """Repeated resets store defaults and invalid forms use normal command diagnostics."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    interaction = Mock()
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock(return_value="applied now")),),
        interaction=interaction,
    )
    manager.call("config", "set loop.debug true scope=session")

    interaction.confirm.side_effect = [True, True, True, True]
    manager.call("config", "reset loop.debug scope=session")
    manager.call("config", "reset unknown.path scope=session")
    manager.call("config", "reset loop.debug scope=session")
    manager.call("config", "reset loop.debug scope=session")
    manager.call("config", "set loop.debug")
    manager.call("config", "set")
    manager.call("config", "get unknown.path")
    manager.call("config", "get")

    assert configuration.effective.loop.debug is False
    assert interaction.report.call_count == 5


def test_config_reset_with_explicit_scope_requires_confirmation(tmp_path):
    """An explicit reset scope is confirmed before changing configuration."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    configuration.set_session("loop.debug", True)
    interaction = Mock()
    interaction.confirm.return_value = False
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock(return_value="applied now")),),
        interaction=interaction,
    )

    manager.call("config", "reset loop.debug scope=session")

    assert configuration.effective.loop.debug is True
    interaction.confirm.assert_called_once_with("Reset loop.debug for session?", default=False)


def test_config_reset_all_and_validation_failures_leave_state_consistent(tmp_path):
    """Reset without a path clears every selected-scope override and invalid writes fail safely."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    interaction = Mock()
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock(return_value="applied now")),),
        interaction=interaction,
    )

    interaction.prompt.side_effect = ["session", "session", "session", "cancel", "cancel", "cancel"]
    manager.call("config", "set loop.debug true")
    manager.call("config", "set loop.max_agent_turns -1")
    manager.call("config", "reset")
    manager.call("config", "reset")
    manager.call("config", "reset loop.debug")
    manager.call("config", "set loop.debug true")

    assert configuration.effective.loop.debug is False
    assert interaction.report.call_count == 1


def test_config_scope_picker_cancels_reset_all(tmp_path):
    """The cancel choice leaves all session overrides untouched."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    configuration.set_session("loop.debug", True)
    interaction = Mock()
    interaction.prompt.return_value = "cancel"
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock(return_value="applied now")),),
        interaction=interaction,
    )

    manager.call("config", "reset")

    assert configuration.effective.loop.debug is True
    interaction.prompt.assert_called_once_with(
        "Choose configuration scope:",
        choices={
            "file": "Save to .loop/config.toml",
            "session": "This session only",
            "cancel": "Cancel",
        },
        index={"file": "f", "session": "s", "cancel": "c"},
    )


def test_config_completion_excludes_secret_paths_from_get_and_set(tmp_path):
    """Get and set complete only non-secret schema-derived configuration paths."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    commands = CommandManager(providers=(ConfigurationCommands(configuration, Mock()),))
    completion = CompletionManager((CommandCompletionAdapter(lambda: commands.commands),))

    values = list(completion.get_completions(Document("/config get loop.d"), Mock()))

    assert [value.text for value in values] == ["loop.debug"]
    values = list(completion.get_completions(Document("/config set backend.a"), Mock()))

    assert not values


def test_config_secret_completion_suggests_only_secret_paths(tmp_path):
    """The secret action completes the entries that require masked input."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    commands = CommandManager(providers=(ConfigurationCommands(configuration, Mock()),))
    completion = CompletionManager((CommandCompletionAdapter(lambda: commands.commands),))

    values = list(completion.get_completions(Document("/config secret backend.a"), Mock()))

    assert [value.text for value in values] == ["backend.api_key"]


def test_config_completion_suggests_scope_values_after_edit_arguments(tmp_path):
    """Editable operations complete the explicit file and session scope arguments."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    commands = CommandManager(providers=(ConfigurationCommands(configuration, Mock()),))
    completion = CompletionManager((CommandCompletionAdapter(lambda: commands.commands),))

    values = list(
        completion.get_completions(Document("/config set loop.debug false scope="), Mock())
    )

    assert [value.text for value in values] == ["scope=file", "scope=session"]


def test_config_reset_completion_suggests_scope_without_a_path(tmp_path):
    """Reset-all completion includes the file and session scope arguments."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    commands = CommandManager(providers=(ConfigurationCommands(configuration, Mock()),))
    completion = CompletionManager((CommandCompletionAdapter(lambda: commands.commands),))

    values = list(completion.get_completions(Document("/config reset scope="), Mock()))

    assert [value.text for value in values] == ["scope=file", "scope=session"]


def test_config_reset_completion_suggests_scope_after_a_path(tmp_path):
    """Single-entry reset completion includes the file and session scope arguments."""
    configuration = ConfigurationManager(tmp_path)
    configuration.load()
    commands = CommandManager(providers=(ConfigurationCommands(configuration, Mock()),))
    completion = CompletionManager((CommandCompletionAdapter(lambda: commands.commands),))

    values = list(completion.get_completions(Document("/config reset loop.debug scope="), Mock()))

    assert [value.text for value in values] == ["scope=file", "scope=session"]
