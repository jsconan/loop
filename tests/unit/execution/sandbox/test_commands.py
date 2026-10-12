"""Verify sandbox-owned command intent, fresh consent and sanitized management failures."""

from unittest.mock import Mock

import pytest
from prompt_toolkit.document import Document

from loop import CommandCompletionAdapter, CompletionManager
from loop.commands import CommandArgumentError, CommandContext, CommandManager
from loop.execution.sandbox import SandboxCommands
from loop.execution.sandbox import commands as module
from loop.execution.sandbox.linux import management as linux_management
from loop.execution.sandbox.linux import setup as linux_setup


@pytest.fixture
def commands(monkeypatch):
    """Provide sandbox commands with default non-elevating library authority."""
    monkeypatch.setattr(linux_management, "installation_status", Mock(return_value="Inventory"))
    return SandboxCommands()


def test_non_linux_host_does_not_register_or_run_linux_management(commands, monkeypatch):
    """Non-Linux hosts expose no Linux management command or authority."""
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")

    assert commands.get_commands() == ()
    with pytest.raises(CommandArgumentError, match="unavailable on this host"):
        commands.sandbox(CommandContext("sandbox", Mock()))


def test_sandbox_status_delegates_to_management_owner(commands, monkeypatch):
    """App status displays the owner's safe result without reconstructing Linux authority."""
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")

    status = Mock(return_value="authenticated; unavailable")
    monkeypatch.setattr(linux_management, "installation_status", status)
    context = CommandContext("sandbox", Mock())
    commands.sandbox(context)
    status.assert_called_once_with(details=False)
    context.interaction.info.assert_called_once_with("authenticated; unavailable")
    assert commands.get_commands()[0].name == "sandbox"


def test_sandbox_status_can_request_diagnostics(commands, monkeypatch):
    """The name=value slash-command syntax exposes diagnostics as an explicit opt-in."""
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    status = Mock(return_value="Files and settings")
    monkeypatch.setattr(linux_management, "installation_status", status)
    interaction = Mock()
    manager = CommandManager(
        commands=commands.get_commands(), interaction=interaction, exit_command_names=()
    )

    manager.call("sandbox", "status details=true")

    status.assert_called_once_with(details=True)
    interaction.info.assert_called_once_with("Files and settings")


def test_sandbox_status_completion_offers_opt_in_diagnostics(commands, monkeypatch):
    """Status argument completion offers the opt-in inventory switch."""
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    manager = CommandManager(commands=commands.get_commands(), exit_command_names=())
    completion = CompletionManager((CommandCompletionAdapter(lambda: manager.commands),))

    values = completion.get_completions(Document("/sandbox status details=t"), Mock())

    assert [value.text for value in values] == ["details=true"]


def test_sandbox_details_option_is_restricted_to_status(commands, monkeypatch):
    """Diagnostic inventory is not accepted as an unrelated management argument."""
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")

    with pytest.raises(CommandArgumentError, match="only with /sandbox status"):
        commands.sandbox(CommandContext("sandbox", Mock()), "install", details=True)


@pytest.mark.parametrize("decision", [False, None, True])
def test_sandbox_uninstall_delegates_intent_and_fresh_human_review(commands, monkeypatch, decision):
    """The app owns human consent and audits while management owns construction and execution."""
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")

    interaction = Mock()
    interaction.confirm.return_value = decision
    audit = Mock()
    monkeypatch.setattr(module, "telemetry_audit", audit)

    def manage(action, review, *, main_app):
        """Exercise the app callback through the public management intent boundary."""
        assert main_app is False
        assert action == "uninstall"
        return review("exact reviewed action")

    monkeypatch.setattr(linux_setup, "manage_setup", manage)
    commands.sandbox(CommandContext("sandbox", interaction), "uninstall")
    assert interaction.info.call_args_list[0].args[0] == "exact reviewed action"
    interaction.confirm.assert_called_once_with("Uninstall the Linux sandbox?", default=False)
    audit.assert_any_call(
        "sandbox.management.decided", action="uninstall", approved=decision is True
    )
    if decision is True:
        audit.assert_any_call("sandbox.management.finished", action="uninstall", success=True)


@pytest.mark.parametrize("message", ["Changed authority", "", "\ninternal detail"])
def test_sandbox_uninstall_owner_refusal_reports_unavailable(commands, monkeypatch, message):
    """Owner refusals show one bounded reason and current inventory without another action."""
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")

    manage = Mock(side_effect=ValueError(message))
    monkeypatch.setattr(linux_setup, "manage_setup", manage)
    monkeypatch.setattr(linux_management, "installation_status", Mock(return_value="Inventory"))
    interaction = Mock()
    with pytest.raises(CommandArgumentError, match="Linux remains unavailable"):
        commands.sandbox(CommandContext("sandbox", interaction), "uninstall")
    interaction.confirm.assert_not_called()
    assert interaction.info.call_args.args[0] == (
        f"Uninstall: {message if message == 'Changed authority' else 'Native management failed.'}"
        "\nInventory"
    )
    manage.assert_called_once()


def test_sandbox_install_reports_distinct_grouped_failures(commands, monkeypatch):
    """A failed activation and admission closure show both causes and one inventory."""
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        linux_setup,
        "manage_setup",
        Mock(
            side_effect=ExceptionGroup(
                "activation",
                [
                    ValueError("activation changed"),
                    OSError("closure failed"),
                    ValueError("activation changed"),
                ],
            )
        ),
    )
    monkeypatch.setattr(linux_management, "installation_status", Mock(return_value="Inventory"))
    interaction = Mock()

    with pytest.raises(CommandArgumentError, match="Linux remains unavailable"):
        commands.sandbox(CommandContext("sandbox", interaction), "install")

    interaction.info.assert_called_once_with(
        "Install: activation changed\nInstall: closure failed\nInventory"
    )


@pytest.mark.parametrize("action", ["install"])
def test_sandbox_setup_delegates_one_intent_and_consent(monkeypatch, action):
    """The existing provider forwards setup intent and literal human consent to its domain owner."""
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")

    monkeypatch.setattr(linux_management, "installation_status", Mock(return_value="Inventory"))
    interaction = Mock()
    interaction.confirm.return_value = True

    def manage(operation, review, *, main_app):
        """Exercise the public setup boundary without constructing installation models."""
        assert operation == action
        assert main_app is True
        return review("exact setup bytes")

    monkeypatch.setattr(linux_setup, "manage_setup", manage)
    refresh_backend = Mock()
    commands = SandboxCommands(main_app=True, refresh_backend=refresh_backend)
    commands.sandbox(CommandContext("sandbox", interaction), action)
    refresh_backend.assert_called_once_with()
    assert interaction.info.call_args_list[0].args[0] == "exact setup bytes"
    interaction.confirm.assert_called_once_with("Install the Linux sandbox?", default=False)
    assert [value.value for value in commands.get_commands()[0].completion.values] == [
        "status",
        "install",
        "reinstall",
        "refresh",
        "uninstall",
    ]


@pytest.mark.parametrize("uninstall", ["completed", "incomplete", "declined"])
@pytest.mark.parametrize("action", ["reinstall", "refresh"])
def test_reinstall_runs_reviewed_steps_in_order(monkeypatch, uninstall, action):
    """Reinstall attempts install after incomplete removal and stops after declined consent."""
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(linux_management, "installation_status", Mock(return_value="Inventory"))
    interaction = Mock()
    interaction.confirm.return_value = uninstall != "declined"
    refresh_backend = Mock()
    audit = Mock()
    monkeypatch.setattr(module, "telemetry_audit", audit)
    calls = []

    def manage(action, review, *, main_app):
        """Observe ordered management intents at the package boundary."""
        assert main_app is True
        calls.append(action)
        if action == "uninstall" and uninstall == "incomplete":
            raise OSError("removal incomplete")
        return review(f"review {action}")

    monkeypatch.setattr(linux_setup, "manage_setup", manage)
    commands = SandboxCommands(main_app=True, refresh_backend=refresh_backend)
    commands.sandbox(CommandContext("sandbox", interaction), action)

    assert calls == (["uninstall"] if uninstall == "declined" else ["uninstall", "install"])
    assert "Inventory" in interaction.info.call_args.args[0]
    assert interaction.info.call_args.args[0].count("Inventory") == 1
    if uninstall == "completed":
        assert [call.args[0] for call in interaction.confirm.call_args_list] == [
            "Uninstall the Linux sandbox?",
            "Install the Linux sandbox?",
        ]
        audit.assert_any_call("sandbox.management.finished", action=action, success=True)
        assert refresh_backend.call_count == 2
    elif uninstall == "incomplete":
        interaction.confirm.assert_called_once_with("Install the Linux sandbox?", default=False)
        audit.assert_any_call("sandbox.management.finished", action=action, success=True)
        refresh_backend.assert_called_once_with()
    else:
        interaction.confirm.assert_called_once_with("Uninstall the Linux sandbox?", default=False)
        refresh_backend.assert_not_called()


@pytest.mark.parametrize("uninstall_fails", [False, True])
def test_reinstall_reports_both_steps_when_install_is_incomplete(monkeypatch, uninstall_fails):
    """A failed install reports both attempted steps and the current installation state."""
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(linux_management, "installation_status", Mock(return_value="Inventory"))
    interaction = Mock()
    interaction.confirm.return_value = True
    audit = Mock()
    monkeypatch.setattr(module, "telemetry_audit", audit)
    calls = []

    def manage(action, review, *, main_app):
        """Fail the second step after the first reviewed action completes."""
        assert main_app is True
        calls.append(action)
        if action == "install" or uninstall_fails:
            raise OSError(f"{action} incomplete")
        return review("review uninstall")

    monkeypatch.setattr(linux_setup, "manage_setup", manage)
    with pytest.raises(CommandArgumentError, match="Native reinstall incomplete"):
        SandboxCommands(main_app=True).sandbox(CommandContext("sandbox", interaction), "reinstall")

    assert calls == ["uninstall", "install"]
    assert interaction.info.call_args.args[0] == (
        f"Uninstall: {'incomplete (uninstall incomplete)' if uninstall_fails else 'completed'}"
        "\nInstall: incomplete (install incomplete)\nInventory"
    )
    audit.assert_any_call("sandbox.management.finished", action="reinstall", success=False)


def test_reinstall_reports_grouped_install_failure(monkeypatch):
    """Reinstall preserves each owner failure while rendering one final inventory."""
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(linux_management, "installation_status", Mock(return_value="Inventory"))

    def manage(action, review, *, main_app):
        """Complete uninstall and raise a grouped install failure at the owner boundary."""
        assert main_app is True
        if action == "install":
            raise ExceptionGroup(
                "install", [ValueError("policy changed"), OSError("closure failed")]
            )
        return review("review uninstall")

    monkeypatch.setattr(linux_setup, "manage_setup", manage)
    interaction = Mock()
    interaction.confirm.return_value = True

    with pytest.raises(CommandArgumentError, match="Native reinstall incomplete"):
        SandboxCommands(main_app=True).sandbox(CommandContext("sandbox", interaction), "reinstall")

    interaction.info.assert_called_with(
        "Uninstall: completed\nInstall: incomplete (policy changed)"
        "\nInstall: incomplete (closure failed)\nInventory"
    )
