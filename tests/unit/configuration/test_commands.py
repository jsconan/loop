"""Tests for interactive application-configuration commands."""

import asyncio
import json
import os
from threading import Event, get_ident
from unittest.mock import AsyncMock, Mock, call

import pytest
from prompt_toolkit.document import Document

from loop import Interaction, PermissionManager
from loop.commands import CommandArgumentError, CommandContext, CommandManager
from loop.completion import CommandCompletionAdapter, CompletionManager
from loop.configuration import ConfigurationCommands, ConfigurationManager
from loop.interaction import ConsoleInteraction
from loop.models import CommandCapability
from loop.permissions.managed import ManagedPolicyStore
from tests.scheduling import deliver_callbacks


def test_yolo_command_defaults_to_session_and_displays_activation_and_status(tmp_path):
    """The human config command explains authority before enabling and reports both lifetimes."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    interaction = Mock(spec=ConsoleInteraction)
    events = []
    interaction.warning.side_effect = lambda message: events.append(("warning", message))
    apply = Mock(side_effect=lambda path, settings: events.append(("apply", path)) or "applied now")
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply),), interaction=interaction
    )
    config = next(item for item in manager.active_commands if item.name == "config")
    assert "private data" in config.description

    assert manager.handle_user_command("/config set app.yolo true")
    assert configuration.effective.app.yolo is True
    assert configuration.source_for("app.yolo") == "session"
    assert configuration.saved_yolo is False
    assert events[0][0] == "warning"
    assert "private data" in events[0][1]
    assert events[1] == ("apply", "app.yolo")
    interaction.prompt.assert_not_called()
    assert "YOLO active: True" in interaction.info.call_args.args[0]

    assert manager.handle_user_command("/config set app.yolo true scope=user")
    assert configuration.saved_yolo is True
    assert manager.handle_user_command("/config set app.yolo false")
    assert configuration.effective.app.yolo is False
    assert "will become active on restart" in interaction.info.call_args.args[0]
    assert manager.handle_user_command("/config set app.yolo true scope=session")
    assert manager.handle_user_command("/config set app.yolo false scope=user")
    assert configuration.effective.app.yolo is True
    assert "Disable with scope=session" in interaction.info.call_args.args[0]
    assert manager.handle_user_command("/config get app.yolo")
    assert "saved user value: False" in interaction.info.call_args.args[0]
    assert manager.handle_user_command("/config set app.yolo false scope=workspace")
    assert "supports session and user" in interaction.report.call_args.args[0].detail


@pytest.mark.parametrize("scope", ["session", "user"])
def test_yolo_command_checks_managed_policy_before_changing_configuration(tmp_path, scope):
    """Administrator denial cannot change active or persisted YOLO configuration."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    before = configuration.path.read_bytes()
    interaction = Mock(spec=ConsoleInteraction)
    apply = Mock()
    validate = Mock(side_effect=ValueError("Administrator policy prohibits YOLO activation."))
    provider = ConfigurationCommands(configuration, apply, validate_yolo=validate)
    manager = CommandManager(providers=(provider,), interaction=interaction)
    assert manager.handle_user_command(f"/config set app.yolo true scope={scope}")
    assert configuration.effective.app.yolo is False
    assert configuration.saved_yolo is False
    assert configuration.path.read_bytes() == before
    assert "prohibits YOLO" in interaction.report.call_args.args[0].detail
    apply.assert_not_called()
    validate.assert_called_once()


@pytest.mark.parametrize("scope", ["session", "user"])
def test_yolo_activation_failure_does_not_publish_or_overwrite_concurrent_config(tmp_path, scope):
    """A later managed denial rolls back runtime without publishing the pending setting."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    concurrent = ConfigurationManager(configuration.path)
    concurrent.load({})
    interaction = Mock(spec=ConsoleInteraction)
    applied = []

    def apply(path, settings):
        """Change an unrelated saved setting while runtime activation is being applied."""
        applied.append(settings.app.yolo)
        if settings.app.yolo:
            concurrent.set("app.debug", True, scope="user")
        return "applied now"

    validate = Mock(side_effect=[None, ValueError("Administrator policy became stale.")])
    provider = ConfigurationCommands(configuration, apply, validate_yolo=validate)
    manager = CommandManager(providers=(provider,), interaction=interaction)
    assert manager.handle_user_command(f"/config set app.yolo true scope={scope}")
    assert applied == [True, False]
    assert "policy became stale" in interaction.report.call_args.args[0].detail
    concurrent.reload({})
    assert concurrent.saved_yolo is False
    assert concurrent.effective.app.debug is True
    assert configuration.effective.app.yolo is False


@pytest.mark.parametrize("policy_fails", [False, True])
def test_cancelled_yolo_activation_settles_its_commit_or_rollback(tmp_path, policy_fails):
    """Cancellation waits for publication or rollback when managed validation changes."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    started = asyncio.Event()
    release = asyncio.Event()
    applied = []

    async def apply_async(_path, settings):
        """Block only the first activation so cancellation can arrive during application."""
        applied.append(settings.app.yolo)
        if settings.app.yolo:
            started.set()
            await release.wait()
        return "applied now"

    validate = Mock(side_effect=[None, ValueError("policy changed")] if policy_fails else None)
    provider = ConfigurationCommands(
        configuration, Mock(), apply_async=apply_async, validate_yolo=validate
    )
    manager = CommandManager(providers=(provider,), interaction=Mock(spec=ConsoleInteraction))

    async def dispatch():
        """Cancel after runtime work begins and allow the transaction to settle."""
        pending = asyncio.create_task(manager.call_async("config", "set app.yolo true scope=user"))
        await started.wait()
        pending.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await pending
        if policy_fails:
            assert isinstance(captured.value.__cause__, ValueError)

    asyncio.run(dispatch())
    assert configuration.saved_yolo is (not policy_fails)
    assert applied == ([True, False] if policy_fails else [True])
    manager.close()


def test_independently_cancelled_yolo_application_rolls_back(tmp_path):
    """A cancelled runtime callback cannot leave a committed or active activation."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    applied = []

    async def apply_async(_path, settings):
        """Cancel the activation independently and accept the explicit rollback."""
        applied.append(settings.app.yolo)
        if settings.app.yolo:
            raise asyncio.CancelledError("runtime cancelled")
        return "applied now"

    provider = ConfigurationCommands(
        configuration, Mock(), apply_async=apply_async, validate_yolo=Mock()
    )
    manager = CommandManager(providers=(provider,), interaction=Mock(spec=ConsoleInteraction))

    async def dispatch():
        """Observe the callback cancellation after the compensating runtime action."""
        with pytest.raises(asyncio.CancelledError, match="runtime cancelled"):
            await manager.call_async("config", "set app.yolo true scope=user")

    asyncio.run(dispatch())
    assert applied == [True, False]
    assert configuration.saved_yolo is False
    assert configuration.effective.app.yolo is False
    manager.close()


@pytest.mark.parametrize("fail_save", [False, True])
def test_yolo_publication_gates_automatic_decisions_and_restores_failed_save(
    tmp_path, monkeypatch, fail_save
):
    """An uncommitted runtime activation cannot approve work and a failed save restores state."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    before = configuration.path.read_bytes()
    permissions = PermissionManager(tmp_path)
    started, release = asyncio.Event(), asyncio.Event()

    async def apply_async(_path, settings):
        """Apply the requested mode, then hold activation before publishing it."""
        permissions.set_yolo(settings.app.yolo)
        if settings.app.yolo:
            started.set()
            await release.wait()
        return "applied now"

    if fail_save:
        monkeypatch.setattr(configuration, "_save_scope", Mock(side_effect=OSError("save failed")))
    provider = ConfigurationCommands(
        configuration,
        Mock(),
        apply_async=apply_async,
        validate_yolo=permissions.validate_yolo_activation,
        yolo_active=lambda: permissions.yolo,
        yolo_gate=permissions.suspend_yolo_decisions,
    )
    manager = CommandManager(providers=(provider,), interaction=Mock(spec=ConsoleInteraction))

    async def dispatch():
        """Check the live permission boundary while the activation is not yet committed."""
        pending = asyncio.create_task(manager.call_async("config", "set app.yolo true scope=user"))
        await started.wait()
        assert permissions.yolo is False
        assert permissions.confirmation_decision() is None
        release.set()
        if fail_save:
            with pytest.raises(OSError, match="save failed"):
                await pending
        else:
            await pending

    asyncio.run(dispatch())
    assert permissions.yolo is not fail_save
    assert configuration.effective.app.yolo is not fail_save
    assert configuration.saved_yolo is not fail_save
    if fail_save:
        assert configuration.path.read_bytes() == before
    manager.close()
    permissions.close()


@pytest.mark.parametrize("scope", ["session", "user"])
def test_reapplying_yolo_preserves_decisions_during_publication(tmp_path, scope):
    """An already committed activation remains available while its update is pending."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    configuration.set_session("app.yolo", True)
    permissions = PermissionManager(tmp_path, yolo=True)
    started, release = asyncio.Event(), asyncio.Event()

    async def apply_async(_path, settings):
        """Hold publication after the runtime sees the already-active setting."""
        permissions.set_yolo(settings.app.yolo)
        started.set()
        await release.wait()
        return "applied now"

    provider = ConfigurationCommands(
        configuration,
        Mock(),
        apply_async=apply_async,
        validate_yolo=permissions.validate_yolo_activation,
        yolo_active=lambda: permissions.yolo,
        yolo_gate=permissions.suspend_yolo_decisions,
    )
    manager = CommandManager(providers=(provider,), interaction=Mock(spec=ConsoleInteraction))

    async def dispatch():
        """Authorize while the config command has not yet completed."""
        pending = asyncio.create_task(
            manager.call_async("config", f"set app.yolo true scope={scope}")
        )
        await started.wait()
        assert permissions.yolo is True
        assert permissions.confirmation_decision() is True
        release.set()
        await pending

    asyncio.run(dispatch())
    assert permissions.yolo is True
    assert configuration.effective.app.yolo is True
    manager.close()
    permissions.close()


@pytest.mark.parametrize("revoke", [False, True])
def test_failed_yolo_update_does_not_restore_revoked_authority(tmp_path, monkeypatch, revoke):
    """A failed update restores active authority only if managed policy never revoked it."""
    directory = tmp_path / "system" / "loop"
    directory.mkdir(parents=True)
    directory.parent.chmod(0o755)
    directory.chmod(0o755)
    identity = "a" * 32
    expectation = directory.parent / "loop.managed.expect.json"
    expectation.write_text(json.dumps({"version": 1, "identity": identity, "minimum_revision": 1}))
    policy = directory / "managed-policy.json"
    document = {"version": 1, "identity": identity, "revision": 1, "forbid_yolo": False}
    policy.write_text(json.dumps(document))
    expectation.chmod(0o644)
    policy.chmod(0o644)
    store = ManagedPolicyStore(directory, trust_anchor=tmp_path, owner_uid=os.getuid())
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    configuration.set_session("app.yolo", True)
    permissions = PermissionManager(tmp_path, managed_policy_store=store, yolo=True)
    started, release = asyncio.Event(), asyncio.Event()
    applied = []

    async def apply_async(_path, settings):
        """Hold the runtime update before the user configuration save."""
        applied.append(settings.app.yolo)
        permissions.set_yolo(settings.app.yolo)
        if len(applied) == 1:
            started.set()
            await release.wait()
        return "applied now"

    monkeypatch.setattr(configuration, "_save_scope", Mock(side_effect=OSError("save failed")))
    provider = ConfigurationCommands(
        configuration,
        Mock(),
        apply_async=apply_async,
        validate_yolo=permissions.validate_yolo_activation,
        yolo_active=lambda: permissions.yolo,
        yolo_gate=permissions.suspend_yolo_decisions,
    )
    manager = CommandManager(providers=(provider,), interaction=Mock(spec=ConsoleInteraction))

    async def dispatch():
        """Change managed authority while the original publication is pending."""
        pending = asyncio.create_task(manager.call_async("config", "set app.yolo true scope=user"))
        await started.wait()
        if revoke:
            document.update(revision=2, forbid_yolo=True)
            policy.write_text(json.dumps(document))
            permissions.reload()
            assert permissions.yolo is False
            document.update(revision=3, forbid_yolo=False)
            policy.write_text(json.dumps(document))
            permissions.reload()
            assert permissions.yolo is False
        release.set()
        with pytest.raises(OSError, match="save failed"):
            await pending

    asyncio.run(dispatch())
    assert permissions.yolo is not revoke
    assert permissions.confirmation_decision() is (None if revoke else True)
    assert configuration.saved_yolo is False
    manager.close()
    permissions.close()


def test_yolo_rollback_without_permission_gate_preserves_original_error(tmp_path):
    """An independent configuration adapter retains both errors when no gate is supplied."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    apply = Mock(side_effect=[ValueError("activation failed"), OSError("rollback failed")])
    provider = ConfigurationCommands(configuration, apply, validate_yolo=Mock())
    context = CommandContext("config", Mock(spec=ConsoleInteraction))
    with pytest.raises(CommandArgumentError, match="activation failed") as captured:
        asyncio.run(provider.config_async(context, "set", "app.yolo", "true", "session"))
    assert any("rollback failed" in note for note in captured.value.__cause__.__notes__)
    assert configuration.effective.app.yolo is False
    assert configuration.saved_yolo is False


@pytest.mark.parametrize("initially_active", [False, True])
def test_yolo_rollback_failure_disables_automatic_approval(tmp_path, monkeypatch, initially_active):
    """Unconfirmed egress revocation cannot expose YOLO after a failed configuration write."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    permissions = PermissionManager(tmp_path, yolo=initially_active)
    lease = Mock()
    lease.pause.return_value = False
    lease.revoke_revision.return_value = False
    applied = []

    async def apply_async(_path, settings):
        """Bind an unresponsive mediator after pre-application and before the failed save."""
        applied.append(settings.app.yolo)
        permissions.set_yolo(settings.app.yolo)
        if len(applied) == 1:
            permissions.bind_egress(permissions.egress_policy_revision(), lease)
        return "applied now"

    monkeypatch.setattr(configuration, "_save_scope", Mock(side_effect=OSError("save failed")))
    provider = ConfigurationCommands(
        configuration,
        Mock(),
        apply_async=apply_async,
        validate_yolo=permissions.validate_yolo_activation,
        yolo_active=lambda: permissions.yolo,
        yolo_gate=permissions.suspend_yolo_decisions,
    )
    manager = CommandManager(providers=(provider,), interaction=Mock(spec=ConsoleInteraction))

    async def dispatch():
        """Keep the original save failure and the unconfirmed rollback diagnostic."""
        with pytest.raises(OSError, match="save failed") as captured:
            await manager.call_async("config", "set app.yolo true scope=user")
        assert any(
            "egress admission pause is uncertain" in note for note in captured.value.__notes__
        )

    asyncio.run(dispatch())
    assert permissions.yolo is False
    assert permissions.confirmation_decision() is None
    assert configuration.saved_yolo is False
    lease.revoke_revision.assert_called()
    with pytest.raises(OSError, match="egress closure could not be confirmed"):
        permissions.close()
    manager.close()


@pytest.mark.parametrize("cancel_first", [False, True])
def test_concurrent_yolo_commands_apply_in_order(tmp_path, cancel_first):
    """A later session disable takes effect after the pending user activation settles."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    started, release, second_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    applied = []

    async def apply_async(_path, settings):
        """Hold the first activation while another command enters dispatch."""
        applied.append(settings.app.yolo)
        if settings.app.yolo:
            started.set()
            await release.wait()
        return "applied now"

    provider = ConfigurationCommands(
        configuration, Mock(), apply_async=apply_async, validate_yolo=Mock()
    )
    manager = CommandManager(providers=(provider,), interaction=Mock(spec=ConsoleInteraction))

    async def disable():
        """Signal entry before contending for the mutation transaction."""
        second_started.set()
        await manager.call_async("config", "set app.yolo false scope=session")

    async def dispatch():
        """Interleave an activation and later disable, optionally cancelling the first."""
        first = asyncio.create_task(manager.call_async("config", "set app.yolo true scope=user"))
        await started.wait()
        second = asyncio.create_task(disable())
        await second_started.wait()
        assert not second.done()
        if cancel_first:
            first.cancel()
        release.set()
        if cancel_first:
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            await first
        await second

    asyncio.run(dispatch())
    assert applied == [True, False]
    assert configuration.saved_yolo is True
    assert configuration.effective.app.yolo is False
    manager.close()


def test_cancelled_yolo_waiter_does_not_retain_the_mutation_lock(tmp_path):
    """Cancelling a competing command cannot strand or steal another transaction's lock."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    started, release, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def apply_async(_path, settings):
        """Hold the first activation while the competing command is cancelled."""
        if settings.app.yolo:
            started.set()
            await release.wait()
        return "applied now"

    provider = ConfigurationCommands(
        configuration, Mock(), apply_async=apply_async, validate_yolo=Mock()
    )
    manager = CommandManager(providers=(provider,), interaction=Mock(spec=ConsoleInteraction))

    async def competing():
        """Signal entry before waiting for the shared mutation transaction."""
        waiting.set()
        await manager.call_async("config", "set app.yolo false scope=session")

    async def dispatch():
        """Cancel the waiter, finish the owner, and then start another mutation."""
        first = asyncio.create_task(manager.call_async("config", "set app.yolo true scope=user"))
        await started.wait()
        second = asyncio.create_task(competing())
        await waiting.wait()
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        release.set()
        await first
        await manager.call_async("config", "set app.yolo false scope=session")

    asyncio.run(dispatch())
    assert configuration.saved_yolo is True
    assert configuration.effective.app.yolo is False
    manager.close()


def test_user_yolo_preapply_preserves_masking_session_override(tmp_path):
    """Saving a user activation cannot override an active false session setting."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    configuration.set_session("app.yolo", False)
    apply = Mock(return_value="applied now")
    provider = ConfigurationCommands(configuration, apply, validate_yolo=Mock())
    manager = CommandManager(providers=(provider,), interaction=Mock(spec=ConsoleInteraction))
    assert manager.handle_user_command("/config set app.yolo true scope=user")
    assert configuration.saved_yolo is True
    assert configuration.effective.app.yolo is False
    assert apply.call_args.args[1].app.yolo is False


def test_yolo_reload_cannot_activate_a_forbidden_saved_value(tmp_path):
    """An externally saved activation remains ineffective when managed policy rejects it."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    writer = ConfigurationManager(configuration.path)
    writer.load({})
    writer.set("app.yolo", True, scope="user")
    interaction = Mock(spec=ConsoleInteraction)
    apply = Mock(side_effect=ValueError("Administrator policy prohibits YOLO activation."))
    provider = ConfigurationCommands(configuration, apply, yolo_active=lambda: False)
    manager = CommandManager(providers=(provider,), interaction=interaction)
    assert manager.handle_user_command("/config reload")
    assert "prohibits YOLO" in interaction.report.call_args.args[0].detail
    assert manager.handle_user_command("/config get app.yolo")
    assert "YOLO active: False; source: inactive" in interaction.info.call_args.args[0]


@pytest.mark.parametrize("async_dispatch", [False, True])
def test_yolo_reload_warns_before_enabling_without_an_extra_confirmation(tmp_path, async_dispatch):
    """A newly effective trusted value displays the disclaimer before runtime activation."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    writer = ConfigurationManager(configuration.path)
    writer.load({})
    writer.set("app.yolo", True, scope="user")
    interaction = Mock(spec=ConsoleInteraction)
    events = []
    interaction.warning.side_effect = lambda message: events.append(("warning", message))
    apply = Mock(side_effect=lambda path, settings: events.append(("apply", path)) or "applied now")
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply),), interaction=interaction
    )
    if async_dispatch:
        asyncio.run(manager.call_async("config", "reload"))
    else:
        manager.call("config", "reload")
    assert events[0][0] == "warning"
    assert "private data" in events[0][1]
    assert "directly on the host" in events[0][1]
    assert events[1] == ("apply", "app.yolo")
    interaction.prompt.assert_not_called()
    interaction.confirm.assert_not_called()
    manager.close()


@pytest.mark.parametrize("transition", ["unchanged", "disable", "masked"])
def test_yolo_reload_does_not_warn_without_activation(tmp_path, transition):
    """Unchanged, disabled, and session-masked settings do not produce activation warnings."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    if transition == "disable":
        configuration.set("app.yolo", True, scope="user")
    configuration.load({})
    if transition == "masked":
        configuration.set_session("app.yolo", False)
    writer = ConfigurationManager(configuration.path)
    writer.load({})
    if transition != "unchanged":
        writer.set("app.yolo", transition == "masked", scope="user")
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock(return_value="applied now")),),
        interaction=interaction,
    )
    manager.call("config", "reload")
    assert configuration.effective.app.yolo is False
    interaction.warning.assert_not_called()
    manager.close()


def test_yolo_status_reports_managed_suppression(tmp_path):
    """Display effective mode when a loaded setting is administratively blocked."""
    configuration = ConfigurationManager(tmp_path / "config" / "config.toml")
    configuration.initialize()
    configuration.load({})
    configuration.set_session("app.yolo", True)
    interaction = Mock(spec=ConsoleInteraction)
    provider = ConfigurationCommands(configuration, Mock(), yolo_active=lambda: False)
    manager = CommandManager(providers=(provider,), interaction=interaction)
    assert manager.handle_user_command("/config get app.yolo")
    assert "YOLO active: False; source: inactive" in interaction.info.call_args.args[0]


def test_yolo_deactivation_reports_active_run_authority(tmp_path):
    """Disabling no-prompt mode during a run explains that admitted jobs may continue."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.load({})
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock(return_value="applied now")),),
        interaction=interaction,
    )
    manager.set_capability_provider(
        lambda: frozenset(
            {
                CommandCapability.CONTEXT_READ,
                CommandCapability.CONTEXT_UPDATE,
                CommandCapability.RUN_ACTIVE,
            }
        )
    )
    assert manager.handle_user_command("/config set app.yolo true scope=session")
    assert manager.handle_user_command("/config set app.yolo false scope=session")
    assert any("running jobs" in call.args[0] for call in interaction.warning.call_args_list)


def test_safe_config_changes_apply_during_runs_and_hide_unsafe_values(tmp_path):
    """Run-safe settings apply immediately while backend changes vanish from completion."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.load()
    provider = ConfigurationCommands(configuration, Mock(return_value="applied now"))
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(providers=(provider,), interaction=interaction)
    manager.set_capability_provider(
        lambda: frozenset(
            {
                CommandCapability.CONTEXT_READ,
                CommandCapability.CONTEXT_UPDATE,
                CommandCapability.CONVERSATION_ACTIVE,
                CommandCapability.RUN_ACTIVE,
            }
        )
    )

    config_command = next(item for item in manager.active_commands if item.name == "config")

    assert config_command.name == "config"
    assert manager.handle_user_command("/config set app.max_agent_turns 50 scope=session")
    assert configuration.effective.app.max_agent_turns == 50
    provider._apply.assert_called_once_with("app.max_agent_turns", configuration.effective)

    completion = CompletionManager(
        (
            CommandCompletionAdapter(
                lambda: manager.active_commands,
                providers=(provider,),
                capabilities=lambda: manager.capabilities,
            ),
        )
    )
    action_values = list(completion.get_completions(Document("/config "), None))
    setting_values = list(completion.get_completions(Document("/config set app."), None))
    assert {item.text for item in action_values} == {"get", "set"}
    assert "app.max_agent_turns" in {item.text for item in setting_values}
    assert not any(item.text.startswith("backend.") for item in setting_values)
    assert "tools.command_timeout" in {
        item.text for item in completion.get_completions(Document("/config set tools."), Mock())
    }

    manager.handle_user_command("/config set backend.base_url https://new.example scope=session")
    assert "cannot be changed while the agent is working" in (
        interaction.report.call_args.args[0].detail
    )


def test_config_run_restrictions_preserve_scope_prompt_and_reject_unsafe_operations(tmp_path):
    """Scope selection remains interactive while runtime replacement operations stay blocked."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),),
        interaction=interaction,
    )
    manager.set_capability_provider(
        lambda: frozenset(
            {
                CommandCapability.CONTEXT_READ,
                CommandCapability.CONTEXT_UPDATE,
                CommandCapability.RUN_ACTIVE,
            }
        )
    )

    interaction.prompt.return_value = "session"
    assert manager.handle_user_command("/config set app.debug true")
    interaction.prompt.assert_called_once_with(
        "Choose configuration scope:",
        choices={
            "user": "Save for current user",
            "workspace": "Save for current workspace",
            "session": "This session only",
            "cancel": "Cancel",
        },
        index={"user": "u", "workspace": "w", "session": "s", "cancel": "c"},
    )
    assert configuration.effective.app.debug is True
    assert manager.handle_user_command("/config set app.model test scope=session")
    assert "cannot be changed while the agent is working" in (
        interaction.report.call_args.args[0].detail
    )
    assert manager.handle_user_command("/config set app.unknown value scope=session")
    assert "Unknown configuration field" in interaction.report.call_args.args[0].detail
    assert manager.handle_user_command("/config reset app.debug scope=session")
    assert "cannot run while the agent is working" in (interaction.report.call_args.args[0].detail)
    assert manager.handle_user_command("/config reload")
    assert "cannot run while the agent is working" in (interaction.report.call_args.args[0].detail)
    assert manager.handle_user_command("/config secret backend.api_key")
    assert "cannot run while the agent is working" in (interaction.report.call_args.args[0].detail)
    assert manager.handle_user_command("/config set scope=session")
    assert "requires a setting path" in interaction.report.call_args.args[0].detail


def test_sync_command_manager_keeps_sync_configuration_prompts(tmp_path):
    """The synchronous hub invokes configuration commands with synchronous prompts."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.load()
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "session"
    interaction.confirm.return_value = True
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),),
        interaction=interaction,
    )

    manager.call("config", "set app.debug true")
    manager.call("config", "reset app.debug scope=session")

    assert configuration.effective.app.debug is False
    interaction.prompt.assert_called_once()
    interaction.confirm.assert_called_once_with("Reset app.debug for session?", default=False)
    manager.close()


def test_async_config_command_offloads_persistence_and_applies_on_owner(tmp_path, monkeypatch):
    """Configuration file work runs in a worker and runtime changes return to the owner."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    started, release = Event(), Event()
    set_threads = []
    apply_threads = []
    original_set = configuration.set

    def wait_then_set(path, value, *, scope="user"):
        """Pause persistence until the async owner gets a scheduling turn."""
        set_threads.append(get_ident())
        started.set()
        assert release.wait(timeout=5)
        return original_set(path, value, scope=scope)

    def apply(path, settings):
        """Record the thread that applies the committed configuration snapshot."""
        apply_threads.append(get_ident())
        return "applied now"

    monkeypatch.setattr(configuration, "set", wait_then_set)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply),),
        interaction=interaction,
    )

    async def dispatch():
        """Release synchronous persistence from a scheduled owner-loop callback."""
        owner_thread = get_ident()
        pending = asyncio.create_task(manager.call_async("config", "set app.debug true scope=user"))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            assert not pending.done()
            release.set()
            await pending
        finally:
            release.set()
        assert apply_threads == [owner_thread]

    asyncio.run(dispatch())

    assert set_threads[0] != apply_threads[0]
    assert configuration.effective.app.debug is True


def test_config_dispatch_uses_the_callback_for_its_execution_mode(tmp_path):
    """Sync dispatch stays synchronous while async dispatch awaits its native callback."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    apply = Mock(return_value="applied now")
    apply_async = AsyncMock(return_value="applied now")
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply, apply_async=apply_async),),
        interaction=interaction,
    )

    manager.call("config", "set app.debug true scope=session")

    apply.assert_called_once_with("app.debug", configuration.effective)
    apply_async.assert_not_awaited()

    async def dispatch():
        """Run a configuration update through the asynchronous command API."""
        await manager.call_async("config", "set app.debug false scope=session")

    asyncio.run(dispatch())

    apply_async.assert_awaited_once_with("app.debug", configuration.effective)
    apply.assert_called_once()
    assert configuration.effective.app.debug is False
    manager.close()


def test_async_config_reload_uses_the_async_batch_applicator(tmp_path):
    """Async reload delegates all changed paths to its async batch callback."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.initialize()
    configuration.load()
    configuration.path.write_text("[app]\ndebug = true\n", encoding="utf-8")
    interaction = Mock(spec=ConsoleInteraction)
    apply_many = Mock()
    apply_many_async = AsyncMock(return_value={"app.debug": "applied now"})
    manager = CommandManager(
        providers=(
            ConfigurationCommands(
                configuration,
                Mock(),
                apply_many,
                apply_many_async=apply_many_async,
            ),
        ),
        interaction=interaction,
    )

    async def dispatch():
        """Reload durable settings through asynchronous command dispatch."""
        await manager.call_async("config", "reload")

    asyncio.run(dispatch())

    apply_many.assert_not_called()
    apply_many_async.assert_awaited_once_with(("app.debug",), configuration.effective)
    interaction.info.assert_called_once_with("Configuration reloaded: applied now: app.debug.")
    manager.close()


def test_async_config_cancellation_settles_async_runtime_effect(tmp_path):
    """Cancellation waits for an async runtime effect after changing session settings."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    started = asyncio.Event()
    release = asyncio.Event()
    applied = []

    async def apply_async(path, settings):
        """Hold runtime application until the test cancels the configuration command."""
        started.set()
        await release.wait()
        applied.append((path, settings.app.debug))
        return "applied now"

    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock(), apply_async=apply_async),),
        interaction=interaction,
    )

    async def dispatch():
        """Cancel a session update while its async runtime callback is pending."""
        pending = asyncio.create_task(
            manager.call_async("config", "set app.debug true scope=session")
        )
        await started.wait()
        pending.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(dispatch())

    assert applied == [("app.debug", True)]
    assert configuration.effective.app.debug is True
    manager.close()


def test_async_config_cancellation_preserves_failure_from_runtime_effect(tmp_path):
    """A runtime failure is attached when cancellation arrives during async application."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    started = asyncio.Event()
    release = asyncio.Event()

    async def fail_after_release(_path, _settings):
        """Fail runtime application only after the test cancels the command."""
        started.set()
        await release.wait()
        raise ValueError("runtime update failed")

    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock(), apply_async=fail_after_release),),
        interaction=interaction,
    )

    async def dispatch():
        """Cancel the command before releasing its failing runtime callback."""
        pending = asyncio.create_task(
            manager.call_async("config", "set app.debug true scope=session")
        )
        await started.wait()
        pending.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await pending
        assert any("ValueError" in note for note in captured.value.__notes__)

    asyncio.run(dispatch())
    manager.close()


def test_async_config_propagates_runtime_effect_failure_without_cancellation(tmp_path):
    """An uncancelled runtime callback failure reaches the command caller."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)

    async def fail(_path, _settings):
        """Raise the runtime error produced while applying a valid setting."""
        raise RuntimeError("runtime update failed")

    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock(), apply_async=fail),),
        interaction=interaction,
    )

    async def dispatch():
        """Dispatch the setting and observe its uncancelled runtime failure."""
        with pytest.raises(RuntimeError, match="runtime update failed"):
            await manager.call_async("config", "set app.debug true scope=session")

    asyncio.run(dispatch())
    manager.close()


def test_async_config_cancellation_settles_persistence_and_owner_effect(tmp_path, monkeypatch):
    """Cancellation waits for an in-flight setting write and applies its committed value."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    started = Event()
    release = Event()
    apply_threads = []
    original_set = configuration.set

    def wait_then_set(path, value, *, scope="user"):
        """Hold the file transaction until the test cancels its async command."""
        started.set()
        assert release.wait(timeout=5)
        return original_set(path, value, scope=scope)

    def apply(_path, _settings):
        """Record that committed runtime state is applied on the async owner."""
        apply_threads.append(get_ident())
        return "applied now"

    monkeypatch.setattr(configuration, "set", wait_then_set)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply),),
        interaction=interaction,
    )

    async def dispatch():
        """Cancel twice while persistence runs, then release it to settle."""
        owner_thread = get_ident()
        pending = asyncio.create_task(manager.call_async("config", "set app.debug true scope=user"))
        assert await asyncio.to_thread(started.wait, 5)
        pending.cancel()
        await deliver_callbacks()
        pending.cancel()
        await deliver_callbacks()
        assert not pending.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert apply_threads == [owner_thread]

    asyncio.run(dispatch())

    assert configuration.effective.app.debug is True


def test_async_config_cancellation_preserves_cancel_when_persistence_fails(tmp_path, monkeypatch):
    """Cancellation remains primary when a settled configuration write also fails."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    started = Event()
    release = Event()

    def fail_after_release(path, value, *, scope="user"):
        """Fail the worker operation only after the command has been cancelled."""
        started.set()
        assert release.wait(timeout=1)
        raise OSError("configuration write failed")

    monkeypatch.setattr(configuration, "set", fail_after_release)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),),
        interaction=interaction,
    )

    async def dispatch():
        """Cancel during persistence, then check the worker failure note."""
        pending = asyncio.create_task(manager.call_async("config", "set app.debug true scope=user"))
        assert await asyncio.to_thread(started.wait, 5)
        pending.cancel()
        await deliver_callbacks()
        pending.cancel()
        await deliver_callbacks()
        assert not pending.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await pending
        assert any("OSError" in note for note in captured.value.__notes__)

    asyncio.run(dispatch())


def test_async_reset_all_offloads_persistence_and_applies_on_owner(tmp_path, monkeypatch):
    """Persisted reset-all work runs off-loop and runtime effects return to its owner."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.initialize()
    configuration.load()
    configuration.set("app.debug", True, scope="user")
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm_async.return_value = True
    reset_threads = []
    apply_threads = []
    original_reset_all = configuration.reset_all

    def record_reset_all(*, scope):
        """Record where persistent reset work executes."""
        reset_threads.append(get_ident())
        return original_reset_all(scope=scope)

    def apply(_path, _settings):
        """Record where live settings are updated."""
        apply_threads.append(get_ident())
        return "applied now"

    monkeypatch.setattr(configuration, "reset_all", record_reset_all)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply),),
        interaction=interaction,
    )

    async def dispatch():
        """Dispatch a persisted reset and capture its owner thread."""
        owner_thread = get_ident()
        await manager.call_async("config", "reset scope=user")
        assert reset_threads[0] != owner_thread
        assert apply_threads
        assert set(apply_threads) == {owner_thread}

    asyncio.run(dispatch())

    assert configuration.effective.app.debug is False


def test_config_reads_remain_available_during_agent_runs(tmp_path):
    """Read-only configuration inspection is still dispatched during an active run."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),),
        interaction=interaction,
    )
    manager.set_capability_provider(
        lambda: frozenset(
            {
                CommandCapability.CONTEXT_READ,
                CommandCapability.CONTEXT_UPDATE,
                CommandCapability.RUN_ACTIVE,
            }
        )
    )

    assert manager.handle_user_command("/config get app.debug")
    assert [item.path for item in interaction.table.call_args.args[0]] == ["app.debug"]


def test_config_set_applies_a_session_override(tmp_path):
    """The direct set form changes the active snapshot and calls its runtime applicator."""
    configuration = ConfigurationManager(
        tmp_path / "config.toml", tmp_path / ".loop" / "config.toml"
    )
    configuration.load()
    apply = Mock(return_value="applied now")
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply),), interaction=interaction
    )

    manager.call("config", "set app.debug true scope=session")

    assert configuration.effective.app.debug is True
    assert configuration.source_for("app.debug") == "session"
    apply.assert_called_once_with("app.debug", configuration.effective)
    interaction.info.assert_called_once_with("Updated app.debug for session: applied now")


def test_config_get_rejects_secret_entries(tmp_path):
    """The read-only get form directs secret edits to the masked secret operation."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
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
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),), interaction=interaction
    )

    manager.call("config", "get app.debug")

    entries = interaction.table.call_args.args[0]
    assert [entry.path for entry in entries] == ["app.debug"]


def test_config_without_arguments_displays_only_non_secret_entries(tmp_path):
    """The read-only config command omits secret configuration entries."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),), interaction=interaction
    )

    manager.call("config", "")

    entries = interaction.table.call_args.args[0]
    assert len(entries) == len(configuration.entries) - 1
    assert all(not entry.secret for entry in entries)


def test_config_set_applies_a_file_override(tmp_path):
    """The direct set form persists the selected scope without a separate command."""
    configuration = ConfigurationManager(
        tmp_path / "config.toml", tmp_path / ".loop" / "config.toml"
    )
    configuration.load()
    apply = Mock(return_value="applied now")
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply),), interaction=interaction
    )

    manager.call("config", "set app.stream false scope=workspace")

    assert configuration.effective.app.stream is False
    assert (tmp_path / ".loop" / "config.toml").exists()
    assert apply.called


def test_config_reload_applies_effective_user_and_workspace_changes(tmp_path):
    """Reload applies only changed effective paths after rereading both durable scopes."""
    user_path = tmp_path / "config.toml"
    workspace_path = tmp_path / ".loop" / "config.toml"
    configuration = ConfigurationManager(user_path, workspace_path)
    configuration.initialize()
    configuration.load()
    user_path.write_text("[app]\nstream = false\n", encoding="utf-8")
    workspace_path.parent.mkdir()
    workspace_path.write_text("[app]\ndebug = true\n", encoding="utf-8")
    apply = Mock(return_value="applied now")
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply),), interaction=interaction
    )

    manager.call("config", "reload")

    assert configuration.effective.app.debug is True
    assert configuration.effective.app.stream is False
    assert apply.call_args_list == [
        call("app.stream", configuration.effective),
        call("app.debug", configuration.effective),
    ]
    interaction.info.assert_called_once_with(
        "Configuration reloaded: applied now: app.stream, app.debug."
    )


def test_config_reload_delegates_multiple_changes_to_the_batch_applicator(tmp_path):
    """Reload uses the runtime batch applicator when one is supplied."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.initialize()
    configuration.load()
    configuration.path.write_text("[app]\ndebug = true\n", encoding="utf-8")
    apply = Mock()
    apply_many = Mock(return_value={"app.debug": "applied now"})
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply, apply_many),),
        interaction=interaction,
    )

    manager.call("config", "reload")

    apply.assert_not_called()
    apply_many.assert_called_once_with(("app.debug",), configuration.effective)


def test_config_reload_reports_no_effective_change(tmp_path):
    """Reload reports success without applying settings when disk inputs resolve identically."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.initialize()
    configuration.load()
    apply = Mock()
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply),), interaction=interaction
    )

    manager.call("config", "reload")

    apply.assert_not_called()
    interaction.info.assert_called_once_with("Configuration reloaded: no effective changes.")


def test_config_reload_rejects_arguments_and_invalid_disk_content(tmp_path):
    """Reload rejects arguments and reports an invalid external file without applying it."""
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.initialize()
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),), interaction=interaction
    )

    manager.call("config", "reload app.debug")
    configuration.path.write_text('[app]\ndebug = "invalid"\n', encoding="utf-8")
    manager.call("config", "reload")

    assert interaction.report.call_count == 2
    assert "does not accept arguments" in interaction.report.call_args_list[0].args[0].detail
    assert "app.debug" in interaction.report.call_args_list[1].args[0].detail
    assert configuration.effective.app.debug is False


def test_config_secret_prompts_before_selecting_scope_and_applies_value(tmp_path):
    """The secret form reads masked input before selecting its destination scope."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    apply = Mock(return_value="backend replaced")
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt.side_effect = ["new-api-key", "user"]
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
                "user": "Save for current user",
                "workspace": "Save for current workspace",
                "session": "This session only",
                "cancel": "Cancel",
            },
            index={"user": "u", "workspace": "w", "session": "s", "cancel": "c"},
        ),
    ]
    apply.assert_called_once_with("backend.api_key", configuration.effective)
    interaction.info.assert_called_once_with("Updated backend.api_key for user: backend replaced")


def test_config_set_rejects_secret_entries_and_secret_rejects_plain_entries(tmp_path):
    """Normal set and secret actions accept only their matching entry types."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),), interaction=interaction
    )

    manager.call("config", "set backend.api_key exposed scope=session")
    manager.call("config", "secret app.debug scope=session")
    manager.call("config", "secret backend.api_key exposed scope=session")
    manager.call("config", "set unknown.path ignored scope=session")
    manager.call("config", "secret")

    assert configuration.effective.backend.api_key.get_secret_value() == "local-api-key"
    assert interaction.report.call_count == 5
    interaction.prompt_async.assert_not_called()


def test_config_secret_cancellation_leaves_the_existing_value_unchanged(tmp_path):
    """Cancelling masked input or the later scope selection changes no configuration."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt.side_effect = [False, "new-api-key", "cancel"]
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),), interaction=interaction
    )

    manager.call("config", "secret backend.api_key")
    manager.call("config", "secret backend.api_key")

    assert configuration.effective.backend.api_key.get_secret_value() == "local-api-key"
    assert interaction.info.call_args_list[0] == call("Secret backend.api_key was not changed.")


def test_config_reset_and_invalid_direct_forms_report_argument_errors(tmp_path):
    """Repeated resets store defaults and invalid forms use normal command diagnostics."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt.return_value = "cancel"
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock(return_value="applied now")),),
        interaction=interaction,
    )
    manager.call("config", "set app.debug true scope=session")

    interaction.confirm.side_effect = [True, True, True, True]
    manager.call("config", "reset app.debug scope=session")
    manager.call("config", "reset unknown.path scope=session")
    manager.call("config", "reset app.debug scope=session")
    manager.call("config", "reset app.debug scope=session")
    manager.call("config", "set app.debug")
    manager.call("config", "set")
    manager.call("config", "get unknown.path")
    manager.call("config", "get")

    assert configuration.effective.app.debug is False
    assert interaction.report.call_count == 5


@pytest.mark.parametrize("response", [False, object()])
@pytest.mark.parametrize("path", ["app.debug", ""])
def test_config_reset_with_explicit_scope_requires_confirmation(tmp_path, response, path):
    """Only explicit approval permits resetting a setting or an entire scope."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    configuration.set_session("app.debug", True)
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm.return_value = response
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock(return_value="applied now")),),
        interaction=interaction,
    )

    manager.call("config", f"reset {path} scope=session")

    assert configuration.effective.app.debug is True
    target = path or "all configuration entries"
    interaction.confirm.assert_called_once_with(f"Reset {target} for session?", default=False)


def test_async_config_reset_rejects_invalid_confirmation(tmp_path):
    """The native async interaction cannot approve reset with a truthy non-boolean."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    configuration.set_session("app.debug", True)
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm_async = AsyncMock(return_value=object())
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),), interaction=interaction
    )

    asyncio.run(manager.call_async("config", "reset app.debug scope=session"))

    assert configuration.effective.app.debug is True
    interaction.confirm_async.assert_awaited_once_with(
        "Reset app.debug for session?", default=False
    )
    manager.close()


def test_config_workspace_reset_reveals_user_value(tmp_path):
    """Reset removes a workspace override so the user configuration wins."""
    configuration = ConfigurationManager(
        tmp_path / "config.toml", tmp_path / "project" / ".loop" / "config.toml"
    )
    configuration.initialize()
    configuration.load()
    configuration.set("app.debug", True, scope="user")
    configuration.set("app.debug", False, scope="workspace")
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm.return_value = True
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock(return_value="applied now")),),
        interaction=interaction,
    )

    manager.call("config", "reset app.debug scope=workspace")

    assert configuration.effective.app.debug is True
    assert configuration.source_for("app.debug") == "user"
    assert not (tmp_path / "project" / ".loop" / "config.toml").exists()


def test_config_reset_all_and_validation_failures_leave_state_consistent(tmp_path):
    """Reset without a path clears every selected-scope override and invalid writes fail safely."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm.return_value = True
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock(return_value="applied now")),),
        interaction=interaction,
    )

    manager.call("config", "set app.debug true scope=session")
    manager.call("config", "set app.max_agent_turns -1 scope=session")
    manager.call("config", "reset scope=session")
    manager.call("config", "reset app.debug scope=session")

    assert configuration.effective.app.debug is False
    assert interaction.report.call_count == 1


def test_config_scope_picker_cancels_reset_all(tmp_path):
    """The cancel choice leaves all session overrides untouched."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    configuration.set_session("app.debug", True)
    interaction = Mock(spec=ConsoleInteraction)
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock(return_value="applied now")),),
        interaction=interaction,
    )

    manager.call("config", "reset")

    assert configuration.effective.app.debug is True
    interaction.prompt.assert_called_once()


def test_config_explicit_user_reset_can_be_declined(tmp_path):
    """An explicitly scoped reset-all remains protected by confirmation."""
    configuration = ConfigurationManager(
        tmp_path / "config.toml", tmp_path / ".loop" / "config.toml"
    )
    configuration.load()
    interaction = Mock(spec=ConsoleInteraction)
    interaction.confirm.return_value = False
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, Mock()),), interaction=interaction
    )

    manager.call("config", "reset scope=user")

    interaction.confirm.assert_called_once()


def test_config_omitted_scope_prompts_and_supports_workspace_or_cancel(tmp_path):
    """Omitted scope remains interactive, including workspace selection and cancellation."""
    configuration = ConfigurationManager(
        tmp_path / "config.toml", tmp_path / ".loop" / "config.toml"
    )
    configuration.load()
    apply = Mock(return_value="applied")
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt.side_effect = ["workspace", "cancel", "session"]
    manager = CommandManager(
        providers=(ConfigurationCommands(configuration, apply),), interaction=interaction
    )

    manager.call("config", "set app.debug true")
    manager.call("config", "set app.stream false")
    manager.call("config", "reset app.debug")

    assert configuration.source_for("app.debug") == "session"
    assert configuration.effective.app.stream is True


def test_config_completion_excludes_secret_paths_from_get_and_set(tmp_path):
    """Get and set complete only non-secret schema-derived configuration paths."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    provider = ConfigurationCommands(configuration, Mock())
    commands = CommandManager(providers=(provider,))
    completion = CompletionManager(
        (CommandCompletionAdapter(lambda: commands.commands, providers=(provider,)),)
    )

    values = list(completion.get_completions(Document("/config get app.d"), Mock()))

    assert [value.text for value in values] == ["app.debug"]
    values = list(completion.get_completions(Document("/config set backend.a"), Mock()))

    assert not values


def test_config_secret_completion_suggests_only_secret_paths(tmp_path):
    """The secret action completes the entries that require masked input."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    provider = ConfigurationCommands(configuration, Mock())
    commands = CommandManager(providers=(provider,))
    completion = CompletionManager(
        (CommandCompletionAdapter(lambda: commands.commands, providers=(provider,)),)
    )

    values = list(completion.get_completions(Document("/config secret backend.a"), Mock()))

    assert [value.text for value in values] == ["backend.api_key"]


def test_config_completion_suggests_scope_values_after_edit_arguments(tmp_path):
    """Editable operations complete explicit session and workspace scope arguments."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    provider = ConfigurationCommands(configuration, Mock())
    commands = CommandManager(providers=(provider,))
    completion = CompletionManager(
        (CommandCompletionAdapter(lambda: commands.commands, providers=(provider,)),)
    )

    values = list(
        completion.get_completions(Document("/config set app.debug false scope="), Mock())
    )
    positional_values = list(
        completion.get_completions(Document("/config set app.debug false "), Mock())
    )

    assert [value.text for value in values] == [
        "scope=session",
        "scope=user",
        "scope=workspace",
    ]
    assert {value.text for value in positional_values} == {"session", "user", "workspace"}


def test_config_reset_completion_suggests_scope_without_a_path(tmp_path):
    """Reset-all completion includes the session and workspace scope arguments."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    provider = ConfigurationCommands(configuration, Mock())
    commands = CommandManager(providers=(provider,))
    completion = CompletionManager(
        (CommandCompletionAdapter(lambda: commands.commands, providers=(provider,)),)
    )

    values = list(completion.get_completions(Document("/config reset scope="), Mock()))

    assert [value.text for value in values] == [
        "scope=session",
        "scope=user",
        "scope=workspace",
    ]


def test_config_reset_completion_suggests_scope_after_a_path(tmp_path):
    """Single-entry reset completion includes the session and workspace scope arguments."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    provider = ConfigurationCommands(configuration, Mock())
    commands = CommandManager(providers=(provider,))
    completion = CompletionManager(
        (CommandCompletionAdapter(lambda: commands.commands, providers=(provider,)),)
    )

    values = list(completion.get_completions(Document("/config reset app.debug scope="), Mock()))

    assert [value.text for value in values] == [
        "scope=session",
        "scope=user",
        "scope=workspace",
    ]


def test_config_completion_lists_arguments_for_selected_operation(tmp_path):
    """Configuration completion shows values and remaining names for the chosen action."""
    configuration = ConfigurationManager(tmp_path / ".loop" / "config.toml")
    configuration.load()
    provider = ConfigurationCommands(configuration, Mock())
    commands = CommandManager(providers=(provider,))
    completion = CompletionManager(
        (CommandCompletionAdapter(lambda: commands.commands, providers=(provider,)),)
    )

    actions = list(completion.get_completions(Document("/config "), Mock()))
    get_arguments = list(completion.get_completions(Document("/config get "), Mock()))
    set_arguments = list(completion.get_completions(Document("/config set "), Mock()))
    secret_arguments = list(completion.get_completions(Document("/config secret "), Mock()))
    reset_arguments = list(completion.get_completions(Document("/config reset "), Mock()))
    reload_arguments = list(completion.get_completions(Document("/config reload "), Mock()))

    assert {value.text for value in actions} == {"get", "set", "secret", "reset", "reload"}
    assert {value.text for value in get_arguments} >= {"app.debug"}
    assert "path=" not in {value.text for value in get_arguments}
    assert "backend.api_key" not in {value.text for value in get_arguments}
    assert "path=" not in {value.text for value in set_arguments}
    assert {value.text for value in set_arguments} >= {
        "value=",
        "scope=",
        "app.debug",
    }
    assert "backend.api_key" not in {value.text for value in set_arguments}
    assert {value.text for value in secret_arguments} >= {"scope=", "backend.api_key"}
    assert "path=" not in {value.text for value in secret_arguments}
    assert "value=" not in {value.text for value in secret_arguments}
    assert "path=" not in {value.text for value in reset_arguments}
    assert {value.text for value in reset_arguments} >= {
        "scope=",
        "backend.api_key",
    }
    assert list(reload_arguments) == []
