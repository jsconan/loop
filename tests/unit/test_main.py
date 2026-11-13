"""Verify the application command-line composition root."""

import asyncio
import runpy
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call

import pytest

from loop import ShutdownRequested
from loop import main as application
from loop.application.runtime import ApplicationRuntime as RealApplicationRuntime
from loop.configuration import ApplicationSettings
from loop.configuration.manager import ConfigurationManager as RealConfigurationManager
from loop.interaction import ConsoleInteraction
from loop.permissions import PermissionNamespaceError
from loop.workspace import Workspace, WorkspaceSwitchRequested


def runtime_factory(create):
    """Compose test runtimes through the real recovery coordinator."""
    runtime = Mock(create=create)

    async def create_interactively(*args):
        return await RealApplicationRuntime.create_interactively_async.__func__(runtime, *args)

    runtime.create_interactively_async = AsyncMock(side_effect=create_interactively)
    return runtime


@pytest.fixture(autouse=True)
def isolate_main(monkeypatch):
    """Replace filesystem discovery and composition with isolated doubles."""
    monkeypatch.setattr(application, "load_dotenv", Mock())
    workspace = Workspace(Path("/project"), Path("/project"))
    initialized = Workspace(Path("/project"), Path("/project"), "id", "project", "directory", 1, 1)
    monkeypatch.setattr(application, "Workspace", Mock(discover=Mock(return_value=workspace)))
    paths = Mock()
    paths.user_configuration = Path("/config/config.toml")
    paths.workspace_catalog = Path("/data/workspaces.db")
    paths.for_workspace.return_value = Mock()
    monkeypatch.setattr(application, "ApplicationPaths", Mock(discover=Mock(return_value=paths)))
    configuration = Mock()
    configuration.load.return_value = ApplicationSettings()

    async def load_interactively(interaction):
        return await RealConfigurationManager.load_interactively_async(configuration, interaction)

    configuration.load_interactively_async = AsyncMock(side_effect=load_interactively)
    monkeypatch.setattr(application, "ConfigurationManager", Mock(return_value=configuration))
    repository = Mock()
    repository.initialize.return_value = initialized
    monkeypatch.setattr(application, "WorkspaceRepository", Mock(return_value=repository))
    monkeypatch.setattr(application, "ApplicationMigration", Mock())
    monkeypatch.setattr(application, "set_telemetry", Mock())


def test_run_application_async_closes_runtime(monkeypatch):
    """The async entry point runs and closes its composed runtime directly."""
    interaction = Mock(spec=ConsoleInteraction)
    runtime = Mock()
    runtime.run_async = AsyncMock()
    runtime.close_async = AsyncMock()
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(
        application, "ApplicationRuntime", runtime_factory(Mock(return_value=runtime))
    )
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())

    asyncio.run(application.run_application_async())

    interaction.info.assert_called_once_with("Hello from loop!")
    runtime.run_async.assert_awaited_once_with()
    runtime.close_async.assert_awaited_once_with()


@pytest.mark.parametrize("choice", ["retry", "continue", "exit"])
def test_startup_configuration_recovery_requires_explicit_selection(monkeypatch, choice):
    """Startup reports bad settings and only proceeds after the selected recovery action."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt_async = AsyncMock(return_value=choice)
    runtime = Mock()
    runtime.run_async = AsyncMock()
    runtime.close_async = AsyncMock()
    configuration = application.ConfigurationManager.return_value
    settings = ApplicationSettings()
    configuration.load.side_effect = [ValueError("private value"), settings]
    configuration.use_defaults.return_value = settings
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(
        application, "ApplicationRuntime", runtime_factory(Mock(return_value=runtime))
    )
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())
    monkeypatch.setattr(application, "_LOGGER", Mock())

    application.run_application()

    problem = interaction.report.call_args.args[0]
    assert problem.code == "configuration.startup_failed"
    assert "private value" not in problem.detail
    interaction.prompt_async.assert_awaited_once()
    if choice == "exit":
        runtime.run_async.assert_not_awaited()
    else:
        runtime.run_async.assert_awaited_once()
    assert configuration.use_defaults.call_count == (choice == "continue")


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt, ShutdownRequested])
def test_run_application_gracefully_handles_shutdown(monkeypatch, interruption):
    """Interruptions stop and close a constructed runtime with a friendly message."""
    interaction = Mock(spec=ConsoleInteraction)
    runtime = Mock()
    runtime.run_async = AsyncMock(side_effect=interruption)
    runtime.close_async = AsyncMock()
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(
        application, "ApplicationRuntime", runtime_factory(Mock(return_value=runtime))
    )
    signals = Mock()
    monkeypatch.setattr(application, "register_shutdown_signals", signals)

    application.run_application()

    signals.assert_called_once_with()
    runtime.run.assert_not_called()
    runtime.stop.assert_called_once_with()
    runtime.close_async.assert_awaited_once_with()
    assert interaction.info.call_args_list == [
        call("Hello from loop!"),
        call("\nStopping loop. Goodbye!"),
    ]


def test_run_application_reports_runtime_failure_and_closes(monkeypatch):
    """Unexpected runtime failures become fatal problems before close."""
    interaction = Mock(spec=ConsoleInteraction)
    runtime = Mock()
    runtime.run_async = AsyncMock(side_effect=RuntimeError("private"))
    runtime.close_async = AsyncMock()
    logger = Mock()
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(
        application, "ApplicationRuntime", runtime_factory(Mock(return_value=runtime))
    )
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())
    monkeypatch.setattr(application, "_LOGGER", logger)

    application.run_application()

    logger.log.assert_called_once()
    interaction.report.assert_called_once()
    assert interaction.report.call_args.args[0].severity == "fatal"
    runtime.close_async.assert_awaited_once_with()


def test_policy_recovery_reports_repair_path_without_starting_model(monkeypatch):
    """CLI startup reports a policy failure and halts when the user declines retry."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt_async = AsyncMock(return_value="exit")
    error = PermissionNamespaceError(Path("/policy"), "Restore or repair the policy.")
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    runtime = runtime_factory(Mock(side_effect=error))
    monkeypatch.setattr(application, "ApplicationRuntime", runtime)
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())
    monkeypatch.setattr(application, "_LOGGER", Mock())

    application.run_application()

    problem = interaction.report.call_args.args[0]
    assert problem.code == "permission.policy_recovery_required"
    assert problem.severity == "error"
    assert "/policy" in problem.detail
    assert "Restore or repair" in problem.detail
    runtime.create.assert_called_once()


def test_policy_recovery_retries_after_repair(monkeypatch):
    """Startup retries protected policy loading after the user chooses reload."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt_async = AsyncMock(return_value="retry")
    error = PermissionNamespaceError(Path("/policy"), "Restore or repair the policy.")
    runtime = Mock()
    runtime.run_async = AsyncMock()
    runtime.close_async = AsyncMock()
    create = Mock(side_effect=[error, runtime])
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(application, "ApplicationRuntime", runtime_factory(create))
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())
    monkeypatch.setattr(application, "_LOGGER", Mock())

    application.run_application()

    assert create.call_count == 2
    interaction.prompt_async.assert_awaited_once()
    runtime.run_async.assert_awaited_once()


def test_policy_recovery_before_console_setup_still_closes_cleanly(monkeypatch):
    """An early policy failure needs no console to release process telemetry."""
    error = PermissionNamespaceError(Path("/policy"), "Restore the policy.")
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(side_effect=error))
    monkeypatch.setattr(application, "_LOGGER", Mock())

    application.run_application()

    application.set_telemetry.assert_called_once_with(None)


def test_run_application_clears_telemetry_when_failure_prevents_runtime(monkeypatch):
    """Failure before runtime ownership clears any process telemetry facade."""
    monkeypatch.setattr(
        application, "ConsoleInteraction", Mock(side_effect=RuntimeError("terminal"))
    )
    monkeypatch.setattr(application, "_LOGGER", Mock())

    application.run_application()

    application.set_telemetry.assert_called_once_with(None)


def test_run_application_rejects_an_active_event_loop():
    """The CLI root does not create a nested event loop or an unawaited coroutine."""

    async def scenario():
        """Attempt to start the synchronous CLI entry point from async code."""
        with pytest.raises(RuntimeError, match=r"await run_application_async\(\) instead"):
            application.run_application()

    asyncio.run(scenario())


def test_headless_policy_initialization_uses_explicit_host_cli(monkeypatch):
    """The direct CLI operation provisions a named namespace without a model or prompt."""
    manager = Mock()
    monkeypatch.setattr(application, "PermissionManager", Mock(return_value=manager))
    monkeypatch.setattr(application.sys, "argv", ["loop", "--initialize-policy", "workspace", "id"])

    application.run_application()

    manager.initialize_policy_namespace.assert_called_once_with("workspace", "id", authorized=True)
    manager.close.assert_called_once_with()


@pytest.mark.parametrize(
    "arguments", [["loop", "--initialize-policy"], ["loop", "--initialize-policy", "user"]]
)
def test_headless_policy_initialization_requires_scope_and_identity(monkeypatch, arguments):
    """Incomplete provisioning fails with a non-success CLI result."""
    monkeypatch.setattr(application.sys, "argv", arguments)
    with pytest.raises(SystemExit, match="Usage: loop --initialize-policy"):
        application.run_application()


def test_headless_policy_initialization_rejects_namespace_errors(monkeypatch):
    """A registry or identity failure remains a non-success, actionable CLI outcome."""
    manager = Mock()
    manager.initialize_policy_namespace.side_effect = PermissionNamespaceError(
        Path("/policy"), "Identity mismatch."
    )
    monkeypatch.setattr(application, "PermissionManager", Mock(return_value=manager))
    monkeypatch.setattr(application.sys, "argv", ["loop", "--initialize-policy", "user", "wrong"])

    with pytest.raises(SystemExit, match="Identity mismatch"):
        application.run_application()
    manager.close.assert_called_once_with()


def test_headless_policy_initialization_requires_workspace_identity(monkeypatch):
    """Provisioning cannot create policy without a registered workspace UUID."""
    repository = application.WorkspaceRepository.return_value
    repository.initialize.return_value.id = None
    with pytest.raises(ValueError, match="Workspace identity is unavailable"):
        application._initialize_policy_namespace("workspace", "id")


@pytest.mark.parametrize("with_interaction", [False, True])
def test_run_application_handles_shutdown_before_runtime_ownership(monkeypatch, with_interaction):
    """Early shutdown needs neither a runtime nor an initialized interaction."""
    if with_interaction:
        interaction = Mock(spec=ConsoleInteraction)
        monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
        monkeypatch.setattr(
            application, "register_shutdown_signals", Mock(side_effect=ShutdownRequested)
        )
    else:
        monkeypatch.setattr(application, "ConsoleInteraction", Mock(side_effect=ShutdownRequested))

    application.run_application()

    application.set_telemetry.assert_called_once_with(None)
    if with_interaction:
        interaction.info.assert_called_once_with("\nStopping loop. Goodbye!")


def test_module_runs_application_entry_point(monkeypatch):
    """Executing the source module as a script invokes its entry point."""
    monkeypatch.setattr("loop.interaction.ConsoleInteraction", Mock(return_value=Mock()))
    runtime = Mock()
    runtime.run_async = AsyncMock()
    runtime.close_async = AsyncMock()
    monkeypatch.setattr(
        "loop.application.ApplicationRuntime.create_interactively_async",
        AsyncMock(return_value=runtime),
    )
    monkeypatch.setattr("loop.utils.find_project_root", Mock(return_value=Path.cwd()))
    monkeypatch.setattr("loop.utils.register_shutdown_signals", Mock())
    configuration = Mock()
    configuration.load_interactively_async = AsyncMock(return_value=ApplicationSettings())
    monkeypatch.setattr("loop.configuration.ConfigurationManager", Mock(return_value=configuration))
    monkeypatch.setattr("loop.telemetry.set_telemetry", Mock())

    with pytest.warns(RuntimeWarning, match="'loop.main' found in sys.modules"):
        runpy.run_module("loop.main", run_name="__main__")


def test_run_application_closes_runtime_before_workspace_switch(monkeypatch):
    """A switch signal tears down global owners before composing the target runtime."""
    target = Workspace(Path("/target"), Path("/target"), "target", "target", "directory", 2, 2)
    first = Mock()
    first.run_async = AsyncMock(side_effect=WorkspaceSwitchRequested(target))
    first.close_async = AsyncMock()
    second = Mock()
    second.close_async = AsyncMock()
    factory = Mock(side_effect=[first, second])
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=Mock()))
    monkeypatch.setattr(application, "ApplicationRuntime", runtime_factory(factory))
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())

    application.run_application()

    first.close_async.assert_awaited_once_with()
    assert factory.call_args_list[1].args[0] is target
    second.close_async.assert_awaited_once_with()


def test_run_application_restores_workspace_when_switch_rebuild_fails(monkeypatch):
    """A failed target composition rebuilds and continues the previous workspace."""
    target = Workspace(Path("/target"), Path("/target"), "target", "target", "directory", 2, 2)
    first = Mock()
    first.run_async = AsyncMock(side_effect=WorkspaceSwitchRequested(target))
    first.close_async = AsyncMock()
    restored = Mock()
    restored.close_async = AsyncMock()
    factory = Mock(side_effect=[first, RuntimeError("target failed"), restored])
    interaction = Mock(spec=ConsoleInteraction)
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(application, "ApplicationRuntime", runtime_factory(factory))
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())

    application.run_application()

    assert factory.call_args_list[1].args[0] is target
    assert factory.call_args_list[2].args[0].id == "id"
    assert "restored workspace id" in interaction.warning.call_args.args[0]
    restored.close_async.assert_awaited_once_with()


@pytest.mark.parametrize("failure", ["rollback", "halt"])
def test_workspace_switch_does_not_rebuild_after_unsafe_or_declined_recovery(monkeypatch, failure):
    """A failed rollback or an explicit halt cannot trigger another runtime construction."""
    target = Workspace(Path("/target"), Path("/target"), "target", "target", "directory", 2, 2)
    first = Mock()
    first.run_async = AsyncMock(side_effect=WorkspaceSwitchRequested(target))
    first.close_async = AsyncMock()
    error = (
        application.RuntimeRollbackError("rollback failed")
        if failure == "rollback"
        else PermissionNamespaceError(Path("/policy"), "Restore the policy.")
    )
    create = Mock(side_effect=[first, error])
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt_async = AsyncMock(return_value="exit")
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(application, "ApplicationRuntime", runtime_factory(create))
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())
    monkeypatch.setattr(application, "_LOGGER", Mock())

    application.run_application()

    assert create.call_count == 2
    first.close_async.assert_awaited_once()


@pytest.mark.parametrize("choice", ["retry", "continue", "exit"])
def test_startup_recovers_invalid_preset_catalog(monkeypatch, choice):
    """Startup reports every preset failure before retrying, excluding, or stopping."""
    from loop.permissions import PermissionLoadFailure, PermissionPresetError

    failure = PermissionLoadFailure(source="preset", path="presets/bad.yaml", message="bad field")
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt_async = AsyncMock(return_value=choice)
    runtime = Mock()
    runtime.run_async = AsyncMock()
    runtime.close_async = AsyncMock()
    create = Mock(side_effect=[PermissionPresetError((failure,)), runtime])
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(application, "ApplicationRuntime", runtime_factory(create))
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())

    application.run_application()

    assert "presets/bad.yaml: bad field" in interaction.report.call_args.args[0].detail
    if choice == "exit":
        assert create.call_count == 1
    else:
        assert create.call_count == 2
        assert create.call_args.kwargs["accepted_preset_failures"] == (
            (failure,) if choice == "continue" else ()
        )


def test_startup_reports_a_namespace_failure_before_runtime(monkeypatch, caplog):
    """A policy namespace failure before runtime composition receives a fatal diagnostic."""
    interaction = Mock(spec=ConsoleInteraction)
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    application.WorkspaceRepository.return_value.initialize.side_effect = PermissionNamespaceError(
        "/policy", "invalid registry"
    )
    asyncio.run(application.run_application_async())
    interaction.report.assert_not_called()
    assert "Permission policy recovery required" in caplog.text


def test_namespace_failure_during_runtime_is_reported_and_closed(monkeypatch):
    """An unexpected namespace failure is reported after the runtime has started."""
    interaction = Mock(spec=ConsoleInteraction)
    runtime = Mock()
    runtime.run_async = AsyncMock(side_effect=PermissionNamespaceError("/policy", "invalid"))
    runtime.close_async = AsyncMock()
    monkeypatch.setattr(application, "ConsoleInteraction", Mock(return_value=interaction))
    monkeypatch.setattr(
        application, "ApplicationRuntime", runtime_factory(Mock(return_value=runtime))
    )
    monkeypatch.setattr(application, "register_shutdown_signals", Mock())
    asyncio.run(application.run_application_async())
    assert interaction.report.call_args.args[0].code == "permission.policy_recovery_required"
    runtime.close_async.assert_awaited_once()
