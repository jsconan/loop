"""Verify application runtime composition and shutdown."""

import asyncio
import gc
import logging
import warnings
from threading import Event, get_ident
from unittest.mock import AsyncMock, MagicMock, Mock, call

import pytest

import loop.application.runtime as runtime_module
from loop import ConsoleInteraction
from loop.application import ApplicationPaths
from loop.application.commands import ApplicationCommands
from loop.application.runtime import ApplicationRuntime
from loop.configuration import ApplicationSettings, ConfigurationCommands
from loop.conversation import ConversationCommands
from loop.telemetry import SQLiteTelemetryAdapter, Telemetry
from loop.workspace import Workspace, WorkspaceCommands


@pytest.fixture
def assembled(tmp_path):
    """Return initialized workspace, immutable paths, and settings."""
    project = tmp_path / "project"
    project.mkdir()
    workspace = Workspace(project, project, "workspace-id", "project", "directory", 1, 1)
    paths = ApplicationPaths(tmp_path / "config", tmp_path / "data", tmp_path / "state")
    workspace_paths = paths.for_workspace(workspace.id, project)
    settings = ApplicationSettings()
    return workspace, paths, workspace_paths, settings


@pytest.fixture
def dependencies(monkeypatch):
    """Replace runtime collaborators with composition-boundary doubles."""
    names = (
        "OpenAIBackend",
        "SQLiteTelemetryAdapter",
        "Telemetry",
        "create_default_tool_registry",
        "SQLiteSessionStore",
        "SessionManager",
        "PermissionManager",
        "CommandExecutionService",
    )
    result = {name: Mock(return_value=Mock()) for name in names}
    loop = Mock()
    active_loop = Mock()
    loop.create_default.return_value = active_loop
    result["Loop"] = loop
    for name in (
        "configure_operational_logging",
        "get_telemetry",
        "set_telemetry",
        "telemetry_activity",
    ):
        result[name] = Mock()
    active_telemetry = {"value": None}
    result["get_telemetry"].side_effect = lambda: active_telemetry["value"]
    result["set_telemetry"].side_effect = lambda value: active_telemetry.__setitem__("value", value)
    for name, replacement in result.items():
        monkeypatch.setattr(runtime_module, name, replacement)
    return result


def test_create_composes_runtime_from_bound_references(dependencies, assembled):
    """One settings snapshot and reference graph configures every runtime component."""
    workspace, paths, workspace_paths, settings = assembled
    settings = settings.model_copy(
        update={
            "loop": settings.loop.model_copy(
                update={"temperature": 0.2, "reasoning_effort": "medium"}
            ),
        }
    )
    configuration = Mock()
    repository = Mock()
    interaction = Mock()

    runtime = ApplicationRuntime.create(
        workspace, paths, workspace_paths, settings, configuration, repository, interaction
    )

    active_loop = dependencies["Loop"].create_default.return_value
    assert [
        type(item.args[0]) for item in active_loop.register_command_provider.call_args_list
    ] == [
        ConfigurationCommands,
        ApplicationCommands,
        WorkspaceCommands,
    ]

    logging_path = dependencies["configure_operational_logging"].call_args.args[0]
    assert logging_path == paths.operational_log
    telemetry_path = dependencies["SQLiteTelemetryAdapter"].call_args.args[0]
    session_path = dependencies["SQLiteSessionStore"].call_args.args[0]
    assert telemetry_path == paths.telemetry
    assert session_path == workspace_paths.sessions
    assert dependencies["PermissionManager"].call_args.kwargs["user_configuration_path"] == (
        paths.user_permissions
    )
    loop_kwargs = dependencies["Loop"].create_default.call_args.kwargs
    tool_kwargs = dependencies["create_default_tool_registry"].call_args.kwargs
    assert tool_kwargs["settings"].user_agent == settings.web.user_agent
    assert tool_kwargs["settings"].command_timeout == settings.tools.command_timeout
    assert tool_kwargs["permission_manager"] is dependencies["PermissionManager"].return_value
    assert loop_kwargs["permission_manager"] is tool_kwargs["permission_manager"]
    assert loop_kwargs["refresh_sandbox_backend"] == (
        dependencies["CommandExecutionService"].for_host.return_value.refresh_backend
    )
    assert loop_kwargs["working_directory"] is workspace.working_directory
    assert loop_kwargs["stream"] is settings.loop.stream
    assert loop_kwargs["temperature"] == 0.2
    assert loop_kwargs["reasoning_effort"] == "medium"
    assert loop_kwargs["compaction_threshold"] == settings.loop.compaction_threshold
    loop_kwargs["on_model_select"]("selected")
    configuration.set.assert_called_once_with("loop.model", "selected")
    dependencies["set_telemetry"].assert_called_once_with(dependencies["Telemetry"].return_value)

    runtime.run()
    runtime.stop()
    runtime.close()

    dependencies["Loop"].create_default.return_value.run.assert_called_once_with()
    dependencies["telemetry_activity"].assert_has_calls(
        [
            call("application.stopping", severity="info", reason="interrupted"),
            call("application.stopped", severity="info", component="main"),
        ]
    )
    dependencies["set_telemetry"].assert_called_with(None)


def test_async_runtime_registers_and_binds_the_session_command_provider(
    dependencies, assembled, monkeypatch
):
    """Async runtimes share the loop command manager with the registered controller."""
    workspace, paths, workspace_paths, settings = assembled
    interaction = Mock(spec=ConsoleInteraction)
    controller = Mock()
    controller_class = Mock(return_value=controller)
    monkeypatch.setattr(runtime_module, "SessionController", controller_class)

    runtime = ApplicationRuntime.create(
        workspace, paths, workspace_paths, settings, Mock(), Mock(), interaction
    )

    active_loop = dependencies["Loop"].create_default.return_value
    controller_class.assert_called_once_with(interaction)
    assert isinstance(
        active_loop.register_command_provider.call_args_list[0].args[0], ConversationCommands
    )
    session_manager = dependencies["Loop"].create_default.call_args.kwargs["session_manager"]
    controller.bind_session_manager.assert_called_once_with(session_manager)
    controller.bind_command_manager.assert_called_once_with(active_loop.command_manager)
    controller.running = False
    runtime.close()


def test_runtime_async_path_without_controller_awaits_loop_directly():
    """Legacy interaction compositions can use the runtime async facade directly."""
    loop = Mock()
    loop.run_async = AsyncMock()
    runtime = ApplicationRuntime(loop, Mock(), 1, cleanup_callbacks=[])

    asyncio.run(runtime.run_async())

    loop.run_async.assert_awaited_once_with()
    runtime.close()


def test_create_requires_identity_and_skips_environment_model_callback(dependencies, assembled):
    """Runtime assembly requires identity and does not persist environment-owned models."""
    workspace, paths, workspace_paths, settings = assembled
    configuration = Mock()
    repository = Mock()
    configuration.source_for.return_value = "environment"

    ApplicationRuntime.create(
        workspace, paths, workspace_paths, settings, configuration, repository, Mock()
    )
    assert dependencies["Loop"].create_default.call_args.kwargs["on_model_select"] is None

    with pytest.raises(ValueError, match="initialized"):
        ApplicationRuntime.create(
            Workspace(workspace.root, workspace.working_directory),
            paths,
            workspace_paths,
            settings,
            configuration,
            repository,
            Mock(),
        )


def test_create_accepts_execution_service_without_close_hook(dependencies, assembled, monkeypatch):
    """Composition leaves ownership with the registry when a service has no close hook."""
    workspace, paths, workspace_paths, settings = assembled
    service = object()
    monkeypatch.setattr(
        runtime_module, "CommandExecutionService", Mock(for_host=Mock(return_value=service))
    )

    runtime = ApplicationRuntime.create(
        workspace, paths, workspace_paths, settings, Mock(), Mock(), Mock()
    )

    assert (
        dependencies["create_default_tool_registry"].call_args.kwargs["execution_service"]
        is service
    )
    runtime.close()


def test_apply_configuration_routes_aggregate_domain_and_scalar_changes(dependencies):
    """Backend and model operations remain explicit while scalar references update directly."""
    active_loop = Mock()
    settings = ApplicationSettings.model_validate({"loop": {"debug": True, "model": "model"}})
    active_loop.apply_runtime_settings.side_effect = lambda path, _settings: (
        "saved; restart required" if path.startswith("telemetry.") else "applied now"
    )
    runtime = ApplicationRuntime(active_loop, Mock(), 2.0)

    assert runtime.apply_configuration("backend.default_model", settings) == "applied now"
    assert runtime.apply_configuration("loop.temperature", settings) == "applied now"
    assert runtime.apply_configuration("loop.model", settings) == "applied now"
    assert runtime.apply_configuration("loop.debug", settings) == "applied now"
    assert (
        runtime.apply_configuration("telemetry.batch_size", settings) == "saved; restart required"
    )

    active_loop.replace_backend.assert_called_once_with(dependencies["OpenAIBackend"].return_value)
    assert active_loop.apply_runtime_settings.call_count == 4


def test_apply_configuration_changes_replaces_the_backend_once(dependencies):
    """Reload batches backend changes while retaining per-path runtime statuses."""
    active_loop = Mock()
    active_loop.apply_runtime_settings.return_value = "applied now"
    runtime = ApplicationRuntime(active_loop, Mock(), 2.0)

    statuses = runtime.apply_configuration_changes(
        ("backend.base_url", "backend.default_model", "loop.debug"), ApplicationSettings()
    )

    assert statuses == {
        "backend.base_url": "applied now",
        "backend.default_model": "applied now",
        "loop.debug": "applied now",
    }
    active_loop.replace_backend.assert_called_once_with(dependencies["OpenAIBackend"].return_value)
    active_loop.apply_runtime_settings.assert_called_once_with("loop.debug", ApplicationSettings())


def test_apply_configuration_changes_skips_backend_replacement_without_backend_changes(
    dependencies,
):
    """Reload applies scalar-only changes without reconstructing the backend."""
    active_loop = Mock()
    active_loop.apply_runtime_settings.return_value = "applied now"
    runtime = ApplicationRuntime(active_loop, Mock(), 2.0)

    statuses = runtime.apply_configuration_changes(("loop.debug",), ApplicationSettings())

    assert statuses == {"loop.debug": "applied now"}
    active_loop.replace_backend.assert_not_called()


@pytest.mark.parametrize("after_telemetry", [False, True])
def test_create_closes_only_successfully_created_telemetry(
    dependencies, assembled, after_telemetry
):
    """Failed composition closes telemetry only after telemetry construction succeeded."""
    workspace, paths, workspace_paths, settings = assembled
    error = RuntimeError("composition failed")
    if after_telemetry:
        dependencies["Loop"].create_default.side_effect = error
    else:
        dependencies["OpenAIBackend"].side_effect = error

    with pytest.raises(RuntimeError, match="composition failed"):
        ApplicationRuntime.create(
            workspace, paths, workspace_paths, settings, Mock(), Mock(), Mock()
        )

    if after_telemetry:
        dependencies["Telemetry"].return_value.close.assert_called_once_with(
            timeout=settings.telemetry.shutdown_timeout
        )
    else:
        dependencies["Telemetry"].return_value.close.assert_not_called()
    dependencies["configure_operational_logging"].return_value.close.assert_called_once_with()


def test_close_without_owned_log_handler_remains_safe():
    """Injected runtimes can close telemetry without owning process logging."""
    telemetry = Mock()
    runtime = ApplicationRuntime(Mock(), telemetry, 1.0)

    runtime.close()

    telemetry.close.assert_called_once_with(timeout=1.0)


def test_failed_create_without_logging_or_telemetry_owners_is_safe(dependencies, assembled):
    """Early composition failure needs no cleanup when global owners were unavailable."""
    workspace, paths, workspace_paths, settings = assembled
    dependencies["configure_operational_logging"].return_value = None
    dependencies["OpenAIBackend"].side_effect = RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        ApplicationRuntime.create(
            workspace, paths, workspace_paths, settings, Mock(), Mock(), Mock()
        )


def test_backend_failure_rolls_back_only_owned_logging(dependencies, assembled):
    """A backend failure removes and closes the handler acquired for the attempt."""
    workspace, paths, workspace_paths, settings = assembled
    handler = dependencies["configure_operational_logging"].return_value
    dependencies["OpenAIBackend"].side_effect = RuntimeError("backend failed")

    with pytest.raises(RuntimeError, match="backend failed"):
        ApplicationRuntime.create(
            workspace, paths, workspace_paths, settings, Mock(), Mock(), Mock()
        )

    handler.close.assert_called_once_with()
    dependencies["set_telemetry"].assert_not_called()


def test_backend_failure_restores_owned_root_logging_state(dependencies, assembled):
    """Rollback removes the installed handler and restores the prior root level."""
    workspace, paths, workspace_paths, settings = assembled
    root = logging.getLogger()
    previous_level = root.level
    installed_level = logging.ERROR if previous_level != logging.ERROR else logging.DEBUG
    handler = Mock(spec=logging.Handler)

    def install_logging(*args, **kwargs):
        root.addHandler(handler)
        root.setLevel(installed_level)
        return handler

    dependencies["configure_operational_logging"].side_effect = install_logging
    dependencies["OpenAIBackend"].side_effect = RuntimeError("backend failed")

    try:
        with pytest.raises(RuntimeError, match="backend failed"):
            ApplicationRuntime.create(
                workspace, paths, workspace_paths, settings, Mock(), Mock(), Mock()
            )

        assert handler not in root.handlers
        assert root.level == previous_level
        handler.close.assert_called_once_with()
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


def test_command_failure_closes_all_acquired_resources(dependencies, assembled):
    """Command-registration failure closes sessions, telemetry, backend, and logging."""
    workspace, paths, workspace_paths, settings = assembled
    active_loop = dependencies["Loop"].create_default.return_value
    active_loop.register_command_provider.side_effect = RuntimeError("commands failed")

    with pytest.raises(RuntimeError, match="commands failed"):
        ApplicationRuntime.create(
            workspace, paths, workspace_paths, settings, Mock(), Mock(), Mock()
        )

    dependencies["SQLiteSessionStore"].return_value.close.assert_called_once_with()
    dependencies["Telemetry"].return_value.close.assert_called_once_with(
        timeout=settings.telemetry.shutdown_timeout
    )
    dependencies["OpenAIBackend"].return_value.close.assert_called_once_with()
    dependencies["configure_operational_logging"].return_value.close.assert_called_once_with()
    dependencies["set_telemetry"].assert_not_called()


def test_late_startup_failure_unpublishes_before_rollback(dependencies, assembled):
    """A failure after publication restores the facade and shuts telemetry down."""
    workspace, paths, workspace_paths, settings = assembled
    prior = Mock()
    dependencies["set_telemetry"](prior)
    dependencies["set_telemetry"].reset_mock()
    dependencies["telemetry_activity"].side_effect = RuntimeError("start event failed")

    with pytest.raises(RuntimeError, match="start event failed"):
        ApplicationRuntime.create(
            workspace, paths, workspace_paths, settings, Mock(), Mock(), Mock()
        )

    assert dependencies["set_telemetry"].call_args_list == [
        call(dependencies["Telemetry"].return_value),
        call(prior),
    ]
    dependencies["Telemetry"].return_value.close.assert_called_once_with(
        timeout=settings.telemetry.shutdown_timeout
    )


def test_runtime_restores_prior_telemetry_owner(dependencies, assembled):
    """Shutdown restores a pre-existing facade without overwriting a later owner."""
    workspace, paths, workspace_paths, settings = assembled
    prior = Mock()
    later = Mock()
    dependencies["set_telemetry"](prior)

    runtime = ApplicationRuntime.create(
        workspace, paths, workspace_paths, settings, Mock(), Mock(), Mock()
    )
    runtime.close()
    assert dependencies["get_telemetry"]() is prior

    dependencies["set_telemetry"](prior)
    runtime = ApplicationRuntime.create(
        workspace, paths, workspace_paths, settings, Mock(), Mock(), Mock()
    )
    dependencies["set_telemetry"](later)
    runtime.close()
    assert dependencies["get_telemetry"]() is later


def test_publication_follows_complete_command_registration(dependencies, assembled):
    """Telemetry becomes global only after every command group is registered."""
    workspace, paths, workspace_paths, settings = assembled
    events = []
    dependencies["Loop"].create_default.return_value.register_command_provider.side_effect = (
        lambda provider: events.append(("register", provider))
    )
    dependencies["set_telemetry"].side_effect = lambda value: events.append(("publish", value))

    runtime = ApplicationRuntime.create(
        workspace, paths, workspace_paths, settings, Mock(), Mock(), Mock()
    )

    assert [name for name, _value in events] == ["register", "register", "register", "publish"]
    runtime.close()


def test_close_is_idempotent_and_context_exit_closes_resources():
    """Explicit and context-managed shutdown close each owned resource exactly once."""
    telemetry = Mock()
    handler = Mock(spec=logging.Handler)

    with ApplicationRuntime(Mock(), telemetry, 1.0, logging_handler=handler) as runtime:
        assert runtime is not None
    runtime.close()

    telemetry.close.assert_called_once_with(timeout=1.0)
    handler.close.assert_called_once_with()


def test_async_close_uses_async_resource_cleanup_and_is_idempotent():
    """Async shutdown closes loop-affine resources through their async lifecycle once."""

    class Resource:
        """Represent a resource whose initialized state requires async shutdown."""

        def __init__(self):
            self.sync_closes = 0
            self.async_closes = 0

        @property
        def async_close_required(self):
            """Report whether the resource needs its owner event loop for cleanup."""
            return True

        def close(self):
            """Record an invalid synchronous cleanup attempt."""
            self.sync_closes += 1

        async def close_async(self):
            """Record async cleanup on the runtime's active event loop."""
            self.async_closes += 1

    resource = Resource()
    runtime = ApplicationRuntime(Mock(), Mock(), 1.0, cleanup_callbacks=[resource.close])

    with pytest.raises(RuntimeError, match="close_async"):
        runtime.close()

    async def close_twice():
        """Close the resource and verify a repeated request is a no-op."""
        await runtime.close_async()
        await runtime.close_async()

    asyncio.run(close_twice())

    assert resource.sync_closes == 0
    assert resource.async_closes == 1


def test_async_close_runs_sync_cleanup_off_the_event_loop():
    """Synchronous resource cleanup cannot block the runtime's owner loop."""
    cleanup_threads = []
    runtime = ApplicationRuntime(
        Mock(),
        Mock(),
        1.0,
        cleanup_callbacks=[lambda: cleanup_threads.append(get_ident())],
    )

    async def close_runtime():
        """Return the event-loop thread after runtime cleanup settles."""
        owner_thread = get_ident()
        await runtime.close_async()
        return owner_thread

    owner_thread = asyncio.run(close_runtime())

    assert len(cleanup_threads) == 1
    assert cleanup_threads[0] != owner_thread


def test_async_close_awaits_async_cleanup_callback_on_the_owner_loop():
    """Standalone async cleanup callbacks run on the active event loop."""
    cleanup_threads = []

    async def cleanup():
        """Record the thread that runs asynchronous cleanup."""
        cleanup_threads.append(get_ident())

    runtime = ApplicationRuntime(Mock(), Mock(), 1.0, cleanup_callbacks=[cleanup])

    async def close_runtime():
        """Close the runtime and return its event-loop thread."""
        owner_thread = get_ident()
        await runtime.close_async()
        return owner_thread

    owner_thread = asyncio.run(close_runtime())

    assert cleanup_threads == [owner_thread]


def test_async_close_settles_sync_cleanup_after_cancellation():
    """Cancellation waits until synchronous cleanup has released its worker."""
    started, release, settled = Event(), Event(), Event()

    def cleanup():
        """Hold cleanup until the async shutdown receives cancellation."""
        started.set()
        release.wait(timeout=1.0)
        settled.set()

    runtime = ApplicationRuntime(Mock(), Mock(), 1.0, cleanup_callbacks=[cleanup])

    async def scenario():
        """Cancel shutdown while cleanup is active, then verify it settles."""
        closing = asyncio.create_task(runtime.close_async())
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 1.0), timeout=2.0)
        closing.cancel()
        asyncio.get_running_loop().call_later(0.01, release.set)
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert settled.is_set()

    asyncio.run(scenario())


def test_async_close_rejects_a_different_resource_loop_before_cleanup(dependencies):
    """Wrong-loop shutdown leaves every resource available for its owner to close."""

    class Resource:
        """Represent an async resource that reports its owning event loop."""

        async_close_loop: asyncio.AbstractEventLoop
        async_closes: int

        def __init__(self, owner_loop):
            self.async_close_loop = owner_loop
            self.async_closes = 0

        def close(self):
            """Provide the synchronous cleanup handle registered by the runtime."""

        async def close_async(self):
            """Count an async close on the resource owner loop."""
            self.async_closes += 1

    owner_loop = asyncio.new_event_loop()
    resource = Resource(owner_loop)
    runtime = ApplicationRuntime(Mock(), Mock(), 1.0, cleanup_callbacks=[resource.close])

    try:
        with pytest.raises(RuntimeError, match="owning loop"):
            asyncio.run(runtime.close_async())

        assert resource.async_closes == 0

        owner_loop.run_until_complete(runtime.close_async())
        assert resource.async_closes == 1
    finally:
        owner_loop.close()


def test_async_close_settles_cleanup_after_cancellation():
    """Cancellation cannot abandon a loop-affine resource close."""

    class Resource:
        """Hold async cleanup until the test releases its owner task."""

        def __init__(self, started, release, settled):
            self._started = started
            self._release = release
            self._settled = settled

        @property
        def async_close_required(self):
            """Require async cleanup for this resource."""
            return True

        def close(self):
            """Reject synchronous cleanup for this resource."""
            raise RuntimeError("Synchronous cleanup is unsafe.")

        async def close_async(self):
            """Signal when cleanup starts and settle after the release event."""
            self._started.set()
            await self._release.wait()
            self._settled.set()
            raise OSError("resource failed after cancellation")

    async def scenario():
        """Cancel shutdown while cleanup is active, then let the owned close finish."""
        started, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        runtime = ApplicationRuntime(
            Mock(),
            Mock(),
            1.0,
            cleanup_callbacks=[Resource(started, release, settled).close],
        )
        closing = asyncio.create_task(runtime.close_async())
        await started.wait()
        closing.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await closing
        assert settled.is_set()
        assert raised.value.__notes__ == [
            "Runtime cleanup also failed: OSError: resource failed after cancellation"
        ]

    asyncio.run(scenario())


def test_async_close_tries_remaining_resources_after_cleanup_failure():
    """Async shutdown reports its first resource failure after trying every owner."""
    synchronous = Mock(return_value=None)
    first = Mock(side_effect=OSError("first close failed"))
    second = Mock(side_effect=RuntimeError("second close failed"))
    runtime = ApplicationRuntime(
        Mock(), Mock(), 1.0, cleanup_callbacks=[synchronous, second, first]
    )

    with pytest.raises(OSError, match="first close failed") as raised:
        asyncio.run(runtime.close_async())

    assert raised.value.__notes__ == [
        "Additional runtime cleanup failed: RuntimeError: second close failed"
    ]
    first.assert_called_once_with()
    second.assert_called_once_with()
    synchronous.assert_called_once_with()


def test_async_close_preserves_shutdown_event_failure_during_cleanup(dependencies):
    """A stopped-event failure stays primary while asynchronous cleanup still runs."""
    dependencies["telemetry_activity"].side_effect = RuntimeError("event failed")

    class Resource:
        """Hold cleanup until shutdown has recorded cancellation."""

        def __init__(self, started, release):
            self.started = started
            self.release = release

        @property
        def async_close_required(self):
            """Require cleanup on the active event loop."""
            return True

        def close(self):
            """Provide the runtime's synchronous cleanup handle."""

        async def close_async(self):
            """Wait for cancellation, then report the resource close failure."""
            self.started.set()
            await self.release.wait()
            raise OSError("resource failed")

    async def scenario():
        """Cancel async shutdown and preserve its original telemetry failure."""
        started, release = asyncio.Event(), asyncio.Event()
        runtime = ApplicationRuntime(
            Mock(), Mock(), 1.0, cleanup_callbacks=[Resource(started, release).close]
        )
        closing = asyncio.create_task(runtime.close_async())
        await started.wait()
        closing.cancel()
        release.set()
        with pytest.raises(RuntimeError, match="event failed") as raised:
            await closing
        assert raised.value.__notes__ == [
            "Runtime rollback also failed: OSError: resource failed",
            "Runtime cleanup completed after cancellation was requested.",
        ]

    asyncio.run(scenario())


def test_async_close_preserves_shutdown_event_failure_without_cancellation(dependencies):
    """A stopped-event failure remains primary after successful async cleanup."""
    dependencies["telemetry_activity"].side_effect = RuntimeError("event failed")
    cleanup = Mock(return_value=None)
    runtime = ApplicationRuntime(Mock(), Mock(), 1.0, cleanup_callbacks=[cleanup])

    with pytest.raises(RuntimeError, match="event failed") as raised:
        asyncio.run(runtime.close_async())

    cleanup.assert_called_once_with()
    assert getattr(raised.value, "__notes__", []) == []


def test_backend_replacements_remain_owned_until_runtime_shutdown(dependencies):
    """Every live backend replacement is closed when its runtime is released."""
    active_loop = Mock()
    first, second = Mock(), Mock()
    dependencies["OpenAIBackend"].side_effect = [first, second]
    runtime = ApplicationRuntime(active_loop, Mock(), 1.0, cleanup_callbacks=[])

    runtime.apply_configuration("backend.base_url", ApplicationSettings())
    runtime.apply_configuration_changes(("backend.default_model",), ApplicationSettings())
    runtime.close()

    first.close.assert_called_once_with()
    second.close.assert_called_once_with()
    assert active_loop.replace_backend.call_count == 2


def test_close_reports_telemetry_shutdown_timeout():
    """Runtime shutdown exposes telemetry flush or adapter failure to its caller."""
    telemetry = Mock()
    telemetry.close.return_value = False
    runtime = ApplicationRuntime(Mock(), telemetry, 1.0)

    with pytest.raises(TimeoutError, match="1.0 seconds"):
        runtime.close()

    runtime.close()


def test_close_cleans_up_when_shutdown_activity_fails(dependencies):
    """A stopped-event failure remains primary while owned resources still close."""
    telemetry = Mock()
    dependencies["telemetry_activity"].side_effect = RuntimeError("event failed")
    runtime = ApplicationRuntime(Mock(), telemetry, 1.0)

    with pytest.raises(RuntimeError, match="event failed"):
        runtime.close()

    telemetry.close.assert_called_once_with(timeout=1.0)


def test_runtime_accepts_resources_without_close_contract(dependencies, assembled):
    """Composition tracks only resources that advertise deterministic cleanup."""
    workspace, paths, workspace_paths, settings = assembled
    dependencies["SQLiteTelemetryAdapter"].return_value = object()
    dependencies["SQLiteSessionStore"].return_value = object()

    runtime = ApplicationRuntime.create(
        workspace, paths, workspace_paths, settings, Mock(), Mock(), Mock()
    )
    runtime.close()

    dependencies["Telemetry"].return_value.close.assert_called_once_with(
        timeout=settings.telemetry.shutdown_timeout
    )


def test_rollback_preserves_primary_error_when_cleanup_fails(dependencies, assembled):
    """Secondary close failures annotate rather than replace the startup error."""
    workspace, paths, workspace_paths, settings = assembled
    dependencies["Loop"].create_default.side_effect = RuntimeError("startup failed")
    dependencies["Telemetry"].return_value.close.side_effect = OSError("close failed")

    with pytest.raises(RuntimeError, match="startup failed") as raised:
        ApplicationRuntime.create(
            workspace, paths, workspace_paths, settings, Mock(), Mock(), Mock()
        )

    assert raised.value.__notes__ == ["Runtime rollback also failed: OSError: close failed"]


def test_close_reports_first_cleanup_error_and_notes_additional_failures():
    """Normal shutdown attempts every callback and reports all cleanup failures."""
    first = Mock(side_effect=OSError("first"))
    second = Mock(side_effect=RuntimeError("second"))
    runtime = ApplicationRuntime(Mock(), Mock(), 1.0, cleanup_callbacks=[second, first])

    with pytest.raises(OSError, match="first") as raised:
        runtime.close()

    assert raised.value.__notes__ == ["Additional runtime cleanup failed: RuntimeError: second"]
    first.assert_called_once_with()
    second.assert_called_once_with()


def test_context_close_releases_real_sqlite_connection_without_resource_warning(tmp_path):
    """Context shutdown explicitly closes the runtime-owned SQLite telemetry connection."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        adapter = SQLiteTelemetryAdapter(tmp_path / "telemetry.sqlite3", workspace_id="workspace")
        telemetry = Telemetry(adapter, workspace_id="workspace")
        with ApplicationRuntime(Mock(), telemetry, 1.0):
            telemetry.activity("runtime.test")
        del telemetry
        del adapter
        gc.collect()

    assert not [warning for warning in caught if warning.category is ResourceWarning]


def test_async_runtime_binds_all_services_to_its_controller(dependencies, assembled, monkeypatch):
    """The async runtime uses one mailbox and stop control across the complete service graph."""
    workspace, paths, workspace_paths, settings = assembled
    source = MagicMock(spec=ConsoleInteraction)
    controller = Mock()
    controller.running = False
    receive = Mock()

    async def run_async(conversation, *, route_idle_commands):
        """Execute the runtime callback through its supplied awaitable input reader."""
        await conversation(receive)
        assert route_idle_commands is True

    controller.run_async = AsyncMock(side_effect=run_async)
    create = Mock(return_value=controller)
    monkeypatch.setattr(runtime_module, "SessionController", create)
    runtime = ApplicationRuntime.create(
        workspace,
        paths,
        workspace_paths,
        settings,
        Mock(),
        Mock(),
        source,
    )
    active = dependencies["Loop"].create_default.return_value
    active.run_async = AsyncMock()
    try:
        asyncio.run(runtime.run_async())
        runtime.stop()
    finally:
        runtime.close()

    create.assert_called_once_with(source)
    for dependency in ("PermissionManager", "SessionManager", "create_default_tool_registry"):
        assert dependencies[dependency].call_args.kwargs["interaction"] is controller.interaction
    assert (
        dependencies["Loop"].create_default.call_args.kwargs["interaction"]
        is controller.interaction
    )
    assert dependencies["Loop"].create_default.call_args.kwargs["run_control"] is controller.control
    active.run_async.assert_awaited_once_with(input_source=receive)
    controller.stop.assert_called_once_with()
    controller.run_async.assert_awaited_once()


def test_runtime_refuses_to_close_services_while_execution_is_active():
    """Owned resources stay live until conversation execution has settled."""
    controller = Mock(running=True)
    resource = Mock()
    runtime = ApplicationRuntime(
        Mock(), Mock(), 1, cleanup_callbacks=[resource], session_controller=controller
    )

    with pytest.raises(RuntimeError, match="settle before closing"):
        runtime.close()
    with pytest.raises(RuntimeError, match="settle before closing"):
        asyncio.run(runtime.close_async())
    resource.assert_not_called()
    controller.running = False
    runtime.close()
    resource.assert_called_once_with()


def test_sync_runtime_keeps_its_native_path_when_a_controller_is_bound():
    """The synchronous runtime never routes the loop through its async controller."""
    loop = Mock()
    controller = Mock(running=False)
    runtime = ApplicationRuntime(
        loop, Mock(), 1, cleanup_callbacks=[], session_controller=controller
    )

    runtime.run()

    runtime.close()
    loop.run.assert_called_once_with()
    controller.run_async.assert_not_called()
