"""Tests for user-command registration and dispatch."""

import asyncio
from threading import Barrier, Event, Lock, Thread, get_ident
from unittest.mock import Mock

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from pydantic import ValidationError

from loop import (
    Command,
    CommandCapability,
    CommandCompletion,
    CommandContext,
    CommandManager,
    CommandRegistration,
    CommandRegistrationError,
    CommandsProvider,
    CompletionManager,
    CompletionProviderRegistration,
    CompletionValue,
    ConsoleInteraction,
    Interaction,
    Problem,
    ProblemException,
    command,
)
from loop.commands import command_manager as module
from loop.commands.utils import get_command_arguments_model
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry


def declared_command(name: str = "test") -> Command:
    """Build a parameterless command with explicit metadata."""

    def function() -> None:
        pass

    return Command(
        function,
        name=name,
        description="Test command.",
        arguments_model=get_command_arguments_model(function, name),
    )


def test_manager_registers_explicit_discovered_and_configured_commands():
    """Every registration form preserves explicit catalog order."""
    explicit = declared_command("explicit")

    def discovered() -> None:
        """Run a discovered command."""

    configured = CommandRegistration(discovered, name="configured")
    manager = CommandManager(
        (
            CommandRegistration(
                explicit.function, name=explicit.name, description=explicit.description
            ),
            discovered,
            configured,
        ),
        Mock(spec=Interaction),
    )

    assert [item.name for item in manager.commands] == [
        "help",
        "exit",
        "quit",
        "explicit",
        "discovered",
        "configured",
    ]
    assert manager.commands[3].name == explicit.name
    assert manager.exit_requested is False


def test_interaction_property_can_be_replaced_and_cleared():
    """The default interaction can be read, replaced, and cleared."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager()

    assert manager.interaction is None
    manager.interaction = interaction
    assert manager.interaction is interaction
    manager.interaction = None
    assert manager.interaction is None


def test_command_dispatch_records_success_and_sanitized_validation_failure():
    """Central dispatch emits lifecycle activity and typed failures without argument contents."""
    interaction = Mock(spec=Interaction)

    def configured(value: int) -> None:
        """Accept one integer."""

    manager = CommandManager([CommandRegistration(configured, name="configured")], interaction)
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    try:
        manager.call("configured", "1")
        manager.call("configured", "private-invalid")
        assert telemetry.close(1)
    finally:
        set_telemetry(None)

    assert [record.event_name for record in adapter.records] == [
        "command.started",
        "command.completed",
        "command.started",
        "command.failed",
    ]
    assert "private-invalid" not in repr(adapter.records)


@pytest.mark.parametrize("has_interaction", [False, True])
def test_manager_reports_structured_operational_failure(monkeypatch, has_interaction):
    """Dispatch reports operational problems without misclassifying them as bad arguments."""
    problem = Problem(
        code="sandbox.management.incomplete",
        title="Native reinstall incomplete",
        detail="Install: policy changed\nLinux execution unavailable.",
        operation="sandbox.reinstall",
    )
    error = ProblemException(problem)
    interaction = Mock(spec=Interaction) if has_interaction else None
    log_problem = Mock()
    monkeypatch.setattr(module, "log_problem", log_problem)

    def configured() -> None:
        """Represent one failed operational command."""
        raise error

    manager = CommandManager([CommandRegistration(configured, name="configured")], interaction)

    if interaction is None:
        with pytest.raises(ProblemException, match="Install: policy changed"):
            manager.call("configured")
    else:
        manager.call("configured")
        interaction.report.assert_called_once_with(problem)

    log_problem.assert_called_once()
    assert log_problem.call_args.args[1:] == (problem, error)


def test_manager_registers_structural_command_providers():
    """Providers contribute registrations without inheriting a framework base class."""

    class Provider:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_commands(self):
            """Return one configured command."""

            def supplied() -> None:
                """Run a supplied command."""

            return (CommandRegistration(supplied, name=self.name),)

    provider: CommandsProvider = Provider("first")
    manager = CommandManager(providers=(provider,))
    manager.register_providers((Provider("second"),))

    assert [command.name for command in manager.commands] == [
        "help",
        "exit",
        "quit",
        "first",
        "second",
    ]


def test_register_resolves_declared_metadata_and_registration_options():
    """Registration resolves passive declarations and explicit options."""
    manager = CommandManager()

    @command
    def discovered(enabled: bool = False) -> None:
        """Discover this description."""

    def target() -> None:
        """Run the target command."""

    manager.register(discovered)
    manager.register(target, name="other", description="Declared description.")

    assert discovered is manager.commands[-2].function
    assert manager.commands[-2].description == "Discover this description."
    assert target is manager.commands[-1].function
    assert manager.commands[-1].name == "other"
    assert manager.commands[-1].description == "Declared description."


def test_command_capabilities_are_inherited_and_can_be_overridden():
    """Declarations and registrations preserve explicit runtime capability requirements."""
    manager = CommandManager()

    @command(capabilities=frozenset({CommandCapability.CONTEXT_READ}))
    def inherited() -> None:
        """Run a context-readable declared command."""

    def configured() -> None:
        """Run a command with registration metadata."""

    manager.register(inherited)
    manager.register(
        CommandRegistration(
            configured,
            capabilities=frozenset({CommandCapability.CONTEXT_UPDATE}),
        )
    )
    manager.register(CommandRegistration(configured, name="serialized"))

    assert manager.commands[-3].capabilities == frozenset({CommandCapability.CONTEXT_READ})
    assert manager.commands[-2].capabilities == frozenset({CommandCapability.CONTEXT_UPDATE})
    assert manager.commands[-1].capabilities == frozenset({CommandCapability.SESSION_IDLE})

    override = CommandManager()
    override.register(
        CommandRegistration(
            configured,
            capabilities=frozenset({CommandCapability.CONTEXT_UPDATE}),
        ),
        capabilities=frozenset({CommandCapability.CONTEXT_READ}),
    )

    assert override.commands[-1].capabilities == frozenset({CommandCapability.CONTEXT_READ})


def test_command_sync_call_rejects_async_and_awaitable_results():
    """Direct sync command calls reject both coroutine callbacks and returned awaitables."""

    async def asynchronous() -> None:
        """Provide a native coroutine callback."""

    async def result() -> None:
        """Provide an awaitable returned by a sync callback."""

    def returning_awaitable() -> object:
        """Return a coroutine to exercise sync result validation."""
        return result()

    with pytest.raises(ProblemException) as captured:
        Command(asynchronous).registered(name="asynchronous").call("")
    assert captured.value.problem.code == "command.async_required"

    with pytest.raises(ProblemException) as captured:
        Command(returning_awaitable).registered(name="returning_awaitable").call("")
    assert captured.value.problem.code == "command.async_required"


def test_capabilities_filter_dispatch_and_consume_inactive_slash_commands():
    """Active commands run immediately while inactive and unknown slash input is consumed."""
    interaction = Mock(spec=Interaction)
    calls = []

    def immediate(context: CommandContext) -> None:
        """Record direct dispatch and inspect the current command catalog."""
        assert context.interaction is interaction
        assert context.available_commands is not None
        assert context.execution_mode == "sync"
        calls.append(context.name)

    def idle_only() -> None:
        """Represent a command that requires an idle session owner."""

    manager = CommandManager(interaction=interaction)
    manager.register(
        CommandRegistration(
            immediate,
            capabilities=frozenset({CommandCapability.CONTEXT_UPDATE}),
        )
    )
    manager.register(idle_only)
    manager.set_capability_provider(
        lambda: frozenset(
            {
                CommandCapability.CONTEXT_READ,
                CommandCapability.CONTEXT_UPDATE,
                CommandCapability.CONVERSATION_ACTIVE,
                CommandCapability.CONVERSATION_CONTROL,
                CommandCapability.RUN_ACTIVE,
            }
        )
    )

    assert manager.handle_user_command("/idle_only") is True
    assert manager.handle_user_command("/missing") is True
    assert manager.handle_user_command("/immediate") is True
    assert calls == ["immediate"]
    interaction.warning.assert_called_once_with(
        "Command '/idle_only' is unavailable in the current conversation state. "
        "Type /help to see commands that can run now."
    )
    assert interaction.report.call_args.args[0].detail == (
        "Unknown command '/missing'. Type /help for available commands."
    )


def test_help_and_completion_share_context_capability_filtering():
    """Help and slash completion expose the same capability-filtered command catalog."""
    interaction = Mock(spec=Interaction)

    def hidden() -> None:
        """Never display this command while active."""

    def readable() -> None:
        """Remain available to inspect the active context."""

    manager = CommandManager(interaction=interaction)
    manager.register(hidden)
    manager.register(
        CommandRegistration(
            readable,
            capabilities=frozenset({CommandCapability.CONTEXT_READ}),
        )
    )
    manager.set_capability_provider(lambda: frozenset({CommandCapability.CONTEXT_READ}))

    assert [item.name for item in manager.active_commands] == [
        "help",
        "readable",
    ]
    assert manager.handle_user_command("/help", interaction) is True
    visible = [item.name for item in interaction.table.call_args.args[0]]
    assert visible == ["help", "readable"]
    completion = CompletionManager((manager.completion_adapter,))
    suggestions = list(completion.get_completions(Document("/"), CompleteEvent()))
    assert {item.text for item in suggestions} == {"/help", "/readable"}


def test_manager_registers_provider_completion_sources():
    """Command providers add named value sources to the manager-owned command adapter."""

    class Provider:
        def get_commands(self):
            """Return one command using a named value source."""

            def choose(value: str) -> None:
                """Accept one completed value."""

            return (
                CommandRegistration(
                    choose,
                    completion=CommandCompletion(provider="choices"),
                ),
            )

        def get_completion_providers(self):
            """Return values referenced by the command's grammar."""
            return (
                CompletionProviderRegistration(
                    "choices",
                    lambda: (CompletionValue("blue"), CompletionValue("red")),
                ),
            )

    manager = CommandManager(providers=(Provider(),))
    completion = CompletionManager((manager.completion_adapter,))

    suggestions = list(completion.get_completions(Document("/choose b"), CompleteEvent()))

    assert [item.text for item in suggestions] == ["blue"]


def test_inactive_command_requires_an_interaction():
    """An inactive command requires an interaction to report its current availability."""

    def unavailable() -> None:
        """Represent an operation disabled during active execution."""

    manager = CommandManager(exit_command_names=None)
    manager.register(unavailable)
    manager.set_capability_provider(lambda: frozenset())

    with pytest.raises(ValueError, match="requires an Interaction"):
        manager.handle_user_command("/unavailable")


def test_register_retains_explicit_and_function_declared_completion_metadata():
    """Registration carries completion grammars from options or decorated functions."""
    manager = CommandManager()
    explicit = CommandCompletion(values=(CompletionValue("explicit"),))
    inherited = CommandCompletion(values=(CompletionValue("inherited"),))

    @command(completion=explicit)
    def first(value: str) -> None:
        """Select the first value."""

    def second(value: str) -> None:
        """Select the second value."""

    CommandCompletion.set_completion(second, inherited)
    manager.register(first)
    manager.register(second)

    assert manager.commands[-2].completion is explicit
    assert manager.commands[-1].completion is inherited


@pytest.mark.parametrize("name", ["", "/test", "bad name", "bad\tname"])
def test_register_rejects_invalid_command_names(name):
    """Declared command names must be useful slash-free tokens without whitespace."""
    manager = CommandManager()

    with pytest.raises(ValueError, match="Invalid command name"):
        manager.register(
            CommandRegistration(
                declared_command(name).function,
                name=name,
                description="Test command.",
            )
        )


def test_register_rejects_duplicates_and_allows_registration_metadata_overrides():
    """Registration rejects duplicates and allows metadata overrides on registrations."""
    manager = CommandManager()
    manager.register(
        CommandRegistration(
            declared_command("test").function, name="test", description="Test command."
        )
    )

    with pytest.raises(ValueError, match="already registered"):
        manager.register(
            CommandRegistration(
                declared_command("test").function, name="test", description="Test command."
            )
        )
    manager.register(
        CommandRegistration(
            declared_command().function,
            name="renamed",
            description="Test command.",
        )
    )

    def configured() -> None:
        """Run the configured command."""

    completion = CommandCompletion(values=(CompletionValue("configured"),))
    manager.register(
        CommandRegistration(configured, name="configured", description="Run."),
        name="other",
        description="Run another command.",
        completion=completion,
    )
    assert manager.commands[-1].name == "other"
    assert manager.commands[-1].description == "Run another command."
    assert manager.commands[-1].completion is completion


def test_register_rejects_missing_descriptions_and_invalid_parameters():
    """Discovery requires documentation and schema-compatible annotated parameters."""
    manager = CommandManager()

    def undocumented() -> None:
        pass

    def untyped(value) -> None:
        """Use an untyped value."""

    with pytest.raises(CommandRegistrationError, match="must have a docstring"):
        manager.register(undocumented)
    with pytest.raises(CommandRegistrationError, match="needs a type annotation"):
        manager.register(untyped)


def test_handle_user_command_deserializes_arguments_and_injects_context():
    """Slash input detects a command while dispatch uses its slash-free name and typed values."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)

    calls = []

    def inspect_context(context: CommandContext, count: int) -> None:
        """Inspect one invocation."""
        assert context.interaction is interaction
        assert context.execution_mode == "sync"
        calls.append((context.name, count))

    manager.register(inspect_context)

    assert manager.is_user_command("ordinary /inspect_context") is False
    assert manager.is_user_command("/inspect_context 3") is True
    assert manager.is_user_command("/") is True
    assert manager.handle_user_command("ordinary /inspect_context") is False
    assert manager.handle_user_command("/inspect_context 3") is True
    assert manager.call("inspect_context", "count=4") is None
    assert calls == [("inspect_context", 3), ("inspect_context", 4)]


def test_handle_user_command_async_dispatches_sync_callbacks_to_workers():
    """Async command dispatch adapts an ordinary sync callback without registration metadata."""
    interaction = Mock(spec=Interaction)
    interaction.worker_calls_routed = True
    manager = CommandManager()
    observed = []
    owner_thread = get_ident()

    def probe(context: CommandContext) -> None:
        """Record the invocation interaction and thread."""
        observed.append((context.interaction, context.execution_mode, get_ident()))

    manager.register(probe)

    async def scenario():
        """Dispatch ordinary text and one synchronous command."""
        assert await manager.handle_user_command_async("hello", interaction) is False
        assert await manager.handle_user_command_async("/probe", interaction) is True

    asyncio.run(scenario())
    assert len(observed) == 1
    assert observed[0][:2] == (interaction, "async")
    assert observed[0][2] != owner_thread


def test_sync_dispatch_routes_async_prompt_through_sync_interaction_owner():
    """A dual sync/async interaction does not leak its loop-bound prompt into the runner."""
    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt.return_value = "session"
    answers = []

    async def select_scope_async(context: CommandContext) -> None:
        """Use the sync dispatch adapter to obtain a scope choice."""
        answers.append(await context.interaction.prompt_async("Choose a scope:"))

    manager = CommandManager(
        commands=(CommandRegistration(select_scope_async, name="select_scope"),),
        interaction=interaction,
    )
    try:
        manager.call("select_scope")
    finally:
        manager.close()

    assert answers == ["session"]
    interaction.prompt.assert_called_once_with("Choose a scope:")
    interaction.prompt_async.assert_not_called()


def test_sync_blocking_operation_runs_on_the_bounded_worker():
    """Synchronous UI filesystem work uses the command manager's worker pool."""
    manager = CommandManager()
    owner_thread = get_ident()

    worker_thread, value = manager.run_blocking(lambda: (get_ident(), "visible"))

    assert worker_thread != owner_thread
    assert value == "visible"


def test_sync_blocking_operation_rejects_event_loop_and_closed_manager():
    """Synchronous UI work fails clearly on an event loop or after manager shutdown."""
    manager = CommandManager()

    async def reject_event_loop():
        """Reject blocking completion while an event loop is active."""
        with pytest.raises(ProblemException) as captured:
            manager.run_blocking(lambda: None)
        assert captured.value.problem.code == "command.async_required"

    asyncio.run(reject_event_loop())
    manager.close()

    with pytest.raises(ProblemException) as captured:
        manager.run_blocking(lambda: None)
    assert captured.value.problem.code == "command.dispatch_busy"


def test_sync_blocking_operation_reuses_existing_worker_and_releases_failed_admission(
    monkeypatch,
):
    """Nested work stays in its worker and submission failures release bounded capacity."""
    manager = CommandManager()
    worker_thread = asyncio.run(manager.run_blocking_async(manager.run_blocking, get_ident))
    assert worker_thread != get_ident()

    with monkeypatch.context() as scoped:
        scoped.setattr(
            module.ThreadPoolExecutor, "submit", Mock(side_effect=RuntimeError("submit"))
        )
        with pytest.raises(RuntimeError, match="submit"):
            manager.run_blocking(lambda: None)

    assert manager.run_blocking(lambda: "available") == "available"


def test_sync_blocking_operation_from_another_manager_uses_its_own_worker():
    """A different command manager does not mistake another pool's worker for its own."""
    first = CommandManager()
    second = CommandManager()

    def invoke_second_manager():
        """Call the second manager while executing on the first manager's worker."""
        first_worker = get_ident()
        second_worker = second.run_blocking(get_ident)
        return first_worker, second_worker

    try:
        first_worker, second_worker = asyncio.run(first.run_blocking_async(invoke_second_manager))
    finally:
        first.close()
        second.close()

    assert first_worker != second_worker


def test_sync_blocking_operation_reports_worker_failures():
    """Synchronous bounded work propagates its original operation error."""
    manager = CommandManager()

    def fail():
        """Raise a specific failure from the worker."""
        raise ValueError("completion failed")

    with pytest.raises(ValueError, match="completion failed"):
        manager.run_blocking(fail)


def test_async_command_context_requires_a_manager_for_sync_leaf_work():
    """An unowned async context refuses to run synchronous work on its event loop."""
    context = CommandContext(
        name="probe",
        interaction=Mock(spec=Interaction),
        execution_mode="async",
    )

    async def scenario():
        """Reject direct blocking work without a command-manager worker."""
        with pytest.raises(ProblemException) as captured:
            await context.run_blocking_async(lambda: None)
        assert captured.value.problem.code == "command.blocking_dispatch_unavailable"

    asyncio.run(scenario())


def test_call_async_consumes_an_unknown_command_after_reporting_it():
    """Async dispatch reports an unknown command without attempting callback execution."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)

    asyncio.run(manager.call_async("missing"))

    interaction.report.assert_called_once()
    assert interaction.report.call_args.args[0].code == "command.unknown"


def test_async_dispatch_awaits_native_commands_on_the_owner_loop():
    """Async callbacks run and settle on the active command owner loop."""
    manager = CommandManager()
    observed = []

    async def probe() -> None:
        """Record the loop and thread that own this command callback."""
        observed.append((asyncio.get_running_loop(), get_ident()))

    manager.register(probe)

    async def scenario():
        """Dispatch the native async command from its active event loop."""
        owner_loop = asyncio.get_running_loop()
        owner_thread = get_ident()
        await manager.handle_user_command_async("/probe")
        assert observed == [(owner_loop, owner_thread)]

    asyncio.run(scenario())


def test_sync_dispatch_reuses_its_runner_for_async_commands():
    """A synchronous host reuses one owned event loop for async command callbacks."""
    manager = CommandManager()
    observed = []

    async def probe() -> None:
        """Record the event loop that hosts this callback."""
        observed.append(asyncio.get_running_loop())

    manager.register(probe)
    manager.call("probe")
    manager.call("probe")

    assert observed[0] is observed[1]
    manager.close()


def test_sync_command_result_awaitable_is_resolved_on_the_async_owner():
    """A synchronous callback may return a coroutine that async dispatch resolves natively."""
    manager = CommandManager()
    observed = []

    async def result() -> None:
        """Capture the loop that receives a coroutine returned by sync command code."""
        observed.append((asyncio.get_running_loop(), get_ident()))

    def callback():
        """Return coroutine work without changing the callback's declared sync shape."""
        return result()

    manager.register(callback)

    async def scenario():
        """Await the returned coroutine on the manager's async caller."""
        owner = (asyncio.get_running_loop(), get_ident())
        await manager.call_async("callback")
        assert observed == [owner]

    asyncio.run(scenario())


def test_sync_command_rejects_a_future_from_another_loop():
    """Async dispatch reports a sync callback future that belongs to another loop."""
    other_loop = asyncio.new_event_loop()
    future = other_loop.create_future()
    manager = CommandManager()

    def callback():
        """Return a future that the current command owner cannot safely await."""
        return future

    manager.register(callback)
    try:
        with pytest.raises(ProblemException) as captured:
            asyncio.run(manager.call_async("callback"))
    finally:
        manager.close()
        other_loop.close()

    assert captured.value.problem.code == "command.loop_mismatch"


def test_sync_dispatch_rejects_runner_use_from_a_different_thread():
    """A manager-owned runner remains bound to the thread that first created it."""
    manager = CommandManager()

    async def probe() -> None:
        """Provide a native callback for runner thread ownership."""

    manager.register(probe)
    manager.call("probe")
    failures = []

    def dispatch() -> None:
        """Capture the required-async error from the non-owning thread."""
        try:
            manager.call("probe")
        except ProblemException as error:
            failures.append(error)

    thread = Thread(target=dispatch)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()

    assert len(failures) == 1
    assert failures[0].problem.code == "command.async_required"


def test_sync_runner_is_created_once_for_concurrent_callers(monkeypatch, contended_lock):
    """Concurrent sync callers cannot race manager-owned runner initialization."""
    manager = CommandManager()
    start = Barrier(3)
    runner_creation_started = Event()
    release_runner_creation = Event()
    monkeypatch.setattr(manager, "_runner_lock", contended_lock.lock)
    created_runners = []
    created_runners_lock = Lock()
    results = []

    def create_runner():
        """Hold the first runner creation to expose an unsynchronized check-then-act race."""
        with created_runners_lock:
            runner = Mock()
            created_runners.append(runner)
            first_runner = len(created_runners) == 1
        if first_runner:
            runner_creation_started.set()
            assert release_runner_creation.wait(timeout=3)
        return runner

    monkeypatch.setattr(module.asyncio, "Runner", create_runner)

    def get_runner():
        """Request the manager runner after both callers reach the same start gate."""
        start.wait(timeout=3)
        results.append(manager._sync_runner())

    threads = [Thread(target=get_runner) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        start.wait(timeout=3)
        assert runner_creation_started.wait(timeout=3)
        assert contended_lock.contended.wait(timeout=3)
        assert len(created_runners) == 1
    finally:
        release_runner_creation.set()
        for thread in threads:
            thread.join(timeout=3)
        manager.close()

    assert all(not thread.is_alive() for thread in threads)
    assert manager._sync_runner() is None
    assert len(created_runners) == 1
    assert len(results) == 2
    assert sum(result is not None for result in results) == 1


def test_sync_dispatch_reports_async_editor_state_unavailable_to_legacy_interaction():
    """A synchronous input adapter rejects async-only draft restoration explicitly."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)

    async def probe(context: CommandContext) -> None:
        """Request async editor state unavailable from a synchronous interaction."""
        await context.interaction.prompt_async(initial="draft")

    manager.register(probe)
    manager.call("probe")

    interaction.prompt.assert_not_called()
    interaction.report.assert_called_once()
    assert interaction.report.call_args.args[0].code == "interaction.async_required"
    manager.close()


def test_sync_dispatch_requires_async_api_inside_a_running_loop():
    """A sync call cannot start the manager-owned runner inside a caller event loop."""
    manager = CommandManager()

    async def probe() -> None:
        """Provide a native async callback for the sync boundary check."""

    manager.register(probe)

    async def scenario():
        """Reject sync dispatch before invoking the asynchronous callback."""
        with pytest.raises(ProblemException) as captured:
            manager.call("probe")
        assert captured.value.problem.code == "command.async_required"
        assert "await manager.call_async(...)" in captured.value.problem.detail

    asyncio.run(scenario())


def test_sync_command_dispatch_rejects_callbacks_inside_a_running_loop():
    """Sync callbacks are never invoked by the synchronous hub on an event-loop thread."""
    invoked = []
    manager = CommandManager()

    def probe() -> None:
        """Record whether synchronous dispatch reaches the callback."""
        invoked.append(True)

    manager.register(probe)

    async def scenario():
        """Require callers with a running loop to use the async dispatch API."""
        assert manager._sync_runner() is None
        with pytest.raises(ProblemException) as captured:
            manager.call("probe")
        assert captured.value.problem.code == "command.async_required"

    asyncio.run(scenario())

    assert not invoked
    manager.close()


def test_sync_command_leaves_the_async_loop_responsive():
    """An unmarked synchronous callback runs in a worker while the loop continues to run."""
    manager = CommandManager()
    release = Event()
    signal_started = []

    def blocked() -> None:
        """Hold worker execution until the async owner has made progress."""
        signal_started[0]()
        assert release.wait(timeout=5), "test synchronization was not released"

    manager.register(blocked)

    async def scenario():
        """Observe owner-loop progress while one blocking callback is held."""
        loop = asyncio.get_running_loop()
        worker_started = asyncio.Event()
        signal_started.append(lambda: loop.call_soon_threadsafe(worker_started.set))

        task = asyncio.create_task(manager.handle_user_command_async("/blocked"))
        await worker_started.wait()
        pulse = loop.create_future()
        loop.call_soon(pulse.set_result, None)
        await pulse
        assert not task.done()
        release.set()
        assert await task is True

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        manager.close()


def test_sync_command_requires_a_routed_context_interaction():
    """A worker command cannot receive a live interaction without owner routing."""
    interaction = Mock(spec=Interaction)
    interaction.worker_calls_routed = False
    called = []
    manager = CommandManager(interaction=interaction)

    def blocked(context: CommandContext) -> None:
        """Record accidental execution without an owner-routed interaction."""
        called.append(context.name)

    manager.register(blocked)
    asyncio.run(manager.call_async("blocked"))

    interaction.report.assert_called_once()
    assert interaction.report.call_args.args[0].code == "command.context_relay_unavailable"
    assert called == []


def test_sync_command_admission_is_bounded():
    """Worker and queued command admission rejects work beyond the fixed capacity."""
    manager = CommandManager()
    release = Event()

    def blocked() -> None:
        """Wait until all admitted work is queued for bounded-admission coverage."""
        assert release.wait(timeout=5), "test synchronization was not released"

    manager.register(blocked)

    async def scenario():
        """Fill the worker and queue slots before requesting one more command."""
        loop = asyncio.get_running_loop()
        tasks = [asyncio.create_task(manager.call_async("blocked")) for _ in range(4)]
        tick = loop.create_future()
        loop.call_soon(tick.set_result, None)
        await tick
        with pytest.raises(ProblemException) as captured:
            await manager.call_async("blocked")
        assert captured.value.problem.code == "command.dispatch_busy"
        release.set()
        await asyncio.gather(*tasks)

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        manager.close()


def test_sync_command_releases_admission_when_executor_submission_fails(monkeypatch):
    """A failed executor submission releases the bounded command slot and propagates its error."""
    manager = CommandManager()

    def blocked() -> None:
        """Provide one synchronous command callback."""

    manager.register(blocked)

    async def scenario():
        """Fail before worker start and prove the manager preserves that failure."""
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(
            loop,
            "run_in_executor",
            Mock(side_effect=RuntimeError("executor unavailable")),
        )
        with pytest.raises(RuntimeError, match="executor unavailable"):
            await manager.call_async("blocked")

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [False, True])
def test_handle_user_command_async_settles_after_repeated_cancellation(monkeypatch, failure):
    """Cancelled dispatch retains command ownership until cleanup and preserves cancellation."""
    manager = CommandManager()

    def probe() -> None:
        """Occupy a synchronous command until its worker is released."""

    manager.register(probe)

    async def scenario():
        """Cancel twice while a controlled worker still owns command cleanup."""
        started, release = asyncio.Event(), asyncio.Event()

        async def dispatch(_executor, function, *args):
            """Hold the command until cleanup is released, optionally exposing its failure."""
            started.set()
            await release.wait()
            if failure:
                raise RuntimeError("command failed")
            function(*args)
            return True

        monkeypatch.setattr(
            asyncio.get_running_loop(),
            "run_in_executor",
            lambda *args: asyncio.create_task(dispatch(*args)),
        )
        task = asyncio.create_task(manager.handle_user_command_async("/probe"))
        await started.wait()
        task.cancel()
        tick = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(tick.set_result, None)
        await tick
        task.cancel()
        tick = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(tick.set_result, None)
        await tick
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await task
        if failure:
            assert captured.value.__notes__ == ["Command also stopped with RuntimeError."]

    asyncio.run(scenario())


def test_handle_user_command_async_preserves_completed_command_cancellation(monkeypatch):
    """Cancellation arriving with command completion still propagates to its owner."""
    manager = CommandManager()

    def probe() -> None:
        """Provide one synchronous command for the cancellation boundary."""

    manager.register(probe)

    async def scenario():
        """Finish the worker as its awaiting owner receives cancellation."""
        owner = asyncio.current_task()

        async def dispatch(_executor, function, *args):
            """Request caller cancellation immediately before reporting command completion."""
            owner.cancel()
            function(*args)
            return True

        monkeypatch.setattr(
            asyncio.get_running_loop(),
            "run_in_executor",
            lambda *args: asyncio.create_task(dispatch(*args)),
        )
        with pytest.raises(asyncio.CancelledError):
            await manager.handle_user_command_async("/probe")

    asyncio.run(scenario())


def test_handle_user_command_async_preserves_cancellation_of_worker_future(monkeypatch):
    """A cancelled worker future does not replace its owner's cancellation."""
    manager = CommandManager()

    def probe() -> None:
        """Provide one synchronous command for the cancellation boundary."""

    manager.register(probe)

    async def scenario():
        """Cancel both the owner and its executor future while work is pending."""
        loop = asyncio.get_running_loop()
        pending = loop.create_future()
        submitted = asyncio.Event()

        def submit(*_args):
            """Expose a controlled executor future to command dispatch."""
            submitted.set()
            return pending

        monkeypatch.setattr(loop, "run_in_executor", submit)
        task = asyncio.create_task(manager.call_async("probe"))
        await submitted.wait()
        task.cancel()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_handle_user_command_async_preserves_worker_keyboard_interrupt(monkeypatch):
    """A worker interruption reaches its awaiting owner without escaping an auxiliary task."""
    manager = CommandManager()

    def probe() -> None:
        """Provide one synchronous command for the worker exception boundary."""

    manager.register(probe)

    async def scenario():
        """Expose an interrupted executor future at the native worker boundary."""
        loop = asyncio.get_running_loop()
        pending = loop.create_future()
        pending.set_exception(KeyboardInterrupt())
        monkeypatch.setattr(loop, "run_in_executor", lambda *args: pending)
        with pytest.raises(KeyboardInterrupt):
            await manager.handle_user_command_async("/probe")

    asyncio.run(scenario())


def test_handle_user_command_reports_invalid_and_unknown_commands():
    """Invalid arguments and unknown slash commands, including a bare slash, are reported."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)

    def pair(first: int, second: int) -> None:
        """Add a pair."""

    manager.register(pair)

    assert manager.handle_user_command("/pair invalid") is True
    assert interaction.report.call_args.args[0].detail.startswith(
        "Invalid arguments for command '/pair'"
    )
    assert manager.handle_user_command("/pair unknown=1") is True
    assert "Unknown parameter 'unknown'" in interaction.report.call_args.args[0].detail
    assert manager.handle_user_command("/missing argument") is True
    assert interaction.report.call_args.args[0].detail == (
        "Unknown command '/missing'. Type /help for available commands."
    )
    assert manager.handle_user_command("/") is True
    assert interaction.report.call_args.args[0].detail == (
        "Unknown command '/'. Type /help for available commands."
    )


def test_call_requires_interaction_only_when_dispatch_needs_it():
    """Context-free calls work alone while contextual, unknown, and invalid calls need output."""
    manager = CommandManager()

    plain_called = []

    def plain() -> None:
        """Perform a plain side effect."""
        plain_called.append(True)

    def contextual(context: CommandContext) -> None:
        """Use context."""

    def count(value: int) -> None:
        """Parse a count."""

    manager.register(plain)
    manager.register(contextual)
    manager.register(count)

    assert manager.call("plain") is None
    assert plain_called == [True]
    with pytest.raises(ValueError, match="requires an Interaction"):
        manager.call("contextual")
    with pytest.raises(ValueError, match="dispatch requires an Interaction"):
        manager.call("missing")
    with pytest.raises(ValidationError):
        manager.call("count", "invalid")
    with pytest.raises(ValueError, match="must not start"):
        manager.call("/plain")


def test_help_is_mandatory_and_displays_the_complete_sorted_catalog():
    """Every manager exposes discovery and renders all registered commands alphabetically."""
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction, exit_command_names=None)

    def zebra() -> None:
        """Run the zebra command."""

    manager.register(zebra)

    manager.call("help")

    rows = [command.name for command in interaction.table.call_args.args[0]]
    assert rows == ["help", "zebra"]
    assert interaction.table.call_args.kwargs == {
        "title": "Available commands:",
        "prefix": "  /",
    }


def test_exit_command_names_control_termination_aliases():
    """Configured aliases request exit while None adds no termination command."""
    manager = CommandManager(exit_command_names=("stop", "leave"))

    assert [command.name for command in manager.commands] == ["help", "stop", "leave"]
    manager.call("stop")
    assert manager.exit_requested is True

    without_exit = CommandManager(exit_command_names=None)
    assert [command.name for command in without_exit.commands] == ["help"]


def test_request_exit_changes_manager_state():
    """Termination remains an explicit manager operation independent of command aliases."""
    manager = CommandManager(exit_command_names=None)
    manager.request_exit()
    assert manager.exit_requested is True
